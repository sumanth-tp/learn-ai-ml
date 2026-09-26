"""Application service shared by the CLI and the API: run, stream, resume, inspect."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from research_analyst import __version__
from research_analyst.checkpoint import sqlite_checkpointer
from research_analyst.config import Settings
from research_analyst.deps import Deps, build_deps
from research_analyst.events import ProgressEvent
from research_analyst.graph.supervisor import build_graph
from research_analyst.logging_setup import thread_id_var
from research_analyst.models import FinalReport, Source

log = logging.getLogger(__name__)

Status = Literal["not_found", "running_or_interrupted", "complete"]


class RunStatus(BaseModel):
    thread_id: str
    status: Status
    next_nodes: list[str] = []
    report: FinalReport | None = None


class ResearchService:
    """Owns the checkpointer and compiled graph. Use as ``async with ResearchService(s) as svc``."""

    def __init__(
        self,
        settings: Settings,
        deps: Deps | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.settings = settings
        self._deps = deps
        self._checkpointer = checkpointer
        self._stack = AsyncExitStack()
        self.graph: CompiledStateGraph | None = None

    async def __aenter__(self) -> ResearchService:
        self.settings.apply_tracing_env()
        if self._checkpointer is None:
            self._checkpointer = await self._stack.enter_async_context(
                sqlite_checkpointer(self.settings.checkpoint_db)
            )
        self._deps = self._deps or build_deps(self.settings)
        self.graph = build_graph(self._deps, checkpointer=self._checkpointer)
        return self

    async def __aexit__(self, *exc: object) -> None:
        close = getattr(self._deps.search, "aclose", None) if self._deps else None
        if close:
            await close()
        await self._stack.aclose()

    # ------------------------------------------------------------------ helpers
    def _config(self, thread_id: str, question: str | None = None) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "run_name": "research_report",
            "tags": ["research-analyst", self.settings.mode, f"v{__version__}"],
            "metadata": {
                "thread_id": thread_id,
                "question": question or "",
                "llm_model": self.settings.llm_model,
            },
            "max_concurrency": self.settings.max_parallel_workers,
            "recursion_limit": 80,
        }

    async def status(self, thread_id: str) -> RunStatus:
        assert self.graph is not None
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snap.values:
            return RunStatus(thread_id=thread_id, status="not_found")
        report = snap.values.get("report")
        if report is not None and not snap.next:
            return RunStatus(thread_id=thread_id, status="complete", report=report)
        return RunStatus(
            thread_id=thread_id, status="running_or_interrupted", next_nodes=list(snap.next)
        )

    async def sources(self, thread_id: str) -> dict[str, Source]:
        assert self.graph is not None
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return dict(snap.values.get("sources") or {})

    async def _drive(
        self, graph_input: dict | None, thread_id: str, question: str | None
    ) -> AsyncIterator[ProgressEvent]:
        assert self.graph is not None
        token = thread_id_var.set(thread_id)
        started = time.perf_counter()
        try:
            async for _ns, mode, data in self.graph.astream(
                graph_input,
                self._config(thread_id, question),
                stream_mode=["custom", "updates"],
                subgraphs=True,
            ):
                if mode == "custom":
                    yield ProgressEvent.model_validate(data)
        except Exception as exc:
            log.exception("run failed")
            yield ProgressEvent(
                type="run_failed",
                data={
                    "thread_id": thread_id,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "resume": f"research-analyst resume {thread_id}",
                },
            )
            raise
        finally:
            log.info("run ended", extra={"latency_s": round(time.perf_counter() - started, 3)})
            thread_id_var.reset(token)

    # ------------------------------------------------------------------ public API
    async def stream(
        self, question: str, thread_id: str | None = None
    ) -> AsyncIterator[ProgressEvent]:
        """Start a new run (or return the finished one: same thread id = idempotent)."""
        thread_id = thread_id or uuid.uuid4().hex
        current = await self.status(thread_id)
        if current.status == "complete":
            yield ProgressEvent(type="report_ready", data={"thread_id": thread_id, "cached": True})
            return
        if current.status == "running_or_interrupted":
            async for ev in self.resume(thread_id):
                yield ev
            return
        async for ev in self._drive(
            {"question": question, "thread_id": thread_id}, thread_id, question
        ):
            yield ev

    async def resume(self, thread_id: str) -> AsyncIterator[ProgressEvent]:
        """Continue from the last checkpoint. Completed nodes are not re-run."""
        yield ProgressEvent(type="run_resumed", data={"thread_id": thread_id})
        async for ev in self._drive(None, thread_id, None):
            yield ev

    async def run(self, question: str, thread_id: str | None = None) -> FinalReport:
        thread_id = thread_id or uuid.uuid4().hex
        async for _ in self.stream(question, thread_id):
            pass
        st = await self.status(thread_id)
        if st.report is None:
            raise RuntimeError(f"run {thread_id} did not produce a report (status={st.status})")
        return st.report

    # ------------------------------------------------------------------ retention
    async def delete(self, thread_id: str) -> bool:
        """Erase every checkpoint of a run (right-to-erasure, or a poisoned run)."""
        assert self._checkpointer is not None
        existed = (await self.status(thread_id)).status != "not_found"
        await self._checkpointer.adelete_thread(thread_id)
        return existed

    async def purge(self, older_than_days: float) -> list[str]:
        """Delete runs whose most recent checkpoint is older than the retention window."""
        assert self._checkpointer is not None
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        latest: dict[str, datetime] = {}
        async for tup in self._checkpointer.alist(None):
            tid = tup.config["configurable"]["thread_id"]
            ts = datetime.fromisoformat(tup.checkpoint["ts"])
            if tid not in latest or ts > latest[tid]:
                latest[tid] = ts
        expired = sorted(tid for tid, ts in latest.items() if ts < cutoff)
        for tid in expired:
            await self._checkpointer.adelete_thread(tid)
        log.info("purge complete", extra={"deleted": len(expired), "days": older_than_days})
        return expired
