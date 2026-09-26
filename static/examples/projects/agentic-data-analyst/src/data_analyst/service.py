"""The application service: one object the CLI, the API and the evals all share.

It owns the compiled graph and the checkpointer, turns LangGraph's stream into a small
typed event protocol, records metrics, and exposes time travel over checkpoints.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command
from pydantic import BaseModel

from data_analyst import metrics
from data_analyst.config import Settings
from data_analyst.executor import WarehouseExecutor
from data_analyst.graph.builder import build_graph
from data_analyst.graph.nodes import Deps
from data_analyst.llm import StructuredLLM, build_embeddings, build_llm
from data_analyst.logging_setup import get_logger
from data_analyst.prompts import PROMPT_VERSION
from data_analyst.retrieval import SchemaIndex, SemanticCache
from data_analyst.sandbox.chart import ChartSandbox
from data_analyst.validator import SQLValidator
from data_analyst.warehouse.catalog import ALLOWED_TABLES, schema_version
from data_analyst.warehouse.seed import ensure_warehouse

log = get_logger(__name__)
EventType = Literal["progress", "node", "approval_required", "final"]


class Event(BaseModel):
    type: EventType
    data: dict[str, Any]


def build_deps(settings: Settings, llm: StructuredLLM | None = None) -> Deps:
    ensure_warehouse(settings.warehouse_path)
    embeddings = build_embeddings(settings)
    cache = (
        SemanticCache(
            settings.cache_path,
            embeddings,
            threshold=settings.cache_similarity,
            ttl_s=settings.cache_ttl_s,
            schema=schema_version(),
        )
        if settings.cache_enabled
        else None
    )
    return Deps(
        settings=settings,
        llm=llm or build_llm(settings),
        index=SchemaIndex(embeddings),
        validator=SQLValidator(ALLOWED_TABLES, settings.default_limit, settings.max_limit),
        executor=WarehouseExecutor(
            settings.warehouse_path,
            timeout_s=settings.query_timeout_s,
            row_cap=settings.row_cap,
            memory_limit=settings.duckdb_memory_limit,
            threads=settings.duckdb_threads,
        ),
        charts=ChartSandbox(settings.chart_timeout_s, settings.chart_memory_mb),
        cache=cache,
    )


def sqlite_checkpointer(settings: Settings) -> SqliteSaver:
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    return SqliteSaver(conn)


class AnalystService:
    def __init__(self, deps: Deps, checkpointer: Checkpointer) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.checkpointer = checkpointer
        self.graph: CompiledStateGraph = build_graph(deps, checkpointer)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        llm: StructuredLLM | None = None,
        persistent: bool = True,
    ) -> AnalystService:
        saver = sqlite_checkpointer(settings) if persistent else InMemorySaver()
        return cls(build_deps(settings, llm), saver)

    def close(self) -> None:
        conn = getattr(self.checkpointer, "conn", None)
        if conn is not None:
            with suppress(Exception):
                conn.close()

    # ------------------------------------------------------------------ config
    def config(self, thread_id: str, checkpoint_id: str | None = None) -> RunnableConfig:
        configurable: dict[str, Any] = {"thread_id": thread_id}
        if checkpoint_id:
            configurable |= {"checkpoint_id": checkpoint_id, "checkpoint_ns": ""}
        return {
            "configurable": configurable,
            "run_name": "analyst-turn",
            "tags": ["data-analyst", f"llm:{self.settings.llm_mode}"],
            "metadata": {
                "thread_id": thread_id,
                "model": self.settings.llm_model,
                "prompt_version": PROMPT_VERSION,
                "schema_version": schema_version(),
            },
            "recursion_limit": 50,
        }

    # ------------------------------------------------------------------ running
    def ask(self, thread_id: str, question: str) -> Iterator[Event]:
        yield from self._run({"question": question}, self.config(thread_id))

    def resume(
        self, thread_id: str, approved: bool, reviewer: str = "cli", comment: str = ""
    ) -> Iterator[Event]:
        metrics.APPROVALS.labels(outcome="approved" if approved else "rejected").inc()
        payload = {"approved": approved, "reviewer": reviewer, "comment": comment}
        yield from self._run(Command(resume=payload), self.config(thread_id))

    def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        snap = self.graph.get_state(self.config(thread_id))
        return snap.interrupts[0].value if snap.interrupts else None

    def _run(self, inp: Any, config: RunnableConfig) -> Iterator[Event]:
        start = time.perf_counter()
        for mode, chunk in self.graph.stream(inp, config, stream_mode=["updates", "custom"]):
            if mode == "custom":
                yield Event(type="progress", data=chunk)
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    metrics.APPROVALS.labels(outcome="requested").inc()
                    yield Event(type="approval_required", data=update[0].value)
                else:
                    yield Event(type="node", data={"node": node, "keys": sorted(update or {})})
        # Read the thread's LATEST checkpoint: after a replay or fork, `config` still pins
        # the old checkpoint we started from.
        snap = self.graph.get_state(self.config(config["configurable"]["thread_id"]))
        if snap.next:  # paused at an interrupt; the final event comes after resume
            return
        summary = self.summarise(snap.values)
        summary["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
        self._record(summary)
        yield Event(type="final", data=summary)

    def run_to_end(
        self, thread_id: str, question: str, *, auto_approve: bool | None = None
    ) -> dict[str, Any]:
        """Non-interactive: run a question, answering any approval with ``auto_approve``.
        With ``auto_approve=None`` a pending approval is returned as-is."""
        final: dict[str, Any] | None = None
        pending: dict[str, Any] | None = None
        for ev in self.ask(thread_id, question):
            if ev.type == "final":
                final = ev.data
            elif ev.type == "approval_required":
                pending = ev.data
        if final is None and pending is not None and auto_approve is not None:
            for ev in self.resume(thread_id, auto_approve, reviewer="auto"):
                if ev.type == "final":
                    final = ev.data
        if final is None:
            return {"status": "awaiting_approval", "approval": pending}
        return final

    def summarise(self, v: dict[str, Any]) -> dict[str, Any]:
        tin, tout = v.get("input_tokens", 0), v.get("output_tokens", 0)
        attempts = v.get("attempts", 0)
        return {
            "status": v.get("status"),
            "question": v.get("question"),
            "standalone_question": v.get("standalone_question"),
            "answer": v.get("answer"),
            "sql": v.get("sql"),
            "sql_source": v.get("sql_source"),
            "cache_similarity": v.get("cache_similarity"),
            "tables": v.get("tables", []),
            "attempts": attempts,
            "retries": max(attempts - 1, 0),
            "errors": v.get("errors", []),
            "warnings": v.get("warnings", []),
            "estimate": v.get("estimate"),
            "approval": v.get("approval"),
            "result": v.get("result"),
            "chart_png_base64": v.get("chart_png_base64"),
            "chart_error": v.get("chart_error"),
            "input_tokens": tin,
            "output_tokens": tout,
            "llm_calls": v.get("llm_calls", 0),
            "cost_usd": round(self.settings.cost_usd(tin, tout), 6),
        }

    def _record(self, s: dict[str, Any]) -> None:
        metrics.TURNS.labels(status=s["status"], sql_source=s["sql_source"] or "none").inc()
        metrics.RETRIES.observe(s["retries"])
        metrics.LATENCY.observe(s["elapsed_ms"] / 1000)
        metrics.TOKENS.labels(direction="input").inc(s["input_tokens"])
        metrics.TOKENS.labels(direction="output").inc(s["output_tokens"])
        metrics.COST.inc(s["cost_usd"])
        if s["chart_error"]:
            metrics.CHART_FAILURES.inc()
        if s["result"]:
            metrics.REDACTIONS.inc(s["result"].get("redactions", 0))
        log.info(
            "turn_summary",
            status=s["status"],
            retries=s["retries"],
            elapsed_ms=s["elapsed_ms"],
            cost_usd=s["cost_usd"],
            source=s["sql_source"],
        )

    # ------------------------------------------------------------------ state and time travel
    def state(self, thread_id: str) -> dict[str, Any]:
        snap = self.graph.get_state(self.config(thread_id))
        return {
            "next": list(snap.next),
            "values": self.summarise(snap.values),
            "history": snap.values.get("history", []),
        }

    def history(self, thread_id: str) -> list[dict[str, Any]]:
        """Every checkpoint of the thread, newest first: the audit trail of a bad run."""
        out = []
        for snap in self.graph.get_state_history(self.config(thread_id)):
            v = snap.values
            out.append(
                {
                    "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                    "step": snap.metadata.get("step") if snap.metadata else None,
                    "next": list(snap.next),
                    "created_at": snap.created_at,
                    "question": v.get("question"),
                    "sql": v.get("sql"),
                    "attempts": v.get("attempts", 0),
                    "last_error": v.get("last_error"),
                    "status": v.get("status"),
                }
            )
        return out

    def replay(self, thread_id: str, checkpoint_id: str) -> Iterator[Event]:
        """Re-run from a past checkpoint. Steps after it execute again, creating a fork."""
        yield from self._run(None, self.config(thread_id, checkpoint_id))

    def fork_with_sql(self, thread_id: str, checkpoint_id: str, sql: str) -> Iterator[Event]:
        """Time travel with an edit: pretend generate_sql produced ``sql`` at that point,
        then continue. This is how you test a fix against the exact failing state."""
        cfg = self.config(thread_id, checkpoint_id)
        new_cfg = self.graph.update_state(
            cfg, {"sql": sql, "sql_source": "llm", "last_error": None}, as_node="generate_sql"
        )
        yield from self._run(
            None, {**self.config(thread_id), **{"configurable": new_cfg["configurable"]}}
        )
