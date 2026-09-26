"""SupportRunner: the one API every entry point (HTTP, CLI, evals) uses to drive the graph.

It owns the concerns that sit around the graph rather than inside it:
thread ownership, one run per thread at a time, refusing new messages while an
approval is pending, turning LangGraph stream parts into client events, the
review queue, history and time travel.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy import select

from support_agent import metrics
from support_agent.container import Container
from support_agent.db import PendingApproval, Thread, session_scope
from support_agent.errors import PermissionDeniedError, SupportError
from support_agent.graph import STREAMED_NODES, ApprovalDecision
from support_agent.logging_setup import request_id_var, thread_id_var
from support_agent.prompts import APPROVAL_PENDING_MESSAGE
from support_agent.state import Context
from support_agent.tracing import build_callbacks, run_config

log = logging.getLogger(__name__)


class ThreadBusyError(SupportError):
    """Another run on this thread is in progress."""


class NoPendingApprovalError(SupportError):
    """Resume was called but nothing is waiting (already resumed, or never paused)."""


class UnknownThreadError(SupportError):
    pass


@dataclass
class Event:
    type: str  # metadata | token | update | interrupt | message | done | error
    data: dict[str, Any] = field(default_factory=dict)


def message_to_dict(m: BaseMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"id": m.id, "role": m.type, "content": m.content}
    if isinstance(m, AIMessage) and m.tool_calls:
        out["tool_calls"] = [
            {"name": t["name"], "args": t["args"], "id": t["id"]} for t in m.tool_calls
        ]
    if isinstance(m, ToolMessage):
        out["tool_call_id"] = m.tool_call_id
        out["status"] = m.status
    return out


class SupportRunner:
    def __init__(self, container: Container, graph: CompiledStateGraph) -> None:
        self.c = container
        self.graph = graph
        self.settings = container.settings
        self._callbacks = build_callbacks(self.settings)
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ---- ownership ---------------------------------------------------------

    def ensure_thread(self, thread_id: str, user_id: str, create: bool = True) -> None:
        with session_scope(self.c.sessions) as s:
            row = s.get(Thread, thread_id)
            if row is None:
                if not create:
                    raise UnknownThreadError(thread_id)
                s.add(Thread(id=thread_id, user_id=user_id))
                return
            if row.user_id != user_id:
                raise PermissionDeniedError("This conversation belongs to another customer.")

    def thread_owner(self, thread_id: str) -> str:
        with session_scope(self.c.sessions) as s:
            row = s.get(Thread, thread_id)
            if row is None:
                raise UnknownThreadError(thread_id)
            return row.user_id

    def _config(
        self, thread_id: str, user_id: str, kind: str, checkpoint_id: str | None = None
    ) -> dict[str, Any]:
        return run_config(
            self.settings,
            self._callbacks,
            thread_id=thread_id,
            request_id=request_id_var.get(),
            user_id=user_id,
            kind=kind,
            checkpoint_id=checkpoint_id,
        )

    async def pending_interrupts(self, thread_id: str) -> list[dict[str, Any]]:
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return [{"id": i.id, "value": i.value} for i in snap.interrupts]

    # ---- turns -------------------------------------------------------------

    def is_busy(self, thread_id: str) -> bool:
        return self._locks[thread_id].locked()

    def _acquire(self, thread_id: str) -> asyncio.Lock:
        """One run per thread. A second concurrent request gets 409, it does not queue."""
        lock = self._locks[thread_id]
        if lock.locked():
            raise ThreadBusyError(thread_id)
        return lock

    async def stream_turn(
        self, thread_id: str | None, user_id: str, message: str
    ) -> AsyncIterator[Event]:
        thread_id = thread_id or f"th_{uuid.uuid4().hex[:16]}"
        self.ensure_thread(thread_id, user_id)
        async with self._acquire(thread_id):
            yield Event("metadata", {"thread_id": thread_id})
            pending = await self.pending_interrupts(thread_id)
            if pending:
                # The graph is parked on an approval: a new input would abandon it.
                text = (
                    "Your refund for order "
                    f"{pending[0]['value'].get('order_id')} is still being reviewed. "
                    "I'll update you here as soon as it's decided."
                )
                yield Event("message", {"content": text})
                yield Event(
                    "done",
                    {
                        "thread_id": thread_id,
                        "answer": text,
                        "outcome": "awaiting_approval",
                        "interrupted": True,
                    },
                )
                return
            cfg = self._config(thread_id, user_id, "turn")
            async for ev in self._run(
                {"messages": [HumanMessage(message)]}, cfg, thread_id, user_id
            ):
                yield ev

    async def stream_resume(
        self, thread_id: str, decision: ApprovalDecision, interrupt_id: str | None = None
    ) -> AsyncIterator[Event]:
        user_id = self.thread_owner(thread_id)
        # The pending check and the resume happen under one lock, so two reviewers
        # clicking "approve" at once cannot both resume the same interrupt.
        async with self._acquire(thread_id):
            pending = await self.pending_interrupts(thread_id)
            if not pending:
                raise NoPendingApprovalError(thread_id)
            target = (
                next((p for p in pending if p["id"] == interrupt_id), None)
                if interrupt_id
                else pending[0]
            )
            if target is None:
                raise NoPendingApprovalError(f"interrupt {interrupt_id} is not pending")
            yield Event("metadata", {"thread_id": thread_id, "resumed": target["id"]})
            cfg = self._config(thread_id, user_id, "resume")
            cmd = Command(resume={target["id"]: decision.model_dump()})
            async for ev in self._run(cmd, cfg, thread_id, user_id):
                yield ev
            with session_scope(self.c.sessions) as s:
                row = s.get(PendingApproval, target["id"])
                if row is not None:
                    row.status, row.decided_by = "resolved", decision.reviewer

    async def _run(
        self, graph_input: Any, cfg: dict[str, Any], thread_id: str, user_id: str
    ) -> AsyncIterator[Event]:
        """Drive one graph run. The caller must hold the thread lock."""
        thread_id_var.set(thread_id)
        started = time.perf_counter()
        first_token = True
        interrupted = False
        ctx = Context(user_id=user_id, request_id=request_id_var.get())
        async for part in self.graph.astream(
            graph_input,
            cfg,
            context=ctx,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if part["type"] == "messages":
                chunk, meta = part["data"]
                node = meta.get("langgraph_node")
                if node in STREAMED_NODES and isinstance(chunk.content, str) and chunk.content:
                    if first_token:
                        metrics.FIRST_TOKEN.observe(time.perf_counter() - started)
                        first_token = False
                    yield Event("token", {"text": chunk.content, "node": node})
            elif part["type"] == "updates":
                for node, update in part["data"].items():
                    if node == "__interrupt__":
                        interrupted = True
                        for intr in update:
                            self._record_approval(thread_id, intr.id, intr.value)
                            yield Event("interrupt", {"id": intr.id, "value": intr.value})
                        yield Event("message", {"content": APPROVAL_PENDING_MESSAGE})
                    elif node != "__metadata__":
                        keys = sorted((update or {}).keys()) if isinstance(update, dict) else []
                        yield Event("update", {"node": node, "keys": keys})
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        metrics.LATENCY.observe(time.perf_counter() - started)
        values = snap.values
        last_ai = next(
            (
                m
                for m in reversed(values.get("messages", []))
                if isinstance(m, AIMessage) and m.content
            ),
            None,
        )
        answer = (
            APPROVAL_PENDING_MESSAGE if interrupted else (str(last_ai.content) if last_ai else "")
        )
        yield Event(
            "done",
            {
                "thread_id": thread_id,
                "answer": answer,
                "intent": values.get("intent"),
                "outcome": "awaiting_approval" if interrupted else values.get("outcome"),
                "interrupted": interrupted,
                "steps": values.get("steps", 0),
                "tokens": values.get("tokens", 0),
            },
        )

    def _record_approval(self, thread_id: str, interrupt_id: str, value: dict[str, Any]) -> None:
        with session_scope(self.c.sessions) as s:
            if s.get(PendingApproval, interrupt_id) is None:
                s.add(
                    PendingApproval(interrupt_id=interrupt_id, thread_id=thread_id, payload=value)
                )
                metrics.INTERRUPTS.inc()

    async def run_turn(self, thread_id: str | None, user_id: str, message: str) -> dict[str, Any]:
        """Non-streaming convenience: collect events and return the final one."""
        done: dict[str, Any] = {}
        async for ev in self.stream_turn(thread_id, user_id, message):
            if ev.type == "done":
                done = ev.data
        return done

    async def resume(self, thread_id: str, decision: ApprovalDecision) -> dict[str, Any]:
        done: dict[str, Any] = {}
        async for ev in self.stream_resume(thread_id, decision):
            if ev.type == "done":
                done = ev.data
        return done

    # ---- review queue, history and time travel -----------------------------

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        with session_scope(self.c.sessions) as s:
            rows = s.scalars(
                select(PendingApproval)
                .where(PendingApproval.status == "pending")
                .order_by(PendingApproval.created_at)
            ).all()
            return [
                {
                    "interrupt_id": r.interrupt_id,
                    "thread_id": r.thread_id,
                    "payload": r.payload,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def history(self, thread_id: str) -> dict[str, Any]:
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = snap.values or {}
        return {
            "thread_id": thread_id,
            "messages": [message_to_dict(m) for m in values.get("messages", [])],
            "summary": values.get("summary", ""),
            "intent": values.get("intent"),
            "next": list(snap.next),
            "pending_interrupts": [{"id": i.id, "value": i.value} for i in snap.interrupts],
        }

    async def checkpoints(self, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for snap in self.graph.aget_state_history(
            {"configurable": {"thread_id": thread_id}}, limit=limit
        ):
            meta = snap.metadata or {}
            msgs = (snap.values or {}).get("messages", [])
            out.append(
                {
                    "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                    "parent_id": (snap.parent_config or {})
                    .get("configurable", {})
                    .get("checkpoint_id"),
                    "step": meta.get("step"),
                    "source": meta.get("source"),
                    "next": list(snap.next),
                    "created_at": snap.created_at,
                    "messages": len(msgs),
                    "pending_tools": [tc["name"] for tc in getattr(msgs[-1], "tool_calls", [])]
                    if msgs
                    else [],
                }
            )
        return out

    async def replay(self, thread_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Re-run the graph from a past checkpoint (same inputs). Side effects are
        protected by idempotency keys, so a replayed refund is not paid twice."""
        user_id = self.thread_owner(thread_id)
        cfg = self._config(thread_id, user_id, "replay", checkpoint_id=checkpoint_id)
        done: dict[str, Any] = {}
        async with self._acquire(thread_id):
            async for ev in self._run(None, cfg, thread_id, user_id):
                if ev.type == "done":
                    done = ev.data
        return done

    async def fork(self, thread_id: str, checkpoint_id: str, message: str) -> dict[str, Any]:
        """Branch from a past checkpoint with a different user message."""
        user_id = self.thread_owner(thread_id)
        cfg = self._config(thread_id, user_id, "fork", checkpoint_id=checkpoint_id)
        done: dict[str, Any] = {}
        async with self._acquire(thread_id):
            async for ev in self._run(
                {"messages": [HumanMessage(message)]}, cfg, thread_id, user_id
            ):
                if ev.type == "done":
                    done = ev.data
        return done
