"""AgentService: one request in, one fully traced and persisted TraceRecord out."""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from agentmon.agent.backend import FakeBankBackend
from agentmon.agent.graph import AgentGraph
from agentmon.agent.intents import classify_intent
from agentmon.agent.tools import ToolRegistry
from agentmon.clock import Clock
from agentmon.config import Settings
from agentmon.models import ToolCallRecord, TraceRecord
from agentmon.store import Store
from agentmon.tracing import Tracing

log = logging.getLogger("agentmon.service")


class TraceSink(Protocol):
    def ingest(self, trace: TraceRecord) -> None: ...


class ChatResponse(BaseModel):
    trace_id: str
    request_id: str
    answer: str
    blocked: bool
    flags: list[str]
    replayed: bool = False


class AgentService:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        backend: FakeBankBackend,
        model: BaseChatModel,
        tracing: Tracing,
        clock: Clock,
        sink: TraceSink | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.backend = backend
        self.tracing = tracing
        self.clock = clock
        self.sink = sink
        self.prompt_version = settings.prompt_version
        self.agent = AgentGraph(settings, model, ToolRegistry(backend), tracing, clock)

    def deploy_prompt(self, version: str, note: str = "") -> None:
        """Change the live prompt and record it, so root-cause analysis can find it."""
        old, self.prompt_version = self.prompt_version, version
        self.store.record_deployment(self.clock.now(), "prompt_version", old, version, note)
        log.info("deploy.prompt", extra={"old": old, "new": version})

    def handle(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        request_id = request_id or uuid.uuid4().hex
        existing = self.store.get_trace_by_request(request_id)
        if existing is not None:  # idempotent replay: a retried HTTP call never re-runs tools
            return ChatResponse(
                trace_id=existing.trace_id,
                request_id=request_id,
                answer=existing.output,
                blocked=existing.status == "blocked",
                flags=existing.flags,
                replayed=True,
            )
        session_id = session_id or request_id
        intent = classify_intent(message)
        t0 = self.clock.now()
        with self.tracing.span(
            "agent.run",
            **{
                "app.user_id": user_id,
                "app.session_id": session_id,
                "app.request_id": request_id,
                "app.prompt_version": self.prompt_version,
                "app.intent": intent,
            },
        ) as root:
            trace_id = f"{root.get_span_context().trace_id:032x}"
            error: str | None = None
            try:
                state = self.agent.invoke(user_id, request_id, message, self.prompt_version)
            except Exception as exc:
                log.exception("agent.failed", extra={"request_id": request_id})
                error = f"{type(exc).__name__}: {exc}"
                state = {
                    "messages": [],
                    "final": "Sorry, something went wrong on our side.",
                    "flags": ["agent_exception"],
                }
            record = self._record(
                trace_id, request_id, user_id, session_id, message, intent, t0, state, error
            )
            Tracing.set(
                root,
                **{
                    "app.status": record.status,
                    "gen_ai.usage.input_tokens": record.input_tokens,
                    "gen_ai.usage.output_tokens": record.output_tokens,
                    "app.cost_usd": record.cost_usd,
                    "app.flags": record.flags,
                },
            )
        self.store.insert_trace(record)
        log.info(
            "trace.recorded",
            extra={
                "trace_id": trace_id,
                "status": record.status,
                "latency_ms": round(record.latency_ms, 1),
            },
        )
        if self.sink is not None:
            self.sink.ingest(record)
        return ChatResponse(
            trace_id=trace_id,
            request_id=request_id,
            answer=record.output,
            blocked=record.status == "blocked",
            flags=record.flags,
        )

    def _record(
        self,
        trace_id: str,
        request_id: str,
        user_id: str,
        session_id: str,
        message: str,
        intent: str,
        t0: float,
        state: dict,
        error: str | None,
    ) -> TraceRecord:
        messages = state.get("messages", [])
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        calls: list[ToolCallRecord] = []
        in_tok = out_tok = 0
        for m in messages:
            if isinstance(m, AIMessage):
                usage = m.usage_metadata or {}
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
                for tc in m.tool_calls:
                    res = results.get(tc["id"])
                    calls.append(
                        ToolCallRecord(
                            name=tc["name"],
                            args=tc["args"],
                            status=res.status if res is not None else "error",
                            output=str(res.content) if res is not None else "not executed",
                        )
                    )
        flags = list(dict.fromkeys(state.get("flags", [])))
        if error:
            status = "error"
        elif state.get("blocked"):
            status = "blocked"
        elif any(f in flags for f in ("tool_error:timeout", "loop_detected", "step_limit")):
            status = "error"
        else:
            status = "ok"
        return TraceRecord(
            trace_id=trace_id,
            request_id=request_id,
            ts=t0,
            user_id=user_id,
            session_id=session_id,
            input=message,
            output=state.get("final", ""),
            intent=intent,
            prompt_version=self.prompt_version,
            model=self.agent.model_name,
            status=status,
            error=error,
            latency_ms=(self.clock.now() - t0) * 1000,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self.settings.cost_usd(in_tok, out_tok),
            steps=int(state.get("steps", 0)),
            tool_calls=calls,
            flags=flags,
        )
