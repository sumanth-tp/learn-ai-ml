"""The LangGraph agent: input guard -> (model <-> tools)* -> output guard.

Every node opens an OpenTelemetry span. The tools node is hand-written rather than
the prebuilt ToolNode because it has to do five things per call: validate, apply the
action guard, detect loops, run with timeout+retry, and sanitise the result."""

from __future__ import annotations

import json
import logging
import operator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentmon.agent.backend import BackendError, BackendTimeout
from agentmon.agent.guardrails import (
    REFUSAL,
    ActionGuard,
    InputGuard,
    OutputGuard,
    sanitise_tool_output,
)
from agentmon.agent.prompts import system_prompt
from agentmon.agent.tools import ToolArgsError, ToolContext, ToolRegistry, tool_specs
from agentmon.clock import Clock
from agentmon.config import Settings
from agentmon.tracing import Tracing

log = logging.getLogger("agentmon.agent")
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")

STEP_LIMIT_ANSWER = "I couldn't finish that request safely. A colleague will follow up."
LOOP_REPEAT_LIMIT = 3


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    request_id: str
    prompt_version: str
    steps: int
    stop: bool
    blocked: bool
    final: str
    flags: Annotated[list[str], operator.add]


def _last_human(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


class AgentGraph:
    def __init__(
        self,
        settings: Settings,
        model: BaseChatModel,
        registry: ToolRegistry,
        tracing: Tracing,
        clock: Clock,
    ) -> None:
        self.settings = settings
        self.model = model.bind_tools(tool_specs())
        self.model_name = getattr(model, "model_name", None) or settings.model_name
        self.registry = registry
        self.tracing = tracing
        self.clock = clock
        self.input_guard = InputGuard()
        self.action_guard = ActionGuard()
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def input_guard_node(self, state: AgentState) -> dict[str, Any]:
        text = _last_human(state["messages"])
        with self.tracing.span(
            "guard.input", **{"guard.enabled": self.settings.guardrails_enabled}
        ) as sp:
            if not self.settings.guardrails_enabled:
                return {}
            decision = self.input_guard.check(text)
            Tracing.set(
                sp, **{"guard.allowed": decision.allowed, "guard.category": decision.category}
            )
            if decision.allowed:
                return {}
            return {
                "blocked": True,
                "final": REFUSAL,
                "flags": [f"blocked:{decision.category}"],
                "messages": [AIMessage(content=REFUSAL)],
            }

    def agent_node(self, state: AgentState) -> dict[str, Any]:
        msgs = [SystemMessage(content=system_prompt(state["prompt_version"])), *state["messages"]]
        with self.tracing.span(
            "llm.chat",
            **{
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self.model_name,
                "gen_ai.system": self.settings.llm_provider,
                "agent.step": state.get("steps", 0),
            },
        ) as sp:
            reply = self.model.invoke(msgs)
            usage = getattr(reply, "usage_metadata", None) or {}
            Tracing.set(
                sp,
                **{
                    "gen_ai.usage.input_tokens": usage.get("input_tokens", 0),
                    "gen_ai.usage.output_tokens": usage.get("output_tokens", 0),
                    "gen_ai.response.tool_calls": len(getattr(reply, "tool_calls", []) or []),
                },
            )
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def tools_node(self, state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        last = messages[-1]
        assert isinstance(last, AIMessage)
        user_text = _last_human(messages)
        ctx = ToolContext(user_id=state["user_id"], request_id=state["request_id"])
        history = [
            (tc["name"], json.dumps(tc["args"], sort_keys=True))
            for m in messages[:-1]
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
        ]
        out: list[ToolMessage] = []
        flags: list[str] = []
        stop = False
        for tc in last.tool_calls:
            key = (tc["name"], json.dumps(tc["args"], sort_keys=True))
            with self.tracing.span(
                "tool.call",
                **{
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tc["name"],
                    "gen_ai.tool.call.id": tc["id"],
                    "tool.args": tc["args"],
                },
            ) as sp:
                status, payload = self._execute(tc, ctx, user_text, history, key, flags)
                if payload.get("error") == "loop_detected":
                    stop = True
                content = json.dumps(payload, default=str)
                if status == "success" and self.settings.guardrails_enabled:
                    content, modified = sanitise_tool_output(content)
                    if modified:
                        flags.append("sanitised:tool_output_injection")
                        Tracing.set(sp, **{"guard.sanitised": True})
                Tracing.set(sp, **{"tool.status": status, "tool.output_chars": len(content)})
            history.append(key)
            out.append(
                ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"], status=status)
            )
        return {"messages": out, "flags": flags, "stop": stop}

    def _execute(
        self,
        tc: dict[str, Any],
        ctx: ToolContext,
        user_text: str,
        history: list[tuple[str, str]],
        key: tuple[str, str],
        flags: list[str],
    ) -> tuple[Literal["success", "error"], dict[str, Any]]:
        if history.count(key) >= LOOP_REPEAT_LIMIT - 1:
            flags.append("loop_detected")
            return "error", {"error": "loop_detected", "detail": "identical call repeated"}
        if self.settings.guardrails_enabled:
            decision = self.action_guard.check(tc["name"], tc["args"], user_text)
            if not decision.allowed:
                flags.append(f"blocked:{decision.category}")
                return "error", {"error": "blocked_by_policy", "detail": decision.reason}
        try:
            return "success", self._run_with_retry(tc["name"], tc["args"], ctx)
        except ToolArgsError as exc:
            return "error", {"error": exc.code, "detail": str(exc)}
        except BackendError as exc:
            if exc.transient:
                flags.append("tool_error:timeout")
            return "error", {"error": exc.code, "detail": str(exc)}

    def _run_with_retry(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        attempts = self.settings.tool_max_retries + 1
        for attempt in range(attempts):
            future = _POOL.submit(self.registry.run, name, args, ctx)
            try:
                return future.result(timeout=self.settings.tool_timeout_s)
            except (BackendTimeout, FutureTimeout) as exc:
                log.warning("tool.retry", extra={"tool": name, "attempt": attempt + 1})
                if attempt == attempts - 1:
                    if isinstance(exc, FutureTimeout):
                        raise BackendTimeout(
                            f"{name} exceeded {self.settings.tool_timeout_s}s"
                        ) from exc
                    raise
                self.clock.sleep(self.settings.tool_backoff_base_s * (2**attempt))
        raise BackendTimeout(name)  # unreachable, keeps type checkers honest

    def output_guard_node(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        flags: list[str] = []
        if isinstance(last, AIMessage) and last.tool_calls:
            text = STEP_LIMIT_ANSWER
            flags.append("step_limit")
        else:
            text = str(last.content)
        with self.tracing.span(
            "guard.output", **{"guard.enabled": self.settings.guardrails_enabled}
        ):
            if self.settings.guardrails_enabled:
                text, out_flags = OutputGuard().apply(text)
                flags.extend(out_flags)
        return {"final": text, "flags": flags}

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _after_input(state: AgentState) -> str:
        return END if state.get("blocked") else "agent"

    def _after_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.get("steps", 0) >= self.settings.max_agent_steps:
                return "output_guard"
            return "tools"
        return "output_guard"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("input_guard", self.input_guard_node)
        g.add_node("agent", self.agent_node)
        g.add_node("tools", self.tools_node)
        g.add_node("output_guard", self.output_guard_node)
        g.add_edge(START, "input_guard")
        g.add_conditional_edges("input_guard", self._after_input, ["agent", END])
        g.add_conditional_edges("agent", self._after_agent, ["tools", "output_guard"])
        g.add_edge("tools", "agent")
        g.add_edge("output_guard", END)
        return g.compile()

    def invoke(
        self, user_id: str, request_id: str, message: str, prompt_version: str
    ) -> AgentState:
        limit = 2 * self.settings.max_agent_steps + 6
        return self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "request_id": request_id,
                "prompt_version": prompt_version,
                "steps": 0,
                "flags": [],
            },
            config={
                "recursion_limit": limit,
                "run_name": "banking-agent",
                "metadata": {"prompt_version": prompt_version, "request_id": request_id},
                "tags": [f"prompt:{prompt_version}"],
            },
        )
