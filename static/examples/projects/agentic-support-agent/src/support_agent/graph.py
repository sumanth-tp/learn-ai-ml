"""The support graph: guard -> memory -> intent routing -> agent/tool loop or FAQ RAG
-> human approval for large refunds -> finalise (long-term memory) -> summarise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field

from support_agent import metrics
from support_agent.config import Settings
from support_agent.guardrails import check_input, refusal_for, scrub_output
from support_agent.intent import Intent, IntentClassifier, last_human_text
from support_agent.memory import (
    extract_preferences,
    load_user_context,
    record_issue,
    save_preferences,
)
from support_agent.prompts import (
    BUDGET_MESSAGE,
    HANDOFF_MESSAGE,
    SUMMARY_PROMPT,
    render_agent_prompt,
    render_faq_prompt,
)
from support_agent.services.refunds import money
from support_agent.state import RESET, Context, SupportState
from support_agent.tools import TOOLS_BY_INTENT, ToolServices, build_tool_node, build_tools

log = logging.getLogger(__name__)

AGENT_INTENTS = {Intent.ORDER_STATUS, Intent.RETURNS, Intent.REFUND}
STREAMED_NODES = frozenset({"agent", "faq_answer"})


class ApprovalDecision(BaseModel):
    """What a reviewer sends back to resume a paused refund."""

    approved: bool
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=500)


@dataclass
class GraphDeps:
    settings: Settings
    model: BaseChatModel
    classifier: IntentClassifier
    tools: ToolServices


def is_transient_llm_error(exc: BaseException) -> bool:
    names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
    }
    return isinstance(exc, TimeoutError | ConnectionError) or type(exc).__name__ in names


def _usage(msg: AnyMessage) -> int:
    usage = getattr(msg, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0))


def _turn_start(messages: list[AnyMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def build_graph(
    deps: GraphDeps,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    settings = deps.settings
    threshold = money(settings.refund_approval_threshold)
    tools = build_tools(deps.tools)
    tool_node = build_tool_node(tools, settings)
    by_name = {t.name: t for t in tools}
    # Bind once per intent: each specialist sees only its permitted tools.
    bound: dict[str, Runnable[Any, Any]] = {
        intent: deps.model.bind_tools([by_name[n] for n in sorted(names)])
        for intent, names in TOOLS_BY_INTENT.items()
    }
    llm_retry = RetryPolicy(
        max_attempts=settings.llm_node_retry_attempts,
        initial_interval=settings.llm_node_retry_initial_s,
        retry_on=is_transient_llm_error,
    )

    # ---- nodes -------------------------------------------------------------

    def guard_input(state: SupportState) -> dict[str, Any]:
        verdict = check_input(last_human_text(state["messages"]))
        if not verdict.allowed:
            metrics.GUARDRAIL_BLOCKS.labels(reason=str(verdict.reason).split(":")[0]).inc()
            log.warning("input blocked", extra={"reason": verdict.reason})
        # Every new turn starts with a fresh per-request budget.
        return {
            "steps": RESET,
            "tokens": RESET,
            "outcome": None,
            "blocked_reason": None if verdict.allowed else verdict.reason,
        }

    async def load_memory(state: SupportState, runtime: Runtime[Context]) -> dict[str, Any]:
        if runtime.store is None:
            return {"user_context": ""}
        user_id = runtime.context.user_id
        await save_preferences(
            runtime.store, user_id, extract_preferences(last_human_text(state["messages"]))
        )
        return {"user_context": await load_user_context(runtime.store, user_id)}

    async def classify_intent(state: SupportState) -> dict[str, Any]:
        previous = state.get("intent")
        decision = await deps.classifier.classify(
            state["messages"], Intent(previous) if previous else None
        )
        log.info(
            "intent", extra={"intent": decision.intent.value, "confidence": decision.confidence}
        )
        update: dict[str, Any] = {"intent": decision.intent.value}
        if decision.order_id:
            update["order_id"] = decision.order_id
        if decision.intent == Intent.OFF_TOPIC:
            update["blocked_reason"] = "off_topic"
        return update

    async def agent(state: SupportState) -> dict[str, Any]:
        intent = state.get("intent") or Intent.ORDER_STATUS.value
        system = SystemMessage(
            render_agent_prompt(
                intent, float(threshold), state.get("user_context", ""), state.get("summary", "")
            )
        )
        history = trim_messages(
            state["messages"],
            max_tokens=settings.max_context_tokens,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        if not history:  # one huge turn: keep at least the current turn
            history = state["messages"][_turn_start(state["messages"]) :]
        response = await bound[intent].ainvoke([system, *history])
        tokens = _usage(response)
        metrics.TOKENS.labels(node="agent").inc(tokens)
        return {"messages": [response], "steps": 1, "tokens": tokens}

    def needs_approval(state: SupportState) -> list[dict[str, Any]]:
        approvals = state.get("approvals") or {}
        last = state["messages"][-1]
        return [
            tc
            for tc in getattr(last, "tool_calls", []) or []
            if tc["name"] == "issue_refund"
            and money(tc["args"].get("amount", 0)) > threshold
            and tc["id"] not in approvals
        ]

    def route_after_agent(
        state: SupportState,
    ) -> Literal["tools", "human_approval", "handoff", "finalize"]:
        if tools_condition(state) == END:
            return "finalize"
        if state.get("steps", 0) >= settings.max_steps:
            metrics.BUDGET_EXCEEDED.labels(kind="steps").inc()
            return "handoff"
        if state.get("tokens", 0) >= settings.max_tokens_per_request:
            metrics.BUDGET_EXCEEDED.labels(kind="tokens").inc()
            return "handoff"
        if needs_approval(state):
            return "human_approval"
        return "tools"

    def human_approval(state: SupportState, runtime: Runtime[Context]) -> Command[Any]:
        decisions: dict[str, Any] = {}
        for tc in needs_approval(state):
            # One interrupt per refund. On resume the node re-runs from the top and
            # each interrupt() returns its resume value in order.
            decision = interrupt(
                {
                    "type": "refund_approval",
                    "tool_call_id": tc["id"],
                    "user_id": runtime.context.user_id,
                    "order_id": tc["args"].get("order_id"),
                    "amount": str(money(tc["args"].get("amount", 0))),
                    "reason": str(tc["args"].get("reason", ""))[:200],
                    "threshold": str(threshold),
                },
                response_schema=ApprovalDecision,
            )
            decisions[tc["id"]] = ApprovalDecision.model_validate(decision).model_dump()
        return Command(goto="tools", update={"approvals": decisions})

    async def faq_answer(state: SupportState) -> dict[str, Any]:
        question = last_human_text(state["messages"])
        passages = deps.tools.faq.search(question, k=3)
        response = await deps.model.ainvoke(
            [SystemMessage(render_faq_prompt(passages)), HumanMessage(question)]
        )
        tokens = _usage(response)
        metrics.TOKENS.labels(node="faq_answer").inc(tokens)
        outcome = "resolved" if passages else "no_answer"
        return {"messages": [response], "tokens": tokens, "outcome": outcome}

    def refuse(state: SupportState) -> dict[str, Any]:
        reason = state.get("blocked_reason") or "off_topic"
        out: list[AnyMessage] = []
        last = state["messages"][-1]
        if reason.startswith("prompt_injection") and isinstance(last, HumanMessage):
            # Same id => add_messages replaces it, so the attack never reaches
            # a future prompt through the conversation history.
            out.append(HumanMessage(content="[message removed by safety filter]", id=last.id))
        out.append(AIMessage(content=refusal_for(reason)))
        label = "off_topic" if reason == "off_topic" else "blocked"
        metrics.REQUESTS.labels(intent=label, outcome="refused").inc()
        return {"messages": out, "outcome": "refused"}

    def handoff(state: SupportState) -> dict[str, Any]:
        over_budget = (
            state.get("steps", 0) >= settings.max_steps
            or state.get("tokens", 0) >= settings.max_tokens_per_request
        )
        out: list[AnyMessage] = [
            # Close every open tool call, or the next model call is rejected.
            ToolMessage(
                content="Not executed: handed over to a person.",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            )
            for tc in getattr(state["messages"][-1], "tool_calls", []) or []
        ]
        out.append(AIMessage(content=BUDGET_MESSAGE if over_budget else HANDOFF_MESSAGE))
        return {"messages": out, "outcome": "budget_exceeded" if over_budget else "handoff"}

    async def finalize(state: SupportState, runtime: Runtime[Context]) -> dict[str, Any]:
        update: dict[str, Any] = {}
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            clean = scrub_output(str(last.content))
            if clean != last.content:
                update["messages"] = [AIMessage(content=clean, id=last.id)]
        outcome = state.get("outcome") or "resolved"
        update["outcome"] = outcome
        intent = state.get("intent") or "unknown"
        if runtime.store is not None and intent in {i.value for i in AGENT_INTENTS}:
            await record_issue(
                runtime.store, runtime.context.user_id, intent, state.get("order_id"), outcome
            )
        metrics.REQUESTS.labels(intent=intent, outcome=outcome).inc()
        return update

    def route_after_finalize(state: SupportState) -> Literal["summarise", "__end__"]:
        return "summarise" if len(state["messages"]) > settings.summarise_after_messages else END

    async def summarise(state: SupportState) -> dict[str, Any]:
        msgs = state["messages"]
        cut = max(0, len(msgs) - settings.keep_last_messages)
        # Move the cut back to a human message so no tool result loses its call.
        while cut > 0 and not isinstance(msgs[cut], HumanMessage):
            cut -= 1
        if cut == 0:
            return {}
        older = msgs[:cut]
        transcript = "\n".join(
            f"{'customer' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}"
            for m in older
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content)
        )
        response = await deps.model.ainvoke(
            [
                SystemMessage(SUMMARY_PROMPT.format(existing=state.get("summary") or "none")),
                HumanMessage(transcript),
            ]
        )
        return {
            "summary": str(response.content),
            "messages": [RemoveMessage(id=m.id) for m in older if m.id],
        }

    # ---- wiring ------------------------------------------------------------

    def route_after_guard(state: SupportState) -> Literal["refuse", "load_memory"]:
        return "refuse" if state.get("blocked_reason") else "load_memory"

    def route_by_intent(
        state: SupportState,
    ) -> Literal["agent", "faq_answer", "handoff", "refuse"]:
        intent = state.get("intent")
        if intent in {i.value for i in AGENT_INTENTS}:
            return "agent"
        if intent == Intent.FAQ.value:
            return "faq_answer"
        if intent == Intent.HUMAN.value:
            return "handoff"
        return "refuse"

    g = StateGraph(SupportState, context_schema=Context)
    g.add_node("guard_input", guard_input)
    g.add_node("load_memory", load_memory)
    g.add_node("classify_intent", classify_intent, retry_policy=llm_retry)
    g.add_node("agent", agent, retry_policy=llm_retry)
    g.add_node("tools", tool_node)
    g.add_node("human_approval", human_approval, destinations=("tools",))
    g.add_node("faq_answer", faq_answer, retry_policy=llm_retry)
    g.add_node("refuse", refuse)
    g.add_node("handoff", handoff)
    g.add_node("finalize", finalize)
    g.add_node("summarise", summarise, retry_policy=llm_retry)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", route_after_guard)
    g.add_edge("load_memory", "classify_intent")
    g.add_conditional_edges("classify_intent", route_by_intent)
    g.add_conditional_edges("agent", route_after_agent)
    g.add_edge("tools", "agent")
    g.add_edge("faq_answer", "finalize")
    g.add_edge("handoff", "finalize")
    g.add_edge("refuse", END)
    g.add_conditional_edges("finalize", route_after_finalize)
    g.add_edge("summarise", END)

    return g.compile(checkpointer=checkpointer, store=store)
