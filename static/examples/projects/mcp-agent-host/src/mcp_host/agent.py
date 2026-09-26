"""The LangGraph agent: agent -> guard -> tools -> agent.

``guard`` is the policy enforcement point. It runs *between* the model proposing tool
calls and the host executing them, so no model output can reach a server without
passing it. It blocks unknown and denied tools, blocks cross-server calls made after
untrusted output was flagged, and pauses on ``interrupt()`` for destructive tools.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from mcp_host.host import McpHost
from mcp_host.registry import Decision, ToolRegistry
from mcp_host.safety import SPOTLIGHT_RULES

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    verdicts: dict[str, str]  # tool_call_id -> "allow" | reason it was blocked
    tainted: list[str]  # servers whose output was flagged during this turn
    suspect_tools: list[str]  # tools named inside flagged output during this turn
    steps: int


class ApprovalDecision(BaseModel):
    """What a human sends back to resume an approval interrupt."""

    approve: list[str] = Field(default_factory=list, description="tool_call ids to run")
    note: str | None = None


SYSTEM_TEMPLATE = """You are a work assistant with tools from several MCP servers.
Tool names are namespaced as <server>__<tool>. Today is {today} (UTC).

{spotlight}

{availability}
Destructive tools (deleting notes, cancelling events) need the user's approval; the
host will ask them. Use host__read_resource for full documents. Be concise and cite
document slugs like [leave-policy] when you answer from company docs."""


def system_prompt(registry: ToolRegistry) -> SystemMessage:
    return SystemMessage(
        content=SYSTEM_TEMPLATE.format(
            today=datetime.now(UTC).date().isoformat(),
            spotlight=SPOTLIGHT_RULES,
            availability=registry.availability_note(),
        )
    )


def build_graph(
    host: McpHost,
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    registry = host.registry
    max_steps = host.settings.max_agent_steps

    async def agent(state: AgentState) -> dict[str, Any]:
        steps = state.get("steps", 0)
        if steps >= max_steps:
            return {"messages": [AIMessage(content=f"I stopped after {max_steps} tool rounds.")]}
        system = system_prompt(registry)
        # Re-bind every turn: the catalogue can change between turns (list_changed,
        # a server going down) and the model must only see what exists right now.
        model = llm.bind_tools(registry.openai_tools())
        reply = await model.ainvoke([system, *state["messages"]])
        return {"messages": [reply]}

    def guard(state: AgentState) -> dict[str, Any]:
        # This node may run twice (interrupt() re-executes it on resume), so it must
        # be pure: decide, never act.
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        tainted = set(state.get("tainted", []))
        suspects = set(state.get("suspect_tools", []))
        verdicts: dict[str, str] = {}
        pending: list[dict[str, Any]] = []
        for call in last.tool_calls:
            cid, name = call["id"] or "", call["name"]
            entry = registry.resolve(name)
            if entry is None:
                verdicts[cid] = f"unknown or unavailable tool {name!r}"
                continue
            decision = registry.decide(entry)
            target = host.effective_server(entry, call["args"])
            if decision == Decision.DENY:
                verdicts[cid] = f"tool {name!r} is denied by host policy"
            elif name in suspects or (tainted - {target}):
                verdicts[cid] = (
                    f"blocked: untrusted output from {sorted(tainted) or 'a server'} "
                    f"cannot trigger a call to {target!r}. Ask the user to confirm in a "
                    "new message if this action is really wanted."
                )
            elif decision == Decision.APPROVE:
                pending.append(
                    {"id": cid, "tool": name, "server": entry.server, "args": call["args"]}
                )
            else:
                verdicts[cid] = "allow"
        if pending:
            answer = interrupt(
                {"kind": "approval_required", "calls": pending},
                response_schema=ApprovalDecision,
            )
            approved = set(answer.approve)
            for item in pending:
                verdicts[item["id"]] = "allow" if item["id"] in approved else "rejected by the user"
            log.info("approval resolved", extra={"approved": sorted(approved)})
        return {"verdicts": verdicts}

    async def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        verdicts = state.get("verdicts", {})
        tainted = set(state.get("tainted", []))
        suspects = set(state.get("suspect_tools", []))

        async def run(call: dict[str, Any]) -> ToolMessage:
            cid, name = call["id"] or "", call["name"]
            verdict = verdicts.get(cid, "no verdict")
            if verdict != "allow":
                return ToolMessage(
                    content=f"NOT EXECUTED: {verdict}", tool_call_id=cid, name=name, status="error"
                )
            entry = registry.resolve(name)
            if entry is None:  # vanished between guard and tools (list_changed, crash)
                return ToolMessage(
                    content=f"NOT EXECUTED: {name!r} is no longer available",
                    tool_call_id=cid,
                    name=name,
                    status="error",
                )
            outcome = await host.call(entry, dict(call["args"]))
            if outcome.flagged:
                tainted.add(outcome.server)
                suspects.update(outcome.mentioned_tools)
                log.warning(
                    "tool output flagged",
                    extra={
                        "tool": name,
                        "reasons": outcome.reasons,
                        "mentions": outcome.mentioned_tools,
                    },
                )
            return ToolMessage(
                content=outcome.content,
                tool_call_id=cid,
                name=name,
                status="error" if outcome.is_error else "success",
                artifact={
                    "server": outcome.server,
                    "flagged": outcome.flagged,
                    "truncated": outcome.truncated,
                },
            )

        # Calls in one step are independent by construction; run them concurrently.
        results = await asyncio.gather(*(run(c) for c in last.tool_calls))
        return {
            "messages": list(results),
            "tainted": sorted(tainted),
            "suspect_tools": sorted(suspects),
            "steps": state.get("steps", 0) + 1,
        }

    def route(state: AgentState) -> Literal["guard", "__end__"]:
        last = state["messages"][-1]
        return "guard" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("guard", guard)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("guard", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
