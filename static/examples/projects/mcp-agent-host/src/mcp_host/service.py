"""ChatService: runs turns on the graph and turns them into a stream of UI events.

The API and the CLI are thin shells over this class, so both front ends get the same
behaviour: token streaming, tool events, approval pauses, per-thread locking and
conversation history loaded from the checkpointer.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from opentelemetry import trace

from mcp_host.agent import ApprovalDecision
from mcp_host.host import McpHost

log = logging.getLogger(__name__)
Event = dict[str, Any]


class ThreadBusy(Exception):
    """A second request arrived for a thread that is mid-turn."""


class ChatService:
    def __init__(
        self, host: McpHost, graph: CompiledStateGraph, checkpointer: BaseCheckpointSaver
    ) -> None:
        self.host = host
        self.graph = graph
        self.checkpointer = checkpointer
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 4 * self.host.settings.max_agent_steps + 5,
            "run_name": "mcp-agent-turn",
            "metadata": {"thread_id": thread_id},
        }

    def is_busy(self, thread_id: str) -> bool:
        return self._locks[thread_id].locked()

    async def send(self, thread_id: str, text: str) -> AsyncIterator[Event]:
        async for event in self.send_messages(thread_id, [HumanMessage(content=text)]):
            yield event

    async def send_prompt(
        self, thread_id: str, server: str, name: str, arguments: dict[str, str]
    ) -> AsyncIterator[Event]:
        messages = await self.host.get_prompt(server, name, arguments)
        async for event in self.send_messages(thread_id, list(messages)):
            yield event

    async def send_messages(
        self, thread_id: str, messages: list[BaseMessage]
    ) -> AsyncIterator[Event]:
        if await self.pending_approval(thread_id):
            yield {"type": "error", "message": "this thread is waiting for an approval decision"}
            return
        # A new user turn resets the per-turn taint: the user has spoken again.
        start = {"messages": messages, "tainted": [], "suspect_tools": [], "steps": 0}
        async for event in self._run(thread_id, start):
            yield event

    async def resume(self, thread_id: str, decision: ApprovalDecision) -> AsyncIterator[Event]:
        if not await self.pending_approval(thread_id):
            yield {"type": "error", "message": "nothing is waiting for approval"}
            return
        async for event in self._run(thread_id, Command(resume=decision.model_dump())):
            yield event

    async def _run(self, thread_id: str, graph_input: Any) -> AsyncIterator[Event]:
        lock = self._locks[thread_id]
        if lock.locked():
            raise ThreadBusy(thread_id)
        async with lock:
            tracer = trace.get_tracer("mcp_host")
            with tracer.start_as_current_span("agent.turn") as span:
                span.set_attribute("thread.id", thread_id)
                try:
                    async for event in self._stream(thread_id, graph_input):
                        yield event
                except Exception as exc:
                    log.exception("turn failed", extra={"thread_id": thread_id})
                    span.set_attribute("error.type", type(exc).__name__)
                    yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            yield {"type": "done", "thread_id": thread_id}

    async def _stream(self, thread_id: str, graph_input: Any) -> AsyncIterator[Event]:
        async for mode, chunk in self.graph.astream(
            graph_input, self._config(thread_id), stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                message, meta = chunk
                if (
                    meta.get("langgraph_node") == "agent"
                    and isinstance(message, AIMessageChunk)
                    and isinstance(message.content, str)
                    and message.content
                ):
                    yield {"type": "token", "text": message.content}
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    for item in update:
                        yield {"type": "approval_required", **item.value}
                elif node == "agent":
                    for msg in update["messages"]:
                        if msg.tool_calls:
                            for call in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "id": call["id"],
                                    "tool": call["name"],
                                    "args": call["args"],
                                }
                        else:
                            yield {"type": "message", "text": msg.content}
                elif node == "tools":
                    for msg in update["messages"]:
                        art = msg.artifact or {}
                        yield {
                            "type": "tool_result",
                            "id": msg.tool_call_id,
                            "tool": msg.name,
                            "status": msg.status,
                            "flagged": art.get("flagged", False),
                            "truncated": art.get("truncated", False),
                            "preview": str(msg.content)[:300],
                        }

    async def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        for interrupt_ in state.interrupts:
            return dict(interrupt_.value)
        return None

    async def history(self, thread_id: str) -> list[dict[str, Any]]:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        out = []
        for msg in state.values.get("messages", []):
            item: dict[str, Any] = {"role": msg.type, "content": msg.content}
            if isinstance(msg, AIMessage) and msg.tool_calls:
                item["tool_calls"] = [
                    {"name": c["name"], "args": c["args"]} for c in msg.tool_calls
                ]
            if isinstance(msg, ToolMessage):
                item["name"], item["status"] = msg.name, msg.status
            out.append(item)
        return out

    async def threads(self, limit: int = 50) -> list[str]:
        seen: list[str] = []
        async for tup in self.checkpointer.alist(None, limit=500):
            tid = tup.config["configurable"]["thread_id"]
            if tid not in seen:
                seen.append(tid)
            if len(seen) >= limit:
                break
        return seen
