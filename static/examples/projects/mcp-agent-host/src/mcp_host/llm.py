"""Chat-model factory plus deterministic fakes that support tool calling.

LangChain's ``GenericFakeChatModel`` cannot ``bind_tools``, and a tool-calling agent
is useless without it. ``FakeToolModel`` is a real ``BaseChatModel`` whose replies come
from a *responder*: a script (tests) or a keyword router (offline demo and eval
baseline). Both see the same messages and tool schemas the real model would.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field

from mcp_host.settings import Settings

log = logging.getLogger(__name__)
Responder = Callable[[list[BaseMessage], list[dict[str, Any]]], AIMessage]


class FakeToolModel(BaseChatModel):
    """A chat model driven by a Python function. Supports ``bind_tools`` and streaming."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    responder: Any
    tools: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str = "fake-tool-model"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.model_copy(update={"tools": [t for t in tools if isinstance(t, dict)]})

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        reply = self.responder(messages, self.tools)
        for call in reply.tool_calls:
            call.setdefault("id", f"call_{uuid.uuid4().hex[:8]}")
        return reply

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Native async: the responder runs on the event loop, not in a worker thread,
        # so scripted steps in tests can touch asyncio objects safely.
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    @staticmethod
    def _chunks(reply: AIMessage) -> Iterator[ChatGenerationChunk]:
        if reply.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=str(reply.content),
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": json.dumps(c["args"]),
                            "id": c["id"],
                            "index": i,
                        }
                        for i, c in enumerate(reply.tool_calls)
                    ],
                )
            )
            return
        for word in re.split(r"(\s+)", str(reply.content)):
            if word:
                yield ChatGenerationChunk(message=AIMessageChunk(content=word))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._chunks(self._reply(messages)):
            if run_manager and chunk.text:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._chunks(self._reply(messages)):
            if run_manager and chunk.text:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


class ScriptedResponder:
    """Replays a list of replies in order. A callable entry is called with the messages."""

    def __init__(self, script: Sequence[AIMessage | Callable[[list[BaseMessage]], AIMessage]]):
        self.script = list(script)
        self.calls: list[list[BaseMessage]] = []

    def __call__(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        self.calls.append(list(messages))
        if not self.script:
            return AIMessage(content="(script exhausted)")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


# ---------------------------------------------------------------- keyword router
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
NOTE_RE = re.compile(r"\bnote[s]?\s+(?:called\s+|named\s+)?[\"']?([a-z0-9][a-z0-9_-]*)", re.I)
TAG_RE = re.compile(r"<tool_output[^>]*>\n?(.*?)\n?</tool_output[^>]*>", re.S)

# (keywords, tool suffix, args builder). First rule with any keyword hit wins, so the
# order encodes priority: specific verbs before generic "search".
RULES: list[tuple[tuple[str, ...], str, Callable[[str], dict[str, Any]]]] = [
    (
        ("summarise my day", "summarize my day", "summary of", "summarise the day"),
        "calendar__summarise_day",
        lambda q: {"day": _date(q)},
    ),
    (
        ("free slot", "free time", "when am i free", "find a slot"),
        "calendar__find_free_slot",
        lambda q: {"day": _date(q), "duration_minutes": 30},
    ),
    (("cancel",), "calendar__cancel_event", lambda q: {"event_id": _word_after(q, "event")}),
    (
        ("schedule", "book a meeting", "create an event", "set up a meeting"),
        "calendar__create_event",
        lambda q: {
            "title": "Meeting",
            "start": f"{_date(q)}T15:00:00+00:00",
            "idempotency_key": f"router-{abs(hash(q)) % 10**8}",
        },
    ),
    (
        ("calendar", "meetings", "events", "agenda"),
        "calendar__list_events",
        lambda q: {"day": _date(q)},
    ),
    (
        ("delete note", "remove note", "delete the note", "remove the note"),
        "notes__delete_note",
        lambda q: {"name": _note(q)},
    ),
    (
        ("write a note", "save a note", "take a note", "write note", "save note"),
        "notes__write_note",
        lambda q: {"name": _note(q), "content": q},
    ),
    (("tag",), "notes__enable_tag_tools", lambda q: {}),
    (
        ("list my notes", "what notes", "which notes", "all notes"),
        "notes__list_notes",
        lambda q: {},
    ),
    (
        ("read note", "read my note", "open note", "show note", "read the note"),
        "notes__read_note",
        lambda q: {"name": _note(q)},
    ),
    (("in my notes", "my notes"), "notes__search", lambda q: {"query": _topic(q)}),
    (
        (
            "policy",
            "handbook",
            "allowance",
            "leave",
            "expense",
            "on-call",
            "vendor",
            "security",
            "company",
            "how many days",
            "rule",
        ),
        "docs__search",
        lambda q: {"query": q, "k": 3},
    ),
]


def _date(q: str) -> str:
    m = DATE_RE.search(q)
    return m.group(1) if m else "2026-10-05"


def _note(q: str) -> str:
    m = NOTE_RE.search(q)
    return m.group(1).lower() if m else "scratch"


def _word_after(q: str, word: str) -> str:
    m = re.search(rf"\b{word}\s+([A-Za-z0-9_]+)", q)
    return m.group(1) if m else "unknown"


def _topic(q: str) -> str:
    m = re.search(r"\babout\s+(.+?)(?:\s+in my notes|\?|$)", q, re.I)
    return (m.group(1) if m else q).strip()


def keyword_router(messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
    """Deterministic stand-in for an LLM: route by keywords, then summarise tool output."""
    available = {t["function"]["name"] for t in tools if "function" in t}
    last = messages[-1] if messages else HumanMessage(content="")
    if isinstance(last, ToolMessage):
        outputs = []
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                break
            found = TAG_RE.search(str(msg.content))
            body = found.group(1) if found else str(msg.content)
            body = re.sub(r"^\[host notice:[^\]]*\]\n", "", body)
            outputs.append(body.strip()[:400])
        return AIMessage(content="Here is what I found:\n" + "\n---\n".join(reversed(outputs)))
    text = str(last.content)
    if not tools:  # sampling or plain chat: a short extractive "summary"
        lines = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
        body = "; ".join(lines[1:] or lines)[:300]
        return AIMessage(content=f"Summary: {body}")
    lowered = text.lower()
    for keywords, tool, build in RULES:
        if tool in available and any(k in lowered for k in keywords):
            return AIMessage(
                content="", tool_calls=[{"name": tool, "args": build(text), "id": None}]
            )
    return AIMessage(content="I can help with your notes, calendar and the company docs.")


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Return the configured model. ``auto`` means OpenAI when a key exists, else the fake."""
    provider = settings.llm_provider
    if provider == "auto":
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "fake"
        if provider == "fake":
            log.warning("OPENAI_API_KEY not set: using the offline keyword-router model")
    if provider == "fake":
        return FakeToolModel(responder=keyword_router)
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model,
        model_provider=provider,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def to_langchain(messages: Sequence[Any], system_prompt: str | None) -> list[BaseMessage]:
    """Convert MCP ``SamplingMessage`` objects to LangChain messages (text only)."""
    from mcp import types

    out: list[BaseMessage] = [SystemMessage(content=system_prompt)] if system_prompt else []
    for message in messages:
        texts = [b.text for b in message.content_as_list if isinstance(b, types.TextContent)]
        cls = HumanMessage if message.role == "user" else AIMessage
        out.append(cls(content="\n".join(texts)))
    return out
