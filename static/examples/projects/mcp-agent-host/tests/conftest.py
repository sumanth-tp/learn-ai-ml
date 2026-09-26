"""Shared fixtures: in-process MCP servers, fast settings, a scripted fake LLM."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from demo_servers import calendar_server, docs_server, notes_server
from mcp_host.llm import FakeToolModel, ScriptedResponder
from mcp_host.runtime import Runtime, open_runtime
from mcp_host.settings import ServersFile, Settings
from mcp_host.tracing import configure_tracing
from mcp_host.transports import InProcessServer

ROOT = Path(__file__).resolve().parents[1]
SPANS = InMemorySpanExporter()
configure_tracing("none", "mcp-agent-host-tests", extra=SPANS)


def call(name: str, args: dict[str, Any] | None = None, cid: str | None = None) -> AIMessage:
    """A model reply that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": cid or name}])


def servers_file(**policy_overrides: dict[str, Any]) -> ServersFile:
    raw: dict[str, Any] = {
        "servers": {
            "notes": {
                "connection": {"transport": "stdio", "command": "python", "args": []},
                "policy": {"destructive": ["delete_note"]},
            },
            "calendar": {
                "connection": {"transport": "http", "url": "http://calendar.invalid/mcp"},
                "policy": {"allow_sampling": True},
                "timeout_s": 2,
            },
            "docs": {
                "connection": {"transport": "http", "url": "http://docs.invalid/mcp"},
                "max_output_chars": 1500,
            },
        }
    }
    for name, policy in policy_overrides.items():
        raw["servers"][name]["policy"] = {**raw["servers"][name].get("policy", {}), **policy}
    return ServersFile.model_validate(raw)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        checkpoint_db=":memory:",
        connect_timeout_s=3,
        call_timeout_s=2,
        ping_interval_s=0.2,
        ping_timeout_s=0.5,
        backoff_initial_s=0.05,
        backoff_max_s=0.2,
        max_tool_output_chars=4000,
        log_json=False,
    )


@pytest.fixture
def data_dirs(tmp_path: Path) -> dict[str, Path]:
    notes = tmp_path / "notes"
    shutil.copytree(ROOT / "data" / "notes", notes)
    return {"notes": notes, "docs": ROOT / "data" / "docs", "calendar": tmp_path / "calendar.json"}


@pytest.fixture
def inproc(data_dirs: dict[str, Path]) -> dict[str, InProcessServer]:
    return {
        "notes": InProcessServer(notes_server.create_server(data_dirs["notes"])),
        "calendar": InProcessServer(
            calendar_server.create_server(
                data_dirs["calendar"], ROOT / "data" / "calendar_seed.json"
            )
        ),
        "docs": InProcessServer(docs_server.create_server(data_dirs["docs"])),
    }


RuntimeFactory = Callable[..., AbstractAsyncContextManager[Runtime]]


@pytest.fixture
def make_runtime(settings: Settings, inproc: dict[str, InProcessServer]) -> RuntimeFactory:
    @asynccontextmanager
    async def factory(
        script: Sequence[AIMessage | Callable[[list[BaseMessage]], AIMessage]] = (),
        *,
        servers: ServersFile | None = None,
        responder: Any = None,
        extra: dict[str, InProcessServer] | None = None,
        **overrides: Any,
    ) -> AsyncIterator[Runtime]:
        cfg = settings.model_copy(update=overrides)
        llm = FakeToolModel(responder=responder or ScriptedResponder(script))
        all_servers = {**inproc, **(extra or {})}
        async with open_runtime(
            cfg,
            servers=servers or servers_file(),
            llm=llm,
            factories={n: s.connect for n, s in all_servers.items()},
        ) as rt:
            yield rt

    return factory


async def collect(stream: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event async for event in stream]


@pytest.fixture
def spans() -> InMemorySpanExporter:
    SPANS.clear()
    return SPANS
