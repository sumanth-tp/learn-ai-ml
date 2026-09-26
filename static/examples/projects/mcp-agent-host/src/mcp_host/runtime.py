"""Wiring: settings -> LLM -> host -> checkpointer -> graph -> chat service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mcp_host.agent import build_graph
from mcp_host.host import McpHost
from mcp_host.llm import build_chat_model
from mcp_host.service import ChatService
from mcp_host.settings import ServersFile, Settings, load_servers
from mcp_host.tracing import configure_tracing
from mcp_host.transports import TransportFactory


@asynccontextmanager
async def open_checkpointer(target: str) -> AsyncIterator[BaseCheckpointSaver]:
    """``:memory:`` for tests; otherwise a SQLite file that survives restarts."""
    if target == ":memory:":
        yield InMemorySaver()
        return
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(target) as saver:
        await saver.setup()
        yield saver


@dataclass
class Runtime:
    settings: Settings
    host: McpHost
    service: ChatService


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    servers: ServersFile | None = None,
    llm: BaseChatModel | None = None,
    factories: Mapping[str, TransportFactory] | None = None,
) -> AsyncIterator[Runtime]:
    configure_tracing(settings.otel_exporter, settings.service_name)
    model = llm or build_chat_model(settings)
    host = McpHost(settings, servers or load_servers(settings.servers_file), model, factories)
    async with open_checkpointer(settings.checkpoint_db) as saver, host:
        graph = build_graph(host, model, saver)
        yield Runtime(settings=settings, host=host, service=ChatService(host, graph, saver))
