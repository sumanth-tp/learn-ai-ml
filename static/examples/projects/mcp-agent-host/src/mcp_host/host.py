"""The MCP host: owns every server connection, the registry, and the safe call path."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.prompts import convert_mcp_prompt_message_to_langchain_message

from mcp_host.connection import HostError, ServerConnection
from mcp_host.registry import HOST, ToolEntry, ToolRegistry
from mcp_host.safety import render_resource, render_result, scan, spotlight, truncate
from mcp_host.sampling import make_sampling_callback
from mcp_host.settings import ServersFile, Settings
from mcp_host.transports import TransportFactory, factory_for

log = logging.getLogger(__name__)


@dataclass
class ToolOutcome:
    """What the agent gets back from one tool call, plus the flags the guard needs."""

    content: str
    server: str
    is_error: bool = False
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)
    mentioned_tools: list[str] = field(default_factory=list)
    truncated: bool = False


class McpHost:
    def __init__(
        self,
        settings: Settings,
        servers: ServersFile,
        llm: BaseChatModel,
        factories: Mapping[str, TransportFactory] | None = None,
    ) -> None:
        self.settings = settings
        self.servers = {n: c for n, c in servers.servers.items() if c.enabled}
        self.llm = llm
        self.registry = ToolRegistry({n: c.policy for n, c in self.servers.items()})
        self.connections: dict[str, ServerConnection] = {}
        overrides = dict(factories or {})
        model_label = getattr(llm, "model_name", None) or settings.llm_model
        for name, cfg in self.servers.items():
            self.connections[name] = ServerConnection(
                name,
                overrides.get(name) or factory_for(cfg, settings.connect_timeout_s),
                settings,
                timeout_s=cfg.timeout_s,
                sampling_callback=make_sampling_callback(
                    name, cfg.policy, llm, settings.sampling_max_tokens, str(model_label)
                ),
                on_change=self._on_change,
            )

    def _on_change(self, _server: str) -> None:
        snap = self.registry.rebuild(self.connections)
        log.debug("registry rebuilt", extra={"version": snap.version, "tools": len(snap.tools)})

    async def start(self) -> None:
        """Connect to every server in parallel; do not let one slow server block start-up."""
        for conn in self.connections.values():
            conn.start()
        ready = await asyncio.gather(
            *(c.wait_ready(self.settings.connect_timeout_s) for c in self.connections.values())
        )
        self.registry.rebuild(self.connections)
        down = [n for n, ok in zip(self.connections, ready, strict=True) if not ok]
        log.info(
            "host started",
            extra={"ready": [n for n in self.connections if n not in down], "degraded": down},
        )

    async def stop(self) -> None:
        await asyncio.gather(*(c.stop() for c in self.connections.values()))

    async def __aenter__(self) -> McpHost:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---------------------------------------------------------------- calling
    def _limit(self, server: str) -> int:
        cfg = self.servers.get(server)
        return (cfg.max_output_chars if cfg else None) or self.settings.max_tool_output_chars

    def _package(self, source: str, server: str, text: str, is_error: bool) -> ToolOutcome:
        text, cut = truncate(text, self._limit(server))
        result = scan(text, set(self.registry.snapshot.tools) | self._offline_tools())
        return ToolOutcome(
            content=spotlight(source, text, result),
            server=server,
            is_error=is_error,
            flagged=result.flagged,
            reasons=result.reasons,
            mentioned_tools=result.mentioned_tools,
            truncated=cut,
        )

    def _offline_tools(self) -> set[str]:
        return {t for tools in self.registry.snapshot.missing_tools.values() for t in tools}

    def effective_server(self, entry: ToolEntry, args: Mapping[str, Any]) -> str:
        """The server a call really touches. ``host__read_resource`` inherits the URI's."""
        if entry.server == HOST and entry.tool == "read_resource":
            return self.registry.server_for_uri(str(args.get("uri", ""))) or HOST
        return entry.server

    async def call(self, entry: ToolEntry, args: dict[str, Any]) -> ToolOutcome:
        if entry.server == HOST:
            return await self._host_tool(entry, args)
        conn = self.connections[entry.server]
        source = f"{entry.server}.{entry.tool}"
        try:
            result = await conn.call_tool(entry.tool, args)
        except HostError as exc:
            return ToolOutcome(content=f"ERROR: {exc}", server=entry.server, is_error=True)
        return self._package(source, entry.server, render_result(result), result.isError)

    async def _host_tool(self, entry: ToolEntry, args: dict[str, Any]) -> ToolOutcome:
        if entry.tool == "list_resources":
            listing = {
                name: [str(r.uri) for r in c.catalogue.resources]
                + [t.uriTemplate for t in c.catalogue.resource_templates]
                for name, c in self.connections.items()
                if name not in self.registry.snapshot.unavailable
            }
            return ToolOutcome(content=json.dumps(listing), server=HOST)
        uri = str(args.get("uri", ""))
        server = self.registry.server_for_uri(uri)
        if server is None:
            return ToolOutcome(
                content=f"ERROR: no server serves {uri!r}", server=HOST, is_error=True
            )
        try:
            result = await self.connections[server].read_resource(uri)
        except HostError as exc:
            return ToolOutcome(content=f"ERROR: {exc}", server=server, is_error=True)
        return self._package(f"{server}:{uri}", server, render_resource(result), False)

    # ---------------------------------------------------------------- prompts
    def list_prompts(self) -> list[dict[str, Any]]:
        return [
            {
                "server": name,
                "name": p.name,
                "description": p.description,
                "arguments": [a.name for a in (p.arguments or [])],
            }
            for name, conn in self.connections.items()
            for p in conn.catalogue.prompts
        ]

    async def get_prompt(
        self, server: str, name: str, arguments: dict[str, str]
    ) -> list[BaseMessage]:
        """Fetch a server prompt as LangChain messages, ready to start a turn with."""
        if server not in self.connections:
            raise HostError(f"unknown server {server!r}")
        result = await self.connections[server].get_prompt(name, arguments)
        return [convert_mcp_prompt_message_to_langchain_message(m) for m in result.messages]

    def status(self) -> dict[str, Any]:
        snap = self.registry.snapshot
        return {
            "servers": [c.status() for c in self.connections.values()],
            "tools": sorted(snap.tools),
            "unavailable": snap.unavailable,
            "collisions": snap.collisions,
            "registry_version": snap.version,
        }
