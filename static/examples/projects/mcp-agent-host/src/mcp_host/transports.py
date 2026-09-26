"""Transport factories: each returns an async context manager yielding (read, write) streams.

The connection supervisor does not care how bytes move. It asks a factory for a fresh
pair of streams on every (re)connect, which is what makes reconnection one code path
for stdio subprocesses, remote HTTP servers and the in-process servers used in tests.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.message import SessionMessage

from mcp_host.settings import HttpServer, ServerConfig, StdioServer

Streams = tuple[
    MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]
]
TransportFactory = Callable[[], AbstractAsyncContextManager[Streams]]


def stdio_factory(cfg: StdioServer) -> TransportFactory:
    # "python" means "the interpreter running the host", so a uv venv just works.
    command = sys.executable if cfg.command in {"python", "python3"} else cfg.command
    params = StdioServerParameters(
        command=command,
        args=cfg.args,
        # Pass PATH and friends through, then the server's own variables.
        env={**{k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG"}}, **cfg.env},
        cwd=cfg.cwd,
    )

    @asynccontextmanager
    async def connect() -> AsyncIterator[Streams]:
        async with stdio_client(params) as (read, write):
            yield read, write

    return connect


def http_factory(cfg: HttpServer, connect_timeout_s: float) -> TransportFactory:
    @asynccontextmanager
    async def connect() -> AsyncIterator[Streams]:
        # The SDK's old ``streamablehttp_client(url, headers=...)`` is deprecated; the
        # current API takes a configured httpx client. Reads stay open for SSE streams.
        timeout = httpx.Timeout(connect_timeout_s, read=300.0)
        async with (
            httpx.AsyncClient(headers=cfg.headers, timeout=timeout) as client,
            streamable_http_client(cfg.url, http_client=client) as (read, write, _session_id),
        ):
            yield read, write

    return connect


def factory_for(cfg: ServerConfig, connect_timeout_s: float) -> TransportFactory:
    conn = cfg.connection
    if isinstance(conn, StdioServer):
        return stdio_factory(conn)
    return http_factory(conn, connect_timeout_s)


class InProcessServer:
    """Runs a FastMCP server in the host's event loop over memory streams.

    Used by the tests (and handy in notebooks). ``crash()`` kills every live session the
    way a dying process would: the server stops and its side of the pipe closes.
    ``revive()`` lets new connections succeed again.
    """

    def __init__(self, server: FastMCP) -> None:
        self.server = server
        self.alive = True
        self.connections = 0
        self._scopes: list[anyio.CancelScope] = []

    def crash(self) -> None:
        self.alive = False
        for scope in self._scopes:
            scope.cancel()
        self._scopes.clear()

    def revive(self) -> None:
        self.alive = True

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Streams]:
        if not self.alive:
            raise ConnectionRefusedError(f"in-process server {self.server.name!r} is down")
        self.connections += 1
        low: Any = self.server._mcp_server
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            scope = anyio.CancelScope()
            self._scopes.append(scope)

            async def run() -> None:
                with scope:
                    await low.run(server_read, server_write, low.create_initialization_options())
                # Closing our write side is what the client sees as "process exited".
                await server_write.aclose()
                await server_read.aclose()

            async with anyio.create_task_group() as tg:
                tg.start_soon(run)
                try:
                    yield client_streams
                finally:
                    tg.cancel_scope.cancel()
