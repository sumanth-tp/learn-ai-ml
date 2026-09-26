"""One long-lived, self-healing MCP session per server.

A supervisor task owns the transport and the ``ClientSession``: it connects,
initialises, discovers capabilities, then watches for three things: a stop request, a
``*/list_changed`` notification (re-discover) and a broken connection (reconnect with
exponential backoff and jitter). Callers never touch the session directly; they go
through :meth:`ServerConnection.request`, which adds a timeout, sends
``notifications/cancelled`` when the host gives up, and turns transport failures into
``ServerUnavailable`` so the agent can degrade instead of crash.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar

import anyio
from mcp import ClientSession, McpError, types
from mcp.client.session import SamplingFnT
from mcp.shared.session import RequestResponder
from pydantic import AnyUrl

from mcp_host.settings import Settings
from mcp_host.tracing import mcp_span
from mcp_host.transports import TransportFactory

log = logging.getLogger(__name__)
T = TypeVar("T")

CLIENT_INFO = types.Implementation(name="mcp-agent-host", version="0.1.0")
LIST_CHANGED = (
    types.ToolListChangedNotification,
    types.ResourceListChangedNotification,
    types.PromptListChangedNotification,
)
TRANSPORT_ERRORS = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)


class ServerState(StrEnum):
    CONNECTING = "connecting"
    READY = "ready"
    DOWN = "down"
    STOPPED = "stopped"


class HostError(Exception):
    """Base class for errors the agent is allowed to see (as a tool result)."""


class ServerUnavailable(HostError):
    def __init__(self, server: str, reason: str) -> None:
        super().__init__(f"server {server!r} is unavailable: {reason}")
        self.server, self.reason = server, reason


class RequestTimeout(HostError):
    def __init__(self, server: str, method: str, seconds: float) -> None:
        super().__init__(f"{server} {method} timed out after {seconds:g}s and was cancelled")
        self.server, self.method, self.seconds = server, method, seconds


class RequestFailed(HostError):
    def __init__(self, server: str, method: str, message: str) -> None:
        super().__init__(f"{server} {method} failed: {message}")


class _ConnectionLost(Exception):
    pass


@dataclass
class Catalogue:
    """What one server offers, as last discovered."""

    tools: list[types.Tool] = field(default_factory=list)
    resources: list[types.Resource] = field(default_factory=list)
    resource_templates: list[types.ResourceTemplate] = field(default_factory=list)
    prompts: list[types.Prompt] = field(default_factory=list)


class ServerConnection:
    def __init__(
        self,
        name: str,
        factory: TransportFactory,
        settings: Settings,
        *,
        timeout_s: float | None = None,
        sampling_callback: SamplingFnT | None = None,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self._factory = factory
        self._settings = settings
        self.timeout_s = timeout_s or settings.call_timeout_s
        self._sampling_callback = sampling_callback
        self._on_change = on_change or (lambda _name: None)

        self.state = ServerState.CONNECTING
        self.catalogue = Catalogue()
        self.server_info: types.Implementation | None = None
        self.capabilities: types.ServerCapabilities | None = None
        self.last_error: str | None = None
        self.connected_at: datetime | None = None
        self.reconnects = 0

        self._session: ClientSession | None = None
        self._ready = asyncio.Event()
        self._wake = asyncio.Event()
        self._stopping = False
        self._broken: str | None = None
        self._refresh = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._supervise(), name=f"mcp-{self.name}")

    async def wait_ready(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
            return True
        except TimeoutError:
            return False

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self._set_state(ServerState.STOPPED)

    def _set_state(self, state: ServerState) -> None:
        if state != self.state:
            log.info("server state change", extra={"server": self.name, "state": str(state)})
            self.state = state
            self._on_change(self.name)

    async def _supervise(self) -> None:
        attempt = 0
        while not self._stopping:
            if self.state != ServerState.DOWN:  # stay "down" while retrying, no flapping
                self._set_state(ServerState.CONNECTING)
            try:
                async with (
                    self._factory() as (read, write),
                    ClientSession(
                        read,
                        write,
                        sampling_callback=self._sampling_callback,
                        message_handler=self._on_message,
                        client_info=CLIENT_INFO,
                    ) as session,
                ):
                    with mcp_span(self.name, "initialize"):
                        with anyio.fail_after(self._settings.connect_timeout_s):
                            init = await session.initialize()
                    self.server_info, self.capabilities = init.serverInfo, init.capabilities
                    await self._discover(session)
                    self._session, self._broken, self.last_error = session, None, None
                    if self.connected_at is not None:
                        self.reconnects += 1
                    self.connected_at = datetime.now(UTC)
                    attempt = 0
                    self._ready.set()
                    self._set_state(ServerState.READY)
                    await self._watch(session)
            except Exception as exc:
                self.last_error = _describe(exc)
                log.warning(
                    "server connection failed",
                    extra={"server": self.name, "error": self.last_error, "attempt": attempt},
                )
            finally:
                self._session = None
                self._ready.clear()
            if self._stopping:
                break
            self._set_state(ServerState.DOWN)
            delay = min(self._settings.backoff_max_s, self._settings.backoff_initial_s * 2**attempt)
            delay *= random.uniform(0.5, 1.0)  # jitter: many hosts must not reconnect in lockstep
            attempt += 1
            self._wake.clear()
            with anyio.move_on_after(delay):
                while not self._stopping:
                    await self._wake.wait()
                    self._wake.clear()

    async def _watch(self, session: ClientSession) -> None:
        """Park until stop, list_changed or a broken pipe; ping when idle."""
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), self._settings.ping_interval_s)
            except TimeoutError:
                try:
                    with (
                        mcp_span(self.name, "ping"),
                        anyio.fail_after(self._settings.ping_timeout_s),
                    ):
                        await session.send_ping()
                except Exception as exc:
                    raise _ConnectionLost(f"ping failed: {_describe(exc)}") from exc
                continue
            self._wake.clear()
            if self._stopping:
                return
            if self._broken:
                raise _ConnectionLost(self._broken)
            if self._refresh:
                self._refresh = False
                await self._discover(session)
                self._on_change(self.name)

    async def _on_message(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult]
        | types.ServerNotification
        | Exception,
    ) -> None:
        # Runs inside the session's receive loop. Never await a request here: the
        # response would have to come through this same loop, so it would deadlock.
        # Flag the work and let the supervisor do it.
        if isinstance(message, RuntimeError):
            # Protocol-level oddities, not a dead pipe. The common one: after we send
            # notifications/cancelled, the SDK server still answers the cancelled
            # request, and our session reports "response with an unknown request ID".
            log.debug("protocol notice", extra={"server": self.name, "error": str(message)})
        elif isinstance(message, Exception):
            self._mark_broken(f"transport error: {_describe(message)}")
        elif isinstance(message, types.ServerNotification) and isinstance(
            message.root, LIST_CHANGED
        ):
            log.info("list changed", extra={"server": self.name, "kind": message.root.method})
            self._refresh = True
            self._wake.set()

    def _mark_broken(self, reason: str) -> None:
        if self._broken is None:
            self._broken = reason
            self.last_error = reason
            self._wake.set()

    # ------------------------------------------------------------------ discovery
    async def _discover(self, session: ClientSession) -> None:
        caps = self.capabilities or types.ServerCapabilities()
        cat = Catalogue()
        with mcp_span(self.name, "discover") as span:
            if caps.tools is not None:
                cat.tools = await _paginate(session.list_tools, "tools")
            if caps.resources is not None:
                cat.resources = await _paginate(session.list_resources, "resources")
                cat.resource_templates = await _paginate(
                    session.list_resource_templates, "resourceTemplates"
                )
            if caps.prompts is not None:
                cat.prompts = await _paginate(session.list_prompts, "prompts")
            span.set_attribute("mcp.tools.count", len(cat.tools))
        self.catalogue = cat

    # ------------------------------------------------------------------ requests
    async def request(
        self,
        method: str,
        call: Callable[[ClientSession], Awaitable[T]],
        *,
        timeout_s: float | None = None,
        **span_attrs: Any,
    ) -> T:
        """Run one MCP request with timeout, cancellation and failure mapping."""
        session = self._session
        if session is None or self.state != ServerState.READY:
            raise ServerUnavailable(self.name, self.last_error or str(self.state))
        timeout = timeout_s or self.timeout_s
        started = time.perf_counter()
        with mcp_span(self.name, method, **span_attrs) as span:
            # The SDK numbers requests from this counter and has no public way to learn
            # the id it will use. Read it immediately before the call: nothing can
            # interleave, because there is no await between here and the send.
            request_id = session._request_id
            try:
                with anyio.fail_after(timeout):
                    result = await call(session)
            except TimeoutError:
                await self._cancel_remote(session, request_id, f"host timeout {timeout:g}s")
                raise RequestTimeout(self.name, method, timeout) from None
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    await self._cancel_remote(session, request_id, "cancelled by host")
                raise
            except McpError as exc:
                if exc.error.code == types.CONNECTION_CLOSED:
                    self._mark_broken("connection closed")
                    raise ServerUnavailable(self.name, "connection closed mid-request") from exc
                raise RequestFailed(self.name, method, exc.error.message) from exc
            except TRANSPORT_ERRORS as exc:
                self._mark_broken(_describe(exc))
                raise ServerUnavailable(self.name, "transport closed") from exc
            except RuntimeError as exc:  # the SDK raises this for output-schema violations
                raise RequestFailed(self.name, method, str(exc)) from exc
            finally:
                span.set_attribute("mcp.duration_ms", round((time.perf_counter() - started) * 1000))
            return result

    async def _cancel_remote(self, session: ClientSession, request_id: int, reason: str) -> None:
        """Tell the server to stop working on a request we no longer want (best effort)."""
        try:
            with anyio.move_on_after(1):
                await session.send_notification(
                    types.ClientNotification(
                        types.CancelledNotification(
                            params=types.CancelledNotificationParams(
                                requestId=request_id, reason=reason
                            )
                        )
                    )
                )
        except Exception as exc:
            log.debug("cancel notification failed", extra={"server": self.name, "error": str(exc)})

    async def call_tool(
        self, tool: str, arguments: dict[str, Any], timeout_s: float | None = None
    ) -> types.CallToolResult:
        return await self.request(
            "tools/call",
            lambda s: s.call_tool(tool, arguments),
            timeout_s=timeout_s,
            tool=tool,
        )

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        return await self.request(
            "resources/read", lambda s: s.read_resource(AnyUrl(uri)), resource_uri=uri
        )

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> types.GetPromptResult:
        return await self.request(
            "prompts/get", lambda s: s.get_prompt(name, arguments), prompt=name
        )

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": str(self.state),
            "server": self.server_info.name if self.server_info else None,
            "tools": [t.name for t in self.catalogue.tools],
            "resources": len(self.catalogue.resources) + len(self.catalogue.resource_templates),
            "prompts": [p.name for p in self.catalogue.prompts],
            "last_error": self.last_error,
            "reconnects": self.reconnects,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
        }


async def _paginate(fn: Callable[..., Awaitable[Any]], attr: str) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None
    for _ in range(100):  # a server that never stops paginating must not hang discovery
        params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
        page = await fn(params=params)
        items.extend(getattr(page, attr))
        cursor = page.nextCursor
        if not cursor:
            break
    return items


def _describe(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _describe(exc.exceptions[0])
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
