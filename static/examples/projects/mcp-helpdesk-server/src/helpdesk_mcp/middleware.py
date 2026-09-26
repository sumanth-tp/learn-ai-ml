"""FastMCP middleware: the cross-cutting concerns every request goes through.

Order (outermost first), set in ``server.build_server``:

1. ``ObservabilityMiddleware``: request id, structured log line, metrics.
2. ``RateLimitMiddleware``: per-user token bucket.
3. ``RoleVisibilityMiddleware``: hide tools the caller's role cannot use.
4. ``TimeoutMiddleware``: a hard ceiling on every tool call.

Observability is outermost so that rate-limited and timed-out calls are still
counted and logged.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any

import anyio
import structlog
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.middleware.rate_limiting import TokenBucketRateLimiter

from helpdesk_mcp.config import Settings
from helpdesk_mcp.errors import HelpdeskError, RateLimited, Timeout
from helpdesk_mcp.identity import current_identity
from helpdesk_mcp.logging_setup import get_logger
from helpdesk_mcp.metrics import Metrics

log = get_logger("helpdesk_mcp.access")

# Protocol plumbing that should never be rate limited or counted as "work".
_EXEMPT = {"initialize", "server/discover", "ping", "notifications/initialized"}
# Methods that do real work and are charged against the caller's rate limit.
# List calls are cheap and clients issue them implicitly (FastMCP's client
# lists tools to learn output schemas), so charging them surprises users.
_METERED = {"tools/call", "resources/read", "prompts/get"}


def _component(context: MiddlewareContext) -> str:
    msg = context.message
    for attr in ("name", "uri"):
        value = getattr(msg, attr, None)
        if value is not None:
            return str(value)
    return "-"


def current_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


class ObservabilityMiddleware(Middleware):
    def __init__(self, settings: Settings, metrics: Metrics) -> None:
        self.settings = settings
        self.metrics = metrics

    async def on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        method = context.method or "unknown"
        component = _component(context)
        headers = get_http_headers()  # {} outside HTTP (stdio, in-memory)
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        user = tenant = "-"
        try:
            who = current_identity(self.settings)
            user, tenant = who.user, who.tenant
        except HelpdeskError:
            pass  # unauthenticated; the tool itself will refuse
        structlog.contextvars.bind_contextvars(
            request_id=request_id, user=user, tenant=tenant, method=method, component=component
        )
        start = time.perf_counter()
        self.metrics.in_flight.inc()
        outcome = "ok"
        try:
            return await call_next(context)
        except Exception as exc:
            outcome = "error"
            error = getattr(exc, "code", None) or type(exc).__name__
            self.metrics.errors.labels(method, component, str(error)).inc()
            log.warning("mcp.request_failed", error=str(error), detail=str(exc)[:300])
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.metrics.in_flight.dec()
            if method not in _EXEMPT:
                self.metrics.requests.labels(method, component, outcome).inc()
                self.metrics.latency.labels(method, component).observe(elapsed)
            log.info("mcp.request", outcome=outcome, duration_ms=round(elapsed * 1000, 2))
            structlog.contextvars.unbind_contextvars(
                "request_id", "user", "tenant", "method", "component"
            )


class RateLimitMiddleware(Middleware):
    """Per-(tenant, user) token bucket.

    FastMCP ships ``RateLimitingMiddleware``, but its per-client store is an
    unbounded ``defaultdict``: one bucket per user forever. This version bounds
    memory with an LRU of buckets and reports rejections in metrics. It is
    per-process; with several replicas, move the buckets to Redis.
    """

    def __init__(self, settings: Settings, metrics: Metrics, max_clients: int = 10_000) -> None:
        self.settings = settings
        self.metrics = metrics
        self.capacity = settings.rate_limit_burst
        self.refill_per_s = settings.rate_limit_per_minute / 60.0
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, TokenBucketRateLimiter] = OrderedDict()

    def _bucket(self, key: str) -> TokenBucketRateLimiter:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucketRateLimiter(self.capacity, self.refill_per_s)
            self._buckets[key] = bucket
            if len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return bucket

    async def on_request(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        if context.method not in _METERED:
            return await call_next(context)
        try:
            who = current_identity(self.settings)
            key, tenant = f"{who.tenant}:{who.user}", who.tenant
        except HelpdeskError:
            key, tenant = "anonymous", "-"
        if not await self._bucket(key).consume():
            self.metrics.rate_limited.labels(tenant).inc()
            retry = max(1, round(1 / self.refill_per_s))
            raise RateLimited(f"Too many requests. Retry in about {retry}s.")
        return await call_next(context)


class RoleVisibilityMiddleware(Middleware):
    """Only list tools the caller can use. Tools carry a ``min_role:<role>`` tag.

    This is least privilege for the *model*: a requester's assistant never sees
    ``delete_ticket``, so it cannot be talked into calling it. It is not the
    security boundary; each tool re-checks the role before acting.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def on_list_tools(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        tools = await call_next(context)
        try:
            who = current_identity(self.settings)
        except HelpdeskError:
            return []
        visible = []
        for tool in tools:
            needed = next(
                (t.split(":", 1)[1] for t in tool.tags if t.startswith("min_role:")), None
            )
            if needed is None or who.has_role(needed):
                visible.append(tool)
        return visible


class TimeoutMiddleware(Middleware):
    """Cancel any tool call that exceeds ``tool_timeout_s`` and say so clearly."""

    def __init__(self, settings: Settings) -> None:
        self.timeout_s = settings.tool_timeout_s

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        try:
            with anyio.fail_after(self.timeout_s):
                return await call_next(context)
        except TimeoutError:
            raise Timeout(
                f"Tool '{_component(context)}' exceeded {self.timeout_s:g}s and was cancelled. "
                "Narrow the request (smaller window or page) and try again."
            ) from None
