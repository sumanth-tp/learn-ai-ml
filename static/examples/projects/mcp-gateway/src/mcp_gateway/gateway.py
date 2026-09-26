"""Assemble the gateway: auth, one ProxyProvider per upstream, the enforcement
middleware, and the operational HTTP routes."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.providers.proxy import ProxyProvider
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from mcp_gateway.audit import AuditLog
from mcp_gateway.config import Settings
from mcp_gateway.identity import build_verifier
from mcp_gateway.llm_scanner import DescriptionJudge, LLMDescriptionJudge, build_chat_model
from mcp_gateway.middleware import Components, GatewayMiddleware
from mcp_gateway.observability import Metrics
from mcp_gateway.policy import PolicyStore
from mcp_gateway.ratelimit import Limiter
from mcp_gateway.resilience import BreakerState, CircuitBreaker, TTLCache
from mcp_gateway.secret_broker import SecretBroker, build_broker
from mcp_gateway.state import PinStore, QuotaStore, StateDB
from mcp_gateway.upstreams import (
    ClientFactory,
    UpstreamSpec,
    build_client_factory,
    load_upstreams,
)

INSTRUCTIONS = (
    "Enterprise MCP gateway. Tools are namespaced by upstream (e.g. docs_read_doc). "
    "You only see tools your groups may call; denials explain which rule applied."
)


class Readiness:
    """Pings each required upstream at most every ``ttl`` seconds."""

    def __init__(self, factories: dict[str, ClientFactory], specs: dict[str, UpstreamSpec],
                 breakers: dict[str, CircuitBreaker], ttl: float = 5.0) -> None:
        self.factories, self.specs, self.breakers, self.ttl = factories, specs, breakers, ttl
        self._cached: tuple[float, dict[str, str]] | None = None

    async def _ping(self, name: str) -> str:
        if self.breakers[name].state is BreakerState.OPEN:
            return "circuit_open"
        try:
            async with asyncio.timeout(3.0), self.factories[name]() as client:
                await client.ping()
            return "ok"
        except Exception as exc:
            return f"unreachable: {type(exc).__name__}"

    async def check(self) -> dict[str, str]:
        now = time.monotonic()
        if self._cached and now - self._cached[0] < self.ttl:
            return self._cached[1]
        names = list(self.specs)
        results = await asyncio.gather(*(self._ping(n) for n in names))
        status = dict(zip(names, results, strict=True))
        self._cached = (now, status)
        return status


def build_gateway(
    settings: Settings,
    *,
    specs: list[UpstreamSpec] | None = None,
    broker: SecretBroker | None = None,
    server_overrides: dict[str, FastMCP[Any]] | None = None,
    judge: DescriptionJudge | None = None,
) -> tuple[FastMCP[Any], Components]:
    specs = specs if specs is not None else load_upstreams(settings.upstreams_file)
    broker = broker or build_broker(settings.secrets_backend, settings.secrets_dir)
    overrides = server_overrides or {}
    if judge is None and settings.llm_scanner:
        judge = LLMDescriptionJudge(build_chat_model(settings))

    db = StateDB(settings.state_db)
    breakers = {
        s.name: CircuitBreaker(s.name, settings.breaker_failure_threshold,
                               settings.breaker_reset_seconds)
        for s in specs
    }
    components = Components(
        settings=settings,
        specs={s.name: s for s in specs},
        policy=PolicyStore(settings.policy_file),
        audit=AuditLog(settings.audit_path),
        pins=PinStore(db),
        limiter=Limiter(QuotaStore(db)),
        cache=TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
        breakers=breakers,
        metrics=Metrics(),
        judge=judge,
    )

    gateway: FastMCP[Any] = FastMCP(
        "mcp-gateway",
        instructions=INSTRUCTIONS,
        auth=build_verifier(settings),
        middleware=[GatewayMiddleware(components)],
        mask_error_details=True,  # upstream stack traces never reach clients
    )
    factories: dict[str, ClientFactory] = {}
    for spec in specs:
        factory = build_client_factory(
            spec, broker, default_timeout=settings.upstream_timeout_seconds,
            server_override=overrides.get(spec.name),
        )
        factories[spec.name] = factory
        # cache_ttl bounds how stale a definition can be before a rug pull
        # is noticed on the call path.
        provider = ProxyProvider(factory, cache_ttl=settings.definition_refresh_seconds)
        components.providers[spec.name] = provider
        gateway.add_provider(provider, namespace=spec.name)

    readiness = Readiness(factories, components.specs, breakers)

    @gateway.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @gateway.custom_route("/readyz", methods=["GET"])
    async def readyz(_: Request) -> Response:
        upstreams = await readiness.check()
        required_ok = all(
            upstreams[n] == "ok" for n, s in components.specs.items() if s.required
        )
        policy_error = components.policy.last_error
        ready = required_ok and policy_error is None
        return JSONResponse(
            {"ready": ready, "upstreams": upstreams, "policy_error": policy_error},
            status_code=200 if ready else 503,
        )

    @gateway.custom_route("/metrics", methods=["GET"])
    async def metrics(request: Request) -> Response:
        token = settings.metrics_token
        if token is not None:
            if request.headers.get("authorization") != f"Bearer {token.get_secret_value()}":
                return PlainTextResponse("unauthorized", status_code=401)
        return Response(generate_latest(components.metrics.registry),
                        media_type=CONTENT_TYPE_LATEST)

    return gateway, components
