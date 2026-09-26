"""Prometheus metrics, one registry per server instance.

A private ``CollectorRegistry`` (instead of the global default) means two
servers in one test process do not collide, and ``/metrics`` exposes only
what this service owns.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "helpdesk_mcp_requests_total",
            "MCP requests by method, component and outcome.",
            ["method", "component", "outcome"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "helpdesk_mcp_request_duration_seconds",
            "MCP request latency by method and component.",
            ["method", "component"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "helpdesk_mcp_errors_total",
            "Failed MCP requests by error type.",
            ["method", "component", "error"],
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "helpdesk_mcp_rate_limited_total",
            "Requests rejected by the per-user rate limiter.",
            ["tenant"],
            registry=self.registry,
        )
        self.llm_fallbacks = Counter(
            "helpdesk_mcp_triage_fallback_total",
            "Triage suggestions served by rules because the LLM failed.",
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "helpdesk_mcp_in_flight_requests",
            "Requests currently being handled.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
