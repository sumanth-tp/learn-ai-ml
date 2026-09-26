"""Prometheus metrics and structured (JSON) logging."""

from __future__ import annotations

import logging
import sys

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    """One registry per gateway instance, so tests can build many gateways."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.calls = Counter(
            "mcp_gateway_calls_total", "Tool calls and resource reads by outcome",
            ["upstream", "target", "decision"], registry=self.registry,
        )
        self.latency = Histogram(
            "mcp_gateway_upstream_latency_seconds", "Upstream call latency",
            ["upstream"], registry=self.registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
        )
        self.denials = Counter(
            "mcp_gateway_denials_total", "Denied requests by reason class",
            ["reason"], registry=self.registry,
        )
        self.security_alerts = Counter(
            "mcp_gateway_security_alerts_total", "Rug pulls, poisoned descriptions, injected outputs",
            ["kind", "upstream"], registry=self.registry,
        )
        self.cache = Counter(
            "mcp_gateway_cache_total", "Result cache lookups", ["result"], registry=self.registry,
        )
        self.breaker_state = Gauge(
            "mcp_gateway_breaker_state", "0=closed 1=half-open 2=open",
            ["upstream"], registry=self.registry,
        )
        self.result_bytes = Histogram(
            "mcp_gateway_result_bytes", "Result size returned to clients", registry=self.registry,
            buckets=(256, 1024, 4096, 16384, 65536, 262144),
        )


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stderr, format="%(levelname)s %(name)s %(message)s")
