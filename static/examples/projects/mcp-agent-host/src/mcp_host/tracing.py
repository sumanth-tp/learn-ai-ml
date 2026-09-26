"""OpenTelemetry spans for every MCP call.

LangSmith (enabled by ``LANGSMITH_TRACING=true``) already traces the LangGraph run
and the LLM calls. It does not see the MCP wire: which server, which method, how
long, what error code. These spans do, and they nest under whatever span is active.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode

TRACER_NAME = "mcp_host"
_configured = False


def configure_tracing(
    exporter: str, service_name: str, extra: SpanExporter | None = None
) -> TracerProvider:
    """Install a tracer provider once per process. ``extra`` lets tests capture spans."""
    global _configured
    current = trace.get_tracer_provider()
    if _configured and isinstance(current, TracerProvider):
        if extra is not None:
            current.add_span_processor(SimpleSpanProcessor(extra))
        return current
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        # Reads OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318).
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if extra is not None:
        provider.add_span_processor(SimpleSpanProcessor(extra))
    trace.set_tracer_provider(provider)
    _configured = True
    return provider


@contextmanager
def mcp_span(server: str, method: str, **attributes: Any) -> Iterator[Span]:
    """One span per MCP request. Attribute names follow the OTel MCP draft conventions."""
    tracer = trace.get_tracer(TRACER_NAME)
    name = f"mcp {method}"
    if "tool" in attributes:
        name = f"{name} {server}.{attributes['tool']}"
    with tracer.start_as_current_span(name, record_exception=False) as span:
        span.set_attribute("mcp.server.name", server)
        span.set_attribute("mcp.method.name", method)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"mcp.{key}", value)
        try:
            yield span
        except BaseException as exc:
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("error.type", type(exc).__name__)
            raise
