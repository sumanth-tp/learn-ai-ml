"""OpenTelemetry tracing with a SQLite exporter, so every span is queryable offline.

Span attributes follow the OpenTelemetry GenAI semantic conventions
(gen_ai.request.model, gen_ai.usage.input_tokens, ...) so the same spans can be
shipped to any OTLP backend (Jaeger, Tempo, Honeycomb, Langfuse) unchanged.
LangSmith tracing is separate and switched on by LangChain's own env vars."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

from agentmon.clock import Clock
from agentmon.store import Store

log = logging.getLogger("agentmon.tracing")


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, store: Store) -> None:
        self.store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for s in spans:
            ctx = s.get_span_context()
            if ctx is None:
                continue
            rows.append(
                {
                    "span_id": f"{ctx.span_id:016x}",
                    "trace_id": f"{ctx.trace_id:032x}",
                    "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                    "name": s.name,
                    "start_ns": s.start_time,
                    "end_ns": s.end_time,
                    "status": s.status.status_code.name,
                    "attributes": json.dumps(dict(s.attributes or {}), default=str),
                }
            )
        try:
            self.store.insert_spans(rows)
        except Exception:  # an exporter must never take the request down
            log.exception("span.export_failed")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def build_tracer_provider(
    store: Store, service_name: str, otlp_endpoint: str | None = None
) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # Simple (synchronous) processor: the span is in SQLite before the request returns,
    # so an evaluator that reads it immediately never races the exporter.
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(store)))
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            log.warning("otlp.exporter_missing", extra={"hint": "uv sync --extra otlp"})
        else:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
    return provider


def _attr(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    return json.dumps(value, default=str)


class Tracing:
    """Thin wrapper that takes span timestamps from our Clock, so simulated traffic
    has simulated span times and latencies."""

    def __init__(self, provider: TracerProvider, clock: Clock) -> None:
        self.provider = provider
        self.tracer = provider.get_tracer("agentmon")
        self.clock = clock

    def _now_ns(self) -> int:
        return int(self.clock.now() * 1e9)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        with self.tracer.start_as_current_span(
            name,
            start_time=self._now_ns(),
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, _attr(value))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end(end_time=self._now_ns())

    @staticmethod
    def set(span: Span, **attributes: Any) -> None:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, _attr(value))
