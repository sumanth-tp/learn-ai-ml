"""Progress events. Nodes emit them through LangGraph's custom stream channel."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

log = logging.getLogger("research_analyst.events")

EventType = Literal[
    "run_started",
    "plan_ready",
    "worker_started",
    "crag_verdict",
    "web_fallback",
    "worker_finished",
    "worker_failed",
    "sources_consolidated",
    "draft_ready",
    "critique",
    "budget_degraded",
    "verification",
    "report_ready",
    "run_failed",
    "run_resumed",
]


class ProgressEvent(BaseModel):
    type: EventType
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


def emit(type_: EventType, **data: Any) -> None:
    """Emit to the stream if running inside a graph; always log it."""
    event = ProgressEvent(type=type_, data=data)
    log.info(type_, extra={"event": type_, **{f"ev_{k}": v for k, v in data.items()}})
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a graph run (unit tests of helpers)
        return
    writer(event.model_dump())
