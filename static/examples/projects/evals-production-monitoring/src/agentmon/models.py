"""Shared record types that move between the agent, the store and the evaluators."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TraceStatus = Literal["ok", "error", "blocked"]


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"] = "success"
    output: str = ""


class TraceRecord(BaseModel):
    trace_id: str
    request_id: str
    ts: float
    user_id: str
    session_id: str
    input: str
    output: str
    intent: str
    prompt_version: str
    model: str
    status: TraceStatus
    error: str | None = None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    trace_id: str
    evaluator: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    tier: Literal["heuristic", "judge", "cascade"] = "heuristic"
    cost_usd: float = 0.0
    ts: float = 0.0


class Feedback(BaseModel):
    trace_id: str
    rating: Literal[-1, 1]
    correction: str | None = None
    ts: float = 0.0
