"""Graph state. Values are plain JSON types so any checkpointer can store them."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Status = Literal["running", "answered", "rejected", "failed"]


class Turn(TypedDict):
    question: str
    standalone_question: str
    sql: str | None
    status: Status
    answer: str


class AnalystState(TypedDict, total=False):
    # input
    question: str
    # conversation memory: completed turns in this thread (appended by finalize)
    history: Annotated[list[Turn], operator.add]
    # per-turn working state (reset by contextualise at the start of every turn)
    standalone_question: str
    tables: list[str]
    table_scores: list[dict[str, Any]]
    schema_context: str
    plan: dict[str, Any] | None
    sql: str | None
    sql_source: Literal["llm", "cache"] | None
    cache_similarity: float | None
    attempts: int
    errors: list[str]
    last_error: str | None
    warnings: list[str]
    estimate: dict[str, int] | None
    approval: dict[str, Any] | None
    result: dict[str, Any] | None
    answer: str | None
    chart_recommended: bool
    chart_png_base64: str | None
    chart_error: str | None
    status: Status
    # accounting, summed over the turn
    input_tokens: int
    output_tokens: int
    llm_calls: int
