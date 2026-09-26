"""Domain models shared by every layer: benchmark items, model outputs, scores."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Label(StrEnum):
    BILLING = "billing"
    REFUND = "refund"
    SHIPPING = "shipping"
    ACCOUNT_ACCESS = "account_access"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    CANCELLATION = "cancellation"
    OTHER = "other"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


Split = Literal["dev", "test", "private"]
Task = Literal["classify", "extract", "reply"]
TASKS: tuple[Task, ...] = ("classify", "extract", "reply")


class TicketFields(BaseModel):
    """The extraction schema. ``extra='forbid'`` makes invented keys a schema failure."""

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{5}$")
    product: str | None = None
    amount: float | None = Field(default=None, ge=0)
    priority: Priority
    sentiment: Sentiment


FIELD_NAMES: tuple[str, ...] = tuple(TicketFields.model_fields)


class BenchmarkItem(BaseModel):
    id: str
    split: Split
    customer_name: str
    ticket: str
    label: Label
    fields: TicketFields
    reference_reply: str
    tags: list[str] = Field(default_factory=list)
    """Slices such as ``ambiguous`` or ``no_order_id``, used for per-slice reporting."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Completion(BaseModel):
    """One LLM call as the harness saw it."""

    model_id: str
    text: str
    usage: Usage
    latency_ms: float
    cost_usd: float
    cached: bool = False
    attempts: int = 1
    response_metadata: dict[str, object] = Field(default_factory=dict)


class Prediction(BaseModel):
    """One candidate model's answer to one task on one item."""

    run_id: str
    model_id: str
    item_id: str
    split: Split
    task: Task
    output: str
    usage: Usage
    latency_ms: float
    cost_usd: float
    cached: bool
    error: str | None = None


class ItemScore(BaseModel):
    """Per-item scores for one model. Paired tests need these, not just the means."""

    model_id: str
    item_id: str
    split: Split
    label_pred: str
    label_correct: bool
    json_valid: bool
    field_accuracy: float
    reply_score: float | None
    """Judge score on the 1..5 scale after swap/probability weighting."""
    composite: float
    latency_ms: float
    cost_usd: float
    tags: list[str] = Field(default_factory=list)


class HumanLabel(BaseModel):
    """One row of the human-labelled calibration file."""

    id: str
    item_id: str
    ticket: str
    reference_reply: str
    reply_a: str
    reply_b: str
    author_a: str
    author_b: str
    ratings_a: list[int] = Field(min_length=2)
    ratings_b: list[int] = Field(min_length=2)
    preference: Literal["A", "B", "tie"]
