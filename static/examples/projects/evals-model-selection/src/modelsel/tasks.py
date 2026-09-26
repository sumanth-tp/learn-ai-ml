"""Prompts and output parsers for the three sub-tasks of ticket triage.

Every candidate gets byte-identical prompts. ``PROMPT_VERSION`` is part of the
cache key and the run manifest: change a prompt and old results stop being
comparable, which is exactly what you want.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from modelsel.schemas import BenchmarkItem, Label, Task, TicketFields

PROMPT_VERSION = "v3"

LABEL_LIST = ", ".join(label.value for label in Label)

CLASSIFY_SYSTEM = f"""You are a support-ticket triage assistant.
TASK: classify
Choose exactly one label for the ticket from: {LABEL_LIST}.
Rules: a duplicate charge where the customer asks for money back is `refund`;
a wrong amount on an invoice is `billing`; ending a plan is `cancellation`.
Answer with the label only, in lower case, nothing else."""

EXTRACT_SYSTEM = """You are a support-ticket triage assistant.
TASK: extract
Return ONLY a JSON object with exactly these keys:
  "order_id": string like "ORD-12345" or null,
  "product": the product name as written, or null,
  "amount": number in pounds without the currency sign, or null,
  "priority": one of "low", "medium", "high", "urgent",
  "sentiment": one of "negative", "neutral", "positive".
No prose, no markdown fences, no extra keys."""

REPLY_SYSTEM = """You are a support agent drafting a reply for a human to review.
TASK: reply
Greet the customer by name, acknowledge the product, state the concrete next
action, cite the order number when there is one, and close politely.
Keep it under 120 words. Never promise anything not in the policy:
refunds within five working days, courier trace within 24 hours,
password reset links by email, cancellations take effect immediately."""

SYSTEMS: dict[Task, str] = {"classify": CLASSIFY_SYSTEM, "extract": EXTRACT_SYSTEM, "reply": REPLY_SYSTEM}
MAX_TOKENS: dict[Task, int] = {"classify": 8, "extract": 200, "reply": 300}


def build_messages(task: Task, item: BenchmarkItem) -> list[BaseMessage]:
    return [SystemMessage(content=SYSTEMS[task]), HumanMessage(content=f"TICKET:\n{item.ticket}")]


def parse_label(text: str) -> str:
    """Return the first valid label in the output, or ``invalid``.

    Lenient on case and punctuation, strict on vocabulary: "Refund." is a refund,
    "money back" is not a label and counts as wrong.
    """
    for token in re.findall(r"[a-z_]+", text.strip().lower()):
        if token in _VALID_LABELS:
            return token
    return "invalid"


_VALID_LABELS = frozenset(label.value for label in Label)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_fields(text: str) -> tuple[TicketFields | None, str | None]:
    """Parse and schema-validate extraction output. Returns (fields, error)."""
    raw = _FENCE.sub("", text.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode: {exc.msg}"
    if not isinstance(data, dict):
        return None, "json_not_object"
    try:
        return TicketFields.model_validate(data), None
    except ValidationError as exc:
        return None, f"schema: {exc.errors()[0]['type']}"
