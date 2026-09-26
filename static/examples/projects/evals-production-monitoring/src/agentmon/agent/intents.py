"""A cheap, deterministic intent classifier. It feeds stratified sampling, the intent
histogram used for drift detection, and the reference-free 'required tool' check."""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal["balance", "transactions", "transfer", "faq", "greeting", "out_of_scope", "other"]
INTENTS: tuple[Intent, ...] = (
    "balance",
    "transactions",
    "transfer",
    "faq",
    "greeting",
    "out_of_scope",
    "other",
)

# Which tool a correct answer to each intent must be grounded in.
REQUIRED_TOOL: dict[str, str] = {
    "balance": "get_balance",
    "transactions": "list_transactions",
    "transfer": "transfer_funds",
    "faq": "search_help_center",
}

_RULES: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "out_of_scope",
        re.compile(
            r"\b(crypto|bitcoin|stock tips?|invest(ing|ment)? advice|weather|poem|recipe|joke|"
            r"write (me )?(some )?code|football|horoscope)\b",
            re.I,
        ),
    ),
    ("transfer", re.compile(r"\b(transfer|send|pay|move)\b.*\bACC-\d{4}\b", re.I)),
    (
        "transactions",
        re.compile(
            r"\b(transactions?|statement|spent|spending|payments? (in|out)|recent activity)\b", re.I
        ),
    ),
    ("balance", re.compile(r"\b(balance|how much (money )?(do i have|is in|is left))\b", re.I)),
    (
        "faq",
        re.compile(
            r"\b(fee|fees|card|lost|stolen|declined|limit|interest|overdraft|dispute|refund|"
            r"freeze|how do i|how can i|what is the)\b",
            re.I,
        ),
    ),
    (
        "greeting",
        re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks?)\b", re.I),
    ),
]


def classify_intent(text: str) -> Intent:
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent
    return "other"
