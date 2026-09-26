"""PII detection and redaction, shared by logging and trace anonymisation."""

from __future__ import annotations

import re
from typing import Any

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
PHONE_RE = re.compile(r"(?<![\w-])\+?\d[\d ().-]{8,}\d(?![\w-])")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[CARD]" if _luhn_ok(digits) else match.group()


def _phone(match: re.Match[str]) -> str:
    # At least 10 digits: an ISO date (8 digits) or an amount is left alone.
    return "[PHONE]" if len(re.sub(r"\D", "", match.group())) >= 10 else match.group()


def redact(text: str) -> str:
    """Replace PII with typed placeholders such as [EMAIL] or [CARD].

    Cards are checked with the Luhn algorithm so long order numbers are not
    mislabelled, and anything card-shaped that fails Luhn still gets caught
    by the phone rule if it is long enough.
    """
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = CARD_RE.sub(_card, text)
    text = IBAN_RE.sub("[IBAN]", text)
    return PHONE_RE.sub(_phone, text)


def redact_obj(value: Any) -> Any:
    """Recursively redact strings inside dicts, lists and tuples."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(redact_obj(v) for v in value)
    return value
