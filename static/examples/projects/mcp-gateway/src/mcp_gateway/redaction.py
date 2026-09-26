"""PII and secret redaction for anything the gateway writes to logs or audit.

Regexes are the right first layer here: they are fast (sub-millisecond on a
64 KB payload), deterministic and auditable. They miss free-text PII such as
names, which is why the audit log stores an argument *hash*, not the
arguments themselves, and only a short redacted preview.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("JWT", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b")),
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk|ghp|xox[abp])[-_][A-Za-z0-9_-]{12,}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    # international format only (leading +), so order ids and amounts are not eaten
    ("PHONE", re.compile(r"(?<!\w)\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")),
]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact_text(text: str) -> str:
    for label, pattern in _PATTERNS:
        if label == "CARD":
            def _card(m: re.Match[str]) -> str:
                digits = re.sub(r"\D", "", m.group(0))
                # only real card numbers (Luhn-valid) are redacted, so order ids survive
                return "[REDACTED:CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else m.group(0)

            text = pattern.sub(_card, text)
        else:
            text = pattern.sub(f"[REDACTED:{label}]", text)
    return text


SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "apikey", "authorization", "ssn"}


def redact_value(value: Any) -> Any:
    """Recursively redact strings; blank out values under sensitive keys."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            k: "[REDACTED:KEY]" if str(k).lower() in SENSITIVE_KEYS else redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_value(v) for v in value]
    return value
