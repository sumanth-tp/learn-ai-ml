"""PII detection and redaction, shared by ingestion, the output guard and the safety metric."""

from __future__ import annotations

import re

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?:\+44\s?\d{2}|\b0\d{2,4})[\s-]?\d{3,4}[\s-]?\d{3,4}\b"),
    "NI_NUMBER": re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def find_pii(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for kind, pattern in PII_PATTERNS.items():
        found.extend((kind, m.group(0)) for m in pattern.finditer(text))
    return found


def redact(text: str) -> str:
    for kind, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{kind}]", text)
    return text


def leaked_pii(answer: str, question: str) -> list[tuple[str, str]]:
    """PII in the answer that the user did not supply themselves in the question."""
    supplied = {value for _, value in find_pii(question)}
    return [(kind, value) for kind, value in find_pii(answer) if value not in supplied]
