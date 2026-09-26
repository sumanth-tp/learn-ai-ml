"""Input guard: refuse obvious prompt-injection and bulk-PII requests before any LLM call."""

from __future__ import annotations

import re

_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(your\s+|the\s+|previous\s+|prior\s+)*"
    r"(instructions|rules|prompt)|system\s+prompt|developer\s+mode|jailbreak",
    re.IGNORECASE,
)
_PII_REQUEST = re.compile(
    r"(home\s+address|phone\s+number|mobile\s+number|national\s+insurance|salary\s+of|"
    r"personal\s+email|direct\s+line)",
    re.IGNORECASE,
)


def check_input(question: str) -> str | None:
    """Return a reason string when the question must be refused, else None."""
    if _INJECTION.search(question):
        return "prompt_injection"
    if _PII_REQUEST.search(question):
        return "personal_data_request"
    return None
