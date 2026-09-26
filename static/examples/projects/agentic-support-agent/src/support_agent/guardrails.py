"""Input and output guardrails: prompt-injection blocking and card-number scrubbing.

These are cheap deterministic checks that run before any LLM call. They are a
first line, not the only line: tools are also scoped per intent and per user,
and refunds above the threshold need a human, so a prompt that slips past
these patterns still cannot move money on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent.pii import CARD_RE, _card

MAX_INPUT_CHARS = 4000

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|earlier)\b.{0,20}"
            r"\b(instructions?|rules|prompts?|messages?)\b",
            re.I | re.S,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|leak|output)\b.{0,40}\b(system|hidden|initial)\s+"
            r"(prompt|instructions?|message)",
            re.I | re.S,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"\byou are (now|no longer)\b|\bact as (an? )?(admin|developer|system)\b|"
            r"\b(developer|god|dan|jailbreak) mode\b",
            re.I,
        ),
    ),
    ("fake_role_tag", re.compile(r"(^|\n)\s*(system|assistant)\s*:|<\s*/?\s*system\s*>", re.I)),
    (
        "tool_forcing",
        re.compile(
            r"\b(call|invoke|run|execute)\b.{0,20}\b(issue_refund|create_return|tool)\b.{0,40}"
            r"\b(without|skip|bypass|no)\b.{0,20}\b(approval|check|review|limit)",
            re.I | re.S,
        ),
    ),
    (
        "approval_bypass",
        re.compile(
            r"\b(bypass|skip|override|disable)\b.{0,30}\b(approval|threshold|limit|review|guardrails?)\b",
            re.I | re.S,
        ),
    ),
]


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str | None = None


def check_input(text: str) -> GuardResult:
    if not text.strip():
        return GuardResult(False, "empty")
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, "too_long")
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardResult(False, f"prompt_injection:{name}")
    return GuardResult(True)


def scrub_output(text: str) -> str:
    """The assistant must never echo a card number, even one the user typed."""
    return CARD_RE.sub(_card, text)


REFUSAL_MESSAGES = {
    "off_topic": "I can help with orders, deliveries, returns, refunds and our store "
    "policies. I can't help with that one, but ask me anything about your orders.",
    "prompt_injection": "I can't act on that request. I can help with your orders, "
    "returns and refunds.",
    "too_long": "That message is too long for me to process. Could you shorten it?",
    "empty": "I didn't catch a question there. How can I help with your order?",
}


def refusal_for(reason: str) -> str:
    return REFUSAL_MESSAGES.get(reason.split(":")[0], REFUSAL_MESSAGES["prompt_injection"])
