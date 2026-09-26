"""Guardrails, each a small, testable function.

- InputGuard: blocks prompt injection, jailbreaks, toxic requests and out-of-scope asks.
- sanitise_tool_output: strips instruction-like text from tool results (indirect injection).
- ActionGuard: a side-effecting tool call must be traceable to the user's own words.
- OutputGuard: redacts third-party PII and blocks system-prompt leakage (canary).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentmon.agent.intents import classify_intent
from agentmon.agent.prompts import CANARY

REFUSAL = (
    "I'm sorry, I can't help with that. I can help with your balances, transactions, "
    "transfers between accounts and help-centre questions."
)

_INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above|your) (instructions|rules)|"
    r"(reveal|print|show|repeat) (me )?(your|the) (system )?(prompt|instructions)|"
    r"system prompt|you are now|developer mode|\bDAN\b|jailbreak|no restrictions|"
    r"disregard (the|your) (rules|policy))",
    re.I,
)
_TOXIC = re.compile(
    r"\b(insult|abusive|humiliate|threaten|hate message|slur|harass|bully)\w*", re.I
)
_TOOL_INSTRUCTION = re.compile(
    r"([^.]*\b(system note|ignore (all |any )?previous instructions|assistant must|"
    r"call transfer_funds|immediately call)\b[^.]*\.?)",
    re.I,
)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z]{4}\d{10,14}\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


@dataclass
class GuardDecision:
    allowed: bool
    category: str | None = None
    reason: str = ""


class InputGuard:
    def check(self, text: str) -> GuardDecision:
        if _INJECTION.search(text):
            return GuardDecision(False, "prompt_injection", "instruction-override pattern")
        if _TOXIC.search(text):
            return GuardDecision(False, "toxic_request", "request for abusive content")
        if classify_intent(text) == "out_of_scope":
            return GuardDecision(False, "out_of_scope", "not a banking request")
        return GuardDecision(True)


def sanitise_tool_output(content: str) -> tuple[str, bool]:
    """Remove sentences that try to instruct the assistant. Returns (clean, was_modified)."""
    cleaned, n = _TOOL_INSTRUCTION.subn(" [removed: instruction-like text in tool output]", content)
    return cleaned, n > 0


class ActionGuard:
    """Side effects need provenance: the destination account and the transfer intent
    must both appear in the customer's own message, not just in retrieved text."""

    def check(self, tool_name: str, args: dict[str, Any], user_message: str) -> GuardDecision:
        if tool_name != "transfer_funds":
            return GuardDecision(True)
        if classify_intent(user_message) != "transfer":
            return GuardDecision(False, "unrequested_action", "user did not ask for a transfer")
        if str(args.get("to_account", "")) not in user_message:
            return GuardDecision(False, "unrequested_action", "destination not named by user")
        return GuardDecision(True)


@dataclass
class OutputGuard:
    allowed_pii: set[str] = field(default_factory=set)

    def apply(self, text: str) -> tuple[str, list[str]]:
        flags: list[str] = []
        if CANARY in text:
            return REFUSAL, ["system_prompt_leak_blocked"]

        def _mask(kind: str) -> Any:
            def repl(m: re.Match[str]) -> str:
                if m.group(0) in self.allowed_pii:
                    return m.group(0)
                flags.append(f"redacted_{kind}")
                return f"[redacted {kind}]"

            return repl

        text = EMAIL.sub(_mask("email"), text)
        text = IBAN.sub(_mask("iban"), text)
        text = CARD.sub(_mask("card"), text)
        return text, sorted(set(flags))
