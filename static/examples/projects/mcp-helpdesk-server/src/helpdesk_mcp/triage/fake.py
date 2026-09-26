"""A deterministic, offline stand-in for the LLM provider.

It is a real LangChain ``BaseChatModel``, so the triage service calls it
through exactly the same interface as ``ChatOpenAI``. It reads the ticket text
out of the last human message and answers with the JSON the real model is
prompted to produce, using keyword rules. Tests and ``make demo`` run with no
API key and no network.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Order matters: the first matching category wins.
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("access", ("password", "locked out", "mfa", "2fa", "permission", "access to", "sso")),
    ("email", ("outlook", "email", "mailbox", "calendar", "inbox")),
    ("network", ("vpn", "wifi", "wi-fi", "network", "internet", "dns", "proxy")),
    ("hardware", ("laptop", "monitor", "keyboard", "printer", "battery", "screen", "dock")),
    ("software", ("install", "licence", "license", "crash", "update", "excel", "app ")),
]
P1_WORDS = ("everyone", "all users", "whole office", "outage", "down for", "production down")
P2_WORDS = ("team", "urgent", "cannot work", "can't work", "blocked", "deadline")
P4_WORDS = ("how do i", "question", "when convenient", "minor", "nice to have")


def classify(text: str) -> tuple[str, str]:
    t = text.lower()
    category = next((c for c, words in CATEGORY_KEYWORDS if any(w in t for w in words)), "other")
    if any(w in t for w in P1_WORDS):
        priority = "p1"
    elif any(w in t for w in P2_WORDS):
        priority = "p2"
    elif any(w in t for w in P4_WORDS):
        priority = "p4"
    else:
        priority = "p3"
    return category, priority


class KeywordTriageChatModel(BaseChatModel):
    """Answers triage prompts with rule-derived JSON. Never calls the network."""

    @property
    def _llm_type(self) -> str:
        return "keyword-triage-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = str(messages[-1].content)
        # The prompt wraps the ticket between markers; classify only that part.
        if "<ticket>" in text and "</ticket>" in text:
            text = text.split("<ticket>", 1)[1].split("</ticket>", 1)[0]
        category, priority = classify(text)
        answer = {
            "category": category,
            "priority": priority,
            "rationale": f"Keyword rules matched category '{category}' and priority '{priority}'.",
        }
        return ChatResult(generations=[ChatGeneration(message=AIMessage(json.dumps(answer)))])
