"""Long-term memory in a LangGraph Store, namespaced per user.

Namespaces:
  ("users", <user_id>, "preferences")  key "profile"  -> preferred name, channel
  ("users", <user_id>, "issues")       key <digest>   -> one record per (intent, order)

Writes are deduplicated: an issue key is a digest of (intent, order_id), and a
write is skipped when the stored value already says the same thing.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore


def prefs_ns(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "preferences")


def issues_ns(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "issues")


_NAME = re.compile(r"\b(?:call me|my name is|i'?m called)\s+([A-Z][a-z]{1,30})\b", re.I)
_CHANNEL = re.compile(
    r"\b(?:prefer|contact me by|reach me by|use)\s+(email|sms|text|phone)\b", re.I
)


def extract_preferences(text: str) -> dict[str, str]:
    """Rule-based extraction of stable preferences from one user message."""
    found: dict[str, str] = {}
    if m := _NAME.search(text):
        found["preferred_name"] = m.group(1).capitalize()
    if m := _CHANNEL.search(text):
        found["contact_channel"] = {"text": "sms"}.get(m.group(1).lower(), m.group(1).lower())
    return found


def issue_key(intent: str, order_id: str | None) -> str:
    return hashlib.sha256(f"{intent}|{order_id or '-'}".encode()).hexdigest()[:16]


async def load_user_context(store: BaseStore, user_id: str, limit: int = 3) -> str:
    """Render what we remember about the user as a short prompt section."""
    lines: list[str] = []
    profile = await store.aget(prefs_ns(user_id), "profile")
    if profile and profile.value:
        if name := profile.value.get("preferred_name"):
            lines.append(f"preferred name: {name}")
        if channel := profile.value.get("contact_channel"):
            lines.append(f"preferred contact channel: {channel}")
    issues = await store.asearch(issues_ns(user_id), limit=20)
    issues.sort(key=lambda it: it.updated_at, reverse=True)
    for it in issues[:limit]:
        v = it.value
        lines.append(
            f"past issue: {v['intent']} on {v.get('order_id') or 'no order'} ({v['outcome']})"
        )
    return "\n".join(lines)


async def save_preferences(store: BaseStore, user_id: str, found: dict[str, str]) -> bool:
    if not found:
        return False
    current = await store.aget(prefs_ns(user_id), "profile")
    merged = {**(current.value if current else {}), **found}
    if current and current.value == merged:
        return False  # dedupe: nothing new
    await store.aput(prefs_ns(user_id), "profile", merged)
    return True


async def record_issue(
    store: BaseStore, user_id: str, intent: str, order_id: str | None, outcome: str
) -> bool:
    key = issue_key(intent, order_id)
    current = await store.aget(issues_ns(user_id), key)
    value: dict[str, Any] = {"intent": intent, "order_id": order_id, "outcome": outcome}
    if current and {k: current.value.get(k) for k in value} == value:
        return False
    value["last_seen"] = datetime.now(UTC).isoformat()
    await store.aput(issues_ns(user_id), key, value)
    return True
