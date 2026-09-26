from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from support_agent.memory import (
    extract_preferences,
    issues_ns,
    load_user_context,
    record_issue,
    save_preferences,
)


def test_extract_preferences() -> None:
    assert extract_preferences("Hi, please call me asha and contact me by text") == {
        "preferred_name": "Asha",
        "contact_channel": "sms",
    }
    assert extract_preferences("where is my order") == {}


async def test_writes_are_deduplicated() -> None:
    store = InMemoryStore()
    assert await save_preferences(store, "u1", {"preferred_name": "Asha"}) is True
    assert await save_preferences(store, "u1", {"preferred_name": "Asha"}) is False
    assert await record_issue(store, "u1", "refund", "ORD-1", "resolved") is True
    assert await record_issue(store, "u1", "refund", "ORD-1", "resolved") is False
    assert await record_issue(store, "u1", "refund", "ORD-1", "handoff") is True  # changed
    assert len(await store.asearch(issues_ns("u1"))) == 1


async def test_namespaces_isolate_users() -> None:
    store = InMemoryStore()
    await save_preferences(store, "u1", {"preferred_name": "Asha"})
    await record_issue(store, "u1", "returns", "ORD-1", "resolved")
    assert "Asha" in await load_user_context(store, "u1")
    assert await load_user_context(store, "u2") == ""
