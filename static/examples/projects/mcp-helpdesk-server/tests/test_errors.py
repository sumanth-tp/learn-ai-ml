"""Error paths: validation, conflicts, bad cursors and masked internal errors."""

from __future__ import annotations

import uuid

import pytest
from fastmcp import Client

from helpdesk_mcp.server import HelpdeskApp
from tests.conftest import text_of


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"title": "Hi", "description": "long enough description"}, "title"),
        ({"title": "Valid title", "description": "long enough", "priority": "urgent"}, "priority"),
        (
            {"title": "Valid title", "description": "long enough", "idempotency_key": "x"},
            "idempotency_key",
        ),
        (
            {
                "title": "Valid title",
                "description": "long enough",
                "idempotency_key": "has spaces in it",
            },
            "idempotency_key",
        ),
        ({"description": "no title at all"}, "title"),
    ],
    ids=["short-title", "bad-enum", "short-key", "bad-key-chars", "missing-required"],
)
async def test_input_validation(local_client: Client, args: dict, needle: str) -> None:
    r = await local_client.call_tool("create_ticket", args, raise_on_error=False)
    assert r.is_error and needle in text_of(r)


async def test_ticket_id_must_be_positive(local_client: Client) -> None:
    r = await local_client.call_tool("get_ticket", {"ticket_id": 0}, raise_on_error=False)
    assert r.is_error and "greater than or equal to 1" in text_of(r)


async def test_version_conflict(local_client: Client) -> None:
    await local_client.call_tool(
        "update_ticket", {"ticket_id": 1, "status": "in_progress", "expected_version": 1}
    )
    r = await local_client.call_tool(
        "update_ticket",
        {"ticket_id": 1, "status": "resolved", "expected_version": 1},
        raise_on_error=False,
    )
    assert r.is_error and text_of(r).startswith("[version_conflict]")


async def test_update_with_nothing_to_change(local_client: Client) -> None:
    r = await local_client.call_tool("update_ticket", {"ticket_id": 1}, raise_on_error=False)
    assert r.is_error and "Nothing to update" in text_of(r)


async def test_idempotency_key_reused_for_different_request(local_client: Client) -> None:
    key = f"k-{uuid.uuid4()}"
    base = {"title": "Laptop fan loud", "description": "Fan is very loud.", "idempotency_key": key}
    await local_client.call_tool("create_ticket", base)
    r = await local_client.call_tool(
        "create_ticket", {**base, "description": "Something else entirely."}, raise_on_error=False
    )
    assert r.is_error and text_of(r).startswith("[idempotency_conflict]")


async def test_tampered_cursor_rejected(local_client: Client) -> None:
    page = (await local_client.call_tool("search_tickets", {"limit": 1})).structured_content
    body, sig = page["next_cursor"].split(".")
    forged = f"{body}.{'0' * len(sig)}"
    r = await local_client.call_tool(
        "search_tickets", {"limit": 1, "cursor": forged}, raise_on_error=False
    )
    assert r.is_error and "Invalid cursor" in text_of(r)


async def test_cursor_bound_to_its_query(local_client: Client) -> None:
    page = (await local_client.call_tool("search_tickets", {"limit": 1})).structured_content
    r = await local_client.call_tool(
        "search_tickets",
        {"limit": 1, "status": "open", "cursor": page["next_cursor"]},
        raise_on_error=False,
    )
    assert r.is_error and "does not belong to this query" in text_of(r)


async def test_unexpected_errors_are_masked(
    local_app: HelpdeskApp, local_client: Client, monkeypatch
) -> None:
    async def boom(*_a, **_k):
        raise RuntimeError("password=hunter2 at db-primary-7")

    monkeypatch.setattr(local_app.repo, "search", boom)
    r = await local_client.call_tool("search_tickets", {}, raise_on_error=False)
    assert r.is_error
    assert "hunter2" not in text_of(r) and "db-primary" not in text_of(r)


async def test_unknown_tool(local_client: Client) -> None:
    r = await local_client.call_tool_mcp("drop_all_tables", {})
    assert r.is_error
