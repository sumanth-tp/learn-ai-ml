"""Multi-tenant isolation and per-role data scoping."""

from __future__ import annotations

import json

import pytest
from fastmcp.exceptions import ClientError
from mcp import MCPError

from tests.conftest import text_of

# Seed: acme has tickets 1-4 (alice: 1 and 3, bob: 2, carol: 4); globex has 5 (dave).


async def test_other_tenant_ticket_looks_missing(as_user) -> None:
    async with as_user("dave", tenant="globex", role="admin") as c:
        r = await c.call_tool("get_ticket", {"ticket_id": 1}, raise_on_error=False)
        assert r.is_error and text_of(r) == "[not_found] Ticket 1 not found."
        r = await c.call_tool(
            "update_ticket", {"ticket_id": 1, "status": "closed"}, raise_on_error=False
        )
        assert r.is_error and "[not_found]" in text_of(r)
        r = await c.call_tool("delete_ticket", {"ticket_id": 1}, raise_on_error=False)
        assert r.is_error and "[not_found]" in text_of(r)
        with pytest.raises((MCPError, ClientError)):
            await c.read_resource("helpdesk://tickets/1")


async def test_search_never_crosses_tenants(as_user) -> None:
    async with as_user("dave", tenant="globex", role="admin") as c:
        items = (await c.call_tool("search_tickets", {})).structured_content["items"]
    assert [t["id"] for t in items] == [5]


async def test_requester_sees_only_own_tickets(as_user) -> None:
    async with as_user("alice", role="requester") as c:
        mine = (await c.call_tool("search_tickets", {})).structured_content["items"]
        assert {t["requester"] for t in mine} == {"alice"}
        r = await c.call_tool("get_ticket", {"ticket_id": 2}, raise_on_error=False)  # bob's
        assert r.is_error and "[not_found]" in text_of(r)


async def test_agent_sees_whole_tenant(as_user) -> None:
    async with as_user("sam", role="agent") as c:
        items = (await c.call_tool("search_tickets", {})).structured_content["items"]
    assert {t["id"] for t in items} == {1, 2, 3, 4}


async def test_internal_comments_hidden_from_requester(as_user) -> None:
    async with as_user("sam", role="agent") as c:
        await c.call_tool(
            "add_comment", {"ticket_id": 1, "body": "Reset via AD.", "internal": True}
        )
        await c.call_tool("add_comment", {"ticket_id": 1, "body": "Try again now please."})
    async with as_user("alice", role="requester") as c:
        detail = (await c.call_tool("get_ticket", {"ticket_id": 1})).structured_content
        [content] = await c.read_resource("helpdesk://tickets/1")
    assert [cm["body"] for cm in detail["comments"]] == ["Try again now please."]
    assert len(json.loads(content.text)["comments"]) == 1


async def test_requester_cannot_post_internal_note(as_user) -> None:
    async with as_user("alice", role="requester") as c:
        r = await c.call_tool(
            "add_comment", {"ticket_id": 1, "body": "x", "internal": True}, raise_on_error=False
        )
    assert r.is_error and "[permission_denied]" in text_of(r)


async def test_kb_is_per_tenant(app, as_user) -> None:
    from helpdesk_mcp.db.models import KBArticle
    from helpdesk_mcp.db.session import make_session_factory

    async with make_session_factory(app.engine).begin() as s:
        s.add(
            KBArticle(
                tenant_id="globex",
                slug="globex-only",
                title="Globex secret",
                body="internal",
                category="other",
            )
        )
    async with as_user("alice", role="requester") as c:
        [index] = await c.read_resource("helpdesk://kb/articles")
        with pytest.raises((MCPError, ClientError)):
            await c.read_resource("helpdesk://kb/articles/globex-only")
    assert "globex-only" not in index.text


async def test_audit_trail_records_actor_and_tenant(app, as_user) -> None:
    async with as_user("sam", role="agent") as c:
        await c.call_tool("assign_ticket", {"ticket_id": 2, "assignee": "sam"})
    events = await app.repo.audit_events("acme")
    assert events[-1].action == "ticket.update" and events[-1].actor == "sam"
    assert await app.repo.audit_events("globex") == []
