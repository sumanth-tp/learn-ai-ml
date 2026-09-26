"""Every tool, happy path, through an in-memory MCP client (local admin identity)."""

from __future__ import annotations

import uuid

from fastmcp import Client

EXPECTED_TOOLS = {
    "create_ticket",
    "get_ticket",
    "search_tickets",
    "update_ticket",
    "assign_ticket",
    "add_comment",
    "delete_ticket",
    "suggest_triage",
    "incident_report",
}


async def _new_ticket(c: Client, **kw) -> dict:
    args = {
        "title": "Printer on floor 3 jams",
        "description": "Every job jams the printer.",
        "category": "hardware",
        **kw,
    }
    return (await c.call_tool("create_ticket", args)).structured_content


async def test_all_tools_listed_with_annotations(local_client: Client) -> None:
    tools = {t.name: t for t in await local_client.list_tools()}
    assert set(tools) == EXPECTED_TOOLS
    assert tools["delete_ticket"].annotations.destructive_hint is True
    assert tools["search_tickets"].annotations.read_only_hint is True
    assert tools["create_ticket"].annotations.idempotent_hint is True
    assert tools["suggest_triage"].annotations.open_world_hint is True
    # Descriptions are written for the model and must survive into the schema.
    assert "Search first" in tools["create_ticket"].description
    assert tools["create_ticket"].input_schema["properties"]["priority"]["enum"] == [
        "p1",
        "p2",
        "p3",
        "p4",
    ]


async def test_create_and_get(local_client: Client) -> None:
    created = await _new_ticket(local_client, priority="p2")
    assert created["status"] == "open" and created["priority"] == "p2"
    assert created["uri"] == f"helpdesk://tickets/{created['id']}"
    got = await local_client.call_tool("get_ticket", {"ticket_id": created["id"]})
    assert got.structured_content["title"] == "Printer on floor 3 jams"
    assert got.structured_content["comments"] == []


async def test_search_filters(local_client: Client) -> None:
    res = await local_client.call_tool("search_tickets", {"query": "vpn"})
    titles = [t["title"] for t in res.structured_content["items"]]
    assert titles == ["VPN down for the whole office"]
    res = await local_client.call_tool("search_tickets", {"status": "resolved"})
    assert [t["status"] for t in res.structured_content["items"]] == ["resolved"]


async def test_update_assign_comment(local_client: Client) -> None:
    t = await _new_ticket(local_client)
    up = await local_client.call_tool(
        "update_ticket", {"ticket_id": t["id"], "priority": "p1", "expected_version": 1}
    )
    assert up.structured_content["priority"] == "p1"
    assert up.structured_content["version"] == 2
    asg = await local_client.call_tool("assign_ticket", {"ticket_id": t["id"], "assignee": "sam"})
    assert asg.structured_content["assignee"] == "sam"
    assert asg.structured_content["status"] == "in_progress"
    com = await local_client.call_tool(
        "add_comment",
        {"ticket_id": t["id"], "body": "On it.", "idempotency_key": f"k-{uuid.uuid4()}"},
    )
    assert com.structured_content["body"] == "On it."
    detail = await local_client.call_tool("get_ticket", {"ticket_id": t["id"]})
    assert len(detail.structured_content["comments"]) == 1


async def test_suggest_triage_uses_model_and_kb(local_client: Client) -> None:
    t = await _new_ticket(
        local_client,
        title="VPN keeps failing",
        description="The VPN client cannot connect from home.",
    )
    res = await local_client.call_tool("suggest_triage", {"ticket_id": t["id"]})
    s = res.structured_content
    assert s["category"] == "network" and s["source"] == "llm"
    assert "vpn-troubleshooting" in s["suggested_kb_slugs"]


async def test_incident_report(local_client: Client) -> None:
    res = await local_client.call_tool("incident_report", {"window_hours": 48})
    r = res.structured_content
    assert r["tickets_scanned"] == 4  # acme seed tickets only
    assert r["top_category"] in {"network", "access", "email", "hardware"}
    assert len(r["open_p1_ids"]) == 1


async def test_delete_two_step_without_elicitation(local_client: Client) -> None:
    t = await _new_ticket(local_client)
    first = await local_client.call_tool("delete_ticket", {"ticket_id": t["id"]})
    body = first.structured_content
    assert body["status"] == "confirmation_required" and body["confirm_token"]
    # Nothing is deleted until the token comes back.
    assert (await local_client.call_tool("get_ticket", {"ticket_id": t["id"]})).structured_content
    done = await local_client.call_tool(
        "delete_ticket", {"ticket_id": t["id"], "confirm_token": body["confirm_token"]}
    )
    assert done.structured_content["status"] == "deleted"
    gone = await local_client.call_tool("get_ticket", {"ticket_id": t["id"]}, raise_on_error=False)
    assert gone.is_error
