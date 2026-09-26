"""Tool schemas are a public API: the snapshot gate catches breaking changes."""

from __future__ import annotations

import copy
from pathlib import Path

from helpdesk_mcp.schema_compat import breaking_changes, load, snapshot
from helpdesk_mcp.server import HelpdeskApp

SNAPSHOT = Path(__file__).parent / "snapshots" / "tool_schemas.json"


async def test_no_breaking_changes_against_committed_snapshot(local_app: HelpdeskApp) -> None:
    current = await snapshot(local_app.mcp)
    problems = breaking_changes(load(SNAPSHOT), current)
    assert problems == [], (
        "Breaking tool-schema change. Either keep it compatible, or add a new tool "
        "version and regenerate with `make schema-snapshot`:\n" + "\n".join(problems)
    )


def _sample() -> dict:
    return {
        "search_tickets": {
            "input": {
                "type": "object",
                "properties": {
                    "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "status": {
                        "anyOf": [{"enum": ["open", "closed"], "type": "string"}, {"type": "null"}]
                    },
                },
                "required": [],
            },
            "output": {"type": "object", "properties": {"items": {"type": "array"}}},
            "annotations": {"read_only_hint": True},
        },
        "delete_ticket": {
            "input": {
                "type": "object",
                "properties": {"ticket_id": {"type": "integer"}},
                "required": ["ticket_id"],
            },
            "output": None,
            "annotations": {"destructive_hint": True},
        },
    }


def test_additive_changes_are_allowed() -> None:
    old, new = _sample(), _sample()
    new["search_tickets"]["input"]["properties"]["assignee"] = {"type": "string"}
    new["search_tickets"]["output"]["properties"]["total"] = {"type": "integer"}
    new["create_ticket"] = {"input": {}, "output": None, "annotations": {}}
    assert breaking_changes(old, new) == []


def test_breaking_changes_are_detected() -> None:
    old = _sample()
    new = copy.deepcopy(old)
    del new["search_tickets"]["input"]["properties"]["query"]
    new["search_tickets"]["input"]["required"] = ["status"]
    new["search_tickets"]["input"]["properties"]["status"]["anyOf"][0]["enum"] = ["open"]
    del new["search_tickets"]["output"]["properties"]["items"]
    new["delete_ticket"]["annotations"] = {}
    problems = breaking_changes(old, new)
    assert "search_tickets: input 'query' removed" in problems
    assert "search_tickets: input 'status' became required" in problems
    assert any("lost enum values" in p for p in problems)
    assert "search_tickets: output 'items' removed" in problems
    assert "delete_ticket: destructive_hint removed" in problems


def test_removed_tool_is_breaking() -> None:
    old = _sample()
    new = {"search_tickets": old["search_tickets"]}
    assert breaking_changes(old, new) == ["delete_ticket: tool removed"]
