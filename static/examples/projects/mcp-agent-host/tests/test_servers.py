"""The demo servers on their own, through a real MCP client session in memory."""

from __future__ import annotations

from datetime import timedelta

from mcp.shared.memory import create_connected_server_and_client_session

from demo_servers import calendar_server, docs_server, notes_server
from tests.conftest import ROOT


async def test_notes_refuses_path_traversal(data_dirs):
    server = notes_server.create_server(data_dirs["notes"])
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("read_note", {"name": "../../etc/passwd"})
        assert result.isError and "invalid note name" in result.content[0].text


async def test_notes_write_is_idempotent(data_dirs):
    server = notes_server.create_server(data_dirs["notes"])
    async with create_connected_server_and_client_session(server) as client:
        first = await client.call_tool("write_note", {"name": "n1", "content": "hello"})
        second = await client.call_tool("write_note", {"name": "n1", "content": "hello"})
        assert first.structuredContent["status"] == "created"
        assert second.structuredContent["status"] == "unchanged"


async def test_calendar_create_event_idempotency_key(tmp_path):
    server = calendar_server.create_server(
        tmp_path / "c.json", ROOT / "data" / "calendar_seed.json"
    )
    async with create_connected_server_and_client_session(server) as client:
        args = {"title": "Retro", "start": "2026-10-06T15:00:00+00:00", "idempotency_key": "k-1"}
        a = await client.call_tool("create_event", args)
        b = await client.call_tool("create_event", args)  # a retry after a timeout
        assert a.structuredContent["id"] == b.structuredContent["id"]
        listed = await client.call_tool("list_events", {"day": "2026-10-06"})
        assert len(listed.structuredContent["result"]) == 1


async def test_calendar_free_slot_skips_meetings(tmp_path):
    server = calendar_server.create_server(
        tmp_path / "c.json", ROOT / "data" / "calendar_seed.json"
    )
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(
            "find_free_slot", {"day": "2026-10-05", "duration_minutes": 45}
        )
        assert result.structuredContent["result"] == "2026-10-05T09:15:00+00:00"


async def test_summarise_day_falls_back_without_sampling(tmp_path):
    server = calendar_server.create_server(
        tmp_path / "c.json", ROOT / "data" / "calendar_seed.json"
    )
    async with create_connected_server_and_client_session(
        server, read_timeout_seconds=timedelta(seconds=5)
    ) as client:
        result = await client.call_tool("summarise_day", {"day": "2026-10-05"})
        assert "Team stand-up" in result.structuredContent["result"]


async def test_docs_search_ranks_the_right_section():
    server = docs_server.create_server(ROOT / "data" / "docs")
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("search", {"query": "meal allowance", "k": 1})
        hit = result.structuredContent["result"][0]
        assert (hit["slug"], hit["section"]) == ("expenses-policy", "Meals")
        index = await client.read_resource("docs://index")
        assert "leave-policy" in index.contents[0].text
        prompt = await client.get_prompt("answer_with_citations", {"question": "q?"})
        assert "docs__search" in prompt.messages[0].content.text
