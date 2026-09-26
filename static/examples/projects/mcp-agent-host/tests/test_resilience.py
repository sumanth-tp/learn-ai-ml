"""Failure paths: slow servers, crashes, reconnection, degradation, list_changed."""

from __future__ import annotations

import asyncio

import anyio
import pytest
from langchain_core.messages import AIMessage
from mcp.server.fastmcp import FastMCP

from mcp_host.connection import RequestTimeout, ServerConnection, ServerState, ServerUnavailable
from mcp_host.transports import InProcessServer
from tests.conftest import call, collect, servers_file


def slow_server(cancelled: list[str]) -> FastMCP:
    mcp = FastMCP("slow")

    @mcp.tool()
    async def wait(seconds: float) -> str:
        """Sleep, then answer."""
        try:
            await anyio.sleep(seconds)
        except anyio.get_cancelled_exc_class():
            cancelled.append("cancelled")
            raise
        return "finished"

    return mcp


async def until(predicate, timeout: float = 3.0) -> None:
    with anyio.fail_after(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


async def test_slow_tool_times_out_and_server_is_told_to_cancel(settings):
    cancelled: list[str] = []
    server = InProcessServer(slow_server(cancelled))
    conn = ServerConnection("slow", server.connect, settings, timeout_s=0.3)
    conn.start()
    try:
        assert await conn.wait_ready(3)
        with pytest.raises(RequestTimeout):
            await conn.call_tool("wait", {"seconds": 5})
        await until(lambda: cancelled == ["cancelled"])
        # The session is still healthy after a timeout: the next call works.
        result = await conn.call_tool("wait", {"seconds": 0})
        assert result.structuredContent == {"result": "finished"}
    finally:
        await conn.stop()


async def test_host_cancellation_propagates_to_server(settings):
    cancelled: list[str] = []
    server = InProcessServer(slow_server(cancelled))
    conn = ServerConnection("slow", server.connect, settings, timeout_s=10)
    conn.start()
    try:
        assert await conn.wait_ready(3)
        task = asyncio.create_task(conn.call_tool("wait", {"seconds": 5}))
        await asyncio.sleep(0.2)
        task.cancel()  # e.g. the user closed the browser tab
        with pytest.raises(asyncio.CancelledError):
            await task
        await until(lambda: cancelled == ["cancelled"])
    finally:
        await conn.stop()


async def test_crash_mid_call_raises_unavailable_then_reconnects(settings):
    server = InProcessServer(slow_server([]))
    conn = ServerConnection("slow", server.connect, settings, timeout_s=5)
    conn.start()
    try:
        assert await conn.wait_ready(3)
        task = asyncio.create_task(conn.call_tool("wait", {"seconds": 5}))
        await asyncio.sleep(0.2)
        server.crash()
        with pytest.raises(ServerUnavailable):
            await task
        await until(lambda: conn.state == ServerState.DOWN)
        with pytest.raises(ServerUnavailable):
            await conn.call_tool("wait", {"seconds": 0})
        server.revive()
        assert await conn.wait_ready(3)
        assert conn.reconnects == 1
        assert server.connections >= 2
    finally:
        await conn.stop()


async def test_idle_crash_detected_by_ping(settings):
    server = InProcessServer(slow_server([]))
    conn = ServerConnection("slow", server.connect, settings)
    conn.start()
    try:
        assert await conn.wait_ready(3)
        server.crash()
        await until(lambda: conn.state == ServerState.DOWN)
        assert conn.last_error
    finally:
        await conn.stop()


async def test_agent_is_told_which_tools_are_unavailable(make_runtime, inproc):
    seen: list[str] = []

    def spy(messages):
        seen.append(str(messages[0].content))
        return AIMessage(content="The calendar is down, so I cannot check it.")

    async with make_runtime([spy]) as rt:
        inproc["calendar"].crash()
        await until(lambda: "calendar" in rt.host.registry.snapshot.unavailable)
        assert "calendar__list_events" not in rt.host.registry.snapshot.tools
        await collect(rt.service.send("d1", "what's on my calendar?"))
        assert "UNAVAILABLE" in seen[0] and "calendar__list_events" in seen[0]
        # Other servers keep working.
        assert "docs__search" in rt.host.registry.snapshot.tools
        inproc["calendar"].revive()
        await until(lambda: "calendar__list_events" in rt.host.registry.snapshot.tools)


async def test_server_down_at_startup_degrades_instead_of_failing(make_runtime, inproc):
    inproc["docs"].crash()
    async with make_runtime(connect_timeout_s=0.5) as rt:
        assert "docs" in rt.host.registry.snapshot.unavailable
        assert "notes__list_notes" in rt.host.registry.snapshot.tools


async def test_tool_that_vanishes_mid_turn_returns_error_message(make_runtime, inproc):
    def crash_then_call(messages):
        inproc["calendar"].crash()
        return call("calendar__list_events", {"day": "2026-10-05"}, "c1")

    async with make_runtime([crash_then_call, AIMessage(content="sorry")]) as rt:
        events = await collect(rt.service.send("d2", "calendar?"))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["status"] == "error"
        assert "unavailable" in result["preview"] or "NOT EXECUTED" in result["preview"]


async def test_list_changed_adds_tools_to_registry(make_runtime):
    script = [call("notes__enable_tag_tools", {}, "e1"), AIMessage(content="enabled")]
    async with make_runtime(script) as rt:
        assert "notes__tag_note" not in rt.host.registry.snapshot.tools
        await collect(rt.service.send("l1", "turn on tagging"))
        await until(lambda: "notes__tag_note" in rt.host.registry.snapshot.tools)


async def test_capabilities_advertise_list_changed(make_runtime):
    async with make_runtime() as rt:
        caps = rt.host.connections["notes"].capabilities
        assert caps is not None and caps.tools is not None and caps.tools.listChanged is True


async def test_unknown_server_policy_defaults(make_runtime):
    async with make_runtime(servers=servers_file()) as rt:
        entry = rt.host.registry.resolve("notes__list_notes")
        assert entry is not None and entry.read_only
