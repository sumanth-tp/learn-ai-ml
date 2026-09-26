"""Whole-system tests: a real stdio subprocess, and the full HTTP demo on a real port."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from helpdesk_mcp.db.migrate import upgrade
from helpdesk_mcp.demo import run_demo


async def test_stdio_server_subprocess(tmp_path: Path) -> None:
    """What Claude Desktop does: spawn the server and talk MCP over stdin/stdout."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'stdio.db'}"
    await asyncio.to_thread(upgrade, url)
    env = {
        **os.environ,
        "HELPDESK_DATABASE_URL": url,
        "HELPDESK_LOCAL_USER": "desk-user",
        "HELPDESK_LOCAL_TENANT": "acme",
        "HELPDESK_LOCAL_ROLES": '["agent"]',
        "HELPDESK_LLM_PROVIDER": "fake",
    }
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "helpdesk_mcp.cli", "serve", "--transport", "stdio"],
        env=env,
    )
    async with Client(transport) as c:
        names = {t.name for t in await c.list_tools()}
        assert "update_ticket" in names and "delete_ticket" not in names  # agent role
        created = await c.call_tool(
            "create_ticket",
            {
                "title": "Created over stdio",
                "description": "Spawned like Claude Desktop does.",
            },
        )
        assert created.structured_content["requester"] == "desk-user"
        [me] = await c.read_resource("helpdesk://me")
        assert '"user":"desk-user"' in me.text


async def test_full_demo_over_http() -> None:
    summary = await run_demo()
    assert summary == {
        "ticket_id": summary["ticket_id"],
        "idempotent": True,
        "cross_tenant_blocked": True,
        "open_p1": summary["open_p1"],
        "internal_hidden": True,
        "deleted": True,
    }
    assert len(summary["open_p1"]) == 1
