"""End-to-end demo over real Streamable HTTP on a local port.

Migrates and seeds a throwaway SQLite database, starts the server with
uvicorn, then plays four users through a realistic morning at the helpdesk
with the FastMCP client and signed JWTs. Works offline (fake triage model);
set HELPDESK_LLM_PROVIDER=openai and OPENAI_API_KEY to use a real model.
"""

from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastmcp import Client

from helpdesk_mcp.config import Settings
from helpdesk_mcp.db.migrate import upgrade
from helpdesk_mcp.logging_setup import configure_logging
from helpdesk_mcp.seed import seed
from helpdesk_mcp.server import build_server
from helpdesk_mcp.tokens import mint_token


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _say(step: str, detail: object = "") -> None:
    text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    print(f"\n== {step}\n{text}")


async def _progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"   progress {progress}/{total or '?'} {message or ''}")


async def _server_log(msg) -> None:
    print(f"   server log [{msg.level}] {msg.data}")


async def run_demo(base: Settings | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="helpdesk-demo-"))
    port = _free_port()
    overrides = {
        "database_url": f"sqlite+aiosqlite:///{tmp / 'demo.db'}",
        "port": port,
        "auth_mode": "jwt",
        "transport": "http",
        "log_level": "WARNING",
    }
    settings = (base or Settings()).model_copy(update=overrides)
    configure_logging("WARNING", json=True)  # keep the story readable
    await asyncio.to_thread(upgrade, settings.database_url)
    app = build_server(settings)
    await seed(app.engine)

    server = uvicorn.Server(
        uvicorn.Config(
            app.mcp.http_app(path=settings.mcp_path),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
    )
    serve_task = asyncio.create_task(server.serve())
    # uvicorn exposes startup only as a flag, so poll it (bounded: 10s).
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("demo server did not start within 10s")
    url = f"http://127.0.0.1:{port}{settings.mcp_path}"

    def token(user: str, tenant: str, role: str) -> str:
        return mint_token(settings, user=user, tenant=tenant, roles=[role])

    alice = token("alice", "acme", "requester")
    sam = token("sam", "acme", "agent")
    ada = token("ada", "acme", "admin")
    dave = token("dave", "globex", "requester")
    summary: dict = {}

    try:
        async with Client(url, auth=alice) as c:
            names = sorted(t.name for t in await c.list_tools())
            _say("alice (requester) sees these tools", names)
            key = f"demo-{uuid.uuid4()}"
            args = {
                "title": "Cannot connect to VPN from home",
                "description": "VPN client says certificate expired since this morning.",
                "category": "network",
                "idempotency_key": key,
            }
            first = (await c.call_tool("create_ticket", args)).structured_content
            again = (await c.call_tool("create_ticket", args)).structured_content
            _say(
                "alice creates a ticket, then her client retries with the same key",
                {"first_id": first["id"], "retry_id": again["id"]},
            )
            summary["ticket_id"] = tid = first["id"]
            summary["idempotent"] = first["id"] == again["id"]

        async with Client(url, auth=dave) as c:
            r = await c.call_tool("get_ticket", {"ticket_id": tid}, raise_on_error=False)
            _say("dave (another tenant) asks for alice's ticket", r.content[0].text)
            summary["cross_tenant_blocked"] = r.is_error

        async with Client(url, auth=sam, progress_handler=_progress, log_handler=_server_log) as c:
            page = await c.call_tool("search_tickets", {"status": "open", "limit": 2})
            data = page.structured_content
            _say(
                "sam (agent) searches open tickets, page 1",
                {"ids": [t["id"] for t in data["items"]], "has_next": bool(data["next_cursor"])},
            )
            triage = (await c.call_tool("suggest_triage", {"ticket_id": tid})).structured_content
            _say("sam asks for a triage suggestion", triage)
            current = (await c.call_tool("get_ticket", {"ticket_id": tid})).structured_content
            updated = (
                await c.call_tool(
                    "update_ticket",
                    {
                        "ticket_id": tid,
                        "priority": triage["priority"],
                        "category": triage["category"],
                        "expected_version": current["version"],
                    },
                )
            ).structured_content
            stale = await c.call_tool(
                "update_ticket",
                {
                    "ticket_id": tid,
                    "status": "resolved",
                    "expected_version": current["version"],
                },
                raise_on_error=False,
            )
            _say(
                "sam updates it, then a stale second update is rejected",
                {"new_version": updated["version"], "stale_error": stale.content[0].text},
            )
            await c.call_tool(
                "add_comment",
                {
                    "ticket_id": tid,
                    "body": "Cert re-enrolment needed; see vpn-troubleshooting.",
                    "internal": True,
                    "idempotency_key": f"note-{uuid.uuid4()}",
                },
            )
            await c.call_tool("assign_ticket", {"ticket_id": tid, "assignee": "sam"})
            report = (await c.call_tool("incident_report", {"window_hours": 24})).structured_content
            _say("sam runs the incident report", report)
            prompt = await c.get_prompt("incident_summary", {"window_hours": "24"})
            _say("incident_summary prompt text", prompt.messages[0].content.text[:160] + "...")
            summary["open_p1"] = report["open_p1_ids"]

        async with Client(url, auth=alice) as c:
            mine = (await c.call_tool("get_ticket", {"ticket_id": tid})).structured_content
            _say(
                "alice re-reads her ticket: the internal note is hidden",
                {
                    "status": mine["status"],
                    "assignee": mine["assignee"],
                    "comments_visible": len(mine["comments"]),
                },
            )
            summary["internal_hidden"] = len(mine["comments"]) == 0

        async with Client(url, auth=ada) as c:
            step1 = (await c.call_tool("delete_ticket", {"ticket_id": tid})).structured_content
            _say(
                "ada (admin) asks to delete; her client has no elicitation, so a token",
                {k: step1[k] for k in ("status", "expires_in_s", "message")},
            )
            step2 = (
                await c.call_tool(
                    "delete_ticket",
                    {
                        "ticket_id": tid,
                        "confirm_token": step1["confirm_token"],
                    },
                )
            ).structured_content
            _say("after the user says yes, ada calls again with the token", step2)
            summary["deleted"] = step2["status"] == "deleted"

        async with httpx.AsyncClient() as h:
            ready = (await h.get(f"http://127.0.0.1:{port}/readyz")).json()
            metrics = (await h.get(f"http://127.0.0.1:{port}/metrics")).text
        calls = [
            ln
            for ln in metrics.splitlines()
            if ln.startswith("helpdesk_mcp_requests_total{") and "tools/call" in ln
        ]
        _say("readiness and a slice of /metrics", {"readyz": ready, "tool_call_series": calls[:4]})
    finally:
        server.should_exit = True
        await serve_task
        await app.engine.dispose()
    _say("summary", summary)
    return summary
