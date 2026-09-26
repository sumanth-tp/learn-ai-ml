"""Health, readiness, metrics and request-id propagation."""

from __future__ import annotations

from fastmcp.utilities.tests import ASGIServer

from helpdesk_mcp.server import HelpdeskApp

BASE = "http://127.0.0.1"


async def test_health_and_readiness(http: ASGIServer) -> None:
    async with http.http_client() as h:
        live = await h.get(f"{BASE}/healthz")
        ready = await h.get(f"{BASE}/readyz")
    assert live.status_code == 200 and live.json()["status"] == "ok"
    assert ready.status_code == 200 and ready.json()["database"] == "up"


async def test_readiness_fails_when_database_is_gone(http: ASGIServer, monkeypatch) -> None:
    async def db_down(_engine) -> bool:
        raise OSError("connection refused")

    monkeypatch.setattr("helpdesk_mcp.server.ping", db_down)
    async with http.http_client() as h:
        ready = await h.get(f"{BASE}/readyz")
        live = await h.get(f"{BASE}/healthz")
    assert ready.status_code == 503 and ready.json()["database"] == "down"
    assert live.status_code == 200  # liveness must not depend on the DB, or k8s restarts pods


async def test_metrics_count_calls_errors_and_latency(http: ASGIServer, as_user) -> None:
    async with as_user("sam", role="agent") as c:
        await c.call_tool("search_tickets", {})
        await c.call_tool("get_ticket", {"ticket_id": 999}, raise_on_error=False)
    async with http.http_client() as h:
        text = (await h.get(f"{BASE}/metrics")).text
    assert (
        'helpdesk_mcp_requests_total{component="search_tickets",method="tools/call",'
        'outcome="ok"} 1.0'
    ) in text
    assert (
        'helpdesk_mcp_errors_total{component="get_ticket",error="not_found",'
        'method="tools/call"} 1.0'
    ) in text
    assert 'helpdesk_mcp_request_duration_seconds_bucket{component="search_tickets"' in text


async def test_request_id_header_reaches_audit_log(
    app: HelpdeskApp, http: ASGIServer, token
) -> None:
    async with http.client(
        auth=token("sam", role="agent"), headers={"X-Request-ID": "req-abc-123"}
    ) as c:
        await c.call_tool("assign_ticket", {"ticket_id": 1, "assignee": "sam"})
    events = await app.repo.audit_events("acme")
    assert events[-1].request_id == "req-abc-123"
