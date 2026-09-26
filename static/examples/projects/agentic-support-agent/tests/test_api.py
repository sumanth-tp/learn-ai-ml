"""HTTP tests: auth, SSE framing, resume semantics, history, time travel, metrics."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from support_agent.api.app import create_app
from support_agent.config import Settings

CUSTOMER = {"Authorization": "Bearer dev-customer-token", "X-User-Id": "cust_001"}
REVIEWER = {"Authorization": "Bearer dev-reviewer-token", "X-Reviewer": "lead@example.com"}


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def chat(
    client: httpx.AsyncClient,
    message: str,
    thread_id: str | None = None,
    headers: dict[str, str] = CUSTOMER,
) -> tuple[int, list[tuple[str, dict]]]:
    body: dict[str, Any] = {"message": message}
    if thread_id:
        body["thread_id"] = thread_id
    resp = await client.post("/v1/chat", json=body, headers=headers)
    return resp.status_code, parse_sse(resp.text) if resp.status_code == 200 else []


async def test_health_and_ready(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    assert (await client.get("/readyz")).json() == {"status": "ready"}
    assert "Support agent" in (await client.get("/")).text


async def test_auth_is_required(client: httpx.AsyncClient) -> None:
    assert (await client.post("/v1/chat", json={"message": "hi"})).status_code == 401
    bad = {"Authorization": "Bearer nope", "X-User-Id": "cust_001"}
    assert (await client.post("/v1/chat", json={"message": "hi"}, headers=bad)).status_code == 401
    # A customer token cannot approve refunds.
    assert (await client.get("/v1/approvals", headers=CUSTOMER)).status_code == 401


async def test_chat_streams_sse_events(client: httpx.AsyncClient) -> None:
    status, events = await chat(client, "Where is my order ORD-1003?", "thread_a1")
    assert status == 200
    types = [t for t, _ in events]
    assert types[0] == "metadata" and types[-1] == "done"
    assert "token" in types and "update" in types
    assert "RM555GB" in events[-1][1]["answer"]


async def test_refund_approval_flow_over_http(client: httpx.AsyncClient) -> None:
    _, events = await chat(client, "I want a refund for ORD-1002", "thread_r1")
    assert any(t == "interrupt" for t, _ in events)
    queue = (await client.get("/v1/approvals", headers=REVIEWER)).json()
    assert queue[0]["thread_id"] == "thread_r1"

    resp = await client.post(
        "/v1/threads/thread_r1/resume",
        headers=REVIEWER,
        json={"approved": True, "interrupt_id": queue[0]["interrupt_id"]},
    )
    assert resp.status_code == 200
    assert "refunded 249.00" in parse_sse(resp.text)[-1][1]["answer"]

    again = await client.post(
        "/v1/threads/thread_r1/resume", headers=REVIEWER, json={"approved": True}
    )
    assert again.status_code == 409  # resumed twice: nothing runs
    assert (await client.get("/v1/approvals", headers=REVIEWER)).json() == []


async def test_history_is_owner_only(client: httpx.AsyncClient) -> None:
    await chat(client, "Where is ORD-1003?", "thread_h1")
    ok = await client.get("/v1/threads/thread_h1/history", headers=CUSTOMER)
    assert ok.status_code == 200 and len(ok.json()["messages"]) >= 2
    other = {**CUSTOMER, "X-User-Id": "cust_002"}
    assert (await client.get("/v1/threads/thread_h1/history", headers=other)).status_code == 403
    status, _ = await chat(client, "hi", "thread_h1", headers=other)
    assert status == 403
    assert (await client.get("/v1/threads/nope/history", headers=CUSTOMER)).status_code == 404


async def test_checkpoints_and_fork_endpoints(client: httpx.AsyncClient) -> None:
    await chat(client, "Where is ORD-1003?", "thread_t1")
    cps = (await client.get("/v1/threads/thread_t1/checkpoints", headers=REVIEWER)).json()
    assert len(cps) > 3 and cps[0]["next"] == []
    start = next(c for c in cps if c["next"] == ["tools"])
    replay = await client.post(
        "/v1/threads/thread_t1/replay",
        headers=REVIEWER,
        json={"checkpoint_id": start["checkpoint_id"]},
    )
    assert "RM555GB" in replay.json()["answer"]
    fork = await client.post(
        "/v1/threads/thread_t1/fork",
        headers=REVIEWER,
        json={"checkpoint_id": cps[-1]["checkpoint_id"], "message": "Where is ORD-1001?"},
    )
    assert "ORD-1001" in fork.json()["answer"]


async def test_metrics_exposed(client: httpx.AsyncClient) -> None:
    await chat(client, "How long do refunds take?", "thread_m1")
    body = (await client.get("/metrics")).text
    assert "support_requests_total" in body and "support_turn_latency_seconds" in body


async def test_validation_errors(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/chat", json={"message": ""}, headers=CUSTOMER)
    assert resp.status_code == 422
    resp = await client.post(
        "/v1/chat", json={"message": "hi", "thread_id": "../etc"}, headers=CUSTOMER
    )
    assert resp.status_code == 422
