"""FastAPI endpoints through TestClient (lifespan included)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from research_analyst.api import create_app
from research_analyst.checkpoint import memory_checkpointer
from research_analyst.service import ResearchService
from tests.conftest import SODIUM_Q


@pytest.fixture
def client(settings, deps):
    svc = ResearchService(settings, deps=deps, checkpointer=memory_checkpointer())
    with TestClient(create_app(settings, service=svc)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_create_report_and_fetch_status(client):
    r = client.post("/v1/reports", json={"question": SODIUM_Q, "thread_id": "api-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thread_id"] == "api-1" and body["references"]
    st = client.get("/v1/reports/api-1").json()
    assert st["status"] == "complete" and st["report"]["title"] == body["title"]
    # idempotent replay returns the same report without re-running
    again = client.post("/v1/reports", json={"question": SODIUM_Q, "thread_id": "api-1"})
    assert again.json() == body


def test_stream_report_sse(client):
    with client.stream(
        "POST", "/v1/reports/stream", json={"question": SODIUM_Q, "thread_id": "sse-1"}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]
    assert events[0] == "run" and "plan_ready" in events and events[-1] == "report"
    last_data = [line for line in text.splitlines() if line.startswith("data: ")][-1]
    assert json.loads(last_data.removeprefix("data: "))["thread_id"] == "sse-1"


def test_validation_and_not_found(client):
    assert client.post("/v1/reports", json={"question": "short"}).status_code == 422
    assert (
        client.post("/v1/reports", json={"question": SODIUM_Q, "thread_id": "bad id!"}).status_code
        == 422
    )
    assert client.get("/v1/reports/nope").status_code == 404
    assert client.post("/v1/reports/nope/resume").status_code == 404


def test_api_key_is_enforced(settings, deps):
    s = settings.model_copy(update={"api_key": SecretStr("secret")})
    svc = ResearchService(s, deps=deps, checkpointer=memory_checkpointer())
    with TestClient(create_app(s, service=svc)) as c:
        assert c.post("/v1/reports", json={"question": SODIUM_Q}).status_code == 401
        ok = c.post("/v1/reports", json={"question": SODIUM_Q}, headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        assert c.get("/healthz").status_code == 200  # probes stay unauthenticated


def test_delete_report(client):
    client.post("/v1/reports", json={"question": SODIUM_Q, "thread_id": "del-1"})
    assert client.delete("/v1/reports/del-1").status_code == 204
    assert client.get("/v1/reports/del-1").status_code == 404
    assert client.delete("/v1/reports/del-1").status_code == 404
