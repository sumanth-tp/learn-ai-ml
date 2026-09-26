import pytest
from fastapi.testclient import TestClient

from agentmon.api import create_app


@pytest.fixture
def client(settings, rt):
    app = create_app(settings.model_copy(update={"inprocess_workers": True}), runtime=rt)
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["provider"] == "fake"


def test_chat_trace_feedback_review_flow(client) -> None:
    r = client.post(
        "/v1/chat",
        json={
            "user_id": "CUST-1",
            "message": "What's the balance on ACC-1001?",
            "request_id": "abc",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "4,210.55" in body["answer"]
    again = client.post("/v1/chat", json={"user_id": "CUST-1", "message": "x", "request_id": "abc"})
    assert again.json()["replayed"] is True and again.json()["trace_id"] == body["trace_id"]

    t = client.get(f"/v1/traces/{body['trace_id']}").json()
    assert t["trace"]["tool_calls"][0]["name"] == "get_balance"
    assert any(s["name"] == "tool.call" for s in t["spans"])

    fb = client.post(
        "/v1/feedback", json={"trace_id": body["trace_id"], "rating": -1, "correction": "wrong"}
    )
    assert fb.status_code == 202
    queue = client.get("/v1/review").json()
    assert queue[0]["trace_id"] == body["trace_id"] and queue[0]["draft"]["expected_calls"]
    ok = client.post(f"/v1/review/{body['trace_id']}/approve", json={"reviewer": "sam"})
    assert ok.json()["status"] == "approved"


def test_validation_and_not_found(client) -> None:
    assert client.post("/v1/chat", json={"user_id": "bob", "message": "hi"}).status_code == 422
    assert client.post("/v1/feedback", json={"trace_id": "nope", "rating": 1}).status_code == 404
    assert client.get("/v1/traces/nope").status_code == 404
    assert client.post("/v1/feedback", json={"trace_id": "x", "rating": 3}).status_code == 422


def test_metrics_alerts_dashboard(client, rt) -> None:
    client.post("/v1/chat", json={"user_id": "CUST-1", "message": "hi"})
    rt.clock.advance(60)
    metrics = client.get("/v1/metrics", params={"window_hours": 1, "lookback_hours": 2}).json()
    assert sum(m["n"] for m in metrics) == 1
    assert client.post("/v1/alerts/evaluate").status_code == 200
    assert client.get("/v1/alerts").json() == []
    page = client.get("/dashboard")
    assert page.status_code == 200 and "<svg" in page.text
