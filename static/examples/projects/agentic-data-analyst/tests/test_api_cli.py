from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from data_analyst.api.app import create_app
from data_analyst.cli import app as cli_app
from data_analyst.service import AnalystService


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    s = settings.model_copy(update={"chart_enabled": False})
    svc = AnalystService.from_settings(s, persistent=False)
    with TestClient(create_app(s, svc)) as c:
        yield c


def test_health_ready_metrics_and_ui(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    assert "Data analyst agent" in client.get("/").text
    client.post("/v1/threads/m1/ask", json={"question": "How many customers do we have?"})
    assert "analyst_turns_total" in client.get("/metrics").text


def test_ask_streams_events(client: TestClient) -> None:
    r = client.post("/v1/threads/t1/ask", json={"question": "Number of customers by segment"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[-1][0] == "final" and events[-1][1]["status"] == "answered"
    assert any(kind == "progress" for kind, _ in events)


def test_approval_flow_over_http(client: TestClient) -> None:
    q = {"question": "List every product paired with every customer"}
    events = parse_sse(client.post("/v1/threads/t2/ask", json=q).text)
    assert events[-1][0] == "approval_required"
    # a second question on a paused thread is refused
    assert client.post("/v1/threads/t2/ask", json=q).status_code == 409
    r = client.post("/v1/threads/t2/approval", json={"approved": True, "reviewer": "carol"})
    final = parse_sse(r.text)[-1]
    assert final[0] == "final" and final[1]["approval"]["reviewer"] == "carol"
    assert (
        client.post("/v1/threads/t2/approval", json={"approved": True, "reviewer": "x"}).status_code
        == 409
    )


def test_history_and_state(client: TestClient) -> None:
    client.post("/v1/threads/t3/ask", json={"question": "How many customers do we have?"})
    hist = client.get("/v1/threads/t3/history").json()
    assert len(hist) > 5 and hist[0]["status"] == "answered"
    assert client.get("/v1/threads/t3").json()["values"]["status"] == "answered"


def test_input_validation(client: TestClient) -> None:
    assert client.post("/v1/threads/bad%20id/ask", json={"question": "hello"}).status_code == 422
    assert client.post("/v1/threads/t4/ask", json={"question": "x"}).status_code == 422


def test_api_key_is_enforced(settings) -> None:
    from data_analyst.config import Settings

    s = Settings(
        _env_file=None,
        data_dir=settings.data_dir,
        api_key="s3cret",  # type: ignore[call-arg]
        chart_enabled=False,
    )
    svc = AnalystService.from_settings(s, persistent=False)
    with TestClient(create_app(s, svc)) as c:
        body = {"question": "How many customers do we have?"}
        assert c.post("/v1/threads/a/ask", json=body).status_code == 401
        assert (
            c.post("/v1/threads/a/ask", json=body, headers={"X-API-Key": "wrong"}).status_code
            == 401
        )
        assert (
            c.post("/v1/threads/a/ask", json=body, headers={"X-API-Key": "s3cret"}).status_code
            == 200
        )


def test_cli_seed_ask_and_history(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYST_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("ANALYST_CHART_ENABLED", "false")
    runner = CliRunner()
    assert "exists" in runner.invoke(cli_app, ["seed"]).output
    r = runner.invoke(cli_app, ["ask", "Total revenue in 2024", "--thread", "c1"])
    assert r.exit_code == 0, r.output
    assert "[answered]" in r.output and "11064659.76" in r.output
    r = runner.invoke(
        cli_app,
        ["ask", "List every product paired with every customer", "--thread", "c2", "--auto-reject"],
    )
    assert "[rejected]" in r.output
    r = runner.invoke(cli_app, ["history", "c1"])
    assert r.exit_code == 0 and "step=" in r.output
