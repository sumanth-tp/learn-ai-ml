"""Integration: the whole pipeline, the HTTP API and the CLI, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelsel.api import create_app
from modelsel.cli import main
from modelsel.config import Settings
from modelsel.llm.registry import load_catalogue
from modelsel.pipeline import run_selection
from modelsel.store import RunStore


async def test_full_run_end_to_end(settings: Settings) -> None:
    s = settings.model_copy(update={"allow_private": True})
    store = RunStore(s.db_path)
    summary = await run_selection(s, store, include_private=True, run_id="run-e2e")

    rec = summary["recommendation"]["model_id"]
    assert rec is not None and rec != "fake:leaky-tuned", "a contaminated model must never be recommended"
    assert summary["contamination"]["fake:leaky-tuned"]["probe_flagged"]
    assert summary["contamination"]["fake:leaky-tuned"]["gap_flagged"]
    by_id = {m["model_id"]: m for m in summary["models"]}
    assert not by_id["fake:local-8b"]["passes"], "local model fails the JSON validity gate"
    assert set(summary["tests_vs_baseline"]) == set(by_id) - {summary["baseline"]}
    assert summary["sample_size"]["n_needed_composite"] > 0
    assert summary["calibration"]["trusted"]

    run = store.get_run("run-e2e")
    assert run is not None and run["status"] == "succeeded" and run["dataset_hash"] == summary["dataset_hash"]
    assert store.count_predictions("run-e2e") == 5 * 110 * 3
    assert "Recommendation" in (s.reports_dir / "run-e2e.md").read_text()
    assert "<html" in (s.reports_dir / "latest.html").read_text()

    # Re-running is free: everything is served from the cache.
    again = await run_selection(s, store, include_private=True, run_id="run-e2e-2")
    assert again["billed_usd"] == 0.0
    assert again["recommendation"] == summary["recommendation"]
    store.close()


async def test_adding_a_model_only_pays_for_the_new_model(settings: Settings, tmp_path: Path) -> None:
    store = RunStore(settings.db_path)
    first = ["fake:balanced-mini", "fake:local-8b"]
    await run_selection(settings, store, candidates=first, run_id="r1")
    second = await run_selection(settings, store, candidates=[*first, "fake:frontier-large"], run_id="r2")
    cat = load_catalogue(settings.models_file)
    assert second["billed_usd"] > 0
    costs = {m["model_id"]: m["cost_per_1k_usd"] for m in second["models"]}
    # billed spend is the new model's candidate calls plus judge calls on its replies, nothing for the old ones
    assert costs["fake:frontier-large"] > 0 and cat.spec("fake:local-8b").input_per_mtok == 0
    store.close()


def test_run_marks_failure_in_store(settings: Settings) -> None:
    import asyncio

    bad = settings.model_copy(update={"models_file": settings.data_dir / "missing.toml"})
    store = RunStore(bad.db_path)
    store.create_run("r-bad", "offline", ["test"])
    with pytest.raises(FileNotFoundError):
        asyncio.run(run_selection(bad, store, run_id="r-bad"))
    store.close()


def test_api_lifecycle(settings: Settings) -> None:
    app = create_app(settings)
    with TestClient(app) as http:
        assert http.get("/healthz").json() == {"status": "ok"}
        r = http.post("/runs", json={"models": ["fake:balanced-mini", "fake:frontier-large"]}, headers={"Idempotency-Key": "k1"})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        # TestClient runs background tasks before returning, so the run is finished here
        body = http.get(f"/runs/{run_id}").json()
        assert body["status"] == "succeeded" and body["summary"]["recommendation"]["model_id"]
        assert "<html" in http.get(f"/runs/{run_id}/report").text
        again = http.post("/runs", json={}, headers={"Idempotency-Key": "k1"})
        assert again.json()["run_id"] == run_id, "same idempotency key, same run"
        assert http.post("/runs", json={"include_private": True}).status_code == 403
        assert http.get("/runs/nope").status_code == 404
        assert any(x["run_id"] == run_id for x in http.get("/runs").json())


def test_api_report_conflict_while_running(settings: Settings) -> None:
    app = create_app(settings)
    with TestClient(app) as http:
        app.state.store.create_run("r-queued", "offline", ["test"])
        assert http.get("/runs/r-queued/report").status_code == 409


def test_cli_sample_size(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sample-size", "--delta", "0.1", "--sd", "0.5", "--discordant", "0.2", "--n", "80"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_for_mean_diff"] == 197 and out["n_for_accuracy_mcnemar"] > 0 and out["mde_at_n"] > 0.1


def test_cli_refuses_private_without_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELSEL_ALLOW_PRIVATE", "false")
    assert main(["run", "--include-private"]) == 2
