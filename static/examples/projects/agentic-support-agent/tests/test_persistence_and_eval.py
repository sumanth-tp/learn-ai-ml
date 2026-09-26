"""Durable checkpoints across a restart, and the offline eval regression gate."""

from __future__ import annotations

import json
from pathlib import Path

from support_agent.config import Settings
from support_agent.container import build_container
from support_agent.evaluation import run_eval
from support_agent.graph import ApprovalDecision, build_graph
from support_agent.persistence import open_persistence
from support_agent.runner import SupportRunner

ROOT = Path(__file__).resolve().parents[1]


async def test_sqlite_checkpoints_survive_a_restart(settings: Settings) -> None:
    s = settings.model_copy(update={"checkpoint_backend": "sqlite"})
    c = build_container(s)
    async with open_persistence(s) as (saver, store):
        runner = SupportRunner(c, build_graph(c.deps, saver, store))
        done = await runner.run_turn("t1", "cust_001", "Refund ORD-1002 please")
        assert done["interrupted"]
    # "Restart": new connections, new graph, same files. The approval is still waiting.
    async with open_persistence(s) as (saver, store):
        runner = SupportRunner(c, build_graph(c.deps, saver, store))
        assert await runner.pending_interrupts("t1")
        done = await runner.resume("t1", ApprovalDecision(approved=True, reviewer="lead"))
        assert "refunded 249.00" in done["answer"]
    c.engine.dispose()


async def test_offline_eval_passes_regression_gate(settings: Settings) -> None:
    report = await run_eval(ROOT / "evals" / "dataset.jsonl", settings)
    thresholds = json.loads((ROOT / "evals" / "thresholds.json").read_text())
    assert report.gate(thresholds) == [], [r.failures for r in report.results if not r.success]
    assert len(report.results) >= 20


def test_gate_reports_regressions() -> None:
    from support_agent.evaluation import CaseResult, EvalReport

    bad = EvalReport([CaseResult("x", False, True, False, "faq", [], "")])
    assert bad.gate({"routing_accuracy": 0.9}) == ["routing_accuracy=0.0 < 0.9"]


async def test_retention_purge_and_user_erasure(runner: SupportRunner) -> None:
    await runner.run_turn("t1", "cust_001", "Please call me Asha. Where is ORD-1003?")
    await runner.run_turn("t2", "cust_002", "Where is ORD-2001?")
    assert await runner.purge_threads(older_than_days=30) == 0  # nothing old yet
    result = await runner.forget_user("cust_001")
    assert result["threads"] == 1 and result["memories"] >= 1
    assert (await runner.history("t1"))["messages"] == []
    assert (await runner.history("t2"))["messages"]  # other customers untouched
    assert await runner.purge_threads(older_than_days=-1) == 1  # everything is "old"
