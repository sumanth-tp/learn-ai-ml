"""Metric unit tests and the regression gate that CI enforces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from data_analyst.evals import (
    Baseline,
    gate,
    load_baseline,
    load_golden,
    results_match,
    run_eval,
)
from data_analyst.executor import WarehouseExecutor
from data_analyst.llm import OfflineAnalystLLM
from data_analyst.schemas import SQLDraft
from data_analyst.service import AnalystService

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden.jsonl"
BASELINE = ROOT / "evals" / "baseline.json"


@pytest.mark.parametrize(
    ("pred", "gold", "ordered", "expected"),
    [
        ([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], False, True),  # row order ignored
        ([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]], True, False),  # ...unless it matters
        ([["a", 1], ["b", 2]], [[1, "a"], [2, "b"]], True, True),  # column order ignored
        ([[1.004]], [[1.0]], False, True),  # float noise
        ([[1.02]], [[1.0]], False, False),
        ([[1], [1]], [[1]], False, False),  # multiset, not set
        ([[1, 2]], [[1]], False, False),  # extra column
        ([], [], False, True),
        ([["2024-01-01"]], [["2024-01-01"]], False, True),
    ],
)
def test_results_match(pred: Any, gold: Any, ordered: bool, expected: bool) -> None:
    assert results_match(pred, gold, ordered=ordered) is expected


def test_golden_expected_results_match_current_data(settings) -> None:
    """If the seed or views change, the stored answers are stale: fail loudly."""
    ex = WarehouseExecutor(settings.warehouse_path, timeout_s=10, row_cap=1000)
    for case in load_golden(GOLDEN):
        res = ex.run(case.sql)
        assert case.expected is not None
        assert results_match(res.rows, case.expected["rows"], ordered=case.ordered), case.id


def test_gate_reports_each_violation() -> None:
    base = Baseline(
        min_execution_accuracy=0.9,
        min_validity_rate=1.0,
        max_mean_retries=0.5,
        max_p95_latency_ms=100,
        max_cost_per_question_usd=0.001,
        passing_ids=["q01", "q02"],
    )
    from data_analyst.evals import EvalReport

    rep = EvalReport(
        n=2,
        execution_accuracy=0.5,
        validity_rate=0.5,
        mean_retries=1.0,
        p50_latency_ms=10,
        p95_latency_ms=500,
        total_cost_usd=0.01,
        cost_per_question_usd=0.005,
        passing_ids=["q01"],
        cases=[],
    )
    problems = gate(rep, base)
    assert len(problems) == 6 and any("q02" in p for p in problems)


def test_regression_gate_passes_for_current_system(settings) -> None:
    """The CI gate: the offline system must meet the committed baseline."""
    svc = AnalystService.from_settings(
        settings.model_copy(update={"cache_enabled": False, "chart_enabled": False}),
        persistent=False,
    )
    report = run_eval(svc, load_golden(GOLDEN))
    assert gate(report, load_baseline(BASELINE)) == []
    assert report.execution_accuracy >= 0.9 and report.validity_rate == 1.0
    assert "q15" not in report.passing_ids  # the known-wrong offline answer is detected


class DegradedLLM(OfflineAnalystLLM):
    """Simulates a bad prompt change: every query forgets the status filter."""

    def _sql(self, h: Any) -> SQLDraft:
        draft = super()._sql(h)
        return SQLDraft(
            sql=draft.sql.replace("o.status = 'completed' AND", "").replace(
                "WHERE o.status = 'completed'", "WHERE 1 = 1"
            ),
            explanation=draft.explanation,
        )


def test_regression_gate_catches_a_degraded_model(settings) -> None:
    llm = DegradedLLM(OfflineAnalystLLM.from_package().script)
    svc = AnalystService.from_settings(
        settings.model_copy(update={"cache_enabled": False, "chart_enabled": False}),
        llm=llm,
        persistent=False,
    )
    report = run_eval(svc, load_golden(GOLDEN))
    problems = gate(report, load_baseline(BASELINE))
    assert any("previously passing" in p for p in problems)
    assert any("execution accuracy" in p for p in problems)
