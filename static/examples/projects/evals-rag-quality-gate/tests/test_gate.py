import pytest

from ragate.evaluation.gate import (
    ExitCode,
    GateConfig,
    MetricRule,
    compare,
    load_gate_config,
    load_noise,
)
from ragate.evaluation.results import ItemResult, JudgeInfo, RunResult
from ragate.evaluation.stats import paired_bootstrap
from ragate.metrics.aggregate import AGGREGATES, aggregate_all
from tests.conftest import ROOT

JUDGE = JudgeInfo(model_id="heuristic-v1", backend="native", geval_model="stub",
                  prompts={}, deepeval_version="x")


def items(recall: list[float], leak_at: int | None = None) -> list[ItemResult]:
    out = []
    for n, r in enumerate(recall):
        out.append(ItemResult(item_id=f"i{n}", question_type="factoid", expected_behaviour="answer",
                              question="q", scores={"recall_at_k": r, "faithfulness": 1.0},
                              latency_ms=100 + n, pii_leaks=["EMAIL"] if n == leak_at else []))
    return out


def run(run_id: str, its: list[ItemResult], dataset: str = "v1", judge: JudgeInfo = JUDGE):
    return RunResult(run_id=run_id, name=run_id, created_at="t", git_sha="g", config={"k": 4},
                     config_hash="h", dataset_version=dataset, dataset_sha="s" + dataset,
                     judge=judge, provider={}, duration_s=0.0, items=its,
                     aggregates=aggregate_all(its))


CFG = GateConfig(metrics={
    "recall_at_k": MetricRule(direction="higher", min_delta=0.03, floor=0.5),
    "pii_leak_rate": MetricRule(direction="lower", ceiling=0.0),
    "latency_p95_ms": MetricRule(direction="lower", rel_delta=0.1, enforce="warn"),
})
BASE = run("base", items([1.0] * 30))


def test_identical_runs_promote() -> None:
    result = compare(BASE, run("cand", items([1.0] * 30)), CFG)
    assert result.decision == "promote" and result.exit_code == ExitCode.PROMOTE


def test_significant_regression_blocks() -> None:
    cand = run("cand", items([1.0] * 18 + [0.0] * 12))
    result = compare(BASE, cand, CFG)
    v = next(v for v in result.verdicts if v.metric == "recall_at_k")
    assert v.status == "regressed" and v.ci_high is not None and v.ci_high < 0
    assert result.exit_code == ExitCode.BLOCK


def test_drop_within_tolerance_passes() -> None:
    cand = run("cand", items([1.0] * 29 + [0.5]))
    assert compare(BASE, cand, CFG).decision == "promote"


def test_measured_noise_widens_the_tolerance() -> None:
    cand = run("cand", items([1.0] * 25 + [0.5] * 5))  # delta -0.083
    assert compare(BASE, cand, CFG).decision == "block"
    assert compare(BASE, cand, CFG, noise={"recall_at_k": 0.05}).decision == "promote"


def test_floor_blocks_regardless_of_baseline() -> None:
    weak_base = run("base", items([0.4] * 30))
    cand = run("cand", items([0.45] * 30))
    v = next(v for v in compare(weak_base, cand, CFG).verdicts if v.metric == "recall_at_k")
    assert v.status == "limit"


def test_any_pii_leak_blocks() -> None:
    result = compare(BASE, run("cand", items([1.0] * 30, leak_at=3)), CFG)
    assert result.decision == "block" and any("pii_leak_rate" in r for r in result.reasons)


def test_warn_only_metric_never_blocks() -> None:
    slow = items([1.0] * 30)
    for i in slow:
        i.latency_ms *= 5
    result = compare(BASE, run("cand", slow), CFG)
    v = next(v for v in result.verdicts if v.metric == "latency_p95_ms")
    assert v.status == "regressed" and result.decision == "promote"


def test_different_dataset_or_judge_is_an_error() -> None:
    other_ds = compare(BASE, run("cand", items([1.0] * 30), dataset="v2"), CFG)
    assert other_ds.exit_code == ExitCode.ERROR and "dataset differs" in other_ds.reasons[0]
    other_judge = JUDGE.model_copy(update={"model_id": "gpt-4o-mini@t0"})
    result = compare(BASE, run("cand", items([1.0] * 30), judge=other_judge), CFG)
    assert result.exit_code == ExitCode.ERROR


def test_unknown_metric_in_config_is_an_error() -> None:
    cfg = GateConfig(metrics={"recal_at_k": MetricRule(direction="higher")})
    assert compare(BASE, BASE, cfg).exit_code == ExitCode.ERROR


def test_bootstrap_is_deterministic_and_brackets_the_delta() -> None:
    cand = items([1.0] * 20 + [0.0] * 10)
    a = paired_bootstrap(BASE.items, cand, AGGREGATES["recall_at_k"], seed=3)
    b = paired_bootstrap(BASE.items, cand, AGGREGATES["recall_at_k"], seed=3)
    assert a == b and a is not None
    assert a.low <= a.delta <= a.high and a.delta == pytest.approx(-1 / 3)


def test_shipped_gate_config_is_valid_and_complete() -> None:
    cfg = load_gate_config(ROOT / "config" / "gate.yaml")
    assert set(cfg.metrics) <= set(AGGREGATES)
    assert cfg.metrics["pii_leak_rate"].ceiling == 0.0


def test_noise_file_for_another_judge_is_ignored(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "noise.json"
    path.write_text('{"judge": "gpt-4o-mini@t0", "metrics": {"faithfulness": {"std": 0.02}}}')
    assert load_noise(path, "gpt-4o-mini@t0") == {"faithfulness": 0.02}
    assert load_noise(path, "heuristic-v1") == {}
