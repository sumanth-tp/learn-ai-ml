import math

import pytest
from test_judges_cascade import make_trace

from agentmon.models import EvalResult, Feedback
from agentmon.monitoring.alerts import (
    AlertEngine,
    AlertRules,
    BurnRateRule,
    DriftRule,
    ThresholdRule,
    load_rules,
)
from agentmon.monitoring.dashboard import render_dashboard
from agentmon.monitoring.drift import kl, psi, score_histogram
from agentmon.monitoring.metrics import Row, TraceFrame, aggregate, percentile, window_metrics
from agentmon.monitoring.rootcause import analyse
from agentmon.store import Store

H = 3600.0


def row(i: int, ts: float, good: bool, version: str = "v1", intent: str = "balance") -> Row:
    t = make_trace(
        trace_id=f"t{i}",
        request_id=f"r{i}",
        ts=ts,
        prompt_version=version,
        intent=intent,
        latency_ms=1000 + i,
    )
    ev = EvalResult(
        trace_id=t.trace_id, evaluator="required_tool_called", score=float(good), passed=good, ts=ts
    )
    return Row(t, {ev.evaluator: ev})


def test_psi_and_kl() -> None:
    same = {"a": 50, "b": 50}
    assert psi(same, same) == pytest.approx(0.0)
    shifted = {"a": 80, "b": 20}
    assert psi(same, shifted) == pytest.approx(
        (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    )
    assert kl(shifted, same) > 0 and kl(shifted, same) != kl(same, shifted)
    assert psi({"a": 10}, {"b": 10}) > 1  # a brand-new category is a big shift, not a crash


def test_score_histogram_and_percentile() -> None:
    assert score_histogram([0.0, 0.5, 1.0], bins=5) == {"0": 1, "1": 0, "2": 1, "3": 0, "4": 1}
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 95) == 0.0


def test_window_metrics_and_feedback_makes_trace_bad() -> None:
    r = row(1, 0, True)
    r.feedback = Feedback(trace_id="t1", rating=-1)
    m = window_metrics([r, row(2, 1, True)])
    assert m["n"] == 2 and m["good_rate"] == 0.5 and m["thumbs_down_rate"] == 0.5


def _frame(regress_at: float, end: float, per_hour: int = 10) -> TraceFrame:
    rows, i = [], 0
    t = 0.0
    while t < end:
        for k in range(per_hour):
            ts = t + k * (H / per_hour)
            after = ts >= regress_at
            good = not (after and k % 2 == 0)  # 50% failure after the change
            rows.append(
                row(i, ts, good, "v2" if after else "v1", "balance" if k % 2 == 0 else "faq")
            )
            i += 1
        t += H
    return TraceFrame(rows)


def test_aggregate_windows() -> None:
    frame = _frame(regress_at=10 * H, end=20 * H)
    windows = aggregate(frame, 0, 20 * H, 10 * H)
    assert [w["n"] for w in windows] == [100, 100]
    assert windows[0]["required_tool_rate"] == 1.0 and windows[1]["required_tool_rate"] == 0.5


def test_threshold_alert_fires_after_regression_and_holds_with_hysteresis() -> None:
    frame = _frame(regress_at=24 * H, end=48 * H)
    rules = AlertRules(
        threshold=[
            ThresholdRule(
                name="rt",
                metric="required_tool_rate",
                op="<",
                threshold=0.95,
                window_hours=2,
                min_samples=10,
            )
        ]
    )
    alerts = AlertEngine(rules, frame, []).replay(0, 48 * H)
    assert len(alerts) == 1
    assert 24 * H < alerts[0]["fired_ts"] <= 26 * H


def test_burn_rate_needs_both_windows() -> None:
    frame = _frame(regress_at=24 * H, end=30 * H)
    rule = BurnRateRule(
        name="burn",
        slo_target=0.9,
        long_window_hours=6,
        short_window_hours=1,
        burn_threshold=3,
        severity="ticket",
    )
    engine = AlertEngine(AlertRules(burn_rate=[rule]), frame, [])
    before = engine.evaluate_at(24 * H)[0]
    after = engine.evaluate_at(30 * H)[0]
    assert not before.firing and after.firing
    assert after.value == pytest.approx(5.0)  # 50% bad / 10% budget


def test_drift_rule_on_intent_mix() -> None:
    rows = [row(i, i * 60.0, True, intent="balance" if i % 2 else "faq") for i in range(4320)]
    rows += [row(10_000 + i, 72 * H + i * 60.0, True, intent="faq") for i in range(1440)]
    frame = TraceFrame(rows)
    rule = DriftRule(name="d", kind="intent", baseline_hours=72, window_hours=24, threshold=0.2)
    engine = AlertEngine(AlertRules(drift=[rule]), frame, [])
    assert not engine.evaluate_at(80 * H)[0].firing  # overlaps baseline: never compared
    ev = engine.evaluate_at(96 * H)[0]
    assert ev.firing and ev.value > 0.2


def test_root_cause_points_at_the_deployment() -> None:
    frame = _frame(regress_at=24 * H, end=30 * H)
    deps = [{"ts": 24 * H, "component": "prompt_version", "old_value": "v1", "new_value": "v2"}]
    rc = analyse(frame, 30 * H, 24 * H, deps)
    top = rc["slices"][0]
    assert (top["dimension"], top["value"], top["matches_change"]) == ("prompt_version", "v2", True)
    assert "Suspect change: prompt_version=v2" in rc["summary"]
    assert "intent=balance" in rc["summary"]


def test_alert_rules_file_loads() -> None:
    from conftest import ROOT

    rules = load_rules(ROOT / "config/alerts.toml")
    assert rules.threshold and rules.burn_rate and rules.drift


def test_dashboard_renders(rt) -> None:
    for q in ["What's the balance on ACC-1001?", "hi", "What are the overdraft fees?"]:
        rt.service.handle("CUST-1", q)
    page = render_dashboard(rt.store)
    assert page.startswith("<!doctype html>") and "<svg" in page and "Alerts" in page
    assert "No traces yet" in render_dashboard(Store(":memory:"))
