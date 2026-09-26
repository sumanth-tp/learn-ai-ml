"""The whole story on a smaller week: regression on day 4 must page, point at the
prompt deployment, and the gate must block v2 once production cases are promoted."""

import asyncio

from agentmon.clock import SimClock
from agentmon.evals.regression import gate, load_cases, load_thresholds, run_suite
from agentmon.feedback.review import ReviewQueue, simulated_review
from agentmon.monitoring.alerts import load_rules, run_alerts
from agentmon.runtime import build_runtime
from agentmon.simulation import WEEK_START, simulate_week
from agentmon.store import Store


def test_simulated_week_detects_and_traces_the_regression(settings) -> None:
    s = settings.model_copy(update={"backend_fault_rate": 0.03})
    clock = SimClock(WEEK_START)
    rt = build_runtime(s, clock=clock, store=Store(":memory:"))
    summary = simulate_week(rt, per_day=120)
    assert summary["requests"] > 700
    assert rt.store.job_counts().get("pending", 0) == 0

    alerts = run_alerts(rt.store, load_rules(s.alerts_path))
    deploy = summary["deploy_ts"]
    pages = [a for a in alerts if a["rule"] == "required_tool_rate_low"]
    assert pages and all(a["fired_ts"] > deploy for a in pages)
    assert pages[0]["fired_ts"] - deploy < 6 * 3600  # detected within hours, not days
    rc = pages[0]["evidence"]["root_cause"]
    assert rc["changes"][0]["new_value"] == "v2"
    assert any(a["rule"] == "intent_mix_drift" for a in alerts)
    assert not any(a["fired_ts"] < deploy and a["severity"] == "page" for a in alerts)

    q = ReviewQueue(rt.store, clock, s.golden_seed_path, s.golden_production_path)
    assert simulated_review(q).get("approved", 0) > 5
    cases = load_cases(s.golden_seed_path, s.golden_production_path)
    judges = rt.cascade.judges
    thresholds = load_thresholds(s.thresholds_path)
    assert gate(asyncio.run(run_suite(cases, s, judges, "v1")), thresholds).passed
    assert not gate(asyncio.run(run_suite(cases, s, judges, "v2")), thresholds).passed
