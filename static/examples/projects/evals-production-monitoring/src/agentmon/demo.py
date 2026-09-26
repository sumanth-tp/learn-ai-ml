"""One command, the whole story: simulate a week (regression on day 4, drift on days
6-7), run the online evals, replay the alert rules, triage the root cause, red-team
with and without guardrails, promote flagged traces through the review queue, run the
offline regression gate on v1 and v2, and write the dashboard."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentmon.clock import SimClock
from agentmon.config import Settings
from agentmon.evals.regression import gate, load_cases, load_thresholds, run_suite
from agentmon.feedback.review import ReviewQueue, simulated_review
from agentmon.monitoring.alerts import load_rules, run_alerts
from agentmon.monitoring.dashboard import render_dashboard
from agentmon.runtime import build_runtime
from agentmon.safety.redteam import load_attacks, load_benign, redteam_gate, run_redteam
from agentmon.simulation import WEEK_START, simulate_week
from agentmon.store import Store


def _stamp(ts: float | None) -> str:
    return "-" if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%a %d %b %H:%M")


def run_demo(settings: Settings, per_day: int = 160, fresh: bool = True) -> dict[str, Any]:
    out = settings.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(str(settings.db_path) + suffix).unlink(missing_ok=True)
        settings.golden_production_path.unlink(missing_ok=True)
    settings = settings.model_copy(
        update={"backend_fault_rate": 0.03, "eval_backoff_base_s": 0.0, "prompt_version": "v1"}
    )
    clock = SimClock(WEEK_START)
    rt = build_runtime(settings, clock=clock, store=Store(settings.db_path))
    report: dict[str, Any] = {}

    print("1/6 simulating a week of traffic (prompt v2 ships Thursday 09:00) ...")
    report["simulation"] = simulate_week(rt, per_day=per_day)

    print("2/6 replaying alert rules hour by hour ...")
    alerts = run_alerts(rt.store, load_rules(settings.alerts_path))
    report["alerts"] = alerts
    for a in alerts:
        print(f"   ALERT {a['rule']:<28} fired {_stamp(a['fired_ts'])}  {a['message']}")
        if "root_cause" in a["evidence"]:
            print(f"         root cause: {a['evidence']['root_cause']['summary']}")

    print("3/6 red-teaming with guardrails off and on ...")
    attacks, benign = load_attacks(settings.redteam_path), load_benign(settings.benign_path)
    off = run_redteam(settings, attacks, benign, guardrails=False)
    on = run_redteam(settings, attacks, benign, guardrails=True)
    report["redteam"] = {
        "off": off.model_dump(),
        "on": on.model_dump(),
        "gate_failures_on": redteam_gate(on),
    }
    print(
        f"   ASR off={off.asr:.0%} on={on.asr:.0%}; false refusals on={on.false_refusal_rate:.0%}"
    )

    print("4/6 promoting flagged traces through the review queue ...")
    queue = ReviewQueue(rt.store, clock, settings.golden_seed_path, settings.golden_production_path)
    report["review"] = simulated_review(queue)
    print(f"   {report['review']}")

    print("5/6 offline regression gate on v1 and v2 ...")
    cases = load_cases(settings.golden_seed_path, settings.golden_production_path)
    thresholds = load_thresholds(settings.thresholds_path)
    reg: dict[str, Any] = {}
    baseline = None
    for version in ("v1", "v2"):
        suite = asyncio.run(run_suite(cases, settings, rt.cascade.judges, version))
        g = gate(suite, thresholds, baseline)
        if version == "v1":
            baseline = suite.metrics
        reg[version] = {
            "n": suite.n,
            "metrics": suite.metrics,
            "gate": g.model_dump(),
            "failing_cases": [c.model_dump() for c in suite.cases if c.failures][:10],
        }
        verdict = "PASS" if g.passed else "FAIL"
        print(
            f"   {version}: {verdict} on {suite.n} cases "
            f"({sum(c.source == 'production' for c in cases)} from production) {g.failures[:3]}"
        )
    report["regression"] = reg

    print("6/6 writing dashboard and reports ...")
    (out / "dashboard.html").write_text(
        render_dashboard(rt.store, {"redteam": report["redteam"], "regression": reg})
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"   open {out / 'dashboard.html'}")
    rt.store.close()
    return report
