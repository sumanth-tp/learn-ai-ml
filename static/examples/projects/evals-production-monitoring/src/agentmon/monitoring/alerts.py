"""Alert rules: static thresholds, multi-window SLO burn rates, and drift.

Burn rate = (bad fraction in window) / (error budget, 1 - SLO target). A burn rate of
1 spends the budget exactly over the SLO period; the fast-burn rule pages only when
a long AND a short window both burn hot, which filters blips yet resets quickly
after a fix (the multiwindow pattern from the Google SRE workbook)."""

from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from agentmon.monitoring.drift import kl, psi, score_histogram, top_movers
from agentmon.monitoring.metrics import TraceFrame, window_metrics
from agentmon.monitoring.rootcause import analyse
from agentmon.store import Store

HOUR = 3600.0


class ThresholdRule(BaseModel):
    name: str
    metric: str
    op: Literal["<", ">"]
    threshold: float
    window_hours: float
    min_samples: int = 10
    severity: Literal["page", "ticket"] = "ticket"


class BurnRateRule(BaseModel):
    name: str
    slo_target: float
    long_window_hours: float
    short_window_hours: float
    burn_threshold: float
    min_samples: int = 5
    severity: Literal["page", "ticket"] = "page"


class DriftRule(BaseModel):
    name: str
    kind: Literal["intent", "score"]
    evaluator: str = "judge_groundedness"
    baseline_hours: float = 72
    window_hours: float = 24
    threshold: float = 0.2
    min_samples: int = 30
    severity: Literal["page", "ticket"] = "ticket"


class AlertRules(BaseModel):
    threshold: list[ThresholdRule] = []
    burn_rate: list[BurnRateRule] = []
    drift: list[DriftRule] = []


def load_rules(path: Path) -> AlertRules:
    return AlertRules.model_validate(tomllib.loads(path.read_text()))


class Evaluation(BaseModel):
    rule: str
    severity: str
    firing: bool
    value: float | None
    threshold: float
    message: str
    insufficient: bool = False  # too little data to decide: leave alert state unchanged
    evidence: dict[str, Any] = {}


def _sample_bad(rows: list, k: int = 5) -> list[str]:
    return [r.trace.trace_id for r in rows if not r.good][:k]


class AlertEngine:
    def __init__(self, rules: AlertRules, frame: TraceFrame, deployments: list[dict]) -> None:
        self.rules = rules
        self.frame = frame
        self.deployments = deployments
        self.origin = frame.span[0]

    def _threshold(self, r: ThresholdRule, t: float) -> Evaluation:
        rows = self.frame.window(t - r.window_hours * HOUR, t)
        m = window_metrics(rows)
        value = m.get(r.metric)
        samples = (
            m["judged_n"]
            if r.metric.startswith(("groundedness", "helpfulness", "policy"))
            else m["required_tool_n"]
            if r.metric == "required_tool_rate"
            else m["n"]
        )
        firing = (
            value is not None
            and samples >= r.min_samples
            and (value < r.threshold if r.op == "<" else value > r.threshold)
        )
        return Evaluation(
            rule=r.name,
            severity=r.severity,
            firing=firing,
            value=value,
            threshold=r.threshold,
            insufficient=value is None or samples < r.min_samples,
            message=f"{r.metric} {'n/a' if value is None else f'{value:.3f}'} {r.op} "
            f"{r.threshold} over {r.window_hours:g}h (n={samples})",
            evidence={"bad_traces": _sample_bad(rows), "prompt_versions": m["prompt_versions"]},
        )

    def _burn(self, r: BurnRateRule, t: float) -> Evaluation:
        budget = 1 - r.slo_target
        out = {}
        for label, hours in (("long", r.long_window_hours), ("short", r.short_window_hours)):
            rows = self.frame.window(t - hours * HOUR, t)
            bad = sum(not x.good for x in rows)
            out[label] = ((bad / len(rows)) / budget if rows else 0.0, len(rows))
        (long_b, long_n), (short_b, short_n) = out["long"], out["short"]
        firing = (
            long_n >= r.min_samples
            and short_n >= r.min_samples
            and long_b > r.burn_threshold
            and short_b > r.burn_threshold
        )
        rows = self.frame.window(t - r.long_window_hours * HOUR, t)
        return Evaluation(
            rule=r.name,
            severity=r.severity,
            firing=firing,
            value=long_b,
            threshold=r.burn_threshold,
            insufficient=long_n < r.min_samples or short_n < r.min_samples,
            message=f"burn rate {long_b:.1f}x ({r.long_window_hours:g}h) and {short_b:.1f}x "
            f"({r.short_window_hours:g}h) vs {r.burn_threshold}x, SLO {r.slo_target:.0%}",
            evidence={
                "bad_traces": _sample_bad(rows),
                "prompt_versions": dict(Counter(x.trace.prompt_version for x in rows)),
            },
        )

    def _drift(self, r: DriftRule, t: float) -> Evaluation:
        base = self.frame.window(self.origin, self.origin + r.baseline_hours * HOUR)
        cur = self.frame.window(t - r.window_hours * HOUR, t)
        if t - r.window_hours * HOUR < self.origin + r.baseline_hours * HOUR:
            cur = []  # never compare a window with the baseline it overlaps
        if r.kind == "intent":
            b = dict(Counter(x.trace.intent for x in base))
            c = dict(Counter(x.trace.intent for x in cur))
            n = len(cur)
        else:
            b = score_histogram(
                [x.evals[r.evaluator].score for x in base if r.evaluator in x.evals]
            )
            vals = [x.evals[r.evaluator].score for x in cur if r.evaluator in x.evals]
            c = score_histogram(vals)
            n = len(vals)
        if n < r.min_samples or not base:
            return Evaluation(
                rule=r.name,
                severity=r.severity,
                firing=False,
                value=None,
                threshold=r.threshold,
                message=f"insufficient data (n={n})",
                insufficient=True,
            )
        value = psi(b, c)
        return Evaluation(
            rule=r.name,
            severity=r.severity,
            firing=value > r.threshold,
            value=value,
            threshold=r.threshold,
            message=f"{r.kind} PSI {value:.3f} (KL {kl(c, b):.3f}) vs {r.threshold} "
            f"over {r.window_hours:g}h",
            evidence={"top_movers": top_movers(b, c)},
        )

    def evaluate_at(self, t: float) -> list[Evaluation]:
        return (
            [self._threshold(r, t) for r in self.rules.threshold]
            + [self._burn(r, t) for r in self.rules.burn_rate]
            + [self._drift(r, t) for r in self.rules.drift]
        )

    def replay(
        self, start: float, end: float, step_s: float = HOUR, resolve_after: int = 6
    ) -> list[dict[str, Any]]:
        """Evaluate every rule at every step, turning edges into fired/resolved alerts.

        Hysteresis: an alert resolves only after `resolve_after` consecutive clear
        evaluations, so a metric hovering at the threshold does not page repeatedly."""
        active: dict[str, dict[str, Any]] = {}
        clear: dict[str, int] = {}
        alerts: list[dict[str, Any]] = []
        t = start + step_s
        while t <= end + step_s:
            for ev in self.evaluate_at(t):
                if ev.insufficient and not ev.firing:
                    continue
                if ev.firing:
                    clear[ev.rule] = 0
                if ev.firing and ev.rule not in active:
                    alert = {
                        "rule": ev.rule,
                        "severity": ev.severity,
                        "fired_ts": t,
                        "resolved_ts": None,
                        "value": ev.value or 0.0,
                        "threshold": ev.threshold,
                        "message": ev.message,
                        "evidence": dict(ev.evidence),
                    }
                    if ev.severity == "page" and ev.rule.find("drift") < 0:
                        alert["evidence"]["root_cause"] = analyse(
                            self.frame, t, 24 * HOUR, self.deployments
                        )
                    active[ev.rule] = alert
                    alerts.append(alert)
                elif not ev.firing and ev.rule in active:
                    clear[ev.rule] = clear.get(ev.rule, 0) + 1
                    if clear[ev.rule] >= resolve_after:
                        active.pop(ev.rule)["resolved_ts"] = t
            t += step_s
        return alerts


def run_alerts(store: Store, rules: AlertRules, step_s: float = HOUR) -> list[dict[str, Any]]:
    frame = TraceFrame.load(store)
    if not frame.rows:
        return []
    start, end = frame.span
    alerts = AlertEngine(rules, frame, store.deployments()).replay(start, end, step_s)
    store.clear_alerts()
    for a in alerts:
        store.insert_alert(a)
    return alerts
