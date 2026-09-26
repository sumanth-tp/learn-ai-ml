"""Metric aggregation into time windows. The store is loaded once into a TraceFrame
(traces joined with evals and feedback, sorted by time) so alert evaluation over
hundreds of windows is fast and consistent."""

from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentmon.models import EvalResult, Feedback, TraceRecord
from agentmon.store import Store

# Evaluators whose failure makes a trace a "bad event" for the quality SLO.
# latency has its own SLO; the cascade marker is bookkeeping, not a verdict.
NOT_QUALITY = {"latency_slo", "cascade"}


@dataclass
class Row:
    trace: TraceRecord
    evals: dict[str, EvalResult] = field(default_factory=dict)
    feedback: Feedback | None = None

    @property
    def good(self) -> bool:
        if self.feedback is not None and self.feedback.rating < 0:
            return False
        return all(e.passed for k, e in self.evals.items() if k not in NOT_QUALITY)

    def failed_evaluators(self) -> list[str]:
        out = [k for k, e in self.evals.items() if k not in NOT_QUALITY and not e.passed]
        if self.feedback is not None and self.feedback.rating < 0:
            out.append("thumbs_down")
        return out


class TraceFrame:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = sorted(rows, key=lambda r: r.trace.ts)
        self._ts = [r.trace.ts for r in self.rows]

    @classmethod
    def load(cls, store: Store, start: float = 0.0, end: float = math.inf) -> TraceFrame:
        traces = store.traces_between(start, end if math.isfinite(end) else 1e12)
        evals: dict[str, dict[str, EvalResult]] = {}
        for e in store.evals_between(start, end if math.isfinite(end) else 1e12):
            evals.setdefault(e.trace_id, {})[e.evaluator] = e
        fb = store.all_feedback()
        return cls([Row(t, evals.get(t.trace_id, {}), fb.get(t.trace_id)) for t in traces])

    def window(self, start: float, end: float) -> list[Row]:
        return self.rows[bisect.bisect_left(self._ts, start) : bisect.bisect_left(self._ts, end)]

    @property
    def span(self) -> tuple[float, float]:
        return (self._ts[0], self._ts[-1]) if self._ts else (0.0, 0.0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, math.ceil(q / 100 * len(s)) - 1)  # nearest-rank
    return s[k]


def _rate(rows: list[Row], evaluator: str) -> tuple[float | None, int]:
    judged = [r.evals[evaluator].passed for r in rows if evaluator in r.evals]
    return (sum(judged) / len(judged) if judged else None), len(judged)


def window_metrics(rows: list[Row]) -> dict[str, Any]:
    n = len(rows)
    lat = [r.trace.latency_ms for r in rows]
    fb = [r.feedback for r in rows if r.feedback is not None]
    g_rate, g_n = _rate(rows, "judge_groundedness")
    h_rate, _ = _rate(rows, "judge_helpfulness")
    p_rate, _ = _rate(rows, "judge_policy")
    req_rate, req_n = _rate(rows, "required_tool_called")
    g_scores = [
        r.evals["judge_groundedness"].score for r in rows if "judge_groundedness" in r.evals
    ]
    cost = sum(r.trace.cost_usd for r in rows)
    judge_cost = sum(e.cost_usd for r in rows for e in r.evals.values() if e.tier == "judge")
    return {
        "n": n,
        "error_rate": sum(r.trace.status == "error" for r in rows) / n if n else None,
        "block_rate": sum(r.trace.status == "blocked" for r in rows) / n if n else None,
        "p50_latency_ms": percentile(lat, 50) if n else None,
        "p95_latency_ms": percentile(lat, 95) if n else None,
        "cost_usd": cost,
        "cost_per_request": cost / n if n else None,
        "judge_cost_usd": judge_cost,
        "good_rate": sum(r.good for r in rows) / n if n else None,
        "judged_n": g_n,
        "groundedness_pass_rate": g_rate,
        "groundedness_mean_score": sum(g_scores) / len(g_scores) if g_scores else None,
        "helpfulness_pass_rate": h_rate,
        "policy_pass_rate": p_rate,
        "required_tool_rate": req_rate,
        "required_tool_n": req_n,
        "feedback_n": len(fb),
        "thumbs_down_rate": sum(f.rating < 0 for f in fb) / n if n else None,
        "intent_counts": dict(Counter(r.trace.intent for r in rows)),
        "prompt_versions": dict(Counter(r.trace.prompt_version for r in rows)),
    }


def aggregate(frame: TraceFrame, start: float, end: float, step_s: float) -> list[dict[str, Any]]:
    out = []
    t = start
    while t < end:
        m = window_metrics(frame.window(t, t + step_s))
        m["start"] = t
        m["end"] = t + step_s
        out.append(m)
        t += step_s
    return out
