"""Root-cause triage for a quality alert: where do the bad traces concentrate, and
which change landed just before the alert? Slices are ranked by
(share of bad traces in the slice) x (lift = slice bad rate / overall bad rate)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agentmon.monitoring.metrics import Row, TraceFrame

# Candidate causes: things someone can change or roll back, or a traffic segment.
DIMENSIONS: dict[str, Callable[[Row], str]] = {
    "prompt_version": lambda r: r.trace.prompt_version,
    "model": lambda r: r.trace.model,
    "intent": lambda r: r.trace.intent,
}
# Symptoms: reported for context, never ranked as the cause.
SYMPTOMS: dict[str, Callable[[Row], str]] = {
    "first_tool": lambda r: r.trace.tool_calls[0].name if r.trace.tool_calls else "none",
    "status": lambda r: r.trace.status,
}


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%a %d %b %H:%M UTC")


def analyse(
    frame: TraceFrame,
    end_ts: float,
    lookback_s: float,
    deployments: list[dict[str, Any]],
    min_n: int = 3,
) -> dict[str, Any]:
    changes = [d for d in deployments if end_ts - 3 * lookback_s <= d["ts"] <= end_ts]
    changes.sort(key=lambda d: -d["ts"])
    # Stretch the window back past the most recent change, so traffic from before and
    # after it is compared; otherwise a change that affects 100% of the window has lift 1.
    start = end_ts - lookback_s
    if changes:
        start = min(start, changes[0]["ts"] - lookback_s)
    rows = frame.window(start, end_ts)
    bad = [r for r in rows if not r.good]
    if not rows or not bad:
        return {"summary": "no bad traces in window", "slices": []}
    overall = len(bad) / len(rows)
    slices = []
    for dim, fn in (DIMENSIONS | SYMPTOMS).items():
        groups: dict[str, list[Row]] = {}
        for r in rows:
            groups.setdefault(fn(r), []).append(r)
        for value, members in groups.items():
            if len(members) < min_n:
                continue
            nb = sum(not m.good for m in members)
            rate = nb / len(members)
            share = nb / len(bad)
            slices.append(
                {
                    "dimension": dim,
                    "kind": "cause" if dim in DIMENSIONS else "symptom",
                    "value": value,
                    "n": len(members),
                    "bad_rate": round(rate, 3),
                    "lift": round(rate / overall, 2),
                    "share_of_bad": round(share, 3),
                    "rank": round(share * rate / overall, 3),
                }
            )
    # A slice that IS a recent change (dimension and value match a deployment) is the
    # prime suspect: it is both correlated and actionable (roll it back).
    changed = {(c["component"], c["new_value"]) for c in changes}
    for sl in slices:
        sl["matches_change"] = (sl["dimension"], sl["value"]) in changed and sl["lift"] > 1.2
    slices.sort(key=lambda s: (not s["matches_change"], -s["rank"]))
    failed = Counter(e for r in bad for e in r.failed_evaluators())
    top = next((s for s in slices if s["kind"] == "cause"), None)
    segment = (
        next(
            (
                s
                for s in slices
                if s["kind"] == "cause"
                and s is not top
                and s["dimension"] != top["dimension"]
                and s["lift"] > 1.2
            ),
            None,
        )
        if top
        else None
    )
    exemplars = []
    if top:
        fn = DIMENSIONS[top["dimension"]]
        for r in bad:
            if fn(r) == top["value"] and len(exemplars) < 3:
                exemplars.append(
                    {
                        "trace_id": r.trace.trace_id,
                        "input": r.trace.input,
                        "output": r.trace.output,
                        "failed": r.failed_evaluators(),
                    }
                )
    hours = (end_ts - start) / 3600
    parts = [f"Bad-event rate {overall:.1%} over the last {hours:.0f}h."]
    if top:
        label = "Suspect change" if top["matches_change"] else "Failures concentrate in"
        parts.append(
            f"{label}: {top['dimension']}={top['value']} "
            f"({top['share_of_bad']:.0%} of bad traces, lift {top['lift']}x)."
        )
    if segment:
        parts.append(
            f"Most affected segment: {segment['dimension']}={segment['value']} "
            f"(lift {segment['lift']}x)."
        )
    if failed:
        parts.append(
            "Top failing checks: " + ", ".join(f"{k} ({v})" for k, v in failed.most_common(3)) + "."
        )
    if changes:
        c = changes[0]
        ago = (end_ts - c["ts"]) / 3600
        parts.append(
            f"Nearest change: {c['component']} {c['old_value']} -> {c['new_value']} at "
            f"{_fmt(c['ts'])}, {ago:.1f}h before the alert."
        )
    return {
        "summary": " ".join(parts),
        "slices": slices[:8],
        "failed_checks": dict(failed),
        "changes": changes[:3],
        "exemplars": exemplars,
    }
