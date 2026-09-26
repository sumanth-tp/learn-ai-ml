"""Markdown reports: the gate report posted on the PR, and the experiments table."""

from __future__ import annotations

from ragate.evaluation.gate import GateResult
from ragate.evaluation.results import RunResult
from ragate.metrics.aggregate import AGGREGATES

BREAKDOWN_METRICS = ["recall_at_k", "faithfulness", "correctness", "refusal_rate"]


def fmt(value: float | None, metric: str = "") -> str:
    if value is None:
        return "n/a"
    if metric.endswith("_usd"):
        return f"{value:.6f}"
    if metric.endswith("_ms") or metric.startswith("tokens"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def _signed(value: float | None, metric: str) -> str:
    if value is None:
        return ""
    return ("+" if value >= 0 else "") + fmt(value, metric)


def gate_report(result: GateResult, base: RunResult, cand: RunResult) -> str:
    lines = [
        f"# Eval gate: **{result.decision.upper()}** (exit {result.exit_code})",
        "",
        f"- Baseline: `{base.run_id}` ({base.git_sha}), candidate: `{cand.run_id}` "
        f"({cand.git_sha})",
        f"- Dataset: `{cand.dataset_version}` ({len(cand.items)} items), judge: "
        f"`{cand.judge.model_id}` / backend `{cand.judge.backend}`",
        f"- Generator: `{cand.provider.get('generator')}`, k={cand.config.get('k')}, "
        f"chunk_size={cand.config.get('chunk_size')}, reranker={cand.config.get('reranker')}",
        "",
        "## Decision",
        "",
        *[f"- {r}" for r in result.reasons],
        "",
    ]
    if result.verdicts:
        lines += [
            "## Metrics",
            "",
            "| Metric | Better | Baseline | Candidate | Delta | 95% CI | Tolerance | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for v in result.verdicts:
            ci = (f"[{_signed(v.ci_low, v.metric)}, {_signed(v.ci_high, v.metric)}]"
                  if v.ci_low is not None else "")
            status = v.status + (" (warn)" if v.enforce == "warn" and v.status != "pass" else "")
            lines.append(
                f"| {v.metric} | {v.direction} | {fmt(v.baseline, v.metric)} | "
                f"{fmt(v.candidate, v.metric)} | {_signed(v.delta, v.metric)} | {ci} | "
                f"{fmt(v.tolerance, v.metric)} | {status} |"
            )
        lines += ["", "## By question type (candidate vs baseline)", "",
                  "| Type | " + " | ".join(BREAKDOWN_METRICS) + " |",
                  "| --- |" + " --- |" * len(BREAKDOWN_METRICS)]
        types = sorted({i.question_type for i in cand.items})
        for qt in types:
            b_items = [i for i in base.items if i.question_type == qt]
            c_items = [i for i in cand.items if i.question_type == qt]
            cells = []
            for m in BREAKDOWN_METRICS:
                b, c = AGGREGATES[m](b_items), AGGREGATES[m](c_items)
                change = _signed(None if b is None else c - b, m) if c is not None else ""
                cells.append("n/a" if c is None else f"{fmt(c)} ({change or 'new'})")
            lines.append(f"| {qt} | " + " | ".join(cells) + " |")
        lines += ["", "## Items that got worse", ""]
        lines += worst_items(base, cand) or ["None."]
    return "\n".join(lines) + "\n"


def worst_items(base: RunResult, cand: RunResult, limit: int = 5) -> list[str]:
    by_id = {i.item_id: i for i in base.items}
    drops = []
    for c in cand.items:
        b = by_id.get(c.item_id)
        if b is None:
            continue
        for metric in ("correctness", "recall_at_k", "faithfulness"):
            bv, cv = b.scores.get(metric), c.scores.get(metric)
            if bv is not None and cv is not None and cv < bv:
                drops.append((bv - cv, c, metric, bv, cv))
        if b.refused != c.refused:
            drops.append((1.0, c, "refused", float(b.refused), float(c.refused)))
    drops.sort(key=lambda d: -d[0])
    return [
        f"- `{c.item_id}` ({c.question_type}) {metric}: {bv:.2f} -> {cv:.2f}. Q: {c.question}"
        for _, c, metric, bv, cv in drops[:limit]
    ]


def experiments_table(runs: list[RunResult], metrics: list[str]) -> str:
    header = "| Variant | chunk | k | hybrid | reranker | model | " + " | ".join(metrics) + " |"
    lines = [header, "| --- " * (6 + len(metrics)) + "|"]
    for r in runs:
        cfg = r.config
        model = r.provider.get("generator", "").split(":", 1)[-1]
        cells = [fmt(r.aggregates.get(m), m) for m in metrics]
        lines.append(
            f"| {r.name} | {cfg['chunk_size']} | {cfg['k']} | {cfg['hybrid']} | "
            f"{cfg['reranker']} | {model} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"
