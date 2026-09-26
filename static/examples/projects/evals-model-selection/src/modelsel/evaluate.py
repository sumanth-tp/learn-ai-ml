"""Turn predictions and judge scores into per-item scores and per-model summaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from modelsel.decision import ModelSummary
from modelsel.llm.registry import DecisionConfig
from modelsel.metrics.classification import exact_match, macro_f1
from modelsel.metrics.extraction import field_accuracy
from modelsel.metrics.operational import cost_per_1k, percentile
from modelsel.schemas import BenchmarkItem, ItemScore, Label, Prediction
from modelsel.stats import bootstrap_ci, bootstrap_ci_indices
from modelsel.tasks import parse_fields, parse_label

LABELS = [lb.value for lb in Label]


def composite(cfg: DecisionConfig, correct: bool, field_acc: float, reply_score: float | None) -> float:
    """One number per item so paired tests have something to pair. Reply score is
    mapped from 1..5 to 0..1; a missing judge score counts as the worst grade."""
    qw = cfg.quality_weights
    reply01 = ((reply_score or 1.0) - 1) / 4
    total = qw["classification"] + qw["extraction"] + qw["reply"]
    return (qw["classification"] * float(correct) + qw["extraction"] * field_acc + qw["reply"] * reply01) / total


def score_items(
    items: list[BenchmarkItem],
    preds: list[Prediction],
    reply_scores: dict[tuple[str, str], float | None],
    cfg: DecisionConfig,
) -> list[ItemScore]:
    by_key: dict[tuple[str, str, str], Prediction] = {(p.model_id, p.item_id, p.task): p for p in preds}
    models = sorted({p.model_id for p in preds})
    out: list[ItemScore] = []
    for m in models:
        for it in items:
            c = by_key.get((m, it.id, "classify"))
            e = by_key.get((m, it.id, "extract"))
            r = by_key.get((m, it.id, "reply"))
            if c is None or e is None or r is None:
                continue
            label = parse_label(c.output)
            fields, _err = parse_fields(e.output)
            fa = field_accuracy(it.fields, fields)
            rs = reply_scores.get((m, it.id))
            correct = label == it.label.value
            out.append(
                ItemScore(
                    model_id=m,
                    item_id=it.id,
                    split=it.split,
                    label_pred=label,
                    label_correct=correct,
                    json_valid=fields is not None,
                    field_accuracy=fa,
                    reply_score=rs,
                    composite=composite(cfg, correct, fa, rs),
                    # classify and extract run in parallel in production, then the reply
                    latency_ms=max(c.latency_ms, e.latency_ms) + r.latency_ms,
                    cost_usd=c.cost_usd + e.cost_usd + r.cost_usd,
                    tags=it.tags,
                )
            )
    return out


def summarise(
    scores: list[ItemScore], items: list[BenchmarkItem], preds: list[Prediction], *, resamples: int, seed: int
) -> list[ModelSummary]:
    gold = {it.id: it.label.value for it in items}
    by_model: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        by_model[s.model_id].append(s)
    errors: dict[str, int] = defaultdict(int)
    for p in preds:
        if p.error:
            errors[p.model_id] += 1
    out: list[ModelSummary] = []
    for m, rows in sorted(by_model.items()):
        rows.sort(key=lambda r: r.item_id)
        g = [gold[r.item_id] for r in rows]
        pr = [r.label_pred for r in rows]
        q = bootstrap_ci([r.composite for r in rows], resamples=resamples, seed=seed)
        reply = [r.reply_score for r in rows if r.reply_score is not None]
        out.append(
            ModelSummary(
                model_id=m,
                quality=q.point,
                quality_low=q.low,
                quality_high=q.high,
                accuracy=exact_match(g, pr),
                macro_f1=macro_f1(g, pr, labels=sorted(set(g))),
                json_validity=float(np.mean([r.json_valid for r in rows])),
                field_accuracy=float(np.mean([r.field_accuracy for r in rows])),
                reply_score=float(np.mean(reply)) if reply else 0.0,
                p50_latency_ms=percentile([r.latency_ms for r in rows], 50),
                p95_latency_ms=percentile([r.latency_ms for r in rows], 95),
                cost_per_1k_usd=cost_per_1k([r.cost_usd for r in rows]),
                errors=errors[m],
            )
        )
    return out


def macro_f1_ci(
    rows: list[ItemScore], gold: dict[str, str], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Macro-F1 is not a mean of per-item values, so bootstrap the indices and recompute."""
    g = np.array([gold[r.item_id] for r in rows])
    p = np.array([r.label_pred for r in rows])
    labels = sorted(set(g.tolist()))

    def stat(idx: np.ndarray) -> float:
        return macro_f1(g[idx].tolist(), p[idx].tolist(), labels=labels)

    ci = bootstrap_ci_indices(len(rows), stat, resamples=resamples, seed=seed)
    return ci.point, ci.low, ci.high


def per_slice(scores: list[ItemScore], tag: str) -> dict[str, float]:
    """Mean composite per model on items carrying ``tag`` (e.g. ``ambiguous``)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        if tag in s.tags:
            acc[s.model_id].append(s.composite)
    return {m: float(np.mean(v)) for m, v in sorted(acc.items())}
