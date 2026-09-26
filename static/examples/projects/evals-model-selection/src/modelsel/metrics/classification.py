"""Exact match and macro-F1, written out so the arithmetic is inspectable.

Macro-F1 averages per-class F1 with equal weight. It is the right headline for
triage because the rare classes (account takeover, cancellation) are the ones
where a miss costs most; accuracy lets a model ignore them and still look good.
"""

from __future__ import annotations

from collections.abc import Sequence


def exact_match(gold: Sequence[str], pred: Sequence[str]) -> float:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def per_class_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> dict[str, float]:
    classes = list(labels) if labels is not None else sorted(set(gold))
    out: dict[str, float] = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[c] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Classes are taken from the gold labels, so an invented label only costs precision."""
    scores = per_class_f1(gold, pred, labels)
    return sum(scores.values()) / len(scores) if scores else 0.0


def confusion(gold: Sequence[str], pred: Sequence[str]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for g, p in zip(gold, pred, strict=True):
        counts[(g, p)] = counts.get((g, p), 0) + 1
    return counts
