"""Run-level metrics as functions of the per-item results.

Every aggregate is a pure function of a list of ItemResults. That is what makes the
paired bootstrap possible: resample items, recompute the aggregate, repeat.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ragate.evaluation.results import ItemResult

Aggregate = Callable[[list[ItemResult]], float | None]


def mean_of(metric: str) -> Aggregate:
    def agg(items: list[ItemResult]) -> float | None:
        vals = [i.scores[metric] for i in items if i.scores.get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    return agg


def rate(select: Callable[[ItemResult], bool], hit: Callable[[ItemResult], bool]) -> Aggregate:
    def agg(items: list[ItemResult]) -> float | None:
        chosen = [i for i in items if select(i)]
        return sum(hit(i) for i in chosen) / len(chosen) if chosen else None

    return agg


def percentile(field: str, q: float) -> Aggregate:
    def agg(items: list[ItemResult]) -> float | None:
        vals = [getattr(i, field) for i in items]
        return float(np.percentile(vals, q)) if vals else None

    return agg


def mean_field(field: str) -> Aggregate:
    def agg(items: list[ItemResult]) -> float | None:
        vals = [getattr(i, field) for i in items]
        return float(np.mean(vals)) if vals else None

    return agg


PER_ITEM_METRICS = [
    # retriever (reference-based)
    "recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k", "contextual_precision",
    "contextual_recall",
    # generator
    "faithfulness", "answer_relevancy", "correctness", "citation_validity",
    "citation_precision", "citation_recall",
    # the RAG triad's third leg (the other two are faithfulness and answer_relevancy)
    "context_relevance",
]

AGGREGATES: dict[str, Aggregate] = {m: mean_of(m) for m in PER_ITEM_METRICS}
AGGREGATES.update({
    "refusal_rate": rate(lambda i: i.expected_behaviour == "refuse", lambda i: i.refused),
    "false_refusal_rate": rate(lambda i: i.expected_behaviour != "refuse", lambda i: i.refused),
    "pii_leak_rate": rate(lambda i: True, lambda i: bool(i.pii_leaks)),
    "judge_error_rate": rate(lambda i: True, lambda i: bool(i.errors)),
    "latency_p50_ms": percentile("latency_ms", 50),
    "latency_p95_ms": percentile("latency_ms", 95),
    "cost_per_query_usd": mean_field("cost_usd"),
    "tokens_per_query": mean_field("total_tokens"),
})


def aggregate_all(items: list[ItemResult]) -> dict[str, float | None]:
    return {name: fn(items) for name, fn in AGGREGATES.items()}
