"""Paired bootstrap confidence intervals for the difference of two runs' aggregates."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ragate.evaluation.results import ItemResult
from ragate.metrics.aggregate import Aggregate


class DeltaCI(BaseModel):
    delta: float
    low: float
    high: float
    n: int


def paired_bootstrap(
    baseline: list[ItemResult],
    candidate: list[ItemResult],
    agg: Aggregate,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> DeltaCI | None:
    """Resample *items* (the same indices for both runs) and recompute the aggregate.

    Pairing removes item difficulty from the variance: a hard question is hard for both
    runs. That makes the interval far tighter than comparing two independent means.
    """
    b_by_id = {i.item_id: i for i in baseline}
    c_by_id = {i.item_id: i for i in candidate}
    ids = sorted(b_by_id.keys() & c_by_id.keys())
    base = [b_by_id[i] for i in ids]
    cand = [c_by_id[i] for i in ids]
    b0, c0 = agg(base), agg(cand)
    if b0 is None or c0 is None:
        return None
    rng = np.random.default_rng(seed)
    n = len(ids)
    deltas = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        b, c = agg([base[j] for j in idx]), agg([cand[j] for j in idx])
        if b is not None and c is not None:
            deltas.append(c - b)
    if not deltas:
        return None
    alpha = (1 - confidence) / 2
    low, high = np.quantile(deltas, [alpha, 1 - alpha])
    return DeltaCI(delta=c0 - b0, low=float(low), high=float(high), n=n)
