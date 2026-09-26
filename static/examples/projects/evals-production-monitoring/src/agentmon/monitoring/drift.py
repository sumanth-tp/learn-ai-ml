"""Drift detection on the input-topic distribution (intent histogram) and on quality
scores, with PSI and KL divergence.

PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""

from __future__ import annotations

import math
from collections.abc import Iterable

EPS = 1e-4


def _dist(counts: dict[str, float], keys: Iterable[str]) -> dict[str, float]:
    total = sum(counts.values()) or 1.0
    return {k: max(counts.get(k, 0.0) / total, EPS) for k in keys}


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    keys = sorted(set(expected) | set(actual))
    e, a = _dist(expected, keys), _dist(actual, keys)
    return sum((a[k] - e[k]) * math.log(a[k] / e[k]) for k in keys)


def kl(p_counts: dict[str, float], q_counts: dict[str, float]) -> float:
    """KL(P || Q): P is the current window, Q the baseline. Asymmetric, unlike PSI."""
    keys = sorted(set(p_counts) | set(q_counts))
    p, q = _dist(p_counts, keys), _dist(q_counts, keys)
    return sum(p[k] * math.log(p[k] / q[k]) for k in keys)


def score_histogram(scores: list[float], bins: int = 5) -> dict[str, float]:
    h = {str(i): 0.0 for i in range(bins)}
    for s in scores:
        h[str(min(int(s * bins), bins - 1))] += 1
    return h


def top_movers(expected: dict[str, float], actual: dict[str, float], k: int = 3) -> list[dict]:
    keys = sorted(set(expected) | set(actual))
    e, a = _dist(expected, keys), _dist(actual, keys)
    moves = sorted(keys, key=lambda x: -abs(a[x] - e[x]))[:k]
    return [{"key": x, "baseline": round(e[x], 3), "current": round(a[x], 3)} for x in moves]
