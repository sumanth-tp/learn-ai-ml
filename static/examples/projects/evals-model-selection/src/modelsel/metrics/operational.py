"""Latency percentiles and cost per 1,000 tickets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def cost_per_1k(costs_per_ticket: Sequence[float]) -> float:
    """A ticket is three calls (classify, extract, reply); the input is already summed per ticket."""
    if not costs_per_ticket:
        return 0.0
    return float(np.mean(costs_per_ticket) * 1000)
