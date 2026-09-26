"""Token and cost budget, enforced from graph state.

The budget is *data in state* (a summed ``Usage``), not a global counter, so it
survives checkpoints, adds up correctly across parallel workers and is visible
in every trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from research_analyst.models import Usage

# USD per 1M tokens (input, output). Update when provider prices change.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "fake": (0.15, 0.60),
}
DEFAULT_PRICE = (1.00, 4.00)  # deliberately pessimistic for unknown models


def price_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES_PER_M.get(model, DEFAULT_PRICE)
    return round(input_tokens * pin / 1_000_000 + output_tokens * pout / 1_000_000, 6)


class BudgetMode(StrEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"  # past the soft limit: skip optional work
    EXHAUSTED = "exhausted"  # past the hard limit: no more paid calls


@dataclass(frozen=True)
class Budget:
    max_cost_usd: float
    max_tokens: int
    soft_ratio: float = 0.8

    def fraction_used(self, used: Usage) -> float:
        return max(used.cost_usd / self.max_cost_usd, used.total_tokens / self.max_tokens)

    def mode(self, used: Usage) -> BudgetMode:
        frac = self.fraction_used(used)
        if frac >= 1.0:
            return BudgetMode.EXHAUSTED
        if frac >= self.soft_ratio:
            return BudgetMode.DEGRADED
        return BudgetMode.NORMAL

    def remaining_usd(self, used: Usage) -> float:
        return max(0.0, self.max_cost_usd - used.cost_usd)

    def share(self, used: Usage, workers: int) -> Budget:
        """A worker's slice of what is left, so N parallel workers cannot overspend N times."""
        n = max(1, workers)
        return Budget(
            max_cost_usd=max(1e-6, self.remaining_usd(used) / n),
            max_tokens=max(1, (self.max_tokens - used.total_tokens) // n),
            soft_ratio=self.soft_ratio,
        )
