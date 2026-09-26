"""Statistics you can defend in a design review.

* Bootstrap confidence intervals for any per-item metric (including macro-F1,
  which is not a mean, so the resampling recomputes it).
* Paired tests: every model answers the same items, so compare per-item
  differences. Pairing removes item difficulty from the noise and typically
  needs far fewer items than an unpaired test.
* McNemar for paired correct/incorrect outcomes, paired bootstrap and a
  sign-flip permutation test for continuous scores, Holm correction when you
  compare several candidates against one baseline.
* Sample-size estimates, so "we need more items" is a number, not a feeling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class CI:
    point: float
    low: float
    high: float

    def fmt(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> CI:
    """Percentile bootstrap over items."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(float("nan"), float("nan"), float("nan"))
    stat = statistic or (lambda x: float(np.mean(x)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    boots = np.array([stat(arr[row]) for row in idx])
    return CI(stat(arr), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def bootstrap_ci_indices(
    n: int,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> CI:
    """Bootstrap a statistic that needs several aligned arrays (e.g. macro-F1 over gold and pred)."""
    rng = np.random.default_rng(seed)
    full = statistic(np.arange(n))
    boots = np.array([statistic(rng.integers(0, n, size=n)) for _ in range(resamples)])
    return CI(full, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci: CI
    p_value: float
    test: str


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, resamples: int = 2000, seed: int = 7) -> PairedResult:
    """Bootstrap the mean of per-item differences (a - b). The p-value is two-sided:
    twice the share of resampled means on the other side of zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("no paired items")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    point = float(d.mean())
    p = 2 * float(np.mean(boots <= 0) if point >= 0 else np.mean(boots >= 0))
    ci = CI(point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return PairedResult(point, ci, min(1.0, p), "paired bootstrap")


def permutation_test(a: Sequence[float], b: Sequence[float], *, resamples: int = 5000, seed: int = 7) -> PairedResult:
    """Sign-flip permutation test on paired differences. Under H0 (no difference) the
    sign of each item's difference is a coin flip; count how often random signs give a
    mean at least as extreme as the observed one."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(resamples, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= observed - 1e-12) + 1) / (resamples + 1)
    ci = bootstrap_ci(d, resamples=min(resamples, 2000), seed=seed)
    return PairedResult(float(d.mean()), ci, float(p), "sign-flip permutation")


@dataclass(frozen=True)
class McNemarResult:
    only_a_correct: int
    only_b_correct: int
    statistic: float
    p_value: float
    exact: bool


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """Only discordant items carry information. Exact binomial when they are few (< 25),
    chi-square with continuity correction otherwise."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcomes must align")
    b01 = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = b01 + b10
    if n == 0:
        return McNemarResult(b01, b10, 0.0, 1.0, True)
    if n < 25:
        p = float(sps.binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue)
        return McNemarResult(b01, b10, float(min(b01, b10)), p, True)
    stat = (abs(b01 - b10) - 1) ** 2 / n
    return McNemarResult(b01, b10, float(stat), float(sps.chi2.sf(stat, df=1)), False)


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: which comparisons stay significant after testing several at once."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    result: dict[str, bool] = {}
    still = True
    for i, (name, p) in enumerate(ordered):
        still = still and p <= alpha / (m - i)
        result[name] = still
    return result


def n_paired_proportions(delta: float, discordant_rate: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed for McNemar to detect an accuracy difference ``delta`` (Connor, 1987).

    ``discordant_rate`` is the share of items where exactly one model is right; take it
    from a pilot run. More disagreement between models means more items are needed.
    """
    if not 0 < abs(delta) <= discordant_rate <= 1:
        raise ValueError("need 0 < |delta| <= discordant_rate <= 1")
    za = sps.norm.ppf(1 - alpha / 2)
    zb = sps.norm.ppf(power)
    n = (za * np.sqrt(discordant_rate) + zb * np.sqrt(discordant_rate - delta**2)) ** 2 / delta**2
    return int(np.ceil(n))


def n_paired_means(delta: float, sd_diff: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed to detect a mean score difference ``delta`` given the SD of per-item differences."""
    if delta <= 0 or sd_diff <= 0:
        raise ValueError("delta and sd_diff must be positive")
    za = sps.norm.ppf(1 - alpha / 2)
    zb = sps.norm.ppf(power)
    return int(np.ceil(((za + zb) * sd_diff / delta) ** 2))


def minimum_detectable_effect(n: int, sd_diff: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """The smallest mean difference this many paired items can reliably detect."""
    za = sps.norm.ppf(1 - alpha / 2)
    zb = sps.norm.ppf(power)
    return float((za + zb) * sd_diff / np.sqrt(max(n, 1)))
