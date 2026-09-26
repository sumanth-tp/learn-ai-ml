"""Judge calibration against the human-labelled subset.

A judge is a measuring instrument. Before its numbers pick a model, show:

1. **Agreement**: it ranks replies like humans do (Spearman) and lands on the same
   grade (quadratic-weighted Cohen's kappa). Fleiss' kappa among the humans is the
   ceiling: a judge cannot be expected to agree with humans more than they agree
   with each other.
2. **Bias**: position (verdict flips when A and B swap), verbosity (padding raises the
   score), self-preference (it favours text in its own family's style).
3. **Mitigations work**: the same agreement numbers with anchors, reference and swap
   switched off, so each mitigation's value is measured rather than assumed.
4. **Spot checks**: a stronger meta-judge audits the cases where the judge and the
   humans disagree most (judge-of-judges).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sps

from modelsel.harness.client import LLMClient
from modelsel.judge.judge import Judge
from modelsel.llm.registry import Catalogue
from modelsel.logging_setup import log_event
from modelsel.schemas import HumanLabel
from modelsel.stats import permutation_test

logger = logging.getLogger(__name__)

FILLER = (
    "We truly appreciate your patience and loyalty, and we want you to know that your "
    "satisfaction is our highest priority. Our dedicated team works around the clock to make "
    "sure every customer enjoys a seamless, delightful experience with every single product."
)


def cohen_kappa(
    a: Sequence[Any], b: Sequence[Any], *, labels: Sequence[Any] | None = None, weights: str | None = None
) -> float:
    """Cohen's kappa; ``weights='quadratic'`` for ordinal grades (a 4 vs 5 miss is not a 1 vs 5 miss)."""
    cats = list(labels) if labels is not None else sorted(set(a) | set(b))
    k = len(cats)
    if k < 2:
        return 1.0
    pos = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    if weights == "quadratic":
        w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    else:
        w = 1 - np.eye(k)
    denom = float((w * exp).sum())
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


def fleiss_kappa(ratings: Sequence[Sequence[Any]], labels: Sequence[Any]) -> float:
    """Fleiss' kappa for N items each rated by the same number of raters."""
    cats = list(labels)
    counts = np.array([[sum(1 for r in row if r == c) for c in cats] for row in ratings], dtype=float)
    n_raters = counts.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] < 2:
        raise ValueError("every item needs the same number (>= 2) of ratings")
    n = n_raters[0]
    p_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = float((p_j**2).sum())
    return 1.0 if p_e == 1 else float((p_bar - p_e) / (1 - p_e))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    rho = sps.spearmanr(a, b).statistic
    return float(0.0 if np.isnan(rho) else rho)


def pad_reply(reply: str) -> str:
    """Add two paragraphs of content-free filler before the sign-off."""
    head, sep, tail = reply.rpartition("\n\n")
    if not sep:
        return f"{reply}\n\n{FILLER}\n\n{FILLER}"
    return f"{head}\n\n{FILLER}\n\n{FILLER}\n\n{tail}"


def _pref(score_a: float) -> str:
    return "A" if score_a > 0.5 else "B" if score_a < 0.5 else "tie"


@dataclass
class CalibrationReport:
    judge_id: str
    n_rows: int
    spearman: float
    weighted_kappa: float
    human_fleiss: float
    human_plus_judge_fleiss: float
    pairwise_kappa_single: float
    pairwise_kappa_swapped: float
    position_consistency: float
    first_position_win_rate_when_inconsistent: float
    verbosity_delta: float
    verbosity_p: float
    self_preference_delta: float | None
    self_preference_p: float | None
    ablation: dict[str, float] = field(default_factory=dict)
    meta_checks: list[dict[str, Any]] = field(default_factory=list)
    meta_agreement: float | None = None
    trusted: bool = False
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _pointwise_all(judge: Judge, rows: list[HumanLabel], which: str) -> list[float]:
    async def one(r: HumanLabel) -> float:
        reply = r.reply_a if which == "a" else r.reply_b
        res = await judge.pointwise(r.ticket, reply, r.reference_reply)
        return float(res.score) if res.score is not None else 3.0

    return list(await asyncio.gather(*(one(r) for r in rows)))


async def calibrate(
    client: LLMClient,
    catalogue: Catalogue,
    judge_id: str,
    meta_judge_id: str,
    rows: list[HumanLabel],
    *,
    spot_checks: int = 8,
    seed: int = 7,
) -> CalibrationReport:
    judge = Judge(client, judge_id)
    judge_family = catalogue.spec(judge_id).family

    scores_a, scores_b = await asyncio.gather(_pointwise_all(judge, rows, "a"), _pointwise_all(judge, rows, "b"))
    judge_scores = scores_a + scores_b
    human_means = [float(np.mean(r.ratings_a)) for r in rows] + [float(np.mean(r.ratings_b)) for r in rows]
    human_medians = [int(np.median(r.ratings_a)) for r in rows] + [int(np.median(r.ratings_b)) for r in rows]
    grades = [1, 2, 3, 4, 5]
    rounded = [int(min(5, max(1, round(s)))) for s in judge_scores]

    human_matrix = [r.ratings_a for r in rows] + [r.ratings_b for r in rows]
    human_fleiss = fleiss_kappa(human_matrix, grades)
    with_judge = fleiss_kappa([[*h, j] for h, j in zip(human_matrix, rounded, strict=True)], grades)

    single = await asyncio.gather(
        *(judge.pairwise(r.ticket, r.reply_a, r.reply_b, r.reference_reply, swap=False) for r in rows)
    )
    swapped = await asyncio.gather(*(judge.pairwise(r.ticket, r.reply_a, r.reply_b, r.reference_reply) for r in rows))
    prefs = [r.preference for r in rows]
    k_single = cohen_kappa([_pref(s.score_a) for s in single], prefs, labels=["A", "B", "tie"])
    k_swap = cohen_kappa([_pref(s.score_a) for s in swapped], prefs, labels=["A", "B", "tie"])
    consistency = float(np.mean([s.consistent for s in swapped]))
    inconsistent = [s for s in swapped if not s.consistent]
    first_wins = (
        float(np.mean([(s.verdicts[0] == "A") + (s.verdicts[1] == "B") for s in inconsistent]) / 2)
        if inconsistent
        else 0.0
    )

    padded = await asyncio.gather(*(judge.pointwise(r.ticket, pad_reply(r.reply_a), r.reference_reply) for r in rows))
    padded_scores = [float(p.score) if p.score is not None else 3.0 for p in padded]
    verbosity = permutation_test(padded_scores, scores_a, seed=seed)

    def family_of(author: str) -> str | None:
        return catalogue.models[author].family if author in catalogue.models else None

    authors = [r.author_a for r in rows] + [r.author_b for r in rows]
    residual = np.array(judge_scores) - np.array(human_means)
    own = [float(x) for x, au in zip(residual, authors, strict=True) if family_of(au) == judge_family]
    other = [float(x) for x, au in zip(residual, authors, strict=True) if family_of(au) != judge_family]
    if len(own) >= 3 and len(other) >= 3:
        self_delta: float | None = float(np.mean(own) - np.mean(other))
        self_p: float | None = float(sps.mannwhitneyu(own, other, alternative="two-sided").pvalue)
    else:
        self_delta, self_p = None, None

    ablation: dict[str, float] = {"anchored+reference (default)": spearman(judge_scores, human_means)}
    for name, anchored, ref in [
        ("no anchors", False, True),
        ("no reference", True, False),
        ("bare prompt", False, False),
    ]:
        variant = Judge(client, judge_id, anchored=anchored, reference_guided=ref)
        va, vb = await asyncio.gather(_pointwise_all(variant, rows, "a"), _pointwise_all(variant, rows, "b"))
        ablation[name] = spearman(va + vb, human_means)
    ablation["pairwise kappa: single order"] = k_single
    ablation["pairwise kappa: swap-and-average"] = k_swap

    worst = sorted(range(len(judge_scores)), key=lambda i: -abs(judge_scores[i] - human_means[i]))[:spot_checks]
    meta = Judge(client, meta_judge_id)
    checks: list[dict[str, Any]] = []
    for i in worst:
        r = rows[i % len(rows)]
        reply = r.reply_a if i < len(rows) else r.reply_b
        audit = await meta.audit(r.ticket, reply, r.reference_reply, judge_scores[i])
        checks.append(
            {
                "row": r.id,
                "reply": "a" if i < len(rows) else "b",
                "judge": round(judge_scores[i], 2),
                "human_mean": round(human_means[i], 2),
                "meta_agrees": audit.agree,
                "meta_score": audit.corrected,
            }
        )
    meta_agreement = float(np.mean([c["meta_agrees"] for c in checks])) if checks else None

    report = CalibrationReport(
        judge_id=judge_id,
        n_rows=len(rows),
        spearman=spearman(judge_scores, human_means),
        weighted_kappa=cohen_kappa(rounded, human_medians, labels=grades, weights="quadratic"),
        human_fleiss=human_fleiss,
        human_plus_judge_fleiss=with_judge,
        pairwise_kappa_single=k_single,
        pairwise_kappa_swapped=k_swap,
        position_consistency=consistency,
        first_position_win_rate_when_inconsistent=first_wins,
        verbosity_delta=verbosity.mean_diff,
        verbosity_p=verbosity.p_value,
        self_preference_delta=self_delta,
        self_preference_p=self_p,
        ablation=ablation,
        meta_checks=checks,
        meta_agreement=meta_agreement,
    )
    report.caveats = judge_caveats(report)
    report.trusted = report.spearman >= 0.6 and report.weighted_kappa >= 0.4
    log_event(logger, "judge_calibrated", judge_id=judge_id, spearman=round(report.spearman, 3), trusted=report.trusted)
    return report


def judge_caveats(r: CalibrationReport) -> list[str]:
    out: list[str] = []
    if r.spearman < 0.6:
        out.append(f"Judge ranks replies only loosely like humans (Spearman {r.spearman:.2f} < 0.60).")
    if r.weighted_kappa < 0.4:
        out.append(f"Judge grades disagree with human medians (weighted kappa {r.weighted_kappa:.2f} < 0.40).")
    if r.position_consistency < 0.8:
        out.append(
            f"Position bias: {1 - r.position_consistency:.0%} of pairwise verdicts flip when the order is swapped; "
            "pairwise results use swap-and-average."
        )
    if r.verbosity_p < 0.05 and r.verbosity_delta > 0.1:
        out.append(
            f"Verbosity bias: content-free padding raises scores by {r.verbosity_delta:+.2f} (p={r.verbosity_p:.3f})."
        )
    if r.self_preference_delta is not None and r.self_preference_p is not None and r.self_preference_delta > 0.2:
        strength = "confirmed" if r.self_preference_p < 0.05 else "suggested but not significant"
        out.append(
            f"Self-preference {strength} (p={r.self_preference_p:.3f}): the judge over-scores its own family's "
            f"replies by {r.self_preference_delta:+.2f} relative to humans; "
            "treat that family's reply scores as optimistic."
        )
    if r.meta_agreement is not None and r.meta_agreement < 0.5:
        out.append(f"Meta-judge disagrees with the judge on {1 - r.meta_agreement:.0%} of the worst cases it audited.")
    return out
