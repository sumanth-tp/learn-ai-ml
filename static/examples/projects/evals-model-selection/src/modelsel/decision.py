"""From per-model metrics to a recommendation: gates, Pareto frontier, weighted matrix.

Order matters:

1. **Hard gates** remove models that cannot ship (invalid JSON too often, too slow,
   too expensive). No weight can buy back a failed gate.
2. The **Pareto frontier** shows every model that is not beaten on all of quality,
   cost and latency at once. Anything off the frontier is dominated: never choose it.
3. The **weighted matrix** picks one point on the frontier using the business's
   weights. Change the weights and the choice may move along the frontier; that is
   the conversation to have with stakeholders, not a bug.
4. **Significance** decides whether the winner's quality lead is real. If it is not,
   the cheaper model of the statistically tied pair is recommended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelsel.llm.registry import DecisionConfig


@dataclass
class ModelSummary:
    model_id: str
    quality: float
    quality_low: float
    quality_high: float
    accuracy: float
    macro_f1: float
    json_validity: float
    field_accuracy: float
    reply_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_usd: float
    errors: int = 0
    gate_failures: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return not self.gate_failures


def apply_gates(models: list[ModelSummary], cfg: DecisionConfig, blocked: dict[str, str] | None = None) -> None:
    """Hard gates from the decision policy, plus ``blocked`` reasons from other checks
    (a contamination flag is a gate: a model whose score is inflated cannot be ranked)."""
    blocked = blocked or {}
    for m in models:
        m.gate_failures = [blocked[m.model_id]] if m.model_id in blocked else []
        if m.json_validity < cfg.min_json_validity:
            m.gate_failures.append(f"JSON validity {m.json_validity:.1%} < {cfg.min_json_validity:.0%}")
        if m.p95_latency_ms > cfg.max_p95_latency_ms:
            m.gate_failures.append(f"p95 latency {m.p95_latency_ms:.0f} ms > {cfg.max_p95_latency_ms:.0f} ms")
        if m.cost_per_1k_usd > cfg.max_cost_per_1k_tickets_usd:
            m.gate_failures.append(f"cost ${m.cost_per_1k_usd:.2f}/1k > ${cfg.max_cost_per_1k_tickets_usd:.2f}/1k")


def dominates(a: ModelSummary, b: ModelSummary) -> bool:
    """a dominates b if it is at least as good on every axis and strictly better on one."""
    ge = a.quality >= b.quality and a.cost_per_1k_usd <= b.cost_per_1k_usd and a.p95_latency_ms <= b.p95_latency_ms
    gt = a.quality > b.quality or a.cost_per_1k_usd < b.cost_per_1k_usd or a.p95_latency_ms < b.p95_latency_ms
    return ge and gt


def pareto_frontier(models: list[ModelSummary]) -> list[str]:
    return [m.model_id for m in models if not any(dominates(o, m) for o in models if o is not m)]


def _minmax(values: list[float], invert: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [((hi - v) if invert else (v - lo)) / (hi - lo) for v in values]


def weighted_matrix(models: list[ModelSummary], cfg: DecisionConfig) -> dict[str, dict[str, float]]:
    """Min-max normalise each criterion across the candidates (cost and latency inverted),
    then weight. Scores are relative to this shortlist, so add or remove a model and
    everyone's normalised score moves; the ranking of the others does not."""
    if not models:
        return {}
    q = _minmax([m.quality for m in models], invert=False)
    c = _minmax([m.cost_per_1k_usd for m in models], invert=True)
    lat = _minmax([m.p95_latency_ms for m in models], invert=True)
    w = cfg.weights
    total_w = sum(w.values()) or 1.0
    out: dict[str, dict[str, float]] = {}
    for m, qn, cn, ln in zip(models, q, c, lat, strict=True):
        total = (w.get("quality", 0) * qn + w.get("cost", 0) * cn + w.get("latency", 0) * ln) / total_w
        out[m.model_id] = {"quality": qn, "cost": cn, "latency": ln, "total": total}
    return out


@dataclass
class Recommendation:
    model_id: str | None
    runner_up: str | None
    reason: str
    significant: bool | None
    caveats: list[str]


def recommend(
    models: list[ModelSummary],
    cfg: DecisionConfig,
    *,
    p_values_vs: dict[tuple[str, str], float],
    blocked: dict[str, str] | None = None,
) -> Recommendation:
    """``p_values_vs[(a, b)]`` is the paired-test p-value for composite quality of a vs b."""
    apply_gates(models, cfg, blocked)
    eligible = [m for m in models if m.passes]
    if not eligible:
        return Recommendation(
            None, None, "No candidate passes the hard gates.", None, ["Relax a gate or add candidates."]
        )
    frontier = set(pareto_frontier(eligible))
    matrix = weighted_matrix(eligible, cfg)
    ranked = sorted((m for m in eligible if m.model_id in frontier), key=lambda m: -matrix[m.model_id]["total"])
    best = ranked[0]
    caveats: list[str] = []
    if len(ranked) == 1:
        return Recommendation(
            best.model_id, None, "Only model on the Pareto frontier that passes every gate.", None, caveats
        )
    second = ranked[1]
    hi, lo = (best, second) if best.quality >= second.quality else (second, best)
    p = p_values_vs.get((hi.model_id, lo.model_id), p_values_vs.get((lo.model_id, hi.model_id), 1.0))
    significant = p < 0.05
    if not significant and lo.cost_per_1k_usd < hi.cost_per_1k_usd and best is hi:
        caveats.append(
            f"{hi.model_id} leads {lo.model_id} on quality by {hi.quality - lo.quality:+.3f} but the difference is "
            f"not significant (p={p:.3f}); the cheaper model is recommended."
        )
        return Recommendation(
            lo.model_id, hi.model_id, "Statistically tied on quality with the leader, and cheaper.", False, caveats
        )
    reason = (
        f"Highest weighted score on the Pareto frontier ({matrix[best.model_id]['total']:.2f} vs "
        f"{matrix[second.model_id]['total']:.2f} for {second.model_id})."
    )
    if not significant:
        caveats.append(f"Quality difference between {hi.model_id} and {lo.model_id} is not significant (p={p:.3f}).")
    return Recommendation(best.model_id, second.model_id, reason, significant, caveats)
