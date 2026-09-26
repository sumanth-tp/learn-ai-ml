"""Report-level metrics. The judge is a separate ``Brain`` from the one that wrote the report."""

from __future__ import annotations

import asyncio
import math

from pydantic import BaseModel

from research_analyst.models import FinalReport, Source
from research_analyst.providers.llm import Brain
from research_analyst.text import content_tokens


class OutlineItem(BaseModel):
    topic: str
    keywords: list[str]


class EvalCase(BaseModel):
    id: str
    question: str
    reference_outline: list[OutlineItem]


class CaseResult(BaseModel):
    id: str
    citation_precision: float
    claim_support_rate: float
    coverage: float
    draft_support_rate: float
    claims: int
    gaps: int
    cost_usd: float
    tokens: int
    latency_s: float


async def judge_report(
    report: FinalReport, sources: dict[str, Source], judge: Brain
) -> tuple[float, float]:
    """Re-check every (claim, citation) pair with an independent judge.

    citation precision = citations the judge finds supporting / all citations
    claim support rate = claims with at least one fully supporting citation / all claims
    """
    pairs = [(c.text, sid) for sec in report.sections for c in sec.claims for sid in c.citations]
    if not pairs:
        return 0.0, 0.0
    judgements = await asyncio.gather(
        *(
            judge.check_support(text, sources[sid].content if sid in sources else "")
            for text, sid in pairs
        )
    )
    levels = [j.level for j, _ in judgements]
    precision = sum(1 for lv in levels if lv != "no_support") / len(levels)
    by_claim: dict[str, bool] = {}
    for (text, _), lv in zip(pairs, levels, strict=True):
        by_claim[text] = by_claim.get(text, False) or lv == "fully_supported"
    support = sum(by_claim.values()) / len(by_claim)
    return round(precision, 3), round(support, 3)


def coverage(report: FinalReport, outline: list[OutlineItem]) -> float:
    """Share of reference-outline topics the report covers (>= half their keywords present)."""
    text = content_tokens(" ".join(c.text for s in report.sections for c in s.claims))
    covered = 0
    for item in outline:
        kws = set().union(*(content_tokens(k) for k in item.keywords))
        if kws and len(kws & text) / len(kws) >= 0.5:
            covered += 1
    return round(covered / len(outline), 3) if outline else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
