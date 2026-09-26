"""Eval metrics and the regression gate."""

from __future__ import annotations

import json

from research_analyst.evals.metrics import OutlineItem, coverage, p95
from research_analyst.evals.run import format_summary, run_eval
from research_analyst.models import FinalReport, ReportMetrics, ReportSection, Usage, VerifiedClaim


def _report(texts: list[str]) -> FinalReport:
    claims = [
        VerifiedClaim(
            sub_question_id="sq1", text=t, citations=["S1"], checks=[], status="supported"
        )
        for t in texts
    ]
    return FinalReport(
        thread_id="t",
        question="q",
        title="t",
        sections=[ReportSection(sub_question_id="sq1", heading="h", claims=claims)],
        references=[],
        gaps=[],
        degraded=False,
        degradation_notes=[],
        metrics=ReportMetrics(
            claims_drafted=1,
            claims_kept=1,
            claims_removed=0,
            claim_support_rate=1,
            citation_precision=1,
            revisions=0,
            critic_score=None,
        ),
        usage=Usage(),
        markdown="",
    )


def test_coverage_counts_outline_topics():
    r = _report(["LFP cost per kWh fell to 92 USD.", "Cold climate performance is good."])
    outline = [
        OutlineItem(topic="cost", keywords=["lfp", "cost", "kwh"]),
        OutlineItem(topic="cold", keywords=["cold", "climate"]),
        OutlineItem(topic="safety", keywords=["thermal", "runaway"]),
    ]
    assert coverage(r, outline) == round(2 / 3, 3)


def test_p95():
    assert p95([1.0, 2.0, 3.0]) == 3.0
    assert p95([]) == 0.0


async def test_offline_eval_passes_gate(settings):
    summary = await run_eval(settings)
    assert summary.passed, summary.failures
    assert len(summary.cases) == 3
    # the pipeline's verifier removed claims the draft could not support
    assert all(c.draft_support_rate < 1.0 for c in summary.cases)
    assert "GATE: PASS" in format_summary(summary)


async def test_gate_fails_on_strict_thresholds(settings, tmp_path):
    strict = tmp_path / "t.json"
    strict.write_text(
        json.dumps(
            {
                "min_citation_precision": 1.01,
                "min_claim_support_rate": 0,
                "min_coverage": 0,
                "max_mean_cost_usd": 1,
                "max_p95_latency_s": 100,
            }
        )
    )
    summary = await run_eval(settings, thresholds=strict)
    assert not summary.passed and "citation_precision" in summary.failures[0]
