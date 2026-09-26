"""Offline evaluation + regression gate. ``research-analyst eval`` exits non-zero on regression."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from research_analyst.checkpoint import memory_checkpointer
from research_analyst.config import Settings
from research_analyst.evals.metrics import CaseResult, EvalCase, coverage, judge_report, p95
from research_analyst.providers.llm import build_brain
from research_analyst.service import ResearchService

EVAL_DIR = Path(__file__).resolve().parent


class Thresholds(BaseModel):
    min_citation_precision: float
    min_claim_support_rate: float
    min_coverage: float
    max_mean_cost_usd: float
    max_p95_latency_s: float


class EvalSummary(BaseModel):
    cases: list[CaseResult]
    citation_precision: float
    claim_support_rate: float
    coverage: float
    mean_cost_usd: float
    p95_latency_s: float
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_cases(path: Path | None = None) -> list[EvalCase]:
    path = path or EVAL_DIR / "dataset.jsonl"
    return [
        EvalCase.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def load_thresholds(path: Path | None = None) -> Thresholds:
    return Thresholds.model_validate_json((path or EVAL_DIR / "thresholds.json").read_text())


async def run_eval(
    settings: Settings,
    dataset: Path | None = None,
    thresholds: Path | None = None,
    service: ResearchService | None = None,
) -> EvalSummary:
    cases = load_cases(dataset)
    limits = load_thresholds(thresholds)
    judge = build_brain(settings, judge=True)
    results: list[CaseResult] = []
    svc = service or ResearchService(settings, checkpointer=memory_checkpointer())
    async with svc:
        for case in cases:
            started = time.perf_counter()
            report = await svc.run(case.question, thread_id=f"eval-{case.id}-{time.time_ns()}")
            latency = time.perf_counter() - started
            sources = await svc.sources(report.thread_id)
            precision, support = await judge_report(report, sources, judge)
            results.append(
                CaseResult(
                    id=case.id,
                    citation_precision=precision,
                    claim_support_rate=support,
                    coverage=coverage(report, case.reference_outline),
                    draft_support_rate=report.metrics.claim_support_rate,
                    claims=report.metrics.claims_kept,
                    gaps=len(report.gaps),
                    cost_usd=report.usage.cost_usd,
                    tokens=report.usage.total_tokens,
                    latency_s=round(latency, 3),
                )
            )

    def mean(field: str) -> float:
        return round(sum(getattr(r, field) for r in results) / len(results), 4)

    summary = EvalSummary(
        cases=results,
        citation_precision=mean("citation_precision"),
        claim_support_rate=mean("claim_support_rate"),
        coverage=mean("coverage"),
        mean_cost_usd=mean("cost_usd"),
        p95_latency_s=p95([r.latency_s for r in results]),
        failures=[],
    )
    checks = [
        (
            summary.citation_precision >= limits.min_citation_precision,
            f"citation_precision {summary.citation_precision} < {limits.min_citation_precision}",
        ),
        (
            summary.claim_support_rate >= limits.min_claim_support_rate,
            f"claim_support_rate {summary.claim_support_rate} < {limits.min_claim_support_rate}",
        ),
        (
            summary.coverage >= limits.min_coverage,
            f"coverage {summary.coverage} < {limits.min_coverage}",
        ),
        (
            summary.mean_cost_usd <= limits.max_mean_cost_usd,
            f"mean_cost_usd {summary.mean_cost_usd} > {limits.max_mean_cost_usd}",
        ),
        (
            summary.p95_latency_s <= limits.max_p95_latency_s,
            f"p95_latency_s {summary.p95_latency_s} > {limits.max_p95_latency_s}",
        ),
    ]
    summary.failures = [msg for ok, msg in checks if not ok]
    return summary


def format_summary(summary: EvalSummary) -> str:
    head = (
        f"{'case':<20}{'cit.prec':>9}{'support':>9}{'draft':>7}{'cover':>7}"
        f"{'claims':>7}{'gaps':>5}{'cost$':>9}{'lat.s':>7}"
    )
    rows = [head, "-" * len(head)]
    for r in summary.cases:
        rows.append(
            f"{r.id:<20}{r.citation_precision:>9.2f}{r.claim_support_rate:>9.2f}"
            f"{r.draft_support_rate:>7.2f}{r.coverage:>7.2f}{r.claims:>7}{r.gaps:>5}"
            f"{r.cost_usd:>9.4f}{r.latency_s:>7.2f}"
        )
    rows.append("-" * len(head))
    rows.append(
        f"mean citation precision {summary.citation_precision:.3f} | claim support "
        f"{summary.claim_support_rate:.3f} | coverage {summary.coverage:.3f} | "
        f"mean cost ${summary.mean_cost_usd:.4f} | p95 latency {summary.p95_latency_s:.2f}s"
    )
    rows.append(
        "GATE: PASS" if summary.passed else "GATE: FAIL\n  " + "\n  ".join(summary.failures)
    )
    return "\n".join(rows)


def write_results(summary: EvalSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.model_dump(), indent=2))
