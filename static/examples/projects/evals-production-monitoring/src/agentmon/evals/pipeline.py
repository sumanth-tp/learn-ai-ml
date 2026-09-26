"""The online evaluation pipeline.

ingest(trace)      -> tier-0 heuristics inline, sampling decision, enqueue a judge job,
                      and open a review item for anything that failed or was flagged.
on_feedback(...)   -> store it; negative feedback always enqueues and opens a review.
CascadingEvaluator -> what a worker runs for one job: skip the judges if heuristics already
                      decided, respect the daily judge budget, else run all judges."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agentmon.clock import Clock
from agentmon.evals.heuristics import decided_by_heuristics, run_heuristics
from agentmon.evals.judges import LLMJudge
from agentmon.evals.sampling import Sampler
from agentmon.models import EvalResult, Feedback, TraceRecord
from agentmon.store import Store

log = logging.getLogger("agentmon.evals")
DAY = 86_400.0


@dataclass
class CascadeOutcome:
    results: list[EvalResult]
    status: str  # done | decided_by_heuristics | skipped_budget | skipped_blocked


class CascadingEvaluator:
    def __init__(self, store: Store, judges: list[LLMJudge], budget_usd_per_day: float) -> None:
        self.store = store
        self.judges = judges
        self.budget = budget_usd_per_day

    async def evaluate(self, trace: TraceRecord) -> CascadeOutcome:
        heuristics = [e for e in self.store.evals_for(trace.trace_id) if e.tier == "heuristic"]
        if decided_by_heuristics(heuristics):
            marker = EvalResult(
                trace_id=trace.trace_id,
                evaluator="cascade",
                score=0.0,
                passed=False,
                reason="decided by tier-0 heuristics",
                tier="cascade",
                ts=trace.ts,
            )
            return CascadeOutcome([marker], "decided_by_heuristics")
        if trace.status == "blocked":
            marker = EvalResult(
                trace_id=trace.trace_id,
                evaluator="cascade",
                score=1.0,
                passed=True,
                reason="blocked by input guard; refusal is policy",
                tier="cascade",
                ts=trace.ts,
            )
            return CascadeOutcome([marker], "skipped_blocked")
        day_start = trace.ts - (trace.ts % DAY)
        if self.store.judge_spend_between(day_start, day_start + DAY) >= self.budget:
            return CascadeOutcome([], "skipped_budget")
        results = await asyncio.gather(*(j.aevaluate(trace) for j in self.judges))
        return CascadeOutcome(list(results), "done")


class OnlineEvalPipeline:
    def __init__(self, store: Store, sampler: Sampler, clock: Clock, latency_slo_ms: float) -> None:
        self.store = store
        self.sampler = sampler
        self.clock = clock
        self.latency_slo_ms = latency_slo_ms

    def ingest(self, trace: TraceRecord) -> None:
        heuristics = run_heuristics(trace, self.latency_slo_ms)
        self.store.upsert_evals(heuristics)
        decision = self.sampler.decide(trace)
        if decision.sampled:
            self.store.enqueue_eval(trace.trace_id, decision.reason, trace.ts)
        failed = [h.evaluator for h in heuristics if not h.passed]
        if failed:
            self.store.add_review(trace.trace_id, "heuristic:" + ",".join(failed), trace.ts)
        elif any(f.startswith(("blocked:", "sanitised:", "redacted_")) for f in trace.flags):
            self.store.add_review(trace.trace_id, "guardrail:" + ",".join(trace.flags), trace.ts)

    def on_feedback(self, fb: Feedback) -> None:
        trace = self.store.get_trace(fb.trace_id)
        if trace is None:
            raise KeyError(fb.trace_id)
        self.store.upsert_feedback(fb)
        if fb.rating < 0:
            self.store.enqueue_eval(fb.trace_id, "always:negative_feedback", fb.ts)
            self.store.add_review(fb.trace_id, "feedback:thumbs_down", fb.ts)
            log.info("feedback.negative", extra={"trace_id": fb.trace_id})
