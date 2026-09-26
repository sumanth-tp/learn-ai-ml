"""Asynchronous evaluator workers.

Jobs live in SQLite (eval_jobs), so a crash loses nothing: a stale 'running' job is
requeued. Each job gets a timeout and retries with exponential backoff; after the last
attempt it goes to 'failed' (the dead-letter state) with the error kept for triage.
Results are keyed (trace_id, evaluator), so a re-run overwrites rather than duplicates."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from agentmon.clock import Clock
from agentmon.evals.judges import JudgeError
from agentmon.evals.pipeline import CascadingEvaluator
from agentmon.store import Store

log = logging.getLogger("agentmon.workers")


class EvalWorkerPool:
    def __init__(
        self,
        store: Store,
        cascade: CascadingEvaluator,
        clock: Clock,
        *,
        concurrency: int = 4,
        timeout_s: float = 20.0,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
    ) -> None:
        self.store = store
        self.cascade = cascade
        self.clock = clock
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.stats: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    async def _process(self, trace_id: str) -> None:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            self.store.finish_job(trace_id, "failed", self.clock.now(), "trace not found")
            return
        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                outcome = await asyncio.wait_for(self.cascade.evaluate(trace), self.timeout_s)
            except (TimeoutError, JudgeError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._count("retries")
                log.warning(
                    "eval.retry",
                    extra={"trace_id": trace_id, "attempt": attempt + 1, "error": last_error},
                )
                await asyncio.sleep(self.backoff_base_s * (2**attempt))
                continue
            self.store.upsert_evals(outcome.results)
            if any(not r.passed for r in outcome.results if r.tier == "judge"):
                failed = [r.evaluator for r in outcome.results if not r.passed]
                self.store.add_review(trace_id, "judge:" + ",".join(failed), trace.ts)
            self.store.finish_job(trace_id, outcome.status, self.clock.now())
            self._count(outcome.status)
            return
        self.store.finish_job(trace_id, "failed", self.clock.now(), last_error)
        self._count("failed")

    async def run_once(self, batch: int = 64) -> int:
        """Drain the queue once. Returns the number of jobs processed."""
        sem = asyncio.Semaphore(self.concurrency)
        processed = 0

        async def guarded(tid: str) -> None:
            async with sem:
                await self._process(tid)

        while jobs := self.store.claim_jobs(batch, self.clock.now()):
            await asyncio.gather(*(guarded(j["trace_id"]) for j in jobs))
            processed += len(jobs)
        return processed

    async def run_forever(self, stop: asyncio.Event, poll_s: float = 1.0) -> None:
        self.store.requeue_stale_jobs(self.clock.now() - 300)
        while not stop.is_set():
            try:
                n = await self.run_once()
            except Exception:
                log.exception("workers.loop_error")
                n = 0
            if n == 0:  # idle: sleep until the next poll, or wake at once on shutdown
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), poll_s)
