"""Run N candidate models over benchmark items, concurrently, through one client.

A failed call does not abort the run: it becomes a ``Prediction`` with ``error`` set
and an empty output, which the metrics score as wrong. A model that times out
10% of the time should lose points for it, not crash the comparison.
"""

from __future__ import annotations

import asyncio
import logging

from modelsel.harness.client import LLMCallError, LLMClient
from modelsel.logging_setup import log_event
from modelsel.schemas import TASKS, BenchmarkItem, Prediction, Task, Usage
from modelsel.tasks import MAX_TOKENS, PROMPT_VERSION, build_messages

logger = logging.getLogger(__name__)


async def predict_one(client: LLMClient, run_id: str, model_id: str, item: BenchmarkItem, task: Task) -> Prediction:
    try:
        c = await client.complete(
            model_id,
            build_messages(task, item),
            prompt_version=PROMPT_VERSION,
            max_tokens=MAX_TOKENS[task],
            tags=[task, item.split],
        )
    except LLMCallError as exc:
        return Prediction(
            run_id=run_id,
            model_id=model_id,
            item_id=item.id,
            split=item.split,
            task=task,
            output="",
            usage=Usage(),
            latency_ms=0.0,
            cost_usd=0.0,
            cached=False,
            error=str(exc),
        )
    return Prediction(
        run_id=run_id,
        model_id=model_id,
        item_id=item.id,
        split=item.split,
        task=task,
        output=c.text,
        usage=c.usage,
        latency_ms=c.latency_ms,
        cost_usd=c.cost_usd,
        cached=c.cached,
    )


async def run_candidates(
    client: LLMClient, run_id: str, model_ids: list[str], items: list[BenchmarkItem]
) -> list[Prediction]:
    """Fan out model x item x task. Concurrency is bounded inside the client
    (global semaphore + per-model token bucket), so gathering everything is safe."""
    jobs = [predict_one(client, run_id, m, it, t) for m in model_ids for it in items for t in TASKS]
    preds = await asyncio.gather(*jobs)
    for m in model_ids:
        mine = [p for p in preds if p.model_id == m]
        log_event(
            logger,
            "candidate_done",
            model_id=m,
            calls=len(mine),
            errors=sum(p.error is not None for p in mine),
            cache_hits=sum(p.cached for p in mine),
            cost_usd=round(sum(p.cost_usd for p in mine), 6),
        )
    return list(preds)
