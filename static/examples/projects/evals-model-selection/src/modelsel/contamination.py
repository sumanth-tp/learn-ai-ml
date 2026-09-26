"""Contamination checks: has a candidate seen the benchmark before?

Two signals, both cheap:

1. **Completion probe.** Give the model the first half of a ticket and ask it to
   continue verbatim. A model that reproduces the true second half (high 8-gram
   overlap) has memorised it. Unseen text gets a generic continuation.
2. **Public-private gap.** Score the public test split and the held-out private
   split. A model whose quality drops much more than others on private items has
   probably been tuned on the public ones.

Neither proves innocence. They catch the common, accidental case: benchmark files
scraped into a training set or used for prompt tuning.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from modelsel.harness.client import LLMClient
from modelsel.schemas import BenchmarkItem

PROBE_SYSTEM = """TASK: continue
Continue the text exactly as it would appear in the original document. Output only the continuation."""


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap(truth: str, produced: str, n: int = 8) -> float:
    """Share of the true continuation's n-grams the model reproduced. Falls back to
    shorter n for short texts so a 12-word suffix can still be measured."""
    for size in (n, 5, 3):
        t = ngrams(truth, size)
        if t:
            return len(t & ngrams(produced, size)) / len(t)
    return 0.0


@dataclass(frozen=True)
class ProbeResult:
    model_id: str
    split: str
    mean_overlap: float
    flagged: bool


async def completion_probe(
    client: LLMClient, model_id: str, items: list[BenchmarkItem], *, threshold: float = 0.5, split: str = ""
) -> ProbeResult:
    async def one(it: BenchmarkItem) -> float:
        cut = len(it.ticket) // 2
        prefix, truth = it.ticket[:cut], it.ticket[cut:]
        msgs = [SystemMessage(content=PROBE_SYSTEM), HumanMessage(content=f"PREFIX:\n{prefix}")]
        c = await client.complete(model_id, msgs, prompt_version="probe-v1", max_tokens=120, tags=["probe"])
        return overlap(truth, c.text)

    scores = await asyncio.gather(*(one(it) for it in items))
    mean = sum(scores) / len(scores) if scores else 0.0
    return ProbeResult(model_id, split, mean, mean >= threshold)
