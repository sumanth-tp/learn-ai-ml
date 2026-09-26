"""Which traces get the expensive judges. Decisions are a pure function of the trace
(hash-based, not random.random()), so replays and backfills sample the same traces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from agentmon.models import TraceRecord


@dataclass(frozen=True)
class SampleDecision:
    sampled: bool
    reason: str


def unit_hash(key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class Sampler(Protocol):
    def decide(self, trace: TraceRecord) -> SampleDecision: ...


class AlwaysSampleRules:
    """Errors and guardrail hits are always worth a look; they are also rare."""

    def decide(self, trace: TraceRecord) -> SampleDecision:
        if trace.status == "error":
            return SampleDecision(True, "always:error")
        if any(f.startswith(("blocked:", "sanitised:", "redacted_")) for f in trace.flags):
            return SampleDecision(True, "always:guardrail")
        return SampleDecision(False, "")


class StratifiedSampler:
    """Per-intent rates so rare-but-risky intents (transfers) are not drowned out by
    high-volume ones (balance checks)."""

    def __init__(self, rates: dict[str, float], seed: str) -> None:
        self.rates = rates
        self.seed = seed

    def decide(self, trace: TraceRecord) -> SampleDecision:
        rate = self.rates.get(trace.intent)
        if rate is not None and unit_hash(trace.trace_id, self.seed + ":strat") < rate:
            return SampleDecision(True, f"stratified:{trace.intent}")
        return SampleDecision(False, "")


class RandomSampler:
    def __init__(self, rate: float, seed: str) -> None:
        self.rate = rate
        self.seed = seed

    def decide(self, trace: TraceRecord) -> SampleDecision:
        if unit_hash(trace.trace_id, self.seed) < self.rate:
            return SampleDecision(True, "random")
        return SampleDecision(False, "")


class CompositeSampler:
    def __init__(self, samplers: list[Sampler]) -> None:
        self.samplers = samplers

    def decide(self, trace: TraceRecord) -> SampleDecision:
        for s in self.samplers:
            d = s.decide(trace)
            if d.sampled:
                return d
        return SampleDecision(False, "not_sampled")
