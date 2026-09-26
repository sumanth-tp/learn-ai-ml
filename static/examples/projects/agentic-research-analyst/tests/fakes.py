"""Test doubles layered on the offline fakes to trigger each failure path on purpose."""

from __future__ import annotations

import asyncio
from collections import Counter

from research_analyst.models import Claim, CriterionScore, Critique, RevisionRequest, SectionDraft
from research_analyst.providers.llm import HeuristicBrain
from research_analyst.providers.search import SearchError, WebResult


class CountingBrain(HeuristicBrain):
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    async def plan(self, *a, **k):
        self.calls["plan"] += 1
        return await super().plan(*a, **k)

    async def grade(self, *a, **k):
        self.calls["grade"] += 1
        return await super().grade(*a, **k)

    async def write_section(self, *a, **k):
        self.calls["write_section"] += 1
        return await super().write_section(*a, **k)

    async def critique(self, *a, **k):
        self.calls["critique"] += 1
        return await super().critique(*a, **k)


class SimulatedCrash(RuntimeError):
    """Stands in for the process dying mid-run (OOM kill, deploy, node restart)."""


class CrashOnceInWriter(CountingBrain):
    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    async def write_section(self, *a, **k):
        if not self.crashed:
            self.crashed = True
            raise SimulatedCrash("process killed during write")
        return await super().write_section(*a, **k)


class SlowOnRisks(CountingBrain):
    """The 'risks' worker hangs, e.g. a provider that never answers."""

    async def grade(self, question, docs):
        if "risk" in question.lower():
            await asyncio.sleep(5)
        return await super().grade(question, docs)


class BrokenOnCosts(CountingBrain):
    async def grade(self, question, docs):
        if "cost" in question.lower() and "economics" in question.lower():
            raise ValueError("provider returned malformed JSON")
        return await super().grade(question, docs)


class NeverSatisfiedCritic(CountingBrain):
    async def critique(self, question, sections):
        self.calls["critique"] += 1
        return Critique(
            scores=[CriterionScore(criterion="coverage", score=2.0)],
            overall=2.0,
            revision_requests=[
                RevisionRequest(sub_question_id=s.sub_question_id, instruction="more detail")
                for s in sections
            ],
        ), (await HeuristicBrain.critique(self, question, sections))[1]


class HallucinatingWriter(CountingBrain):
    async def write_section(self, sq, evidence, feedback):
        draft, usage = await super().write_section(sq, evidence, feedback)
        claims = [
            *draft.claims,
            Claim(text="Sodium-ion costs 10 USD per kWh.", citations=["S-made-up-id"]),
        ]
        return SectionDraft(sub_question_id=sq.id, heading=draft.heading, claims=claims), usage


class FailingSearch:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, k: int) -> list[WebResult]:
        self.calls += 1
        raise SearchError("upstream 503 after retries")


class EmptyIndex:
    async def search(self, query: str, k: int):
        return []
