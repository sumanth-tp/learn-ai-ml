"""Integration tests of the whole supervisor graph with offline fakes, plus failure paths."""

from __future__ import annotations

import dataclasses

from research_analyst.checkpoint import memory_checkpointer
from research_analyst.service import ResearchService
from tests.conftest import HYDROGEN_Q, SODIUM_Q
from tests.fakes import (
    BrokenOnCosts,
    CountingBrain,
    HallucinatingWriter,
    NeverSatisfiedCritic,
    SlowOnRisks,
)


async def _run(settings, deps, question=SODIUM_Q):
    events = []
    async with ResearchService(settings, deps=deps, checkpointer=memory_checkpointer()) as svc:
        async for ev in svc.stream(question, "t-1"):
            events.append(ev)
        st = await svc.status("t-1")
        sources = await svc.sources("t-1")
    return st.report, events, sources


async def test_end_to_end_report_is_fully_cited(settings, deps):
    report, _events, sources = await _run(settings, deps)
    assert report is not None and not report.degraded
    assert len(report.sections) == 4 and report.gaps == []
    ref_ids = {r.source_id for r in report.references}
    for sec in report.sections:
        for claim in sec.claims:
            assert claim.citations, "every kept claim has a citation"
            assert set(claim.citations) <= ref_ids <= set(sources)
            assert claim.status in ("supported", "partial")
    # the fake writer's unsupported 'synthesis' sentences were caught by Self-RAG checks
    assert report.metrics.claims_removed > 0
    assert all("consensus" not in c.text for s in report.sections for c in s.claims)
    assert [r.number for r in report.references] == list(range(1, len(report.references) + 1))
    assert "## References" in report.markdown and "[1]" in report.markdown


async def test_stream_emits_progress_in_order(settings, deps):
    _, events, _ = await _run(settings, deps)
    types = [e.type for e in events]
    assert types[0] == "run_started" and types[-1] == "report_ready"
    assert types.index("plan_ready") < types.index("worker_started")
    assert types.count("worker_started") == 4 == types.count("worker_finished")
    assert types.index("sources_consolidated") > max(
        i for i, t in enumerate(types) if t == "worker_finished"
    )
    assert "verification" in types and "crag_verdict" in types


async def test_web_fallback_and_citations_for_out_of_index_topic(settings, deps):
    report, events, _ = await _run(settings, deps, HYDROGEN_Q)
    assert any(e.type == "web_fallback" for e in events)
    assert any(r.origin.value == "web" for r in report.references)


async def test_worker_timeout_becomes_gap_and_report_completes(settings, deps):
    s = settings.model_copy(update={"worker_timeout_s": 0.3})
    d = dataclasses.replace(deps, settings=s, brain=SlowOnRisks())
    report, events, _ = await _run(s, d)
    assert report is not None and report.degraded
    kinds = {g.kind for g in report.gaps}
    assert "timeout" in kinds
    assert len(report.sections) == 3
    assert "## Gaps" in report.markdown
    assert any(e.type == "worker_failed" for e in events)


async def test_worker_exception_becomes_gap(settings, deps):
    d = dataclasses.replace(deps, brain=BrokenOnCosts())
    report, _, _ = await _run(settings, d)
    gap = next(g for g in report.gaps if g.kind == "error")
    assert "malformed JSON" in gap.reason
    assert len(report.sections) == 3


async def test_budget_exhaustion_degrades_gracefully(settings, deps):
    s = settings.model_copy(update={"max_cost_usd": 0.001})
    brain = CountingBrain()
    d = dataclasses.replace(deps, settings=s, brain=brain)
    report, events, _ = await _run(s, d)
    assert report is not None and report.degraded
    assert brain.calls["critique"] == 0  # the optional critic loop was skipped
    assert any(n.startswith("budget") for n in report.degradation_notes)
    checks = [ck for sec in report.sections for c in sec.claims for ck in c.checks]
    assert checks and all(ck.method == "lexical" for ck in checks)
    assert any(e.type == "budget_degraded" for e in events)


async def test_revision_loop_is_bounded_by_max_revisions(settings, deps):
    s = settings.model_copy(update={"max_revisions": 2})
    brain = NeverSatisfiedCritic()
    d = dataclasses.replace(deps, settings=s, brain=brain)
    report, _, _ = await _run(s, d)
    assert report.metrics.revisions == 2
    assert brain.calls["critique"] == 3  # draft + 2 revisions, then stop
    assert report.metrics.critic_score == 2.0


async def test_hallucinated_citation_ids_are_dropped(settings, deps):
    d = dataclasses.replace(deps, brain=HallucinatingWriter())
    report, _, _ = await _run(settings, d)
    texts = [c.text for s in report.sections for c in s.claims]
    assert "Sodium-ion costs 10 USD per kWh." not in texts


async def test_same_thread_id_is_idempotent(settings, deps):
    brain = CountingBrain()
    d = dataclasses.replace(deps, brain=brain)
    async with ResearchService(settings, deps=d, checkpointer=memory_checkpointer()) as svc:
        first = await svc.run(SODIUM_Q, "same")
        second = await svc.run(SODIUM_Q, "same")
    assert first == second
    assert brain.calls["plan"] == 1
