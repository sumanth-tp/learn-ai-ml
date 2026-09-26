"""The CRAG researcher subgraph, tested on its own state."""

from __future__ import annotations

import dataclasses

from research_analyst.graph.researcher import build_researcher
from research_analyst.models import SubQuestion
from research_analyst.providers.search import StubWebSearch
from tests.fakes import EmptyIndex, FailingSearch


def _sq(q: str, query: str) -> SubQuestion:
    return SubQuestion(id="sq1", question=q, search_queries=[query])


def _input(sq: SubQuestion) -> dict:
    return {"sub_question": sq, "max_cost_usd": 1.0, "max_tokens": 1_000_000}


async def test_correct_verdict_uses_internal_only(deps):
    search = StubWebSearch(deps.settings.corpus_dir / "web.jsonl")
    g = build_researcher(dataclasses.replace(deps, search=search))
    out = await g.ainvoke(
        _input(
            _sq(
                "What are the costs and economics of LFP batteries?",
                "lfp battery cost kwh procurement",
            )
        )
    )
    assert out["verdict"] == "correct"
    assert search.calls == []  # no web fallback on the correct path
    assert out["evidence"] and all(e.sub_question_id == "sq1" for e in out["evidence"])


async def test_incorrect_verdict_falls_back_to_web(deps):
    search = StubWebSearch(deps.settings.corpus_dir / "web.jsonl")
    g = build_researcher(dataclasses.replace(deps, search=search, index=EmptyIndex()))
    out = await g.ainvoke(
        _input(_sq("How will electrolyser cost change by 2030?", "electrolyser cost 2030"))
    )
    assert out["verdict"] == "incorrect"
    assert len(search.calls) >= 1 and out["rewrites"] >= 1
    assert out["evidence"] and all(d.origin.value == "web" for d in out["kept"])


async def test_rewrite_loop_is_bounded(deps):
    search = FailingSearch()
    settings = deps.settings.model_copy(update={"max_query_rewrites": 1})
    g = build_researcher(
        dataclasses.replace(deps, settings=settings, search=search, index=EmptyIndex())
    )
    out = await g.ainvoke(_input(_sq("Something nobody has written about?", "zzz qqq")))
    assert search.calls == 2  # first rewrite + one retry, then give up
    assert out["evidence"] == []
    assert any("web search failed" in n for n in out["notes"])


async def test_low_quality_web_sources_are_dropped(deps):
    search = StubWebSearch(deps.settings.corpus_dir / "web.jsonl")
    g = build_researcher(dataclasses.replace(deps, search=search, index=EmptyIndex()))
    out = await g.ainvoke(
        _input(
            _sq("Top 10 sodium ion batteries to buy now deals?", "top 10 sodium ion batteries buy")
        )
    )
    assert all("best-battery-deals" not in d.url for d in out["kept"])
    assert any("low-quality" in n for n in out["notes"])


async def test_prompt_injection_document_is_quarantined(deps):
    g = build_researcher(deps)
    out = await g.ainvoke(
        _input(
            _sq(
                "What does the sodium-ion supplier brochure say?",
                "sodium-ion battery supplier brochure email",
            )
        )
    )
    assert any("quarantined" in n for n in out["notes"])
    assert not any("IGNORE" in e.text for e in out["evidence"])
