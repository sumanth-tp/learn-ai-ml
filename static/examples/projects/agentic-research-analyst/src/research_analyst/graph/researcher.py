"""Researcher subgraph: Corrective RAG for one sub-question.

    retrieve_internal -> grade_internal --correct--------------------------> refine
                                       \\-incorrect/ambiguous-> rewrite -> web_search
                                                                 ^            |
                                                                 +--retry-- grade_web -> refine

CRAG verdicts (Yan et al., 2024):
* correct   - at least one internal doc scores >= upper: use internal docs only;
* incorrect - every internal doc scores < lower: discard them, rewrite, search the web;
* ambiguous - otherwise: keep the passable internal docs *and* add web results.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_analyst.budget import Budget, BudgetMode
from research_analyst.deps import Deps
from research_analyst.events import emit
from research_analyst.graph.state import ResearcherState
from research_analyst.models import Evidence, Origin, Source, Usage, Verdict
from research_analyst.providers.search import SearchError
from research_analyst.quality import looks_like_injection, quality_score, source_id_for
from research_analyst.text import overlap, sentences

MAX_EVIDENCE_PER_WORKER = 8


def _grading_question(state: ResearcherState) -> str:
    sq = state["sub_question"]
    return f"{sq.question} {' '.join(sq.search_queries)}"


def _budget_mode(state: ResearcherState) -> BudgetMode:
    budget = Budget(max_cost_usd=state["max_cost_usd"], max_tokens=state["max_tokens"])
    return budget.mode(state.get("usage") or Usage())


def build_researcher(deps: Deps) -> CompiledStateGraph:
    s = deps.settings

    async def retrieve_internal(state: ResearcherState) -> dict:
        sq = state["sub_question"]
        query = sq.search_queries[0]
        docs = await deps.index.search(query, k=s.retrieval_k)
        return {"query": query, "candidates": docs, "rewrites": 0, "kept": []}

    async def grade_internal(state: ResearcherState) -> dict:
        docs = state["candidates"]
        if not docs:
            return {"verdict": Verdict.INCORRECT, "scores": {}, "kept": []}
        grades, usage = await deps.brain.grade(_grading_question(state), docs)
        scores = {g.source_id: g.score for g in grades.grades}
        best = max(scores.values(), default=0.0)
        if best >= s.crag_upper:
            verdict = Verdict.CORRECT
        elif best < s.crag_lower:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.AMBIGUOUS
        kept = (
            []
            if verdict is Verdict.INCORRECT
            else [d for d in docs if scores.get(d.id, 0.0) >= s.crag_lower]
        )
        emit(
            "crag_verdict",
            sub_question_id=state["sub_question"].id,
            verdict=verdict.value,
            best_score=round(best, 3),
        )
        return {"verdict": verdict, "scores": scores, "kept": kept, "usage": usage}

    def after_internal(state: ResearcherState) -> Literal["refine", "rewrite_query"]:
        if state["verdict"] is Verdict.CORRECT:
            return "refine"
        if _budget_mode(state) is BudgetMode.EXHAUSTED:
            return "refine"
        return "rewrite_query"

    async def rewrite_query(state: ResearcherState) -> dict:
        rq, usage = await deps.brain.rewrite(state["sub_question"].question, state["query"])
        return {"query": rq.query, "rewrites": state.get("rewrites", 0) + 1, "usage": usage}

    async def web_search(state: ResearcherState) -> dict:
        emit("web_fallback", sub_question_id=state["sub_question"].id, query=state["query"])
        try:
            results = await deps.search.search(state["query"], k=s.web_k)
        except SearchError as exc:
            note = f"{state['sub_question'].id}: web search failed ({exc})"
            return {"candidates": [], "notes": [note]}
        web_docs = []
        for r in results:
            src = Source(
                id=source_id_for(r.url),
                url=r.url,
                title=r.title,
                origin=Origin.WEB,
                content=r.content,
                published=r.published,
            )
            src.quality = quality_score(src)
            web_docs.append(src)
        usage = Usage(search_calls=1, cost_usd=s.search_cost_usd)
        return {"candidates": web_docs, "usage": usage}

    async def grade_web(state: ResearcherState) -> dict:
        docs = [d for d in state["candidates"] if d.quality >= s.min_source_quality]
        dropped = len(state["candidates"]) - len(docs)
        notes = (
            [f"{state['sub_question'].id}: dropped {dropped} low-quality web source(s)"]
            if dropped
            else []
        )
        if not docs:
            return {"notes": notes}
        grades, usage = await deps.brain.grade(_grading_question(state), docs)
        scores = {**state.get("scores", {}), **{g.source_id: g.score for g in grades.grades}}
        known = {d.id for d in state["kept"]}
        kept = state["kept"] + [
            d for d in docs if scores.get(d.id, 0.0) >= s.crag_lower and d.id not in known
        ]
        return {"scores": scores, "kept": kept, "usage": usage, "notes": notes}

    def after_web(state: ResearcherState) -> Literal["refine", "rewrite_query"]:
        good = any(state["scores"].get(d.id, 0.0) >= s.crag_upper for d in state["kept"])
        if good:
            return "refine"
        # the first rewrite is part of the fallback; max_query_rewrites counts retries after it
        if state["rewrites"] > s.max_query_rewrites:
            return "refine"
        if _budget_mode(state) is not BudgetMode.NORMAL:
            return "refine"
        return "rewrite_query"

    async def refine(state: ResearcherState) -> dict:
        """CRAG knowledge refinement: decompose docs into strips, keep the relevant ones."""
        sq = state["sub_question"]
        question = _grading_question(state)
        scores = state.get("scores", {})
        strips: list[Evidence] = []
        suspicious = [d for d in state["kept"] if looks_like_injection(d.content)]
        kept = [d for d in state["kept"] if d not in suspicious]
        notes = [
            f"{sq.id}: quarantined possible prompt injection in '{d.title}'" for d in suspicious
        ]
        for doc in kept:
            for sent in sentences(doc.content):
                rel = overlap(question, sent)
                if rel >= 0.2:
                    strips.append(
                        Evidence(
                            sub_question_id=sq.id,
                            source_id=doc.id,
                            text=sent,
                            relevance=round(0.5 * rel + 0.5 * scores.get(doc.id, 0.0), 3),
                        )
                    )
        strips.sort(key=lambda e: (-e.relevance, e.source_id))
        return {"evidence": strips[:MAX_EVIDENCE_PER_WORKER], "kept": kept, "notes": notes}

    g = StateGraph(ResearcherState)
    g.add_node("retrieve_internal", retrieve_internal)
    g.add_node("grade_internal", grade_internal)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search)
    g.add_node("grade_web", grade_web)
    g.add_node("refine", refine)
    g.add_edge(START, "retrieve_internal")
    g.add_edge("retrieve_internal", "grade_internal")
    g.add_conditional_edges("grade_internal", after_internal, ["refine", "rewrite_query"])
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "grade_web")
    g.add_conditional_edges("grade_web", after_web, ["refine", "rewrite_query"])
    g.add_edge("refine", END)
    return g.compile(name="researcher")
