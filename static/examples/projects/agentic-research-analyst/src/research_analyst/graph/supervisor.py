"""The supervisor graph: plan, fan out researchers, consolidate, write/critique, verify.

START -> plan --Send x N--> research_worker --> consolidate -> write -> critique
                                                               ^         |
                                                               +-revise--+--> verify -> finalize
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import httpx
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

from research_analyst.budget import Budget, BudgetMode
from research_analyst.deps import Deps
from research_analyst.events import emit
from research_analyst.graph.researcher import build_researcher
from research_analyst.graph.state import ResearchState, WorkerInput
from research_analyst.models import (
    CitationCheck,
    Claim,
    Evidence,
    Gap,
    SectionDraft,
    Source,
    SubQuestion,
    Usage,
    VerifiedClaim,
)
from research_analyst.providers.llm import lexical_support
from research_analyst.quality import deduplicate
from research_analyst.report import build_report

log = logging.getLogger(__name__)

VERIFY_CONCURRENCY = 8


def is_transient(exc: Exception) -> bool:
    """Retry network-ish failures only. Validation errors and bugs must fail fast."""
    if isinstance(exc, NodeTimeoutError | TimeoutError | httpx.TransportError):
        return True
    try:
        import openai

        if isinstance(
            exc, openai.APIConnectionError | openai.RateLimitError | openai.InternalServerError
        ):
            return True
    except ImportError:  # pragma: no cover - openai ships with langchain-openai
        pass
    return False


def build_graph(deps: Deps, checkpointer=None) -> CompiledStateGraph:
    s = deps.settings
    budget = Budget(
        max_cost_usd=s.max_cost_usd, max_tokens=s.max_tokens, soft_ratio=s.soft_budget_ratio
    )
    researcher = build_researcher(deps)
    retry = RetryPolicy(
        max_attempts=3, initial_interval=1.0, backoff_factor=2.0, retry_on=is_transient
    )

    def mode(state: ResearchState) -> BudgetMode:
        return budget.mode(state.get("usage") or Usage())

    # ------------------------------------------------------------------ supervisor
    async def plan(state: ResearchState) -> dict:
        emit("run_started", question=state["question"])
        result, usage = await deps.brain.plan(state["question"], s.max_sub_questions)
        subs = result.sub_questions[: s.max_sub_questions]
        ids = [sq.id for sq in subs]
        if len(set(ids)) != len(ids):  # models sometimes repeat ids; make them unique
            subs = [sq.model_copy(update={"id": f"sq{i + 1}"}) for i, sq in enumerate(subs)]
        result = result.model_copy(update={"sub_questions": subs})
        emit(
            "plan_ready",
            title=result.title,
            sub_questions=[{"id": q.id, "question": q.question} for q in subs],
        )
        return {"plan": result, "usage": usage, "revisions": 0, "critique": None}

    def fan_out(state: ResearchState) -> list[Send]:
        subs = state["plan"].sub_questions
        if mode(state) is not BudgetMode.NORMAL:
            subs = subs[:2]
        share = budget.share(state.get("usage") or Usage(), len(subs))
        return [
            Send(
                "research_worker",
                WorkerInput(
                    question=state["question"],
                    sub_question=sq,
                    max_cost_usd=share.max_cost_usd,
                    max_tokens=share.max_tokens,
                ),
            )
            for sq in subs
        ]

    # ------------------------------------------------------------------ worker
    async def research_worker(payload: WorkerInput) -> dict:
        """Runs the researcher subgraph with its own state. Never raises: failures become gaps."""
        sq: SubQuestion = payload["sub_question"]
        emit("worker_started", sub_question_id=sq.id, question=sq.question)
        try:
            out = await asyncio.wait_for(
                researcher.ainvoke(
                    {
                        "sub_question": sq,
                        "max_cost_usd": payload["max_cost_usd"],
                        "max_tokens": payload["max_tokens"],
                    }
                ),
                timeout=s.worker_timeout_s,
            )
        except TimeoutError:
            emit("worker_failed", sub_question_id=sq.id, reason="timeout")
            return {
                "gaps": [
                    Gap(
                        sub_question_id=sq.id,
                        question=sq.question,
                        kind="timeout",
                        reason=f"researcher exceeded {s.worker_timeout_s:.0f}s",
                    )
                ]
            }
        except Exception as exc:
            log.exception("worker failed", extra={"sub_question_id": sq.id})
            emit("worker_failed", sub_question_id=sq.id, reason=type(exc).__name__)
            return {
                "gaps": [
                    Gap(
                        sub_question_id=sq.id,
                        question=sq.question,
                        kind="error",
                        reason=f"{type(exc).__name__}: {exc}"[:300],
                    )
                ]
            }
        evidence: list[Evidence] = out.get("evidence", [])
        cited = {e.source_id for e in evidence}
        sources = {d.id: d for d in out.get("kept", []) if d.id in cited}
        update: dict = {
            "raw_evidence": evidence,
            "raw_sources": sources,
            "usage": out.get("usage") or Usage(),
            "notes": out.get("notes", []),
        }
        if not evidence:
            update["gaps"] = [
                Gap(
                    sub_question_id=sq.id,
                    question=sq.question,
                    kind="no_evidence",
                    reason="no relevant internal or web evidence found",
                )
            ]
        emit(
            "worker_finished",
            sub_question_id=sq.id,
            evidence=len(evidence),
            sources=len(sources),
            verdict=str(out.get("verdict", "")),
        )
        return update

    # ------------------------------------------------------------------ consolidate
    async def consolidate(state: ResearchState) -> dict:
        raw: dict[str, Source] = state.get("raw_sources") or {}
        good = [src for src in raw.values() if src.quality >= s.min_source_quality]
        survivors, alias = deduplicate(good)
        evidence: list[Evidence] = []
        seen: set[tuple[str, str, str]] = set()
        for ev in state.get("raw_evidence") or []:
            if ev.source_id not in alias:
                continue  # its source was dropped for low quality
            ev = ev.model_copy(update={"source_id": alias[ev.source_id]})
            key = (ev.sub_question_id, ev.source_id, ev.text)
            if key not in seen:
                seen.add(key)
                evidence.append(ev)
        merged = len(good) - len(survivors)
        emit(
            "sources_consolidated",
            raw=len(raw),
            kept=len(survivors),
            merged_duplicates=merged,
            dropped_low_quality=len(raw) - len(good),
        )
        return {"sources": {src.id: src for src in survivors}, "evidence": evidence}

    # ------------------------------------------------------------------ write / critique
    async def write(state: ResearchState) -> dict:
        subs = {sq.id: sq for sq in state["plan"].sub_questions}
        by_sq: dict[str, list[Evidence]] = {}
        for ev in state.get("evidence") or []:
            by_sq.setdefault(ev.sub_question_id, []).append(ev)
        sections = dict(state.get("sections") or {})
        critique = state.get("critique")
        if critique is None:  # first draft: every sub-question with evidence
            targets = {sid: None for sid in subs if by_sq.get(sid)}
        else:  # revision: only the sections the critic asked for
            targets = {
                r.sub_question_id: r.instruction
                for r in critique.revision_requests
                if r.sub_question_id in subs and by_sq.get(r.sub_question_id)
            }

        async def one(sid: str, feedback: str | None) -> tuple[SectionDraft, Usage]:
            draft, usage = await deps.brain.write_section(subs[sid], by_sq[sid], feedback)
            allowed = {e.source_id for e in by_sq[sid]}
            claims = []
            for c in draft.claims:  # citations to ids not in the evidence are hallucinated
                valid = [cid for cid in dict.fromkeys(c.citations) if cid in allowed]
                if valid:
                    claims.append(Claim(text=c.text.strip(), citations=valid))
            return draft.model_copy(update={"sub_question_id": sid, "claims": claims}), usage

        results = await asyncio.gather(*(one(sid, fb) for sid, fb in targets.items()))
        usage = Usage()
        for draft, u in results:
            sections[draft.sub_question_id] = draft
            usage = usage + u
        revisions = state.get("revisions", 0) + (1 if critique is not None else 0)
        emit("draft_ready", sections=len(sections), rewritten=list(targets), revision=revisions)
        return {
            "sections": sections,
            "usage": usage,
            "revisions": revisions,
            "claims_drafted": sum(len(x.claims) for x in sections.values()),
        }

    async def critique(state: ResearchState) -> dict:
        if mode(state) is not BudgetMode.NORMAL:
            emit("budget_degraded", skipped="critique")
            return {"critique": None, "notes": ["budget soft limit reached: critic loop skipped"]}
        sections = list(state["sections"].values())
        try:
            crit, usage = await deps.brain.critique(state["question"], sections)
        except Exception as exc:  # the critic is an optimisation, not a dependency
            log.warning("critic failed", exc_info=True)
            return {"critique": None, "notes": [f"critic unavailable ({type(exc).__name__})"]}
        emit(
            "critique",
            overall=crit.overall,
            scores={c.criterion: c.score for c in crit.scores},
            revision_requests=[r.sub_question_id for r in crit.revision_requests],
        )
        return {"critique": crit, "usage": usage}

    def after_critique(state: ResearchState) -> Literal["write", "verify"]:
        crit = state.get("critique")
        if crit is None or crit.overall >= s.critic_pass_score or not crit.revision_requests:
            return "verify"
        if state.get("revisions", 0) >= s.max_revisions or mode(state) is not BudgetMode.NORMAL:
            return "verify"
        return "write"

    # ------------------------------------------------------------------ Self-RAG verification
    async def verify(state: ResearchState) -> dict:
        sources = state.get("sources") or {}
        exhausted = mode(state) is BudgetMode.EXHAUSTED
        sem = asyncio.Semaphore(VERIFY_CONCURRENCY)
        notes: list[str] = []
        if exhausted:
            emit("budget_degraded", skipped="llm verification")
            notes.append("budget exhausted: claims verified lexically, not by the LLM")

        async def check(claim: str, sid: str) -> tuple[CitationCheck, Usage]:
            text = sources[sid].content if sid in sources else ""
            if exhausted:
                return CitationCheck(
                    source_id=sid, level=lexical_support(claim, text).level, method="lexical"
                ), Usage()
            async with sem:
                try:
                    j, u = await asyncio.wait_for(
                        deps.brain.check_support(claim, text), timeout=s.claim_check_timeout_s
                    )
                    return CitationCheck(source_id=sid, level=j.level, method="llm"), u
                except Exception:
                    log.warning("claim check failed; lexical fallback", exc_info=True)
                    return CitationCheck(
                        source_id=sid, level=lexical_support(claim, text).level, method="lexical"
                    ), Usage()

        jobs = [(sec.sub_question_id, c) for sec in state["sections"].values() for c in sec.claims]
        results = await asyncio.gather(
            *(asyncio.gather(*(check(c.text, sid) for sid in c.citations)) for _, c in jobs)
        )
        verified: list[VerifiedClaim] = []
        usage = Usage()
        for (sqid, claim), checks in zip(jobs, results, strict=True):
            cks = [ck for ck, _ in checks]
            for _, u in checks:
                usage = usage + u
            levels = {ck.level for ck in cks}
            status = (
                "supported"
                if "fully_supported" in levels
                else "partial"
                if "partially_supported" in levels
                else "unsupported"
            )
            verified.append(
                VerifiedClaim(
                    sub_question_id=sqid,
                    text=claim.text,
                    checks=cks,
                    status=status,
                    citations=[ck.source_id for ck in cks if ck.level != "no_support"],
                )
            )
        counts = {
            k: sum(1 for v in verified if v.status == k)
            for k in ("supported", "partial", "unsupported")
        }
        emit("verification", **counts)
        return {"verified": verified, "usage": usage, "notes": notes}

    async def finalize(state: ResearchState) -> dict:
        report = build_report(state)
        emit(
            "report_ready",
            title=report.title,
            claims=report.metrics.claims_kept,
            gaps=len(report.gaps),
            cost_usd=report.usage.cost_usd,
            degraded=report.degraded,
        )
        return {"report": report}

    g = StateGraph(ResearchState)
    timeout = s.node_timeout_s
    g.add_node("plan", plan, retry_policy=retry, timeout=timeout)
    g.add_node("research_worker", research_worker)
    g.add_node("consolidate", consolidate)
    g.add_node("write", write, retry_policy=retry, timeout=timeout)
    g.add_node("critique", critique, timeout=timeout)
    g.add_node("verify", verify, retry_policy=retry, timeout=timeout)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["research_worker"])
    g.add_edge("research_worker", "consolidate")
    g.add_edge("consolidate", "write")
    g.add_edge("write", "critique")
    g.add_conditional_edges("critique", after_critique, ["write", "verify"])
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, name="research_analyst")
