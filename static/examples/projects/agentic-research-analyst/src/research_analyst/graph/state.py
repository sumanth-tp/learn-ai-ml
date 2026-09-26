"""Graph state schemas and reducers.

Keys written by parallel workers need reducers (otherwise LangGraph raises
``InvalidUpdateError`` when two workers write in the same super-step). Keys that
a single node owns are plain values and simply overwrite.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from research_analyst.models import (
    Critique,
    Evidence,
    FinalReport,
    Gap,
    ResearchPlan,
    SectionDraft,
    Source,
    SubQuestion,
    Usage,
    Verdict,
    VerifiedClaim,
)
from research_analyst.quality import merge_sources


def add_usage(left: Usage | None, right: Usage | None) -> Usage:
    return (left or Usage()) + (right or Usage())


class ResearchState(TypedDict, total=False):
    # input
    question: str
    thread_id: str
    # supervisor
    plan: ResearchPlan
    # fan-in from workers (reducers)
    raw_sources: Annotated[dict[str, Source], merge_sources]
    raw_evidence: Annotated[list[Evidence], operator.add]
    gaps: Annotated[list[Gap], operator.add]
    usage: Annotated[Usage, add_usage]
    notes: Annotated[list[str], operator.add]
    # consolidated (single writer)
    sources: dict[str, Source]
    evidence: list[Evidence]
    # write / critique loop
    sections: dict[str, SectionDraft]
    critique: Critique | None
    revisions: int
    # verification and output
    claims_drafted: int
    verified: list[VerifiedClaim]
    report: FinalReport


class WorkerInput(TypedDict):
    """Payload of each ``Send``: a worker sees only its slice, not the whole state."""

    question: str
    sub_question: SubQuestion
    max_cost_usd: float
    max_tokens: int


class ResearcherState(TypedDict, total=False):
    """Private state of the researcher subgraph (CRAG loop for one sub-question)."""

    sub_question: SubQuestion
    max_cost_usd: float
    max_tokens: int
    query: str
    rewrites: int
    candidates: list[Source]
    kept: list[Source]
    scores: dict[str, float]
    verdict: Verdict
    evidence: list[Evidence]
    usage: Annotated[Usage, add_usage]
    notes: Annotated[list[str], operator.add]
