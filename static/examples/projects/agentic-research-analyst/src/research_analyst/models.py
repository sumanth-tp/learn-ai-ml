"""Domain models. Every value that crosses a node boundary is one of these."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- plan


class SubQuestion(BaseModel):
    id: str = Field(description="Short stable id such as 'sq1'.")
    question: str = Field(description="One focused research question.")
    search_queries: list[str] = Field(
        min_length=1, max_length=3, description="Keyword queries for retrieval and web search."
    )
    rationale: str = Field(default="", description="Why this sub-question matters.")


class ResearchPlan(BaseModel):
    """What the supervisor produces. Structured so it can be validated and fanned out."""

    title: str
    objective: str
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=8)


# ----------------------------------------------------------------------------- sources


class Origin(StrEnum):
    INTERNAL = "internal"
    WEB = "web"


class Source(BaseModel):
    id: str
    url: str
    title: str
    origin: Origin
    content: str
    published: date | None = None
    quality: float = 0.0
    aliases: list[str] = Field(default_factory=list, description="Ids merged into this one.")


class Evidence(BaseModel):
    """A refined strip of a source, relevant to one sub-question (CRAG knowledge refinement)."""

    sub_question_id: str
    source_id: str
    text: str
    relevance: float


class Verdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class DocGrade(BaseModel):
    source_id: str
    score: float = Field(ge=0, le=1)
    reason: str = ""


class DocGrades(BaseModel):
    grades: list[DocGrade]


class RewrittenQuery(BaseModel):
    query: str


# ----------------------------------------------------------------------------- writing


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list, description="Source ids backing the claim.")


class SectionDraft(BaseModel):
    sub_question_id: str
    heading: str
    claims: list[Claim]


class RevisionRequest(BaseModel):
    sub_question_id: str
    instruction: str


class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=1, le=5)


class Critique(BaseModel):
    # a list, not dict[str, float]: OpenAI strict JSON schema forbids free-form object keys
    scores: list[CriterionScore] = Field(description="One score 1..5 per rubric criterion.")
    overall: float = Field(ge=1, le=5)
    revision_requests: list[RevisionRequest] = Field(default_factory=list)
    summary: str = ""


SupportLevel = Literal["fully_supported", "partially_supported", "no_support"]


class SupportJudgement(BaseModel):
    """Self-RAG ISSUP token as a structured judgement."""

    level: SupportLevel
    reason: str = ""


class CitationCheck(BaseModel):
    source_id: str
    level: SupportLevel
    method: Literal["llm", "lexical"] = "llm"


class VerifiedClaim(BaseModel):
    sub_question_id: str
    text: str
    citations: list[str]
    checks: list[CitationCheck]
    status: Literal["supported", "partial", "unsupported"]


# ----------------------------------------------------------------------------- run metadata


class Usage(BaseModel):
    """Summed with ``+``; used as a LangGraph reducer so parallel workers add up."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    search_calls: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            search_calls=self.search_calls + other.search_calls,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Gap(BaseModel):
    sub_question_id: str
    question: str
    reason: str
    kind: Literal["timeout", "error", "no_evidence", "budget"]


class Reference(BaseModel):
    number: int
    source_id: str
    title: str
    url: str
    origin: Origin
    quality: float


class ReportSection(BaseModel):
    sub_question_id: str
    heading: str
    claims: list[VerifiedClaim]


class ReportMetrics(BaseModel):
    claims_drafted: int
    claims_kept: int
    claims_removed: int
    claim_support_rate: float
    citation_precision: float
    revisions: int
    critic_score: float | None


class FinalReport(BaseModel):
    thread_id: str
    question: str
    title: str
    sections: list[ReportSection]
    references: list[Reference]
    gaps: list[Gap]
    degraded: bool
    degradation_notes: list[str]
    metrics: ReportMetrics
    usage: Usage
    markdown: str
