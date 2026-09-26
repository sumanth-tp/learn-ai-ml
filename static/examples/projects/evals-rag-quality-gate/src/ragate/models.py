"""Domain models shared across the RAG app, the dataset tooling and the eval harness."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    owner: str = ""
    updated: str = ""
    access: Literal["public", "restricted"] = "public"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    source: str = "dense"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StageTimings(BaseModel):
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms + self.generate_ms


class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    timings: StageTimings = Field(default_factory=StageTimings)


# ---------------------------------------------------------------- golden dataset


class QuestionType(StrEnum):
    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class ExpectedBehaviour(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    CORRECT_PREMISE = "correct_premise"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_EDIT = "needs_edit"
    REJECTED = "rejected"


class Evidence(BaseModel):
    """A span of a source document that supports the reference answer.

    Evidence is anchored to (doc_id, quote), not to chunk ids, so that relevance
    labels survive re-chunking. Chunk-size experiments would be meaningless otherwise.
    """

    doc_id: str
    quote: str


class GoldenItem(BaseModel):
    item_id: str
    question: str
    question_type: QuestionType
    expected_behaviour: ExpectedBehaviour = ExpectedBehaviour.ANSWER
    reference_answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: Literal["human", "synthetic"] = "human"
    review_status: ReviewStatus = ReviewStatus.APPROVED
    reviewer: str = ""
    notes: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.expected_behaviour != ExpectedBehaviour.REFUSE and bool(self.evidence)
