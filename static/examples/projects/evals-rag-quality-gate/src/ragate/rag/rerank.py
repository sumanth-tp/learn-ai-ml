"""Rerankers. All share one interface so the pipeline config can swap them."""

from __future__ import annotations

from itertools import pairwise
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ragate.log import get_logger
from ragate.models import RetrievedChunk
from ragate.retry import RetryExhaustedError, call_with_retries
from ragate.text import content_tokens, coverage

log = get_logger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        ...


def _reranked(chunks: list[RetrievedChunk], scores: list[float], top_n: int, src: str):
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i].rank))[:top_n]
    return [
        chunks[i].model_copy(update={"score": scores[i], "rank": r, "source": src})
        for r, i in enumerate(order, start=1)
    ]


class NoReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        return chunks[:top_n]


class LexicalReranker:
    """Cheap cross-attention stand-in: query-term coverage plus a bigram phrase bonus.

    Scores the (query, chunk) pair jointly, which first-stage retrieval does not.
    """

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        q = content_tokens(query)
        q_bigrams = set(pairwise(q))
        scores = []
        for rc in chunks:
            toks = content_tokens(rc.chunk.title + " " + rc.chunk.text)
            bigrams = set(pairwise(toks))
            phrase = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            prior = 1.0 / (1 + rc.rank)  # keep a little of the first-stage ordering
            scores.append(coverage(query, rc.chunk.title + " " + rc.chunk.text)
                          + 0.5 * phrase + 0.1 * prior)
        return _reranked(chunks, scores, top_n, "lexical-rerank")


class _Scores(BaseModel):
    scores: list[int] = Field(description="0-10 relevance score for each passage, in order")


class LLMReranker:
    """Listwise LLM reranker. On failure it degrades to first-stage order, never errors."""

    def __init__(self, model: BaseChatModel, attempts: int = 2) -> None:
        self.model = model
        self.attempts = attempts

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        passages = "\n".join(f"{i}. {c.chunk.text}" for i, c in enumerate(chunks))
        prompt = (
            "Score how useful each passage is for answering the question, 0-10.\n"
            f"Question: {query}\nPassages:\n{passages}"
        )
        try:
            structured = self.model.with_structured_output(_Scores)
            result = call_with_retries(
                lambda: structured.invoke(prompt), what="llm_rerank", attempts=self.attempts
            )
            scores = [float(s) for s in result.scores]  # type: ignore[union-attr]
            if len(scores) != len(chunks):
                raise ValueError("reranker returned the wrong number of scores")
        except (RetryExhaustedError, ValueError, NotImplementedError) as exc:
            log.warning("rerank_fallback", error=repr(exc))
            return chunks[:top_n]
        return _reranked(chunks, scores, top_n, "llm-rerank")
