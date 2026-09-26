"""Reference-based retrieval metrics over evidence quotes.

A retrieved chunk is *relevant* to an evidence item when it comes from the same
document and contains most of the quote's content words. Labels are therefore
independent of chunk ids, so chunk-size experiments stay comparable.
"""

from __future__ import annotations

import math

from ragate.models import Evidence, RetrievedChunk
from ragate.text import coverage

RELEVANCE_THRESHOLD = 0.6


def relevant_to(chunk: RetrievedChunk, evidence: Evidence) -> bool:
    return (
        chunk.chunk.doc_id == evidence.doc_id
        and coverage(evidence.quote, chunk.chunk.text) >= RELEVANCE_THRESHOLD
    )


def relevance_vector(chunks: list[RetrievedChunk], evidence: list[Evidence]) -> list[bool]:
    return [any(relevant_to(c, e) for e in evidence) for c in chunks]


def recall_at_k(chunks: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    """Share of evidence items found in at least one of the top-k chunks."""
    if not evidence:
        raise ValueError("recall is undefined without evidence")
    top = chunks[:k]
    return sum(any(relevant_to(c, e) for c in top) for e in evidence) / len(evidence)


def precision_at_k(chunks: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    top = chunks[:k]
    if not top:
        return 0.0
    return sum(relevance_vector(top, evidence)) / len(top)


def hit_at_k(chunks: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    return 1.0 if any(relevance_vector(chunks[:k], evidence)) else 0.0


def reciprocal_rank(chunks: list[RetrievedChunk], evidence: list[Evidence]) -> float:
    for rank, rel in enumerate(relevance_vector(chunks, evidence), start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    """Binary-relevance nDCG. The ideal ranking puts min(|evidence|, k) relevant chunks first."""
    rels = relevance_vector(chunks[:k], evidence)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(evidence), k)))
    return dcg / ideal if ideal else 0.0


def contextual_precision(chunks: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    """Rank-aware precision, as defined by DeepEval/Ragas but with reference labels.

    mean over relevant positions r of precision@r. Rewards putting relevant chunks
    *first*, which matters because models attend most to early context.
    """
    rels = relevance_vector(chunks[:k], evidence)
    hits, total = 0, 0.0
    for i, rel in enumerate(rels, start=1):
        if rel:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
