"""Generator metrics: citation accuracy (deterministic) and judge-based quality."""

from __future__ import annotations

from ragate.judge.base import Judge
from ragate.judge.prompts import REGISTRY
from ragate.models import Evidence, RagAnswer


def context_text(answer: RagAnswer) -> str:
    return "\n".join(f"[{c.chunk.doc_id}] {c.chunk.text}" for c in answer.contexts)


def citation_scores(answer: RagAnswer, evidence: list[Evidence]) -> dict[str, float]:
    """validity: cited ids are among retrieved docs (no invented sources);
    precision: cited ids are gold evidence docs; recall: gold docs that were cited."""
    cited = set(answer.citations)
    retrieved = {c.chunk.doc_id for c in answer.contexts}
    gold = {e.doc_id for e in evidence}
    if not cited:
        return {"citation_validity": 0.0, "citation_precision": 0.0, "citation_recall": 0.0}
    return {
        "citation_validity": len(cited & retrieved) / len(cited),
        "citation_precision": len(cited & gold) / len(cited),
        "citation_recall": len(cited & gold) / len(gold) if gold else 0.0,
    }


def faithfulness(judge: Judge, answer: RagAnswer) -> float:
    return judge.evaluate(
        REGISTRY["faithfulness"], {"context": context_text(answer), "answer": answer.answer}
    ).score


def answer_relevancy(judge: Judge, answer: RagAnswer) -> float:
    return judge.evaluate(
        REGISTRY["answer_relevancy"], {"question": answer.question, "answer": answer.answer}
    ).score


def context_relevance(judge: Judge, answer: RagAnswer) -> float:
    return judge.evaluate(
        REGISTRY["context_relevance"],
        {"question": answer.question, "context": context_text(answer)},
    ).score


def contextual_recall(judge: Judge, answer: RagAnswer, reference: str) -> float:
    return judge.evaluate(
        REGISTRY["contextual_recall"],
        {"reference": reference, "context": context_text(answer)},
    ).score
