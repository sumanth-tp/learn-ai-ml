"""Dataset quality, contamination and leakage checks.

Errors block an eval run (the numbers would be wrong); warnings are printed for a human.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel

from ragate.models import Document, ExpectedBehaviour, GoldenItem, QuestionType
from ragate.rag.generator import FEW_SHOT_QUESTIONS
from ragate.text import content_tokens, coverage, jaccard, normalise, sentences, shingles


class Finding(BaseModel):
    level: Literal["error", "warning"]
    check: str
    item_id: str
    message: str


def run_checks(
    items: list[GoldenItem], docs: list[Document], *, min_per_type: int = 5
) -> list[Finding]:
    by_id = {d.doc_id: d for d in docs}
    corpus_sents = [s for d in docs for s in sentences(d.text)]
    corpus_shingles = set().union(*(shingles(s, 5) for s in corpus_sents))
    out: list[Finding] = []

    def add(level: Literal["error", "warning"], check: str, item: str, msg: str) -> None:
        out.append(Finding(level=level, check=check, item_id=item, message=msg))

    for it in items:
        # 1. schema consistency between behaviour and labels
        if it.expected_behaviour == ExpectedBehaviour.REFUSE and it.evidence:
            add("error", "schema", it.item_id, "refuse items must not carry evidence")
        if it.expected_behaviour != ExpectedBehaviour.REFUSE and not (
            it.evidence and it.reference_answer
        ):
            add("error", "schema", it.item_id, "answerable items need evidence and a reference")

        # 2. evidence integrity: the quote must exist verbatim in the cited document
        for ev in it.evidence:
            doc = by_id.get(ev.doc_id)
            if doc is None:
                add("error", "evidence", it.item_id, f"unknown doc_id {ev.doc_id}")
            elif normalise(ev.quote) not in normalise(doc.text):
                add("error", "evidence", it.item_id, f"quote not found in {ev.doc_id} (stale?)")
            elif doc.access == "restricted":
                add("error", "evidence", it.item_id, f"{ev.doc_id} is restricted, not retrievable")

        # 3. prompt contamination: golden items must not be the prompt's few-shot examples
        for shot in FEW_SHOT_QUESTIONS:
            if jaccard(shingles(it.question), shingles(shot)) >= 0.5:
                add("error", "prompt_contamination", it.item_id, "matches a few-shot example")

        # 4. answer leakage: the question must not already contain the answer
        if it.reference_answer and it.expected_behaviour == ExpectedBehaviour.ANSWER:
            novel = set(content_tokens(it.reference_answer)) - set(content_tokens(it.question))
            if not novel:
                add("error", "answer_leakage", it.item_id, "question contains the answer")

        # 5. verbatim leakage: copied corpus text makes lexical retrieval trivially perfect
        q_sh = shingles(it.question, 5)
        if q_sh and len(q_sh & corpus_shingles) / len(q_sh) >= 0.6:
            add("warning", "verbatim_leakage", it.item_id, "question copies corpus text")

        # 6. unanswerable sanity: does the corpus in fact answer it?
        if it.question_type == QuestionType.UNANSWERABLE:
            best = max((coverage(it.question, s) for s in corpus_sents), default=0.0)
            if best >= 0.75:
                add("warning", "maybe_answerable", it.item_id, f"corpus covers {best:.0%} of it")

    # 7. near-duplicates inside the set (inflate the weight of one behaviour)
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if jaccard(shingles(items[a].question), shingles(items[b].question)) >= 0.8:
                add("error", "near_duplicate", items[b].item_id, f"near-dup of {items[a].item_id}")

    # 8. stratification: every question type needs enough items to mean anything
    counts = Counter(i.question_type for i in items)
    for qt in QuestionType:
        if counts.get(qt, 0) < min_per_type:
            add("error", "stratification", "*", f"{qt.value}: {counts.get(qt, 0)} < {min_per_type}")
    return out


def has_errors(findings: list[Finding]) -> bool:
    return any(f.level == "error" for f in findings)
