"""Small inspectable retrieval baseline, not a semantic embedding or LLM service."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
import re

@dataclass(frozen=True)
class Document:
    id: str
    tenant: str
    text: str
    revision: int = 1

def tokens(text):
    return re.findall(r"[\w-]+", text.casefold())

def retrieve(query, documents, tenant, k=3):
    if k < 1:
        raise ValueError("k must be positive")
    latest = {}
    for doc in documents:
        if doc.tenant != tenant:  # Before any scoring or text disclosure.
            continue
        if doc.id not in latest or doc.revision > latest[doc.id].revision:
            latest[doc.id] = doc
    terms = Counter(tokens(query))
    scored = [(sum((terms & Counter(tokens(d.text))).values()), d)
              for d in latest.values()]
    return [d for score, d in sorted(scored, key=lambda row: (-row[0], row[1].id))
            if score > 0][:k]

def rrf(rankings, c=60):
    if c <= 0:
        raise ValueError("c must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (c + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))

def metrics(ranked, grades, k):
    """Judged IDs only; unjudged treatment must be specified before using real data.

    Precision uses k slots (missing results count as missed slots).
    No relevant judgments: recall/MRR/nDCG are None, not perfect retrieval.
    """
    if k < 1 or any(g < 0 for g in grades.values()):
        raise ValueError("invalid k or grade")
    ranked = list(dict.fromkeys(ranked))[:k]
    relevant = {key for key, grade in grades.items() if grade > 0}
    hits = sum(key in relevant for key in ranked)
    rr = next((1 / i for i, key in enumerate(ranked, 1) if key in relevant), 0.0)
    dcg = sum((2 ** grades.get(key, 0) - 1) / log2(i + 1)
              for i, key in enumerate(ranked, 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / log2(i + 1) for i, grade in enumerate(ideal, 1))
    return {"precision": hits / k, "recall": hits / len(relevant) if relevant else None,
            "hit": int(hits > 0), "rr": rr if relevant else None,
            "ndcg": dcg / idcg if idcg else None}

def answer(query, documents, tenant):
    found = retrieve(query, documents, tenant)
    if not found:
        return {"status": "no_evidence", "evidence": []}
    # Return evidence verbatim. Do not label lexical overlap as proof of an answer.
    return {"status": "evidence_only", "evidence": [
        {"id": d.id, "revision": d.revision, "text": d.text} for d in found]}

if __name__ == "__main__":
    corpus = [Document("refund", "acme", "Refund requests require a receipt."),
              Document("private", "other", "Refund secret: OTHER_TENANT_TOKEN")]
    print(answer("refund", corpus, "acme"))
    print(metrics(["x", "a", "b"], {"a": 1, "b": 1, "c": 1}, 3))
