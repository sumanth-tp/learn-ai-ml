import math
import re
from collections import Counter


QUERY = "shipping refund damaged item"
DOCUMENTS = [
    "Damaged parcel refund includes shipping fees",
    "Refund reaches the original card within fourteen days",
    "Delivery fees are returned when goods arrive broken",
    "Standard shipping takes three business days",
    "Damaged goods can be replaced",
    "An account password can be reset",
]
RELEVANT = {0, 2}
DENSE_FEATURES = [
    (1.0, 0.8, 0.8),
    (0.0, 1.0, 0.0),
    (0.9, 0.9, 0.8),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0),
]
INTENTS = [
    {"shipping", "delivery", "fees"},
    {"refund", "refunded", "returned"},
    {"damaged", "broken"},
]


def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def bm25(query, documents, k1=1.5, b=0.75):
    rows = [tokens(document) for document in documents]
    n = len(rows)
    average_length = sum(map(len, rows)) / n
    frequency = Counter(term for row in rows for term in set(row))
    scores = []
    for row in rows:
        counts = Counter(row)
        score = 0.0
        for term in set(tokens(query)):
            count = counts[term]
            if count == 0:
                continue
            idf = math.log(1 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * count * (k1 + 1) / (count + k1 * (1 - b + b * len(row) / average_length))
        scores.append(score)
    return scores


def cosine(left, right):
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / (left_length * right_length) if left_length and right_length else 0.0


def rank(scores):
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def compare():
    lexical = bm25(QUERY, DOCUMENTS)
    dense = [cosine((1.0, 1.0, 1.0), vector) for vector in DENSE_FEATURES]
    lexical_rank = rank(lexical)
    dense_rank = rank(dense)
    lexical_places = {doc_id: place for place, doc_id in enumerate(lexical_rank, 1)}
    dense_places = {doc_id: place for place, doc_id in enumerate(dense_rank, 1)}
    fused = [1 / (3 + lexical_places[doc_id]) + 1 / (3 + dense_places[doc_id]) for doc_id in range(len(DOCUMENTS))]
    hybrid_rank = rank(fused)
    candidates = hybrid_rank[:4]
    coverage = {doc_id: sum(bool(set(tokens(DOCUMENTS[doc_id])) & group) for group in INTENTS) for doc_id in candidates}
    reranked = sorted(candidates, key=lambda doc_id: (-coverage[doc_id], -fused[doc_id], doc_id)) + hybrid_rank[4:]
    rankings = {"BM25": lexical_rank, "dense toy vectors": dense_rank, "hybrid RRF": hybrid_rank, "reranked": reranked}
    return rankings


def precision_at_two(ranking):
    return sum(doc_id in RELEVANT for doc_id in ranking[:2]) / 2


if __name__ == "__main__":
    for name, ranking in compare().items():
        print(f"{name:18} top 3={ranking[:3]} P@2={precision_at_two(ranking):.2f}")
