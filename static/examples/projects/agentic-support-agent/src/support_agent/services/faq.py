"""FAQ retrieval over the help-centre articles, behind LangChain's Embeddings interface."""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib import resources
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "is",
    "are",
    "i",
    "my",
    "me",
    "you",
    "your",
    "we",
    "our",
    "in",
    "on",
    "for",
    "it",
    "do",
    "does",
    "can",
    "how",
    "what",
    "when",
    "with",
    "be",
    "if",
    "at",
    "by",
    "this",
    "that",
    "long",
    "much",
    "there",
    "any",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Crude stemming so "refunds", "refunded" and "refund" share a bucket.
    return [re.sub(r"(ing|ed|es|s)$", "", w) or w for w in words if w not in _STOP]


class HashingEmbeddings(Embeddings):
    """A local, deterministic bag-of-words embedding (feature hashing).

    It is lexical, not semantic, but it is a real retriever: queries that
    share words with an article score higher. It keeps tests and the offline
    demo meaningful without an embeddings API.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def load_faq_articles() -> list[dict[str, str]]:
    raw = resources.files("support_agent.data").joinpath("faq.json").read_text("utf-8")
    return json.loads(raw)


class FaqRetriever:
    def __init__(self, embeddings: Embeddings, min_score: float = 0.15) -> None:
        self._store = InMemoryVectorStore(embedding=embeddings)
        self._min_score = min_score
        articles = load_faq_articles()
        self._store.add_documents(
            [
                Document(page_content=f"{a['title']}. {a['text']}", metadata={"id": a["id"]})
                for a in articles
            ],
            ids=[a["id"] for a in articles],
        )

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        hits = self._store.similarity_search_with_score(query, k=k)
        return [
            {"id": doc.metadata["id"], "text": doc.page_content, "score": round(score, 3)}
            for doc, score in hits
            if score >= self._min_score
        ]
