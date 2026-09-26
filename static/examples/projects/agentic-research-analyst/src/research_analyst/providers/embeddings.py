"""Embedding providers: OpenAI in live mode, a deterministic hashing model offline.

``HashingEmbeddings`` is a real bag-of-words model (feature hashing + L2 norm), so
cosine similarity still means "shares vocabulary". That keeps offline retrieval
meaningful, unlike random fake vectors.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from research_analyst.config import Settings
from research_analyst.text import content_tokens


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in content_tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.mode == "offline":
        return HashingEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        max_retries=settings.llm_max_retries,
        request_timeout=settings.http_timeout_s,
    )
