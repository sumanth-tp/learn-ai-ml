"""Dense, sparse and hybrid (reciprocal rank fusion) retrieval."""

from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings

from ragate.models import RetrievedChunk
from ragate.rag.index import HandbookIndex
from ragate.text import content_tokens


class Retriever:
    def __init__(self, index: HandbookIndex, emb: Embeddings, *, hybrid: bool, rrf_k: int) -> None:
        self.index = index
        self.emb = emb
        self.hybrid = hybrid
        self.rrf_k = rrf_k

    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        vec = np.asarray([self.emb.embed_query(query)], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores, ids = self.index.faiss.search(vec, min(k, len(self.index.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.index.bm25.get_scores(content_tokens(query) or ["_"])
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def retrieve(self, query: str, fetch_k: int) -> list[RetrievedChunk]:
        dense = self.dense(query, fetch_k)
        if not self.hybrid:
            ranked = dense
            source = "dense"
        else:
            # RRF: score = sum 1/(k + rank). Scale-free, so BM25 and cosine need no calibration.
            fused: dict[int, float] = {}
            for results in (dense, self.sparse(query, fetch_k)):
                for rank, (idx, _) in enumerate(results, start=1):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
            ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:fetch_k]
            source = "hybrid"
        return [
            RetrievedChunk(chunk=self.index.chunks[i], score=s, rank=r, source=source)
            for r, (i, s) in enumerate(ranked, start=1)
        ]
