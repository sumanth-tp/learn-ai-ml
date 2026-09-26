"""Persistent hybrid index: FAISS (dense, inner product on normalised vectors) plus BM25."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from ragate.config import PipelineConfig
from ragate.log import get_logger
from ragate.models import Chunk, Document
from ragate.rag.chunking import chunk_corpus
from ragate.rag.corpus import corpus_hash
from ragate.text import content_tokens

log = get_logger(__name__)


def _embed_text(chunk: Chunk) -> str:
    # A contextual header: the title disambiguates chunks like "must be used by 31 March".
    return f"{chunk.title}. {chunk.text}"


class HandbookIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, key: str) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors are out of step")
        self.chunks = chunks
        self.key = key
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss.add(vectors)
        self._vectors = vectors
        self.bm25 = BM25Okapi([content_tokens(_embed_text(c)) or ["_"] for c in chunks])

    @staticmethod
    def index_key(docs: list[Document], config: PipelineConfig, embedding_id: str) -> str:
        parts = {
            "corpus": corpus_hash(docs),
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "include_restricted": config.include_restricted,
            "pii_redaction": config.pii_redaction,
            "embeddings": embedding_id,
        }
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def build(
        cls, docs: list[Document], config: PipelineConfig, emb: Embeddings, embedding_id: str
    ) -> HandbookIndex:
        chunks = chunk_corpus(
            docs,
            config.chunk_size,
            config.chunk_overlap,
            include_restricted=config.include_restricted,
            pii_redaction=config.pii_redaction,
        )
        vectors = np.asarray(emb.embed_documents([_embed_text(c) for c in chunks]), "float32")
        faiss.normalize_L2(vectors)
        return cls(chunks, vectors, cls.index_key(docs, config, embedding_id))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks]), encoding="utf-8"
        )
        (directory / "manifest.json").write_text(json.dumps({"key": self.key}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> HandbookIndex:
        key = json.loads((directory / "manifest.json").read_text())["key"]
        raw = json.loads((directory / "chunks.json").read_text())
        chunks = [Chunk.model_validate(c) for c in raw]
        return cls(chunks, np.load(directory / "vectors.npy"), key)

    @classmethod
    def build_or_load(
        cls,
        docs: list[Document],
        config: PipelineConfig,
        emb: Embeddings,
        embedding_id: str,
        root: Path,
    ) -> HandbookIndex:
        """Idempotent: one directory per index key, rebuilt only when an input changes."""
        key = cls.index_key(docs, config, embedding_id)
        directory = root / key
        if (directory / "manifest.json").exists():
            log.info("index_loaded", key=key)
            return cls.load(directory)
        index = cls.build(docs, config, emb, embedding_id)
        index.save(directory)
        log.info("index_built", key=key, chunks=len(index.chunks))
        return index
