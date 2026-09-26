"""Schema retrieval and the semantic question-to-SQL cache.

Both are embedding lookups. At this size a brute-force cosine over a few thousand
vectors is sub-millisecond; swap in an ANN index (pgvector, FAISS) past ~100k entries.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from data_analyst.warehouse.catalog import METRICS, TABLES, close_join_paths


def _unit(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class RetrievedTable(BaseModel):
    name: str
    score: float


class SchemaIndex:
    """Embeds one document per table; returns the top-k plus any join bridges."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.names = [t.name for t in TABLES]
        self.matrix = _unit(embeddings.embed_documents([t.document() for t in TABLES]))

    def search(self, question: str, k: int) -> list[RetrievedTable]:
        q = _unit([self.embeddings.embed_query(question)])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [RetrievedTable(name=self.names[i], score=round(float(scores[i]), 4)) for i in order]

    def retrieve(
        self, question: str, k: int, min_ratio: float = 0.5
    ) -> tuple[list[str], list[RetrievedTable]]:
        """Top-k tables, dropping weak hits (below ``min_ratio`` of the best score), then
        adding the bridge tables their joins need."""
        hits = self.search(question, k)
        if hits:
            floor = hits[0].score * min_ratio
            hits = [h for h in hits if h.score >= floor]
        names = [h.name for h in hits]
        lowered = question.lower()
        for term, tables in METRICS.items():
            if term in lowered:
                names.extend(t for t in tables if t not in names)
        return close_join_paths(names), hits


class CacheHit(BaseModel):
    question: str
    sql: str
    similarity: float


class SemanticCache:
    """Question -> validated SQL. Stores SQL only, never results, so every hit re-runs
    through validation, masking and the approval gate with current data."""

    def __init__(
        self, path: Path, embeddings: Embeddings, *, threshold: float, ttl_s: int, schema: str
    ) -> None:
        self.path = path
        self.embeddings = embeddings
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.schema = schema
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache (id INTEGER PRIMARY KEY, question TEXT,"
                " embedding TEXT, sql TEXT, schema TEXT, created REAL, hits INTEGER DEFAULT 0)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def lookup(self, question: str) -> CacheHit | None:
        cutoff = time.time() - self.ttl_s
        with closing(self._conn()) as con:
            rows = con.execute(
                "SELECT id, question, embedding, sql FROM cache WHERE schema = ? AND created > ?",
                (self.schema, cutoff),
            ).fetchall()
        if not rows:
            return None
        q = _unit([self.embeddings.embed_query(question)])[0]
        matrix = _unit([json.loads(r[2]) for r in rows])
        scores = matrix @ q
        best = int(np.argmax(scores))
        if float(scores[best]) < self.threshold:
            return None
        row = rows[best]
        with closing(self._conn()) as con, con:
            con.execute("UPDATE cache SET hits = hits + 1 WHERE id = ?", (row[0],))
        return CacheHit(question=row[1], sql=row[3], similarity=round(float(scores[best]), 4))

    def store(self, question: str, sql: str) -> None:
        vec = self.embeddings.embed_query(question)
        with closing(self._conn()) as con, con:
            con.execute(
                "DELETE FROM cache WHERE question = ? AND schema = ?", (question, self.schema)
            )
            con.execute(
                "INSERT INTO cache (question, embedding, sql, schema, created) VALUES (?,?,?,?,?)",
                (question, json.dumps(vec), sql, self.schema, time.time()),
            )

    def invalidate(self, sql: str) -> None:
        with closing(self._conn()) as con, con:
            con.execute("DELETE FROM cache WHERE sql = ?", (sql,))

    def size(self) -> int:
        with closing(self._conn()) as con:
            return int(con.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
