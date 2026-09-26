"""SQLite run store: every run, its predictions, per-item scores and the final summary.

Keeping per-item scores (not just the means) is what lets a later run be compared
item-by-item with an old one, and lets you re-run the statistics without re-calling
any model.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelsel.schemas import ItemScore, Prediction

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    profile TEXT NOT NULL,
    splits TEXT NOT NULL,
    dataset_hash TEXT,
    idempotency_key TEXT UNIQUE,
    summary TEXT,
    report_md TEXT,
    report_html TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL, model_id TEXT NOT NULL, item_id TEXT NOT NULL, task TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, item_id, task)
);
CREATE TABLE IF NOT EXISTS item_scores (
    run_id TEXT NOT NULL, model_id TEXT NOT NULL, item_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, model_id, item_id)
);
"""

UPDATABLE = frozenset({"dataset_hash", "summary", "report_md", "report_html", "error"})


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def create_run(self, run_id: str, profile: str, splits: list[str], idempotency_key: str | None = None) -> str:
        """Insert a run, or return the existing run for the same idempotency key."""
        with self._lock:
            if idempotency_key:
                row = self._conn.execute(
                    "SELECT run_id FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if row:
                    return str(row["run_id"])
            self._conn.execute(
                "INSERT INTO runs (run_id, created_at, status, profile, splits, idempotency_key) VALUES (?,?,?,?,?,?)",
                (run_id, datetime.now(UTC).isoformat(), "queued", profile, json.dumps(splits), idempotency_key),
            )
            self._conn.commit()
        return run_id

    def set_status(self, run_id: str, status: str, **fields: Any) -> None:
        unknown = set(fields) - UPDATABLE
        if unknown:
            raise ValueError(f"cannot update columns {sorted(unknown)}")
        cols = ["status = ?"] + [f"{k} = ?" for k in fields]
        values = [status, *(json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()), run_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {', '.join(cols)} WHERE run_id = ?", values
            )  # columns checked against UPDATABLE
            self._conn.commit()

    def save_predictions(self, preds: list[Prediction]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)",
                [(p.run_id, p.model_id, p.item_id, p.task, p.model_dump_json()) for p in preds],
            )
            self._conn.commit()

    def save_scores(self, run_id: str, scores: list[ItemScore]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO item_scores VALUES (?,?,?,?)",
                [(run_id, s.model_id, s.item_id, s.model_dump_json()) for s in scores],
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["splits"] = json.loads(out["splits"])
        out["summary"] = json.loads(out["summary"]) if out["summary"] else None
        return out

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_at, status, profile, splits, dataset_hash"
                " FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) | {"splits": json.loads(r["splits"])} for r in rows]

    def load_scores(self, run_id: str) -> list[ItemScore]:
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM item_scores WHERE run_id = ?", (run_id,)).fetchall()
        return [ItemScore.model_validate_json(r["payload"]) for r in rows]

    def count_predictions(self, run_id: str) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (run_id,)).fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
