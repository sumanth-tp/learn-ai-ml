"""SQLite persistence for eval runs (the API and dashboard read from here)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ragate.evaluation.results import RunResult


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
              config_hash TEXT NOT NULL, dataset_version TEXT NOT NULL,
              aggregates TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS gate_decisions (
              id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              baseline TEXT NOT NULL, candidate TEXT NOT NULL, decision TEXT NOT NULL,
              report TEXT NOT NULL);
            """
        )

    def save(self, run: RunResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run.run_id, run.name, run.created_at, run.config_hash, run.dataset_version,
                 json.dumps(run.aggregates), run.model_dump_json()),
            )
            self._conn.commit()

    def get(self, run_id: str) -> RunResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return RunResult.model_validate_json(row[0]) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, name, created_at, config_hash, dataset_version, aggregates"
                " FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"run_id": r[0], "name": r[1], "created_at": r[2], "config_hash": r[3],
             "dataset_version": r[4], "aggregates": json.loads(r[5])}
            for r in rows
        ]

    def record_decision(self, baseline: str, candidate: str, decision: str, report: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO gate_decisions (baseline, candidate, decision, report)"
                " VALUES (?, ?, ?, ?)", (baseline, candidate, decision, report)
            )
            self._conn.commit()

    def decisions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT created_at, baseline, candidate, decision FROM gate_decisions"
                " ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"created_at": r[0], "baseline": r[1], "candidate": r[2], "decision": r[3]}
                for r in rows]


def load_run_file(path: Path) -> RunResult:
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def write_run_file(run: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=1) + "\n", encoding="utf-8")
