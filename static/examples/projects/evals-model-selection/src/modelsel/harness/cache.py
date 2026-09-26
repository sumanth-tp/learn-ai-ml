"""A SQLite response cache keyed by everything that can change an answer.

Re-running the report after a new model ships should only pay for the new model.
The key covers model id, prompt version, messages and call parameters, so a changed
prompt or temperature is a cache miss, never a stale hit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from modelsel.schemas import Completion


def cache_key(model_id: str, prompt_version: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"m": model_id, "v": prompt_version, "msgs": messages, "p": params}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, model_id TEXT NOT NULL, payload TEXT NOT NULL,"
                " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self._conn.commit()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        completion = Completion.model_validate_json(row[0])
        return completion.model_copy(update={"cached": True})

    def put(self, key: str, completion: Completion) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model_id, payload) VALUES (?, ?, ?)",
                (key, completion.model_id, completion.model_dump_json()),
            )
            self._conn.commit()

    def count(self, model_id: str | None = None) -> int:
        with self._lock:
            if model_id is None:
                return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
            return int(
                self._conn.execute("SELECT COUNT(*) FROM responses WHERE model_id = ?", (model_id,)).fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
