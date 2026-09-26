"""SQLite cache for judge verdicts.

Keyed on everything that can change a verdict: judge model, temperature, prompt name,
version and fingerprint, and the rendered inputs. Re-running an unchanged item is free
and bit-for-bit identical, which removes judge noise from items that did not change.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from ragate.judge.base import Judge, Verdict
from ragate.judge.prompts import JudgePrompt


class JudgeCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
            " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model_id: str, prompt: JudgePrompt, variables: dict[str, str]) -> str:
        payload = {
            "model": model_id,
            "prompt": [prompt.name, prompt.version, prompt.fingerprint],
            "vars": variables,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> Verdict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM verdicts WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return Verdict.model_validate_json(row[0]).model_copy(update={"cached": True})

    def put(self, key: str, verdict: Verdict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts (key, value) VALUES (?, ?)",
                (key, verdict.model_dump_json()),
            )
            self._conn.commit()

    def get_raw(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM verdicts WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put_raw(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts (key, value) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()


class CachedJudge:
    """Decorator that adds caching to any Judge."""

    def __init__(self, inner: Judge, cache: JudgeCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    def evaluate(self, prompt: JudgePrompt, variables: dict[str, str]) -> Verdict:
        key = JudgeCache.key(self.model_id, prompt, variables)
        if (hit := self.cache.get(key)) is not None:
            return hit
        verdict = self.inner.evaluate(prompt, variables)
        self.cache.put(key, verdict)
        return verdict
