"""Durable gateway state in SQLite: tool pins and daily quota counters.

SQLite in WAL mode is enough for one gateway replica (thousands of writes a
second). With several replicas, move both tables to Postgres or Redis; the
two classes below are the only code that would change.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PinStatus = Literal["approved", "pending", "changed", "quarantined"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pins (
    tool TEXT PRIMARY KEY,
    upstream TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    seen_sha256 TEXT,
    findings TEXT NOT NULL DEFAULT '',
    first_seen REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quotas (
    user TEXT NOT NULL,
    scope TEXT NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (user, scope, day)
);
"""


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self.lock = threading.Lock()

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        with self.lock:
            return self._conn.execute(sql, params).fetchall()


@dataclass(frozen=True)
class Pin:
    tool: str
    upstream: str
    sha256: str
    status: PinStatus
    seen_sha256: str | None
    findings: str


class PinStore:
    def __init__(self, db: StateDB) -> None:
        self.db = db

    def get(self, tool: str) -> Pin | None:
        rows = self.db.execute(
            "SELECT tool, upstream, sha256, status, seen_sha256, findings FROM pins WHERE tool=?",
            (tool,),
        )
        return Pin(*rows[0]) if rows else None  # type: ignore[arg-type]

    def put(self, tool: str, upstream: str, sha: str, status: PinStatus, findings: str) -> None:
        now = time.time()
        self.db.execute(
            "INSERT INTO pins VALUES (?,?,?,?,NULL,?,?,?) ON CONFLICT(tool) DO UPDATE SET "
            "sha256=excluded.sha256, status=excluded.status, seen_sha256=NULL, "
            "findings=excluded.findings, updated=excluded.updated",
            (tool, upstream, sha, status, findings, now, now),
        )

    def mark_changed(self, tool: str, seen_sha: str, findings: str) -> None:
        self.db.execute(
            "UPDATE pins SET status='changed', seen_sha256=?, findings=?, updated=? WHERE tool=?",
            (seen_sha, findings, time.time(), tool),
        )

    def approve(self, tool: str) -> bool:
        """Accept the currently served definition as the new pin."""
        pin = self.get(tool)
        if pin is None:
            return False
        new_sha = pin.seen_sha256 or pin.sha256
        self.db.execute(
            "UPDATE pins SET sha256=?, seen_sha256=NULL, status='approved', updated=? WHERE tool=?",
            (new_sha, time.time(), tool),
        )
        return True

    def all(self) -> list[Pin]:
        rows = self.db.execute(
            "SELECT tool, upstream, sha256, status, seen_sha256, findings FROM pins ORDER BY tool"
        )
        return [Pin(*r) for r in rows]  # type: ignore[arg-type]


class QuotaStore:
    """Daily counters that survive restarts (a restart must not reset a quota)."""

    def __init__(self, db: StateDB) -> None:
        self.db = db

    @staticmethod
    def today(now: float | None = None) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))

    def try_consume(
        self, user: str, scopes: list[tuple[str, int]], now: float | None = None
    ) -> str | None:
        """Atomically consume one unit from every (scope, limit) pair.

        Returns None on success, or the first exhausted scope (and consumes
        nothing), so a call denied by the per-tool quota does not also burn
        the user's overall quota.
        """
        day = self.today(now)
        with self.db.lock:
            conn = self.db._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                for scope, limit in scopes:
                    row = conn.execute(
                        "SELECT count FROM quotas WHERE user=? AND scope=? AND day=?",
                        (user, scope, day),
                    ).fetchone()
                    if (row[0] if row else 0) >= limit:
                        conn.execute("ROLLBACK")
                        return scope
                for scope, _ in scopes:
                    conn.execute(
                        "INSERT INTO quotas VALUES (?,?,?,1) ON CONFLICT(user,scope,day) "
                        "DO UPDATE SET count=count+1",
                        (user, scope, day),
                    )
                conn.execute("COMMIT")
                return None
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def used(self, user: str, scope: str, now: float | None = None) -> int:
        rows = self.db.execute(
            "SELECT count FROM quotas WHERE user=? AND scope=? AND day=?",
            (user, scope, self.today(now)),
        )
        return int(rows[0][0]) if rows else 0  # type: ignore[call-overload]
