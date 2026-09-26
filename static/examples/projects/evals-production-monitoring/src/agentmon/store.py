"""SQLite persistence for traces, spans, evals, feedback, the eval job queue,
the review queue, deployments and alerts. Works offline; WAL mode lets the API
and a separate worker process share one file."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from agentmon.models import EvalResult, Feedback, ToolCallRecord, TraceRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces(
  trace_id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, ts REAL NOT NULL,
  user_id TEXT, session_id TEXT, input TEXT, output TEXT, intent TEXT,
  prompt_version TEXT, model TEXT, status TEXT, error TEXT, latency_ms REAL,
  input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, steps INTEGER,
  tool_calls TEXT, flags TEXT);
CREATE INDEX IF NOT EXISTS ix_traces_ts ON traces(ts);
CREATE TABLE IF NOT EXISTS spans(
  span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_id TEXT, name TEXT,
  start_ns INTEGER, end_ns INTEGER, status TEXT, attributes TEXT);
CREATE INDEX IF NOT EXISTS ix_spans_trace ON spans(trace_id);
CREATE TABLE IF NOT EXISTS evals(
  trace_id TEXT NOT NULL, evaluator TEXT NOT NULL, score REAL, passed INTEGER,
  reason TEXT, tier TEXT, cost_usd REAL, ts REAL, PRIMARY KEY(trace_id, evaluator));
CREATE INDEX IF NOT EXISTS ix_evals_ts ON evals(ts);
CREATE TABLE IF NOT EXISTS eval_jobs(
  trace_id TEXT PRIMARY KEY, reason TEXT, status TEXT NOT NULL, attempts INTEGER DEFAULT 0,
  enqueued_ts REAL, updated_ts REAL, last_error TEXT);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON eval_jobs(status);
CREATE TABLE IF NOT EXISTS feedback(
  trace_id TEXT PRIMARY KEY, ts REAL, rating INTEGER, correction TEXT);
CREATE TABLE IF NOT EXISTS review_queue(
  trace_id TEXT PRIMARY KEY, reason TEXT, status TEXT NOT NULL, created_ts REAL,
  reviewed_ts REAL, reviewer TEXT, golden_id TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS deployments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, component TEXT, old_value TEXT,
  new_value TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, rule TEXT, severity TEXT, fired_ts REAL,
  resolved_ts REAL, value REAL, threshold REAL, message TEXT, evidence TEXT);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    # --- traces -------------------------------------------------------------------
    def insert_trace(self, t: TraceRecord) -> bool:
        """Insert a trace. Returns False when the request_id was already stored (replay)."""
        cur = self._exec(
            "INSERT OR IGNORE INTO traces VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.trace_id,
                t.request_id,
                t.ts,
                t.user_id,
                t.session_id,
                t.input,
                t.output,
                t.intent,
                t.prompt_version,
                t.model,
                t.status,
                t.error,
                t.latency_ms,
                t.input_tokens,
                t.output_tokens,
                t.cost_usd,
                t.steps,
                json.dumps([c.model_dump() for c in t.tool_calls]),
                json.dumps(t.flags),
            ),
        )
        return cur.rowcount == 1

    @staticmethod
    def _trace(row: sqlite3.Row) -> TraceRecord:
        d = dict(row)
        d["tool_calls"] = [ToolCallRecord(**c) for c in json.loads(d["tool_calls"] or "[]")]
        d["flags"] = json.loads(d["flags"] or "[]")
        return TraceRecord(**d)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        rows = self._rows("SELECT * FROM traces WHERE trace_id=?", (trace_id,))
        return self._trace(rows[0]) if rows else None

    def get_trace_by_request(self, request_id: str) -> TraceRecord | None:
        rows = self._rows("SELECT * FROM traces WHERE request_id=?", (request_id,))
        return self._trace(rows[0]) if rows else None

    def traces_between(self, start: float, end: float) -> list[TraceRecord]:
        rows = self._rows("SELECT * FROM traces WHERE ts>=? AND ts<? ORDER BY ts", (start, end))
        return [self._trace(r) for r in rows]

    def trace_time_range(self) -> tuple[float, float] | None:
        row = self._rows("SELECT MIN(ts) AS a, MAX(ts) AS b FROM traces")[0]
        return None if row["a"] is None else (row["a"], row["b"])

    # --- spans --------------------------------------------------------------------
    def insert_spans(self, spans: Iterable[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES(:span_id,:trace_id,:parent_id,:name,"
                ":start_ns,:end_ns,:status,:attributes)",
                list(spans),
            )

    def spans_for(self, trace_id: str) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM spans WHERE trace_id=? ORDER BY start_ns, parent_id IS NOT NULL, end_ns", (trace_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["attributes"] = json.loads(d["attributes"] or "{}")
            out.append(d)
        return out

    # --- evals --------------------------------------------------------------------
    def upsert_evals(self, results: Iterable[EvalResult]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO evals VALUES(?,?,?,?,?,?,?,?)",
                [
                    (
                        r.trace_id,
                        r.evaluator,
                        r.score,
                        int(r.passed),
                        r.reason,
                        r.tier,
                        r.cost_usd,
                        r.ts,
                    )
                    for r in results
                ],
            )

    def evals_for(self, trace_id: str) -> list[EvalResult]:
        rows = self._rows("SELECT * FROM evals WHERE trace_id=?", (trace_id,))
        return [EvalResult(**{**dict(r), "passed": bool(r["passed"])}) for r in rows]

    def evals_between(self, start: float, end: float) -> list[EvalResult]:
        rows = self._rows("SELECT * FROM evals WHERE ts>=? AND ts<?", (start, end))
        return [EvalResult(**{**dict(r), "passed": bool(r["passed"])}) for r in rows]

    def judge_spend_between(self, start: float, end: float) -> float:
        row = self._rows(
            "SELECT COALESCE(SUM(cost_usd),0) AS c FROM evals WHERE tier='judge' AND ts>=? AND ts<?",
            (start, end),
        )[0]
        return float(row["c"])

    # --- eval job queue -----------------------------------------------------------
    def enqueue_eval(self, trace_id: str, reason: str, ts: float) -> bool:
        cur = self._exec(
            "INSERT OR IGNORE INTO eval_jobs(trace_id,reason,status,attempts,enqueued_ts,updated_ts)"
            " VALUES(?,?, 'pending', 0, ?, ?)",
            (trace_id, reason, ts, ts),
        )
        return cur.rowcount == 1

    def claim_jobs(self, limit: int, ts: float) -> list[dict[str, Any]]:
        """Atomically move up to `limit` pending jobs to running, so two workers never
        evaluate the same trace."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = list(
                    self._conn.execute(
                        "SELECT * FROM eval_jobs WHERE status='pending' ORDER BY enqueued_ts LIMIT ?",
                        (limit,),
                    ).fetchall()
                )
                self._conn.executemany(
                    "UPDATE eval_jobs SET status='running', attempts=attempts+1, updated_ts=?"
                    " WHERE trace_id=?",
                    [(ts, r["trace_id"]) for r in rows],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return [dict(r) for r in rows]

    def finish_job(self, trace_id: str, status: str, ts: float, error: str | None = None) -> None:
        self._exec(
            "UPDATE eval_jobs SET status=?, updated_ts=?, last_error=? WHERE trace_id=?",
            (status, ts, error, trace_id),
        )

    def requeue_stale_jobs(self, older_than_ts: float) -> int:
        cur = self._exec(
            "UPDATE eval_jobs SET status='pending' WHERE status='running' AND updated_ts<?",
            (older_than_ts,),
        )
        return cur.rowcount

    def job_counts(self) -> dict[str, int]:
        rows = self._rows("SELECT status, COUNT(*) AS n FROM eval_jobs GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    def get_job(self, trace_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM eval_jobs WHERE trace_id=?", (trace_id,))
        return dict(rows[0]) if rows else None

    # --- feedback -----------------------------------------------------------------
    def upsert_feedback(self, f: Feedback) -> None:
        self._exec(
            "INSERT OR REPLACE INTO feedback VALUES(?,?,?,?)",
            (f.trace_id, f.ts, f.rating, f.correction),
        )

    def feedback_for(self, trace_id: str) -> Feedback | None:
        rows = self._rows("SELECT * FROM feedback WHERE trace_id=?", (trace_id,))
        return Feedback(**dict(rows[0])) if rows else None

    def all_feedback(self) -> dict[str, Feedback]:
        return {r["trace_id"]: Feedback(**dict(r)) for r in self._rows("SELECT * FROM feedback")}

    # --- review queue -------------------------------------------------------------
    def add_review(self, trace_id: str, reason: str, ts: float) -> bool:
        cur = self._exec(
            "INSERT OR IGNORE INTO review_queue(trace_id,reason,status,created_ts)"
            " VALUES(?,?,'pending',?)",
            (trace_id, reason, ts),
        )
        return cur.rowcount == 1

    def review_items(self, status: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if status:
            rows = self._rows(
                "SELECT * FROM review_queue WHERE status=? ORDER BY created_ts LIMIT ?",
                (status, limit),
            )
        else:
            rows = self._rows("SELECT * FROM review_queue ORDER BY created_ts LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def set_review(
        self,
        trace_id: str,
        status: str,
        reviewer: str,
        ts: float,
        golden_id: str | None = None,
        note: str | None = None,
    ) -> None:
        self._exec(
            "UPDATE review_queue SET status=?, reviewer=?, reviewed_ts=?, golden_id=?, note=?"
            " WHERE trace_id=?",
            (status, reviewer, ts, golden_id, note, trace_id),
        )

    # --- deployments and alerts ---------------------------------------------------
    def record_deployment(
        self, ts: float, component: str, old: str, new: str, note: str = ""
    ) -> None:
        self._exec(
            "INSERT INTO deployments(ts,component,old_value,new_value,note) VALUES(?,?,?,?,?)",
            (ts, component, old, new, note),
        )

    def deployments(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows("SELECT * FROM deployments ORDER BY ts")]

    def insert_alert(self, alert: dict[str, Any]) -> int:
        cur = self._exec(
            "INSERT INTO alerts(rule,severity,fired_ts,resolved_ts,value,threshold,message,evidence)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                alert["rule"],
                alert["severity"],
                alert["fired_ts"],
                alert.get("resolved_ts"),
                alert["value"],
                alert["threshold"],
                alert["message"],
                json.dumps(alert.get("evidence", {})),
            ),
        )
        return int(cur.lastrowid or 0)

    def alerts(self) -> list[dict[str, Any]]:
        out = []
        for r in self._rows("SELECT * FROM alerts ORDER BY fired_ts"):
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"] or "{}")
            out.append(d)
        return out

    def clear_alerts(self) -> None:
        self._exec("DELETE FROM alerts")
