"""Sandboxed, read-only query execution against DuckDB, plus EXPLAIN-based cost estimates.

Layers of defence, from outermost in:
1. read_only connection: no writes, whatever the SQL.
2. enable_external_access = false: no files, no network, no extensions.
3. lock_configuration: SQL cannot switch 2 back on with SET.
4. memory_limit and threads: one query cannot starve the host.
5. a wall-clock timeout that interrupts the query.
6. a row cap: we fetch at most row_cap + 1 rows, whatever LIMIT says.
7. a DLP pass that redacts email and phone patterns in results.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from pydantic import BaseModel

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+\d{1,3}[\s.-]\(?\d{2,4}\)?[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?")


class QueryError(Exception):
    """The database rejected the query. The message is fed back to the model."""


class QueryTimeout(QueryError):
    pass


class CostEstimate(BaseModel):
    estimated_rows: int
    max_intermediate_rows: int
    scanned_rows: int


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    elapsed_ms: float
    redactions: int = 0


def to_jsonable(value: Any) -> Any:
    """Make a DuckDB value safe for JSON, checkpoints and comparisons."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date | dtime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def redact(value: Any) -> tuple[Any, int]:
    """Mask anything that looks like an email or phone number that slipped through."""
    if not isinstance(value, str):
        return value, 0
    if value.startswith("***"):
        return value, 0
    new, n1 = EMAIL_RE.subn(lambda m: "***@" + m.group(0).split("@", 1)[1], value)
    new, n2 = PHONE_RE.subn(lambda m: "***-***-" + re.sub(r"\D", "", m.group(0))[-4:], new)
    return new, n1 + n2


class WarehouseExecutor:
    def __init__(
        self,
        path: Path,
        *,
        timeout_s: float,
        row_cap: int,
        memory_limit: str = "512MB",
        threads: int = 2,
    ) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self.row_cap = row_cap
        self.memory_limit = memory_limit
        self.threads = threads

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.path),
            read_only=True,
            config={
                "enable_external_access": False,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "memory_limit": self.memory_limit,
                "threads": self.threads,
                "lock_configuration": True,
            },
        )

    def ping(self) -> bool:
        con = self._connect()
        try:
            return con.execute("SELECT 1").fetchone() == (1,)
        finally:
            con.close()

    def estimate(self, sql: str) -> CostEstimate:
        """Ask the optimiser for its cardinality estimates without running the query.

        This also catches binder errors (unknown column, bad join) cheaply, before any
        data is scanned, so the self-correction loop gets feedback for free.
        """
        con = self._connect()
        try:
            plan_json = con.execute(f"EXPLAIN (FORMAT JSON) {_without_limit(sql)}").fetchall()[0][1]
        except duckdb.Error as e:
            raise QueryError(_clean(e)) from e
        finally:
            con.close()
        plan = json.loads(plan_json)
        roots = plan if isinstance(plan, list) else [plan]
        cards: list[int] = []
        root = sum(_estimate(r, cards) for r in roots)
        scans = [_card(n) or 0 for n in _walk(plan) if "SCAN" in str(n.get("name", ""))]
        return CostEstimate(
            estimated_rows=root,
            max_intermediate_rows=max(cards, default=0),
            scanned_rows=sum(scans),
        )

    def run(self, sql: str) -> QueryResult:
        con = self._connect()
        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            con.interrupt()

        timer = threading.Timer(self.timeout_s, _kill)
        start = time.perf_counter()
        timer.start()
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description or []]
            raw_rows = cur.fetchmany(self.row_cap + 1)
        except duckdb.Error as e:
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded the {self.timeout_s:.0f}s timeout") from e
            raise QueryError(_clean(e)) from e
        finally:
            timer.cancel()
            con.close()
        elapsed = (time.perf_counter() - start) * 1000
        truncated = len(raw_rows) > self.row_cap
        rows: list[list[Any]] = []
        redactions = 0
        for raw in raw_rows[: self.row_cap]:
            out = []
            for v in raw:
                clean, n = redact(to_jsonable(v))
                redactions += n
                out.append(clean)
            rows.append(out)
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=round(elapsed, 2),
            redactions=redactions,
        )


def _without_limit(sql: str) -> str:
    """Estimate the query as asked, not as capped: LIMIT hides the real cardinality."""
    stmt = sqlglot.parse_one(sql, read="duckdb")
    stmt.set("limit", None)
    return stmt.sql(dialect="duckdb")


def _estimate(node: dict[str, Any], seen: list[int]) -> int:
    """Cardinality of a plan node. DuckDB omits it on some operators (CROSS_PRODUCT,
    LIMIT), so fall back to the product (cross join) or max (others) of the children."""
    children = [_estimate(c, seen) for c in node.get("children", [])]
    card = _card(node)
    if card == 0 and children:
        card = None  # DuckDB reports 0 on some top operators (ORDER_BY); do not trust it
    if card is None:
        if node.get("name") == "CROSS_PRODUCT" and children:
            card = 1
            for c in children:
                card *= max(c, 1)
        else:
            card = max(children, default=0)
    seen.append(card)
    return card


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for n in node:
            yield from _walk(n)
    elif isinstance(node, dict):
        yield node
        for child in node.get("children", []):
            yield from _walk(child)


def _card(node: dict[str, Any]) -> int | None:
    raw = (node.get("extra_info") or {}).get("Estimated Cardinality")
    try:
        return int(str(raw).lstrip("~"))
    except (TypeError, ValueError):
        return None


def _clean(e: Exception) -> str:
    """First meaningful lines of a DuckDB error, without the caret diagram."""
    lines = [ln for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("^")]
    return " ".join(lines[:2])[:400]
