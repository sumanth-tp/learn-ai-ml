"""Offline evaluation: execution accuracy over a golden set, plus a regression gate.

Execution accuracy compares RESULT SETS, not SQL strings. ``SELECT COUNT(*) FROM
customers`` and ``SELECT COUNT(customer_id) AS n FROM customers c`` are both right;
string match would fail one of them, and would pass a query with the right shape but
the wrong filter.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from data_analyst.executor import WarehouseExecutor, to_jsonable
from data_analyst.service import AnalystService

FLOAT_DP = 2


class GoldenCase(BaseModel):
    id: str
    question: str
    sql: str
    ordered: bool = False
    tags: list[str] = []
    expected: dict[str, Any] | None = None  # {"columns": [...], "rows": [[...]]}


class CaseResult(BaseModel):
    id: str
    question: str
    status: str
    correct: bool
    valid: bool
    retries: int
    latency_ms: float
    cost_usd: float
    sql: str | None
    error: str | None = None


class EvalReport(BaseModel):
    n: int
    execution_accuracy: float
    validity_rate: float
    mean_retries: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    cost_per_question_usd: float
    passing_ids: list[str]
    cases: list[CaseResult]


# ---------------------------------------------------------------- result matching


def _norm(v: Any) -> Any:
    v = to_jsonable(v)
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        f = float(v)
        return round(f, FLOAT_DP) if math.isfinite(f) else None
    return str(v)


def _rows(rows: list[list[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(_norm(v) for v in r) for r in rows]


def results_match(pred_rows: list[list[Any]], gold_rows: list[list[Any]], *, ordered: bool) -> bool:
    """True if two result sets are equal up to column order, column names and float noise.

    Rows are compared as a multiset unless the question implies an order. Columns may be
    permuted: we look for a mapping of predicted columns onto gold columns whose value
    multisets agree, then compare rows under that mapping.
    """
    pred, gold = _rows(pred_rows), _rows(gold_rows)
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if any(len(r) != width for r in pred):
        return False

    def cols(rows: list[tuple[Any, ...]]) -> list[Counter[Any]]:
        return [Counter(r[i] for r in rows) for i in range(width)]

    pcols, gcols = cols(pred), cols(gold)
    mapping: list[int] = []
    used: set[int] = set()
    for g in range(width):
        match = next((p for p in range(width) if p not in used and pcols[p] == gcols[g]), None)
        if match is None:
            return False
        mapping.append(match)
        used.add(match)
    remapped = [tuple(r[m] for m in mapping) for r in pred]
    return remapped == gold if ordered else Counter(remapped) == Counter(gold)


# ---------------------------------------------------------------- golden set IO


def load_golden(path: Path) -> list[GoldenCase]:
    return [
        GoldenCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def refresh_expected(path: Path, executor: WarehouseExecutor) -> list[GoldenCase]:
    """Run each gold SQL and store its result. Re-run when the seed or schema changes,
    review the diff, then commit it: the expected results are part of the test."""
    cases = load_golden(path)
    for c in cases:
        res = executor.run(c.sql)
        c.expected = {"columns": res.columns, "rows": res.rows}
    path.write_text("".join(c.model_dump_json() + "\n" for c in cases))
    return cases


# ---------------------------------------------------------------- running


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return round(ordered[k], 1)


def run_eval(service: AnalystService, cases: list[GoldenCase]) -> EvalReport:
    """Each case runs in a fresh thread (no memory bleed) and approvals are auto-granted
    so the gate measures SQL quality, not reviewer availability."""
    results: list[CaseResult] = []
    for case in cases:
        if case.expected is None:
            raise ValueError(f"{case.id} has no expected result; run `analyst eval --refresh`")
        start = time.perf_counter()
        out = service.run_to_end(
            f"eval-{case.id}-{uuid.uuid4().hex[:8]}", case.question, auto_approve=True
        )
        latency = (time.perf_counter() - start) * 1000
        result = out.get("result")
        valid = out.get("status") == "answered" and result is not None
        correct = valid and results_match(
            result["rows"], case.expected["rows"], ordered=case.ordered
        )
        results.append(
            CaseResult(
                id=case.id,
                question=case.question,
                status=str(out.get("status")),
                correct=bool(correct),
                valid=valid,
                retries=int(out.get("retries", 0)),
                latency_ms=round(latency, 1),
                cost_usd=float(out.get("cost_usd", 0.0)),
                sql=out.get("sql"),
                error=(out.get("errors") or [None])[-1],
            )
        )
    n = len(results)
    latencies = [r.latency_ms for r in results]
    total_cost = sum(r.cost_usd for r in results)
    return EvalReport(
        n=n,
        execution_accuracy=round(sum(r.correct for r in results) / n, 4) if n else 0.0,
        validity_rate=round(sum(r.valid for r in results) / n, 4) if n else 0.0,
        mean_retries=round(statistics.fmean(r.retries for r in results), 3) if n else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        total_cost_usd=round(total_cost, 6),
        cost_per_question_usd=round(total_cost / n, 6) if n else 0.0,
        passing_ids=sorted(r.id for r in results if r.correct),
        cases=results,
    )


# ---------------------------------------------------------------- regression gate


class Baseline(BaseModel):
    min_execution_accuracy: float
    min_validity_rate: float
    max_mean_retries: float
    max_p95_latency_ms: float
    max_cost_per_question_usd: float
    passing_ids: list[str]


def gate(report: EvalReport, baseline: Baseline) -> list[str]:
    """Return the list of violations. Empty means the change may ship."""
    problems: list[str] = []
    if report.execution_accuracy < baseline.min_execution_accuracy:
        problems.append(
            f"execution accuracy {report.execution_accuracy:.3f} < "
            f"{baseline.min_execution_accuracy:.3f}"
        )
    if report.validity_rate < baseline.min_validity_rate:
        problems.append(f"validity {report.validity_rate:.3f} < {baseline.min_validity_rate:.3f}")
    if report.mean_retries > baseline.max_mean_retries:
        problems.append(f"mean retries {report.mean_retries} > {baseline.max_mean_retries}")
    if report.p95_latency_ms > baseline.max_p95_latency_ms:
        problems.append(f"p95 latency {report.p95_latency_ms}ms > {baseline.max_p95_latency_ms}ms")
    if report.cost_per_question_usd > baseline.max_cost_per_question_usd:
        problems.append(
            f"cost/question ${report.cost_per_question_usd} > ${baseline.max_cost_per_question_usd}"
        )
    regressed = sorted(set(baseline.passing_ids) - set(report.passing_ids))
    if regressed:
        problems.append(f"previously passing cases now fail: {', '.join(regressed)}")
    return problems


def load_baseline(path: Path) -> Baseline:
    return Baseline.model_validate_json(path.read_text())


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2))
