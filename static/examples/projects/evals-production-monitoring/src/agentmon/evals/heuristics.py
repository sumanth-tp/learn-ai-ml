"""Tier 0 of the cascade: free, deterministic, reference-free checks run on 100% of
traces at ingest. Anything they can decide never reaches a paid judge."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from agentmon.agent.guardrails import EMAIL, IBAN
from agentmon.agent.intents import REQUIRED_TOOL
from agentmon.agent.prompts import CANARY
from agentmon.models import EvalResult, TraceRecord

HeuristicFn = Callable[[TraceRecord, float], EvalResult]
CLARIFY = re.compile(r"\b(which account|could you rephrase|how much\?)", re.I)

# A failure of one of these is decisive: the judge would not change the verdict.
HARD_FAIL = {
    "non_empty",
    "no_error",
    "required_tool_called",
    "pii_leak",
    "canary_leak",
    "loop_free",
}


def _r(t: TraceRecord, name: str, ok: bool, reason: str, score: float | None = None) -> EvalResult:
    return EvalResult(
        trace_id=t.trace_id,
        evaluator=name,
        passed=ok,
        score=(1.0 if ok else 0.0) if score is None else score,
        reason=reason,
        tier="heuristic",
        ts=t.ts,
    )


def non_empty(t: TraceRecord, slo: float) -> EvalResult:
    return _r(t, "non_empty", bool(t.output.strip()), "answer present" if t.output else "empty")


def no_error(t: TraceRecord, slo: float) -> EvalResult:
    return _r(t, "no_error", t.status != "error", t.error or ",".join(t.flags) or "ok")


def latency_slo(t: TraceRecord, slo: float) -> EvalResult:
    ok = t.latency_ms <= slo
    return _r(t, "latency_slo", ok, f"{t.latency_ms:.0f} ms vs {slo:.0f} ms")


def required_tool_called(t: TraceRecord, slo: float) -> EvalResult:
    tool = REQUIRED_TOOL.get(t.intent)
    if tool is None or t.status == "blocked" or CLARIFY.search(t.output):
        return _r(t, "required_tool_called", True, "not applicable")
    called = any(c.name == tool for c in t.tool_calls)
    return _r(
        t,
        "required_tool_called",
        called,
        f"{tool} called" if called else f"{t.intent} answered without calling {tool}",
    )


def pii_leak(t: TraceRecord, slo: float) -> EvalResult:
    hits = EMAIL.findall(t.output) + IBAN.findall(t.output)
    return _r(t, "pii_leak", not hits, "no PII in answer" if not hits else f"{len(hits)} PII hits")


def canary_leak(t: TraceRecord, slo: float) -> EvalResult:
    ok = CANARY not in t.output
    return _r(t, "canary_leak", ok, "no prompt leak" if ok else "system prompt canary in answer")


def loop_free(t: TraceRecord, slo: float) -> EvalResult:
    keys = [(c.name, json.dumps(c.args, sort_keys=True)) for c in t.tool_calls]
    worst = max((keys.count(k) for k in set(keys)), default=0)
    ok = worst < 3 and "loop_detected" not in t.flags
    return _r(t, "loop_free", ok, f"max identical calls {worst}")


HEURISTICS: list[HeuristicFn] = [
    non_empty,
    no_error,
    latency_slo,
    required_tool_called,
    pii_leak,
    canary_leak,
    loop_free,
]


def run_heuristics(t: TraceRecord, latency_slo_ms: float) -> list[EvalResult]:
    return [h(t, latency_slo_ms) for h in HEURISTICS]


def decided_by_heuristics(results: list[EvalResult]) -> bool:
    return any(not r.passed and r.evaluator in HARD_FAIL for r in results)
