"""The offline regression suite: golden cases (seeded + promoted from production) run
against a candidate config, scored with trajectory metrics, task checks and the same
judges used online, then gated against thresholds and a stored baseline."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentmon.agent.backend import FakeBankBackend
from agentmon.agent.guardrails import REFUSAL
from agentmon.agent.service import AgentService
from agentmon.clock import SimClock
from agentmon.config import Settings
from agentmon.evals.judges import LLMJudge
from agentmon.evals.trajectory import (
    ExpectedCall,
    TrajectoryMode,
    detect_loop,
    step_efficiency,
    tool_call_accuracy,
    trajectory_match,
)
from agentmon.llm.factory import build_agent_model
from agentmon.store import Store
from agentmon.tracing import Tracing, build_tracer_provider


class GoldenCase(BaseModel):
    id: str
    input: str
    user_id: str = "CUST-1"
    expected_calls: list[ExpectedCall] = Field(default_factory=list)
    trajectory_mode: TrajectoryMode = "in_order"
    answer_contains: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    forbid_transfer_to: list[str] = Field(default_factory=list)
    source: str = "seed"
    tags: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    id: str
    source: str
    tool_call_accuracy: float
    trajectory_ok: bool
    step_efficiency: float
    loop: bool
    task_completed: bool
    grounded: bool
    policy_ok: bool
    output: str
    failures: list[str]


class SuiteReport(BaseModel):
    prompt_version: str
    n: int
    metrics: dict[str, float]
    cases: list[CaseResult]


class GateResult(BaseModel):
    passed: bool
    failures: list[str]


def load_cases(*paths: Path) -> list[GoldenCase]:
    cases: dict[str, GoldenCase] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                c = GoldenCase.model_validate_json(line)
                cases[c.id] = c
    return list(cases.values())


def _task_checks(
    case: GoldenCase, output: str, blocked: bool, backend: FakeBankBackend
) -> list[str]:
    fails = []
    low = output.lower()
    fails += [f"missing '{s}'" for s in case.answer_contains if s.lower() not in low]
    fails += [f"contains '{s}'" for s in case.must_not_contain if s.lower() in low]
    if case.expect_refusal and not (blocked or output == REFUSAL):
        fails.append("expected a refusal")
    for acc in case.forbid_transfer_to:
        if any(t["to_account"] == acc for t in backend.transfers):
            fails.append(f"money moved to {acc}")
    return fails


async def run_suite(
    cases: list[GoldenCase],
    settings: Settings,
    judges: list[LLMJudge],
    prompt_version: str | None = None,
) -> SuiteReport:
    version = prompt_version or settings.prompt_version
    seed = json.loads(Path(settings.seed_path).read_text())
    store = Store(":memory:")
    clock = SimClock(1_790_000_000.0)
    tracing = Tracing(build_tracer_provider(store, "regression"), clock)
    model = build_agent_model(settings, clock)
    by_name = {j.criterion: j for j in judges}
    results: list[CaseResult] = []
    for case in cases:
        backend = FakeBankBackend(seed)  # fresh state per case: side effects never leak
        service = AgentService(settings, store, backend, model, tracing, clock)
        service.prompt_version = version
        resp = service.handle(case.user_id, case.input, request_id=f"reg-{version}-{case.id}")
        trace = store.get_trace(resp.trace_id)
        assert trace is not None
        actual = [c.model_dump() for c in trace.tool_calls]
        grounded = (await by_name["groundedness"].aevaluate(trace)).passed
        policy_ok = (await by_name["policy"].aevaluate(trace)).passed
        tca = tool_call_accuracy(case.expected_calls, actual)
        traj = trajectory_match(case.expected_calls, actual, case.trajectory_mode)
        fails = _task_checks(case, resp.answer, resp.blocked, backend)
        failures = list(fails)
        if tca < 1.0:
            failures.append(f"tool_call_accuracy={tca:.2f}")
        if not traj:
            failures.append(
                f"trajectory({case.trajectory_mode}) mismatch: {[a['name'] for a in actual]}"
            )
        if not grounded:
            failures.append("ungrounded")
        if not policy_ok:
            failures.append("policy")
        results.append(
            CaseResult(
                id=case.id,
                source=case.source,
                tool_call_accuracy=tca,
                trajectory_ok=traj,
                step_efficiency=step_efficiency(case.expected_calls, actual),
                loop=detect_loop(actual),
                task_completed=not fails,
                grounded=grounded,
                policy_ok=policy_ok,
                output=resp.answer,
                failures=failures,
            )
        )
    n = max(len(results), 1)
    metrics = {
        "tool_call_accuracy": sum(r.tool_call_accuracy for r in results) / n,
        "trajectory_match": sum(r.trajectory_ok for r in results) / n,
        "task_completion": sum(r.task_completed for r in results) / n,
        "groundedness": sum(r.grounded for r in results) / n,
        "policy": sum(r.policy_ok for r in results) / n,
        "step_efficiency": sum(r.step_efficiency for r in results) / n,
        "loop_rate": sum(r.loop for r in results) / n,
    }
    store.close()
    return SuiteReport(prompt_version=version, n=len(results), metrics=metrics, cases=results)


def load_thresholds(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())["gate"]


def gate(
    report: SuiteReport, thresholds: dict[str, Any], baseline: dict[str, float] | None = None
) -> GateResult:
    failures = []
    for metric, floor in thresholds.items():
        if metric in ("max_loop_rate", "max_drop_vs_baseline"):
            continue
        value = report.metrics.get(metric)
        if value is not None and value < floor:
            failures.append(f"{metric}={value:.3f} < {floor}")
    if report.metrics["loop_rate"] > thresholds.get("max_loop_rate", 0.0):
        failures.append(f"loop_rate={report.metrics['loop_rate']:.3f}")
    if baseline:
        drop = thresholds.get("max_drop_vs_baseline", 0.03)
        for metric, base in baseline.items():
            if metric == "loop_rate" or metric not in report.metrics:
                continue
            if base - report.metrics[metric] > drop:
                failures.append(f"{metric} dropped {base - report.metrics[metric]:.3f} vs baseline")
    return GateResult(passed=not failures, failures=failures)
