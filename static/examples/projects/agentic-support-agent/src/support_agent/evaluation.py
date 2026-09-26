"""Offline evaluation: routing accuracy, tool-call correctness and task success.

Each case runs end to end through SupportRunner (guardrails, routing, tools,
approvals, persistence) on a fresh database and thread. The same harness runs
against the fakes in CI and against a real model with --live.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from support_agent.config import Settings
from support_agent.container import build_container
from support_agent.graph import ApprovalDecision, build_graph
from support_agent.persistence import open_persistence
from support_agent.runner import SupportRunner


@dataclass
class CaseResult:
    id: str
    routing_ok: bool
    tools_ok: bool
    success: bool
    intent: str
    tools: list[str]
    answer: str
    failures: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[CaseResult]

    def _rate(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.results) / max(1, len(self.results))

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "routing_accuracy": round(self._rate("routing_ok"), 3),
            "tool_call_accuracy": round(self._rate("tools_ok"), 3),
            "task_success": round(self._rate("success"), 3),
        }

    def gate(self, thresholds: dict[str, float]) -> list[str]:
        """Return the metrics below threshold. Empty means the gate passes."""
        m = self.metrics
        return [f"{k}={m[k]} < {v}" for k, v in thresholds.items() if m.get(k, 0.0) < v]


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_case(case: dict[str, Any], base: Settings) -> CaseResult:
    with tempfile.TemporaryDirectory() as tmp:
        settings = base.model_copy(
            update={
                "database_url": f"sqlite:///{tmp}/eval.db",
                "checkpoint_backend": "memory",
                "tool_backoff_initial_s": 0.01,
                "llm_node_retry_initial_s": 0.01,
            }
        )
        container = build_container(settings)
        try:
            async with open_persistence(settings) as (saver, store):
                runner = SupportRunner(container, build_graph(container.deps, saver, store))
                thread_id = f"eval_{case['id']}"
                done: dict[str, Any] = {}
                for turn in case["turns"]:
                    done = await runner.run_turn(thread_id, case["user_id"], turn)
                interrupted = bool(done.get("interrupted"))
                if interrupted and case.get("approve") is not None:
                    done = await runner.resume(
                        thread_id,
                        ApprovalDecision(
                            approved=case["approve"],
                            reviewer="eval-bot",
                            note="" if case["approve"] else "Outside policy.",
                        ),
                    )
                snap = await runner.graph.aget_state({"configurable": {"thread_id": thread_id}})
                tools = [
                    tc["name"]
                    for m in snap.values.get("messages", [])
                    if isinstance(m, AIMessage)
                    for tc in m.tool_calls
                ]
                intent = snap.values.get("intent") or "blocked"
                if snap.values.get("blocked_reason", "") and str(
                    snap.values.get("blocked_reason")
                ).startswith("prompt_injection"):
                    intent = "blocked"
                exp = case["expect"]
                answer = str(done.get("answer", ""))
                failures: list[str] = []
                routing_ok = intent == exp["intent"]
                if not routing_ok:
                    failures.append(f"intent {intent} != {exp['intent']}")
                # Only the last turn's tools count for single-turn cases; for
                # multi-turn cases the expectation lists the tools of the last turn too.
                last_turn_tools = tools[-len(exp["tools"]) :] if exp["tools"] else []
                tools_ok = set(last_turn_tools) == set(exp["tools"]) and (
                    "issue_refund" in tools
                ) == ("issue_refund" in exp["tools"])
                if not tools_ok:
                    failures.append(f"tools {tools} != {exp['tools']}")
                for needle in exp.get("contains", []):
                    if needle.lower() not in answer.lower():
                        failures.append(f"answer missing {needle!r}")
                if "interrupted" in exp and interrupted != exp["interrupted"]:
                    failures.append(f"interrupted={interrupted}")
                for order_id, count in exp.get("refunds", {}).items():
                    got = container.deps.tools.refunds.count_succeeded(order_id)
                    if got != count:
                        failures.append(f"{order_id} refunds {got} != {count}")
                return CaseResult(
                    case["id"], routing_ok, tools_ok, not failures, intent, tools, answer, failures
                )
        finally:
            container.engine.dispose()


async def run_eval(dataset: Path, settings: Settings) -> EvalReport:
    return EvalReport([await run_case(c, settings) for c in load_cases(dataset)])
