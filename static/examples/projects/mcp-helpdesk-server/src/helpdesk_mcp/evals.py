"""Offline evaluation of triage quality, with a regression gate.

Metrics:

* ``category_accuracy``: share of cases with the right category.
* ``priority_accuracy``: share with the right priority.
* ``p1_recall``: of the true P1 outages, how many we called P1. This is the
  one that matters most: under-calling an outage costs far more than
  over-calling a password reset, so it has the strictest threshold.
* ``fallback_rate``: share answered by rules because the model failed.

The same dataset runs against the offline fake (in CI, every commit) and the
real model (``make eval-live``, before changing model or prompt).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from helpdesk_mcp.triage import TriageService

THRESHOLDS = {
    "category_accuracy": 0.80,
    "priority_accuracy": 0.70,
    "p1_recall": 0.95,
    "fallback_rate_max": 0.10,
}


@dataclass
class EvalReport:
    cases: int
    category_accuracy: float
    priority_accuracy: float
    p1_recall: float
    fallback_rate: float
    failures: list[dict]

    def gate(self) -> list[str]:
        """Return the list of thresholds this run violates (empty means pass)."""
        problems = []
        for metric in ("category_accuracy", "priority_accuracy", "p1_recall"):
            if getattr(self, metric) < THRESHOLDS[metric]:
                problems.append(f"{metric}={getattr(self, metric):.2f} < {THRESHOLDS[metric]}")
        if self.fallback_rate > THRESHOLDS["fallback_rate_max"]:
            problems.append(f"fallback_rate={self.fallback_rate:.2f} too high")
        return problems


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run_eval(service: TriageService, cases: list[dict]) -> EvalReport:
    cat_ok = pri_ok = fallbacks = p1_total = p1_hit = 0
    failures: list[dict] = []
    for case in cases:
        result = await service.suggest(case["title"], case["description"])
        c_ok = result.category.value == case["category"]
        p_ok = result.priority.value == case["priority"]
        cat_ok += c_ok
        pri_ok += p_ok
        fallbacks += result.source == "fallback"
        if case["priority"] == "p1":
            p1_total += 1
            p1_hit += result.priority.value == "p1"
        if not (c_ok and p_ok):
            failures.append(
                {
                    "id": case["id"],
                    "expected": [case["category"], case["priority"]],
                    "got": [result.category.value, result.priority.value],
                }
            )
    n = len(cases)
    return EvalReport(
        cases=n,
        category_accuracy=cat_ok / n,
        priority_accuracy=pri_ok / n,
        p1_recall=p1_hit / p1_total if p1_total else 1.0,
        fallback_rate=fallbacks / n,
        failures=failures,
    )


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**asdict(report), "gate": report.gate()}, indent=2) + "\n")
