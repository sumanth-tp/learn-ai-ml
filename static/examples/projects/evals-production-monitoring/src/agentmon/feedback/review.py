"""The feedback loop: flagged and failing production traces become golden-dataset
candidates. A human approves or rejects each; approved ones are appended to the
production golden file that the offline regression suite reads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from agentmon.agent.intents import REQUIRED_TOOL
from agentmon.clock import Clock
from agentmon.evals.regression import GoldenCase, load_cases
from agentmon.evals.trajectory import ExpectedCall
from agentmon.models import TraceRecord
from agentmon.store import Store

ACC = re.compile(r"ACC-\d{4}")


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 -]", " ", text.lower()).split())


def draft_case(trace: TraceRecord) -> GoldenCase | None:
    """Turn a trace into a proposed regression case. Returns None when the trace has
    nothing a test could pin down (small talk, 'other')."""
    cid = "prod-" + hashlib.sha256(normalise(trace.input).encode()).hexdigest()[:10]
    common: dict[str, Any] = {
        "id": cid,
        "input": trace.input,
        "user_id": trace.user_id,
        "source": "production",
        "tags": [trace.intent, *trace.flags],
    }
    if trace.status == "blocked" or trace.intent == "out_of_scope":
        return GoldenCase(**common, expect_refusal=True)
    if any(f.startswith("sanitised:") for f in trace.flags):
        return GoldenCase(
            **common,
            expected_calls=[ExpectedCall(name="search_help_center")],
            trajectory_mode="subset",
            forbid_transfer_to=["ACC-9999"],
        )
    tool = REQUIRED_TOOL.get(trace.intent)
    if tool is None:
        return None
    accounts = ACC.findall(trace.input)
    if tool in ("get_balance", "list_transactions"):
        if not accounts:
            return None
        call = ExpectedCall(name=tool, args={"account_id": accounts[0]})
    elif tool == "transfer_funds":
        if len(accounts) < 2:
            return None
        call = ExpectedCall(
            name=tool, args={"from_account": accounts[0], "to_account": accounts[1]}
        )
    else:
        call = ExpectedCall(name=tool)
    return GoldenCase(**common, expected_calls=[call], trajectory_mode="in_order")


class ReviewQueue:
    def __init__(self, store: Store, clock: Clock, seed_path: Path, production_path: Path) -> None:
        self.store = store
        self.clock = clock
        self.seed_path = seed_path
        self.production_path = production_path

    def pending(self, limit: int = 50) -> list[dict[str, Any]]:
        out = []
        for item in self.store.review_items("pending", limit):
            trace = self.store.get_trace(item["trace_id"])
            if trace is None:
                continue
            draft = draft_case(trace)
            out.append(
                {
                    **item,
                    "input": trace.input,
                    "output": trace.output,
                    "intent": trace.intent,
                    "prompt_version": trace.prompt_version,
                    "draft": draft.model_dump() if draft else None,
                }
            )
        return out

    def _known_inputs(self) -> set[str]:
        return {normalise(c.input) for c in load_cases(self.seed_path, self.production_path)}

    def approve(self, trace_id: str, reviewer: str, case: GoldenCase | None = None) -> str:
        """Returns the resulting status: approved, duplicate or unusable."""
        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(trace_id)
        case = case or draft_case(trace)
        now = self.clock.now()
        if case is None:
            self.store.set_review(
                trace_id, "unusable", reviewer, now, note="no testable expectation"
            )
            return "unusable"
        if normalise(case.input) in self._known_inputs():
            self.store.set_review(trace_id, "duplicate", reviewer, now, golden_id=case.id)
            return "duplicate"
        self.production_path.parent.mkdir(parents=True, exist_ok=True)
        with self.production_path.open("a") as fh:
            fh.write(case.model_dump_json() + "\n")
        self.store.set_review(trace_id, "approved", reviewer, now, golden_id=case.id)
        return "approved"

    def reject(self, trace_id: str, reviewer: str, note: str = "") -> None:
        self.store.set_review(trace_id, "rejected", reviewer, self.clock.now(), note=note)


def simulated_review(queue: ReviewQueue, limit: int = 200) -> dict[str, int]:
    """Stands in for the human reviewer in the demo: approve whatever has a usable
    draft. A real reviewer also edits the draft (for example the expected answer)."""
    counts: dict[str, int] = {}
    for item in queue.pending(limit):
        status = queue.approve(item["trace_id"], "simulated-reviewer")
        counts[status] = counts.get(status, 0) + 1
    return counts
