"""Versioned judge prompts.

A judge prompt is part of the measuring instrument. If its wording changes, scores
shift even when the RAG app did not, so every prompt carries a semantic version and
a fingerprint of its template. `judge_prompts.lock.json` records both, and a unit
test fails when a template changes without a version bump.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

LOCK_PATH = Path(__file__).with_name("judge_prompts.lock.json")


class JudgePrompt(BaseModel):
    name: str
    version: str
    template: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, variables: dict[str, str]) -> str:
        return self.template.format(**variables)


_SCALE = (
    "Think about it briefly, then give an integer score from 0 to 10 and a one-sentence "
    "reason. 10 means fully satisfied, 0 means not at all."
)

FAITHFULNESS = JudgePrompt(
    name="faithfulness",
    version="1.0.0",
    template=(
        "You check whether an ANSWER is supported by CONTEXT. Split the answer into "
        "factual claims. The score is the share of claims directly supported by the "
        "context (ignore citation markers like [doc-id]).\n"
        f"{_SCALE}\n\nCONTEXT:\n{{context}}\n\nANSWER:\n{{answer}}"
    ),
)

ANSWER_RELEVANCY = JudgePrompt(
    name="answer_relevancy",
    version="1.0.0",
    template=(
        "You check whether an ANSWER addresses the QUESTION. Penalise statements that are "
        "off-topic or that do not help answer it. Correctness does not matter here.\n"
        f"{_SCALE}\n\nQUESTION:\n{{question}}\n\nANSWER:\n{{answer}}"
    ),
)

CONTEXT_RELEVANCE = JudgePrompt(
    name="context_relevance",
    version="1.0.0",
    template=(
        "You check whether retrieved CONTEXT passages are relevant to the QUESTION. The "
        "score is the share of passages that contain information useful for answering.\n"
        f"{_SCALE}\n\nQUESTION:\n{{question}}\n\nCONTEXT:\n{{context}}"
    ),
)

CONTEXTUAL_RECALL = JudgePrompt(
    name="contextual_recall",
    version="1.0.0",
    template=(
        "You check whether the CONTEXT contains everything needed for the REFERENCE "
        "answer. Split the reference into statements; the score is the share of "
        "statements attributable to the context.\n"
        f"{_SCALE}\n\nREFERENCE:\n{{reference}}\n\nCONTEXT:\n{{context}}"
    ),
)

# G-Eval correctness is run through DeepEval; its criteria, steps and rubric are
# versioned here in the same way so they are part of the run's judge fingerprint.
CORRECTNESS_GEVAL = JudgePrompt(
    name="correctness_geval",
    version="1.0.0",
    template=json.dumps(
        {
            "criteria": "Is the actual output factually correct and complete compared with "
            "the expected output, for an employee asking about company policy?",
            "steps": [
                "Identify every fact (numbers, deadlines, approvers, conditions) in the "
                "expected output.",
                "Check whether the actual output states each fact correctly.",
                "Penalise contradictions heavily and omissions moderately.",
                "Do not penalise extra correct detail or different wording.",
                "If the expected output corrects a false premise, the actual output must "
                "also reject the premise.",
            ],
            "rubric": [
                [0, 2, "Contradicts the expected output or answers a different question."],
                [3, 6, "Partly correct: key facts missing or one fact wrong."],
                [7, 8, "Correct with a minor omission."],
                [9, 10, "Fully correct and complete."],
            ],
        },
        indent=1,
    ),
)

REGISTRY: dict[str, JudgePrompt] = {
    p.name: p
    for p in (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_RELEVANCE, CONTEXTUAL_RECALL,
              CORRECTNESS_GEVAL)
}


def fingerprint_all() -> dict[str, dict[str, str]]:
    return {n: {"version": p.version, "sha": p.fingerprint} for n, p in sorted(REGISTRY.items())}


def write_lock() -> None:
    LOCK_PATH.write_text(json.dumps(fingerprint_all(), indent=2) + "\n")


def lock_violations() -> list[str]:
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}
    problems = []
    for name, current in fingerprint_all().items():
        locked = lock.get(name)
        if locked and locked["sha"] != current["sha"] and locked["version"] == current["version"]:
            problems.append(f"{name}: template changed but version is still {current['version']}")
    return problems
