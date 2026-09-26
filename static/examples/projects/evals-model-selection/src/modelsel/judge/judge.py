"""LLM-as-judge for drafted replies: G-Eval-style pointwise scoring, pairwise with swap, and audits.

Three mitigations are switches here so calibration can measure what each buys:

* ``anchored``: the rubric describes what a 1, 3 and 5 look like. Without anchors
  judges compress scores towards the middle and disagree with each other more.
* ``reference_guided``: the judge sees the reference reply. It no longer has to
  know the refund policy to spot a wrong promise.
* ``swap``: pairwise comparisons run in both orders and are averaged, cancelling
  position bias at the price of twice the calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from modelsel.harness.client import LLMClient

JUDGE_PROMPT_VERSION = "judge-v2"

CRITERIA = (
    "Helpfulness and correctness of a drafted customer-support reply: it must greet the "
    "customer by name, acknowledge their product, state the correct next action under "
    "policy, cite the order number when one exists, and close politely. Extra length "
    "that adds no information is not a virtue."
)

STEPS = """1. Read the ticket and identify the customer's actual request.
2. If a reference reply is given, identify the next action it commits to.
3. Check the response for: greeting by name, acknowledgement, the correct next action, order reference, polite close.
4. A wrong or invented next action is a severe error even if the tone is good.
5. Ignore length and style unless they hide or contradict the action."""

ANCHORS = """ANCHORS:
1 = wrong or missing next action, or promises something against policy.
2 = next action vague; several required parts missing.
3 = correct next action but two parts missing (e.g. no name, no order number).
4 = correct next action, one minor part missing.
5 = correct next action and every required part present, nothing invented."""


def pointwise_system(anchored: bool) -> str:
    parts = [
        "You are an impartial evaluator of customer-support replies.",
        "TASK: judge-pointwise",
        f"CRITERIA: {CRITERIA}",
        f"EVALUATION STEPS:\n{STEPS}",
    ]
    if anchored:
        parts.append(ANCHORS)
    parts.append("Write your step-by-step checks, then a final line exactly `Score: N` with N in 1..5.")
    return "\n\n".join(parts)


def pairwise_system(anchored: bool) -> str:
    parts = [
        "You are an impartial evaluator comparing two customer-support replies.",
        "TASK: judge-pairwise",
        f"CRITERIA: {CRITERIA}",
        "Do not let the order of the responses or their length influence you.",
    ]
    if anchored:
        parts.append(ANCHORS)
    parts.append("Explain briefly, then a final line exactly `Verdict: A`, `Verdict: B` or `Verdict: tie`.")
    return "\n\n".join(parts)


AUDIT_SYSTEM = f"""You audit another model's grading of a customer-support reply.
TASK: judge-audit
CRITERIA: {CRITERIA}

{ANCHORS}

Say whether you agree with the judge's score (within one point), then give your own.
Final lines exactly: `Assessment: agree|disagree` and `Corrected score: N`."""


def _user(ticket: str, reference: str | None, **responses: str) -> str:
    blocks = [f"TICKET:\n{ticket}"]
    if reference:
        blocks.append(f"REFERENCE:\n{reference}")
    for header, body in responses.items():
        blocks.append(f"{header.replace('_', ' ').upper()}:\n{body}")
    return "\n\n".join(blocks)


def probability_weighted_score(logprobs: Any) -> float | None:
    """G-Eval's trick: E[score] = sum_s s * p(s) over the score token's top alternatives.

    It turns a coarse 1..5 into a continuous score and removes most sampling variance.
    Works with the OpenAI logprobs shape that LangChain surfaces in ``response_metadata``.
    """
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content") or []
    for entry in reversed(content):
        if str(entry.get("token", "")).strip() in {"1", "2", "3", "4", "5"}:
            probs: dict[int, float] = {}
            for alt in entry.get("top_logprobs") or []:
                tok = str(alt.get("token", "")).strip()
                if tok in {"1", "2", "3", "4", "5"}:
                    probs[int(tok)] = probs.get(int(tok), 0.0) + math.exp(float(alt["logprob"]))
            total = sum(probs.values())
            if total > 0:
                return sum(s * p for s, p in probs.items()) / total
    return None


_SCORE = re.compile(r"Score:\s*([1-5](?:\.\d+)?)", re.I)
_VERDICT = re.compile(r"Verdict:\s*(A|B|tie)\b", re.I)


@dataclass(frozen=True)
class JudgeScore:
    score: float | None
    weighted: bool
    raw: str
    cost_usd: float


@dataclass(frozen=True)
class PairwiseResult:
    score_a: float
    """Share of A's wins across orders, ties counted as 0.5."""
    consistent: bool
    verdicts: tuple[str, ...]
    cost_usd: float


@dataclass(frozen=True)
class AuditResult:
    agree: bool
    corrected: float | None


Verdict = Literal["A", "B", "tie"]


class Judge:
    def __init__(
        self,
        client: LLMClient,
        judge_id: str,
        *,
        anchored: bool = True,
        reference_guided: bool = True,
    ) -> None:
        self.client = client
        self.judge_id = judge_id
        self.anchored = anchored
        self.reference_guided = reference_guided

    @property
    def version(self) -> str:
        return f"{JUDGE_PROMPT_VERSION}-a{int(self.anchored)}-r{int(self.reference_guided)}"

    async def pointwise(self, ticket: str, reply: str, reference: str | None) -> JudgeScore:
        ref = reference if self.reference_guided else None
        messages = [
            SystemMessage(content=pointwise_system(self.anchored)),
            HumanMessage(content=_user(ticket, ref, response=reply)),
        ]
        c = await self.client.complete(
            self.judge_id, messages, prompt_version=self.version, max_tokens=400, tags=["judge", "pointwise"]
        )
        weighted = probability_weighted_score(c.response_metadata.get("logprobs"))
        if weighted is not None:
            return JudgeScore(weighted, True, c.text, c.cost_usd)
        match = _SCORE.search(c.text)
        return JudgeScore(float(match.group(1)) if match else None, False, c.text, c.cost_usd)

    async def _verdict(self, ticket: str, a: str, b: str, reference: str | None) -> tuple[Verdict | None, float]:
        ref = reference if self.reference_guided else None
        messages = [
            SystemMessage(content=pairwise_system(self.anchored)),
            HumanMessage(content=_user(ticket, ref, response_a=a, response_b=b)),
        ]
        c = await self.client.complete(
            self.judge_id, messages, prompt_version=self.version, max_tokens=300, tags=["judge", "pairwise"]
        )
        match = _VERDICT.search(c.text)
        if not match:
            return None, c.cost_usd
        v = match.group(1)
        return ("tie" if v.lower() == "tie" else v.upper()), c.cost_usd  # type: ignore[return-value]

    async def pairwise(
        self, ticket: str, a: str, b: str, reference: str | None, *, swap: bool = True
    ) -> PairwiseResult:
        first, cost1 = await self._verdict(ticket, a, b, reference)
        points = {"A": 1.0, "tie": 0.5, "B": 0.0, None: 0.5}
        if not swap:
            return PairwiseResult(points[first], True, (str(first),), cost1)
        second_raw, cost2 = await self._verdict(ticket, b, a, reference)
        second = {"A": "B", "B": "A", "tie": "tie", None: None}[second_raw]  # map back to original labels
        score = (points[first] + points[second]) / 2
        return PairwiseResult(score, first == second, (str(first), str(second)), cost1 + cost2)

    async def audit(self, ticket: str, reply: str, reference: str | None, judge_score: float) -> AuditResult:
        user = _user(ticket, reference, response=reply) + f"\n\nJUDGE SCORE:\n{judge_score:.1f}"
        c = await self.client.complete(
            self.judge_id,
            [SystemMessage(content=AUDIT_SYSTEM), HumanMessage(content=user)],
            prompt_version=JUDGE_PROMPT_VERSION,
            max_tokens=300,
            tags=["judge", "audit"],
        )
        agree = bool(re.search(r"Assessment:\s*agree", c.text, re.I))
        m = re.search(r"Corrected score:\s*([1-5])", c.text)
        return AuditResult(agree, float(m.group(1)) if m else None)
