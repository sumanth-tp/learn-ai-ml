"""Deterministic offline stand-ins for candidate models and judges.

They subclass LangChain's ``BaseChatModel``, so they go through exactly the same
harness path (caching, rate limiting, retries, usage accounting, tracing) as the
real providers. Behaviour is a pure function of (model id, prompt): runs are
reproducible and the cache is meaningful.

The fakes are not random noise. Each one has a *skill* that controls how often it
makes each kind of mistake a real model makes (wrong label on ambiguous tickets,
invalid JSON, missing the next action in a reply), and the judge has the three
biases the calibration step must detect: position, verbosity and self-preference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from abc import abstractmethod
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

from modelsel.dataset import ACTIONS, CLOSE, PRODUCTS, compose_reply
from modelsel.llm.registry import FakeProfile, JudgeProfile
from modelsel.schemas import BenchmarkItem, Label, Priority, Sentiment


class FakeRateLimitError(Exception):
    """Shaped like a provider 429 so the harness's transient-error check treats it the same."""

    status_code = 429


def _rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _section(prompt: str, name: str) -> str:
    """Return the text after ``NAME:`` up to the next upper-case header or the end."""
    match = re.search(rf"^{name}:\n(.*?)(?=^\s*[A-Z_ ]+:\n|\Z)", prompt, re.S | re.M)
    return match.group(1).strip() if match else ""


KEYWORDS: dict[Label, tuple[str, ...]] = {
    Label.REFUND: ("refund", "money back", "returned"),
    Label.CANCELLATION: ("cancel", "close my subscription", "stop charging"),
    Label.ACCOUNT_ACCESS: ("log in", "password", "locked", "email address on my account", "logged into"),
    Label.SHIPPING: ("arrived", "tracking", "deliver", "in transit", "crushed"),
    Label.BUG_REPORT: ("crash", "firmware", "error", "wrong time zone", "bricked"),
    Label.FEATURE_REQUEST: ("would be great", "any chance", "please offer", "adding"),
    Label.BILLING: ("invoice", "charged", "billed", "bill", "vat"),
}


def heuristic_label(ticket: str) -> Label:
    low = ticket.lower()
    for label, words in KEYWORDS.items():
        if any(w in low for w in words):
            return label
    return Label.OTHER


def heuristic_fields(ticket: str) -> dict[str, Any]:
    low = ticket.lower()
    order = re.search(r"ORD-\d{5}", ticket)
    amount = re.search(r"£(\d+(?:\.\d{2})?)", ticket)
    product = next((p for p in PRODUCTS if p in ticket), None)
    if any(w in low for w in ("urgent", "tomorrow", "today", "someone else")):
        priority = Priority.URGENT
    elif any(w in low for w in ("not acceptable", "stopped working", "nine days", "bricked", "immediately", "cannot", "error", "twice", "done with")):
        priority = Priority.HIGH
    elif any(w in low for w in ("?", "please", "could", "how do")) and not any(w in low for w in ("why", "still")):
        priority = Priority.LOW
    else:
        priority = Priority.MEDIUM
    if any(w in low for w in ("love", "brilliant", "great", "thanks to")):
        sentiment = Sentiment.POSITIVE
    elif any(w in low for w in ("not", "never", "crash", "damaged", "wrong", "why", "done with", "bricked", "!")):
        sentiment = Sentiment.NEGATIVE
    else:
        sentiment = Sentiment.NEUTRAL
    return {
        "order_id": order.group(0) if order else None,
        "product": product,
        "amount": float(amount.group(1)) if amount else None,
        "priority": priority.value,
        "sentiment": sentiment.value,
    }


class _FakeBase(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    sleep_scale: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "modelsel-fake"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id}

    @abstractmethod
    def _respond(self, system: str, user: str) -> tuple[str, float, dict[str, Any]]:
        """Return (text, simulated latency in ms, extra response metadata)."""

    def _result(self, messages: list[BaseMessage]) -> tuple[ChatResult, float]:
        system = next((str(m.content) for m in messages if m.type == "system"), "")
        user = "\n".join(str(m.content) for m in messages if m.type == "human")
        text, latency_ms, extra = self._respond(system, user)
        in_tok = _tokens(system + user)
        out_tok = _tokens(text)
        msg = AIMessage(
            content=text,
            usage_metadata={"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok},
            response_metadata={"model_name": self.model_id, "simulated_latency_ms": latency_ms, **extra},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)]), latency_ms

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(messages)[0]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result, latency_ms = self._result(messages)
        if self.sleep_scale:
            await asyncio.sleep(latency_ms / 1000 * self.sleep_scale)
        return result


class FakeTicketModel(_FakeBase):
    """A candidate model. ``skill`` near 1 behaves like a frontier model, near 0.5 like a small local one."""

    profile: FakeProfile = Field(default_factory=FakeProfile)
    memorised: dict[str, BenchmarkItem] = Field(default_factory=dict)
    """Tickets this model 'saw in training'. Simulates benchmark contamination."""
    calls: dict[str, int] = Field(default_factory=dict)

    def _respond(self, system: str, user: str) -> tuple[str, float, dict[str, Any]]:
        key = hashlib.sha256((system + user).encode()).hexdigest()
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.calls[key] <= self.profile.transient_failures:
            raise FakeRateLimitError(f"{self.model_id}: 429 rate limited (simulated)")

        rng = _rng(self.model_id, system, user)
        ticket = _section(user, "TICKET") or user
        seen = self.memorised.get(ticket)
        p = self.profile
        base_latency = p.latency_ms * (0.7 + 0.6 * rng.random())

        if "TASK: continue" in system:
            half = _section(user, "PREFIX")
            if seen is not None:
                return seen.ticket[len(half):], base_latency, {}
            return " and I would like some help with this please.", base_latency, {}

        if "TASK: classify" in system:
            if seen is not None:
                return seen.label.value, base_latency * 0.3, {}
            label = heuristic_label(ticket)
            if rng.random() > p.skill:
                label = rng.choice([lb for lb in Label if lb != label])
            return label.value, base_latency * 0.3, {}

        if "TASK: extract" in system:
            if seen is not None:
                return seen.fields.model_dump_json(), base_latency * 0.6, {}
            fields = heuristic_fields(ticket)
            if rng.random() > p.skill:
                key_to_break = rng.choice(["priority", "sentiment", "product", "amount"])
                fields[key_to_break] = {"priority": "normal", "sentiment": "angry"}.get(key_to_break)
            if rng.random() < p.json_error_rate:
                return "Sure! Here is the JSON:\n" + json.dumps(fields)[:-1], base_latency * 0.6, {}
            return json.dumps(fields), base_latency * 0.6, {}

        name_match = re.search(r"From: (\w+)", ticket)
        name = name_match.group(1) if name_match else "there"
        fields = heuristic_fields(ticket)
        label = seen.label if seen is not None else heuristic_label(ticket)
        parts = [part for part in ("greeting", "ack", "action", "order", "close") if rng.random() < 0.55 + 0.45 * p.skill]
        wrong = seen is None and rng.random() > p.skill
        text = compose_reply(
            name, label, fields["product"], fields["order_id"], parts=parts,
            action_label=rng.choice([lb for lb in Label if lb != label]) if wrong else None,
            filler_paragraphs=p.verbosity, close=p.signature or CLOSE,
        )
        return text, base_latency * (1 + 0.4 * p.verbosity), {}


_STOP = frozenset("a an and the to of your you for on in is it this i have will we be with our by at as".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if w not in _STOP and len(w) > 2}


class FakeJudgeModel(_FakeBase):
    """An LLM judge with tunable biases, driven by the rubric prompts in ``modelsel.judge``."""

    profile: JudgeProfile = Field(default_factory=JudgeProfile)

    def quality(self, reply: str, ticket: str, reference: str, anchored: bool) -> float:
        """Estimate quality on 1..5 from what a good reply must contain."""
        if not reply.strip():
            return 1.0
        rw = _content_words(reply)
        if reference:
            action = next((a for a in ACTIONS.values() if a in reference), "")
            action_hit = 1.0 if action and action in reply else 0.0
            ref_words = _content_words(reference.replace(action, "")) if action else _content_words(reference)
            recall = len(ref_words & rw) / max(1, len(ref_words))
            raw = 0.55 * action_hit + 0.45 * recall
        else:
            guessed = ACTIONS[heuristic_label(ticket)]
            action_hit = 1.0 if guessed in reply else 0.0
            recall = len(_content_words(ticket) & rw) / max(1, len(_content_words(ticket)))
            raw = 0.45 * action_hit + 0.55 * min(1.0, recall * 1.6)
        score = 1 + 4 * raw
        extra_words = max(0, len(reply.split()) - 70)
        score += self.profile.verbosity_bias * min(1.0, extra_words / 60)
        if self.profile.self_bonus and "Warm regards" in reply:
            score += self.profile.self_bonus
        if not anchored:
            score = 3 + (score - 3) * 0.6  # without anchors, judges drift to the middle
        return max(1.0, min(5.0, score))

    def _respond(self, system: str, user: str) -> tuple[str, float, dict[str, Any]]:
        rng = _rng(self.model_id, system, user)
        anchored = "ANCHORS:" in system
        noise = self.profile.noise * (1.0 if anchored else 1.8)
        ticket = _section(user, "TICKET")
        reference = _section(user, "REFERENCE")
        latency = 900 * (0.7 + 0.6 * rng.random())

        if "TASK: judge-pairwise" in system:
            qa = self.quality(_section(user, "RESPONSE A"), ticket, reference, anchored) + rng.gauss(0, noise)
            qb = self.quality(_section(user, "RESPONSE B"), ticket, reference, anchored) + rng.gauss(0, noise)
            if abs(qa - qb) < 0.9 and rng.random() < self.profile.position_bias:
                verdict = "A"
            elif abs(qa - qb) < 0.35:
                verdict = "tie"
            else:
                verdict = "A" if qa > qb else "B"
            return f"Reasoning: compared both against the rubric.\nVerdict: {verdict}", latency, {}

        if "TASK: judge-audit" in system:
            true_q = self.quality(_section(user, "RESPONSE"), ticket, reference, True) + rng.gauss(0, noise * 0.5)
            claimed = re.search(r"JUDGE SCORE:\n(\d(?:\.\d+)?)", user)
            claimed_v = float(claimed.group(1)) if claimed else 3.0
            agree = abs(true_q - claimed_v) < 1.0
            return (
                f"Assessment: {'agree' if agree else 'disagree'}\nCorrected score: {round(max(1, min(5, true_q)))}",
                latency * 1.5,
                {},
            )

        q = self.quality(_section(user, "RESPONSE"), ticket, reference, anchored) + rng.gauss(0, noise)
        q = max(1.0, min(5.0, q))
        # Emit OpenAI-shaped logprobs over the score token so G-Eval weighting is exercised offline.
        logits = {str(s): -((s - q) ** 2) / 0.5 for s in range(1, 6)}
        z = math.log(sum(math.exp(v) for v in logits.values()))
        top = [{"token": t, "logprob": v - z} for t, v in sorted(logits.items(), key=lambda kv: -kv[1])]
        best = top[0]["token"]
        text = f"Steps: checked greeting, acknowledgement, next action, order reference, close.\nScore: {best}"
        logprobs = {"content": [{"token": best, "logprob": top[0]["logprob"], "top_logprobs": top, "position": "score"}]}
        return text, latency, {"logprobs": logprobs}
