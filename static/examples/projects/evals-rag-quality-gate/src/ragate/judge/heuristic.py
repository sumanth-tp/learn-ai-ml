"""Deterministic judge stub used offline and in CI.

It answers the same prompts as the LLM judge using lexical evidence, so scores move
in the right direction when the pipeline gets better or worse. It is not a substitute
for an LLM judge's quality; it is a substitute for its *interface*, with zero noise.
An optional seeded jitter simulates judge noise to exercise the noise-band machinery.
"""

from __future__ import annotations

import hashlib
import random
import re

from ragate.judge.base import JudgeError, Verdict
from ragate.judge.prompts import JudgePrompt
from ragate.text import content_tokens, coverage, sentences

_SUPPORTED = 0.6


def _strip_citations(text: str) -> str:
    return re.sub(r"\[[a-z0-9-]+\]", "", text)


def _claims(text: str) -> list[str]:
    return [s for s in sentences(_strip_citations(text)) if content_tokens(s)]


def _share(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


class HeuristicJudge:
    def __init__(self, jitter: float = 0.0, seed: int = 0) -> None:
        self.jitter = jitter
        self.seed = seed

    @property
    def model_id(self) -> str:
        return "heuristic-v1" if not self.jitter else f"heuristic-v1~{self.jitter:g}"

    def _score(self, prompt: JudgePrompt, v: dict[str, str]) -> tuple[float, str]:
        match prompt.name:
            case "faithfulness":
                claims = _claims(v["answer"])
                ok = [coverage(c, v["context"]) >= _SUPPORTED for c in claims]
                return _share(ok), f"{sum(ok)}/{len(claims)} claims supported by context"
            case "answer_relevancy":
                claims = _claims(v["answer"])
                ok = [coverage(v["question"], c) >= 0.3 for c in claims]
                return _share(ok), f"{sum(ok)}/{len(claims)} statements address the question"
            case "context_relevance":
                passages = [p for p in v["context"].split("\n") if p.strip()]
                ok = [coverage(v["question"], p) >= 0.3 for p in passages]
                return _share(ok), f"{sum(ok)}/{len(passages)} passages relevant"
            case "contextual_recall":
                stmts = _claims(v["reference"])
                ok = [coverage(s, v["context"]) >= _SUPPORTED for s in stmts]
                return _share(ok), f"{sum(ok)}/{len(stmts)} reference statements in context"
        raise JudgeError(f"heuristic judge has no rule for prompt {prompt.name!r}")

    def evaluate(self, prompt: JudgePrompt, variables: dict[str, str]) -> Verdict:
        score, reason = self._score(prompt, variables)
        if self.jitter:
            # Seeded per (seed, input) so a given repeat is reproducible.
            h = hashlib.sha256(f"{self.seed}|{prompt.name}|{sorted(variables.items())}".encode())
            rnd = random.Random(int(h.hexdigest(), 16))
            score = min(1.0, max(0.0, score + rnd.gauss(0, self.jitter)))
        # Quantise to the 0-10 integer scale the LLM judge uses, so both behave alike.
        return Verdict(score=round(score * 10) / 10, reason=reason)
