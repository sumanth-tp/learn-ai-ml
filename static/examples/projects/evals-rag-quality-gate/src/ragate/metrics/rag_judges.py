"""Two interchangeable backends for the judge-based RAG metrics.

``NativeRagMetrics`` uses our versioned prompts through any Judge (LLM or heuristic).
``DeepEvalRagMetrics`` uses DeepEval's claim-level metrics with a real model. Both
return scores in 0..1 so the runner, report and gate do not care which one ran.
"""

from __future__ import annotations

from typing import Protocol

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from ragate.judge.base import Judge, JudgeError
from ragate.metrics import generation
from ragate.models import RagAnswer


class RagJudgeMetrics(Protocol):
    backend: str

    def faithfulness(self, answer: RagAnswer) -> float: ...
    def answer_relevancy(self, answer: RagAnswer) -> float: ...
    def context_relevance(self, answer: RagAnswer) -> float: ...
    def contextual_recall(self, answer: RagAnswer, reference: str) -> float: ...


class NativeRagMetrics:
    backend = "native"

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def faithfulness(self, answer: RagAnswer) -> float:
        return generation.faithfulness(self.judge, answer)

    def answer_relevancy(self, answer: RagAnswer) -> float:
        return generation.answer_relevancy(self.judge, answer)

    def context_relevance(self, answer: RagAnswer) -> float:
        return generation.context_relevance(self.judge, answer)

    def contextual_recall(self, answer: RagAnswer, reference: str) -> float:
        return generation.contextual_recall(self.judge, answer, reference)


class DeepEvalRagMetrics:
    backend = "deepeval"

    def __init__(self, model: DeepEvalBaseLLM) -> None:
        kw = {"model": model, "async_mode": False, "include_reason": False}
        self._faith = FaithfulnessMetric(**kw)
        self._relevancy = AnswerRelevancyMetric(**kw)
        self._ctx_rel = ContextualRelevancyMetric(**kw)
        self._recall = ContextualRecallMetric(**kw)

    @staticmethod
    def _case(answer: RagAnswer, reference: str = "") -> LLMTestCase:
        return LLMTestCase(
            input=answer.question,
            actual_output=answer.answer,
            expected_output=reference or None,
            retrieval_context=[c.chunk.text for c in answer.contexts] or [""],
        )

    @staticmethod
    def _run(metric, case: LLMTestCase) -> float:  # type: ignore[no-untyped-def]
        try:
            metric.measure(case)
        except Exception as exc:
            raise JudgeError(f"{type(metric).__name__} failed: {exc!r}") from exc
        return float(metric.score or 0.0)

    def faithfulness(self, answer: RagAnswer) -> float:
        return self._run(self._faith, self._case(answer))

    def answer_relevancy(self, answer: RagAnswer) -> float:
        return self._run(self._relevancy, self._case(answer))

    def context_relevance(self, answer: RagAnswer) -> float:
        return self._run(self._ctx_rel, self._case(answer))

    def contextual_recall(self, answer: RagAnswer, reference: str) -> float:
        return self._run(self._recall, self._case(answer, reference))
