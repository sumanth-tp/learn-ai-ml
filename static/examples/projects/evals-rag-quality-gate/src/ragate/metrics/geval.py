"""Answer correctness with DeepEval's G-Eval, a pinned rubric and pinned evaluation steps.

Pinning ``evaluation_steps`` matters: left empty, G-Eval asks the judge to invent the
steps on every run, which is one more source of run-to-run noise.
"""

from __future__ import annotations

import json

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from ragate.judge.base import JudgeError
from ragate.judge.prompts import CORRECTNESS_GEVAL
from ragate.metrics.deepeval_models import CURRENT_CASE


class CorrectnessGEval:
    def __init__(self, model: DeepEvalBaseLLM) -> None:
        spec = json.loads(CORRECTNESS_GEVAL.template)
        self.model_id = model.get_model_name()
        self.metric = GEval(
            name="Correctness",
            evaluation_steps=spec["steps"],
            rubric=[
                Rubric(score_range=(lo, hi), expected_outcome=text)
                for lo, hi, text in spec["rubric"]
            ],
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            model=model,
            async_mode=False,
            threshold=0.7,
        )

    def score(self, question: str, answer: str, reference: str) -> float:
        token = CURRENT_CASE.set((answer, reference))
        try:
            self.metric.measure(
                LLMTestCase(input=question, actual_output=answer, expected_output=reference)
            )
        except Exception as exc:  # DeepEval raises bare exceptions on bad judge output
            raise JudgeError(f"G-Eval failed: {exc!r}") from exc
        finally:
            CURRENT_CASE.reset(token)
        if self.metric.score is None:
            raise JudgeError("G-Eval returned no score")
        return float(self.metric.score)
