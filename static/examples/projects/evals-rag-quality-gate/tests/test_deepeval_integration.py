"""Contract tests against the installed DeepEval: they fail loudly if its API drifts."""

import typing
from pathlib import Path

from deepeval.models import DeepEvalBaseLLM
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from ragate.judge.cache import JudgeCache
from ragate.metrics.deepeval_models import LangChainDeepEvalModel, StubGEvalModel
from ragate.metrics.geval import CorrectnessGEval
from ragate.metrics.rag_judges import DeepEvalRagMetrics
from ragate.models import RagAnswer
from tests.conftest import make_chunk

REF = "Hotels in London are capped at 180 GBP per night."


def test_geval_with_rubric_scores_correct_above_wrong() -> None:
    geval = CorrectnessGEval(StubGEvalModel())
    right = geval.score("hotel cap?", "Hotel rates are capped at 180 GBP per night in London.",
                        REF)
    wrong = geval.score("hotel cap?", "Laptops are refreshed every 3 years.", REF)
    assert 0.0 <= wrong < 0.5 < right <= 1.0


class _Stub(DeepEvalBaseLLM):
    """Fills any DeepEval schema with neutral values: enough to exercise the wiring."""

    def __init__(self) -> None:
        super().__init__("schema-stub")

    def load_model(self):  # type: ignore[no-untyped-def]
        return self

    def get_model_name(self) -> str:
        return "schema-stub"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_):  # type: ignore[no-untyped-def]
        assert schema is not None, "DeepEval should always request a schema"
        values = {}
        for name, field in schema.model_fields.items():
            origin = typing.get_origin(field.annotation)
            values[name] = [] if origin in (list, typing.List) else (  # noqa: UP006
                "stub" if field.annotation is str else 1)
        return schema.model_construct(**values)

    async def a_generate(self, prompt: str, schema=None, **kw):  # type: ignore[no-untyped-def]
        return self.generate(prompt, schema, **kw)


def test_deepeval_rag_metrics_run_against_installed_version() -> None:
    metrics = DeepEvalRagMetrics(_Stub())
    ans = RagAnswer(question="hotel cap?", answer="180 GBP [travel]",
                    contexts=[make_chunk("travel", "Hotels are capped at 180 GBP.", 1)])
    for score in (metrics.faithfulness(ans), metrics.answer_relevancy(ans),
                  metrics.context_relevance(ans), metrics.contextual_recall(ans, REF)):
        assert 0.0 <= score <= 1.0


class _Schema(BaseModel):
    score: float
    reason: str


class _StructuredFake:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, schema):  # type: ignore[no-untyped-def]
        def run(_):  # type: ignore[no-untyped-def]
            self.calls += 1
            return schema(score=7, reason="ok")

        return RunnableLambda(run)


def test_langchain_adapter_caches_structured_calls(tmp_path: Path) -> None:
    fake = _StructuredFake()
    model = LangChainDeepEvalModel(fake, "m", JudgeCache(tmp_path / "c.db"))  # type: ignore[arg-type]
    a = model.generate("prompt", schema=_Schema)
    b = model.generate("prompt", schema=_Schema)
    assert a == b == _Schema(score=7, reason="ok") and fake.calls == 1
