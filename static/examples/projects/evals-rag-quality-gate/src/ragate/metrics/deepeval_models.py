"""DeepEval model adapters.

DeepEval metrics call ``model.generate(prompt, schema=SomePydanticModel)``. The adapter
below lets any LangChain chat model serve as DeepEval's judge, adds the same SQLite
cache as our own judge, and pins temperature. The stub is the offline stand-in.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from ragate.judge.cache import JudgeCache
from ragate.text import coverage


class LangChainDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self, model: BaseChatModel, model_name: str, cache: JudgeCache | None) -> None:
        self._lc = model
        self._name = model_name
        self._cache = cache
        super().__init__(model_name)

    def load_model(self) -> BaseChatModel:
        return self._lc

    def get_model_name(self) -> str:
        return self._name

    def _key(self, prompt: str, schema: type[BaseModel] | None) -> str:
        raw = json.dumps([self._name, schema.__name__ if schema else None, prompt])
        return "deepeval:" + hashlib.sha256(raw.encode()).hexdigest()

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        key = self._key(prompt, schema)
        if self._cache and (hit := self._cache.get_raw(key)) is not None:
            return schema.model_validate_json(hit) if schema else json.loads(hit)
        if schema is not None:
            result: Any = self._lc.with_structured_output(schema).invoke(prompt)
            if not isinstance(result, schema):
                result = schema.model_validate(result)
            stored = result.model_dump_json()
        else:
            result = str(self._lc.invoke(prompt).content)
            stored = json.dumps(result)
        if self._cache:
            self._cache.put_raw(key, stored)
        return result

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None,
                         **kw: Any) -> Any:
        return self.generate(prompt, schema, **kw)


# The item being scored, so the offline stub can read actual/expected directly
# instead of parsing DeepEval's prompt text (which changes between releases).
CURRENT_CASE: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "current_geval_case", default=None
)


class StubGEvalModel(DeepEvalBaseLLM):
    """Deterministic offline G-Eval judge.

    Mirrors the rubric's intent: reward covering the expected facts (recall), and
    penalise padding lightly (precision), so extra correct detail costs little.
    """

    def __init__(self) -> None:
        super().__init__("stub-geval")

    def load_model(self) -> StubGEvalModel:
        return self

    def get_model_name(self) -> str:
        return "stub-geval"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any) -> Any:
        from deepeval.metrics.g_eval.schema import ReasonScore, Steps

        if schema is Steps:
            return Steps(steps=["Compare the facts in the actual and expected outputs."])
        case = CURRENT_CASE.get()
        if schema is ReasonScore and case is not None:
            actual, expected = (re.sub(r"\[[a-z0-9-]+\]", "", t) for t in case)
            recall = coverage(expected, actual)
            precision = coverage(actual, expected)
            value = 0.8 * recall + 0.2 * precision
            return ReasonScore(score=round(value * 10),
                               reason=f"fact recall {recall:.2f}, precision {precision:.2f}")
        raise RuntimeError(f"stub G-Eval model cannot answer schema {schema}")

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None,
                         **kw: Any) -> Any:
        return self.generate(prompt, schema, **kw)
