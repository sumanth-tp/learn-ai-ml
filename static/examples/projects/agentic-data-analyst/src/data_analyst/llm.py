"""Model access behind two small interfaces, each with a real and an offline implementation.

``StructuredLLM`` returns a validated Pydantic object plus token usage. The real one
wraps any LangChain chat model (OpenAI by default, swappable by config); the offline
one is deterministic and scripted, so tests and CI need no keys and no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any, Protocol, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from data_analyst.config import Settings
from data_analyst.logging_setup import get_logger
from data_analyst.schemas import (
    ChartCode,
    Interpretation,
    QueryPlan,
    SQLDraft,
    StandaloneQuestion,
)

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )


class LLMOutputError(Exception):
    """The model did not return output matching the schema."""


class StructuredLLM(Protocol):
    def generate(
        self,
        step: str,
        schema: type[T],
        messages: Sequence[BaseMessage],
        hints: Mapping[str, Any],
    ) -> tuple[T, Usage]:
        """Return ``schema`` filled in by the model.

        ``hints`` carries the structured inputs the prompt was built from. The real model
        ignores them (it reads the prompt); the offline model reads them instead of
        parsing prose.
        """
        ...


class LangChainStructuredLLM:
    """Any LangChain chat model with native structured output."""

    def __init__(self, model: BaseChatModel, parse_retries: int = 1) -> None:
        self.model = model
        self.parse_retries = parse_retries

    def generate(
        self,
        step: str,
        schema: type[T],
        messages: Sequence[BaseMessage],
        hints: Mapping[str, Any],
    ) -> tuple[T, Usage]:
        runnable = self.model.with_structured_output(schema, include_raw=True)
        last_error: Exception | None = None
        for attempt in range(self.parse_retries + 1):
            out = runnable.invoke(list(messages), config={"run_name": f"llm:{step}"})
            parsed = out.get("parsed")
            raw = out.get("raw")
            meta = getattr(raw, "usage_metadata", None) or {}
            usage = Usage(
                input_tokens=int(meta.get("input_tokens", 0)),
                output_tokens=int(meta.get("output_tokens", 0)),
                calls=1,
            )
            if isinstance(parsed, schema):
                return parsed, usage
            last_error = out.get("parsing_error") or ValueError("no parsed output")
            log.warning("structured_output_parse_failed", step=step, attempt=attempt)
            time.sleep(0.5 * (2**attempt))
        raise LLMOutputError(f"{step}: {last_error}")


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Provider-agnostic: ANALYST_LLM_PROVIDER=anthropic / ollama / ... also works once the
    matching langchain-<provider> package is installed."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_mode == "offline":
        return HashingEmbeddings()
    from langchain.embeddings import init_embeddings

    return init_embeddings(settings.embedding_model, provider=settings.embedding_provider)


def build_llm(settings: Settings) -> StructuredLLM:
    if settings.llm_mode == "offline":
        return OfflineAnalystLLM.from_package()
    return LangChainStructuredLLM(build_chat_model(settings))


# --------------------------------------------------------------------------- offline


def normalise(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", question.lower())).strip()


FOLLOW_UP_CUES = ("now ", "only ", "and ", "what about", "same ", "instead", "just ", "but ")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OfflineAnalystLLM:
    """Deterministic stand-in for the model.

    SQL comes from a script keyed by the normalised standalone question. A script entry
    is a list of attempts, so a scripted first attempt can be wrong on purpose to
    exercise the self-correction loop. Unscripted questions get a safe row-count query
    over the most relevant table and say so in the explanation.
    """

    def __init__(self, script: Mapping[str, list[str]]) -> None:
        self.script = {normalise(k): v for k, v in script.items()}

    @classmethod
    def from_package(cls) -> OfflineAnalystLLM:
        text = resources.files("data_analyst").joinpath("offline_script.json").read_text()
        return cls(json.loads(text))

    def generate(
        self,
        step: str,
        schema: type[T],
        messages: Sequence[BaseMessage],
        hints: Mapping[str, Any],
    ) -> tuple[T, Usage]:
        handler = getattr(self, f"_{step}")
        result: BaseModel = handler(hints)
        if not isinstance(result, schema):
            raise LLMOutputError(f"offline model returned {type(result).__name__} for {step}")
        prompt_text = "".join(str(m.content) for m in messages)
        usage = Usage(
            input_tokens=_estimate_tokens(prompt_text),
            output_tokens=_estimate_tokens(result.model_dump_json()),
            calls=1,
        )
        return result, usage

    def _rewrite(self, h: Mapping[str, Any]) -> StandaloneQuestion:
        question: str = h["question"]
        history: list[dict[str, Any]] = h.get("history", [])
        lowered = question.lower().strip()
        if history and lowered.startswith(FOLLOW_UP_CUES):
            previous = history[-1]["standalone_question"].rstrip("?. ")
            return StandaloneQuestion(question=f"{previous}, {question}", is_follow_up=True)
        return StandaloneQuestion(question=question, is_follow_up=False)

    def _plan(self, h: Mapping[str, Any]) -> QueryPlan:
        tables = list(h.get("tables", []))
        return QueryPlan(
            tables=tables,
            steps=[f"read {', '.join(tables)}", "apply filters", "aggregate", "order"],
            metric_definition="as defined in the table descriptions",
        )

    def _sql(self, h: Mapping[str, Any]) -> SQLDraft:
        attempts = self.script.get(normalise(h["question"]))
        if attempts:
            idx = min(int(h.get("attempt", 1)) - 1, len(attempts) - 1)
            return SQLDraft(sql=attempts[idx], explanation="scripted offline answer")
        table = (h.get("tables") or ["orders"])[0]
        return SQLDraft(
            sql=f"SELECT COUNT(*) AS row_count FROM {table}",
            explanation=f"offline model has no script for this question; counted rows of {table}",
        )

    def _interpret(self, h: Mapping[str, Any]) -> Interpretation:
        columns: list[str] = h["columns"]
        rows: list[list[Any]] = h["rows"]
        total: int = h["total"]
        if not rows:
            return Interpretation(answer="The query returned no rows.", chart_recommended=False)
        if total == 1:
            pairs = ", ".join(f"{c} = {v}" for c, v in zip(columns, rows[0], strict=False))
            return Interpretation(answer=f"Result: {pairs}.", chart_recommended=False)
        first = ", ".join(f"{c} = {v}" for c, v in zip(columns, rows[0], strict=False))
        last = rows[0][-1]
        numeric = isinstance(last, int | float) and not isinstance(last, bool)
        return Interpretation(
            answer=f"The query returned {total} rows. The first row is {first}.",
            chart_recommended=numeric and len(columns) >= 2,
        )

    def _chart(self, h: Mapping[str, Any]) -> ChartCode:
        columns: list[str] = h["columns"]
        x, y = columns[0], columns[-1]
        code = (
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            f"ax.bar(df[{x!r}].astype(str), df[{y!r}])\n"
            f"ax.set_xlabel({x!r})\n"
            f"ax.set_ylabel({y!r})\n"
            "ax.tick_params(axis='x', rotation=45)\n"
            f"ax.set_title({h['question'][:60]!r})\n"
            "fig.tight_layout()\n"
        )
        return ChartCode(code=code, title=h["question"][:60])


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "many",
        "me",
        "much",
        "of",
        "on",
        "or",
        "per",
        "show",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "which",
        "who",
        "with",
        "our",
        "us",
        "all",
        "each",
        "every",
    ]
)


class HashingEmbeddings(Embeddings):
    """Deterministic lexical embeddings: hashed word unigrams, crude stemming, L2 norm.

    Not semantic, but similar wording gives similar vectors, which is what schema
    retrieval and the semantic cache need in tests.
    """

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        return [
            w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS
        ]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dims] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
