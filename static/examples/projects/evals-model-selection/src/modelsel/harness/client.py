"""One provider-agnostic call path: cache -> rate limit -> timeout -> retry -> usage -> cost.

Every LLM call in the project (candidates, judge, meta-judge, contamination probe)
goes through ``LLMClient.complete``. That is what makes the numbers comparable:
the same timeout, the same retry policy and the same token accounting for every model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from tenacity import AsyncRetrying, RetryError, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from modelsel.config import Settings
from modelsel.harness.cache import ResponseCache, cache_key
from modelsel.harness.costs import cost_usd
from modelsel.harness.ratelimit import AsyncTokenBucket
from modelsel.llm.registry import Catalogue, ModelSpec, build_chat_model
from modelsel.logging_setup import log_event
from modelsel.schemas import BenchmarkItem, Completion, Usage

logger = logging.getLogger(__name__)

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMCallError(RuntimeError):
    """A call that failed after all retries, or failed with a non-retryable error."""

    def __init__(self, model_id: str, message: str, attempts: int) -> None:
        super().__init__(f"{model_id}: {message} (attempts={attempts})")
        self.model_id = model_id
        self.attempts = attempts


def is_transient(exc: BaseException) -> bool:
    """Retry timeouts, connection drops, 429 and 5xx. Never retry 400/401/404: they will not heal."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and status in TRANSIENT_STATUS


def _serialise(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [{"role": m.type, "content": str(m.content)} for m in messages]


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        catalogue: Catalogue,
        cache: ResponseCache | None,
        *,
        memorised: dict[str, dict[str, BenchmarkItem]] | None = None,
    ) -> None:
        self.settings = settings
        self.catalogue = catalogue
        self.cache = cache
        self._memorised = memorised or {}
        self._models: dict[str, BaseChatModel] = {}
        self._buckets: dict[str, AsyncTokenBucket] = {}
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self.billed_usd = 0.0
        """Money actually spent in this process (cache hits cost nothing)."""

    def register_model(self, model_id: str, chat_model: BaseChatModel) -> None:
        """Use a pre-built chat model for ``model_id`` (tests, or a custom gateway client).
        The id must still exist in the catalogue: prices and rate limits come from there."""
        self.catalogue.spec(model_id)
        self._models[model_id] = chat_model

    def model(self, model_id: str) -> BaseChatModel:
        if model_id not in self._models:
            spec = self.catalogue.spec(model_id)
            self._models[model_id] = build_chat_model(spec, self.settings, memorised=self._memorised.get(model_id))
        return self._models[model_id]

    def _bucket(self, spec: ModelSpec) -> AsyncTokenBucket:
        if spec.id not in self._buckets:
            self._buckets[spec.id] = AsyncTokenBucket(spec.rpm, burst=max(1, min(spec.rpm // 10, 20)))
        return self._buckets[spec.id]

    def _bound(self, spec: ModelSpec, max_tokens: int) -> Any:
        model = self.model(spec.id)
        if spec.provider == "fake":
            return model
        if spec.provider == "ollama":
            return model.bind(num_predict=max_tokens)
        return model.bind(max_tokens=max_tokens)

    async def complete(
        self,
        model_id: str,
        messages: list[BaseMessage],
        *,
        prompt_version: str,
        max_tokens: int = 300,
        tags: list[str] | None = None,
        use_cache: bool = True,
    ) -> Completion:
        spec = self.catalogue.spec(model_id)
        key = cache_key(model_id, prompt_version, _serialise(messages), {"max_tokens": max_tokens, "temperature": 0})
        if use_cache and self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit

        runnable = self._bound(spec, max_tokens)
        config = {
            "run_name": f"{spec.provider}.{tags[0] if tags else 'call'}",
            "tags": [model_id, *(tags or [])],
            "metadata": {"model_id": model_id, "prompt_version": prompt_version},
        }
        attempts = 0
        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.settings.max_attempts),
                wait=wait_exponential_jitter(initial=self.settings.backoff_initial_s, max=self.settings.backoff_max_s),
                retry=retry_if_exception(is_transient),
                reraise=False,
            ):
                with attempt:
                    attempts += 1
                    waited = await self._bucket(spec).acquire()
                    if waited > 0.5:
                        log_event(logger, "rate_limited", model_id=model_id, waited_s=round(waited, 2))
                    started = time.perf_counter()
                    async with self._semaphore:
                        message = await asyncio.wait_for(
                            runnable.ainvoke(messages, config=config), timeout=self.settings.request_timeout_s
                        )
                    if attempts > 1:
                        log_event(logger, "retry_succeeded", model_id=model_id, attempts=attempts)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            log_event(
                logger, "llm_call_failed", logging.WARNING, model_id=model_id, attempts=attempts, error=repr(cause)
            )
            raise LLMCallError(model_id, repr(cause), attempts) from cause
        except Exception as exc:  # non-transient: surface immediately with context
            log_event(logger, "llm_call_failed", logging.WARNING, model_id=model_id, attempts=attempts, error=repr(exc))
            raise LLMCallError(model_id, repr(exc), attempts) from exc

        measured_ms = (time.perf_counter() - started) * 1000
        completion = self._to_completion(spec, message, measured_ms, attempts)
        self.billed_usd += completion.cost_usd
        if use_cache and self.cache is not None:
            self.cache.put(key, completion)
        return completion

    @staticmethod
    def _to_completion(spec: ModelSpec, message: AIMessage, measured_ms: float, attempts: int) -> Completion:
        meta = dict(message.response_metadata or {})
        um = message.usage_metadata or {}
        usage = Usage(input_tokens=int(um.get("input_tokens", 0)), output_tokens=int(um.get("output_tokens", 0)))
        simulated = meta.get("simulated_latency_ms")
        latency = float(simulated) if isinstance(simulated, (int, float)) else measured_ms
        keep = {k: meta[k] for k in ("logprobs", "model_name", "finish_reason", "stop_reason") if k in meta}
        return Completion(
            model_id=spec.id,
            text=message.text,
            usage=usage,
            latency_ms=latency,
            cost_usd=cost_usd(spec, usage),
            attempts=attempts,
            response_metadata=keep,
        )
