"""The harness failure paths: cache, retries, non-retryable errors, timeouts, rate limits, cost."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import Field

from modelsel.config import Settings
from modelsel.dataset import load_split
from modelsel.harness.client import LLMCallError, LLMClient, is_transient
from modelsel.harness.ratelimit import AsyncTokenBucket
from modelsel.harness.runner import run_candidates
from modelsel.llm.fakes import FakeRateLimitError, FakeTicketModel
from modelsel.llm.registry import Catalogue, FakeProfile

MSGS = [SystemMessage(content="TASK: classify"), HumanMessage(content="TICKET:\nFrom: Tom\n\nPlease refund me.")]


class _Status(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status


class Flaky(GenericFakeChatModel):
    """Raises the queued exceptions first, then answers."""

    errors: list[Any] = Field(default_factory=list)
    calls: int = 0

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return await super()._agenerate(*args, **kwargs)


class Slow(GenericFakeChatModel):
    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(10)
        return await super()._agenerate(*args, **kwargs)


def _reply(text: str = "refund") -> Any:
    return iter(
        [AIMessage(content=text, usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500})] * 5
    )


def test_is_transient_classification() -> None:
    assert is_transient(TimeoutError())
    assert is_transient(TimeoutError())
    assert is_transient(_Status(429)) and is_transient(_Status(503))
    assert not is_transient(_Status(400)) and not is_transient(_Status(401))
    assert not is_transient(ValueError("bad"))


async def test_cost_from_usage_and_cache_hit_is_free(client: LLMClient) -> None:
    client.register_model("fake:scripted", GenericFakeChatModel(messages=_reply()))
    first = await client.complete("fake:scripted", MSGS, prompt_version="t")
    # 1000 in x $1/M + 500 out x $2/M
    assert first.cost_usd == pytest.approx(0.002)
    assert not first.cached and client.billed_usd == pytest.approx(0.002)
    second = await client.complete("fake:scripted", MSGS, prompt_version="t")
    assert second.cached and second.text == first.text
    assert client.billed_usd == pytest.approx(0.002), "cache hits must not be billed"
    third = await client.complete("fake:scripted", MSGS, prompt_version="t2")
    assert not third.cached, "a new prompt version must miss the cache"


async def test_transient_errors_are_retried(client: LLMClient) -> None:
    model = Flaky(messages=_reply(), errors=[_Status(429), TimeoutError()])
    client.register_model("fake:scripted", model)
    c = await client.complete("fake:scripted", MSGS, prompt_version="t", use_cache=False)
    assert c.attempts == 3 and model.calls == 3


async def test_non_transient_error_fails_fast(client: LLMClient) -> None:
    model = Flaky(messages=_reply(), errors=[_Status(400)])
    client.register_model("fake:scripted", model)
    with pytest.raises(LLMCallError) as info:
        await client.complete("fake:scripted", MSGS, prompt_version="t", use_cache=False)
    assert info.value.attempts == 1 and model.calls == 1


async def test_retries_exhausted_raise_with_attempt_count(client: LLMClient) -> None:
    model = Flaky(messages=_reply(), errors=[_Status(503)] * 5)
    client.register_model("fake:scripted", model)
    with pytest.raises(LLMCallError) as info:
        await client.complete("fake:scripted", MSGS, prompt_version="t", use_cache=False)
    assert info.value.attempts == 3  # settings.max_attempts


async def test_timeout_is_enforced(settings: Settings, catalogue: Catalogue) -> None:
    s = settings.model_copy(update={"request_timeout_s": 0.05, "max_attempts": 2})
    client = LLMClient(s, catalogue, None)
    client.register_model("fake:scripted", Slow(messages=_reply()))
    with pytest.raises(LLMCallError) as info:
        await client.complete("fake:scripted", MSGS, prompt_version="t")
    assert info.value.attempts == 2


async def test_fake_transient_failures_recover_through_retries(client: LLMClient) -> None:
    model = FakeTicketModel(model_id="fake:balanced-mini", profile=FakeProfile(transient_failures=2))
    client.register_model("fake:balanced-mini", model)
    c = await client.complete("fake:balanced-mini", MSGS, prompt_version="t", use_cache=False)
    assert c.attempts == 3
    assert FakeRateLimitError.status_code == 429


async def test_token_bucket_throttles() -> None:
    now = [0.0]
    bucket = AsyncTokenBucket(60, burst=1, clock=lambda: now[0])  # 1 per second
    assert await bucket.acquire() == 0.0

    async def advance() -> None:
        await asyncio.sleep(0)
        now[0] += 1.0

    waiter = asyncio.create_task(bucket.acquire())
    await advance()
    waited = await waiter
    assert waited > 0  # had to wait for the refill


async def test_failed_calls_become_error_predictions(client: LLMClient, settings: Settings) -> None:
    items = load_split(settings.data_dir, "test")[:2]
    client.register_model("fake:scripted", Flaky(messages=_reply(), errors=[_Status(401)] * 10))
    preds = await run_candidates(client, "r1", ["fake:scripted"], items)
    assert len(preds) == 6
    assert all(p.error and p.output == "" for p in preds), "a failing model is scored, not crashed"


def test_real_provider_factory_without_network(
    settings: Settings, catalogue: Catalogue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing real clients needs keys but no network; nothing is called here."""
    from modelsel.llm.registry import build_chat_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-real")
    mini = build_chat_model(catalogue.spec("openai:gpt-4o-mini"), settings)
    assert type(mini).__name__ == "ChatOpenAI" and mini.max_retries == 0, "the harness owns retries"
    judge = build_chat_model(catalogue.spec("openai:gpt-4o"), settings)
    assert judge.logprobs is True and judge.top_logprobs == 5
    sonnet = build_chat_model(catalogue.spec("anthropic:claude-sonnet-5"), settings)
    assert sonnet.temperature is None, "sonnet-5 rejects sampling parameters"
    local = build_chat_model(catalogue.spec("ollama:llama3.1:8b"), settings)
    assert type(local).__name__ == "ChatOllama"
