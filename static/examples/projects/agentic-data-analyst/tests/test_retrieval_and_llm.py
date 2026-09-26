from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from data_analyst.llm import (
    HashingEmbeddings,
    LangChainStructuredLLM,
    LLMOutputError,
    OfflineAnalystLLM,
)
from data_analyst.retrieval import SchemaIndex, SemanticCache
from data_analyst.schemas import SQLDraft, StandaloneQuestion


@pytest.fixture
def index() -> SchemaIndex:
    return SchemaIndex(HashingEmbeddings())


@pytest.mark.parametrize(
    ("question", "must_include"),
    [
        ("How many customers do we have?", {"customers"}),
        ("Revenue by product category", {"products", "order_items", "orders"}),
        ("What share of web sessions converted, by device?", {"web_sessions"}),
        ("Average ticket resolution time in hours by priority", {"support_tickets"}),
        ("Which suppliers supply the most products?", {"suppliers", "products"}),
        ("Top 5 customers by revenue", {"customers", "orders", "order_items"}),
    ],
)
def test_schema_retrieval(index: SchemaIndex, question: str, must_include: set[str]) -> None:
    tables, _ = index.retrieve(question, k=3)
    assert must_include <= set(tables)


def test_retrieval_prunes_irrelevant_tables(index: SchemaIndex) -> None:
    tables, _ = index.retrieve("Total marketing budget by channel", k=3)
    assert tables == ["marketing_campaigns"]


def test_cache_round_trip_and_threshold(tmp_path) -> None:
    cache = SemanticCache(
        tmp_path / "c.sqlite", HashingEmbeddings(), threshold=0.9, ttl_s=60, schema="v1"
    )
    cache.store("Total revenue by year", "SELECT 1 FROM orders")
    hit = cache.lookup("total revenue by year?")
    assert hit is not None and hit.sql == "SELECT 1 FROM orders" and hit.similarity > 0.99
    assert cache.lookup("Number of customers by segment") is None


def test_cache_ignores_other_schema_versions_and_expired(tmp_path) -> None:
    path = tmp_path / "c.sqlite"
    SemanticCache(path, HashingEmbeddings(), threshold=0.9, ttl_s=60, schema="old").store(
        "Total revenue by year", "SELECT 1 FROM orders"
    )
    assert (
        SemanticCache(path, HashingEmbeddings(), threshold=0.9, ttl_s=60, schema="new").lookup(
            "Total revenue by year"
        )
        is None
    )
    expired = SemanticCache(path, HashingEmbeddings(), threshold=0.9, ttl_s=0, schema="old")
    time.sleep(0.01)
    assert expired.lookup("Total revenue by year") is None


def test_cache_invalidate(tmp_path) -> None:
    cache = SemanticCache(
        tmp_path / "c.sqlite", HashingEmbeddings(), threshold=0.9, ttl_s=60, schema="v1"
    )
    cache.store("q one", "SELECT 1 FROM orders")
    cache.invalidate("SELECT 1 FROM orders")
    assert cache.size() == 0


def test_offline_rewrite_uses_history(offline_llm: OfflineAnalystLLM) -> None:
    history = [{"standalone_question": "Total revenue by year"}]
    out, usage = offline_llm.generate(
        "rewrite",
        StandaloneQuestion,
        [HumanMessage("x")],
        {"question": "now only for 2024", "history": history},
    )
    assert out.question == "Total revenue by year, now only for 2024" and out.is_follow_up
    assert usage.calls == 1 and usage.input_tokens > 0


def test_offline_attempts_follow_script() -> None:
    llm = OfflineAnalystLLM({"q": ["SELECT bad", "SELECT good"]})
    first, _ = llm.generate("sql", SQLDraft, [], {"question": "q", "attempt": 1})
    second, _ = llm.generate("sql", SQLDraft, [], {"question": "Q?", "attempt": 2})
    third, _ = llm.generate("sql", SQLDraft, [], {"question": "q", "attempt": 9})
    assert (first.sql, second.sql, third.sql) == ("SELECT bad", "SELECT good", "SELECT good")


def test_offline_unscripted_question_is_safe() -> None:
    out, _ = OfflineAnalystLLM({}).generate(
        "sql", SQLDraft, [], {"question": "anything", "tables": ["products"]}
    )
    assert out.sql == "SELECT COUNT(*) AS row_count FROM products"


class FakeStructuredChat:
    """Minimal stand-in for a chat model's with_structured_output(include_raw=True)."""

    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs

    def with_structured_output(self, schema: type, include_raw: bool = False) -> RunnableLambda:
        return RunnableLambda(lambda _messages, config=None: self.outputs.pop(0))


def test_langchain_adapter_reads_usage() -> None:
    raw = AIMessage(
        "", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    )
    chat = FakeStructuredChat(
        [{"raw": raw, "parsed": SQLDraft(sql="SELECT 1", explanation="x"), "parsing_error": None}]
    )
    out, usage = LangChainStructuredLLM(chat).generate("sql", SQLDraft, [], {})  # type: ignore[arg-type]
    assert out.sql == "SELECT 1" and (usage.input_tokens, usage.output_tokens) == (120, 30)


def test_langchain_adapter_retries_then_raises_on_parse_errors() -> None:
    bad = {"raw": AIMessage("not json"), "parsed": None, "parsing_error": ValueError("bad")}
    chat = FakeStructuredChat([dict(bad), dict(bad)])
    with pytest.raises(LLMOutputError):
        LangChainStructuredLLM(chat, parse_retries=1).generate("sql", SQLDraft, [], {})  # type: ignore[arg-type]
    assert chat.outputs == []
