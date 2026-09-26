"""Provider tests: the real LangChain brain against a fake chat model, the stub and
Tavily search (with a mocked transport), and the internal index."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from research_analyst.index import InternalIndex, chunk
from research_analyst.models import ResearchPlan
from research_analyst.providers.embeddings import HashingEmbeddings
from research_analyst.providers.llm import HeuristicBrain, LangChainBrain, lexical_support
from research_analyst.providers.search import SearchError, StubWebSearch, TavilySearch


async def test_langchain_brain_parses_structured_output_from_fake_model():
    plan = {
        "title": "T",
        "objective": "O",
        "sub_questions": [
            {"id": "sq1", "question": "What is X?", "search_queries": ["x overview"]}
        ],
    }
    model = FakeListChatModel(responses=[json.dumps(plan)])
    brain = LangChainBrain(model, "gpt-4o-mini")
    result, usage = await brain.plan("What is X?", 3)
    assert isinstance(result, ResearchPlan) and result.sub_questions[0].id == "sq1"
    assert usage.llm_calls == 1 and usage.input_tokens > 0 and usage.cost_usd > 0


async def test_langchain_brain_rejects_invalid_output():
    brain = LangChainBrain(FakeListChatModel(responses=["not json"]), "gpt-4o-mini")
    with pytest.raises(Exception):  # noqa: B017 - parser raises OutputParserException
        await brain.plan("What is X?", 3)


async def test_heuristic_brain_plan_is_deterministic():
    b = HeuristicBrain()
    p1, _ = await b.plan("How are electrolyser costs expected to change by 2030?", 3)
    p2, _ = await b.plan("How are electrolyser costs expected to change by 2030?", 3)
    assert p1 == p2 and [q.id for q in p1.sub_questions] == ["sq1", "sq2", "sq3"]


def test_lexical_support_levels():
    src = "LFP cells retained 60 percent of rated capacity at minus 20 degrees Celsius."
    assert lexical_support("LFP cells retained 60 percent capacity", src).level == "fully_supported"
    assert lexical_support("Sodium-ion is cheaper than LFP everywhere", src).level == "no_support"


def test_hashing_embeddings_are_normalised_and_meaningful():
    e = HashingEmbeddings()
    a, b, c = e.embed_documents(
        ["sodium-ion battery cost", "battery cost sodium-ion", "heat pump radiator"]
    )
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert abs(dot(a, a) - 1) < 1e-9
    assert dot(a, b) > 0.99 and dot(a, c) < 0.3


def test_chunk_respects_paragraphs_and_limit():
    text = "\n\n".join(["para " * 50] * 4)
    parts = chunk(text, max_chars=600)
    assert all(len(p) <= 600 for p in parts) and len(parts) >= 2


async def test_index_build_save_load_search(settings, tmp_path):
    idx = InternalIndex.build(settings.corpus_dir / "internal", HashingEmbeddings())
    path = tmp_path / "idx.json"
    idx.save(path)
    loaded = InternalIndex.load_or_build(
        path, settings.corpus_dir / "internal", HashingEmbeddings()
    )
    hits = await loaded.search("heat pump radiator installer", k=2)
    assert hits and hits[0].origin.value == "internal"
    assert "heat pump" in hits[0].content.lower()


async def test_stub_web_search_ranks_relevant_first(settings):
    ws = StubWebSearch(settings.corpus_dir / "web.jsonl")
    res = await ws.search("electrolyser cost 2030", k=3)
    assert "electrolyser" in res[0].content.lower()
    assert ws.calls == ["electrolyser cost 2030"]


async def test_tavily_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["Authorization"] == "Bearer k"
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://a.example.gov/x",
                        "title": "X",
                        "content": "c",
                        "published_date": "2025-01-02T00:00:00",
                    }
                ]
            },
        )

    ts = TavilySearch(
        "k",
        "https://api.test/search",
        5,
        transport=httpx.MockTransport(handler),
        max_backoff_s=0.01,
    )
    res = await ts.search("q", 3)
    await ts.aclose()
    assert calls["n"] == 3 and res[0].published.year == 2025


async def test_tavily_gives_up_with_search_error():
    ts = TavilySearch(
        "k",
        "https://api.test/search",
        5,
        attempts=2,
        max_backoff_s=0.01,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    with pytest.raises(SearchError):
        await ts.search("q", 3)
    await ts.aclose()


async def test_tavily_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401)

    ts = TavilySearch(
        "k",
        "https://api.test/search",
        5,
        max_backoff_s=0.01,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SearchError):
        await ts.search("q", 3)
    await ts.aclose()
    assert calls["n"] == 1


def test_live_mode_requires_key_and_wires_real_providers(tmp_path):
    from pydantic import SecretStr, ValidationError

    from research_analyst.config import Settings
    from research_analyst.providers.llm import build_brain
    from research_analyst.providers.search import build_web_search

    with pytest.raises(ValidationError):
        Settings(_env_file=None, mode="live")
    s = Settings(_env_file=None, mode="live", OPENAI_API_KEY="sk-test-not-real")
    brain = build_brain(s)
    assert isinstance(brain, LangChainBrain) and brain.model_name == "gpt-4o-mini"
    assert type(brain._model).__name__ == "ChatOpenAI"  # constructed, never called
    assert isinstance(build_web_search(s), StubWebSearch)  # no TAVILY_API_KEY -> stub
    s2 = s.model_copy(update={"tavily_api_key": SecretStr("tvly-test")})
    assert isinstance(build_web_search(s2), TavilySearch)


@pytest.mark.parametrize(
    "schema_name",
    ["ResearchPlan", "DocGrades", "RewrittenQuery", "SectionDraft", "Critique", "SupportJudgement"],
)
def test_llm_schemas_are_strict_json_schema_compatible(schema_name):
    """OpenAI strict structured output rejects free-form objects (dict fields)."""
    from research_analyst import models

    def walk(node):
        if isinstance(node, dict):
            extra = node.get("additionalProperties")
            assert extra in (None, False), f"{schema_name} has a free-form object: {node}"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(getattr(models, schema_name).model_json_schema())
