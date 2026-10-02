---
id: lc-adv-caching
title: "Caching — Exact-Match and Semantic Response Caching"
sidebar_label: "9 · Caching"
sidebar_position: 9
slug: /genai/langchain-advanced/caching
description: "set_llm_cache for exact-match response caching, and an embedding-based semantic cache for near-duplicate queries, with runnable code for both."
tags: [langchain, caching, cost, performance]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent). [Advanced concepts](/docs/genai/advanced-concepts) mentions prompt caching and semantic caching in one line each under cost control; this chapter is the code.
:::

**In one line.** `set_llm_cache` makes LangChain skip the model call entirely when it sees an identical prompt again — turning a paid, slow API call into a free, instant lookup.

## Exact-match caching

Every model built from `ChatAnthropic`, `ChatOpenAI`, and the rest reads from a **global cache** if one is set. The simplest is in-process and disappears when the program exits — good for development and for tests that call the same prompt repeatedly:

```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

model = ChatAnthropic(model="claude-sonnet-4-5")

model.invoke("What is the capital of France?")  # hits the API
model.invoke("What is the capital of France?")  # cache hit, no API call, instant
```

A cache that survives restarts needs a real backend. SQLite is the easiest local option:

```python
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

Redis scales further, and is the natural choice once more than one process needs to share a cache:

```python
from langchain_community.cache import RedisCache
from redis import Redis

set_llm_cache(RedisCache(redis_=Redis()))
```

Exact-match caching only helps when the *exact same string* is sent twice — including whitespace and punctuation. It is a strong win for repeated system prompts, evaluation runs, and demo scripts; it does nothing for two users asking the same thing in different words.

## Semantic caching — matching near-duplicate queries

"What's the capital of France?" and "capital of France?" are different strings but the same question. A semantic cache stores past query embeddings alongside their answers, and on a new query, checks whether anything sufficiently similar was already asked — the same nearest-neighbour idea as the [vector stores chapter](/docs/genai/vector-stores), applied to caching instead of retrieval.

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

set_llm_cache(
    RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=OpenAIEmbeddings(),
        score_threshold=0.2,   # lower = stricter match required
    )
)

model.invoke("What is the capital of France?")   # miss — calls the API, caches the answer
model.invoke("Tell me France's capital city")     # semantic hit — same cached answer, no API call
```

`score_threshold` is the only knob that matters here, and it trades directly against correctness: too loose, and unrelated questions return a stale cached answer; too strict, and you rarely get a hit at all. Tune it against real traffic, not a hunch.

:::warning A semantic cache can return a wrong answer
Exact-match caching can only ever return an answer to the *exact* question asked. A semantic cache can return an answer to a *similar-sounding but different* question if the threshold is too loose — verify against real query pairs before trusting it in production.
:::

## Per-call cache control

Caching is enabled globally by `set_llm_cache`, but any single call can opt out — useful for a request that must always hit the live model, such as one involving current data:

```python
model.invoke("What is the capital of France?")                      # uses the cache
model.invoke("What time is it right now?", config={"cache": False})  # always live
```

## When caching pays for itself

| Situation | Worth caching? |
|---|---|
| Repeated system prompt across many requests | Yes — exact-match, large win |
| A FAQ-style support bot | Yes — semantic, catches paraphrases |
| Evaluation suite run on every CI build | Yes — exact-match, huge speed-up |
| Personalised answers ("what did I ask yesterday?") | No — each answer is meant to differ |
| Anything time-sensitive (weather, stock price, "now") | No — a cache hit would return stale data |

## Checklist

- [ ] I can enable exact-match caching with `set_llm_cache` and a real backend (SQLite/Redis)
- [ ] I can explain why exact-match caching misses two differently-worded versions of the same question
- [ ] I can set up a semantic cache and explain what `score_threshold` trades off
- [ ] I can identify which of my own use cases should never be cached

## Summary table

| Topic | Summary |
| --- | --- |
| Exact cache | Reuse a response for the same prompt and model configuration. |
| Semantic cache | Match similar queries using embeddings and a chosen similarity threshold. |
| Policy | Cache only where staleness, privacy and context rules permit reuse. |
| Exact key | An exact cache reuses a result when the prompt and relevant model settings match. |
| Semantic match | A semantic cache embeds a query and searches for close earlier requests. |
| Threshold | A strict similarity threshold reduces false reuse but lowers the cache hit rate. |
| Scope | Keep personalised, time-sensitive or sensitive answers out of shared caches. |
