---
id: lc-adv-runnable-resilience
title: "Runnable Resilience — Retries, Fallbacks, and Configurable Chains"
sidebar_label: "7 · Runnable resilience"
sidebar_position: 7
slug: /genai/langchain-advanced/runnable-resilience
description: "with_retry, with_fallbacks, and configurable_fields/configurable_alternatives — making an LCEL chain survive a bad response or a down provider, and swap models at call time."
tags: [langchain, lcel, runnables, retries, fallbacks]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent). These methods live on every `Runnable`, so they apply directly to the chains built in [runnables part 1](/docs/genai/runnables-part-1) and [part 2](/docs/genai/runnables-part-2) — the playlist covers what runnables *are*, not how to make them fault-tolerant.
:::

**In one line.** `.with_retry()`, `.with_fallbacks()`, and `.configurable_fields()` are methods on every `Runnable` that add resilience or runtime flexibility without changing how the chain is built or piped together.

## `.with_retry()` — survive a transient failure

Every model call can fail: a rate limit, a network blip, a provider outage. Retrying with backoff turns most of those from a hard failure into a slower success.

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5")

resilient_model = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

chain = prompt | resilient_model | StrOutputParser()
```

`.with_retry()` works on any runnable, not only a model — put it on the whole chain if a downstream step (a tool call, a parser) is what's flaky:

```python
resilient_chain = chain.with_retry(stop_after_attempt=3)
```

## `.with_fallbacks()` — survive a down provider

Retrying the same model does not help when the provider itself is down. `.with_fallbacks()` tries a list of alternative runnables, in order, the moment the primary one raises:

```python
from langchain_openai import ChatOpenAI

primary = ChatAnthropic(model="claude-sonnet-4-5")
backup = ChatOpenAI(model="gpt-4.1-mini")

model_with_fallback = primary.with_fallbacks([backup])

chain = prompt | model_with_fallback | StrOutputParser()
result = chain.invoke({"topic": "monsoon"})  # tries Anthropic, then OpenAI if it fails
```

Fallbacks stack: chain several cheaper or smaller models behind a strong primary, in decreasing preference order.

```python
model_with_fallbacks = primary.with_fallbacks([backup, ChatAnthropic(model="claude-haiku-4-5")])
```

:::tip Combine retry and fallback
`model.with_retry(stop_after_attempt=2).with_fallbacks([backup])` retries the primary briefly for transient errors, then falls back only if it is still failing after those retries — cheaper than falling back on the very first blip.
:::

## `.configurable_fields()` — change a parameter at call time

A chain is normally built once, with fixed parameters. `configurable_fields` marks specific parameters as changeable *per invocation*, without rebuilding the chain:

```python
from langchain_core.runnables import ConfigurableField

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="Higher = more creative, lower = more deterministic",
    )
)

chain = prompt | model | StrOutputParser()

# Default temperature (0)
chain.invoke({"topic": "monsoon"})

# Override per call
chain.invoke(
    {"topic": "monsoon"},
    config={"configurable": {"temperature": 0.9}},
)
```

## `.configurable_alternatives()` — swap the whole model at call time

Where `configurable_fields` tweaks a parameter, `configurable_alternatives` swaps the entire runnable — useful for routing between providers or model tiers per request, without an `if`/`else` around chain construction:

```python
from langchain_core.runnables import ConfigurableField

model = ChatAnthropic(model="claude-sonnet-4-5").configurable_alternatives(
    ConfigurableField(id="model"),
    default_key="anthropic",
    openai=ChatOpenAI(model="gpt-4.1-mini"),
    local=ChatOllama(model="llama3.1"),
)

chain = prompt | model | StrOutputParser()

chain.invoke({"topic": "monsoon"})  # uses claude-sonnet-4-5, the default
chain.invoke({"topic": "monsoon"}, config={"configurable": {"model": "openai"}})
chain.invoke({"topic": "monsoon"}, config={"configurable": {"model": "local"}})
```

This is the LCEL-native version of the "model routing" idea from [advanced concepts](/docs/genai/advanced-concepts) — cheap requests to a small model, hard ones to a frontier model — expressed as configuration rather than branching code.

## Putting it together

A production-shaped chain typically combines all three:

```python
model = (
    ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    .configurable_fields(temperature=ConfigurableField(id="temperature"))
    .with_retry(stop_after_attempt=3)
    .with_fallbacks([ChatOpenAI(model="gpt-4.1-mini")])
)

chain = prompt | model | StrOutputParser()
```

## Checklist

- [ ] I can add retry-with-backoff to a model or a chain with `.with_retry()`
- [ ] I can add a backup provider with `.with_fallbacks()`, and explain why retry alone doesn't cover a provider outage
- [ ] I can make one parameter (like temperature) overridable per call with `configurable_fields`
- [ ] I can swap the entire model per call with `configurable_alternatives`, and see how it relates to model routing
