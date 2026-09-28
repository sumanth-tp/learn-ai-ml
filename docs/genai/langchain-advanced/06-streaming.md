---
id: lc-adv-streaming
title: "Streaming — Tokens, Steps, and Custom Progress"
sidebar_label: "6 · Streaming"
sidebar_position: 6
slug: /genai/langchain-advanced/streaming
description: "The stream_mode options for both LCEL runnables and create_agent — token streaming, step updates, and custom progress events."
tags: [langchain, streaming, runnables, agents, lcel]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent), built from LangChain's [streaming docs](https://docs.langchain.com/oss/python/langchain/streaming). Every chain and agent built in the playlist used `.invoke()`, which waits for the whole answer — this chapter covers `.stream()` and friends, which do not.
:::

**In one line.** Every runnable and every agent supports `.stream()` / `.astream()`; the only decision is *what kind* of thing you want streamed — tokens, step updates, or your own custom events — via `stream_mode`.

## Why `.invoke()` alone is not enough

`.invoke()` returns once the entire answer exists. For a chatbot UI, that means staring at a blank screen for however long generation takes. Streaming sends partial output as it is produced — the same experience as ChatGPT's typing effect.

## Streaming a plain LCEL chain

The simplest case: a chain built the way [runnables part 1](/docs/genai/runnables-part-1) and [part 2](/docs/genai/runnables-part-2) built them, streamed token by token.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatAnthropic(model="claude-sonnet-4-5")
prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")
chain = prompt | model | StrOutputParser()

for chunk in chain.stream({"topic": "the monsoon"}):
    print(chunk, end="", flush=True)
```

`StrOutputParser` streams cleanly because it just passes text through. A parser that needs the *whole* output before it can parse (like `PydanticOutputParser` from the [output parsers chapter](/docs/genai/output-parsers)) cannot stream partial results the same way — it can only yield once, at the end.

## Stream modes for `create_agent`

An agent has more moving parts than a chain — model calls, tool calls, state updates — so `stream_mode` picks which of those you see:

| Mode | Streams |
|---|---|
| `"updates"` | state after each agent step (a tool call happened, a message was added) |
| `"messages"` | `(token, metadata)` pairs as any LLM node generates them |
| `"custom"` | arbitrary data your own tools push mid-execution |

Pass one mode, or a list of modes at once.

### Token streaming — `"messages"`

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    return f"It's sunny in {city}."

agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode="messages",
):
    token, metadata = chunk
    if token.content:
        print(token.content, end="", flush=True)
```

### Step-by-step progress — `"updates"`

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode="updates",
):
    for node, update in chunk.items():
        print(f"[{node}] {update['messages'][-1].content!r}")
```

This is what shows a UI "Calling get_weather..." followed by "Generating answer..." instead of one silent pause.

### Custom progress from inside a tool — `"custom"`

A tool can push its own progress messages while it runs, using `get_stream_writer`:

```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    writer = get_stream_writer()
    writer(f"Looking up weather for {city}...")
    result = f"It's sunny in {city}."
    writer(f"Done: {result}")
    return result

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode="custom",
):
    print("custom:", chunk)
```

### Combining modes

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode=["messages", "updates"],
):
    kind, data = chunk["type"], chunk["data"]
    if kind == "messages":
        token, _ = data
        if token.content:
            print(token.content, end="", flush=True)
    elif kind == "updates":
        for node, update in data.items():
            if node not in ("__start__",):
                print(f"\n[step: {node}]")
```

## Streaming reasoning/thinking tokens

For models that expose an internal reasoning trace (extended thinking), filter for it explicitly rather than assuming every token is the final answer:

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    thinking={"type": "enabled", "budget_tokens": 5000},
)
agent = create_agent(model=model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode="messages",
):
    token, _ = chunk
    for block in token.content_blocks:
        if block.get("type") == "reasoning":
            print(f"[thinking] {block['reasoning']}", end="")
        elif block.get("type") == "text":
            print(block["text"], end="", flush=True)
```

## Async

Every `.stream(...)` has an `.astream(...)` twin, for use inside an async web handler (FastAPI, for example — see the [deployment section of advanced concepts](/docs/genai/advanced-concepts)):

```python
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    stream_mode="messages",
):
    token, _ = chunk
    if token.content:
        print(token.content, end="", flush=True)
```

## Disabling streaming

Some deployments (behind certain proxies, or when a provider's streaming is unreliable) need it off entirely:

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5", streaming=False)
```

:::tip Streaming changes perceived latency, not billed cost
As [advanced concepts](/docs/genai/advanced-concepts) notes under cost and latency: you pay for the same tokens either way. Streaming only changes when the user sees them.
:::

## Checklist

- [ ] I can stream a plain LCEL chain token by token
- [ ] I can pick the right `stream_mode` for an agent: tokens, step updates, or custom
- [ ] I can push custom progress from inside a tool with `get_stream_writer`
- [ ] I know why a parser like `PydanticOutputParser` cannot stream partial output the way `StrOutputParser` can
