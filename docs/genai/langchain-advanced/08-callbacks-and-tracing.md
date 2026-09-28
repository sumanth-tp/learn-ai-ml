---
id: lc-adv-callbacks-and-tracing
title: "Callbacks and Tracing — Watching a Chain from the Inside"
sidebar_label: "8 · Callbacks & tracing"
sidebar_position: 8
slug: /genai/langchain-advanced/callbacks-and-tracing
description: "Writing a custom callback handler to observe every step of a chain or agent, and turning on LangSmith tracing with three environment variables."
tags: [langchain, callbacks, langsmith, tracing, observability]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent). [Advanced concepts](/docs/genai/advanced-concepts) mentions LangSmith tracing in one line under evaluation; this chapter is the code — both LangSmith and rolling your own callback handler.
:::

**In one line.** A callback handler is an object with `on_*` methods that LangChain calls automatically as a chain runs — every LLM call, every tool call, every chain step — which is how both custom logging and LangSmith's tracing actually work under the hood.

## Why not just `print()`

You could sprinkle `print()` statements through custom code, but LangChain's own building blocks — `ChatAnthropic`, `RunnableSequence`, a `@tool`-decorated function — don't expose hooks for you to edit. Callbacks solve that: attach an observer once, and it fires for every step, in every chain, without touching that chain's code.

## Writing a custom callback handler

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class TimingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        print(f"[llm start] {len(prompts)} prompt(s)")

    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage = response.llm_output.get("usage", {}) if response.llm_output else {}
        print(f"[llm end] tokens used: {usage}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        print(f"[tool start] {serialized.get('name')}({input_str})")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"[tool end] {output}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"[error] {error}")
```

Attach it at invocation time — no need to rebuild the chain:

```python
chain = prompt | model | StrOutputParser()
chain.invoke({"topic": "monsoon"}, config={"callbacks": [TimingHandler()]})
```

Or attach it once, to every future call, when constructing a runnable:

```python
model = ChatAnthropic(model="claude-sonnet-4-5", callbacks=[TimingHandler()])
```

The same handler works on an agent built with `create_agent` — callbacks fire for model calls and tool calls inside the agent loop exactly as they do for a plain chain:

```python
agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[get_weather])
agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in Chennai?"}]},
    config={"callbacks": [TimingHandler()]},
)
```

## The events you can hook

| Method | Fires when |
|---|---|
| `on_llm_start` / `on_llm_end` / `on_llm_error` | a model call begins / completes / fails |
| `on_chain_start` / `on_chain_end` / `on_chain_error` | any runnable in a chain runs |
| `on_tool_start` / `on_tool_end` / `on_tool_error` | a tool executes |
| `on_retriever_start` / `on_retriever_end` | a retriever from the [retrievers chapter](/docs/genai/retrievers) runs |
| `on_agent_action` / `on_agent_finish` | an agent decides to act, or is done |

Only implement the ones you need; unimplemented hooks default to doing nothing.

## Turning on LangSmith without writing a handler

LangSmith is LangChain's own tracing platform — the same events above, sent to a hosted UI, with no handler code required:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="ls__..."
export LANGSMITH_PROJECT="genai-course"
```

```python
# no code changes needed — every chain and agent invocation now appears in LangSmith
chain.invoke({"topic": "monsoon"})
```

Each trace shows the full call tree: the prompt actually sent, retrieved chunks if any, every tool call and its result, token counts, latency, and cost per step — the debugging view [advanced concepts](/docs/genai/advanced-concepts) says you will need "the first time someone reports a bad answer."

## Local debug output without LangSmith

For a quick look during development, LangChain can print every step to the console directly, with no account or API key needed:

```python
from langchain.globals import set_debug, set_verbose

set_debug(True)     # full internal detail — prompts, raw responses, every step
# or, less noisy:
set_verbose(True)   # just the high-level steps
```

:::warning Turn debug mode off again
`set_debug(True)` is global and very loud — it also prints full prompts and responses, which may include sensitive data. Fine for local development; never leave it on in a deployed service.
:::

## Checklist

- [ ] I can write a `BaseCallbackHandler` and attach it to a chain or an agent
- [ ] I can name three `on_*` hooks and what triggers them
- [ ] I can turn on LangSmith tracing with three environment variables, no code changes
- [ ] I can turn on local `set_debug`/`set_verbose` output, and know why to turn it back off
