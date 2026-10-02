---
id: lc-adv-memory
title: "Memory — Short-Term Threads and Long-Term Storage"
sidebar_label: "4 · Memory"
sidebar_position: 4
slug: /genai/langchain-advanced/memory
description: "Checkpointers and threads for within-conversation memory, and the cross-session Store for long-term memory, with real code for both."
tags: [langchain, agents, memory, langgraph, checkpointer]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent), built from LangChain's [short-term](https://docs.langchain.com/oss/python/langchain/short-term-memory) and [long-term](https://docs.langchain.com/oss/python/langchain/long-term-memory) memory docs. The [document loaders chapter](/docs/genai/document-loaders) already noted that "memory is moving into LangGraph" — this is that move, with code. The [advanced concepts chapter](/docs/genai/advanced-concepts) covers memory *strategies* conceptually; this chapter is the API that implements them.
:::

**In one line.** Short-term memory keeps one conversation's history alive across turns via a **checkpointer** and a **thread ID**; long-term memory keeps facts alive *across* conversations via a **store**.

## Short-term memory: checkpointer + thread

A `checkpointer` saves the agent's state after every step. A `thread_id` says which conversation you are continuing — the same idea as an email thread grouping messages.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread = {"configurable": {"thread_id": "conversation-1"}}

agent.invoke({"messages": [{"role": "user", "content": "Hi! My name is Priya."}]}, thread)
result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, thread)
print(result["messages"][-1].content)  # remembers Priya, same thread_id
```

`InMemorySaver` disappears when the process exits — fine for a demo, useless in production. Swap in a real backend without touching the agent code:

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=[get_user_info], checkpointer=checkpointer)
```

## Extending state beyond messages

The default state is just `{"messages": [...]}`. Add fields with a custom `state_schema`:

```python
from langchain.agents import create_agent, AgentState

class CustomState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_info],
    state_schema=CustomState,
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"},
    },
    {"configurable": {"thread_id": "conversation-1"}},
)
```

Tools read that extra state through `ToolRuntime`, and can write back to it with a `Command`:

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up the current user."""
    return f"user_id is {runtime.state['user_id']}"

@tool
def remember_preference(key: str, value: str, runtime: ToolRuntime) -> Command:
    """Save a user preference."""
    prefs = {**runtime.state.get("preferences", {}), key: value}
    return Command(update={"preferences": prefs})
```

## Keeping conversations bounded

Every message stays in state and, therefore, in the model's context window. Left unmanaged, a long thread eventually blows the context budget. Three fixes, in order of how much they lose:

**Trim** — keep the system message plus the most recent N, drop the rest:

```python
from typing import Any
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langgraph.runtime import Runtime

@before_model
def trim_history(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= 12:
        return None
    kept = [messages[0], *messages[-10:]]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}

agent = create_agent(model="...", tools=[...], middleware=[trim_history], checkpointer=InMemorySaver())
```

**Summarize** — replace old turns with an LLM-written summary instead of dropping them outright:

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="anthropic:claude-haiku-4-5",   # a cheaper model for the summary itself
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

**Delete outright** — for when history genuinely no longer matters:

```python
def clear_thread(state: AgentState) -> dict:
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

This is the coded version of the buffer / window / summary trade-off table in [advanced concepts](/docs/genai/advanced-concepts) — trim ≈ window, `SummarizationMiddleware` ≈ summary.

## Long-term memory: the Store

A checkpointer remembers *this thread*. A `Store` remembers facts *across every thread for a given user* — what the playlist's memory table calls "entity" memory. Data lives under a `(namespace, key) → value` structure, namespace acting like a folder:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("users", "user_123"), "profile", {"name": "Priya", "tz": "Asia/Kolkata"})

item = store.get(("users", "user_123"), "profile")
print(item.value)  # {'name': 'Priya', 'tz': 'Asia/Kolkata'}
```

Wire it into an agent, and tools can read and write it through `runtime.store`:

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up saved facts about the current user."""
    info = runtime.store.get(("users",), runtime.context["user_id"])
    return str(info.value) if info else "No saved info yet."

@tool
def save_user_info(fact: str, runtime: ToolRuntime) -> str:
    """Save a fact about the current user for future conversations."""
    runtime.store.put(("users",), runtime.context["user_id"], {"fact": fact})
    return "Saved."

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_info, save_user_info],
    store=store,
)
```

Production stores back onto Postgres the same way checkpointers do (`from langgraph.store.postgres import PostgresStore`), and can be given an embedding function to support **semantic search over memories** rather than only exact-key lookup:

```python
from langgraph.store.base import IndexConfig
from langchain_anthropic import AnthropicEmbeddings  # any embeddings model works

store = InMemoryStore(index=IndexConfig(embed=AnthropicEmbeddings(), dims=1536))
results = store.search(("users", "user_123"), query="what language does this user prefer?")
```

## Short-term vs long-term, side by side

| | Short-term (checkpointer) | Long-term (store) |
|---|---|---|
| Scope | one thread | across every thread |
| Backing | `InMemorySaver`, `PostgresSaver`, ... | `InMemoryStore`, `PostgresStore`, ... |
| Access in tools | `runtime.state` | `runtime.store` |
| Typical content | this conversation's messages | user profile, preferences, durable facts |
| What it replaces | `ConversationBufferMemory` (legacy) | the "entity memory" row of the old memory table |

## Checklist

- [ ] I can start and continue a conversation using `checkpointer` + `thread_id`
- [ ] I can extend agent state with a custom `state_schema` and read/write it from a tool
- [ ] I can trim, summarize, or clear a growing message history with middleware
- [ ] I can tell short-term (checkpointer) and long-term (store) memory apart, and pick the right one

## Summary table

| Topic | Summary |
| --- | --- |
| Conversation | A checkpointer and thread ID retain short-term conversation state. |
| Long-term data | A store keeps information across threads or sessions. |
| Boundaries | Trim, summarise or clear growing history to control context size. |
| Short-term state | A checkpointer and thread ID preserve messages across turns in one conversation. |
| Custom state | A state schema can hold application fields beyond messages. |
| Long-term store | A store keeps selected facts across threads and sessions. |
| Growth control | Trim, summarise or clear history to keep model context bounded. |
