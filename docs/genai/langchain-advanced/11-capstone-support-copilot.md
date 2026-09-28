---
id: lc-adv-capstone-support-copilot
title: "Capstone: A Production Support Copilot"
sidebar_label: "11 · Capstone project"
sidebar_position: 11
slug: /genai/langchain-advanced/capstone-support-copilot
description: "One project that exercises every chapter in this subfolder — create_agent, middleware, human-in-the-loop, memory, multi-agent supervisors, retrieval, resilience, streaming, tracing and caching — building a multi-specialist customer support agent."
tags: [langchain, capstone, project, agents, multi-agent, rag, memory, langgraph]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent). This capstone plays the same role as the course's own [Research Copilot capstone](/docs/genai/capstone) — one project, every chapter — but exercises the ten chapters in *this* subfolder instead: the `create_agent` architecture, middleware, human-in-the-loop, memory, multi-agent, streaming, resilience, tracing, caching, and advanced retrievers.
:::

**In one line.** Build **Aegis**, a customer-support copilot for a fictional SaaS product: a supervisor routes each conversation to a billing or technical specialist, each specialist answers from a real knowledge base, sensitive actions pause for a human, the conversation remembers the customer across sessions, and every request is cached, retried, and traced.

## What you are building

```mermaid
flowchart TB
    U(["Customer message"]) --> SUP["Supervisor agent<br/>(create_agent)"]
    SUP -->|billing question| BA["Billing subagent"]
    SUP -->|technical question| TA["Tech-support subagent"]
    BA --> KB1[("Billing docs<br/>Ensemble retriever")]
    TA --> KB2[("Product docs<br/>Parent-document retriever")]
    BA -->|refund > $50| HITL["Human-in-the-loop<br/>approval gate"]
    HITL --> DB[("Order database")]
    SUP <--> STORE[("Long-term store<br/>customer profile")]
    SUP <--> CKPT[("Checkpointer<br/>this conversation's thread")]
    SUP --> STREAM(["Streamed reply"])
```

Every box in that diagram maps to one chapter you have already read:

| Box | Chapter |
|---|---|
| Supervisor + subagents | [`create_agent`](/docs/genai/langchain-advanced/create-agent), [multi-agent systems](/docs/genai/langchain-advanced/multi-agent-systems) |
| Ensemble / parent-document retrievers | [advanced retrievers](/docs/genai/langchain-advanced/advanced-retrievers) |
| Human-in-the-loop approval gate | [human-in-the-loop](/docs/genai/langchain-advanced/human-in-the-loop) |
| Long-term store + checkpointer | [memory](/docs/genai/langchain-advanced/memory) |
| Streamed reply | [streaming](/docs/genai/langchain-advanced/streaming) |
| Retries, fallback model | [runnable resilience](/docs/genai/langchain-advanced/runnable-resilience) |
| PII scrubbing, summarization | [middleware](/docs/genai/langchain-advanced/middleware) |
| Tracing + cost logging | [callbacks & tracing](/docs/genai/langchain-advanced/callbacks-and-tracing) |
| FAQ cache | [caching](/docs/genai/langchain-advanced/caching) |
| FastAPI + Docker serving | Milestone 10 — deployment, below |

## Prerequisites

All ten preceding chapters, plus the playlist's [tools](/docs/genai/tools), [vector stores](/docs/genai/vector-stores), and [text splitters](/docs/genai/text-splitters) chapters — this project assumes you can already build a tool and a vector store, and focuses on the parts that are new here.

```bash
pip install langchain langchain-classic langchain-anthropic langchain-openai \
            langgraph langchain-chroma rank_bm25
```

## Milestone 1 — The knowledge base, indexed two ways

Aegis answers from two document sets: product documentation (long, structured — needs parent-document retrieval so an answer keeps its surrounding context) and a billing FAQ (short, keyword-heavy — needs hybrid search so exact terms like invoice numbers are not lost to paraphrase).

```python
from langchain_classic.retrievers import ParentDocumentRetriever, BM25Retriever, EnsembleRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Product docs: parent-document retrieval — small chunks to search, full sections to read
product_retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings(), collection_name="product_docs"),
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
product_retriever.add_documents(product_docs)  # loaded via DirectoryLoader, from the loaders chapter

# Billing FAQ: hybrid search — BM25 catches exact invoice/plan codes, vectors catch paraphrase
billing_vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), collection_name="billing_faq")
billing_vectorstore.add_documents(billing_faq_docs)
billing_bm25 = BM25Retriever.from_documents(billing_faq_docs)
billing_bm25.k = 5

billing_retriever = EnsembleRetriever(
    retrievers=[billing_bm25, billing_vectorstore.as_retriever(search_kwargs={"k": 5})],
    weights=[0.4, 0.6],
)
```

## Milestone 2 — Two specialist subagents

Each specialist is a plain `create_agent`, with a tool that wraps its retriever — the same "retriever inside a tool" pattern from the playlist's [RAG chapter](/docs/genai/rag), unchanged.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_product_docs(query: str) -> str:
    """Search product documentation for an answer."""
    docs = product_retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def search_billing_faq(query: str) -> str:
    """Search billing FAQs and past invoice explanations."""
    docs = billing_retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def issue_refund(order_id: str, amount_usd: float) -> str:
    """Issue a refund for an order. Requires approval above $50."""
    process_refund(order_id, amount_usd)  # your own billing-system call
    return f"Refunded ${amount_usd} for order {order_id}."

tech_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search_product_docs],
    system_prompt="You are a technical support specialist. Answer only from search_product_docs.",
    name="tech_support",
)

billing_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search_billing_faq, issue_refund],
    system_prompt="You are a billing specialist. Look up facts before answering; use issue_refund only when asked to refund.",
    name="billing",
)
```

## Milestone 3 — Human-in-the-loop on the refund tool

A refund is money leaving the business — exactly the kind of action from the [human-in-the-loop chapter](/docs/genai/langchain-advanced/human-in-the-loop) that should not run unsupervised. Gate it only above a threshold, using a `when` predicate, so small refunds do not create approval fatigue.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def refund_needs_approval(request) -> bool:
    return request.tool_call["args"].get("amount_usd", 0) > 50

billing_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[search_billing_faq, issue_refund],
    system_prompt="You are a billing specialist.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": refund_needs_approval,
                },
            },
        ),
    ],
    checkpointer=InMemorySaver(),
    name="billing",
)
```

## Milestone 4 — Middleware: redact PII, keep the thread bounded, retry

Support conversations routinely contain emails, phone numbers, and card fragments — [middleware](/docs/genai/langchain-advanced/middleware) scrubs it before it ever reaches a model call log. Long-running tickets need [summarization](/docs/genai/langchain-advanced/memory) so the transcript does not blow the context window, and a flaky model call should retry before it becomes a customer-visible error.

```python
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware, ModelRetryMiddleware

shared_middleware = [
    PIIMiddleware("email"),
    PIIMiddleware("credit_card"),
    ModelRetryMiddleware(max_retries=3),
    SummarizationMiddleware(
        model="anthropic:claude-haiku-4-5",
        trigger=("tokens", 4000),
        keep=("messages", 20),
    ),
]

tech_agent = create_agent(..., middleware=shared_middleware)
billing_agent = create_agent(..., middleware=[*shared_middleware, HumanInTheLoopMiddleware(...)])
```

## Milestone 5 — Memory: this conversation, and this customer

Two different kinds of memory, doing two different jobs — the distinction the [memory chapter](/docs/genai/langchain-advanced/memory) draws between a checkpointer and a store:

- **Checkpointer** — remembers *this ticket's* back-and-forth, so the customer never repeats themselves mid-conversation.
- **Store** — remembers *this customer* across tickets: their plan, past issues, tone preference.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime

checkpointer = InMemorySaver()   # swap for PostgresSaver in production
store = InMemoryStore()          # swap for PostgresStore in production

@tool
def get_customer_profile(runtime: ToolRuntime) -> str:
    """Look up what we already know about this customer."""
    profile = store.get(("customers",), runtime.context["customer_id"])
    return str(profile.value) if profile else "New customer, no history."

@tool
def save_customer_note(note: str, runtime: ToolRuntime) -> str:
    """Save a durable note about this customer for future tickets."""
    store.put(("customers",), runtime.context["customer_id"], {"note": note})
    return "Saved."
```

## Milestone 6 — The supervisor: wiring the subagents together

The [multi-agent systems chapter](/docs/genai/langchain-advanced/multi-agent-systems)'s single-dispatch pattern: one `task` tool, one enum of specialists, a supervisor that only routes and never answers technical or billing questions itself.

```python
from enum import Enum

class Specialist(str, Enum):
    TECH = "tech_support"
    BILLING = "billing"

SPECIALISTS = {"tech_support": tech_agent, "billing": billing_agent}

@tool
def task(specialist: Specialist, request: str) -> str:
    """Delegate the customer's request to the right specialist."""
    result = SPECIALISTS[specialist].invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content

supervisor = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[task, get_customer_profile, save_customer_note],
    system_prompt=(
        "You are Aegis, a support router. Look up the customer profile first. "
        "Delegate technical questions to tech_support and billing questions to billing. "
        "Never answer either kind of question yourself."
    ),
    checkpointer=checkpointer,
    store=store,
    name="supervisor",
)
```

```mermaid
sequenceDiagram
    participant C as Customer
    participant S as Supervisor
    participant P as get_customer_profile
    participant B as Billing subagent
    participant H as Human reviewer

    C->>S: "Refund my last invoice, it's wrong"
    S->>P: look up customer_id
    P-->>S: plan=Pro, 2 past tickets
    S->>B: task(billing, "refund last invoice")
    B->>B: search_billing_faq(...)
    B->>S: proposes issue_refund($120)
    S->>H: interrupt — $120 > $50 threshold
    H-->>S: approve
    S-->>C: "Refunded $120. Anything else?"
```

## Milestone 7 — Resilience: a fallback model

A production support bot cannot go down because one provider had an outage. The [resilience chapter](/docs/genai/langchain-advanced/runnable-resilience) pattern applies to the underlying chat model each agent is built on:

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

resilient_model = (
    ChatAnthropic(model="claude-sonnet-4-5")
    .with_retry(stop_after_attempt=2)
    .with_fallbacks([ChatOpenAI(model="gpt-4.1-mini")])
)

supervisor = create_agent(model=resilient_model, tools=[task, get_customer_profile, save_customer_note], ...)
```

## Milestone 8 — Streaming the reply

A support widget cannot show a spinner for ten seconds. [Streaming](/docs/genai/langchain-advanced/streaming) turns the final answer into tokens as they are generated, and surfaces routing/tool steps separately so the UI can show "Checking billing records..." instead of silence.

```python
config = {"configurable": {"thread_id": ticket_id}}
context = {"customer_id": customer_id}

for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Refund my last invoice, it's wrong"}]},
    config=config,
    context=context,
    stream_mode=["messages", "updates"],
):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
        if token.content:
            print(token.content, end="", flush=True)
    elif chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
        notify_human_reviewer(chunk["data"]["__interrupt__"])
```

## Milestone 9 — Tracing and caching

Two chapters, two different jobs: [tracing](/docs/genai/langchain-advanced/callbacks-and-tracing) tells you *why* a specific ticket went wrong; [caching](/docs/genai/langchain-advanced/caching) stops you paying for the same FAQ answer twice.

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "aegis-support"

from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

# Billing FAQ answers repeat across thousands of tickets — semantic caching pays for itself fast.
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.15,
))
```

:::warning Do not cache the supervisor's own calls
The supervisor's answers depend on `customer_id` and live order data — caching them risks serving one customer's refund confirmation to another. Scope caching to the FAQ-lookup calls inside the billing subagent, not the supervisor's routing decisions.
:::

## Milestone 10 — Deploying Aegis

Everything so far ran in a Python process on your laptop, with `InMemorySaver`/`InMemoryStore` losing every ticket the moment it restarted — fine for building, not for customers. Deployment has four parts: a real API in front of the agent, durable backends behind it, a container to ship, and the config that tells the container which backends to use.

### Serving the agent behind FastAPI

Wrap `supervisor` in a thin API. The [streaming chapter](/docs/genai/langchain-advanced/streaming)'s `.stream()` loop becomes a Server-Sent-Events response, so the same token-by-token UI you built locally works over HTTP unchanged.

```python
# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class TicketMessage(BaseModel):
    ticket_id: str
    customer_id: str
    message: str

async def event_stream(payload: TicketMessage):
    config = {"configurable": {"thread_id": payload.ticket_id}}
    context = {"customer_id": payload.customer_id}
    async for chunk in supervisor.astream(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
        context=context,
        stream_mode=["messages", "updates"],
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if token.content:
                yield f"data: {token.content}\n\n"
        elif chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
            yield f"event: interrupt\ndata: {chunk['data']['__interrupt__']}\n\n"

@app.post("/chat")
async def chat(payload: TicketMessage):
    return StreamingResponse(event_stream(payload), media_type="text/event-stream")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
```

Run it with an ASGI server that supports streaming responses:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

:::warning One worker per checkpointer connection pool, not one process for everything
`PostgresSaver`/`PostgresStore` manage their own connection pool. Multiple `uvicorn` workers each need their own pool, not a pool created once and shared across forked processes — construct the checkpointer and store inside each worker's startup, not at import time, if you scale past `--workers 1`.
:::

### Swapping in durable backends

Every backend used so far has a production-grade drop-in replacement, already named in the [memory](/docs/genai/langchain-advanced/memory) and [caching](/docs/genai/langchain-advanced/caching) chapters. Deployment is where you actually flip the switch:

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = os.environ["AEGIS_DB_URI"]

checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = checkpointer_cm.__enter__()
checkpointer.setup()

store_cm = PostgresStore.from_conn_string(DB_URI)
store = store_cm.__enter__()
store.setup()
```

### Containerising it

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  aegis:
    build: .
    ports: ["8000:8000"]
    environment:
      AEGIS_DB_URI: postgresql://aegis:aegis@postgres:5432/aegis
      REDIS_URL: redis://redis:6379
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      LANGSMITH_TRACING: "true"
      LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
      LANGSMITH_PROJECT: aegis-support
    depends_on: [postgres, redis]
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: aegis
      POSTGRES_PASSWORD: aegis
      POSTGRES_DB: aegis
    volumes: ["pgdata:/var/lib/postgresql/data"]
  redis:
    image: redis:7
volumes:
  pgdata:
```

```bash
docker compose up --build
```

### What deployment changes, and what it doesn't

| Concern | Local (this capstone so far) | Deployed |
|---|---|---|
| Checkpointer / store | `InMemorySaver` / `InMemoryStore` | `PostgresSaver` / `PostgresStore`, one shared DB |
| Cache | in-process or a local Redis | managed Redis, same `RedisSemanticCache` code |
| Secrets | hardcoded or a local `.env` | injected as container environment variables, never committed |
| Interrupts | printed to the console | `event: interrupt` pushed to whatever queues your human-reviewer dashboard |
| Scaling | one Python process | multiple `uvicorn` workers/replicas behind a load balancer, sharing the same Postgres/Redis |

The agent code itself — `create_agent`, the middleware list, the tools — does not change between local and deployed. Only the objects passed as `checkpointer=`, `store=`, and the cache backend change, which is the entire point of LangGraph's storage interfaces being swappable.

## Architecture, end to end

```mermaid
flowchart LR
    subgraph Request
        C(["Customer"]) --> W["Chat widget"]
    end
    subgraph Serving["Serving — Docker container"]
        W -->|SSE stream| API["FastAPI + uvicorn"]
    end
    subgraph Aegis
        API --> SUP["Supervisor<br/>+ resilient model"]
        SUP --> TASK["task tool"]
        TASK --> BA["Billing agent<br/>+ HITL + retry"]
        TASK --> TA["Tech agent<br/>+ retry"]
        BA --> ER[("Ensemble retriever")]
        TA --> PR[("Parent-doc retriever")]
        BA -.cache.-> CACHE[("Semantic cache")]
    end
    subgraph Persistence
        SUP <--> CK[("PostgresSaver")]
        SUP <--> ST[("PostgresStore")]
    end
    subgraph Ops
        SUP -.trace.-> LS["LangSmith"]
        BA -.interrupt.-> HR["Human reviewer"]
    end
```

## Grading rubric

| Criterion | What "done" looks like |
|---|---|
| Routing | Supervisor never answers directly; every domain question reaches the right subagent |
| Retrieval | Product answers cite the parent chunk; billing answers survive an exact invoice-number query |
| Approval | A refund over $50 pauses and only proceeds after `approve`/`edit` |
| Memory | The customer's name/plan persists across two separate `thread_id`s (long-term) but each ticket's own history resets on a new thread (short-term) |
| Resilience | Killing the primary model's API key still produces an answer, via fallback |
| Cost | A repeated FAQ question is a cache hit, visible in LangSmith as near-zero latency |
| Deployment | `docker compose up` serves a working `/chat` endpoint; restarting the `aegis` container mid-ticket does not lose that ticket's history |

## Extensions

- Add a third subagent (account management) and route to it via [multi-agent discovery](/docs/genai/langchain-advanced/multi-agent-systems) instead of a fixed enum, once there are too many specialists to list in a prompt.
- Add a `SelfQueryRetriever` over ticket history, so the supervisor can answer "how many refunds has this customer had this year?" directly from metadata.
- Put the deployment behind a real load balancer and scale `uvicorn` to multiple replicas; confirm two replicas serving the same `thread_id` still see a consistent conversation, since both read/write the same Postgres checkpointer.
- Add a `/metrics` endpoint (request count, cache hit rate, interrupt rate) so the rubric's cost and deployment criteria can be checked without opening LangSmith.

## Checklist

- [ ] I can point to the specific chapter behind every box in the architecture diagram
- [ ] I can explain why the refund threshold uses a `when` predicate instead of gating every call to `issue_refund`
- [ ] I can explain why caching is scoped to the FAQ lookup and not the supervisor's own answers
- [ ] I could extend this project with a new subagent without restructuring the supervisor
- [ ] I can explain what changes (and what doesn't) between running this locally and running it in `docker compose`
