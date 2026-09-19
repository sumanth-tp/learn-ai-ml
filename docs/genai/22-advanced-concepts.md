---
id: advanced-concepts
title: "Advanced Concepts (beyond the playlist)"
sidebar_label: "22 · Advanced concepts"
sidebar_position: 22
slug: /genai/advanced-concepts
description: "What sits between a working demo and a production system — advanced RAG, evaluation, LangGraph, memory, MCP, guardrails, caching, cost control and LLMOps."
tags: [genai, advanced-rag, langgraph, evaluation, ragas, llmops, guardrails, mcp, multi-agent]
---

:::note Not from the playlist
The 21 videos end with the agent build. This chapter and the [capstone](/docs/genai/capstone) are **additions** — the topics the course itself points to as next steps, gathered in one place.

Several are named explicitly in the source videos: the [RAG project](/docs/genai/youtube-chatbot) lists advanced-RAG improvements and RAGAS, the [agents chapter](/docs/genai/ai-agent) ends by recommending LangGraph, and the [document loaders chapter](/docs/genai/document-loaders) notes memory moving into LangGraph.
:::

**In one line.** Everything so far builds a system that works; this chapter is about making one that keeps working.

## 1. Advanced RAG

A naive RAG pipeline fails in predictable ways. Each stage has known fixes.

```mermaid
flowchart TB
    subgraph I["Indexing"]
        I1["Clean the source<br/>fix OCR/ASR errors, translate"]
        I2["Semantic chunking<br/>instead of fixed-size"]
        I3["Parent-child chunks<br/>small to search, large to read"]
        I4["Enrich metadata<br/>source, date, section, author"]
    end
    subgraph PRE["Pre-retrieval"]
        P1["Query rewriting"]
        P2["Multi-query generation"]
        P3["HyDE<br/>embed a hypothetical answer"]
        P4["Domain routing"]
    end
    subgraph RET["Retrieval"]
        R1["Hybrid search<br/>BM25 keyword + semantic"]
        R2["MMR for diversity"]
        R3["Metadata filtering"]
    end
    subgraph POST["Post-retrieval"]
        O1["Re-ranking<br/>cross-encoder or LLM"]
        O2["Contextual compression"]
        O3["Deduplication"]
    end
    I --> PRE --> RET --> POST
```

**Query rewriting.** Users type three words. Have an LLM expand the query before it reaches the retriever.

**HyDE — Hypothetical Document Embeddings.** Ask the LLM to *invent* an answer, embed that, and search with it. A hypothetical answer is lexically closer to real answers than the question was — often a large gain for little work.

**Hybrid retrieval.** Semantic search misses exact tokens: product codes, error numbers, surnames. Keyword search (BM25) misses paraphrase. Run both and merge with Reciprocal Rank Fusion. One of the highest-value upgrades available.

**Re-ranking.** Retrieve twenty candidates cheaply, then re-score with a cross-encoder that reads query and document *together*. Slower per item, far more accurate than embedding similarity. Keep the top five.

**Parent-child chunking.** Index small chunks for precise matching, but return the larger parent chunk for context.

## 2. Evaluating a RAG system

You cannot improve what you do not measure, and "the answers look good" is not measurement.

**RAGAS** scores four things:

| Metric | The question it answers |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? |
| **Answer relevance** | Does the answer address the question asked? |
| **Context precision** | Of what was retrieved, how much was useful? |
| **Context recall** | Of what was needed, how much was retrieved? |

The last two diagnose **retrieval**; the first two diagnose **generation**. Low context recall means the retriever is missing documents — no amount of prompt engineering fixes that.

**LangSmith** traces every step: which chunks came back, what the prompt looked like, tokens, latency, cost.

**Build a golden set.** Thirty to a hundred question-answer pairs written by someone who knows the domain. Run it after every change. Without a regression set you are guessing.

## 3. LangGraph

LangChain expresses **chains** — directed acyclic graphs. Agents are not acyclic: they loop, branch, backtrack, pause for approval and resume.

LangGraph models a workflow as a **state graph**: nodes are steps, edges are transitions, and a shared state object flows through.

```mermaid
flowchart LR
    S(["START"]) --> A["agent node<br/>the LLM decides"]
    A -->|needs a tool| T["tool node"]
    T --> A
    A -->|needs approval| H["human review"]
    H --> A
    A -->|done| E(["END"])
```

What it adds over chains:

- **Cycles** — the fundamental agent shape
- **Persistence** — checkpoint state, resume after a crash or days later
- **Human-in-the-loop** — pause before consequential actions
- **Streaming intermediate state** — show progress, not just the final answer
- **Time travel** — rewind and take a different branch

Everything learned about ReAct, scratchpads and tool calling applies directly. The framework changes; the ideas do not.

## 4. Memory

LangChain's memory components are migrating to LangGraph — the reason memory was skipped in the playlist. The strategies matter more than the API.

| Strategy | Keeps | Trade-off |
|---|---|---|
| Buffer | everything | grows without bound |
| Window | the last *N* turns | bounded, forgets |
| Summary | an LLM summary of history | cheap, lossy |
| Vector-backed | embeds turns, retrieves relevant ones | scalable, more complex |
| Entity | facts about people and things | precise, needs design |

Production systems usually **combine** them: recent turns verbatim, older turns summarised, durable facts in a separate store.

## 5. Multi-agent systems

One agent with twenty tools picks the wrong one. Several specialist agents, each with three or four, do better.

**Supervisor pattern** — a coordinator routes work to specialists and assembles the result.
**Hierarchical** — supervisors of supervisors, for genuinely large problems.
**Peer collaboration** — agents debate or critique each other. A generator plus a critic often beats a single pass.

The costs are real: more tokens, more latency, harder debugging, failures that compound across handoffs. Reach for multi-agent when one agent measurably cannot cope — not because it sounds sophisticated.

## 6. Model Context Protocol (MCP)

Every framework invented its own tool format, so integrations never transferred. MCP is an open standard: expose a capability once as an MCP server, and any compatible client can use it.

```mermaid
flowchart LR
    H["Host application"] --> C["MCP client"]
    C --> S1["MCP server<br/>filesystem"]
    C --> S2["MCP server<br/>database"]
    C --> S3["MCP server<br/>your internal API"]
```

Servers expose **tools** (callable functions), **resources** (readable data) and **prompts** (reusable templates). Practically: write one integration instead of one per framework.

## 7. Guardrails and safety

**Input guardrails** — prompt-injection detection, PII scrubbing, topic restriction, rate limiting.

**Output guardrails** — check for hallucination against the retrieved context, block unsafe content, validate schema, enforce citations.

**Grounding and citations.** Require the model to cite which chunk each claim came from. Users can verify, and models cite less confidently when inventing.

:::danger Prompt injection is unsolved
If your system reads untrusted content — web pages, uploaded files, user text — that content can contain instructions aimed at your model. Treat every retrieved document as untrusted input. Defences reduce risk; none eliminate it.
:::

## 8. Cost and latency

**Prompt caching.** Providers cache repeated prefixes. Put stable content first and variable content last — a large saving for near-zero effort.

**Semantic caching.** Cache answers by query embedding, so a near-identical question skips the model call.

**Model routing.** Easy requests to a small fast model, hard ones to a frontier model. Classification, extraction and formatting rarely need your most expensive option.

**Streaming.** Does not reduce cost; transforms perceived latency.

**Batching.** Use `batch()` rather than looping `invoke()`.

**Context trimming.** Every retrieved token is billed. Compression and re-ranking pay for themselves.

## 9. Fine-tuning, revisited

RAG adds knowledge. Fine-tuning changes **behaviour**. Consider it for a consistent format or tone, a domain vocabulary the base model lacks, or distillation — teaching a small fast model to imitate a large one.

**LoRA / QLoRA** train a small adapter while freezing base weights — orders of magnitude cheaper than full fine-tuning, and usually sufficient.

Most production systems use both: fine-tune for behaviour, RAG for facts.

## 10. Deployment and LLMOps

**Serving.** FastAPI plus LangServe exposes chains as endpoints. Stream with SSE or WebSockets.

**Observability.** Trace every request. Log prompts, retrieved context, token counts, latency, cost. You will need this the first time someone reports a bad answer.

**Versioning.** Prompts and schemas are code. Version them, review them, roll them back.

**Evaluation in CI.** Run your golden set on every change. Block merges that regress faithfulness.

**Canary releases.** Model and prompt changes are behaviour changes. Ship to a slice first.

**Feedback loops.** Collect thumbs up/down with the trace attached. That becomes your next evaluation set.

## 11. Multimodal and beyond

**Multimodal RAG** — index images and diagrams alongside text. Essential for technical manuals where the diagram carries the meaning.

**Agentic RAG** — an agent that decides *whether* to retrieve, *which* index to use, and whether to search the web instead.

**Memory-augmented RAG** — personalised systems that recall what a specific user asked last week.

## A maturity ladder

| Level | You have |
|---|---|
| 0 | A prompt and a model |
| 1 | Naive RAG: load, split, embed, retrieve, generate |
| 2 | Better retrieval: hybrid search, re-ranking, metadata filters |
| 3 | Evaluation: a golden set, RAGAS, tracing |
| 4 | Agents: tool calling, ReAct, LangGraph |
| 5 | Production: guardrails, caching, cost control, CI evaluation, feedback |

The playlist takes you to level 1, with a taste of 4. Most **value** appears between 2 and 5.

## Checklist

- [ ] I can name three advanced-RAG techniques and the failure each fixes
- [ ] I can explain the four RAGAS metrics and which stage each diagnoses
- [ ] I can explain why agents need LangGraph rather than chains
- [ ] I can name three memory strategies and their trade-offs
- [ ] I can explain prompt injection and why it is not solved
- [ ] I can list three ways to cut cost without hurting quality
- [ ] I can place my own project on the maturity ladder
