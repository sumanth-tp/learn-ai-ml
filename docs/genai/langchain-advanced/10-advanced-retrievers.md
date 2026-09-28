---
id: lc-adv-advanced-retrievers
title: "Advanced Retrievers — Self-Query, Parent-Document, and Ensemble"
sidebar_label: "10 · Advanced retrievers"
sidebar_position: 10
slug: /genai/langchain-advanced/advanced-retrievers
description: "Three retrievers beyond the ones the playlist demos: SelfQueryRetriever for metadata filters written in plain English, ParentDocumentRetriever for small-chunk search with full-document context, and EnsembleRetriever for hybrid keyword + semantic search."
tags: [langchain, retrievers, rag, hybrid-search, self-query]
---

:::note Addition — not from the playlist
Part of [LangChain Advanced Topics](/docs/genai/langchain-advanced/create-agent). The [retrievers chapter](/docs/genai/retrievers) covers `WikipediaRetriever`, the vector store retriever, MMR, `MultiQueryRetriever`, and `ContextualCompressionRetriever`. These three are the retrievers [advanced concepts](/docs/genai/advanced-concepts) names under hybrid search and parent-child chunking, but never shows in code.
:::

:::warning These moved to `langchain_classic`
As of LangChain 1.0, `SelfQueryRetriever`, `ParentDocumentRetriever`, and `EnsembleRetriever` live in the `langchain_classic` package, not `langchain`. Install it alongside `langchain`: `pip install langchain-classic`.
:::

**In one line.** Each of these three fixes a specific failure mode a plain vector-store retriever (from the [retrievers chapter](/docs/genai/retrievers)) has: it can't filter on metadata from a natural-language query, it forces a trade-off between chunk size for search vs chunk size for context, and it misses exact keywords that paraphrase-based semantic search was never going to catch.

## SelfQueryRetriever — metadata filters from plain English

A user asking *"comedies from the 1990s rated above 8"* is really asking for a semantic search on *"comedies"* plus a structured filter on `year` and `rating`. `SelfQueryRetriever` uses an LLM to split a natural-language query into exactly that: a search string and a metadata filter, then applies both.

```python
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.structured_query import AttributeInfo
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

metadata_field_info = [
    AttributeInfo(name="genre", description="The movie genre", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="rating", description="A 1-10 IMDb-style rating", type="float"),
]

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./movies_db")

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents="Brief summary of a movie",
    metadata_field_info=metadata_field_info,
)

results = retriever.invoke("comedies from the 1990s rated above 8")
# internally: search="comedy", filter=(genre == "comedy") AND (year >= 1990) AND (year < 2000) AND (rating > 8)
```

This only works when your documents actually carry that metadata — populate it at ingestion time in the [document loaders chapter](/docs/genai/document-loaders), because the retriever can only filter on fields it was told exist.

## ParentDocumentRetriever — small chunks to search, large chunks to read

The [text splitters chapter](/docs/genai/text-splitters) already names this trade-off: small chunks match precisely but lack context; large chunks carry context but drown a precise match in irrelevant text. `ParentDocumentRetriever` keeps both — small chunks are what gets embedded and searched, but the retriever returns the larger parent chunk they came from.

```python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=OpenAIEmbeddings(), collection_name="parent_child"),
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)  # splits into parents, then children, embeds only the children

results = retriever.invoke("what caused the outage?")
# search matched a precise 400-char child chunk, but you get the full 2000-char parent back
```

## EnsembleRetriever — hybrid keyword + semantic search

Semantic search misses exact tokens: product codes, error numbers, surnames — because embeddings capture meaning, not spelling. Keyword search (BM25) misses paraphrase, for the opposite reason. `EnsembleRetriever` runs both and merges the results with Reciprocal Rank Fusion, which [advanced concepts](/docs/genai/advanced-concepts) calls "one of the highest-value upgrades available."

```python
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],   # semantic weighted slightly higher; tune against your own queries
)

results = ensemble_retriever.invoke("error code E-4521 on the payment gateway")
# BM25 catches the exact code; the vector retriever catches paraphrases of the same issue
```

Any number of retrievers can be ensembled this way, not just two — add a `SelfQueryRetriever` into the same list if metadata filtering and hybrid search are both needed at once.

## Which one, when

| Symptom | Retriever |
|---|---|
| Users describe filters in words ("recent", "under $50", "by Author X") | `SelfQueryRetriever` |
| Small chunks match well but lose surrounding context | `ParentDocumentRetriever` |
| Exact tokens (codes, IDs, names) get missed by semantic search alone | `EnsembleRetriever` (with BM25) |

These compose with the retrievers already covered — an `EnsembleRetriever` can wrap a `SelfQueryRetriever` as one of its inputs, and any of them can feed into `ContextualCompressionRetriever` from the [retrievers chapter](/docs/genai/retrievers) to re-rank before generation.

## Checklist

- [ ] I can explain what `SelfQueryRetriever` needs at ingestion time to work (metadata fields)
- [ ] I can explain the small-chunk-search / large-chunk-context trade-off `ParentDocumentRetriever` solves
- [ ] I can set up hybrid search with `EnsembleRetriever` and explain what BM25 catches that embeddings miss
- [ ] I know these three now live in `langchain_classic`, not `langchain`
