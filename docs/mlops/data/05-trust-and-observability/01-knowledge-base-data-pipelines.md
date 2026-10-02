---
id: dm-knowledge-base-data-pipelines
title: "Data Management · Lecture 14 — Knowledge Base Data Pipelines"
sidebar_label: "14 · LLM data pipelines"
sidebar_position: 1
slug: /mlops/data/knowledge-base-data-pipelines
description: "Build a versioned RAG knowledge base and interpret cosine retrieval and ANN trade-offs."
tags: [data-management, rag, vector-search, llm-data]
---

import Infographic from '@site/src/components/Infographic';
import CosineRetrievalLab from '@site/src/components/viz/CosineRetrievalLab';

**In one line.** An LLM knowledge base is a maintained data product: source, chunk, embed, index, retrieve and verify.

## The idea in plain words

A retrieval-augmented generation application depends on a current, authorised collection of source material. Documents arrive from files, help centres or databases; the pipeline extracts their text, creates chunks, computes embeddings, indexes them and later retrieves candidates for a question. The model can only ground an answer in what the retrieval step actually supplies. A beautiful prompt cannot repair an outdated, duplicated or inaccessible knowledge base.

The lecture names the core sequence: **load → chunk → embed → store → retrieve**. Treat each arrow as a data contract. Loading preserves document identity, source version and permission. Chunking preserves enough context to make a passage meaningful. Embedding stores both the vector and the model version that produced it. Indexing supports search under filters and updates. Retrieval decides which chunks can enter a prompt, often after ranking and a relevance check. Downstream answer generation and citation checking make the full RAG system, but this chapter focuses on the data pipeline beneath them.

<Infographic src="/img/dm/llm-pipelines.svg" alt="A RAG data pipeline loads and cleans documents, chunks them, embeds and indexes the chunks, then retrieves filtered candidates; the worked cosine between two four-dimensional vectors is two thirds." caption="A vector score is one stage of retrieval, not a guarantee that a chunk is used in an answer." />

The source's numerical example uses a query vector **[1, 0, 1, 1]** and a chunk vector **[1, 1, 1, 0]**. Their dot product is 2 and each norm is √3, so cosine similarity is **2/3 ≈ 0.667**. That is a correct score. It does not establish that this chunk is retrieved: top-k selection, a threshold, other candidate scores, metadata filters, access control and approximate-index recall can all change the returned set.

:::note Beyond the lecture

The source introduces embeddings, vector stores, cosine and ANN indexes. The sections below add source permissions, stable chunk IDs, embedding-version migrations, deletion and freshness, filtered search, exact-search comparisons and retrieval evaluation. The source's blanket sub-linear ANN claim is replaced with measured recall and latency for the chosen index and workload.

:::

The default lab reproduces the source calculation: **dot = 2**, both norms are √3, and **cosine = 0.667**. Change one chunk component or the illustrative threshold to see how the score and a threshold decision move. The threshold is not a universal relevance rule; it must be chosen on labelled queries and the actual corpus.

<CosineRetrievalLab />

## How it works

### Chunk, embed, index

Load documents → chunk (with overlap) → embed → store in a vector DB → retrieve semantically. Freshness, dedup and quality govern answers.

### Cosine & ANN

Embeddings place similar text nearby; rank by cosine similarity. An exact scan compares eligible vectors directly; ANN indexes such as HNSW and IVF can reduce latency but need measured recall, memory and build-time trade-offs.

:::tip

**Worked.** q=[1,0,1,1], d=[1,1,1,0] → cos = 2/3 = 0.667 → retrieved.

:::


## A real system that works this way

**Qdrant** is a concrete vector search engine. Its documentation describes dense-vector HNSW indexing, exact search, payload indexes for metadata filters and collection configuration. A vector index speeds similarity search for a large collection, while payload indexes help restrict candidates by tenant, document type or permission attributes. Qdrant can choose a full scan for small filtered sets; index choice is a workload decision rather than a claim that every query is sub-linear.

Imagine an internal support assistant with policy documents for several regions. The ingestion job records each document's ID, region, effective date, access group and content hash. It extracts text, strips navigation chrome, creates chunks and embeds them with a named embedding model version. Each indexed point carries its source document ID and span, so a retrieved passage can be cited. The query path applies the requester's access group and region filters before choosing candidate chunks. A similarity result from an inaccessible policy is not a valid answer source, however close its vector is.

When a policy is revised, the job must replace or delete the old chunks. If the embedding model changes, old and new vectors may inhabit different spaces even if their dimensions match. Build a new index or collection for the new model, evaluate it against a fixed query set and switch the serving pointer after validation. Index freshness is measured from source publication to searchable, approved chunk, not from the last successful embed call alone.

## Code you can run

Reproduce the lecture's cosine result with standard-library Python. Explicitly reject a zero vector because its cosine is undefined.

```python
import math

def cosine(left, right):
    assert len(left) == len(right)
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value ** 2 for value in left))
    right_norm = math.sqrt(sum(value ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine needs nonzero vectors")
    return dot / (left_norm * right_norm)

query = [1, 0, 1, 1]
chunk = [1, 1, 1, 0]
score = cosine(query, chunk)
print(f"dot=2, cosine={score:.3f}")
assert math.isclose(score, 2 / 3)
```

The next example chooses from a small in-memory corpus after applying a region and access filter. It shows why a good score alone does not imply selection. In production, the filter must be enforced at the retrieval boundary, not merely described in the prompt.

```python
import math

def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value ** 2 for value in left))
    right_norm = math.sqrt(sum(value ** 2 for value in right))
    return dot / (left_norm * right_norm)

query = [1, 0, 1, 1]
chunks = [
    {"id": "a", "vector": [1, 1, 1, 0], "region": "EU", "access": "public"},
    {"id": "b", "vector": [1, 0, 1, 1], "region": "US", "access": "public"},
    {"id": "c", "vector": [1, 0, 0, 1], "region": "EU", "access": "restricted"},
]
eligible = [item for item in chunks if item["region"] == "EU" and item["access"] == "public"]
ranked = sorted(eligible, key=lambda item: cosine(query, item["vector"]), reverse=True)
print([(item["id"], round(cosine(query, item["vector"]), 3)) for item in ranked])
assert [item["id"] for item in ranked] == ["a"]
assert math.isclose(cosine(query, chunks[1]["vector"]), 1)
```

Chunk `b` is the exact vector match, but it belongs to a different region and is absent from the eligible result. This example assumes the same embedding space and a simple exact scan; an ANN index needs a measured recall check against an exact baseline.

## Designing with it

### Establish source identity and rights

Keep the document ID, source system, revision or content hash, publication time, access policy and deletion state. A file name alone may be reused for new content. A page URL may redirect, change language or be removed. The pipeline should know whether a newly fetched document replaces an old version, adds a version or is a duplicate. Deduplicate by stable source identity and meaningful content hashes, but avoid collapsing two documents whose text matches while their access rules differ. Preserve a link from each chunk to its source and span for citation and deletion.

An internal corpus can still contain material a user is not allowed to see. Attach access metadata at ingestion and apply it as a hard retrieval filter for every query. A prompt instruction such as "do not reveal private text" is not an access control boundary. Check that a changed permission propagates to all old chunks. Deletion needs a measurable completion path through raw storage, vector index, caches and generated excerpts where policy requires it.

### Chunk for retrieval and citation

Chunk size is a trade-off. Short passages make semantic search focused but may omit a qualifier or table heading. Long passages carry context but can dilute the signal and consume prompt tokens. Overlap can preserve a sentence across boundaries while duplicating content and increasing index size. Start with document structure such as headings and paragraphs, then test sizes against actual user questions. Keep titles and section paths as metadata; they often provide context more cheaply than a very large chunk.

Use a stable chunk ID derived from document version and span or deterministic segmentation. If a parser inserts one extra paragraph, downstream IDs may all change; decide whether the system tolerates that reindexing. Preserve source offsets when possible so a citation resolves to the exact passage. Tables, images and code require dedicated extraction policies. Flattening a table into arbitrary text can remove its row and column meaning; a multimodal figure may need a caption or separate representation.

### Version embeddings and indexes

Record embedding model, revision, dimensionality, normalisation and preprocessing with every index build. A new model can produce vectors with the same dimensions but different geometry, so mixing versions can corrupt rankings. Test a migration in a shadow index. Compare retrieval recall, latency, filter correctness and index freshness on a labelled query set. Keep the old index available until the new one is approved. If sources update continually, reconcile the delta stream during the switch so the new collection does not start already stale.

Exact vector search compares a query with each eligible candidate and is a useful quality baseline, especially for small filtered sets. ANN methods such as HNSW or IVF trade index memory and construction cost against query latency and recall. Their observed complexity depends on data distribution, index parameters, filtering and hardware; a blanket sub-linear promise is not a service guarantee. Measure recall at k against exact results, p95 latency, update delay and memory. A faster search that misses the only policy clause needed by a user is not an improvement.

### Evaluate the retrieved evidence

Build a golden set of questions with relevant passage IDs, including ambiguous and unanswerable cases. Measure whether the needed source appears in top k, whether forbidden sources are excluded and whether the cited span really supports the answer. Analyse misses by source, language, length and recency. A high average cosine score may reflect duplicated boilerplate. A low score can still accompany a useful exact identifier match, so hybrid lexical and vector search may help. Keep answer quality and retrieval quality separate so a generation error is not blamed on indexing.

## Update a policy without leaking an old version

A regional policy changes at noon. The source system emits revision 12 with a publication timestamp and a stable document ID. The ingestion job compares its hash with revision 11, extracts the new text and records its access group. It then chunks the revised sections, embeds them with the active model and writes candidates to a staging collection. A reconciliation check counts expected chunks, verifies every chunk points to revision 12 and confirms that the previous revision's chunks are marked for removal. Only then is the new version made searchable.

Consider a partial failure. Half of revision 12's chunks index successfully and the job crashes. If the application searches the live collection immediately, it may combine old and new policies in one answer. Use a versioned publication pointer or an atomic batch operation supported by the storage system so a query sees one approved revision for a document. A retry should use stable chunk IDs and replace its staged records rather than append duplicates. If the source is corrected again during recovery, choose a new revision identity and record why the previous candidate was abandoned.

Now an employee loses access to that region. The permission update must reach the search index quickly. An access filter on a stale payload is dangerous. Either query a current authorisation service for each result, maintain a strictly bounded metadata propagation delay with a fail-closed policy, or design another approved enforcement path. Test this with a user whose access is revoked while old chunks remain indexed. The answer generator must never receive the forbidden text.

The team changes embedding models next month. It does not simply append new vectors to the old collection. It builds a new collection with the same source snapshot and permissions, calculates exact and ANN retrieval baselines, and compares top-k recall on the golden questions. A result with cosine 0.667 under one model is not calibrated against 0.667 under another. After approval, the serving pointer switches to the new index; recent source changes are replayed up to a recorded watermark. The old collection is retained only as long as rollback and retention policy require.

### Diagnose a poor answer

A user asks about a refund exception. The assistant gives an outdated rule. First inspect the prompt context: which chunks and source revisions were supplied? If revision 11 was retrieved, check index freshness and deletion. If revision 12 was indexed but ranked below generic boilerplate, inspect chunk boundaries, duplicate text, title metadata and lexical terms. If the correct chunk was retrieved but the answer contradicted it, the generation and citation stage needs work. Each failure has a different owner and repair.

Check the query's filter path as carefully as its vector score. A region filter applied after top-k retrieval can discard all initial results even when highly relevant allowed chunks exist lower in the corpus. Prefer a search strategy that respects filters during candidate generation, and evaluate its recall. Qdrant documents both payload indexing and plan changes for small filtered sets; inspect the actual query plan and response rather than assuming every vector search follows one path.

Finally, monitor the knowledge base as data. Report source-to-index delay, number of indexed documents and chunks, parser failures, duplicate rates, deleted-source residue, query latency and gold-set retrieval quality. A successful embedding job can still leave a stale source, a permission mistake or a broken citation. Version each report against the index and embedding model so a regression can be tied to the change that caused it.

## Where this stands in 2026

:::info Industry view

- Vector search engines such as Qdrant combine dense indexes with payload filtering and can use exact scans for small candidate sets.
- ANN quality is assessed by recall and latency on representative queries, not by assuming one complexity class guarantees relevance.
- Knowledge-base freshness, source permissions and deletion have become operational data responsibilities for RAG applications.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What are the stages of a RAG data pipeline?</summary>

Load documents → chunk (with overlap) → embed → store in a vector database → retrieve by semantic similarity.<br /><em>Lecture 14 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is an embedding and how does retrieval use it?</summary>

A dense vector where semantically similar text is nearby; retrieval ranks chunks by cosine similarity to the query embedding.<br /><em>Lecture 14 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Query q=[1,0,1,1], chunk d=[1,1,1,0]. Compute cosine similarity.</summary>

The dot product is 2 and both norms are √3, so cosine = 2/3 ≈ 0.667. Retrieval still depends on top-k, thresholds, filters, competing chunks and index recall.<br /><em>Lecture 14 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why use ANN indexes instead of exact search?</summary>

Exact scanning compares eligible vectors directly. ANN methods such as HNSW and IVF may lower latency with a measured recall, memory and build-time trade-off; sub-linear behaviour is not guaranteed for every workload.<br /><em>Lecture 14 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Name two operational concerns for an LLM knowledge base.</summary>

Any of: index freshness (re-embed changed docs), deduplication, chunk size/overlap tuning, retrieval-quality and cost monitoring.<br /><em>Lecture 14 · conceptual</em>

</details>

## Go deeper

- [Qdrant indexing](https://qdrant.tech/documentation/manage-data/indexing/) documents HNSW and payload indexes.
- [Qdrant search](https://qdrant.tech/documentation/search/search/) documents exact search, limits and filtering.
- Built from the course lecture "dm-l14-llm-pipelines" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can explain the source-to-chunk-to-index-to-retrieval lifecycle.
- [ ] I can calculate the lecture's dot product and cosine of 2/3 without calling it an automatic retrieval decision.
- [ ] I can version chunks and embeddings, propagate permissions and remove stale source revisions.
- [ ] I can compare ANN results with an exact baseline and locate a RAG failure by stage.
