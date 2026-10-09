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

:::tip Before you start

**You should already know**

- What an embedding vector is and what cosine similarity measures ([Session 5, vector space and term weighting](/docs/theory/ir/vector-space-and-term-weighting)).
- Why a pipeline stage needs a stable identity so a retry does not duplicate data ([Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery)).

**Reading time:** about 45 minutes, plus about a minute to run the experiment.

**After this chapter you can**

- Count the chunks, words and embeddings a chunking choice produces before you run it.
- Measure retrieval recall for several chunking strategies and say which cost was worth paying.
- Explain why the way you count top results can reverse the ranking of strategies.

:::

## In 30 seconds

A question-answering system cannot read your whole library each time. It cuts the documents into pieces, turns each piece into a list of numbers that stands for its meaning, and later finds the pieces closest to the question. The cutting is a data decision, and it changes what the system can find.

Think of an index card for each page of a book versus one for each paragraph. Page cards are easy to file and miss details. Paragraph cards are precise and there are many more of them.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Chunk | One piece of a document that is indexed on its own | 100 words |
| Overlap | Words repeated between neighbouring chunks | 50 shared words |
| Embedding | A vector that stands for the meaning of a text | 384 numbers |
| Top-k | The k best-scoring items returned | The best 5 chunks |
| Recall at k | Share of relevant documents found in the top k | 0.88 |
| Pooling by document | Keep the best chunk per document before cutting at k | 5 distinct documents |
| Token limit | The longest input the model reads, longer text is cut off | 256 word pieces |
| Index freshness | Delay from source change to searchable chunk | 10 minutes |


## The idea in plain words

A retrieval-augmented generation application depends on a current, authorised collection of source material. Documents arrive from files, help centres or databases; the pipeline extracts their text, creates chunks, computes embeddings, indexes them and later retrieves candidates for a question. The model can only ground an answer in what the retrieval step actually supplies. A beautiful prompt cannot repair an outdated, duplicated or inaccessible knowledge base.

The core sequence is **load → chunk → embed → store → retrieve**. Treat each arrow as a data contract. Loading preserves document identity, source version and permission. Chunking preserves enough context to make a passage meaningful. Embedding stores both the vector and the model version that produced it. Indexing supports search under filters and updates. Retrieval decides which chunks can enter a prompt, often after ranking and a relevance check. Downstream answer generation and citation checking make the full RAG system, but this chapter focuses on the data pipeline beneath them.

<Infographic src="/img/dm/llm-pipelines.svg" alt="A RAG data pipeline loads and cleans documents, chunks them, embeds and indexes the chunks, then retrieves filtered candidates; the worked cosine between two four-dimensional vectors is two thirds." caption="A vector score is one stage of retrieval, not a guarantee that a chunk is used in an answer." />

The worked numerical example uses a query vector **[1, 0, 1, 1]** and a chunk vector **[1, 1, 1, 0]**. Their dot product is 2 and each norm is √3, so cosine similarity is **2/3 ≈ 0.667**. That is a correct score. It does not establish that this chunk is retrieved: top-k selection, a threshold, other candidate scores, metadata filters, access control and approximate-index recall can all change the returned set.

:::note Added for this site

The course material covers embeddings, vector stores, cosine and ANN indexes. The sections below add source permissions, stable chunk IDs, embedding-version migrations, deletion and freshness, filtered search, exact-search comparisons, retrieval evaluation and a measured comparison of chunking strategies.

:::

:::note Correction

It is sometimes said that ANN search is always sub-linear. Whether an index is faster, and how much recall it gives up, depends on the data, the parameters and the filters, so measure recall and latency for the chosen index and workload.

:::

The default lab reproduces the worked calculation: **dot = 2**, both norms are √3, and **cosine = 0.667**. Change one chunk component or the illustrative threshold to see how the score and a threshold decision move. The threshold is not a universal relevance rule; it must be chosen on labelled queries and the actual corpus.

<CosineRetrievalLab />

## Worked example, step by step

One abstract has an 8-word title and a 240-word body. Count what four chunking choices produce.

1. **Whole document.** 1 chunk of 8 + 240 = 248 words.
2. **100-word windows, no overlap.** Windows start at word 0, 100 and 200, so 3 chunks of 100, 100 and 40 words: 240 words in all.
3. **100-word windows, step 50.** Windows start at 0, 50, 100 and 150, so 4 chunks of 100, 100, 100 and 90 words: 390 words, which is 390 / 240 = 1.625 times the text.
4. **Title on every no-overlap chunk.** 240 + 3 x 8 = 264 words, 10% more than the text.
5. **Top-5 crowding.** If the overlapping chunks of a relevant document score highest, all 4 can take 4 of the 5 result slots, leaving 1 slot for every other document. Pooling by document first keeps 5 distinct documents instead.
6. **Cosine of the worked vectors.** A query [1, 0, 1, 1] against a chunk [1, 1, 1, 0] gives 2 / (sqrt 3 x sqrt 3) = 0.667, as shown above.

In words: more overlap means more words to embed and store, and a result list counted in chunks can fill up with copies of one document. The first block below prints steps 1 to 4.

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

Reproduce the cosine result with standard-library Python. Explicitly reject a zero vector because its cosine is undefined.

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

### The worked example in code

This block counts the chunks and words for the 240-word abstract with the same windowing rule the experiment uses.

```python
def windows(words, size, step):
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - size + step), step)]

body = ["w"] * 240
title = ["t"] * 8
cases = {
    "whole": [" ".join(title + body)],
    "100 no overlap": windows(body, 100, 100),
    "100 step 50": windows(body, 100, 50),
    "title on each": [" ".join(title) + " " + c for c in windows(body, 100, 100)],
}
for name, chunks in cases.items():
    print(name, len(chunks), sum(len(c.split()) for c in chunks), [len(c.split()) for c in chunks])
```

**Reading the output.** It prints `whole 1 248`, `100 no overlap 3 240`, `100 step 50 4 390` with sizes 100, 100, 100 and 90, and `title on each 3 264`. These are steps 1 to 4.

### An experiment on chunking

Which chunking strategy finds the right document most often, and what does it cost? The block below takes the SciFact scientific abstracts: the 283 abstracts that are labelled relevant to the 300 test queries plus 250 random others, 533 in all. It cuts them five ways, then scores each strategy with BM25 (word matching) and with a small embedding model, `all-MiniLM-L6-v2`. It reports recall at 5 in two ways: counting the top 5 chunks, and keeping the best chunk of each document first so that the top 5 are distinct documents.

Versions used: Python 3.14.6, sentence-transformers 6.1.0, scikit-learn 1.9.1, datasets 5.0.1, NumPy 2.5.3. The corpus is a small sample, so recall is higher than it would be on all 5,183 abstracts. It runs in about a minute on a CPU, plus a one-off download of the model and data.

```python
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]
relevant = defaultdict(set)
for row in qrels:
    if row["score"] > 0:
        relevant[str(row["query-id"])].add(str(row["corpus-id"]))
qids = [q["_id"] for q in queries if q["_id"] in relevant]
qtexts = [q["text"] for q in queries if q["_id"] in relevant]
needed = set().union(*relevant.values())
rng = np.random.default_rng(14)
others = [d for d in corpus if d["_id"] not in needed]
docs = [d for d in corpus if d["_id"] in needed] + [others[i] for i in rng.choice(len(others), 250, replace=False)]

def windows(words, size, step):
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - size + step), step)]

strategies = {
    "whole abstract": lambda d: [d["title"] + ". " + d["text"]],
    "40 words, no overlap": lambda d: windows(d["text"].split(), 40, 40),
    "100 words, no overlap": lambda d: windows(d["text"].split(), 100, 100),
    "100 words, overlap 50": lambda d: windows(d["text"].split(), 100, 50),
    "100 words + title on each": lambda d: [d["title"] + ". " + c for c in windows(d["text"].split(), 100, 100)],
}
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
query_vectors = model.encode(qtexts, normalize_embeddings=True)
tokens = np.array([len(model.tokenizer(d["title"] + ". " + d["text"])["input_ids"]) for d in docs])
print("documents", len(docs), "queries", len(qids), "model limit", model.max_seq_length, "mean tokens", round(tokens.mean()), "share over the limit", round((tokens > model.max_seq_length).mean(), 3))
counter = CountVectorizer(token_pattern=r"[a-z0-9]+", stop_words="english")

def recall_at(scores, owner, k=5, pooled=False):
    hits = []
    for qid, row in zip(qids, scores):
        if pooled:
            best = {}
            for i in np.argsort(-row, kind="stable"):
                best.setdefault(owner[i], row[i])
                if len(best) == k:
                    break
            found = set(best)
        else:
            found = {owner[i] for i in np.argsort(-row, kind="stable")[:k]}
        hits.append(len(found & relevant[qid]) / len(relevant[qid]))
    return float(np.mean(hits))

print(f"{'strategy':26}{'chunks':>7}{'words':>8}{'BM25':>7}{'dense':>7}{'dense, 5 docs':>15}")
for name, split in strategies.items():
    chunks, owner = [], []
    for d in docs:
        for c in split(d):
            chunks.append(c)
            owner.append(d["_id"])
    tf = counter.fit_transform(chunks).tocsr().astype(float)
    length = np.asarray(tf.sum(axis=1)).ravel()
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
    rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
    tf.data = tf.data * 2.2 / (tf.data + 1.2 * (0.25 + 0.75 * length[rows] / length.mean())) * idf[tf.indices]
    sparse = ((counter.transform(qtexts) > 0).astype(float) @ tf.T).toarray()
    vectors = model.encode(chunks, batch_size=64, normalize_embeddings=True)
    dense = query_vectors @ vectors.T
    words = sum(len(c.split()) for c in chunks)
    print(f"{name:26}{len(chunks):7d}{words:8d}{recall_at(sparse, owner):7.3f}{recall_at(dense, owner):7.3f}{recall_at(dense, owner, pooled=True):15.3f}")
```

The output of the run:

```text
documents 533 queries 300 model limit 256 mean tokens 353 share over the limit 0.724
strategy                   chunks   words   BM25  dense  dense, 5 docs
whole abstract                533  118466  0.855  0.884          0.884
40 words, no overlap         3025  111534  0.816  0.859          0.896
100 words, no overlap        1393  111534  0.824  0.861          0.891
100 words, overlap 50        1946  182184  0.809  0.848          0.908
100 words + title on each    1393  130211  0.832  0.879          0.903
```

**Reading the output.** `chunks` and `words` are the size of the index to build and embed. `BM25` and `dense` are recall at 5 counted over chunks. `dense, 5 docs` is recall at 5 distinct documents after keeping each document's best chunk. For the whole-abstract row the two dense numbers are equal because each document has one chunk. The first line shows that 72.4% of the abstracts are longer than the model's 256-token limit.

**Line by line.**

- `windows(words, size, step)` slides a window of `size` words forward by `step`. A step smaller than the size creates overlap.
- `tf.data * 2.2 / (tf.data + 1.2 * (0.25 + 0.75 * length[rows] / length.mean())) * idf[tf.indices]` is the BM25 weight with k1 = 1.2 and b = 0.75, applied to every non-zero cell.
- `best.setdefault(owner[i], row[i])` keeps the first, and so best, score seen for each document while walking the ranking from the top.

### What the numbers say

Counted in chunks, no chunking strategy beat embedding the whole abstract. The whole abstract scored 0.884 with the dense model, against 0.859, 0.861, 0.848 and 0.879 for the four chunked variants. BM25 agreed: 0.855 for whole abstracts against 0.809 to 0.832. This is a surprise given that 72.4% of abstracts exceed the model's input limit, so the whole-abstract vector represents only the first part of most abstracts.

The overlap row explains why counting matters. With overlap of 50 the dense score fell to 0.848 counted in chunks, the lowest of all, because neighbouring chunks of the same document filled several of the five slots. Pooled by document the same strategy scored 0.908, the highest. Every chunked variant beat the whole abstract once pooled (0.891 to 0.908 against 0.884).

Context was cheaper than overlap. Putting the title on each 100-word chunk raised dense recall from 0.861 to 0.879, and pooled recall from 0.891 to 0.903, for 16.7% more words (130,211 against 111,534). Overlap 50 cost 63.3% more words (182,184) for the best pooled score, a gain of 0.017 over the plain 100-word chunks.

Limits: a 533-document sample, abstracts rather than long manuscripts, one embedding model, one seed for the distractors, and recall measured at the document level because these labels do not mark the evidence sentence. Re-run on your own documents.

<Infographic src="/img/dm-enrich/dm2-chunking-recall.svg" alt="Dense recall at 5 for five chunking strategies counted over chunks and over distinct documents, with the words each index must embed." caption="Look first at how the two bars for the overlap strategy swap places: counting chunks and counting documents give opposite answers." />

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Chunking by default, whatever the documents | Every RAG tutorial chunks | Measure against the whole document first. Counted in chunks, whole abstracts scored 0.884 against 0.848 to 0.879 |
| Counting top-k over chunks when the unit is a document | The index holds chunks, so k counts chunks | Pool by document before cutting, or the same document fills the slots. Overlap 50 went from 0.848 to 0.908 |
| Adding overlap to fix lost context | A shared boundary seems safe | Overlap cost 63.3% more words to embed. A title on each chunk cost 16.7% and recovered most of the loss |
| Ignoring the embedding model's input limit | The model accepts any text | Text beyond the limit is dropped. 72.4% of these abstracts exceeded 256 tokens |
| Trusting a score from another model | 0.667 is 0.667 | Compare scores only within one model and index version |

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

<details>
<summary><strong>Q6.</strong> (Medium) A 300-word document is cut into 100-word windows with step 50. How many chunks and how many words are embedded?</summary>

Windows start at 0, 50, 100, 150 and 200, because the rule runs while the start is below 300 - 100 + 50 = 250. That gives 5 chunks of 100 words, so 500 words are embedded, 1.67 times the 300 words of text.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) In the experiment, dense recall at 5 over chunks was lowest for the overlap strategy, and highest when pooled by document. Explain, and say which number to use when users see documents.</summary>

With overlap, neighbouring chunks of the same document are near-duplicates, so a good document can occupy several of the five slots and push others out. Pooling by document keeps only the best chunk per document, so the five slots hold five distinct documents. Use the pooled number when the user sees documents, and the chunk-level number when the model reads five chunks as context, because in that case duplicated text really does waste the prompt.

</details>

## Go deeper

- [Qdrant indexing](https://qdrant.tech/documentation/manage-data/indexing/) documents HNSW and payload indexes.
- [Qdrant search](https://qdrant.tech/documentation/search/search/) documents exact search, limits and filtering.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), opened 2026-10-09: inputs are truncated at 256 word pieces, the output is 384-dimensional, licence Apache 2.0.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact), opened 2026-10-09: 5,183 corpus documents, 1,109 queries, licence CC BY-SA 4.0. The experiment uses the 300 test queries with labelled abstracts.
- Built from the course lecture "dm-l14-llm-pipelines" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can explain the source-to-chunk-to-index-to-retrieval lifecycle.
- [ ] I can calculate the dot product of 2 and the cosine of 2/3 without calling it an automatic retrieval decision.
- [ ] I can version chunks and embeddings, propagate permissions and remove stale source revisions.
- [ ] I can compare ANN results with an exact baseline and locate a RAG failure by stage.
- [ ] I can count the chunks and words that a window size and a step produce for a given document.
- [ ] I can explain why overlap looked worst when counting chunks and best when pooling by document.
- [ ] I can say what an embedding model's input limit does to a long document, and why a title on each chunk is a cheap fix.

## Where to go next

Next: [Lecture 15, data privacy and governance](/docs/mlops/data/privacy-and-governance), which decides which chunks a person may see. Related: [Session 5, vector space and term weighting](/docs/theory/ir/vector-space-and-term-weighting), where the same SciFact abstracts test BM25 against tf-idf.
