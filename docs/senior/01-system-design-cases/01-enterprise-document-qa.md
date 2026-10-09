---
id: senior-case-document-qa
title: "Design an Enterprise Document Q&A System"
sidebar_label: "1 · Document Q&A"
sidebar_position: 1
slug: /senior/design-enterprise-document-qa
description: "A whole-system design for question answering over two million internal documents: sizing, the retrieval funnel measured on real data, permissions, context budgets, cost, failure modes and rollout."
tags: [system-design, rag, retrieval, hybrid-search, reranking, access-control, cost-estimation]
---

import Infographic from '@site/src/components/Infographic';
import RetrievalFunnelLab from '@site/src/components/viz/RetrievalFunnelLab';

**In one line.** An enterprise document Q&A system is a retrieval funnel with permissions bolted into the middle of it and a token bill at the end, so the design questions are how deep each stage looks, who is allowed to see what, and how much context the model is fed.

:::note Not from a lecture
Written for this site from the public sources listed under Further reading. The company and its numbers are an invented teaching scenario; every figure marked as measured comes from code in this chapter, and prices are placeholders you replace.
:::

## The idea in plain words

You are asked to build "ask the company's documents a question and get a sourced answer". The scenario for this chapter: 20,000 employees, two million documents (policies, contracts, wikis, tickets, slide decks), spread across systems that each have their own permissions. Answers must cite the passages they came from, and nobody may be shown or answered from a document they could not open themselves.

A model cannot read two million documents per question, so the system **retrieves** a few passages and asks the model to answer from them. The retrieval idea goes back to the RAG paper (Lewis et al., 2020), which paired a generator with a dense index of passages. Everything that makes the enterprise version hard sits around that core:

- the corpus is large, so the index size and the cost of re-embedding it matter;
- queries come in two flavours, exact ("clause HR-4471") and paraphrased ("can I keep unused holiday"), and no single retriever handles both;
- permissions differ per user, and a vector index knows nothing about users;
- the prompt is paid for per token, and long contexts are used unevenly.

<Infographic src="/img/senior/enterprise-document-qa-architecture.svg" alt="Two paths: an offline ingestion path from connectors to chunk, embed and three stores, and an online query path from authentication through retrieval, fusion, an access filter, reranking, context packing and the answer." caption="The whole design on one page, with the sizes printed by the first code block." />

## How it works

### Requirements, with numbers

| Requirement | Value used in this chapter |
| --- | --- |
| Corpus | 2,000,000 documents, about 15 chunks of 350 tokens each |
| Users | 20,000 employees, 35% active a day, 6 questions each |
| Latency | first token in about 2 seconds, full answer in under 10 |
| Freshness | a changed document searchable within an hour |
| Safety | no answer built from a document the asker cannot open; every claim cites a chunk |
| Quality | the right passage reaches the model for at least 90% of test questions |

### Back-of-envelope sizing

The first code block turns those requirements into a memory, storage and cost estimate. The headline is that **traffic is small and memory is large**: 42,000 questions a day is about 5 questions a second at peak, which one modest service handles, while 30 million vectors at 1,024 dimensions are 122.9 GB as float32. Quantising the vectors to int8 cuts that to 30.7 GB, which is why quantisation is a first-week decision rather than an optimisation. The HNSW graph (Malkov and Yashunin) adds links per vector on top; the estimator treats those as a parameter.

### Data flow

**Ingestion.** Connectors pull documents together with their access lists. Each file is parsed, split into chunks and embedded, and written to three places: the vector index, a keyword (BM25) index, and a chunk store holding text and metadata, including the groups allowed to read it. The access list travels with every chunk because it is cheaper to store than to look up later, and because a chunk without it can leak.

**Query.** The user is authenticated and their groups resolved. The question goes to the keyword index and the vector index in parallel. The two ranked lists are merged with **reciprocal rank fusion**, a rule that scores each document by the sum of `1 / (rank + 60)` over the lists it appears in; Microsoft's documentation for Azure AI Search describes the formula and notes that a small constant such as 60 performed best in experiments. Permissions are applied inside the search, a **cross-encoder** reranks the top 20, and a packer fits the best chunks into a token budget. The model answers with citations, and a final check confirms each cited chunk supports its claim.

### The decisions that matter

**1. Which retriever?** The BEIR benchmark paper found that BM25 is a robust zero-shot baseline, that dense and sparse neural retrievers are cheaper to run but often underperform other approaches, and that rerankers and late-interaction models score best on average at a higher compute cost. The second code block measures this on real questions.

| Option | Strength | Weakness | Use when |
| --- | --- | --- | --- |
| BM25 only | exact terms, codes, names; no model to host | misses paraphrase | the corpus is full of identifiers |
| Dense only | paraphrase and meaning | misses rare exact tokens | users ask in their own words |
| Hybrid with rank fusion | each covers the other's misses; no score calibration | two indexes to run | the default for an enterprise |

**2. Rerank, and how deep?** A cross-encoder reads the question and a passage together, so it is more accurate than comparing two stored vectors, but scoring a whole corpus with it is impractical, which is why the Sentence Transformers documentation uses it only on a retrieved shortlist. Depth trades recall for latency and cost. The measurement below is a warning: a generic reranker is not automatically an improvement.

**3. Where do permissions apply?**

| Option | How | Cost | Risk |
| --- | --- | --- | --- |
| Post-filter | retrieve, then drop what the user cannot see | cheap to build | can leave fewer than k results; the OpenFGA guidance suggests asking for two to three times as many |
| Pre-filter | pass allowed groups or document ids into the search | filter support in the index; stale lists matter | none if lists are fresh |

**4. How much context?** Every chunk sent costs money and, past a point, accuracy: Liu et al. ("Lost in the Middle") found that models use information at the start and end of a long context better than information in the middle. The third code block shows a packer that respects a budget, limits chunks per document, and puts the strongest chunks at both ends.

### Failure modes and mitigations

| Failure | What it looks like | Mitigation |
| --- | --- | --- |
| Evidence never retrieved | confident answer from nothing | measure retrieval hit rate separately; answer "not found" below a score threshold |
| Permission leak | a user sees a summary of a restricted file | filter inside the search; test with canary documents per group; log which chunks fed each answer |
| Stale index | answers quote a superseded policy | update on change events; store a version and date with every chunk |
| Poisoned document | text inside a file tries to instruct the model | treat retrieved text as data in the prompt; no tools in the answering step |

### Evaluation and rollout

Build a set of 200 to 500 real questions with the document that answers each, from search logs and subject-matter experts. Track three numbers separately: **retrieval hit rate** (is the evidence in the top k), **faithfulness** (does the answer follow from the chunks) and **citation precision**. Roll out in rings: the team that owns the documents, one department, then everyone, with a thumbs-down button that stores the question, the chunks and the permissions in force. See [RAG evaluation in the LLM evals course](/docs/llm-evals/llm-eval-methods) and [the RAG project](/docs/projects/enterprise-rag/session-1) for the evaluation side in depth.

## A real system that works this way

Several public sources describe pieces of this design, and none describes a whole company's system.

- **Hybrid retrieval and reranking.** Anthropic's "Contextual Retrieval" post (19 September 2024) reports that adding chunk-specific context before embedding cut the top-20 retrieval failure rate by 35% (5.7% to 3.7%); combining that with BM25 cut it by 49% (to 2.9%); and adding reranking cut it by 67% (to 1.9%). These are Anthropic's own measurements on their evaluation, not a guarantee for your corpus.
- **Permission-aware retrieval.** The OpenFGA documentation lays out pre-filtering and post-filtering as the two patterns for authorising RAG results, with the trade-offs in the table above.

## Code you can run

Everything here runs on CPU. Versions used: Python 3.14.6, sentence-transformers 6.1.0, transformers 5.18.0, datasets 5.0.1, numpy 2.5.3, run on 2 October 2026. The models (`all-MiniLM-L6-v2` and `ms-marco-MiniLM-L6-v2`, both Apache 2.0) and the SciFact data download from the Hugging Face Hub the first time.

#### 1. Sizing and cost estimator

All inputs are named parameters at the top. The prices are placeholders, not any provider's price list.

```python
DOCUMENTS = 2_000_000
CHUNKS_PER_DOCUMENT = 15
CHUNK_TOKENS = 350
EMBEDDING_DIM = 1024
HNSW_LINKS_PER_VECTOR = 32
METADATA_BYTES_PER_CHUNK = 200
EMPLOYEES = 20_000
ACTIVE_SHARE = 0.35
QUESTIONS_PER_ACTIVE_USER = 6
BUSIEST_HOUR_SHARE = 0.15
PEAK_SECOND_FACTOR = 3
WORKING_DAYS = 22
PROMPT_OVERHEAD_TOKENS = 400
QUESTION_TOKENS = 60
ANSWER_TOKENS = 350
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0

chunks = DOCUMENTS * CHUNKS_PER_DOCUMENT
GB = 1e9
float32_gb = chunks * EMBEDDING_DIM * 4 / GB
int8_gb = chunks * EMBEDDING_DIM * 1 / GB
graph_gb = chunks * HNSW_LINKS_PER_VECTOR * 4 / GB
metadata_gb = chunks * METADATA_BYTES_PER_CHUNK / GB
text_gb = chunks * CHUNK_TOKENS * 4 / GB
print(f"chunks to index: {chunks:,}")
print(f"vectors as float32: {float32_gb:,.1f} GB   as int8: {int8_gb:,.1f} GB   graph links: {graph_gb:,.1f} GB   metadata: {metadata_gb:,.1f} GB")
print(f"chunk text in the document store (about 4 bytes per token): {text_gb:,.1f} GB")
print(f"tokens to embed once: {chunks * CHUNK_TOKENS / 1e9:,.2f} billion")

daily = EMPLOYEES * ACTIVE_SHARE * QUESTIONS_PER_ACTIVE_USER
peak_qps = daily * BUSIEST_HOUR_SHARE / 3600 * PEAK_SECOND_FACTOR
print(f"\nquestions per day: {daily:,.0f}   peak questions per second: {peak_qps:.1f}")

print("\nchunks in the prompt   input tokens   cost per question   cost per month")
for k in (3, 6, 10):
    tokens_in = PROMPT_OVERHEAD_TOKENS + QUESTION_TOKENS + k * CHUNK_TOKENS
    per_question = (tokens_in * PRICE_IN_PER_MTOK + ANSWER_TOKENS * PRICE_OUT_PER_MTOK) / 1e6
    print(f"{k:20d}   {tokens_in:12,d}   {per_question:17.4f}   {per_question * daily * WORKING_DAYS:14,.0f}")
```

Read the output as a design argument. The corpus needs 30 million chunks, 122.9 GB of float32 vectors or 30.7 GB as int8, and 10.50 billion tokens to embed once. Traffic is 42,000 questions a day and 5.2 per second at the busiest moment. The money is in the prompt: at the assumed prices, six chunks per question cost \$0.0129 and \$11,947 a month, three chunks \$9,037 and ten chunks \$15,828. Choosing k is a budget decision as much as a quality one.

#### 2. The retrieval funnel on real questions

SciFact, from the BEIR collection, is 100 scientific claims, each with a labelled evidence abstract. The block builds a 1,500-document index (every relevant document plus random others), compares BM25, a dense retriever and hybrid fusion, reranks the hybrid top 10 and top 20, and then tests two ways of applying permissions. The metric is the share of questions with at least one relevant document in the top k. This is a scientific corpus, not a company's documents, and 100 questions give roughly three points of noise either way.

```python
import math
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]

query_text = {str(r["_id"]): r["text"] for r in queries}
relevant = defaultdict(set)
for r in qrels:
    if r["score"] > 0:
        relevant[str(r["query-id"])].add(str(r["corpus-id"]))
qids = sorted(relevant, key=int)[:100]

doc_text = {str(r["_id"]): (r["title"] + ". " + r["text"]).strip() for r in corpus}
must = sorted({d for q in qids for d in relevant[q]})
rng = np.random.default_rng(0)
others = sorted(set(doc_text) - set(must))
extra = rng.choice(others, size=1500 - len(must), replace=False).tolist()
doc_ids = sorted(must + extra)
texts = [doc_text[d] for d in doc_ids]
index_of = {d: i for i, d in enumerate(doc_ids)}
gold = [[index_of[d] for d in relevant[q]] for q in qids]
print(f"{len(qids)} queries, {len(doc_ids)} documents, {len(must)} relevant documents")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        self.length = np.array([len(tokens(d)) for d in docs], dtype=float)
        self.avg = self.length.mean()
        for i, d in enumerate(docs):
            for term, f in Counter(tokens(d)).items():
                self.postings[term].append((i, f))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.n = n

    def scores(self, query):
        s = np.zeros(self.n)
        for term in set(tokens(query)):
            for i, f in self.postings.get(term, []):
                norm = f + self.k1 * (1 - self.b + self.b * self.length[i] / self.avg)
                s[i] += self.idf[term] * f * (self.k1 + 1) / norm
        return s


bm25 = BM25(texts)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_vec = encoder.encode(texts, batch_size=64, normalize_embeddings=True)
q_vec = encoder.encode([query_text[q] for q in qids], normalize_embeddings=True)

bm_order = [np.argsort(-bm25.scores(query_text[q]), kind="stable") for q in qids]
de_order = [np.argsort(-(doc_vec @ q_vec[i]), kind="stable") for i in range(len(qids))]


def rrf(orders, depth=100, k=60):
    score = defaultdict(float)
    for order in orders:
        for rank, d in enumerate(order[:depth]):
            score[int(d)] += 1.0 / (k + rank + 1)
    fused = sorted(score, key=lambda d: (-score[d], d))
    rest = [int(d) for d in orders[0] if int(d) not in score]
    return np.array(fused + rest)


hy_order = [rrf([bm_order[i], de_order[i]]) for i in range(len(qids))]


def best_rank(order, targets):
    position = {int(d): r + 1 for r, d in enumerate(order)}
    return min(position[t] for t in targets)


ranks = {
    "BM25": [best_rank(bm_order[i], gold[i]) for i in range(len(qids))],
    "dense (MiniLM)": [best_rank(de_order[i], gold[i]) for i in range(len(qids))],
    "hybrid (RRF, k=60)": [best_rank(hy_order[i], gold[i]) for i in range(len(qids))],
}
cuts = (1, 5, 10, 20, 50)
print("\nshare of queries with a relevant document in the top k")
print(f"{'retriever':22s}" + "".join(f"  top {k:<3d}" for k in cuts))
for name, r in ranks.items():
    print(f"{name:22s}" + "".join(f"  {np.mean(np.array(r) <= k):7.2f}" for k in cuts))

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
depth = 20
pairs = [(query_text[qids[i]], texts[int(d)]) for i in range(len(qids)) for d in hy_order[i][:depth]]
scores = reranker.predict(pairs, batch_size=64).reshape(len(qids), depth)


def reranked_rank(limit):
    out = []
    for i in range(len(qids)):
        order = [int(d) for d in hy_order[i][:limit][np.argsort(-scores[i][:limit], kind="stable")]]
        hits = [order.index(t) + 1 for t in gold[i] if t in order]
        out.append(min(hits) if hits else 0)
    return np.array(out)


print("\nhybrid top k, then cross-encoder rerank: relevant document in the top 1, 3, 5")
for limit in (10, 20):
    r = reranked_rank(limit)
    print(f"  rerank the top {limit}:  " + "   ".join(f"{np.mean((r > 0) & (r <= k)):.2f}" for k in (1, 3, 5)))
print(f"  no rerank, plain hybrid:  {np.mean(np.array(ranks['hybrid (RRF, k=60)']) <= 1):.2f}   {np.mean(np.array(ranks['hybrid (RRF, k=60)']) <= 3):.2f}   {np.mean(np.array(ranks['hybrid (RRF, k=60)']) <= 5):.2f}")
print(f"  ceiling set by retrieval depth 20: {np.mean(np.array(ranks['hybrid (RRF, k=60)']) <= 20):.2f}")

visible_share = 0.30
acl_rng = np.random.default_rng(1)
post_hit = pre_hit = 0
post_returned = []
for i in range(len(qids)):
    visible = acl_rng.random(len(doc_ids)) < visible_share
    visible[gold[i]] = True
    order = hy_order[i]
    top10 = [int(d) for d in order[:10]]
    kept = [d for d in top10 if visible[d]]
    post_returned.append(len(kept))
    post_hit += any(d in gold[i] for d in kept[:5])
    allowed = [int(d) for d in order if visible[int(d)]]
    pre_hit += any(d in gold[i] for d in allowed[:5])
print(f"\nuser sees {visible_share:.0%} of documents (plus the evidence for their question)")
print(f"post-filter: retrieve 10, drop unauthorised: {np.mean(post_returned):.1f} results left on average, hit@5 {post_hit / len(qids):.2f}")
print(f"pre-filter: filter inside the search, then take 5: hit@5 {pre_hit / len(qids):.2f}")
```

What it shows, in order. BM25 is stronger in the top 5 and top 10 (0.91 and 0.93 against 0.86 and 0.90 for dense) but flat after rank 10 (0.93 at 20), while dense is slightly ahead at rank 1 (0.75 against 0.71) and better at depth (0.98 at 50); hybrid fusion reaches 0.98 by rank 20 and 0.99 by 50, so **fusion recovers what each retriever misses**. The reranker is the surprise: reranking the top 10 gives 0.91 in the top 5 and the top 20 gives 0.89, against 0.92 for plain hybrid. On this data a generic MS MARCO cross-encoder did not help, and its extra latency would have bought nothing. That is a result about one model on one corpus, and the correct lesson is to **measure a reranker on your own questions** rather than adding one by default.

Permissions behave as the table predicted. When the user can see 30% of documents, retrieving 10 and dropping the rest leaves 3.7 results on average and a hit rate at 5 of 0.94; filtering inside the search gives 0.97.

The lab replays these ranks. Its defaults (hybrid, depth 20, rerank on, 5 chunks) show 0.89, the printed figure; switch the rerank off for 0.92.

<RetrievalFunnelLab />

<Infographic src="/img/senior/enterprise-document-qa-funnel.svg" alt="Four panels: hit rates for BM25, dense and hybrid retrieval; the reranker result; post-filter against pre-filter; and the monthly cost of three, six and ten chunks." caption="The measured funnel and its costs, with the figures printed by the first two code blocks." />

#### 3. Packing context under a token budget

The chunk texts are an invented leave-policy example; the token counts come from the SmolLM2 tokenizer. The packer takes chunks by score, skips any that would overflow the budget or give one document more than two slots, and orders the survivors so the best are first and last.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

retrieved = [
    ("leave-policy", 0.91, "Employees accrue 1.75 days of annual leave per month. Unused leave up to 10 days carries over into the next leave year; anything above that lapses on 31 March."),
    ("leave-policy", 0.88, "Carry-over requests above 10 days need written approval from the department head before 28 February and are granted only for approved project work."),
    ("travel-policy", 0.74, "Flights over six hours may be booked in premium economy. Hotel limits are set per city band and listed in the travel rate card."),
    ("leave-faq", 0.71, "Q: Does unused leave carry over? A: Up to 10 days carry over. See the leave policy for the lapse date and the approval route for exceptions."),
    ("expenses", 0.55, "Expense claims must be submitted within 30 days with receipts. Meals are reimbursed up to the daily allowance for the city band."),
    ("onboarding", 0.52, "New joiners receive their leave balance in the first payroll cycle, prorated from the start date to the end of the leave year."),
    ("leave-policy", 0.49, "Sick leave is separate from annual leave and is not carried over. A medical certificate is required after three consecutive days."),
    ("travel-policy", 0.41, "Travel booked less than seven days ahead needs a manager's note. Rail is preferred for trips under four hours."),
]

BUDGET = 150
MAX_PER_DOCUMENT = 2


def pack(chunks, budget, max_per_document):
    chosen, used, per_doc = [], 0, {}
    for doc, score, text in sorted(chunks, key=lambda c: -c[1]):
        n = len(tokenizer(text)["input_ids"])
        if per_doc.get(doc, 0) >= max_per_document or used + n > budget:
            continue
        chosen.append((doc, score, n))
        per_doc[doc] = per_doc.get(doc, 0) + 1
        used += n
    return chosen, used


def edge_order(chosen):
    front, back = [], []
    for i, item in enumerate(chosen):
        (front if i % 2 == 0 else back).append(item)
    return front + back[::-1]


print("all eight chunks:", sum(len(tokenizer(t)["input_ids"]) for _, _, t in retrieved), "tokens")
chosen, used = pack(retrieved, BUDGET, MAX_PER_DOCUMENT)
print(f"budget {BUDGET} tokens, at most {MAX_PER_DOCUMENT} chunks per document -> {used} tokens used")
print("kept by score:", [(d, s, n) for d, s, n in chosen])
print("order sent to the model (best at both ends):", [f"{d}:{s}" for d, s, _ in edge_order(chosen)])
```

All eight chunks would cost 242 tokens; the budget of 150 keeps four chunks using 137. The third leave-policy chunk (score 0.49) is dropped by the per-document cap, so a single long document cannot crowd out the others. The order sent to the model puts the 0.91 chunk first and the 0.88 chunk last.

## Designing with it

**Cost estimate as a formula.**

`monthly cost = questions per month x (input tokens x input price + output tokens x output price) + vector memory x memory price + embedding tokens x embedding price (once, and again per re-index)`

with input tokens equal to prompt overhead plus question plus `k x chunk tokens`. Only the first term moves with usage; the others are close to fixed.

**What I would build first.**

1. Hybrid retrieval over one department's documents with permissions from the source system, no reranker.
2. The 200-question evaluation set and the three metrics, before any tuning.
3. Citations and a "not found" path.
4. Only then a reranker, if the measurement says it helps, and pre-filtering if post-filtering leaves too few results.
5. Quantisation and the ingestion pipeline for the whole corpus.

## Where this stands in 2026

:::info Industry view

- **Hybrid retrieval plus a reranking option is the standard shape.** The Azure AI Search documentation, last updated in September 2026, treats keyword and vector queries fused by RRF as the normal hybrid mode, with semantic ranking as a later step.
- **Permission awareness is now a product feature, not an afterthought.** The same page describes Azure AI Search as underpinning Foundry IQ, a managed knowledge layer for "permission-aware knowledge bases for agents".
- **Rerankers are not free wins.** BEIR reports the best average zero-shot results for them; this chapter's measurement shows a small one that did not help. Evaluate on your data.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why is memory, not queries per second, the main sizing driver for this system?</summary>

At 42,000 questions a day the peak is about 5 per second, which is small. The index is large: 30 million chunks at 1,024 dimensions are 122.9 GB as float32 (30.7 GB as int8) plus graph links and metadata, and it must sit in memory for fast search. The footprint is set by corpus size and vector precision, so quantisation and dimension choices dominate the infrastructure bill.

</details>

<details>
<summary><strong>Q2.</strong> Hybrid retrieval reached 0.98 at depth 20 while BM25 stayed at 0.93 and dense at 0.92. What does that tell you about the two retrievers?</summary>

They miss different questions. BM25 fails when the question's wording differs from the document; dense retrieval fails on rare exact tokens. Fusing the ranked lists lets each cover the other's misses, and rank fusion needs no calibration between the two score scales.

</details>

<details>
<summary><strong>Q3.</strong> The reranker lowered the top-5 hit rate from 0.92 to 0.91 (top 10) and 0.89 (top 20). Do you drop it?</summary>

Not on this evidence alone: 100 questions carry about three points of noise, the corpus is scientific rather than enterprise, and one small model was tested. But the default of "add a reranker" is not supported. I would repeat the test on real questions, try a stronger or domain-tuned reranker, and keep it only if the hit rate gain justifies its latency and cost.

</details>

<details>
<summary><strong>Q4.</strong> A user can see 30% of the corpus. Compare post-filtering and pre-filtering.</summary>

Post-filtering retrieves first, so about 70% of the shortlist is discarded: 3.7 of 10 survived here and the hit rate at 5 was 0.94. Pre-filtering restricts the search itself and gave 0.97. Post-filtering needs over-fetching (two to three times) and wastes work; pre-filtering needs the index to support filters and the access lists to be fresh.

</details>

<details>
<summary><strong>Q5.</strong> The monthly bill doubled after launch with no change in users. Where do you look?</summary>

Tokens per question: the number of chunks, chunk size, prompt overhead, answer length, retries and any new feature that adds context. In the estimator, going from three to ten chunks raises the monthly cost from \$9,037 to \$15,828 with traffic unchanged. Alert on tokens per question and cap chunks and tokens per answer.

</details>

<details>
<summary><strong>Q6.</strong> How do you prove no permission leak exists?</summary>

You cannot prove it, but you can make it detectable: plant canary documents readable by one group, query as members of other groups and assert they never appear in chunks or answers; filter inside the search; log the chunk ids that fed each answer; and sync access lists from the source system on change events rather than nightly.

</details>

## Further reading

- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)](https://arxiv.org/abs/2005.11401): the original generator plus dense-index design.
- [Anthropic, "Contextual Retrieval" (19 September 2024)](https://www.anthropic.com/news/contextual-retrieval): the failure-rate figures quoted above.
- [Liu et al., "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172): position effects in long prompts.
- [Microsoft Learn, "Hybrid search scoring (RRF)" in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking): the fusion formula and constant.
- [Thakur et al., "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models"](https://arxiv.org/abs/2104.08663): BM25, dense and reranking compared.
- [Sentence Transformers, "Retrieve and re-rank"](https://www.sbert.net/examples/applications/retrieve_rerank/README.html): why cross-encoders are used on a shortlist.
- [OpenFGA, "RAG authorization"](https://openfga.dev/docs/modeling/agents/rag-authorization): pre-filter and post-filter patterns.
- [Malkov and Yashunin, hierarchical navigable small world graphs (arXiv 1603.09320)](https://arxiv.org/abs/1603.09320): the HNSW index.
- Site chapters: [retrieval theory](/docs/theory/ir/neural-retrieval-and-reranking), [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), [the enterprise RAG project](/docs/projects/enterprise-rag/session-1), [architectural patterns for ML systems](/docs/theory/seml/architectural-patterns).

## Check yourself

- I can size a document Q&A system from documents, chunk size and dimensions, and say which number sets the infrastructure floor.
- I can explain why hybrid retrieval with rank fusion covers the misses of BM25 and dense search.
- I can measure a retrieval stage as a hit rate at k and name the ceiling the retrieval depth imposes.
- I can explain why a reranker needs testing on my own questions before it is trusted.
- I can compare post-filtering and pre-filtering for permissions and say what each costs.
- I can budget prompt tokens, cap chunks per document and order context so the strongest evidence sits at the edges.
- I can write the cost formula for the system and name the term that moves with usage.
