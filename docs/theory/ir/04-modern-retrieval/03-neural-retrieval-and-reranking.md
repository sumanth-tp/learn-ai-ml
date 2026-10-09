---
id: ir-neural-retrieval-reranking
title: "Information Retrieval · Session 15 — Neural Retrieval and Reranking"
sidebar_label: "15 · Neural IR"
sidebar_position: 3
slug: /theory/ir/neural-retrieval-and-reranking
description: "How dual encoders, lexical retrieval, fusion, cross-encoder reranking and RAG fit into one measured retrieval pipeline."
tags: [information-retrieval, dense-retrieval, reranking, rag]
---

import Infographic from '@site/src/components/Infographic';
import RetrievalPipelineLab from '@site/src/components/viz/RetrievalPipelineLab';

**In one line.** Fast lexical and dense methods find candidates; a more expensive interaction model can reorder a shortlist, but only retrieved evidence can reach the answer stage.

:::tip Before you start

**You should already know**

- How BM25 scores a document and what nDCG@10 measures ([Session 5](/docs/theory/ir/vector-space-and-term-weighting) and [Session 7](/docs/theory/ir/evaluating-ranked-retrieval)).
- That a dual encoder puts two kinds of item in one vector space ([Session 13](/docs/theory/ir/multimodal-retrieval-and-clip)).

**Reading time:** about 60 minutes, plus about 2 minutes to run the code (the first run downloads SciFact and one small model).

**After this chapter you can**

- Fuse a keyword ranking and a dense ranking with reciprocal rank fusion, by hand.
- Say which kinds of query favour BM25 and which favour a dense encoder, and test it on your own data.
- Explain why a hybrid is a safe average, not a guaranteed win.

:::

## In 30 seconds

Two helpers search a library. The first matches your exact words, so it is superb when your words appear in the book and useless when they do not. The second matches meaning, so it finds a book that says the same thing in other words, but it can fumble an exact code or a very short query.

A hybrid asks both and merges their lists. Because it listens to both, it is rarely the worst. It also usually fails to beat whichever helper is better for that particular question. A reranker then reads the top few results carefully, but it can only reorder what the first stage returned.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| BM25 | Keyword scoring with saturation and length control | A claim word appears twice in an abstract |
| Dense retrieval | Matching by the closeness of learned vectors | `all-MiniLM-L6-v2`, 384 numbers per text |
| Cross-encoder | A model that reads the query and one document together | Scores a shortlist of 20 |
| Reciprocal rank fusion (RRF) | Merging lists by adding $1/(60 + \text{rank})$ from each | Rank 1 gives 0.0164 |
| Hybrid retrieval | A first stage that uses both a keyword and a dense list | BM25 plus dense, merged by RRF |
| nDCG@10 | A rank-aware score of the top 10, 1.0 is perfect | 0.670 for BM25 on SciFact claims |
| Word overlap | Share of a claim's words found in its right abstract | 0.38 means 38% |
| Candidate recall | Share of right documents the first stage returns at all | A cap on every later stage |

## The idea in plain words

Classical lexical ranking uses terms and corpus statistics. Neural retrieval learns representations so a query can match a passage even when the two express the same idea with different words. The main architecture splits the job into **retrieve** and **rerank**. A dual encoder processes query and document separately, allowing document vectors to be stored and searched quickly. A cross encoder examines the query and one candidate document together, allowing richer interaction at higher serving cost.

Neither replaces exact lexical evidence. A semantic model can retrieve a broad paraphrase but miss a product code, a date or a negation. BM25 can recognise those exact strings but may miss a true synonym. A hybrid system can gather candidates from both paths, combine their ranks and rerank a manageable subset. Its quality depends on what entered that subset. A cross encoder cannot promote a relevant passage it was never given.

The same architecture feeds **retrieval-augmented generation (RAG)**: retrieve passages, then provide them to a generator as context for an answer. Retrieval can make an answer grounded in external material, but passing a passage does not guarantee the answer uses it faithfully. The retrieval result, the generated claim and the cited source need separate checks. See the site's [RAG explanation](/docs/genai/rag) and [enterprise RAG build](/docs/projects/enterprise-rag/session-1) for the downstream system.

<Infographic src="/img/ir/neural-retrieval.svg" alt="BM25 and separate-encoder semantic candidates are fused with RRF, then a joint query-document stage reranks a shortlist; the teaching example improves P at two from 0.50 to 1.00." caption="The toy six-document example demonstrates stage roles. Its dense vectors and final reranker are hand-designed proxies, not trained models." />

:::note Added for this site

The four-method runnable comparison, candidate-cutoff failure and RAG evidence checks extend the core outline. Its dense features and reranker are deliberately transparent toy mechanisms and must not be read as a neural model benchmark.

:::

Select BM25, illustrative dense vectors, hybrid RRF or the intent-coverage reranker. The fixed six-document corpus has relevant IDs **0 and 2**. The reranked default places them first and second, giving **P@2 = 1.00** and **AP = 1.000**. The table shows every rank and relevance label.

<RetrievalPipelineLab />

## Worked example, step by step

A query has three candidate documents X, Y and Z, and one extra document W. BM25 ranks X first, Y second, Z third. The dense encoder ranks Y first, W second, X third, and does not return Z at all. RRF with the usual constant 60 adds $1/(60 + \text{rank})$ from each list that contains the document.

1. $X$: $1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266$.
2. $Y$: $1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522$.
3. $Z$: only in the BM25 list, $1/63 = 0.015873$. $W$: only in the dense list, $1/62 = 0.016129$.
4. Fused order: Y, X, W, Z. Y wins by a hair because both lists like it.
5. nDCG@10 with one right document is $1/\log_2(\text{rank} + 1)$: rank 1 gives 1.000, rank 2 gives 0.631, rank 3 gives 0.500, rank 4 gives 0.431, and absent gives 0.
6. Suppose the right document is X in one query, Y in a second and Z in a third. BM25 ranks them 1, 2, 3, scoring $(1 + 0.631 + 0.5)/3 = 0.710$. Dense ranks them 3, 1, absent, scoring $(0.5 + 1 + 0)/3 = 0.500$. The hybrid ranks them 2, 1, 4, scoring $(0.631 + 1 + 0.431)/3 = 0.687$.

In words: the hybrid is between the two lists in this toy, not above both. It beats dense and loses to BM25, because BM25 placed the right document higher than the hybrid did in two of the three queries. The experiment checks whether real data behave like this.

<Infographic src="/img/ir-enrich/ir2-neural-overlap.svg" alt="Bars of nDCG at 10 for BM25, dense and hybrid in three groups of claims by word overlap with the right abstract, and a table for claims cut to their first 3, 5 and 8 words." caption="Look first at the lowest third: dense scores 0.402 against 0.284 for BM25. In the highest third the order reverses, and the hybrid wins neither." />

## How it works

### Representation vs interaction

- **Dual-encoder (Siamese)**; Encode separately → dot product → dense retrieval + ANN (fast).
- **Cross-encoder**; BERT over query+doc → accurate re-ranking (slow).

:::note

**Combine.** Lexical (BM25) + semantic: cheap retrieval → neural re-ranker.

:::

### The frontier

Generative IR / RAG retrieve passages and feed an LLM to generate a grounded, cited answer. Conversational/QA-based IR (ChatGPT-style) handles multi-turn needs.


The whole of IR in one arc; from **Boolean retrieval** to **neural IR**.

### Classic core → neural

S1–8: Boolean, dictionary/tolerant, index/compression, VSM, classification/clustering, evaluation. S9–16: web search, crawling, link analysis, CLIR, CLIP, recommenders, neural IR/RAG.

### Ideas that recur

- **Vectors + cosine**; tf-idf, CLIP, dense embeddings all rank by cosine.
- **Combine signals**; content + links + semantics + personalisation; evaluated by MAP/NDCG.

:::tip

**Formulas.** tf-idf = tf·log(N/df); cosine; PageRank = (1−d)/N + d·Σ PR/L; AP = mean precision at relevant ranks.

:::


## A real system that works this way

**Azure AI Search** documents a current [hybrid ranking pipeline](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking): full-text BM25 and vector queries each produce ranked results, reciprocal rank fusion combines their positions, and a semantic ranker can rerank the merged text-bearing candidates. Its [vector-search quickstart](https://learn.microsoft.com/en-us/azure/search/search-get-started-vector) walks through a hotel query where semantic reranking changes which hotel appears first after hybrid fusion. This is a named production product workflow with public documentation. Its internal semantic ranker should not be equated with the simple intent rule in our code.

The [Dense Passage Retrieval paper](https://arxiv.org/abs/2004.04906) is an original research example of separate query and passage encoders for open-domain question answering. A separate representation permits indexed passage vectors and fast candidate search. [ColBERT](https://arxiv.org/abs/2004.12832) illustrates a further design point: late interaction retains token-level representations and performs more interaction than a single-vector dual encoder, while avoiding the full per-candidate joint encoding pattern of a cross encoder. These are model families, not a claim that one wins every domain.

An enterprise policy search might combine exact policy identifiers with paraphrase recall, then rerank passages for the user's question. Permission filtering must apply before a result is exposed. If a RAG assistant follows, it should retain passage IDs and versioned citations so an answer can be checked against the retrieved evidence. A good retrieval score is not a licence to invent a missing exception.

## Code you can run

The first block reuses the checked-in six-document comparison from Sessions 5 and 7. Run it from the repository root. The dense vectors and reranker are **hand-designed teaching proxies**, not outputs from a trained model. The real BM25 computation, toy dense ranking, RRF and reranking use the same two relevance labels.

```python
import runpy

demo = runpy.run_path("scripts/ir_comparison.py")
rankings = demo["compare"]()
relevant = demo["RELEVANT"]

def average_precision(ranking):
    found = 0
    total = 0.0
    for position, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            found += 1
            total += found / position
    return total / len(relevant)

for method, ranking in rankings.items():
    print(f"{method:18} top 3={ranking[:3]} P@2={demo['precision_at_two'](ranking):.2f} AP={average_precision(ranking):.3f}")
assert rankings["reranked"][:2] == [0, 2]
assert average_precision(rankings["reranked"]) == 1.0
```

The second block isolates the **candidate cutoff** problem. If the BM25 first stage sends only its top two IDs, document 2 never reaches a reranker, no matter how good that reranker is. Reranking can reorder a shortlist; it cannot add an unseen document.

```python
import runpy

demo = runpy.run_path("scripts/ir_comparison.py")
rankings = demo["compare"]()
bm25_shortlist = rankings["BM25"][:2]
relevant = demo["RELEVANT"]
best_possible = sorted(bm25_shortlist, key=lambda doc_id: doc_id not in relevant)
print("BM25 shortlist:", bm25_shortlist)
print("best possible rerank of that shortlist:", best_possible)
print("P@2 ceiling:", demo["precision_at_two"](best_possible))
assert 2 not in bm25_shortlist
assert demo["precision_at_two"](best_possible) == 0.5
```

The example reranker improves the larger hybrid candidate set, not this truncated BM25 list. In a real service, tune candidate depth against recall, reranker latency and cost using many judged queries. One query's perfect AP is a mechanism demonstration, not a deployment result.

### The worked example in code

This block reproduces the fusion scores and the three nDCG averages.

```python
import numpy as np

bm25 = {"X": 1, "Y": 2, "Z": 3}
dense = {"Y": 1, "W": 2, "X": 3}
fused = {d: sum(1 / (60 + r[d]) for r in (bm25, dense) if d in r) for d in "XYZW"}
print({d: round(s, 6) for d, s in fused.items()}, sorted(fused, key=fused.get, reverse=True))
gain = lambda rank: 0.0 if rank is None else 1 / np.log2(rank + 1)
hybrid_rank = {d: i + 1 for i, d in enumerate(sorted(fused, key=fused.get, reverse=True))}
for name, ranks in (("BM25", (1, 2, 3)), ("dense", (3, 1, None)), ("hybrid", (hybrid_rank["X"], hybrid_rank["Y"], hybrid_rank["Z"]))):
    print(name, round(sum(gain(r) for r in ranks) / 3, 3))
```

**Reading the output.** The scores print as X 0.032266, Y 0.032522, Z 0.015873, W 0.016129, in the order Y, X, W, Z. The three averages print as 0.71, 0.5 and 0.687.

### An experiment: where does each method win?

The senior case and the reranking chapter already compare BM25, dense and a cross-encoder on SciFact. This block asks a different question: for which claims does each method win, and what happens to short queries? It uses the 300 SciFact test claims, 5,183 abstracts, `rank-bm25` BM25 and `all-MiniLM-L6-v2` dense retrieval, with RRF to merge them. It splits the claims into thirds by word overlap, the share of a claim's words that appear in its best right abstract. Then it cuts every claim to its first 3, 5 and 8 words. Versions used: Python 3.14.6, sentence-transformers 6.1.0, rank-bm25 0.2.2, datasets 5.0.1, NumPy 2.5.3, on CPU. The model is Apache 2.0 and the SciFact data is CC BY-SA 4.0. The run takes about a minute, with the data and model cached.

```python
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
relevant = defaultdict(set)
for row in load_dataset("BeIR/scifact-qrels")["test"]:
    if row["score"] > 0:
        relevant[str(row["query-id"])].add(str(row["corpus-id"]))
ids = np.array([d["_id"] for d in corpus])
texts = [d["title"] + ". " + d["text"] for d in corpus]
qids = [q["_id"] for q in queries if q["_id"] in relevant]
qtexts = [q["text"] for q in queries if q["_id"] in relevant]

analyse = CountVectorizer(stop_words="english").build_analyzer()
bm25 = BM25Okapi([analyse(t) for t in texts])
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
documents = encoder.encode(texts, batch_size=64, normalize_embeddings=True)

def ndcg(scores):
    out = []
    for qid, row in zip(qids, scores):
        top = ids[np.argsort(-row, kind="stable")[:10]]
        gains = np.array([d in relevant[qid] for d in top], dtype=float)
        discount = 1 / np.log2(np.arange(2, 12))
        ideal = discount[: min(10, len(relevant[qid]))].sum()
        out.append((gains * discount).sum() / ideal)
    return np.array(out)

def rrf(first, second, k=60):
    fused = np.zeros_like(first)
    for scores in (first, second):
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
        fused += 1 / (k + ranks + 1)
    return fused

def systems(question_texts):
    lexical = np.array([bm25.get_scores(analyse(q)) for q in question_texts])
    dense = encoder.encode(question_texts, normalize_embeddings=True) @ documents.T
    return {"BM25": ndcg(lexical), "dense": ndcg(dense), "hybrid RRF": ndcg(rrf(lexical, dense))}

full = systems(qtexts)
overlap = []
position = {d: i for i, d in enumerate(ids)}
for qid, text in zip(qids, qtexts):
    words = set(analyse(text))
    best = max(len(words & set(analyse(texts[position[d]]))) / len(words) for d in relevant[qid] if d in position)
    overlap.append(best)
overlap = np.array(overlap)
edges = np.quantile(overlap, [1 / 3, 2 / 3])
bucket = np.digitize(overlap, edges)
print(f"{'slice of the 300 claims':<46}{'BM25':>7}{'dense':>7}{'hybrid':>8}")
print(f"{'all claims':<46}" + "".join(f"{v.mean():>{w}.3f}" for v, w in zip(full.values(), (7, 7, 8))))
for b, name in enumerate(("lowest third of word overlap", "middle third", "highest third")):
    mask = bucket == b
    label = f"{name} ({overlap[mask].min():.2f} to {overlap[mask].max():.2f})"
    print(f"{label:<46}" + "".join(f"{v[mask].mean():>{w}.3f}" for v, w in zip(full.values(), (7, 7, 8))))

for k in (3, 5, 8):
    short = systems([" ".join(q.split()[:k]) for q in qtexts])
    print(f"{f'claim cut to its first {k} words':<46}" + "".join(f"{v.mean():>{w}.3f}" for v, w in zip(short.values(), (7, 7, 8))))
```

The output of the run:

```text

slice of the 300 claims                          BM25  dense  hybrid
all claims                                      0.670  0.648   0.686
lowest third of word overlap (0.00 to 0.38)     0.284  0.402   0.360
middle third (0.38 to 0.64)                     0.765  0.697   0.776
highest third (0.67 to 1.00)                    0.949  0.838   0.913
claim cut to its first 3 words                  0.360  0.254   0.358
claim cut to its first 5 words                  0.471  0.363   0.470
claim cut to its first 8 words                  0.579  0.526   0.587
```

**Reading the output.** Each cell is mean nDCG@10. The first row is all claims. The next three rows are thirds of the claims by overlap, with the overlap range in brackets. The last three rows rerun everything with the claim cut short.

**Line by line.**

- `rrf` turns each score matrix into ranks with a double `argsort`, then adds $1/(60 + \text{rank})$ from each list. This is the arithmetic of the worked example.
- `overlap` uses the labelled right abstract, so it is a way to slice the results, not something a system can compute at query time.
- `ndcg` divides by the best possible value for each claim, using up to 10 relevant documents.

### What the numbers say

Over all 300 claims BM25 scored 0.670, dense 0.648 and the hybrid 0.686. The BM25 figure matches the `rank-bm25` result of 0.670 in [Session 5](/docs/theory/ir/vector-space-and-term-weighting), an independent check of the pipeline. The averages hide a sharp split. When a claim shared few words with its right abstract (overlap up to 0.38), dense scored 0.402 against 0.284 for BM25. When overlap was high (0.67 and above), BM25 scored 0.949 against 0.838 for dense. Each method wins where its theory says it should.

The hybrid is the surprise. In both outer thirds it lost to the better single method: 0.360 against 0.402 in the lowest, 0.913 against 0.949 in the highest. It won only in the middle, by 0.011 (0.776 against 0.765), which is within what I would treat as a tie. Its overall lead of 0.016 over BM25 comes from the low-overlap third (+0.076 over BM25), partly cancelled by the high-overlap third (-0.036). It is never the worst and is rarely the best, so a hybrid works as insurance.

The second belief to drop is that dense retrieval suits short queries. Cut to 3 words, BM25 scored 0.360 and dense 0.254, the hybrid 0.358. At 8 words it was 0.579, 0.526 and 0.587. The gap narrows as the claim grows, because a longer claim gives the encoder more to work with.

Limits: one encoder (a 6-layer model), one collection of scientific claims, no query is a real keyword query, no intervals (each third has about 100 claims, so the large outer gaps of 0.1 or more are likely real and the 0.011 middle gap is not), and the overlap measure uses the answer. Nothing here tests a cross-encoder; see [reranking with a cross-encoder](/docs/genai/rag-advanced/contextual-retrieval-and-reranking) for that.

## Designing with it

### Put each model where its cost fits

A dual encoder computes document vectors ahead of time. At request time it encodes a query and performs nearest-neighbour search. That scales candidate generation across a large collection. A cross encoder jointly processes a query with each candidate, so it can compare terms and conditions directly but must run per pair. Use it on a limited shortlist or accept substantial latency. A late-interaction model is another point on this spectrum, retaining multiple token vectors and more matching detail at additional storage and serving cost.

BM25 remains a valuable baseline. Its term matching can be essential for names, codes, quoted phrases and exact constraints. Dense retrieval can capture paraphrases but may confuse related topics with the required answer. Keep stage results separately observable: lexical candidates, dense candidates, fused ranking, reranked order and final answer context. If the result is wrong, those traces show where the relevant passage disappeared.

| Stage | Stored representation | Query-time work | Failure to watch |
| --- | --- | --- | --- |
| BM25 | Inverted term index | Term lookup and lexical scoring | Paraphrases with no shared terms |
| Dual encoder | Document vectors | Query vector and ANN search | Exact codes, fine distinctions |
| RRF fusion | Two ordered candidate lists | Rank combination | Poor input depth or list weighting |
| Cross encoder | Text-bearing candidates | Joint score per candidate | Relevant item absent from shortlist |
| Answer generator | Selected passages | Produce answer and citations | Unsupported or misread claims |

### Guard candidate recall

Evaluate recall at the reranker's cutoff, not only P@2 after reranking. If the right passage is missing from the first-stage union, no downstream scoring can recover it. Increase candidate depth only if the extra recall justifies latency and noise. Test exact identifiers, paraphrases, long documents, multilingual wording and policy exceptions separately. Different query types may need different candidate sources or weights.

The six-document script intentionally puts document 2 in the hybrid top three but not in BM25's top two. The reranker can promote it only when it receives at least that broader candidate set. A model's apparent failure can therefore be an upstream recall failure. Log candidate IDs, method membership and cutoff values with each evaluation run.

### Keep scores and ranks distinct

BM25 scores, cosine similarities and semantic-ranker scores have different scales. Adding them raw can make one source dominate for accidental numerical reasons. Reciprocal rank fusion combines positions and avoids that direct scale comparison, but it has its own constant and depth choices. Session 5's code uses a small constant for a six-document lesson; the Azure product has its own implementation and settings. If using score blending, calibrate and test it explicitly.

### Treat RAG as another measured stage

For RAG, check whether the retrieved context contains all facts needed for the question. Then check whether the generated answer states only facts supported by that context and cites the right passage. A result may be relevant in topic but lack the exact condition, date or exception. Source freshness and permissions apply before generation, and a citation should point to the document version actually retrieved. See the site's [RAG lesson](/docs/genai/rag) for the generate stage; this chapter focuses on retrieval quality feeding it.

Conversational retrieval adds context from earlier turns. Resolve references such as "what about the second policy?" carefully, then evaluate the rewritten search query and retrieved passages. Conversation history can help disambiguate but can also bias a later question toward an old topic. Keep the interpreted query visible in logs and allow correction when the user meant something else.

## Trace the six-document shortlist

The query is `shipping refund damaged item`, and relevant documents are IDs 0 and 2. Document 0 repeats the query's terms. BM25 ranks it first. Document 2 paraphrases the answer with delivery, returned fees and broken goods, so the toy lexical analyser gives it little direct overlap. BM25 places several near-misses above it. The hand-designed three-dimensional features recognise the paraphrase but also overvalue a generic shipping page, document 3. These are controlled examples of lexical and semantic failure modes, not measured behaviour of a neural encoder.

RRF combines the two orderings. The hybrid top three includes documents 0, 3 and 2, so both relevant documents become available for a final stage. The intent-coverage reranker asks whether a candidate covers shipping, refund and damage concepts, then orders 0 and 2 first. P@2 rises to 1.00 and AP to 1.000 for this single toy query. The lab lets the reader switch stages and see which document moves. Its ranking arrays are the outputs of the checked-in comparison script, and its relevance labels match the code.

If only BM25's top two were sent onward, they would be 0 and 4. Document 2 would be absent. An ideal reranker restricted to those two could at best put 0 first and 4 second, keeping P@2 at 0.50. That is why candidate recall must be measured at the cutoff. A cross encoder can be powerful but has no way to inspect a passage it never receives.

### Compare model interaction patterns

A dual encoder can precompute each document's vector independently of the query. The query vector is compared with many stored vectors. This is fast and indexable, but compresses each passage into a representation that may blur subtle conditions. A cross encoder sees query and candidate together, enabling token-level attention and richer relevance scoring, but repeats computation for every candidate. Late-interaction approaches such as ColBERT retain token-level vectors for documents and combine them with query tokens at search time, offering another quality-cost balance. Choose based on measured relevance, memory and latency, not a label such as "neural" alone.

The model's training pairs and negatives matter. If all negatives are obviously unrelated, a dense retriever may learn broad topic matching and fail to distinguish a refund policy from a shipping timetable. Hard negatives that share vocabulary or topic can teach the distinction, but they can also accidentally include a truly relevant passage. Validate labels and evaluate on held-out query types. A reranker can likewise overfit the candidate distribution it was trained on, so changes to first-stage retrieval may require renewed evaluation.

### Analyse a hybrid failure

Suppose an exact product code is present in one policy but the dense route retrieves semantically broad pages about a different product. The lexical result might be excellent in its own list yet lose rank after fusion if many near-duplicate dense candidates dominate. Inspect method membership, candidate depth and duplicate groups. Apply exact product constraints as filters where appropriate, rather than hoping the semantic model will infer them. Conversely, if the relevant passage uses a paraphrase absent from lexical postings, ensure the dense path has enough depth to include it.

RRF makes raw score scales less troublesome, but it cannot judge content quality by itself. A document ranked moderately in both lists can outrank a document that is excellent in one. Deduplication and sensible list weighting may help, but every adjustment should be tested on the same query set. Session 7's AP and NDCG show ordering; candidate recall and failure examples show what the order missed.

### Keep the answer stage honest

When retrieved passages feed a generator, the system has two related but distinct questions: were the needed passages retrieved, and did the answer use them correctly? A passage about a general refund may be high in semantic similarity but omit a shipping-fee exception. An answer generator could confidently assert the wrong rule, even while citing that passage. Require support for each material claim and check citations at the passage level. If no retrieved evidence supports a requested fact, the answer should expose that uncertainty rather than invent a citation.

Freshness is another link in the chain. A policy indexed last week may have changed today. Store source version and retrieval time, refresh important documents, and ensure permission changes remove content from both vector and lexical indexes. A high-quality reranker cannot compensate for stale or unauthorised evidence. Evaluate the complete path with real task examples, then localise failures by stage before replacing a model.

### Decide what to build first

Start with a lexical baseline and a judged query set. Add a dense candidate route if paraphrase recall is measurably weak. Fuse results and inspect which queries improve or regress. Add a reranker if candidate recall is already adequate but ordering is poor. For an answer system, check passage sufficiency and generated-claim support after retrieval improves. This staged approach makes the added computation accountable. The toy comparison is useful because every list and label is inspectable; a production decision needs more queries, real encoders and measured costs.

## Where this stands in 2026

:::info Industry view

- Azure AI Search documents current BM25/vector fusion with RRF and an optional semantic reranking stage, illustrating the retrieve-then-rerank architecture in a real product.
- Neural IR now spans single-vector dual encoders, late-interaction designs and joint cross encoders; each trades representation cost for matching detail.
- RAG systems require retrieval and answer evaluation separately. Retrieved context creates an opportunity for grounding but does not guarantee a faithful answer.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Choosing between BM25 and dense on one average | 0.670 against 0.648 looks decisive | Split by query type. Dense won the low-overlap third by 0.118 and lost the high-overlap third by 0.111 |
| Assuming a hybrid beats both | It uses more information | It lost to the better method in both outer thirds. Treat it as insurance, and measure it |
| Expecting dense retrieval to help very short queries | Meaning should beat missing words | At 3 words dense scored 0.254 against 0.360 for BM25 |
| Using an answer-based slice as a router | It separated the methods cleanly | Overlap uses the right abstract, which is unknown at query time. Use a cheap query feature instead, such as length or presence of codes |
| Spending on a reranker before measuring candidate recall | Reranking is the clever part | A reranker only reorders what the first stage returned. Measure recall at the cutoff first |

## Practice questions

<details>
<summary><strong>Q1.</strong> What does neural IR add over tf-idf/BM25?</summary>

Learned semantic matching; capturing synonyms and paraphrase via embeddings, beyond exact lexical overlap.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast dual-encoder (Siamese) and cross-encoder models.</summary>

Dual-encoders encode query and document separately → dot product → fast dense retrieval (ANN); cross-encoders (BERT over query+doc) are accurate but slow; used for re-ranking.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Contrast lexical and semantic matching.</summary>

Lexical (BM25) is precise for exact terms; semantic (dense embeddings) handles synonyms/paraphrase but can drift; modern systems combine them.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> What is RAG?</summary>

Retrieval-Augmented Generation: retrieve relevant passages and feed them to an LLM to generate a grounded, cited answer.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Describe the standard retrieve-then-re-rank pipeline.</summary>

A cheap first stage (BM25 or dense dual-encoder) retrieves top candidates, then an expensive cross-encoder re-ranks them for final relevance.<br /><em>Session 15 · conceptual</em>

</details>

Comprehensive synthesis questions (Sessions 1-16).

<details>
<summary><strong>Q1.</strong> Outline the IR course arc from classic to modern.</summary>

Classic core (S1–8): Boolean → dictionary/tolerant → index/compression → VSM → classification/clustering → evaluation; modern (S9–16): web search, crawling, link analysis, CLIR, CLIP, recommenders, neural IR.<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What single idea underlies tf-idf, CLIP and dense retrieval?</summary>

Representing items and queries as vectors and ranking by cosine similarity in an embedding space.<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Trace how one query flows across modules.</summary>

Normalise (S3) → match in inverted index (S2/S4) → rank by tf-idf cosine (S5) or neural re-ranker (S15) → boost by PageRank (S11) → evaluate by MAP/NDCG (S7).<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Which formulas should you memorise for the comprehensive exam?</summary>

tf-idf = tf·log(N/df); cosine = q·d/(‖q‖‖d‖); PageRank = (1−d)/N + d·Σ PR/L; AP = mean precision at relevant ranks.<br /><em>Session 16 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> How do modern IR systems combine lexical and neural methods?</summary>

Cheap lexical/dense retrieval of top candidates, then an expensive neural cross-encoder re-ranker; RAG then grounds an LLM answer on the results.<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q11.</strong> (Easy) Compute the RRF score of a document ranked 1st by BM25 and 3rd by the dense retriever, with constant 60.</summary>

$1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266$. A document that appears in only one list gets just one of the two terms, so a document that is good in both lists beats a document that is excellent in one.

</details>

<details>
<summary><strong>Q12.</strong> (Medium) Dense retrieval won the low-overlap third (0.402 against 0.284) and lost the high-overlap third (0.838 against 0.949). Explain both results.</summary>

When the claim and its evidence share few words, BM25 has little to match and the encoder can still link them by meaning, so dense wins. When they share many words, BM25 gets most of the signal directly, while the encoder compresses the abstract into one vector and can blur exact terms, so BM25 wins. The result matches the usual advice to combine the two, but it also shows that a combination gives up something in each regime.

</details>

<details>
<summary><strong>Q13.</strong> (Stretch) The hybrid scored 0.686 overall but lost to the best single method in two of three slices. How can both be true, and what would you check before shipping it?</summary>

Both are true because of how the slices add up. Against BM25 the hybrid gained 0.076 in the low-overlap third, 0.011 in the middle and lost 0.036 in the high-overlap third, so the overall mean rose by 0.016 even though it was not the best in the outer thirds. Before shipping, check that the gain over BM25 (0.016) survives a bootstrap interval, that candidate recall at the reranker cutoff improved, and how the added latency and index cost compare with the gain.

</details>

## Go deeper

- [Dense Passage Retrieval paper](https://arxiv.org/abs/2004.04906); separate query and passage encoders.
- [ColBERT paper](https://arxiv.org/abs/2004.12832); token-level late interaction.
- [Azure AI Search hybrid scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking); current fusion and reranking product workflow.
- [Original RAG paper](https://arxiv.org/abs/2005.11401); retrieval combined with generation.
- [BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models](https://arxiv.org/abs/2104.08663) (opened 2026-10-09); its abstract reports BM25 as a robust baseline against dense, late-interaction and re-ranking systems.
- [Wadden et al.: Fact or Fiction, Verifying Scientific Claims](https://arxiv.org/abs/2004.14974) (opened 2026-10-09); the SciFact claims and evidence abstracts used in the experiment.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (opened 2026-10-09); the encoder used, Apache 2.0.
- Built from the course lectures "ir-s15-neural-ir" and "ir-s16-review" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can place BM25, a dual encoder, RRF and a cross encoder in a retrieve-then-rerank pipeline.
- [ ] I can explain why reranking cannot recover a document absent from its shortlist.
- [ ] I can compare the four toy rankings without calling their dense features a trained model.
- [ ] I can separate retrieval quality, passage sufficiency and answer faithfulness in RAG.
- [ ] I can trace Session 16's complete arc from Boolean candidates and vector ranking through web, multimodal and neural retrieval to evaluation.
- [ ] I can compute reciprocal rank fusion and nDCG@10 for a small example by hand.
- [ ] I can say which claims favour BM25 and which favour a dense encoder, and quote the measured split.
- [ ] I can explain why the hybrid did not win the outer thirds and what it is good for instead.
- [ ] I can say why a dense encoder is not automatically better for short queries, with the measured numbers.

## Where to go next

Next: the [question bank](/docs/theory/ir/question-bank), which revises the whole course. Related: [reranking with a cross-encoder](/docs/genai/rag-advanced/contextual-retrieval-and-reranking), and the [RAG overview](/docs/genai/rag) for what happens after retrieval.
