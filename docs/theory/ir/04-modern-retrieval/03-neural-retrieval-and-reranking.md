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

## The idea in plain words

Classical lexical ranking uses terms and corpus statistics. Neural retrieval learns representations so a query can match a passage even when the two express the same idea with different words. The lecture's main architecture splits the job into **retrieve** and **rerank**. A dual encoder processes query and document separately, allowing document vectors to be stored and searched quickly. A cross encoder examines the query and one candidate document together, allowing richer interaction at higher serving cost.

Neither replaces exact lexical evidence. A semantic model can retrieve a broad paraphrase but miss a product code, a date or a negation. BM25 can recognise those exact strings but may miss a true synonym. A hybrid system can gather candidates from both paths, combine their ranks and rerank a manageable subset. Its quality depends on what entered that subset. A cross encoder cannot promote a relevant passage it was never given.

The lecture also points to **retrieval-augmented generation (RAG)**: retrieve passages, then provide them to a generator as context for an answer. Retrieval can make an answer grounded in external material, but passing a passage does not guarantee the answer uses it faithfully. The retrieval result, the generated claim and the cited source need separate checks. See the site's [RAG explanation](/docs/genai/rag) and [enterprise RAG build](/docs/projects/enterprise-rag/session-1) for the downstream system.

<Infographic src="/img/ir/neural-retrieval.svg" alt="BM25 and separate-encoder semantic candidates are fused with RRF, then a joint query-document stage reranks a shortlist; the teaching example improves P at two from 0.50 to 1.00." caption="The toy six-document example demonstrates stage roles. Its dense vectors and final reranker are hand-designed proxies, not trained models." />

:::note Beyond the lecture

The four-method runnable comparison, candidate-cutoff failure and RAG evidence checks extend the lecture's outline. Its dense features and reranker are deliberately transparent toy mechanisms and must not be read as a neural model benchmark.

:::

Select BM25, illustrative dense vectors, hybrid RRF or the intent-coverage reranker. The fixed six-document corpus has relevant IDs **0 and 2**. The reranked default places them first and second, giving **P@2 = 1.00** and **AP = 1.000**. The table shows every rank and relevance label.

<RetrievalPipelineLab />

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

## Go deeper

- [Dense Passage Retrieval paper](https://arxiv.org/abs/2004.04906); separate query and passage encoders.
- [ColBERT paper](https://arxiv.org/abs/2004.12832); token-level late interaction.
- [Azure AI Search hybrid scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking); current fusion and reranking product workflow.
- [Original RAG paper](https://arxiv.org/abs/2005.11401); retrieval combined with generation.
- Built from the course lecture "ir-s15-neural-ir" (Lecture Library series).
- Built from the course lecture "ir-s16-review" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can place BM25, a dual encoder, RRF and a cross encoder in a retrieve-then-rerank pipeline.
- [ ] I can explain why reranking cannot recover a document absent from its shortlist.
- [ ] I can compare the four toy rankings without calling their dense features a trained model.
- [ ] I can separate retrieval quality, passage sufficiency and answer faithfulness in RAG.
- [ ] I can trace Session 16's complete arc from Boolean candidates and vector ranking through web, multimodal and neural retrieval to evaluation.
