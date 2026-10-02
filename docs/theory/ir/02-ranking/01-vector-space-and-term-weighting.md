---
id: ir-vector-space-term-weighting
title: "Information Retrieval · Session 5 — Vector Space and Term Weighting"
sidebar_label: "5 · Vector space"
sidebar_position: 1
slug: /theory/ir/vector-space-and-term-weighting
description: "How tf-idf and cosine turn term overlap into a ranked result list, with a runnable comparison against BM25, toy dense retrieval, fusion and reranking."
tags: [information-retrieval, tf-idf, cosine-similarity, ranking]
---

import Infographic from '@site/src/components/Infographic';
import TfIdfLab from '@site/src/components/viz/TfIdfLab';

**In one line.** Weight terms by how informative they are, then compare query and document vectors to rank candidate documents.

## The idea in plain words

Boolean retrieval answers whether a document matches a term expression. A person looking at a long result set needs a different answer: *which matching document is most useful first?* The vector-space model represents each document and query with one coordinate per term. A term with more evidence gets more weight, and cosine similarity measures how closely the weighted vectors point in the same direction.

The lecture uses **term frequency times inverse document frequency**. A term appearing often in one document is stronger evidence for that document, but a term found in nearly every document discriminates poorly. With 1,000 documents and a term in 100 of them, base-10 IDF is $\log_{10}(1000/100)=1.0$. If the term occurs three times in a document, its simple tf-idf weight is $3\times1.0=3.0$.

The lecture's second worked example uses query vector $q=[1,0,1,0]$ and document vector $d=[1,1,1,0]$. Their dot product is 2, while their lengths are $\sqrt{2}$ and $\sqrt{3}$. The cosine is $2/(\sqrt{2}\sqrt{3})\approx0.816$. Length normalisation keeps a long document from winning simply because it contains more words. It does not prove the document satisfies the user's intent; it only measures this vector representation.

<Infographic src="/img/ir/vector-space.svg" alt="The corpus gives base-10 IDF 1.0, term frequency three gives tf-idf 3.0, and vector cosine gives 0.816." caption="Two calculations from the lecture: term weighting and length-normalised vector similarity." />

:::note Beyond the lecture

The comparison with BM25, dense toy vectors, fusion and reranking below connects this classical model to a current retrieval pipeline. The fixed dense vectors are hand-designed teaching features, not embeddings from a trained model.

:::

Adjust the corpus size, document frequency and term frequency. The defaults reproduce **IDF 1.0**, **tf-idf 3.0** and the fixed cosine example **0.816**.

<TfIdfLab />

## How it works

### tf-idf

weight = tf × idf; idf = log(N/df) upweights rare, discriminating terms.

:::tip

**Worked.** N=1000, df=100 → idf = 1.0; tf=3 → tf-idf = 3.0.

:::

### Cosine similarity

Cosine of the angle (dot / magnitudes) normalises for length; rank by descending cosine.

:::tip

**Worked.** q=[1,0,1,0], d=[1,1,1,0] → cos = 0.816.

:::


## A real system that works this way

**Apache Lucene** offers several scoring models. Its [BM25Similarity API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html) documents parameters for term-frequency saturation and document-length normalisation, plus a smoothed IDF formula. BM25 is a different lexical ranking model from the lecture's simple tf-idf cosine, but it uses the same corpus evidence: term frequency, document frequency and length. Its IDF uses a natural logarithm and smoothing, so the worked base-10 IDF value should not be copied directly into BM25.

**Azure AI Search** provides a production example of combining lexical and vector result lists. Its [hybrid ranking documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) describes reciprocal rank fusion, which adds a rank-based contribution from each list. This avoids assuming a BM25 score and a vector similarity live on directly comparable scales. A later reranker can examine a smaller candidate set more carefully.

For an internal policy query such as "shipping refund damaged item", lexical matching recognises exact words in one document. A paraphrase such as "delivery fees are returned when goods arrive broken" may have no exact query term while expressing the same answer. Combining the two candidate sources can raise both documents. The toy code below makes the distinction visible and evaluates all four stages against the same two relevance labels.

## Code you can run

The first block independently verifies the lecture's two numerical examples. The logarithm base matters for the printed IDF value.

```python
from math import log10, sqrt

documents, document_frequency, term_frequency = 1000, 100, 3
idf = log10(documents / document_frequency)
weight = term_frequency * idf
query = [1, 0, 1, 0]
document = [1, 1, 1, 0]
dot = sum(a * b for a, b in zip(query, document))
query_length = sqrt(sum(value * value for value in query))
document_length = sqrt(sum(value * value for value in document))
cosine = dot / (query_length * document_length)
print(f"IDF={idf:.3f} tf-idf={weight:.3f} cosine={cosine:.3f}")
assert (round(idf, 3), round(weight, 3), round(cosine, 3)) == (1.0, 3.0, 0.816)
```

The next block runs from the repository root and uses the checked-in, dependency-free example in `scripts/ir_comparison.py`. It ranks a six-document corpus using BM25, hand-designed three-dimensional dense features, reciprocal rank fusion and a simple intent-coverage reranker. The features and reranker are **illustrative proxies**, not learned neural models. Relevant documents are IDs 0 and 2.

```python
import runpy

demo = runpy.run_path("scripts/ir_comparison.py")
rankings = demo["compare"]()
precision_at_two = demo["precision_at_two"]
for name, ranking in rankings.items():
    print(f"{name:18} top 3={ranking[:3]} P@2={precision_at_two(ranking):.2f}")
assert rankings["reranked"][:2] == [0, 2]
```

On this deliberately difficult query, BM25, the toy dense vectors and hybrid RRF each score **P@2 = 0.50**, while reranking reaches **1.00**. The hybrid top three includes both relevant documents but still places an irrelevant document second. This is one toy case, not a claim that these methods have that performance on real collections. In a deployed system, the dense features would come from an encoder and the reranker from a model or validated rule set.

## Designing with it

### Get the representation right first

The vector coordinates depend on tokenisation, normalisation and vocabulary. If the query contains `refund` but the document says `returned`, a pure lexical vector has no shared coordinate for that idea. Stemming, synonyms or semantic retrieval can bridge it, each with different failure modes. If a field contains product codes or names, exact tokens may be more valuable than broad semantic similarity. Index and query analysis still need to agree, as Session 3 showed.

The lecture's weight is intentionally simple. Raw term frequency can let repeated words dominate; variants use sublinear tf or another saturation function. A rare term gets high IDF, but a misspelled one-off token should not automatically outweigh every other signal. Document length normalisation also matters: cosine normalises vector magnitude, while BM25 models length as part of its scoring formula. Choose and tune on judged queries rather than trusting a formula because it is familiar.

### Keep candidate generation and reranking distinct

Lexical BM25 can be excellent for exact identifiers and phrasing. Dense retrieval can recognise paraphrases but may retrieve a semantically related document that lacks the required fact. Reciprocal rank fusion merges their ranks without mixing incomparable scores. A reranker can then inspect the query and each shortlisted document together. This layered design spends expensive computation only on a small set; it also makes stage-level failures visible.

| Stage | Strength | Typical miss |
| --- | --- | --- |
| Tf-idf cosine | Interpretable term weighting | Synonyms with no shared term |
| BM25 | Strong lexical baseline with length and tf handling | Paraphrases and omitted context |
| Dense candidates | Semantic similarity | Exact codes, negation and subtle conditions |
| Hybrid fusion | Finds candidates from both paths | Can still order near-misses too high |
| Reranking | More careful ordering of shortlisted results | Cannot recover a relevant document never shortlisted |

For the toy example, the dense vectors are fixed by hand and one irrelevant shipping page is intentionally close to the query. The reranker uses explicit concept coverage. That design is transparent enough to teach the stages, but it is not a substitute for an evaluated production model. Keep the demonstration's mechanism and its limits together when comparing it to real system results.

### Evaluate the actual use case

A ranking score alone is not a quality metric. Label the top results for realistic queries, including exact codes and paraphrases, then compute rank-aware measures from Session 7. Check false positives, false negatives and latency. When a retrieval system feeds an answer generator, inspect whether the selected passages contain sufficient evidence and whether the answer uses it correctly. Compare against a lexical baseline before adding dense indexing or a reranker; complexity should buy measured value.

## Diagnose the six-document comparison

The toy corpus has two relevant documents for "shipping refund damaged item". Document 0 uses the query's words: damaged parcel, refund and shipping fees. Document 2 paraphrases the same policy using delivery, returned and broken. The BM25 path ranks document 0 first, which is good, but puts document 2 below several lexical near-misses because it has no shared query token under the demo's simple analyser. That is not a BM25 bug; it is a candidate-vocabulary limitation.

The hand-designed dense features give each document three broad semantic dimensions: shipping, refund and damage. This helps document 2 enter the shortlist even though its surface words differ. It also makes document 3, a page about standard shipping time, look too close. A real encoder can make subtler mistakes of the same kind. The dense method is valuable because it expands recall, but high cosine does not establish that the answer's required conditions are present.

Hybrid RRF uses each method's ordering, not the raw score values. In the checked-in implementation, the constant is small so the ranking differences are easy to see in a six-document lesson. Production systems often use a larger constant and tune the candidate depths; the Azure documentation explains its own RRF choice. The fused top three contains both relevant documents, improving the *candidate set* for the next stage, but it still puts the generic shipping page ahead of the paraphrased refund policy.

The illustrative reranker looks for coverage of all three query concepts: shipping, refund and damage. Both relevant documents cover all three; the generic shipping page covers only one. It therefore moves document 2 above document 3, raising P@2 from 0.50 to 1.00 in this example. The rule does not understand arbitrary English. If the wording changes outside its small synonym sets, it may fail. A production reranker must be evaluated on a held-out set and checked for latency, cost and permission handling.

### Keep score scales separate

BM25 scores are not probabilities, and their magnitude depends on corpus statistics and query terms. Cosine similarity has a different range and meaning. Adding a BM25 number directly to a cosine number gives an arbitrary weighting unless scores are calibrated and the choice validated. RRF is attractive precisely because it uses rank positions instead. It has its own assumptions: a document in two mediocre positions can outrank a document that is excellent in one list, and candidate depth changes the result. Inspect fused rankings instead of assuming fusion always helps.

### Use the smallest effective pipeline

For a collection dominated by exact error codes, lexical search may already satisfy users. For a policy library with many paraphrases, dense candidates may add recall. For an answer generator, a reranker may improve the quality of the few passages sent to the model, but only if the correct passages enter the candidate pool. Build a judged sample that includes both exact and semantic cases, run the baseline and add one stage at a time. Record P@k, recall at the reranker cutoff, latency and failure examples after each stage. The six-document script teaches the mechanics; the deployment decision needs real evidence.

### Watch the corner cases in the formula

The simple lecture IDF is zero when a term appears in every document, because $\log_{10}(N/N)=0$. That is sensible as a discrimination signal: the term does not help select one document over another. It is undefined when document frequency is zero, so a query term absent from the index needs a separate path, such as no postings or a tolerant expansion from Session 3. The lab prevents zero document frequency for that reason. A practical implementation may smooth the formula and handle out-of-vocabulary terms explicitly.

Raw tf-idf also makes a term that occurs ten times weigh ten times as much as one occurrence, even when the repetitions are boilerplate. Saturating term frequency reduces that effect. A document that contains hundreds of unrelated words can still achieve a large raw dot product; cosine divides by vector lengths so the score emphasises direction rather than scale. Yet cosine is not a cure for every length problem: field boundaries, repeated navigation text and short snippets can still change the result. Remove obvious boilerplate at ingestion and evaluate fields separately where appropriate.

Query weighting deserves attention too. A short query often has one occurrence of each term, so its tf does little. IDF can still make a rare technical term dominate, which is helpful for an exact product name but harmful if it is a typo. A user who supplies a quoted phrase or an explicit filter is expressing structure beyond a bag of weighted terms; preserve that structure rather than flattening everything into one vector. The vector-space model is a ranking foundation, not a complete query language.

Finally, a cosine value is relative to the representation. A score of 0.816 for the lecture's four-dimensional vectors is exact arithmetic, but it is not a universal "good relevance" threshold. Changing tokenisation, IDF, corpus, field weights or query formulation changes the space. Compare scores within a controlled query and index version, then judge actual results with the test collection. Avoid presenting a raw score as a probability that the document answers the question.

## Where this stands in 2026

:::info Industry view

- Classical tf-idf remains the clearest model for teaching why rare terms matter and why vector length is normalised; production lexical search often uses BM25 or a tuned variant.
- Current Lucene documentation exposes BM25's term-frequency and length parameters, while Azure AI Search documents rank fusion for combined lexical and vector queries.
- A modern retrieval stack is usually evaluated in stages: candidate recall, fused ranking, reranked top results and the downstream user or answer outcome.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Define tf-idf weighting.</summary>

weight = tf × idf, idf = log(N/df) upweighting rare, discriminating terms.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> N=1000, df=100. Compute idf, and tf-idf for tf=3.</summary>

idf = log₁₀(10) = 1.0; tf-idf = 3 × 1.0 = 3.0.<br /><em>Session 5 · numeric</em>

</details>

<details>
<summary><strong>Q3.</strong> Why cosine similarity rather than raw dot product?</summary>

Cosine normalises for vector length so long and short documents compare fairly.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> q=[1,0,1,0], d=[1,1,1,0]. Compute cosine similarity.</summary>

cos = 2/(√2·√3) = 0.816.<br /><em>Session 5 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> How does the VSM improve on Boolean retrieval?</summary>

It ranks documents by relevance (cosine of weighted vectors) rather than returning an unranked exact set.<br /><em>Session 5 · conceptual</em>

</details>

## Go deeper

- [Stanford IR book: vector-space scoring](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html); the weighting and cosine model.
- [Apache Lucene BM25Similarity](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html); current lexical scoring parameters.
- [Azure AI Search hybrid scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking); a production RRF implementation.
- Built from the course lecture "ir-s5-vsm" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can compute the lecture's base-10 IDF, tf-idf and cosine values.
- [ ] I can explain how tf-idf ranks documents and what a zero shared-term vector misses.
- [ ] I can distinguish a lexical score, dense similarity, rank fusion and reranking in the tiny corpus.
- [ ] I can state why a reranker cannot fix a relevant document missing from the candidate set.
