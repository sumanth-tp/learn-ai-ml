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

:::tip Before you start

**You should already know**

- What an inverted index and a postings list are ([Session 2](/docs/theory/ir/boolean-retrieval)).
- What precision and recall mean ([Session 1](/docs/theory/ir/what-information-retrieval-is)).

**Reading time:** about 45 minutes, plus a minute to run the code.

**After this chapter you can**

- Compute idf, tf-idf and BM25's saturating term weight by hand.
- Say which weighting choices move retrieval quality and which barely do.
- Run a parameter grid on a real test collection and read it without over-claiming.

:::

## In 30 seconds

Some words help you find a document and some do not. "The" appears everywhere and tells you nothing. "Anisotropy" appears in a handful of abstracts and tells you a lot. TF-IDF scores a document by rewarding words that are frequent in it but rare in the collection.

BM25 adds two refinements. Repeating a word helps less and less, like the first coffee of the day compared with the fifth. And a long document is not allowed to win only because it has more words in it.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Term frequency (tf) | How often a word occurs in one document | `baggage` appears 3 times |
| Document frequency (df) | How many documents contain a word | 100 of 1,000 documents |
| Inverse document frequency (idf) | A rarity score: higher for words in fewer documents | $\log_{10}(1000/100) = 1.0$ |
| Cosine similarity | Angle-based match between two weighted vectors, ignoring length | 0.816 for the worked vectors |
| Sublinear tf | Damping repeats, for example $1 + \ln(\text{tf})$ | tf 10 counts as 3.303, not 10 |
| Saturation | A ceiling on how much repeating a word can add | BM25's tf weight never exceeds $k_1 + 1$ |
| $k_1$, $b$ | BM25's two knobs: how fast tf saturates, and how strongly length is penalised | $k_1 = 1.2$, $b = 0.75$ |
| nDCG@10 | A rank-aware score for the top 10 results, 1.0 is perfect | 0.669 in the experiment below |


## The idea in plain words

Boolean retrieval answers whether a document matches a term expression. A person looking at a long result set needs a different answer: *which matching document is most useful first?* The vector-space model represents each document and query with one coordinate per term. A term with more evidence gets more weight, and cosine similarity measures how closely the weighted vectors point in the same direction.

The classic weight is **term frequency times inverse document frequency**. A term appearing often in one document is stronger evidence for that document, but a term found in nearly every document discriminates poorly. With 1,000 documents and a term in 100 of them, base-10 IDF is $\log_{10}(1000/100)=1.0$. If the term occurs three times in a document, its simple tf-idf weight is $3\times1.0=3.0$.

A second worked example uses query vector $q=[1,0,1,0]$ and document vector $d=[1,1,1,0]$. Their dot product is 2, while their lengths are $\sqrt{2}$ and $\sqrt{3}$. The cosine is $2/(\sqrt{2}\sqrt{3})\approx0.816$. Length normalisation keeps a long document from winning simply because it contains more words. It does not prove the document satisfies the user's intent; it only measures this vector representation.

<Infographic src="/img/ir/vector-space.svg" alt="The corpus gives base-10 IDF 1.0, term frequency three gives tf-idf 3.0, and vector cosine gives 0.816." caption="Two calculations: term weighting and length-normalised vector similarity." />

:::note Added for this site

The comparison with BM25, dense toy vectors, fusion and reranking below connects this classical model to a current retrieval pipeline. The fixed dense vectors are hand-designed teaching features, not embeddings from a trained model.

:::

Adjust the corpus size, document frequency and term frequency. The defaults reproduce **IDF 1.0**, **tf-idf 3.0** and the fixed cosine example **0.816**.

<TfIdfLab />

## Worked example, step by step

A word occurs 1, 2 and then 10 times in a document. How much should each count add to the score? Compare three rules, with BM25's $k_1 = 1.2$ and length normalisation off.

1. **Raw tf** adds exactly the count: 1, 2 and 10. Ten repeats count ten times as much as one.
2. **Sublinear tf** adds $1 + \ln(\text{tf})$: $1 + \ln 2 = 1.693$ and $1 + \ln 10 = 3.303$. Ten repeats count a little over three times as much.
3. **BM25** adds $\text{tf} \times (k_1 + 1) / (\text{tf} + k_1)$. For tf = 1 that is $2.2 / 2.2 = 1.000$. For tf = 2 it is $4.4 / 3.2 = 1.375$. For tf = 10 it is $22 / 11.2 = 1.964$.
4. As tf grows without limit, BM25's weight approaches $k_1 + 1 = 2.2$ and never passes it.

In words: all three agree that more repeats help, and they disagree about how much. BM25 says ten repeats are worth under twice as much as one, which blunts keyword stuffing. The first block below prints these numbers.

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

**Apache Lucene** offers several scoring models. Its [BM25Similarity API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html) documents parameters for term-frequency saturation and document-length normalisation, plus a smoothed IDF formula. BM25 is a different lexical ranking model from the simple tf-idf cosine, but it uses the same corpus evidence: term frequency, document frequency and length. Its IDF uses a natural logarithm and smoothing, so the worked base-10 IDF value should not be copied directly into BM25.

**Azure AI Search** provides a production example of combining lexical and vector result lists. Its [hybrid ranking documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) describes reciprocal rank fusion, which adds a rank-based contribution from each list. This avoids assuming a BM25 score and a vector similarity live on directly comparable scales. A later reranker can examine a smaller candidate set more carefully.

For an internal policy query such as "shipping refund damaged item", lexical matching recognises exact words in one document. A paraphrase such as "delivery fees are returned when goods arrive broken" may have no exact query term while expressing the same answer. Combining the two candidate sources can raise both documents. The toy code below makes the distinction visible and evaluates all four stages against the same two relevance labels.

## Code you can run

The first block independently verifies the two numerical examples. The logarithm base matters for the printed IDF value.

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

### The worked example in code

This block prints the three weighting rules for tf of 1, 2 and 10.

```python
import numpy as np

tf = np.array([1, 2, 10])
k1 = 1.2
print("raw tf          ", tf.tolist())
print("sublinear 1+ln  ", np.round(1 + np.log(tf), 3).tolist())
print("BM25 saturation ", np.round(tf * (k1 + 1) / (tf + k1), 3).tolist())
print("BM25 ceiling    ", k1 + 1)
```

**Reading the output.** The raw row is `[1, 2, 10]`, the sublinear row is `[1.0, 1.693, 3.303]`, the BM25 row is `[1.0, 1.375, 1.964]` and the ceiling is 2.2, matching steps 1 to 4.

### An experiment on a real test collection

Which choice matters more: the tf-idf variant, or BM25's two knobs? The block below uses the 300 test claims of SciFact with their labelled abstracts. It scores four tf-idf variants with scikit-learn, then builds BM25 on a sparse matrix and scores a grid of seven values of $k_1$ and five of $b$. For a cross-check it also runs the `rank-bm25` library at its defaults. Every score is nDCG@10, averaged over the claims.

My BM25 uses the smoothed IDF $\ln(1 + (N - n + 0.5) / (n + 0.5))$ that Lucene documents. The `rank-bm25` library uses an unsmoothed IDF with a floor, so the two give slightly different numbers.

Versions used: Python 3.14.6, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.3, rank-bm25 0.2.2. The run takes about 10 seconds.

```python
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]
relevant = defaultdict(set)
for row in qrels:
    if row["score"] > 0:
        relevant[str(row["query-id"])].add(str(row["corpus-id"]))
ids = np.array([d["_id"] for d in corpus])
texts = [d["title"] + " " + d["text"] for d in corpus]
qids = [q["_id"] for q in queries if q["_id"] in relevant]
qtexts = [q["text"] for q in queries if q["_id"] in relevant]

def ndcg_at_10(scores):
    values = []
    for qid, row in zip(qids, scores):
        top = ids[np.argsort(-row, kind="stable")[:10]]
        gains = np.array([d in relevant[qid] for d in top], dtype=float)
        ideal = np.ones(min(10, len(relevant[qid])))
        discount = 1 / np.log2(np.arange(2, 12))
        values.append((gains * discount).sum() / (ideal * discount[: len(ideal)]).sum())
    return float(np.mean(values))

settings = {
    "tf-idf, no length normalisation": dict(norm=None),
    "tf-idf cosine": dict(norm="l2"),
    "tf-idf cosine, sublinear tf": dict(norm="l2", sublinear_tf=True),
    "tf-idf cosine, sublinear, no idf": dict(norm="l2", sublinear_tf=True, use_idf=False),
}
for name, options in settings.items():
    tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9]+", stop_words="english", **options)
    docs = tfidf.fit_transform(texts)
    scores = (tfidf.transform(qtexts) @ docs.T).toarray()
    print(f"{name:36} nDCG@10 = {ndcg_at_10(scores):.3f}")

counter = CountVectorizer(token_pattern=r"[a-z0-9]+", stop_words="english")
tf = counter.fit_transform(texts).tocsr().astype(float)
queries_matrix = (counter.transform(qtexts) > 0).astype(float)
length = np.asarray(tf.sum(axis=1)).ravel()
df = np.asarray((tf > 0).sum(axis=0)).ravel()
idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))

def bm25(k1, b):
    weighted = tf.copy()
    norm = k1 * (1 - b + b * length[rows] / length.mean())
    weighted.data = tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.indices]
    return (queries_matrix @ weighted.T).toarray()

analyse = counter.build_analyzer()
reference = BM25Okapi([analyse(t) for t in texts])
library = np.array([reference.get_scores(analyse(q)) for q in qtexts])
print(f"{'rank_bm25 BM25Okapi (k1=1.5, b=0.75)':36} nDCG@10 = {ndcg_at_10(library):.3f}")

print("BM25 nDCG@10, rows k1, columns b")
bs = (0.0, 0.25, 0.5, 0.75, 1.0)
print("       " + "".join(f"{b:7.2f}" for b in bs))
for k1 in (0.2, 0.5, 0.9, 1.2, 1.5, 2.0, 3.0):
    print(f"k1={k1:3.1f} " + "".join(f"{ndcg_at_10(bm25(k1, b)):7.3f}" for b in bs))
```

The output of the run:

```text
tf-idf, no length normalisation      nDCG@10 = 0.518
tf-idf cosine                        nDCG@10 = 0.580
tf-idf cosine, sublinear tf          nDCG@10 = 0.632
tf-idf cosine, sublinear, no idf     nDCG@10 = 0.571
rank_bm25 BM25Okapi (k1=1.5, b=0.75) nDCG@10 = 0.670
BM25 nDCG@10, rows k1, columns b
          0.00   0.25   0.50   0.75   1.00
k1=0.2   0.638  0.642  0.648  0.648  0.650
k1=0.5   0.646  0.656  0.657  0.663  0.665
k1=0.9   0.659  0.664  0.669  0.671  0.668
k1=1.2   0.657  0.661  0.667  0.669  0.673
k1=1.5   0.653  0.662  0.664  0.666  0.666
k1=2.0   0.652  0.662  0.665  0.668  0.665
k1=3.0   0.643  0.655  0.669  0.669  0.656
```

**Reading the output.** The first four rows are tf-idf variants. The `norm=None` row is a raw dot product with no length normalisation. The grid shows nDCG@10 for each pair of $k_1$ (rows) and $b$ (columns); $b = 0$ turns length normalisation off and $b = 1$ applies it fully.

**Line by line.**

- `ndcg_at_10` takes the top 10 documents per claim, gives gain 1 to relevant ones, discounts by $1/\log_2(\text{rank} + 1)$ and divides by the best possible value, which is the same sum with the relevant documents first.
- `tf.data * (k1 + 1) / (tf.data + norm)` applies BM25's saturating weight to every non-zero cell at once. `norm` contains the length term, so changing `b` changes only that array.
- `queries_matrix @ weighted.T` scores all 300 claims against all 5,183 abstracts in one sparse product.

### What the numbers say

The tf-idf variant mattered a lot. Removing length normalisation cost 0.062 (0.580 down to 0.518), switching to sublinear tf gained 0.052 (0.580 up to 0.632), and removing idf cost 0.061 (0.632 down to 0.571). Between the worst and best tf-idf variant the spread was 0.114.

BM25's parameters mattered much less. The whole 35-cell grid spans 0.638 to 0.673, a spread of 0.035. At the common defaults $k_1 = 1.2$ and $b = 0.75$ the score is 0.669, and the `rank-bm25` library at $k_1 = 1.5$ and $b = 0.75$ scores 0.670. At every value of $k_1$, $b = 0.75$ beat $b = 0$ by between 0.010 and 0.026.

The surprise is how flat the grid is. Careful tuning of BM25 changes the result by about a third of what a sloppy tf-idf choice costs. Limits: one collection of scientific abstracts, 300 claims with about 1.1 relevant abstracts each, no stemming, and no held-out set. The grid was scored on the same queries it is read on, so the best cell (0.673) is optimistic. Quote the spread, not the winner.

<Infographic src="/img/ir-enrich/ir1-weighting.svg" alt="Bars of nDCG@10 for four tf-idf variants and three BM25 settings on 300 SciFact claims." caption="Look first at the red and orange bars: tf-idf choices span 0.114, while all BM25 grid cells sit within 0.035." />

## Designing with it

### Get the representation right first

The vector coordinates depend on tokenisation, normalisation and vocabulary. If the query contains `refund` but the document says `returned`, a pure lexical vector has no shared coordinate for that idea. Stemming, synonyms or semantic retrieval can bridge it, each with different failure modes. If a field contains product codes or names, exact tokens may be more valuable than broad semantic similarity. Index and query analysis still need to agree, as Session 3 showed.

The basic weight is intentionally simple. Raw term frequency can let repeated words dominate; variants use sublinear tf or another saturation function. A rare term gets high IDF, but a misspelled one-off token should not automatically outweigh every other signal. Document length normalisation also matters: cosine normalises vector magnitude, while BM25 models length as part of its scoring formula. Choose and tune on judged queries rather than trusting a formula because it is familiar.

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

The simple IDF is zero when a term appears in every document, because $\log_{10}(N/N)=0$. That is sensible as a discrimination signal: the term does not help select one document over another. It is undefined when document frequency is zero, so a query term absent from the index needs a separate path, such as no postings or a tolerant expansion from Session 3. The lab prevents zero document frequency for that reason. A practical implementation may smooth the formula and handle out-of-vocabulary terms explicitly.

Raw tf-idf also makes a term that occurs ten times weigh ten times as much as one occurrence, even when the repetitions are boilerplate. Saturating term frequency reduces that effect. A document that contains hundreds of unrelated words can still achieve a large raw dot product; cosine divides by vector lengths so the score emphasises direction rather than scale. Yet cosine is not a cure for every length problem: field boundaries, repeated navigation text and short snippets can still change the result. Remove obvious boilerplate at ingestion and evaluate fields separately where appropriate.

Query weighting deserves attention too. A short query often has one occurrence of each term, so its tf does little. IDF can still make a rare technical term dominate, which is helpful for an exact product name but harmful if it is a typo. A user who supplies a quoted phrase or an explicit filter is expressing structure beyond a bag of weighted terms; preserve that structure rather than flattening everything into one vector. The vector-space model is a ranking foundation, not a complete query language.

Finally, a cosine value is relative to the representation. A score of 0.816 for these four-dimensional vectors is exact arithmetic, but it is not a universal "good relevance" threshold. Changing tokenisation, IDF, corpus, field weights or query formulation changes the space. Compare scores within a controlled query and index version, then judge actual results with the test collection. Avoid presenting a raw score as a probability that the document answers the question.

## Where this stands in 2026

:::info Industry view

- Classical tf-idf remains the clearest model for teaching why rare terms matter and why vector length is normalised; production lexical search often uses BM25 or a tuned variant.
- Current Lucene documentation exposes BM25's term-frequency and length parameters, while Azure AI Search documents rank fusion for combined lexical and vector queries.
- A modern retrieval stack is usually evaluated in stages: candidate recall, fused ranking, reranked top results and the downstream user or answer outcome.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Tuning $k_1$ and $b$ on the queries you report | The best cell looks like the answer | Hold out queries for the final number. A 35-cell grid on 300 claims overfits a little by construction |
| Using a raw dot product between tf-idf vectors | Cosine feels like an optional polish | Normalise length. Without it nDCG@10 fell from 0.580 to 0.518 |
| Spending a day on BM25 parameters before checking the basics | They are the famous knobs | Check length normalisation, sublinear tf and idf first. Those moved the score by 0.05 to 0.11, the grid by 0.035 |
| Comparing raw scores across queries | A higher score looks more relevant | Compare scores only within one query and one index. Scores depend on query length and corpus statistics |
| Copying the textbook IDF into BM25 | They are both called IDF | BM25 libraries use natural logarithms and smoothing. The base-10 value from the worked example is not interchangeable |

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

<details>
<summary><strong>Q6.</strong> (Medium) Why did tf-idf without length normalisation score 0.518 while cosine scored 0.580?</summary>

A raw dot product adds up weights for every query word a document contains, so a long abstract that happens to contain more of them gets a higher score for no better reason. Cosine divides by the vector length, so a long document must be proportionally richer in query words to win. On SciFact abstracts, which vary in length, that correction was worth 0.062 nDCG@10.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) With $k_1 = 1.2$ a word occurring 10 times scores 1.964 against 1.000 for once. What does that imply for a page that repeats a keyword 50 times?</summary>

The BM25 term weight cannot exceed $k_1 + 1 = 2.2$, so 50 repeats are worth at most 2.2 times one occurrence, and in practice less than that. Repeating a keyword has sharply diminishing returns, which is the point of saturation. Raw tf-idf, by contrast, would score 50 repeats fifty times higher.

</details>

## Go deeper

- [Stanford IR book: vector-space scoring](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html); the weighting and cosine model.
- [Apache Lucene BM25Similarity](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html); current lexical scoring parameters.
- [Azure AI Search hybrid scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking); a production RRF implementation.
- [Apache Lucene BM25Similarity](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/similarities/BM25Similarity.html); the smoothed IDF and the $k_1$ and $b$ parameters used in the experiment.
- [BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models](https://arxiv.org/abs/2104.08663); the benchmark that SciFact belongs to. Its abstract describes BM25 as a robust baseline.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact); corpus, queries and the CC BY-SA 4.0 licence.
- Built from the course lecture "ir-s5-vsm" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can compute the base-10 IDF, tf-idf and cosine values.
- [ ] I can explain how tf-idf ranks documents and what a zero shared-term vector misses.
- [ ] I can distinguish a lexical score, dense similarity, rank fusion and reranking in the tiny corpus.
- [ ] I can state why a reranker cannot fix a relevant document missing from the candidate set.
- [ ] I can compute raw, sublinear and BM25 term weights for a word occurring 1, 2 and 10 times.
- [ ] I can explain why length normalisation and idf mattered more than BM25's $k_1$ and $b$ in this experiment.
- [ ] I can read a parameter grid, say what it shows, and say why its best cell is optimistic.

## Where to go next

Next: [Session 6, classification and clustering](/docs/theory/ir/document-classification-and-clustering), which uses the same weighted vectors to group documents. Related: [Evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), which puts intervals around scores like the ones above.
