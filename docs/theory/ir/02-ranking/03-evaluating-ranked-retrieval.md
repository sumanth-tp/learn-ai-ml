---
id: ir-evaluating-ranked-retrieval
title: "Information Retrieval · Session 7 — Evaluating Ranked Retrieval"
sidebar_label: "7 · Evaluate ranking"
sidebar_position: 3
slug: /theory/ir/evaluating-ranked-retrieval
description: "How test collections and rank-aware metrics such as AP, MAP, MRR and NDCG reveal the quality of a search result order."
tags: [information-retrieval, evaluation, average-precision, ndcg]
---

import Infographic from '@site/src/components/Infographic';
import RankingMetricsLab from '@site/src/components/viz/RankingMetricsLab';

**In one line.** A search test must judge both which documents were found and where the useful ones appeared in the ranking.

:::tip Before you start

**You should already know**

- What precision and recall mean ([Session 1](/docs/theory/ir/what-information-retrieval-is)).
- How BM25 and tf-idf rank documents ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).

**Reading time:** about 50 minutes, plus a few seconds to run the code.

**After this chapter you can**

- Compute DCG and nDCG for a graded ranking by hand.
- Put a bootstrap interval on a mean score and run a paired test between two systems.
- Explain why overlapping intervals can still hide a real difference, and why a small test set cannot confirm a small gain.

:::

## In 30 seconds

Two search systems can return the same documents and still feel different, because people read from the top. Rank-aware metrics reward useful documents that appear early: a perfect answer at rank 1 is worth more than the same answer at rank 9.

One average over 300 queries is also only a sample. Ask how much it would move if you had picked other queries, by resampling the queries many times. And compare systems query by query, not by comparing two separate averages: the paired comparison is much sharper.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Relevance judgement (qrels) | A stored label saying which documents answer which query | Query 7, document 31 = relevant |
| P@k | Share of the first k results that are relevant | 2 relevant in the top 5 = 0.4 |
| AP / MAP | Average of precision at each relevant rank; MAP averages that over queries | Ranks 1, 3, 5 of 3 relevant = 0.756 |
| MRR | The mean of 1 / rank of the first relevant result | First hit at rank 3 = 0.333 |
| DCG / nDCG | Graded gain, discounted by rank, then divided by the best possible value | 8.887 / 9.393 = 0.946 |
| Bootstrap interval | The range of a mean when queries are resampled with replacement | 0.632 with interval [0.589, 0.677] |
| Paired test | Compares two systems on the same queries, using each query's difference | Wilcoxon signed-rank test |
| p-value | How surprising the observed difference would be if the systems were equal | p = 0.0004 |


## The idea in plain words

A search system can return the same five documents in two orders and give users very different experiences. If useful documents appear at ranks 1, 3 and 5, a person may succeed quickly. If they all appear after irrelevant material, the set is unchanged but the ranking is poor. Precision and recall over the whole set ignore this difference. Ranked retrieval needs measures that reward early useful results.

The worked example has relevant documents at ranks **1, 3 and 5**. Precision at those ranks is $1/1=1.0$, $2/3\approx0.667$, and $3/5=0.6$. **Average Precision (AP)** averages these three values, giving $0.756$ rounded to three decimals. This calculation assumes those are the three known relevant documents for the query. If the collection contains additional relevant documents that were not retrieved, the AP denominator should include them and the value would be lower.

<Infographic src="/img/ir/ranking-evaluation.svg" alt="Relevant documents at ranks 1, 3 and 5 give precisions 1.000, 0.667 and 0.600, whose average is 0.756." caption="The AP example rewards relevant documents at the ranks where they are found." />

No single measure answers every product question. **P@k** describes the cleanliness of the first $k$ results. **MRR** focuses on the first relevant result, useful when one answer suffices. **AP** rewards finding all known relevant documents early, and **MAP** averages AP across queries. **NDCG** permits graded relevance: a document that answers the question fully can count more than a partial match, with a discount for lower ranks. The choice of $k$, labels and query set matters as much as the metric's formula.

:::note Added for this site

The product evaluation workflow and four-method comparison below extend the metric outline. They show how relevance judgements, incomplete pools and uncertainty affect an engineering decision.

:::

Move a result up or down. With the defaults, relevant documents sit at ranks **1, 3, 5**, so AP is **0.756**. The lab also shows P@3, MRR and graded NDCG; its data view exposes every rank's precision and discounted gain.

<RankingMetricsLab />

## Worked example, step by step

A ranking of five results has relevance grades 3, 0, 2, 0, 1 (3 is a full answer, 0 is useless). Compute nDCG, which uses gain $2^g - 1$ and a discount of $1 / \log_2(\text{rank} + 1)$.

1. **Gains.** Grade 3 gives 7, grade 2 gives 3, grade 1 gives 1, grade 0 gives 0. The list of gains is 7, 0, 3, 0, 1.
2. **Discounts** for ranks 1 to 5: 1.000, 0.631, 0.500, 0.431, 0.387.
3. **DCG** is the sum of gain times discount: $7 \times 1.000 + 0 + 3 \times 0.500 + 0 + 1 \times 0.387 = 7 + 1.5 + 0.387 = 8.887$.
4. **Ideal DCG** puts the gains in the best order, 7, 3, 1, 0, 0: $7 \times 1.000 + 3 \times 0.631 + 1 \times 0.500 = 7 + 1.893 + 0.5 = 9.393$.
5. **nDCG** is $8.887 / 9.393 = 0.946$. The ranking loses a little because the grade-2 result sits at rank 3 and not rank 2.

In words: nDCG compares your ranking with the best possible one for the same documents, so 1.0 means nothing could be better. The lab above uses the same formula, so its default grades give 0.946. The first block below prints these numbers.

## How it works

### Ranked-retrieval measures

- **P@k / AP / MAP**; Precision in top-k; average over relevant ranks; over queries.
- **NDCG**; Graded relevance, rank-discounted.
- **MRR**; Reciprocal rank of first relevant.

:::tip

**Worked.** relevant at ranks 1,3,5 → P@1=1, P@3=0.667, P@5=0.6 → AP=0.756.

:::

### Pooling & significance

Relevance judgments are costly; pooling judges the top results of many systems. Check inter-annotator agreement (kappa) and significance of metric differences.


## A real system that works this way

**NIST's TREC** is a real evaluation programme built around document collections, topics and relevance judgements. Its [how-to guide](https://trec.nist.gov/howto.html) explains those parts of a test collection. Teams submit ranked runs against shared tasks so methods can be compared on the same material. This is a disciplined alternative to choosing a ranking model because one demo query looked good.

**Azure Databricks AI Search** provides a current product example of automated retrieval-quality evaluation. Its [documentation](https://learn.microsoft.com/en-us/azure/databricks/ai-search/retrieval-quality-eval) describes comparing full-text, vector, hybrid and reranked search strategies on the same generated queries, grading query-document pairs, and reporting ranked metrics with confidence intervals. The feature is documented as beta. A generated query and an LLM judge can speed up diagnosis, but they should not be treated as unquestionable ground truth for a high-impact application; review samples and check the task's own users and documents.

For a company policy search, a practical evaluation set might include exact policy names, a broad question such as "Can I claim a damaged bag?", multilingual variants and product codes. Freeze the document snapshot, write the intended information need for each query and have reviewers judge candidate results. The ranking report should show both aggregate scores and the worst individual queries so a team can understand *why* one method wins.

## Code you can run

The first block reproduces the AP example. It assumes exactly three relevant documents exist in the judged collection for this query.

```python
relevant_at_rank = [True, False, True, False, True]
relevant_total = 3
seen = 0
precision_at_relevant = []
for rank, relevant in enumerate(relevant_at_rank, 1):
    if relevant:
        seen += 1
        precision_at_relevant.append(seen / rank)
average_precision = sum(precision_at_relevant) / relevant_total
print("P at relevant ranks:", [round(value, 3) for value in precision_at_relevant])
print(f"AP={average_precision:.3f}")
assert round(average_precision, 3) == 0.756
```

The second block uses the same checked-in six-document comparison as Session 5. Run it from the repository root. It calculates AP for each ranking against the two known relevant documents, IDs 0 and 2. That gives a rank-aware view of BM25, illustrative dense vectors, hybrid RRF and the intent-coverage reranker.

```python
import runpy

demo = runpy.run_path("scripts/ir_comparison.py")
rankings = demo["compare"]()
relevant = demo["RELEVANT"]

def average_precision(ranking, relevant):
    if not relevant:
        return 0.0
    seen = 0
    total = 0.0
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            seen += 1
            total += seen / rank
    return total / len(relevant)

for name, ranking in rankings.items():
    print(f"{name:18} top 3={ranking[:3]} AP={average_precision(ranking, relevant):.3f}")
assert average_precision(rankings["reranked"], relevant) == 1.0
```

The reranked toy list reaches AP 1.0 because both relevant documents occupy the first two positions. The other methods place at least one irrelevant document before the second relevant result. This illustrates how to compare methods on one query; a real conclusion requires many held-out queries and uncertainty estimates.

### The worked example in code

This block computes gains, discounts, DCG, ideal DCG and nDCG for the grades above.

```python
import numpy as np

grades = np.array([3, 0, 2, 0, 1])
discount = 1 / np.log2(np.arange(2, len(grades) + 2))
gain = 2.0 ** grades - 1
dcg = (gain * discount).sum()
ideal = (np.sort(gain)[::-1] * discount).sum()
print("gains", gain.tolist(), "discounts", np.round(discount, 3).tolist())
print(f"DCG {dcg:.3f}  ideal DCG {ideal:.3f}  nDCG {dcg / ideal:.3f}")
```

**Reading the output.** The gains and discounts match steps 1 and 2, and the last line prints `DCG 8.887  ideal DCG 9.393  nDCG 0.946`.

### An experiment on real runs with intervals and a paired test

The final block compares three real systems on the 300 SciFact test claims: tf-idf cosine with sublinear tf, BM25 with $k_1 = 1.2$ and $b = 0.75$, and BM25 with $b = 0$ (no length normalisation). It computes nDCG@10, MAP over the full ranking and MRR@10 per query, puts a 95 per cent bootstrap interval on each mean using 10,000 resamples of the queries, and compares systems by their per-query differences with a bootstrap interval and a Wilcoxon signed-rank test. A last line repeats one comparison on only 50 random queries.

Versions used: Python 3.14.6, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.3. The run takes about 5 seconds.

```python
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from scipy.stats import wilcoxon
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

counter = CountVectorizer(token_pattern=r"[a-z0-9]+", stop_words="english")
tf = counter.fit_transform(texts).tocsr().astype(float)
length = np.asarray(tf.sum(axis=1)).ravel()
df = np.asarray((tf > 0).sum(axis=0)).ravel()
idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
query_matrix = (counter.transform(qtexts) > 0).astype(float)

def bm25(k1, b):
    weighted = tf.copy()
    weighted.data = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * length[rows] / length.mean())) * idf[tf.indices]
    return (query_matrix @ weighted.T).toarray()

tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9]+", stop_words="english", sublinear_tf=True)
docs = tfidf.fit_transform(texts)
runs = {
    "tf-idf cosine": (tfidf.transform(qtexts) @ docs.T).toarray(),
    "BM25 b=0.75": bm25(1.2, 0.75),
    "BM25 b=0": bm25(1.2, 0.0),
}

def per_query(scores):
    ndcg, ap, rr = [], [], []
    discount = 1 / np.log2(np.arange(2, 12))
    for qid, row in zip(qids, scores):
        order = ids[np.argsort(-row, kind="stable")]
        hit = np.array([d in relevant[qid] for d in order])
        ranks = np.flatnonzero(hit) + 1
        ndcg.append((hit[:10] * discount).sum() / discount[: min(10, len(relevant[qid]))].sum())
        ap.append(np.sum(np.arange(1, len(ranks) + 1) / ranks) / len(relevant[qid]))
        rr.append(1 / ranks[0] if ranks[0] <= 10 else 0.0)
    return {"nDCG@10": np.array(ndcg), "MAP": np.array(ap), "MRR@10": np.array(rr)}

rng = np.random.default_rng(0)
resamples = rng.integers(0, len(qids), size=(10000, len(qids)))
results = {name: per_query(scores) for name, scores in runs.items()}
for metric in ("nDCG@10", "MAP", "MRR@10"):
    for name in runs:
        values = results[name][metric]
        low, high = np.percentile(values[resamples].mean(axis=1), [2.5, 97.5])
        print(f"{metric:8}{name:15} {values.mean():.3f}  95% CI [{low:.3f}, {high:.3f}]")

def compare(a, b, metric="nDCG@10"):
    diff = results[a][metric] - results[b][metric]
    low, high = np.percentile(diff[resamples].mean(axis=1), [2.5, 97.5])
    p = wilcoxon(diff[diff != 0]).pvalue
    print(f"{a} minus {b}: {diff.mean():+.3f}  CI [{low:+.3f}, {high:+.3f}]  Wilcoxon p={p:.4f}  queries changed: {(diff != 0).sum()} of {len(diff)}")

compare("BM25 b=0.75", "tf-idf cosine")
compare("BM25 b=0.75", "BM25 b=0")

small = rng.choice(len(qids), size=50, replace=False)
diff = (results["BM25 b=0.75"]["nDCG@10"] - results["tf-idf cosine"]["nDCG@10"])[small]
low, high = np.percentile(diff[rng.integers(0, 50, size=(10000, 50))].mean(axis=1), [2.5, 97.5])
print(f"same comparison on 50 random queries: {diff.mean():+.3f}  CI [{low:+.3f}, {high:+.3f}]")
```

The output of the run:

```text
nDCG@10 tf-idf cosine   0.632  95% CI [0.589, 0.677]
nDCG@10 BM25 b=0.75     0.669  95% CI [0.624, 0.714]
nDCG@10 BM25 b=0        0.657  95% CI [0.609, 0.703]
MAP     tf-idf cosine   0.585  95% CI [0.539, 0.632]
MAP     BM25 b=0.75     0.630  95% CI [0.583, 0.677]
MAP     BM25 b=0        0.624  95% CI [0.576, 0.672]
MRR@10  tf-idf cosine   0.591  95% CI [0.544, 0.639]
MRR@10  BM25 b=0.75     0.637  95% CI [0.589, 0.685]
MRR@10  BM25 b=0        0.628  95% CI [0.579, 0.676]
BM25 b=0.75 minus tf-idf cosine: +0.037  CI [+0.017, +0.056]  Wilcoxon p=0.0004  queries changed: 98 of 300
BM25 b=0.75 minus BM25 b=0: +0.013  CI [-0.000, +0.025]  Wilcoxon p=0.0311  queries changed: 59 of 300
same comparison on 50 random queries: +0.024  CI [-0.031, +0.079]
```

**Reading the output.** "CI" is the 95 per cent interval from the bootstrap. In the comparison lines, "minus" is a mean of per-query differences, and "queries changed" counts queries whose nDCG@10 differs at all between the two systems. The p-value is the Wilcoxon signed-rank test on the non-zero differences.

**Line by line.**

- `resamples = rng.integers(0, len(qids), size=(10000, len(qids)))` draws 10,000 sets of 300 query indices with replacement. `values[resamples].mean(axis=1)` is then the mean of each resample, and the 2.5th and 97.5th percentiles give the interval.
- The same `resamples` array is reused for every metric and comparison, so intervals and differences are consistent with each other.
- `ap` divides by `len(relevant[qid])`, the number of known relevant documents, so a missed relevant document costs AP, as the chapter's denominator warning requires.
- `diff[diff != 0]` drops tied queries before the Wilcoxon test, which ignores zero differences.

### What the numbers say

The three systems' separate intervals overlap heavily: tf-idf [0.589, 0.677] and BM25 with $b = 0.75$ [0.624, 0.714]. A reader comparing the two ranges would call it a tie. The paired comparison says otherwise. BM25 gained 0.037 on average, the interval of the per-query differences is [+0.017, +0.056], which excludes zero, and the Wilcoxon p-value is 0.0004. The reason is that the two systems fail and succeed on mostly the same queries. Only 98 of 300 queries differed at all, so the noise in the difference is far smaller than the noise in each score.

The comparison between $b = 0.75$ and $b = 0$ is the awkward case. The mean gain of 0.013 has an interval of [-0.000, +0.025], which touches zero, while the Wilcoxon test gives p = 0.0311. The two checks disagree at the boundary, so I would not claim a win from this data. On 50 random queries the same tf-idf comparison gave +0.024 with interval [-0.031, +0.079], consistent with no difference. A test set that small cannot confirm a gain of this size.

With about 1.13 relevant abstracts per claim, MAP (0.630) and MRR@10 (0.637) land close together for the same system. The metrics agree here because there is rarely a second relevant document to find. Limits: one collection, 300 queries, incomplete judgements (an unjudged abstract counts as not relevant), a resampling model that treats queries as independent, and two comparisons run without correcting for multiple tests.

<Infographic src="/img/ir-enrich/ir1-confidence.svg" alt="Three nDCG@10 means with overlapping 95 per cent intervals, beside three cards giving paired differences, p-values and a 50-query repeat." caption="Look first at the three overlapping intervals on the left, then at the green card: the paired difference still excludes zero." />

## Designing with it

### Construct a test collection deliberately

A test collection is a document snapshot, a set of queries or topics, and relevance judgements for query-document pairs. Sample queries from the workflow you intend to improve, not only from convenient document titles. Include common, rare and failure-prone cases. Record user intent, collection version, query text, filters and the judgement guide. If a policy changes, version the labels rather than silently scoring new documents against old judgements.

Exhaustively judging every document for every query is usually impossible. **Pooling** collects the top results from several systems and judges that union. It improves coverage but introduces bias toward systems similar to those that supplied the pool. A document outside the pool is unjudged, not proven irrelevant. Report how unjudged results are handled, and add new candidates to the pool when testing a method that retrieves different material.

### Select measures by user behaviour

| Measure | What it rewards | Limit to remember |
| --- | --- | --- |
| P@k | Relevant results in the visible top $k$ | Ignores relevant documents below $k$ and collection recall |
| AP / MAP | Relevant documents found early, across one / many queries | Needs a known relevant count; binary relevance |
| MRR | The first relevant hit | Ignores whether later results are useful |
| NDCG@k | Graded gains high in the ranking | Depends on grade definitions and the ideal-ranking denominator |
| Recall@k | Known relevant documents entering the top $k$ | Sensitive to missing relevance judgements |

For a question-answering assistant, the first useful passage may matter most, but one passage might not contain every fact needed. Track both early precision and candidate recall at the generator's context cutoff. For exploratory research, several relevant documents across the first page may matter more than the first hit alone. Set $k$ to match the actual interface or downstream budget.

### Compare paired queries, not just totals

When method B improves mean NDCG by a small amount, inspect the per-query differences. Did many queries improve a little, or did one common query dominate? Query-level paired bootstrap or randomisation can estimate uncertainty without pretending each document is an independent experiment. If several people judge relevance, sample disagreements and measure consistency; kappa can be a diagnostic but is affected by class prevalence. Resolve a rubric problem before treating disagreements as noise.

### Join offline and online evidence

Offline relevance judgements are repeatable and fast for development. They cannot fully capture snippets, layout, trust, latency or the consequence of an answer in context. A production search team should monitor queries with no useful click, rapid reformulation, abandonment and user reports, while remembering that clicks are biased by position. An online A/B test can estimate product impact after an offline gate, provided permissions and latency remain safe. Keep a set of regression queries for high-impact failures even if the overall average rises.

## Read the metric disagreement

Suppose two rankings each return three relevant documents in their top five. Set-level precision is 3/5 for both, and recall is the same if there are exactly three relevant documents in the collection. The first ranking places them at 1, 3 and 5, giving AP 0.756. Another ranking might place them at 3, 4 and 5: its precision values at relevant ranks are 1/3, 2/4 and 3/5, so AP is about 0.478. The same documents are found, yet a user must work harder to reach them. That is the core reason to evaluate order.

MRR would tell a different story. The first ranking's top result is relevant, so reciprocal rank is 1.0. The second ranking's first relevant result is at rank 3, so reciprocal rank is 1/3. MRR is especially clear for a task where one correct result ends the search. But if a question needs evidence from several documents, a system with one excellent top result and many useless following results can still have a perfect reciprocal rank. Pair MRR with recall or AP when completeness matters.

NDCG adds a second dimension: quality grades. In the lab, results A, C and E have grades 3, 2 and 1. A grade-3 hit contributes gain $2^3-1=7$ before discount; a grade-1 hit contributes 1. The ideal ordering sorts the available graded results from highest to lowest, and NDCG divides the observed discounted gain by that ideal. Changing the judgement rubric changes the metric. If graders cannot reliably distinguish 2 from 3, a finely graded score can give a false sense of precision.

### Examine the four-method example by query

The six-document comparison is intentionally one query with two known relevant documents. BM25 gets document 0 first but misses the paraphrase near the top. Toy dense features recover the paraphrase but overvalue a generic shipping page. Fusion puts both relevant documents in its top three, improving the shortlist for reranking, yet still has the generic page in second position. The reranker checks concept coverage and promotes the paraphrase. AP exposes the position of both relevant documents, not just whether the first result was correct.

That result is a useful **unit example**, not a benchmark. Change the query to an exact product code and the dense path may be harmful; change the corpus and the hand-designed features cease to represent it. A real comparison needs many queries split by type, labels made independently of candidate method, and a frozen configuration for each run. Record not only mean AP or NDCG but the examples where the new method loses, because those losses may be concentrated in the queries the product values most.

### Make incomplete judgements visible

Pooling is practical because many systems surface the same obvious documents. It can miss a relevant document found by a new system with a different retrieval representation. If that document is treated as nonrelevant, the new system is penalised for discovering it. A defensible workflow flags unjudged top results for review, reports judgement coverage, and reruns metrics after adding high-impact candidates to the pool. The denominator in AP is particularly sensitive to whether all relevant documents are known.

The relevance judgement itself is contextual. A page that mentions a policy but lacks the exception may be partially relevant to an exploratory search and insufficient for an answer that must quote the exception. Write task-specific grades. Keep the assessor blind to system identity where feasible, and resolve disagreements with the rubric. Evaluation is an engineering instrument for deciding what to change; its credibility comes from the collection and labels as much as from the code implementing the formula.

### Keep evaluation reproducible

Store the query set, document snapshot identifier, judgement file, run configuration and scorer version together. A changed analyser, embedding model or filtering rule can alter a ranking even when the visible query is identical. Note whether scores are calculated before or after permission filtering, deduplication and reranking. If one system has access to more documents, compare it in a separate experiment rather than presenting the difference as a ranking gain.

When reporting a result, include the number of queries, the judgement coverage of each system's top results and at least one representative win and loss. A mean score without that context can conceal a method that serves popular easy queries while failing rare safety or policy queries. Those details make the next investigation possible.

## Session 8 synthesis: connect the classic pipeline

A review of Sessions 1 to 7 links them: define the information need, normalise text, use the dictionary and inverted index for candidates, compress postings to make lookup practical, rank with weighted vectors, optionally classify or cluster documents, then evaluate results. The formula list includes precision, recall, F1, IDF, tf-idf, cosine, Heaps' vocabulary-growth law and AP. Those calculations answer different questions; a working index does not by itself prove a useful ranking.

A worked review chain has **five retrieved**, relevant hits at **ranks 1, 3 and 5**, and **four relevant in the collection**. Precision is $3/5=0.60$ and recall is $3/4=0.75$. A common answer is AP **0.756**, which divides the precisions $1$, $2/3$ and $3/5$ by the three *retrieved* relevant hits. Under the standard AP definition used in Session 7, the denominator is the four **known relevant documents in the collection**, so AP is $(1+2/3+3/5)/4\approx0.567$. The unreturned fourth relevant document contributes zero. The value 0.756 is correct only for Session 7's separate example where exactly three relevant documents exist. State the denominator with any AP calculation.

<details>
<summary><strong>Review Q1.</strong> Name the classic IR pipeline in order.</summary>

Introduction and information need; Boolean and inverted-index candidates; dictionary and tolerant retrieval; index construction and compression; vector ranking; optional classification or clustering; evaluation. <em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Review Q2.</strong> Write the IDF, tf-idf and cosine formulas.</summary>

IDF is log(N/df) in its simple form; tf-idf is tf times IDF; cosine is the vector dot product divided by both vector lengths. <em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Review Q3.</strong> Five retrieved, three relevant at ranks 1, 3 and 5, of four relevant in the collection: give precision, recall and AP.</summary>

Precision is 3/5 = 0.60; recall is 3/4 = 0.75; standard AP is (1 + 2/3 + 3/5) / 4 = 0.567. The figure 0.756 uses a three-document denominator and does not match the four-relevant premise. <em>Session 8 · numeric</em>

</details>

<details>
<summary><strong>Review Q4.</strong> Why can a vector-space model rank results that Boolean retrieval leaves unranked?</summary>

It assigns weighted term coordinates and compares each candidate to the query by cosine. Boolean retrieval first determines set membership, so it needs an additional scoring rule to order that set. <em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Review Q5.</strong> What is the role of compression in indexing?</summary>

Gap and variable-byte encoding reduce postings storage; dictionary techniques such as front-coding reduce vocabulary storage. Smaller structures can improve memory use and I/O when their decode costs are acceptable. <em>Session 8 · conceptual</em>

</details>

Built from the course lecture "ir-s8-midsem-review" (Lecture Library series).

## Where this stands in 2026

:::info Industry view

- TREC still organises retrieval comparisons around shared documents, topics and relevance judgements; modern tracks include richer and graded tasks.
- Current AI Search evaluation tools compare lexical, vector, hybrid and reranked strategies with ranked metrics and confidence intervals, but generated queries and model-based judges require human spot checks.
- RAG evaluation needs both retrieval quality and answer quality. A high NDCG for passages does not prove the generated answer used them faithfully.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Declaring a tie because two intervals overlap | The ranges visibly touch | Interval the per-query difference. Here the systems' intervals overlapped yet the paired interval was [+0.017, +0.056] |
| Declaring a win from a single p-value | 0.031 is below 0.05 | Check that an interval on the same difference agrees. For $b = 0.75$ against $b = 0$ it did not (the interval touched zero), so no claim |
| Evaluating on a few dozen queries | Labelling is expensive | With 50 queries the gain of +0.024 had an interval of [-0.031, +0.079]. Plan the query count before you begin |
| Reporting AP with the retrieved count as denominator | It gives a higher number | Divide by the known relevant count, as in the Session 8 note, and state it |
| Reading unjudged documents as irrelevant without saying so | The label file lists only relevant ones | Report judgement coverage, and treat a method that finds new documents as possibly under-scored |

## Practice questions

<details>
<summary><strong>Q1.</strong> What does a test collection consist of?</summary>

A set of documents, queries (topics), and relevance judgments (which documents are relevant to which query).<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Why are unranked metrics insufficient for ranked retrieval?</summary>

They ignore the order of results; ranking quality needs rank-aware metrics that reward putting relevant documents higher.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Relevant docs at ranks 1, 3, 5 of 5. Compute Average Precision.</summary>

P@1=1.0, P@3=2/3=0.667, P@5=3/5=0.6 → AP = (1.0+0.667+0.6)/3 = 0.756.<br /><em>Session 7 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What does NDCG add over MAP?</summary>

NDCG uses graded relevance and a rank discount (a hit at rank 1 beats one at rank 10), normalised by the ideal ranking.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is pooling, and why is it used?</summary>

Judging only the top results pooled from many systems, because exhaustively judging every document is infeasible.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) Compute nDCG for grades 3, 0, 2, 0, 1 using gain $2^g - 1$ and discount $1 / \log_2(\text{rank} + 1)$.</summary>

DCG = 7 x 1.000 + 3 x 0.500 + 1 x 0.387 = 8.887. The ideal order 7, 3, 1 gives 7 + 3 x 0.631 + 1 x 0.500 = 9.393. nDCG = 8.887 / 9.393 = 0.946.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) The nDCG@10 intervals for tf-idf and BM25 overlap, yet the paired interval for their difference is [+0.017, +0.056]. How can both be true?</summary>

Each interval reflects how much a system's own average moves with the choice of queries, and some queries are simply hard for everyone. The two systems succeed and fail on mostly the same queries (only 98 of 300 changed), so those shared swings cancel in the per-query difference. The difference varies much less than either score, and its interval is narrower than the gap between the systems' intervals suggests.

</details>

## Go deeper

- [Stanford IR book: evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html); test collections and ranking measures.
- [NIST TREC how-to](https://trec.nist.gov/howto.html); real benchmark construction and judged runs.
- [Azure Databricks AI Search quality evaluation](https://learn.microsoft.com/en-us/azure/databricks/ai-search/retrieval-quality-eval); a current product workflow, documented as beta.
- [Smucker, Allan and Carterette (CIKM 2007), a comparison of statistical significance tests for information retrieval evaluation](https://uwaterloo.ca/data-systems-group/node/4915); compares the paired t-test, Wilcoxon test, sign test, bootstrap and randomisation test on TREC runs.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact); the corpus, claims and labels used in the experiment.
- [BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models](https://arxiv.org/abs/2104.08663); how the collection is used to compare retrievers.
- Built from the course lecture "ir-s7-evaluation" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can calculate AP for relevant results at ranks 1, 3 and 5 and state the denominator assumption.
- [ ] I can choose among P@k, MAP, MRR, NDCG and Recall@k for a user task.
- [ ] I can explain how pooling creates incomplete judgements and how that affects comparisons.
- [ ] I can compare retrieval methods on the same held-out queries and inspect per-query losses.
- [ ] I can compute DCG, ideal DCG and nDCG for a graded ranking by hand.
- [ ] I can build a bootstrap interval over queries and say what it does and does not cover.
- [ ] I can explain why overlapping system intervals do not rule out a real paired difference, and why two tests that disagree mean no claim.
- [ ] I can say why 50 queries cannot confirm a gain of 0.03.

## Where to go next

Next: [Web search at scale](/docs/theory/ir/web-search-at-scale), which moves from a fixed collection to an open, changing web. Related: [Neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), where these metrics and tests compare modern retrievers.
