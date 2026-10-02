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

## The idea in plain words

A search system can return the same five documents in two orders and give users very different experiences. If useful documents appear at ranks 1, 3 and 5, a person may succeed quickly. If they all appear after irrelevant material, the set is unchanged but the ranking is poor. Precision and recall over the whole set ignore this difference. Ranked retrieval needs measures that reward early useful results.

The lecture's example has relevant documents at ranks **1, 3 and 5**. Precision at those ranks is $1/1=1.0$, $2/3\approx0.667$, and $3/5=0.6$. **Average Precision (AP)** averages these three values, giving $0.756$ rounded to three decimals. This calculation assumes those are the three known relevant documents for the query. If the collection contains additional relevant documents that were not retrieved, the AP denominator should include them and the value would be lower.

<Infographic src="/img/ir/ranking-evaluation.svg" alt="Relevant documents at ranks 1, 3 and 5 give precisions 1.000, 0.667 and 0.600, whose average is 0.756." caption="The lecture's AP example rewards relevant documents at the ranks where they are found." />

No single measure answers every product question. **P@k** describes the cleanliness of the first $k$ results. **MRR** focuses on the first relevant result, useful when one answer suffices. **AP** rewards finding all known relevant documents early, and **MAP** averages AP across queries. **NDCG** permits graded relevance: a document that answers the question fully can count more than a partial match, with a discount for lower ranks. The choice of $k$, labels and query set matters as much as the metric's formula.

:::note Beyond the lecture

The product evaluation workflow and four-method comparison below extend the lecture's metric outline. They show how relevance judgements, incomplete pools and uncertainty affect an engineering decision.

:::

Move a result up or down. With the defaults, relevant documents sit at ranks **1, 3, 5**, so AP is **0.756**. The lab also shows P@3, MRR and graded NDCG; its data view exposes every rank's precision and discounted gain.

<RankingMetricsLab />

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

The first block reproduces the lecture's AP example. It assumes exactly three relevant documents exist in the judged collection for this query.

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

Suppose two rankings each return three relevant documents in their top five. Set-level precision is 3/5 for both, and recall is the same if there are exactly three relevant documents in the collection. The lecture's ranking places them at 1, 3 and 5, giving AP 0.756. Another ranking might place them at 3, 4 and 5: its precision values at relevant ranks are 1/3, 2/4 and 3/5, so AP is about 0.478. The same documents are found, yet a user must work harder to reach them. That is the core reason to evaluate order.

MRR would tell a different story. The lecture's first result is relevant, so reciprocal rank is 1.0. The second ranking's first relevant result is at rank 3, so reciprocal rank is 1/3. MRR is especially clear for a task where one correct result ends the search. But if a question needs evidence from several documents, a system with one excellent top result and many useless following results can still have a perfect reciprocal rank. Pair MRR with recall or AP when completeness matters.

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

The review lecture links Sessions 1 to 7: define the information need, normalise text, use the dictionary and inverted index for candidates, compress postings to make lookup practical, rank with weighted vectors, optionally classify or cluster documents, then evaluate results. Its formula list includes precision, recall, F1, IDF, tf-idf, cosine, Heaps' vocabulary-growth law and AP. Those calculations answer different questions; a working index does not by itself prove a useful ranking.

The review's worked chain says **five retrieved**, relevant hits at **ranks 1, 3 and 5**, and **four relevant in the collection**. Precision is $3/5=0.60$ and recall is $3/4=0.75$. The review prints AP **0.756**, which divides the precisions $1$, $2/3$ and $3/5$ by the three *retrieved* relevant hits. Under the standard AP definition used in Session 7, the denominator is the four **known relevant documents in the collection**, so AP is $(1+2/3+3/5)/4\approx0.567$. The unreturned fourth relevant document contributes zero. The value 0.756 is correct only for Session 7's separate example where exactly three relevant documents exist. State the denominator with any AP calculation.

<details>
<summary><strong>Review Q1.</strong> Name the classic IR pipeline in order.</summary>

Introduction and information need; Boolean and inverted-index candidates; dictionary and tolerant retrieval; index construction and compression; vector ranking; optional classification or clustering; evaluation. <em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Review Q2.</strong> Write the IDF, tf-idf and cosine formulas.</summary>

IDF is log(N/df) in the lecture's simple form; tf-idf is tf times IDF; cosine is the vector dot product divided by both vector lengths. <em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Review Q3.</strong> Five retrieved, three relevant at ranks 1, 3 and 5, of four relevant in the collection: give precision, recall and AP.</summary>

Precision is 3/5 = 0.60; recall is 3/4 = 0.75; standard AP is (1 + 2/3 + 3/5) / 4 = 0.567. The review source's 0.756 uses a three-document denominator and does not match its four-relevant premise. <em>Session 8 · numeric</em>

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

## Go deeper

- [Stanford IR book: evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html); test collections and ranking measures.
- [NIST TREC how-to](https://trec.nist.gov/howto.html); real benchmark construction and judged runs.
- [Azure Databricks AI Search quality evaluation](https://learn.microsoft.com/en-us/azure/databricks/ai-search/retrieval-quality-eval); a current product workflow, documented as beta.
- Built from the course lecture "ir-s7-evaluation" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can calculate AP for relevant results at ranks 1, 3 and 5 and state the denominator assumption.
- [ ] I can choose among P@k, MAP, MRR, NDCG and Recall@k for a user task.
- [ ] I can explain how pooling creates incomplete judgements and how that affects comparisons.
- [ ] I can compare retrieval methods on the same held-out queries and inspect per-query losses.
