---
id: ir-what-information-retrieval-is
title: "Information Retrieval · Session 1 — What Search Must Do"
sidebar_label: "1 · What IR is"
sidebar_position: 1
slug: /theory/ir/what-information-retrieval-is
description: "How a search system turns a text collection into ranked results and how precision, recall and F1 describe its errors."
tags: [information-retrieval, search, precision, recall]
---

import Infographic from '@site/src/components/Infographic';
import RetrievalMetricsLab from '@site/src/components/viz/RetrievalMetricsLab';

**In one line.** Information retrieval turns an uncertain information need into a ranked list, then checks whether that list helped.

## The idea in plain words

A user asks a search system for *information*, usually in a few ambiguous words. The system has a collection of documents, not a clean table whose rows can be selected by an exact predicate. Its job is to find useful documents and present the strongest candidates first.

Think of a company knowledge base. An employee types "expense policy for delayed baggage". Relevant material may say *travel reimbursement* or *lost luggage*, live in a PDF, and contain several unrelated policies. An exact database lookup is excellent when the employee knows a claim number. It is a poor substitute for a search engine when the employee needs to discover which document answers the question.

The search loop has two time scales. **Indexing** happens when documents arrive or change: decode text, choose searchable fields, split text into terms, normalise those terms, and record where each term occurs. **Querying** happens whenever a person searches: analyse the query in the same way, find candidates using the index, score them, and show a small ranked set. A mismatch between indexing and query analysis can make an existing document invisible.

<Infographic src="/img/ir/ir-pipeline.svg" alt="Documents are analysed into an inverted index, then candidates are scored into ranked results and evaluated." caption="The two paths in an information retrieval system: prepare the collection, then answer and assess queries." />

The lecture's worked set contains 50 retrieved documents, 40 of which are relevant, with 60 relevant documents in the entire collection. **Precision** is 40/50 = 0.800: four in five shown results are useful. **Recall** is 40/60 = 0.667: the system found two thirds of everything useful. Their harmonic mean, **F1**, is about 0.727. None of these numbers says whether the first result was good; ranked measures appear in Session 7.

:::note Beyond the lecture

Real search systems also need a choice of document unit, relevance labels, access control and a latency budget. Those engineering decisions are explained below; the lecture introduces the retrieval and metric vocabulary.

:::

Move either slider. The defaults reproduce the lecture's 40-of-50, 60-total example: precision **0.800**, recall **0.667** and F1 **0.727**.

<RetrievalMetricsLab />

## How it works

### IR vs data retrieval

IR finds unstructured text ranked by relevance (web/enterprise search); data retrieval returns exact structured records.

### Precision & recall

Precision = relevant retrieved / retrieved; recall = relevant retrieved / relevant; F1 balances them.

:::tip

**Worked.** 40 of 50 retrieved relevant, 60 relevant total → P=0.80, R=0.667, F1=0.727.

:::


## A real system that works this way

**Apache Lucene** is a concrete example of this architecture. Its core inverted index maps a term in a field to an ordered postings list of documents. A search application can combine those postings with query objects, score the candidates and retrieve stored fields to show a title or excerpt. The index is not itself the whole search product: an application still chooses the analysis chain, which fields to expose, who may see which document, how to rank and how to measure success. The [Lucene index API](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/index/package-summary.html) documents the distinction between term postings and stored fields.

Consider an internal travel-policy search. A PDF parser extracts the title and body. The application indexes both, keeping a stable external document ID and version. A query for "delayed baggage expenses" retrieves pages containing those terms and possibly related wording. The ranking layer promotes the exact policy page above a travel newsletter. A permission filter removes results the employee cannot open. The user sees a title and a short matching passage rather than an opaque document ID.

There are two separate success questions. **Can the right page enter the candidate set?** That is a recall problem, and bad extraction or missing vocabulary can defeat it before ranking begins. **Does it appear near the top?** That is a ranking problem, and a good candidate set can still fail the user if the right page sits below irrelevant results. A search team needs both kinds of measurement.

## Code you can run

The first block reproduces every number in the lecture. It also checks that the counts describe a possible result set.

```python
def retrieval_metrics(retrieved, relevant_retrieved, relevant_total):
    if retrieved < 0 or relevant_total < 0:
        raise ValueError("Counts must be non-negative")
    if relevant_retrieved < 0 or relevant_retrieved > min(retrieved, relevant_total):
        raise ValueError("Relevant retrieved cannot exceed either total")
    precision = relevant_retrieved / retrieved if retrieved else 0.0
    recall = relevant_retrieved / relevant_total if relevant_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

p, r, f1 = retrieval_metrics(50, 40, 60)
print(f"precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
assert (round(p, 3), round(r, 3), round(f1, 3)) == (0.800, 0.667, 0.727)
```

The next block shows what an inverted index changes. A forward view answers "which terms are in this document?"; an inverted view answers "which documents contain this term?". The example deliberately uses a tiny corpus so every posting can be inspected.

```python
import re
from collections import defaultdict

documents = {
    1: "Delayed baggage expenses are reimbursed",
    2: "Travel expenses require an approved claim",
    3: "The baggage desk tracks delayed bags",
}

def terms(text):
    return re.findall(r"[a-z]+", text.lower())

postings = defaultdict(set)
for doc_id, body in documents.items():
    for term in set(terms(body)):
        postings[term].add(doc_id)

print("delayed:", sorted(postings["delayed"]))
print("expenses:", sorted(postings["expenses"]))
print("both:", sorted(postings["delayed"] & postings["expenses"]))
assert sorted(postings["delayed"] & postings["expenses"]) == [1]
```

The first result is `precision=0.800 recall=0.667 F1=0.727`. The second block yields `delayed: [1, 3]`, `expenses: [1, 2]`, and `both: [1]`. The corpus and labels are illustrative; these examples demonstrate the mechanics, not a production-quality ranking.

## Designing with it

### Choose the searchable unit

Before choosing a ranking formula, decide what a result *is*. A web page, a section of a long handbook and a 400-token passage are different retrieval units. A whole document provides context but may hide the relevant paragraph in hundreds of pages. A passage retrieves precisely but can remove the sentence that qualifies a policy. Keep a link from each passage to its parent document so the result remains interpretable.

Stable external IDs matter. If a page moves or an extraction pipeline changes its chunk boundaries, users' bookmarks, relevance labels and evaluation history can break. Version the content, preserve source identity and define what happens when a document is deleted. Internal Lucene doc IDs can change during segment merges; do not expose them as the application's permanent identity. An index that continues to serve a deleted document is an operational and sometimes an access-control failure.

### Define relevance before measuring it

Precision and recall require a set of **relevance judgements**. For a small benchmark, write realistic queries, freeze the collection version and have a reviewer identify useful documents. The denominator for recall is only meaningful if that judgement set is sufficiently complete. In a large collection, exhaustive judgements are expensive; pooling results from several systems is a common way to identify likely relevant documents. Treat an unjudged document as *unknown*, not automatically irrelevant.

The same query may have several legitimate intents. "Leave policy" might mean annual leave, parental leave, or how to quit. A single label per query will hide that ambiguity. Record a short description of the user's task, and include queries from the vocabulary of actual users rather than only from document titles.

### Let the product cost decide the metric

| Situation | Error that hurts most | First measure to watch |
| --- | --- | --- |
| A legal discovery team must find every responsive document | Missing a relevant document | Recall, with a review workflow |
| A user scans only the first few search results | Irrelevant items at the top | Precision at a small cutoff and ranked measures |
| An automated answer uses retrieved passages | Missing evidence or adding distracting evidence | Recall and precision of the context set, then answer quality |

F1 is useful when both kinds of error matter and neither has a special cost. It is not a universal objective. It also ignores rank: two result lists with the same 40 relevant items can have identical precision and recall even when one puts all useful documents first and the other puts them last.

### Keep the index and the permission model aligned

Searchable text is often derived from PDFs, HTML and database records. Extraction failures, stale indexes and mismatched permissions create failure modes that a ranking algorithm cannot repair. Log the source version, indexed version and query path. Filter by permissions before showing snippets as well as before opening documents, because a snippet can itself reveal private text. Test additions, updates and deletions, not just the initial bulk import.

## Work through a search diagnosis

Suppose the travel-policy system receives the query "Can I claim for a delayed bag?" and returns a newsletter, a baggage-status page and the actual reimbursement policy in third place. That is a more useful diagnostic example than a single global F1 number, because it forces us to inspect the stages in order.

First, confirm the reimbursement policy entered the index. Check the extraction output, document ID, version and permissions. If a PDF parser dropped the paragraph containing "delayed baggage", no ranking change can recover those words through lexical matching. The fix belongs in ingestion or extraction. If the indexed text is correct, compare the analysed query terms with the indexed terms. `bag` and `baggage` are different tokens unless the analysis or expansion policy connects them. That is a vocabulary problem.

Next, inspect the candidate set. If the policy page is absent, there is a recall failure at candidate generation. If it is present but third, the problem is ranking. Perhaps the newsletter repeats "baggage" in a sidebar while the policy says it once in the heading. Field weights or term-frequency handling may need review. Perhaps the policy is older and the ranker overweights recency. These are distinct hypotheses, and a result trace should let the engineer test each one instead of guessing from the visible title.

Finally, check the product outcome. A third-place policy may be acceptable on a desktop page where all three results are visible, but poor in a mobile interface where only the first result fits. A reviewer can label all three as broadly related while the user still fails to answer the reimbursement question. That is why the evaluation set needs task context and rank-sensitive metrics. It is also why a small browser check belongs beside the offline benchmark.

### Build a small, honest judgement set

Start with a fixed snapshot of perhaps a few hundred documents and a deliberately varied list of queries. Include exact document titles, vague questions, paraphrases, misspellings and identifiers. For each query, write down the intent in a sentence. Reviewers should judge whether a document actually satisfies that intent, not merely whether it shares query words. Keep a record of disagreements; they often reveal ambiguous information needs or inadequate policy wording rather than a simple labelling error.

For the lecture's metric example, imagine that the 60 relevant documents were all identified by such a review. A system retrieves 50 and gets 40 right. There are 10 false positives and 20 false negatives. An engineer can now inspect a sample of each error type: were the false positives caused by a common word, a stale document or a weak title boost? Were the false negatives caused by missing extraction, synonyms or a restrictive filter? The scalar numbers tell us *how much* error there is; the examples suggest *where* to act.

If reviewers have judged only a pool of top results, the assertion that exactly 60 relevant documents exist may be false. Recall then becomes an estimate conditional on the judged pool. Report that limitation instead of treating unreviewed material as irrelevant. Hold back some queries for later checks so repeated tuning does not simply memorise the small evaluation set.

### Connect search quality to operations

A successful offline score can still fail at launch if an index refresh takes hours, a document disappears from the source but remains searchable, or a permission change reaches the index later than the application. Define a freshness target for additions, edits and deletions; monitor each path. Log query latency separately for parsing, candidate lookup and scoring so a quality improvement does not silently break the response-time budget. Relevance and reliability are parts of the same user experience: a perfect result arriving after a timeout is no result at all.

## Where this stands in 2026

:::info Industry view

- A modern search application may combine term postings with vector candidates, but the need for document identity, filtering and relevance judgements remains. The lexical index is still useful for exact terms such as IDs and policy names.
- Lucene's current API exposes term dictionaries, postings and stored fields as separate structures. That separation mirrors the distinction between finding candidates and displaying results.
- For applications that feed retrieved text to an LLM, retrieval metrics alone are insufficient: the answer must also be checked for evidence use and factual correctness. See the site's [RAG chapter](/docs/genai/rag).

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> How does IR differ from database data retrieval?</summary>

IR finds unstructured text ranked by relevance and tolerates ambiguity; data retrieval returns exact structured records.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is an inverted index?</summary>

A map from each term to its postings list (documents containing it); the core data structure that makes search scale.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> 50 retrieved, 40 relevant, 60 relevant total. Compute P, R, F1.</summary>

P = 40/50 = 0.80; R = 40/60 = 0.667; F1 = 0.727.<br /><em>Session 1 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Name the stages of an IR system.</summary>

Preprocess → index (inverted index) → parse query → match → rank by relevance.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why does IR rank rather than return a set?</summary>

Because information needs are graded and ambiguous; users want the most relevant documents first, not an unordered exact-match set.<br /><em>Session 1 · conceptual</em>

</details>

## Go deeper

- [Introduction to Information Retrieval, chapters 1 and 8](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html); the foundational account of indexes and evaluation.
- [Apache Lucene index API](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/index/package-summary.html); a production library's term, postings and stored-field concepts.
- Built from the course lecture "ir-s1-intro" (Lecture Library series).

## Check your understanding

- [ ] I can explain why unstructured search returns ranked candidates instead of exact database rows.
- [ ] I can draw the indexing and query paths and identify where a relevant document can disappear.
- [ ] I can calculate precision, recall and F1 from counts and state what each metric omits.
- [ ] I can choose a retrieval unit and explain how relevance labels and permissions affect evaluation.
