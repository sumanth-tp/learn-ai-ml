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

:::tip Before you start

**You should already know**

- How to read a short Python loop, a set and a dictionary.
- What a ratio is. No probability is needed.

**Reading time:** about 40 minutes, plus a few minutes to run the code.

**After this chapter you can**

- Calculate precision, recall and F1 from counts, and say what each one hides.
- Explain why a strict keyword filter and a ranked list behave so differently on real questions.
- Run a small experiment that measures both on a real collection.

:::

## In 30 seconds

Searching is not looking up a row. You describe what you need in a few words, and the system has to guess which documents help. Picture asking a librarian for "the rules on delayed baggage". She cannot hand over one exact match, so she brings a short stack with the best book on top.

Information retrieval is the craft of building that stack and of checking whether it helped. Two numbers describe a stack: how much of it is useful (precision) and how much of the useful material it found (recall).

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Information need | What the person is really after, which the query only hints at | "Can I claim for a lost suitcase?" |
| Query | The words typed into the search box | `delayed baggage claim` |
| Collection (corpus) | The set of documents being searched | 5,183 paper abstracts |
| Relevance judgement | A human label saying a document answers a query | Document 4 is relevant to query 1 |
| Precision | Share of the returned documents that are relevant | 2 relevant out of 3 returned = 0.667 |
| Recall | Share of all relevant documents that were returned | 2 found out of 3 existing = 0.667 |
| F1 | One number that blends precision and recall | 2 x 1.0 x 0.667 / 1.667 = 0.800 |
| Ranked retrieval | Returning documents in order of estimated usefulness | Best match first, then the next best |
| Inverted index | A map from each word to the documents containing it | `baggage` -> documents 1, 2, 4 |


## The idea in plain words

A user asks a search system for *information*, usually in a few ambiguous words. The system has a collection of documents, not a clean table whose rows can be selected by an exact predicate. Its job is to find useful documents and present the strongest candidates first.

Think of a company knowledge base. An employee types "expense policy for delayed baggage". Relevant material may say *travel reimbursement* or *lost luggage*, live in a PDF, and contain several unrelated policies. An exact database lookup is excellent when the employee knows a claim number. It is a poor substitute for a search engine when the employee needs to discover which document answers the question.

The search loop has two time scales. **Indexing** happens when documents arrive or change: decode text, choose searchable fields, split text into terms, normalise those terms, and record where each term occurs. **Querying** happens whenever a person searches: analyse the query in the same way, find candidates using the index, score them, and show a small ranked set. A mismatch between indexing and query analysis can make an existing document invisible.

<Infographic src="/img/ir/ir-pipeline.svg" alt="Documents are analysed into an inverted index, then candidates are scored into ranked results and evaluated." caption="The two paths in an information retrieval system: prepare the collection, then answer and assess queries." />

The worked set contains 50 retrieved documents, 40 of which are relevant, with 60 relevant documents in the entire collection. **Precision** is 40/50 = 0.800: four in five shown results are useful. **Recall** is 40/60 = 0.667: the system found two thirds of everything useful. Their harmonic mean, **F1**, is about 0.727. None of these numbers says whether the first result was good; ranked measures appear in Session 7.

:::note Added for this site

Real search systems also need a choice of document unit, relevance labels, access control and a latency budget. Those engineering decisions are explained below; the sections above introduce the retrieval and metric vocabulary.

:::

Move either slider. The defaults reproduce the 40-of-50, 60-total example: precision **0.800**, recall **0.667** and F1 **0.727**.

<RetrievalMetricsLab />

## Worked example, step by step

Take five tiny documents and the query `delayed baggage claim`. Documents 2, 4 and 5 are the relevant ones. Document 5 says "lost luggage reimbursed", so it shares no word with the query.

1. Count the query words each document contains: document 1 has one (`baggage`), document 2 has three, document 3 has one (`claim`), document 4 has three and document 5 has none.
2. **AND** keeps documents with all three words: `{2, 4}`. Precision is 2/2 = 1.000. Recall is 2/3 = 0.667. F1 is 2 x 1.000 x 0.667 / (1.000 + 0.667) = 0.800.
3. **OR** keeps documents with at least one word: `{1, 2, 3, 4}`. Precision is 2/4 = 0.500. Recall is still 2/3 = 0.667.
4. **Ranked top 3** sorts by the count from step 1, breaking ties by document number: 2, 4, 1. Precision is 2/3 = 0.667 and recall is 2/3 = 0.667.
5. No method finds document 5. The words are different, so word matching cannot reach it. That gap is called vocabulary mismatch.

In words: AND is precise but strict, OR is generous but noisy, and ranking lets you take the best few. The first block under "Code you can run" below reproduces these numbers.

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

The first block reproduces every number in the worked set above. It also checks that the counts describe a possible result set.

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

### The worked example in code

This block repeats the five-document example with sets, so the arithmetic above can be checked line by line.

```python
documents = {
    1: {"baggage", "policy"},
    2: {"delayed", "baggage", "claim", "form"},
    3: {"travel", "claim"},
    4: {"delayed", "baggage", "claim"},
    5: {"lost", "luggage", "reimbursed"},
}
relevant = {2, 4, 5}
query = {"delayed", "baggage", "claim"}

overlap = {doc: len(query & terms) for doc, terms in documents.items()}
results = {
    "AND": {d for d, n in overlap.items() if n == len(query)},
    "OR": {d for d, n in overlap.items() if n >= 1},
    "ranked top 3": set(sorted(overlap, key=lambda d: (-overlap[d], d))[:3]),
}
for name, found in results.items():
    hits = len(found & relevant)
    print(f"{name:13} returns {sorted(found)}  precision {hits / len(found):.3f}  recall {hits / len(relevant):.3f}")
```

**Reading the output.** The three rows match steps 2 to 4: AND returns `[2, 4]` with precision 1.000 and recall 0.667, OR returns four documents at precision 0.500, and the top 3 gives 0.667 for both. If recall ever exceeded 1.0 or precision came out above 1.0, the counts would be wrong.

### An experiment on real documents: keyword matching against ranking

Five documents hide the real problem. Now the same comparison runs on SciFact, a collection of 5,183 scientific abstracts with 300 test claims that each have at least one labelled supporting abstract. Each claim is a full sentence, which is a harsher test for exact matching than a two-word query. The block compares a strict AND, a "half of the terms" rule, an OR, and the top 1 and top 10 of BM25 (a ranking formula covered in [Session 5](/docs/theory/ir/vector-space-and-term-weighting)).

The data is cached by the Hugging Face `datasets` library. The first run downloads it. Versions used: Python 3.14.6, scikit-learn 1.9.1, NumPy 2.5.3, rank-bm25 0.2.2 and datasets 5.0.1.

```python
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]

relevant = defaultdict(set)
for row in qrels:
    if row["score"] > 0:
        relevant[str(row["query-id"])].add(str(row["corpus-id"]))
ids = [d["_id"] for d in corpus]
texts = [d["title"] + " " + d["text"] for d in corpus]
qtext = {q["_id"]: q["text"] for q in queries if q["_id"] in relevant}

vectoriser = CountVectorizer(binary=True, stop_words="english", token_pattern=r"[a-z0-9]+")
matrix = vectoriser.fit_transform(texts).tocsc()
analyse = vectoriser.build_analyzer()
bm25 = BM25Okapi([analyse(t) for t in texts])

def score(returned, truth):
    hits = len(set(returned) & truth)
    p = hits / len(returned) if returned else 0.0
    r = hits / len(truth)
    return p, r, 2 * p * r / (p + r) if p + r else 0.0

rows, sizes = defaultdict(list), defaultdict(list)
for qid, text in qtext.items():
    terms = sorted(set(analyse(text)))
    cols = [vectoriser.vocabulary_[t] for t in terms if t in vectoriser.vocabulary_]
    counts = np.asarray(matrix[:, cols].sum(axis=1)).ravel()
    order = np.argsort(-bm25.get_scores(analyse(text)))
    results = {
        "AND": [ids[i] for i in np.where(counts == len(terms))[0]],
        "half of terms": [ids[i] for i in np.where(counts >= max(1, len(terms) // 2))[0]],
        "OR": [ids[i] for i in np.where(counts >= 1)[0]],
        "BM25 top 1": [ids[order[0]]],
        "BM25 top 10": [ids[i] for i in order[:10]],
    }
    for name, returned in results.items():
        rows[name].append(score(returned, relevant[qid]))
        sizes[name].append(len(returned))

print(len(qtext), "queries,", len(ids), "documents,", sum(map(len, relevant.values())), "relevant pairs")
print(f"{'method':14}{'median size':>12}{'empty':>7}{'P':>7}{'R':>7}{'F1':>7}")
for name in rows:
    p, r, f = np.mean(rows[name], axis=0)
    print(f"{name:14}{np.median(sizes[name]):12.0f}{sum(s == 0 for s in sizes[name]):7d}{p:7.3f}{r:7.3f}{f:7.3f}")
```

The output of the run:

```text
300 queries, 5183 documents, 339 relevant pairs
method         median size  empty      P      R     F1
AND                      0    290  0.027  0.028  0.027
half of terms            6     26  0.179  0.634  0.223
OR                    1754      0  0.001  0.978  0.002
BM25 top 1               1      0  0.547  0.526  0.531
BM25 top 10             10      0  0.088  0.794  0.157
```

**Reading the output.** "Median size" is the number of documents the method returns for a typical query, and "empty" counts the queries that got nothing back. An empty answer scores precision 0 and recall 0 by the convention used here. The columns P, R and F1 are averages over the 300 queries.

**Line by line.**

- `CountVectorizer(binary=True, stop_words="english")` builds a document-by-word presence matrix. `build_analyzer()` reuses the same cleaning for queries, because index and query analysis must agree.
- `counts == len(terms)` is the AND rule. A query word missing from the whole vocabulary keeps the count short, so AND returns nothing.
- `BM25Okapi` ranks every document for each query. Taking `order[:1]` and `order[:10]` turns one ranking into two result sets of fixed size.
- `score()` returns precision 0 for an empty set, so a method that returns nothing is punished instead of ignored.

### What the numbers say

The strict AND returned nothing for 290 of 300 claims. A claim has many words, and one word that no abstract contains is enough to empty the answer. The ten queries that did return something averaged about 0.8 precision (0.027 x 300 / 10), but the other 290 count as zero, so the mean is only 0.027. The "half of the terms" rule rescues most queries (26 still come back empty) and reaches recall 0.634 at precision 0.179.

OR does the opposite. It finds almost every relevant abstract (recall 0.978) by returning a median of 1,754 of the 5,183 documents, so precision falls to 0.001. Nobody reads 1,754 results.

Ranking gets the best of both. BM25 top 1 reaches F1 0.531, far above every set-based rule, and the top 10 raises recall to 0.794. Precision at 10 looks poor at 0.088, but 339 relevant pairs over 300 queries means only about 1.13 relevant documents per query, so the ceiling for P@10 is 0.113. The system is at about 78 per cent of that ceiling.

The surprise is how little the "exact" Boolean method gives for natural-language questions. Limits: one collection, scientific abstracts only, full-sentence queries, no stemming, a single run with the library's default BM25 settings, and relevance labels that cover only a few documents per claim. Short keyword queries would make AND less harsh.

<Infographic src="/img/ir-enrich/ir1-keyword-vs-ranked.svg" alt="Bar charts of precision and recall for AND, half of the terms, OR and BM25 top 1 and top 10 on 300 SciFact claims." caption="Look first at the AND row: almost empty bars, because 290 of 300 queries returned nothing." />

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

For the metric example, imagine that the 60 relevant documents were all identified by such a review. A system retrieves 50 and gets 40 right. There are 10 false positives and 20 false negatives. An engineer can now inspect a sample of each error type: were the false positives caused by a common word, a stale document or a weak title boost? Were the false negatives caused by missing extraction, synonyms or a restrictive filter? The scalar numbers tell us *how much* error there is; the examples suggest *where* to act.

If reviewers have judged only a pool of top results, the assertion that exactly 60 relevant documents exist may be false. Recall then becomes an estimate conditional on the judged pool. Report that limitation instead of treating unreviewed material as irrelevant. Hold back some queries for later checks so repeated tuning does not simply memorise the small evaluation set.

### Connect search quality to operations

A successful offline score can still fail at launch if an index refresh takes hours, a document disappears from the source but remains searchable, or a permission change reaches the index later than the application. Define a freshness target for additions, edits and deletions; monitor each path. Log query latency separately for parsing, candidate lookup and scoring so a quality improvement does not silently break the response-time budget. Relevance and reliability are parts of the same user experience: a perfect result arriving after a timeout is no result at all.

## Where this stands in 2026

:::info Industry view

- A modern search application may combine term postings with vector candidates, but the need for document identity, filtering and relevance judgements remains. The lexical index is still useful for exact terms such as IDs and policy names.
- Lucene's current API exposes term dictionaries, postings and stored fields as separate structures. That separation mirrors the distinction between finding candidates and displaying results.
- For applications that feed retrieved text to an LLM, retrieval metrics alone are insufficient: the answer must also be checked for evidence use and factual correctness. See the site's [RAG chapter](/docs/genai/rag).

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Quoting precision and recall for a ranked list without a cutoff | They are the two famous metrics | State the cutoff, such as P@10 or recall@10, and add a rank-aware measure from Session 7 |
| Averaging precision only over queries that returned something | Empty answers look like "no data" | Count an empty answer as a failure, as the experiment does, and report how many there were |
| Treating unjudged documents as irrelevant | The label file lists only the relevant ones | Say that recall is measured against the judged set, and expect it to be optimistic or pessimistic accordingly |
| Using F1 to compare two rankings | One number is convenient | F1 ignores order. Two lists with the same documents in different orders get the same F1 |
| Using strict AND as the default for natural-language questions | Exact logic feels safe | Use ranking, or keep AND for filters such as tenant and status |

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

<details>
<summary><strong>Q6.</strong> (Medium) On SciFact, a strict AND of all query terms returned nothing for 290 of 300 claims. Give two reasons and one remedy.</summary>

Claims are full sentences with many terms, so a single word that appears in no abstract empties the AND. The abstracts also use different vocabulary from the claim (for example "luggage" for "baggage"). A remedy is to rank by overlap instead of demanding all terms, or to require only a fraction of the terms, as the "half of terms" row does.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) BM25 top 10 has precision 0.088 and recall 0.794. Is that a weak system?</summary>

Not necessarily. There are 339 relevant pairs for 300 queries, about 1.13 relevant documents per query, so the best possible P@10 averages 1.13 / 10 = 0.113. A score of 0.088 is about 78 per cent of that ceiling. The recall of 0.794 says that roughly four in five relevant abstracts appear in the first ten results.

</details>

## Go deeper

- [Introduction to Information Retrieval, chapters 1 and 8](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html); the foundational account of indexes and evaluation.
- [Apache Lucene index API](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/index/package-summary.html); a production library's term, postings and stored-field concepts.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact); the 5,183 abstracts and 1,109 claims used in the experiment. The dataset card lists the licence as CC BY-SA 4.0.
- [BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models](https://arxiv.org/abs/2104.08663); the benchmark paper, whose abstract calls BM25 a robust baseline.
- Built from the course lecture "ir-s1-intro" (Lecture Library series).

## Check yourself

- [ ] I can explain why unstructured search returns ranked candidates instead of exact database rows.
- [ ] I can draw the indexing and query paths and identify where a relevant document can disappear.
- [ ] I can calculate precision, recall and F1 from counts and state what each metric omits.
- [ ] I can choose a retrieval unit and explain how relevance labels and permissions affect evaluation.
- [ ] I can explain why a strict AND returned nothing for 290 of 300 SciFact claims and what a ranker does differently.
- [ ] I can read a table of precision, recall and F1 for set-based and ranked methods and name what each one hides.
- [ ] I can say why P@10 cannot exceed about 0.113 on a collection with 1.13 relevant documents per query.

## Where to go next

Next: [Session 2, Boolean retrieval](/docs/theory/ir/boolean-retrieval), where the index that makes AND and OR fast is built from sorted postings. Related: [Evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval) for metrics that reward order.
