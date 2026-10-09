---
id: ir-boolean-retrieval
title: "Information Retrieval · Session 2 — Boolean Retrieval"
sidebar_label: "2 · Boolean retrieval"
sidebar_position: 2
slug: /theory/ir/boolean-retrieval
description: "How an inverted index answers AND, OR and NOT queries by merging sorted postings, and why ranking remains necessary."
tags: [information-retrieval, boolean-search, inverted-index, postings]
---

import Infographic from '@site/src/components/Infographic';
import BooleanPostingsLab from '@site/src/components/viz/BooleanPostingsLab';

**In one line.** A Boolean query is a set expression over documents, evaluated efficiently with sorted postings lists.

:::tip Before you start

**You should already know**

- What precision and recall mean ([Session 1](/docs/theory/ir/what-information-retrieval-is)).
- How a Python list and a `while` loop with two index variables work.

**Reading time:** about 40 minutes, plus a few minutes to run the code.

**After this chapter you can**

- Intersect two sorted postings lists by hand and count the comparisons.
- Say when skip pointers help and when they cost more than they save.
- Compare an index lookup with a scan of every document on a real collection.

:::

## In 30 seconds

The index at the back of a textbook lists, for each word, the pages where it appears. Asking for "delayed AND baggage" means taking two page lists and keeping the pages that appear in both. Because each list is sorted, you walk down both together and never step backwards. That costs far less than reading every page.

A skip pointer is a shortcut printed in the margin. When one list is short and the other is long, it lets you jump over several entries at once, because you know the target is further down.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Inverted index | A map from each term to the documents that contain it | `delayed` -> 2, 4, 7 |
| Postings list | The sorted list of document IDs for one term | `[1, 2, 4, 11, 31]` |
| Intersection (AND) | Documents present in both lists | `[1, 2, 4, 31]` |
| Union (OR) | Documents present in either list | `[1, 2, 4, 5, 11, 31]` |
| Merge | Walking two sorted lists together with two pointers | Advance the pointer at the smaller ID |
| Comparison | One test of one ID against another; the unit of work counted here | `a[i] == b[j]` |
| Skip pointer | A forward link that jumps several entries in a long list | From position 0 straight to position 4 |
| Selectivity | How few documents a term matches; rare terms are selective | A term in 200 of 100,000 documents |


## The idea in plain words

Imagine six documents. Rather than opening all six for every query, keep a dictionary from each term to the IDs of documents containing it. That dictionary and its postings lists are an **inverted index**. A query such as `insurance AND claim` asks for the intersection of two postings lists. `insurance OR claim` asks for their union. `insurance NOT claim` asks for documents in the first list but outside the second.

Take two lists, A = `[1, 2, 4, 11, 31]` and B = `[1, 2, 4, 5, 31]`. Their intersection is `[1, 2, 4, 31]`. Because both lists are sorted, two pointers can find that answer without looking at every document in the collection. When IDs are equal, emit one and advance both. When they differ, advance the pointer at the smaller ID: that ID can never match a larger one in the other list.

<Infographic src="/img/ir/boolean-postings.svg" alt="Two sorted postings lists intersect through an AND operation to return document IDs 1, 2, 4 and 31." caption="A sorted-postings example. The pointers meet on four document IDs." />

The output is an **unranked set**. For a strict compliance filter, an exact set can be ideal. For a user asking a broad question, it is rarely enough: 500 documents that satisfy an expression are not equally useful. Modern search often combines Boolean filters with a ranked query, applying exact conditions for permissions or dates while ordering the surviving documents by relevance.

:::note Added for this site

The query-planning and permission examples below extend the set-operation example. They explain how the same mechanics become useful inside a larger search system.

:::

The default AND query returns **1, 2, 4, 31**, matching the Python example. Toggle a document in either list or change the set operation to see exactly which IDs survive.

<BooleanPostingsLab />

## Worked example, step by step

Let list A be the numbers 1 to 32, and list B be the single number 31. We want A AND B and we count comparisons. Each trip round the loop counts one comparison. Each test of a skip target counts one more.

1. **Plain merge.** A's pointer starts at 1 and B's at 31. Every A value below 31 is smaller, so A advances one step each time: 30 comparisons for the values 1 to 30. The 31st comparison finds 31 = 31. Total: 31.
2. **Skip pointers every 4 entries.** At position 0 (value 1), compare with 31 (1 comparison), then test the skip target: A's entry four places on is 5, which is at most 31, so jump (1 more). Each jump of four entries costs 2. Jumps start at positions 0, 4, 8, 12, 16, 20 and 24: that is 7 jumps, 14 comparisons, landing on position 28 (value 29).
3. At position 28 there is no skip target left, so compare 29 < 31 (comparison 15), then 30 < 31 (16), then 31 = 31 (17). Total: 17.
4. **Skip pointers every 8 entries.** Three jumps (positions 0, 8, 16) cost 6 comparisons and land on value 25. Then 25, 26, 27, 28, 29 and 30 each need one comparison (6 more), and 31 = 31 needs one: total 13.

In words: when one list is tiny and the other is long, jumping saves most of the work, and a longer span saves more up to a point. The first block below prints 31, 17 and 13.

## How it works

### Boolean & inverted index

Documents are term sets; AND/OR/NOT return an unranked exact set via an inverted index (term → sorted postings).

:::tip

**Worked.** [1,2,4,11,31] AND [1,2,4,5,31] = [1,2,4,31] in x+y=10 steps.

:::

### Why ranking wins

Boolean = exact, controllable, but unranked and expert-only. Ranked retrieval scores documents so the best appear first; the modern default.


## A real system that works this way

In **Apache Lucene**, terms are associated with postings. Its query classes let applications compose term queries under Boolean conditions; a searcher then evaluates and scores matching documents. The [search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html) describes term queries, query composition and scoring. A user-facing product built on Lucene still decides how to parse a typed query, which conditions are filters, and which fields deserve a relevance score.

Consider a policy search with fields `tenant`, `status`, `title` and `body`. The product might require `tenant = west` AND `status = published`, then rank the matching documents for the words `travel expenses`. The first two conditions are hard eligibility rules; they must not be softened by a semantic match. The text query is a relevance request; its best answer should come first. Treating the whole request as one rigid Boolean expression would force users to know the exact wording in the policy. Treating the tenant rule as a ranking hint would leak material from another tenant.

The postings abstraction is useful because it separates *which documents might match* from *what a document is*. A posting can contain more than an ID: term frequency, positions and other metadata support phrase queries and scoring. A stored field can then supply the title or snippet for a matched ID. The tiny lists in the lab include only IDs, which is exactly the information needed to understand Boolean matching.

## Code you can run

A two-pointer merge gives the expected answer. Counting comparisons clarifies its cost: the two list lengths sum to 10, which is an **upper bound** on pointer advances, while this particular intersection needs six ID comparisons.

```python
def intersect(left, right):
    i = j = comparisons = 0
    matches = []
    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] == right[j]:
            matches.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return matches, comparisons

a = [1, 2, 4, 11, 31]
b = [1, 2, 4, 5, 31]
answer, comparisons = intersect(a, b)
print(answer, comparisons, "ID comparisons")
assert answer == [1, 2, 4, 31]
assert comparisons == 6
```

The program prints `[1, 2, 4, 31] 6 ID comparisons`. The algorithm is $O(|A|+|B|)$ in the worst case; it does not promise exactly ten comparisons for these inputs.

:::note Correction

A common statement of this example is that it takes `x+y=10 steps`. Ten is the simple upper bound from the two list lengths. Six ID comparisons suffice for these particular values, as the runnable trace shows.

:::

The same IDs demonstrate all three operators. Here `NOT` is relative to A. A free-standing `NOT claim` instead needs a defined collection universe and should usually be paired with a positive term or filter.

```python
a = {1, 2, 4, 11, 31}
b = {1, 2, 4, 5, 31}
print("AND:", sorted(a & b))
print("OR:", sorted(a | b))
print("A NOT B:", sorted(a - b))
assert sorted(a - b) == [11]
```

This set-based block is convenient for a small demonstration. A production index keeps lists ordered or uses other compressed representations so it can avoid materialising every intermediate set.

### The worked example in code

This block runs the same two lists with no skips, a span of 4 and a span of 8.

```python
def walk(a, b, span):
    i = j = steps = 0
    while i < len(a) and j < len(b):
        steps += 1
        if a[i] == b[j]:
            i += 1; j += 1
        elif a[i] < b[j]:
            if span and i % span == 0 and i + span < len(a):
                steps += 1
                if a[i + span] <= b[j]:
                    i += span
                    continue
            i += 1
        else:
            j += 1
    return steps

long_list = list(range(1, 33))
short_list = [31]
print("plain merge steps:", walk(long_list, short_list, 0))
print("skip every 4 steps:", walk(long_list, short_list, 4))
print("skip every 8 steps:", walk(long_list, short_list, 8))
```

**Reading the output.** The three lines print 31, 17 and 13, matching steps 1 to 4. `walk` counts one comparison per loop and one more per skip test, the same rule as the steps.

**Line by line.**

- `if span and i % span == 0 and i + span < len(a)` allows a skip only from positions that carry a pointer (every `span` entries) and only if the target exists.
- `continue` after a jump re-compares at the new position instead of also stepping one forward.

### An experiment on real postings: scan, merge and skip pointers

Real queries have lists of very different lengths. This block builds an inverted index over the 5,183 SciFact abstracts, takes every pair of distinct words in each of the 1,109 SciFact queries (keeping every seventh pair, 12,015 in all), and measures three things. First, a linear scan of every document against the same AND. Second, a plain merge. Third, merges with skip pointers: one every $\sqrt{L}$ entries (the textbook rule) and fixed spans of 4, 16 and 64. No stop words are removed, so very common words give long lists, as in a real index.

Versions used: Python 3.14.6, scikit-learn 1.9.1, NumPy 2.5.3. Timings depend on the machine, so the comparison counts carry the lesson.

```python
from itertools import combinations
from math import isqrt
from time import perf_counter

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
texts = [d["title"] + " " + d["text"] for d in corpus]

vectoriser = CountVectorizer(binary=True, token_pattern=r"[a-z0-9]+")
matrix = vectoriser.fit_transform(texts).tocsc()
vocab = vectoriser.vocabulary_
postings = {t: matrix.indices[matrix.indptr[c]:matrix.indptr[c + 1]].tolist() for t, c in vocab.items()}
term_sets = [set(row) for row in vectoriser.inverse_transform(matrix.tocsr())]

def merge(a, b, span=None):
    sa = span or max(1, isqrt(len(a)))
    sb = span or max(1, isqrt(len(b)))
    i = j = steps = 0
    out = []
    while i < len(a) and j < len(b):
        steps += 1
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            if span != 0 and i % sa == 0 and i + sa < len(a):
                steps += 1
                if a[i + sa] <= b[j]:
                    i += sa
                    continue
            i += 1
        else:
            if span != 0 and j % sb == 0 and j + sb < len(b):
                steps += 1
                if b[j + sb] <= a[i]:
                    j += sb
                    continue
            j += 1
    return out, steps

analyse = vectoriser.build_analyzer()
pairs = []
for q in queries:
    terms = sorted({t for t in analyse(q["text"]) if t in postings})
    pairs += list(combinations(terms, 2))
pairs = pairs[::7]
lists = [sorted((postings[a], postings[b]), key=len) for a, b in pairs]
print(len(pairs), "term pairs from real queries,", len(postings), "terms,", len(texts), "documents")

def timed(fn):
    best = 1e9
    for _ in range(3):
        start = perf_counter()
        out = [fn(a, b) for a, b in lists]
        best = min(best, (perf_counter() - start) * 1e6 / len(lists))
    return out, best

plain, t_plain = timed(lambda a, b: merge(a, b, 0))
skips, t_skip = timed(merge)
start = perf_counter()
scan = [[d for d, t in enumerate(term_sets) if x in t and y in t] for x, y in pairs]
t_scan = (perf_counter() - start) * 1e6 / len(pairs)
assert all(p[0] == s[0] == r for p, s, r in zip(plain, skips, scan))
print(f"linear scan {t_scan:7.1f} us | merge {t_plain:6.1f} us | skip pointers {t_skip:6.1f} us per query")

ratio = np.array([len(b) / len(a) for a, b in lists])
cost = {"merge": np.array([s for _, s in plain]), "skip sqrt(L)": np.array([s for _, s in skips])}
for span in (4, 16, 64):
    cost[f"skip span {span}"] = np.array([merge(a, b, span)[1] for a, b in lists])
for lo, hi in ((1, 4), (4, 30), (30, 1e9)):
    mask = (ratio >= lo) & (ratio < hi)
    row = "  ".join(f"{name} {values[mask].mean():7.1f}" for name, values in cost.items())
    print(f"length ratio {lo:>2}-{min(hi, 999):<3} n={mask.sum():5d}  mean comparisons: {row}")
```

The output of one run (timings differ between runs):

```text
12015 term pairs from real queries, 35734 terms, 5183 documents
linear scan   633.8 us | merge  162.7 us | skip pointers  153.7 us per query
length ratio  1-4   n= 3876  mean comparisons: merge  1532.4  skip sqrt(L)  1558.8  skip span 4  1637.1  skip span 16  1582.5  skip span 64  1545.2
length ratio  4-30  n= 4544  mean comparisons: merge  2122.5  skip sqrt(L)  2089.9  skip span 4  1565.7  skip span 16  1726.8  skip span 64  2105.6
length ratio 30-999 n= 3595  mean comparisons: merge  2710.8  skip sqrt(L)  1228.6  skip span 4  1440.5  skip span 16   735.3  skip span 64  1236.3
```

**Reading the output.** "Length ratio" is the longer list divided by the shorter one, so the three rows go from balanced pairs to very uneven ones. Each number is a mean count of comparisons per query pair. The assertion in the code confirms that all three methods return identical document sets.

**Line by line.**

- `postings` slices the sparse matrix's column index arrays, which are already sorted document numbers, so no extra sorting is needed.
- `sorted((postings[a], postings[b]), key=len)` puts the shorter list first, the shortest-first plan from the Designing section.
- `merge(a, b, 0)` is the plain merge: `span != 0` switches skipping off. With `span=None` the function uses $\lfloor\sqrt{L}\rfloor$ for each list.
- `timed` runs each method three times and keeps the fastest, which damps noise from other processes.

### What the numbers say

Skip pointers are not a free win. For balanced lists (ratio 1 to 4) the $\sqrt{L}$ rule needed 1,558.8 comparisons against 1,532.4 for the plain merge, because the skip tests cost more than the jumps save. For very uneven lists (ratio 30 or more) the same rule cut the work from 2,710.8 to 1,228.6, a fall of 55 per cent. A span of 16 did better still at 735.3, so the textbook span is a sensible default and not the optimum for this collection.

On wall time the picture is flatter. Across six runs made while developing the script, the scan took 612 to 897 microseconds per query, the merge 139 to 163 and the skip version 138 to 167. So the index made AND between 3.8 and 6.0 times faster than the scan on 5,183 documents, while skip pointers were indistinguishable from the plain merge in Python.

The surprise is that the saving in comparisons did not appear as saved time. A likely reason is interpreter overhead per comparison, though I did not profile it. In a compiled engine the saving may well show, and the scan gap would also grow with the number of documents, because the scan reads all of them while the merge reads only the postings. Limits: a small collection, pure Python, in-memory uncompressed lists, evenly spaced skips and one query set.

<Infographic src="/img/ir-enrich/ir1-skip-pointers.svg" alt="Bars of mean comparisons for plain merge, skips every square root of L and skips every 16 across three list-length ratios." caption="Look first at the bottom group: for very uneven lists the skip bars are far shorter than the plain merge bar." />

## Designing with it

### Plan multi-term queries from the rare terms

For `A AND B AND C`, order the intersections so the shortest postings lists meet early. A rare term produces a small intermediate result; each later merge then has less work. For example, if `travel` occurs in 100,000 documents, `baggage` in 2,000 and `delayed` in 200, start with `delayed AND baggage`. Starting with the common term can create a large intermediate list that the rare term later discards. This is a planning rule, not a semantic one: intersection is commutative, so the final answer stays the same.

The actual best plan also depends on index layout, skips, cache and term statistics. The simple shortest-first rule teaches the key idea: use cheap selectivity estimates to reduce unnecessary work. The [Stanford IR book's Boolean chapter](https://nlp.stanford.edu/IR-book/html/htmledition/boolean-retrieval-1.html) develops the postings representation and multi-term processing.

### Decide which conditions are hard

| Condition | Typical role | What goes wrong if treated the other way |
| --- | --- | --- |
| Tenant or document permission | Mandatory filter | Cross-tenant or unauthorised text can appear |
| Publication status | Mandatory filter | Deleted or draft content can leak into results |
| Query words from a person | Candidate generation and scoring | A strict AND can return zero useful results for a paraphrase |
| Date range selected explicitly | Usually a filter | Old material may crowd out the requested period |
| A preferred document type | Usually a boost | A hard filter can hide the best answer in another format |

The query parser must be explicit about grouping. `A OR B AND C` is ambiguous to a reader even if an implementation has a precedence rule. Present or log the parsed form, such as `A OR (B AND C)`. When a product offers an advanced syntax, test unbalanced parentheses, empty operands and escaping. Do not silently broaden a malformed permission expression.

### Understand the set model's limits

Exact membership is useful for constraints but brittle as an entire relevance model. A document containing a term in a footnote and a document centred on that term both enter the same set. Synonyms, spelling errors and morphology can exclude the best answer. OR broadens the set but may admit noise; AND narrows it but may miss paraphrases. Later chapters add tolerant term matching, term weighting and ranked evaluation.

The index is also a snapshot. A document added after the last refresh may be absent; a deleted document may remain until the update propagates. Define freshness targets, test the deletion path and keep permission filtering correct even during an index lag. A Boolean answer can be logically correct against the index and still be wrong against the current business state.

### Make the result auditable

For regulated or internal search, it is useful to record which term and filter clauses matched an ID. That trace helps explain why a document appeared and diagnose accidental exclusion. It should be kept separate from user-visible ranking explanations, which need plain language and should not expose private terms or fields. A query log should record enough structure to reproduce a failure without storing sensitive raw text longer than necessary.

## Trace a three-term query

Consider `delayed AND baggage AND policy`. Suppose their postings lengths are 200, 2,000 and 100,000. The final intersection is independent of order, but the amount of intermediate work is not. Intersecting the two short lists first can produce, say, 35 IDs. Intersecting those 35 with the policy list is then cheap relative to intersecting a 100,000-ID list with a 2,000-ID list first. The numbers here illustrate the planner's reasoning; a real engine estimates cost using the current index's statistics and its own data structures.

The two-pointer algorithm also makes a useful invariant visible. At any moment, all IDs smaller than the current pointer in either list have been fully considered. If A points at 11 and B points at 5, document 5 cannot occur later in A because A is sorted and already at 11. Advancing B is therefore safe. When both pointers show 31, 31 belongs to the intersection and both pointers advance. This argument is what makes the code correct, not merely the fact that it produced the expected answer once.

OR needs a different merge: emit the smaller current ID, advance its pointer, and emit a shared ID only once. A NOT B keeps an A ID only if B has passed it or B is exhausted. All three benefit from sorted postings. A free-standing NOT over a huge collection can be expensive because it describes almost everything; pairing exclusion with a selective positive term is usually more useful. For example, `travel AND NOT draft` starts with travel candidates and removes the draft IDs.

### Make the parser part of the contract

Boolean syntax is powerful because it lets a user specify exact logic, but a product should never assume that all users will write it correctly. Decide whether quoted phrases, parentheses, field names and implicit AND are supported. Show clear errors for invalid expressions. A query like `claims OR expenses AND travel` should be displayed or logged in its parsed grouping so a result can be explained. A training page can teach syntax, but the default search box should still be useful for ordinary words.

The analyser must also agree with the syntax. `New York` as two term operands is different from an exact phrase query. A query containing a hyphen might be one token, two tokens or a negation operator depending on the parser. Those details should be tested using the same analyser and parser configuration that the live application uses. A mismatch can make a Boolean expression appear logically wrong when the real problem is tokenisation.

### Separate eligibility from relevance

An access-control filter should apply to every result, including snippets and cached answers. It must not be replaced by a low score for forbidden documents: even one high-scoring forbidden item would be a leak. Conversely, forcing all relevance words into a mandatory AND often makes recall collapse. A sensible search request can therefore have two layers: a hard filter expression for tenant, status and explicit constraints, and a scoring expression for the information need. Keeping these layers distinct also makes debugging easier, because a missing result can be traced to either eligibility or relevance.

Some data changes faster than the text index. If a permission is revoked before the next index refresh, a filter stored only in the stale index may be insufficient. The application may need to check the authoritative permission service at result time or guarantee a suitably fast update path. This is a system-design choice, not a Boolean algebra problem, yet it determines whether the apparently correct set answer is safe to serve.

### Test the edge cases

The example lists have several common IDs, making the intersection easy to see. Tests should also cover disjoint lists, an empty list, duplicate IDs in malformed input, one list exhausted before the other, and a very common term paired with a rare one. A sorted-list merge expects sorted, deduplicated inputs; validate that precondition when building a teaching implementation. In a production index, the format and iterator APIs enforce their own guarantees. For a query planner, measure both the count of matches and the work done: two plans can return the same IDs while having very different latency.

## Where this stands in 2026

:::info Industry view

- Boolean matching remains a useful part of search in 2026 because exact permissions, tenant boundaries and publication states are set conditions even when text relevance is scored by a richer model.
- Lucene's search API includes term, Boolean, phrase, prefix, wildcard and fuzzy query classes. Those are distinct operations with different cost and recall characteristics.
- Hybrid retrieval still benefits from Boolean filters: a vector candidate that violates a hard access rule cannot be rescued by a high similarity score.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Adding skip pointers to every list | They sound like pure speed | Keep them for long lists that meet short ones; on balanced lists the extra tests cost more than they save |
| Reading comparison counts as latency | Fewer comparisons must mean faster | Measure time on the real engine as well. Here the count fell by 55 per cent for uneven lists and the Python timing did not move |
| Intersecting terms in the order the user typed them | The query reads left to right | Order by list length, shortest first, so intermediate results stay small |
| Running a standalone `NOT` | It is a valid set operation | Pair it with a selective positive term, such as `travel AND NOT draft` |
| Merging lists that are unsorted or hold duplicates | The code works on a test with clean data | Validate the precondition when you build the index |

## Practice questions

<details>
<summary><strong>Q1.</strong> How does the Boolean model answer a query and what are its weaknesses?</summary>

Combines terms with AND/OR/NOT as set intersect/union/complement on postings, returning an unranked exact set; weaknesses: no ranking, brittle, expert syntax.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What does an inverted index store?</summary>

For each term, a sorted postings list of document IDs containing it.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Intersect [1,2,4,11,31] and [1,2,4,5,31]; give the cost.</summary>

[1, 2, 4, 31] via a linear merge in O(x+y) = 10 steps.<br /><em>Session 2 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> For a multi-term AND, which term is processed first and why?</summary>

The rarest (shortest postings list), to minimise intermediate result sizes and total work.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Contrast Boolean and ranked retrieval.</summary>

Boolean: exact, controllable, unranked, expert-only. Ranked: scores documents by relevance so the best appear first (better for end users).<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) In the experiment, skip pointers every square root of L cut comparisons by more than half for one group of pairs and raised them for another. Which groups, and why?</summary>

For lists with a length ratio of 30 or more they fell from 2,710.8 to 1,228.6 (55 per cent), because most of the long list can be jumped over. For balanced lists (ratio 1 to 4) they rose from 1,532.4 to 1,558.8, because both pointers advance together and each skip test is wasted work.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) A linear scan took about 600 to 900 microseconds per query against 140 to 160 for the merge on 5,183 documents. Why would the gap be much larger on 5 million documents?</summary>

The scan touches every document, so its cost grows with the collection size. The merge touches only the postings of the query terms, which grow far more slowly for most terms. Both figures here are Python timings on a small collection, so the exact ratio will not transfer, but the direction will.

</details>

## Go deeper

- [Stanford IR book: Boolean retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/boolean-retrieval-1.html); the original postings-list model and query processing.
- [Apache Lucene search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html); the production query classes that compose these operations.
- [Stanford IR book: faster postings list intersection via skip pointers](https://nlp.stanford.edu/IR-book/html/htmledition/faster-postings-list-intersection-via-skip-pointers-1.html); the square-root placement heuristic and the trade-off between skip count and span.
- [SciFact in the BEIR collection](https://huggingface.co/datasets/BeIR/scifact); the corpus and queries used in the experiment.
- Built from the course lecture "ir-s2-boolean" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can construct an inverted index and read a sorted postings list.
- [ ] I can intersect two postings lists with pointers and explain the worst-case cost.
- [ ] I can distinguish an exact eligibility filter from a relevance score.
- [ ] I can explain why Boolean matching alone does not order useful results first.
- [ ] I can count the comparisons of a plain merge and a skip-pointer merge for two small lists.
- [ ] I can explain why skip pointers helped by 55 per cent on very uneven lists but not on balanced ones.
- [ ] I can say why a drop in comparisons does not always become a drop in latency.

## Where to go next

Next: [Session 3, dictionaries and tolerant search](/docs/theory/ir/dictionaries-and-tolerant-search), where a query word that misses the dictionary is repaired before the lookup. Related: [Index construction and compression](/docs/theory/ir/index-construction-and-compression), which shrinks the postings lists used here.
