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

## The idea in plain words

Imagine six documents. Rather than opening all six for every query, keep a dictionary from each term to the IDs of documents containing it. That dictionary and its postings lists are an **inverted index**. A query such as `insurance AND claim` asks for the intersection of two postings lists. `insurance OR claim` asks for their union. `insurance NOT claim` asks for documents in the first list but outside the second.

The lecture's lists are A = `[1, 2, 4, 11, 31]` and B = `[1, 2, 4, 5, 31]`. Their intersection is `[1, 2, 4, 31]`. Because both lists are sorted, two pointers can find that answer without looking at every document in the collection. When IDs are equal, emit one and advance both. When they differ, advance the pointer at the smaller ID: that ID can never match a larger one in the other list.

<Infographic src="/img/ir/boolean-postings.svg" alt="Two sorted postings lists intersect through an AND operation to return document IDs 1, 2, 4 and 31." caption="The lecture's sorted-postings example. The pointers meet on four document IDs." />

The output is an **unranked set**. For a strict compliance filter, an exact set can be ideal. For a user asking a broad question, it is rarely enough: 500 documents that satisfy an expression are not equally useful. Modern search often combines Boolean filters with a ranked query, applying exact conditions for permissions or dates while ordering the surviving documents by relevance.

:::note Beyond the lecture

The query-planning and permission examples below extend the lecture's set-operation demonstration. They explain how the same mechanics become useful inside a larger search system.

:::

The default AND query returns **1, 2, 4, 31**, matching the Python example. Toggle a document in either list or change the set operation to see exactly which IDs survive.

<BooleanPostingsLab />

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

A two-pointer merge gives the lecture's answer. Counting comparisons clarifies its cost: the two list lengths sum to 10, which is an **upper bound** on pointer advances, while this particular intersection needs six ID comparisons.

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

:::note Correction to the lecture

The lecture says this example takes `x+y=10 steps`. Ten is the simple upper bound from the two list lengths. Six ID comparisons suffice for these particular values, as the runnable trace shows.

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

The two-pointer algorithm also makes a useful invariant visible. At any moment, all IDs smaller than the current pointer in either list have been fully considered. If A points at 11 and B points at 5, document 5 cannot occur later in A because A is sorted and already at 11. Advancing B is therefore safe. When both pointers show 31, 31 belongs to the intersection and both pointers advance. This argument is what makes the code correct, not merely the fact that it produced the lecture's answer once.

OR needs a different merge: emit the smaller current ID, advance its pointer, and emit a shared ID only once. A NOT B keeps an A ID only if B has passed it or B is exhausted. All three benefit from sorted postings. A free-standing NOT over a huge collection can be expensive because it describes almost everything; pairing exclusion with a selective positive term is usually more useful. For example, `travel AND NOT draft` starts with travel candidates and removes the draft IDs.

### Make the parser part of the contract

Boolean syntax is powerful because it lets a user specify exact logic, but a product should never assume that all users will write it correctly. Decide whether quoted phrases, parentheses, field names and implicit AND are supported. Show clear errors for invalid expressions. A query like `claims OR expenses AND travel` should be displayed or logged in its parsed grouping so a result can be explained. A training page can teach syntax, but the default search box should still be useful for ordinary words.

The analyser must also agree with the syntax. `New York` as two term operands is different from an exact phrase query. A query containing a hyphen might be one token, two tokens or a negation operator depending on the parser. Those details should be tested using the same analyser and parser configuration that the live application uses. A mismatch can make a Boolean expression appear logically wrong when the real problem is tokenisation.

### Separate eligibility from relevance

An access-control filter should apply to every result, including snippets and cached answers. It must not be replaced by a low score for forbidden documents: even one high-scoring forbidden item would be a leak. Conversely, forcing all relevance words into a mandatory AND often makes recall collapse. A sensible search request can therefore have two layers: a hard filter expression for tenant, status and explicit constraints, and a scoring expression for the information need. Keeping these layers distinct also makes debugging easier, because a missing result can be traced to either eligibility or relevance.

Some data changes faster than the text index. If a permission is revoked before the next index refresh, a filter stored only in the stale index may be insufficient. The application may need to check the authoritative permission service at result time or guarantee a suitably fast update path. This is a system-design choice, not a Boolean algebra problem, yet it determines whether the apparently correct set answer is safe to serve.

### Test the edge cases

The lecture's lists have several common IDs, making the intersection easy to see. Tests should also cover disjoint lists, an empty list, duplicate IDs in malformed input, one list exhausted before the other, and a very common term paired with a rare one. A sorted-list merge expects sorted, deduplicated inputs; validate that precondition when building a teaching implementation. In a production index, the format and iterator APIs enforce their own guarantees. For a query planner, measure both the count of matches and the work done: two plans can return the same IDs while having very different latency.

## Where this stands in 2026

:::info Industry view

- Boolean matching remains a useful part of search in 2026 because exact permissions, tenant boundaries and publication states are set conditions even when text relevance is scored by a richer model.
- Lucene's search API includes term, Boolean, phrase, prefix, wildcard and fuzzy query classes. Those are distinct operations with different cost and recall characteristics.
- Hybrid retrieval still benefits from Boolean filters: a vector candidate that violates a hard access rule cannot be rescued by a high similarity score.

:::

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

## Go deeper

- [Stanford IR book: Boolean retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/boolean-retrieval-1.html); the original postings-list model and query processing.
- [Apache Lucene search API](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/package-summary.html); the production query classes that compose these operations.
- Built from the course lecture "ir-s2-boolean" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can construct an inverted index and read a sorted postings list.
- [ ] I can intersect two postings lists with pointers and explain the worst-case cost.
- [ ] I can distinguish an exact eligibility filter from a relevance score.
- [ ] I can explain why Boolean matching alone does not order useful results first.
