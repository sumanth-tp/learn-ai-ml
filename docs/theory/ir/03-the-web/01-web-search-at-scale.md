---
id: ir-web-search-at-scale
title: "Information Retrieval · Session 9 — Web Search at Scale"
sidebar_label: "9 · Web search"
sidebar_position: 1
slug: /theory/ir/web-search-at-scale
description: "How web scale, query intent, spam, links and index overlap change the retrieval problem."
tags: [information-retrieval, web-search, index-sampling]
---

import Infographic from '@site/src/components/Infographic';
import IndexOverlapLab from '@site/src/components/viz/IndexOverlapLab';

**In one line.** Web search applies retrieval to a vast, changing and adversarial collection, so its quality depends on discovery, intent, duplicate control and multiple ranking signals.

## The idea in plain words

A library search system can usually enumerate its documents and decide when they change. A web engine cannot assume either. Pages appear and vanish, links lead to new pages, one article may exist at many URLs, and publishers may actively try to influence ranking. The engine must find pages, decide which copies to index, and make useful results available for a query that may be only two words long.

The lecture identifies three common query intents. An **informational** query asks for knowledge, such as how to repair a bicycle tyre. A **navigational** query seeks a particular site or page. A **transactional** query suggests an action such as buying a part or making a booking. These are useful lenses rather than rigid labels: `passport renewal` can mean instructions, an official portal or a fee payment. The intended result changes with the user and context.

Web pages form a directed graph. A link can help the crawler discover a target and can provide evidence that another page considers it useful. Neither the link nor term frequency is a verdict. A link farm can be built to manipulate authority; a page can repeat a phrase to match queries. Content, links, freshness and observed user behaviour may all inform ranking, but each can be noisy or biased.

<Infographic src="/img/ir/web-search.svg" alt="Web search connects a changing linked corpus to crawling, indexing and different query intents; the worked overlap probabilities imply a size ratio of 1.25." caption="The web adds discovery and adversarial signals to retrieval. The lower card reproduces the lecture's index-size example." />

:::note Beyond the lecture

The production pipeline, sampling caveats and evaluation workflow below extend the lecture's outline. The exact Google Search process described here is supported by Google's public documentation; ranking details it does not publish are not inferred.

:::

Move the two overlap probabilities. At the lecture defaults, $p_A=0.4$ and $p_B=0.5$, the estimated ratio is **$|A|/|B|=1.25$**. The bars show the direction of each sample; the table exposes the denominator for each probability.

<IndexOverlapLab />

## How it works

### Characteristics & queries

Billions of pages, a bow-tie graph, spam, and short informational/navigational/transactional queries. Ranking fuses content, link authority and usage signals.

### Sampling the web

Index size is estimated by sampling: a random A-page is in B with prob p_A; then |A|/|B| = p_B/p_A.

:::tip

**Worked.** p_A=0.4, p_B=0.5 → |A|/|B| = 0.5/0.4 = 1.25 (A is 25% larger).

:::


## A real system that works this way

**Google Search** publicly describes three stages in its [Search Central guide](https://developers.google.com/search/docs/fundamentals/how-search-works): crawling, indexing and serving. Googlebot discovers URLs from known pages, links and sitemaps. Google says its crawler chooses which sites to fetch and how often, and can slow down when a server signals trouble. The indexing stage analyses content and groups duplicate or similar pages to choose a canonical representative. When a user searches, the serving stage retrieves and ranks pages relevant to the query.

This is a production example of why a web engine cannot begin with a complete, fixed document collection. A URL in a sitemap is a discovery hint; it is not a guarantee of crawling or indexing. A crawled page is not automatically a result. The engine may select another URL as the canonical copy, and a page may remain unseen if it is inaccessible or low quality. Google's [canonicalisation documentation](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls) distinguishes redirects, canonical annotations and sitemap inclusion as signals of different strength, while saying the search engine can still choose a different canonical.

For an organisation running site search, the same separation is useful even at smaller scale. A connector discovers a document, an ingestion job extracts and deduplicates it, an index stores searchable representations, and a query service decides which result to show. Monitor each stage. A missing result can be a discovery or permissions failure, not a ranking failure.

## Code you can run

The first block verifies the lecture's ratio from two overlap probabilities. It also constructs exact finite sets with the same values, making the denominator visible: 20 shared pages out of 50 in A and 20 out of 40 in B.

```python
index_a = set(range(50))
index_b = set(range(30, 70))
shared = len(index_a & index_b)
p_a = shared / len(index_a)
p_b = shared / len(index_b)
estimated_ratio = p_b / p_a
actual_ratio = len(index_a) / len(index_b)
print(f"shared={shared} p_A={p_a:.1f} p_B={p_b:.1f}")
print(f"estimated |A|/|B|={estimated_ratio:.2f}; actual={actual_ratio:.2f}")
assert (shared, p_a, p_b, estimated_ratio) == (20, 0.4, 0.5, 1.25)
```

The identity follows because $p_A=|A\cap B|/|A|$ and $p_B=|A\cap B|/|B|$. Dividing $p_B$ by $p_A$ cancels the shared count and leaves $|A|/|B|$. It assumes the samples represent the two indexes and that membership can be measured consistently. A zero observed overlap gives no usable ratio.

The second block labels a tiny query set by its intended task. It is a transparent example of **human labels**, not a classifier. The same words can support several intents, so evaluation must specify what each query is meant to accomplish.

```python
queries = [
    ("how to replace bicycle tyre", "informational", "repair guide"),
    ("city library opening hours", "navigational", "official library page"),
    ("buy bicycle tyre", "transactional", "product checkout"),
]
for query, intent, desired_result in queries:
    print(f"{intent:13} {query} → {desired_result}")
assert {intent for _, intent, _ in queries} == {
    "informational", "navigational", "transactional"
}
```

These labels help assemble test queries; they do not establish that every query in a real log has one intent. A person searching for opening hours may want an answer card, a map or the official page. Ask what would count as success before scoring the result list.

## Designing with it

### Separate coverage from ranking

An index can only return pages it has discovered, fetched and accepted. If an important page is missing, raising its hypothetical ranking score will do nothing. Diagnose coverage with a stage trace: known URL, allowed fetch, successful response, extracted content, duplicate group, indexed representation and query match. Only then inspect the final score. The pipeline board is deliberately sequential because these failures have different owners and remedies.

The web's bow-tie graph is a structural reminder that a crawler following links from one seed cannot reach every page. A strongly connected core has pages that can reach one another, while IN pages can reach the core and OUT pages are reachable from it. Tendrils and disconnected components fall outside those paths. Sitemaps, submissions and diverse seeds help with discovery, but no public web index can honestly promise complete coverage.

### Treat adversarial content as part of the data

On the open web, an author can change a title, repeat keywords, create many near-identical pages or coordinate links. A ranker that treats every occurrence and link as independent evidence can be gamed. Deduplicate before measuring diversity; discount boilerplate; evaluate link signals alongside content and user utility. Spam detection is itself imperfect, so inspect false positives that might hide legitimate new sites or minority-language content.

Usage signals also have feedback loops. Top results receive more attention because of position, so raw clicks can reinforce an early ranking mistake. A navigational query may produce one click and immediate satisfaction; an informational query may be answered on the results page without a click. Use judged queries and carefully interpreted behaviour rather than declaring all clicks relevant and all unclicked pages bad.

### Design the result for the intent

An informational query may benefit from several independent sources, snippets and a clear answer location. A navigational query values the correct official page near the top. A transactional query may need current product or service details and a trustworthy destination. The same ranking measures from Session 7 still apply, but the relevance rubric should describe the user's task. A page that mentions a shop is not necessarily useful for completing a purchase.

| Intent | Example | Likely useful result | Common failure |
| --- | --- | --- | --- |
| Informational | `how to change a tyre` | Reliable instructions | Thin page repeating the query |
| Navigational | `city library website` | Official destination | Directory page outranks the institution |
| Transactional | `book a train seat` | Working booking flow | Stale or misleading landing page |

### Use index-size estimates cautiously

The lecture's overlap formula is exact for complete sets. A measured version uses samples and has uncertainty. If only a handful of sampled pages overlap, a small counting change can move the ratio substantially. URL normalisation, canonical groups, inaccessible pages and differing definitions of a "page" make membership ambiguous. Some pages exist in both indexes under different URLs; some copies share a URL but contain different content snapshots. State the sampling unit, collection time and matching rule.

A ratio also says nothing by itself about quality. Index A may contain 25% more pages than B while missing many pages users care about. Evaluate coverage by topic, language, region and update age. A larger index can even add low-quality duplicates that make ranking harder. The ratio is a way to reason about scale, not a product score.

### Protect latency and trust

Web serving must handle many short queries quickly. Candidate retrieval, expensive ranking and presentation therefore have distinct budgets. A system can cache common results or precompute static signals, while using query-dependent evidence at serving time. Keep permission checks and removal requests outside ranking heuristics: content that must not be served needs an enforced rule, not merely a low score. Preserve result provenance, canonical URL and crawl age so users and operators can judge whether a result is current.

## Trace one page from discovery to a result

Imagine a local council publishes a new page explaining a grant. A link from its news page exposes the URL to a crawler. The URL frontier schedules a fetch subject to host access rules and politeness. The response must be successful, the main text must be extractable, and the page may be grouped with a PDF copy or a print view. The index stores a chosen representation. Only when someone searches does query matching and ranking put it in a result list. A missing result might therefore mean the link was never seen, the crawl was blocked, the content was not extracted, a duplicate was selected, or the query did not match. Each case has a different fix.

The council example also shows why freshness matters. A grant deadline may change after the initial crawl. Re-fetching every URL every minute would overload sites and waste resources; never refreshing gives stale answers. A crawler can prioritise important or frequently changing pages and record when each indexed copy was last observed. A search product should avoid presenting an old deadline as current without checking the source. The exact refresh policy depends on host capacity, change frequency and user impact.

### See where the overlap formula can mislead

Suppose a sample from A has 40% of its pages in B and a sample from B has 50% in A. The lecture gives $|A|/|B|=1.25$. That is a ratio of the indexed sets under a consistent matching rule. If B's sample happens to contain many popular pages found everywhere, while A's sample includes more obscure pages, the two estimates may not represent uniform samples. If one engine counts a mobile URL and desktop URL separately and the other consolidates them, the sets are not even defined the same way. A robust study repeats samples, reports uncertainty and checks strata rather than treating 1.25 as an exact measurement of the live web.

The shared count cancels algebraically, but sampling error does not. An overlap of 20 gives the exact toy result because we constructed full sets. In a real estimate, 20 observed matches out of 50 and 20 out of 40 are noisy proportions. Their errors can be correlated because both concern the same overlap. If the indexes change while sampling is underway, the target ratio changes too. Record timestamps and prefer a defined snapshot whenever possible.

### Connect this chapter to the rest of retrieval

The inverted index from Sessions 1 to 4 still supports fast term lookup. The scoring models from Session 5 still help rank candidates. Session 7 still supplies measures for judged query results. What changes is the environment: documents are discovered through links, many copies refer to one content item, site owners may be adversarial, and the collection is never fully known. Session 10 explains the crawler; Session 11 explains one link signal. Neither replaces relevance judgement. A page can have excellent link authority and still fail a specific query, while a new page with few links can be the right answer.

For a retrieval-augmented assistant built on web pages, the same chain affects the answer. If the index uses stale content or collapses a crucial page into an unsuitable canonical copy, the generator will not see the right evidence. If a spam page enters the top passages, its text can contaminate an answer. Evaluate retrieval separately from generation, inspect sources and keep an audit trail from answer to indexed document and crawl time. The pipeline is useful precisely because it lets a team locate the failure before changing the model.

### Decide what a good search session means

One query is not always the whole task. A person may begin with broad information, reformulate with a product name and then navigate to a transaction. A web-search evaluation set should include those journeys as well as single queries. For a navigational query, success may be the correct destination at rank one. For a broad information query, credible diversity can matter more than one exact page. For a transaction, a result that opens a working flow is more useful than a page that merely contains the right words. This context determines which results should be labelled relevant and which offline metric deserves weight.

The lecture's three intent names are a starting vocabulary. Some queries mix them, and some users change intent mid-session. Keep the rubric open to ambiguity: multiple result types can be acceptable if they serve plausible needs. When an evaluation score falls, inspect the actual result, query and user task before deciding whether the ranker, crawler or interface needs work.

## Where this stands in 2026

:::info Industry view

- Google's current public Search guide still separates crawling, indexing and serving; it also says none of the stages is guaranteed for every URL.
- Canonicalisation remains part of index quality, because multiple URLs may represent the same or similar page.
- Search quality is now evaluated across different user intents and collection slices, while adversarial pages and feedback bias still need explicit attention.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> How does web search differ from classic IR?</summary>

It operates at web scale (billions of pages) with spam, hyperlinks, duplication and huge query volume, fusing content, link and usage signals.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Name the three web query types.</summary>

Informational, navigational, and transactional.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A random A-indexed page is in B with prob 0.4; a random B page is in A with 0.5. Estimate |A|/|B|.</summary>

|A|/|B| = p_B/p_A = 0.5/0.4 = 1.25; A's index is 25% larger.<br /><em>Session 9 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why must web ranking resist spam?</summary>

Because content is adversarial; keyword stuffing and link farms try to manipulate rankings; so signals must be robust and combined.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is the bow-tie structure of the web graph?</summary>

A large strongly-connected core with IN and OUT components (pages that only link into or out of the core) plus tendrils/disconnected pages.<br /><em>Session 9 · conceptual</em>

</details>

## Go deeper

- [Stanford IR book: web search basics](https://nlp.stanford.edu/IR-book/html/htmledition/web-search-basics-1.html); the classical web-search framing.
- [Google Search Central: how Search works](https://developers.google.com/search/docs/fundamentals/how-search-works); a current public pipeline description.
- [Google Search Central: canonical URLs](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls); duplicate consolidation signals.
- Built from the course lecture "ir-s9-web-search" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can explain why web retrieval needs discovery, duplicate control and spam resistance in addition to ranking.
- [ ] I can distinguish informational, navigational and transactional intents without assuming every query has one fixed label.
- [ ] I can derive the 1.25 index-size ratio and state its sampling assumptions.
- [ ] I can trace a missing result through crawling, indexing and serving before changing its score.
