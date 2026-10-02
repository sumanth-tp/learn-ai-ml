---
id: ir-web-crawling-distributed-indexes
title: "Information Retrieval · Session 10 — Web Crawling and Distributed Indexes"
sidebar_label: "10 · Crawl the web"
sidebar_position: 2
slug: /theory/ir/web-crawling-and-distributed-indexes
description: "How a URL frontier, host politeness, duplicate control and distributed indexing make a web search collection possible."
tags: [information-retrieval, web-crawling, robots, distributed-index]
---

import Infographic from '@site/src/components/Infographic';
import CrawlerRateLab from '@site/src/components/viz/CrawlerRateLab';

**In one line.** A crawler repeatedly discovers and fetches URLs, but it must coordinate host access and index construction so scale does not become abuse or duplication.

## The idea in plain words

A web index begins with a crawler. It takes seed URLs, fetches allowed pages, extracts links, normalises their URLs and adds unseen candidates to a **frontier**. The frontier is more than a simple first-in queue: it must decide which page to fetch next, which host may receive a request now, and when an old page deserves another visit. A fetch can discover many links, so the frontier can grow far faster than a single worker can drain it.

The lecture's central scaling rule is to add **independent hosts**, not to hammer one host faster. At one request per second to each of 500 active hosts, idealised throughput is about **500 pages per second**. That arithmetic assumes each host has an eligible page and the network, parser and index can keep up. It does not authorise one request per second to every host: the operator's robots rules, server health and stated access policy still govern.

The fetched corpus then needs an index. One distributed design assigns subsets of documents to nodes and asks several nodes for each query before merging their top results. Another assigns terms to nodes and distributes a query's term lookups. The first is common because each node can score its local documents, but it has fan-out and merge costs. The second can concentrate hot terms and require cross-node work for multi-term queries. The choice changes query serving as well as ingestion.

<Infographic src="/img/ir/web-crawling.svg" alt="Seed URLs feed a host-aware frontier, polite fetchers and an index; 500 hosts at one request per second yield about 500 pages per second before bottlenecks." caption="The crawler's loop discovers URLs, while host queues constrain when each destination can be contacted." />

:::note Beyond the lecture

The scheduling example below is a local simulation: it makes no network requests. The discussion of standard robots semantics and Google's current crawler is added from the linked primary documentation.

:::

Change the number of active hosts and the minimum per-host delay. The lecture default shows **500 hosts × 1 page per host per second = 500 pages per second**. The data view separates the per-host rate from the aggregate idealisation.

<CrawlerRateLab />

## How it works

### Architecture & politeness

From seeds → fetch → extract links → URL frontier. Obey robots.txt; politeness limits per-host rate; freshness policies re-crawl; dedupe URLs and content.

:::tip

**Worked.** 1 req/host/s × 500 parallel hosts ≈ 500 pages/s; scale by adding hosts, not per-host rate.

:::

### Partitioning

- **Document partitioning**; Each node indexes a subset of docs; broadcast query, merge. Common.
- **Term partitioning**; Each node holds some terms' full postings.


## A real system that works this way

**Googlebot** is a real production crawler described in Google's [Search Central pipeline guide](https://developers.google.com/search/docs/fundamentals/how-search-works). Google says it discovers URLs from links and sitemaps, chooses which sites to crawl and how often, and tries to avoid crawling so fast that it overloads a site. Responses such as server errors can cause it to slow down. Its crawling is followed by indexing and serving; discovering a URL does not guarantee any of those later stages.

The [Robots Exclusion Protocol, RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html), defines a way for a service to publish access rules for automated clients. A crawler identifies its user agent, retrieves the site's robots file, matches the applicable allow and disallow rules, and caches the result within the standard's rules. Robots directives control crawler requests; they are not access control for confidential content. The server must still enforce authentication and authorisation for private resources.

Google also explains [canonical URL signals](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls). A crawler may fetch several URLs that contain the same page, while the indexer may choose one representative. Robots rules, canonicalisation and `noindex` have different jobs. A page blocked from being fetched cannot be read for its content; a canonical hint concerns which copy should represent a duplicate group; a `noindex` instruction concerns whether a readable page may appear in results. Treating them as interchangeable produces surprises.

## Code you can run

The first block reproduces the lecture's throughput arithmetic and shows what happens if the same hosts are given a three-second interval. It is a capacity bound, not a measured crawl rate.

```python
hosts = 500
requests_per_host_per_second = 1
ideal_pages_per_second = hosts * requests_per_host_per_second
slower_host_interval = 3
slower_aggregate_rate = hosts / slower_host_interval
print(f"default ideal rate={ideal_pages_per_second} pages/s")
print(f"at {slower_host_interval}s per host={slower_aggregate_rate:.1f} pages/s")
assert ideal_pages_per_second == 500
```

This second block simulates a small host-aware frontier without contacting the web. Three hosts each have two URLs. A priority queue stores the earliest eligible time for each host. The event list shows that every host is visited at time 0 and again at time 1, but never twice in the same second.

```python
from heapq import heappop, heappush

frontier = []
for host in ("a.example", "b.example", "c.example"):
    heappush(frontier, (0, host, 2))
events = []
while frontier:
    ready_at, host, remaining = heappop(frontier)
    events.append((ready_at, host))
    if remaining > 1:
        heappush(frontier, (ready_at + 1, host, remaining - 1))
print(events)
assert len(events) == 6
for host in ("a.example", "b.example", "c.example"):
    assert [time for time, name in events if name == host] == [0, 1]
```

A real frontier must also handle response status, redirects, DNS, robots updates, retries, changing pages, duplicate URLs and a much larger scheduling state. The small heap shows only the timing invariant. It intentionally does not fetch `example` hosts or claim that a site has approved any crawl.

## Designing with it

### Maintain a host-aware frontier

Normalise a discovered URL before deduplication so trivial spelling differences do not create repeated fetches. Preserve enough information to avoid merging distinct resources: case sensitivity in a path, query parameters that select content and internationalised domains can matter. A seen-URL store prevents repeated scheduling, while a content fingerprint helps detect different URLs serving the same text. These are separate deduplication tasks.

Keep a per-host or per-service queue with an earliest eligible time. A global priority queue can then choose among hosts that are ready. High-priority pages should not bypass a host's delay. A queue can balance freshness, expected value and fairness: revisiting frequently updated important pages is useful, but a new site must still get discovery capacity. Do not let one host with millions of generated URLs occupy the entire frontier.

### Apply rules before requests

Fetch and interpret the applicable robots rules before requesting a page. Match rules against the crawler's user agent and the URL path as the current protocol specifies. Cache the file for an appropriate period and handle its errors and redirects according to the standard. Operationally, also identify the crawler and offer a contact path; site operators need a way to report overload. Back off when a host signals distress, even when a fixed delay would otherwise permit the next request.

The lecture says "obey robots.txt" and "limit per-host rate"; those are minimum obligations, not a complete safety model. A robot can still overload a small host if it opens too many simultaneous connections or downloads large files. Byte budgets, timeouts, response sizes and retry limits matter. Respect an operator's authenticated boundaries rather than treating a robots allow rule as permission to access private content.

### Keep discovery and indexing separate

After a fetch, extract links and main content. Some HTML contains navigation, scripts and repeated footer text that should not dominate the index. A PDF or a JavaScript-rendered page may need a different extractor. Store fetch time, response status, content hash, canonical hints and provenance. A document should enter the searchable index only after extraction and quality checks succeed; a URL's presence in the frontier is not enough.

Recrawling policy trades freshness against load. A news homepage may change frequently; an old reference paper may rarely change. Prioritise with observed change rates, link importance and user value, then check the actual update age of surfaced results. Conditional requests can reduce transfer when supported, but they still consume a host request and must follow the same access policy.

### Choose an index partition deliberately

| Partition | Data on each node | Query shape | Main risk |
| --- | --- | --- | --- |
| By document | A subset of documents and their local inverted index | Fan out query, score locally, merge top results | Tail latency and uneven shard sizes |
| By term | Full postings for a subset of terms | Contact term owners, combine postings and scores | Hot terms and network-heavy joins |

Document partitioning makes ingestion and rebalancing relatively direct: route each document to a shard and build its local index. A serving tier sends the query to the relevant shards and merges results. If one shard is slow, the whole query may wait; replicas, timeouts and partial-result policy matter. Term partitioning can make one term's postings easy to find but may require moving large candidate sets for an AND query or scoring across several owners. The right partition depends on query distribution, update pattern and hardware, not on a single universal rule.

### Measure the pipeline, not only its output

Track discovered URLs, allowed fetches, successful responses, extraction failures, duplicate rates, indexed documents and recrawl age separately. Track these by host, content type and language. A sudden drop in search results could be caused by robots changes, extractor failure or shard lag. If only one metric exists, all three look like "search quality got worse". The source trace from Session 9 helps localise the failure.

## Follow one host through the frontier

Suppose three hosts each have hundreds of eligible pages. A naive global queue could pop several URLs from the same host in succession and overload it, even though many other hosts are idle. A host-aware frontier keeps each host's next allowed time. After fetching one page from host A at time 0, it makes A ineligible until at least time 1 in the lecture's example. Hosts B and C can be fetched during that interval. This is the mechanism behind scaling across hosts rather than raising one site's request rate.

The simple scheduler's timestamp is the earliest allowed time, not the guaranteed completion time. DNS lookup, network latency, page size and parsing can delay actual progress. If a host returns a server error or asks the crawler to slow down, the queue should move its eligible time further out. If robots rules disallow a path, the URL should not be fetched at all. A priority score for a popular page cannot override either constraint.

### Avoid the infinite URL trap

Some sites generate unbounded URL spaces through calendars, search result pages, faceted filters or session identifiers. A crawler that follows every newly seen link can spend its entire budget on near-duplicates. Normalise known tracking parameters, limit path patterns with low value, detect repeated content and enforce per-host budgets. These are heuristic controls; validate them against a sample so legitimate deep pages are not cut off. A new product catalogue may have many similar URLs because each product is genuinely distinct.

Robots rules can help a site operator exclude trap paths, but the crawler should still protect itself and the host. A site may have no robots file, stale rules or a dynamic URL space that changes faster than the operator can describe it. Treat crawling as a cooperative distributed system: client behaviour must remain bounded even when server guidance is incomplete.

### Distinguish URL and content identity

Two URLs can be byte-identical, near-duplicate, or intentionally localised versions of one article. The URL store answers whether this exact normalised address has been scheduled. A content fingerprint answers whether two fetched bodies look alike. A canonical decision answers which version should represent a duplicate group in the index. These layers should not be collapsed into one `seen` Boolean. Otherwise, a new URL can be skipped without knowing whether its content changed, or two distinct language versions can be merged because their templates match.

For each indexed document, retain the mapping back to the fetched URL and snapshot. This helps correct removals, inspect stale results and explain why a query returned a page. If a source changes, refresh or retire the old document. A search system that knows only the extracted text cannot tell a user whether the cited page still exists or whether the correct language was preserved.

### Calculate ideal throughput honestly

At 500 hosts with one eligible request per host per second, the arithmetic gives 500 requests per second. If the average response takes two seconds and only 200 fetch workers are available, the worker pool may cap throughput near 100 requests per second even though host politeness permits more. If extraction or indexing is slower than fetching, the queue between stages grows. Storage and network limits may lower it further. Capacity planning should measure each stage and backpressure the previous one when downstream work cannot keep up.

The rate lab changes a minimum delay, so a three-second interval across the same 500 hosts gives roughly 167 ideal pages per second. This is still an upper bound. The bar is not a prediction of a real web crawl. Its purpose is to make the lecture's scaling direction visible: multiplying independent hosts can increase aggregate work while keeping each host's request schedule bounded.

### Read distributed-index trade-offs with a query

Consider the query `repair bicycle tyre`. Under document partitioning, each shard has complete term postings for only its own documents. Every shard scores its local candidates and sends a small top list to a coordinator, which merges those lists. If one shard contains unusually many popular pages or is slow, its work can dominate response time. Replication can improve availability and serving capacity, but updates must reach replicas coherently.

Under term partitioning, the postings for `repair`, `bicycle` and `tyre` may live on three nodes. A Boolean intersection or ranking calculation must combine information across them. A very common term can have a huge postings list and become a hot node. Term partitioning may suit specialised workloads, but the network cost is tied to the query's term pattern. This is why the lecture calls document partitioning common. Benchmark both under expected query and update distributions before deciding.

### Carry the crawl result into evaluation

Session 7's ranking metrics assume the relevant documents are in the collection. A crawler changes that denominator. If a relevant page is missing because it was never discovered or fetched, a perfect ranker cannot return it. Evaluate **coverage** and **freshness** alongside P@k or NDCG. In a controlled site-search environment, enumerate expected documents and test ingestion completeness. On the open web, use samples and slices, acknowledge incomplete ground truth, and review important missing pages. Crawler quality and ranking quality need separate reports.

## Where this stands in 2026

:::info Industry view

- RFC 9309 gives a current standard for robots rules; a responsible crawler also monitors load and responds to host distress.
- Google publicly describes URL discovery, adaptive crawling and later indexing as distinct stages, while warning that discovery does not guarantee inclusion.
- Distributed indexing still requires a deliberate partition, replication and merge policy; apparent crawl throughput is only one part of end-to-end freshness.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> How does a web crawler work?</summary>

From seed URLs, it fetches pages, extracts links, and adds new URLs to a frontier queue; a large breadth-first traversal.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is crawler politeness and how is it enforced?</summary>

Not overloading servers by limiting the request rate per host (a per-host delay) and obeying robots.txt.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> How do crawlers scale while staying polite?</summary>

By crawling many hosts in parallel (e.g. 500 hosts × 1 req/s ≈ 500 pages/s), not by raising the per-host rate.<br /><em>Session 10 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Contrast document and term partitioning of a distributed index.</summary>

Document partitioning: each node indexes a subset of docs (broadcast query, merge; common). Term partitioning: each node holds some terms' full postings.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why must a crawler detect duplicates?</summary>

The web has heavy duplication and near-duplication; detecting it avoids wasting fetch/index resources and skewing results.<br /><em>Session 10 · conceptual</em>

</details>

## Go deeper

- [Stanford IR book: web crawling and indexes](https://nlp.stanford.edu/IR-book/html/htmledition/web-crawling-and-indexes-1.html); crawler frontier and partition background.
- [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html); current access-rule standard.
- [Google Search Central: how Search works](https://developers.google.com/search/docs/fundamentals/how-search-works); an operating crawler's public process.
- Built from the course lecture "ir-s10-web-crawling" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can describe the seed, frontier, fetch, extract and index loop.
- [ ] I can calculate the ideal 500-pages-per-second example and explain why it is an upper bound.
- [ ] I can explain why robots rules, host delays and authorisation have different roles.
- [ ] I can compare document and term partitioning for a multi-term query.
