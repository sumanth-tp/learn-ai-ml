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

:::tip Before you start

**You should already know**

- What a postings list is and why an index is built from fetched text ([Session 2](/docs/theory/ir/boolean-retrieval) and [Session 4](/docs/theory/ir/index-construction-and-compression)).
- Why a web engine cannot assume it holds every page ([Session 9](/docs/theory/ir/web-search-at-scale)).

**Reading time:** about 50 minutes, plus about 15 seconds to run the code.

**After this chapter you can**

- Work out how fast a polite crawler can go, and which limit binds first.
- Explain why a crawl queue has to be organised by host and ranked by priority.
- Compare ways of splitting an index across machines, and measure how unevenly each loads them.

:::

## In 30 seconds

A crawler is a very fast reader who must not be rude. It follows links from page to page, but each website is like a shop with one door: if a thousand people squeeze through at once, the shop closes. So the crawler keeps a separate polite waiting line for every site and rotates between the lines.

Once pages are fetched, the index is too big for one machine and is split. One way is to give each machine a share of the documents. Another is to give each machine a share of the words. Both need a rule for who owns what, and a good rule moves little when a machine joins.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Frontier | The queue of URLs waiting to be fetched | 8,000 discovered links, none fetched yet |
| Politeness delay | The minimum gap between two requests to one host | 1 second per host |
| robots.txt | A file where a site lists what crawlers may fetch | `Disallow: /private/` |
| Crawl-delay | A non-standard robots line asking for a longer gap | `Crawl-delay: 3` |
| Document partitioning | Each machine indexes a subset of documents | Machine 3 holds documents 30,000 to 39,999 |
| Term partitioning | Each machine holds the full postings of some words | Machine 1 holds every posting for `bicycle` |
| Consistent hashing | Mapping keys and machines onto a ring so that few keys move when machines change | Add a machine, move about 1 key in 11 |
| Virtual node | One machine placed at many points on the ring to even out load | 100 points per machine |

## The idea in plain words

A web index begins with a crawler. It takes seed URLs, fetches allowed pages, extracts links, normalises their URLs and adds unseen candidates to a **frontier**. The frontier is more than a simple first-in queue: it must decide which page to fetch next, which host may receive a request now, and when an old page deserves another visit. A fetch can discover many links, so the frontier can grow far faster than a single worker can drain it.

The central scaling rule is to add **independent hosts**, not to hammer one host faster. At one request per second to each of 500 active hosts, idealised throughput is about **500 pages per second**. That arithmetic assumes each host has an eligible page and the network, parser and index can keep up. It does not authorise one request per second to every host: the operator's robots rules, server health and stated access policy still govern.

The fetched corpus then needs an index. One distributed design assigns subsets of documents to nodes and asks several nodes for each query before merging their top results. Another assigns terms to nodes and distributes a query's term lookups. The first is common because each node can score its local documents, but it has fan-out and merge costs. The second can concentrate hot terms and require cross-node work for multi-term queries. The choice changes query serving as well as ingestion.

<Infographic src="/img/ir/web-crawling.svg" alt="Seed URLs feed a host-aware frontier, polite fetchers and an index; 500 hosts at one request per second yield about 500 pages per second before bottlenecks." caption="The crawler's loop discovers URLs, while host queues constrain when each destination can be contacted." />

:::note Added for this site

The scheduling example below is a local simulation: it makes no network requests. The discussion of standard robots semantics and Google's current crawler is added from the linked primary documentation.

:::

Change the number of active hosts and the minimum per-host delay. The default setting shows **500 hosts × 1 page per host per second = 500 pages per second**. The data view separates the per-host rate from the aggregate idealisation.

<CrawlerRateLab />

## Worked example, step by step

**Part 1: scheduling.** Three hosts (a, b, c) each have three pages. The politeness delay is 2 seconds and the crawler has only 2 fetch workers, so at most 2 pages per second can start.

1. At time 0 hosts a and b are eligible. Fetch one page from each. Host c waits because both workers are busy.
2. At time 1 a and b must wait until time 2. Host c fetches its first page.
3. At time 2 a and b fetch again. At time 3 c fetches. At time 4 a and b fetch their third pages. At time 5 c fetches its third.
4. Nine pages took until time 5, which is 6 seconds counting time 0. That is 1.5 pages per second, and 3 hosts divided by a 2-second delay is also 1.5.

In words: with many workers and few hosts, the hosts set the speed. Adding workers would not help here.

**Part 2: consistent hashing on a ring of 100 positions.** Machines sit at positions 10, 40 and 70. A key belongs to the first machine at or after its hash, wrapping round past 99.

5. Keys with hashes 15, 45, 75 and 5 go to machines at 40, 70, 10 and 10.
6. Add a machine at 55. Only keys in the range 41 to 55 change owner. Key 45 moves to 55. One key of four moved.
7. Compare modulo hashing. With 3 machines the owners are hash mod 3: 0, 0, 0 and 2. With 4 machines they are 3, 1, 3 and 1. All four keys moved.

In words: on a ring, a new machine takes over only the slice just before it. With modulo, almost everything is reassigned.

<Infographic src="/img/ir-enrich/ir2-crawl-and-ring.svg" alt="Left, a table of crawl throughput by number of workers and politeness delay. Right, bars of load imbalance and keys moved for one point, ten, a hundred and a thousand points per machine, and for modulo hashing." caption="Look first at the left: ten times more workers barely changes the crawl. On the right, a single ring point per machine leaves one machine with 2.79 times the average load." />

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

The first block reproduces the throughput arithmetic and shows what happens if the same hosts are given a three-second interval. It is a capacity bound, not a measured crawl rate.

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

### The worked example in code

This block replays both parts of the worked example with only the standard library.

```python
from bisect import bisect_left

hosts = {"a": 3, "b": 3, "c": 3}
delay, workers, ready_at, tick, order = 2, 2, {h: 0 for h in hosts}, 0, []
while any(hosts.values()):
    eligible = [h for h in hosts if hosts[h] and ready_at[h] <= tick][:workers]
    for h in eligible:
        hosts[h] -= 1
        ready_at[h] = tick + delay
        order.append((tick, h))
    tick += 1
print(order)

def owner(machines, key):
    points = sorted(machines)
    return points[bisect_left(points, key) % len(points)]

keys = [15, 45, 75, 5]
print([owner([10, 40, 70], k) for k in keys], [owner([10, 40, 55, 70], k) for k in keys])
print([k % 3 for k in keys], [k % 4 for k in keys])
```

**Reading the output.** The schedule prints a and b at times 0, 2 and 4, and c at times 1, 3 and 5. The ring owners are `[40, 70, 10, 10]` before and `[40, 55, 10, 10]` after. The modulo owners are `[0, 0, 0, 2]` and `[3, 1, 3, 1]`.

### An experiment: workers, politeness and queue order

What really limits a crawl: the number of workers, the delay, or the order of the queue? The block builds a synthetic web of 12,000 pages on 100 hosts. Host sizes follow a power law, so the biggest host has 2,885 pages. Each page links to six others, 85% within its own host. A third of the hosts disallow `/private/` in a real `robots.txt` that `urllib.robotparser` reads, and every fifth host asks for a crawl delay of 3 seconds. The crawler runs 600 simulated seconds from 20 random seeds. "Important pages" are the 120 pages with the most in-links. The frontier is either first in, first out, or ranked by how many in-links the crawler has seen so far. Nothing touches the network. Versions used: Python 3.14.6, NetworkX 3.6.1, NumPy 2.5.3. The run takes about 3 seconds.

```python
import heapq
from urllib.robotparser import RobotFileParser

import networkx as nx
import numpy as np

rng = np.random.default_rng(3)
hosts, pages = 100, 12_000
share = 1 / np.arange(1, hosts + 1) ** 1.1
host_of = rng.choice(hosts, pages, p=share / share.sum())
members = [np.flatnonzero(host_of == h) for h in range(hosts)]
appeal = rng.pareto(1.5, pages) + 1
web = nx.DiGraph()
web.add_nodes_from(range(pages))
for page in range(pages):
    for _ in range(6):
        local = members[host_of[page]]
        pool = local if rng.random() < 0.85 else np.arange(pages)
        web.add_edge(page, int(rng.choice(pool, p=appeal[pool] / appeal[pool].sum())))
private = rng.random(pages) < 0.12
rules = {}
for h in range(hosts):
    text = ["User-agent: *"]
    if h % 3 == 0:
        text.append("Disallow: /private/")
    if h % 5 == 0:
        text.append("Crawl-delay: 3")
    parser = RobotFileParser()
    parser.parse(text)
    rules[h] = parser
important = set(np.argsort(-np.array([web.in_degree(p) for p in range(pages)]))[: pages // 100].tolist())

def crawl(mode, workers, delay, horizon=600):
    done, blocked, queued = set(), set(), set()
    heaps = {h: [] for h in range(hosts)}
    score = {}
    ready_at = np.zeros(hosts)
    wait = {h: max(delay, rules[h].crawl_delay("study-bot") or 0) for h in range(hosts)}
    order = 0

    def push(page):
        nonlocal order
        order += 1
        score[page] = score.get(page, 0) + 1
        key = -score[page] if mode == "priority" else order
        heapq.heappush(heaps[host_of[page]], (key, page))

    for page in rng.choice(pages, 20, replace=False):
        push(int(page))
    marks = {}
    for t in range(1, horizon + 1):
        ready = sorted((h for h in range(hosts) if heaps[h] and ready_at[h] <= t), key=lambda h: heaps[h][0][0])
        for h in ready[:workers]:
            while heaps[h]:
                _, page = heapq.heappop(heaps[h])
                if page in done or page in blocked:
                    continue
                path = f"/private/{page}" if private[page] else f"/p/{page}"
                if not rules[h].can_fetch("study-bot", path):
                    blocked.add(page)
                    continue
                done.add(page)
                ready_at[h] = t + wait[h]
                for target in web.successors(page):
                    push(target)
                break
        if t in (100, 300, 600):
            marks[t] = (len(done), len(done & important))
    return marks, len(done & set(members[0].tolist())), len(blocked)

print(f"pages {pages}, hosts {hosts}, biggest host {len(members[0])} pages, important pages {len(important)}")
print(f"{'frontier':<10}{'workers':>8}{'delay':>6}{'fetched t=100':>15}{'t=300':>7}{'t=600':>7}{'important t=100':>17}{'t=600':>7}{'biggest host':>14}{'blocked':>9}")
for mode, workers, delay in (("fifo", 50, 1), ("priority", 50, 1), ("priority", 500, 1), ("priority", 50, 5)):
    marks, big, blocked = crawl(mode, workers, delay)
    print(f"{mode:<10}{workers:>8}{delay:>6}{marks[100][0]:>15}{marks[300][0]:>7}{marks[600][0]:>7}{marks[100][1]:>17}{marks[600][1]:>7}{big:>14}{blocked:>9}")
```

The output of the run:

```text
pages 12000, hosts 100, biggest host 2885 pages, important pages 120
frontier   workers delay  fetched t=100  t=300  t=600  important t=100  t=600  biggest host  blocked
fifo            50     1           4063   6167   7474               75     97           200      308
priority        50     1           4062   6159   7470              101    115           200      311
priority       500     1           4158   6159   7470              104    115           200      311
priority        50     5           1364   3691   4968               71    108           120      217
```

**Reading the output.** Each row is one crawler setting. The `fetched` columns count pages really fetched by that second. `important` counts how many of the 120 most-linked pages have been fetched. `biggest host` is how many pages of the 2,885 on the largest site were fetched in 600 seconds. `blocked` counts URLs that robots rules forbade.

**Line by line.**

- `ready_at[h] = t + wait[h]` is the whole politeness mechanism: a host cannot be chosen again until that second, whatever its priority.
- `key = -score[page] if mode == "priority" else order` makes the heap pop the most-linked page first, or the oldest one.
- `rules[h].can_fetch(...)` is checked before the fetch, and a forbidden URL is dropped without using the host's slot.

### An experiment: who owns what

The second block measures the index partitioning choices. It hashes 100,000 documents onto a ring of 10 machines with 1, 10, 100 and 1,000 points per machine, compares with plain modulo hashing, and measures how many documents move when an eleventh machine joins. It then places 5,000 words with Zipf-shaped query frequency on 10 machines and measures the busiest machine.

```python
import hashlib

import numpy as np

def point(text):
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "big")

documents = np.array([point(f"doc-{i}") for i in range(100_000)], dtype=np.uint64)

def ring_owner(nodes, virtual):
    pairs = sorted((point(f"node-{n}#{v}"), n) for n in range(nodes) for v in range(virtual))
    points = np.array([p for p, _ in pairs], dtype=np.uint64)
    owners = np.array([n for _, n in pairs])
    return owners[np.searchsorted(points, documents) % len(points)]

print(f"{'virtual points':>15}{'max/mean load':>15}{'min/mean load':>15}{'moved, 10 to 11 nodes':>23}")
for virtual in (1, 10, 100, 1000):
    before, after = ring_owner(10, virtual), ring_owner(11, virtual)
    load = np.bincount(before, minlength=10)
    print(f"{virtual:>15}{load.max() / load.mean():>15.3f}{load.min() / load.mean():>15.3f}{(before != after).mean():>23.3f}")

modulo_before, modulo_after = documents % 10, documents % 11
load = np.bincount(modulo_before.astype(int), minlength=10)
print(f"{'modulo':>15}{load.max() / load.mean():>15.3f}{load.min() / load.mean():>15.3f}{(modulo_before != modulo_after).mean():>23.3f}")
print(f"ideal share of keys that must move: {1 / 11:.3f}")

frequency = 1 / np.arange(1, 5001)
ratios = []
for salt in range(300):
    owner = np.array([point(f"{salt}-term-{r}") % 10 for r in range(5000)])
    load = np.bincount(owner, weights=frequency, minlength=10)
    ratios.append(load.max() / load.mean())
print(f"term partition, 5,000 Zipf terms on 10 nodes: max/mean query load {np.mean(ratios):.3f} on average, {np.percentile(ratios, 95):.3f} at the 95th percentile of 300 hash salts")
print(f"hottest single term carries {frequency[0] / frequency.sum():.3f} of all term lookups, against {1 / 10:.3f} for a perfectly even node")
```

```text
 virtual points  max/mean load  min/mean load  moved, 10 to 11 nodes
              1          2.789          0.029                  0.114
             10          1.626          0.740                  0.093
            100          1.102          0.897                  0.076
           1000          1.053          0.960                  0.094
         modulo          1.020          0.989                  0.910
ideal share of keys that must move: 0.091
term partition, 5,000 Zipf terms on 10 nodes: max/mean query load 2.012 on average, 2.536 at the 95th percentile of 300 hash salts
hottest single term carries 0.110 of all term lookups, against 0.100 for a perfectly even node
```

**Reading the output.** `max/mean load` is the busiest machine's share of documents divided by the average, so 1.000 is perfectly even. `moved` is the share of documents that changed owner when the eleventh machine joined; the ideal is 1/11, or 0.091.

### What the numbers say

Workers were not the bottleneck. With 50 workers and a 1-second delay the crawler had fetched 4,062 pages by second 100. With 500 workers it fetched 4,158, a gain of 96 pages (2.4%), and from second 300 onwards the two runs matched at 6,159 and 7,470. The delay mattered far more: a 5-second delay gave 1,364 pages at second 100 and 4,968 at second 600, against 7,470.

The queue order changed what the crawler found, not how many pages it fetched. By second 100 the priority frontier had fetched 101 of the 120 most-linked pages against 75 for first in, first out, with almost the same page count (4,062 against 4,063). By second 600 the gap had closed to 115 against 97.

The surprise is the biggest site. After 600 seconds the crawler had fetched only 200 of its 2,885 pages, 6.9%, because that host asks for a 3-second delay and no queue order or worker count can speed one host up. With a 5-second delay it was 120 pages. A crawl is finished host by host, and the largest host decides when.

For partitioning, one ring point per machine was poor: the busiest machine held 2.789 times the average and the emptiest 0.029. With 100 points the busiest was at 1.102, and with 1,000 at 1.053. When an eleventh machine joined, modulo hashing moved 0.910 of all documents while the ring moved 0.076 to 0.114, near the ideal 0.091. In a term-partitioned index, the busiest of 10 machines carried 2.012 times the average query load on average and 2.536 at the 95th percentile of 300 hash seeds, because the hottest term alone takes 0.110 of all lookups.

Limits: synthetic hosts, links and robots rules; a crawler that fetches one page per host per slot and ignores DNS, latency and failures; one seed per scenario; and a hash from `hashlib.md5` standing in for whatever a production system uses. The document-partitioned load is flat because every query visits every machine, which is by design and was not measured here.

## Designing with it

### Maintain a host-aware frontier

Normalise a discovered URL before deduplication so trivial spelling differences do not create repeated fetches. Preserve enough information to avoid merging distinct resources: case sensitivity in a path, query parameters that select content and internationalised domains can matter. A seen-URL store prevents repeated scheduling, while a content fingerprint helps detect different URLs serving the same text. These are separate deduplication tasks.

Keep a per-host or per-service queue with an earliest eligible time. A global priority queue can then choose among hosts that are ready. High-priority pages should not bypass a host's delay. A queue can balance freshness, expected value and fairness: revisiting frequently updated important pages is useful, but a new site must still get discovery capacity. Do not let one host with millions of generated URLs occupy the entire frontier.

### Apply rules before requests

Fetch and interpret the applicable robots rules before requesting a page. Match rules against the crawler's user agent and the URL path as the current protocol specifies. Cache the file for an appropriate period and handle its errors and redirects according to the standard. Operationally, also identify the crawler and offer a contact path; site operators need a way to report overload. Back off when a host signals distress, even when a fixed delay would otherwise permit the next request.

"Obey robots.txt" and "limit per-host rate" are minimum obligations, not a complete safety model. A robot can still overload a small host if it opens too many simultaneous connections or downloads large files. Byte budgets, timeouts, response sizes and retry limits matter. Respect an operator's authenticated boundaries rather than treating a robots allow rule as permission to access private content.

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

Suppose three hosts each have hundreds of eligible pages. A naive global queue could pop several URLs from the same host in succession and overload it, even though many other hosts are idle. A host-aware frontier keeps each host's next allowed time. After fetching one page from host A at time 0, it makes A ineligible until at least time 1 in the worked example. Hosts B and C can be fetched during that interval. This is the mechanism behind scaling across hosts rather than raising one site's request rate.

The simple scheduler's timestamp is the earliest allowed time, not the guaranteed completion time. DNS lookup, network latency, page size and parsing can delay actual progress. If a host returns a server error or asks the crawler to slow down, the queue should move its eligible time further out. If robots rules disallow a path, the URL should not be fetched at all. A priority score for a popular page cannot override either constraint.

### Avoid the infinite URL trap

Some sites generate unbounded URL spaces through calendars, search result pages, faceted filters or session identifiers. A crawler that follows every newly seen link can spend its entire budget on near-duplicates. Normalise known tracking parameters, limit path patterns with low value, detect repeated content and enforce per-host budgets. These are heuristic controls; validate them against a sample so legitimate deep pages are not cut off. A new product catalogue may have many similar URLs because each product is genuinely distinct.

Robots rules can help a site operator exclude trap paths, but the crawler should still protect itself and the host. A site may have no robots file, stale rules or a dynamic URL space that changes faster than the operator can describe it. Treat crawling as a cooperative distributed system: client behaviour must remain bounded even when server guidance is incomplete.

### Distinguish URL and content identity

Two URLs can be byte-identical, near-duplicate, or intentionally localised versions of one article. The URL store answers whether this exact normalised address has been scheduled. A content fingerprint answers whether two fetched bodies look alike. A canonical decision answers which version should represent a duplicate group in the index. These layers should not be collapsed into one `seen` Boolean. Otherwise, a new URL can be skipped without knowing whether its content changed, or two distinct language versions can be merged because their templates match.

For each indexed document, retain the mapping back to the fetched URL and snapshot. This helps correct removals, inspect stale results and explain why a query returned a page. If a source changes, refresh or retire the old document. A search system that knows only the extracted text cannot tell a user whether the cited page still exists or whether the correct language was preserved.

### Calculate ideal throughput honestly

At 500 hosts with one eligible request per host per second, the arithmetic gives 500 requests per second. If the average response takes two seconds and only 200 fetch workers are available, the worker pool may cap throughput near 100 requests per second even though host politeness permits more. If extraction or indexing is slower than fetching, the queue between stages grows. Storage and network limits may lower it further. Capacity planning should measure each stage and backpressure the previous one when downstream work cannot keep up.

The rate lab changes a minimum delay, so a three-second interval across the same 500 hosts gives roughly 167 ideal pages per second. This is still an upper bound. The bar is not a prediction of a real web crawl. Its purpose is to make the scaling direction visible: multiplying independent hosts can increase aggregate work while keeping each host's request schedule bounded.

### Read distributed-index trade-offs with a query

Consider the query `repair bicycle tyre`. Under document partitioning, each shard has complete term postings for only its own documents. Every shard scores its local candidates and sends a small top list to a coordinator, which merges those lists. If one shard contains unusually many popular pages or is slow, its work can dominate response time. Replication can improve availability and serving capacity, but updates must reach replicas coherently.

Under term partitioning, the postings for `repair`, `bicycle` and `tyre` may live on three nodes. A Boolean intersection or ranking calculation must combine information across them. A very common term can have a huge postings list and become a hot node. Term partitioning may suit specialised workloads, but the network cost is tied to the query's term pattern. This is why document partitioning is the common choice. Benchmark both under expected query and update distributions before deciding.

### Carry the crawl result into evaluation

Session 7's ranking metrics assume the relevant documents are in the collection. A crawler changes that denominator. If a relevant page is missing because it was never discovered or fetched, a perfect ranker cannot return it. Evaluate **coverage** and **freshness** alongside P@k or NDCG. In a controlled site-search environment, enumerate expected documents and test ingestion completeness. On the open web, use samples and slices, acknowledge incomplete ground truth, and review important missing pages. Crawler quality and ranking quality need separate reports.

## Where this stands in 2026

:::info Industry view

- RFC 9309 gives a current standard for robots rules; a responsible crawler also monitors load and responds to host distress.
- Google publicly describes URL discovery, adaptive crawling and later indexing as distinct stages, while warning that discovery does not guarantee inclusion.
- Distributed indexing still requires a deliberate partition, replication and merge policy; apparent crawl throughput is only one part of end-to-end freshness.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Adding fetch workers to crawl faster | More workers means more pages per second | Add independent hosts, or negotiate a shorter delay. Here 500 workers gave only 96 more pages than 50 by second 100, and none later |
| Using one global first-in, first-out queue | It is the simplest queue | Keep a queue per host with an earliest-eligible time, and rank across hosts by priority. Priority found 101 of 120 key pages by second 100, against 75 |
| Treating `Disallow` as security | The file says "disallow" | RFC 9309 states that the rules are not a form of access authorisation. Enforce authentication on the server |
| Assigning documents by `hash mod machines` | It is one line of code | Adding one machine moved 0.910 of the documents. Use a ring with many points per machine |
| Placing one point per machine on the ring | It matches the textbook picture | One point left the busiest machine at 2.79 times the average. Use 100 or more |
| Assuming `Crawl-delay` is universal | Python's parser reads it | It is not part of the standard, and Google lists it among the fields it does not support. Throttle on the client regardless |

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

<details>
<summary><strong>Q6.</strong> (Medium) A crawl has 40 hosts, a 2-second delay per host and 500 workers. What is the highest steady throughput, and what would double it?</summary>

Each host can supply one page every 2 seconds, so 40 hosts give 20 pages per second at most. The 500 workers are idle most of the time. Doubling it needs either 80 hosts with pages waiting, or a 1-second delay that the sites accept. More workers change nothing. This is the same bound as the worked example: hosts divided by delay.

</details>

<details>
<summary><strong>Q7.</strong> (Medium) With one point per machine on a ring of 10 machines, the emptiest held 0.029 of the average load. Why does adding virtual points fix that?</summary>

With one random point each, the gaps between points on the ring vary a lot, and a machine owns the gap before its point, so a machine with a tiny gap owns almost nothing. With 100 points each, a machine owns the sum of 100 small gaps, and sums of many random gaps are close to their mean. The measured busiest-to-average ratio fell from 2.789 to 1.102. The cost is a larger ring to store and search.

</details>

<details>
<summary><strong>Q8.</strong> (Stretch) In the run, the priority frontier fetched the same number of pages as first in, first out but found more of the key pages. Why, and when would this not help?</summary>

Both obey the same host delays, so they fetch the same number of pages per second. Priority chooses which page each free host slot spends: pages already linked from many fetched pages are likely to be well linked overall. It helps when a few pages attract most links and the crawl will be cut short. It would not help in a crawl that runs to completion, as the gap closing from 101 against 75 at second 100 to 115 against 97 at second 600 suggests, or on a site whose links carry no importance signal.

</details>

## Go deeper

- [Stanford IR book: web crawling and indexes](https://nlp.stanford.edu/IR-book/html/htmledition/web-crawling-and-indexes-1.html); crawler frontier and partition background.
- [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html); current access-rule standard.
- [Google Search Central: how Search works](https://developers.google.com/search/docs/fundamentals/how-search-works); an operating crawler's public process.
- [Python documentation: urllib.robotparser](https://docs.python.org/3/library/urllib.robotparser.html) (opened 2026-10-09); `can_fetch` and `crawl_delay`, used in the experiment.
- [Google Search Central: robots.txt specification](https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt) (opened 2026-10-09); lists `crawl-delay` among fields Google does not support.
- [Dynamo: Amazon's Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) (SOSP 2007, opened 2026-10-09); explains why one random ring position per node gives uneven load and how virtual nodes fix it.
- Built from the course lecture "ir-s10-web-crawling" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can describe the seed, frontier, fetch, extract and index loop.
- [ ] I can calculate the ideal 500-pages-per-second example and explain why it is an upper bound.
- [ ] I can explain why robots rules, host delays and authorisation have different roles.
- [ ] I can compare document and term partitioning for a multi-term query.
- [ ] I can work out a polite crawl's throughput from hosts and delay, and say why more workers did not help in the experiment.
- [ ] I can explain why the largest host decides when a crawl finishes.
- [ ] I can place keys on a hash ring by hand, add a machine, and say how many keys move compared with modulo hashing.
- [ ] I can say how many ring points per machine I would use and quote the measured imbalance.

## Where to go next

Next: [Session 11, link analysis](/docs/theory/ir/link-analysis-pagerank-and-hits), which uses the crawled link graph to rank pages. Related: [Session 4, index construction](/docs/theory/ir/index-construction-and-compression), where the partitioned index is built.
