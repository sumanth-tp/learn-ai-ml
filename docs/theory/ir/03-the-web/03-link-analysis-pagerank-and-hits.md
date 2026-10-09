---
id: ir-link-analysis-pagerank-hits
title: "Information Retrieval · Session 11 — Link Analysis, PageRank and HITS"
sidebar_label: "11 · Link analysis"
sidebar_position: 3
slug: /theory/ir/link-analysis-pagerank-and-hits
description: "How PageRank's random surfer and HITS hub-authority updates use a hyperlink graph as ranking evidence."
tags: [information-retrieval, pagerank, hits, link-analysis]
---

import Infographic from '@site/src/components/Infographic';
import PageRankLab from '@site/src/components/viz/PageRankLab';

**In one line.** Hyperlinks provide a graph signal that can help identify authoritative pages, provided the system accounts for graph structure and manipulation.

:::tip Before you start

**You should already know**

- What a hyperlink graph is and how a crawler finds it ([Session 10](/docs/theory/ir/web-crawling-and-distributed-indexes)).
- That a ranking can be judged by where the right page lands ([Session 7](/docs/theory/ir/evaluating-ranked-retrieval)).

**Reading time:** about 55 minutes, plus about 30 seconds to run the code (the first run downloads 9.5 MB).

**After this chapter you can**

- Compute PageRank updates by hand and say why one update is not the answer.
- Run PageRank and HITS on a real link graph and read how damping changes the result.
- Show how a link farm lifts a page, and say which defence helps and by how much.

:::

## In 30 seconds

Imagine a person clicking links at random for hours. Pages that many other pages point to are visited often, and pages pointed to by those busy pages are visited even more. Ranking pages by how often the surfer lands on them is PageRank. Now and then the surfer gets bored and jumps to a random page, which keeps the walk from getting stuck.

HITS asks a slightly different question. Some pages are good lists of links, and some are the things worth linking to. A good list points at good things, and a good thing is pointed at by good lists.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| PageRank | The long-run share of time a random surfer spends on a page | 0.617 after one update in the worked graph |
| Damping factor $d$ | The probability that the surfer follows a link instead of jumping | $d = 0.85$ |
| Teleport | A random jump to any page | Each page gets $(1-d)/N$ per step |
| Dangling page | A page with no outgoing links | A PDF with no links on it |
| Hub | A page that points at many good pages | A curated directory |
| Authority | A page that many good hubs point at | The official regulation |
| Link farm | Many pages built only to link to one target | 200 spam pages all pointing at one shop |
| Spearman correlation | How similarly two rankings order the same items, from -1 to 1 | 0.966 between PageRank and in-links below |

## The idea in plain words

A hyperlink can be read as a page referring readers to another page. If many important pages refer to one destination, that destination may be worth examining. This signal is different from a document's words: the target could be described by other authors even if it does not repeat the query phrase. Two classical ways to use that graph are **PageRank** and **HITS**.

PageRank imagines a random surfer who usually follows a link and sometimes jumps to a random page. A page receives a share of each linking page's current rank, divided by that linking page's number of outgoing links. The jump or **teleport** term prevents the walk from being trapped in a disconnected region under the usual complete model. With three pages, follow-link probability $d=0.85$, and two incoming pages each at rank $1/3$ with one outgoing link, the single update is $(1-0.85)/3+0.85(1/3+1/3)\approx 0.617$. One update is not a final stationary score.

HITS treats **hubs** and **authorities** as two roles. A good hub links to good authorities; a good authority receives links from good hubs. Repeated updates reinforce both, usually within a query-selected subgraph. A directory page can be a useful hub without being the definitive answer itself. Neither score proves factual correctness, and both can be manipulated by artificial links.

<Infographic src="/img/ir/link-analysis.svg" alt="Pages A and B each contribute one-third of their current PageRank to P, yielding a first update of 0.617 with damping 0.85 and three pages." caption="The PageRank arithmetic is one synchronous update on a fully specified three-page graph." />

<Infographic src="/img/ir/hub-authority.svg" alt="Two initial hubs point to two authorities; first HITS updates give authority scores two and one, then hub scores three and two before normalisation." caption="HITS keeps link-giving hubs and link-receiving authorities as different scores." />

:::note Added for this site

The explicit three-page graph, runnable HITS numbers, the real-graph experiment and the discussion of spam, edge meaning and modern ranking are additions. The original search paper below is historical context, not a description of a current private ranking formula.

:::

Run PageRank one step at a time on the board's graph: A→P, B→P and P→A. At the default settings, P moves from **0.333** to **0.617** after the first step. Adjust damping to see the balance between links and teleportation; the table shows each incoming contribution.

<PageRankLab />

## Worked example, step by step

The graph is the one on the board: A links to P, B links to P, and P links to A. There are 3 pages, $d = 0.85$, and every page starts at $1/3$. Each step gives every page $(1 - 0.85)/3 = 0.05$ from teleporting, plus $0.85$ times the rank it receives from links.

1. **Step 1, page P.** It receives all of A's rank and all of B's: $0.05 + 0.85 \times (1/3 + 1/3) = 0.05 + 0.5667 = 0.6167$.
2. **Step 1, page A.** It receives all of P's rank: $0.05 + 0.85 \times 1/3 = 0.3333$.
3. **Step 1, page B.** Nobody links to B: $0.05$. The three values add to $1.0000$.
4. **Step 2, page P.** It now uses the step-1 ranks of A and B, not their starting values: $0.05 + 0.85 \times (0.3333 + 0.05) = 0.3758$.
5. **Step 2, page A.** $0.05 + 0.85 \times 0.6167 = 0.5742$. Page B stays at $0.05$.
6. Page P fell from 0.617 to 0.376 in one step and A rose from 0.333 to 0.574. The ranks bounce and settle, so the first update is a step, not the answer.

In words: each page keeps a small floor from teleporting and passes most of its rank along its links. Repeating this until the numbers stop moving gives the stationary ranks.

<Infographic src="/img/ir-enrich/ir2-pagerank-spam.svg" alt="Left, iterations to converge for five damping values on 4,592 Wikipedia articles. Right, the rank of one target article as a link farm grows, with and without a trusted-teleport defence." caption="Look first at the right: ten spam pages move the target from 2,297th to 442nd, and 200 make it first. The trusted-teleport line stays far lower." />

## How it works

### The random surfer

PR(p) = (1−d)/N + d·Σ PR(q)/L(q): a surfer follows links with prob d and teleports with 1−d. Iterate to the stationary distribution.

:::tip

**Worked.** N=3, d=0.85, two in-links of PR 1/3 (out-degree 1) → PR = 0.05 + 0.85·0.667 = 0.617.

:::

### Hubs & authorities

HITS gives each page an authority score (pointed to by good hubs) and a hub score (points to good authorities), reinforcing over iterations; query-dependent, unlike PageRank.


## A real system that works this way

The 1998 paper [*The Anatomy of a Large-Scale Hypertextual Web Search Engine*](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/) describes the early Google search prototype and its use of hypertext structure in a large crawl and index. It is a concrete historical system in which links supported discovery and search ranking. It does **not** establish how today's Google Search scores any given page; the current product uses a changing combination of signals whose full formula is not public.

The [Stanford IR book's link-analysis chapter](https://nlp.stanford.edu/IR-book/html/htmledition/link-analysis-1.html) explains why links can be useful and why raw in-link count is fragile. Many low-quality pages can be made to point to a target. PageRank weights an incoming edge by the source page's rank and distributes that rank across its outgoing edges. HITS distinguishes pages that collect references from pages that point to several useful resources. Both are graph computations, not a direct substitute for term matching, relevance judgements or source-quality review.

For an internal knowledge base, links between policy pages can be a useful navigation and discovery signal. A link to a policy may indicate that teams use it, but a recently created policy could be essential despite having few links. A page that became a documentation hub may link to many useful answers yet contain little answer text. Combine graph evidence with query relevance and freshness rather than turning link rank into a hard filter.

## Code you can run

The first block runs the one-update calculation on a complete three-page graph. The graph has no dangling node: A and B link to P; P links to A. All three start at $1/3$. Repeated synchronous updates keep the total rank at one.

```python
pages = ["A", "B", "P"]
outlinks = {"A": ["P"], "B": ["P"], "P": ["A"]}
damping = 0.85
ranks = {page: 1 / len(pages) for page in pages}

def update(ranks):
    next_ranks = {page: (1 - damping) / len(pages) for page in pages}
    for source, targets in outlinks.items():
        for target in targets:
            next_ranks[target] += damping * ranks[source] / len(targets)
    return next_ranks

ranks = update(ranks)
print("first P update:", f"{ranks['P']:.3f}")
assert round(ranks["P"], 3) == 0.617
for _ in range(20):
    ranks = update(ranks)
print("later ranks:", {page: round(ranks[page], 3) for page in pages})
assert abs(sum(ranks.values()) - 1) < 1e-12
```

The second block reproduces the HITS board's **unnormalised first round**. Two hubs begin with score one. H1 points to A1 and A2; H2 points only to A1. First calculate authorities from incoming hubs, then hubs from those new authority scores. Production HITS iterations normalise scores to prevent uncontrolled growth.

```python
links = {"H1": ["A1", "A2"], "H2": ["A1"]}
hub = {"H1": 1, "H2": 1}
authority = {"A1": 0, "A2": 0}
for source, targets in links.items():
    for target in targets:
        authority[target] += hub[source]
new_hub = {source: sum(authority[target] for target in targets)
           for source, targets in links.items()}
print("authority:", authority, "hub:", new_hub)
assert authority == {"A1": 2, "A2": 1}
assert new_hub == {"H1": 3, "H2": 2}
```

These calculations show graph mechanics only. They do not say that A1 is twice as relevant as A2 to every query, or that H1's score is directly comparable with a PageRank probability. Each method has a different normalisation and interpretation.

### The worked example in code

This block repeats steps 1 to 6 and prints the three ranks after each of two steps.

```python
pages = ["A", "B", "P"]
outlinks = {"A": ["P"], "B": ["P"], "P": ["A"]}
ranks = {page: 1 / 3 for page in pages}
for step in (1, 2):
    nxt = {page: 0.15 / 3 for page in pages}
    for source, targets in outlinks.items():
        for target in targets:
            nxt[target] += 0.85 * ranks[source] / len(targets)
    ranks = nxt
    print(step, {page: round(value, 4) for page, value in ranks.items()}, round(sum(ranks.values()), 4))
```

**Reading the output.** Step 1 prints P 0.6167, A 0.3333, B 0.05, total 1.0. Step 2 prints P 0.3758, A 0.5742, B 0.05.

### An experiment: PageRank on a real link graph

Does the damping factor matter, and is PageRank anything more than counting links? The block loads the Wikispeedia graph: 4,604 articles from a school edition of Wikipedia with 119,882 hyperlinks (West and Leskovec, 2012). 4,592 articles appear in the link list and 5 of them have no outgoing link. The code downloads the 9.5 MB archive once. The dataset page gives no licence, asks users to cite the two papers named in the file header, and the code does not redistribute the data. The block writes its own power iteration with SciPy, checks it against `networkx.pagerank`, and sweeps $d$. Versions used: Python 3.14.6, NetworkX 3.6.1, SciPy 1.18.1, NumPy 2.5.3.

```python
import os
import tarfile
import tempfile
import urllib.parse
import urllib.request

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
CACHE = os.path.join(tempfile.gettempdir(), "wikispeedia.tar.gz")
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with tarfile.open(CACHE) as archive:
    rows = archive.extractfile("wikispeedia_paths-and-graph/links.tsv").read().decode().splitlines()
edges = [tuple(urllib.parse.unquote(part) for part in row.split("\t")) for row in rows if row and not row.startswith("#")]
graph = nx.DiGraph(edges)
nodes = list(graph)
print(f"{graph.number_of_nodes()} articles, {graph.number_of_edges()} links, {sum(1 for n in nodes if graph.out_degree(n) == 0)} dangling")

matrix = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float).T.tocsr()
out_degree = np.asarray(graph.out_degree(nodes))[:, 1].astype(float)
dangling = out_degree == 0
inverse = np.divide(1.0, out_degree, out=np.zeros_like(out_degree), where=~dangling)

def power_iteration(damping, tolerance=1e-10):
    n = len(nodes)
    rank = np.full(n, 1 / n)
    for step in range(1, 1000):
        spread = matrix @ (rank * inverse) + rank[dangling].sum() / n
        updated = damping * spread + (1 - damping) / n
        change = np.abs(updated - rank).sum()
        rank = updated
        if change < tolerance:
            return rank, step

baseline, _ = power_iteration(0.85)
reference = nx.pagerank(graph, alpha=0.85, tol=1e-12)
print("largest gap to networkx.pagerank:", f"{max(abs(baseline[i] - reference[n]) for i, n in enumerate(nodes)):.2e}")
in_degree = np.array([graph.in_degree(n) for n in nodes])
top_baseline = set(np.argsort(-baseline)[:20])
print(f"{'damping':>8}{'iterations':>12}{'Spearman vs 0.85':>18}{'top-20 shared':>15}{'top-1 article':>18}")
for damping in (0.5, 0.7, 0.85, 0.95, 0.99):
    rank, steps = power_iteration(damping)
    rho = spearmanr(rank, baseline)[0]
    print(f"{damping:>8}{steps:>12}{rho:>18.3f}{len(top_baseline & set(np.argsort(-rank)[:20])):>15}{nodes[int(np.argmax(rank))]:>18}")
print("Spearman of PageRank (0.85) with in-degree:", f"{spearmanr(baseline, in_degree)[0]:.3f}")
print("top 5 by PageRank:", [nodes[i] for i in np.argsort(-baseline)[:5]])
print("top 5 by in-degree:", [nodes[i] for i in np.argsort(-in_degree)[:5]])
```

The output of the run:

```text
4592 articles, 119882 links, 5 dangling
largest gap to networkx.pagerank: 5.32e-11
 damping  iterations  Spearman vs 0.85  top-20 shared     top-1 article
     0.5          21             0.986             17     United_States
     0.7          32             0.997             17     United_States
    0.85          46             1.000             20     United_States
    0.95          62             0.997             18     United_States
    0.99          71             0.993             18     United_States
Spearman of PageRank (0.85) with in-degree: 0.966
top 5 by PageRank: ['United_States', 'France', 'Europe', 'United_Kingdom', 'English_language']
top 5 by in-degree: ['United_States', 'United_Kingdom', 'France', 'Europe', 'England']
```

**Reading the output.** `iterations` is how many updates until the total change fell below $10^{-10}$. `Spearman vs 0.85` compares each ranking with the $d = 0.85$ ranking; 1.000 means the same order. `top-20 shared` counts articles common to the two top-20 lists.

**Line by line.**

- `rank[dangling].sum() / n` spreads the rank of pages with no out-links evenly over all pages, so the total stays at 1. Without it rank would leak away.
- `matrix @ (rank * inverse)` gives every page the sum of rank over out-degree from the pages that link to it, in one sparse product.
- The gap of 5.32e-11 to `networkx.pagerank` is the check that the hand-written iteration is the same algorithm.

### An experiment: a link farm and one defence

How much does a spam farm move a page, and does a trusted-teleport defence help? The second block picks the article that sits exactly in the middle of the PageRank order, `M25_motorway`, with 6 in-links. It adds 10, 50, 200 and 1,000 fake pages that link to it and are linked back from it. It measures the target's rank three ways: plain PageRank; PageRank where the random jump lands only on the 50 top-ranked pages (a simplified version of the seed-set idea in TrustRank, Gyöngyi, Garcia-Molina and Pedersen, 2004, not that algorithm); and HITS authority on the whole graph.

```python
import io
import os
import tarfile
import tempfile
import urllib.parse
import urllib.request

import networkx as nx
import numpy as np

URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
CACHE = os.path.join(tempfile.gettempdir(), "wikispeedia.tar.gz")
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with tarfile.open(CACHE) as archive:
    rows = archive.extractfile("wikispeedia_paths-and-graph/links.tsv").read().decode().splitlines()
edges = [tuple(urllib.parse.unquote(part) for part in row.split("\t")) for row in rows if row and not row.startswith("#")]
graph = nx.DiGraph(edges)

def rank_of(scores, node):
    return 1 + sum(1 for value in scores.values() if value > scores[node])

base = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)
ordered = sorted(base, key=lambda n: (-round(base[n], 12), n))
target = ordered[len(ordered) // 2]
print(f"target '{target}': PageRank rank {rank_of(base, target)} of {len(base)}, in-links {graph.in_degree(target)}")
trusted = ordered[:50]

def with_farm(size):
    farm = nx.DiGraph(graph)
    for i in range(size):
        farm.add_edge(f"spam-{i}", target)
        farm.add_edge(target, f"spam-{i}")
    return farm

print(f"{'farm pages':>11}{'rank, plain':>13}{'rank, trusted teleport':>24}{'HITS authority rank':>21}")
for size in (0, 10, 50, 200, 1000):
    farm = with_farm(size)
    plain = nx.pagerank(farm, alpha=0.85, tol=1e-12, max_iter=1000)
    seed = {n: (1 / len(trusted) if n in trusted else 0.0) for n in farm}
    guarded = nx.pagerank(farm, alpha=0.85, personalization=seed, tol=1e-12, max_iter=1000)
    _, authority = nx.hits(farm, max_iter=1000, tol=1e-10)
    print(f"{size:>11}{rank_of(plain, target):>13}{rank_of(guarded, target):>24}{rank_of(authority, target):>21}")
```

```text
target 'M25_motorway': PageRank rank 2297 of 4592, in-links 6
 farm pages  rank, plain  rank, trusted teleport  HITS authority rank
          0         2297                    1980                 2404
         10          442                    1679                 2404
         50           20                    1296                 2402
        200            1                    1057                 2384
       1000            1                     950                 2295
```

**Reading the output.** Each number is a rank among all pages, so 1 is the top and larger is worse. The `0` row is the graph with no farm.

### What the numbers say

Damping changed the cost far more than the ranking. Iterations to converge grew from 21 at $d = 0.5$ to 32, 46, 62 and 71 at $d = 0.99$. The order barely moved: Spearman correlation with the $d = 0.85$ ranking was 0.986 at $d = 0.5$ and 0.993 at $d = 0.99$, and `United_States` was first at every value. The top-20 lists shared 17 to 20 articles.

The surprise is how close PageRank came to counting in-links. The Spearman correlation between PageRank at $d = 0.85$ and in-degree was 0.966, and four of the top five articles were the same (`United_States`, `France`, `Europe`, `United_Kingdom` against `United_States`, `United_Kingdom`, `France`, `Europe`, `England`). On an honest graph the extra machinery changes the order of a few pages, not its shape.

The farm shows why it still matters. Ten fake pages lifted the target from 2,297th to 442nd of 4,592, fifty to 20th, and two hundred to 1st. The trusted-teleport variant started the target at 1,980th and ended at 1,679th, 1,296th, 1,057th and 950th, so the farm still bought about 1,030 places at 1,000 pages. The defence blunted the attack but did not remove it. HITS run over the whole graph hardly noticed: authority rank went from 2,404 to 2,295.

Limits: one graph of encyclopaedia links, which is cleaner than the open web; a farm that links both ways and nothing else; one target; a HITS run on the whole graph rather than on a query-selected neighbourhood, which is how HITS is meant to be used; and no held-out check of whether either ranking helps a searcher.

## Designing with it

### Define the graph before calculating

State what a node and edge represent. On the open web, a node may be a URL, a canonical page or an entire host. A link from a page to itself, repeated navigation links and links created by a template need a deliberate rule. Duplicate pages can multiply votes if they are all counted independently. In an internal collection, permission boundaries may make some links invisible to one user but visible to another. The graph definition changes the score before any formula is applied.

The formula assumes incoming sources have an out-degree $L(q)$ greater than zero. A page with no outgoing links is **dangling**. A full PageRank implementation redistributes its rank, often through the teleport distribution, so probability mass is not lost. The tiny graph avoids this case to keep the arithmetic faithful to the worked example. The damping term alone should not be described as handling dangling nodes without that redistribution step.

### Understand damping and iteration

At $d=0$, every page gets equal teleport probability in the uniform version. At a higher $d$, link paths matter more. At $d=1$, disconnected components or cycles can prevent the stable, unique behaviour the teleport model is designed to supply. The familiar 0.85 is a modelling choice, not a universal optimum. Run synchronous updates until a convergence tolerance is met, not for an arbitrary fixed number of steps in production. The lab lets the effect of each update be seen before convergence.

The PageRank result is query-independent in the simple global form: it says something about graph position, not whether a page answers `passport renewal fee`. A search engine combines it with content and other query-dependent evidence. A high-authority page about bicycles should not outrank a clear council fee page for that query simply because many pages link to it.

### Use HITS for a query-selected neighbourhood

HITS is commonly explained as starting from pages found by a text query and expanding a local base set via links. It then updates authority scores from hubs and hub scores from authorities in that subset. The selected set therefore matters: a poor root query or a link-spam neighbourhood can change the result. A hub may be valuable because it organises links, not because it contains the sought fact itself. A direct answer and a directory should be judged with different user tasks in mind.

| Question | PageRank | HITS |
| --- | --- | --- |
| Main score | One rank per page | Hub and authority scores |
| Usual graph | Global link graph | Query-selected neighbourhood |
| Update | Distribute rank along outgoing links plus teleport | Alternate incoming-authority and outgoing-hub sums |
| Typical failure | Manipulated links, stale graph, topic mismatch | Base-set drift and topic or link spam |

### Resist link manipulation

An edge can be a citation, a navigation link, an advertisement, a cross-site template or a paid placement. Counting all as sincere endorsements invites abuse. Link farms create many pages that point at one target; reciprocal clusters can amplify each other. Deduplication, host-level patterns, link attributes, spam analysis and human review can reduce these effects, but every defence can make mistakes. Track the source of suspicious rank changes rather than hiding the graph signal inside an unexplained composite score.

### Check value against user relevance

Test link features on judged queries after controlling for lexical and semantic relevance. Compare gains by site age and topic, because new legitimate pages start with few links. Inspect a high-rank false positive and ask whether the links express authority for the *query*, or merely popularity in another domain. For a RAG system, page authority can help prioritise candidates, but an answer still needs passages containing the required facts. Graph rank cannot certify the truth of a claim.

## Work through the random-surfer graph

At step zero, A, B and P each have rank $1/3$. A passes its entire follow-link share to P, because A has one out-link. B does the same. P passes its follow-link share to A. Every page also receives $(1-d)/3=0.05$ from uniform teleportation. After one update, P has $0.05+0.85(1/3+1/3)\approx 0.617$, A has $0.05+0.85(1/3)\approx 0.333$, and B has only $0.05$. The total is one. The lab's first button press displays those values.

On the second update, P uses the *previous step's* A and B ranks, not values that are being changed mid-loop. This synchronous rule matters. Updating a dictionary in place can accidentally make later pages in the loop see newer ranks than earlier ones, creating an order-dependent algorithm. The code constructs a fresh mapping each time. Continue the updates and the ranks approach a stable distribution; the first value 0.617 is a worked step, not the final authority of P.

### Interpret the HITS numbers carefully

H1 points to A1 and A2, while H2 points to A1. Both hubs start at one. A1 receives one from each, so its unnormalised authority becomes two. A2 receives only H1's one, so its authority becomes one. The next hub update sums those authority scores across each hub's out-links: H1 becomes three and H2 becomes two. A1 gains from being cited by two hubs; H1 gains from linking to both authorities. That mutual reinforcement is the distinctive HITS idea.

The raw numbers would grow if this unnormalised cycle continued. Implementations normalise each vector, often by its Euclidean length, before the next iteration. Normalisation changes their numeric scale but preserves the comparative pattern of a single update. A hub score is not a probability and cannot be added directly to PageRank without calibration. The board intentionally labels this as the first round.

### Locate failure modes in the graph

Suppose a new official emergency page has no in-links yet. A graph-only rank can undervalue it even when the query exactly names the incident. Conversely, a popular old page can keep many links after its advice becomes obsolete. A relevant-rank evaluation set should include fresh and rare pages so link features cannot win only on popular long-lived topics. The serving system needs a path for explicit source trust, freshness and exact query intent.

Suppose hundreds of pages from one controlled site point to a target. A simple count says the target is popular. A PageRank-like walk may reduce the effect if the supporting pages have little rank, but a sufficiently structured link scheme can still manipulate graph algorithms. Treat the link graph as adversarial data. Inspect domains, templates, duplication and time patterns for suspicious growth. Avoid assuming that a sophisticated graph formula automatically solves spam.

### Connect graph analysis to crawling

Session 10's crawler needs to choose which discovered URL to fetch next. Links can help prioritise: a page linked from useful sources may be worth crawling soon. But a crawl frontier built only from current link authority can entrench old well-connected regions and ignore new sites. Reserve capacity for exploration and diverse seeds. Link analysis depends on the crawl, and the crawl can in turn be guided by link analysis, so the two form a feedback loop. Measure whether that loop improves coverage of relevant pages rather than just reinforcing existing hubs.

The graph also helps explain why canonicalisation matters. If ten duplicate URLs all point to one target, treating them as ten independent endorsers overstates support. If a canonical page changes after a site migration, old links may resolve through redirects. Preserve URL-to-canonical mappings and recalculate graph signals when they change. This is operationally harder than the three-node lesson, but the mathematical principle is the same: the graph you build is the graph your formula scores.

### Decide whether to use a link signal at all

An internal knowledge base with many thoughtfully maintained cross-references may benefit from link-aware ranking. A collection of independent PDFs with few links will not. A news search might use link evidence as one static prior but depend more on recency and content. A question-answering system might use links to expand candidates from a found source, then verify the newly discovered passages. The feature should earn its place against a controlled baseline on the product's judged queries. Keep its provenance visible so a team can explain why a page moved.

## Where this stands in 2026

:::info Industry view

- Classical PageRank and HITS remain useful models for understanding graph evidence, but the exact current ranking formula of a commercial web engine should not be inferred from historical papers.
- Link analysis remains sensitive to spam and duplicate structure; source quality and query relevance need separate checks.
- Graph signals can help discover and prioritise connected content beyond the open web, including documentation, provided permission and freshness rules are respected.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Stopping after one update and quoting 0.617 | It is the number in the formula | Iterate until the change is tiny. In the worked graph P went from 0.617 to 0.376 on the next step |
| Spending effort tuning $d$ for ranking quality | It is the one parameter | On 4,592 articles the order barely changed (Spearman 0.986 to 0.997 across $d$). Tune it for speed: 21 to 71 iterations |
| Treating PageRank as authority on its own | It sounds like quality | It was 0.966 correlated with raw in-links here. A farm of 200 pages made an unremarkable page first |
| Describing teleport as the fix for dangling pages | Both are in the formula | Teleport makes the walk well-behaved. Dangling pages need their rank redistributed, as `rank[dangling].sum() / n` does |
| Running HITS on the whole graph | It is the same code | HITS is defined on a query-selected neighbourhood. Whole-graph scores are dominated by the densest region |

:::note Correction

The practice answers below say teleport "handles dangling nodes". Teleport guarantees that the walk can reach every page and has one stationary distribution. A page with no out-links still has no outgoing share to give, so implementations redistribute its rank, usually uniformly or by the teleport distribution.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What idea underlies link analysis?</summary>

A hyperlink is an endorsement (vote); pages linked by many/important pages are more authoritative; a content-independent signal.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the PageRank formula and explain damping/teleport.</summary>

PR(p) = (1−d)/N + d·Σ_\{q→p\} PR(q)/L(q). The surfer follows links with prob d and teleports with 1−d, guaranteeing every page some rank and the walk a single stationary distribution. Pages with no out-links are a separate case: their rank is redistributed explicitly, as the correction note above explains.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> N=3, d=0.85; page P gets links from two pages each PR=1/3, out-degree 1. Compute PR(P) (one update).</summary>

PR(P) = 0.15/3 + 0.85·(1/3 + 1/3) = 0.05 + 0.85·0.667 = 0.617.<br /><em>Session 11 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How does HITS differ from PageRank?</summary>

HITS computes query-dependent hub and authority scores that reinforce each other; PageRank is a single query-independent global score.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why include a teleport term in PageRank?</summary>

To keep the walk well-behaved on graphs with disconnected parts or cycles, so the Markov chain is ergodic with a unique stationary distribution. Dangling nodes (no out-links) are a separate problem: their rank is redistributed explicitly.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) After two updates on the worked graph, P has 0.3758. Show the arithmetic and explain why it is lower than after one update.</summary>

Page P receives the step-1 ranks of A and B, which are 0.3333 and 0.05: $0.05 + 0.85 \times (0.3333 + 0.05) = 0.3758$. In step 1, P received $1/3$ from each of A and B because both started at $1/3$. In step 2 B's rank had fallen to 0.05, because nobody links to B, so P's inflow dropped. The ranks settle only after several more updates.

</details>

<details>
<summary><strong>Q7.</strong> (Medium) Changing $d$ from 0.5 to 0.99 took 21 and then 71 iterations but changed the ranking only slightly. Explain both facts.</summary>

At $d = 0.5$ half of the rank is reset to uniform every step, so errors die out quickly. At $d = 0.99$ the reset is only 1%, so rank must travel along many links before it settles and more steps are needed. The ranking changes little because the order is driven by who links to whom in both cases, and the teleport share is a small uniform addition. A graph with long chains or a few dominant hubs can be more sensitive than this one.

</details>

<details>
<summary><strong>Q8.</strong> (Stretch) A farm of 200 pages made a 6-in-link page rank first. A trusted-teleport variant left it 1,057th. What does the variant assume, and where does it break?</summary>

It sends all random jumps to a small set of pages judged trustworthy, so fake pages receive no teleport mass of their own and only get what trusted pages pass on. It breaks if a trusted page links to the farm, if the trusted set is too small to reach most of the good pages (ranks of unrelated pages shift too, as the no-farm row moving from 2,297 to 1,980 shows), or if spammers earn links from trusted pages. It reduced the gain but did not remove it: 1,000 farm pages still gained about 1,030 places.

</details>

## Go deeper

- [Stanford IR book: link analysis](https://nlp.stanford.edu/IR-book/html/htmledition/link-analysis-1.html); PageRank, HITS and graph interpretation.
- [Brin and Page: Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/); historical system paper.
- [Wikispeedia navigation paths (SNAP)](https://snap.stanford.edu/data/wikispeedia.html) (opened 2026-10-09); the 4,604-article, 119,882-link Wikipedia graph, with the two papers to cite: West and Leskovec, WWW 2012; West, Pineau and Precup, IJCAI 2009.
- [Gyöngyi, Garcia-Molina and Pedersen: Combating Web Spam with TrustRank](https://www.vldb.org/conf/2004/RS15P3.PDF) (VLDB 2004, opened 2026-10-09); describes link farms as many bogus pages pointing at one target and proposes a small seed set of vetted pages.
- [Kleinberg: Authoritative Sources in a Hyperlinked Environment](https://www.cs.cornell.edu/home/kleinber/auth.pdf) (opened 2026-10-09); the HITS paper.
- [NetworkX: pagerank](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html) (opened 2026-10-09); the default damping of 0.85 and the handling of dangling nodes and personalisation.
- Built from the course lecture "ir-s11-link-analysis" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can compute the first PageRank update and distinguish it from convergence.
- [ ] I can explain how HITS gives separate hub and authority scores.
- [ ] I can state how dangling nodes, damping and graph definition affect PageRank.
- [ ] I can explain why links contribute evidence without proving query relevance or correctness.
- [ ] I can run two PageRank updates by hand on a three-page graph and check that the total stays at 1.
- [ ] I can say what damping changes in practice (iterations, not order) and quote the measured range.
- [ ] I can describe what a link farm does to a page's rank and how much a trusted-teleport defence recovers.
- [ ] I can say why HITS needs a query-selected neighbourhood.

## Where to go next

Next: [Session 12, cross-language retrieval](/docs/theory/ir/cross-language-retrieval). Related: [Collaborative filtering](/docs/theory/recsys/collaborative-filtering), where a graph of users and items is ranked in a similar spirit.
