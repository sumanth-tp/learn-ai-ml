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

## The idea in plain words

A hyperlink can be read as a page referring readers to another page. If many important pages refer to one destination, that destination may be worth examining. This signal is different from a document's words: the target could be described by other authors even if it does not repeat the query phrase. The lecture introduces two classical ways to use that graph, **PageRank** and **HITS**.

PageRank imagines a random surfer who usually follows a link and sometimes jumps to a random page. A page receives a share of each linking page's current rank, divided by that linking page's number of outgoing links. The jump or **teleport** term prevents the walk from being trapped in a disconnected region under the usual complete model. With three pages, follow-link probability $d=0.85$, and two incoming pages each at rank $1/3$ with one outgoing link, the lecture's single update is $(1-0.85)/3+0.85(1/3+1/3)pprox0.617$. One update is not a final stationary score.

HITS treats **hubs** and **authorities** as two roles. A good hub links to good authorities; a good authority receives links from good hubs. Repeated updates reinforce both, usually within a query-selected subgraph. A directory page can be a useful hub without being the definitive answer itself. Neither score proves factual correctness, and both can be manipulated by artificial links.

<Infographic src="/img/ir/link-analysis.svg" alt="Pages A and B each contribute one-third of their current PageRank to P, yielding a first update of 0.617 with damping 0.85 and three pages." caption="The lecture's PageRank arithmetic is one synchronous update on a fully specified three-page graph." />

<Infographic src="/img/ir/hub-authority.svg" alt="Two initial hubs point to two authorities; first HITS updates give authority scores two and one, then hub scores three and two before normalisation." caption="HITS keeps link-giving hubs and link-receiving authorities as different scores." />

:::note Beyond the lecture

The explicit three-page graph, runnable HITS numbers and discussion of spam, edge meaning and modern ranking are teaching extensions. The original search paper below is historical context, not a description of a current private ranking formula.

:::

Run PageRank one step at a time on the board's graph: A→P, B→P and P→A. At the lecture defaults, P moves from **0.333** to **0.617** after the first step. Adjust damping to see the balance between links and teleportation; the table shows each incoming contribution.

<PageRankLab />

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

The first block runs the lecture's one-update calculation on a complete three-page graph. The graph has no dangling node: A and B link to P; P links to A. All three start at $1/3$. Repeated synchronous updates keep the total rank at one.

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

## Designing with it

### Define the graph before calculating

State what a node and edge represent. On the open web, a node may be a URL, a canonical page or an entire host. A link from a page to itself, repeated navigation links and links created by a template need a deliberate rule. Duplicate pages can multiply votes if they are all counted independently. In an internal collection, permission boundaries may make some links invisible to one user but visible to another. The graph definition changes the score before any formula is applied.

The lecture's formula assumes incoming sources have an out-degree $L(q)$ greater than zero. A page with no outgoing links is **dangling**. A full PageRank implementation redistributes its rank, often through the teleport distribution, so probability mass is not lost. The tiny graph avoids this case to keep the arithmetic faithful to the lecture. The damping term alone should not be described as handling dangling nodes without that redistribution step.

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

At step zero, A, B and P each have rank $1/3$. A passes its entire follow-link share to P, because A has one out-link. B does the same. P passes its follow-link share to A. Every page also receives $(1-d)/3=0.05$ from uniform teleportation. After one update, P has $0.05+0.85(1/3+1/3)pprox0.617$, A has $0.05+0.85(1/3)pprox0.333$, and B has only $0.05$. The total is one. The lab's first button press displays those values.

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

## Practice questions

<details>
<summary><strong>Q1.</strong> What idea underlies link analysis?</summary>

A hyperlink is an endorsement (vote); pages linked by many/important pages are more authoritative; a content-independent signal.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the PageRank formula and explain damping/teleport.</summary>

PR(p) = (1−d)/N + d·Σ_\{q→p\} PR(q)/L(q). The surfer follows links with prob d and teleports with 1−d, guaranteeing every page some rank (handling dangling nodes).<br /><em>Session 11 · conceptual</em>

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

To handle dangling nodes (no out-links) and disconnected components, ensuring the Markov chain is ergodic with a unique stationary distribution.<br /><em>Session 11 · conceptual</em>

</details>

## Go deeper

- [Stanford IR book: link analysis](https://nlp.stanford.edu/IR-book/html/htmledition/link-analysis-1.html); PageRank, HITS and graph interpretation.
- [Brin and Page: Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/); historical system paper.
- Built from the course lecture "ir-s11-link-analysis" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can compute the lecture's first PageRank update and distinguish it from convergence.
- [ ] I can explain how HITS gives separate hub and authority scores.
- [ ] I can state how dangling nodes, damping and graph definition affect PageRank.
- [ ] I can explain why links contribute evidence without proving query relevance or correctness.
