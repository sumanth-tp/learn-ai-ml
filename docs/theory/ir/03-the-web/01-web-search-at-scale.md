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

:::tip Before you start

**You should already know**

- What an inverted index is and why a missing document cannot be ranked ([Session 2](/docs/theory/ir/boolean-retrieval)).
- What precision and recall mean ([Session 1](/docs/theory/ir/what-information-retrieval-is)).
- How ranked results are scored ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).

**Reading time:** about 45 minutes, plus about 10 seconds to run the code.

**After this chapter you can**

- Estimate how much bigger one search index is than another from two samples, and say when the estimate fails.
- Trace a missing result back through discovery, indexing and serving.
- Explain why an overlap between two engines cannot, by itself, tell you the size of the web.

:::

## In 30 seconds

A library knows every book it owns. A web search engine does not know how many pages exist, so it cannot simply count what it is missing. Instead it works like an ecologist counting fish: catch some, tag them, release them, catch again, and see how many tags come back. If half of your second catch is tagged, your first catch was about half the pond.

Two engines can be compared the same way. Pick pages from one and check whether the other holds them. The trick is that the answer is only as honest as the sampling. Pages that every engine loves are caught in every net.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Crawl | Fetching pages by following links | A bot follows a link from a news page |
| Index | The searchable copy an engine builds from what it fetched | Engine A indexes 50 pages |
| Canonical URL | The one address chosen to stand for a group of duplicates | `/print/page` folds into `/page` |
| Overlap | Pages held by both engines | 20 pages in both A and B |
| Capture-recapture | Estimating a population from the overlap of two samples | Tag 50 fish, recapture 40, find 20 tagged |
| Query intent | What the person is trying to do | Informational, navigational or transactional |
| Link farm | Many pages built only to point at one target | 200 spam pages all linking to one shop |
| Coverage | Share of the pages that matter that an index actually holds | 60% of the council's pages are indexed |

## The idea in plain words

A library search system can usually enumerate its documents and decide when they change. A web engine cannot assume either. Pages appear and vanish, links lead to new pages, one article may exist at many URLs, and publishers may actively try to influence ranking. The engine must find pages, decide which copies to index, and make useful results available for a query that may be only two words long.

Web queries fall into three common intents. An **informational** query asks for knowledge, such as how to repair a bicycle tyre. A **navigational** query seeks a particular site or page. A **transactional** query suggests an action such as buying a part or making a booking. These are useful lenses rather than rigid labels: `passport renewal` can mean instructions, an official portal or a fee payment. The intended result changes with the user and context.

Web pages form a directed graph. A link can help the crawler discover a target and can provide evidence that another page considers it useful. Neither the link nor term frequency is a verdict. A link farm can be built to manipulate authority; a page can repeat a phrase to match queries. Content, links, freshness and observed user behaviour may all inform ranking, but each can be noisy or biased.

<Infographic src="/img/ir/web-search.svg" alt="Web search connects a changing linked corpus to crawling, indexing and different query intents; the worked overlap probabilities imply a size ratio of 1.25." caption="The web adds discovery and adversarial signals to retrieval. The lower card reproduces the index-size example." />

:::note Added for this site

The production pipeline, sampling caveats and evaluation workflow below extend the core outline. The exact Google Search process described here is supported by Google's public documentation; ranking details it does not publish are not inferred.

:::

Move the two overlap probabilities. At the default settings, $p_A=0.4$ and $p_B=0.5$, the estimated ratio is **$|A|/|B|=1.25$**. The bars show the direction of each sample; the table exposes the denominator for each probability.

<IndexOverlapLab />

## Worked example, step by step

A tiny web has 100 pages. Engine A indexes 50 of them and engine B indexes 40. We do not know the 100, and we cannot see inside either engine. We only draw random pages from each and check the other.

1. **Independent engines.** Suppose 20 pages are in both. A random page from A is in B with probability $p_A = 20/50 = 0.4$. A random page from B is in A with probability $p_B = 20/40 = 0.5$.
2. **Size ratio.** $p_B / p_A = 0.5 / 0.4 = 1.25$, and indeed $50 / 40 = 1.25$. The shared 20 cancels.
3. **Size of the web.** $p_B$ is the share of the web that A covers, so the web is about $|A| / p_B = 50 / 0.5 = 100$ pages. Correct.
4. **Engines that both favour popular pages.** Suppose B's 40 pages are mostly the ones A also holds, so 35 are shared. Now $p_A = 35/50 = 0.7$ and $p_B = 35/40 = 0.875$.
5. **The ratio survives.** $0.875 / 0.7 = 1.25$ again. The shared count still cancels.
6. **The web size does not.** $|A| / p_B = 50 / 0.875 = 57.1$ pages, not 100. The estimate has shrunk to 57% of the truth.

In words: the size ratio only needs fair samples from each engine. The size of the whole web also needs the two engines to choose their pages independently, and real engines do not.

<Infographic src="/img/ir-enrich/ir2-index-size.svg" alt="Bars for the ratio estimate and the web-size estimate in three scenarios: independent engines, popularity-tilted engines, and popularity-biased sampling." caption="Look first at the middle group: the ratio bar stays near the truth while the web-size bar falls to 0.566 of the truth." />

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

The first block verifies the worked ratio from two overlap probabilities. It also constructs exact finite sets with the same values, making the denominator visible: 20 shared pages out of 50 in A and 20 out of 40 in B.

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

### The worked example in code

This block reproduces steps 1 to 6 for both scenarios.

```python
for label, shared in (("independent", 20), ("popularity-tilted", 35)):
    size_a, size_b = 50, 40
    p_a, p_b = shared / size_a, shared / size_b
    print(f"{label:18} p_A={p_a:.3f} p_B={p_b:.3f} ratio={p_b / p_a:.2f} web size={size_a / p_b:.1f}")
```

**Reading the output.** Both rows print a ratio of 1.25. The web size is 100.0 for the independent row and 57.1 for the tilted row.

### An experiment: when does the overlap estimate fail?

How accurate is the overlap estimate, and which assumption breaks it first? The block builds a synthetic web of 200,000 pages, each with a popularity score. Engine A holds about 40% of the pages and engine B about 25%. In the independent case, inclusion ignores popularity. In the tilted case, both engines prefer popular pages, which is how real crawlers behave. We draw uniform samples of 50, 500 and 5,000 pages from each engine, 400 times, and also a popularity-biased sample, as a query-based sampler would produce. Versions used: Python 3.14.6, NumPy 2.5.3, SciPy 1.18.1. The run takes about 4 seconds.

```python
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(7)
universe = 200_000
z = rng.normal(size=universe)
popularity = np.exp(1.5 * z)

def build_engine(coverage, tilt):
    low, high = -20.0, 20.0
    for _ in range(60):
        middle = (low + high) / 2
        if expit(middle + tilt * z).mean() < coverage:
            low = middle
        else:
            high = middle
    return rng.random(universe) < expit(middle + tilt * z)

def estimate(engine_a, engine_b, size, trials, sample_power):
    members = (np.flatnonzero(engine_a), np.flatnonzero(engine_b))
    ratios, web_sizes = [], []
    for _ in range(trials):
        picks = []
        for pool in members:
            weight = popularity[pool] ** sample_power
            picks.append(rng.choice(pool, size, p=weight / weight.sum()))
        p_a = engine_b[picks[0]].mean()
        p_b = engine_a[picks[1]].mean()
        ratios.append(p_b / p_a)
        web_sizes.append(engine_a.sum() / p_b)
    return np.array(ratios), np.array(web_sizes)

print(f"{'engines':<19}{'sampling':<16}{'n':>5}{'true A/B':>10}{'mean':>7}{'5th to 95th':>14}{'web size est/true':>19}")
for tilt, label in ((0.0, "independent"), (2.0, "popularity-tilted")):
    engine_a, engine_b = build_engine(0.40, tilt), build_engine(0.25, tilt)
    true_ratio = engine_a.sum() / engine_b.sum()
    for power, how, sizes in ((0.0, "uniform", (50, 500, 5000)), (0.5, "popular-biased", (500,))):
        for size in sizes:
            ratios, web_sizes = estimate(engine_a, engine_b, size, 400, power)
            low, high = np.percentile(ratios, [5, 95])
            print(f"{label:<19}{how:<16}{size:>5}{true_ratio:>10.3f}{ratios.mean():>7.3f}{f'{low:.2f} to {high:.2f}':>14}{web_sizes.mean() / universe:>19.3f}")
```

The output of the run:

```text
engines            sampling            n  true A/B   mean   5th to 95th  web size est/true
independent        uniform            50     1.604  1.694  0.94 to 2.78              1.034
independent        uniform           500     1.604  1.610  1.38 to 1.87              1.001
independent        uniform          5000     1.604  1.605  1.53 to 1.68              0.998
independent        popular-biased    500     1.604  1.624  1.40 to 1.87              1.001
popularity-tilted  uniform            50     1.602  1.639  1.20 to 2.25              0.568
popularity-tilted  uniform           500     1.602  1.610  1.46 to 1.76              0.566
popularity-tilted  uniform          5000     1.602  1.606  1.56 to 1.65              0.564
popularity-tilted  popular-biased    500     1.602  1.369  1.28 to 1.47              0.486
```

**Reading the output.** `true A/B` is the real size ratio. `mean` and `5th to 95th` describe the 400 estimates of $p_B / p_A$. The last column divides the estimated size of the whole web by the true 200,000, so 1.000 is perfect.

**Line by line.**

- `build_engine` finds, by bisection, the intercept that gives the requested coverage for a given `tilt`. A tilt of 0 ignores popularity.
- `sample_power` weights the sample by popularity to the given power. Zero is a fair sample.
- `engine_b[picks[0]].mean()` checks the pages sampled from A for membership in B, which is $p_A$.

### How wide is the uncertainty from one sample?

A real study has one sample, not 400. The next block draws one sample of each size from two independent synthetic engines, builds a 90 per cent bootstrap interval for the ratio, and repeats that 300 times to see how often the interval contains the truth.

```python
import numpy as np

rng = np.random.default_rng(11)
universe = 100_000
engine_a = rng.random(universe) < 0.40
engine_b = rng.random(universe) < 0.25
members_a, members_b = np.flatnonzero(engine_a), np.flatnonzero(engine_b)
true_ratio = engine_a.sum() / engine_b.sum()

def interval(size, resamples=500, level=0.90):
    from_a = engine_b[rng.choice(members_a, size)]
    from_b = engine_a[rng.choice(members_b, size)]
    draws = []
    for _ in range(resamples):
        p_a = from_a[rng.integers(0, size, size)].mean()
        p_b = from_b[rng.integers(0, size, size)].mean()
        draws.append(p_b / p_a if p_a > 0 else np.inf)
    tail = (1 - level) / 2
    return np.quantile(draws, [tail, 1 - tail])

print(f"true |A|/|B| = {true_ratio:.3f}; 90 per cent bootstrap interval from ONE sample per engine, 300 repeats")
print(f"{'sample size':>12}{'covers truth':>14}{'mean width':>12}")
for size in (50, 100, 400):
    results = [interval(size) for _ in range(300)]
    covered = np.mean([low <= true_ratio <= high for low, high in results])
    width = np.mean([high - low for low, high in results])
    print(f"{size:>12}{covered:>14.3f}{width:>12.3f}")
```

```text
true |A|/|B| = 1.616; 90 per cent bootstrap interval from ONE sample per engine, 300 repeats
 sample size  covers truth  mean width
          50         0.870       2.110
         100         0.903       1.249
         400         0.883       0.575
```

**Reading the output.** `covers truth` should sit near 0.90 if the interval is honest. `mean width` is the average gap between the interval's ends.

### What the numbers say

The ratio estimator held up. With fair samples of 500 from independent engines the mean estimate was 1.610 against a true 1.604, with 90% of estimates between 1.38 and 1.87. At 5,000 the spread shrank to 1.53 to 1.68. At 50 the mean drifted up to 1.694 and the spread ran from 0.94 to 2.78, so a small sample can even suggest the wrong engine is bigger.

The surprise is that popularity-tilted engines did not hurt the ratio at all: 1.610 against 1.602 at 500 samples. They wrecked the web-size estimate instead, which came out at 0.566 of the truth, because the independence assumption was false. The textbook method states that assumption openly and calls it far from true for real engines (Stanford IR book, opened 2026-10-09).

Sampling bias is the other trap. Drawing popular-biased samples from the tilted engines pulled the ratio down to 1.369 (range 1.28 to 1.47) against a true 1.602, a shortfall of 0.23 that the interval does not show, because the interval measures noise, not bias. With independent engines the same biased sampler was harmless (1.624). The bootstrap intervals covered the truth 0.870, 0.903 and 0.883 of the time at sample sizes 50, 100 and 400, near the nominal 0.90; with 300 repeats each, a coverage figure has about 0.017 of noise of its own.

Limits: a synthetic web with a log-normal popularity and logistic inclusion, one seed per scenario, engines that differ only in coverage, and no duplicate pages or dynamic URLs.

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

The overlap formula is exact for complete sets. A measured version uses samples and has uncertainty. If only a handful of sampled pages overlap, a small counting change can move the ratio substantially. URL normalisation, canonical groups, inaccessible pages and differing definitions of a "page" make membership ambiguous. Some pages exist in both indexes under different URLs; some copies share a URL but contain different content snapshots. State the sampling unit, collection time and matching rule.

A ratio also says nothing by itself about quality. Index A may contain 25% more pages than B while missing many pages users care about. Evaluate coverage by topic, language, region and update age. A larger index can even add low-quality duplicates that make ranking harder. The ratio is a way to reason about scale, not a product score.

### Protect latency and trust

Web serving must handle many short queries quickly. Candidate retrieval, expensive ranking and presentation therefore have distinct budgets. A system can cache common results or precompute static signals, while using query-dependent evidence at serving time. Keep permission checks and removal requests outside ranking heuristics: content that must not be served needs an enforced rule, not merely a low score. Preserve result provenance, canonical URL and crawl age so users and operators can judge whether a result is current.

## Trace one page from discovery to a result

Imagine a local council publishes a new page explaining a grant. A link from its news page exposes the URL to a crawler. The URL frontier schedules a fetch subject to host access rules and politeness. The response must be successful, the main text must be extractable, and the page may be grouped with a PDF copy or a print view. The index stores a chosen representation. Only when someone searches does query matching and ranking put it in a result list. A missing result might therefore mean the link was never seen, the crawl was blocked, the content was not extracted, a duplicate was selected, or the query did not match. Each case has a different fix.

The council example also shows why freshness matters. A grant deadline may change after the initial crawl. Re-fetching every URL every minute would overload sites and waste resources; never refreshing gives stale answers. A crawler can prioritise important or frequently changing pages and record when each indexed copy was last observed. A search product should avoid presenting an old deadline as current without checking the source. The exact refresh policy depends on host capacity, change frequency and user impact.

### See where the overlap formula can mislead

Suppose a sample from A has 40% of its pages in B and a sample from B has 50% in A. The worked example gives $|A|/|B|=1.25$. That is a ratio of the indexed sets under a consistent matching rule. If B's sample happens to contain many popular pages found everywhere, while A's sample includes more obscure pages, the two estimates may not represent uniform samples. If one engine counts a mobile URL and desktop URL separately and the other consolidates them, the sets are not even defined the same way. A robust study repeats samples, reports uncertainty and checks strata rather than treating 1.25 as an exact measurement of the live web.

The shared count cancels algebraically, but sampling error does not. An overlap of 20 gives the exact toy result because we constructed full sets. In a real estimate, 20 observed matches out of 50 and 20 out of 40 are noisy proportions. Their errors can be correlated because both concern the same overlap. If the indexes change while sampling is underway, the target ratio changes too. Record timestamps and prefer a defined snapshot whenever possible.

### Connect this chapter to the rest of retrieval

The inverted index from Sessions 1 to 4 still supports fast term lookup. The scoring models from Session 5 still help rank candidates. Session 7 still supplies measures for judged query results. What changes is the environment: documents are discovered through links, many copies refer to one content item, site owners may be adversarial, and the collection is never fully known. Session 10 explains the crawler; Session 11 explains one link signal. Neither replaces relevance judgement. A page can have excellent link authority and still fail a specific query, while a new page with few links can be the right answer.

For a retrieval-augmented assistant built on web pages, the same chain affects the answer. If the index uses stale content or collapses a crucial page into an unsuitable canonical copy, the generator will not see the right evidence. If a spam page enters the top passages, its text can contaminate an answer. Evaluate retrieval separately from generation, inspect sources and keep an audit trail from answer to indexed document and crawl time. The pipeline is useful precisely because it lets a team locate the failure before changing the model.

### Decide what a good search session means

One query is not always the whole task. A person may begin with broad information, reformulate with a product name and then navigate to a transaction. A web-search evaluation set should include those journeys as well as single queries. For a navigational query, success may be the correct destination at rank one. For a broad information query, credible diversity can matter more than one exact page. For a transaction, a result that opens a working flow is more useful than a page that merely contains the right words. This context determines which results should be labelled relevant and which offline metric deserves weight.

The three intent names are a starting vocabulary. Some queries mix them, and some users change intent mid-session. Keep the rubric open to ambiguity: multiple result types can be acceptable if they serve plausible needs. When an evaluation score falls, inspect the actual result, query and user task before deciding whether the ranker, crawler or interface needs work.

## Where this stands in 2026

:::info Industry view

- Google's current public Search guide still separates crawling, indexing and serving; it also says none of the stages is guaranteed for every URL.
- Canonicalisation remains part of index quality, because multiple URLs may represent the same or similar page.
- Search quality is now evaluated across different user intents and collection slices, while adversarial pages and feedback bias still need explicit attention.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Reading the overlap ratio as the size of the web | The same two numbers produce both | Use the ratio for two engines only. Total size needs independent engines, and this chapter's tilted case estimated 0.566 of the truth |
| Sampling pages from what the engine shows first | Top results are easy to collect | Top results are popular pages. Popular-biased samples dragged the ratio from 1.602 to 1.369 |
| Quoting a ratio from 50 samples | The formula is exact for complete sets | Report an interval. At 50 samples the 90% range ran from 0.94 to 2.78 |
| Counting URLs as pages | A URL is the thing you can see | Match on canonical pages, or one engine's mobile and desktop URLs count twice |
| Debugging ranking when a page is missing | Ranking is the visible part | Trace discovery, fetch, extraction, duplicate choice and indexing first |

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

<details>
<summary><strong>Q6.</strong> (Medium) Both engines hold the most popular pages, yet the size-ratio estimate stayed accurate while the web-size estimate was 43% too small. Why?</summary>

The ratio is $p_B / p_A = (|A \cap B| / |B|) / (|A \cap B| / |A|)$, and the shared count cancels whatever the overlap is. It only needs a fair sample from each engine. The web-size estimate treats the share of B found in A as A's coverage of the whole web. If both prefer popular pages, B's pages are unusually likely to be in A, that share is too high, and the estimate of the web is too low. In the run it was 0.566 of the truth.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) A study draws its sample from pages returned for random queries and finds a ratio of 1.37 where a fair sample would give 1.60. Name the cause and one way to check it.</summary>

Query results favour popular and long pages, so the sample is not uniform. Popular pages are more likely to be in both engines, which pushes both overlap fractions towards 1 and the ratio towards 1. To check it, repeat the study on a stratum of rare pages, or compare against a sample drawn without using ranking, for example from a crawl of random hosts. The bootstrap interval will not reveal it, because the interval measures noise, not bias.

</details>

## Go deeper

- [Stanford IR book: web search basics](https://nlp.stanford.edu/IR-book/html/htmledition/web-search-basics-1.html); the classical web-search framing.
- [Google Search Central: how Search works](https://developers.google.com/search/docs/fundamentals/how-search-works); a current public pipeline description.
- [Google Search Central: canonical URLs](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls); duplicate consolidation signals.
- [Stanford IR book: index size and estimation](https://nlp.stanford.edu/IR-book/html/htmledition/index-size-and-estimation-1.html) (opened 2026-10-09); the capture-recapture method, its independence and uniformity assumptions, and the bias of random-query sampling.
- [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html) (opened 2026-10-09); states that robots rules are not a form of access authorisation.
- Built from the course lecture "ir-s9-web-search" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can explain why web retrieval needs discovery, duplicate control and spam resistance in addition to ranking.
- [ ] I can distinguish informational, navigational and transactional intents without assuming every query has one fixed label.
- [ ] I can derive the 1.25 index-size ratio and state its sampling assumptions.
- [ ] I can trace a missing result through crawling, indexing and serving before changing its score.
- [ ] I can compute the overlap ratio and the web-size estimate by hand, and say which one needs independent engines.
- [ ] I can explain why popularity-biased samples shrink the ratio towards 1 and why an interval will not reveal it.
- [ ] I can say how many samples I would want before quoting an index-size ratio, and why 50 is too few.

## Where to go next

Next: [Session 10, web crawling and distributed indexes](/docs/theory/ir/web-crawling-and-distributed-indexes), which builds the machinery that decides what an index holds. Related: [Evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), which puts intervals around scores.
