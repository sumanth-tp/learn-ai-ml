---
id: senior-case-ranking
title: "Design Search and Recommendation Ranking"
sidebar_label: "3 · Ranking"
sidebar_position: 3
slug: /senior/design-search-and-recommendation-ranking
description: "A whole-system design for ranking a huge catalogue: the retrieve, light-rank, heavy-rank and re-rank funnel, its compute budget, the ceiling retrieval puts on quality, and the position bias hidden in click data, all simulated in code."
tags: [system-design, ranking, recommendation, search, candidate-generation, position-bias, learning-to-rank]
---

import Infographic from '@site/src/components/Infographic';
import RankingFunnelLab from '@site/src/components/viz/RankingFunnelLab';

**In one line.** A ranking system for a large catalogue is a funnel of ever more expensive models over ever fewer items, so the design questions are how many items each stage keeps, what the early stages permanently throw away, and whether the clicks you train on measure relevance or only position.

:::note Not from a lecture
Written for this site from the public sources listed under Further reading. The catalogue, users and costs are an invented teaching scenario; the funnel and bias results come from simulations in this chapter, and unit costs are parameters you replace with measurements.
:::

## The idea in plain words

Imagine a marketplace with 50 million items, and a page that shows ten. Whether the page is a search result or a "recommended for you" shelf, the job is the same: **pick the ten items this person is most likely to value, in about a tenth of a second.**

The best model you can build for that is expensive per item: it reads rich features about the user, the item and their interaction. Running it on 50 million items for every request would take thousands of CPU seconds. So the system never tries. It works as a funnel:

1. **Retrieve** a thousand plausible items cheaply, usually by nearest-neighbour search over embeddings.
2. **Light-rank** them with a small model and keep a couple of hundred.
3. **Heavy-rank** those with the expensive model.
4. **Re-rank** with rules for diversity, freshness and business limits, and show ten.

Google's recommendation course describes the same three stages (candidate generation, scoring, re-ranking), and the YouTube paper of 2016 (Covington et al.) describes a deep candidate-generation model followed by a separate deep ranking model. The trap in a funnel is that **an item dropped early cannot be recovered later**, so the quality of the whole system is capped by its first stage.

<Infographic src="/img/senior/search-and-recommendation-ranking-funnel.svg" alt="A funnel from 50 million items through retrieval of 1,000, a light ranker, a heavy ranker scoring 200 and a final slate of 10, with the CPU cost per stage, a simulated precision table and the retrieval ceiling for a perfect ranker." caption="The funnel, its compute cost and the simulated results printed by the code blocks below." />

## How it works

### Requirements, with numbers

| Requirement | Value used in this chapter |
| --- | --- |
| Catalogue | 50,000,000 items, thousands added a day |
| Load | 20,000 requests per second at peak |
| Latency | 150 ms end to end, so about 20 ms of model compute per request |
| Output | a slate of 10, with no more than a few from one seller or topic |
| Quality | precision and NDCG at 10 offline; engagement and long-term retention online |
| Freshness | a new item can be retrieved within minutes |

### Back-of-envelope sizing

The first code block prices the funnel with **assumed** unit costs: 2 microseconds per item for the light ranker, 60 for the heavy ranker, 3 ms for retrieval. Scoring the full catalogue with the heavy ranker would cost 3,000 CPU seconds per request. The funnel retrieving 1,000, light-ranking 1,000 and heavy-ranking 200 costs 17.0 CPU ms, which is 680 cores at 20,000 requests per second and half-loaded cores. The sensitivity table makes the point that the heavy ranker's depth dominates: keeping 800 instead of 200 costs 53.0 ms and 2,120 cores.

### Data flow

**Offline.** Interaction logs are used to train the embedding models, the rankers and the item indexes (embeddings in an approximate-nearest-neighbour index, features in a store). **Online.** A request carries a user and context. Retrieval returns candidates from one or more sources; the two rankers score them with features fetched from the store; the re-ranker applies rules; the slate is returned and the impression is logged with its position, because that log is the next training set.

Meta's engineering post on Instagram Explore (9 August 2023) describes four such stages: retrieval from several sources (heuristic, learned and real-time) producing hundreds of candidates from billions, a lightweight first-stage ranker over thousands of candidates trained to imitate the heavier second stage, a heavier multi-task second-stage ranker over about 100 candidates, and a final re-ranking with rules such as not showing consecutive items from the same author.

### The decisions that matter

**1. How deep does each stage go?**

| Choice | Effect | Cost |
| --- | --- | --- |
| Shallow retrieval (hundreds) | cheap; ceiling can bind | the ranker never sees the best items |
| Deep retrieval (thousands) | higher ceiling | the light ranker works harder; latency |
| Wide heavy-ranker stage | better final precision | cost grows linearly with items kept |

The second code block measures this tradeoff.

**2. How does retrieval find candidates?**

| Source | Strength | Weakness |
| --- | --- | --- |
| Embedding nearest neighbours (two-tower) | personalised, fast with an ANN index | misses what the embedding does not encode |
| Popularity, trending, recent | covers cold start and freshness | not personal |
| Co-engagement and item-to-item lists | strong "people who liked this" signal | needs enough history |
| Keyword retrieval (search) | exact match on the query | misses paraphrase |

Real systems union several. The Instagram post lists heuristic, learned and real-time sources feeding one funnel.

**3. What is the ranking objective?** Predicting clicks is easy and misleading. Instagram's post describes a multi-task model whose predicted probabilities are combined by a value formula, expected value equal to a weighted sum of positive actions minus a weighted negative one such as "see less". The weights are product decisions, which makes the objective explicit and tunable without retraining.

**4. How do you handle position bias in training data?**

| Option | How | Catch |
| --- | --- | --- |
| Clicks at face value | train on clicks | learns the old ranker's order |
| Inverse propensity weighting | weight each click by one over its examination probability | needs propensities, adds variance |
| Randomised slice of traffic | shuffle a small share, log those clicks | costs a little engagement, gives clean data |

Joachims, Swaminathan and Schnabel (2016) formalise the propensity approach for learning to rank from biased clicks. The third code block shows all three.

### Failure modes and mitigations

| Failure | Looks like | Mitigation |
| --- | --- | --- |
| Retrieval ceiling | rankers improve and the metric does not | measure recall of the relevant set per stage; add sources |
| Stage disagreement | the light ranker drops what the heavy ranker loves | train the light stage to imitate the heavy one; track survival rate |
| Feedback loop | items never shown never get clicks | exploration traffic; propensity-aware training |
| Popularity bias | head items crowd out the long tail | diversity rules; calibrate on exposure |
| Tail latency | one slow feature fetch stalls the page | parallel fetch, timeouts, cached fallback slates (see the [fraud case](/docs/senior/design-real-time-fraud-scoring)) |
| Stale embeddings | new items invisible, old taste persists | incremental index updates; retrain schedule |

### Evaluation and rollout

Offline, evaluate each stage on its own: recall of relevant items at retrieval, survival through the light ranker, and precision or NDCG at 10 at the end. Offline numbers use biased logs, so treat them as a filter, not a verdict. Online, run controlled experiments on engagement and on guardrails (latency, diversity, complaints), ramp slowly, and keep a randomised slice for unbiased measurement. The [evaluating ranked retrieval chapter](/docs/theory/ir/evaluating-ranked-retrieval) covers the metrics.

## A real system that works this way

- **YouTube (2016).** The paper "Deep Neural Networks for YouTube Recommendations" (Covington, Adams and Sargin, RecSys 2016) describes a two-stage system, a deep candidate-generation model and a separate deep ranking model, built for a very large corpus.
- **Instagram Explore (2023).** The four-stage funnel above, including the distillation of the second-stage ranker into the first stage and the value formula that combines predicted actions.
- **Airbnb search (2018).** In "Applying Deep Learning to Airbnb Search" (Haldar et al.) the team describes moving search ranking from gradient-boosted trees to neural networks after the tree models' gains had plateaued, and frames the paper as lessons from the transition.
- **The general pattern.** Google's course and Eugene Yan's June 2021 write-up on discovery systems both describe a fast, coarse retrieval step followed by a slower, more precise ranking step.

## Code you can run

Python 3.14.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, run on 2 October 2026. The data in the second and third blocks is **simulated**, with a known ground truth, so the funnel's losses can be measured exactly. The result is a mechanism, not a benchmark.

#### 1. Compute budget of the funnel

```python
CATALOGUE = 50_000_000
REQUESTS_PER_SECOND = 20_000
UTILISATION = 0.5
LIGHT_US_PER_ITEM = 2
HEAVY_US_PER_ITEM = 60
SLATE = 10

retrieval_ms = 3.0
light_ms = 1000 * LIGHT_US_PER_ITEM / 1000
heavy_ms = 200 * HEAVY_US_PER_ITEM / 1000
funnel_ms = retrieval_ms + light_ms + heavy_ms
brute_force_s = CATALOGUE * HEAVY_US_PER_ITEM / 1e6
print(f"scoring the whole catalogue with the heavy ranker: {brute_force_s:,.0f} CPU seconds per request")
print(f"funnel: retrieve 1,000, light-rank 1,000, heavy-rank 200, show {SLATE}")
print(f"  retrieval {retrieval_ms:.1f} ms + light {light_ms:.1f} ms + heavy {heavy_ms:.1f} ms = {funnel_ms:.1f} CPU ms per request")
cores = REQUESTS_PER_SECOND * funnel_ms / 1000 / UTILISATION
print(f"  at {REQUESTS_PER_SECOND:,} requests per second and {UTILISATION:.0%} utilisation: {cores:,.0f} cores")
print(f"  the catalogue is cut by a factor of {CATALOGUE / SLATE:,.0f} before the user sees anything")

print("\nwhere the CPU goes as the heavy ranker keeps more items")
print("heavy ranker keeps   CPU ms per request   cores")
for keep in (50, 100, 200, 400, 800):
    ms = retrieval_ms + light_ms + keep * HEAVY_US_PER_ITEM / 1000
    print(f"{keep:18d}   {ms:18.1f}   {REQUESTS_PER_SECOND * ms / 1000 / UTILISATION:5,.0f}")
```

The whole catalogue through the heavy ranker is 3,000 CPU seconds per request, a factor of five million more than the funnel's 17.0 ms. The sensitivity table is the budget conversation: 50 heavy-ranked items need 320 cores, 200 need 680 and 800 need 2,120.

#### 2. A simulated funnel with a known ground truth

8,000 items and 4,000 users with hidden tastes. Each user's **relevant** set is the 80 items of highest true utility (1% of the catalogue). Retrieval uses 32-dimensional embeddings factorised from logged clicks and takes the dot product. The light ranker is a gradient-boosted model on three cheap features (embedding score, item popularity, the user's affinity for the item's category). The heavy ranker adds one expensive feature, the similarity between the item's content vector and the average of what the user clicked. Rankers train on logged impressions and clicks; evaluation uses true utility on 300 users.

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

USERS, ITEMS, DIM, CATS, LOGGED, EVAL, RELEVANT = 4000, 8000, 16, 8, 120, 300, 80
rng = np.random.default_rng(11)

item_vec = rng.normal(0, 1, (ITEMS, DIM))
item_cat = rng.integers(0, CATS, ITEMS)
quality = rng.normal(0, 1, ITEMS)
user_vec = rng.normal(0, 1, (USERS, DIM))
favourites = rng.integers(0, CATS, USERS)


def utility(users):
    dot = user_vec[users] @ item_vec.T / np.sqrt(DIM)
    liked = item_cat[None, :] == favourites[users][:, None]
    return dot + 0.5 * quality[None, :] + 1.5 * liked


exposure = np.exp(0.6 * quality)
exposure /= exposure.sum()
rows, cols, impressions = [], [], []
for u in range(USERS):
    shown = rng.choice(ITEMS, LOGGED, replace=False, p=exposure)
    util = utility(np.array([u]))[0, shown]
    clicked = rng.random(LOGGED) < 1 / (1 + np.exp(-(1.6 * util - 2.5)))
    for i, c in zip(shown, clicked):
        impressions.append((u, i, c))
        if c:
            rows.append(u)
            cols.append(i)
imp = np.array(impressions)
print(f"{USERS} users, {ITEMS:,} items, {len(imp):,} logged impressions, {len(rows):,} clicks ({len(rows) / len(imp):.1%})")

R = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(USERS, ITEMS))
svd = TruncatedSVD(n_components=32, random_state=0).fit(R)
item_emb = svd.components_.T
user_emb = R @ item_emb
item_pop = np.log1p(np.asarray(R.sum(axis=0)).ravel())
onehot = csr_matrix((np.ones(ITEMS), (np.arange(ITEMS), item_cat)), shape=(ITEMS, CATS))
affinity_counts = (R @ onehot).toarray() + 0.5
affinity = affinity_counts / affinity_counts.sum(axis=1, keepdims=True)


clicked_content = R @ item_vec
clicked_content = clicked_content / np.maximum(np.asarray(R.sum(axis=1)), 1)


def pair_features(users, items, heavy):
    dot = np.einsum("ij,ij->i", user_emb[users], item_emb[items])
    cols_ = [dot, item_pop[items], affinity[users, item_cat[items]]]
    if heavy:
        cols_.append(np.einsum("ij,ij->i", clicked_content[users], item_vec[items]))
    return np.column_stack(cols_)


u_i, i_i, c_i = imp[:, 0].astype(int), imp[:, 1].astype(int), imp[:, 2].astype(int)
light = HistGradientBoostingClassifier(max_iter=80, random_state=0).fit(pair_features(u_i, i_i, False), c_i)
heavy = HistGradientBoostingClassifier(max_iter=80, random_state=0).fit(pair_features(u_i, i_i, True), c_i)

eval_users = np.arange(EVAL)
util_eval = utility(eval_users)
relevant = np.argsort(-util_eval, axis=1)[:, :RELEVANT]
retrieval_score = user_emb[eval_users] @ item_emb.T

K1S, K2S = (100, 300, 1000, 2000), (50, 100, 200)
results = {}
for k1 in K1S:
    for k2 in K2S:
        if k2 > k1:
            continue
        stage1 = stage2 = final_p = final_ndcg = 0.0
        for n, u in enumerate(eval_users):
            cand = np.argpartition(-retrieval_score[n], k1)[:k1]
            rel = set(relevant[n])
            stage1 += len(rel & set(cand)) / RELEVANT
            us = np.full(len(cand), u)
            keep = cand[np.argsort(-light.predict_proba(pair_features(us, cand, False))[:, 1])[:k2]]
            stage2 += len(rel & set(keep)) / RELEVANT
            ranked = keep[np.argsort(-heavy.predict_proba(pair_features(np.full(len(keep), u), keep, True))[:, 1])][:10]
            hit = np.array([r in rel for r in ranked], dtype=float)
            final_p += hit.mean()
            final_ndcg += (hit / np.log2(np.arange(2, 12))).sum() / (1 / np.log2(np.arange(2, 12))).sum()
        results[(k1, k2)] = tuple(round(v / EVAL, 3) for v in (stage1, stage2, final_p, final_ndcg))

print("\nretrieve K1 -> light ranker keeps K2 -> heavy ranker shows 10")
print("   K1    K2   recall@K1   recall@K2   precision@10   NDCG@10")
for (k1, k2), (a, b, p, n) in results.items():
    print(f"{k1:5d} {k2:5d}   {a:9.3f}   {b:9.3f}   {p:12.3f}   {n:7.3f}")
print("\nceiling: precision@10 of a perfect ranker applied to the K1 candidates")
for k1 in K1S:
    best = 0.0
    for n in range(EVAL):
        cand = np.argpartition(-retrieval_score[n], k1)[:k1]
        best += min(10, len(set(relevant[n]) & set(cand))) / 10
    print(f"  K1 = {k1:4d}: {best / EVAL:.3f}")
```

From 480,000 logged impressions and 119,033 clicks the funnel gives a table in which three effects are visible.

- **Depth helps, then plateaus.** With 100 kept, precision at 10 rises from 0.238 (retrieve 100) to 0.330 (retrieve 1,000). With 200 kept, going from 1,000 to 2,000 candidates adds only 0.006 (0.345 to 0.351) for twice the light-ranker work.
- **The heavy stage's width matters.** At 1,000 retrieved, keeping 50, 100 and 200 gives 0.289, 0.330 and 0.345.
- **Retrieval caps the system, but only when shallow.** A perfect ranker applied to 100 candidates could reach a precision at 10 of at most 0.576; at 300 the cap is 0.954 and from 1,000 it is 1.000. Past 300 the gap to 1.0 is **ranker quality**, not retrieval, and recall of the 80 relevant items is only 0.335 at 1,000 because the cap needs only 10 hits, not all 80.

The lab replays this table. Its defaults (1,000 retrieved, 200 kept) show precision at 10 of 0.345 and NDCG at 10 of 0.337, and 17.0 CPU ms and 680 cores at the same assumed costs as the first block.

<RankingFunnelLab />

#### 3. Clicks are not relevance

Each query has 20 candidate items with a known true click probability. Users examine lower positions less. An old ranker (which mis-weights two features) chose the order, clicks were logged, and a new ranker is trained on those clicks three ways, then scored on fresh slates by NDCG at 5 against the true probabilities.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

rng = np.random.default_rng(5)
FEATURES, SLATE, QUERIES = 6, 20, 6000
w_true = np.array([1.2, 0.9, 0.0, 1.4, -0.8, 0.5])
w_old = np.array([1.2, 0.9, 1.5, 0.0, -0.8, 0.5])
examine = 1.0 / (1.0 + np.arange(SLATE)) ** 0.8


def make_slates(n):
    x = rng.normal(0, 1, (n, SLATE, FEATURES))
    p_true = 1 / (1 + np.exp(-(x @ w_true - 1.5)))
    return x, p_true


def log_clicks(x, p_true, order):
    n = len(x)
    pos = np.empty((n, SLATE), dtype=int)
    for q in range(n):
        pos[q, order[q]] = np.arange(SLATE)
    clicks = rng.random((n, SLATE)) < examine[pos] * p_true
    return pos, clicks


def ndcg5(x, p_true, model):
    score = model.predict_proba(x.reshape(-1, FEATURES))[:, 1].reshape(len(x), SLATE)
    gains, ideal = 0.0, 0.0
    disc = 1 / np.log2(np.arange(2, 7))
    for q in range(len(x)):
        top = np.argsort(-score[q])[:5]
        gains += (p_true[q, top] * disc).sum()
        ideal += (np.sort(p_true[q])[::-1][:5] * disc).sum()
    return gains / ideal


x, p_true = make_slates(QUERIES)
old_score = x @ w_old + rng.normal(0, 0.5, (QUERIES, SLATE))
old_order = np.argsort(-old_score, axis=1)
pos, clicks = log_clicks(x, p_true, old_order)
rand_order = np.argsort(rng.random((QUERIES, SLATE)), axis=1)
rpos, rclicks = log_clicks(x, p_true, rand_order)
xt, pt = make_slates(1500)

flat = x.reshape(-1, FEATURES)
y = clicks.reshape(-1)
weights = (1.0 / examine[pos]).reshape(-1)
fit = lambda feats, labels, w=None: HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(feats, labels, sample_weight=w)

old_rank_model = type("M", (), {"predict_proba": lambda self, f: np.column_stack([np.zeros(len(f)), f @ w_old])})()
oracle_model = type("M", (), {"predict_proba": lambda self, f: np.column_stack([np.zeros(len(f)), f @ w_true])})()
print("examination probability by position (the model of attention): ", " ".join(f"{examine[i]:.2f}" for i in (0, 1, 4, 9, 19)), "for positions 1, 2, 5, 10, 20")
print(f"click rate on randomly ordered slates: position 1 {rclicks[rpos == 0].mean():.3f}, position 20 {rclicks[rpos == 19].mean():.3f}")
print("\nranker trained on                                  NDCG@5 on fresh slates (true relevance)")
print(f"{'old ranker itself':50s} {ndcg5(xt, pt, old_rank_model):.4f}")
print(f"{'clicks, taken at face value':50s} {ndcg5(xt, pt, fit(flat, y)):.4f}")
print(f"{'clicks, weighted by 1 / P(examined)':50s} {ndcg5(xt, pt, fit(flat, y, weights)):.4f}")
print(f"{'clicks from randomly ordered slates':50s} {ndcg5(xt, pt, fit(flat, rclicks.reshape(-1))):.4f}")
print(f"{'perfect ordering':50s} {ndcg5(xt, pt, oracle_model):.4f}")
```

Attention drops fast: in random order, position 1 gets a click rate of 0.308 and position 20 gets 0.026. The old ranker itself scores 0.7176. A new ranker trained on clicks **at face value** reaches 0.9229, **weighted by one over the examination probability** 0.9331, and trained on **randomly ordered** slates 0.9776. The ordering of results is the lesson: the randomised slice is cleanest, the weighting helps a little, and face value inherits the old ranker's blind spots. The weighting here uses the true examination model, which real systems must estimate, and that estimate adds error.

<Infographic src="/img/senior/search-and-recommendation-ranking-position-bias.svg" alt="Two tables: examination probability and click rate by position, and ranker NDCG at 5 when trained on face-value clicks, weighted clicks, randomised clicks and compared with the old ranker." caption="The position-bias results printed by the third code block." />

## Designing with it

**Cost estimate as a formula.**

`cores = requests per second x (retrieval ms + K1 x light microseconds / 1000 + K2 x heavy microseconds / 1000) / 1000 / utilisation`

plus the index memory (items x embedding size) and the feature store. Every term is a parameter you can replace with a measurement; the K2 term usually dominates.

**What I would build first.**

1. Popularity plus a simple item-to-item retrieval source, with every impression and its position logged.
2. A single ranker trained on clicks, with a small randomised slice from day one.
3. An embedding retrieval source and an ANN index, measured by recall of items users went on to engage with.
4. A light ranker only when the heavy ranker's cost forces it, trained to imitate the heavy one.
5. A value formula and re-ranking rules last, when product goals beyond clicks are clear.

## Where this stands in 2026

:::info Industry view

- **The funnel is the documented pattern.** Google's course, the YouTube paper and Meta's 2023 post agree on stages that trade cost for precision. I did not find or verify newer public descriptions for this chapter, so recheck before quoting them as current.
- **Distillation between stages is part of the design.** Meta describes training the first-stage model to predict the second stage's outputs, a way to reduce stage disagreement.
- **Objectives are explicit value functions.** The Instagram formula combines predicted probabilities with product-set weights.
- **Bias-aware training is established.** Propensity-weighted learning from clicks is described in the 2016 paper cited here; randomised exploration remains the simplest source of unbiased signal.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why not run the best ranker over the whole catalogue?</summary>

At 60 microseconds per item, 50 million items cost 3,000 CPU seconds per request, millions of times the 17.0 ms the funnel uses. The funnel spends heavy compute only on items that cheaper stages kept.

</details>

<details>
<summary><strong>Q2.</strong> In the simulation, retrieval depth 2,000 gives only 0.006 more precision than 1,000. What do you do with that?</summary>

Stay at about 1,000. The extra candidates mostly add light-ranker work and latency. The remaining gap to a perfect 1.0 is the rankers' quality, so spend effort on features and models there, and revisit depth if the catalogue or user mix changes.

</details>

<details>
<summary><strong>Q3.</strong> Retrieving 100 items caps precision at 10 at 0.576 for even a perfect ranker. How would you find this in a real system?</summary>

Measure recall per stage on a labelled set: take items users engaged with, check whether retrieval returned them, and compute what a ranker with perfect knowledge could achieve given the candidates. If the ceiling is low, improve or add retrieval sources before touching the ranker.

</details>

<details>
<summary><strong>Q4.</strong> Why is the light ranker trained to imitate the heavy one?</summary>

If the two disagree, the light stage discards items the heavy stage would have ranked highly, and those losses are invisible downstream. Training the light model on the heavy model's outputs, as the Instagram post describes, aligns what survives with what the final stage wants.

</details>

<details>
<summary><strong>Q5.</strong> A click-trained ranker performs well offline and no better online. What could explain it?</summary>

Offline clicks come from the old ranker's slates and are biased by position, so offline evaluation rewards agreeing with the old order. Online, the new order is different, and items the old system never showed have no click history. Fix with propensity-aware evaluation and training, and a randomised slice that exposes unseen items.

</details>

<details>
<summary><strong>Q6.</strong> What changes for search compared with recommendations?</summary>

Search has a query, so retrieval must respect it (keyword and semantic retrieval from the query) and relevance to the query is non-negotiable; recommendations have no query and rely on user history and context. The funnel, the position bias and the cost arithmetic are the same.

</details>

## Further reading

- [Covington, Adams and Sargin, "Deep Neural Networks for YouTube Recommendations" (RecSys 2016)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/): candidate generation and ranking at scale.
- [Meta Engineering, Instagram Explore recommendations system (9 August 2023)](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/): the four-stage funnel, distillation and the value formula.
- [Google for Developers, recommendation systems overview](https://developers.google.com/machine-learning/recommendation/overview/types): candidate generation, scoring and re-ranking.
- [Haldar et al., "Applying Deep Learning To Airbnb Search"](https://arxiv.org/abs/1810.09591): moving from boosted trees to neural networks.
- [Joachims, Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback"](https://arxiv.org/abs/1608.04468): position bias and propensity weighting.
- [Eugene Yan, write-up on system design for discovery (June 2021)](https://eugeneyan.com/writing/system-design-for-discovery/): two-stage design with company examples.
- Site chapters: [recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval), [neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), [web search at scale](/docs/theory/ir/web-search-at-scale), [document Q&A design](/docs/senior/design-enterprise-document-qa).

## Check yourself

- I can draw the retrieve, light-rank, heavy-rank and re-rank funnel and say what each stage trades.
- I can compute the CPU and cores a funnel needs from items per stage and unit costs.
- I can explain why retrieval caps the system and how to measure the cap.
- I can explain why depth beyond a point adds cost without quality.
- I can say why the light ranker should imitate the heavy one.
- I can explain position bias, and compare face-value clicks, propensity weighting and randomised exploration.
- I can write the ranking cost formula and say which term dominates.
