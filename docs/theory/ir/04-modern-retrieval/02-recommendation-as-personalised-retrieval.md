---
id: ir-recommendation-personalised-retrieval
title: "Information Retrieval · Session 14 — Recommendation as Personalised Retrieval"
sidebar_label: "14 · Recommend"
sidebar_position: 2
slug: /theory/ir/recommendation-as-personalised-retrieval
description: "How collaborative and content-based recommendation treat a user's history as an implicit query, with a worked neighbour rating and cold-start design."
tags: [information-retrieval, recommender-systems, collaborative-filtering, cold-start]
---

import Infographic from '@site/src/components/Infographic';
import NeighbourRatingLab from '@site/src/components/viz/NeighbourRatingLab';

**In one line.** A recommender retrieves items for a particular user, using their history and context as an implicit query, then ranks candidates by expected value.

:::tip Before you start

**You should already know**

- How tf-idf weights rare terms more than common ones ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).
- Why a missing interaction is not a dislike ([Feedback and objectives](/docs/theory/recsys/feedback-and-objectives)).

**Reading time:** about 50 minutes, plus a few seconds to run the code (the first run downloads 5 MB).

**After this chapter you can**

- Treat a user's history as a search query and retrieve films with item-to-item similarity.
- Say how much history that query needs, and what inverse-frequency weighting does to it.
- Show that a brand-new item cannot be retrieved by interaction data, and measure what content features recover.

:::

## In 30 seconds

A search engine finds documents that match your words. A recommender has no words, only what you have watched, so it treats that list as the query. If you watched two films that many people watched together with a third, the third is a good candidate.

Search ideas carry over, and each is a trade. Weight rare films higher, as tf-idf weights rare words, and you recommend more of the catalogue. Use only your last film as the query and you get a weak result. A film nobody has rated yet has no one to be similar to, so it cannot be found at all until you describe it with content.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Implicit query | The user's history, used as if it were a typed query | The last 20 films watched |
| Item-item similarity | How often two items are used by the same people, scaled by popularity | Cosine of two film columns |
| Inverse user frequency | Idf for items: rare items count more | A film seen by 3 of 4 users scores 0.288 |
| Cold-start item | A new item with no interactions | A film added today |
| Content-based retrieval | Matching item features to a user profile | Genre flags |
| Precision at 10 (P@10) | Share of the top 10 that the user later liked | 0.131 |
| Coverage | Share of the catalogue that is ever recommended | 0.205 |

## The idea in plain words

Search usually begins with a query typed by a user. A recommender often begins with no explicit query at all. The user's interactions, profile or current context provide a signal of what might be useful. The system still has a retrieval problem: among many eligible items, which few should be shown, and in what order? This is why recommendation is treated as **personalised information retrieval**.

**Collaborative filtering** finds patterns in a user-item interaction matrix. User-based methods look for people with similar histories; item-based methods find items that tend to be liked or used together. **Content-based** methods represent an item by its attributes, text or media and compare it with a user's profile. A hybrid can draw candidates from both. A newly added item has little interaction history, so collaborative methods face an item cold start; a content representation can still make it retrievable. A new user creates a different cold start, because there may be no known preferences yet.

The worked example predicts a rating from two neighbours. They rated an item **4** and **5**, with similarity weights **0.8** and **0.6**. The weighted mean is $(0.8\times4+0.6\times5)/(0.8+0.6)=4.43$ after rounding. This is a transparent prediction rule, not a complete recommender. It assumes ratings share a scale, the similarity weights are meaningful and positive, and the two neighbours represent enough evidence.

<Infographic src="/img/ir/recommender-methods.svg" alt="Collaborative filtering, content-based matching and a hybrid route support personalised retrieval; neighbour ratings four and five weighted by 0.8 and 0.6 predict 4.43." caption="The weighted-neighbour calculation sits inside a wider candidate and ranking system." />

:::note Added for this site

The discussion of implicit feedback, exploration, exposure bias and staged ranking is an addition. The YouTube papers are historical production examples from 2016, not claims about its exact current recommendation algorithm.

:::

Move the two ratings or similarity weights. At the default settings, the numerator is **6.2**, the total weight is **1.4**, and the predicted rating is **4.43**. The data table shows each neighbour's contribution separately.

<NeighbourRatingLab />

## Worked example, step by step

Four users used four films. U1 saw A and B. U2 saw A, B and C. U3 saw B, C and D. U4 saw C and D. A new user has seen A and B, and we want to rank C and D.

1. **Counts.** Film A was seen by 2 users, B by 3, C by 3, D by 2. Pairs seen together: AB 2, AC 1, AD 0, BC 2, BD 1, CD 2.
2. **Item-item cosine** is the pair count over the square root of the product of the two counts. $\cos(A,C) = 1/\sqrt{2 \times 3} = 0.408$. $\cos(B,C) = 2/\sqrt{3 \times 3} = 0.667$. $\cos(A,D) = 0$. $\cos(B,D) = 1/\sqrt{3 \times 2} = 0.408$.
3. **The history is the query.** Score C as $0.408 + 0.667 = 1.075$ and D as $0 + 0.408 = 0.408$. Recommend C.
4. **Weight by rarity.** Inverse frequency is $\ln(4/\text{count})$: A and D get 0.693, B and C get 0.288. Now C scores $0.693 \times 0.408 + 0.288 \times 0.667 = 0.475$ and D scores $0.693 \times 0 + 0.288 \times 0.408 = 0.117$. C still wins, but the rare film A now carries most of the weight.
5. **A one-film query.** With only B as the query, C scores 0.667 and D 0.408. The order is the same here, but one film is a thin basis for most users, as the experiment below shows.

In words: each film in the history casts a vote for similar films, and weighting changes whose vote counts most.

<Infographic src="/img/ir-enrich/ir2-recsys.svg" alt="Left, precision at 10 against how many recent films form the query, for plain and inverse-frequency weighting, with the popularity baseline. Right, precision at 10 for cold films: random, genre content, and hindsight popularity." caption="Look first at the left curve: it peaks near the last 20 films at 0.131 and the two weightings almost coincide. On the right, genre content lifts cold films from 0.054 to 0.090." />

## How it works

### Similar users/items

CF uses the rating matrix: user-based (similar tastes) or item-based (similarly-rated items), with cosine/Pearson similarity and matrix factorisation. Cold-start hurts new users/items.

:::tip

**Worked.** ratings 4,5 with similarities 0.8,0.6 → (0.8·4+0.6·5)/(0.8+0.6) = 4.43.

:::

### Features & combining

Content-based matches item features to a user profile (tf-idf over a user model); no cold-start for new items but limited serendipity. Hybrid combines CF + content-based.


## A real system that works this way

Google Research's [2016 YouTube recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) describes a large system with two stages: candidate generation followed by a separate ranking model. This is a historical, published production example of the same retrieval distinction used in search. Candidate generation narrows a vast catalogue; a more detailed ranker decides which of those candidates to show. The paper does not specify YouTube's exact system in 2026.

A companion [content-based related-video paper](https://research.google/pubs/content-based-related-video-recommendations/) examines new-video cold start. Co-watch patterns are weak for fresh uploads, so the authors use visual content features to represent a video and find related candidates. This supports the point that content features can help when collaborative evidence is absent. It does not mean content alone is always enough or that an untested new item should be shown to everyone.

An online learning library is a smaller example. Collaborative signals can suggest the next lesson taken by similar learners; content features can connect a new lesson to its prerequisite topics; an eligibility filter can prevent suggesting a lesson already completed. The ranking stage should consider both usefulness and learning sequence. A high click rate on a dramatic title may not mean the lesson helped the learner finish the course.

## Code you can run

The first block reproduces the weighted-neighbour prediction. The denominator must be nonzero; otherwise this fallback has no neighbours and cannot make a supported prediction.

```python
ratings = [4, 5]
similarities = [0.8, 0.6]
weighted_total = sum(rating * weight for rating, weight in zip(ratings, similarities))
weight_total = sum(similarities)
prediction = weighted_total / weight_total
print(f"weighted total={weighted_total:.1f}; weight total={weight_total:.1f}")
print(f"predicted rating={prediction:.2f}")
assert round(prediction, 2) == 4.43
```

The second block shows a simple content-based fallback for a new item with no ratings. The item and user profile are hand-written topic vectors; the score is their cosine. This is a transparent teaching mechanism, not a trained embedding or an estimate of a real user's preference.

```python
from math import sqrt

user_profile = [3, 1, 0]
new_items = {"retrieval lesson": [2, 1, 0], "vision lesson": [0, 0, 2]}

def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)

scores = {name: cosine(user_profile, vector) for name, vector in new_items.items()}
print({name: round(score, 3) for name, score in scores.items()})
assert scores["retrieval lesson"] > scores["vision lesson"]
```

The new item is retrievable because it has content features despite no interaction history. Its high topic similarity does not establish that it is pedagogically right for the learner; prerequisites and outcomes still need a separate check.

### The worked example in code

This block reproduces steps 1 to 4.

```python
import numpy as np

seen = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=float)
counts = seen.sum(0)
sim = (seen.T @ seen) / np.sqrt(np.outer(counts, counts))
idf = np.log(4 / counts)
for name, w in (("plain", np.ones(4)), ("idf", idf)):
    score = (w[[0, 1]] @ sim[[0, 1]]).round(3)
    print(name, "C =", score[2], "D =", score[3])
```

**Reading the output.** The plain row prints C = 1.075 and D = 0.408. The idf row prints C = 0.475 and D = 0.117.

### An experiment: how much history does the query need?

Does the idf trick from search help recommendation, and how long should the implicit query be? The block uses MovieLens 100K (943 users, 1,682 films, ratings from 1997 and 1998). Each user's last 20% of ratings are held out, and a held-out film counts as liked if it was rated 4 or 5. The query is the user's most recent $m$ training films. Each query film votes for films with item-item cosine, optionally multiplied by its inverse user frequency $\ln(\text{users} / \text{film count})$, which is the adaptation to item queries of the "inverse user frequency" that Breese and colleagues (1998) applied to votes. Films the user has already rated are excluded. The data may be used for research with acknowledgement, must not be redistributed and must not be used commercially without permission; the code downloads it at run time. Versions used: Python 3.14.6, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1. The run takes about 2 seconds.

```python
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
CACHE = os.path.join(tempfile.gettempdir(), "ml-100k.zip")
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with zipfile.ZipFile(CACHE) as z:
    df = pd.read_csv(z.open("ml-100k/u.data"), sep="\t", names=["u", "i", "r", "t"])
df["u"] -= 1
df["i"] -= 1
n_users, n_items = df.u.max() + 1, df.i.max() + 1
df = df.sort_values(["u", "t"], kind="stable")
cut = (df.groupby("u").u.transform("size") * 0.8).astype(int)
test = df[df.groupby("u").cumcount() >= cut]
train = df.drop(test.index)

B = np.zeros((n_users, n_items))
B[train.u, train.i] = 1
liked = np.zeros((n_users, n_items), bool)
liked[test.u[test.r >= 4], test.i[test.r >= 4]] = True
users = np.flatnonzero(liked.any(1))
recent = {u: g.i.to_numpy()[::-1] for u, g in train.groupby("u")}

Bn = normalize(B, axis=0)
sim = Bn.T @ Bn
np.fill_diagonal(sim, 0)
idf = np.log(n_users / np.maximum(B.sum(0), 1))

def precision_at_10(history_length, power):
    hits, shown = 0, set()
    for u in users:
        query = recent[u][:history_length]
        weights = idf[query] ** power
        scores = weights @ sim[query]
        scores[B[u] > 0] = -np.inf
        top = np.argpartition(-scores, 10)[:10]
        hits += liked[u, top].sum()
        shown.update(top.tolist())
    return hits / (10 * len(users)), len(shown) / n_items

popularity = B.sum(0)
base = np.mean([liked[u, np.argsort(-np.where(B[u] > 0, -1, popularity))[:10]].mean() for u in users])
print(f"{len(users)} users, {n_items} films; popularity P@10 = {base:.3f}")
print(f"{'history used as query':<24}{'sum of similarities':>22}{'idf-weighted':>16}{'coverage (plain, idf)':>24}")
for length in (1, 3, 5, 10, 20, 50, 1000):
    plain, cover_plain = precision_at_10(length, 0)
    weighted, cover_idf = precision_at_10(length, 1)
    label = "all" if length == 1000 else f"last {length}"
    print(f"{label:<24}{plain:>22.3f}{weighted:>16.3f}{f'{cover_plain:.3f}, {cover_idf:.3f}':>24}")
```

The output of the run:

```text
907 users, 1682 films; popularity P@10 = 0.079
history used as query      sum of similarities    idf-weighted   coverage (plain, idf)
last 1                                   0.089           0.089            0.678, 0.678
last 3                                   0.122           0.115            0.425, 0.533
last 5                                   0.126           0.122            0.346, 0.451
last 10                                  0.129           0.126            0.261, 0.342
last 20                                  0.131           0.130            0.205, 0.269
last 50                                  0.127           0.127            0.162, 0.215
all                                      0.126           0.129            0.127, 0.182
```

**Reading the output.** Each row is a query length. The two accuracy columns are precision at 10 averaged over the 907 users with at least one liked held-out film. The coverage column gives the share of all 1,682 films that appear in anyone's top 10, for the plain and idf versions. The first line gives the popularity baseline, the same 10 unseen popular films for everyone.

**Line by line.**

- `recent[u] = g.i.to_numpy()[::-1]` lists a user's training films newest first, so `[:history_length]` is the most recent films.
- `weights @ sim[query]` adds up the similarity rows of the query films, scaled by their weight. With `power` 0 the weights are 1.
- `np.argpartition(-scores, 10)[:10]` picks the top 10 without sorting everything.

### An experiment: a film nobody has rated

How much can content recover for a new item? The second block hides 10% of the films that have at least 20 ratings (93 films) from training, which makes them cold. Every rating of those films becomes test data. It ranks only the cold films for each user who liked at least one, using genre flags: the user profile is the sum of the genre vectors of the films they liked, and the score is cosine. Item-item collaborative filtering has nothing to say about these films, and the block checks that.

```python
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
CACHE = os.path.join(tempfile.gettempdir(), "ml-100k.zip")
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with zipfile.ZipFile(CACHE) as z:
    df = pd.read_csv(z.open("ml-100k/u.data"), sep="\t", names=["u", "i", "r", "t"])
    films = pd.read_csv(z.open("ml-100k/u.item"), sep="|", header=None, encoding="latin-1")
df["u"] -= 1
df["i"] -= 1
n_users, n_items = df.u.max() + 1, df.i.max() + 1
genres = films.iloc[:, 5:].to_numpy(float)

rng = np.random.default_rng(0)
counts = df.groupby("i").size()
eligible = counts[counts >= 20].index.to_numpy()
cold = np.sort(rng.choice(eligible, len(eligible) // 10, replace=False))
is_cold = np.zeros(n_items, bool)
is_cold[cold] = True
warm = df[~df.i.isin(cold)]
cold_events = df[df.i.isin(cold)]

liked_warm = np.zeros((n_users, n_items))
liked_warm[warm.u[warm.r >= 4], warm.i[warm.r >= 4]] = 1
profile = normalize(liked_warm @ genres)
cold_liked = np.zeros((n_users, len(cold)), bool)
position = {item: k for k, item in enumerate(cold)}
for u, i, r in zip(cold_events.u, cold_events.i, cold_events.r):
    cold_liked[u, position[i]] = r >= 4
users = np.flatnonzero(cold_liked.any(1))
print(f"{len(cold)} cold films, {len(users)} users who liked at least one; chance that a random cold film is liked: {cold_liked[users].mean():.3f}")

content = profile @ normalize(genres[cold]).T
rng = np.random.default_rng(1)
random_scores = rng.random(content.shape)
true_popularity = cold_liked.sum(0)[None, :] * np.ones((n_users, 1))

def precision_at_10(scores):
    top = np.argsort(-scores[users], axis=1)[:, :10]
    return np.take_along_axis(cold_liked[users], top, axis=1).mean()

warm_matrix = np.zeros((n_users, n_items))
warm_matrix[warm.u, warm.i] = 1
similarity = normalize(warm_matrix, axis=0).T @ normalize(warm_matrix, axis=0)
print(f"item-item collaborative filtering: cold films with any non-zero similarity = {int((similarity[:, cold] > 0).any(0).sum())} of {len(cold)}")
print(f"random order       P@10 = {np.mean([precision_at_10(rng.random(content.shape)) for _ in range(20)]):.3f}")
print(f"genre content      P@10 = {precision_at_10(content):.3f}")
print(f"hindsight popular  P@10 = {precision_at_10(true_popularity):.3f}  (uses the very ratings it is graded on)")
```

```text
93 cold films, 862 users who liked at least one; chance that a random cold film is liked: 0.055
item-item collaborative filtering: cold films with any non-zero similarity = 0 of 93
random order       P@10 = 0.054
genre content      P@10 = 0.090
hindsight popular  P@10 = 0.196  (uses the very ratings it is graded on)
```

**Reading the output.** The first line gives the chance that a random cold film is liked, 0.055. The next line counts cold films with any non-zero similarity to a warm film. Then come precision at 10 for a random order, for genre content, and for a ranking by the films' own test popularity, which uses the ratings it is graded on and is only an upper reference.

### What the numbers say

The popularity baseline scored 0.079. A query of the single most recent film gave 0.089, hardly above it. Precision rose to 0.122 at the last 3 films, 0.129 at 10 and 0.131 at 20, then fell slightly to 0.127 at 50 and 0.126 for the whole history. More history was not better: the most recent 20 films did as well as or better than everything, probably because tastes drift, though this run does not isolate that cause.

Weighting by rarity did not help accuracy. At 3 films idf scored 0.115 against 0.122, at 20 it was 0.130 against 0.131, and with the whole history it was 0.129 against 0.126. These gaps of 0.007 or less are ties; I computed no intervals. It changed what was recommended: coverage with the whole history rose from 0.127 to 0.182, and at 3 films from 0.425 to 0.533. Search's most reliable trick bought breadth, not precision.

Cold films were invisible to collaborative filtering: 0 of the 93 had any non-zero similarity. Genre content lifted precision at 10 from 0.054 (random, close to the 0.055 chance rate) to 0.090. Ranking by hindsight popularity reached 0.196, which says a handful of early ratings would beat the genre guess, and that genres are a weak description of a film.

Limits: one time-split of one old dataset, no intervals, 19 coarse genre flags, a popularity baseline that uses the same history, and precision at 10 as the only accuracy measure.

## Designing with it

### Define the event and objective

An explicit rating records a user's stated preference on a scale. An implicit event such as a click, watch or purchase has a different meaning. A click may reflect curiosity, a misleading thumbnail or position in a list; an absence of click may mean the item was never displayed. Treating every interaction as a clean positive or every unobserved item as a negative creates biased training data. Define the product objective first: completion, satisfaction, discovery, revenue or some balance. The useful ranking depends on it.

For a course, the next lesson should be understandable and relevant, not merely likely to be clicked. For a marketplace, availability and policy eligibility are exact constraints. For news, source diversity and freshness can matter alongside immediate engagement. State such constraints separately from predicted preference so they can be audited and enforced.

### Choose candidate sources

User-based collaborative filtering can surface items liked by similar users, but similarity is unstable with sparse histories. Item-based methods use co-interaction patterns and can be efficient when item relationships are stable. Matrix factorisation learns low-dimensional user and item factors from interactions, improving generalisation but still struggling with new entities. Content-based methods can represent new items immediately, provided their features actually capture what the user values. A hybrid can union candidates from these routes before ranking.

| Candidate path | Signal | Good use | Weak point |
| --- | --- | --- | --- |
| Similar users | Shared interaction patterns | Discovering beyond a user's own item types | Sparse or noisy user histories |
| Similar items | Co-ratings or co-use | More of a known interest | New-item cold start |
| Matrix factors | Learned interaction structure | Broad catalogue patterns | Little evidence for new users/items |
| Content | Item features and user profile | Fresh items with descriptions | Over-specialisation and weak serendipity |

Candidate generation and final ranking are distinct. A content source may provide coverage for new items while a collaborative source provides highly relevant established items. The ranker can then consider context, diversity and calibrated relevance. If a candidate never enters the shortlist, the ranker cannot recover it; measure recall by item age and user segment.

### Handle cold start explicitly

A new user may need onboarding preferences, contextual defaults or a diverse exploration set. A new item needs an initial representation from text, metadata or media and a controlled chance to be shown so interaction evidence can accumulate. Popularity is a useful baseline when information is sparse, but if the system only shows popular items, new items never receive feedback. Exploration has a cost, so bound it and evaluate its benefit rather than scattering random items into every session.

Content-based methods are said to have no new-item cold start, but that means only that an item with useful features can receive a score before interactions. A blank, misleading or very poor description remains a cold start for the content model. Also, a new user's profile may not exist even when every item is richly described. Distinguish item and user cold start in reports.

### Evaluate recommendation as a ranking problem

Offline precision, recall and NDCG need a careful definition of relevance and candidate exposure. A held-out interaction is not proof that unclicked items were irrelevant. Temporal splits prevent using future behaviour to predict the past. Report performance by new users, new items and long-tail topics as well as an overall mean. An online test can assess whether a system improves the intended user outcome, with guardrails for diversity, complaints or fatigue.

Use feedback responsibly. A history of watched or bought items can reveal sensitive interests. Limit access, retention and unnecessary exposure. Give users understandable controls to correct poor suggestions. Do not assume a model's inferred profile is a fact about a person; it is a fallible prediction derived from behaviour.

## Work through the weighted prediction

Neighbour one rated the item 4 and has similarity 0.8 to the target user, contributing $0.8\times4=3.2$. Neighbour two rated it 5 with similarity 0.6, contributing $0.6\times5=3.0$. Their contributions sum to 6.2. The weights sum to 1.4, so the weighted mean is about 4.4286, displayed as 4.43. The lab exposes both contributions. Changing a similarity alters both the numerator and denominator; increasing the weight of the neighbour who rated 5 should pull the estimate upward.

The formula assumes positive weights and comparable ratings. Some users rate everything generously while others use low scores; a raw neighbour average does not correct those personal baselines. With only two neighbours, the estimate can also be unstable. A production model may centre ratings by user or item, include confidence from interaction count, or learn latent factors. This simple equation is valuable because it makes the source of a prediction visible, not because it settles the modelling choice.

### See the implicit query

Imagine a learner has completed an introductory retrieval lesson and repeatedly opens examples about search evaluation. No text query is entered, but the behaviour suggests an interest. Candidate sources might include lessons taken next by similar learners, lessons tagged with evaluation topics, and a curated path prerequisite map. The recommendation is personalised retrieval because the user's context selects and orders items from a catalogue. The context must be time-bounded: an old interest may not describe today's task.

An explicit search query and an implicit profile can also coexist. If the learner types `PageRank`, the system should respect that immediate request even if their history is mostly about neural networks. Personalisation can improve tie-breaking or filter completed material, but it should not silently replace a clear query with inferred interests. Evaluate both search and recommendation modes against their own user goals.

### Diagnose two cold starts

For a new user, collaborative filtering has no interactions to calculate neighbours or factors from. A brief preference selection, a contextual popular list or a diverse starter set can help. Once feedback arrives, the profile can adapt. For a new item, established users still exist, but the item has no co-interaction vector. Its title, description, media or metadata can place it in content-based candidate lists. A controlled exploration policy gives it exposure so collaborative evidence can eventually develop.

The two cold starts often occur together when a product launches. Then a simple content baseline and editorial rules may outperform a complex collaborative model trained on very little data. Do not claim a learned user-item matrix has predictive power before there are interactions. Measure coverage: how many users and items can each candidate source serve? A method can have a good mean score on active users while leaving new users with no results.

### Read the content fallback

The second code block gives the user profile weight three on a retrieval topic, one on a second topic and zero on vision. A newly described retrieval lesson has a vector pointing in a similar direction, while the vision lesson has no overlap. Cosine prefers the former. This toy representation is hand-designed, so its result is expected; it demonstrates that content can supply a score without ratings. A real representation might use text features or embeddings and must be checked for missing or misleading descriptions.

Content similarity tends to recommend more of what the user has already consumed. That can be useful for mastery but poor for discovery. A curriculum may intentionally introduce an adjacent new topic even if it is less similar to the learner's history. Diversity constraints, pathways and exploration can counter that effect. Measure whether recommendations lead to productive learning rather than only whether they resemble past clicks.

### Account for who was shown what

Suppose item A received many clicks and item B none. If A was shown at the top to thousands of users and B was never displayed, the raw counts tell little about relative preference. This is **exposure bias**. An offline evaluation based on historical interactions can reward a system for repeating yesterday's ranking. Record impressions and positions, use appropriate counterfactual or randomised evaluation where justified, and inspect long-tail coverage. Online experiments are valuable because they expose alternative items under controlled conditions, but they must monitor user cost and safety.

The feedback loop affects supply as well as demand. If popular items get all impressions, new creators or lessons cannot collect the data needed to become popular. A system should decide deliberately how much exploration and diversity it supports, then measure the resulting user outcome. The right balance depends on the product; it is not encoded in the weighted average.

### Connect to retrieval stages

Session 5's BM25 and vector representations can retrieve candidates for a recommender, particularly content-based ones. Session 7's ranked metrics can evaluate what was shown, with exposure caveats. Session 13's image-text embeddings can represent new visual items. The next chapter's dual-encoder and reranking pattern resembles large recommender pipelines, but a recommendation objective also includes user history and repeated interactions. These connections make the "history as query" idea operational while keeping the distinct evaluation risks visible.

## Where this stands in 2026

:::info Industry view

- Two-stage candidate generation and ranking is a documented production design in the historical YouTube paper and remains a useful general pattern for large catalogues.
- Content representations can give fresh items a route into candidate retrieval before collaborative signals accumulate, as the related-video paper demonstrates.
- Modern recommendation evaluation must account for exposure and feedback loops; a high offline interaction score alone does not settle the product decision.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Using the whole history as the query | More evidence should help | Precision peaked at the last 20 films (0.131) and was 0.126 for everything. Try recent windows |
| Using one item as the query | It is the simplest "more like this" | One film gave 0.089, close to the 0.079 popularity baseline |
| Assuming idf will raise accuracy because it does in search | It is the standard trick | It was a tie here (0.129 against 0.126). It did raise coverage from 0.127 to 0.182, which may be what you want |
| Expecting interactions to rank new items | The model is "collaborative" | The 93 cold films had no similarity at all. Give new items content features and a controlled chance to be shown |
| Trusting a hindsight popularity number | It is easy to compute | It uses the ratings being graded (0.196). Use it as a ceiling, never as a result |

## Practice questions

<details>
<summary><strong>Q1.</strong> How is a recommender system like/unlike IR?</summary>

It is personalised retrieval with no explicit query; the user's history/profile acts as the query.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast user-based and item-based collaborative filtering.</summary>

User-based recommends items liked by similar users; item-based recommends items rated similarly to ones the user liked.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Two neighbours rated an item 4 and 5, with similarities 0.8 and 0.6. Predict the rating.</summary>

(0.8·4 + 0.6·5)/(0.8+0.6) = 6.2/1.4 = 4.43 (similarity-weighted average).<br /><em>Session 14 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is the cold-start problem?</summary>

New users or items have no ratings, so collaborative filtering can't recommend for/of them; content-based or hybrid methods help.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What does content-based recommendation match, and its limitation?</summary>

It matches item features to the user's profile (no new-item cold-start) but has limited serendipity (recommends only similar items).<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) In the worked example, why does C beat D for the history A and B, and what does weighting by rarity change?</summary>

C is seen with both A and B (cosines 0.408 and 0.667), while D is seen with B only (0.408), so C collects more votes: 1.075 against 0.408. Weighting by inverse frequency multiplies A's vote by 0.693 and B's by 0.288, because A is seen by fewer users. C still wins, at 0.475 against 0.117, but the rare film A now determines the score.

</details>

<details>
<summary><strong>Q7.</strong> (Medium) Precision at 10 rose from 0.089 (last film) to 0.131 (last 20 films) and then fell to 0.126 (all films). Give two reasons and say which one the run can support.</summary>

More films give more votes, which reduces noise, so precision rises at first. Older films may describe tastes that have changed, and a long history is dominated by popular films that vote for popular candidates, so precision falls slightly. The run shows the rise and fall but does not separate these causes, so any explanation is a hypothesis. A test would hold the history length fixed and compare recent against old films.

</details>

<details>
<summary><strong>Q8.</strong> (Stretch) Genre content gave 0.090 against 0.054 random for cold films. A product manager asks whether to ship it. What do you say?</summary>

It is better than random by 0.036, which is real but small, and it is evaluated only on 93 films in a 1990s catalogue. The comparison that matters is against the alternative you will actually have: showing new items in a short exploration slot and learning from their first ratings, which this run suggests is more informative (0.196 for hindsight popularity, an upper bound). Ship genre content as a stop-gap for the first impressions and replace it as ratings arrive, and measure on your own catalogue before deciding.

</details>

## Go deeper

- [YouTube recommendation system paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/); historical two-stage production design.
- [Content-based related-video paper](https://research.google/pubs/content-based-related-video-recommendations/); a new-item cold-start example.
- [Stanford IR book: text classification and clustering](https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html); neighbouring document-representation ideas.
- [Breese, Heckerman and Kadie: Empirical Analysis of Predictive Algorithms for Collaborative Filtering](https://arxiv.org/abs/1301.7363) (UAI 1998, opened 2026-10-09); defines inverse user frequency as the idf analogue for votes.
- [GroupLens MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) (opened 2026-10-09); licence conditions and the citation Harper and Konstan, ACM TiiS 5(4), 2015.
- Built from the course lecture "ir-s14-recommender" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can calculate the 4.43 weighted-neighbour prediction and state its assumptions.
- [ ] I can distinguish user-based, item-based, content-based and hybrid candidates.
- [ ] I can describe user and item cold start separately and propose a fallback for each.
- [ ] I can explain why exposure bias limits offline interaction metrics.
- [ ] I can compute item-item cosines and rank candidates for a small history by hand, with and without inverse-frequency weights.
- [ ] I can say how long a history query should be and quote the measured peak near 20 films.
- [ ] I can explain why idf raised coverage but not precision in the experiment.
- [ ] I can show that a cold item has no collaborative signal and say what content recovered.

## Where to go next

Next: [Session 15, neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking). Related: [Retrieval, ranking and reranking for recommenders](/docs/theory/recsys/retrieval-ranking-and-reranking), which runs the full funnel on the same data.
