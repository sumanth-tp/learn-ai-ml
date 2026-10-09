---
title: "Recommenders · Collaborative filtering and factors"
sidebar_label: "Collaborative filtering"
sidebar_position: 2
slug: /theory/recsys/collaborative-filtering
description: "Neighbour methods, matrix factorisation and confidence-weighted implicit feedback."
tags: [recommender-systems, collaborative-filtering, matrix-factorisation]
---

import Infographic from '@site/src/components/Infographic';
import MatrixFactorLab from '@site/src/components/viz/MatrixFactorLab';

**In one line.** Collaborative filtering uses interaction patterns shared across users and items to suggest something a user has not already found.

:::tip Before you start
**You should already know**

- Feedback types and why a missing event is not a dislike: [feedback and objectives](/docs/theory/recsys/feedback-and-objectives).
- How to read precision at ten and why a time split is used: [feedback and objectives](/docs/theory/recsys/feedback-and-objectives) ran the same split.
- Cosine similarity of two vectors, as in [vector space and term weighting](/docs/theory/ir/vector-space-and-term-weighting).

**Reading time.** About 35 minutes, plus a few minutes to run the code.

**After this chapter you can**

- compute a user-neighbour score by hand and say why overlap matters,
- compare popularity, user-kNN, item-kNN and truncated SVD on held-out ranking metrics, including catalogue coverage,
- explain why adding factors can make a recommender worse.
:::

## In 30 seconds

Ask a friend who shares your taste for a film, and you are doing collaborative filtering. Find the people whose past choices overlap with yours, and borrow what they liked that you have not seen. Matrix factorisation does the same thing with a short list of numbers per person and per film instead of a whole friend group. The always-available competitor is "recommend what everyone watches". It is easy to beat, but not by as much as people expect, and it is the right yardstick for every fancier model.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Neighbour | A user (or item) whose history is most similar to the target | Ben is Ana's nearest neighbour |
| Cosine similarity | Angle between two history vectors, 1 for the same, 0 for no overlap | Ana and Ben: 0.816 |
| Latent factor | A learned hidden dimension in a short vector | A 10-number vector per film |
| Truncated SVD | A way to find the best low-rank approximation of a matrix | Keep the top 10 directions of the user-film matrix |
| Precision at 10 | Share of the top 10 that the user later liked | 1.3 of 10 on average is 0.13 |
| NDCG at 10 | Like precision, but a hit near the top counts more | A hit at rank 1 beats a hit at rank 9 |
| Catalogue coverage | Share of all items that appear in anybody's list | 0.043 means 4.3% of films ever show up |
| Popularity baseline | Recommend the most-consumed unseen items to everyone | The same 10 films for every user |

## The idea in plain words

Imagine a matrix whose rows are users and columns are items. Its entries are ratings, interactions or missing values. If two people have liked many of the same lessons, a lesson known only to one may be a candidate for the other. If two lessons are often completed by the same learners, one may be a useful follow-up to the other. These are **neighbour methods**: find similar users or similar items, then transfer evidence. They can reveal useful associations that content tags missed, but their similarity estimates become unstable when overlap is sparse.

**Matrix factorisation** compresses the matrix. Instead of storing an independent prediction for every user-item pair, learn a short vector for each user and each item. Their dot product becomes a compatibility score. The dimensions are latent: a dimension may correlate with a topic or style, but its numerical axis is not guaranteed to have a clean human label. A low-dimensional model shares statistical strength across interactions, yet a new user or item has no reliable learned vector until there is evidence or a feature-based way to construct one.

<Infographic src="/img/recsys/matrix-factors.svg" alt="Four cards explain a sparse interaction matrix, hand-set factor vectors with scores three and two, a training objective, and a cold-start fallback." caption="Factor scores are dot products; learning those factors requires an objective and feedback policy." />

## Worked example, step by step

Four films A, B, C, D and three users. Ana has seen A and B. Ben has seen A, B and C. Cat has seen C and D. We score Ana's unseen films by user-neighbour voting. The first block under "Code you can run" reproduces this.

1. Write each history as a 0 or 1 vector over (A, B, C, D): Ana (1, 1, 0, 0), Ben (1, 1, 1, 0), Cat (0, 0, 1, 1).
2. Cosine similarity is $\dfrac{x \cdot y}{\lVert x \rVert \, \lVert y \rVert}$. Ana and Ben share 2 films, so $x \cdot y = 2$. The lengths are $\sqrt{2} = 1.414$ and $\sqrt{3} = 1.732$, so the similarity is $2/(1.414 \times 1.732) = 2/2.449 = 0.816$.
3. Ana and Cat share nothing, so $x \cdot y = 0$ and the similarity is 0.
4. Each film's score is the sum of neighbour similarities that watched it. Film C: $0.816 \times 1 + 0 \times 1 = 0.816$. Film D: $0.816 \times 0 + 0 \times 1 = 0$.
5. Ana has seen A and B, so they are excluded. C scores 0.816 and D scores 0, and C is recommended.

In words: a film is recommended when people who resemble you watched it. Cat watched D but looks nothing like Ana, so D gets no vote. With very few shared films a similarity like 0.816 is fragile, which is why real systems require a minimum overlap or shrink the value.

## How it works

### Neighbour predictions

User-based collaborative filtering forms a neighbourhood around the target user from overlapping histories. A weighted rating estimate can be a similarity-weighted mean of neighbours' ratings. The earlier [Information Retrieval chapter](/docs/theory/ir/recommendation-as-personalised-retrieval) works through ratings four and five with similarities 0.8 and 0.6, producing $(0.8\cdot4+0.6\cdot5)/(0.8+0.6)=4.43$. That number is a prediction for a particular item under a small example, not a ranked catalogue. Similarity can be cosine, correlation or a domain-specific measure. If the denominator is zero, the system needs a fallback.

Users who have rated only one common item can appear perfectly similar by a naive measure. Require sufficient overlap or shrink similarity toward a prior. Centre ratings if one person systematically rates higher than another. Avoid comparing an old interaction from a different catalogue regime with a recent one without deciding whether recency matters. Item-based neighbours can be easier to cache when items are more stable than user histories. Neither neighbour method resolves new-item cold start without content or exploration.

### Factorise observed interactions

Let $U\in\mathbb R^{m\times d}$ contain user vectors and $V\in\mathbb R^{n\times d}$ item vectors. The pair score is $\hat r_{ui}=U_u\cdot V_i$. For an explicit-rating task, a basic objective minimises squared error over observed ratings plus regularisation, for example $\sum_{(u,i)\in\Omega}(r_{ui}-U_u\cdot V_i)^2+\lambda(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$. Only observed ratings appear in the data-fit term. User and item biases can model broad rating tendencies and popularity separately from the interaction factors. The regulariser limits extreme vectors in a sparse matrix.

For the lab's hand-set user vector $(1,1)$, item A $(2,1)$ scores $1\cdot2+1\cdot1=3$, while item B $(0,2)$ scores $2$. This only demonstrates the dot product. It does not claim those factors were learned, nor that a score of three is a three-star rating. If the first user factor increases, A's score rises faster than B's, which is exactly what the lab control shows.

The [Google matrix-factorisation guide](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) distinguishes objectives and optimisation choices. Stochastic gradient descent can update factors from sampled interactions; alternating least squares fixes one set of factors while solving for the other, then alternates. Both require a defined treatment of missing data. Explicit ratings are usually sparse because users choose what to rate; observed entries are not a random sample of every preference. Fit quality on observed ratings alone can overstate recommendation quality on unseen items.

### Implicit feedback needs a different loss

For implicit observations, “no event” is not an explicit negative preference. [Hu, Koren and Volinsky](https://yifanhu.net/PUB/cf.pdf) separate a binary preference from confidence. A simplified objective weights every user-item pair by $c_{ui}$ while fitting $p_{ui}$ with factor dot products. Observed repeated interactions have higher confidence; missing pairs have lower baseline confidence, not certainty of dislike. The full user-by-item space can be enormous, so implementations exploit the structure of the low baseline weight. A chosen confidence function shapes what the model learns; larger weight on frequent interactions may amplify exposure and popularity.

Other implicit objectives use pairwise comparisons: for a user, score an interacted item above a sampled item without an interaction. The sampled item is only a training contrast, not a proven negative. Sampling distributions matter. Uniform sampling may produce very easy negatives; popularity-based sampling may create harder but more biased comparisons. Evaluate the retrieval or ranking task directly instead of assuming a low training loss means good recommendations.

### Understand factor geometry and limits

Dot products reflect both vector angle and vector norm. An item with a large norm can score highly for many users, which can encode popularity. Cosine similarity removes magnitude and changes the ranking. Normalising vectors at serving time without matching the training objective can alter model behaviour. Factors also change under retraining: two equally good factorisations can rotate their axes without changing dot products. Avoid building a brittle rule that treats “dimension three” as a permanent semantic category.

Factorisation can be a candidate source rather than the final policy. It is weak when a new item has no interactions, when a user's short-term intent differs from their long history, or when eligibility changes rapidly. Add content candidates, session-based retrieval or a popularity fallback, then let a common ranker compare eligible options. If the same user sees only items from one inferred factor, the system can over-specialise and gather little evidence about other interests.

## A real system that works this way

The [Google recommendation course](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) presents matrix factors as a compact way to predict user-item compatibility. A learning site can use completed-lesson patterns to suggest a plausible next lesson among eligible options, while keeping prerequisite and completion filters outside the factor score. A fresh lesson should be introduced through metadata-based candidates because it lacks interaction factors. A historical [2016 Google Research paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) shows how large recommendation systems can separate candidate generation and deeper ranking; it does not establish that a particular matrix-factor model is YouTube's current implementation.

## Code you can run

```python
user = (1, 1)
items = {'A': (2, 1), 'B': (0, 2)}
scores = {name: sum(a * b for a, b in zip(user, vector)) for name, vector in items.items()}
print(scores)
print('ranked:', sorted(scores, key=scores.get, reverse=True))
```

This prints scores `{'A': 3, 'B': 2}` and ranking `['A', 'B']`.

<MatrixFactorLab />

Move the user's first factor. At the default $(1,1)$, the data table decomposes A's score into $2+1=3$ and B's into $0+2=2$. The vectors are fixed teaching values.

```python
ratings = [4, 5]
weights = [0.8, 0.6]
denominator = sum(weights)
prediction = sum(rating * weight for rating, weight in zip(ratings, weights)) / denominator
print('weighted neighbour rating:', round(prediction, 2))
```

This reproduces the introductory Information Retrieval chapter's 4.43 rating, linking the neighbour method to the factor method without treating them as the same algorithm.

### Experiment: neighbours and factors against popularity

This experiment reuses the MovieLens 100K download and the per-user time split from [the previous chapter](/docs/theory/recsys/feedback-and-objectives): each user's last 20% of ratings are held out and a held-out film counts as relevant if it was rated 4 or 5. Models see only the training interactions as 0 and 1, so this is the implicit setting. The data licence is described there: research use, acknowledgement, no redistribution, no commercial use without permission. The code downloads the archive when you run it.

Block one is the by-hand example. Block two compares six recommenders with precision at 10, NDCG at 10 and catalogue coverage, then splits users by how much history they have.

```python
import numpy as np

films = ['A', 'B', 'C', 'D']
ana = np.array([1, 1, 0, 0])
ben = np.array([1, 1, 1, 0])
cat = np.array([0, 0, 1, 1])
cosine = lambda x, y: x @ y / np.sqrt((x @ x) * (y @ y))
for name, other in (('Ben', ben), ('Cat', cat)):
    print(f'similarity Ana to {name}: {cosine(ana, other):.3f}')
sims = np.array([cosine(ana, ben), cosine(ana, cat)])
votes = sims @ np.array([ben, cat])
for film, v in zip(films, votes):
    print(film, 'neighbour score', round(float(v), 3), '(already seen)' if ana[films.index(film)] else '')
```

The printed similarities are 0.816 and 0.000, and film C scores 0.816 while D scores 0.0.

```python
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
CACHE = os.path.join(tempfile.gettempdir(), 'ml-100k.zip')
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
with zipfile.ZipFile(CACHE) as z:
    df = pd.read_csv(z.open('ml-100k/u.data'), sep='\t', names=['u', 'i', 'r', 't'])
df['u'] -= 1
df['i'] -= 1
n_users, n_items = df.u.max() + 1, df.i.max() + 1

df = df.sort_values(['u', 't'], kind='stable')
cut = (df.groupby('u').u.transform('size') * 0.8).astype(int)
test = df[df.groupby('u').cumcount() >= cut]
train = df.drop(test.index)
B = np.zeros((n_users, n_items))
B[train.u, train.i] = 1
seen = B > 0
liked = np.zeros((n_users, n_items), bool)
liked[test.u[test.r >= 4], test.i[test.r >= 4]] = True
users = np.flatnonzero(liked.any(1))
discount = 1 / np.log2(np.arange(2, 12))
ideal = np.array([discount[:min(10, liked[u].sum())].sum() for u in users])

def top_ten(S):
    return np.argsort(-np.where(seen, -np.inf, S), axis=1)[:, :10]

def evaluate(S):
    top = top_ten(S)
    hit = np.take_along_axis(liked, top, axis=1)[users]
    return hit.mean(), ((hit * discount).sum(1) / ideal).mean(), len(np.unique(top[users])) / n_items

def keep_top(sim, k):
    sim = sim.copy()
    np.fill_diagonal(sim, 0)
    keep = np.argpartition(-sim, k, axis=1)[:, :k]
    mask = np.zeros_like(sim)
    np.put_along_axis(mask, keep, 1, axis=1)
    return sim * mask

Bu, Bi = normalize(B), normalize(B, axis=0)
results = {'popularity': np.tile(B.sum(0), (n_users, 1))}
results['user-kNN k=50'] = keep_top(Bu @ Bu.T, 50) @ B
results['item-kNN k=50'] = B @ keep_top(Bi.T @ Bi, 50).T
for d in (10, 50, 200):
    svd = TruncatedSVD(d, random_state=0).fit(B)
    results[f'SVD d={d}'] = svd.transform(B) @ svd.components_
for name, S in results.items():
    p, n, c = evaluate(S)
    print(f'{name:15s} P@10 {p:.3f}  NDCG@10 {n:.3f}  coverage {c:.3f}')

history = B.sum(1)
groups = {'under 25 train items': history < 25, '25 to 59': (history >= 25) & (history < 60), '60 or more': history >= 60}
per_user = {name: np.take_along_axis(liked, top_ten(results[name]), axis=1).mean(1) for name in ('popularity', 'SVD d=10')}
for label, mask in groups.items():
    sel = mask & liked.any(1)
    print(f'{label:21s} users {sel.sum():3d}  popularity {per_user["popularity"][sel].mean():.3f}  SVD d=10 {per_user["SVD d=10"][sel].mean():.3f}')
```

**Reading the output.** Each row is one recommender. `P@10` is precision at 10 over users who have at least one liked held-out film. `NDCG@10` rewards early hits, with 1.0 meaning the ideal ordering of that user's liked films. `coverage` is the share of the 1,682 films that appears in at least one list. The last three rows split users by training history length and compare popularity with the 10-factor SVD.

**Line by line.**

- `keep_top` zeroes everything except each row's `k` largest similarities, which is what makes it a neighbourhood and not an average over everyone.
- `results['item-kNN k=50'] = B @ keep_top(...).T` scores an item by the similarity of its neighbours that the user has seen. `results['user-kNN k=50'] = keep_top(...) @ B` scores by what neighbouring users have seen.
- `TruncatedSVD(d)` reduces the matrix to `d` directions. `svd.transform(B) @ svd.components_` rebuilds a full score for every pair from only those directions.
- `np.where(seen, -np.inf, S)` stops a model recommending what the user already has.

The printed output was:

```text
popularity      P@10 0.079  NDCG@10 0.099  coverage 0.043
user-kNN k=50   P@10 0.126  NDCG@10 0.170  coverage 0.178
item-kNN k=50   P@10 0.121  NDCG@10 0.160  coverage 0.259
SVD d=10        P@10 0.126  NDCG@10 0.166  coverage 0.213
SVD d=50        P@10 0.120  NDCG@10 0.166  coverage 0.332
SVD d=200       P@10 0.076  NDCG@10 0.105  coverage 0.483
under 25 train items  users 196  popularity 0.039  SVD d=10 0.065
25 to 59              users 286  popularity 0.057  SVD d=10 0.098
60 or more            users 425  popularity 0.111  SVD d=10 0.173
```

**What the numbers say.** Popularity, which shows every user the same unseen top films, gets 0.079 precision at 10. User-kNN and the 10-factor SVD both reach 0.126, about 60% more, and item-kNN reaches 0.121. Those three are within a hair of each other, so on this dataset the choice between neighbour and factor models is a matter of serving cost, not accuracy. All three beat popularity for every history-length group, with gains from 0.039 to 0.065 for light users and from 0.111 to 0.173 for heavy ones. Popularity also covers only 4.3% of the catalogue against 21.3% for SVD with 10 factors.

The surprise is the last SVD row. With 200 factors the model can reproduce the training matrix almost exactly, so its scores for unseen films are noisy: precision falls to 0.076, below popularity, while coverage climbs to 0.483. High coverage did not mean good recommendations, because recommending more different films is easy if you recommend them badly. Limits: one split, no tuning of `k` or the factor count beyond three values, binary training data only, and a 1997 to 1998 catalogue. A tuned popularity-plus-recency baseline, or an ALS model, would change the margins.

<Infographic src="/img/recsys-enrich/cf-baselines.svg" alt="Bars show precision at 10 for popularity, two neighbour models and three SVD sizes; cards show the 200-factor failure, gains by history length and catalogue coverage." caption="Look first at the red SVD d=200 bar: it falls below the grey popularity bar." />

## Designing with it

### Work through an explicit-rating case

Imagine three users and three films. The first two users have both rated films A and B, while only the second has rated C. A user-neighbour method compares their common ratings, then uses the second user's C rating to estimate the first user's reaction. If the users use the rating scale differently, raw cosine or an uncentred weighted mean can mislead. One user may reserve five stars for rare favourites while another gives five stars to most acceptable films. Centring by user mean and shrinking similarities with little overlap are practical corrections. They do not remove selection bias: users rated the films they chose to watch, not a random sample of the catalogue.

An item-neighbour method reverses the view. If films A and C receive similar ratings from enough shared users, C can be suggested to someone who liked A. This relationship can be cached and explained as “people who liked A also liked C,” but it may be driven by popularity or a shared promotion rather than intrinsic similarity. A new film has few co-ratings, so content and editorial metadata remain useful. A hybrid candidate pool lets the ranker choose between an established interaction match and a fresh content match.

### Work through an implicit-confidence case

For an implicit matrix, suppose a user watched item A twice and never encountered item B. With $\alpha=2$, A has preference one and confidence five, while B has preference zero and confidence one under the simplified construction. The optimisation penalises an error on A more strongly, but B still contributes a low-confidence term. If B was shown many times and explicitly dismissed, it should probably be represented differently from a never-shown B. The simple count scheme cannot express that distinction by itself. Add event type and exposure information, or use a loss designed for the actual recommendation task.

Weighted alternating least squares exploits the factor structure to update all user vectors while holding item vectors fixed, then all item vectors while holding users fixed. This is a training strategy, not a guarantee that the final ranking serves the product objective. Choice of factor dimension controls capacity: too few dimensions can miss distinct tastes, while too many can memorise sparse events. Regularisation, confidence weight and dimension interact, so select them on chronological held-out ranking quality, not only reconstruction loss. A model can reconstruct frequent interactions very well and still be poor at discovering new items.

### Check geometry at serving time

Suppose user vector $(1,1)$ scores A $(2,1)$ as three and B $(0,2)$ as two. If A's vector is doubled, its dot score doubles even though its direction stays the same. This illustrates how norm can encode frequency or confidence as well as direction. A cosine search would change that relationship. If the retrieval index uses cosine but the model was trained and assessed with raw dot products, the served candidate order may differ. Record the similarity function and any normalisation at training, offline evaluation and serving. A vector index is not a neutral storage detail.

User factors can also become stale. A learner who recently switched from introductory maths to computer vision may be represented by a months-old average preference. Weighting recent interactions, using a session encoder or blending a current-item query with the long-term factor can help. Each creates a new evaluation task: does the system adapt to short-term intent without forgetting stable interests or overreacting to one accidental click? Segment results by history length and recency. New users need a fallback before any factor exists.

### Distinguish predictions from explanations

Latent dimensions are not reliably named concepts. Rotating all vectors in a factorisation can preserve dot products while changing each coordinate. A claim such as “factor two means science fiction” may be tempting after inspecting a few high-scoring items, but it is unstable across retraining and may be wrong for individual users. If a product needs explanations, use verifiable item attributes or a separately tested explanation layer. Also avoid saying a high factor score means a person will enjoy an item with certainty. It is a ranking signal conditioned on incomplete historical data and the previous exposure policy.

### Set a maintenance threshold

Compare the factor model with a content baseline, a popularity baseline and the existing production policy. Report candidate recall, final-slate quality, new-item exposure, latency and resource cost. If factorisation wins only on established users while content handles new items, a hybrid route is justified. If the gain is tiny and the embedding refresh system is fragile, a simpler model may be the better service. Write down the failure fallback and index-version policy before launch so an out-of-date vector does not silently produce plausible but degraded recommendations.

Keep the score's meaning explicit in that comparison: a factor dot product is a ranking signal until calibration and the exposure conditions behind its training labels have been examined.

Start with the feedback type and serving role. If the task is explicit rating prediction for existing users and items, evaluate rating error and downstream list quality separately. If the task is implicit top-$K$ retrieval, do not use rating RMSE as the only success measure. Compare a factor model with popularity, content and neighbour baselines. Use chronological holdouts and remove already consumed or unavailable items before computing displayed-list metrics. Record the full candidate catalogue at the evaluation origin.

For scale, store and refresh item vectors in a retrieval index, while user vectors can be updated from recent history or learned in batch. Decide what happens if a user vector is absent, old or corrupted. Version vectors and index together; mixing model versions can make dot products meaningless. Monitor item coverage and new-item retrieval alongside aggregate click or purchase rates. A model that improves average engagement but never surfaces new catalogue entries may undermine discovery and future data quality.

## Where this stands in 2026

Collaborative filtering remains a foundational recommendation idea, even when modern encoders replace fixed identifier factors with neural or content-derived vectors. The key abstraction is still a user or query representation compared with item representations, followed by evaluation of a ranked list. The historical factor methods are valuable because they expose assumptions about missing data, confidence and popularity that remain relevant inside larger systems. They are also useful baselines when a complex two-tower model is proposed.

## Common mistakes

1. **Skipping the popularity baseline.** It feels too simple to count. It costs one line and tells you whether a model earns its complexity. Here popularity got 0.079 against 0.126 for the best models, a real but modest margin.
2. **Raising the factor count to fit more.** More capacity looks like more accuracy. With 200 factors precision dropped to 0.076. Pick the dimension on held-out ranking, not on reconstruction.
3. **Judging by coverage alone.** A model that recommends nearly the whole catalogue looks diverse. The 200-factor SVD covered 0.483 and was worse than popularity. Read coverage next to precision.
4. **Computing similarity on tiny overlap.** Two users with one shared film can look identical. Require a minimum overlap or shrink similarities toward zero.
5. **Letting the model recommend what the user already has.** Mask training items before taking the top 10, or the model spends the list on films already seen.

## Practice questions

<details>
<summary>Why can two users with one shared rating appear misleadingly similar?</summary>

A similarity statistic can be extreme on tiny overlap. Require support, shrink toward a prior, or use a model that shares information more broadly.

</details>

<details>
<summary>What does the dot product of user (1,1) and item A (2,1) equal?</summary>

It is $1\cdot2+1\cdot1=3$. This is a compatibility score under the example's chosen vectors.

</details>

<details>
<summary>Why is every unobserved implicit item not a hard negative?</summary>

The user may never have seen it. An implicit objective should treat absence with limited confidence or use carefully designed sampled contrasts.

</details>

<details>
<summary>What changes if item factors are normalised to unit length only at serving time?</summary>

Dot-product rankings can change because norm information disappears. The serving score must match the intended training geometry and be evaluated end to end.

</details>

<details>
<summary><strong>Q5. (Easy)</strong> Ana has seen A and B, and Dev has seen only A. What is their cosine similarity?</summary>

The dot product is 1, Ana's length is $\sqrt{2}$ and Dev's is 1. The similarity is $1/\sqrt{2} = 0.707$. It is high even though they share one film, which is why overlap thresholds matter.

</details>

<details>
<summary><strong>Q6. (Medium)</strong> Popularity scored 0.079 and SVD with 10 factors 0.126. Is the SVD "60% better" a safe claim for your product?</summary>

The ratio is 0.126 / 0.079 = 1.59, so the arithmetic holds, but the claim belongs to this dataset, one split and these settings. Check it on your catalogue, on a time split, with a tuned popularity baseline, and with a confidence interval over users before quoting it.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> Why did SVD with 200 factors have the highest coverage and the lowest precision among the SVD sizes?</summary>

Many factors let the model memorise each user's training history, so scores for unseen films are close to noise and spread across the catalogue. Spread-out noise raises coverage and removes the signal that precision measures.

</details>

## Further reading

- [Google: collaborative filtering basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) introduces neighbour and embedding ideas.
- [Google: matrix factorisation](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) develops objectives and optimisation.
- [Hu, Koren and Volinsky](https://yifanhu.net/PUB/cf.pdf) gives the original confidence-weighted implicit formulation.
- [Recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) contains the worked neighbour example.
- [Dacrema, Cremonesi and Jannach, "Are We Really Making Much Progress?"](https://arxiv.org/abs/1907.06902) (RecSys 2019, opened 2026-10-09) tried to reproduce 18 recent neural recommenders; only 7 could be reproduced with reasonable effort, and six of those were often beaten by simpler nearest-neighbour or graph-based methods.
- [GroupLens MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) (opened 2026-10-08) carries the licence terms and the citation Harper and Konstan, ACM TiiS 5(4), 2015.
- scikit-learn 1.9.1 `TruncatedSVD` and NumPy 2.5.3 were the versions run for the experiment.

## Check yourself

- I can compute both a weighted-neighbour estimate and a factor dot product.
- I can explain why an implicit zero differs from an explicit negative rating.
- I can state what regularisation, confidence weighting and negative sampling each change.
- I can design cold-start and popularity baselines for a factor-based candidate source.
- I can compute a cosine similarity and a neighbour vote for a small history by hand.
- I can say why popularity must be measured first and report the margin of a personalised model over it.
- I can explain why SVD with too many factors got worse on held-out ranking while its coverage rose.

## Where to go next

Next is [retrieval, ranking and reranking](/docs/theory/recsys/retrieval-ranking-and-reranking), which turns these scores into a staged system. Go back to [feedback and objectives](/docs/theory/recsys/feedback-and-objectives) if the implicit-feedback assumptions here feel unfamiliar.
