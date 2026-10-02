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

## The idea in plain words

Imagine a matrix whose rows are users and columns are items. Its entries are ratings, interactions or missing values. If two people have liked many of the same lessons, a lesson known only to one may be a candidate for the other. If two lessons are often completed by the same learners, one may be a useful follow-up to the other. These are **neighbour methods**: find similar users or similar items, then transfer evidence. They can reveal useful associations that content tags missed, but their similarity estimates become unstable when overlap is sparse.

**Matrix factorisation** compresses the matrix. Instead of storing an independent prediction for every user-item pair, learn a short vector for each user and each item. Their dot product becomes a compatibility score. The dimensions are latent: a dimension may correlate with a topic or style, but its numerical axis is not guaranteed to have a clean human label. A low-dimensional model shares statistical strength across interactions, yet a new user or item has no reliable learned vector until there is evidence or a feature-based way to construct one.

<Infographic src="/img/recsys/matrix-factors.svg" alt="A sparse user-by-item matrix is represented by short user and item factor vectors; user (1,1) scores item A (2,1) as 3 and item B (0,2) as 2." caption="Factor scores are dot products; learning those factors requires an objective and feedback policy." />

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

## Designing with it

Start with the feedback type and serving role. If the task is explicit rating prediction for existing users and items, evaluate rating error and downstream list quality separately. If the task is implicit top-$K$ retrieval, do not use rating RMSE as the only success measure. Compare a factor model with popularity, content and neighbour baselines. Use chronological holdouts and remove already consumed or unavailable items before computing displayed-list metrics. Record the full candidate catalogue at the evaluation origin.

For scale, store and refresh item vectors in a retrieval index, while user vectors can be updated from recent history or learned in batch. Decide what happens if a user vector is absent, old or corrupted. Version vectors and index together; mixing model versions can make dot products meaningless. Monitor item coverage and new-item retrieval alongside aggregate click or purchase rates. A model that improves average engagement but never surfaces new catalogue entries may undermine discovery and future data quality.

## Where this stands in 2026

Collaborative filtering remains a foundational recommendation idea, even when modern encoders replace fixed identifier factors with neural or content-derived vectors. The key abstraction is still a user or query representation compared with item representations, followed by evaluation of a ranked list. The historical factor methods are valuable because they expose assumptions about missing data, confidence and popularity that remain relevant inside larger systems. They are also useful baselines when a complex two-tower model is proposed.

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

## Further reading

- [Google: collaborative filtering basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) introduces neighbour and embedding ideas.
- [Google: matrix factorisation](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) develops objectives and optimisation.
- [Hu, Koren and Volinsky](https://yifanhu.net/PUB/cf.pdf) gives the original confidence-weighted implicit formulation.
- [Recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) contains the worked neighbour example.

## Check yourself

- I can compute both a weighted-neighbour estimate and a factor dot product.
- I can explain why an implicit zero differs from an explicit negative rating.
- I can state what regularisation, confidence weighting and negative sampling each change.
- I can design cold-start and popularity baselines for a factor-based candidate source.
