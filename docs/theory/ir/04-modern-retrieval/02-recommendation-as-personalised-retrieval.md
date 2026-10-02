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

## The idea in plain words

Search usually begins with a query typed by a user. A recommender often begins with no explicit query at all. The user's interactions, profile or current context provide a signal of what might be useful. The system still has a retrieval problem: among many eligible items, which few should be shown, and in what order? This is why the lecture treats recommendation as **personalised information retrieval**.

**Collaborative filtering** finds patterns in a user-item interaction matrix. User-based methods look for people with similar histories; item-based methods find items that tend to be liked or used together. **Content-based** methods represent an item by its attributes, text or media and compare it with a user's profile. A hybrid can draw candidates from both. A newly added item has little interaction history, so collaborative methods face an item cold start; a content representation can still make it retrievable. A new user creates a different cold start, because there may be no known preferences yet.

The lecture's worked example predicts a rating from two neighbours. They rated an item **4** and **5**, with similarity weights **0.8** and **0.6**. The weighted mean is $(0.8\times4+0.6\times5)/(0.8+0.6)=4.43$ after rounding. This is a transparent prediction rule, not a complete recommender. It assumes ratings share a scale, the similarity weights are meaningful and positive, and the two neighbours represent enough evidence.

<Infographic src="/img/ir/recommender-methods.svg" alt="Collaborative filtering, content-based matching and a hybrid route support personalised retrieval; neighbour ratings four and five weighted by 0.8 and 0.6 predict 4.43." caption="The lecture's weighted-neighbour calculation sits inside a wider candidate and ranking system." />

:::note Beyond the lecture

The discussion of implicit feedback, exploration, exposure bias and staged ranking extends the lecture. The YouTube papers are historical production examples from 2016, not claims about its exact current recommendation algorithm.

:::

Move the two ratings or similarity weights. At the lecture defaults, the numerator is **6.2**, the total weight is **1.4**, and the predicted rating is **4.43**. The data table shows each neighbour's contribution separately.

<NeighbourRatingLab />

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

A companion [content-based related-video paper](https://research.google/pubs/content-based-related-video-recommendations/) examines new-video cold start. Co-watch patterns are weak for fresh uploads, so the authors use visual content features to represent a video and find related candidates. This supports the lecture's point that content features can help when collaborative evidence is absent. It does not mean content alone is always enough or that an untested new item should be shown to everyone.

An online learning library is a smaller example. Collaborative signals can suggest the next lesson taken by similar learners; content features can connect a new lesson to its prerequisite topics; an eligibility filter can prevent suggesting a lesson already completed. The ranking stage should consider both usefulness and learning sequence. A high click rate on a dramatic title may not mean the lesson helped the learner finish the course.

## Code you can run

The first block reproduces the lecture's weighted-neighbour prediction. The denominator must be nonzero; otherwise this fallback has no neighbours and cannot make a supported prediction.

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

The lecture says content-based methods have no new-item cold start, but that means only that an item with useful features can receive a score before interactions. A blank, misleading or very poor description remains a cold start for the content model. Also, a new user's profile may not exist even when every item is richly described. Distinguish item and user cold start in reports.

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

The feedback loop affects supply as well as demand. If popular items get all impressions, new creators or lessons cannot collect the data needed to become popular. A system should decide deliberately how much exploration and diversity it supports, then measure the resulting user outcome. The right balance depends on the product; it is not encoded in the lecture's weighted average.

### Connect to retrieval stages

Session 5's BM25 and vector representations can retrieve candidates for a recommender, particularly content-based ones. Session 7's ranked metrics can evaluate what was shown, with exposure caveats. Session 13's image-text embeddings can represent new visual items. The next chapter's dual-encoder and reranking pattern resembles large recommender pipelines, but a recommendation objective also includes user history and repeated interactions. These connections make the lecture's "history as query" idea operational while keeping the distinct evaluation risks visible.

## Where this stands in 2026

:::info Industry view

- Two-stage candidate generation and ranking is a documented production design in the historical YouTube paper and remains a useful general pattern for large catalogues.
- Content representations can give fresh items a route into candidate retrieval before collaborative signals accumulate, as the related-video paper demonstrates.
- Modern recommendation evaluation must account for exposure and feedback loops; a high offline interaction score alone does not settle the product decision.

:::

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

## Go deeper

- [YouTube recommendation system paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/); historical two-stage production design.
- [Content-based related-video paper](https://research.google/pubs/content-based-related-video-recommendations/); a new-item cold-start example.
- [Stanford IR book: text classification and clustering](https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html); neighbouring document-representation ideas.
- Built from the course lecture "ir-s14-recommender" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can calculate the lecture's 4.43 weighted-neighbour prediction and state its assumptions.
- [ ] I can distinguish user-based, item-based, content-based and hybrid candidates.
- [ ] I can describe user and item cold start separately and propose a fallback for each.
- [ ] I can explain why exposure bias limits offline interaction metrics.
