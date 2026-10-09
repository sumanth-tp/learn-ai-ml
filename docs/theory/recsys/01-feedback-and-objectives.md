---
title: "Recommenders · Feedback and objectives"
sidebar_label: "Feedback and objectives"
sidebar_position: 1
slug: /theory/recsys/feedback-and-objectives
description: "Explicit and implicit feedback, exposure, cold start and the objective behind a recommendation."
tags: [recommender-systems, implicit-feedback, product]
---

import Infographic from '@site/src/components/Infographic';
import FeedbackConfidenceLab from '@site/src/components/viz/FeedbackConfidenceLab';

**In one line.** A recommendation system ranks eligible items for a particular context from incomplete, biased evidence about what a user may value.

:::tip Before you start
**You should already know**

- What a matrix is and how to multiply two vectors (the [next chapter](/docs/theory/recsys/collaborative-filtering) uses both heavily).
- What precision and recall mean for a ranked list: [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval) covers them.
- Basic Python and NumPy.

**Reading time.** About 35 minutes, plus a few minutes to run the code.

**After this chapter you can**

- explain why the same log of events trains different models depending on whether you read it as ratings or as interactions,
- say why an unclicked item is not a recorded dislike, and correct a click rate for the position an item was shown in,
- run a rating-error and a ranking comparison on one dataset and say why they disagree.
:::

## In 30 seconds

Picture a shop assistant who watches which shelves customers walk past, pick up, buy and return. A rating card left at the till is rare but clear. Picking an item up is common but vague. A recommender is that assistant: it must guess what to put in front of a person next from clues of very different quality. The clues also depend on where things were placed. A product on the top shelf gets picked up more than the same product at the bottom, so a pick-up count says as much about the shelf as about the product. This chapter shows how to read those clues without fooling yourself.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Explicit feedback | A person states a preference | A 2-star rating |
| Implicit feedback | Behaviour from which a preference is guessed | A click, a watch, a purchase |
| Impression | An item was displayed to a person | A film tile shown on the home page |
| Confidence | How strongly the training objective counts an observation | 21 for a rated film, 1 for an unseen one |
| Exposure | Whether the person had a chance to see the item | Position 1 is seen by nearly everyone, position 10 by few |
| Position bias | Higher slots get more clicks regardless of quality | The same film gets 30 clicks at the top and 7.5 at slot 4 |
| Cold start | No history for a new user or new item | A film added today has no ratings |
| Propensity | The probability that an event could happen under the logging policy | Examination probability 0.25 at slot 4 |

## The idea in plain words

A search system often begins with a query typed by a user. A recommender frequently starts with a visit, a current item, a history or a task: “What should this person see next?” The candidate catalogue may contain courses, films, products, jobs or documents. A prediction of preference is only one input. The system must also respect availability, age suitability, prerequisites, duplicates, freshness and the size of the screen. A high-scoring item that cannot be delivered is not a useful recommendation.

Feedback comes in two broad forms. **Explicit feedback** is a stated preference, such as a rating, like or “not interested” control. It has a declared meaning, but only some users supply it and the scale may be used differently by different people. **Implicit feedback** is observed behaviour, such as an impression, click, watch, completion or purchase. It is plentiful but ambiguous. A click can express curiosity, accidental selection or attraction to a title; a long dwell time can mean interest or confusion. An unclicked item may never have been shown. The feedback label is not the user's true preference.

<Infographic src="/img/recsys/feedback-objective.svg" alt="A board separates explicit ratings, implicit interactions and new-user or new-item cold start; two interactions with confidence weight two produce confidence five." caption="Observed events need an interpretation, an exposure record and a product objective." />

## Worked example, step by step

Maya has rated two films and has never seen a third. The numbers below are small enough to do by hand, and the first block under "Code you can run" reproduces them.

1. Maya gave film A 5 stars and film B 2 stars. Film C was never shown to her.
2. **Explicit reading.** The model is trained to predict stars. It sees targets A = 5 and B = 2. Film C has no label, so it is left out of training. The model learns that Maya liked A and disliked B.
3. **Implicit reading.** The model is trained to predict whether an event happened. With the confidence rule $c = 1 + \alpha \cdot (\text{event count})$ and $\alpha = 20$ (the value the experiment below uses), both A and B have preference 1 and confidence $1 + 20 \cdot 1 = 21$. Film C has preference 0 and confidence 1. The 2-star film B now counts as a strong positive, because Maya chose to rate it.
4. **Position.** Suppose a film's true appeal is 0.30 (30 in 100 people who look at it click). At slot 1 everyone looks, so 100 impressions give $100 \times 1.0 \times 0.30 = 30$ clicks. At slot 4 only a quarter look, so 100 impressions give $100 \times 0.25 \times 0.30 = 7.5$ clicks.
5. The naive click rate is $30/100 = 0.300$ against $7.5/100 = 0.075$, a four-fold difference between identical films. Dividing clicks by the exposure each impression carried gives $30/100 = 0.300$ and $7.5/25 = 0.300$, so the two films agree again.

In words: the label you train on is a choice about how to read the log, and a click count is a product of appeal and visibility.

## How it works

### State the product objective

Before choosing an algorithm, decide what the list should help a person accomplish. A course site may favour sustained learning and prerequisite fit over a short-lived click. A shopping service may care about useful purchases and returns, not clicks alone. A video product may optimise time spent, but should inspect satisfaction, fatigue and diversity. A workplace knowledge tool may prioritise task completion and trustworthy sources. Different outcomes can disagree, so record a primary objective and guardrails rather than hiding a policy inside a training label.

The display surface changes the task. A “related item” carousel has the current item as a strong query; a homepage has a broad user context; a new-user page may have no personal history. Candidate generation for each surface can differ. The same trained user factor may be relevant in one context and irrelevant in another. Decide whether to recommend from all items, only currently eligible items, or a smaller editorial collection. Ineligible items should be filtered at a defined stage, and the evaluation denominator should reflect the actual opportunity set.

### Read implicit evidence carefully

An interaction matrix $r_{ui}$ may contain counts for user $u$ and item $i$. One simple implicit-feedback construction sets preference $p_{ui}=1$ when $r_{ui}>0$ and $0$ otherwise, with confidence $c_{ui}=1+\alpha r_{ui}$. For count two and $\alpha=2$, preference is one and confidence is five. This is the lab's default and a simplified form of the confidence idea in [Hu, Koren and Volinsky's original work](https://yifanhu.net/PUB/cf.pdf). The formula does not prove that the user likes the item five times as much. It says the optimiser gives a repeated interaction more weight under a chosen scheme. The paper discusses confidence as distinct from binary preference because missing observations are uncertain.

Event weighting needs care. A purchase usually carries more commitment than a click, but a purchase may be for someone else and may not predict another purchase of the same item. Repeated views could indicate strong interest or a video that autoplayed. A negative explicit action should be treated differently from a missing event. Deduplicate noisy events, define session boundaries and preserve the type and timestamp. Do not flatten all behaviour into one positive flag without measuring what is lost.

An **impression** records that an item was displayed to the user. It is not perfect proof that the user noticed it, but it is essential context. A missing interaction for an exposed item differs from a missing interaction for an item never shown. Position matters too: the first item is more likely to be seen than the tenth. If only clicked items are logged, a dataset hides the system's earlier selection policy. A recommender trained on that dataset may learn the previous system's exposure patterns rather than users' full interests.

### Handle cold start as separate cases

A new user has little or no personal history. Possible starting points include stated interests, session context, language, location where appropriate, broad popularity and a diverse starter set. Each is a fallback, not a substitute for learning a preference after valid interactions arrive. A new item has no interaction evidence. Content features such as title, topic, creator, price or media embeddings can make it retrievable before collaborative signals accumulate. A new item with poor metadata still has a content cold start. New users and new items require different tests and may need different serving paths.

Exploration gives less-known items a controlled opportunity to collect evidence. It has an opportunity cost because some exploratory items will perform worse immediately. Keep exposure probability and selection policy in logs so later analysis can distinguish “never tried” from “tried and rejected.” Exploration should be bounded by eligibility and quality rules. A popularity-only fallback can work for a first session but, if left as the permanent policy, can starve the long tail and reinforce the existing catalogue winners.

### Define what a score means

A rating-prediction model estimates a stated numeric rating under a selection process. A click-through model estimates a click conditional on an impression, placement and context. A purchase model estimates a different event over a different horizon. A composite score may blend these with business rules. Do not compare raw scores from two candidate sources unless they are calibrated or passed to a common ranker. A dot product from one embedding model and cosine from another have no shared unit merely because both are numbers.

The task is also constrained by privacy and data governance. An interaction history can reveal sensitive interests. Collect only data needed for the feature and evaluation plan, limit retention, protect identifiers and provide controls to correct or reset a preference where the product calls for them. Avoid treating an inferred profile as a fact about a person. The model is a fallible estimate from behaviour shaped by the interface.

## A real system that works this way

The [Google recommendation course](https://developers.google.com/machine-learning/recommendation/overview/types) describes a candidate-generation, scoring and reranking architecture. A course website could use that pattern: retrieve lessons from a learner's completed topics and current course, rank them for likely educational value, then exclude already-completed lessons and enforce prerequisites. A “clicked lesson” label alone would overvalue catchy titles. The product should observe completion, comprehension or return visits where feasible, and show a transparent path for a learner who wants to choose a different topic. The [Information Retrieval lecture on personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) gives the earlier neighbour-rating example; this track extends it to feedback design and production stages.

## Code you can run

```python
interaction_count = 2
alpha = 2
preference = int(interaction_count > 0)
confidence = 1 + alpha * interaction_count
print('preference:', preference)
print('confidence:', confidence)
```

The output is preference `1` and confidence `5`. These values describe a chosen training representation, not a measured psychological state.

<FeedbackConfidenceLab />

Change the event count and confidence weight. At zero events the binary preference is zero, while the lab labels that case “unknown preference.” The default two interactions and alpha two reproduce the code's confidence five.

```python
events = [
    {'item': 'A', 'impressed': True, 'clicked': False},
    {'item': 'B', 'impressed': True, 'clicked': True},
    {'item': 'C', 'impressed': False, 'clicked': False},
]
for event in events:
    status = 'positive interaction' if event['clicked'] else ('exposed without click' if event['impressed'] else 'unobserved')
    print(event['item'], status)
```

Item A and item C have the same click value but different exposure evidence. Even item A's non-click is not a definitive dislike; placement and attention still matter.

### Experiment: one dataset, two ways to read it

The data are MovieLens 100K from GroupLens: 100,000 ratings from 943 users on 1,682 films, collected from September 1997 to April 1998.

:::note About the data
The GroupLens README (opened 2026-10-08) allows use for research purposes. It forbids stating or implying endorsement, requires acknowledgement and the citation Harper and Konstan (2015), forbids redistribution without permission and forbids commercial use without permission. These notes are for study, and the data are not copied into the site: the code below downloads the archive from GroupLens when you run it and caches it in your temporary folder. Later chapters reuse the same download.
:::

Block one reproduces the worked example by hand-sized arithmetic.

```python
alpha = 20
events = {'A': 5, 'B': 2, 'C': None}
for film, stars in events.items():
    if stars is None:
        print(film, 'explicit: no label | implicit: p=0, c=1')
    else:
        print(film, f'explicit target {stars} | implicit: p=1, c={1 + alpha}')

appeal = 0.30
for position, examine in ((1, 1.0), (4, 0.25)):
    clicks = 100 * examine * appeal
    print(f'position {position}: {clicks:.1f} clicks per 100 impressions, naive rate {clicks / 100:.3f}, per exposure {clicks / (100 * examine):.3f}')
```

The output lists A and B with implicit confidence 21 and C with confidence 1, then 30.0 and 7.5 clicks per 100 impressions with identical rates once exposure is divided out.

The second block trains two matrix-factorisation models on the same events. Each user's last 20% of ratings are held out, so the test is "what will this person rate next". The explicit model predicts stars with user and item biases plus 20 factors. The implicit model treats every rated item as a positive with confidence 21 and every other item as a weak zero. Both are alternating least squares written in NumPy. Items a user already rated are never recommended, and a held-out item counts as relevant if it was rated 4 or 5.

```python
import os
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

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
R = np.zeros((n_users, n_items))
R[train.u, train.i] = train.r
seen = R > 0

def als(rows_of, n_rows, n_cols, solve, k=20, iters=12):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.1, (n_rows, k))
    Y = rng.normal(0, 0.1, (n_cols, k))
    for _ in range(iters):
        for a in range(n_rows):
            X[a] = solve(Y, rows_of[0][a], a, True)
        for b in range(n_cols):
            Y[b] = solve(X, rows_of[1][b], b, False)
    return X, Y

mu = train.r.mean()
cnt_i = seen.sum(0)
bi = np.where(seen, R - mu, 0).sum(0) / (cnt_i + 10)
bu = np.where(seen, R - mu - bi, 0).sum(1) / (seen.sum(1) + 10)
resid = np.where(seen, R - mu - bi - bu[:, None], 0)
by_user = [np.flatnonzero(seen[u]) for u in range(n_users)]
by_item = [np.flatnonzero(seen[:, i]) for i in range(n_items)]

def explicit_solve(Y, idx, a, is_user):
    if len(idx) == 0:
        return np.zeros(Y.shape[1])
    target = resid[a, idx] if is_user else resid[idx, a]
    return np.linalg.solve(Y[idx].T @ Y[idx] + 8.0 * len(idx) * np.eye(Y.shape[1]), Y[idx].T @ target)

def implicit_solve(Y, idx, a, is_user, alpha=20.0, lam=10.0):
    gram = Y.T @ Y + lam * np.eye(Y.shape[1])
    return np.linalg.solve(gram + alpha * Y[idx].T @ Y[idx], (1 + alpha) * Y[idx].sum(0))

Ue, Ve = als((by_user, by_item), n_users, n_items, explicit_solve)
base = mu + bu[:, None] + bi[None, :]
rmse = np.sqrt(np.mean((np.clip(base[test.u, test.i] + np.einsum('ij,ij->i', Ue[test.u], Ve[test.i]), 1, 5) - test.r) ** 2))
item_mean = np.where(cnt_i > 0, R.sum(0) / np.maximum(cnt_i, 1), mu)
print('RMSE explicit ALS:', round(rmse, 3), '| item-mean baseline:', round(np.sqrt(np.mean((item_mean[test.i] - test.r) ** 2)), 3))

Ui, Vi = als((by_user, by_item), n_users, n_items, implicit_solve)
scores = {
    'popularity': np.tile(cnt_i.astype(float), (n_users, 1)),
    'raw item mean rating': np.tile(item_mean, (n_users, 1)),
    'explicit ALS': base + Ue @ Ve.T,
    'implicit ALS': Ui @ Vi.T,
}
liked = np.zeros((n_users, n_items), bool)
liked[test.u[test.r >= 4], test.i[test.r >= 4]] = True
users = np.flatnonzero(liked.any(1))
for name, S in scores.items():
    top = np.argsort(-np.where(seen, -np.inf, S), axis=1)[:, :10]
    precision = np.take_along_axis(liked, top, axis=1)[users].mean()
    coverage = len(np.unique(top[users])) / n_items
    print(f'{name:20s} precision@10 {precision:.3f}  catalogue coverage {coverage:.3f}')
```

**Reading the output.** `RMSE explicit ALS` is the average star error on the held-out ratings, so a lower value is better. `precision@10` is the share of the ten recommended items that the user later rated 4 or 5. `catalogue coverage` is the share of all 1,682 films that appears in at least one user's list.

**Line by line.**

- `cut` and `cumcount` split by time inside each user, so no future rating trains the model.
- The bias terms `bi` and `bu` are shrunk with a count of 10 added to the denominator. Without that shrinkage an item rated once by one person with 5 stars would look perfect.
- `explicit_solve` fits only the observed entries and scales the regulariser by how many each row has. `implicit_solve` uses the confidence trick: it solves for the whole row of zeros and ones by adding a correction only for the items the person interacted with.
- `raw item mean rating` is the tempting shortcut of ranking films by their average stars.

Run on 2026-10-09 with Python 3.14.6, NumPy 2.5.3, pandas 2.3.3, the printed output was:

```text
RMSE explicit ALS: 0.999 | item-mean baseline: 1.074
popularity           precision@10 0.079  catalogue coverage 0.043
raw item mean rating precision@10 0.001  catalogue coverage 0.007
explicit ALS         precision@10 0.049  catalogue coverage 0.032
implicit ALS         precision@10 0.100  catalogue coverage 0.432
```

**What the numbers say.** The explicit model is the better predictor of stars (RMSE 0.999 against 1.074 for the item-average baseline), yet as a recommender it is worse than counting popularity: precision at 10 is 0.049 against 0.079. Ranking films by raw average stars is nearly useless, 0.001, because the highest averages belong to obscure films with a handful of 5-star ratings. The implicit model, which never sees a star, wins with 0.100 and covers 0.432 of the catalogue where popularity covers 0.043.

The surprise is that the model with the lowest rating error is not the model that finds the next film. The reason is the question each model answers. An explicit model asks "how many stars would this person give if they watched it". The test asks "what will this person watch and like next", and what people watch next is driven by what they were shown and what is popular. Limits: one run and one split, hyperparameters (20 factors, regulariser 8, alpha 20) were not tuned, a 1997 to 1998 film site is not a modern feed, and the test counts only films the user chose to rate, which favours a model of choice. Do not read 0.100 against 0.079 as a general verdict on implicit models.

### Experiment: exposure and position in a simulated log

Real position logs are not in MovieLens, so this block simulates one with known truth. Each of 200 items has a hidden appeal. An old policy ranks 40 random candidates per request by a noisy score correlated with appeal and shows the top 10. A user examines slot $k$ with probability $1/k$ and clicks an examined item with probability equal to its appeal. We then estimate appeal two ways.

```python
import numpy as np
from scipy.stats import spearmanr

rng = np.random.default_rng(7)
n_items, n_requests, slate, pool = 200, 30000, 10, 40
appeal = rng.beta(2, 8, n_items)
old_score = 0.5 * (appeal - appeal.mean()) / appeal.std() + rng.normal(0, 1, n_items)
examine = 1 / np.arange(1, slate + 1)

impressions = np.zeros(n_items)
clicks = np.zeros(n_items)
exposure = np.zeros(n_items)
for _ in range(n_requests):
    candidates = rng.choice(n_items, pool, replace=False)
    shown = candidates[np.argsort(-old_score[candidates])[:slate]]
    clicked = (rng.random(slate) < examine) & (rng.random(slate) < appeal[shown])
    impressions[shown] += 1
    clicks[shown] += clicked
    exposure[shown] += examine

enough = impressions >= 30
naive = clicks[enough] / impressions[enough]
debiased = clicks[enough] / exposure[enough]
print('items never shown:', int((impressions == 0).sum()), 'of', n_items)
print('items with at least 30 impressions:', int(enough.sum()))
print('Spearman, naive click rate vs true appeal:', round(spearmanr(naive, appeal[enough])[0], 3))
print('Spearman, exposure-corrected vs true appeal:', round(spearmanr(debiased, appeal[enough])[0], 3))
best = set(np.argsort(-appeal)[:20])
ids = np.flatnonzero(enough)
print('true top 20 recovered, naive:', len(best & set(ids[np.argsort(-naive)[:20]])), '| corrected:', len(best & set(ids[np.argsort(-debiased)[:20]])))
print('true top 20 items that were never shown:', sum(impressions[i] == 0 for i in best))
```

**Reading the output.** The Spearman values are rank correlations between an item's estimated and true appeal (1.0 is a perfect ordering). `true top 20 recovered` counts how many of the 20 truly best items appear in the estimator's top 20 among items with at least 30 impressions.

**Line by line.**

- `examine = 1 / np.arange(1, slate + 1)` is the position model, and the corrected estimator divides by it. In a real log this probability must itself be estimated, which is the hard part.
- `exposure[shown] += examine` accumulates the examination probability, not the impression count, as the denominator.

The printed output was:

```text
items never shown: 100 of 200
items with at least 30 impressions: 83
Spearman, naive click rate vs true appeal: 0.787
Spearman, exposure-corrected vs true appeal: 0.931
true top 20 recovered, naive: 12 | corrected: 15
true top 20 items that were never shown: 3
```

Dividing clicks by exposure lifts the rank correlation from 0.787 to 0.931 and recovers 15 of the true top 20 instead of 12. The honest part is the last two lines: half the catalogue was never shown, and three truly top-20 items were among them. A correction can repair how well you estimate what you saw. It cannot score an item that had no impressions. This is a single seeded simulation with a position model I wrote, so the sizes of the gains are illustrative.

<Infographic src="/img/recsys-enrich/explicit-vs-implicit.svg" alt="Bars compare precision at 10 for popularity, raw item mean, explicit and implicit factor models, with cards for the rating error, position-bias correction and items never shown." caption="Look first at the bars: the explicit model has the lower rating error but ranks below popularity, and the implicit model wins." />

## Designing with it

### An interaction is conditional on opportunity

Consider three learners and one advanced lesson. Learner A saw it in the first position and clicked. Learner B saw it below the fold and did not click. Learner C never had the lesson in their candidate set. Encoding A as one and both B and C as zero discards the opportunity difference. B's non-click is weak evidence because the item may not have been noticed; C's zero is no evidence about reaction to that item. A clean log contains request context, candidate sources, eligible catalogue, shown slate, position, impression timestamp and later action. It also needs a consistent identity policy across devices and sessions so that duplicate views are not mistaken for several independent endorsements.

The same behaviour can mean different things on different surfaces. Repeatedly opening a troubleshooting article may mean it is helpful or that it fails to solve the problem. Watching a tutorial to the end may reflect value, but a required training video can be completed without enthusiasm. Returning an item after purchase changes the interpretation of a purchase label. Choose a label in the context of the user's task, and inspect examples with domain experts before treating it as an objective truth. A more complex model trained on a poor label can optimise the wrong behaviour more efficiently.

### Compare objectives on a concrete slate

Suppose a course homepage can show two lessons. One short, sensational lesson is likely to get a click but rarely leads to completion. A foundational lesson gets fewer immediate clicks but helps learners finish a module. A click model may put the first lesson at the top; a completion model may choose the second. A blended score can represent a product trade-off, but the weights need evidence and ownership. Before experimenting, decide whether completion is the primary outcome and whether click rate is a diagnostic, or vice versa. The measured effect may also depend on learner stage: a new learner needs orientation while an experienced learner may value novelty.

Hard constraints are different from score preferences. A lesson behind a paywall, in a language the user did not select, or requiring an unmet prerequisite may be ineligible even if its predicted click rate is high. Filter or enforce these constraints explicitly. A soft bonus for topic diversity can be tuned; a hard prerequisite should not be traded away because an engagement score is large. Record why an item was filtered so the system can explain a missing recommendation and operators can diagnose unexpected catalogue gaps.

### Cold start is a measurement problem too

A new item cannot collect interactions unless it is shown. A recommender that relies only on historic interaction counts may assign it no score and never show it, then infer from zero interactions that users do not want it. Content-based retrieval can place it in plausible candidate sets; a bounded exploration policy can provide initial exposure. Measure early-item performance by launch cohort and age, not only by all-time totals. Similarly, a new user fallback should be evaluated on actual first sessions rather than on existing users after their histories are artificially hidden, because first-session context and selection can differ.

Popularity is an honest baseline if its limitations are stated. It often works well where many users share interests and labels are scarce. It can also create a feedback loop and a narrow catalogue. Compare it with personalised methods on the same eligible set, and report both immediate engagement and distribution of exposure. If personalisation's gain is small, its additional data collection, privacy cost and serving complexity may not be justified.

### Preserve user agency

An inferred profile should be revisable. A user may buy a gift, research a topic for work or change interests. A “not interested” control is a direct signal that deserves distinct handling, and a reset option can help where long histories trap users in old categories. Show enough context that a person understands why an item appears when the product supports explanation. These controls can improve both experience and data quality, but their events should be logged separately from passive non-clicks. Treating every absence of interaction as rejection undermines that distinction.

Create an event dictionary before training. For each event specify how it is emitted, when it becomes available, what exposure it implies, what it does not imply and which user controls can reverse it. Define a label horizon: a click within ten minutes and a purchase within seven days are not interchangeable targets. De-duplicate bot or retry events, and audit whether the instrumentation changed during the training window. A model trained across a logging migration can mistake the migration for a change in preference.

Separate user cold start, item cold start and both-new cases in evaluation. A random split of existing interactions rarely tests either case. Use a chronological split and a held-out entity slice where needed. Include an eligibility snapshot so a model is not penalised for failing to recommend an item that was unavailable. The evaluation chapter develops ranking metrics and exposure limitations in detail.

Keep a simple baseline, such as eligible popularity conditioned on locale or topic, and measure it on the same surface. The goal is a better experience, not a more complex model. If the baseline wins, examine label quality, exposure bias, stale inventory and whether the intended outcome differs from the optimised event.

## Where this stands in 2026

Modern recommenders can combine behaviour, item content and contextual features, yet explicit and implicit evidence remain different kinds of data. Current production designs still separate broad candidate retrieval from richer scoring and policy-aware reranking. An interaction model has to coexist with privacy choices, new users and newly published items. The most durable design decision is to make the objective, exposure and fallback paths observable.

## Common mistakes

1. **Reading a missing event as a dislike.** It feels natural because the zero sits in the same matrix as the ones. Most zeros are unseen items. Keep impressions and give unobserved pairs low confidence rather than a negative label.
2. **Choosing the model with the lowest RMSE.** Rating error rewards accurate stars, and the product shows a ranked list. In the experiment the lower-error model ranked worse than popularity. Pick the metric from the surface.
3. **Comparing click rates across positions.** A rate per impression mixes appeal with visibility. Log the slot and correct for it, or compare only at equal positions.
4. **Ranking by raw average rating.** Items with two ratings dominate the top. Shrink toward the global mean or require support.
5. **Treating confidence as a probability or a star count.** Confidence 21 is a training weight. It says how hard the loss pushes on that cell, nothing about how much the person liked the film.

## Practice questions

<details>
<summary>Why is an unclicked item not automatically a negative example?</summary>

It may never have been displayed. Even when displayed, position, attention and context affect the chance of a click. Record impressions and interpret non-clicks conditionally.

</details>

<details>
<summary>What is the difference between an explicit rating and an implicit watch?</summary>

A rating is a declared judgement on a scale; a watch is behaviour from which interest is inferred. Both are selected observations and need their collection context.

</details>

<details>
<summary>What does confidence five mean in the example?</summary>

It is the weight assigned by $1+\alpha r$ with count two and alpha two. It is not a five-star rating or a probability of liking the item.

</details>

<details>
<summary>How would you recommend a newly published lesson?</summary>

Use topic and prerequisite metadata to retrieve it, filter for eligibility, give it measured exposure, and collect interaction and outcome evidence. A collaborative factor cannot be learned from interactions that have not happened.

</details>

<details>
<summary><strong>Q5. (Easy)</strong> A user watched an item 3 times. With $c = 1 + \alpha r$ and $\alpha = 20$, what are the preference and the confidence?</summary>

Preference is 1 because the count is positive. Confidence is $1 + 20 \cdot 3 = 61$. The experiment above used a binary event, so there the confidence was 21.

</details>

<details>
<summary><strong>Q6. (Medium)</strong> An item shown at slot 3 (examination probability 1/3) got 12 clicks from 200 impressions. What are the naive and the exposure-corrected rates?</summary>

The naive rate is $12/200 = 0.06$. The exposure is $200 \times 1/3 = 66.67$, so the corrected rate is $12/66.67 = 0.18$. The item is three times as appealing as its raw click rate suggests, because it sat low on the page.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> The exposure correction improved the rank correlation from 0.787 to 0.931. Why can it not fix the three top-20 items that were never shown?</summary>

The corrected estimate is clicks divided by exposure. With zero impressions both numbers are zero, so there is no evidence to rescale. Fixing it needs new exposure, for example a small random slot in some requests, not a better formula.

</details>

## Further reading

- [Google's recommendation overview](https://developers.google.com/machine-learning/recommendation/overview/types) defines the staged architecture.
- [Collaborative filtering basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) distinguishes explicit and implicit feedback.
- [Hu, Koren and Volinsky, implicit-feedback collaborative filtering](https://yifanhu.net/PUB/cf.pdf) separates preference from confidence.
- [Recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) gives this curriculum's introductory worked rating example.
- [GroupLens MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) (opened 2026-10-08) states the licence terms and citation: Harper and Konstan, "The MovieLens Datasets: History and Context", ACM TiiS 5(4), 2015.
- In the [Hu, Koren and Volinsky paper](https://yifanhu.net/PUB/cf.pdf) (text extracted and read 2026-10-09) the authors report setting the confidence weight alpha to 40 in their television-viewing experiments; the experiment here uses 20 on a different dataset.
- [Schnabel, Swaminathan, Singh, Chandak and Joachims, "Recommendations as Treatments"](https://arxiv.org/abs/1602.05352) (ICML 2016, opened 2026-10-09) treats biased feedback with causal-inference estimators.
- [Joachims, Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback"](https://arxiv.org/abs/1608.04468) (2016, opened 2026-10-09) covers position bias and propensity weighting.

## Check yourself

- I can explain why a missing interaction is not a measured dislike.
- I can compute the preference and confidence of the two-event example.
- I can distinguish user cold start from item cold start and choose a fallback for each.
- I can specify an objective, eligibility rules, exposure logs and a label horizon before model training.
- I can explain why a model with the lowest rating error can still rank worse than popularity.
- I can turn a count into a preference and a confidence and say what confidence does not mean.
- I can correct a click rate for the slot an item was shown in, and say what the correction cannot do for an item that was never shown.

## Where to go next

Next is [collaborative filtering](/docs/theory/recsys/collaborative-filtering), which compares neighbour and factor models against the popularity baseline on this same split. For the evaluation side of the exposure problem see [evaluation and feedback loops](/docs/theory/recsys/evaluation-and-feedback-loops).
