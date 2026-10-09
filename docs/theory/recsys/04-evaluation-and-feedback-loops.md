---
title: "Recommenders · Evaluation and feedback loops"
sidebar_label: "Evaluation and feedback"
sidebar_position: 4
slug: /theory/recsys/evaluation-and-feedback-loops
description: "Offline ranking metrics, exposure bias, online experiments, diversity and operations."
tags: [recommender-systems, evaluation, experimentation]
---

import Infographic from '@site/src/components/Infographic';
import SlateDiversityLab from '@site/src/components/viz/SlateDiversityLab';

**In one line.** Evaluate a recommender as a displayed list that changes what users see, what they do, and what evidence the next model can learn.

:::tip Before you start
**You should already know**

- Precision, recall and reciprocal rank on a labelled list: [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval).
- The retrieve, rank, rerank stages: [retrieval, ranking and reranking](/docs/theory/recsys/retrieval-ranking-and-reranking).
- Why logs record what was shown, not what was liked: [feedback and objectives](/docs/theory/recsys/feedback-and-objectives).

**Reading time.** About 40 minutes, plus about half a minute to run the code.

**After this chapter you can**

- simulate a recommender that trains on its own clicks and measure how the catalogue it shows narrows,
- estimate the value of a new policy from old logs with inverse propensity scoring, and say when the estimate cannot be trusted,
- choose between a naive offline number, an IPS estimate and an online test.
:::

## In 30 seconds

A restaurant puts its most-ordered dish on the front of the menu. More people order it, because it is in front. Next month's menu goes with the numbers, and the front dish is the same one. Nothing was learned about the dishes that were never in front. Recommenders do this by default, because they train on clicks that their own lists caused. The fix has two parts: measure how narrow the shown set has become, and when you test a new list on old logs, weight each record by how likely it was to be shown, so a rarely shown dish counts for more when it is clicked.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Feedback loop | The system's output changes the data it learns from next | Popular films get shown, so get more clicks |
| Exposure Gini | A 0 to 1 measure of how unevenly impressions spread across items | 0.963 means almost all impressions go to a few items |
| Off-policy evaluation | Estimating a new policy's value from logs of an old one | Replaying last month's requests under a new ranker |
| Propensity | The probability that the logging policy chose the logged action | 0.10 for a rarely shown film |
| Inverse propensity scoring (IPS) | Weight each logged reward by new probability over logged probability | Weight 6.0 for a rare choice the new policy likes |
| Self-normalised IPS | IPS divided by the sum of the weights | 7.6 / 9.0 = 0.844 |
| Clipping | Cap large weights to tame variance, at the price of bias | Cap at 10 |
| Direct method | Fit a reward model, then score the new policy with it | Per-item click rates |

## The idea in plain words

A recommender's prediction is a ranked slate, not an isolated user-item score. The first position gets more attention; a repeated topic can make a slate feel narrow; a high-scoring but unavailable item cannot be used. Offline ranking metrics help compare candidate models, but the logged interactions were produced by an earlier recommendation policy. Online experiments test effects on users under controlled exposure. Production monitoring catches freshness, inventory and feedback-loop problems after release. All three views matter.

The feedback loop is fundamental. A model shows items; users can only interact with items they encounter; those interactions become future training data. A system that repeatedly shows popular items gathers more evidence for them and less evidence for new or niche items. Observed engagement can therefore rise while catalogue coverage and discovery shrink. To understand an outcome, keep the displayed slate, positions, eligibility snapshot and policy version together with the later event.

<Infographic src="/img/recsys/evaluation-loops.svg" alt="Offline replay, online experiments and production feedback monitoring each assess recommendation quality; a score-first A,B slate shares one topic while a 0.2 topic bonus selects A,C." caption="List quality includes relevance, opportunity, diversity and the behaviour created by display." />

## Worked example, step by step

A log has four requests. The logging policy picked the shown item with probability 0.50, 0.25, 0.10 and 0.40. The new policy would have picked those same items with probability 0.80, 0.10, 0.60 and 0.40. The clicks were 1, 0, 1, 0. The first block under "Code you can run" reproduces these numbers.

1. **Naive mean.** $(1 + 0 + 1 + 0)/4 = 0.5$. This is the old policy's click rate, not the new policy's.
2. **Weights.** Each weight is new probability divided by logged probability: $0.80/0.50 = 1.6$, $0.10/0.25 = 0.4$, $0.60/0.10 = 6.0$ and $0.40/0.40 = 1.0$.
3. **IPS.** Multiply each reward by its weight and average: $(1.6 \cdot 1 + 0.4 \cdot 0 + 6.0 \cdot 1 + 1.0 \cdot 0)/4 = 7.6/4 = 1.9$. A click rate cannot exceed 1, so this estimate is impossible. With four records the single weight of 6.0 dominates.
4. **Self-normalised IPS.** Divide the weighted sum by the sum of the weights: $7.6/(1.6 + 0.4 + 6.0 + 1.0) = 7.6/9.0 = 0.844$. It stays inside 0 and 1.
5. **Share of the total.** The weight-6.0 record supplies $6.0/7.6 = 0.789$ of the IPS total. One rarely logged choice decides the answer.

In words: IPS is unbiased, because each logged record stands in for the records the old policy rarely produced. It is also noisy, because those stand-ins are few. That noise is the price of having no experiment.

## How it works

### Build an honest offline split

Train only on interactions that happened before the evaluation period. Hold out later events and reconstruct which items were eligible at each request. Remove items already consumed where the product would do so. If the model uses user history up to a request, do not let it see the held-out event or a later event from the same user. Evaluate new users and new items separately because random interaction splits predominantly measure existing entities. A candidate source should be assessed for recall before a ranker is credited or blamed for a final result.

Even a chronological split cannot reveal all preferences. A held-out click is a positive signal for an item the old policy showed, but the unshown catalogue lacks labels. An offline metric computed over “all unclicked items are negative” can reward imitation of the old policy. Evaluate on exposed alternatives where possible, document candidate and exposure restrictions, and use online randomisation for claims about user impact. Logging the propensity of a controlled exploration policy can support more advanced counterfactual analysis, but such methods require overlap and careful variance checks. Do not infer causal improvement merely from an offline lift.

### Compute list metrics

Precision@$K$ is the fraction of displayed top-$K$ items labelled relevant; Recall@$K$ is the fraction of all labelled relevant items that appear in those $K$ positions. Precision can be high for a narrow slate while recall is poor. Candidate recall and displayed-list recall use different cutoffs and possibly different candidate universes; name the denominator. Mean reciprocal rank emphasises the first relevant item. Average precision rewards the positions of all relevant hits. Normalised discounted cumulative gain, NDCG, accepts graded relevance and discounts lower ranks. The choice should follow the product: one good answer may be enough for a help query, while a browsing surface benefits from several useful options.

For binary relevance with ranking `[A, B, C, D]`, suppose B and D are relevant. At $K=2$, precision is $1/2$ and recall is $1/2$. The first relevant item is at rank two, so reciprocal rank is $1/2$. For graded labels, DCG uses gains such as $2^{grade}-1$ divided by $\log_2(rank+1)$; NDCG divides by the ideal ordering's DCG. The exact gain convention and treatment of unjudged items must be recorded for comparisons. A metric can change solely because the candidate pool or labelling policy changed.

### Evaluate the slate as a whole

Individual item scores do not capture redundancy. Consider A with topic X and relevance 0.9, B with topic X and 0.8, and C with topic Y and 0.7. Score-first top two is A,B, with one distinct topic. A simple illustrative reranker chooses A first, then adds a 0.2 bonus to a new topic; C's adjusted second-position score becomes 0.9, above B's 0.8, yielding A,C and two topics. This rule trades some base relevance for category coverage. It is not evidence that users prefer A,C. Its effect should be assessed with both relevance and user-outcome measures.

Diversity can be measured in several ways: number of distinct topics, pairwise dissimilarity, exposure distribution across creators, or long-tail catalogue coverage. A superficial “distinct topic count” can be gamed by inconsistent metadata and may not reflect meaningful variety. Set diversity targets with the product context. A medical information tool may require a narrow set of vetted sources; an entertainment feed may benefit from exploration. Freshness likewise should not automatically trump relevance. Reranking constraints should have explicit ownership and audit logs.

### Run an online experiment

Randomly assign users or another appropriate unit to control and treatment. Keep assignment stable across repeated visits when cross-session effects matter. Define a primary outcome and guardrails before examining results, such as task completion, satisfaction, complaints, latency and inventory errors. Make sure the two variants see comparable eligible catalogues, and avoid concurrent experiments that change the same surface without accounting for interaction. A click lift can coexist with worse completion or increased returns; interpret the full set of agreed outcomes.

Measure exposure, not just outcomes. If treatment shows more items or uses a different layout, a raw click count may change for interface reasons. Track per-impression and per-user metrics as appropriate, but do not assume they answer the same question. Short tests can miss long-term fatigue or delayed conversion. A treatment that initially boosts novelty may lose value after repeated sessions, or may help users discover durable interests later. Use a rollout and monitoring plan that matches the expected time scale of benefit and harm.

### Monitor a changing system

After release, monitor request success, latency, index freshness, empty candidate lists, catalogue coverage, filtered-item rate and fallbacks. Track outcome metrics by segment when labels arrive. Compare new users, new items, sparse users and long-tail categories. A surge in clicks with falling distinct-item exposure is a feedback-loop warning. A new ranker version can also change the training distribution for the next version. Preserve policy, model and index versions in logs so a later experiment can reproduce what users saw.

Privacy and user control are part of quality. Avoid exposing sensitive inferred categories, and allow a person to indicate that a suggestion is irrelevant or unwanted where the surface permits. Use that feedback as a distinct signal rather than assuming all non-clicks mean the same thing. Retention and deletion rules should apply to impression and interaction logs as well as model features. A technically accurate ranking can still fail if it violates a user's explicit preference or a product safety rule.

## A real system that works this way

The [Google recommendation course](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) describes reranking for diversity, freshness and fairness after scoring. The [2016 YouTube recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) is a historical example of the candidate/ranker split. A course recommender built on that pattern would log eligible lessons, candidates, scores and final positions; test whether the list helps learners complete appropriate lessons; and guard against repeatedly promoting easy but off-path content. These are design implications of the published architecture, not claims about an unpublished current product.

## Code you can run

```python
ranked = [('A', False), ('B', True), ('C', False), ('D', True)]
k = 2
relevant_total = sum(relevant for _, relevant in ranked)
top = ranked[:k]
hits = sum(relevant for _, relevant in top)
precision = hits / k
recall = hits / relevant_total
first_relevant_rank = next(index for index, (_, relevant) in enumerate(ranked, start=1) if relevant)
print('precision@2:', precision)
print('recall@2:', recall)
print('reciprocal rank:', 1 / first_relevant_rank)
```

The three outputs are `0.5`. They describe a fully labelled four-item illustration, not an unbiased estimate from ordinary impression logs.

<SlateDiversityLab />

Move the new-topic bonus. At zero, the second item is B and the slate contains one topic. At the default 0.2, C's adjusted score is 0.9, so the lab selects A,C and shows two topics. The data table preserves each item's base score and adjusted second-position score.

```python
items = [('A', 'X', 0.9), ('B', 'X', 0.8), ('C', 'Y', 0.7)]
bonus = 0.2
first = items[0]
remaining = items[1:]
second = max(remaining, key=lambda item: item[2] + (bonus if item[1] != first[1] else 0))
slate = [first, second]
print('slate:', [item[0] for item in slate])
print('distinct topics:', len({item[1] for item in slate}))
```

This prints `['A', 'C']` and two topics. It only verifies the deterministic reranking arithmetic; it does not measure user satisfaction.

### Experiment: a recommender that learns from its own clicks

The world is simulated so that the truth is known. There are 600 users and 300 items. Each user and item has four hidden numbers, and the true click probability of a pair comes from their product plus an item quality term. Each round every user sees five items, and the user examines slot $k$ with probability 1.0, 0.8, 0.6, 0.45 and 0.35, then clicks with the pair's true probability. The first round is random. After that the recommender retrains on the accumulated clicks.

Three policies are compared. **Popularity** shows the five items with the most clicks so far. **Greedy MF** fits an 8-dimension truncated SVD to the click matrix and shows each user's top five. **MF with 10% exploration** replaces each slot with a random item with probability 0.10. The value of a round is the sum of the true click probabilities of the five shown items, averaged over users and over the last 10 of 40 rounds. The simulator also knows the best possible value, 4.253, from showing every user their true top five.

```python
import numpy as np

logged_prob = np.array([0.50, 0.25, 0.10, 0.40])
new_prob = np.array([0.80, 0.10, 0.60, 0.40])
reward = np.array([1, 0, 1, 0])
weight = new_prob / logged_prob
print('weights:', weight)
print('naive mean reward:', reward.mean())
print('IPS estimate:', round((weight * reward).mean(), 3))
print('self-normalised IPS:', round((weight * reward).sum() / weight.sum(), 3))
print('share of the IPS total from the largest weight:', round(weight[2] * reward[2] / (weight * reward).sum(), 3))
```

The printed weights are 1.6, 0.4, 6.0 and 1.0, the naive mean 0.5, IPS 1.9, self-normalised IPS 0.844, and the share of the total from the largest weight 0.789.

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD

n_users, n_items, d, slate, rounds = 600, 300, 4, 5, 40
world = np.random.default_rng(3)
Z = world.normal(0, 1, (n_users, d))
V = world.normal(0, 1, (n_items, d))
quality = world.normal(0, 0.5, n_items)
true_p = 1 / (1 + np.exp(-(1.2 * Z @ V.T + quality - 3.0)))
examine = np.array([1.0, 0.8, 0.6, 0.45, 0.35])
oracle = np.sort(true_p, axis=1)[:, -slate:].sum(1).mean()

def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())

def simulate(policy, seed):
    rng = np.random.default_rng(seed)
    clicks = np.zeros((n_users, n_items))
    shown_count = np.zeros(n_items)
    value, distinct = [], []
    for t in range(rounds):
        if t == 0:
            score = rng.random((n_users, n_items))
        elif policy == 'popularity':
            score = np.tile(clicks.sum(0), (n_users, 1)) + 0.01 * rng.random((n_users, n_items))
        else:
            svd = TruncatedSVD(8, random_state=0).fit(clicks)
            score = svd.transform(clicks) @ svd.components_ + 1e-3 * rng.random((n_users, n_items))
        top = np.argsort(-score, axis=1)[:, :slate]
        if policy == 'MF with 10% exploration':
            swap = rng.random((n_users, slate)) < 0.10
            top = np.where(swap, np.argsort(-rng.random((n_users, n_items)), axis=1)[:, :slate], top)
        p = np.take_along_axis(true_p, top, axis=1)
        got = rng.random(p.shape) < p * examine
        np.add.at(clicks, (np.repeat(np.arange(n_users), slate), top.ravel()), got.ravel())
        np.add.at(shown_count, top.ravel(), 1)
        value.append(p.sum(1).mean())
        distinct.append(len(np.unique(top)))
    return np.mean(value[-10:]), distinct, gini(shown_count), np.sort(shown_count)[-10:].sum() / shown_count.sum()

print('best possible value per slate (oracle):', round(oracle, 3))
print('distinct items shown in a round: columns are rounds 2, 6, 11 and 40; values are means of 4 seeds')
print('policy                     value   r2    r6   r11   r40   Gini  top-10 share')
per_seed = {}
for policy in ('popularity', 'greedy MF', 'MF with 10% exploration'):
    runs = [simulate(policy, s) for s in range(4)]
    per_seed[policy] = [round(float(r[0]), 2) for r in runs]
    dist = np.mean([[r[1][t] for t in (1, 5, 10, 39)] for r in runs], axis=0)
    print(f'{policy:25s} {np.mean([r[0] for r in runs]):.3f} {dist[0]:5.0f} {dist[1]:5.0f} {dist[2]:5.0f} {dist[3]:5.0f}  {np.mean([r[2] for r in runs]):.3f}  {np.mean([r[3] for r in runs]):.3f}')
print('value per seed, greedy MF:', per_seed['greedy MF'])
print('value per seed, MF with exploration:', per_seed['MF with 10% exploration'])
```

**Reading the output.** `value` is the mean of true expected clicks in a five-item slate. The columns `r2`, `r6`, `r11` and `r40` count how many distinct items were shown to anyone in that round, out of 300. `Gini` and `top-10 share` describe how impressions were spread over the whole run: a Gini of 0 is perfectly even and 1 is everything on one item.

**Line by line.**

- `np.add.at(clicks, ...)` accumulates clicks per user-item pair even when an item repeats in the same round.
- `p * examine` is the position bias: an item in slot 5 is examined with probability 0.35 however good it is.
- `shown_count` is the whole-run impression count that feeds `gini`.
- The second `for` loop builds the four-seed averages, and the last two prints show per-seed values so that a mean cannot hide disagreement.

The printed output was:

```text
best possible value per slate (oracle): 4.253
distinct items shown in a round: columns are rounds 2, 6, 11 and 40; values are means of 4 seeds
policy                     value   r2    r6   r11   r40   Gini  top-10 share
popularity                1.099     8     5     5     5  0.963  0.976
greedy MF                 1.635   299   262   124    38  0.826  0.393
MF with 10% exploration   1.690   300   284   220   203  0.757  0.392
value per seed, greedy MF: [1.53, 1.49, 1.8, 1.72]
value per seed, MF with exploration: [1.7, 1.43, 1.7, 1.93]
```

**What the numbers say.** Popularity collapses fast. After one round of learning, 8 distinct items are shown to 600 users, and from round 6 onward every user sees the same 5 items: Gini 0.963 and 97.6% of impressions on ten items. Greedy MF is personalised and gets a value of 1.635, but still narrows from 299 distinct items in round 2 to 38 in round 40. Ten percent exploration keeps 203 distinct items in round 40 and a Gini of 0.757.

The honest surprise is the value column. The exploration policy gives a higher mean value (1.690 against 1.635), but it won in two of four seeds and lost in the other two (1.43 against 1.49, and 1.70 against 1.80). So exploration bought a much wider catalogue at no reliable cost or gain in short-run value, and the gap between any of these and the best possible 4.253 is large. Limits: a simulator with a linear taste model that favours factor models, 600 users, 4 seeds, one exploration rate, and a click matrix that only accumulates. The pattern of a narrowing catalogue matches the confounding simulations reported by Chaney, Stewart and Engelhardt, but these numbers are not theirs.

### Experiment: estimating a new policy from old logs

Now one request is one decision. The old policy sees 20 random candidate items for a user and picks one by a softmax over item popularity, with temperature `tau_log`. A lower temperature makes it nearly deterministic. The new policy ranks by the true click logit at temperature 0.5. Both probabilities are exact, so IPS can use the true propensities. The simulator also computes the true value of the new policy from a 60,000-request sample, which plays the role an online A/B test would play in practice.

Five estimators are scored on 200 repeated logs of 2,000 requests each, at three logging temperatures.

```python
import numpy as np

n_users, n_items, d, n_cand = 600, 300, 4, 20
world = np.random.default_rng(3)
Z = world.normal(0, 1, (n_users, d))
V = world.normal(0, 1, (n_items, d))
quality = world.normal(0, 0.5, n_items)
logit = 1.2 * Z @ V.T + quality - 3.0
true_p = 1 / (1 + np.exp(-logit))
pop_score = np.log(true_p.mean(0))

def softmax_policy(scores, tau):
    z = scores / tau
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def log_requests(n, tau_log, rng):
    u = rng.integers(0, n_users, n)
    cand = np.stack([rng.choice(n_items, n_cand, replace=False) for _ in range(n)])
    logged = softmax_policy(pop_score[cand], tau_log)
    new = softmax_policy(logit[u[:, None], cand], 0.5)
    pick = np.array([rng.choice(n_cand, p=row) for row in logged])
    reward = (rng.random(n) < true_p[u, cand[np.arange(n), pick]]).astype(float)
    return u, cand, pick, reward, logged, new

u, cand, _, _, logged, new = log_requests(60000, 1.0, np.random.default_rng(99))
truth = (new * true_p[u[:, None], cand]).sum(1).mean()
print('true value of the new policy:', round(truth, 4), '| of the logging policy:', round((logged * true_p[u[:, None], cand]).sum(1).mean(), 4))

def estimates(n, tau_log, seed, clip=10):
    u, cand, pick, reward, logged, new = log_requests(n, tau_log, np.random.default_rng(seed))
    r = np.arange(n)
    w = new[r, pick] / logged[r, pick]
    shown = cand[r, pick]
    item_rate = np.bincount(shown, weights=reward, minlength=n_items) / np.maximum(np.bincount(shown, minlength=n_items), 1)
    return {
        'naive logged mean': reward.mean(),
        'IPS': (w * reward).mean(),
        f'IPS clipped at {clip}': (np.minimum(w, clip) * reward).mean(),
        'self-normalised IPS': (w * reward).sum() / w.sum(),
        'direct method (item rates)': (new * item_rate[cand]).sum(1).mean(),
    }

for tau_log in (1.0, 0.1, 0.03):
    runs = [estimates(2000, tau_log, s) for s in range(200)]
    print(f'logging temperature {tau_log}, 2000 logged requests, 200 repeats')
    for name in runs[0]:
        v = np.array([run[name] for run in runs])
        print(f'  {name:28s} mean {v.mean():.4f}  bias {v.mean() - truth:+.4f}  std {v.std():.4f}')
```

**Reading the output.** `mean` is the average estimate over 200 repeated logs, `bias` is that mean minus the true value of the new policy, and `std` is the spread of the estimate across logs. A good estimator has both small. The true values are printed first.

**Line by line.**

- `w = new[r, pick] / logged[r, pick]` is the inverse propensity weight.
- `np.minimum(w, clip)` is clipping, with the cap chosen as 10 without tuning.
- `direct method (item rates)` fits one click rate per item from the log and scores the new policy with it. It ignores who the user is.
- `log_requests` is called with a fresh seed for each repeat, so each estimate comes from an independent log.

The printed output was:

```text
true value of the new policy: 0.6516 | of the logging policy: 0.1744
logging temperature 1.0, 2000 logged requests, 200 repeats
  naive logged mean            mean 0.1744  bias -0.4772  std 0.0081
  IPS                          mean 0.6469  bias -0.0047  std 0.0549
  IPS clipped at 10            mean 0.5687  bias -0.0829  std 0.0448
  self-normalised IPS          mean 0.6479  bias -0.0037  std 0.0286
  direct method (item rates)   mean 0.2079  bias -0.4437  std 0.0119
logging temperature 0.1, 2000 logged requests, 200 repeats
  naive logged mean            mean 0.2567  bias -0.3949  std 0.0093
  IPS                          mean 0.6376  bias -0.0139  std 0.2007
  IPS clipped at 10            mean 0.4121  bias -0.2395  std 0.0337
  self-normalised IPS          mean 0.6578  bias +0.0062  std 0.1041
  direct method (item rates)   mean 0.1779  bias -0.4737  std 0.0108
logging temperature 0.03, 2000 logged requests, 200 repeats
  naive logged mean            mean 0.2701  bias -0.3815  std 0.0091
  IPS                          mean 0.4667  bias -0.1849  std 1.0274
  IPS clipped at 10            mean 0.2183  bias -0.4333  std 0.0165
  self-normalised IPS          mean 0.7254  bias +0.0738  std 0.1341
  direct method (item rates)   mean 0.1454  bias -0.5062  std 0.0111
```

**What the numbers say.** The new policy is worth 0.6516 clicks per request and the old one 0.1744. Averaging the logged clicks, which is what an offline replay of the old log does, gives 0.1744 with a bias of -0.4772 and a tiny spread of 0.0081: precisely wrong. The per-item direct method is equally wrong, about 0.21, because it ignores who the user is. IPS recovers the truth (bias -0.0047) when the old policy has broad coverage, with a spread of 0.0549. Self-normalised IPS has the same bias and about half the spread.

Two results run against habit. Clipping the weights at 10, the usual advice for taming variance, made things worse at every temperature, adding bias of -0.0829, -0.2395 and -0.4333. And when the old policy is nearly deterministic (temperature 0.03), plain IPS is unbiased only on paper: its spread of 1.0274 is larger than the quantity being estimated, so one real log would give almost any answer. Self-normalised IPS stays usable at 0.7254 but carries +0.0738 of bias. The practical rule is that IPS needs overlap: the old policy must have shown what the new one would show. Limits: exact propensities, a well-behaved simulator, one clip value, 2,000 requests.

<Infographic src="/img/recsys-enrich/loop-and-ips.svg" alt="Bars compare distinct items shown after 40 rounds for popularity, greedy factor and exploring factor policies, with a table of five estimators of a new policy's value at logging temperature 0.03." caption="Look first at the bars: popularity ends on 5 items and greedy factors on 38, and exploration keeps 203." />

## Designing with it

### Interpret metrics through the logged opportunity set

Suppose a new ranker raises Recall@10 on held-out clicks. Ask which items were eligible and exposed when those clicks were recorded. If the metric treats every unclicked catalogue item as irrelevant, it may favour popular, often-exposed items and punish novel ones that never had an opportunity. A temporal split prevents future interactions from leaking into training, but it does not solve exposure bias. At minimum, report the candidate pool, displayed positions and label construction. When a controlled exploration policy provides known selection probabilities, off-policy estimators may help, but they can become unstable when some actions had tiny probability or were never taken. An online experiment remains the clearer test of an intended product change.

List metrics also need a relevance horizon. For a course, clicking a lesson today is immediate while completing it may take days. A seven-day completion label cannot be evaluated on requests from yesterday. If unfinished recent requests are treated as failures, a model that sends learners to longer, more valuable lessons may look worse. Mature the labels or model the delay explicitly. Report the number of evaluated requests and any exclusions. Likewise, a purchase can be reversed by a return, and a “no purchase” label may depend on inventory and price at the time of display.

### A diversity example with real trade-offs

The lab's A,B versus A,C example shows only category coverage. It deliberately leaves out how useful B and C are to an actual user. To evaluate a diversity bonus, record both immediate engagement and longer-term outcomes, perhaps repeat visits or exploration of new topics. Inspect whether the rule helps users whose histories are narrow, harms users with a specific task, or changes exposure for creators. A one-size-fits-all bonus can be too strong for a focused search-like surface and too weak for a discovery surface. The acceptable relevance loss should be agreed before tuning the bonus to online data.

Topic metadata may be incomplete or manipulable. If a creator can choose a rare tag solely to receive a bonus, the diversity metric can rise without actual variety. Sample final slates for human review and compare semantic or editorial categories where appropriate. Distinct-item coverage across the catalogue is another signal: a model may show many topics but only the same few blockbuster items within each. Measure per-user variety, aggregate catalogue exposure and user value separately because one number cannot represent all three.

### Design an experiment that survives feedback

Randomisation unit matters. If the same user alternates between control and treatment across visits, their history is affected by both policies, making a clean comparison harder. Stable user assignment often suits a personalised surface; other products may need household, organisation or geographic assignment when users share content and influence one another. Predefine the primary outcome, guardrails and duration, and check that instrumentation records both variants consistently. If a new interface changes the number of visible tiles, normalise outcome metrics thoughtfully and report the actual exposure difference.

Early results may be misleading because users explore a new list differently at first. Some benefits require several sessions; some harms, such as fatigue or repetitive content, accumulate. Measure the time scale of the claimed benefit. Do not infer long-term improvement from a short click-rate lift. Keep a holdback or periodic comparison where practical, and monitor after rollout because the training data for the next model version will be produced under the new policy. Version the policy and preserve experiment assignment in the data used for retraining.

### Operate the whole slate pipeline

An incident can come from any stage. An empty candidate list may be an index outage, a new user's missing features or an over-strict eligibility filter. A sudden collapse in long-tail exposure may come from an ANN index change even when the ranker is unchanged. A click spike can come from a layout change. Dashboard stage-level counts, filtered reasons, latency, index age, fallback rate and outcome metrics together. Sample request traces with consent and appropriate privacy controls. When a defect is found, the logs should let operators replay the candidate, ranker and reranker versions that produced the displayed slate.

Define an evaluation card for each recommendation surface: request context, eligible universe, exposure logging, relevance label, label delay, offline split, baseline, list metrics, online primary metric and guardrails. Keep candidate, ranker and reranker versions in a single trace. When a metric moves, inspect stage-level coverage and the actual lists rather than changing the final score blindly. Check whether a data collection or layout change caused the movement.

For an experiment, decide a minimum meaningful effect and an observation window before starting. Do not stop as soon as a noisy daily plot crosses a desired threshold. Segment analysis is useful for diagnosis, but many unplanned subgroup comparisons raise false-discovery risk. Preserve an overall primary decision rule and treat exploratory slices as hypotheses for follow-up. After rollout, maintain a holdback or periodic control where appropriate so the feedback loop does not erase the comparison.

## Where this stands in 2026

As recommendation models and indexes become more capable, the difficult question remains whether a displayed list helps users and supports a healthy catalogue over time. Offline metrics are fast iteration tools, controlled online tests support causal product decisions, and monitoring catches drift and feedback loops. Diversity and fairness are not single universal scores; they require a documented product goal and relevant slices. The model, interface and data-collection policy together determine observed outcomes.

## Common mistakes

1. **Evaluating a new policy by averaging the old log's clicks.** It feels like replay. The average measures the old policy: 0.1744 against a true 0.6516. Weight by propensities or run an experiment.
2. **Clipping IPS weights by default.** Clipping tames variance and so looks safe. At cap 10 it biased every estimate by between -0.08 and -0.43 here. Clip only after checking the weight distribution and the bias you accept.
3. **Trusting IPS when the old policy was nearly deterministic.** The estimate is unbiased on paper. The spread was 1.0274 at temperature 0.03. Check overlap first.
4. **Retraining on your own clicks without exploration.** The top of the list collects the data. Popularity held 5 distinct items from round 6. Reserve a logged random slice of traffic.
5. **Reading a single seed.** Exploration won two seeds and lost two on short-run value. Repeat the run and show per-seed values.

## Practice questions

<details>
<summary>Why can a model improve offline recall while having no online benefit?</summary>

Offline labels reflect the prior exposure policy and may not represent unseen items or the intended user outcome. The displayed slate, latency or eligibility can also differ. Test the complete product effect online.

</details>

<details>
<summary>What are precision@2 and recall@2 for A,B when B and D are relevant?</summary>

One of two displayed items is relevant and one of two relevant catalogue items is displayed, so both are 0.5.

</details>

<details>
<summary>What does the diversity lab demonstrate and what does it not?</summary>

It demonstrates how a new-topic bonus changes a two-item slate from one topic to two. It does not establish that users prefer the reranked slate.

</details>

<details>
<summary>How can a recommender create a popularity feedback loop?</summary>

It shows already-popular items more often, creating more interactions for them, then trains on those interactions. Less-shown items remain data-poor regardless of their potential value.

</details>

<details>
<summary><strong>Q5. (Easy)</strong> A logged click had probability 0.20 under the old policy and 0.50 under the new one. What is its IPS weight?</summary>

0.50 / 0.20 = 2.5. The record stands in for 2.5 clicks of the kind the new policy would produce, because the old policy showed it only 40% as often as the new one would.

</details>

<details>
<summary><strong>Q6. (Medium)</strong> In the worked example the IPS estimate was 1.9. Why is that evidence of a problem, and which estimator would you report?</summary>

A click rate cannot exceed 1, so 1.9 shows variance, driven by one record with weight 6.0. Self-normalised IPS gives 0.844 and stays in range, at the cost of a small bias that disappears with more data. Report that, with the number of records and the largest weight.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> Exploration kept 203 distinct items at round 40 but did not reliably raise short-run value. Is it worth 10% of slots?</summary>

It depends on what the product values. Value here counted only expected clicks in the next rounds, while a wider catalogue supports new items, creators and future training data. Run the experiment with long-term metrics, set the rate by the cost of a wasted slot, and log the exploration probability so IPS can use it later.

</details>

## Further reading

- [Google recommendation overview](https://developers.google.com/machine-learning/recommendation/overview/types) connects the stages of a full system.
- [Google scoring guidance](https://developers.google.com/machine-learning/recommendation/dnn/scoring) discusses common scoring of mixed candidates.
- [Google reranking guidance](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) treats diversity, freshness and fairness.
- [Historical two-stage recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) documents a published large-scale design.
- [Chaney, Stewart and Engelhardt, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility"](https://arxiv.org/abs/1710.11214) (RecSys 2018, opened 2026-10-09) uses simulations to show that training on data produced under an existing recommender homogenises behaviour without raising utility.
- [Schnabel, Swaminathan, Singh, Chandak and Joachims, "Recommendations as Treatments"](https://arxiv.org/abs/1602.05352) (ICML 2016, opened 2026-10-09) applies causal-inference estimators to biased recommendation feedback.
- [Joachims, Swaminathan and Schnabel, "Unbiased Learning-to-Rank with Biased Feedback"](https://arxiv.org/abs/1608.04468) (2016, opened 2026-10-09) derives propensity-weighted learning for position-biased clicks.
- scikit-learn 1.9.1 and NumPy 2.5.3 were the versions run.

## Check yourself

- I can calculate precision, recall and reciprocal rank on a labelled slate.
- I can distinguish candidate recall from displayed-list quality.
- I can explain why observed clicks depend on exposure and position.
- I can design an online experiment with a primary outcome and guardrails.
- I can monitor catalogue coverage, fresh items and fallback use after release.
- I can compute IPS and self-normalised IPS for a small log by hand and say which one stays in range.
- I can simulate a feedback loop and report distinct items shown, an exposure Gini and a top-10 share, not just clicks.
- I can explain why an offline replay of old clicks measures the old policy, and what overlap IPS needs.
- I can say why clipping is a trade of variance for bias and decide whether I accept it.

## Where to go next

This closes the recommender track. For the staged system whose lists feed the loop, revisit [retrieval, ranking and reranking](/docs/theory/recsys/retrieval-ranking-and-reranking). For metric choices on ranked lists in text search, see [evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval).
