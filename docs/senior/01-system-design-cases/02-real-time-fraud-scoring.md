---
id: senior-case-fraud
title: "Design Real-Time Fraud Scoring"
sidebar_label: "2 · Fraud scoring"
sidebar_position: 2
slug: /senior/design-real-time-fraud-scoring
description: "A whole-system design for scoring card transactions in under 100 milliseconds: latency budgets and tail behaviour, fresh features, delayed labels, drift, alert budgets and cost, with every number simulated in code."
tags: [system-design, fraud-detection, latency, feature-store, concept-drift, delayed-labels, cost-estimation]
---

import Infographic from '@site/src/components/Infographic';
import LatencyBudgetLab from '@site/src/components/viz/LatencyBudgetLab';

**In one line.** Real-time fraud scoring is two systems joined by a feature store: a decision path with a hard latency budget that is dominated by its slowest lookup, and a learning path whose labels arrive weeks late and whose attackers change behaviour in between.

:::note Not from a lecture
Written for this site from the public sources listed under Further reading. The payment company is an invented teaching scenario; the latency and fraud numbers come from simulations in this chapter, and the amounts and costs are parameters you replace.
:::

## The idea in plain words

A card payment reaches your system and must be approved, challenged or declined **before the customer's checkout times out**. You have a model that scores risk, but the model is the easy part. The hard parts are around it:

- the whole decision has a budget of about 100 milliseconds, shared by networks, feature lookups, the model and policy rules;
- the best signals are fresh counts ("how many transactions has this card made in ten minutes"), which must be served in milliseconds yet match what the model was trained on;
- the truth, a chargeback or a confirmed fraud, arrives days to weeks later, and only for the transactions someone investigated or disputed;
- fraudsters adapt, so yesterday's data describes yesterday's attacks;
- a missed fraud costs the amount, and a false alarm costs friction and review time, so the threshold is a money decision.

Stripe describes the same shape for its Radar product: it scores each payment from more than 1,000 characteristics of the transaction and decides in under 100 milliseconds (Stripe engineering blog, 29 March 2023).

<Infographic src="/img/senior/real-time-fraud-scoring-architecture.svg" alt="A synchronous decision path from gateway through feature fetch, model, rules and decision with p99 allowances, above an asynchronous learning path from event stream through stream features, late labels, point-in-time training set and canary." caption="Two paths and the numbers printed below: a 66 ms sum of p99 allowances against a 100 ms limit." />

## How it works

### Requirements, with numbers

| Requirement | Value used in this chapter |
| --- | --- |
| Peak load | 5,000 transactions per second |
| Decision latency | 100 ms at p99, measured at the gateway |
| Availability | a decision is always returned; a slow dependency degrades the decision, never blocks it |
| Quality | alert precision and recall at a fixed review capacity, not accuracy |
| Labels | chargebacks arrive after about 14 days or more; reviews cover a small fraction |
| Freshness | velocity features no more than seconds old |

### Back-of-envelope sizing

The first code block works out the budget and the load. The sum of the stage p99 allowances is 66 ms, leaving 34 ms of headroom, but a sum of percentiles is not the percentile of the sum, which is why the second block simulates it. Eight entity lookups per transaction (card, device, merchant, IP, account and so on) at 5,000 per second is 40,000 reads a second against the feature store. Forty features for 50 million active cards at 8 bytes with three replicas is 48.0 GB. The event stream is 173 GB a day if the day averages 40% of peak.

### Data flow

**Decision path.** The gateway sends the transaction. The service fetches features for the entities involved **in parallel**, applies the model, then applies policy rules (blocklists, amount limits, regulatory holds) and thresholds, and returns approve, step up or decline. If the feature fetch overruns a timeout the service scores with default features and flags the decision as degraded.

**Learning path.** Every transaction and outcome goes onto an event stream. A stream processor maintains rolling features (counts, sums, distinct counts over windows) in the online store. Labels come back late and biased, and are joined to the features **as they were at decision time**, which Feast's documentation calls a point-in-time join and describes as the way to prevent data leakage and training-serving skew. New models run in shadow and then as a small canary before they decide anything.

### The decisions that matter

**1. Where does the model run?**

| Option | Latency | Trade-off |
| --- | --- | --- |
| In-process library inside the decision service | lowest, no network hop | model and service deploy together |
| Separate model service | extra network hop and its own tail | independent releases and hardware |
| Decide first, score after authorisation | no budget pressure | fraud is already approved; only holds and recovery remain |

**2. How are features served?**

| Option | Fresh | Consistent with training | Cost |
| --- | --- | --- | --- |
| Compute from raw history at request time | yes | yes, if the code is shared | slow, unbounded fan-out |
| Stream-updated online store | seconds | needs one definition for both paths | stream processor, store, replicas |
| Daily batch features | no | easy | blind to bursts, as the stale-feature table shows |

**3. Which model?** Gradient-boosted trees are the sensible start: small, fast on CPU and easy to inspect. Stripe's blog reports moving in mid-2022 from an ensemble of XGBoost and deep networks to a DNN-only multi-branch design inspired by ResNeXt, cutting training time by more than 85% and running tenfold-larger training experiments with continued gains. That is the choice of a company with enormous data; most teams get more from fresher features and cleaner labels.

**4. What happens when a dependency is slow or down?**

| Policy | Effect |
| --- | --- |
| Fail open (approve) | no lost sales, free money for attackers |
| Fail closed (decline) | safe, and customers leave |
| Degrade to rules and default features | bounded risk, a measured accuracy loss, the usual answer |

### Failure modes and mitigations

| Failure | Looks like | Mitigation |
| --- | --- | --- |
| Tail latency from fan-out | p99 far above the sum of typical times | parallel lookups, hedged requests, timeouts with fallback |
| Training-serving skew | strong offline metric, weak live metric | one feature definition, point-in-time joins, log served features |
| Concept drift | quality decays when attackers change style | time-ordered validation, frequent retraining, drift alerts on inputs and scores |
| Label delay and bias | the model never sees what it let through | randomised holdout of a slice of traffic for labelling; separate delayed labels from investigator feedback |
| Alert overload | reviewers ignore a noisy queue | choose the threshold from review capacity and cost |

### Evaluation and rollout

Evaluate on **time-ordered splits** and report alert precision and recall at the number of alerts reviewers can handle. Run a new model in shadow for a full business cycle, then on 1% to 5% of traffic with the old model as the control. Watch approval rate, degraded rate, review precision and later chargebacks, and keep a rollback that needs no deploy.

## A real system that works this way

- **Stripe Radar.** The 2023 engineering post cited above states the decision is made in under 100 milliseconds from more than 1,000 transaction characteristics, and documents the move to a DNN-only model and shorter training. This chapter's architecture is a generic design, not a description of Stripe's internals.
- **Fan-out and the tail.** In "The Tail at Scale" (Dean and Barroso, CACM, 2013), Google engineers show that if a server answers in 10 ms but is slow one time in a hundred, a request fanned out to 100 such servers is slow **63%** of the time. They also report that a hedged request sent after 10 ms cut the 99.9th percentile for reading 1,000 BigTable values from 1,800 ms to 74 ms while sending only 2% more requests, and that deferring the second request until the 95th percentile has passed caps the extra load at about 5%.
- **Fraud-specific learning problems.** Dal Pozzolo and colleagues (IEEE TNNLS, 2018) describe concept drift, class imbalance and verification latency, note that investigators can check only a few alerts a day, define alert precision as the share of true frauds among the top k alerts, and show that investigator feedback and delayed labels should be handled separately.

## Code you can run

Python 3.14.6, numpy 2.5.3, scikit-learn 1.9.1, run on 2 October 2026. All numbers are seeded and printed. The transaction history is **synthetic**: it exists to show mechanisms, not to estimate real fraud rates.

#### 1. Budget, load and cost estimator

```python
PEAK_TPS = 5_000
ANNUAL_TRANSACTIONS = 800e6
AVERAGE_AMOUNT = 60.0
FRAUD_RATE = 0.0007
ENTITY_LOOKUPS_PER_TRANSACTION = 8
ACTIVE_CARDS = 50e6
FEATURES_PER_CARD = 40
BYTES_PER_FEATURE = 8
REPLICAS = 3
EVENT_BYTES = 1_000
REVIEW_COST = 4.0
FRICTION_COST = 1.0
RECALL = 0.60
ALERT_RATE = 0.002

print("end-to-end budget at p99, in milliseconds")
budget = {"network in": 12, "feature fetch": 25, "model": 14, "rules": 3, "network out": 12}
for stage, ms in budget.items():
    print(f"  {stage:14s} {ms:3d}")
print(f"  {'sum of p99s':14s} {sum(budget.values()):3d}   limit 100   headroom {100 - sum(budget.values())}")

lookups_per_second = PEAK_TPS * ENTITY_LOOKUPS_PER_TRANSACTION
store_gb = ACTIVE_CARDS * FEATURES_PER_CARD * BYTES_PER_FEATURE * REPLICAS / 1e9
stream_gb_per_day = PEAK_TPS * 0.4 * 86400 * EVENT_BYTES / 1e9
print(f"\nfeature store reads at peak: {lookups_per_second:,} per second")
print(f"online store: {store_gb:,.1f} GB with {REPLICAS} replicas")
print(f"event stream if the day averages 40% of peak: {stream_gb_per_day:,.0f} GB per day")

fraud_transactions = ANNUAL_TRANSACTIONS * FRAUD_RATE
fraud_loss = fraud_transactions * AVERAGE_AMOUNT
alerts = ANNUAL_TRANSACTIONS * ALERT_RATE
caught = fraud_transactions * RECALL
missed_loss = (fraud_transactions - caught) * AVERAGE_AMOUNT
review = alerts * REVIEW_COST
friction = (alerts - caught) * FRICTION_COST
print(f"\nfraud transactions per year: {fraud_transactions:,.0f}   fraud loss with no model: {fraud_loss:,.0f}")
print(f"with recall {RECALL:.0%} at an alert rate of {ALERT_RATE:.1%}: missed {missed_loss:,.0f} + review {review:,.0f} + friction {friction:,.0f} = {missed_loss + review + friction:,.0f}")
print(f"alert precision at that point: {caught / alerts:.3f}")
```

Without any model, 560,000 fraudulent transactions a year at an average of \$60 cost \$33,600,000. With a model that catches 60% of fraud while alerting on 0.2% of traffic, the estimate is \$13,440,000 missed, \$6,400,000 of review and \$1,264,000 of friction, \$21,104,000 in total, at an alert precision of 0.210. The formula is `missed fraud amount + alerts x review cost + false alerts x friction cost`, and every term is a parameter.

#### 2. Latency Monte Carlo

Each stage's latency is log-normal, defined by its median and 99th percentile. The feature fetch is eight lookups with a 4 ms median and a 25 ms p99. A seeded generator (the same one the lab uses) draws 20,000 requests, and four designs are compared: lookups in series, in parallel, in parallel with a hedged second call after the single-lookup p95, and the same with a 20 ms timeout that falls back to default features.

```python
import math

SAMPLES = 20000
Z99 = 2.3263478740408408
Z95 = 1.6448536269514722
MASK = 0xFFFFFFFF


class Mulberry32:
    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x6D2B79F5) & MASK
        a = self.state
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0

    def normal(self):
        u1 = max(self.next(), 1e-12)
        u2 = self.next()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def draw_normals(samples, slots, seed=20261002):
    rng = Mulberry32(seed)
    return [[rng.normal() for _ in range(slots)] for _ in range(samples)]


def lognormal(p50, p99, z):
    sigma = math.log(p99 / p50) / Z99
    return p50 * math.exp(sigma * z)


STAGES = {"network in": (3, 12), "model": (6, 14), "rules": (1, 3), "network out": (3, 12)}
LOOKUP_SLOTS = 20
SLOTS = 4 + 2 * LOOKUP_SLOTS
NORMALS = draw_normals(SAMPLES, SLOTS)


def simulate(lookups=8, lookup_p50=4.0, lookup_p99=25.0, parallel=True, hedge=False, timeout=None):
    hedge_delay = lookup_p50 * math.exp(math.log(lookup_p99 / lookup_p50) / Z99 * Z95)
    totals, degraded, hedged_calls = [], 0, 0
    for z in NORMALS:
        calls = []
        for j in range(lookups):
            t = lognormal(lookup_p50, lookup_p99, z[4 + j])
            if hedge and t > hedge_delay:
                hedged_calls += 1
                t = min(t, hedge_delay + lognormal(lookup_p50, lookup_p99, z[4 + LOOKUP_SLOTS + j]))
            calls.append(t)
        fetch = max(calls) if parallel else sum(calls)
        if timeout is not None and fetch > timeout:
            fetch = timeout
            degraded += 1
        total = (lognormal(*STAGES["network in"], z[0]) + lognormal(*STAGES["model"], z[1])
                 + lognormal(*STAGES["rules"], z[2]) + lognormal(*STAGES["network out"], z[3]) + fetch)
        totals.append(total)
    totals.sort()
    pick = lambda p: totals[math.ceil(p * len(totals)) - 1]
    return {
        "p50": pick(0.5), "p95": pick(0.95), "p99": pick(0.99), "p99.9": pick(0.999),
        "over100": sum(t > 100 for t in totals) / len(totals),
        "degraded": degraded / len(totals),
        "extra": hedged_calls / (len(totals) * lookups),
    }


configs = [
    ("8 lookups in series", dict(parallel=False)),
    ("8 lookups in parallel", dict()),
    ("parallel + hedge at p95", dict(hedge=True)),
    ("parallel + hedge + 20 ms timeout", dict(hedge=True, timeout=20.0)),
]
print("p99 of one lookup is 25 ms; sum of the stage p99s is", 12 + 25 + 14 + 3 + 12, "ms")
print(f"{'design':34s}{'p50':>7s}{'p95':>7s}{'p99':>7s}{'p99.9':>8s}{'>100 ms':>9s}{'degraded':>10s}{'extra calls':>13s}")
for name, kw in configs:
    r = simulate(**kw)
    print(f"{name:34s}{r['p50']:7.1f}{r['p95']:7.1f}{r['p99']:7.1f}{r['p99.9']:8.1f}{r['over100']:9.2%}{r['degraded']:10.2%}{r['extra']:13.2%}")

print("\nchance that at least one of n lookups lands in its own slowest 1%")
for n in (1, 8, 20, 100):
    print(f"  n = {n:3d}: {1 - 0.99 ** n:.1%}")
```

The sum of the stage p99s is 66 ms, yet eight lookups in series give a p99 of 104.2 ms and 1.39% of requests over the limit: **series is the first thing to fix**. In parallel the p99 falls to 57.9 ms and 0.07% exceed 100 ms, but p99.9 is still 93.2 ms because the slowest of eight lookups is slow more often than one. Hedging at the single-lookup p95 brings p99 to 43.7 ms and p99.9 to 50.3 ms for 4.93% extra calls, the cost the Tail at Scale paper predicts. The 20 ms timeout trims the tail a little more (p99 42.0 ms) by degrading 5.49% of decisions, a real accuracy price. The last lines show the fan-out arithmetic: with n lookups, the chance that at least one is in its own slowest 1% is 7.7% for 8, 18.2% for 20 and 63.4% for 100, matching the paper.

The lab runs the same 20,000 draws. Its defaults (eight parallel lookups, p99 25 ms) give p99 57.9 ms and 0.07% over 100 ms; switch to series for 104.2 ms and 1.39%.

<LatencyBudgetLab />

#### 3. Drift, delayed labels, stale features and the threshold

A 60-day history of 1,500 cards with two attack styles: fast bursts of small then large charges from a new device, and, for half the attacks after day 40, slower and larger charges. Features are the amount, its ratio to the card's average, hour, category, new-device and foreign flags, and velocity counts over 10 minutes and 24 hours.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DAYS, CARDS, DRIFT_DAY = 60, 1500, 40
rng = np.random.default_rng(7)


def make_world():
    rows = []
    card_scale = rng.lognormal(3.4, 0.5, CARDS)
    for card in range(CARDS):
        n = rng.poisson(1.5 * DAYS)
        t = np.sort(rng.uniform(0, DAYS * 86400, n))
        amount = card_scale[card] * rng.lognormal(0, 0.8, n)
        category = rng.integers(0, 10, n)
        device_new = rng.random(n) < 0.02
        foreign = rng.random(n) < 0.03
        for i in range(n):
            rows.append((t[i], card, amount[i], category[i], device_new[i], foreign[i], 0))
    for card in rng.choice(CARDS, size=int(CARDS * DAYS * 0.0016), replace=True):
        start = rng.uniform(0, DAYS * 86400)
        late = start / 86400 >= DRIFT_DAY and rng.random() < 0.5
        k = rng.integers(3, 7) if late else rng.integers(4, 11)
        gaps = rng.uniform(300, 1800, k) if late else rng.uniform(20, 90, k)
        times = start + np.cumsum(gaps)
        for j, tt in enumerate(times):
            if tt >= DAYS * 86400:
                continue
            base = 110 if late else (4 if j < 2 else 150)
            rows.append((tt, card, base * rng.lognormal(0, 0.5), rng.integers(0, 10),
                         rng.random() < (0.6 if late else 0.8), rng.random() < 0.5, 1))
    rows.sort(key=lambda r: r[0])
    return np.array(rows, dtype=float)


W = make_world()
time_s, card, amount, category, device_new, foreign, label = W.T
card = card.astype(int)
day = (time_s // 86400).astype(int)
print(f"{len(W):,} transactions over {DAYS} days, {int(label.sum()):,} fraud ({label.mean():.2%}), half the attacks switch style on day {DRIFT_DAY}")


def window_count(t, lag, width):
    return np.searchsorted(t, t - lag, side="left") - np.searchsorted(t, t - lag - width, side="left")


def features(lag):
    v10, v24, ratio = np.zeros(len(W)), np.zeros(len(W)), np.ones(len(W))
    for c in range(CARDS):
        idx = np.where(card == c)[0]
        t = time_s[idx]
        v10[idx] = window_count(t, lag, 600)
        v24[idx] = window_count(t, lag, 86400)
        past = np.cumsum(amount[idx]) - amount[idx]
        n = np.arange(len(idx))
        ratio[idx] = np.where(n > 0, amount[idx] / np.maximum(past / np.maximum(n, 1), 1e-9), 1.0)
    hour = (time_s % 86400) // 3600
    return np.column_stack([amount, np.log1p(ratio), hour, category, device_new, foreign, v10, v24])


def fit(X, rows):
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, random_state=0).fit(X[rows], label[rows])


def score(model, X, rows):
    p = model.predict_proba(X[rows])[:, 1]
    return roc_auc_score(label[rows], p), average_precision_score(label[rows], p), p


X = features(0)
shuffle = np.random.default_rng(0).permutation(len(W))
random_train, random_test = shuffle[: int(0.7 * len(W))], shuffle[int(0.7 * len(W)):]
past, future = np.where(day < DRIFT_DAY)[0], np.where(day >= DRIFT_DAY)[0]
print("\nvalidation scheme                         ROC AUC   average precision")
auc, ap, _ = score(fit(X, random_train), X, random_test)
print(f"random 70/30 split of all rows            {auc:8.4f}   {ap:8.4f}")
auc, ap, _ = score(fit(X, past), X, future)
print(f"train days 0-39, test days 40-59          {auc:8.4f}   {ap:8.4f}")
delay = 14
final = np.where(day >= 50)[0]
for name, last_day in (("labels instantly available", 49), (f"labels arrive after {delay} days", 49 - delay)):
    auc, ap, _ = score(fit(X, np.where(day <= last_day)[0]), X, final)
    print(f"test days 50-59, {name} (train to day {last_day})   {auc:8.4f}   {ap:8.4f}")

early = np.where(day < DRIFT_DAY)[0]
split = np.where(day[early] < 30)[0]
tr, te = early[split], early[np.where(day[early] >= 30)[0]]
print("\nstale velocity features (days 0-29 train, 30-39 test)")
print("train lag   serve lag   average precision")
for train_lag, serve_lag in ((0, 0), (0, 300), (0, 1800), (300, 300)):
    m = fit(features(train_lag), tr)
    Xs = features(serve_lag)
    print(f"{train_lag:7d} s   {serve_lag:7d} s   {score(m, Xs, te)[1]:8.4f}")

_, _, p = score(fit(X, past), X, future)
y = label[future]
amt = amount[future]
review_cost, friction_cost = 4.0, 1.0
print("\nthreshold on the days 40-59 scores: expected cost = missed fraud amount + review cost per alert")
print("alert rate   precision   recall   cost per 1,000 transactions")
for rate in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
    k = int(rate * len(p))
    top = np.argsort(-p)[:k]
    caught = y[top].sum()
    missed = amt[(y == 1)].sum() - amt[top][y[top] == 1].sum()
    cost = missed + review_cost * k + friction_cost * (k - caught)
    print(f"{rate:9.1%}   {caught / k:9.2f}   {caught / y.sum():6.2f}   {1000 * cost / len(p):10.1f}")
```

There are 135,819 transactions and 953 frauds (0.70%). **A random split flatters the model**: ROC AUC 0.9755 and average precision 0.8438, because rows from the same burst land on both sides of the split. Training on days 0 to 39 and testing on 40 to 59, when half the attacks have changed, drops them to 0.7243 and 0.5151.

**Label delay slows the recovery.** For days 50 to 59, a model trained to day 49 scores 0.9304 and 0.6349 because it has seen ten days of the new style. With a 14-day label delay it can only train to day 35, has seen none of it, and scores 0.7317 and 0.5104. Drift is detected as late as labels arrive.

**Stale velocity features are worse than no velocity features at the wrong time.** Fresh features score average precision 0.8222. Serving counts that are 5 minutes stale to a model trained on fresh ones gives 0.4853 (and 0.4048 at 30 minutes); retraining on the stale version recovers 0.6848. Skew costs more than staleness, but staleness still costs: burst signals live in minutes.

**The threshold is a cost decision.** On the days 40 to 59 scores the cheapest alert rate is 2.0% (cost 309.1 per 1,000 transactions, precision 0.18, recall 0.66), with 1.0% close behind (313.4); alerting on 5.0% costs 459.1 because reviews pile up while recall stalls at 0.66. A 0.5 probability cut-off would have been chosen by nothing in this table.

<Infographic src="/img/senior/real-time-fraud-scoring-latency-and-drift.svg" alt="Four tables: end-to-end latency for four feature-fetch designs, the chance one of n lookups is slow, validation scores for random and time-ordered splits, and the effect of stale velocity features." caption="The latency and drift results printed by the second and third code blocks." />

## Designing with it

**Cost estimate as a formula.**

`annual cost = missed fraud + review cost x alerts + friction x false alerts + feature store (GB x replicas x price) + stream processing + model serving + engineering`

The first three terms depend on the model and threshold; the rest are roughly fixed. Improving recall at constant alerts, or holding recall with fewer alerts, is where the money is.

**What I would build first.**

1. A rules and velocity baseline with logging of every served feature value.
2. A tree model trained with point-in-time joins and evaluated on time-ordered splits.
3. Parallel feature fetch with a timeout and a rules fallback, before any hedging.
4. A shadow deployment, then a canary, with a randomised holdout for unbiased labels.
5. Only then a larger model, hedging and richer streaming features.

## Where this stands in 2026

:::info Industry view

- **Stripe's public account shows the frontier choice, not the default.** Moving to a single deep model with far more training data suits a processor at that scale; the tail-latency and label-delay problems described here apply at every size.
- **Hedging and timeouts are the tools for fan-out tails.** The simulation here reproduces the 2013 paper's 63% fan-out figure and its roughly 5% hedging overhead.
- **Label delay is the structural problem.** The 2018 fraud-detection paper treats verification latency and investigator bias as central; no architecture removes them, so holdouts and separate handling of delayed labels stay necessary.
- **Point-in-time correctness is expected tooling.** Feast's documentation presents point-in-time joins as the guard against leakage and training-serving skew.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> The stage p99 allowances sum to 66 ms, but eight lookups in series gave a p99 of 104.2 ms. Why?</summary>

The allowance for the feature fetch was 25 ms for one lookup, but eight in series add eight typical times and give eight chances of a slow one: the median of the whole request rises from 26.8 ms in parallel to 56.0 ms, and the tail follows. A budget built by summing per-call percentiles ignores that series multiplies the typical time and the tail. Parallel lookups bring the p99 to 57.9 ms.

</details>

<details>
<summary><strong>Q2.</strong> Why does hedging at the single-lookup p95 cost about 5% extra calls?</summary>

A hedge is sent only when the first call is still outstanding after the p95 delay, which by definition happens for about 5% of calls (4.93% in the simulation). The second call usually finishes before the slow one, so p99 drops from 57.9 ms to 43.7 ms. Sending the hedge earlier would cut more but cost more load.

</details>

<details>
<summary><strong>Q3.</strong> A random split gives average precision 0.8438 and a time split 0.5151. Which do you trust and why?</summary>

The time split. Fraud arrives in bursts, so a random split puts rows of one attack in training and test, and it ignores that attack styles change. The time-ordered split mimics deployment: train on the past, score the future.

</details>

<details>
<summary><strong>Q4.</strong> Labels arrive 14 days late. What does that do to retraining?</summary>

The newest usable training day is 14 days old. In the simulation a model trained to day 35 scored 0.5104 on days 50 to 59 against 0.6349 for one that had seen the new attack style. Mitigate with a randomised holdout of traffic that is labelled quickly, drift monitoring on inputs and scores, and fast-feedback signals such as confirmed reviews, handled separately from delayed chargebacks.

</details>

<details>
<summary><strong>Q5.</strong> The velocity features in production are five minutes stale because of a batch job. What do you expect and what do you do?</summary>

A large drop on burst fraud: 0.8222 to 0.4853 average precision when the model was trained on fresh counts. Either serve fresh counts from the stream, or retrain on the stale version (0.6848) as a stopgap, and log served features so skew is measurable.

</details>

<details>
<summary><strong>Q6.</strong> How do you choose the decision threshold?</summary>

From costs and review capacity: compute expected cost per 1,000 transactions across alert rates, as in the last table, and pick the minimum (2.0% there), then check that reviewers can handle that volume. Re-derive it when amounts, review cost or the fraud rate change.

</details>

## Further reading

- [Stripe, "How we built it: Stripe Radar" (29 March 2023)](https://stripe.dev/blog/how-we-built-it-stripe-radar): the under-100-ms decision, 1,000-plus characteristics and the move to a DNN-only model.
- [Dean and Barroso, "The Tail at Scale", Communications of the ACM, 2013](https://www.barroso.org/publications/TheTailAtScale.pdf): fan-out, hedged and tied requests.
- [Dal Pozzolo et al., "Credit Card Fraud Detection: a Realistic Modeling and a Novel Learning Strategy", IEEE TNNLS, 2018](https://boracchi.faculty.polimi.it/docs/2017_FraudsTNNLS.pdf): drift, imbalance, verification latency and alert precision.
- [Feast documentation, "Point-in-time joins"](https://docs.feast.dev/getting-started/concepts/point-in-time-joins): preventing leakage and skew in training data.
- Site chapters: [serving and release strategies](/docs/theory/seml/serving-and-release-strategies), [quality attributes](/docs/theory/seml/quality-attributes), [document Q&A design](/docs/senior/design-enterprise-document-qa), [ranking design](/docs/senior/design-search-and-recommendation-ranking).

## Check yourself

- I can build a latency budget and explain why summing p99 values is not the p99 of the sum.
- I can compute the chance that a fan-out of n lookups hits a slow one and say what parallelism, hedging and timeouts each buy.
- I can design a decision path that degrades to rules instead of blocking or failing open.
- I can explain training-serving skew and how point-in-time joins and shared feature code prevent it.
- I can evaluate a fraud model on a time-ordered split and report alert precision at a review capacity.
- I can explain how label delay and drift interact, and what a randomised holdout buys.
- I can choose a threshold by expected cost and write the annual cost formula.
