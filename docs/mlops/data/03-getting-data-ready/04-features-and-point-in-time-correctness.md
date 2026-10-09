---
id: dm-features-and-point-in-time-correctness
title: "Data Management · Lecture 11 — Features and Point-in-Time Correctness"
sidebar_label: "11 · Feature preparation"
sidebar_position: 4
slug: /mlops/data/features-and-point-in-time
description: "Engineer and standardise features, then retrieve historical and online values without future leakage or training-serving skew."
tags: [data-management, feature-engineering, feature-store, point-in-time]
---

import Infographic from '@site/src/components/Infographic';
import FeatureCutoffLab from '@site/src/components/viz/FeatureCutoffLab';

**In one line.** A feature is useful only when its definition and time boundary are the same in training and serving.

:::tip Before you start

**You should already know**

- What a join on a key is, and what a timestamp column is ([Lecture 9, analytics engineering and history](/docs/mlops/data/analytics-engineering-history)).
- What an ROC AUC score measures: how well a score ranks positives above negatives ([Model evaluation](/docs/theory/ml/model-evaluation)).

**Reading time:** about 40 minutes, plus a minute to run the code.

**After this chapter you can**

- Pick the feature value that was known at a prediction time, and say why "latest" is wrong.
- Explain why an event-time cutoff alone can still leak, and add an availability cutoff.
- Read an offline score against a served score and know which one to believe.

:::

## In 30 seconds

A feature is a number the model reads before it decides. A fair test asks: what did the model know at that moment? If your training table quietly uses a number that was only filled in afterwards, the model looks brilliant on paper and fails in service.

Think of marking an exam where the answer sheet was stapled to the paper. The marks are high and mean nothing. Point-in-time correctness is removing the answer sheet.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Feature | An input number built from raw events | Refunds in the last 30 days |
| Prediction cutoff | The moment the model must decide | Day 4 |
| Event time | When the thing happened | The refund was on day 3 |
| Availability time | When the feature store could first serve it | Published on day 5 |
| Point-in-time join | Pick, for each row, the newest value known at its cutoff | Day 4 gets the day-3 value |
| Time to live (TTL) | How old a value may be and still count | 3 days |
| Training-serving skew | The same feature differs between training and live use | Offline 12, online 8 |
| Standardise | Subtract the training mean, divide by the training spread | (80 - 70) / 5 = 2.0 |


## The idea in plain words

A model cannot use raw business events directly unless they have been translated into inputs with a stable meaning. Feature preparation can encode categories, scale numeric fields, handle missing values and aggregate histories. A useful feature reflects information available at the decision point. A spectacular offline score is suspect if the training pipeline quietly uses a future event or computes a field differently from the live service.

The standardisation exercise is simple but important: for x = 80, training mean μ = 70 and training standard deviation σ = 5, the z-score is **(80 − 70)/5 = 2.0**. The statistics must be fitted on training data only and reused for validation, test and serving. Fitting again on live data changes what a unit of the feature means; fitting on validation or test leaks their distribution into training. Scaling is not required for every model, but consistency is required whenever it is used.

<Infographic src="/img/dm/feature-preparation.svg" alt="An input of eighty with training mean seventy and standard deviation five becomes a standardised value of two; training statistics are reused later." caption="Feature transforms carry learned state that belongs to the training set." />

<Infographic src="/img/dm/feature-time.svg" alt="A prediction on day four may use the day-three feature value eight; the day-five value twelve would be future leakage." caption="Historical retrieval asks what was known at the prediction cutoff, not what is latest now." />

A **feature store** can register feature definitions, retrieve historical values for training and serve recent values with low latency. Offline and online paths have different workloads: historical joins versus point lookup. Shared definitions and materialisation help reduce skew, but the store does not automatically make every feature correct. Entity keys, timestamps, transformation versions, late arrivals and freshness policies still need design and tests.

:::note Added for this site

The course material covers encoding, scaling, aggregations, offline and online stores and point-in-time correctness. The sections below add feature availability time, freshness limits, skew tests and the distinction between event time and the moment a value became known.

:::

At the default prediction day **4**, the latest feature known by then is day **3**, value **8**. The current latest value **12** belongs to day 5 and would leak if used for the earlier prediction. Change the cutoff or age limit to see when no historical value is eligible.

<FeatureCutoffLab />

## Worked example, step by step

A fraud model scores a payment on day 4. Published `refund_count_30d` values: 4 on day 1, 8 on day 3, 12 on day 5. Raw feature x = 80 has training mean 70 and standard deviation 5.

1. **Standardise.** z = (80 - 70) / 5 = 2.0. If a live batch with mean 75 were used instead, z = (80 - 75) / 5 = 1.0. The same raw value now means something else.
2. **Latest-value join.** The newest value is 12 (day 5). It did not exist on day 4, so this join leaks the future.
3. **Event-time cutoff.** Keep rows with event day ≤ 4. That leaves day 1 (4) and day 3 (8). The newest is 8. Its age is 4 - 3 = 1 day, within a TTL of 3, so the answer is 8.
4. **Availability cutoff.** Suppose the day-3 value was published only on day 5. Now day 3 is not available on day 4. Day 1 is: its age is 4 - 1 = 3, which is within the TTL of 3, so the answer is 4.
5. **No value.** With TTL 0 nothing qualifies and the lookup returns missing.

In words: the model may use only what existed and had been published by its cutoff, and not too long ago. The code block "The worked example in code" below prints steps 1, 3 and 4.

## How it works

### Encode & scale

Encode categoricals (one-hot, target, embeddings), scale numerics (standardise, min-max), handle missing values, build aggregations.

:::tip

**Worked.** x=80, μ=70, σ=5 → z = (80−70)/5 = 2.0 (fit scaler on train only).

:::

### Offline, online, PIT

A feature store centralises features: offline (historical, training) + online (low-latency, serving). It can support point-in-time joins and shared feature definitions when timestamps, materialisation and transformation paths are configured and tested correctly.


## A real system that works this way

**Feast** is a concrete feature-store example. Its documentation describes feature views, historical retrieval with point-in-time joins and an online store that keeps the latest feature value per entity for low-latency serving. In its historical retrieval, an entity row's timestamp and a feature view's time-to-live bound which earlier feature record may join. The online store is not a historical archive. It must be materialised or updated from a source, and serving freshness can differ from offline table freshness.

Imagine a fraud model deciding on a payment at day 4. The customer's `refund_count_30d` has published values 4 on day 1, 8 on day 3 and 12 on day 5. An offline training join for the day-4 payment must pick 8, assuming the day-3 value was actually available by day 4. An online lookup performed on day 6 returns 12. Both operations are correct for their times, but using today's online value to rebuild the old example would leak future information.

A real feature may be computed late. If the day-3 event was published to the feature store only on day 5, even its event time of day 3 is too early for a day-4 prediction. Track both event time and availability time where this matters. Feast's point-in-time event-time join is a useful mechanism, but a team must align ingestion and publication semantics to the real decision boundary.

## Code you can run

The first block reproduces the z-score of 2.0 and shows why the saved training statistics matter. It uses plain arithmetic so the convention is visible.

```python
training_mean = 70
training_std = 5
value = 80
z = (value - training_mean) / training_std
print(f"z = ({value} - {training_mean}) / {training_std} = {z:.1f}")
assert z == 2.0

later_batch_mean = 75
assert (value - later_batch_mean) / training_std != z
```

The second block implements a small as-of lookup with an explicit availability time and age limit. The day-3 value is eligible for a day-4 prediction only if it was published by then. This adds a guarantee beyond a simple event-time comparison.

```python
features = [
    {"event_day": 1, "available_day": 1, "value": 4},
    {"event_day": 3, "available_day": 3, "value": 8},
    {"event_day": 5, "available_day": 5, "value": 12},
]

def as_of(prediction_day, ttl=3):
    eligible = [
        row for row in features
        if row["event_day"] <= prediction_day
        and row["available_day"] <= prediction_day
        and prediction_day - row["event_day"] <= ttl
    ]
    return max(eligible, key=lambda row: row["event_day"])["value"] if eligible else None

print("day 4:", as_of(4), "current latest:", features[-1]["value"])
assert as_of(4) == 8
assert as_of(6) == 12
assert as_of(4, ttl=0) is None
```

If the day-3 feature's availability changes to day 5, the day-4 lookup must fall back to day 1 under the three-day TTL. A production feature system needs to test that rule with late data and the actual publication timestamps.

### The worked example in code

This block runs steps 1, 3 and 4 of the worked example, including the late-published day-3 value.

```python
def z(value, mean, std):
    return (value - mean) / std

def as_of(rows, day, ttl=3, use_availability=True):
    ok = [
        r for r in rows
        if r["event_day"] <= day
        and (not use_availability or r["available_day"] <= day)
        and day - r["event_day"] <= ttl
    ]
    return max(ok, key=lambda r: r["event_day"])["value"] if ok else None

on_time = [(1, 1, 4), (3, 3, 8), (5, 5, 12)]
late = [(1, 1, 4), (3, 5, 8), (5, 5, 12)]
build = lambda rows: [dict(zip(("event_day", "available_day", "value"), r)) for r in rows]
print("z", z(80, 70, 5), "with batch mean 75", z(80, 75, 5))
print("event-time cutoff, on-time feed", as_of(build(on_time), 4, use_availability=False))
print("event-time cutoff, late feed   ", as_of(build(late), 4, use_availability=False))
print("availability cutoff, late feed ", as_of(build(late), 4))
print("ttl 0", as_of(build(on_time), 4, ttl=0))
```

**Reading the output.** It prints z 2.0 and 1.0, then 8, 8 and 4, then None. The second 8 is the point: with a late feed, an event-time cutoff still returns the day-3 value that was not yet published on day 4.

### An experiment on leakage

How much does the join rule change the score, and does the "correct" event-time rule survive late data? The block builds 4,000 users with a daily feature, a decision day for each, and a binary outcome. For users whose outcome is positive, the feature is bumped by 6 after the decision day, and the decision-day value is bumped by 7 and published two days late. That models a case review that updates the record after the fact.

It trains a logistic regression on three versions of the training table: the latest value, an event-time `merge_asof`, and an availability-time `merge_asof` with a tolerance of 5 days. Every model is scored twice on 1,000 held-out users: on its own kind of table (offline AUC) and on the availability table, which is what the live service could see (served AUC). A last step refits a standard scaler on a drifted live batch.

Versions used: Python 3.14.6, pandas 2.3.3, scikit-learn 1.9.1, NumPy 2.5.3. The data is synthetic and the late-publication rule is mine. It runs in about two seconds.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(10)
n_users, days = 4000, 60
risk = rng.normal(size=n_users)
decision = rng.integers(20, 50, n_users)
label = (rng.random(n_users) < 1 / (1 + np.exp(-(0.9 * risk - 1.0)))).astype(int)

rows = []
for u in range(n_users):
    level = 5 + 2 * risk[u]
    for d in range(days):
        value = max(0.0, level + rng.normal(0, 1.5))
        if label[u] and d > decision[u]:
            value += 6
        late = label[u] and d == decision[u]
        rows.append((u, d, d + 2 if late else d, value + 7 if late else value))
events = pd.DataFrame(rows, columns=["user", "event_day", "available_day", "value"])

examples = pd.DataFrame({"user": np.arange(n_users), "t": decision, "y": label})
examples["t"] = examples["t"].astype(int)

def join(kind):
    left = examples.sort_values("t")
    if kind == "latest":
        last = events.sort_values("event_day").groupby("user").tail(1)[["user", "value"]]
        return left.merge(last, on="user").sort_values("user")
    key = "event_day" if kind == "event-time" else "available_day"
    right = events.sort_values(key)
    out = pd.merge_asof(left, right[["user", key, "value"]], left_on="t", right_on=key, by="user", tolerance=5)
    return out.sort_values("user")

tables = {k: join(k) for k in ("latest", "event-time", "availability")}
print({k: int(v.value.isna().sum()) for k, v in tables.items()})

train_users = np.arange(n_users) < 3000
test_users = ~train_users
served = tables["availability"]
print("label rate", round(label.mean(), 3))
print(f"{'training join':14}{'offline AUC':>12}{'served AUC':>12}")
for kind, table in tables.items():
    x = table[["value"]].to_numpy()
    y = table.y.to_numpy()
    model = LogisticRegression().fit(x[train_users], y[train_users])
    offline = roc_auc_score(y[test_users], model.predict_proba(x[test_users])[:, 1])
    xs = served[["value"]].to_numpy()
    live = roc_auc_score(y[test_users], model.predict_proba(xs[test_users])[:, 1])
    print(f"{kind:14}{offline:12.3f}{live:12.3f}")

x = served[["value"]].to_numpy()
y = served.y.to_numpy()
scaler = StandardScaler().fit(x[train_users])
model = LogisticRegression().fit(scaler.transform(x[train_users]), y[train_users])
drifted = x[test_users] + 2.0
kept = model.predict_proba(scaler.transform(drifted))[:, 1]
refit = model.predict_proba(StandardScaler().fit_transform(drifted))[:, 1]
print("true rate", round(y[test_users].mean(), 3), "mean prob training scaler", round(kept.mean(), 3), "refit on live batch", round(refit.mean(), 3))
print("AUC training scaler", round(roc_auc_score(y[test_users], kept), 3), "refit", round(roc_auc_score(y[test_users], refit), 3))
```

The output of the run:

```text
{'latest': 0, 'event-time': 0, 'availability': 0}
label rate 0.292
training join  offline AUC  served AUC
latest               0.991       0.678
event-time           0.995       0.678
availability         0.678       0.678
true rate 0.31 mean prob training scaler 0.4 refit on live batch 0.286
AUC training scaler 0.678 refit 0.678
```

**Reading the output.** No join left a missing value. The latest-value and event-time tables score 0.991 and 0.995 offline and 0.678 once served. The availability table scores 0.678 in both places, so offline equals served. The last two lines show that the scaler choice moves the mean predicted probability (0.400 against 0.286) but leaves AUC at 0.678.

**Line by line.**

- `rows.append((u, d, d + 2 if late else d, value + 7 if late else value))` stores both the event day and the day the value became available. Only the third column differs between the two time rules.
- `pd.merge_asof(..., by="user", tolerance=5)` takes, for each decision day, the latest right row at or before it for the same user, and returns missing if the row is more than 5 days old.
- `key = "event_day" if kind == "event-time" else "available_day"` is the only difference between the last two training tables.

### What the numbers say

A model trained on the latest-value table scored 0.991 offline and 0.678 in service, a drop of 0.313. Nothing was wrong with the model code. The table contained the outcome.

The event-time join is the surprise. It is what the textbook point-in-time join does, and it scored 0.995 offline and also 0.678 served. Because the decision-day value was published two days late, the cutoff on event time let the future through. Only the availability cutoff gave a trustworthy offline number: 0.678, equal to what service would see.

The scaler result is a quieter lesson. Every live value was shifted by 2 and the true rate stayed 0.310. The training scaler read the shift as higher risk and predicted 0.400 on average. The refitted scaler hid the shift and predicted 0.286. AUC stayed at 0.678 either way, because standardising does not change the ranking. Which average is right depends on whether the shift is real or a pipeline fault, and a refit makes that question impossible to see.

Limits: one synthetic dataset, one seed, one feature, and a leak I constructed. The size of the gap depends on how strongly the outcome feeds back into the feature.

<Infographic src="/img/dm-enrich/dm2-pit-auc.svg" alt="Paired bars of offline and served AUC for three training joins: latest value 0.991 and 0.678, event-time 0.995 and 0.678, availability 0.678 and 0.678." caption="Look first at the gap between each pair of bars: only the availability join has none." />

## Designing with it

### Choose a feature that exists at decision time

State the entity, value, event time, availability time and lookback window. For `payments_last_30d`, decide whether refunds count, which time zone defines a day and whether pending payments are included. A feature that uses the label window is leakage, even if its SQL is valid. Use the prediction timestamp, not the eventual label timestamp, as the feature cutoff. A practice answer elsewhere says "label's timestamp"; for a forward-looking label read that as the timestamp of the example or prediction, before the outcome window.

Categorical encoding also learns state. One-hot vocabularies and target encodings must be fitted inside the training split. An unseen category needs a declared behaviour at serving, such as an unknown bucket. Target encoding is particularly leakage-prone because it uses labels; use fold-aware estimation. Missing values may be informative, but imputers must use training statistics and a consistent live policy. A feature store can publish these definitions, yet a model package must still carry the fitted transformation state it needs.

### Compare offline and online paths

An offline store supports large historical joins. An online store supports recent keyed lookup. The two may be fed by different code, update schedules and data sources. **Training-serving skew** appears when the same feature name yields different values for the same entity and time. Build a parity test: replay known events through both paths, compare values within a stated tolerance and inspect differences by source and timestamp. Also monitor online feature age and missingness, not just lookup latency.

Point-in-time correctness is necessary but not sufficient. A historical query can choose the newest feature with event time before the prediction while that feature was only published later. If late publication matters, capture availability time and use the stricter cutoff. A backfill using corrected source history may be ideal for a current report but unsuitable for reconstructing what a live model actually saw months ago. Keep training snapshot and serving logs where auditability is needed.

### Keep freshness and fallback explicit

A time-to-live prevents a very old feature from masquerading as current. If no value falls within the permitted window, return missing and use a model-approved fallback or route to review. Do not quietly use a future value or the latest current value for an old example. Monitor how often the fallback fires by entity group and source, because a global low rate can hide a stalled feed for one region.

Feature stores help with discoverability, reuse and consistency, but they do not remove source ownership. Give each feature a definition, version, owner, freshness target, data-quality checks and affected models. Retire unused features and propagate deletes where required. The cost of a shared feature can be multiplied across many models, so changes need impact review.

## Reconstruct one fraud decision

A payment arrives on 10 June, at two in the afternoon. The fraud model asks for account age, recent refund count and merchant risk. The account-age feature may come from an operational record; refund count may be a streaming aggregate; merchant risk may refresh nightly. Each value has an entity key, source time and publication time. The inference log should record which feature versions or values were actually used, within privacy constraints, so a later review can reconstruct the decision.

For historical training, the team creates an entity dataframe of payments at their decision timestamps. The feature lookup for each row searches backward to the latest eligible value for the right entity, bounded by a TTL. A query run today must not attach today's merchant risk to a payment from last year. If a historical value was corrected after the decision, decide whether the training objective is to learn from the best current reconstruction of the past or to simulate exactly what the live model knew. These are different datasets; name and version them separately.

Next, check the transform. The refund count might be computed from settled refunds offline but from requested refunds online. Both paths could pass schema tests and use the same feature name while disagreeing systematically. Replay a small set of event histories through each path and compare results at several cutoffs, including a late event, a duplicate and a refund reversal. The discrepancy is a semantic skew, not a numerical rounding problem.

Standardisation uses the training mean and standard deviation stored with the model package. If a live service refits those statistics on a recent batch, a raw value of 80 no longer maps to the training z-score of 2.0. The model's coefficients have not changed, but the meaning of its input has. A canary test should send a known raw record through training preprocessing and the live service and compare the final feature vector.

Finally, monitor retrieval health. An online store may be up while its materialisation job is late, so keyed reads return stale values quickly. Track feature age, missing values and the percentage of predictions using fallbacks. When a source changes, identify every model using the feature. A shared store makes reuse easier; it also makes a wrong shared definition a broad incident unless ownership and lineage are clear.

### Test the two time axes

Consider a feature event stamped day 3 that reaches the offline table on day 5. An event-time-only historical join for a day-4 prediction may select it, yet the live service could not have used it. If the goal is to reproduce live decisions, the historical query also needs availability time ≤ day 4. If the goal is to model the best later-known state, the query can use the corrected event but should label that dataset differently. Neither choice is automatically wrong; confusing them produces misleading offline evaluation.

Materialisation has its own clock. A feature source may be current as of 10:00 while the online store last loaded it as of 09:30. An inference five minutes after that sees a stale value despite a correct offline query. Monitor source watermark, online materialisation watermark and lookup time separately. Test a record at the boundary of the TTL: if the feature is exactly three days old, decide whether it remains eligible and implement the same rule in training and serving. A one-second discrepancy at a boundary can affect a large batch of scheduled predictions.

Finally, test transforms on adversarial examples: an unknown category, a missing value, a late event, a duplicate event and a value beyond the training range. The offline and online feature vectors should agree for the same entity and cutoff under the same source snapshot. If they do not, log the differing inputs and transform versions. This is a stronger test than checking only that both sides return a number. Repeat it after every feature-definition release.

## Where this stands in 2026

:::info Industry view

- Feature stores such as Feast separate historical retrieval from low-latency online lookup and document point-in-time joins.
- A feature view and online materialisation can reduce duplicated definitions, but event-time, availability-time and skew tests remain the team's responsibility.
- Training-only transform statistics and explicit unknown-category handling are still basic safeguards in modern ML systems.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Training on the latest feature value | It is the easy join and the newest data is "best" | Join per row at its cutoff. The latest-value table scored 0.991 offline and 0.678 served |
| Trusting an event-time cutoff alone | It is the standard point-in-time rule | Add an availability cutoff. The event-time join scored 0.995 offline and 0.678 served, because a late-published value slipped through |
| Believing a very high offline AUC | A better score is good news | Treat a jump as a leak alarm. Replay the cutoff on a sample of rows by hand |
| Refitting the scaler on live data | The live data is the freshest | Ship the training mean and spread with the model. A refit hid a shift of 2 and changed the mean prediction from 0.400 to 0.286 |
| Never testing the TTL boundary | It is a corner case | Decide whether a value exactly TTL old counts, and use the same rule in training and serving |

## Practice questions

<details>
<summary><strong>Q1.</strong> What is feature engineering and why does it matter?</summary>

Creating the model's inputs; encoding categoricals, scaling numerics, handling missing values, building aggregations. Good features often matter more than the model choice.<br /><em>Lecture 11 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Standardise x=80 given μ=70, σ=5.</summary>

z = (x−μ)/σ = (80−70)/5 = 2.0; two standard deviations above the mean.<br /><em>Lecture 11 · numeric</em>

</details>

<details>
<summary><strong>Q3.</strong> Why fit a scaler on training data only?</summary>

Fitting on val/test/all data leaks their statistics into training (data leakage); using training μ,σ at serving also avoids training–serving skew.<br /><em>Lecture 11 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> What are the offline and online stores in a feature store?</summary>

Offline = historical features for training (high throughput); online = low-latency features for real-time serving.<br /><em>Lecture 11 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is point-in-time correctness?</summary>

Using only feature values known at the prediction or example cutoff, before the outcome window, so future information cannot leak into training.<br /><em>Lecture 11 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) A model scores 0.99 AUC offline and 0.68 in production. Name two checks you would run on the training join before touching the model.</summary>

First, check that every feature row used had an event time at or before the example's cutoff. Second, check that its availability time was also at or before the cutoff, since late publication can pass the first check. Then recompute the offline score on a table built from availability time. In the experiment this brought 0.995 back to 0.678, which matched production.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) A scaler refitted on a live batch leaves AUC unchanged. Why is that still a problem?</summary>

Standardising is a monotone transformation of the input, so the ranking, and therefore AUC, does not change. The probabilities do change: the same model predicted a mean of 0.400 with the training scaler and 0.286 with a refit scaler on the same drifted batch. A refit also removes the drift signal from the model, so a real shift in the data stops being visible in the predictions.

</details>

## Go deeper

- [Feast feature views](https://docs.feast.dev/getting-started/concepts/feature-view) describes shared schema and sources.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) explains historical retrieval and TTL.
- [Feast online store](https://docs.feast.dev/getting-started/components/online-store) describes latest-value serving.
- [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) specifies training statistics reused at transform time.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins), opened 2026-10-09: the join looks back from each row's timestamp, TTL bounds the lookback, and by default only the event timestamp constrains the result. A created-timestamp filter must be enabled to stop later-published values leaking.
- [pandas merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html), opened 2026-10-09: backward matching by default, with `by`, `tolerance` and a sorted-key requirement. The page is for pandas 3.0.6; the run above used pandas 2.3.3.
- Built from the course lecture "dm-l11-feature-preparation" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can compute the z-score of 2.0 and retain the training statistics.
- [ ] I can distinguish an offline historical join from an online latest-value lookup.
- [ ] I can enforce a prediction cutoff with event time, availability time and TTL.
- [ ] I can test train-serving feature parity and identify a stale online feature.
- [ ] I can pick the feature value for a day-4 prediction under an event-time rule and under an availability rule, and show they can differ.
- [ ] I can explain why an event-time point-in-time join scored 0.995 offline and 0.678 served in the experiment.
- [ ] I can say why refitting a scaler on live data leaves AUC unchanged yet changes the predicted probabilities.

## Where to go next

Next: [Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery), which schedules the jobs that make feature values available. Related: [Lecture 9, analytics engineering and history](/docs/mlops/data/analytics-engineering-history), where the same as-of join picks a customer's address.
