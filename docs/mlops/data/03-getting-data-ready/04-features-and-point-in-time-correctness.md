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

## The idea in plain words

A model cannot use raw business events directly unless they have been translated into inputs with a stable meaning. Feature preparation can encode categories, scale numeric fields, handle missing values and aggregate histories. A useful feature reflects information available at the decision point. A spectacular offline score is suspect if the training pipeline quietly uses a future event or computes a field differently from the live service.

The lecture's standardisation exercise is simple but important: for x = 80, training mean μ = 70 and training standard deviation σ = 5, the z-score is **(80 − 70)/5 = 2.0**. The statistics must be fitted on training data only and reused for validation, test and serving. Fitting again on live data changes what a unit of the feature means; fitting on validation or test leaks their distribution into training. Scaling is not required for every model, but consistency is required whenever it is used.

<Infographic src="/img/dm/feature-preparation.svg" alt="An input of eighty with training mean seventy and standard deviation five becomes a standardised value of two; training statistics are reused later." caption="Feature transforms carry learned state that belongs to the training set." />

<Infographic src="/img/dm/feature-time.svg" alt="A prediction on day four may use the day-three feature value eight; the day-five value twelve would be future leakage." caption="Historical retrieval asks what was known at the prediction cutoff, not what is latest now." />

A **feature store** can register feature definitions, retrieve historical values for training and serve recent values with low latency. Offline and online paths have different workloads: historical joins versus point lookup. Shared definitions and materialisation help reduce skew, but the store does not automatically make every feature correct. Entity keys, timestamps, transformation versions, late arrivals and freshness policies still need design and tests.

:::note Beyond the lecture

The source introduces encoding, scaling, aggregations, offline/online stores and point-in-time correctness. The sections below add feature availability time, freshness limits, skew tests and the distinction between event time and the moment a value became known.

:::

At the default prediction day **4**, the latest feature known by then is day **3**, value **8**. The current latest value **12** belongs to day 5 and would leak if used for the earlier prediction. Change the cutoff or age limit to see when no historical value is eligible.

<FeatureCutoffLab />

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

The first block reproduces the lecture's z-score and shows why the saved training statistics matter. It uses plain arithmetic so the convention is visible.

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

## Designing with it

### Choose a feature that exists at decision time

State the entity, value, event time, availability time and lookback window. For `payments_last_30d`, decide whether refunds count, which time zone defines a day and whether pending payments are included. A feature that uses the label window is leakage, even if its SQL is valid. Use the prediction timestamp, not the eventual label timestamp, as the feature cutoff. The lecture's practice answer uses "label's timestamp"; for a forward-looking label this should be read as the timestamp of the example or prediction, before the outcome window.

Categorical encoding also learns state. One-hot vocabularies and target encodings must be fitted inside the training split. An unseen category needs a declared behaviour at serving, such as an unknown bucket. Target encoding is particularly leakage-prone because it uses labels; use fold-aware estimation. Missing values may be informative, but imputers must use training statistics and a consistent live policy. A feature store can publish these definitions, yet a model package must still carry the fitted transformation state it needs.

### Compare offline and online paths

An offline store supports large historical joins. An online store supports recent keyed lookup. The two may be fed by different code, update schedules and data sources. **Training-serving skew** appears when the same feature name yields different values for the same entity and time. Build a parity test: replay known events through both paths, compare values within a stated tolerance and inspect differences by source and timestamp. Also monitor online feature age and missingness, not just lookup latency.

Point-in-time correctness is necessary but not sufficient. A historical query can choose the newest feature with event time before the prediction while that feature was only published later. If late publication matters, capture availability time and use the stricter cutoff. A backfill using corrected source history may be ideal for a current report but unsuitable for reconstructing what a live model actually saw months ago. Keep training snapshot and serving logs where auditability is needed.

### Keep freshness and fallback explicit

A time-to-live prevents a very old feature from masquerading as current. If no value falls within the permitted window, return missing and use a model-approved fallback or route to review. Do not quietly use a future value or the latest current value for an old example. Monitor how often the fallback fires by entity group and source, because a global low rate can hide a stalled feed for one region.

Feature stores help with discoverability, reuse and consistency, but they do not remove source ownership. Give each feature a definition, version, owner, freshness target, data-quality checks and affected models. Retire unused features and propagate deletes where required. The cost of a shared feature can be multiplied across many models, so changes need impact review.

## Reconstruct one fraud decision

A payment arrives at 14:00 on 10 June. The fraud model asks for account age, recent refund count and merchant risk. The account-age feature may come from an operational record; refund count may be a streaming aggregate; merchant risk may refresh nightly. Each value has an entity key, source time and publication time. The inference log should record which feature versions or values were actually used, within privacy constraints, so a later review can reconstruct the decision.

For historical training, the team creates an entity dataframe of payments at their decision timestamps. The feature lookup for each row searches backward to the latest eligible value for the right entity, bounded by a TTL. A query run today must not attach today's merchant risk to a payment from last year. If a historical value was corrected after the decision, decide whether the training objective is to learn from the best current reconstruction of the past or to simulate exactly what the live model knew. These are different datasets; name and version them separately.

Next, check the transform. The refund count might be computed from settled refunds offline but from requested refunds online. Both paths could pass schema tests and use the same feature name while disagreeing systematically. Replay a small set of event histories through each path and compare results at several cutoffs, including a late event, a duplicate and a refund reversal. The discrepancy is a semantic skew, not a numerical rounding problem.

Standardisation uses the training mean and standard deviation stored with the model package. If a live service refits those statistics on a recent batch, a raw value of 80 no longer maps to the training z-score of 2.0. The model's coefficients have not changed, but the meaning of its input has. A canary test should send a known raw record through training preprocessing and the live service and compare the final feature vector.

Finally, monitor retrieval health. An online store may be up while its materialisation job is late, so keyed reads return stale values quickly. Track feature age, missing values and the percentage of predictions using fallbacks. When a source changes, identify every model using the feature. A shared store makes reuse easier; it also makes a wrong shared definition a broad incident unless ownership and lineage are clear.

### Test the two time axes

Consider a feature event stamped day 3 that reaches the offline table on day 5. An event-time-only historical join for a day-4 prediction may select it, yet the live service could not have used it. If the goal is to reproduce live decisions, the historical query also needs availability time ≤ day 4. If the goal is to model the best later-known state, the query can use the corrected event but should label that dataset differently. Neither choice is automatically wrong; confusing them produces misleading offline evaluation.

Materialisation has its own clock. A feature source may be current at 10:00 while the online store last loaded it at 09:30. An inference at 10:05 sees a stale value despite a correct offline query. Monitor source watermark, online materialisation watermark and lookup time separately. Test a record at the boundary of the TTL: if the feature is exactly three days old, decide whether it remains eligible and implement the same rule in training and serving. A one-second discrepancy at a boundary can affect a large batch of scheduled predictions.

Finally, test transforms on adversarial examples: an unknown category, a missing value, a late event, a duplicate event and a value beyond the training range. The offline and online feature vectors should agree for the same entity and cutoff under the same source snapshot. If they do not, log the differing inputs and transform versions. This is a stronger test than checking only that both sides return a number. Repeat it after every feature-definition release.

## Where this stands in 2026

:::info Industry view

- Feature stores such as Feast separate historical retrieval from low-latency online lookup and document point-in-time joins.
- A feature view and online materialisation can reduce duplicated definitions, but event-time, availability-time and skew tests remain the team's responsibility.
- Training-only transform statistics and explicit unknown-category handling are still basic safeguards in modern ML systems.

:::

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

## Go deeper

- [Feast feature views](https://docs.feast.dev/getting-started/concepts/feature-view) describes shared schema and sources.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins) explains historical retrieval and TTL.
- [Feast online store](https://docs.feast.dev/getting-started/components/online-store) describes latest-value serving.
- [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) specifies training statistics reused at transform time.
- Built from the course lecture "dm-l11-feature-preparation" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can compute the lecture's z-score of 2.0 and retain the training statistics.
- [ ] I can distinguish an offline historical join from an online latest-value lookup.
- [ ] I can enforce a prediction cutoff with event time, availability time and TTL.
- [ ] I can test train-serving feature parity and identify a stale online feature.
