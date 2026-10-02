---
id: dm-midsem-solved
title: "Data Management; 2026 Mid-Semester Paper, Solved"
sidebar_label: "2 · Solved mid-sem"
sidebar_position: 2
slug: /mlops/data/midsem-solved
description: "Three scenario questions on healthcare data quality, media data architecture and fraud processing with worked solutions and corrections."
tags: [data-management, practice, mid-semester]
---

import Infographic from '@site/src/components/Infographic';
import RepresentationGapLab from '@site/src/components/viz/RepresentationGapLab';

**In one line.** Solve the three scenario questions from the 2026 mid-semester paper, then check the assumptions in the supplied design answers.

## The paper at a glance

This closed-book paper asks about HealthPredict's healthcare data, StreamFlow's recommendation data and PayFast's fraud architecture. The source includes scanned question-paper images that are unavailable in the converted text. The original board below redraws the facts required by the text questions: demographic shares, missingness, clickstream volume, timing, historical replay and transaction consistency. It does not reproduce the scans.

<Infographic src="/img/dm/midsem-practice.svg" alt="Three exam scenarios compare 10% versus 40% rural shares, 500 GB per day of clickstream, and real-time fraud decisions with historical replay." caption="Name the workload, data contract and failure cost before choosing an architecture." />

:::note Beyond the lecture

The independent checks, representation lab and corrections below are added to test the supplied answers. The source question text and worked answers remain in the answer panels.

:::

## Original questions and worked answers

<details>
<summary><strong>Q1.</strong> HealthPredict data quality & validation Training data: only 10% rural patients (40% of real patients are rural); model accuracy drops after 6 months; some groups have higher false-negative rates; some EHR fields have 30% missing. (a) Three data-quality problems. (b) Two drift-detection metrics. (c) Pre-ingestion vs pre-training validation. (d) A data-validation framework.</summary>

(a) **Sampling bias / fairness** (rural under-represented 10% vs 40% → biased model), **data drift** (accuracy decays after 6 months as the population shifts), and **missing values** (30% null EHR fields; plus the higher FN rate signals a fairness/label-quality problem). (b) **Population Stability Index (PSI)** and **KL-divergence / KS-test** between training and live feature distributions; flag retraining when they exceed a threshold (e.g. PSI>0.2). (c) **Pre-ingestion** validation checks raw data as it arrives (schema, types, ranges, nulls); it would catch the rural under-representation by comparing incoming demographics to expected proportions; **pre-training** validation checks the assembled training set (label balance, leakage, drift) before fitting. (d) Framework: monitor quality dimensions (completeness, consistency, accuracy, timeliness, fairness); automated data tests at each pipeline stage; explicit pre-ingestion vs pre-training checks for healthcare (PII handling, valid clinical codes, demographic balance).

:::note Correction to the source answer

The training sample has 10% rural patients against 40% in the deployment population, a 30 percentage-point gap and one-quarter of the deployment share. This supports an under-representation finding. An accuracy drop alone does not prove input drift; inspect time-based validation, labels, prevalence and group-specific false negatives. A PSI threshold is an investigation trigger, never an automatic retraining instruction. Pre-ingestion checks may compare incoming demographics with a reference, but the assembled training sample needs its own representativeness check.

:::
</details>

<details>
<summary><strong>Q2.</strong> StreamFlow video-recommendation data architecture User history (structured, PostgreSQL), clickstream (semi-structured JSON, 500 GB/day), video metadata/thumbnails (unstructured); 60% of time lost to integration/cleaning; no data versioning. (a) A 4-layer data architecture. (b) Warehouse vs lake vs lakehouse; which fits? (c) How a feature store prevents leakage and ensures point-in-time-correct features.</summary>

(a) **Ingestion layer** (stream clickstream + batch DB/metadata), **storage layer** (raw zone for JSON/video + curated zone), **processing/transformation layer** (clean, join, feature engineering; Spark), **serving layer** (feature store + model training/inference). (b) **Lakehouse** fits best: it stores unstructured video/JSON cheaply (lake) while giving ACID, schema management and **data versioning** (warehouse-like) needed for reproducible ML; pure warehouse can't hold raw video, pure lake lacks governance/versioning. (c) A **feature store** serves the *same* feature definitions to training and inference and supports **point-in-time joins** (only data available before the prediction timestamp), preventing future information (e.g. a user's later watch behaviour) from leaking into training.

:::note Correction to the source answer

A lakehouse is a reasonable design for the stated needs, not the only one. Warehouses can store or reference raw media and lakes can use governance and versioned table formats. A feature store supports historical joins but cannot automatically prevent leakage. Use event time, availability time, a prediction cutoff, suitable TTL and online/offline parity tests.

:::
</details>

<details>
<summary><strong>Q3.</strong> PayFast fraud big-data architecture Millions of transactions/day; needs real-time fraud detection (sub-second), historical backtesting for compliance/training, high availability in peak seasons; current monolithic relational DB can't cope. (a) Big-data architecture (ingestion/fraud/historical/storage). (b) Lambda vs Kappa; justify. (c) ACID properties with examples. (d) Where BASE could be used vs ACID, and the trade-off.</summary>

(a) **Streaming ingestion** (Kafka) → **real-time fraud scoring** (stream processor + model) → **batch historical analysis** (data lake + Spark for backtesting) → **storage** (scalable store for raw + features). (b) **Lambda** architecture fits: it runs a *speed layer* for sub-second fraud alerts *and* a *batch layer* for accurate historical backtesting/compliance; Kappa (stream-only) is simpler but reprocessing years of data for compliance is awkward. (c) **ACID:** Atomicity (a transfer debits and credits or neither), Consistency (balances never violate rules), Isolation (concurrent transactions don't interfere), Durability (a committed payment survives a crash). (d) **BASE** (eventual consistency) suits non-critical, high-volume paths; transaction-history display or an analytics dashboard; for availability/scale; keep **ACID** for the core money-moving transactions. Trade-off: BASE gives availability/throughput at the cost of temporary staleness.

:::note Correction to the source answer

Lambda is a valid choice when separate batch and speed paths are acceptable. Kappa can replay retained or archived event history for backtesting, so it is not inherently unable to handle historical analysis. Keep core money-moving transactions strongly consistent; downstream dashboards may tolerate stale views only within an explicit product and compliance contract.

:::
</details>

## Code you can run

### Check the scenario quantities

The first block verifies the two explicit percentages in HealthPredict. The model's group error rates are not numerically specified, so no false-negative-rate gap can be calculated from the question.

```python
training_rural = 0.10
deployment_rural = 0.40
representation_gap = deployment_rural - training_rural
relative_coverage = training_rural / deployment_rural
missing_ehr = 0.30
print(f"Rural-share gap: {representation_gap * 100:.0f} percentage points")
print(f"Training share as fraction of deployment share: {relative_coverage:.2f}")
print(f"EHR field missingness: {missing_ehr:.0%}")
assert (round(representation_gap, 2), relative_coverage, missing_ehr) == (0.30, 0.25, 0.30)
```

Change the two shares in the lab. Its default **30 percentage-point gap** matches the printed HealthPredict result. A gap calls for analysis of sampling and group outcomes; it does not identify why overall accuracy changed.

<RepresentationGapLab />

The second block illustrates StreamFlow's point-in-time rule. An event must have been both generated and made available before the prediction cutoff. A late-arriving watch event with an old event timestamp must still be excluded from the historical feature at that cutoff.

```python
from datetime import datetime, timezone

cutoff = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
events = [
    ("early watch", datetime(2026, 1, 10, 11, tzinfo=timezone.utc), datetime(2026, 1, 10, 11, 5, tzinfo=timezone.utc)),
    ("late arrival", datetime(2026, 1, 10, 11, 30, tzinfo=timezone.utc), datetime(2026, 1, 10, 12, 5, tzinfo=timezone.utc)),
    ("future watch", datetime(2026, 1, 10, 12, 30, tzinfo=timezone.utc), datetime(2026, 1, 10, 12, 31, tzinfo=timezone.utc)),
]
available = [name for name, event_time, available_time in events if event_time <= cutoff and available_time <= cutoff]
print("Available at prediction cutoff:", available)
assert available == ["early watch"]
```

## Design checks after the paper

For HealthPredict, distinguish sampling imbalance, missing fields, observed group error disparity and unproven drift. For StreamFlow, specify the raw and curated stores, versioned datasets and exact feature availability rule. For PayFast, separate the atomic payment record from lower-risk projections, and state how the chosen event retention supports backfills and audits. These distinctions make the same questions useful beyond one named product or architecture.

## Check yourself

- I can explain why a 30 percentage-point sample gap is an observation, while drift remains a hypothesis.
- I can state the event-time and availability-time conditions for a historical feature.
- I can explain where strong transaction consistency is required and where a stale derived view may be acceptable.
