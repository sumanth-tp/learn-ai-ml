---
id: dm-profiling-validation-and-drift
title: "Data Management · Session 8 — Profiling, Validation and Drift"
sidebar_label: "8 · Profile and validate"
sidebar_position: 2
slug: /mlops/data/profiling-validation-drift
description: "Profile data, enforce explicit rules and interpret population shift without treating one PSI threshold as a retraining command."
tags: [data-management, data-profiling, data-validation, data-drift]
---

import Infographic from '@site/src/components/Infographic';
import ProfilingDriftLab from '@site/src/components/viz/ProfilingDriftLab';

**In one line.** Profile to discover surprises, validate known contracts and investigate distribution changes in context.

## The idea in plain words

A pipeline can finish successfully while its data is unusable. A column may become mostly null, a category may change spelling, a source may stop sending events or the population may shift. These failures need different responses. **Profiling** describes what arrived, **validation** checks it against declared rules, and **drift monitoring** compares it with a reference. None of them alone proves that a model is accurate.

The lecture profiles per-column type, null rate, cardinality, range, quantiles and distribution. A profile is a snapshot of observations, not a judgement. A 15% null rate may be normal for an optional middle name and unacceptable for a required model label. Validation turns a consumer requirement into an assertion: a schema, non-null field, allowed value set, uniqueness condition or relationship. A failure needs a named action such as block, quarantine, warn or investigate.

<Infographic src="/img/dm/profiling-validation.svg" alt="A profile finds 1,500 nulls among 10,000 rows, or 15 per cent; a five per cent null ceiling fails, while PSI is a separate measure of population shift." caption="Missingness, contract failure and distribution change are different observations." />

The lecture's worked batch has 1,500 nulls among 10,000 rows, so the null rate is **0.15**, or **15%**. It fails a 5% ceiling. The source says to quarantine the batch, but the right action depends on the dataset contract and consumer cost. A critical label may justify blocking all publication; malformed optional rows may be isolated while unaffected rows continue. Preserve the failed rows and the decision so a repair can be replayed.

:::note Beyond the lecture

The source introduces PSI bands of 0.1 and 0.25 and says a high score calls for retraining. Those cutoffs are heuristics whose behaviour depends on binning and sample size. This chapter keeps the teaching bands but corrects the action: investigate the data and model outcomes before choosing a retrain.

:::

The default controls reproduce the lecture's **15% null rate** and failed **5%** ceiling. Change the high-bin share to see a two-bin PSI against a fixed 50/50 reference. Equal bins give PSI **0.000**. The score describes a shift but does not name its cause.

<ProfilingDriftLab />

## How it works

### Understand the data

Per-column type, null rate, cardinality, min/max/mean/quantiles, distributions; profiling surfaces surprises before they poison a model.

### Guard and monitor

Validate batches against expectations (schema, not-null, ranges, uniqueness) with Great Expectations; quarantine failures. Watch data/concept drift.

:::tip

**Worked.** 1,500 nulls / 10,000 = 0.15 (fails ≤5%). The source's PSI bands of 0.1 and 0.25 are informal triage cues; investigate a high score before deciding to retrain.

:::


## A real system that works this way

**Great Expectations** is a concrete validation framework. Its current Checkpoint flow runs validation definitions, returns results and can trigger actions such as updating data documentation or sending a notification. A checkpoint can take batch parameters, so the same rule can run on each dated partition. This makes a contract visible in the pipeline. It still needs an owner to interpret a new value that might be a legitimate product change rather than bad data.

Imagine an account-risk feature table. The batch arrives with 15% missing income values, above the agreed 5% ceiling. A validator records which accounts are affected, the batch date, source version and rule version. The owner checks whether one mobile form stopped sending the field or whether the whole population changed. Blocking may be appropriate if income is mandatory for the model; quarantining the affected records may be appropriate if a safe fallback exists and removing them does not bias the published population. The action is a product decision expressed as a data contract.

The same table's income distribution may shift even when no fields are missing. A change in the applicant mix could be valid, while an upstream unit conversion could be a bug. PSI can help prioritise investigation, but it does not distinguish those causes or show whether predictions worsened. Compare source versions, segment profiles, sample sizes and delayed outcome labels before retraining.

## Code you can run

The first calculation implements the lecture's null-rate rule. Keep the denominator and field in the reported result.

```python
rows = 10_000
null_income = 1_500
allowed_rate = 0.05
null_rate = null_income / rows
print(f"income nulls={null_income}/{rows} = {null_rate:.0%}; pass={null_rate <= allowed_rate}")
assert null_rate == 0.15
assert null_rate > allowed_rate
```

PSI sums (current − reference) × log(current / reference) over fixed bins. This toy example has two non-empty bins and a 50/50 reference. Real use needs a documented binning rule, enough observations and handling for empty bins; the formula alone does not supply a retrain policy.

```python
import math

reference = [0.5, 0.5]
current = [0.7, 0.3]

def psi(reference_shares, current_shares):
    if len(reference_shares) != len(current_shares):
        raise ValueError("Both distributions need the same bins")
    if any(p <= 0 for p in reference_shares + current_shares):
        raise ValueError("This simple version requires non-empty bins")
    return sum((q - p) * math.log(q / p) for p, q in zip(reference_shares, current_shares))

score = psi(reference, current)
print(f"two-bin PSI={score:.3f}")
assert round(score, 3) == 0.169
assert psi(reference, reference) == 0
```

The score lies in the lecture's informal "moderate" band. It tells us to inspect the shift, not to train a replacement automatically. If bins were chosen on the current data or their boundaries changed between runs, the comparison would be misleading.

## Designing with it

### Profile before setting thresholds

Start by confirming the unit of observation and data types. Count rows and distinct keys, then examine nulls, unexpected categories, numeric ranges and quantiles by source and time. A mean can hide a heavy tail; a cardinality change can reveal an ID format change. Keep a baseline profile from an approved period rather than treating the first batch as truth. State which fields are required for which consumers; one table can have different valid uses.

Set thresholds with the source owner and consumer. A zero-null rule for an immutable primary key is different from a 5% ceiling for a feature that has a documented fallback. A range check should follow real domain constraints, not merely the minimum and maximum seen last month. Use strict checks for impossible values and monitored expectations for plausible change. Otherwise validation can reject exactly the novel, valid population the model needs to learn about.

### Separate data drift from concept drift

**Data drift** means an input distribution changes: perhaps more applicants have low income, or a sensor begins reporting larger readings. **Concept drift** means the relationship between inputs and target changes: the same pattern of transactions now predicts a different risk. Input-only monitoring can detect some data drift promptly. Concept drift usually needs labels or a proxy outcome, often with a delay. A model can degrade without large marginal input drift, and input drift can be harmless if the model remains robust.

PSI compares shares in fixed bins. The lecture's 0.1 and 0.25 thresholds are commonly repeated heuristics, not a universal significance test. A research analysis of their statistical properties shows why threshold choice should consider sample sizes and bin design. Tiny bins can be noisy; empty bins need a declared smoothing convention. Monitor the score alongside source changes, null rates, performance by segment and outcome labels. A PSI above 0.25 may prompt urgent investigation; it cannot by itself establish that retraining would help.

### Route failure with evidence

For every failed rule, record the rule version, observed numerator and denominator, affected partitions or IDs, source snapshot and chosen action. If a batch is quarantined, preserve a way to repair or replay it. If a warning is allowed, surface the degraded state to consumers. A silent drop can alter class balance and make training data look cleaner than production data. Recheck the corrected batch and its downstream outputs after repair.

Drift alerts also need triage. First rule out instrumentation and schema changes. Then inspect which bins and cohorts moved, whether the shift aligns with a product or seasonal change, and whether model outcomes have deteriorated. Only then decide among changing a rule, fixing a source, recalibrating, retraining or leaving the model in place. The action should address the observed cause.

## Diagnose a null-rate spike and a PSI alert

A daily income feature normally has 2% nulls. Monday's batch has 15% missing values and a two-bin PSI of 0.169 against the reference distribution. Treat these as two observations. The null rule fails the agreed 5% ceiling. The PSI says the distribution of the non-null bins differs under the chosen binning. The first action is to find which records and sources changed, not to label the entire model obsolete.

Check counts by channel. If the missing values all come from one new application form, inspect its schema and recent release. Perhaps the field name changed while the ingestion mapping did not. That is a data contract failure. If the values exist in the source but were discarded by a parser, fix and replay the affected partition. If the form legitimately made income optional, the product and model owners need to decide how serving handles missingness; a new training run alone will not repair an unsupported input path.

Next inspect the PSI bins. Were the bin edges frozen from the approved reference? Are the current shares calculated on all eligible records or only non-null ones? Removing missing values can itself change the apparent distribution. If the source population grew in one region, the score may reflect a valid business shift. Segment by region and channel, but report counts so a tiny group is not overinterpreted. Compare the model's prediction mix and, when labels mature, its error rates for the affected groups.

The response depends on impact. If the model cannot score missing income safely, block the affected requests or route them to a reviewed fallback. If it can score them but performance is uncertain, monitor and perhaps run a shadow evaluation. If corrected data restores the old distribution, retraining would treat the symptom rather than the cause. If the population truly changed and outcome performance fell, a new model or threshold may be warranted. Keep the investigation and decision linked to the data version.

Finally, prevent repetition. Add a schema check before the feature transform, make the missingness dashboard slice by source channel, and require a sample payload in the source release process. Test the failure path: can the team identify the bad form, quarantine only its records, notify the consumer and replay after a fix? A quality system is proven by its response to a bad batch, not by the number of green checks on a good day.

### Make the comparison reproducible

A drift score depends on a reference population. Record the exact reference snapshot, bin edges, missing-value treatment and current window. If the bins are recomputed from each current batch, a change in the population can move the bins and conceal the shift. If an empty bin is smoothed, state the small replacement probability and use the same rule every time. A PSI calculated from two tiny samples may swing widely by chance. Show counts in each bin beside the score so a reviewer can see whether the signal is credible.

Validation and drift can disagree without contradiction. A batch with every status value in the allowed set can still have a drastically different mix of statuses. Conversely, a handful of invalid statuses can fail a strict rule while the overall distribution barely moves. Route the first to an integrity or schema investigation and the second to a population or model review, with overlap when both occur. Do not compress them into one "healthy" light.

Quality monitoring should also account for source silence. A null-rate check on an empty partition may never execute or may misleadingly return zero nulls. Add a row-count or freshness expectation that establishes the batch exists and covers the expected period. A missing source feed can produce a perfectly valid empty output if no rule checks volume. Preserve the logical date and the last source watermark in every quality report so absence is visible. Review the rule after seasonal or product changes. A static threshold can become noisy as the business evolves, but moving it solely to make a dashboard green hides a real change. Keep a decision record with the old and new rule, evidence and owner.

## Where this stands in 2026

:::info Industry view

- Expectation-based validators can run batch-specific checks and return results that drive documented actions.
- PSI remains a useful descriptive shift score, but fixed cutoffs and automatic retraining rules need local validation.
- Data quality, input drift and delayed model outcomes are complementary signals; none is a complete substitute for the others.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is data profiling and what does it compute?</summary>

Computing descriptive statistics to understand a dataset: per-column type, null rate, cardinality, min/max/mean/quantiles and distributions.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is data validation and how does it run?</summary>

Checking each batch against declared expectations (schema, not-null, ranges, uniqueness, referential integrity) in the pipeline (e.g. Great Expectations), quarantining/alerting on failures.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A 10,000-row column has 1,500 nulls. Compute the null rate and its verdict at a 5% threshold.</summary>

1,500/10,000 = 0.15 (15%) > 5%, so the rule fails; quarantine or block the batch according to the consumer contract.<br /><em>Session 8 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Contrast data drift and concept drift.</summary>

Data drift = the input distribution changes; concept drift = the input→label relationship changes. Both degrade models silently.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> How is PSI interpreted for drift?</summary>

The source's 0.1 and 0.25 PSI bands are common heuristics, not universal significance tests. A high PSI calls for investigation of bins, sample size, source changes and model outcomes before deciding to retrain.<br /><em>Session 8 · numeric</em>

</details>

## Go deeper

- [Great Expectations Checkpoints](https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/run_a_checkpoint/) shows validation results and actions.
- [Statistical properties of PSI](https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf) examines common threshold behaviour.
- Built from the course lecture "dm-s8-profiling-validation" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can distinguish a profile, a validation rule, data drift and concept drift.
- [ ] I can compute 1,500/10,000 = 15% and compare it with a 5% ceiling.
- [ ] I can calculate a two-bin PSI and explain why its threshold is not a retrain command.
- [ ] I can route a failed batch with affected IDs, an owner and a replay plan.
