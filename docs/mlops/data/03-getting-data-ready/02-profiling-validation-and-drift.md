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


:::tip Before you start

**You should already know**

- What a null rate and a rule threshold are: [data quality rules](/docs/mlops/data/quality-rules).
- What a normal distribution and a standard deviation are.
- What a p-value says: how surprising the data would be if nothing had changed.

**Reading time.** About 55 minutes, plus about 30 seconds to run the experiments.

**After this chapter you can**

- read a PSI score as a size of shift (about the square of the shift in standard deviations for a mean shift),
- compare PSI and a Kolmogorov-Smirnov test by how often each fires for a given shift and batch size,
- choose a threshold from the no-change distribution and avoid alert storms across many features.

:::

## In 30 seconds

A bakery weighs ten loaves each morning. Today's loaves are slightly heavier. Is the oven drifting, or is it ordinary variation? With ten loaves you cannot tell, with ten thousand you can, but a tiny, harmless difference then looks significant. PSI (population stability index) and the KS (Kolmogorov-Smirnov) test are two ways to ask that question of a column of data, and each has a blind spot: PSI barely notices a few extreme values, and KS notices every tiny shift once the batch is large.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Profile | A set of descriptive statistics for a column | 15% nulls, 6 distinct values |
| Reference | The approved earlier data you compare with | Last quarter's batch |
| PSI | A score of how much bin shares changed | 0.169 for 50/50 moving to 70/30 |
| Bin | A range of values that counts as one bucket | The second decile |
| KS test | A test comparing two samples' whole distributions | p below 0.05 means "differs" |
| Data drift | The input distribution changed | Higher incomes |
| Concept drift | The input to target relationship changed | Same income, different risk |
| False alarm | An alert when nothing has changed | PSI above 0.1 on identical data |

## The idea in plain words

A pipeline can finish successfully while its data is unusable. A column may become mostly null, a category may change spelling, a source may stop sending events or the population may shift. These failures need different responses. **Profiling** describes what arrived, **validation** checks it against declared rules, and **drift monitoring** compares it with a reference. None of them alone proves that a model is accurate.

A profile records per-column type, null rate, cardinality, range, quantiles and distribution. A profile is a snapshot of observations, not a judgement. A 15% null rate may be normal for an optional middle name and unacceptable for a required model label. Validation turns a consumer requirement into an assertion: a schema, non-null field, allowed value set, uniqueness condition or relationship. A failure needs a named action such as block, quarantine, warn or investigate.

<Infographic src="/img/dm/profiling-validation.svg" alt="A profile finds 1,500 nulls among 10,000 rows, or 15 per cent; a five per cent null ceiling fails, while PSI is a separate measure of population shift." caption="Missingness, contract failure and distribution change are different observations." />

A worked batch has 1,500 nulls among 10,000 rows, so the null rate is **0.15**, or **15%**. It fails a 5% ceiling. A common rule is to quarantine the batch, but the right action depends on the dataset contract and consumer cost. A critical label may justify blocking all publication; malformed optional rows may be isolated while unaffected rows continue. Preserve the failed rows and the decision so a repair can be replayed.

:::note Correction

PSI bands of 0.1 and 0.25 are often taught with the advice that a high score calls for retraining. Those cutoffs are heuristics whose behaviour depends on binning and sample size. This chapter keeps the teaching bands but corrects the action: investigate the data and model outcomes before choosing a retrain.

:::

The default controls reproduce the **15% null rate** and failed **5%** ceiling. Change the high-bin share to see a two-bin PSI against a fixed 50/50 reference. Equal bins give PSI **0.000**. The score describes a shift but does not name its cause.

<ProfilingDriftLab />

**What each control does.**

- **null values among ten thousand** sets how many of 10,000 values are missing.
- **allowed null rate** sets the ceiling in per cent.
- **current high-bin percentage** moves the current share of a two-bin variable away from the 50/50 reference.

**Try it yourself.**

1. Defaults: 1,500 nulls is 15.0%, above the 5% ceiling, so the null rule fails, and a 50% high bin gives PSI 0.000.
2. Set the high bin to 70%. PSI becomes 0.169, the hand calculation above. The null rule is unchanged, which shows that the two checks are independent.
3. Set the allowed null rate to 15. The null rule now passes (the test is "at most"). Then set the high bin to 90%: PSI climbs to about 0.88, far beyond the 0.25 line, though no value is missing.

## Worked example, step by step

**A two-bin PSI.** A reference with half the values in a low bin and half in a high bin is compared with a batch that is 70% high and 30% low. PSI is the sum over bins of (current share - reference share) × ln(current share / reference share).

1. High bin: (0.7 - 0.5) × ln(0.7 / 0.5) = 0.2 × 0.3365 = 0.0673.
2. Low bin: (0.3 - 0.5) × ln(0.3 / 0.5) = -0.2 × -0.5108 = 0.1022.
3. PSI = 0.0673 + 0.1022 = 0.169, in the informal 0.10 to 0.25 band.

**The score for no change at all.** With 10 bins, the 9 free bin shares each wobble by about 1 / n for a batch of n rows (and 1 / m for a reference of m rows), so a batch with nothing wrong scores about 9 × (1/n + 1/m).

4. For a batch of 200 against a reference of 20,000: 9 × (0.005 + 0.00005) = 0.045.
5. For a batch of 2,000: 9 × (0.0005 + 0.00005) = 0.005. For 20,000: 0.0009.

**The score for a real shift.** For a normal distribution moved by δ standard deviations, the symmetric divergence behind PSI is δ², so PSI ≈ δ².

6. A shift of 0.25 standard deviations gives about 0.0625, a shift of 0.5 gives about 0.25. A PSI of 0.25 therefore means roughly half a standard deviation.

In words: PSI has a noise floor that falls with batch size and a signal that grows with the square of the shift. The experiments below check steps 4 to 6 on simulated data.

## How it works

### Understand the data

Per-column type, null rate, cardinality, min/max/mean/quantiles, distributions; profiling surfaces surprises before they poison a model.

### Guard and monitor

Validate batches against expectations (schema, not-null, ranges, uniqueness) with Great Expectations; quarantine failures. Watch data/concept drift.

:::tip

**Worked.** 1,500 nulls / 10,000 = 0.15 (fails ≤5%). The PSI bands of 0.1 and 0.25 are informal triage cues; investigate a high score before deciding to retrain.

:::


## A real system that works this way

**Great Expectations** is a concrete validation framework. Its current Checkpoint flow runs validation definitions, returns results and can trigger actions such as updating data documentation or sending a notification. A checkpoint can take batch parameters, so the same rule can run on each dated partition. This makes a contract visible in the pipeline. It still needs an owner to interpret a new value that might be a legitimate product change rather than bad data.

Imagine an account-risk feature table. The batch arrives with 15% missing income values, above the agreed 5% ceiling. A validator records which accounts are affected, the batch date, source version and rule version. The owner checks whether one mobile form stopped sending the field or whether the whole population changed. Blocking may be appropriate if income is mandatory for the model; quarantining the affected records may be appropriate if a safe fallback exists and removing them does not bias the published population. The action is a product decision expressed as a data contract.

The same table's income distribution may shift even when no fields are missing. A change in the applicant mix could be valid, while an upstream unit conversion could be a bug. PSI can help prioritise investigation, but it does not distinguish those causes or show whether predictions worsened. Compare source versions, segment profiles, sample sizes and delayed outcome labels before retraining.

## Code you can run

The first calculation implements the null-rate rule. Keep the denominator and field in the reported result.

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

The score lies in the informal "moderate" band, 0.10 to 0.25. It tells us to inspect the shift, not to train a replacement automatically. If bins were chosen on the current data or their boundaries changed between runs, the comparison would be misleading.

### Experiment: detection power of PSI and KS

The first block measures how often PSI above 0.1, PSI above 0.25 and a KS test at p below 0.05 fire. It uses 200 repeated batches for each combination of shift and batch size. The reference is 20,000 standard normal values, PSI uses 10 bins at the reference deciles, and the shifts are a mean move of 0.05 to 0.5 standard deviations, a spread 1.3 times wider and a contamination of 2% outliers at 6. Run with SciPy 1.18.1 and NumPy 2.5.3.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
reference = rng.normal(size=20_000)
edges = np.quantile(reference, np.linspace(0, 1, 11)[1:-1])
for size in (200, 2_000, 20_000):
    print(f"no change, batch of {size:6d}: predicted PSI {9 * (1 / size + 1 / len(reference)):.4f}")

def psi(current):
    share = lambda x: np.clip(np.bincount(np.searchsorted(edges, x), minlength=10) / len(x), 1e-4, None)
    p, q = share(reference), share(current)
    return float(np.sum((q - p) * np.log(q / p)))

def draw(kind, size):
    base = rng.normal(size=size)
    if kind == "none":
        return base
    if kind.startswith("mean"):
        return base + float(kind[4:])
    if kind == "wider 1.3x":
        return base * 1.3
    if kind == "2% outliers at 6":
        return np.where(rng.random(size) < 0.02, 6.0, base)

print(f"{'shift':18s} {'n':>6s} {'PSI median':>10s} {'PSI>0.1':>8s} {'PSI>0.25':>9s} {'KS p<0.05':>10s}")
for kind in ("none", "mean0.05", "mean0.1", "mean0.25", "mean0.5", "wider 1.3x", "2% outliers at 6"):
    for size in (200, 2_000, 20_000):
        scores, ks_hits = [], 0
        for _ in range(200):
            current = draw(kind, size)
            scores.append(psi(current))
            ks_hits += stats.ks_2samp(reference, current).pvalue < 0.05
        scores = np.array(scores)
        print(f"{kind:18s} {size:6d} {np.median(scores):10.3f} {np.mean(scores > 0.1):8.0%} {np.mean(scores > 0.25):9.0%} {ks_hits / 200:10.0%}")
```

**Reading the output.** Each line is one shift and batch size. `PSI median` is the middle PSI over 200 batches. The next three columns are the share of those batches in which each rule fired.

**Line by line.**

- `np.quantile(reference, ...)` freezes the bin edges from the reference, which stops them moving with the current data.
- `np.clip(..., 1e-4, None)` is the empty-bin convention, a small floor so the logarithm is defined.
- `stats.ks_2samp(reference, current).pvalue < 0.05` is a two-sided test of whether the two samples came from the same distribution.

The printed output:

```text
no change, batch of    200: predicted PSI 0.0454
no change, batch of   2000: predicted PSI 0.0050
no change, batch of  20000: predicted PSI 0.0009
shift                   n PSI median  PSI>0.1  PSI>0.25  KS p<0.05
none                  200      0.045       4%        0%         6%
none                 2000      0.005       0%        0%         7%
none                20000      0.001       0%        0%         5%
mean0.05              200      0.047       4%        0%        11%
mean0.05             2000      0.008       0%        0%        46%
mean0.05            20000      0.003       0%        0%       100%
mean0.1               200      0.054       8%        0%        24%
mean0.1              2000      0.013       0%        0%        96%
mean0.1             20000      0.010       0%        0%       100%
mean0.25              200      0.102      52%        0%        80%
mean0.25             2000      0.062       0%        0%       100%
mean0.25            20000      0.059       0%        0%       100%
mean0.5               200      0.293     100%       71%       100%
mean0.5              2000      0.241     100%       30%       100%
mean0.5             20000      0.235     100%        3%       100%
wider 1.3x            200      0.128      77%        3%        55%
wider 1.3x           2000      0.095      36%        0%       100%
wider 1.3x          20000      0.091       2%        0%       100%
2% outliers at 6      200      0.042       2%        0%         5%
2% outliers at 6     2000      0.009       0%        0%        25%
2% outliers at 6    20000      0.005       0%        0%       100%
```

### Reading the detection experiment

The hand estimates held. With no change the PSI medians were 0.045, 0.005 and 0.001 for batches of 200, 2,000 and 20,000, against 0.045, 0.005 and 0.0009 predicted. For a mean shift the medians sit close to the square of the shift: 0.059 to 0.062 at 0.25 and 0.235 to 0.293 at 0.5. A PSI of 0.25 is about half a standard deviation, not a catastrophe on its own.

The surprise is how rarely the 0.25 rule fires. A half-standard-deviation shift pushed PSI past 0.25 in only 3% of batches of 20,000, because the PSI is near 0.235, just under the line, while KS flagged it every time. The 0.1 rule is the opposite for small batches: at n = 200 a shifted mean of 0.25 fires 52% of the time, close to a coin flip on a moderate shift.

KS reacted to everything once the batch was large: a shift of 0.05 standard deviations was flagged in all 200 batches of 20,000, which is statistically real and practically trivial. PSI was nearly blind to the outliers: 2% of values at 6 gave PSI 0.005 at n = 20,000, because all the contamination falls into one top bin. KS found them at large n but only 25% of the time at n = 2,000.

Limits: one reference, one distribution shape (normal), 200 batches per cell and a binning rule I chose. Real columns are skewed, discrete or heavy-tailed.

### Experiment: thresholds and false alarms

The second block asks what PSI looks like when nothing has changed, and what happens when a KS test runs on 20 unchanged features every day.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(1)
reference = rng.normal(size=20_000)
edges = np.quantile(reference, np.linspace(0, 1, 11)[1:-1])
share = lambda x: np.clip(np.bincount(np.searchsorted(edges, x), minlength=10) / len(x), 1e-4, None)
ref_share = share(reference)

def psi(current):
    q = share(current)
    return float(np.sum((q - ref_share) * np.log(q / ref_share)))

print("PSI when nothing has changed, 1,000 batches per size")
for size in (100, 500, 5_000):
    null = np.array([psi(rng.normal(size=size)) for _ in range(1_000)])
    print(f"batch of {size:5d}: median {np.median(null):.3f}, 99th percentile {np.quantile(null, 0.99):.3f}, share above 0.1 {np.mean(null > 0.1):.1%}")

days, features, size = 100, 20, 1_000
p_values = np.array([[stats.ks_2samp(reference, rng.normal(size=size)).pvalue for _ in range(features)] for _ in range(days)])
print(f"{features} unchanged features, {days} days, KS at 0.05: days with at least one alert {np.mean((p_values < 0.05).any(axis=1)):.0%}, alerts per day {(p_values < 0.05).sum() / days:.2f}")
print(f"same data, threshold 0.05 / {features}: days with an alert {np.mean((p_values < 0.05 / features).any(axis=1)):.0%}")
```

**Reading the output.** The first three lines are the no-change PSI for three batch sizes. The last two compare a plain 0.05 KS threshold with a corrected one across 20 features and 100 days.

**Line by line.**

- `share` and `psi` are redefined here so the block runs on its own.
- `0.05 / features` is the Bonferroni correction: with 20 tests the per-test threshold is divided by 20.

The printed output:

```text
PSI when nothing has changed, 1,000 batches per size
batch of   100: median 0.087, 99th percentile 0.260, share above 0.1 38.0%
batch of   500: median 0.017, 99th percentile 0.045, share above 0.1 0.0%
batch of  5000: median 0.002, 99th percentile 0.006, share above 0.1 0.0%
20 unchanged features, 100 days, KS at 0.05: days with at least one alert 64%, alerts per day 1.02
same data, threshold 0.05 / 20: days with an alert 2%
```

### Reading the threshold experiment

On identical data a batch of 100 had a median PSI of 0.087 and exceeded 0.1 in 38.0% of batches, and its 99th percentile, 0.260, is above the 0.25 "significant change" line. The same rule at 500 rows never fired, and the 99th percentile was 0.045. A fixed cutoff therefore means different things at different batch sizes. Set the threshold from the no-change percentile for your batch size, or drop the PSI for small batches.

Alert volume is the other trap. With 20 unchanged features, a plain KS test at 0.05 raised at least one alert on 64% of days, about one alert per day. Dividing the threshold by 20 brought it to 2% of days. Fewer false alarms cost some sensitivity, so pick the correction with an eye on the shifts you must catch. Limits: independent features (real ones correlate), normal data and a single run.

<Infographic src="/img/dm-enrich/psi-vs-ks.svg" alt="A table compares how often PSI above 0.25, PSI above 0.1 and a KS test fire for several shifts and batch sizes, with cards on the no-change PSI at 100 rows and the alert rate across 20 features." caption="Look first at the mean 0.5 rows: KS fires every time while PSI above 0.25 fires in only 3% of batches of 20,000." />

## Designing with it

### Profile before setting thresholds

Start by confirming the unit of observation and data types. Count rows and distinct keys, then examine nulls, unexpected categories, numeric ranges and quantiles by source and time. A mean can hide a heavy tail; a cardinality change can reveal an ID format change. Keep a baseline profile from an approved period rather than treating the first batch as truth. State which fields are required for which consumers; one table can have different valid uses.

Set thresholds with the source owner and consumer. A zero-null rule for an immutable primary key is different from a 5% ceiling for a feature that has a documented fallback. A range check should follow real domain constraints, not merely the minimum and maximum seen last month. Use strict checks for impossible values and monitored expectations for plausible change. Otherwise validation can reject exactly the novel, valid population the model needs to learn about.

### Separate data drift from concept drift

**Data drift** means an input distribution changes: perhaps more applicants have low income, or a sensor begins reporting larger readings. **Concept drift** means the relationship between inputs and target changes: the same pattern of transactions now predicts a different risk. Input-only monitoring can detect some data drift promptly. Concept drift usually needs labels or a proxy outcome, often with a delay. A model can degrade without large marginal input drift, and input drift can be harmless if the model remains robust.

PSI compares shares in fixed bins. The 0.1 and 0.25 thresholds are commonly repeated heuristics, not a universal significance test. A research analysis of their statistical properties shows why threshold choice should consider sample sizes and bin design. Tiny bins can be noisy; empty bins need a declared smoothing convention. Monitor the score alongside source changes, null rates, performance by segment and outcome labels. A PSI above 0.25 may prompt urgent investigation; it cannot by itself establish that retraining would help.

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

## Common mistakes

1. **Using 0.1 and 0.25 as universal limits.** They look standard. At 100 rows, unchanged data exceeded 0.1 in 38.0% of batches, and a half standard deviation shift stayed under 0.25 at 20,000 rows. Calibrate to your batch size.
2. **Reading PSI as proof that the model is worse.** A PSI of 0.25 is about half a standard deviation. It says the input moved, not that predictions are wrong. Check outcomes and segments before retraining.
3. **Expecting PSI to catch outliers.** 2% of values at 6 gave PSI 0.005 at 20,000 rows. Add range, tail and null-rate rules.
4. **Running one test per feature without a correction.** Twenty clean features raised an alert on 64% of days. Correct the threshold or alert on the number of drifting features.
5. **Recomputing bin edges on each batch.** If the bins follow the current data, the shift can disappear. Freeze the edges from the reference.

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

The 0.1 and 0.25 PSI bands are common heuristics, not universal significance tests. A high PSI calls for investigation of bins, sample size, source changes and model outcomes before deciding to retrain.<br /><em>Session 8 · numeric</em>

</details>

<details>
<summary><strong>Q6. (Medium)</strong> A batch of 200 rows is compared with a 20,000-row reference using 10 bins, and nothing has changed. About what PSI do you expect?</summary>

About 9 × (1/200 + 1/20,000) = 9 × 0.00505 = 0.045. The experiment's median was 0.045. A threshold of 0.1 sits only about twice above that noise floor, so it fires on some unchanged batches (4% in the run).

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> A feature shifts by 0.5 standard deviations. At 20,000 rows KS flags it every time but PSI above 0.25 flags 3% of batches. Which should you trust, and what does each say?</summary>

Both are right about different questions. KS says the distributions differ, which is true and detectable at large n. PSI says the size of the difference is about 0.235 (roughly 0.5 squared), just under the 0.25 line, so it is moderate and not an emergency. Trust KS to detect and PSI to size, then look at model outcomes before acting.

</details>

## Go deeper

- [Great Expectations Checkpoints](https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/run_a_checkpoint/) shows validation results and actions.
- [Statistical properties of PSI](https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf) examines common threshold behaviour.
- [Yurdakul and Naranjo, Statistical properties of the population stability index](https://files.wmich.edu/s3fs-public/attachments/u730/2022/PSIfinal.pdf) (Journal of Risk Model Validation 14(4), 2021, opened 2026-10-09) states that the 0.10 and 0.25 rule of thumb is used without reference to type I or type II error rates, and sets out to give PSI's statistical properties.
- [SciPy ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) (opened 2026-10-09, SciPy 1.18) tests the null hypothesis that the two samples come from the same distribution.
- Library versions run for the experiments: SciPy 1.18.1, NumPy 2.5.3, Python 3.14.6. The data is generated, so no licence applies.
- Built from the course lecture "dm-s8-profiling-validation" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can distinguish a profile, a validation rule, data drift and concept drift.
- [ ] I can compute 1,500/10,000 = 15% and compare it with a 5% ceiling.
- [ ] I can calculate a two-bin PSI and explain why its threshold is not a retrain command.
- [ ] I can route a failed batch with affected IDs, an owner and a replay plan.
- [ ] I can estimate the PSI I should see when nothing has changed, from the batch size and the number of bins.
- [ ] I can say what a PSI of 0.25 means for a mean shift, and why KS and PSI disagree at large batch sizes.
- [ ] I can set a threshold from the no-change distribution and correct for testing many features.

## Where to go next

Next is [analytics engineering and history](/docs/mlops/data/analytics-engineering-history). For monitoring these checks in production, see [observing data in production](/docs/mlops/data/data-observability).
