---
id: dm-observing-data-in-production
title: "Data Management · Lecture 16 — Observing Data in Production"
sidebar_label: "16 · Data observability"
sidebar_position: 3
slug: /mlops/data/data-observability
description: "Monitor freshness, volume, schema, distribution and lineage with actionable consumer promises."
tags: [data-management, data-observability, freshness, data-quality]
---

import Infographic from '@site/src/components/Infographic';
import FreshnessBudgetLab from '@site/src/components/viz/FreshnessBudgetLab';

**In one line.** Data observability turns a consumer's trust requirement into measurable signals and a response path.

:::tip Before you start

**You should already know**

- What a pipeline run and a partition are ([Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery)).
- What a null rate and a quantile are, and what profiling and drift checks do ([Session 8, profiling, validation and drift](/docs/mlops/data/profiling-validation-drift)).

**Reading time:** about 45 minutes, plus a few seconds to run the code.

**After this chapter you can**

- Work out a freshness breach and a volume change by hand, and say why the baseline matters.
- Compare monitors by what they catch, how fast, and how often they cry wolf.
- Choose a threshold knowing the cost of false alarms.

:::

## In 30 seconds

A delivery that arrives is not the same as a delivery that is right. Data observability is the set of checks that tell you the data is recent, the right size, the right shape and the right kind of values, before somebody downstream makes a decision with it.

Think of a smoke alarm. If it is too sensitive it goes off every time you make toast, and soon nobody listens. If it is too dull it misses the fire. Every data monitor sits somewhere on that line.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Freshness | How old the newest data is | Last approved load was 90 minutes ago |
| Volume | How much data arrived | 600 rows against 1,000 expected |
| Schema | The columns and their types | `amount` changed from number to text |
| Distribution | The spread of values, nulls and categories | Null rate jumped from 2% to 12% |
| Baseline | What "normal" looks like for comparison | Same hour, last four weeks |
| False alarm | An alert when nothing is wrong | A volume alert at the daily peak |
| Detection delay | Time from the start of a problem to the first alert | 1 hour |
| Watermark | The newest event time present in the data | Events through 09:30 |


## The idea in plain words

A pipeline can finish without producing usable data. It may load an old partition, omit half of a source, change a type, duplicate rows or publish values from an unexpected population. Job success tells operators that code reached a terminal state; **data observability** asks whether the resulting data still serves its consumers. It combines measurements, context and incident response so that a problem is detected before a dashboard or model quietly makes a wrong decision.

Data health is often organised into five pillars: **freshness, volume, schema, distribution and lineage**. Freshness asks whether data is recent enough for a decision. Volume asks whether the expected amount arrived. Schema checks structural compatibility. Distribution observes values, nulls and proportions. Lineage explains upstream origins and downstream impact. This five-pillar framing is a useful taxonomy associated with Monte Carlo's data-observability work, not a universal standard or a complete proof of quality. Correctness against reality, labels and business meaning may need additional checks.

<Infographic src="/img/dm/data-observability.svg" alt="Five data-health dimensions are freshness, volume, schema, distribution and lineage; a 90-minute data age against a 60-minute limit is a 30-minute breach." caption="Signals diagnose different failure modes; the data contract determines which response is appropriate." />

In a worked example the latest data load was **90 minutes** ago against a **60-minute** freshness limit. Under a contract measuring age since the last approved load, **90 > 60**, so the limit is breached by **30 minutes** and an alert should fire. That calculation does not reveal how old the source events are. A job could load now but contain events from yesterday. For a live model, track a source event-time watermark and the approved publication timestamp so that ingest lag and pipeline lag are distinguishable.

:::note Correction

It is common to say that every breach should alert and quarantine, and that observability should trigger retraining or rollback. That is too blunt, and the sections below make the response conditional on the consumer contract and incident cause. A freshness breach may require an alert or fallback; it does not automatically justify quarantining otherwise valid data. Input drift is one signal, but model retraining requires outcome evidence and a cause analysis.

:::

The lab starts with a **90-minute age** and **60-minute limit**, reporting a **30-minute breach**. Change either control to see the state change. It measures time since the approved load; a source watermark is still needed to tell whether the loaded records themselves are current.

<FreshnessBudgetLab />

## Worked example, step by step

An hourly feed normally carries 1,000 rows on a weekday, with a daily cycle: 1,000 x (1 + 0.5 sin(x)), so 1,500 at the busiest hour and 500 at the quietest. The last approved load is stamped 10:30 and the time now is 12:00.

1. **Freshness.** 12:00 - 10:30 = 90 minutes against a 60-minute limit: breached by 30.
2. **A flat volume band.** Compare each hour with the plain average of 1,000 and alert at 20% off, so outside 800 to 1,200. That means the cycle term must satisfy |0.5 sin(x)| <= 0.2, which is |sin(x)| <= 0.4. The share of the day inside is 4 x arcsin(0.4) / (2 x pi) = 4 x 0.4115 / 6.283 = 26%. So about 74% of perfectly healthy hours raise an alert.
3. **A same-hour baseline.** Compare 6 pm with the median of the previous four 6 pm values. Row counts fluctuate by about the square root of 1,000, which is 32 or 3.2%. A 15% band is 0.15 / 0.032 = 4.7 times that noise, so healthy hours almost never alert.
4. **A p-value threshold.** A test with p below 0.05 flags 5% of healthy batches by design: 24 x 0.05 = 1.2 false alarms per day on hourly batches.
5. **Detection delay.** If an incident starts in hour 100 and the first alert fires on the batch of hour 101, the delay is 1 hour.

In words: a monitor is only as good as its baseline, and a threshold on a p-value is a promise of false alarms. The first block below prints steps 1 to 4.

## How it works

### Five pillars of health

- **Freshness · Volume**; Up to date? Expected amount arrived?
- **Schema · Distribution**; Structure changed? Values in range?
- **Lineage**; Where from / what breaks downstream?

### SLAs & alerting

Instrument useful signals with explicit thresholds and owners. Breaches alert; quarantine or block only when the consumer contract requires it. Data freshness and model drift need separate investigation.

:::tip

**Worked.** Data age 90 min > 60 min SLA → breach → alert.

:::


## A real system that works this way

**Elementary** documents monitors for freshness, volume and custom data metrics in dbt models, plus schema checks and downstream exposure information. These are concrete ways to observe analytical datasets after transformations. A daily customer table can have a freshness monitor for its latest approved partition, a volume monitor for row count, tests for key uniqueness and nulls, and schema change detection. The team's ownership record then maps a failed table to reports and models that use it.

**Great Expectations** offers validation definitions and Checkpoints whose Actions can record results, update documentation or send notifications. A checkpoint can run on a named batch and return the exact rule outcomes. The action is configurable: a critical required-key failure may block publication, while a mild distribution shift may notify an owner for investigation. The system should preserve the failed batch and rule version so a fix can be replayed and verified.

Neither tool can infer a business definition by itself. A volume monitor may see the expected row count even when the wrong region's records arrived. A schema check may pass while an API changes the meaning of a status code. The consumer contract must state the grain, time window, expected source coverage, key semantics and acceptable degraded behaviour. Observability tools make these claims measurable and inspectable.

## Code you can run

The freshness arithmetic is simple, but the clock must be named. This code measures minutes since the last approved load at a chosen observation time.

```python
from datetime import datetime, timedelta, timezone

now = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
last_approved = now - timedelta(minutes=90)
limit_minutes = 60
age_minutes = (now - last_approved).total_seconds() / 60
breach_minutes = max(0, age_minutes - limit_minutes)
print(f"age={age_minutes:.0f} min, breach={breach_minutes:.0f} min")
assert age_minutes == 90
assert breach_minutes == 30
```

A separate source watermark shows why a fresh load can contain stale events. The example also compares volume with a declared baseline instead of treating every change as an error.

```python
from datetime import datetime, timedelta, timezone

now = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
age_minutes = 90
limit_minutes = 60
last_source_event = now - timedelta(minutes=150)
source_age = (now - last_source_event).total_seconds() / 60
expected_rows = 1_000
observed_rows = 600
relative_change = (observed_rows - expected_rows) / expected_rows
checks = {
    "load_age_breach": age_minutes > limit_minutes,
    "source_age_breach": source_age > limit_minutes,
    "volume_outside_20_percent": abs(relative_change) > 0.20,
}
print(checks, f"volume_change={relative_change:.0%}")
assert checks == {
    "load_age_breach": True,
    "source_age_breach": True,
    "volume_outside_20_percent": True,
}
assert relative_change == -0.4
```

A seven-day average can be a useful starting baseline, but a weekday, holiday or product launch may need separate expected ranges. Thresholds should be tested against normal history and the cost of missed or noisy alerts.

### The worked example in code

This block reproduces steps 1 to 4 of the worked example.

```python
import numpy as np

age, limit = 90, 60
print("breach", age - limit)
x = np.linspace(0, 2 * np.pi, 100_000, endpoint=False)
volume = 1000 * (1 + 0.5 * np.sin(x))
outside = (np.abs(volume / 1000 - 1) > 0.20).mean()
print("flat band: inside", round(1 - outside, 3), "alerts", round(outside, 3), "closed form", round(4 * np.arcsin(0.4) / (2 * np.pi), 3))
noise = 1000 ** 0.5 / 1000
print("poisson noise", round(noise, 4), "15% band in noise units", round(0.15 / noise, 2))
print("false alarms per day at p < 0.05", 24 * 0.05)
```

**Reading the output.** It prints a breach of 30, then inside 0.262 and alerts 0.738 against a closed form of 0.262, then Poisson noise 0.0316 with a band of 4.74 noise units, then 1.2 false alarms per day.

### An experiment on a simulated stream

Which monitors catch a problem, how soon, and how often do they alarm when nothing is wrong? The block below simulates 120 days of hourly batches with a daily and weekly volume cycle, then injects 36 incidents of four hours each: late publication, partial volume and a distribution change (more nulls and a shifted amount). Each kind appears 12 times at three severities, one quarter, one half and full strength. Ten monitors watch the stream. The first four weeks are warm-up for the baselines, and false alarms are counted on the remaining healthy batches.

Versions used: Python 3.14.6, SciPy 1.18.1, pandas 2.3.3, NumPy 2.5.3. Everything is simulated, including the incident sizes. It runs in about five seconds.

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

rng = np.random.default_rng(16)
hours, warm = 24 * 120, 24 * 28
t = np.arange(hours)
expected = 1000 * (1 + 0.5 * np.sin((t % 24 - 14) * 2 * np.pi / 24)) * np.where((t // 24) % 7 >= 5, 0.6, 1.0)
rows = rng.poisson(expected).astype(float)
delay = rng.lognormal(np.log(8), 0.5, hours)
null_rate = np.full(hours, 0.02)
scale = np.ones(hours)
kind = np.zeros(hours, int)
starts = np.sort(rng.choice(np.arange(warm + 24, hours - 12, 40), 36, replace=False))
incidents = []
for i, start in enumerate(starts):
    label, severity, span = i % 3 + 1, (0.25, 0.5, 1.0)[(i // 3) % 3], slice(start, start + 4)
    kind[span] = label
    incidents.append((label, severity, start))
    if label == 1:
        delay[span] += 90 * severity
    elif label == 2:
        rows[span] *= 1 - 0.5 * severity
    else:
        null_rate[span], scale[span] = 0.02 + 0.10 * severity, 1 + 0.25 * severity

reference = rng.lognormal(3.0, 0.6, 5000)
ks_p, ks_d, batch_null = np.zeros(hours), np.zeros(hours), np.zeros(hours)
for h in range(hours):
    sample = rng.lognormal(3.0, 0.6, min(int(rows[h]), 1000)) * scale[h]
    result = ks_2samp(sample, reference)
    ks_p[h], ks_d[h] = result.pvalue, result.statistic
    batch_null[h] = rng.binomial(int(rows[h]), null_rate[h]) / rows[h]

trailing = pd.Series(rows).shift(1).rolling(168).mean().to_numpy()
same_hour = np.full(hours, np.nan)
for h in range(168 * 4, hours):
    same_hour[h] = np.median(rows[[h - 168 * k for k in range(1, 5)]])
off = lambda tolerance: np.abs(rows / same_hour - 1) > tolerance
twice = lambda alarm: alarm & np.r_[False, alarm[:-1]]
monitors = {
    "delay over 60 min": (delay > 60, 1),
    "delay over 20 min": (delay > 20, 1),
    "volume 20% off trailing mean": (np.abs(rows / trailing - 1) > 0.20, 2),
    "volume 30% off same hour": (off(0.30), 2),
    "volume 15% off same hour": (off(0.15), 2),
    "volume 15% off, twice in a row": (twice(off(0.15)), 2),
    "KS p below 0.05": (ks_p < 0.05, 3),
    "KS p below 1e-6": (ks_p < 1e-6, 3),
    "KS statistic over 0.10": (ks_d > 0.10, 3),
    "null rate over 6%": (batch_null > 0.06, 3),
}
healthy = (kind == 0) & (t >= 168 * 4)
print("batches", hours, "incidents", len(incidents), "healthy batches scored", int(healthy.sum()))
print(f"{'monitor':32}{'sev 0.25':>9}{'sev 0.5':>9}{'sev 1.0':>9}{'delay h':>9}{'false/1000':>12}")
for name, (alarm, wanted) in monitors.items():
    caught, delays = {0.25: 0, 0.5: 0, 1.0: 0}, []
    for label, severity, start in incidents:
        hits = np.flatnonzero(alarm[start:start + 4])
        if label == wanted and len(hits):
            caught[severity] += 1
            delays.append(hits[0])
    row = "".join(f"{caught[s]:7d}/4" for s in (0.25, 0.5, 1.0))
    print(f"{name:32}{row}{np.mean(delays) if delays else float('nan'):9.2f}{1000 * alarm[healthy].mean():12.1f}")
```

The output of the run:

```text
batches 2880 incidents 36 healthy batches scored 2064
monitor                          sev 0.25  sev 0.5  sev 1.0  delay h  false/1000
delay over 60 min                     0/4      2/4      4/4     0.83         0.5
delay over 20 min                     4/4      4/4      4/4     0.00        29.1
volume 20% off trailing mean          3/4      4/4      4/4     0.45       712.2
volume 30% off same hour              0/4      0/4      4/4     0.00         0.0
volume 15% off same hour              3/4      4/4      4/4     0.27         2.4
volume 15% off, twice in a row        1/4      4/4      4/4     1.22         0.0
KS p below 0.05                       4/4      4/4      4/4     0.50        50.4
KS p below 1e-6                       0/4      2/4      4/4     0.33         0.0
KS statistic over 0.10                0/4      3/4      4/4     0.00         1.0
null rate over 6%                     0/4      4/4      4/4     0.25         0.0
```

**Reading the output.** The three severity columns show how many of the 4 incidents of that severity, for the monitor's own incident kind, were caught inside their four hours. `delay h` is the mean hour of the first alarm after the incident began, over caught incidents. `false/1000` is alarms per 1,000 healthy batches.

**Line by line.**

- `rows[span] *= 1 - 0.5 * severity` removes up to half of the rows in a partial-load incident. At severity 0.25 it removes 12.5%.
- `same_hour[h] = np.median(rows[[h - 168 * k for k in range(1, 5)]])` is the seasonal baseline: the same hour of the week, four weeks back, 168 hours apart.
- `twice = lambda alarm: alarm & np.r_[False, alarm[:-1]]` requires two alarms in a row, which removes one-off blips at the price of an hour's delay.

### What the numbers say

Baseline choice mattered most. The volume monitor that compares each hour with the trailing 7-day average and alerts at 20% raised 712.2 false alarms per 1,000 healthy batches, 71% of them. The daily cycle swings volume by half, so the monitor was alarming on the cycle. Comparing with the same hour of previous weeks at 15% cut that to 2.4 per 1,000 and still caught 3 of 4 of the weakest incidents.

Statistical tests cry wolf by design. `KS p below 0.05` caught every incident, including the weakest, but it raised 50.4 false alarms per 1,000 batches, about one every 20 hours. Tightening to p below 1e-6 removed the false alarms (0.0) and also lost all four of the weakest incidents. A fixed effect-size rule, KS statistic over 0.10, had 1.0 false alarm per 1,000 and caught 3 of 4 at half strength. The plain null-rate rule, over 6%, caught every incident at half strength or more with no false alarm.

Freshness shows the same trade. A 60-minute limit raised 0.5 false alarms per 1,000 but missed the weakest delays (0 of 4); a 20-minute limit caught all of them and raised 29.1.

Requiring two alarms in a row removed the remaining false alarms and added about an hour of delay (1.22 against 0.27 hours) and lost 2 of the 3 weak catches. The same-hour monitor at 30% missed everything except the full-strength incidents.

Limits: simulated incidents with a convenient shape, one seed, four-hour incidents, and a Poisson volume that is steadier than many real feeds. The ranking of thresholds will move with your data.

<Infographic src="/img/dm-enrich/dm2-monitor-tradeoff.svg" alt="False alarms per 1,000 healthy batches and incidents caught for nine monitors on a simulated stream." caption="Look first at the 712 bar: a flat baseline on a cyclical feed alarms most of the time." />

## Designing with it

### Define the consumer promise first

Write down which dataset a consumer reads, its expected partition or event-time coverage, maximum age, schema, key grain and safe fallback. A model making a credit decision may require strict feature freshness; a monthly report may tolerate a delayed table with a visible status. A contract should state the evaluation time zone and boundary: if maximum age is 60 minutes, decide whether exactly 60 passes and what timestamp counts as publication. An alert without a named owner, response deadline and consumer impact is an observation, not an operational promise.

Measure two freshness clocks when sources can lag. The **source watermark** is the latest event time represented in the data. The **publication time** is when an approved version became available. A batch loaded five minutes ago can have a source watermark three hours old. Conversely, a source may be current while a validation task has not published it. Keep clocks and time zones explicit. Do not use a maximum event timestamp from a corrupt future-dated record as proof that a partition is current; validate timestamp plausibility and source coverage.

### Instrument the five dimensions

For volume, record counts by source, partition and important segment. A global count can hide one missing region offset by duplicate rows elsewhere. Compare with a baseline that respects weekday and seasonality, and preserve expected counts from source manifests where available. For schema, detect field additions, removals and type changes, then check whether a semantic contract changed even if the type did not. An integer status field can keep its schema while its code meanings change.

For distribution, monitor null rates, allowed values, quantiles and category proportions. A shift can be a legitimate change in customers or an ingestion bug. Segment results by relevant population but include denominators so small groups are not overinterpreted. For lineage, link each output partition or version to its source inputs, transform and downstream users. Lineage does not say that data is healthy; it tells the responder where to look and whom to notify when a signal fails. Keep the graph current by emitting events from jobs rather than relying only on a manually drawn diagram.

### Alert on impact, not every fluctuation

Set a hard alert for impossible or high-impact conditions, such as a missing mandatory key or a source feed that has not advanced by its deadline. Use warning levels and investigation queues for ambiguous changes. A zero-row result might be valid on a holiday, or it might mean the ingestion task never read the source. A sudden 20% count change may be normal for a new campaign. Tune thresholds on history, review false positives and document the action for each severity. Avoid alert storms where five monitors page five teams for one upstream outage; group by source and incident lineage.

Quarantine or block only when the contract requires it. A bad schema in a critical training label may justify stopping a publish. A harmless new category might be released with a warning and a fallback. Preserve the raw or failed batch within retention limits for repair. Record the rule version, observed value, affected IDs or partitions, chosen action and approving owner. After a fix, re-run validation and reconcile downstream outputs; clearing the alert without replay leaves consumers on bad data.

### Connect data health to model health

Input data drift may precede a model performance problem, but a distribution score alone does not establish one. Labels can arrive weeks later. Monitor prediction mix, missing-feature rate, fallback use and eventual outcomes alongside the data signals. A model can degrade with no obvious marginal input shift, and a shifted input can be harmless. When data is late or corrupted, fix or roll back the pipeline first. Consider model recalibration, retraining or deployment rollback only after evaluating the cause and the candidate's performance on relevant data.

## Triage a stale customer feature table

A customer-feature table is meant to be approved within 60 minutes of the newest source events. At noon the last approved load is from 10:30, so its load age is 90 minutes and the threshold is breached by 30. The alert identifies the table, approved partition, source watermark, last successful run and owner. This is more useful than "pipeline failed" because the pipeline may have succeeded with an old partition.

The responder checks the source watermark and finds events only through 09:30. The source is 150 minutes old, so both source delivery and publication are behind. If the source watermark were 11:55 instead, attention would move to validation or publication. Lineage lists the two model services and one report that read this table. One service has a reviewed fallback for a two-hour gap; the other must route requests to manual review. The report can show a stale-data badge. The same threshold breach produces different consumer actions because their costs differ.

The volume monitor reports 600 rows against an expected 1,000, a 40% shortfall. Counts by region show that all 400 missing rows are from one partner feed. The source owner confirms that its API cursor expired and the collector stopped advancing. A generic retrain alert would be wrong: the model cannot learn missing live records back into existence. The team restores the cursor, replays the missing interval and checks stable IDs to avoid duplicate deliveries. It then validates the repaired partition and publishes it under a new version.

The responder checks schema and distribution too. If row count returns to 1,000 but the null rate of a required feature jumps, the incident is not over. If a partner resent 400 old rows, the count could look right while the watermark remains stale. A count, timestamp and ID reconciliation together give stronger evidence. The incident record keeps the source position, failed and corrected output versions, affected consumers and exact interval of degraded operation.

### Separate an alert from a release decision

A late batch may still be internally consistent and useful for historical analysis. Quarantining it prevents every consumer from seeing it, including those who can tolerate age. Conversely, a fresh batch with wrong labels should be blocked even though its freshness monitor is green. Let the consumer contract choose the publication action; the observability system supplies evidence and routes responsibility. A dashboard can mark a dataset degraded while a training pipeline waits for repair.

An alert threshold should be evaluated at a predictable cadence. If it fires every minute during a known four-hour partner maintenance period, operators may become desensitised. Record planned maintenance and expected delay without hiding the resulting data age from consumers. A suppression of pages is not an assertion that the data is fresh. The status page or dataset metadata should still expose the watermark and age.

### Investigate a distribution change after repair

After the partner feed resumes, a region's average transaction amount rises. The source and schema monitors are green, so the team checks whether the population mix changed, whether currency conversion was applied correctly and whether a few extreme records dominate. Compare fixed reference bins and segment counts. If the increase reflects a legitimate new product, update the expected distribution after review. If it reflects a parser bug, repair and replay. A drift number alone cannot distinguish them.

When labels mature, evaluate model error on the affected region and time window. If outcomes are unchanged and the model remains calibrated, immediate retraining may add cost without benefit. If performance fell, create a candidate with corrected and representative data, test it against the current model and deploy under the established model-governance process. Keep the data incident linked to that decision so the team can explain why a model changed.

### Measure the observability system itself

Track time to detect, time to identify the upstream cause, time to notify consumers and time to repair. Sample incidents that monitors missed and add the smallest useful check. Review monitors that page too often without changing an action. A perfect-looking dashboard with many stale monitors is not reliable. Periodically inject a controlled bad batch: remove a mandatory field, delay a source watermark or duplicate a partition. Verify that the correct alert, owner and consumer status appear, and that a repaired batch clears the condition only after validation.

Use lineage to limit blast radius. A source correction should identify which partitions and model versions consumed the bad data, not force a blind rerun of every job. Conversely, do not assume lineage is complete until a few deployed outputs can be traced backward to exact source versions. That verification turns the five dimensions from labels on a slide into a practical incident system.

## Where this stands in 2026

:::info Industry view

- Monte Carlo's five-pillar taxonomy is a useful teaching model for freshness, volume, schema, distribution and lineage, not a complete definition of data correctness.
- Elementary documents freshness and volume anomaly monitors and schema checks for dbt workflows.
- Great Expectations Checkpoints connect validation results to configurable actions; policy determines whether a failure blocks or warns.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Using a flat trailing average as the volume baseline | A seven-day average is easy | Compare with the same hour of previous weeks. The flat baseline raised 712.2 false alarms per 1,000 batches |
| Alerting on a p-value below 0.05 | It is the standard threshold | A threshold of 0.05 alarms on 5% of healthy batches. Use an effect size or a far smaller p-value, and check what you lose |
| Counting only the strong incidents | The monitor caught the outage | Test the weak ones too. A p-value of 1e-6 missed all four incidents at quarter strength |
| Treating job success as data health | The run was green | Watch the data: freshness, volume and null rate caught every full-strength incident here, while no job failed |
| Retraining the model after a freshness alert | The data changed, so the model must | A late or partial batch needs a pipeline repair. Retrain only with outcome evidence |

## Practice questions

<details>
<summary><strong>Q1.</strong> What is data observability?</summary>

The ability to understand data health and detect problems before consumers do; the data equivalent of application monitoring.<br /><em>Lecture 16 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Name the five pillars of data observability.</summary>

Freshness, volume, schema, distribution, and lineage.<br /><em>Lecture 16 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Data was last loaded 90 minutes ago against a 60-minute freshness SLA. What happens?</summary>

Data age 90 > 60 → the freshness SLA is breached, firing an alert.<br /><em>Lecture 16 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How does observability relate to drift monitoring?</summary>

Observability monitors freshness, volume, schema, distribution and lineage. Combine those signals with model outcomes and incident evidence before deciding whether to repair data, retrain a model or roll back a release.<br /><em>Lecture 16 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is a volume check?</summary>

Verifying the expected amount of data arrived (e.g. row count within ±20% of the 7-day average); catching partial or duplicated loads.<br /><em>Lecture 16 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) A feed with a daily cycle between 500 and 1,500 rows is monitored with a flat band of 20% around 1,000. What share of healthy hours alarm, and why?</summary>

About 74%. The band covers 800 to 1,200, so the cycle term 0.5 sin(x) must lie within plus or minus 0.2, which holds for 4 x arcsin(0.4) / (2 x pi) = 26% of the day. The rest of the day the monitor is alarming on the cycle itself. A same-hour baseline fixes it.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) Hourly batches are tested at p below 0.05. How many false alarms per day should a team expect, and what are two ways to cut them?</summary>

24 x 0.05 = 1.2 per day. One way is to use a much smaller p-value, which loses weak incidents (in the experiment 1e-6 missed all four at quarter strength). Another is to alarm on an effect size such as the KS statistic over 0.10, or to require two alarms in a row, which adds about an hour of delay.

</details>

## Go deeper

- [Monte Carlo's original five-pillar explanation](https://www.montecarlodata.com/wp-content/uploads/2021/10/OReilly-Data-Quality-Fundamentals-early-release.pdf) defines the taxonomy.
- [Elementary documentation](https://docs.elementary-data.com/) describes data monitors and test results.
- [Great Expectations Checkpoint actions](https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/create_a_checkpoint_with_actions/) describes configurable responses.
- [Elementary documentation](https://docs.elementary-data.com/), opened 2026-10-09: anomaly tests for freshness, volume and custom data metrics in dbt models, and schema-change validation.
- [scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), opened 2026-10-09: the two-sample Kolmogorov-Smirnov test, whose statistic is the largest gap between the two empirical distribution functions. SciPy 1.18.1 was run.
- Built from the course lecture "dm-l16-observability" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can distinguish freshness, volume, schema, distribution and lineage signals.
- [ ] I can calculate a 30-minute breach from a 90-minute age and a 60-minute limit.
- [ ] I can separate source watermark age from approved-load age and trace a missing feed.
- [ ] I can route a data incident by consumer impact before choosing repair, fallback or model action.
- [ ] I can compute the share of healthy hours a flat volume band will flag on a cyclical feed.
- [ ] I can explain why a p-value threshold of 0.05 produces about 1.2 false alarms a day on hourly batches.
- [ ] I can read a table of detection rate, delay and false alarms and choose a monitor for a stated consumer cost.

## Where to go next

Next: the [question bank](/docs/mlops/data/question-bank), which tests the whole data management course. Related: [Lecture 9, analytics engineering and history](/docs/mlops/data/analytics-engineering-history), whose validity-window checks are one more kind of monitor.
