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

## The idea in plain words

A pipeline can finish without producing usable data. It may load an old partition, omit half of a source, change a type, duplicate rows or publish values from an unexpected population. Job success tells operators that code reached a terminal state; **data observability** asks whether the resulting data still serves its consumers. It combines measurements, context and incident response so that a problem is detected before a dashboard or model quietly makes a wrong decision.

The lecture organises data health into five pillars: **freshness, volume, schema, distribution and lineage**. Freshness asks whether data is recent enough for a decision. Volume asks whether the expected amount arrived. Schema checks structural compatibility. Distribution observes values, nulls and proportions. Lineage explains upstream origins and downstream impact. This five-pillar framing is a useful taxonomy associated with Monte Carlo's data-observability work, not a universal standard or a complete proof of quality. Correctness against reality, labels and business meaning may need additional checks.

<Infographic src="/img/dm/data-observability.svg" alt="Five data-health dimensions are freshness, volume, schema, distribution and lineage; a 90-minute data age against a 60-minute limit is a 30-minute breach." caption="Signals diagnose different failure modes; the data contract determines which response is appropriate." />

The source's worked example says the latest data load was **90 minutes** ago against a **60-minute** freshness limit. Under a contract measuring age since the last approved load, **90 > 60**, so the limit is breached by **30 minutes** and an alert should fire. That calculation does not reveal how old the source events are. A job could load now but contain events from yesterday. For a live model, track a source event-time watermark and the approved publication timestamp so that ingest lag and pipeline lag are distinguishable.

:::note Beyond the lecture

The source says breaches alert and quarantine, and links observability to retrain or rollback. The sections below make response conditional on the consumer contract and incident cause. A freshness breach may require an alert or fallback; it does not automatically justify quarantining otherwise valid data. Input drift is one signal, but model retraining requires outcome evidence and a cause analysis.

:::

The lab starts with the lecture's **90-minute age** and **60-minute limit**, reporting a **30-minute breach**. Change either control to see the state change. It measures time since the approved load; a source watermark is still needed to tell whether the loaded records themselves are current.

<FreshnessBudgetLab />

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

The lecture's freshness arithmetic is simple, but the clock must be named. This code measures minutes since the last approved load at a chosen observation time.

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

## Go deeper

- [Monte Carlo's original five-pillar explanation](https://www.montecarlodata.com/wp-content/uploads/2021/10/OReilly-Data-Quality-Fundamentals-early-release.pdf) defines the taxonomy.
- [Elementary documentation](https://docs.elementary-data.com/) describes data monitors and test results.
- [Great Expectations Checkpoint actions](https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/create_a_checkpoint_with_actions/) describes configurable responses.
- Built from the course lecture "dm-l16-observability" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can distinguish freshness, volume, schema, distribution and lineage signals.
- [ ] I can calculate a 30-minute breach from a 90-minute age and a 60-minute limit.
- [ ] I can separate source watermark age from approved-load age and trace a missing feed.
- [ ] I can route a data incident by consumer impact before choosing repair, fallback or model action.
