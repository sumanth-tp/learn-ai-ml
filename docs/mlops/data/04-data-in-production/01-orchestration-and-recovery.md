---
id: dm-orchestration-and-recovery
title: "Data Management · Lecture 10 — Orchestration and Recovery"
sidebar_label: "10 · Orchestration"
sidebar_position: 1
slug: /mlops/data/orchestration-and-recovery
description: "Schedule a pipeline DAG, handle dependencies and retries, and make backfills and publication safe."
tags: [data-management, orchestration, airflow, backfills]
---

import Infographic from '@site/src/components/Infographic';
import RetryBackoffLab from '@site/src/components/viz/RetryBackoffLab';

**In one line.** An orchestrator coordinates when work may run and records how it recovered; task code must still make reruns safe.

## The idea in plain words

A data pipeline is more than a list of scripts. One job waits for a source, another validates it, another builds features, and a final step publishes an approved dataset. If the first job runs late, downstream work must not pretend that yesterday's result is today's. If a middle task fails, recovery should resume from a known state. **Orchestration** is the control system for these dependencies, schedules and states.

The lecture models this system as a directed acyclic graph, or **DAG**. Each node is a task and each directed edge is a dependency. A DAG gives a scheduler enough structure to decide what is ready to run; it does not guarantee that a task is correct, that a source record is fresh or that a repeated write is harmless. Those guarantees come from the task's data contract and publication design. The logical date of a run is especially important: a task launched at 01:00 to process yesterday's partition must read and write the named partition, not whatever file currently looks newest.

<Infographic src="/img/dm/orchestration.svg" alt="An hourly trigger leads through a source wait, idempotent build and approved publish step; three retry waits total fourteen seconds." caption="Dependencies coordinate the work, while a stable partition key and publish contract make recovery safe." />

The worked retry example uses a base delay of 2 seconds with exponential backoff. The first three waits are **2, 4 and 8 seconds**, totalling **14 seconds** of deliberate waiting. Execution time, queue time and sensor waiting add to that total. The cron expression `0 * * * *` means minute zero of every hour. In UTC or a fixed-offset clock it names **24 runs per day**; a local day with a daylight-saving transition can have a different number of wall-clock hours.

:::note Beyond the lecture

The source introduces scheduling, sensors, retries, backfills and SLAs. The design sections below add logical dates, watermarks, idempotent publication, bounded sensor waits and the distinction between a task duration target and consumer data freshness.

:::

The retry lab begins with the lecture's base of **2 seconds** and **three retries**, giving **2 + 4 + 8 = 14 seconds**. Change either control to see the delays and total. A longer delay reduces pressure on an unhealthy service, but a retry policy also needs a maximum and an alert path.

<RetryBackoffLab />

## How it works

### DAGs, cron, sensors

Tasks run on a schedule (cron) or trigger; sensors wait for conditions; edges encode dependencies so a task starts only after upstreams succeed.

### Retries, backfills, SLAs

Retry transient failures with exponential backoff; idempotent tasks make backfills safe; SLAs flag late runs.

:::tip

**Worked.** base 2s → retries wait 2, 4, 8 s. Cron 0 * * * * = hourly (24/day).

:::


## A real system that works this way

**Apache Airflow** is a concrete orchestrator whose documentation defines tasks, dependencies, task states and sensors. A task instance belongs to a particular DAG run and logical date. A sensor can wait for a file or another condition before downstream work starts. Airflow can retry failed task instances, while its DAG graph prevents downstream tasks from running before their prerequisites satisfy the chosen trigger rule. This is a useful production control plane, provided the underlying operators use stable input and output identities.

Consider a daily customer-feature build. A source task waits for a dated batch and its manifest; a validation task checks schema and row count; a transform task builds a temporary feature partition; and a publication task makes the partition visible to model consumers. If the transform process dies after writing some files, a retry should replace or clean the temporary partition and publish only after validation. Merely adding a retry to a task that appends rows to a live table can create duplicates. Airflow knows the task failed; it cannot infer which side effects should be undone.

A backfill asks the graph to run for old logical dates. It is useful after a parser fix or an outage, but may compete for the same capacity as today's run. A team needs a policy for source snapshots, historical schema changes, late labels and downstream invalidation. A recovered task can be green while the published data is still late for its consumer, so alert on the age of the approved output as well as task state.

## Code you can run

First reproduce the source arithmetic. The waits are delays before attempts, not a promise about total wall-clock completion time.

```python
base_seconds = 2
retry_count = 3
waits = [base_seconds * 2 ** attempt for attempt in range(retry_count)]
print(waits, sum(waits))
assert waits == [2, 4, 8]
assert sum(waits) == 14
hours_in_utc_day = 24
run_minutes = [hour * 60 for hour in range(hours_in_utc_day)]
assert len(run_minutes) == 24
```

This small task runner separates computation from publication. The partition key is explicit; replacing the same key twice gives the same visible result. It does not model an actual object-store transaction, but it shows the property a production design must provide.

```python
from collections import Counter

source = {"2026-09-30": ["a", "b", "a"], "2026-10-01": ["b", "c"]}
published = {}

def build_partition(day):
    counts = Counter(source[day])
    candidate = tuple(sorted(counts.items()))
    assert sum(count for _, count in candidate) == len(source[day])
    published[day] = candidate
    return candidate

first = build_partition("2026-09-30")
second = build_partition("2026-09-30")
print(published)
assert first == second == (("a", 2), ("b", 1))
assert len(published) == 1
```

The same outcome holds only if the input snapshot and transform version are also fixed. If the source was corrected between attempts, the output may properly change; record that version and decide whether consumers should see a new revision.

## Designing with it

### Name the run and its data window

Give each run a logical date or interval and pass it explicitly into every task. A scheduler timestamp is not always the source event time or partition date. A task that selects `latest/` without a date can silently process the wrong batch during a backfill. Keep the source snapshot ID, input watermark and schema version with the run. When a source is late, decide whether the run waits, publishes a degraded result or skips; make that policy visible to consumers.

A cron schedule answers when a run becomes eligible. It does not mean data has arrived by that time. In Airflow, sensors are tasks that wait for a condition. A sensor needs a bounded wait, a meaningful failure state and a way to avoid occupying scarce execution capacity while an external system is slow. An event trigger may fit better than hourly polling when the source provides a reliable completion signal. If a feed can arrive twice, the signal needs an identity so it cannot launch two conflicting publications.

### Build retryable tasks

Retries help with transient network or service faults. They do not repair a deterministic schema error or a missing mandatory column. Classify failures so a bad batch is quarantined promptly rather than retried for hours. Use bounded exponential backoff with jitter when many tasks could hit the same service together. Set a timeout per attempt and a maximum elapsed time that fits the consumer's freshness budget. A retry log should show the attempt number, input identity and error category without leaking sensitive payloads.

Idempotency means repeating the same operation with the same inputs does not create an additional visible effect. A partition replace, a merge keyed by stable record ID or a versioned publish pointer can provide this property. An append-only insert without a unique key usually cannot. A task may perform several side effects, so the design must address failure between them. Write the candidate to a temporary location, validate it and then atomically advance a pointer or use a table transaction where supported. If publication fails, the previous approved version remains readable.

### Backfill deliberately

A backfill is a historical replay, not a copy of today's result into yesterday's slot. Confirm the historical source version, code version, schema and label availability. Late source corrections may make a corrected history desirable, but a forensic reconstruction of what a model saw at the time requires the old available values. Keep those use cases separate. Rate-limit backfills so they do not starve current scheduled work. State whether a successful backfill should invalidate downstream feature sets, training runs and reports.

### Measure the service promise

The lecture uses **SLA** for late runs. A task-duration promise, an alert deadline and a consumer freshness promise are related but distinct. A DAG can finish inside its own duration target yet publish data from a stale source; a slow task may still meet the consumer deadline if it started early. Measure the age of the latest approved partition at the point of use. Track missed schedules, sensor waits, queue time, retry time, task time and publication time separately. This turns a broad late-data page into a diagnosis.

## Recover a failed daily feature publication

At 01:00 UTC a run for the previous day's account events begins. The source manifest has not arrived, so the sensor waits. At 01:12 the manifest appears with a source snapshot ID, expected row count and checksum. The source task records that identity. The validation task rejects the batch if the checksum or schema is wrong. This creates a clear boundary: waiting was an upstream availability issue, whereas a failed contract is an input-quality issue.

The transform writes a candidate to a versioned temporary path. After half its files are written, a worker crashes. A retry starts with the same logical date and source snapshot. It either replaces the incomplete candidate or chooses a new attempt-specific staging path. It does not append to the approved partition. Once complete, the validator checks row count, key uniqueness, null rules and a few reconciliation totals. The publication step changes the visible version pointer only after those checks pass. Consumers never read the half-written attempt.

Suppose publication succeeds but the orchestrator loses its acknowledgement. The task is retried. A publish operation keyed by partition date and candidate version sees that this version is already active and returns success without duplicating rows. This is an important failure case because a task's state can be uncertain even when its side effect happened. The scheduler's retry mechanism and the storage system's idempotency mechanism must meet at an explicit identity.

At 01:25 the run is green, but the source manifest covers only data through 22:00, three hours behind. A task-state dashboard says success; a freshness monitor says the approved dataset misses the 00:00 target. The owner checks whether the source system stopped producing events or the manifest was generated too early. The pipeline should publish its watermark and degraded status so a downstream model can use a reviewed fallback rather than assuming current data.

Two days later, the source team repairs a missing hour. A backfill for that date uses the corrected snapshot and produces a new partition version. The team evaluates which training datasets or reports depended on the old version and reruns only the affected descendants. If the original decision history must be audited, keep the old version alongside the corrected one. A single green DAG run cannot answer both questions without versioned inputs and outputs.

### Design the incident response

The first alert should identify the logical date, failed task, last source watermark and expected publication deadline. An operator needs to know whether to wait, retry, repair data or run a backfill. For a transient API timeout, a bounded retry is appropriate. For a schema mismatch, immediate quarantine and a source-owner notification are more useful. For an empty but valid batch, the row-count contract must say whether zero is acceptable. Otherwise an empty partition can be published as success and suppress a missing-feed alert.

Make dependencies observable. A downstream task that waits on several upstreams should report which one is holding it and whether that upstream is failed, late or merely unscheduled. If a sensor checks for a file, verify that the file is complete using a manifest or stable completion marker, not just that a path exists. If a human approval gates a sensitive release, represent the approval as an explicit state with an audit record and timeout. Avoid implicit manual steps that leave the graph apparently stuck.

Finally, run a recovery drill. Interrupt a transform after its partial write, retry it, then compare the published partition and row count with a clean run. Interrupt just after publication and check that a retry cannot publish twice. Replay one old date while today's run is active and verify neither overwrites the other's output. These tests exercise the behaviour that a DAG drawing cannot prove. A production orchestrator is useful because it gives scheduling and state; reliability comes from the contracts around every task.

## Where this stands in 2026

:::info Industry view

- Airflow documents DAG tasks, task instances and sensors, giving a control plane for scheduled dependencies and waits.
- Teams increasingly treat approved data freshness and source watermarks as separate service measures from task success.
- A backfill is safe when source identity, logical date, side effects and downstream invalidation are all explicit.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What does a data orchestrator do?</summary>

Schedules and runs a pipeline DAG, tracks task state, and handles dependencies, retries, backfills and alerting (Airflow, Prefect, Dagster).<br /><em>Lecture 10 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What are sensors and backfills?</summary>

Sensors wait for an external condition (e.g. a file to arrive) before a task runs; backfills re-run the pipeline for past dates (safe when tasks are idempotent).<br /><em>Lecture 10 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Base delay 2s, exponential backoff. What are the first three retry delays?</summary>

For n = 0, 1, 2, base·2ⁿ gives 2, 4 and 8 seconds, a total of 14 seconds of waiting before execution and queue time.<br /><em>Lecture 10 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What does the cron '0 * * * *' mean?</summary>

Run at minute 0 of every hour; this is 24 runs per UTC or fixed-offset day, while a daylight-saving local day may differ.<br /><em>Lecture 10 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why must tasks be idempotent for backfills?</summary>

So rerunning a past date with the same input and version does not create duplicate visible effects; publication still needs an atomic or equivalent approval step.<br /><em>Lecture 10 · conceptual</em>

</details>

## Go deeper

- [Apache Airflow tasks](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html) explains task instances and states.
- [Apache Airflow sensors](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html) explains waiting for external conditions.
- Built from the course lecture "dm-l10-orchestration" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can explain what a DAG schedules and what task code must guarantee.
- [ ] I can compute the 2, 4 and 8 second waits and distinguish them from total completion time.
- [ ] I can give a run a logical date, source identity and safe publish key.
- [ ] I can decide how a failed task, a late source and a historical backfill should be handled.
