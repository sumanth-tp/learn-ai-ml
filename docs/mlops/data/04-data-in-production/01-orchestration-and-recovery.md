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

:::tip Before you start

**You should already know**

- What a pipeline stage is and why it can fail ([Session 4, building reliable data pipelines](/docs/mlops/data/reliable-pipelines)).
- What a logical date and a partition are ([Lecture 9, analytics engineering and history](/docs/mlops/data/analytics-engineering-history)).

**Reading time:** about 40 minutes, plus a minute to run the code.

**After this chapter you can**

- Work out the finish time of a retried pipeline by hand, with and without resuming from the failed task.
- Say what a retry schedule can and cannot rescue, and show a backfill that duplicates data.
- Choose between append and replace for a publish step.

:::

## In 30 seconds

A pipeline is a chain of jobs: wait for the data, fetch it, check it, transform it, publish it. Jobs fail for dull reasons, such as a network blip or a service that is briefly down. An orchestrator is the foreman that runs them in order, tries again when one fails and records what happened.

The foreman cannot know whether a job already half-did its work. If a retry adds the same rows twice, the foreman was no help. Safe reruns come from how each job writes, not from the scheduler.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| DAG | A set of tasks joined by "run after" arrows with no loops | extract, then validate, then publish |
| Sensor | A task that waits for a condition | Wait for today's file |
| Retry | Run a failed task again | Up to 3 more attempts |
| Backoff | Wait longer before each retry | 2, 4 then 8 seconds |
| Backfill | Run the pipeline for past dates | Replay the last 30 days |
| Idempotent | Running twice has the same visible effect as once | Replace the 1 October partition |
| Logical date | The data window a run is for, not the clock time it started | The run for 30 September |
| SLA | A deadline for a run or an output | Published by 06:00 |


## The idea in plain words

A data pipeline is more than a list of scripts. One job waits for a source, another validates it, another builds features, and a final step publishes an approved dataset. If the first job runs late, downstream work must not pretend that yesterday's result is today's. If a middle task fails, recovery should resume from a known state. **Orchestration** is the control system for these dependencies, schedules and states.

The usual model for this system is a directed acyclic graph, or **DAG**. Each node is a task and each directed edge is a dependency. A DAG gives a scheduler enough structure to decide what is ready to run; it does not guarantee that a task is correct, that a source record is fresh or that a repeated write is harmless. Those guarantees come from the task's data contract and publication design. The logical date of a run is especially important: a task launched at one in the morning to process yesterday's partition must read and write the named partition, not whatever file currently looks newest.

<Infographic src="/img/dm/orchestration.svg" alt="An hourly trigger leads through a source wait, idempotent build and approved publish step; three retry waits total fourteen seconds." caption="Dependencies coordinate the work, while a stable partition key and publish contract make recovery safe." />

The worked retry example uses a base delay of 2 seconds with exponential backoff. The first three waits are **2, 4 and 8 seconds**, totalling **14 seconds** of deliberate waiting. Execution time, queue time and sensor waiting add to that total. The cron expression `0 * * * *` means minute zero of every hour. In UTC or a fixed-offset clock it names **24 runs per day**; a local day with a daylight-saving transition can have a different number of wall-clock hours.

:::note Added for this site

The course material covers scheduling, sensors, retries, backfills and SLAs. The design sections below add logical dates, watermarks, idempotent publication, bounded sensor waits and the distinction between a task duration target and consumer data freshness.

:::

The retry lab begins with a base of **2 seconds** and **three retries**, giving **2 + 4 + 8 = 14 seconds**. Change either control to see the delays and total. A longer delay reduces pressure on an unhealthy service, but a retry policy also needs a maximum and an alert path.

<RetryBackoffLab />

## Worked example, step by step

A five-task chain takes minutes 5 (wait for source), 10 (extract), 3 (validate), 20 (transform) and 2 (publish), so 40 minutes with no trouble. The transform fails once, at its end. The retry wait is 1 minute.

1. **Whole-DAG restart.** Time spent before the failure is 5 + 10 + 3 + 20 = 38. Wait 1, then rerun everything: 40 more. Finish: 38 + 1 + 40 = 79 minutes, with 78 minutes of compute.
2. **Resume from the failed task.** Finish the same 38, wait 1, then rerun only the transform and publish: 20 + 2 = 22. Finish: 38 + 1 + 22 = 61 minutes, with 60 of compute.
3. **Difference.** 79 - 61 = 18 minutes, the cost of redoing the 5 + 10 + 3 that were already good.
4. **Backoff against an outage.** The source is down from minute 5 to minute 11. Each failed attempt fails fast, in 0.2 minutes. With waits 1, 1, 1 the attempts start at 5.0, 6.2, 7.4 and 8.6. All four land inside the outage, so the run fails. With waits 1, 2, 4 the attempts start at 5.0, 6.2, 8.4 and 12.6. The fourth lands after the outage and succeeds.
5. **Backfill with a lost acknowledgement.** Three days each write 1,000 rows, and day 2 is written twice because the first acknowledgement was lost. Append gives 4,000 rows. Replace by day gives 3,000.

In words: resuming saves the finished work, a longer total wait can outlast an outage, and replace-by-key makes a duplicate write harmless. The first code block below prints these numbers.

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

First reproduce the backoff arithmetic. The waits are delays before attempts, not a promise about total wall-clock completion time.

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

### The worked example in code

This block reproduces the five steps of the worked example with a few lines of arithmetic.

```python
minutes = [5, 10, 3, 20, 2]
before_failure = sum(minutes[:4])
whole = before_failure + 1 + sum(minutes)
resume = before_failure + 1 + sum(minutes[3:])
print("whole", whole, "resume", resume, "saved", whole - resume)

def finish(waits, outage=(5.0, 11.0), fast_fail=0.2):
    clock = 5.0
    for wait in waits + [None]:
        if not outage[0] <= clock < outage[1]:
            return clock
        clock += fast_fail
        if wait is None:
            return None
        clock += wait
    return None

print("waits 1,1,1", finish([1, 1, 1]), "waits 1,2,4", finish([1, 2, 4]))
rows_append = sum([1000, 1000, 1000]) + 1000
print("append", rows_append, "replace", 3 * 1000)
```

**Reading the output.** It prints whole 79, resume 61, saved 18. Then `None` for waits 1, 1, 1 (the run fails) and 12.6 for waits 1, 2, 4 (the fourth attempt starts at minute 12.6). Last, append 4000 against replace 3000.

### An experiment on retry and recovery

Which retry choices matter, and how much? The block below simulates 4,000 runs of the five-task chain for each of five policies in virtual time, so nothing sleeps. A transient blip fails a task at its end with a task-specific chance (15% for extract, 25% for transform). In 20% of runs a source outage of 3 to 10 minutes begins in the first 40 minutes, and `extract` and `publish` fail fast while it lasts. The chain is built with networkx, results are summarised with pandas, and a second part replays a 30-day backfill into two DuckDB tables, one appended to and one replaced by day.

Versions used: Python 3.14.6, networkx 3.6.1, pandas 2.3.3, DuckDB 1.5.6, NumPy 2.5.3. All failures are simulated. It runs in about a second.

```python
import duckdb
import networkx as nx
import numpy as np
import pandas as pd

dag = nx.DiGraph([("wait_source", "extract"), ("extract", "validate"), ("validate", "transform"), ("transform", "publish")])
order = list(nx.topological_sort(dag))
minutes = {"wait_source": 5, "extract": 10, "validate": 3, "transform": 20, "publish": 2}
blip = {"wait_source": 0.02, "extract": 0.15, "validate": 0.02, "transform": 0.25, "publish": 0.05}
outage_tasks = {"extract", "publish"}

def attempt(rng, task, start, outage):
    if outage is not None and task in outage_tasks and outage[0] <= start < outage[0] + outage[1]:
        return "down", 0.2
    if rng.random() < blip[task]:
        return "blip", minutes[task]
    return "ok", minutes[task]

def run(rng, delays, resume):
    outage = (rng.uniform(0, 40), rng.uniform(3, 10)) if rng.random() < 0.2 else None
    clock = compute = 0.0
    done, budget = set(), {}
    while len(done) < len(order):
        for task in order:
            if task in done:
                continue
            kind, cost = attempt(rng, task, clock, outage)
            clock += cost
            compute += cost
            if kind != "ok":
                waits = budget.setdefault(task if resume else "all", list(delays))
                if not waits:
                    return False, clock, compute
                clock += waits.pop(0)
                if not resume:
                    done.clear()
                break
            done.add(task)
    return True, clock, compute

policies = {
    "no retry": ([], True),
    "whole DAG, 3 x 1 min": ([1, 1, 1], False),
    "per task, 3 x 1 min": ([1, 1, 1], True),
    "per task, backoff 1,2,4": ([1, 2, 4], True),
    "per task, backoff 1,2,4,8": ([1, 2, 4, 8], True),
}
rows = []
for name, (delays, resume) in policies.items():
    rng = np.random.default_rng(11)
    results = [run(rng, delays, resume) for _ in range(4000)]
    frame = pd.DataFrame(results, columns=["ok", "finish", "compute"])
    rows.append((name, frame.ok.mean(), frame[frame.ok].finish.mean(), frame.compute.mean()))
print(pd.DataFrame(rows, columns=["policy", "success", "finish_min", "compute_min"]).round(3).to_string(index=False))

con = duckdb.connect()
con.sql("create table appended (day integer, n integer)")
con.sql("create table replaced (day integer primary key, n integer)")
rng = np.random.default_rng(12)
truth = writes_total = 0
for day in range(30):
    n = int(rng.integers(900, 1100))
    truth += n
    writes = 1
    while rng.random() < 0.2:
        writes += 1
    writes_total += writes
    for _ in range(writes):
        con.sql(f"insert into appended values ({day}, {n})")
        con.sql(f"insert or replace into replaced values ({day}, {n})")
print("30-day backfill, rows expected", truth, "writes issued", writes_total)
print("append total", con.sql("select sum(n) from appended").fetchone()[0], "replace total", con.sql("select sum(n) from replaced").fetchone()[0])
print("days duplicated by append", con.sql("select count(*) from (select day from appended group by day having count(*) > 1)").fetchone()[0])
```

The output of the run:

```text
                   policy  success  finish_min  compute_min
                 no retry    0.546      40.000       33.992
     whole DAG, 3 x 1 min    0.963      57.722       58.831
      per task, 3 x 1 min    0.968      49.109       47.872
  per task, backoff 1,2,4    0.988      49.422       48.477
per task, backoff 1,2,4,8    0.998      49.924       48.902
30-day backfill, rows expected 30054 writes issued 39
append total 38951 replace total 30054
days duplicated by append 8
```

**Reading the output.** `success` is the share of runs that finished, `finish_min` is the mean finish time of the successful runs and `compute_min` is the mean minutes of work per run, including failed attempts. With no retry only 54.6% of runs finish. The last three lines belong to the backfill: 39 writes were issued for 30 days, and 8 days were written more than once.

**Line by line.**

- `budget.setdefault(task if resume else "all", list(delays))` gives each task its own retry budget when resuming, and one shared budget for the whole run when a failure restarts everything.
- `if not resume: done.clear()` is the whole difference between the two recovery styles: a restart forgets the finished tasks.
- `insert or replace into replaced` uses the primary key on `day`, so a repeated write of the same day replaces the earlier row.

### What the numbers say

Retrying at all is the big step: 0.546 up to 0.968 with three one-minute retries per task. Resuming from the failed task beat restarting the whole chain at the same retry count (0.968 against 0.963 success), and it finished faster, in 49.1 minutes against 57.7, because it did not redo finished work. It also used about 11 fewer minutes of compute per run, 47.9 against 58.8.

The surprise is the schedule. Backoff with waits of 1, 2 and 4 minutes reached 0.988 against 0.968 for three waits of 1 minute, but the gain came from the longer total wait (7 minutes against 3 minutes), because a 3 to 10 minute outage outlasts three short waits. Adding an 8-minute wait took success to 0.998 and cost half a minute of mean finish time. Backoff is protection against a long outage, and it is not a free improvement.

The backfill is a plain warning. Append produced 38,951 rows against 30,054 expected, 29.6% too many, from 9 duplicate writes. Replace by key produced exactly 30,054.

Limits: the failure rates and outage lengths are my assumptions, one seed per policy, no queueing and no shared capacity. The ranking is a property of those assumptions. Measure your own failure mix before choosing a schedule.

<Infographic src="/img/dm-enrich/dm2-retry-recovery.svg" alt="Bars of success rate for five retry policies and a comparison of backfill rows, 38,951 appended against 30,054 replaced." caption="Look first at how success climbs with total waiting time, then at the backfill row counts on the right." />

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

A **service level agreement (SLA)** flags late runs. A task-duration promise, an alert deadline and a consumer freshness promise are related but distinct. A DAG can finish inside its own duration target yet publish data from a stale source; a slow task may still meet the consumer deadline if it started early. Measure the age of the latest approved partition at the point of use. Track missed schedules, sensor waits, queue time, retry time, task time and publication time separately. This turns a broad late-data page into a diagnosis.

## Recover a failed daily feature publication

A run for the previous day's account events begins at one in the morning UTC. The source manifest has not arrived, so the sensor waits. Twelve minutes later the manifest appears with a source snapshot ID, expected row count and checksum. The source task records that identity. The validation task rejects the batch if the checksum or schema is wrong. This creates a clear boundary: waiting was an upstream availability issue, whereas a failed contract is an input-quality issue.

The transform writes a candidate to a versioned temporary path. After half its files are written, a worker crashes. A retry starts with the same logical date and source snapshot. It either replaces the incomplete candidate or chooses a new attempt-specific staging path. It does not append to the approved partition. Once complete, the validator checks row count, key uniqueness, null rules and a few reconciliation totals. The publication step changes the visible version pointer only after those checks pass. Consumers never read the half-written attempt.

Suppose publication succeeds but the orchestrator loses its acknowledgement. The task is retried. A publish operation keyed by partition date and candidate version sees that this version is already active and returns success without duplicating rows. This is an important failure case because a task's state can be uncertain even when its side effect happened. The scheduler's retry mechanism and the storage system's idempotency mechanism must meet at an explicit identity.

Later that morning the run is green, but the source manifest covers only data through ten at night, three hours behind. A task-state dashboard says success; a freshness monitor says the approved dataset misses the midnight target. The owner checks whether the source system stopped producing events or the manifest was generated too early. The pipeline should publish its watermark and degraded status so a downstream model can use a reviewed fallback rather than assuming current data.

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Restarting the whole DAG on a failure | It is the simplest recovery and always consistent | Retry and resume per task. In the experiment a restart finished in 57.7 minutes against 49.1 |
| Adding retries to a task that appends | Retries fix flaky jobs | Make the write idempotent first. Append inflated a 30-day backfill by 29.6% |
| Using the same short wait for every retry | A fixed delay is easy to reason about | Let the total wait exceed the outages you expect. Waits of 1, 2, 4 reached 0.988 where three of 1 minute reached 0.968 |
| Treating green as fresh | The task succeeded | Alert on the age of the approved output, not only on task state |
| Retrying a bad batch for hours | Persistence feels responsible | Classify failures. A schema error is quarantined and sent to the owner, not retried |

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

<details>
<summary><strong>Q6.</strong> (Medium) A 40-minute chain fails at the end of a 20-minute task after 38 minutes of work. A retry waits 1 minute. What is the finish time if you resume, and if you restart?</summary>

Resume: 38 + 1 + (20 + 2) = 61 minutes, because only the transform and publish steps repeat. Restart: 38 + 1 + 40 = 79 minutes. The 18-minute difference is the finished work (5 + 10 + 3) that the restart redoes.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) An outage lasts 8 minutes from the moment of the first attempt, and each failed attempt takes 0.2 minutes. Can waits of 1, 1, 1 survive it, or 1, 2, 4?</summary>

Neither. With 1, 1, 1 the four attempts start at 0, 1.2, 2.4 and 3.6, all inside the outage. With 1, 2, 4 they start at 0, 1.2, 3.4 and 7.6, and the last is still inside it. To survive you need a longer total wait, for example adding an 8-minute wait, so the fifth attempt starts at 15.8. The lesson is to compare the total wait with the outage length you must survive, not to admire the shape of the schedule.

</details>

## Go deeper

- [Apache Airflow tasks](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html) explains task instances and states.
- [Apache Airflow sensors](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html) explains waiting for external conditions.
- Apache Airflow tasks (the link above), opened 2026-10-09 for Airflow 3.3.2: `retries` caps the attempts, `retry_delay` sets the wait, `retry_exponential_backoff` and `max_retry_delay` exist as settings, and `up_for_retry` is the state of a failed task with attempts left. The page says nothing about idempotency, so that duty stays with the task code.
- [networkx topological_sort](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html), opened 2026-10-09: orders the nodes of a directed acyclic graph so every dependency comes first.
- Built from the course lecture "dm-l10-orchestration" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can explain what a DAG schedules and what task code must guarantee.
- [ ] I can compute the 2, 4 and 8 second waits and distinguish them from total completion time.
- [ ] I can give a run a logical date, source identity and safe publish key.
- [ ] I can decide how a failed task, a late source and a historical backfill should be handled.
- [ ] I can work out by hand the finish time of a retried chain with and without resuming from the failed task.
- [ ] I can say why waits of 1, 2, 4 beat three waits of 1 in the experiment, and what would beat both.
- [ ] I can show that an append-based publish duplicates data on a retry and a keyed replace does not.

## Where to go next

Next: [Lecture 12, experiments, metadata and lineage](/docs/mlops/data/experiments-metadata-lineage), which records what each run read and wrote. Related: [Lecture 16, observing data in production](/docs/mlops/data/data-observability), which watches the freshness this chapter promises.
