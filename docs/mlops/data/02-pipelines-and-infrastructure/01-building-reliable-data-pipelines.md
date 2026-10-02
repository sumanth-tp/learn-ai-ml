---
id: dm-building-reliable-data-pipelines
title: "Data Management · Session 4 — Building Reliable Data Pipelines"
sidebar_label: "4 · Data pipelines"
sidebar_position: 1
slug: /mlops/data/reliable-pipelines
description: "Design data DAGs, choose ETL or ELT and batch or streaming, and make retries and backfills safe."
tags: [data-management, data-pipelines, etl, streaming]
---

import Infographic from '@site/src/components/Infographic';
import PipelineFlowLab from '@site/src/components/viz/PipelineFlowLab';

**In one line.** A pipeline is reliable when its dependencies, retries and data effects are explicit.

## The idea in plain words

A data pipeline turns source events into a dataset that another system can trust. It may extract files, validate a schema, deduplicate records, compute features and publish a table. Drawing arrows between these actions is easy. The engineering work is deciding what happens when an action runs twice, arrives late, partly succeeds or is replayed for a past date.

The lecture models scheduled work as a **directed acyclic graph**, or DAG. Each node is a task and each arrow is a dependency. A validation task cannot inspect an export that has not arrived; a consumer publication should not run before validation. The graph gives the orchestrator an order and a place to retry. It does not describe the contents of a record or guarantee that a task's side effect is safe. A stream processor may have a long-running topology instead of a daily DAG, but the dependency and failure questions remain.

<Infographic src="/img/dm/pipeline-flow.svg" alt="A pipeline moves source events through extraction, idempotent transformation and publication; one hundred events per second at 0.2 seconds mean twenty in flight on average." caption="The data path is a graph of dependencies, with a measurable flow through it." />

Two design choices appear early. **ETL** transforms data before loading it into the analytical destination. **ELT** loads a raw or lightly checked copy first and transforms inside the destination. **Batch** processes a bounded collection on a schedule; **streaming** continuously handles events as they arrive. Neither pair is a quality ranking. Pick the path that meets freshness, governance, recovery and cost needs, then define the same business meaning across every path that computes the dataset.

:::note Beyond the lecture

The lecture introduces DAGs, ETL/ELT, batch/stream and Little's Law. This chapter adds task versus data semantics, failure cases, stable averages, late-arrival policy and a concrete backfill plan.

:::

The default controls reproduce the lecture's Little's Law result: 100 events each second spending 0.2 seconds in the system imply **20 events in flight on average**. Move either control to see the relationship. A stable long-run flow is required; if arrivals exceed capacity and a queue grows without bound, the displayed value is not a prediction of that unstable queue.

<PipelineFlowLab />

## How it works

### Transform before or after loading

- **ETL**; Transform before load; classic, controlled.
- **ELT**; Load raw, transform in-warehouse; cloud default.

### Throughput, latency, idempotency

Batch = throughput; stream = latency. Make stages idempotent, support backfills, handle late/duplicate data.

:::tip

**Worked.** λ=100/s, W=0.2s → L = λW = 20 events in flight.

:::


## A real system that works this way

**Apache Airflow** is a named orchestration example. Its documentation arranges tasks into DAGs with upstream and downstream dependencies, retries and sensors. Its backfill feature creates runs for earlier schedule dates, with controls over whether existing runs may be reprocessed and how many run concurrently. That is valuable when a parsing rule is corrected and last month's partitions must be rebuilt. The DAG schedules and observes work; the author still has to make each task's writes safe to repeat.

Suppose a subscription service exports daily payment events. A task lands the source file with a checksum. Another validates field types and event IDs. A third writes a curated partition. A final task publishes a freshness marker that a fraud-feature job watches. If validation fails, publishing must not declare the day ready. If the publish marker is written before all partition files commit, a consumer can read an incomplete day. Treat the marker or table commit as the consumer-visible boundary.

An operator backfills three days after discovering a currency conversion error. Every run must use the corrected transformation version, the right source snapshot and the intended logical date. A naive append would duplicate yesterday's rows. A deterministic partition replacement, keyed upsert or transactional table commit can make replay safe. Airflow will retry a failed task, but it cannot infer which of those data-write semantics the business intended.

## Code you can run

Little's Law relates average arrival rate, average time in a stable system and average work in flight. The units matter: events per second multiplied by seconds gives events.

```python
arrival_per_second = 100
mean_seconds = 0.2
average_inflight = arrival_per_second * mean_seconds
print(f"{arrival_per_second} events/s × {mean_seconds} s = {average_inflight:g} events")
assert average_inflight == 20

for new_time in (0.1, 0.5):
    print(new_time, arrival_per_second * new_time)
```

The second block models an idempotent batch write. A day is replaced inside one SQLite transaction, so replaying the same corrected events leaves one result per event. This is a small local example, not a distributed transaction across a source, warehouse and message broker.

```python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE daily (day TEXT, event_id TEXT, amount INTEGER, PRIMARY KEY(day, event_id))")

def publish_day(day, events):
    with db:
        db.execute("DELETE FROM daily WHERE day = ?", (day,))
        db.executemany(
            "INSERT INTO daily(day, event_id, amount) VALUES (?, ?, ?)",
            [(day, event_id, amount) for event_id, amount in events],
        )

source = [("E1", 20), ("E2", 30)]
publish_day("2026-02-01", source)
publish_day("2026-02-01", source)
rows = db.execute("SELECT event_id, amount FROM daily ORDER BY event_id").fetchall()
print(rows)
assert rows == source
publish_day("2026-02-01", [("E1", 20), ("E2", 35)])
assert db.execute("SELECT SUM(amount) FROM daily").fetchone()[0] == 55
db.close()
```

The corrected E2 amount replaces the earlier value. In a production system, choose the atomic write mechanism offered by the destination and avoid exposing half a partition while rebuilding it.

## Designing with it

### Choose ETL or ELT by control boundary

ETL is useful when data must be transformed or minimised before entering a destination, perhaps because raw fields are sensitive or the destination's compute is expensive. ELT is useful when a durable raw landing zone supports replay and the analytical engine can transform at scale. The source calls ELT a modern cloud default; that is a tendency, not a universal rule. A team may combine them: validate and remove prohibited fields before loading, then perform business transformations inside a warehouse. Decide where raw data may legally and operationally reside.

The destination should publish a clear contract: schema, key, partition meaning, update policy and freshness. A transformation that changes `amount` from cents to pounds must version or communicate that change. If a batch and a stream both populate the same product, reconcile overlapping dates and specify which one is authoritative after late corrections. Otherwise a dashboard and a model may disagree while both pipelines report success.

### Design retries and backfills before launch

**Idempotent** means the intended output is the same after one run or several runs with the same logical input. An idempotent task may overwrite a partition, merge on a stable event key or produce an immutable version and atomically move a pointer. A task that sends an email or charges a customer is not made idempotent merely by retrying it. Keep an operation key so the downstream effect can reject duplicates.

Backfills replay old logical dates. They need source availability, transformation version, bounded concurrency and a plan for downstream invalidation. If a daily feature changes for 30 dates, cached aggregates and model-training snapshots may also need rebuilding. Test one date first, compare row counts and sampled values, then widen the range. A backfill should leave an audit trail: why it ran, which version it used, what it changed and whether the consumer was paused or notified.

### Measure flow with the right units

Batch usually favours efficient throughput over immediate visibility; streaming generally reduces latency but adds state, ordering and recovery work. Little's Law, L = λW, is a relationship among **long-run averages in a stable system**. In the lecture, 100 events/s and 0.2 s give 20 concurrent events on average. If a stage can process only 80 events/s while 100 arrive, backlog grows; multiplying 100 by a fixed 0.2 s no longer describes the observed end-to-end time. Measure arrival rate, service rate, queue age, processing latency and publication lag separately. A low task runtime can coexist with a long wait in a queue.

## Walk through a failed daily run

At 02:00, the export task lands a file and records its checksum. Validation starts at 02:05 and finds an unexpected currency code. The safe result depends on the contract. If the code is a source typo, quarantine its rows and ask the source owner to correct them. If it is a valid new currency, update the allowed set and the conversion rule before replay. A blanket `except` that discards the rows keeps the DAG green while undercounting revenue. The consumer should see that the day's partition is not yet approved.

At 03:00, the source reissues the export. The landing task should identify whether it is the same file, a corrected version or a duplicate delivery. A content hash and source version help, but neither substitutes for a business event ID. If one file contains the same payment twice, a hash of the entire file does not find the duplicate. Validate keys and define whether corrections have new IDs or versions.

The transformation task then crashes after writing half its output. A transactional replacement or write-to-new-version design keeps the previous approved partition visible until the new one commits. If the destination cannot make the whole write atomic, the pipeline needs a readiness marker that is set only after every part succeeds. A retry must clean or overwrite partial work. Otherwise a consumer can combine half of the old day with half of the corrected one.

After repair, run one logical date and compare it with source counts, accepted counts, quarantine counts and a small set of known payments. Expand the backfill with a concurrency limit so it does not starve current ingestion. Record source watermark and output version for each date. The goal is not merely to make Airflow's boxes green; it is to give the consumer a complete, traceable and correct view of the business day.

### Distinguish clock times

An event may have occurrence time, source commit time, arrival time and processing time. A payment at 23:59 can arrive after midnight; a late correction may arrive days later. Decide which time assigns the business-day partition and how long it remains open for late events. If a stream uses a watermark, document what happens beyond it: drop, quarantine, compensate or backfill. Train/serve feature parity depends on using the same time policy in historical and live paths.

### Treat the pipeline as a data product

The producer should publish more than a table name. Give consumers the row grain, key, field meanings, time basis, expected refresh and quality status. If `daily_revenue` excludes refunded payments, say whether refunds are subtracted from the original day or posted as negative events on the refund day. The answer changes a model's features and an analyst's chart. A versioned contract makes such a change reviewable; an unannounced SQL edit can make every downstream run technically successful but semantically wrong.

Map the failure domain of each stage. A malformed source file should not block unrelated partitions if they can be processed independently. On the other hand, publishing one region's data as if the whole world were complete may be worse than holding the result. A partition-level readiness state can distinguish complete, partial and failed output. Give the consumer a way to query that state, not just an engineer-only task log.

Capacity planning needs service time as well as arrival rate. If arrivals average 100/s but spike to 300/s for ten minutes, the system needs either spare processing capacity or a queue that can absorb the burst. Estimate how long recovery takes after the spike: a worker that can drain only 110/s clears backlog slowly. Monitor the age of the oldest unprocessed event, not only the queue length, because an old small backlog may be more harmful than a young large one. Test throttling and retry behaviour under a burst so failures do not multiply the load.

Finally, rehearse a replay. Save a known input, run the stage twice, compare output IDs and totals, then inject a failure between write and commit. A stage that behaves correctly in the happy path may duplicate data on its second attempt. The cheapest time to discover that is before the first production backfill. Include a corrected input in the rehearsal as well: idempotency with identical input is easier than replacing a formerly approved result without leaving a stale copy in downstream caches.

## Where this stands in 2026

:::info Industry view

- Current Airflow documentation treats task dependencies, retries and backfills as separate orchestration concepts; task code supplies its own data-write semantics.
- ELT is widespread where analytical engines can transform landed data, but sensitive inputs and workload economics may still favour earlier transformation.
- Pipeline operations increasingly measure data freshness and quality at the consumer boundary, not only task completion.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is a data pipeline and how is it modelled?</summary>

A system that moves data from sources to destinations while transforming it, modelled as a DAG (directed acyclic graph) of tasks with dependencies, run on schedule or trigger.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast ETL and ELT.</summary>

ETL transforms before loading; ELT loads a raw or lightly checked copy, then transforms in the destination; it is common in cloud analytics when governance permits raw landing.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Why must pipeline stages be idempotent?</summary>

So re-running (after failure or backfill) yields the same result without duplicates or corruption; essential for reliable recovery.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> A stage processes λ=100 events/s at W=0.2 s each. How many are in flight?</summary>

Little's Law L = λW = 100 × 0.2 = 20 events in flight.<br /><em>Session 4 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Contrast batch and streaming pipelines.</summary>

Batch processes bounded data on a schedule (throughput, latency-tolerant); streaming processes unbounded events in near real time (low latency).<br /><em>Session 4 · conceptual</em>

</details>

## Go deeper

- [Airflow tasks](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html) covers dependencies and retries.
- [Airflow backfill](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/backfill.html) describes historical runs and concurrency controls.
- Built from the course lecture "dm-s4-pipelines" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can draw a task DAG and identify the consumer-visible publish point.
- [ ] I can choose ETL or ELT and batch or streaming from specific constraints.
- [ ] I can make a stage safe to retry and plan a bounded backfill.
- [ ] I can compute L = λW while stating the stable-average assumption.
