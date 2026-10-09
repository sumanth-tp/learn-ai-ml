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


:::tip Before you start

**You should already know**

- What a table with a primary key is, and what an upsert does.
- Where tables sit in a lake or warehouse: [warehouses, lakes and lakehouses](/docs/mlops/data/warehouses-lakes-and-lakehouses).
- Basic probability: a 30% chance means about 3 in 10 attempts.

**Reading time.** About 45 minutes, plus about 12 seconds to run the experiment.

**After this chapter you can**

- explain why a retried task duplicates data unless its write is idempotent,
- compare append, upsert and replace-by-partition writes by the number of duplicate rows they leave,
- say when Little's Law holds and show what happens to it in an overloaded queue.

:::

## In 30 seconds

You press the lift button, nothing seems to happen, so you press it again. A good lift treats the second press as the same request. A bad one sends two lifts. Data pipelines retry tasks the same way, because networks fail after the work is done but before the "done" message arrives. If the task's write is idempotent (the same input twice gives the same result) the retry is harmless. If it simply appends, every retry adds duplicate rows.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| DAG | A graph of tasks with arrows for "run after", and no loops | extract, validate, transform, publish |
| Idempotent | Running twice has the same effect as running once | Replace day 5, not append to it |
| Retry | Running a failed task again | Up to 5 attempts per day |
| Lost acknowledgement | The work succeeded but the caller never heard | The write landed, the reply timed out |
| Backfill | Re-running past dates, for example after a bug fix | Rebuild last week |
| Upsert | Insert a row, or update it if its key exists | `ON CONFLICT DO UPDATE` |
| Little's Law | Average in flight = arrival rate × average time in system | 100 per second × 0.2 s = 20 |
| Throughput and latency | Work finished per second, and time for one item | 100 per second, 0.2 s |

## The idea in plain words

A data pipeline turns source events into a dataset that another system can trust. It may extract files, validate a schema, deduplicate records, compute features and publish a table. Drawing arrows between these actions is easy. The engineering work is deciding what happens when an action runs twice, arrives late, partly succeeds or is replayed for a past date.

Scheduled work is modelled as a **directed acyclic graph**, or DAG. Each node is a task and each arrow is a dependency. A validation task cannot inspect an export that has not arrived; a consumer publication should not run before validation. The graph gives the orchestrator an order and a place to retry. It does not describe the contents of a record or guarantee that a task's side effect is safe. A stream processor may have a long-running topology instead of a daily DAG, but the dependency and failure questions remain.

<Infographic src="/img/dm/pipeline-flow.svg" alt="A pipeline moves source events through extraction, idempotent transformation and publication; one hundred events per second at 0.2 seconds mean twenty in flight on average." caption="The data path is a graph of dependencies, with a measurable flow through it." />

Two design choices appear early. **ETL** transforms data before loading it into the analytical destination. **ELT** loads a raw or lightly checked copy first and transforms inside the destination. **Batch** processes a bounded collection on a schedule; **streaming** continuously handles events as they arrive. Neither pair is a quality ranking. Pick the path that meets freshness, governance, recovery and cost needs, then define the same business meaning across every path that computes the dataset.

:::note Added for this site

The course introduces DAGs, ETL/ELT, batch/stream and Little's Law. This chapter adds task versus data semantics, failure cases, stable averages, late-arrival policy, a concrete backfill plan and a measured comparison of append, upsert and replace writes under retries.

:::

The default controls reproduce the Little's Law result: 100 events each second spending 0.2 seconds in the system imply **20 events in flight on average**. Move either control to see the relationship. A stable long-run flow is required; if arrivals exceed capacity and a queue grows without bound, the displayed value is not a prediction of that unstable queue.

<PipelineFlowLab />

**What each control does.**

- **arrival rate** sets events per second, 20 to 200.
- **mean time in pipeline** sets the average seconds an event spends inside, 0.1 to 1.0.

**Try it yourself.**

1. Defaults: 100 events per second at 0.2 seconds gives 20 events in flight.
2. Double the arrival rate to 200. In flight doubles to 40, if the time per event stays the same.
3. Set the time to 0.1 seconds. In flight falls to 10. In a real overloaded system the time would grow with the rate, which is the 120% busy case in the experiment, where the formula no longer applies.

## Worked example, step by step

One day of data holds 50 events. The write succeeds, but the acknowledgement is lost 30% of the time, so the orchestrator retries.

1. If every attempt can lose its acknowledgement with probability 0.3, the number of attempts until one acknowledgement arrives averages 1 / (1 - 0.3) = 1.43.
2. Over 100 days that is about 100 × 1.43 = 143 attempts, which is 43 extra. The experiment below counted 142.
3. An append write adds the 50 events on every attempt. The 42 extra attempts add 42 × 50 = 2,100 duplicate rows to a table that should hold 100 × 50 = 5,000.
4. An upsert on the key (day, event_id) overwrites the same 50 rows each time, so the table stays at 5,000 rows.
5. A replace-by-day write deletes day d and inserts it again inside one transaction, which also leaves 5,000 rows.
6. Little's Law: with 100 events per second and 0.04 seconds in the system, the average in flight is 100 × 0.04 = 4.

In words: a retry costs nothing if the write has a key to overwrite and costs a duplicate row per event if it does not. The experiment runs these write modes with DuckDB as the sink and then tests Little's Law on a simulated queue.

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

### Experiment: retries, duplicates and a queue that does not settle

The first half simulates 100 days of 50 events with DuckDB 1.5.6 as the sink and acknowledgements lost with probability 0, 0.1 and 0.3. The second half simulates a single-server queue with NumPy 2.5.3 at 80% and 120% load to test Little's Law. Everything is synthetic and seeded.

```python
import duckdb
import numpy as np

DAYS, EVENTS = 100, 50

def run_pipeline(write_mode, ack_lost, seed=3, max_attempts=5):
    rng = np.random.default_rng(seed)
    con = duckdb.connect()
    con.execute("CREATE TABLE sink (day INTEGER, event_id INTEGER, amount INTEGER, PRIMARY KEY (day, event_id))")
    con.execute("CREATE TABLE log (day INTEGER, event_id INTEGER, amount INTEGER)")
    attempts = 0
    for day in range(DAYS):
        batch = [(day, e, int(rng.integers(1, 100))) for e in range(EVENTS)]
        for _ in range(max_attempts):
            attempts += 1
            if write_mode == "append":
                con.executemany("INSERT INTO log VALUES (?, ?, ?)", batch)
            elif write_mode == "upsert":
                con.executemany("INSERT INTO sink VALUES (?, ?, ?) ON CONFLICT DO UPDATE SET amount = excluded.amount", batch)
            else:
                con.execute("BEGIN")
                con.execute("DELETE FROM sink WHERE day = ?", [day])
                con.executemany("INSERT INTO sink VALUES (?, ?, ?)", batch)
                con.execute("COMMIT")
            if rng.random() >= ack_lost:
                break
    table = "log" if write_mode == "append" else "sink"
    rows, total = con.execute(f"SELECT count(*), sum(amount) FROM {table}").fetchone()
    return rows, total, attempts

expected = DAYS * EVENTS
print(f"expected rows {expected}")
totals = {}
for ack_lost in (0.0, 0.1, 0.3):
    for mode in ("append", "upsert", "replace-day"):
        rows, total, attempts = run_pipeline(mode, ack_lost)
        totals[(mode, ack_lost)] = total
        print(f"ack lost {ack_lost:.0%} {mode:12s} attempts {attempts:4d} rows {rows:6d} duplicates {rows - expected:5d}")
print(f"revenue at 30% lost acks: append {totals[('append', 0.3)]}, upsert {totals[('upsert', 0.3)]}, overstated by {totals[('append', 0.3)] / totals[('upsert', 0.3)] - 1:.1%}")

def single_server(rate, mean_service, seconds=300, seed=1):
    rng = np.random.default_rng(seed)
    arrive = np.cumsum(rng.exponential(1 / rate, int(rate * seconds)))
    work = rng.exponential(mean_service, arrive.size)
    finish, free_at = np.empty_like(arrive), 0.0
    for i in range(arrive.size):
        free_at = max(free_at, arrive[i]) + work[i]
        finish[i] = free_at
    return arrive, finish

for label, mean_service in (("80% busy", 0.008), ("120% busy", 0.012)):
    arrive, finish = single_server(100.0, mean_service)
    stay = finish - arrive
    probe = np.linspace(60, 240, 400)
    in_flight = np.searchsorted(arrive, probe) - np.searchsorted(np.sort(finish), probe)
    window = (arrive > 60) & (arrive < 240)
    print(f"{label}: mean stay {stay[window].mean():.3f}s, lambda x W = {window.sum() / 180 * stay[window].mean():.1f}, counted in flight {in_flight.mean():.1f}, stay in the last 30 s {stay[arrive > 270].mean():.3f}s")
```

**Reading the output.** The first table has one line per combination of lost-acknowledgement rate and write mode, with the count of attempts, the final row count and the duplicates. `revenue` compares the append total with the upsert total. The last two lines compare the queue's measured behaviour with Little's Law.

**Line by line.**

- `INSERT ... ON CONFLICT DO UPDATE` uses the primary key `(day, event_id)` so a repeated row overwrites instead of adding.
- The `BEGIN`, `DELETE`, `INSERT`, `COMMIT` sequence is the replace-by-partition pattern. A crash before `COMMIT` leaves the previous day untouched.
- `if rng.random() >= ack_lost: break` models the lost acknowledgement: the write already happened, but the loop retries anyway.
- `np.searchsorted(arrive, probe) - np.searchsorted(np.sort(finish), probe)` counts, at each probe time, the events that have arrived but not finished, an independent measure of items in flight.

The printed output:

```text
expected rows 5000
ack lost 0% append       attempts  100 rows   5000 duplicates     0
ack lost 0% upsert       attempts  100 rows   5000 duplicates     0
ack lost 0% replace-day  attempts  100 rows   5000 duplicates     0
ack lost 10% append       attempts  116 rows   5800 duplicates   800
ack lost 10% upsert       attempts  116 rows   5000 duplicates     0
ack lost 10% replace-day  attempts  116 rows   5000 duplicates     0
ack lost 30% append       attempts  142 rows   7100 duplicates  2100
ack lost 30% upsert       attempts  142 rows   5000 duplicates     0
ack lost 30% replace-day  attempts  142 rows   5000 duplicates     0
revenue at 30% lost acks: append 350753, upsert 247090, overstated by 42.0%
80% busy: mean stay 0.042s, lambda x W = 4.2, counted in flight 4.4, stay in the last 30 s 0.030s
120% busy: mean stay 30.924s, lambda x W = 3117.0, counted in flight 2580.5, stay in the last 30 s 56.803s
```

### Reading the experiment

With no failures all three writes give 5,000 rows, which is why the bug hides in testing. At 10% lost acknowledgements, 16 extra attempts produced 800 duplicate rows under append; at 30%, 42 extra attempts produced 2,100, matching the hand count. Upsert and replace-by-day stayed at exactly 5,000 in every case. The revenue total under append was 350,753 against the true 247,090, overstated by 42.0%, with every task reporting success.

Little's Law is the second lesson. At 80% load it held: arrival rate times mean stay gave 4.2 events in flight and a direct count gave 4.4. At 120% load the queue never settled. The stay time grew to 56.8 seconds by the end, and the law's estimate (3,117) was 21% above the counted 2,580 because the averages were not stable. The earlier 20-in-flight example is therefore a statement about stable systems only.

The surprise is that upsert and replace-day tied: neither duplicated a row. They differ elsewhere. Upsert cannot remove a row that was deleted at the source, while replace-by-day can. Limits: a single process, an in-memory database, a loss model that fires only after the write, and one seed.

<Infographic src="/img/dm-enrich/retry-duplicates.svg" alt="Bars show duplicate rows from append, upsert and replace-by-day writes at three lost-acknowledgement rates, and cards show the revenue overstatement and the overloaded queue." caption="Look first at the append bars: duplicates grow with the retry rate while the other two writes stay at zero." />

## Designing with it

### Choose ETL or ELT by control boundary

ETL is useful when data must be transformed or minimised before entering a destination, perhaps because raw fields are sensitive or the destination's compute is expensive. ELT is useful when a durable raw landing zone supports replay and the analytical engine can transform at scale. ELT is often called a modern cloud default; that is a tendency, not a universal rule. A team may combine them: validate and remove prohibited fields before loading, then perform business transformations inside a warehouse. Decide where raw data may legally and operationally reside.

The destination should publish a clear contract: schema, key, partition meaning, update policy and freshness. A transformation that changes `amount` from cents to pounds must version or communicate that change. If a batch and a stream both populate the same product, reconcile overlapping dates and specify which one is authoritative after late corrections. Otherwise a dashboard and a model may disagree while both pipelines report success.

### Design retries and backfills before launch

**Idempotent** means the intended output is the same after one run or several runs with the same logical input. An idempotent task may overwrite a partition, merge on a stable event key or produce an immutable version and atomically move a pointer. A task that sends an email or charges a customer is not made idempotent merely by retrying it. Keep an operation key so the downstream effect can reject duplicates.

Backfills replay old logical dates. They need source availability, transformation version, bounded concurrency and a plan for downstream invalidation. If a daily feature changes for 30 dates, cached aggregates and model-training snapshots may also need rebuilding. Test one date first, compare row counts and sampled values, then widen the range. A backfill should leave an audit trail: why it ran, which version it used, what it changed and whether the consumer was paused or notified.

### Measure flow with the right units

Batch usually favours efficient throughput over immediate visibility; streaming generally reduces latency but adds state, ordering and recovery work. Little's Law, L = λW, is a relationship among **long-run averages in a stable system**. With 100 events/s and 0.2 s, the average is 20 events in flight. If a stage can process only 80 events/s while 100 arrive, backlog grows; multiplying 100 by a fixed 0.2 s no longer describes the observed end-to-end time. Measure arrival rate, service rate, queue age, processing latency and publication lag separately. A low task runtime can coexist with a long wait in a queue.

## Walk through a failed daily run

By 02:00 the export task has landed a file and recorded its checksum. Validation starts a few minutes later and finds an unexpected currency code. The safe result depends on the contract. If the code is a source typo, quarantine its rows and ask the source owner to correct them. If it is a valid new currency, update the allowed set and the conversion rule before replay. A blanket `except` that discards the rows keeps the DAG green while undercounting revenue. The consumer should see that the day's partition is not yet approved.

An hour later the source reissues the export. The landing task should identify whether it is the same file, a corrected version or a duplicate delivery. A content hash and source version help, but neither substitutes for a business event ID. If one file contains the same payment twice, a hash of the entire file does not find the duplicate. Validate keys and define whether corrections have new IDs or versions.

The transformation task then crashes after writing half its output. A transactional replacement or write-to-new-version design keeps the previous approved partition visible until the new one commits. If the destination cannot make the whole write atomic, the pipeline needs a readiness marker that is set only after every part succeeds. A retry must clean or overwrite partial work. Otherwise a consumer can combine half of the old day with half of the corrected one.

After repair, run one logical date and compare it with source counts, accepted counts, quarantine counts and a small set of known payments. Expand the backfill with a concurrency limit so it does not starve current ingestion. Record source watermark and output version for each date. The goal is not merely to make Airflow's boxes green; it is to give the consumer a complete, traceable and correct view of the business day.

### Distinguish clock times

An event may have occurrence time, source commit time, arrival time and processing time. A payment made one minute before midnight can arrive after midnight; a late correction may arrive days later. Decide which time assigns the business-day partition and how long it remains open for late events. If a stream uses a watermark, document what happens beyond it: drop, quarantine, compensate or backfill. Train/serve feature parity depends on using the same time policy in historical and live paths.

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

## Common mistakes

1. **Appending inside a task that can be retried.** It feels simple and works in every test without failures. At 30% lost acknowledgements it left 2,100 duplicates and overstated revenue by 42.0%. Use a keyed upsert or replace the partition.
2. **Reading "success" as "correct".** All 100 days reported success under append. Compare row counts and totals with the source after a run.
3. **Making only the write idempotent.** A task that sends an email or charges a card is not repaired by a retry-safe table. Carry an operation key to the side effect too.
4. **Applying Little's Law to an overloaded system.** At 120% load the formula was 21% off, and getting worse. Check that arrival rate stays below service rate first.
5. **Backfilling everything at once.** A wide replay can starve current ingestion. Run one date, compare counts, then widen with a concurrency limit.

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

<details>
<summary><strong>Q6. (Medium)</strong> Acknowledgements are lost 30% of the time, 100 days of 50 events each. About how many duplicate rows does an append write leave, and how do you get that by hand?</summary>

Expected attempts per day are 1 / 0.7 = 1.43, so about 143 attempts for 100 days, 43 more than needed. Each extra attempt re-appends 50 events, giving about 43 × 50 = 2,150 duplicates. The run counted 142 attempts and 2,100 duplicates, which is the same estimate with random variation.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> Upsert and replace-by-day both left exactly 5,000 rows. Give a case where they differ.</summary>

Suppose the source deletes an event and the corrected day has 49 events. Replace-by-day deletes the old day and inserts 49, so the stale event disappears. Upsert overwrites 49 keys but leaves the 50th row in place, so the deleted event survives. Upsert is enough for corrections of existing events, while replace-by-partition also handles removals.

</details>

## Go deeper

- [Airflow tasks](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html) covers dependencies and retries.
- [Airflow backfill](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/backfill.html) describes historical runs and concurrency controls.
- [Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html) (opened 2026-10-09) says tasks should produce the same outcome on every re-run, advises against plain inserts on a re-run in favour of an upsert, and recommends reading a fixed partition such as `data_interval_start` instead of the latest data.
- Library versions run for the experiment: DuckDB 1.5.6, NumPy 2.5.3, Python 3.14.6.
- Built from the course lecture "dm-s4-pipelines" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can draw a task DAG and identify the consumer-visible publish point.
- [ ] I can choose ETL or ELT and batch or streaming from specific constraints.
- [ ] I can make a stage safe to retry and plan a bounded backfill.
- [ ] I can compute L = λW while stating the stable-average assumption.
- [ ] I can count the duplicate rows an append write leaves after a given number of retries.
- [ ] I can name a case where replace-by-partition is safer than upsert.
- [ ] I can explain why Little's Law gave the right answer at 80% load and a wrong one at 120%.

## Where to go next

Next is [DataOps and reliability](/docs/mlops/data/dataops-reliability), which turns delivery promises into measured objectives. For recovery of failed runs at scale, see [orchestration and recovery](/docs/mlops/data/orchestration-and-recovery).
