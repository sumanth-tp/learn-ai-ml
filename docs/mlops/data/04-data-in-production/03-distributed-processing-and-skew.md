---
id: dm-distributed-processing-and-skew
title: "Data Management · Lecture 13 — Distributed Processing and Skew"
sidebar_label: "13 · Distributed processing"
sidebar_position: 3
slug: /mlops/data/distributed-processing-skew
description: "Reason about partitions, lazy transformations, shuffles and skew in a distributed data job."
tags: [data-management, spark, distributed-processing, data-skew]
---

import Infographic from '@site/src/components/Infographic';
import PartitionSkewLab from '@site/src/components/viz/PartitionSkewLab';

**In one line.** Splitting data enables parallel work, but movement and uneven keys often decide the job's elapsed time.

## The idea in plain words

When a dataset no longer fits comfortably on one machine, a distributed engine divides it into **partitions** and schedules work on a cluster. Some operations, such as mapping each record or filtering by a field, can stay within a partition. Others need records with the same key to meet. Moving those records across machines is a **shuffle**, often one of the most expensive stages of a job. The lecture's MapReduce pattern captures this: map records, shuffle by key and reduce each group.

**Apache Spark** represents a computation as transformations and actions. Transformations build a plan; an action triggers execution. This lazy evaluation lets the engine combine operations and choose a physical plan before work runs. A useful mental model is a graph of stages separated by data movement, though the actual plan may include broadcast, caching, adaptive execution and other choices. A source code `join` does not always imply the same shuffle plan; a small side may be broadcast to avoid moving a large side.

<Infographic src="/img/dm/distributed-processing.svg" alt="Ten GiB at 128 MiB per target partition gives eighty size-based pieces; mapping stays local, grouping shuffles records, and one key holding half the data can dominate elapsed time." caption="Partition count estimates task pieces, while placement, shuffle and skew determine how they execute." />

The worked arithmetic is **10 GiB ÷ 128 MiB = 80**, because 10 GiB is 10,240 MiB. This is a size-based planning estimate, not a guarantee that a file reader or Spark job creates exactly 80 partitions. Compression, file boundaries, record format and engine settings affect the physical plan. It also does not mean 80 tasks run concurrently: available executor slots and resource limits determine concurrency. A key holding **50%** of the dataset represents **5 GiB** and can make one reduce-side partition a straggler even when the input was evenly split.

:::note Beyond the lecture

The source treats joins as always wide and says repartitioning or a broadcast join can mitigate skew. The sections below distinguish shuffle joins from broadcast joins, explain why repartitioning by the same heavy key may not help and show how salting or pre-aggregation can split a hot key when semantics allow it.

:::

The default lab reproduces **10 GiB / 128 MiB = 80 size-based pieces**. Adjust data size, target size and the largest key's share. At 50%, that key holds **5 GiB**. The display separates an input partition estimate from how many tasks a cluster can execute at one time.

<PartitionSkewLab />

## How it works

### MapReduce & Spark

Map transforms records in parallel; shuffle groups by key; reduce aggregates. Spark generalises this in-memory with transformations (lazy) and actions.

### Parallelism & shuffle

- **Narrow vs wide**; map/filter (no move) vs groupBy/join (shuffle).
- **Lazy eval**; Build the DAG, optimise, then run on an action.

:::tip

**Worked.** 10 GB / 128 MB = 10240/128 = 80 partitions (≈80 parallel tasks).

:::


## A real system that works this way

**Apache Spark** is a concrete distributed engine. Its RDD guide explains transformations, actions and shuffles; its SQL performance documentation covers partitioning, broadcast joins and adaptive query execution. A data team might use Spark to build daily features from many event files. A filter and projection can be pushed toward the source; a group-by account ID may require events for the same account to be exchanged across the cluster. The SQL physical plan and stage metrics show whether that exchange occurred.

Suppose the team counts events per merchant. Most merchants have hundreds of events, but one marketplace merchant has half of the 10 GiB input. An even number of source files does not yield even reduce tasks: the hot merchant's records converge on one grouping key. More workers may finish their assigned partitions and sit idle while one task processes the hot key. Spark's adaptive query execution can change some plans at runtime, but the team should still inspect partition sizes, task duration and shuffle read bytes rather than assuming the engine eliminates every skew pattern.

A small merchant lookup table may be broadcast to workers so each event can be enriched locally. That avoids a large-side shuffle for the join when the lookup fits resource limits. A large-to-large join typically requires exchange or compatible existing partitioning. Choosing between these plans depends on table size, key distribution, memory and downstream grouping, not just the `join` method name.

## Code you can run

Use consistent binary units for the lecture's count. The result is a planning estimate for equal-size input pieces.

```python
import math

gib_to_mib = 1024
input_mib = 10 * gib_to_mib
target_mib = 128
pieces = math.ceil(input_mib / target_mib)
hot_key_mib = input_mib * 0.50
print(pieces, hot_key_mib / gib_to_mib)
assert pieces == 80
assert hot_key_mib / gib_to_mib == 5
```

A local map, group and reduce example makes the key movement visible. It also shows that a hot key remains large after simply hashing by the same key.

```python
from collections import defaultdict

records = [("hot", 1)] * 6 + [("a", 1), ("b", 1)]
workers = [records[:4], records[4:]]
mapped = [[(key, value * 2) for key, value in part] for part in workers]
grouped = defaultdict(list)
for part in mapped:
    for key, value in part:
        grouped[key].append(value)
reduced = {key: sum(values) for key, values in grouped.items()}
print(reduced, {key: len(values) for key, values in grouped.items()})
assert reduced == {"hot": 12, "a": 2, "b": 2}
assert len(grouped["hot"]) == 6
```

The Python example runs on one process. In a real cluster, grouping sends matching keys to reduce-side tasks over the network and often writes intermediate data to disk. A broadcast lookup changes the join plan but does not remove a later group-by shuffle.

## Designing with it

### Choose partitions for the operation

Partition size balances overhead against useful parallelism. Tiny partitions create many scheduling tasks and metadata operations. Huge partitions limit parallelism and can exceed worker memory. Begin with data size and a target piece size, then inspect physical input splits and task metrics. Compressed files may be unsplittable, while columnar formats can expose row groups. After a shuffle, the relevant partition count may differ from the source partition count. Measure bytes and records per task, not only a configured number.

Available slots limit concurrency. If 80 partitions exist and the cluster can run 10 tasks at once, the stage may execute in waves. Adding partitions to 800 does not make it ten times faster; it may make scheduling and shuffle overhead worse. Adding workers helps only when work can be distributed and the bottleneck is compute or memory rather than one key, a slow source or network exchange. Define the target completion time and cost before tuning.

### Understand the shuffle boundary

A map or filter is usually a narrow transformation because each output partition can use one input partition. Grouping by key, repartitioning by key and many joins require moving data across partitions. Shuffle costs include network traffic, sort or hash work, spill to disk and fault recovery. A join with a small side can broadcast that side to workers; a join of two large tables may shuffle both sides unless their existing distribution is compatible. Use the physical plan to confirm what the engine chose. Filtering and selecting only needed columns before a shuffle can reduce bytes moved.

### Treat skew as a semantic issue

If one key owns half the records, hashing all rows by that key still sends them to one partition. Raising the number of reduce partitions does not split that key. For an associative aggregate such as count or sum, pre-aggregate locally, salt the hot key into several partial groups, then combine the partial results. For a join, broadcasting a sufficiently small side or using a skew-aware join strategy may help. Salting has correctness costs: both sides of a join need compatible replication or routing, and non-associative computations need special care. First confirm the hot key is real and not a null or malformed ID created by upstream data quality failure.

### Read the executed plan

Lazy evaluation lets the engine optimise a whole chain before an action. It also means a transformation definition may perform no work until a count, write or collect occurs. A seemingly harmless debugging action can launch an expensive full scan. Inspect the plan and stage metrics for filters, exchange nodes, broadcast, skewed task duration, spill, shuffle read and output size. Cache an intermediate result only when it will be reused enough to justify memory and recomputation trade-offs. A narrow-looking expression may still depend on an upstream wide stage.

## Diagnose a slow merchant aggregation

A feature job reads 10 GiB of events and aims for 128 MiB input pieces. The calculation suggests 80 pieces. The actual scan reports 92 tasks because file layout and reader rules split some files differently. That is not a contradiction; the arithmetic is a planning estimate. Each scan task filters irrelevant event types and projects merchant ID plus amount, reducing the bytes that later stages need. An action to write the final feature table starts execution of the planned graph.

The group-by merchant ID creates an exchange. Most reduce tasks finish in seconds, but one runs for several minutes and spills to disk. Its shuffle read is much larger than its peers. A key-frequency sample shows one marketplace merchant owns half of the eligible events. The team calculates that half of 10 GiB is 5 GiB before filtering; it then measures the actual post-filter bytes for that key. If the hot key is `null`, the investigation turns to source quality rather than compute tuning. If it is a real merchant, the aggregation strategy needs work.

For a sum, map-side partial aggregation can shrink many event rows into one subtotal per merchant per input partition. If the hot key still dominates, split it into several salted partial keys using a deterministic salt, aggregate each, then sum the partial results. Check that the arithmetic type and overflow behaviour match the original. For distinct counts or medians, combining partials requires a different algorithm or exact set handling; blindly summing partial distinct counts would overcount. A correctness test compares the salted result with the original on a manageable sample.

The same job joins the aggregate to a small merchant category table. A broadcast join may copy that table to each worker and avoid shuffling the large aggregate. But if the lookup grows past memory limits, broadcast can cause executor failures. The physical plan, table statistics and observed memory tell the team whether the broadcast choice is appropriate. A later group-by region can still shuffle, even if the join itself did not. Optimise the whole path, not one operator name.

After a change, compare wall time, shuffle bytes, spill, task duration distribution and output checksums or reconciliation totals. A faster run that drops rare merchants is a failed optimisation. Keep the input snapshot and code version fixed while measuring the effect. Data volume may change next month, so set an alert on skew ratio and stage time rather than assuming one tuning parameter remains ideal forever.

### Plan capacity from waves, not just partitions

Suppose the stage has 80 partitions and only eight task slots. If tasks were equal and each took one minute, the stage would need roughly ten waves plus overhead. In reality, task durations differ and network contention increases under load. If one hot-key task takes fifteen minutes, it sets a lower bound on stage completion regardless of how quickly the other 79 finish. More slots can shorten the equal-work portion but do not eliminate that straggler. This is why the longest task and the distribution of bytes per partition are often more useful than an average.

Many tiny input files cause a different problem. Even when their total size is 10 GiB, opening thousands of files adds metadata requests and task overhead. Compacting files into appropriately sized columnar outputs can improve scan efficiency, but compaction itself costs time and should preserve snapshot semantics. Too few huge files reduce parallelism. The correct file size depends on the storage system, reader and workload; the 128 MiB figure in the lecture is an illustrative target, not a universal default.

Finally, distinguish storage partitioning from execution partitions. A table may be partitioned by date in object storage, so a query on one day reads fewer files. The engine then splits those files into scan tasks and may repartition records again for a join or group-by. Calling all three things "a partition" can hide the actual bottleneck. In design reviews, name the storage layout, input split and shuffle partition separately and tie each to a measured stage.

## Where this stands in 2026

:::info Industry view

- Spark documentation defines transformations, actions and shuffle behaviour, and its SQL guide documents broadcast and adaptive planning.
- Real task concurrency depends on executor slots and resource limits; a partition count alone does not state elapsed time.
- Skew diagnosis starts with actual per-task bytes and key frequencies, then chooses a correctness-preserving mitigation.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Describe the MapReduce pattern.</summary>

Map transforms each record in parallel, shuffle groups records by key across the network, reduce aggregates per key.<br /><em>Lecture 13 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast narrow and wide transformations in Spark.</summary>

Map and filter are usually narrow; group-by commonly shuffles. A join may shuffle or broadcast a small side, depending on the physical plan.<br /><em>Lecture 13 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A 10 GB dataset with 128 MB partitions splits into how many partitions?</summary>

With binary units, 10 GiB = 10,240 MiB, and 10,240/128 = 80 size-based pieces. Physical partitions may differ, and available slots limit concurrent tasks.<br /><em>Lecture 13 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is lazy evaluation and why is it useful?</summary>

Transformations build a plan (DAG) and only run on an action; this lets Spark optimise the whole computation before executing.<br /><em>Lecture 13 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is data skew and why is it a problem?</summary>

One key holds far more records than others, creating a straggler. Inspect the plan and key frequencies; use pre-aggregation, salting or a suitable broadcast join when the operation permits. Repartitioning by the same hot key alone does not split it.<br /><em>Lecture 13 · conceptual</em>

</details>

## Go deeper

- [Spark RDD programming guide](https://spark.apache.org/docs/latest/rdd-programming-guide) explains transformations, actions and shuffles.
- [Spark SQL performance tuning](https://spark.apache.org/docs/latest/sql-performance-tuning) documents broadcast joins and adaptive execution.
- Built from the course lecture "dm-l13-distributed-processing" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can calculate 10 GiB / 128 MiB = 80 and state the limits of that estimate.
- [ ] I can explain narrow work, shuffle work and lazy evaluation from an executed plan.
- [ ] I can distinguish input pieces, task concurrency and a hot-key straggler.
- [ ] I can choose a skew response and test that it preserves the result.
