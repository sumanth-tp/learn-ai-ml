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

:::tip Before you start

**You should already know**

- What a group-by does ([Session 1, data representations](/docs/mlops/data/representations-for-ml)).
- What a pipeline stage and a task are ([Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery)).

**Reading time:** about 40 minutes, plus a minute to run the code.

**After this chapter you can**

- Estimate how many pieces a dataset splits into, and how unevenly a hot key will load them.
- Show with numbers why more partitions do not cure a hot key, and what salting does.
- Say which aggregates can be split into partial results safely and which cannot.

:::

## In 30 seconds

To count items quickly, you split the pile among many helpers. That works until one item is half the pile: the helper who gets it is still counting when everyone else has gone home. The job takes as long as its slowest helper, not the average one.

Splitting a pile between helpers is partitioning. Handing everything with the same label to the same helper is a shuffle. A very common label is skew.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Partition | One piece of the data handled by one task | 128 MiB of rows |
| Shuffle | Moving rows between machines so equal keys meet | Group by merchant |
| Narrow, wide | Work that stays in a piece, or needs a shuffle | Filter, or group-by |
| Skew | Some keys hold far more rows than others | One merchant has 50% |
| Straggler | The one slow task that holds up the stage | The hot key's partition |
| Salting | Add a random suffix to a hot key to split it | `hot#0` to `hot#7` |
| Map-side combine | Aggregate inside each piece before the shuffle | One subtotal per merchant per piece |
| Broadcast | Copy a small table to every worker | A merchant lookup |


## The idea in plain words

When a dataset no longer fits comfortably on one machine, a distributed engine divides it into **partitions** and schedules work on a cluster. Some operations, such as mapping each record or filtering by a field, can stay within a partition. Others need records with the same key to meet. Moving those records across machines is a **shuffle**, often one of the most expensive stages of a job. The MapReduce pattern captures this: map records, shuffle by key and reduce each group.

**Apache Spark** represents a computation as transformations and actions. Transformations build a plan; an action triggers execution. This lazy evaluation lets the engine combine operations and choose a physical plan before work runs. A useful mental model is a graph of stages separated by data movement, though the actual plan may include broadcast, caching, adaptive execution and other choices. A source code `join` does not always imply the same shuffle plan; a small side may be broadcast to avoid moving a large side.

<Infographic src="/img/dm/distributed-processing.svg" alt="Ten GiB at 128 MiB per target partition gives eighty size-based pieces; mapping stays local, grouping shuffles records, and one key holding half the data can dominate elapsed time." caption="Partition count estimates task pieces, while placement, shuffle and skew determine how they execute." />

The worked arithmetic is **10 GiB ÷ 128 MiB = 80**, because 10 GiB is 10,240 MiB. This is a size-based planning estimate, not a guarantee that a file reader or Spark job creates exactly 80 partitions. Compression, file boundaries, record format and engine settings affect the physical plan. It also does not mean 80 tasks run concurrently: available executor slots and resource limits determine concurrency. A key holding **50%** of the dataset represents **5 GiB** and can make one reduce-side partition a straggler even when the input was evenly split.

:::note Added for this site

The course material treats joins as always wide and says repartitioning or a broadcast join can mitigate skew. The sections below distinguish shuffle joins from broadcast joins, explain why repartitioning by the same heavy key may not help and show how salting or pre-aggregation can split a hot key when semantics allow it.

:::

The default lab reproduces **10 GiB / 128 MiB = 80 size-based pieces**. Adjust data size, target size and the largest key's share. At 50%, that key holds **5 GiB**. The display separates an input partition estimate from how many tasks a cluster can execute at one time.

<PartitionSkewLab />

## Worked example, step by step

A job counts 2,000,000 events by merchant. One merchant holds 50% of the rows, which is 1,000,000. Hash the merchant to P partitions.

1. **Ideal load.** With P = 16 the ideal partition has 2,000,000 / 16 = 125,000 rows. With P = 80 it is 25,000. With P = 400 it is 5,000.
2. **The hot key cannot be split.** All 1,000,000 hot rows hash to one partition, so the largest partition holds at least 1,000,000. Against the ideal that is at least 8 times at P = 16, 40 times at P = 80 and 200 times at P = 400.
3. **Salting.** Add a salt from 0 to 7 to the key. The hot key becomes 8 groups of about 125,000 rows. At P = 80 the hot pieces are 125,000 / 25,000 = 5 times the ideal each, down from 40.
4. **More salt.** With 64 salts the pieces are 1,000,000 / 64 = 15,625 rows, below the ideal of 25,000.
5. **Map-side combine.** With 80 input pieces and at most 5,000 merchants, no piece sends more than 5,000 subtotals, so at most 80 x 5,000 = 400,000 rows cross the network instead of 2,000,000.
6. **A trap for distinct counts.** Customers A, B, A, C split into partials `{A, B}` and `{A, C}` give 2 + 2 = 4, but the true distinct count is 3.

In words: more partitions shrink the ideal share but not the hot key, salting divides the hot key, and a split only works for results you can add up. The first block below prints steps 1 to 6.

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

Use consistent binary units for the partition count. The result is a planning estimate for equal-size input pieces.

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

### The worked example in code

This block reproduces the arithmetic of steps 1 to 6.

```python
n, hot = 2_000_000, 1_000_000
for partitions in (16, 80, 400):
    ideal = n / partitions
    print(partitions, "ideal", ideal, "hot key at least", hot / ideal, "times ideal")
ideal80 = n / 80
print("salt 8 piece", hot / 8, "ratio at 80", hot / 8 / ideal80)
print("salt 64 piece", hot / 64, "below ideal", hot / 64 < ideal80)
print("combine bound", 80 * 5000)
partials = [{"A", "B"}, {"A", "C"}]
print("sum of partial distincts", sum(len(p) for p in partials), "true", len(set().union(*partials)))
```

**Reading the output.** It prints ratios of 8.0, 40.0 and 200.0, then a salt-8 piece of 125,000 with ratio 5.0, a salt-64 piece of 15,625 that is below the ideal, a combine bound of 400,000, and 4 against a true count of 3.

### An experiment on a hot key

Does salting fix the imbalance, and what does it break? The block below generates 2,000,000 events whose merchants follow a skewed distribution with merchant 0 holding exactly half. It uses DuckDB's `hash` to assign each row to one of P partitions, measures the largest partition against the ideal, and tries three keys: the merchant, the merchant with 8 salts and the merchant with 64 salts. It then counts the rows that survive a map-side combine over 80 input pieces, and checks which aggregates survive being split into 8 salted partials: sum, distinct count and median.

Versions used: Python 3.14.6, DuckDB 1.5.6, pandas 2.3.3, NumPy 2.5.3. The data is synthetic, and "partition" here is a hash bucket on one machine, so the imbalance numbers measure rows per bucket, not wall time on a cluster. It runs in about a second.

```python
import duckdb
import numpy as np
import pandas as pd

rng = np.random.default_rng(13)
n, merchants, parts_in = 2_000_000, 5000, 80
weights = 1 / np.arange(1, merchants + 1) ** 1.2
weights /= weights.sum()
tail = rng.choice(np.arange(1, merchants + 1), size=n, p=weights)
merchant = np.where(rng.random(n) < 0.5, 0, tail)
events = pd.DataFrame({
    "merchant": merchant,
    "customer": rng.integers(0, 200_000, n),
    "amount": rng.lognormal(3.0, 1.0, n).round(2),
    "input_part": rng.integers(0, parts_in, n),
    "salt": rng.integers(0, 64, n),
})
con = duckdb.connect()
con.register("events", events)

def imbalance(key_sql, partitions):
    sizes = con.sql(f"select count(*) as n from events group by hash({key_sql}) % {partitions}").df().n
    return sizes.max() / (n / partitions), sizes.max() / n

print("hot merchant share", round((events.merchant == 0).mean(), 3))
print(f"{'plan':26}{'partitions':>11}{'max / ideal':>12}{'max share':>11}")
plans = [("hash by merchant", "merchant")] + [(f"merchant + salt({k})", f"merchant * 64 + salt % {k}") for k in (8, 64)]
for name, key in plans:
    for partitions in (16, 80, 400):
        ratio, share = imbalance(key, partitions)
        print(f"{name:26}{partitions:11d}{ratio:12.2f}{share:11.3f}")

combined = con.sql("select count(*) from (select input_part, merchant from events group by all)").fetchone()[0]
print("rows shuffled raw", n, "after map-side combine", combined, f"({combined / n:.3f} of raw)")

exact = con.sql("select merchant, sum(amount) s, count(distinct customer) d, median(amount) m from events group by merchant").df().set_index("merchant")
partial = con.sql("select merchant, salt % 8 as salt, sum(amount) s, count(distinct customer) d, median(amount) m from events group by merchant, salt % 8").df()
combined_sum = partial.groupby("merchant").s.sum()
combined_distinct = partial.groupby("merchant").d.sum()
combined_median = partial.groupby("merchant").m.mean()
hot = 0
print("hot merchant sum: exact", round(exact.s[hot], 2), "salted", round(combined_sum[hot], 2))
print("hot merchant distinct customers: exact", int(exact.d[hot]), "sum of partials", int(combined_distinct[hot]))
print("hot merchant median: exact", round(exact.m[hot], 2), "mean of partial medians", round(combined_median[hot], 2))
sizes = events.groupby("merchant").size()
big = sizes[sizes >= 200].index
error = (combined_median[big] / exact.m[big] - 1).abs()
print("merchants with 200+ events", len(big), "median of partial medians off by up to", f"{100 * error.max():.2f}%")
print("largest sum difference over all merchants", round(combined_sum.sub(exact.s).abs().max(), 6))
```

The output of the run:

```text
hot merchant share 0.5
plan                       partitions max / ideal  max share
hash by merchant                   16        9.94      0.621
hash by merchant                   80       40.14      0.502
hash by merchant                  400      200.09      0.500
merchant + salt(8)                 16        2.34      0.146
merchant + salt(8)                 80        5.48      0.069
merchant + salt(8)                400       25.97      0.065
merchant + salt(64)                16        1.51      0.095
merchant + salt(64)                80        2.64      0.033
merchant + salt(64)               400        6.75      0.017
rows shuffled raw 2000000 after map-side combine 126159 (0.063 of raw)
hot merchant sum: exact 33134978.24 salted 33134978.24
hot merchant distinct customers: exact 198666 sum of partials 743776
hot merchant median: exact 20.15 mean of partial medians 20.15
merchants with 200+ events 334 median of partial medians off by up to 14.19%
largest sum difference over all merchants 0.0
```

**Reading the output.** `max / ideal` is the largest bucket divided by rows per bucket if the load were perfectly even; 1.00 is perfect. `max share` is the largest bucket as a share of all rows. The distinct-customer line compares the exact count with the sum of the eight salted partial counts.

**Line by line.**

- `hash({key_sql}) % {partitions}` imitates the hash partitioner of a shuffle: the same key always gives the same bucket.
- `merchant * 64 + salt % {k}` builds a combined key with `k` salt values, so the hot merchant becomes `k` groups.
- `group by all` over `(input_part, merchant)` counts what a map-side combine would send: one row per merchant per input piece.

### What the numbers say

Hashing by merchant left the largest partition at 40.14 times the ideal with 80 partitions, which matches the hand bound of 40. Going from 80 to 400 partitions made the ratio worse, 200.09, because the ideal fell while the hot key stayed whole. The largest partition's share stayed at 0.500 throughout. More partitions did nothing for the straggler.

Salting helped, and its limit is the salt count. Eight salts brought the ratio at 80 partitions to 5.48, and the largest share could not fall below about 1/16 of the data (0.065 at 400 partitions). With 64 salts the ratio was 2.64 at 80 partitions. Map-side combine cut the shuffle to 126,159 rows, 6.3% of the raw 2,000,000, which is below the 400,000 bound because small merchants appear in few input pieces.

The caution is correctness. Sums were identical. The distinct-customer count for the hot merchant was 743,776 after adding partials, against a true 198,666, an overcount of almost four times. The surprise is the median: for the hot merchant the mean of the partial medians matched at 20.15, which would tempt a team to trust it, but over the 334 merchants with at least 200 events it was off by up to 14.19%.

Limits: one synthetic distribution, one seed, rows per bucket as a proxy for time, and no spill or network cost. Check task durations and shuffle bytes in your engine.

<Infographic src="/img/dm-enrich/dm2-skew-salting.svg" alt="Largest partition against ideal for hashing by merchant and by salted merchant at 16, 80 and 400 partitions, plus the overcount of a summed distinct count." caption="Look first at the red bars: adding partitions makes the hot key's ratio worse, and only salting brings it down." />

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

Many tiny input files cause a different problem. Even when their total size is 10 GiB, opening thousands of files adds metadata requests and task overhead. Compacting files into appropriately sized columnar outputs can improve scan efficiency, but compaction itself costs time and should preserve snapshot semantics. Too few huge files reduce parallelism. The correct file size depends on the storage system, reader and workload; the 128 MiB figure here is an illustrative target, not a universal default.

Finally, distinguish storage partitioning from execution partitions. A table may be partitioned by date in object storage, so a query on one day reads fewer files. The engine then splits those files into scan tasks and may repartition records again for a join or group-by. Calling all three things "a partition" can hide the actual bottleneck. In design reviews, name the storage layout, input split and shuffle partition separately and tie each to a measured stage.

## Where this stands in 2026

:::info Industry view

- Spark documentation defines transformations, actions and shuffle behaviour, and its SQL guide documents broadcast and adaptive planning.
- Real task concurrency depends on executor slots and resource limits; a partition count alone does not state elapsed time.
- Skew diagnosis starts with actual per-task bytes and key frequencies, then chooses a correctness-preserving mitigation.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Adding partitions to fix a slow stage | More pieces means more parallelism | If one key dominates, the largest partition does not shrink. The ratio went from 40.14 to 200.09 when partitions rose from 80 to 400 |
| Salting with a small number of salts | Any split helps | The salt count caps the gain. 8 salts left a ratio of 5.48, 64 salts reached 2.64 |
| Summing partial distinct counts | Sums are safe for totals | Distinct counts and medians need other algorithms. The sum of partials overcounted the hot merchant by almost four times |
| Trusting an average over tasks | The mean task is fast | Look at the slowest task and bytes per task. The stage ends with its straggler |
| Treating a hot key as a tuning problem | The job is slow, so tune it | Check whether the hot key is a null or a malformed ID first. That is a data quality failure |

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

<details>
<summary><strong>Q6.</strong> (Medium) 4,000,000 rows, 40 reduce partitions, one key holds 30% of the rows. What is the smallest possible ratio of the largest partition to the ideal?</summary>

The ideal is 4,000,000 / 40 = 100,000 rows. The hot key holds 1,200,000 rows, all in one partition, so the largest is at least 1,200,000 and the ratio is at least 12. More partitions would raise the ratio, since the ideal falls while the hot key stays whole.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) You salt a hot key into 8 groups and sum partial results. Which of sum, count, maximum, distinct count and median are safe to combine from the partials, and how?</summary>

Sum and count: add the partials. Maximum: take the maximum of the partial maxima. Distinct count and median are not safe. Summing partial distinct counts overcounted the hot merchant almost four times in the experiment, and averaging partial medians was off by up to 14.19% on merchants with 200 or more events. Use a mergeable sketch or an exact set union for distinct counts, and an approximate quantile sketch or an exact sort for the median.

</details>

## Go deeper

- [Spark RDD programming guide](https://spark.apache.org/docs/latest/rdd-programming-guide) explains transformations, actions and shuffles.
- [Spark SQL performance tuning](https://spark.apache.org/docs/latest/sql-performance-tuning) documents broadcast joins and adaptive execution.
- Spark SQL performance tuning (the link above), opened 2026-10-09 for Spark 4.2.0: the adaptive skew-join setting is described for sort-merge joins, `autoBroadcastJoinThreshold` defaults to 10 MB, and shuffle partitions default to 200 and can be coalesced.
- [DuckDB utility functions](https://duckdb.org/docs/current/sql/functions/utility), opened 2026-10-09: `hash(value)` returns an unsigned 64-bit integer, and the hash function may change across versions.
- Built from the course lecture "dm-l13-distributed-processing" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can calculate 10 GiB / 128 MiB = 80 and state the limits of that estimate.
- [ ] I can explain narrow work, shuffle work and lazy evaluation from an executed plan.
- [ ] I can distinguish input pieces, task concurrency and a hot-key straggler.
- [ ] I can choose a skew response and test that it preserves the result.
- [ ] I can work out the smallest possible largest-partition ratio from a hot key's share and the partition count.
- [ ] I can explain why 400 partitions made the straggler ratio worse than 80 in the experiment.
- [ ] I can name which aggregates survive a salted split and which need a different method.

## Where to go next

Next: [Lecture 14, knowledge base data pipelines](/docs/mlops/data/knowledge-base-data-pipelines), which prepares text for retrieval. Related: [Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery), which schedules the stages this chapter splits.
