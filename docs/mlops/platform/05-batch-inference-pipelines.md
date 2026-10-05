---
id: plat-batch-inference
title: "Batch Inference Pipelines"
sidebar_label: "Batch inference"
sidebar_position: 5
slug: /mlops/platform/batch-inference-pipelines
description: "Score millions of rows without an endpoint: when batch beats online, how to make a job idempotent and restartable, how to partition and backfill, and how Spark, Ray and Beam compare, with DuckDB and pandas code that runs."
tags: [batch-inference, idempotency, backfill, partitioning, duckdb, spark, ray, beam, ml-platform]
---

import Infographic from '@site/src/components/Infographic';
import BatchWindowLab from '@site/src/components/viz/BatchWindowLab';

**In one line.** A batch inference job scores a whole partition of data on a schedule, and what makes it production-grade is not the model call but that you can run it twice, kill it halfway, or backfill last month and always end up with the same correct table.

:::note Not from a lecture

This chapter is written for this site from the engine and orchestrator documentation listed under Further reading; it is not built from a course lecture.

:::

## The idea in plain words

An endpoint answers one request while a person waits. Many predictions have no one waiting: tonight's churn scores for every customer, next week's demand for every shop, embeddings for a new catalogue of ten million products. The answers will be read tomorrow, by a dashboard or a downstream job. For this work the endpoint is the wrong tool. It idles between requests, it is sized for the peak, and every row pays a network round trip.

**Batch inference** reads a bounded chunk of data, scores it in large vectorised batches, writes the results to a table, and exits. You pay only for the compute you use and you pick the hardware for throughput, not latency. The previous chapter's cost model made the point in numbers: at 30 million requests a month, batch cost 1,667 cost units against 2,920 for always-on and 3,333 for pay-per-use, with the same compute per request.

The price is operational. A batch job runs unattended, on a schedule, over data that changes, and it fails in the middle. So the real design questions are about failure:

| Failure | What goes wrong without a design for it |
| --- | --- |
| The job is re-run after a crash | Duplicate rows, or half-written files |
| A day of source data arrives late | The table silently misses it |
| You must backfill three months | Hundreds of runs, some fail, nobody knows which |
| One partition is huge | Adding workers does not make the job finish sooner |
| The input has very uneven lengths | Most of the GPU time is spent on padding |

<Infographic src="/img/plat/plat-batch-inference-idempotent.svg" alt="Append versus overwrite, a backfill ledger with a failed day, late data rerunning one partition, and a crash during a write" caption="A job you can run twice. Every number is printed by block 2 below." />

<Infographic src="/img/plat/plat-batch-inference-window.svg" alt="Batch size against model calls, padding waste with and without sorting by length, and the makespan table for a partition holding thirty percent of the rows" caption="Making the batch fit its window. The numbers are printed by blocks 1 and 3 below." />

## How it works

### Batch or online?

Ask who is waiting and how stale an answer may be. If a human or another service blocks on the response, you need an endpoint. If the consumer reads results later, you want a batch job, and often a **precompute-then-lookup** design: score everything nightly, write to a table or key-value store, and let the online path just read. The cloud services draw the same line. SageMaker's batch transform runs inference "when you don't need a persistent endpoint"; Azure's documentation recommends batch endpoints for expensive models, large data spread across many files, no low-latency requirement, and inputs in storage; Bedrock's batch inference takes JSONL prompts from S3 and returns outputs to S3 ([batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html), [Azure endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2), [Bedrock batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)). The previous chapter, [cloud ML platforms](/docs/mlops/platform/cloud-ml-platforms), compares them.

### Idempotency: the output is a function of the input

A job is **idempotent** if running it twice produces the same result as running it once. Three habits give you that.

1. **Key everything by the input partition.** Airflow's best-practice guide says to treat a task like a database transaction that produces identical results on every re-run, and to use the data interval start as the partition key for reading and writing instead of the current time: calling `now()` inside a task "leads to different outcomes on each run". The same page warns that an `INSERT` on re-run can duplicate rows and recommends an `UPSERT` ([Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html), version 3.3.2 when read).
2. **Overwrite the partition, never append to it.** The output path is derived from the partition, so a retry replaces the previous attempt. Block 2 shows the cost of getting this wrong: appending twice gives 28,000 rows for 14,000 distinct ids.
3. **Publish atomically.** Spark's documentation notes that its save modes use no locking and are not atomic, and that an overwrite deletes the existing data before writing ([Spark load and save](https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html), Spark 4.2.0). A crash in between leaves you with nothing. The defence is the pattern from block 2: write to a temporary name in the same directory, then rename into place, which a local file system performs as one step. Object stores differ in what they guarantee, so check yours.

### Partitioning, backfills and late data

Partition the data the way you will rerun it, usually by date. Spark discovers partitioning from directory names such as `event_date=2026-09-01/` automatically ([Spark Parquet](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html)) and writes them with `partitionBy`, which its load and save page says has limited applicability to high-cardinality columns. DuckDB writes the same layout with `COPY ... PARTITION_BY`, and its documentation recommends keeping at least 100 MB of data per partition to avoid expensive file operations ([DuckDB partitioned writes](https://duckdb.org/docs/current/data/partitioning/partitioned_writes.html)).

A **backfill** is the same job run over a range of past partitions. It works if each partition is independent and idempotent, and if you keep a **ledger**: a small table recording for each partition the status and a fingerprint of the input (row count and a checksum). A runner then skips partitions whose fingerprint is unchanged, retries the failed ones, and reruns exactly those whose input changed, which is how late data is handled. Block 2 does this with a seven-day backfill in which a worker dies: the first run completes six days, the second runs only the failed day and skips six, the third does nothing.

### Throughput: batching, ordering and skew

Three levers decide whether the job fits its window.

- **Batch size.** Scoring 100,000 rows takes 100 calls at batch size 1,000 or 1 call at 100,000, with identical results (block 1). Larger batches amortise per-call overhead and use vector hardware well, up to memory. Ray Data's documentation says the same: bigger batches run faster because inference is vectorised, and for GPU inference you set an explicit batch size to use the device without exceeding its memory ([Ray Data batch inference](https://docs.ray.io/en/latest/data/batch_inference.html), docs for Ray 2.59 and later).
- **Order.** For variable-length inputs such as text, a batch costs as much as its longest item. In arrival order, block 1 wastes three quarters of the tokens as padding; sorted by length, 99.5% of the compute is useful. Carry a row id so you can restore the original order afterwards.
- **Skew.** The job cannot finish before its largest indivisible piece. With one partition holding 30% of 50 million rows, adding workers stops helping at 2.08 hours (block 3). Cut large partitions into bounded chunks so work spreads.

### Spark, Ray and Beam

The three engines solve the same job with different centres of gravity.

| Engine | Model of work | What its documentation says that matters here |
| --- | --- | --- |
| **Spark** | Partitioned DataFrames over a cluster | Partition discovery and `partitionBy`; save modes are not atomic; partition-specific overwrite uses `insertInto` with dynamic partition overwrite mode, controlled by `spark.sql.sources.partitionOverwriteMode` |
| **Ray Data** | Streaming datasets with actor pools | `map_batches` with a class loads the model once per actor; `num_gpus`, `batch_size` and an actor pool size are the knobs; execution is streaming, so preprocessing of one batch overlaps inference on the previous |
| **Apache Beam** | One pipeline for batch and streaming | `RunInference` is a transform for model inference; it batches dynamically, shares models across threads and processes, supports a dead-letter queue and works in batch and streaming pipelines ([Beam ML](https://beam.apache.org/documentation/ml/about-ml/)) |

Choose by the work around the model. Lots of tabular joins and aggregation before scoring points to Spark. A GPU model with expensive preprocessing points to Ray Data's overlapped stages. A pipeline that must also run on a stream, with the same code, points to Beam. A single machine with DuckDB or pandas and a partitioned Parquet layout is enough more often than teams expect, as block 2 shows.

On Kubernetes, the unit is a Job. An Indexed Job gives every pod a stable index, so "pod 3 scores partition 3", and `backoffLimit` bounds retries; see [Kubernetes for ML](/docs/mlops/platform/kubernetes-for-ml). For orchestration of the daily run, retries and recovery, see [orchestration and recovery](/docs/mlops/data/orchestration-and-recovery), and for skew in distributed data processing generally, [distributed processing and skew](/docs/mlops/data/distributed-processing-skew).

## A real system that works this way

**SageMaker batch transform** is built around exactly these ideas. Its documentation says the job partitions the S3 objects of the input by key and maps objects to instances, so with one input file and many instances only one instance works and the rest sit idle. It splits files into mini-batches when `SplitType` is `Line`, limits `MaxPayloadInMB` to 100, writes one output per input with an `.out` suffix, and lists predictions in the same order as the input records ([batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)). It is the skew problem and the ordering problem with product names.

**Apache Airflow** is the orchestration pattern the ledger imitates. Its guide tells you to make tasks idempotent, to key them by data interval and to avoid computing from the current time, which is what lets its scheduler retry and backfill safely.

## Code you can run

Four blocks, executed with Python 3.14, NumPy 2.5.3, pandas 3.0.6, DuckDB 1.5.6, PyArrow 25.0.1 and scikit-learn. Everything is seeded; there are no timings, so the printed counts are the ones quoted in the text.

### 1. Batch size and padding

The first part scores 100,000 rows with a fitted logistic regression at three batch sizes and checks that the scores are identical. The second counts padded tokens for 20,000 variable-length sequences in batches of 32, in arrival order and sorted by length.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(21)
n_train, n_score, features = 4000, 100_000, 12
w = rng.normal(0, 1, features)
X = rng.normal(0, 1, (n_train, features))
y = (X @ w + rng.normal(0, 1, n_train) > 0).astype(int)
model = LogisticRegression(max_iter=500).fit(X, y)
rows = rng.normal(0, 1, (n_score, features))

def score_in_batches(rows, batch_size):
    out, calls = [], 0
    for start in range(0, len(rows), batch_size):
        out.append(model.predict_proba(rows[start:start + batch_size])[:, 1])
        calls += 1
    return np.concatenate(out), calls

one_shot, _ = score_in_batches(rows, len(rows))
print(f"scoring {n_score:,} rows")
for batch_size in (1_000, 10_000, 100_000):
    scores, calls = score_in_batches(rows, batch_size)
    same = np.allclose(scores, one_shot, rtol=0, atol=1e-12)
    print(f"  batch size {batch_size:>7,}: {calls:>4} model calls, identical scores: {same}")
print(f"  row by row would need {n_score:,} calls; scores above 0.5: {(one_shot > 0.5).sum():,}")

lengths = np.clip(rng.lognormal(mean=4.0, sigma=0.8, size=20_000).astype(int), 8, 512)
batch = 32

def padded_tokens(order):
    total = 0
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        total += chunk.max() * len(chunk)
    return total

real = int(lengths.sum())
arrival = padded_tokens(lengths)
bucketed = padded_tokens(np.sort(lengths))
print(f"\nvariable-length inputs: {len(lengths):,} sequences, batches of {batch}, real tokens {real:,}")
print(f"  arrival order : {arrival:,} padded tokens, {real / arrival:.1%} useful")
print(f"  sorted by length: {bucketed:,} padded tokens, {real / bucketed:.1%} useful")
print(f"  compute saved by sorting: {1 - bucketed / arrival:.1%}")
```

Batching changes the number of calls (100, 10, 1, against 100,000 row by row) but never the answers. The padding result is the bigger surprise: in arrival order only 25.3% of the processed tokens are real; sorted by length, 99.5% are, and 74.6% of the compute disappears. Sorting is free; the cost is keeping row ids to restore order.

### 2. An idempotent, restartable daily job

DuckDB scores seven daily partitions of 2,000 rows with a fixed linear score. The runner writes each partition to a temporary file and renames it, records an input fingerprint in a ledger, and skips partitions it has already done.

```python
import os
import tempfile
import uuid
import duckdb
import numpy as np
import pandas as pd

DAYS = [f"2026-09-0{d}" for d in range(1, 8)]
ROWS_PER_DAY = 2000
SCORE = "1 / (1 + exp(-(f0 * 0.8 - f1 * 0.5 + f2 * 0.3 - 0.1)))"

def make_source(root):
    rng = np.random.default_rng(5)
    for i, day in enumerate(DAYS):
        folder = os.path.join(root, "src", f"event_date={day}")
        os.makedirs(folder)
        frame = pd.DataFrame(rng.normal(0, 1, (ROWS_PER_DAY, 3)), columns=["f0", "f1", "f2"])
        frame.insert(0, "id", np.arange(ROWS_PER_DAY) + i * 100_000)
        frame.to_parquet(os.path.join(folder, "part.parquet"))

def run_day(con, root, day, mode, fail_on=None):
    source = os.path.join(root, "src", f"event_date={day}", "part.parquet")
    folder = os.path.join(root, "preds", f"event_date={day}")
    os.makedirs(folder, exist_ok=True)
    if fail_on == day:
        raise RuntimeError(f"worker lost while scoring {day}")
    query = f"SELECT id, {SCORE} AS score FROM read_parquet('{source}')"
    if mode == "append":
        con.execute(f"COPY ({query}) TO '{folder}/{uuid.uuid4().hex}.parquet' (FORMAT parquet)")
    else:
        tmp = os.path.join(folder, "part.parquet.tmp")
        con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT parquet)")
        os.replace(tmp, os.path.join(folder, "part.parquet"))

def stats(con, root):
    glob = os.path.join(root, "preds", "*", "*.parquet")
    n, distinct, total = con.execute(f"SELECT count(*), count(DISTINCT id), round(sum(score), 6) FROM read_parquet('{glob}')").fetchone()
    return n, distinct, total

def fingerprint(con, root, day):
    source = os.path.join(root, "src", f"event_date={day}", "part.parquet")
    return con.execute(f"SELECT count(*), round(sum(f0), 9) FROM read_parquet('{source}')").fetchone()

def backfill(con, root, ledger, fail_on=None):
    ran, skipped, failed = [], [], []
    for day in DAYS:
        fp = fingerprint(con, root, day)
        if ledger.get(day) == fp:
            skipped.append(day)
            continue
        try:
            run_day(con, root, day, "overwrite", fail_on)
            ledger[day] = fp
            ran.append(day)
        except RuntimeError:
            failed.append(day)
    return ran, skipped, failed

with tempfile.TemporaryDirectory() as root:
    make_source(root)
    con = duckdb.connect()

    for _ in range(2):
        for day in DAYS:
            run_day(con, root, day, "append")
    n, distinct, _ = stats(con, root)
    print(f"append twice:    {n:,} rows written, {distinct:,} distinct ids (duplicates: {n - distinct:,})")

with tempfile.TemporaryDirectory() as root:
    make_source(root)
    con = duckdb.connect()
    sums = []
    for _ in range(2):
        for day in DAYS:
            run_day(con, root, day, "overwrite")
        sums.append(stats(con, root))
    print(f"overwrite twice: {sums[1][0]:,} rows, {sums[1][1]:,} distinct ids, same checksum both runs: {sums[0] == sums[1]}")

with tempfile.TemporaryDirectory() as root:
    make_source(root)
    con = duckdb.connect()
    ledger = {}
    print("\nbackfill of 7 days, a worker dies on 2026-09-04")
    for label, fail in [("first run", "2026-09-04"), ("second run", None), ("third run", None)]:
        ran, skipped, failed = backfill(con, root, ledger, fail)
        print(f"  {label}: ran {len(ran)}, skipped {len(skipped)}, failed {failed or 'none'}")
    print(f"  rows now: {stats(con, root)[0]:,}")

    late = pd.DataFrame(np.random.default_rng(9).normal(0, 1, (300, 3)), columns=["f0", "f1", "f2"])
    late.insert(0, "id", np.arange(300) + 50_000)
    path = os.path.join(root, "src", "event_date=2026-09-02", "part.parquet")
    pd.concat([pd.read_parquet(path), late]).to_parquet(path)
    ran, skipped, failed = backfill(con, root, ledger)
    n, distinct, _ = stats(con, root)
    print(f"\n300 late rows arrive for 2026-09-02: ran {ran}, skipped {len(skipped)}; table has {n:,} rows, {distinct:,} distinct ids")

    good = os.path.join(root, "preds", "event_date=2026-09-01", "part.parquet")
    data = open(good, "rb").read()
    atomic_tmp = good + ".tmp"
    open(atomic_tmp, "wb").write(data[: len(data) // 2])
    after_atomic_crash = con.execute(f"SELECT count(*) FROM read_parquet('{good}')").fetchone()[0]
    open(good, "wb").write(data[: len(data) // 2])
    try:
        con.execute(f"SELECT count(*) FROM read_parquet('{good}')").fetchone()
        after_naive_crash = "readable"
    except duckdb.Error:
        after_naive_crash = "unreadable"
    print(f"crash halfway through a write: temp-then-rename leaves {after_atomic_crash:,} good rows; writing in place leaves a file that is {after_naive_crash}")
```

Read the output top to bottom. Appending twice doubles the table; overwriting twice leaves 14,000 rows with an identical checksum. In the backfill the failed day is the only one rerun, and a third invocation does nothing, which is what lets an orchestrator retry blindly. When 300 late rows arrive for 2 September, the fingerprint changes, only that partition reruns, and the table ends with 14,300 rows and 14,300 distinct ids. The last line is the reason for temp-then-rename: a half-written file is unreadable, while the previous good file survives a crash in the temporary one.

### 3. Will it fit the window?

Parameters are named: the rows per second per worker is an assumed rate you replace with one you measure on your own hardware and model. The block compares a closed-form lower bound with a longest-first scheduling simulation.

```python
import heapq
import math

def makespan_seconds(rows, rate, workers, largest_share, chunk_rows=None):
    biggest = largest_share * rows
    if chunk_rows is not None:
        biggest = min(biggest, chunk_rows)
    return max(rows / (rate * workers), biggest / rate)

def lpt(partitions, workers, rate):
    loads = [0.0] * workers
    heapq.heapify(loads)
    for size in sorted(partitions, reverse=True):
        lightest = heapq.heappop(loads)
        heapq.heappush(loads, lightest + size / rate)
    return max(loads)

rows, rate, window_hours = 50_000_000, 2000, 2.0
window = window_hours * 3600
print(f"{rows:,} rows a night, {rate:,} rows per second per worker (an assumed rate), window {window_hours:g} h")

need = math.ceil(rows / (rate * window))
print(f"workers needed if work splits perfectly: {need}")
print()
print(f"{'workers':>8}{'as partitioned':>16}{'fits':>6}{'chunked at 5M':>15}{'fits':>6}")
for workers in (2, 4, 8):
    skewed = makespan_seconds(rows, rate, workers, 0.30) / 3600
    chunked = makespan_seconds(rows, rate, workers, 0.30, 5_000_000) / 3600
    print(f"{workers:>8}{skewed:>14.2f} h{('yes' if skewed * 3600 <= window else 'no'):>6}{chunked:>12.2f} h{('yes' if chunked * 3600 <= window else 'no'):>6}")

partitions = [15_000_000] + [rows * 0.7 / 24] * 24
chunked_parts = []
for size in partitions:
    while size > 5_000_000:
        chunked_parts.append(5_000_000)
        size -= 5_000_000
    chunked_parts.append(size)
print("\nsimulated with the longest-first rule on 4 workers")
print(f"  25 partitions, one holds 30% of the rows: {lpt(partitions, 4, rate) / 3600:.2f} h")
print(f"  same data cut into chunks of at most 5M rows ({len(chunked_parts)} tasks): {lpt(chunked_parts, 4, rate) / 3600:.2f} h")
print(f"  perfect split: {rows / (rate * 4) / 3600:.2f} h; utilisation with the skew: {rows / (rate * 4) / lpt(partitions, 4, rate):.0%}")
```

Four workers are needed with a perfect split, and four workers do reach it only when the big partition is chunked: 1.74 hours against 2.08. Eight workers do nothing for the unchunked layout (still 2.08 hours) but cut the chunked layout to 0.87 hours. The simulated schedule (1.82 hours for 27 tasks) is a little above the perfect-split bound because task sizes do not divide evenly, and utilisation with the skew is 83%. The lab below runs the same formula; its defaults reproduce the 2.08 hour row, and choosing the 5 million chunk reproduces 1.74 hours.

<BatchWindowLab />

### 4. DuckDB's own partitioned write options

The previous block wrote one file per partition so that overwriting is trivial. DuckDB's `COPY` also supports partitioned output directly; this block checks what its options do, since the three behave very differently on a rerun.

```python
import glob
import os
import tempfile
import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE preds AS SELECT range AS id, '2026-09-0' || (1 + range % 2) AS day FROM range(10)")

with tempfile.TemporaryDirectory() as root:
    out = os.path.join(root, "preds")

    def files():
        return len(glob.glob(out + "/*/*.parquet"))

    def rows():
        return con.execute(f"SELECT count(*) FROM read_parquet('{out}/*/*.parquet')").fetchone()[0]

    con.execute(f"COPY preds TO '{out}' (FORMAT parquet, PARTITION_BY (day))")
    print(f"first write:                 {files()} files, {rows()} rows")
    try:
        con.execute(f"COPY preds TO '{out}' (FORMAT parquet, PARTITION_BY (day))")
    except duckdb.IOException as err:
        print("second plain write:          refused,", str(err).split(':')[0])
    con.execute(f"COPY preds TO '{out}' (FORMAT parquet, PARTITION_BY (day), OVERWRITE_OR_IGNORE)")
    print(f"OVERWRITE_OR_IGNORE rerun:   {files()} files, {rows()} rows")
    con.execute(f"COPY preds TO '{out}' (FORMAT parquet, PARTITION_BY (day), APPEND)")
    print(f"APPEND rerun:                {files()} files, {rows()} rows (duplicates)")
```

A plain rerun refuses to write into a non-empty directory, which is a safe default. `OVERWRITE_OR_IGNORE` replaces the files of the same name, so reruns stay idempotent here; `APPEND` adds a uniquely named file per run and doubles the rows. That matches the documentation's description of both options, and it is the same append-versus-overwrite choice from block 2 made by a flag.

## Production snippets (not run here)

These need a cluster, and use only API names that appear in the documentation cited above.

```python
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
scored.write.mode("overwrite").insertInto("predictions")
```

Not run in this environment.

```python
class Predictor:
    def __init__(self):
        self.model = load_model()

    def __call__(self, batch):
        batch["score"] = self.model.predict(batch["features"])
        return batch

scored = ds.map_batches(
    Predictor,
    compute=ray.data.ActorPoolStrategy(size=4),
    num_gpus=1,
    batch_size=256,
)
```

Not run in this environment. `load_model` stands for your own loader.

```python
scored = rows | RunInference(model_handler)
```

Not run in this environment. `model_handler` is a `ModelHandler` for your framework; reading and writing connectors are separate and are not covered here.

## Designing with it

- **Derive the output path from the input partition,** and write by overwrite, with a temporary file and a rename. Test it by running the job twice and diffing the output.
- **Keep a ledger of partition, status and input fingerprint.** It turns retries, restarts, backfills and late data into one operation.
- **Reconcile after every run.** Compare input and output row counts and distinct keys per partition; a silent loss is worse than a loud failure.
- **Bound partition size in both directions.** Chunk the heavy ones; do not make thousands of tiny ones (DuckDB suggests at least 100 MB per partition).
- **Sort variable-length inputs before batching** and restore the order after.
- **Measure rows per second per worker,** and put it in the capacity model. Re-measure when the model changes.
- **Version the model with the output.** Write the model version into each row or partition so you can tell which scores a rerun would replace.
- **Decide what a late partition means** before it happens: rerun and overwrite, or write a correction.

## Where this stands in 2026

:::info Industry view

- **Versions read for this chapter.** Spark documentation for 4.2.0, Airflow documentation for 3.3.2, Ray Data documentation for Ray 2.59 and later, and Beam's `RunInference` listed as requiring Beam 2.40.0 or later. The current Beam release was not checked.
- **LLM batch APIs are an everyday use.** Bedrock's batch inference and the batch endpoints in Azure Machine Learning make bulk scoring and bulk prompting a managed service; Bedrock's page notes that batch inference does not support tool calling or structured output, so check limits that matter to you.
- **The single-machine option is credible.** Columnar formats and embedded engines like DuckDB handle sizeable partitions on one node; reach for a cluster when the data or the model, not habit, demands it.
- **GPU batch work is a scheduling problem as well as a compute one.** Ray Data's streaming execution and Kubernetes Jobs both target keeping the device busy; see [Kubernetes for ML](/docs/mlops/platform/kubernetes-for-ml) for GPU placement.
- **Not verified here.** Throughput of any engine on real hardware, GPU behaviour, and object-store atomic rename semantics were not measured.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A nightly job appends predictions to a table. After a retry the dashboard shows twice the usual customers. What is wrong and how do you fix it without losing the retry?</summary>

The write is not idempotent: the retry appended again. Make the output path depend on the partition and overwrite it (block 2 gives 14,000 rows after two overwrites versus 28,000 after two appends), and publish by renaming a temporary file so a crash never leaves a partial result.

</details>

<details>
<summary><strong>Q2.</strong> You add workers from 4 to 8 and the job takes exactly as long. Give the most likely cause and a fix.</summary>

One partition is larger than the rest, so the job cannot finish before that partition does (block 3: 2.08 hours at both 4 and 8 workers). Cut large partitions into chunks of bounded size, then more workers help (0.87 hours at 8).

</details>

<details>
<summary><strong>Q3.</strong> Why is the ledger keyed on an input fingerprint and not just on the date?</summary>

Date alone says the partition ran, not that it ran on the data that exists now. A fingerprint (row count and checksum) detects late or corrected input, so only changed partitions rerun. In block 2, 300 late rows to 2 September rerun that one day and skip the other six.

</details>

<details>
<summary><strong>Q4.</strong> A transformer batch job spends most of its GPU time on padding. What do you change, and what must you remember to restore?</summary>

Sort or bucket the inputs by length before batching. In block 1 this lifts the useful share of processed tokens from 25.3% to 99.5%. Carry a row id through the job so results can be put back in the original order.

</details>

<details>
<summary><strong>Q5.</strong> Your task uses `datetime.now()` to decide which day to process. Why is that a bug in a batch pipeline, and what do you use instead?</summary>

It produces a different result on every run, so a retry on the next day processes the wrong partition. Airflow's guide says never to use `now()` inside a task for the critical computation; use the data interval start as the partition key for both reading and writing.

</details>

<details>
<summary><strong>Q6.</strong> When would you pick Ray Data over Spark for batch inference, and when Beam?</summary>

Ray Data when a GPU model with expensive preprocessing needs actor pools that load the model once and overlap preprocessing with inference. Spark when the work is dominated by joins and aggregation over partitioned tables before scoring. Beam when the same pipeline must run on both batch and streaming data. For modest volumes, DuckDB on one machine can beat all three on simplicity.

</details>

## Further reading

- [Apache Airflow best practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html): idempotent tasks, data intervals instead of `now()`, and upserts.
- [Apache Spark: generic load and save functions](https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html) and [Parquet and partition discovery](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html): save modes, atomicity, `partitionBy` and dynamic partition overwrite.
- [Ray Data: batch inference](https://docs.ray.io/en/latest/data/batch_inference.html): actor pools, batch size, GPUs and streaming execution.
- [Apache Beam: ML inference with RunInference](https://beam.apache.org/documentation/ml/about-ml/): model handlers, dynamic batching and dead-letter queues.
- [DuckDB: partitioned writes](https://duckdb.org/docs/current/data/partitioning/partitioned_writes.html): `PARTITION_BY`, `OVERWRITE_OR_IGNORE`, `APPEND` and partition size advice.
- [Amazon SageMaker AI: batch transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html), [Amazon Bedrock: batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) and [Azure Machine Learning: endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2): managed batch scoring.
- [Kubernetes: Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/): indexed completion and retries for sharded batch work.

## Check yourself

- I can decide between a batch job and an endpoint from who is waiting and how stale an answer may be.
- I can make a job idempotent by deriving its output from the input partition, overwriting and publishing atomically.
- I can design a ledger and a backfill runner that retries failures and reruns only partitions whose input changed.
- I can explain why batch size changes the number of calls but not the answers, and why sorting by length saves compute.
- I can tell when more workers will not help and chunk skewed partitions to fix it.
- I can compare Spark, Ray Data and Beam by the work around the model, and say which of their behaviours I took from documentation.
