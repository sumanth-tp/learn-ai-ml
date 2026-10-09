---
id: dm-warehouses-lakes-and-lakehouses
title: "Data Management · Session 3 — Warehouses, Lakes and Lakehouses"
sidebar_label: "3 · Architectures"
sidebar_position: 3
slug: /mlops/data/warehouses-lakes-and-lakehouses
description: "Compare warehouse, lake and lakehouse designs, analytical models and bronze-to-gold refinement."
tags: [data-management, data-warehouse, data-lake, lakehouse]
---

import Infographic from '@site/src/components/Infographic';
import SnapshotTimeTravelLab from '@site/src/components/viz/SnapshotTimeTravelLab';

**In one line.** Choose storage and processing paths from the data's consumers, then make refinement and ownership explicit.


:::tip Before you start

**You should already know**

- Row and columnar layouts, and what a Parquet file is: [data representations](/docs/mlops/data/representations-for-ml).
- What a validation rule does at a boundary: [data quality rules](/docs/mlops/data/quality-rules).
- Basic SQL: `SELECT`, `GROUP BY` and `JOIN`.

**Reading time.** About 45 minutes, plus a second to run the demo.

**After this chapter you can**

- say which guarantees a bare directory of files lacks and a table format adds,
- read a table at a past snapshot and explain why a directory listing is not the same thing,
- predict which schema changes a reader survives and which break it.

:::

## In 30 seconds

Picture a shared folder of spreadsheets that several people update. If you add up every file in the folder you may count a corrected sheet and its older copy twice, or include a sheet someone is still writing. A table format fixes this with one small index file that says "the table is exactly these files". Readers follow the index, so they always see one finished version. Warehouses, lakes and lakehouses differ mainly in how much of that bookkeeping they give you.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Warehouse | Curated tables for SQL analysis, checked before loading | A finance reporting database |
| Lake | Cheap storage for raw and varied files | A bucket of JSON and Parquet |
| Lakehouse | Lake files plus table metadata for snapshots and transactions | Parquet files plus a manifest |
| Snapshot | One committed version of a table, listing its files | Snapshot 3 = `a_fixed`, `b` |
| Manifest | The file that lists a snapshot's data files | `v_3.json` |
| Time travel | Reading the table as of an older snapshot | Read snapshot 1 |
| Schema evolution | Changing a table's columns over time | Adding a `currency` column |
| Compaction | Rewriting many small files as fewer large ones | 300 files into 1 |

## The idea in plain words

An operational service, an analyst and a model trainer can all use the same source events while asking different questions. The service needs small, reliable reads and writes. The analyst wants to scan years of history and group by business dimensions. The trainer needs reproducible snapshots with labels and feature cutoffs. A data architecture joins these workloads without letting one consumer silently change another's meaning.

Three storage designs are worth telling apart: a **warehouse**, a **lake** and a **lakehouse**. A warehouse commonly curates structured tables for SQL analytics. A lake stores varied raw or lightly processed files for flexible later use. A lakehouse applies table management such as snapshots, schema control and transactions to files in a lake. These descriptions are useful starting points, not strict product categories. Warehouses can ingest JSON, lakes can enforce schemas and a lakehouse still needs governance. The choice is about the concrete guarantees the workload requires.

<Infographic src="/img/dm/storage-architectures.svg" alt="A warehouse holds curated analytical tables, a lake holds varied files and a lakehouse adds table metadata; bronze, silver and gold show increasing refinement." caption="Architecture choices and refinement layers answer different questions: where data lives and how its meaning improves." />

Two workload types matter as well: **OLTP** and **OLAP**. Online transaction processing handles many short operations that change current state, such as recording a payment. Online analytical processing scans and aggregates larger collections, often with historical context. Row-oriented and column-oriented layouts often align with these patterns, but layout alone does not define either workload. The operational source is where events happen; analytics and ML generally consume a versioned, transformed view of them.

:::note Added for this site

The course gives the three architecture labels, star schema, medallion layers and Lambda/Kappa patterns. This chapter adds decision boundaries, snapshot semantics, failure handling, an explicit distinction between file format and table-format guarantees, and a measured demo of a versioned table.

:::

The board above separates the storage choice from the refinement path. The code below works through a small fact table and a validated event stream.

## Worked example, step by step

A tiny table keeps `id` and `amount`. We follow it through three commits and one unfinished write, adding the amounts by hand.

1. Commit 1 writes file `a` with amounts 10, 20, 30 and publishes manifest `v_1` listing `[a]`. The table has 3 rows and a total of 60.
2. Commit 2 writes file `b` with amounts 40, 50 and publishes `v_2` listing `[a, b]`. The table has 5 rows and a total of 60 + 90 = 150.
3. Commit 3 corrects one amount. It writes a new file `a_fixed` with 10, 25, 30 and publishes `v_3` listing `[a_fixed, b]`. Old file `a` stays on disk but is no longer listed. The total is 65 + 90 = 155 over 5 rows.
4. Someone starts a fourth write and leaves file `c_half_written` with amount 999 on disk, never committed.
5. A reader who lists the directory finds `a`, `a_fixed`, `b` and `c_half_written`: 3 + 3 + 2 + 1 = 9 rows and 60 + 65 + 90 + 999 = 1,214. A reader who follows the latest manifest sees 5 rows and 155.
6. Time travel is just choosing a different manifest. Snapshot 1 still gives 3 rows and 60.

In words: the manifest is the table. Files are only storage. The experiment below builds exactly this and then breaks it with schema changes.

## How it works

### Warehouse, lake, lakehouse

- **Warehouse**; Structured, schema-on-write, BI.
- **Lake**; Raw, schema-on-read, cheap (swamp risk).
- **Lakehouse**; Lake + ACID/schema (Delta, Iceberg).

### OLTP vs OLAP & modelling

- **OLTP → OLAP**; Many small transactions (row) feed few large analytical queries (columnar).
- **Star schema & medallion**; Fact + dimension tables; Bronze→Silver→Gold layering. Lambda vs Kappa for batch/stream.


## A real system that works this way

**Databricks' medallion guidance** is a named production pattern for organising lakehouse data. Bronze retains source data, silver validates and deduplicates it, and gold presents curated aggregates or business-ready tables. The guidance recommends preserving raw history so later layers can be rebuilt. These are logical quality stages, not a requirement that every organisation name schemas bronze, silver and gold or use one vendor.

Imagine payment events arriving from a transaction service. Bronze records the payload and source metadata as received. Silver parses the amount and time, maps source account IDs, removes duplicate event IDs and quarantines malformed events while retaining detail. Gold publishes daily spending per customer and merchant category for an analyst. A training pipeline may instead consume silver detail and calculate rolling features at a historical cutoff. If it reads a gold table created after the label date without time travel or a cutoff, the model may learn from future information.

**Apache Iceberg** is a concrete open table-format example. Its table metadata tracks snapshots and data files; its API supports transactions and schema changes. The transaction belongs to the table layer, not to a bare directory full of Parquet files. A reader can identify one committed snapshot while a writer prepares a new one. That matters for a model training run that must see one coherent table state rather than a mixture of old and new files. Snapshot retention is still an operational choice: historical versions consume storage and need a policy.

## Code you can run

A tiny SQLite star schema makes the fact-and-dimension idea concrete. The fact table records events and amounts; dimensions attach customer and category descriptions. The query groups by a dimension without repeating that text in every fact row.

```python
import sqlite3

db = sqlite3.connect(":memory:")
db.executescript('''
CREATE TABLE customer (customer_id TEXT PRIMARY KEY, region TEXT NOT NULL);
CREATE TABLE category (category_id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE payment (
    payment_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customer(customer_id),
    category_id TEXT NOT NULL REFERENCES category(category_id),
    amount INTEGER NOT NULL
);
INSERT INTO customer VALUES ('C1', 'North'), ('C2', 'South');
INSERT INTO category VALUES ('G', 'Groceries'), ('T', 'Travel');
INSERT INTO payment VALUES
    ('P1', 'C1', 'G', 20), ('P2', 'C1', 'T', 30), ('P3', 'C2', 'G', 40);
''')
totals = db.execute('''
SELECT c.region, SUM(p.amount)
FROM payment AS p JOIN customer AS c USING (customer_id)
GROUP BY c.region ORDER BY c.region
''').fetchall()
print(totals)
assert totals == [("North", 50), ("South", 40)]
db.close()
```

The next block models bronze-to-silver validation and an idempotent event ID. It deliberately keeps the rejected record instead of pretending that dropping it is harmless. A real pipeline would persist both accepted and quarantined outputs atomically or use a recovery plan.

```python
raw = [
    {"event_id": "E1", "amount": "20"},
    {"event_id": "E1", "amount": "20"},
    {"event_id": "E2", "amount": "bad"},
    {"event_id": "E3", "amount": "40"},
]
accepted = {}
quarantined = []
for event in raw:
    if event["event_id"] in accepted:
        continue
    try:
        amount = int(event["amount"])
        if amount < 0:
            raise ValueError("negative amount")
    except ValueError:
        quarantined.append(event)
        continue
    accepted[event["event_id"]] = amount
gold_total = sum(accepted.values())
print(accepted, quarantined, gold_total)
assert accepted == {"E1": 20, "E3": 40}
assert quarantined == [{"event_id": "E2", "amount": "bad"}]
assert gold_total == 60
```

This is a teaching calculation, not an implementation of Iceberg or a distributed stream processor. It demonstrates the quality decisions that an architecture must preserve when scaled up.

### Experiment: a versioned table made of Parquet files

The demo below uses only pyarrow 25.0.1 and DuckDB 1.5.6. A commit writes a manifest as a temporary file and renames it into place, which is atomic on a normal local file system. It then reads snapshots, compares a manifest read with a directory listing, evolves the schema two ways and times a scan over many small files. This is a teaching model of what Apache Iceberg, Delta Lake and similar formats do, not an implementation of any of them.

```python
import json
import os
import tempfile
import time

import duckdb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

root = tempfile.mkdtemp()
os.makedirs(f"{root}/data")

def write_file(name, rows):
    pq.write_table(pa.table(rows), f"{root}/data/{name}.parquet")
    return f"data/{name}.parquet"

def commit(files):
    versions = sorted(int(f[2:-5]) for f in os.listdir(root) if f.startswith("v_") and f.endswith(".json"))
    version = (versions[-1] + 1) if versions else 1
    with open(f"{root}/v_{version}.json.tmp", "w") as handle:
        json.dump({"version": version, "files": files}, handle)
    os.replace(f"{root}/v_{version}.json.tmp", f"{root}/v_{version}.json")
    return version

def snapshot(version):
    with open(f"{root}/v_{version}.json") as handle:
        return json.load(handle)["files"]

def total(files):
    paths = ", ".join(f"'{root}/{f}'" for f in files)
    return duckdb.sql(f"SELECT count(*), sum(amount) FROM read_parquet([{paths}])").fetchone()

a = write_file("a", {"id": [1, 2, 3], "amount": [10, 20, 30]})
v1 = commit([a])
b = write_file("b", {"id": [4, 5], "amount": [40, 50]})
v2 = commit([a, b])
a_fixed = write_file("a_fixed", {"id": [1, 2, 3], "amount": [10, 25, 30]})
v3 = commit([a_fixed, b])
for version in (v1, v2, v3):
    print(f"snapshot {version}: rows, total = {total(snapshot(version))}")
print("files on disk:", sorted(os.listdir(f"{root}/data")))

orphan = write_file("c_half_written", {"id": [6], "amount": [999]})
latest = max(int(f[2:-5]) for f in os.listdir(root) if f.startswith("v_") and f.endswith(".json"))
listing = [f"data/{f}" for f in os.listdir(f"{root}/data")]
print(f"directory listing sees {total(listing)}, manifest still says {total(snapshot(latest))}")

wide = write_file("d_wide", {"id": [7], "amount": [60], "currency": ["GBP"]})
print("added column, unified schema:", ds.dataset([f"{root}/{f}" for f in (a_fixed, wide)], format="parquet", schema=pa.unify_schemas([pq.read_schema(f"{root}/{a_fixed}"), pq.read_schema(f"{root}/{wide}")])).to_table().to_pydict())
retyped = write_file("e_retyped", {"id": [8], "amount": ["seventy"]})
try:
    ds.dataset([f"{root}/{a_fixed}", f"{root}/{retyped}"], format="parquet").to_table()
except Exception as error:
    print("type change:", type(error).__name__, str(error).splitlines()[0][:90])
try:
    pa.unify_schemas([pq.read_schema(f"{root}/{a_fixed}"), pq.read_schema(f"{root}/{retyped}")])
except Exception as error:
    print("unify:", type(error).__name__)

small = tempfile.mkdtemp()
for i in range(300):
    pq.write_table(pa.table({"id": list(range(i * 100, i * 100 + 100)), "amount": [i] * 100}), f"{small}/p{i:03d}.parquet")
many = ", ".join(f"'{small}/p{i:03d}.parquet'" for i in range(300))
duckdb.sql(f"COPY (SELECT * FROM read_parquet([{many}])) TO '{small}/compact.parquet' (FORMAT parquet)")

def best_scan(source):
    times = []
    for _ in range(3):
        start = time.perf_counter()
        duckdb.sql(f"SELECT sum(amount) FROM read_parquet({source})").fetchall()
        times.append(time.perf_counter() - start)
    return min(times)

t_many, t_one = best_scan(f"[{many}]"), best_scan(f"'{small}/compact.parquet'")
print(f"300 files of 100 rows: {t_many:.4f}s; one compacted file of the same 30,000 rows: {t_one:.4f}s; {t_many / t_one:.0f}x")
```

**Reading the output.** The first three lines are the totals at each snapshot and match the hand calculation. `directory listing sees` against `manifest still says` is the torn-read comparison. The schema lines show a compatible change (a new column) and an incompatible one (a text value in a number column). The last line times a scan over 300 small files against one compacted file.

**Line by line.**

- `os.replace(tmp, final)` is the whole commit protocol. A reader sees either the old manifest or the new one, never half of a manifest.
- `total(snapshot(version))` reads only the files that manifest names, so snapshot 1 still works after commit 3.
- `pa.unify_schemas` merges the old and new schemas and fills missing columns with null. It raises `ArrowTypeError` when the same column has two incompatible types.
- `COPY ... TO ... (FORMAT parquet)` rewrites the 300 files as one, which is compaction.

The printed output:

```text
snapshot 1: rows, total = (3, 60)
snapshot 2: rows, total = (5, 150)
snapshot 3: rows, total = (5, 155)
files on disk: ['a.parquet', 'a_fixed.parquet', 'b.parquet']
directory listing sees (9, 1214), manifest still says (5, 155)
added column, unified schema: {'id': [1, 2, 3, 7], 'amount': [10, 25, 30, 60], 'currency': [None, None, None, 'GBP']}
type change: ArrowInvalid Failed to parse string: 'seventy' as a scalar of type int64
unify: ArrowTypeError
300 files of 100 rows: 0.0099s; one compacted file of the same 30,000 rows: 0.0003s; 29x
```

### Reading the experiment

The totals confirm the hand arithmetic: 60, 150 and 155. The striking line is the directory listing. It returned 9 rows and a total of 1,214 where the committed table holds 5 rows and 155, which is 7.8 times too large. Two things caused it: the superseded file `a` and its corrected twin `a_fixed` were both counted, and the half-written file added 999. Nothing in a bare directory says which files belong together. That is the gap a table format fills, and it is why a training job should read a pinned snapshot.

Schema changes split into two kinds. Adding a `currency` column was harmless: old rows read as null. Putting the text `seventy` in a column that earlier files stored as 64-bit integers failed, and the type-merge raised `ArrowTypeError`. Notice that the failure appears at read time, not at write time, because plain Parquet files do not check each other. A table format would refuse the commit.

The last line is the small-file problem: 300 files of 100 rows took 0.0099 s against 0.0003 s for the same rows in one file, 29 times slower, though the exact ratio varied between 28 and 40 times across runs. Limits: a local disk, a few rows, one writer at a time and no concurrency control. A real format adds optimistic concurrency so two writers cannot silently overwrite each other.

<SnapshotTimeTravelLab />

**What each control does.**

- **snapshot version** picks which committed manifest to read, 1 to 3.
- **read the directory listing instead of the manifest** ignores the manifest and reads every file on disk, including superseded and half-written ones.

**Try it yourself.**

1. Leave the checkbox off and move the slider from 1 to 3. The totals are 60, 150 and 155, the numbers printed above. The slider is time travel.
2. Tick the checkbox. The lab reads all four files and shows 9 rows and 1,214, matching the experiment. The version slider no longer matters because no manifest is being read.
3. Untick it again and set version 2, then 3. Only one file changes between them (`a` becomes `a_fixed`), and the total moves by 5, the correction of 20 to 25.

<Infographic src="/img/dm-enrich/manifest-vs-listing.svg" alt="Two columns compare reading a table through its manifest, 5 rows and 155, with listing the directory, 9 rows and 1,214, and cards show schema evolution and compaction results." caption="Look first at the red card: the directory listing overstates the committed total by 7.8 times." />

## Designing with it

### Distinguish storage from table guarantees

A lake may store Parquet objects cheaply and flexibly, but those objects do not by themselves provide one atomic multi-file table update. A table format such as Iceberg adds metadata and commit rules that identify the active files in a snapshot. A warehouse often provides transactional tables and managed SQL access. Evaluate the actual system's guarantees: isolation during updates, delete propagation, schema evolution, access control, catalogue discovery and recovery. Saying "lakehouse" without specifying these guarantees is not an architecture decision.

The schema-on-write versus schema-on-read distinction is a tendency, not a universal law. A curated warehouse table normally validates fields before publication. A raw lake zone may retain payloads before assigning full meaning. Both can contain layers with stronger or weaker controls. Schema-on-read does not mean schema-free: every successful query still interprets bytes through a schema, even if that schema is applied later. Delay can be useful for exploratory data, but postponing validation can move surprise and cost to consumers.

### Model facts and dimensions around questions

A star schema centres a **fact** table of events or measurements, connected to **dimensions** that describe who, what, where and when. State the grain first: one row per payment event, not one row per customer or one row per day. A dimension table can then supply customer region or product category at a defined point in time. If a customer's region changes, decide whether historical reports use the old region or the current region. That is a business question encoded in slowly changing dimension handling, which Chapter 9 develops.

A star schema is effective for repeated analytical questions, but it is not automatically the right shape for every training job. A model may need event sequence, raw text and point-in-time joins. Keep a trace from a derived feature to the fact rows and dimension versions that produced it. A daily aggregate cannot recreate individual event order after it has discarded that detail.

### Make the layers and batch/stream paths explicit

Bronze, silver and gold can separate retention, validation and consumption. Define the contract at each transition: accepted schema, key, deduplication rule, watermark, output freshness and handling of invalid rows. A direct source-to-gold shortcut may be fine for a small, trusted dataset, but it removes a recovery point if raw history is not retained elsewhere. Keep only the raw data needed for replay and audit within privacy and retention limits.

**Lambda** and **Kappa** are two ways to combine batch and stream. Lambda combines a batch recomputation path with a low-latency stream path; this can provide fresh results and historical correction, but the two implementations may disagree. Kappa processes through a replayable stream path and uses the log to reprocess; that reduces duplicated logic but requires enough retained events and suitable replay semantics. Neither label guarantees exactly-once business outcomes. Stable event IDs, idempotent writes and explicit late-arrival policy still matter. Choose the simplest path that meets latency and correction needs.

## Trace a payment across the architecture

A customer makes a payment at nine in the morning. The transaction system commits a new payment row and responds to the customer. It cannot wait for a warehouse refresh or a feature backfill. A change-data-capture or event export process later publishes the payment with an event ID, source commit time and payload version. The pipeline records it in a raw zone, even if one downstream parser cannot yet recognise a new optional field. That preserves an explanation for what was received.

In silver, a validation task checks the amount, currency and account reference. It may find the payment twice because an exporter retried after a timeout. Deduplication by event ID lets the second delivery be harmless, provided the two payloads are genuinely the same event. If they differ, silently keeping the first may hide a correction; the contract must define whether a corrected event carries a new version or a compensating event. Invalid records go to a visible quarantine with a repair owner. A successful task run is not enough if the quarantine grows unnoticed.

An analyst's gold table groups payments by local business day. The conversion from UTC to local date must be declared; otherwise a late-night payment can shift days across reports. A model feature may need the prior 30 days as known at a prediction cutoff. That requires event time, ingestion time and a consistent historical snapshot. Rebuilding the feature from today's corrected silver table might yield different values than the original training run. Store or identify the snapshot and transformation version so the difference can be explained.

Suppose the silver table is updated while training reads it. Without snapshot isolation, early batches could use yesterday's account mapping and later batches today's. A table snapshot allows the training job to pin a coherent version. It does not solve all leakage: a coherent snapshot taken after the prediction date still contains future data. The training query must enforce a point-in-time cutoff as well. Table consistency and historical correctness are separate guarantees.

Finally, ask whether each consumer can discover the right asset. A catalogue entry should name the table's owner, grain, schema, update schedule, quality status and permitted uses. Access control must cover raw sensitive payloads as well as curated results. The architecture is complete only when a consumer can find a trustworthy dataset, understand its limitations and trace its derived values back to evidence.

### Design for correction, not only first publication

The first successful load is easy to demonstrate. The harder test is a source correction after several consumers have already used the data. Suppose a payment was recorded twice and identified three weeks later. Which bronze records preserve the evidence? Which silver partitions need recomputation? Which gold aggregates and training snapshots include the duplicate? A lineage map from source event to downstream products gives a bounded repair plan. Without it, teams may rebuild everything or quietly leave inconsistent reports in place.

Set a replay policy before failures occur. A batch job can recalculate a date partition; a streaming job may need to replay from an offset or emit compensating changes. Both paths should be idempotent so a retry does not add the same payment twice. If Lambda has separate stream and batch implementations, reconcile their outputs for overlapping periods and state which result becomes authoritative after correction. If Kappa replays a log, retain events for at least the recovery window and test that old schemas can still be decoded.

Finally, control cost and accessibility. Keeping every raw copy indefinitely is not a free form of safety; storage, privacy obligations and catalogue clutter grow. Retain enough history to meet replay and audit needs, then expire snapshots and raw objects under a documented policy. Partitioning can reduce scan cost but overly fine partitions can create many tiny files and metadata overhead. Treat the architecture as an operated product with owners, budgets and recovery exercises rather than a diagram that ends at a gold table.

## Where this stands in 2026

:::info Industry view

- Current lakehouse guidance uses bronze, silver and gold as a logical refinement pattern, with validated detail in silver and business-ready outputs in gold.
- Open table formats such as Apache Iceberg manage snapshots and transactions over data files; snapshot retention and cleanup remain operational concerns.
- Warehouses, lakes and lakehouses overlap in capability. Select on workload, governance and required guarantees instead of assuming a name implies a feature.

:::

## Common mistakes

1. **Reading a directory of files as if it were a table.** It feels natural because the files are right there. It double counted a corrected file and included an unfinished write (1,214 against 155). Read through a manifest or a table format.
2. **Believing a bare lake gives transactions.** Object storage holds files, not commits. Without a table layer, a reader can see half a write.
3. **Assuming a schema change is always safe.** An added column was fine, a changed type broke the read. Check each change for compatibility before publishing.
4. **Keeping every old file and snapshot forever.** Time travel is useful, but each snapshot holds storage and clutter. Set a retention policy.
5. **Writing thousands of tiny files.** The scan was 29 times slower. Batch writes and compact on a schedule.

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast a data warehouse, data lake and lakehouse.</summary>

Warehouse: structured, schema-on-write, BI. Lake: raw, schema-on-read, cheap (swamp risk). Lakehouse: lake storage + ACID/schema (Delta, Iceberg).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast OLTP and OLAP.</summary>

OLTP: many small transactions, row-oriented, normalised, current (operational source). OLAP: few large analytical queries, columnar, denormalised, historical (analytics).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> What is the medallion architecture?</summary>

Layering data by refinement: Bronze (raw) → Silver (cleaned/conformed) → Gold (curated/aggregated, ML/BI-ready).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> What is a star schema?</summary>

A central fact table surrounded by dimension tables, denormalised for fast analytical aggregation.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Contrast Lambda and Kappa architectures.</summary>

Lambda runs parallel batch + speed (stream) layers; Kappa is stream-only, reprocessing from the log; simpler, one code path.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q6. (Medium)</strong> The manifest says 5 rows and 155 but a directory listing found 9 rows and 1,214. Explain each extra row.</summary>

The listing adds file `a` (3 rows, 60), which commit 3 superseded, so those rows are counted twice alongside `a_fixed` (3 rows, 65). It also adds the uncommitted `c_half_written` (1 row, 999). So 5 + 3 + 1 = 9 rows and 155 + 60 + 999 = 1,214.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> A new file stores `amount` as text. Plain Parquet accepted the write. Where should the failure have happened, and what does a table format do?</summary>

It should have failed at commit, before any reader sees the file. A table format checks the incoming schema against the table schema and rejects an incompatible commit, so the manifest never lists the bad file. With bare files the error appears later, on the first reader that merges the schemas.

</details>

## Go deeper

- [Databricks medallion guidance](https://docs.databricks.com/aws/en/lakehouse/medallion) explains raw, validated and curated layers.
- [Apache Iceberg API](https://iceberg.apache.org/docs/latest/api/) describes table metadata and transactions.
- [Apache Iceberg snapshots](https://iceberg.apache.org/docs/latest/branching/) describes snapshot history and retention.
- [Apache Parquet concepts](https://parquet.apache.org/docs/concepts/) (opened 2026-10-09) defines a row group as a horizontal partition holding one column chunk per column.
- [DuckDB file format guide](https://duckdb.org/docs/current/guides/performance/file_formats) (opened 2026-10-09) advises files of roughly 100 MB to 10 GB and row groups of 100,000 to 1,000,000 rows, which is why 300 files of 100 rows is a bad layout.
- The Iceberg pages above were not readable in full when this chapter was revised on 2026-10-09, so the demo claims nothing about Iceberg beyond what the chapter already cited.
- Library versions run for the demo: pyarrow 25.0.1, DuckDB 1.5.6, Python 3.14.6.
- Built from the course lecture "dm-s3-architectures" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can select a warehouse, lake or lakehouse from the consumer's access and governance needs.
- [ ] I can explain OLTP, OLAP, a fact table's grain and a dimension's historical meaning.
- [ ] I can trace a record from raw retention through validation to a curated product.
- [ ] I can distinguish snapshot consistency from a leakage-safe point-in-time query.
- [ ] I can explain why a directory listing and a committed snapshot can give different totals.
- [ ] I can read a table at a past snapshot and say what must be retained to allow it.
- [ ] I can tell a compatible schema change from an incompatible one and where each failure shows up.

## Where to go next

Next is [building reliable data pipelines](/docs/mlops/data/reliable-pipelines), which makes the writes into these tables safe to repeat. For the leakage side of snapshots see [features and point-in-time correctness](/docs/mlops/data/features-and-point-in-time).
