---
id: dm-data-representations-for-ml
title: "Data Management · Session 1 — Data Representations for ML"
sidebar_label: "1 · Representations"
sidebar_position: 1
slug: /mlops/data/representations-for-ml
description: "Choose data models, exchange formats and row or column layouts for machine learning work."
tags: [data-management, parquet, data-models, machine-learning]
---

import Infographic from '@site/src/components/Infographic';
import ColumnProjectionLab from '@site/src/components/viz/ColumnProjectionLab';

**In one line.** Represent data according to the reads, writes and decisions it must support.

## The idea in plain words

A model sees features, but a product begins with events, records, images and documents. Data management decides which facts are retained, how they are identified, how they move and what a reader may assume about them. Poor handling can make a sound model answer the wrong question because its inputs are missing, stale or defined differently from the training set.

The lecture introduces the **data → information → knowledge → wisdom** ladder. A transaction record is data: a customer identifier, a time and an amount. Information appears when we interpret it, for example monthly spending by customer. Knowledge is a tested pattern such as a reliable relation between payment history and default risk. Wisdom is the decision made with that pattern under business, legal and human constraints. The ladder is a teaching aid, not an automatic pipeline: aggregation does not guarantee insight, and a model's prediction does not decide what action is fair or useful.

<Infographic src="/img/dm/representations.svg" alt="Row records, exchange formats and columnar Parquet support different operations; selecting two of fifty equal columns is an ideal four per cent projection." caption="Representation choices follow workload: transactions, interchange and analytical scans ask different questions." />

Data may be **structured**, such as a table with declared fields; **semi-structured**, such as JSON with nested keys that vary by event; or **unstructured**, such as an image or a document. These names describe the form of the payload, not whether it has any metadata. A photograph can still have a structured asset ID, creation time, rights status and label table. Text becomes model-ready only after tokenisation or embedding, yet its source and access rules remain important.

:::note Beyond the lecture

The source gives the representation vocabulary and a numerical scan example. The rest of this chapter adds schema evolution, stable identity, data contracts and a practical choice process for ML systems.

:::

Move the sliders to change the number of fields selected. The lecture assumes a 10 GB row-oriented representation and 2 GB Parquet file, a fivefold size ratio. Under the simplifying assumption of 50 equal-sized columns, reading two gives a **4%** projection and an ideal **0.08 GB** of Parquet column data. Metadata, row groups, compression differences and filters make a real scan different.

<ColumnProjectionLab />

## How it works

### The DIKW hierarchy

Data → Information → Knowledge → Wisdom. ML turns data into information and predictive knowledge; data quality caps model quality.

### Formats, models & structure

- **Structured**; Tables/relations.
- **Semi-structured**; JSON, XML (self-describing).
- **Unstructured**; Text, images, audio.

**Models:** relational, document, key-value, wide-column, graph. **Formats:** CSV, JSON, Parquet, Avro.

### Row vs columnar

- **Row (OLTP)**; Fields of a record together; fast writes & whole-row reads.
- **Columnar (OLAP)**; Each column together; fast scans, big compression (Parquet/ORC).

:::tip

**Worked.** 10 GB → 2 GB Parquet = 5× smaller; a query on 2 of 50 columns reads ~4% of data.

:::


## A real system that works this way

**DuckDB** is a concrete analytical system that can read Parquet files directly. Its documentation describes projection and filter pushdown, so a query selecting a few columns and rows may avoid reading unrelated data. It also explains when loading Parquet into a DuckDB table can improve repeated or join-heavy queries. This is a useful counterexample to the idea that a file format alone determines performance: layout, metadata, query shape and repetition all matter.

Imagine a customer-support team collecting ticket events. The live service writes each ticket update as a small transaction. An export job stores immutable event batches with ticket ID, event time, category, free text and resolution state. Analysts scan only time, category and state to chart backlog. A training job reads text and the historical resolution label, with a cutoff that prevents future events leaking into an earlier example. The same business event crosses transactional, analytical and modelling boundaries, but it must keep its identity and meaning.

A production reader needs more than a path to a file. It needs a schema, field definitions, time zone, null semantics and a version. Does a missing resolution state mean "open", "unknown" or "not collected"? Does event time mean occurrence or ingestion? These distinctions alter the label and the model's features. Document them where the data is published and test them at the boundary. A compressed, fast file containing ambiguous fields remains poor training data.

## Code you can run

The lecture's arithmetic is deliberately simple. Compression ratio compares physical sizes. The selected-column fraction is a *logical* fraction, not a promise about bytes read by a particular Parquet engine.

```python
row_gb = 10.0
parquet_gb = 2.0
all_columns = 50
selected_columns = 2

compression_ratio = row_gb / parquet_gb
projected_fraction = selected_columns / all_columns
ideal_parquet_gb = parquet_gb * projected_fraction
print(f"size ratio={compression_ratio:.1f}x")
print(f"ideal selected fraction={projected_fraction:.0%}")
print(f"ideal Parquet column bytes={ideal_parquet_gb:.2f} GB")
assert compression_ratio == 5.0
assert projected_fraction == 0.04
assert ideal_parquet_gb == 0.08
```

The next block shows why record identity and time semantics matter before any model is trained. A ticket may have several events, so treating each event as a different ticket changes a ticket-level feature. The cutoff is inclusive and expressed in UTC for this example.

```python
from datetime import datetime, timezone

events = [
    {"ticket_id": "T1", "at": "2026-01-03T10:00:00+00:00", "state": "open"},
    {"ticket_id": "T1", "at": "2026-01-05T10:00:00+00:00", "state": "resolved"},
    {"ticket_id": "T2", "at": "2026-01-04T12:00:00+00:00", "state": "open"},
]
cutoff = datetime(2026, 1, 4, 23, 59, tzinfo=timezone.utc)
known = [event for event in events if datetime.fromisoformat(event["at"]) <= cutoff]
latest = {}
for event in known:
    previous = latest.get(event["ticket_id"])
    if previous is None or event["at"] > previous["at"]:
        latest[event["ticket_id"]] = event
print({ticket: event["state"] for ticket, event in sorted(latest.items())})
assert {ticket: event["state"] for ticket, event in latest.items()} == {"T1": "open", "T2": "open"}
```

At the cutoff, T1 is still open. Reading its later resolution into the earlier feature table would leak future information. The code uses ISO strings only because all example timestamps share the same explicit offset; production pipelines should parse and normalise mixed time zones before ordering.

## Designing with it

### Separate a data model from a file format

A **data model** says how entities and relationships are represented. A relational model makes tables and keys explicit; a document model can keep a nested event together; a key-value model makes retrieval by one key simple; a graph model makes relationships traversable. The lecture also names wide-column stores. Those models guide access patterns and constraints. CSV, JSON, Avro and Parquet are **formats** that encode bytes for exchange or storage. A JSON file may carry document-like objects, but choosing JSON does not itself give you a document database's indexes or transactions.

For ML, a stable entity key often matters more than the initial storage technology. A customer, device or ticket needs an identity that survives exports and recomputations. Define whether two records with the same key represent updates, duplicates or separate events. Keep event time and ingestion time separately when late arrivals are possible. Preserve raw source data long enough to reconstruct a feature after a transformation bug, subject to retention and privacy rules.

### Match layout to the operation

Row-oriented storage keeps a record's fields together, which suits point lookup and updates of a whole record. Column-oriented storage groups values from the same field, which suits wide scans that use only a few fields and often compresses repeated values well. This is a useful default distinction, not a law: row stores can run analytics, column stores can support selective access, and indexes, partitions and caches may dominate a specific query. Benchmark the queries that matter to the product.

The lecture's 10 GB to 2 GB example has two independent effects. Fivefold compression reduces stored bytes. Selecting 2 of 50 equally sized columns yields a 4% *ideal* projection. Do not multiply those into a guaranteed 125-fold speedup. Column widths and compression differ, Parquet reads row groups and metadata, a filter may read additional columns, and execution has CPU and network costs. The [Parquet concepts](https://parquet.apache.org/docs/concepts/) explain that a row group contains one chunk per column, while the [DuckDB guide](https://duckdb.org/docs/current/guides/performance/file_formats) describes which queries benefit from pushdown.

### Put a contract at the handoff

For a published dataset, record the owner, schema, keys, event-time definition, permitted nulls, update rule and freshness expectation. Consumers need to know whether data is a snapshot or a stream of changes. A column renamed from `amount` to `net_amount` might break a model outright; a change from cents to pounds may silently corrupt predictions. Version the contract and test a sample consumer before rollout. A contract is valuable when the provider and the training pipeline are owned by different teams.

Treat sensitive content as a design constraint from the start. A Parquet export can preserve far more fields than a training job needs. Reduce fields, limit access and define deletion propagation. The ability to compress and scan data cheaply does not imply that it should be retained forever.

## Work through a representation decision

Start with the unit of work. A payment service needs to confirm one transaction quickly and durably. A fraud model needs a short, current history for one account. An analyst wants a year of spending grouped by merchant category. A model trainer wants a reproducible snapshot of events and labels. These are four access patterns, and one layout rarely serves all of them best. Draw the handoffs first: operational store → event export → curated historical table → feature generation. Attach identity, time and schema to every arrow.

Next ask what must be preserved. A mutable customer profile is convenient for serving the current state, but it cannot reproduce what the model knew six months ago unless changes were recorded. An event log can keep that history, although replay and deduplication require explicit rules. If a source sends the same event twice, a stable event ID can make ingestion idempotent. If an event is corrected, define whether the new record supersedes it or is an additional fact. These semantics cannot be recovered from file compression alone.

Choose the exchange format with the reader and writer in mind. CSV is easy to inspect but loses nested structure and has weak type information. JSON can carry nested values and is useful at API boundaries, but optional fields and inconsistent number formats need validation. Avro is designed for schema-aware serialisation. Parquet packages columnar data for analytical reads. A team can legitimately use JSON on the event bus and Parquet for historical analysis, while storing transactional rows in a database. The formats are steps in a lifecycle, not mutually exclusive badges.

Then test with a real workload. If a training job scans twenty million historical events and selects five of eighty fields, compare a raw CSV scan with a Parquet scan under the same filter and hardware. Record bytes read, elapsed time, CPU and memory. If the query joins several large tables repeatedly, test whether loading them into an analytical database pays off. Keep the result local to the workload; a file layout choice is not a universal benchmark. Include schema changes and late-arriving events in the trial, because correctness failures can cost more than scan time.

Finally, trace a model feature back to the originating event. If `prior_30_day_refunds` is wrong, can a reviewer find the exact source version, transformation version, cutoff and entity key used? If not, the data representation has hidden a dependency. Reproducibility needs a chain of evidence, not only an efficient file. This chain also gives operations a way to determine which models and reports must be rebuilt after a source correction.

### Check the boundary with a small sample

Before adopting a layout for a whole corpus, choose ten representative records and follow them through export, storage and readback. Include a missing optional field, a non-ASCII name, a late event, a duplicate event ID and a corrected value. Compare values and types after each step, then ask whether the original source record can still be found. A type that changes silently between JSON, CSV and an analytical table can corrupt a feature even when the file parses. This sample is also a good place to test deletion and access: can a user who should not see a field retrieve it through an old extract or a log? Record expected behaviour as executable checks before increasing scale.

## Where this stands in 2026

:::info Industry view

- Parquet remains a common open columnar format. Its row-group and column-chunk structure supports selecting fields without decoding every field in a record.
- DuckDB can query Parquet directly and use projection and filter pushdown; its own guidance recommends loading data for some repeated or join-heavy workloads.
- ML training increasingly spans tabular events and unstructured documents. Both need stable identity, time semantics and provenance even when their payload formats differ.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why does data management matter for ML?</summary>

Model quality is capped by data quality and availability; most production ML effort is data plumbing (collection, storage, transformation, governance), not modelling.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Explain the DIKW hierarchy.</summary>

Data (raw facts) → Information (organised, in context) → Knowledge (with experience/rules) → Wisdom (applied to decide).<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Contrast row-oriented and columnar storage.</summary>

Row keeps a record's fields together (fast writes/whole-row reads; OLTP); columnar keeps each column together (fast analytical scans + strong compression; OLAP; Parquet/ORC).<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> A 10 GB CSV compresses to 2 GB as Parquet, and a query touches 2 of 50 columns. Give the compression ratio and fraction scanned.</summary>

Compression = 10/2 = 5×; columnar reads only 2/50 = 4% of the data.<br /><em>Session 1 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Contrast structured, semi-structured and unstructured data.</summary>

Structured = tables/relations; semi-structured = self-describing JSON/XML; unstructured = text/images/audio.<br /><em>Session 1 · conceptual</em>

</details>

## Go deeper

- [Apache Parquet concepts](https://parquet.apache.org/docs/concepts/) explains row groups and column chunks.
- [DuckDB Parquet documentation](https://duckdb.org/docs/stable/data/parquet/overview) shows direct file queries and pushdown.
- Built from the course lecture "dm-s1-intro" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can explain why the same event may need different transactional and analytical representations.
- [ ] I can distinguish a data model from a storage or exchange format.
- [ ] I can compute the lecture's fivefold size ratio and four per cent ideal projection without claiming a measured speedup.
- [ ] I can preserve entity identity, time meaning and a schema contract across the ML data path.
