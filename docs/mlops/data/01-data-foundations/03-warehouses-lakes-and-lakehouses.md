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

**In one line.** Choose storage and processing paths from the data's consumers, then make refinement and ownership explicit.

## The idea in plain words

An operational service, an analyst and a model trainer can all use the same source events while asking different questions. The service needs small, reliable reads and writes. The analyst wants to scan years of history and group by business dimensions. The trainer needs reproducible snapshots with labels and feature cutoffs. A data architecture joins these workloads without letting one consumer silently change another's meaning.

The lecture contrasts a **warehouse**, a **lake** and a **lakehouse**. A warehouse commonly curates structured tables for SQL analytics. A lake stores varied raw or lightly processed files for flexible later use. A lakehouse applies table management such as snapshots, schema control and transactions to files in a lake. These descriptions are useful starting points, not strict product categories. Warehouses can ingest JSON, lakes can enforce schemas and a lakehouse still needs governance. The choice is about the concrete guarantees the workload requires.

<Infographic src="/img/dm/storage-architectures.svg" alt="A warehouse holds curated analytical tables, a lake holds varied files and a lakehouse adds table metadata; bronze, silver and gold show increasing refinement." caption="Architecture choices and refinement layers answer different questions: where data lives and how its meaning improves." />

The lecture also separates **OLTP** from **OLAP**. Online transaction processing handles many short operations that change current state, such as recording a payment. Online analytical processing scans and aggregates larger collections, often with historical context. Row-oriented and column-oriented layouts often align with these patterns, but layout alone does not define either workload. The operational source is where events happen; analytics and ML generally consume a versioned, transformed view of them.

:::note Beyond the lecture

The lecture gives the three architecture labels, star schema, medallion layers and Lambda/Kappa patterns. This chapter adds decision boundaries, snapshot semantics, failure handling and an explicit distinction between file format and table-format guarantees.

:::

The board above separates the storage choice from the refinement path. The code below works through a small fact table and a validated event stream.

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

A tiny SQLite star schema makes the lecture's fact-and-dimension idea concrete. The fact table records events and amounts; dimensions attach customer and category descriptions. The query groups by a dimension without repeating that text in every fact row.

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

## Designing with it

### Distinguish storage from table guarantees

A lake may store Parquet objects cheaply and flexibly, but those objects do not by themselves provide one atomic multi-file table update. A table format such as Iceberg adds metadata and commit rules that identify the active files in a snapshot. A warehouse often provides transactional tables and managed SQL access. Evaluate the actual system's guarantees: isolation during updates, delete propagation, schema evolution, access control, catalogue discovery and recovery. Saying "lakehouse" without specifying these guarantees is not an architecture decision.

The lecture's schema-on-write versus schema-on-read distinction is a tendency, not a universal law. A curated warehouse table normally validates fields before publication. A raw lake zone may retain payloads before assigning full meaning. Both can contain layers with stronger or weaker controls. Schema-on-read does not mean schema-free: every successful query still interprets bytes through a schema, even if that schema is applied later. Delay can be useful for exploratory data, but postponing validation can move surprise and cost to consumers.

### Model facts and dimensions around questions

A star schema centres a **fact** table of events or measurements, connected to **dimensions** that describe who, what, where and when. State the grain first: one row per payment event, not one row per customer or one row per day. A dimension table can then supply customer region or product category at a defined point in time. If a customer's region changes, decide whether historical reports use the old region or the current region. That is a business question encoded in slowly changing dimension handling, which Chapter 9 develops.

A star schema is effective for repeated analytical questions, but it is not automatically the right shape for every training job. A model may need event sequence, raw text and point-in-time joins. Keep a trace from a derived feature to the fact rows and dimension versions that produced it. A daily aggregate cannot recreate individual event order after it has discarded that detail.

### Make the layers and batch/stream paths explicit

Bronze, silver and gold can separate retention, validation and consumption. Define the contract at each transition: accepted schema, key, deduplication rule, watermark, output freshness and handling of invalid rows. A direct source-to-gold shortcut may be fine for a small, trusted dataset, but it removes a recovery point if raw history is not retained elsewhere. Keep only the raw data needed for replay and audit within privacy and retention limits.

The lecture mentions **Lambda** and **Kappa** architectures. Lambda combines a batch recomputation path with a low-latency stream path; this can provide fresh results and historical correction, but the two implementations may disagree. Kappa processes through a replayable stream path and uses the log to reprocess; that reduces duplicated logic but requires enough retained events and suitable replay semantics. Neither label guarantees exactly-once business outcomes. Stable event IDs, idempotent writes and explicit late-arrival policy still matter. Choose the simplest path that meets latency and correction needs.

## Trace a payment across the architecture

A customer makes a payment at 09:00. The transaction system commits a new payment row and responds to the customer. It cannot wait for a warehouse refresh or a feature backfill. A change-data-capture or event export process later publishes the payment with an event ID, source commit time and payload version. The pipeline records it in a raw zone, even if one downstream parser cannot yet recognise a new optional field. That preserves an explanation for what was received.

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

## Go deeper

- [Databricks medallion guidance](https://docs.databricks.com/aws/en/lakehouse/medallion) explains raw, validated and curated layers.
- [Apache Iceberg API](https://iceberg.apache.org/docs/latest/api/) describes table metadata and transactions.
- [Apache Iceberg snapshots](https://iceberg.apache.org/docs/latest/branching/) describes snapshot history and retention.
- Built from the course lecture "dm-s3-architectures" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can select a warehouse, lake or lakehouse from the consumer's access and governance needs.
- [ ] I can explain OLTP, OLAP, a fact table's grain and a dimension's historical meaning.
- [ ] I can trace a record from raw retention through validation to a curated product.
- [ ] I can distinguish snapshot consistency from a leakage-safe point-in-time query.
