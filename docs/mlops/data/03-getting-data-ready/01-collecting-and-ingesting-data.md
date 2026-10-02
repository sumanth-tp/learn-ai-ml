---
id: dm-collecting-and-ingesting-data
title: "Data Management · Session 7 — Collecting and Ingesting Data"
sidebar_label: "7 · Ingestion"
sidebar_position: 1
slug: /mlops/data/collection-and-ingestion
description: "Collect files, API results, database changes and streams while preserving identity, time and a representative sample."
tags: [data-management, data-ingestion, change-data-capture, sampling]
---

import Infographic from '@site/src/components/Infographic';
import SamplingFractionLab from '@site/src/components/viz/SamplingFractionLab';

**In one line.** Ingestion must carry source changes into a durable, identifiable and replayable form.

## The idea in plain words

A model team often receives data from several places: a transactional database, an API, a file drop and an event stream. Those sources differ in rate, schema, ownership and what a record means. Ingestion is the boundary where the team decides which changes to collect, how to identify them and how to prove that nothing essential was lost or counted twice.

The lecture names three modes. **Batch** reads a bounded set on a schedule. **Streaming** handles an ongoing flow of events. **Change Data Capture** (CDC) follows database changes, often through a transaction or replication log, and publishes inserts, updates and deletes. A source may **push** events to the platform, while a collector may **pull** from an API or table. These terms describe how data arrives, not whether its contents are valid. Every mode needs a schema, source position or version, error handling and a consumer-visible freshness measure.

<Infographic src="/img/dm/ingestion-modes.svg" alt="Batch, streaming and database change capture observe source data in different ways; sampling five thousand of two million rows is 0.25 per cent." caption="Choose an ingestion mode by change semantics and recovery needs, then measure what was actually captured." />

When a source is huge, the lecture proposes sampling. Taking 5,000 rows from 2,000,000 is a fraction of **0.0025**, or **0.25%**. The arithmetic does not establish representativeness. The first 5,000 rows of a time-sorted table might all be old, while a random sample could miss a rare fraud class. A sample design needs a target population, a selection mechanism and a check that important groups are covered.

:::note Beyond the lecture

The source introduces modes, push/pull, connectors and reservoir sampling. The sections below add deletion handling, snapshot-to-stream handoff, backpressure, sampling bias and source-to-landing reconciliation.

:::

The default lab reproduces the lecture's **5,000 / 2,000,000 = 0.0025 = 0.25%**. Move the source and sample sizes to see how the fraction changes. A low fraction can be sufficient for common patterns but still inadequate for a rare label or small subgroup.

<SamplingFractionLab />

## How it works

### Batch, stream & CDC

- **Batch**; Periodic bulk pulls.
- **Streaming**; Continuous events, low latency.
- **CDC**; Read the DB change log to replicate changes.

### Sampling large sources

Push vs pull; managed connectors (Kafka Connect, Fivetran, Airbyte); raw landing zone. When a source is huge, sample it.

:::tip

**Worked.** 5,000 of 2,000,000 rows → fraction 0.0025 (0.25%). Reservoir sampling for streams.

:::


## A real system that works this way

**Debezium** is a concrete CDC system. Its documentation describes connectors that observe row-level database changes and emit a stream. A change event can include operation type, source metadata and before/after row state. The PostgreSQL connector, for example, uses logical replication; the MySQL connector reads the binary log. This is more informative than periodically selecting the current table because an update or deletion can be represented as a change rather than silently replacing state.

Suppose an account table stores a current address. A nightly full-table pull can copy today's state but cannot reconstruct when the address changed unless the source retains history. CDC can emit the update event, letting a downstream system maintain a historical dimension. It must also establish an initial snapshot, remember the log position from which changes continue and recover if the log is no longer retained. Duplicate delivery after a retry is possible, so consumers need a stable key and position or version to avoid double application.

A managed connector can reduce integration effort, but the team still owns the data contract. Does a deleted customer row mean that historical reports must be erased, or only that the active account is gone? Which sensitive fields may leave the transactional store? What happens when the source adds a column? Connector health, source lag and the age of the last accepted event should be monitored separately from downstream transformation success.

## Code you can run

The lecture's sampling fraction is independent of the selection algorithm. Compute it first and label the units clearly.

```python
source_rows = 2_000_000
sample_rows = 5_000
fraction = sample_rows / source_rows
print(f"fraction={fraction:.4f}, percentage={fraction:.2%}")
assert fraction == 0.0025
```

Reservoir sampling gives each item in a one-pass stream the same inclusion probability when the stream length was not known in advance. This seeded example uses a small population so the selected IDs can be inspected. The equal-probability statement concerns the algorithm over possible random draws, not a guarantee that one realised sample balances every subgroup.

```python
import random

def reservoir(stream, size, seed=7):
    rng = random.Random(seed)
    chosen = []
    for index, item in enumerate(stream):
        if index < size:
            chosen.append(item)
            continue
        slot = rng.randrange(index + 1)
        if slot < size:
            chosen[slot] = item
    return chosen

sample = reservoir(range(20), 5)
print(sorted(sample))
assert len(sample) == 5
assert len(set(sample)) == 5
assert all(0 <= item < 20 for item in sample)
assert reservoir(range(20), 5) == sample
```

For a very large source, record the seed, stream ordering, inclusion rule and any stratification so that a reviewer can reproduce the sampled dataset. A true uniform stream sample may still be the wrong training sample if the model must learn a rare outcome.

## Designing with it

### Identify the source state and the change

A batch file should carry a source version, extract time, row count, checksum and schema. A CDC event should carry source table, key, operation, source position and change time. A pushed event should have an event ID that is stable across retries. These fields let a downstream task distinguish a new record from a duplicate delivery or a correction. If a source cannot supply stable IDs, document the surrogate key and its collision risk.

An initial snapshot plus CDC stream needs a handoff point. If the snapshot runs while new transactions commit, a naive start of the stream can miss changes or apply them twice. Use the connector's documented snapshot and log-position mechanism, then reconcile counts or sample IDs. Keep log retention long enough for planned outages and recovery. A connector that reports "running" while its source lag grows is not meeting a low-latency contract.

### Decide between push and pull

Pulling an API gives the collector control over schedules and retries but must respect rate limits, pagination, cursor expiry and partial responses. Pushing events reduces polling but requires the receiver to absorb bursts and report failures so the producer can retry safely. A daily file drop is simple when freshness needs are measured in hours. A stream is useful for minute-scale decisions but introduces ordering and state concerns. CDC is attractive when the database is the source of truth and the downstream needs every committed change, including deletes.

Land a raw copy or a durable change log within retention and privacy limits. This creates a recovery point if a parser or business rule is corrected. Do not assume raw means ungoverned: access to source payloads may be more sensitive than access to curated features. Masking or minimisation may need to occur before landing.

### Sample for a purpose

Uniform row sampling estimates common properties when every row has a similar chance of selection. It may fail for rare events, grouped entities or temporal change. Stratify by a rare label or source when the analysis needs those subgroups, and keep weights if population estimates will be reconstructed. For a time-series model, sample complete entity histories or contiguous windows rather than independent rows that break order. A development sample may use a different strategy from the final training set; label it accordingly.

Reservoir sampling is useful for an unknown-length stream because it uses fixed memory and one pass. It does not solve class imbalance or create an unbiased label when only some events are reviewed by humans. Record the sampling frame and missing population. The right question is not merely "how many rows?" but "which decisions will this sample support, and which cases could it omit?

## Trace an account update into a feature table

An account service changes a customer's address at 11:03. The database commits the transaction, and the CDC connector emits an update with a key, operation, source time and log position. The landing system stores that change without discarding the previous state. A downstream task converts it into a Type 2 address history; a training join can then find the address that was valid when a historical decision happened. A current-state batch pull would only show the new address and could leak it backward into old examples.

Now suppose the connector disconnects for two hours. The source log must still contain those changes when it reconnects. Monitor the oldest unread position and the remaining log-retention margin, not only the connector process status. On restart, some events may be replayed. The history table should reject a duplicate source position or version without losing a genuinely corrected event. A delete must be modelled explicitly so the downstream current-state table does not keep a customer who no longer exists.

Reconcile at more than one level. Compare a source snapshot's key count with the landed current-state count after the stream catches up. Compare counts of inserts, updates and deletes over a window. Investigate a sample of source keys end to end, including one that changed twice quickly. Counts can match even when the wrong records were copied, so identity-based checks add evidence. Record the time at which both sides were compared to avoid calling normal lag a discrepancy.

If the team samples events for model exploration, decide whether to sample before or after deduplication. Sampling deliveries before deduplication gives frequently retried events a higher inclusion chance. Sampling after deduplication gives each logical event one chance. If the source is ordered by account, a first-N sample is not uniform. Use a seeded method or a documented stratified design, then check label rates and time coverage against the full population.

Finally, define consumer freshness from source commit to approved destination, not merely connector receipt. A fast CDC stream followed by a stalled validation task still leaves features stale. Report source lag, landing lag and publication lag separately, with an owner for each stage. That separation makes an incident diagnosable rather than a vague "data is late" alert.

### Check a pull-based API boundary

An API may paginate results with a cursor and impose a rate limit. Persist the cursor only after the page has been durably landed; otherwise a crash can skip a page. If the API may return the same item on overlapping pages or on a retry, deduplicate by its stable ID and version. A time-based cursor needs a lookback window for late updates. Re-reading a short window and applying idempotent writes is often safer than assuming every source timestamp is perfectly ordered.

Test the source's deletion semantics. Some APIs return only active records, so a missing item could mean deletion, a permission change or a partial response. A full snapshot comparison can identify disappeared keys, but should not turn a transient pagination failure into a mass delete. CDC may emit a delete event, yet the downstream retention and privacy policy still determines whether the historical record remains. Record the action and its justification.

### Audit the sample frame

Before using a sample for training, compare its date range, source mix, entity counts and outcome rates with the full eligible population. If labels are delayed, compare only records old enough to have matured; otherwise recent examples look disproportionately negative. If sampling was stratified to include a rare class, raw sample proportions no longer estimate population prevalence. Keep inclusion probabilities or weights when the analysis needs population estimates. A development sample can prioritise rare failure modes for debugging, while final model evaluation must represent the deployment target.

Reservoir sampling assumes a sequence in which each logical item appears once. If a stream carries updates or duplicate deliveries, decide whether the population is event versions, distinct events or current entities before applying the algorithm. A reservoir of deliveries can overrepresent noisy sources. Sampling after identity resolution changes the memory and latency requirements, so make that trade-off explicit. These details are why a correct 0.25% arithmetic result is only the start of sampling design. Document the sample's refresh schedule too: a fixed exploration sample is reproducible, while a rolling sample may track a changing population. Neither should quietly replace the final held-out evaluation set. When the sample is used to estimate a metric, report uncertainty and the effective sample size for the population of interest.

## Where this stands in 2026

:::info Industry view

- Debezium documents row-level change streams with operation and source metadata, supporting historical reconstruction when the source table itself is mutable.
- Batch, streaming and CDC are often combined: a snapshot establishes state, a stream carries changes and periodic reconciliation checks the result.
- Sampling remains a design decision even with cheap compute; a percentage alone does not show coverage of rare or time-dependent cases.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Name four data sources and three ingestion modes.</summary>

Sources: databases, APIs, files, event streams. Modes: batch, streaming, and Change Data Capture (CDC).<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is Change Data Capture?</summary>

Reading a database's change log to replicate inserts/updates/deletes with low latency and without heavy queries on the source.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Contrast push and pull ingestion.</summary>

Pull = the platform polls the source; push = the source sends events to the platform. Managed connectors standardise both.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Draw 5,000 rows from a 2,000,000-row table. Give the sampling fraction.</summary>

5,000/2,000,000 = 0.0025 (0.25%).<br /><em>Session 7 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What is reservoir sampling used for?</summary>

Drawing a fixed-size uniform sample from a stream of unknown length in a single pass (each item retained with equal probability).<br /><em>Session 7 · conceptual</em>

</details>

## Go deeper

- [Debezium documentation](https://debezium.io/documentation/reference/stable/index.html) introduces row-level CDC.
- [Debezium change-event structure](https://debezium.io/documentation/reference/stable/transformations/event-flattening.html) shows operation and before/after fields.
- Built from the course lecture "dm-s7-collection-ingestion" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can distinguish batch, streaming, CDC, push and pull ingestion.
- [ ] I can compute 5,000/2,000,000 and explain why 0.25% alone is not a representativeness claim.
- [ ] I can carry stable identity and source position through a retry or backfill.
- [ ] I can reconcile an initial snapshot with later changes and name the lag that consumers feel.
