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


:::tip Before you start

**You should already know**

- What a primary key is and what an upsert does: [building reliable data pipelines](/docs/mlops/data/reliable-pipelines).
- The difference between when something happened and when you learned of it.
- Basic pandas or NumPy boolean masks.

**Reading time.** About 50 minutes, plus a few seconds to run the experiments.

**After this chapter you can**

- compare full reload, arrival-time incremental and event-time incremental ingestion by rows read and events lost,
- choose a lookback window for late data and say what it costs,
- explain why a sample of 5,000 from 2,000,000 can contain very few rare cases.

:::

## In 30 seconds

You run a corner shop and a courier brings parcels all day, some of them late. You could re-read the whole delivery book each night (safe, slow), or only the lines added since yesterday (fast, but a late parcel entered under an older date is easy to miss). Ingestion is that choice, plus a way to keep a manageable sample when the book has millions of lines. The trick is to track not the date on a line but the moment it arrived, or to re-read a few days back to catch strays.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Batch | Read a bounded set on a schedule | A nightly file |
| Streaming | Handle events as they arrive | Click events |
| CDC | Capture each change from a database log | An address update event |
| Watermark | The position or time up to which data has been loaded | Loaded events up to 2026-03-04 |
| Event time | When the thing happened | 23:58 on 4 March |
| Arrival time | When the system received it | 01:10 on 5 March |
| Lookback window | Re-reading a recent period to catch late data | Re-read the last 24 hours |
| Reservoir sampling | A one-pass uniform sample of fixed size | 5,000 of an endless stream |

## The idea in plain words

A model team often receives data from several places: a transactional database, an API, a file drop and an event stream. Those sources differ in rate, schema, ownership and what a record means. Ingestion is the boundary where the team decides which changes to collect, how to identify them and how to prove that nothing essential was lost or counted twice.

Three modes are common. **Batch** reads a bounded set on a schedule. **Streaming** handles an ongoing flow of events. **Change Data Capture** (CDC) follows database changes, often through a transaction or replication log, and publishes inserts, updates and deletes. A source may **push** events to the platform, while a collector may **pull** from an API or table. These terms describe how data arrives, not whether its contents are valid. Every mode needs a schema, source position or version, error handling and a consumer-visible freshness measure.

<Infographic src="/img/dm/ingestion-modes.svg" alt="Batch, streaming and database change capture observe source data in different ways; sampling five thousand of two million rows is 0.25 per cent." caption="Choose an ingestion mode by change semantics and recovery needs, then measure what was actually captured." />

When a source is huge, sampling is one answer. Taking 5,000 rows from 2,000,000 is a fraction of **0.0025**, or **0.25%**. The arithmetic does not establish representativeness. The first 5,000 rows of a time-sorted table might all be old, while a random sample could miss a rare fraud class. A sample design needs a target population, a selection mechanism and a check that important groups are covered.

:::note Added for this site

The course introduces modes, push/pull, connectors and reservoir sampling. The sections below add deletion handling, snapshot-to-stream handoff, backpressure, sampling bias, source-to-landing reconciliation and a measured comparison of ingestion strategies on late data.

:::

The default lab reproduces **5,000 / 2,000,000 = 0.0025 = 0.25%**. Move the source and sample sizes to see how the fraction changes. A low fraction can be sufficient for common patterns but still inadequate for a rare label or small subgroup.

<SamplingFractionLab />


**What each control does.**

- **source rows (thousands)** sets the size of the source, 100 thousand to 2 million.
- **sample rows** sets how many rows are drawn, 1,000 to 10,000.

**Try it yourself.**

1. Leave the defaults: 5,000 of 2,000,000 gives 0.0025, 0.25%.
2. Set sample rows to 10,000. The fraction doubles to 0.0050, but the expected number of rare cases at a 0.3% rate is only 30, so the estimate is still noisy.
3. Set the source to 100 thousand rows with 5,000 sampled. The fraction is 5.00%. Compare that with the 0.25% case: a sample ten times larger as a share still depends on its design, not just its size.

## Worked example, step by step

A source produces 20,000 events a day for 14 days. About 3% of events arrive more than an hour late, some up to two days.

1. Late events per day: 20,000 × 0.03 = 600, so the experiment's 570 to 635 range is what we expect.
2. A full reload reads every visible event on every run. Over 16 nightly runs the table grows from 20,000 towards 280,000, so the total read is roughly 16 × half of 280,000, which is about 2.2 million rows, the order of the 2,652,720 printed below.
3. An arrival-time watermark reads only events that arrived since the last run, so each event is read once: 280,000 rows.
4. An event-time watermark reads events newer than the latest event time already loaded. A late event has an older event time than the watermark, so it is never read. If 2.4% of events arrive after a later one has already been loaded, about 2.4% are lost.
5. A lookback of L hours re-reads about 20,000 × L / 24 rows per run. For 6 hours that is about 5,000 extra rows per run, 80,000 over 16 runs.
6. Sampling: 5,000 / 2,000,000 = 0.0025 = 0.25%. If 0.3% of rows are rare cases, a sample holds 5,000 × 0.003 = 15 of them on average, with a standard deviation of √(5,000 × 0.003 × 0.997) = 3.9, so a 95% range of roughly 15 ± 7.6.

In words: arrival-time tracking is exact and cheap, event-time tracking is lossy for late data, and a lookback buys back what it re-reads. A small uniform sample holds only a handful of rare cases. The experiments below measure both.

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

The sampling fraction is independent of the selection algorithm. Compute it first and label the units clearly.

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

### Experiment: late data and incremental loads

The simulation creates 14 days of events at 20,000 a day. 97% arrive within a couple of minutes on average, and 3% arrive between one hour and two days late. A job runs each night at half past midnight and loads from the source according to one of six strategies. The metrics are rows read, the share of events eventually loaded, and how short the day's count was at its first publication. Everything is synthetic and seeded; the run uses NumPy 2.5.3.

```python
import numpy as np

rng = np.random.default_rng(11)
DAYS, PER_DAY, DAY = 14, 20_000, 86_400
n = DAYS * PER_DAY
event = np.sort(rng.uniform(0, DAYS * DAY, n))
delay = np.where(rng.random(n) < 0.97, rng.exponential(120, n), rng.uniform(3_600, 2 * DAY, n))
arrival = event + delay
day_of = (event // DAY).astype(int)
truth = np.bincount(day_of, minlength=DAYS)
print(f"{n:,} events over {DAYS} days; {(delay > 3_600).mean():.1%} arrive more than an hour late")

def run(strategy, lookback_hours=0.0):
    loaded = np.zeros(n, bool)
    last_run, watermark, rows_read, first_view = 0.0, 0.0, 0, []
    for k in range(1, DAYS + 3):
        now = k * DAY + 1_800
        visible = arrival <= now
        if strategy == "full":
            batch = visible
        elif strategy == "arrival":
            batch = visible & (arrival > last_run)
        else:
            batch = visible & (event > watermark - lookback_hours * 3_600)
        rows_read += int(batch.sum())
        loaded |= batch
        watermark = max(watermark, event[loaded].max())
        last_run = now
        if k <= DAYS:
            first_view.append(np.sum(loaded & (day_of == k - 1)))
    return rows_read, loaded.mean(), 1 - np.mean(np.array(first_view) / truth)

print(f"{'strategy':28s} {'rows read':>10s} {'events kept':>11s} {'day short at first publish':>27s}")
plans = [("full reload", "full", 0), ("arrival-time watermark", "arrival", 0), ("event-time watermark", "event", 0),
         ("event-time, 6 h lookback", "event", 6), ("event-time, 24 h lookback", "event", 24), ("event-time, 72 h lookback", "event", 72)]
for label, strategy, hours in plans:
    rows_read, kept, short = run(strategy, hours)
    print(f"{label:28s} {rows_read:10,d} {kept:11.2%} {short:27.2%}")
```

**Reading the output.** `rows read` counts every row the job pulled from the source across all runs. `events kept` is the share of all events that ended up loaded. `day short at first publish` is how far below the true event-day count the first published number was.

**Line by line.**

- `visible = arrival <= now` is what the source can show at run time. Events that have not arrived yet cannot be read by any strategy.
- `batch = visible & (event > watermark - lookback_hours * 3_600)` is the event-time rule. With a lookback of 0 it reads only events newer than the newest loaded.
- `loaded |= batch` makes loading idempotent: re-reading an event does not duplicate it, which is what a keyed upsert gives in a real table.

The printed output:

```text
280,000 events over 14 days; 3.0% arrive more than an hour late
strategy                      rows read events kept  day short at first publish
full reload                   2,652,720     100.00%                       2.29%
arrival-time watermark          280,000     100.00%                       2.29%
event-time watermark            273,226      97.58%                       2.42%
event-time, 6 h lookback        347,549      98.11%                       2.29%
event-time, 24 h lookback       571,219      99.30%                       2.29%
event-time, 72 h lookback     1,111,705     100.00%                       2.29%
```

### Reading the ingestion experiment

Full reload and the arrival-time watermark both end with 100.00% of events, but the reload read 2,652,720 rows against 280,000, 9.5 times more. The event-time watermark read nearly the same as the arrival-time one (273,226) and lost 2.42% of events for ever, because a late event is older than the watermark. That is the failure to guard against: the job reports success and a daily aggregate is quietly wrong.

A lookback recovers the late events at a price. Six hours kept 98.11%, 24 hours 99.30% and 72 hours, which covers the longest delay in this data, kept 100.00%, reading 1,111,705 rows, about four times the minimum.

The surprise is the last column. No strategy beat 2.29% for the first published number, because those events had not yet reached the source. Cleverness in ingestion cannot publish what has not arrived. The honest design is to publish early, mark the recent days provisional and restate them when the lookback window closes. Limits: a delay distribution I chose, one seed, whole-day aggregates and an idealised source that supports every query.

### Experiment: how good is a 0.25% sample

This second block draws samples of 5,000 from a synthetic population of 2,000,000 with a rare label at 0.3% and a field that drifts with time. It compares a uniform random sample, a stratified sample and the first 5,000 rows.

```python
import numpy as np

rng = np.random.default_rng(5)
population, sample_size, rare_rate = 2_000_000, 5_000, 0.003
rare = rng.random(population) < rare_rate
trend = np.linspace(0, 1, population) + 0.05 * rng.normal(size=population)
print(f"population {population:,}, rare label rate {rare.mean():.4f}, sample {sample_size:,} = {sample_size / population:.2%}")

counts, estimates = [], []
for _ in range(300):
    pick = rng.choice(population, sample_size, replace=False)
    counts.append(int(rare[pick].sum()))
    estimates.append(rare[pick].mean())
low, high = np.percentile(estimates, [2.5, 97.5])
print(f"uniform sample: rare rows per draw mean {np.mean(counts):.1f}, min {min(counts)}, max {max(counts)}; draws with fewer than 10: {np.mean(np.array(counts) < 10):.0%}")
print(f"rate estimate 95% of draws between {low:.4f} and {high:.4f} (truth {rare.mean():.4f})")

positives, negatives = np.flatnonzero(rare), np.flatnonzero(~rare)
strata = np.r_[rng.choice(positives, 500, replace=False), rng.choice(negatives, sample_size - 500, replace=False)]
weights = np.where(rare[strata], len(positives) / 500, len(negatives) / (sample_size - 500))
print(f"stratified: rare rows {int(rare[strata].sum())}, raw rate {rare[strata].mean():.3f}, weighted rate {np.average(rare[strata], weights=weights):.4f}")

first_n = trend[:sample_size].mean()
uniform = trend[rng.choice(population, sample_size, replace=False)].mean()
print(f"mean of a drifting field: whole population {trend.mean():.3f}, first {sample_size:,} rows {first_n:.3f}, uniform sample {uniform:.3f}")
```

**Reading the output.** The population line states the rare rate. The uniform line gives, over 300 draws, how many rare rows each sample held. The stratified line reports the raw and the weighted rate. The last line shows the first 5,000 rows of a time-ordered table against a uniform sample.

**Line by line.**

- `rng.choice(population, sample_size, replace=False)` is a uniform draw without replacement.
- `weights = len(positives) / 500` and `len(negatives) / (sample_size - 500)` are inclusion weights: each sampled row stands for that many population rows, which is how a stratified sample still estimates prevalence.

The printed output:

```text
population 2,000,000, rare label rate 0.0030, sample 5,000 = 0.25%
uniform sample: rare rows per draw mean 15.0, min 3, max 25; draws with fewer than 10: 9%
rate estimate 95% of draws between 0.0016 and 0.0046 (truth 0.0030)
stratified: rare rows 500, raw rate 0.100, weighted rate 0.0030
mean of a drifting field: whole population 0.500, first 5,000 rows 0.003, uniform sample 0.507
```

### Reading the sampling experiment

A uniform sample held 15.0 rare rows on average, as the hand arithmetic predicted, but the number ranged from 3 to 25 across draws, and 9% of samples held fewer than 10. A 95% range of 0.0016 to 0.0046 around the true 0.0030 means the estimated rate can be half or one and a half times the real one. Compare the 0.25% fraction with 15 cases, not with the 2,000,000 rows.

A stratified sample guaranteed 500 rare rows and, with weights, recovered the rate 0.0030, while its raw rate of 0.100 would be a gross overstatement of prevalence. The first 5,000 rows of a time-ordered table had a mean of 0.003 against a population mean of 0.500, because they came from the beginning of the drift. Limits: the rare rate is constant, there are no duplicates, and the drift is a simple ramp.

<Infographic src="/img/dm-enrich/late-data-ingestion.svg" alt="Bars show events kept and rows read for six ingestion strategies on late data, with cards for the unavoidable 2.29 per cent first-publication gap and the spread of rare rows in a 0.25 per cent sample." caption="Look first at the event-time watermark row: it reads almost as few rows as the arrival-time one but loses 2.42% of events." />

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

An account service changes a customer's address late one morning. The database commits the transaction, and the CDC connector emits an update with a key, operation, source time and log position. The landing system stores that change without discarding the previous state. A downstream task converts it into a Type 2 address history; a training join can then find the address that was valid when a historical decision happened. A current-state batch pull would only show the new address and could leak it backward into old examples.

Now suppose the connector disconnects for two hours. The source log must still contain those changes when it reconnects. Monitor the oldest unread position and the remaining log-retention margin, not only the connector process status. On restart, some events may be replayed. The history table should reject a duplicate source position or version without losing a genuinely corrected event. A delete must be modelled explicitly so the downstream current-state table does not keep a customer who no longer exists.

Reconcile at more than one level. Compare a source snapshot's key count with the landed current-state count after the stream catches up. Compare counts of inserts, updates and deletes over a window. Investigate a sample of source keys end to end, including one that changed twice quickly. Counts can match even when the wrong records were copied, so identity-based checks add evidence. Record the time at which both sides were compared to avoid calling normal lag a discrepancy.

If the team samples events for model exploration, decide whether to sample before or after deduplication. Sampling deliveries before deduplication gives frequently retried events a higher inclusion chance. Sampling after deduplication gives each logical event one chance. If the source is ordered by account, a first-N sample is not uniform. Use a seeded method or a documented stratified design, then check label rates and time coverage against the full population.

Finally, define consumer freshness from source commit to approved destination, not merely connector receipt. A fast CDC stream followed by a stalled validation task still leaves features stale. Report source lag, landing lag and publication lag separately, with an owner for each stage. That separation makes an incident diagnosable rather than a vague "data is late" alert.

### Check a pull-based API boundary

An API may paginate results with a cursor and impose a rate limit. Persist the cursor only after the page has been durably landed; otherwise a crash can skip a page. If the API may return the same item on overlapping pages or on a retry, deduplicate by its stable ID and version. A time-based cursor needs a lookback window for late updates. Re-reading a short window and applying idempotent writes is often safer than assuming every source timestamp is perfectly ordered.

Test how the source handles deletion. Some APIs return only active records, so a missing item could mean deletion, a permission change or a partial response. A full snapshot comparison can identify disappeared keys, but should not turn a transient pagination failure into a mass delete. CDC may emit a delete event, yet the downstream retention and privacy policy still determines whether the historical record remains. Record the action and its justification.

### Audit the sample frame

Before using a sample for training, compare its date range, source mix, entity counts and outcome rates with the full eligible population. If labels are delayed, compare only records old enough to have matured; otherwise recent examples look disproportionately negative. If sampling was stratified to include a rare class, raw sample proportions no longer estimate population prevalence. Keep inclusion probabilities or weights when the analysis needs population estimates. A development sample can prioritise rare failure modes for debugging, while final model evaluation must represent the deployment target.

Reservoir sampling assumes a sequence in which each logical item appears once. If a stream carries updates or duplicate deliveries, decide whether the population is event versions, distinct events or current entities before applying the algorithm. A reservoir of deliveries can overrepresent noisy sources. Sampling after identity resolution changes the memory and latency requirements, so make that trade-off explicit. These details are why a correct 0.25% arithmetic result is only the start of sampling design. Document the sample's refresh schedule too: a fixed exploration sample is reproducible, while a rolling sample may track a changing population. Neither should quietly replace the final held-out evaluation set. When the sample is used to estimate a metric, report uncertainty and the effective sample size for the population of interest.

## Where this stands in 2026

:::info Industry view

- Debezium documents row-level change streams with operation and source metadata, supporting historical reconstruction when the source table itself is mutable.
- Batch, streaming and CDC are often combined: a snapshot establishes state, a stream carries changes and periodic reconciliation checks the result.
- Sampling remains a design decision even with cheap compute; a percentage alone does not show coverage of rare or time-dependent cases.

:::

## Common mistakes

1. **Loading by event time alone.** It feels natural, since event time is the business time. It lost 2.42% of events with no error. Track arrival time or an update timestamp, or add a lookback.
2. **Choosing a lookback by guess.** 6 hours kept 98.11%, 24 hours 99.30%. Measure the delay distribution and set the window to its long tail, then budget the extra reads.
3. **Treating a first publication as final.** Every strategy was 2.29% short on the first day count. Mark recent days provisional and restate them.
4. **Judging a sample by its percentage.** 0.25% sounds small but is fine for common patterns. For a 0.3% rare label it held 15 cases on average and sometimes 3.
5. **Sampling the first N rows.** In a time-ordered table they are the oldest data. The mean was 0.003 against 0.500. Use a seeded uniform or stratified draw.

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

<details>
<summary><strong>Q6. (Medium)</strong> A daily job loads events with event_time greater than the maximum event_time already loaded. 3% of events arrive late. What goes wrong and what are two fixes?</summary>

A late event has an event time older than the watermark, so the job never selects it, and it is lost without any error (2.42% in the experiment). Fixes: filter on arrival time or an update timestamp, which every event gets once; or re-read a lookback window and upsert on the event ID so re-reads do not duplicate.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> You sample 5,000 of 2,000,000 rows and 0.3% are rare. Roughly how many rare rows do you expect, and why is a stratified sample better for a classifier of the rare class?</summary>

Expected: 5,000 × 0.003 = 15, with a standard deviation near 3.9, so as few as 3 in some draws. A stratified sample fixes the number of rare rows (500 here), which gives the model something to learn from, and inclusion weights restore the true prevalence for estimates (0.0030).

</details>

## Go deeper

- [Debezium documentation](https://debezium.io/documentation/reference/stable/index.html) introduces row-level CDC.
- [Debezium change-event structure](https://debezium.io/documentation/reference/stable/transformations/event-flattening.html) shows operation and before/after fields.
- Library versions run for the experiments: NumPy 2.5.3, Python 3.14.6. The data is generated, so no licence applies.
- Built from the course lecture "dm-s7-collection-ingestion" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can distinguish batch, streaming, CDC, push and pull ingestion.
- [ ] I can compute 5,000/2,000,000 and explain why 0.25% alone is not a representativeness claim.
- [ ] I can carry stable identity and source position through a retry or backfill.
- [ ] I can reconcile an initial snapshot with later changes and name the lag that consumers feel.
- [ ] I can say why event-time incremental loading drops late data and name two fixes.
- [ ] I can estimate the extra rows a lookback window reads.
- [ ] I can explain why 0.25% of a source may hold very few rare cases, and when to stratify.

## Where to go next

Next is [profiling, validation and drift](/docs/mlops/data/profiling-validation-drift). For history tracking of changing records, see [analytics engineering and history](/docs/mlops/data/analytics-engineering-history).
