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


:::tip Before you start

**You should already know**

- What a table with rows and columns is, and how to read a CSV file with pandas.
- Basic Python: lists, dictionaries and functions.
- This is the first chapter of the data management series. The next one, [data quality rules](/docs/mlops/data/quality-rules), builds on the vocabulary here.

**Reading time.** About 40 minutes, plus about 30 seconds to run the experiment. It writes roughly 1.4 GB of temporary files, so check your free disk space first.

**After this chapter you can**

- explain why one analytical question reads far less data from a columnar file than from a row file,
- say which numbers you can compute by hand (size ratio, ideal projection) and which you must measure,
- show with a measurement that Parquet compression depends on the data, not only on the format.

:::

## In 30 seconds

A shop keeps every sale as one line in a ledger. To look up one sale you read one line, which is quick. To add up a month of sales by country you only need two columns, yet the ledger makes you read every line in full. Parquet stores each column on its own page instead, like a ledger with one page for countries and another for amounts. Adding up sales now touches two pages. The price is that recording a single new sale touches every page, so the best layout depends on the question you ask most often.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Row-oriented | A record's fields are stored together | One sale: id, date, country, amount |
| Columnar | Each field's values are stored together | All amounts in one block |
| Parquet | An open columnar file format with compression and statistics | `sales.parquet` |
| Row group | A horizontal slice of a Parquet file; each holds one chunk per column | 8 row groups of 250,000 rows |
| Projection | Reading only the columns a query needs | 2 of 20 columns |
| Predicate pushdown | Skipping data that cannot match a filter, using stored minimum and maximum values | One day of a 90-day file |
| Schema | The declared names and types of the fields | `amount` is a decimal number |
| Semi-structured | Self-describing data such as JSON whose keys can vary | `{"id": 7, "tags": ["a"]}` |

## The idea in plain words

A model sees features, but a product begins with events, records, images and documents. Data management decides which facts are retained, how they are identified, how they move and what a reader may assume about them. Poor handling can make a sound model answer the wrong question because its inputs are missing, stale or defined differently from the training set.

The **data → information → knowledge → wisdom** ladder (DIKW) is a useful way to see what data work is for. A transaction record is data: a customer identifier, a time and an amount. Information appears when we interpret it, for example monthly spending by customer. Knowledge is a tested pattern such as a reliable relation between payment history and default risk. Wisdom is the decision made with that pattern under business, legal and human constraints. The ladder is a teaching aid, not an automatic pipeline: aggregation does not guarantee insight, and a model's prediction does not decide what action is fair or useful.

<Infographic src="/img/dm/representations.svg" alt="Row records, exchange formats and columnar Parquet support different operations; selecting two of fifty equal columns is an ideal four per cent projection." caption="Representation choices follow workload: transactions, interchange and analytical scans ask different questions." />

Data may be **structured**, such as a table with declared fields; **semi-structured**, such as JSON with nested keys that vary by event; or **unstructured**, such as an image or a document. These names describe the form of the payload, not whether it has any metadata. A photograph can still have a structured asset ID, creation time, rights status and label table. Text becomes model-ready only after tokenisation or embedding, yet its source and access rules remain important.

:::note Added for this site

The course supplies the representation vocabulary and the scan example. The rest of this chapter adds schema evolution, stable identity, data contracts, a practical choice process for ML systems and a measured experiment on a two-million-row table.

:::

Move the sliders to change the number of fields selected. The example assumes a 10 GB row-oriented representation and 2 GB Parquet file, a fivefold size ratio. Under the simplifying assumption of 50 equal-sized columns, reading two gives a **4%** projection and an ideal **0.08 GB** of Parquet column data. Metadata, row groups, compression differences and filters make a real scan different.

<ColumnProjectionLab />

**What each control does.**

- **columns in the dataset** sets how many equal-sized columns the table has, 10 to 100.
- **columns in the query** sets how many of them the query reads.

**Try it yourself.**

1. Defaults: 2 of 50 columns is 4%, and the ideal read from a 2 GB Parquet file is 0.080 GB, the numbers computed by hand above.
2. Set the query to 10 columns. The fraction rises to 20% and the ideal read to 0.400 GB: five times more columns, five times more data.
3. Set the query to all 50. The fraction is 100% and the read is the full 2 GB. Projection saves nothing when a query needs every column, so only compression is left.

## Worked example, step by step

Take a tiny table of four sales with three columns, and suppose every stored value takes 8 bytes. The question is "total amount per country".

1. A row layout stores 4 rows of 3 values: 4 × 3 × 8 = 96 bytes. To answer the question the reader must walk all 96 bytes, because the values of one row sit together.
2. A column layout stores three blocks of 4 values, 32 bytes each. The question needs `country` and `amount`, so it reads 2 × 32 = 64 bytes, which is 64 / 96 = 67% of the data.
3. With 20 columns instead of 3 the same question reads 2 / 20 = 10% of the values. With the 50 columns of the example above it is 2 / 50 = 4%, and 2 GB × 0.04 = 0.08 GB.
4. Compression is a separate effect. 10 GB stored as 2 GB is a ratio of 10 / 2 = 5. Do not multiply 5 by 25 and promise a 125-fold speedup: the two effects overlap, and the second depends on the data.
5. Filters add a third effect. Suppose a file holds 90 days of events in 8 row groups. If events are stored in time order, one day (1/90 of the rows) lives in one group, so a reader can skip 7 of 8 groups, reading 12.5%. If events are shuffled, every group holds some of every day, so the reader must open all 8, reading 100%.

In words: columns decide how much of each row you read, compression decides how many bytes those values take, and sorting decides how many row groups a filter can skip. The experiment below measures all three on two million rows.

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

The first block's arithmetic is deliberately simple. Compression ratio compares physical sizes. The selected-column fraction is a *logical* fraction, not a promise about bytes read by a particular Parquet engine.

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

### Experiment: what a few million rows actually do

The by-hand numbers above are ideals. This experiment builds a table of two million rows and 20 columns (a time stamp, a country, an amount, two identifiers and fifteen random floats), writes it as CSV and as Parquet, and measures size, query time and how many row groups a one-day filter can skip. The data is synthetic, generated from a fixed seed, so there is no licence to worry about. Run it with DuckDB 1.5.6, pandas 2.3.3, pyarrow 25.0.1 and NumPy 2.5.3.

```python
import os
import tempfile
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

rows = 2_000_000
rng = np.random.default_rng(0)
frame = pd.DataFrame({"event_id": np.arange(rows), "user_id": rng.integers(0, 50_000, rows)})
frame["ts"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 90 * 86_400, rows)), unit="s")
frame["country"] = rng.choice(["GB", "IN", "US", "DE", "FR", "BR"], rows, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
frame["amount"] = rng.gamma(2.0, 20.0, rows).round(2)
for i in range(15):
    frame[f"f{i:02d}"] = rng.normal(size=rows)

folder = tempfile.mkdtemp()
paths = {name: f"{folder}/{name}" for name in ("t.csv", "t.parquet", "shuffled.parquet", "meta.csv", "meta.parquet")}
meta = frame[["event_id", "user_id", "ts", "country", "amount"]]
frame.to_csv(paths["t.csv"], index=False)
frame.to_parquet(paths["t.parquet"], compression="zstd", row_group_size=250_000)
frame.sample(frac=1.0, random_state=0).to_parquet(paths["shuffled.parquet"], compression="zstd", row_group_size=250_000)
meta.to_csv(paths["meta.csv"], index=False)
meta.to_parquet(paths["meta.parquet"], compression="zstd")
mb = lambda key: os.path.getsize(paths[key]) / 1e6
print(f"all 20 columns: csv {mb('t.csv'):.0f} MB, parquet {mb('t.parquet'):.0f} MB, ratio {mb('t.csv') / mb('t.parquet'):.2f}x")
print(f"5 compressible columns: csv {mb('meta.csv'):.0f} MB, parquet {mb('meta.parquet'):.0f} MB, ratio {mb('meta.csv') / mb('meta.parquet'):.2f}x")

def best(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)

con = duckdb.connect()
group_sql = "SELECT country, SUM(amount) FROM {src} GROUP BY country"
t_csv = best(lambda: con.sql(group_sql.format(src=f"read_csv('{paths['t.csv']}')")).fetchall())
t_pq = best(lambda: con.sql(group_sql.format(src=f"read_parquet('{paths['t.parquet']}')")).fetchall())
sum_all = "SELECT " + ", ".join(f"sum(f{i:02d})" for i in range(15)) + f", sum(amount) FROM read_parquet('{paths['t.parquet']}')"
t_all = best(lambda: con.sql(sum_all).fetchall())
print(f"group by on 2 of 20 columns: csv {t_csv:.3f}s, parquet {t_pq:.3f}s, parquet is {t_csv / t_pq:.0f}x faster")
print(f"parquet, 16 columns summed {t_all:.3f}s against 2 columns {t_pq:.3f}s: {t_all / t_pq:.1f}x")
t_all_arrow = best(lambda: pq.read_table(paths["t.parquet"]), 2)
t_two_arrow = best(lambda: pq.read_table(paths["t.parquet"], columns=["country", "amount"]), 2)
print(f"pyarrow read_table: all columns {t_all_arrow:.3f}s, two columns {t_two_arrow:.3f}s")

low, high = pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-11")
for name in ("t.parquet", "shuffled.parquet"):
    meta_pq = pq.ParquetFile(paths[name]).metadata
    ts_index = meta_pq.schema.names.index("ts")
    kept = 0
    for g in range(meta_pq.num_row_groups):
        stats = meta_pq.row_group(g).column(ts_index).statistics
        kept += not (stats.max < low or stats.min >= high)
    sql = f"SELECT count(*) FROM read_parquet('{paths[name]}') WHERE ts >= TIMESTAMP '2026-01-10' AND ts < TIMESTAMP '2026-01-11'"
    print(f"{name:17s} row groups that can match one day: {kept} of {meta_pq.num_row_groups}, query {best(lambda: con.sql(sql).fetchall()):.4f}s")
```

**Reading the output.** The first two lines compare file sizes. `group by on 2 of 20 columns` times the same SQL query on the CSV file and the Parquet file. The next line compares two queries on the Parquet file only, one touching 2 columns and one touching 16. The two `row groups that can match` lines count, from the file's own statistics, how many of the 8 row groups could contain the chosen day.

**Line by line.**

- `row_group_size=250_000` fixes 8 row groups for 2,000,000 rows, which makes the skip count easy to read.
- `best` takes the minimum of three runs, so the first read from disk does not dominate. All files are probably in the operating system's cache, which favours both formats.
- The loop over `metadata.row_group(g).column(ts_index).statistics` reads the minimum and maximum of `ts` that Parquet stored for each group. That is the information a reader uses to skip.
- `frame.sample(frac=1.0, ...)` writes the same rows in random order, so the only difference between the two Parquet files is sorting.

The printed output on the run that the prose below quotes:

```text
all 20 columns: csv 673 MB, parquet 285 MB, ratio 2.36x
5 compressible columns: csv 84 MB, parquet 19 MB, ratio 4.42x
group by on 2 of 20 columns: csv 0.380s, parquet 0.004s, parquet is 107x faster
parquet, 16 columns summed 0.052s against 2 columns 0.004s: 14.6x
pyarrow read_table: all columns 0.084s, two columns 0.004s
t.parquet         row groups that can match one day: 1 of 8, query 0.0031s
shuffled.parquet  row groups that can match one day: 8 of 8, query 0.0050s
```

### Reading the experiment

The size ratio is the surprise. The example earlier in the chapter says 10 GB becomes 2 GB, a ratio of 5. On this table the ratio is 2.36 (673 MB to 285 MB), because 15 of the 20 columns are random floating-point numbers, which have no repeated values to compress. Take those out and the five realistic columns (identifiers, a time stamp, a country and an amount) compress 4.42 times, from 84 MB to 19 MB. The ratio belongs to the data, not to the format, so measure it on your own table before you plan storage.

Projection is the second lesson. Two of 20 columns is 10% of the values, yet Parquet answered the query about 107 times faster than the CSV file. Most of that gap is not projection: CSV has to parse all 673 MB of text to find two columns. The fairer projection comparison is inside Parquet, where touching 2 columns instead of 16 took 14.6 times less time. That is more than the ideal 8 times because the two queries do different work, so treat it as "roughly the ideal, not a law".

Filtering depends on sorting. The sorted file let 1 of 8 row groups match the day, the shuffled file all 8. Yet the query times differ only slightly (0.0031 s against 0.0050 s), because two million rows is small and DuckDB is fast. Skipping matters when the data does not fit in memory or sits on remote storage.

Limits: one machine, warm cache, one seed, one table shape, and timings that move between runs (the same query came out 72 to 109 times faster across four runs, so quote the order of magnitude, not the digit).

<Infographic src="/img/dm-enrich/representations-measured.svg" alt="Cards show the measured CSV and Parquet sizes with their ratio, the speed of projection, and how sorting changes the number of row groups a filter can skip." caption="Look first at the left card: the size ratio is 2.36 on random floats and 4.42 on compressible columns, not the example's 5." />

## Designing with it

### Separate a data model from a file format

A **data model** says how entities and relationships are represented. A relational model makes tables and keys explicit; a document model can keep a nested event together; a key-value model makes retrieval by one key simple; a graph model makes relationships traversable. Wide-column stores are a fifth common model. Those models guide access patterns and constraints. CSV, JSON, Avro and Parquet are **formats** that encode bytes for exchange or storage. A JSON file may carry document-like objects, but choosing JSON does not itself give you a document database's indexes or transactions.

For ML, a stable entity key often matters more than the initial storage technology. A customer, device or ticket needs an identity that survives exports and recomputations. Define whether two records with the same key represent updates, duplicates or separate events. Keep event time and ingestion time separately when late arrivals are possible. Preserve raw source data long enough to reconstruct a feature after a transformation bug, subject to retention and privacy rules.

### Match layout to the operation

Row-oriented storage keeps a record's fields together, which suits point lookup and updates of a whole record. Column-oriented storage groups values from the same field, which suits wide scans that use only a few fields and often compresses repeated values well. This is a useful default distinction, not a law: row stores can run analytics, column stores can support selective access, and indexes, partitions and caches may dominate a specific query. Benchmark the queries that matter to the product.

The 10 GB to 2 GB example has two independent effects. Fivefold compression reduces stored bytes. Selecting 2 of 50 equally sized columns yields a 4% *ideal* projection. Do not multiply those into a guaranteed 125-fold speedup. Column widths and compression differ, Parquet reads row groups and metadata, a filter may read additional columns, and execution has CPU and network costs. The [Parquet concepts](https://parquet.apache.org/docs/concepts/) explain that a row group contains one chunk per column, while the [DuckDB guide](https://duckdb.org/docs/current/guides/performance/file_formats) describes which queries benefit from pushdown.

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

## Common mistakes

1. **Quoting a compression ratio from someone else's table.** A ratio of 5 sounds like a property of Parquet. It is a property of the data: 2.36 here with random floats, 4.42 without them. Measure on your own columns.
2. **Multiplying compression by projection.** 5 × 25 = 125 looks like a speedup. Reading fewer columns and storing fewer bytes overlap, and metadata, row groups and CPU decoding remain. Benchmark the query you actually run.
3. **Treating CSV against Parquet as a projection test.** The 107 times gap includes parsing text. Compare projection inside one format if you want to isolate it.
4. **Shuffling before you write.** A file in random order keeps every row group alive for every filter (8 of 8 here). Sort or partition by the column you filter on most.
5. **Choosing a format before choosing the key and the time meaning.** A fast file with an ambiguous `amount` or `ts` still trains a wrong model. Write the schema contract first.

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

<details>
<summary><strong>Q6. (Medium)</strong> The experiment gave a CSV to Parquet ratio of 2.36 on 20 columns and 4.42 on five of them. Why does the ratio differ, and what would you check before planning storage from the example's ratio of 5?</summary>

Fifteen columns are random floats, which have almost no repeated values, so a compressor cannot shrink them. The other five (identifiers, time stamps, a six-valued country, an amount rounded to cents) repeat or have small ranges. The ratio is therefore a property of the data. Before planning storage, write a sample of your own table to Parquet with the codec you intend to use and measure the ratio.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> The sorted file let a one-day filter skip 7 of 8 row groups, but the query was only 0.0031 s against 0.0050 s. Is the sorting worthless?</summary>

No. Two million rows fit in memory and the file is cached, so even reading all 8 groups is quick. The saving that matters is bytes read and decoded, which grows with table size and with slow storage such as object stores. The honest conclusion is that the effect is real in the file statistics (1 of 8 against 8 of 8) but small in wall-clock time at this scale. Repeat the test on a table too large for memory before deciding.

</details>

## Go deeper

- [Apache Parquet concepts](https://parquet.apache.org/docs/concepts/) explains row groups and column chunks.
- [DuckDB Parquet documentation](https://duckdb.org/docs/stable/data/parquet/overview) shows direct file queries and pushdown.
- [DuckDB file format performance guide](https://duckdb.org/docs/current/guides/performance/file_formats) (opened 2026-10-09) says DuckDB can apply projection and filter pushdown on Parquet, works best with row groups of 100,000 to 1,000,000 rows, and recommends loading data into a table for join-heavy or repeated queries.
- Library versions run for the experiment: DuckDB 1.5.6, pandas 2.3.3, pyarrow 25.0.1, NumPy 2.5.3, Python 3.14.6.
- Built from the course lecture "dm-s1-intro" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can explain why the same event may need different transactional and analytical representations.
- [ ] I can distinguish a data model from a storage or exchange format.
- [ ] I can compute the fivefold size ratio and four per cent ideal projection without claiming a measured speedup.
- [ ] I can preserve entity identity, time meaning and a schema contract across the ML data path.
- [ ] I can explain why a measured compression ratio belongs to the data and not to the file format.
- [ ] I can say how sorting changes the number of Parquet row groups a filter can skip, and why wall-clock gains depend on table size.
- [ ] I can separate parsing cost from projection benefit when comparing CSV with Parquet.

## Where to go next

Next is [data quality rules](/docs/mlops/data/quality-rules), which turns the schema and null semantics from this chapter into checks. For the storage side of the same ideas, see [warehouses, lakes and lakehouses](/docs/mlops/data/warehouses-lakes-and-lakehouses).
