---
id: dm-question-bank
title: "Data Management; Question Bank"
sidebar_label: "1 · Question bank"
sidebar_position: 1
slug: /mlops/data/question-bank
description: "Twenty-six data-management questions grouped by the lecture sequence, with source answers, verified calculations and corrections."
tags: [data-management, practice, question-bank]
---

import Infographic from '@site/src/components/Infographic';
import FreshnessBudgetLab from '@site/src/components/viz/FreshnessBudgetLab';

**In one line.** Practise the data lifecycle with all 26 unique questions from the two course banks, and check the assumptions behind each worked answer.

## How to use this bank

The comprehensive bank repeats questions 11–26 of the main bank verbatim, including their answers. Each appears once below. The groups follow the lecture sequence, with a final calculation group spanning the course. Attempt each question before opening its answer. The source answer is retained; a labelled correction follows where its wording makes a claim stronger than the evidence supports.

<Infographic src="/img/dm/question-bank.svg" alt="Twenty-six questions span foundations, production data systems and ten worked calculations; sixteen repeated questions appear only once." caption="Use the lecture groups to revise concepts, then check the units and assumptions in the calculations." />

## Questions by lecture group

### Sessions 1–4: foundations, architecture and pipelines

<details>
<summary><strong>Q1.</strong> What is data engineering for ML, and why does it matter?</summary>

Building the systems that collect, store, transform and serve data reliably for ML. It matters because model quality is capped by data quality and availability; most production ML effort is data plumbing, not modelling.

</details>

<details>
<summary><strong>Q2.</strong> Contrast a data warehouse, data lake, and lakehouse.</summary>

A warehouse stores structured, schema-on-write data for analytics; a lake stores raw, schema-on-read data of any type cheaply; a lakehouse combines lake storage with warehouse-style management (ACID, schema); e.g. Delta Lake.

:::note Correction to the source answer

These are useful patterns, not rigid storage rules. Warehouses can hold or reference semi-structured and media data; a lake can enforce schemas and transactions through a table format.

:::
</details>

<details>
<summary><strong>Q3.</strong> Contrast batch and streaming data pipelines.</summary>

Batch processes bounded data on a schedule (high throughput, latency-tolerant); streaming processes unbounded events in near real time (low latency). The choice follows the freshness requirement of the consumer.

</details>

<details>
<summary><strong>Q4.</strong> What is the difference between ETL and ELT?</summary>

ETL transforms data before loading it into the warehouse; ELT loads raw data first and transforms inside the warehouse using its compute. ELT suits modern cloud warehouses with cheap storage and scalable SQL compute.

:::note Correction to the source answer

ELT is a common choice when the destination can execute the transformations. It is not universally cheaper or better; privacy, source limits and compute cost can favour ETL.

:::
</details>

### Sessions 5–8: lifecycle, quality and ingestion

<details>
<summary><strong>Q5.</strong> What are the stages of the ML data lifecycle?</summary>

Collection/ingestion → storage → profiling & validation → transformation/feature engineering → training → serving → monitoring; a loop, since monitoring feeds new data back in.

</details>

<details>
<summary><strong>Q6.</strong> What is data profiling and data validation?</summary>

Profiling summarises a dataset's structure and statistics (types, ranges, nulls, distributions); validation checks it against expectations (schema, constraints, distribution drift) so bad data is caught before it reaches training.

</details>

<details>
<summary><strong>Q7.</strong> What is schema-on-read vs schema-on-write?</summary>

Schema-on-write enforces structure when data is stored (warehouses); schema-on-read applies structure when data is queried (lakes); trading upfront rigidity for flexibility and cheaper ingestion.

</details>

<details>
<summary><strong>Q8.</strong> What is data lineage and why is it important?</summary>

A record of where data came from and how it was transformed. It is essential for debugging, reproducibility, impact analysis, and governance/compliance (knowing what fed a model).

</details>

<details>
<summary><strong>Q9.</strong> What is DataOps, and how do CI/CD and Continuous Training (CT) apply to data pipelines?</summary>

DataOps applies DevOps practices to data: automation, CI/CD, testing, monitoring and collaboration across the pipeline. CI/CD builds and tests pipeline/transform code on every change; Continuous Training (CT) automatically retrains models on fresh, validated data; together giving reproducible, observable, continuously-updated data+ML pipelines.

:::note Correction to the source answer

Continuous training is an optional, governed workflow. Fresh data alone must not trigger deployment; evaluate data validity, model quality, safety and approval criteria first.

:::
</details>

<details>
<summary><strong>Q10.</strong> What is modern data infrastructure / the 'modern data stack'?</summary>

A cloud-native, modular stack: managed ingestion → cheap object/lakehouse storage → ELT transformation (dbt) → orchestration (Airflow) → serving to BI and ML; automated and observable, replacing monolithic ETL with composable, scalable services.

</details>

### Lectures 9–16: orchestration, features and governance

<details>
<summary><strong>Q11.</strong> What does a workflow orchestrator (e.g. Airflow) provide?</summary>

It schedules and runs pipeline tasks as a DAG, handling dependencies, retries, backfills and monitoring; turning ad-hoc scripts into reliable, observable, repeatable workflows.

</details>

<details>
<summary><strong>Q12.</strong> What is a feature store and what problem does it solve?</summary>

A central system for computing, storing and serving features consistently for training and inference, preventing training-serving skew and enabling feature reuse and point-in-time correctness.

:::note Correction to the source answer

A feature store can support shared definitions and historical retrieval, but cannot guarantee training-serving parity or prevent leakage on its own. Event time, availability time and parity tests must be designed explicitly.

:::
</details>

<details>
<summary><strong>Q13.</strong> What is training-serving skew?</summary>

When features are computed differently (or from different data) at training vs serving time, degrading live performance. Feature stores and shared transformation code prevent it.

:::note Correction to the source answer

Shared transformation code helps reduce skew but does not prove its absence. Check source timing, missing-value behaviour and online/offline parity.

:::
</details>

<details>
<summary><strong>Q14.</strong> Name two techniques for privacy-preserving / responsible data handling.</summary>

Anonymisation/pseudonymisation, differential privacy (adding calibrated noise), access controls and data minimisation; plus governance practices like consent tracking and audit logs.

</details>

<details>
<summary><strong>Q15.</strong> What should ML observability monitor beyond model accuracy?</summary>

Data drift (input distribution changes), prediction drift, feature health (nulls, ranges), pipeline latency/throughput/failures, and; when labels arrive; live accuracy, with alerting and retraining triggers.

:::note Correction to the source answer

A drift alert is evidence for investigation, not an automatic retraining command. Check labels, data quality, impact and approved response rules.

:::
</details>

<details>
<summary><strong>Q16.</strong> Name the five pillars of data observability.</summary>

Freshness (up to date?), volume (expected amount arrived?), schema (structure changed?), distribution (values in range?), and lineage (provenance / downstream impact).

</details>

### Worked calculations across the course

<details>
<summary><strong>Q17.</strong> A 10 GB CSV compresses to 2 GB as Parquet; a query touches 2 of 50 columns. Give the compression ratio and fraction scanned.</summary>

Compression = 10/2 = 5×; columnar reads only 2/50 = 4% of the data.

:::note Correction to the source answer

Two of fifty columns are 4% of the column count, not necessarily 4% of stored bytes or I/O. Column widths, compression, metadata and row-group pruning determine bytes read.

:::
</details>

<details>
<summary><strong>Q18.</strong> A 1000-row column has 50 nulls. Compute completeness; does it pass a 99% rule?</summary>

Completeness = (1000−50)/1000 = 0.95 (95%) &lt; 99% → fails; quarantine or fix before training.

:::note Correction to the source answer

The 99% rule fails. Quarantine, repair or a documented exception depends on the data contract and the downstream risk.

:::
</details>

<details>
<summary><strong>Q19.</strong> A stream stage processes λ=100 events/s at W=0.2 s each. How many are in flight (Little's Law)?</summary>

L = λW = 100 × 0.2 = 20 events in flight.

:::note Correction to the source answer

Little’s Law gives 20 only for a stable system using long-run average arrival rate and end-to-end time over the same boundary.

:::
</details>

<details>
<summary><strong>Q20.</strong> A 99.9% availability SLA allows how much downtime per year?</summary>

(1−0.999)×365×24 = 0.001×8760 = 8.76 hours/year (99.99% → 0.876 h).

:::note Correction to the source answer

The 8.76-hour result assumes a 365-day year. The exact annual budget changes with the chosen calendar and measurement window.

:::
</details>

<details>
<summary><strong>Q21.</strong> Standardise a feature value x=80 given μ=70, σ=5.</summary>

z = (x−μ)/σ = (80−70)/5 = 2.0; two standard deviations above the mean; fit the scaler on training data only.

</details>

<details>
<summary><strong>Q22.</strong> A 10 GB dataset splits into how many 128 MB Spark partitions?</summary>

10×1024/128 = 10240/128 = 80 partitions (≈80 parallel tasks).

:::note Correction to the source answer

This is 80 size-based pieces if GB means GiB and MB means MiB. Actual partition counts also depend on file layout and engine settings; 80 pieces do not imply 80 simultaneous workers.

:::
</details>

<details>
<summary><strong>Q23.</strong> RAG retrieval: query q=[1,0,1,1], chunk d=[1,1,1,0]. Compute cosine similarity.</summary>

q·d = 2, ‖q‖=√3, ‖d‖=√3 → cos = 2/3 = 0.667 → retrieved.

:::note Correction to the source answer

The cosine is 2/3, but a document is retrieved only if it passes the system’s candidate selection and threshold or ranking policy.

:::
</details>

<details>
<summary><strong>Q24.</strong> The smallest \{age,ZIP\} group has 4 people. Give k and the re-identification bound.</summary>

k = 4 (k-anonymity); re-identification probability ≤ 1/4 = 0.25. A group of size 1 must be suppressed/generalised.

:::note Correction to the source answer

The group size establishes k = 4 for those quasi-identifiers. It does not imply a general re-identification probability bound of 1/4; outside information, group homogeneity and sensitive attributes matter.

:::
</details>

<details>
<summary><strong>Q25.</strong> Data was last loaded 90 minutes ago against a 60-minute freshness SLA. What happens?</summary>

Data age 90 > 60 → freshness SLA breached → alert (a core data-observability check).

</details>

<details>
<summary><strong>Q26.</strong> How is PSI interpreted for drift?</summary>

PSI &lt; 0.1 = no significant shift; 0.1–0.25 = moderate; > 0.25 = major shift → retrain.

:::note Correction to the source answer

The quoted PSI cutoffs are heuristics, not statistical guarantees. A high PSI should prompt investigation; retraining depends on verified quality and outcome impact.

:::
</details>

## Code you can run

The first block verifies every arithmetic answer in Q17–Q25. It distinguishes a column fraction from bytes read and a partition count from worker count. It also states the units used for the size calculation.

```python
from math import sqrt

compression = 10 / 2
column_fraction = 2 / 50
completeness = (1000 - 50) / 1000
in_flight = 100 * 0.2
downtime_hours = (1 - 0.999) * 365 * 24
z_score = (80 - 70) / 5
size_pieces = (10 * 1024) / 128
query = [1, 0, 1, 1]
chunk = [1, 1, 1, 0]
cosine = sum(a * b for a, b in zip(query, chunk)) / sqrt(sum(a * a for a in query) * sum(b * b for b in chunk))
k = 4
freshness_breach_minutes = 90 - 60
print(f"Q17 compression {compression:.0f}×; column fraction {column_fraction:.0%}")
print(f"Q18 completeness {completeness:.0%}; passes 99%: {completeness >= 0.99}")
print(f"Q19 in flight {in_flight:.0f}; Q20 downtime {downtime_hours:.2f} h")
print(f"Q21 z {z_score:.1f}; Q22 size pieces {size_pieces:.0f}")
print(f"Q23 cosine {cosine:.3f}; Q24 k {k}; Q25 breach {freshness_breach_minutes} min")
assert (compression, column_fraction, completeness, in_flight) == (5, 0.04, 0.95, 20)
assert round(downtime_hours, 2) == 8.76
assert (z_score, size_pieces, round(cosine, 3), k, freshness_breach_minutes) == (2, 80, 0.667, 4, 30)
```

The second block makes the Q26 PSI threshold discussion concrete. These are invented teaching distributions; no retraining decision follows from the value alone.

```python
from math import log

baseline = [0.5, 0.3, 0.2]
observed = [0.4, 0.35, 0.25]
psi = sum((o - b) * log(o / b) for b, o in zip(baseline, observed))
print(f"Teaching PSI {psi:.3f}")
assert round(psi, 3) == 0.041
```

The lecture's freshness question uses 90 minutes against a 60-minute limit, a **30-minute breach**. Move the approved maximum in this lab to see how the alert changes. Its default matches Q25 and the first block.

<FreshnessBudgetLab />

## Reading the answers responsibly

The bank covers conceptual recall and small arithmetic checks. In production, a threshold or architecture label does not decide the response by itself. Write down the measurement window, units, data contract and downstream consequence before choosing a fix. For further explanations, follow the linked Data Management lectures in this section of the course.

## Check yourself

- I can explain why two of fifty columns does not imply reading exactly 4% of a file's bytes.
- I can explain when Little’s Law gives twenty in-flight events.
- I can explain why a feature store and a drift threshold are supports for a governed decision rather than guarantees.
