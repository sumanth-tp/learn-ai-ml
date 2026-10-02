---
id: dm-analytics-engineering-and-history
title: "Data Management · Lecture 9 — Analytics Engineering and History"
sidebar_label: "9 · Analytics engineering"
sidebar_position: 3
slug: /mlops/data/analytics-engineering-history
description: "Build tested analytical models, shared metrics and Type 2 history that supports point-in-time ML joins."
tags: [data-management, analytics-engineering, dbt, slowly-changing-dimensions]
---

import Infographic from '@site/src/components/Infographic';
import ScdHistoryLab from '@site/src/components/viz/ScdHistoryLab';

**In one line.** Analytical tables need tested transformations and a recorded meaning that survives source changes.

## The idea in plain words

Raw source tables often use names, keys and update rules chosen for an operational product. Analysts and model builders need a stable analytical meaning. **Analytics engineering** treats the transformations between those worlds as maintained code: versioned SQL or Python, declared dependencies, tests, documentation and lineage. A table is not trusted because it has an appealing name; it is trusted when a consumer can find its grain, rules, source and owner.

The lecture describes three layers. **Staging** gives source fields clear types and names while preserving identity. **Intermediate** models join or reshape reusable business entities. **Marts** expose facts, dimensions and metrics for consumers. A semantic layer defines a metric such as "active user" once, including filters and time basis, so reports do not silently disagree. The layers are a design pattern, not a fixed count of directories. A small team may combine them while keeping the same contracts.

<Infographic src="/img/dm/analytics-engineering.svg" alt="Staging, intermediate and mart layers add types, reusable entities and metrics; a Type 2 customer with three address changes has four historical rows." caption="Layered models make transformations inspectable; Type 2 history keeps old facts interpretable." />

The lecture contrasts **Slowly Changing Dimension Type 1** and **Type 2**. Type 1 overwrites a descriptive value, leaving only the latest. Type 2 ends the old version's validity window and adds a new row. One original address plus three changes produces **four rows**. At any historical event time, a point-in-time join should choose the one version valid then. Without this history, a current address can be attached to an old transaction and create a misleading regional report or a leaked model feature.

:::note Beyond the lecture

The source gives the layers, semantic metric, Type 1/2 contrast and row count. The sections below add grain checks, half-open validity windows, late corrections and the distinction between event time and knowledge time.

:::

Move the change count. The default **three changes** create **four Type 2 rows**, matching the lecture. Move the historical time to see which version is valid. The intervals are half-open: a new version owns its start time, so adjacent versions do not both match the boundary.

<ScdHistoryLab />

## How it works

### Layers, semantic layer, SCD

- **Layered models**; Staging → intermediate → marts; semantic layer defines metrics once.
- **SCD**; Type 1 overwrites; Type 2 versions rows (valid-from/to) for history.

:::tip

**Worked.** A customer with 3 address changes → 1+3 = 4 rows in an SCD Type 2 table (point-in-time correct).

:::

### Tests, docs, lineage

Tests (uniqueness, not-null, accepted values, referential integrity), documentation and generated lineage graphs make data trustworthy and changes safe.


## A real system that works this way

**dbt** is the lecture's concrete analytics-engineering example. Its documentation describes models, data tests, snapshots and generated documentation. A dbt snapshot can record changes in a mutable source table as Type 2 history with `dbt_valid_from` and `dbt_valid_to`. A current row may have a null end time. The team still must choose a reliable key and change-detection strategy and run snapshots often enough; history that was overwritten before the next snapshot cannot be recovered automatically.

Consider a customer dimension with a region field. A salesperson moves from North to South. A Type 1 table makes every past sale appear to belong to South when joined to the current customer row. A Type 2 table records North until the change date and South after it. A sales report can then answer either "what was the region at sale time?" or "what is the customer's region now?" without confusing them. The first needs a validity-window join; the second deliberately uses the current row.

dbt's semantic layer is another named implementation. It lets a team define a metric such as net revenue over approved payments and expose that definition to downstream tools. This reduces duplicated SQL, but the metric still needs a clear grain, currency policy, refund treatment and time zone. A shared wrong definition spreads an error consistently. Tests, review and ownership remain necessary.

## Code you can run

The first block constructs four half-open address versions from an original value and three changes. It checks the lecture's row count and selects the version valid at a historical day.

```python
changes = [(0, "North"), (10, "South"), (20, "East"), (30, "West")]
history = [
    {"from": day, "to": changes[index + 1][0] if index + 1 < len(changes) else None, "region": region}
    for index, (day, region) in enumerate(changes)
]

def as_of(day):
    matches = [row for row in history if row["from"] <= day and (row["to"] is None or day < row["to"])]
    if len(matches) != 1:
        raise ValueError("Expected exactly one historical version")
    return matches[0]["region"]

print(history)
print("day 15:", as_of(15))
assert len(history) == 4
assert as_of(15) == "South"
assert as_of(30) == "West"
```

The second block shows why a metric definition needs a clear event set. The same payments produce different totals when refunds are subtracted versus ignored. Neither query is universally correct; the semantic contract decides.

```python
payments = [
    {"id": "P1", "kind": "charge", "amount": 100},
    {"id": "P2", "kind": "charge", "amount": 40},
    {"id": "R1", "kind": "refund", "amount": 20},
]
gross = sum(row["amount"] for row in payments if row["kind"] == "charge")
net = sum(row["amount"] if row["kind"] == "charge" else -row["amount"] for row in payments)
print(f"gross={gross}, net={net}")
assert (gross, net) == (140, 120)
```

In a production mart, these definitions also need currency, status, date and duplicate-event rules. The calculation is small so the semantic choice remains visible.

## Designing with it

### Declare the grain and keys

Before joining models, write what one row represents. A fact may be one payment event; a customer dimension may have one current row or many historical versions. Joining a payment to every historical customer row multiplies amounts. Test uniqueness at the declared key and include a validity condition for Type 2 joins. A source with duplicate keys needs a repair or an explicit deduplication rule, not a hidden `DISTINCT` that happens to make one report look right.

Staging should preserve source identifiers and enough metadata to explain a later value. Intermediate models can standardise entities and resolve relationships. Marts should expose stable consumer contracts. A lineage graph helps reviewers see which reports and model features depend on a changed source or transformation. It does not prove that a business definition is right; document why a metric includes or excludes each event type.

### Choose Type 1 or Type 2 deliberately

Type 1 is suitable when only current state matters or an earlier value was simply wrong and should be corrected everywhere. Type 2 is suitable when historical state matters: customer region at purchase, account tier at application or product category at order time. Every new version needs a business key, valid-from and valid-to. Use half-open windows, `[start, end)`, so exactly one version owns a boundary. Check for gaps and overlaps. Decide how a late correction is inserted and whether earlier reports must be recomputed.

Snapshots of mutable tables have limits. A snapshot taken nightly cannot see a value that changed twice between runs and ended at its initial value. An event log or CDC stream may be necessary for complete change history. Decide whether validity timestamps mean source event time or observation time. For ML, the date a value became known may differ from the date it was said to be valid; both can matter for leakage-safe training.

### Make metrics reusable without hiding assumptions

A semantic metric names its numerator, denominator, filters, time dimension and aggregation. "Active customer" could mean logged in during 30 days, purchased during 30 days or held an open account at month's end. A shared definition is useful only when those conditions are reviewable. Version a breaking metric change, compare old and new values on representative periods and tell consumers which dashboards or features are affected.

Tests should target the model's promises: unique fact IDs, not-null keys, accepted statuses, foreign-key relationships, non-overlapping Type 2 windows and reconciled totals. A green test suite does not rule out a missing source feed unless a freshness or volume check covers it. Pair transformation tests with source monitoring.

## Rebuild a historical customer metric

A team wants revenue by customer region for each month last year. The payment fact table records payment ID, customer ID, event time, amount and refund links. The customer table currently stores only the latest region. Joining those tables directly answers "revenue grouped by customers' present regions," not "revenue by region at the time of payment." Both may be valid questions, but they are not interchangeable. The analysis request must choose one.

To answer the historical question, build a Type 2 customer dimension from reliable changes. Each row has customer ID, region, start and end time. For every payment, join by customer ID and require start ≤ payment time < end, allowing an open end for the current version. Test that each payment matches at most one version. An unmatched payment may reflect an incomplete history or a new customer; a multiple match indicates overlapping validity windows. Do not silently duplicate or drop those facts.

Now account for refunds. A refund may occur in a later month and after a customer moves. Should it reduce the original sale month and region or appear in the refund month under the current region? The business owner must decide, and the metric definition should encode that choice. If reports need both views, name them separately. Documenting "net revenue" without its refund-time semantics is insufficient.

Suppose an address correction arrives two months late and claims the customer actually moved a week earlier than originally recorded. The dimension's valid-time history changes. Reports for affected dates need recomputation. A model that trained on a previous snapshot should still be reproducible; keep the old dataset version and record the corrected one. This distinction between a historical truth update and what was known at prediction time is especially important in ML. A value corrected today must not silently become a feature in an old decision that could not have known it.

Finally, trace the final report or feature back through the mart, intermediate model and staging source. A source schema change can affect only one region or status code, leaving global totals close enough to look plausible. Lineage tells the team where to test; reconciliation and sampled records tell it whether the result is right. Analytics engineering earns trust by making those checks repeatable and reviewable.

### Check snapshot cadence and validity

A nightly snapshot sees the states present when it runs. If an order changes from pending to shipped and then cancelled within one day, the intermediate shipped state may never appear in the snapshot. That may be acceptable for a daily report but not for measuring shipping process time. Compare the required historical resolution with the source's update frequency. Use an event log or CDC if the use case needs every transition. A Type 2 table is only as complete as the changes it observed.

Validity windows also need a boundary convention. Half-open `[valid_from, valid_to)` intervals assign an exact change timestamp to the new row. If a source supplies only dates, decide whether a change at midnight applies to the new business day in the source time zone. Test adjacent windows for gaps and overlaps by key. A join that matches no row should be an explicit unknown state, not silently dropped from a revenue denominator. A join that matches two rows should fail validation before it doubles amounts.

### Version semantic changes

Suppose the organisation changes "active user" from one login in 30 days to one purchase in 30 days. Both definitions are plausible, but replacing the metric in place rewrites dashboards and model features without a clear comparison. Create a versioned definition or a named new metric, calculate both over representative periods and tell consumers the migration date. Track which reports and models depend on the old meaning. Shared metrics reduce duplication only if changes to them are treated as product changes.

The same applies to historical corrections. A late address change can improve the best current reconstruction of the past, yet an earlier model decision should still be reproducible from what was known then. Store the curated table snapshot or a bitemporal record of valid time and knowledge time when the decision demands it. This is more work than a basic Type 2 dimension, but it prevents a corrected fact from becoming an impossible historical input. Keep the original source event ID beside the dimension version for audit.

## Where this stands in 2026

:::info Industry view

- dbt snapshots record Type 2 changes from mutable tables and expose validity columns for historical queries.
- Shared semantic metrics can reduce duplicated definitions across tools, provided the definitions and access rules are governed.
- For ML, historical truth and historical knowledge can differ; point-in-time features need the values actually available at the prediction cutoff.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is analytics engineering and what tool typifies it?</summary>

Turning raw data into clean, tested, documented transformation models analysts and ML trust; typified by dbt (version-controlled SQL with tests and docs).<br /><em>Lecture 9 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is a semantic layer?</summary>

A layer that defines metrics once (e.g. 'active user') so every consumer computes them consistently.<br /><em>Lecture 9 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Contrast SCD Type 1 and Type 2.</summary>

Type 1 overwrites (history lost); Type 2 adds a new versioned row with valid-from/valid-to, preserving history for point-in-time correctness.<br /><em>Lecture 9 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> A customer has 3 address changes in an SCD Type 2 table. How many rows?</summary>

1 original + 3 changes = 4 rows, each valid for a date range.<br /><em>Lecture 9 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What software practices does analytics engineering add to data?</summary>

Version control, automated tests, documentation, and lineage; treating transformations as maintainable code.<br /><em>Lecture 9 · conceptual</em>

</details>

## Go deeper

- [dbt snapshots](https://docs.getdbt.com/docs/build/snapshots) documents Type 2 history and its configuration.
- [dbt Semantic Layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl?version=2) describes shared metric definitions.
- [dbt data tests](https://docs.getdbt.com/docs/build/data-tests?version=1.12) lists core integrity checks.
- Built from the course lecture "dm-l9-analytics-engineering" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can name the grain and key of a fact, dimension and mart before joining them.
- [ ] I can explain how three address changes create four Type 2 rows.
- [ ] I can write a half-open validity join and detect gaps or overlaps.
- [ ] I can state the business rules behind a shared metric and trace its inputs.
