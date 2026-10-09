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

:::tip Before you start

**You should already know**

- What a table, a primary key and a join are ([Session 1, data representations](/docs/mlops/data/representations-for-ml)).
- Why a feature value must be the one known at prediction time ([Lecture 11, features and point-in-time correctness](/docs/mlops/data/features-and-point-in-time)).

**Reading time:** about 40 minutes, plus a minute to run the code.

**After this chapter you can**

- Build a Type 2 history table and join to it without double counting or dropping rows.
- Say how wrong a "latest value" join is on a real-sized table, and why the totals can still look fine.
- Write the checks that catch gaps and overlaps in validity windows.

:::

## In 30 seconds

A customer moves from the north to the south. If your table keeps only the new address, every old order now looks as if it came from the south. Type 2 history keeps one row per address, each with the dates it was true, so an old order finds the address it really had.

Think of a passport with stamped pages instead of one crossed-out line. You can read where the holder was on any given date.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Grain | What one row stands for | One payment, or one customer version |
| Dimension | A table that describes things, such as customers | `customer_id`, `region` |
| Type 1 | Overwrite the old value | North becomes South, North is gone |
| Type 2 | Add a new row, close the old one | North until day 10, South from day 10 |
| Validity window | The dates a row was true, `[valid_from, valid_to)` | `[10, 20)` includes day 10, excludes day 20 |
| Half-open | Start included, end excluded | Day 20 belongs to the next row |
| Point-in-time join | Join each event to the version valid at its time | An order on day 15 gets South |
| Snapshot | A copy of a table taken on a schedule | A nightly copy of the customer table |


## The idea in plain words

Raw source tables often use names, keys and update rules chosen for an operational product. Analysts and model builders need a stable analytical meaning. **Analytics engineering** treats the transformations between those worlds as maintained code: versioned SQL or Python, declared dependencies, tests, documentation and lineage. A table is not trusted because it has an appealing name; it is trusted when a consumer can find its grain, rules, source and owner.

The usual pattern has three layers. **Staging** gives source fields clear types and names while preserving identity. **Intermediate** models join or reshape reusable business entities. **Marts** expose facts, dimensions and metrics for consumers. A semantic layer defines a metric such as "active user" once, including filters and time basis, so reports do not silently disagree. The layers are a design pattern, not a fixed count of directories. A small team may combine them while keeping the same contracts.

<Infographic src="/img/dm/analytics-engineering.svg" alt="Staging, intermediate and mart layers add types, reusable entities and metrics; a Type 2 customer with three address changes has four historical rows." caption="Layered models make transformations inspectable; Type 2 history keeps old facts interpretable." />

Two ways of keeping a changing attribute are **Slowly Changing Dimension Type 1** and **Type 2**. Type 1 overwrites a descriptive value, leaving only the latest. Type 2 ends the old version's validity window and adds a new row. One original address plus three changes produces **four rows**. At any historical event time, a point-in-time join should choose the one version valid then. Without this history, a current address can be attached to an old transaction and create a misleading regional report or a leaked model feature.

:::note Added for this site

The layers, the semantic metric, the Type 1 and Type 2 contrast and the row count come from the course material. The sections below add grain checks, half-open validity windows, late corrections and the distinction between event time and knowledge time.

:::

Move the change count. The default **three changes** create **four Type 2 rows**, matching the worked example below. Move the historical time to see which version is valid. The intervals are half-open: a new version owns its start time, so adjacent versions do not both match the boundary.

<ScdHistoryLab />

## Worked example, step by step

One customer has an original region and three moves. Orders arrive on days 5, 15 and 30. Compare three ways of finding the region.

| `valid_from` | `valid_to` | Region |
| ---: | ---: | --- |
| 0 | 10 | North |
| 10 | 20 | South |
| 20 | 30 | East |
| 30 | open | West |

1. **Count the rows.** One original value plus three changes gives 4 rows.
2. **Half-open join.** The order on day 15 needs `valid_from <= 15 < valid_to`. Only South qualifies, because 10 ≤ 15 < 20. The order on day 30 qualifies for West (30 ≤ 30, open end). East fails because 30 < 30 is false. One match each.
3. **Closed join (`BETWEEN`).** Now East qualifies for day 30 as well, because 30 ≤ 30 is true for the closed end. The £25 order matches two rows. Revenue for the three orders of £10, £40 and £25 becomes 10 + 40 + 25 + 25 = £100 instead of £75.
4. **Latest-row join.** Every order takes West, the current region. Day 5 should be North and day 15 should be South, so 2 of 3 orders are wrong.

In words: the boundary day belongs to exactly one version, and the current row answers a different question from the historical one. The first code block below prints these numbers.

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

**dbt** is the best-known analytics-engineering tool. Its documentation describes models, data tests, snapshots and generated documentation. A dbt snapshot can record changes in a mutable source table as Type 2 history with `dbt_valid_from` and `dbt_valid_to`. A current row may have a null end time. The team still must choose a reliable key and change-detection strategy and run snapshots often enough; history that was overwritten before the next snapshot cannot be recovered automatically.

Consider a customer dimension with a region field. A salesperson moves from North to South. A Type 1 table makes every past sale appear to belong to South when joined to the current customer row. A Type 2 table records North until the change date and South after it. A sales report can then answer either "what was the region at sale time?" or "what is the customer's region now?" without confusing them. The first needs a validity-window join; the second deliberately uses the current row.

dbt's semantic layer is another named implementation. It lets a team define a metric such as net revenue over approved payments and expose that definition to downstream tools. This reduces duplicated SQL, but the metric still needs a clear grain, currency policy, refund treatment and time zone. A shared wrong definition spreads an error consistently. Tests, review and ownership remain necessary.

## Code you can run

The first block constructs four half-open address versions from an original value and three changes. It checks the row count of one original value plus three changes and selects the version valid at a historical day.

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

### The worked example in code

This block runs the three joins from the worked example on the three orders.

```python
history = [(0, 10, "North"), (10, 20, "South"), (20, 30, "East"), (30, None, "West")]
orders = [(5, 10), (15, 40), (30, 25)]

def half_open(day):
    return [r for lo, hi, r in history if lo <= day and (hi is None or day < hi)]

def closed(day):
    return [r for lo, hi, r in history if lo <= day and (hi is None or day <= hi)]

rows_half = sum(len(half_open(day)) for day, _ in orders)
revenue_half = sum(amount * len(half_open(day)) for day, amount in orders)
revenue_closed = sum(amount * len(closed(day)) for day, amount in orders)
wrong_latest = sum(half_open(day)[0] != history[-1][2] for day, _ in orders)
print("rows", len(history), "half-open matches", rows_half, "revenue", revenue_half)
print("closed matches", sum(len(closed(day)) for day, _ in orders), "revenue", revenue_closed)
print("latest-row join wrong for", wrong_latest, "of", len(orders), "orders")
```

**Reading the output.** It prints 4 rows, 3 half-open matches and revenue 75, then 4 closed matches and revenue 100, then 2 of 3 orders wrong for the latest-row join. Each matches steps 1 to 4.

### An experiment at realistic size

Does it matter at 20,000 orders, or only in a toy? The block below builds 2,000 customers with a random number of moves over a year, 20,000 orders, and 300 extra orders placed on the exact day a customer moved. It runs three joins in DuckDB: half-open (correct), closed with `BETWEEN`, and latest-row. It then corrupts 40 validity windows to test a gap and overlap check, and asks how many versions a weekly snapshot would see.

Versions used: Python 3.14.6, DuckDB 1.5.6, pandas 2.3.3, NumPy 2.5.3. The data is synthetic, generated by the seed in the block. It runs in about a second.

```python
import duckdb
import numpy as np
import pandas as pd

rng = np.random.default_rng(9)
regions = ["North", "South", "East", "West"]
n_customers, n_orders = 2000, 20000
days = 365

customers = pd.DataFrame({"customer_id": np.arange(n_customers), "region": rng.choice(regions, n_customers)})
moves = []
for cid, region in zip(customers.customer_id, customers.region):
    current = region
    for day in sorted(rng.choice(np.arange(1, days), size=rng.poisson(0.6), replace=False)):
        new = rng.choice([r for r in regions if r != current])
        moves.append((cid, int(day), new))
        current = new
changes = pd.DataFrame(moves, columns=["customer_id", "valid_from", "region"])
start = customers.assign(valid_from=0)[["customer_id", "valid_from", "region"]]
events = pd.concat([start, changes]).sort_values(["customer_id", "valid_from"]).reset_index(drop=True)
events["valid_to"] = events.groupby("customer_id")["valid_from"].shift(-1)
current_region = events.groupby("customer_id")["region"].last().rename("current_region")

orders = pd.DataFrame({
    "order_id": np.arange(n_orders),
    "customer_id": rng.integers(0, n_customers, n_orders),
    "day": rng.integers(0, days, n_orders),
    "amount": rng.gamma(2.0, 30.0, n_orders).round(2),
})
boundary = events.dropna(subset=["valid_to"]).sample(300, random_state=1)
extra = pd.DataFrame({
    "order_id": np.arange(n_orders, n_orders + len(boundary)),
    "customer_id": boundary.customer_id.values,
    "day": boundary.valid_to.astype(int).values,
    "amount": 50.0,
})
orders = pd.concat([orders, extra], ignore_index=True)

con = duckdb.connect()
con.register("orders", orders)
con.register("scd", events)
con.register("cur", current_region.reset_index())

truth = con.sql("""
    select o.order_id, o.amount, s.region from orders o
    join scd s on o.customer_id = s.customer_id
     and o.day >= s.valid_from and (s.valid_to is null or o.day < s.valid_to)
""").df()
latest = con.sql("select o.order_id, o.amount, c.current_region as region from orders o join cur c using (customer_id)").df()
closed = con.sql("""
    select o.order_id, o.amount, s.region from orders o
    join scd s on o.customer_id = s.customer_id
     and o.day between s.valid_from and coalesce(s.valid_to, 10000)
""").df()

print("orders", len(orders), "scd rows", len(events), "customers with history", int((events.groupby('customer_id').size() > 1).sum()))
print("half-open join rows", len(truth), "closed (BETWEEN) join rows", len(closed))
print("revenue true", round(orders.amount.sum(), 2), "closed join", round(closed.amount.sum(), 2),
      "inflation %", round(100 * (closed.amount.sum() / orders.amount.sum() - 1), 2))

merged = truth.merge(latest, on="order_id", suffixes=("_true", "_latest"))
wrong = (merged.region_true != merged.region_latest)
print("orders attributed to the wrong region by latest-row join", int(wrong.sum()), f"{100 * wrong.mean():.2f}%")

by = pd.DataFrame({
    "true": truth.groupby("region").amount.sum(),
    "latest": latest.groupby("region").amount.sum(),
}).round(0)
by["error_pct"] = (100 * (by.latest / by.true - 1)).round(2)
print(by)

def check(events):
    ordered = events.sort_values(["customer_id", "valid_from"])
    nxt = ordered.groupby("customer_id").valid_from.shift(-1)
    gaps = (ordered.valid_to < nxt).sum()
    overlaps = (ordered.valid_to > nxt).sum()
    return int(gaps), int(overlaps)

print("gaps/overlaps clean", check(events))
broken = events.copy()
idx = broken[broken.valid_to.notna()].sample(40, random_state=2).index
broken.loc[idx[:20], "valid_to"] += 3
broken.loc[idx[20:], "valid_to"] -= 1
print("gaps/overlaps after 40 corruptions", check(broken))

end = events.valid_to.fillna(days)
seen = np.ceil(events.valid_from / 7) * 7 < end
print("weekly snapshot sees", int(seen.sum()), "of", len(events), "versions; misses", int((~seen).sum()))
```

The output of the run:

```text
orders 20300 scd rows 3145 customers with history 862
half-open join rows 20300 closed (BETWEEN) join rows 20638
revenue true 1227800.55 closed join 1245149.75 inflation % 1.41
orders attributed to the wrong region by latest-row join 4807 23.68%
            true    latest  error_pct
region                               
East    316266.0  313417.0      -0.90
North   311459.0  305264.0      -1.99
South   291429.0  293174.0       0.60
West    308646.0  315946.0       2.37
gaps/overlaps clean (0, 0)
gaps/overlaps after 40 corruptions (20, 20)
weekly snapshot sees 3133 of 3145 versions; misses 12
```

**Reading the output.** The half-open join returns exactly one row per order (20,300). The closed join returns 20,638, so 338 orders were counted twice and revenue is inflated by 1.41%. The latest-row join returns one row per order too, so nothing looks broken, but 4,807 orders (23.68%) carry the wrong region. The check function finds all 20 gaps and all 20 overlaps that were injected.

**Line by line.**

- `events.groupby("customer_id")["valid_from"].shift(-1)` gives each version the start of the next one. That is the `valid_to`, and it makes gaps impossible by construction on clean data.
- The `boundary` rows are orders dated exactly on a `valid_to`. Without them the closed join would look harmless, because few random orders land on a move day.
- `np.ceil(events.valid_from / 7) * 7 < end` asks whether any weekly snapshot day falls inside a version's window.

### What the numbers say

The latest-row join put 23.68% of orders in the wrong region, yet region totals moved by only between -1.99% and +2.37%. Errors cancel: customers leave a region and arrive in another, so the totals stay plausible while individual rows are wrong. A dashboard check on totals would not catch this. A model trained on the per-order region would see wrong labels for nearly one order in four.

The closed join looks safe and is not: 338 extra rows and 1.41% too much revenue, from only 300 deliberately placed boundary orders plus a few chance ones. The weekly snapshot was kinder than expected, missing only 12 of 3,145 versions, because most customers stay put for months. Those 12 are the customers who moved twice within a week, and the history for them is simply wrong.

Limits: synthetic customers, one seed, a move rate of about 0.6 per year, and a boundary share that I chose. Real move rates and event timing differ, so quote the mechanism, not the percentages.

<Infographic src="/img/dm-enrich/dm2-scd-history.svg" alt="Bars for the share of orders in the wrong region, the error in each region total, and the revenue inflation from a closed join, measured on 20,300 synthetic orders." caption="Look first at the 23.68% bar: most of the damage hides inside region totals that are only a few percent off." />

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

A nightly snapshot sees the states present when it runs. If an order changes from pending to shipped and then cancelled within one day, the intermediate shipped state may never appear in the snapshot. That may be acceptable for a daily report but not for measuring shipping process time. Compare the required historical resolution with how often the source updates. Use an event log or CDC if the use case needs every transition. A Type 2 table is only as complete as the changes it observed.

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Joining facts to the current customer row | One row per customer is simple and the totals look right | Join to the version valid at event time. In the experiment 23.68%% of orders got the wrong region while totals moved under 2.4%% |
| Using `BETWEEN` on a validity window | `BETWEEN` reads naturally | Use `valid_from <= t AND t < valid_to`. The closed join returned 338 duplicate rows |
| Checking only totals after a join | Totals reconcile, so the join must be right | Also assert one match per fact row and run a gap and overlap check per key |
| Trusting a snapshot to hold every change | The table has a history column, so it must be complete | A snapshot sees the state at run time. Changes between runs are lost, so use change data capture when every transition matters |
| Changing a metric definition in place | Everyone gets the fix at once | Version the metric and compare old and new on a past period before migrating |

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

<details>
<summary><strong>Q6.</strong> (Medium) A join of 20,300 orders to a Type 2 table returns 20,638 rows. What is the most likely cause and how do you find the culprits?</summary>

Some orders match two versions, almost certainly because the window is closed at both ends or windows overlap. Group the joined result by order ID, keep those with a count above 1, and look at their timestamps: if they equal a `valid_to`, the window convention is the cause. Fix with `t < valid_to`.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) Region totals from a latest-row join are within 2.4% of the truth. Is that good enough to ship a model feature built the same way?</summary>

No. The totals are close because moves cancel out between regions. At row level 23.68% of orders carried the wrong region in the experiment. A model learns from rows, not totals, so it would train on wrong labels for roughly one order in four. Judge a join by row-level correctness, not by a reconciled total.

</details>

## Go deeper

- [dbt snapshots](https://docs.getdbt.com/docs/build/snapshots) documents Type 2 history and its configuration.
- [dbt Semantic Layer](https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl?version=2) describes shared metric definitions.
- [dbt data tests](https://docs.getdbt.com/docs/build/data-tests?version=1.12) lists core integrity checks.
- [dbt snapshots](https://docs.getdbt.com/docs/build/snapshots), opened 2026-10-09: the `timestamp` and `check` strategies, `dbt_valid_from` and `dbt_valid_to`, and the warning that snapshots must run on a schedule and miss changes between runs.
- [DuckDB AS OF join](https://duckdb.org/docs/current/guides/sql_features/asof_join), opened 2026-10-09: matches each row to the most recent row at or before its timestamp.
- Built from the course lecture "dm-l9-analytics-engineering" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can name the grain and key of a fact, dimension and mart before joining them.
- [ ] I can explain how three address changes create four Type 2 rows.
- [ ] I can write a half-open validity join and detect gaps or overlaps.
- [ ] I can state the business rules behind a shared metric and trace its inputs.
- [ ] I can show with three orders why a closed `BETWEEN` window double counts on the boundary day.
- [ ] I can explain why a latest-row join can be wrong for about a quarter of rows while region totals stay within a few percent.
- [ ] I can write a gap and overlap check for Type 2 windows and say what a snapshot cadence cannot see.

## Where to go next

Next: [Lecture 11, features and point-in-time correctness](/docs/mlops/data/features-and-point-in-time), which applies the same as-of idea to feature values for training. Related: [Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery), which schedules the snapshots and models this chapter relies on.
