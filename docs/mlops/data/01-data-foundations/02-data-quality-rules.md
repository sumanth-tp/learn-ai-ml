---
id: dm-data-quality-rules
title: "Data Management · Session 2 — Measure Data Quality"
sidebar_label: "2 · Data quality"
sidebar_position: 2
slug: /mlops/data/quality-rules
description: "Measure six data-quality dimensions and turn them into explicit pipeline decisions."
tags: [data-management, data-quality, validation, mlops]
---

import Infographic from '@site/src/components/Infographic';
import DataQualityRulesLab from '@site/src/components/viz/DataQualityRulesLab';

**In one line.** Data quality becomes useful when a rule says what to measure and what happens when it fails.


:::tip Before you start

**You should already know**

- What a column, a null value and a primary key are.
- How a file format fixes a schema: [data representations](/docs/mlops/data/representations-for-ml).
- Basic pandas: filtering a DataFrame with a boolean mask.

**Reading time.** About 40 minutes, plus a second to run the experiment.

**After this chapter you can**

- compute a completeness score and compare it with a threshold,
- measure the precision and recall of a quality rule against defects you injected,
- explain why a coercing schema can hide the very errors it was meant to catch.

:::

## In 30 seconds

A delivery company checks every parcel label before the van leaves. Is the postcode there (completeness)? Is it a real postcode (validity)? Is it the postcode the customer actually lives at (accuracy)? Is the same parcel listed twice (uniqueness)? Each question catches a different mistake, and a label can pass three and fail the fourth. Data quality rules are those label checks, written as code so a pipeline can run them on every batch and act on the answer.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Completeness | Share of required values that are present | 950 of 1,000 present = 95% |
| Validity | A value obeys a declared rule | A country code is one of GB, IN, US, DE |
| Uniqueness | A key appears once | One row per `order_id` |
| Accuracy | A value matches the real world | The amount equals what the customer paid |
| Precision of a rule | Of the rows it flags, the share that are real defects | 180 flagged, 180 real = 1.0 |
| Recall of a rule | Of the real defects, the share it flags | 180 of 200 real = 0.9 |
| Coercion | Converting a value to the declared type, even if it loses information | 19.99 becomes 19 |
| Quarantine | Setting failed rows aside with a reason, instead of dropping them | A `rejected` table |

## The idea in plain words

A model trained on wrong, missing or delayed records can produce plausible predictions for the wrong population. Data quality is therefore **fitness for a particular use**, not a single property that a dataset either has or lacks. A postcode that is absent may be tolerable in one analysis and disqualifying in a delivery task. A purchase event arriving five minutes late may be fine for monthly reporting and too late for a real-time fraud decision.

Six dimensions describe quality: **accuracy, completeness, consistency, timeliness, validity and uniqueness**. Each asks a different question. Validity asks whether a value obeys a declared rule, such as a date being parsable; accuracy asks whether that date matches what happened. A valid but fabricated date is inaccurate. Completeness asks whether required values exist, while uniqueness asks whether records that should have one identity are duplicated. A row can pass one dimension and fail another.

<Infographic src="/img/dm/quality-dimensions.svg" alt="Six data-quality dimensions ask about truth, missingness, agreement, freshness, rules and duplicate keys; 950 of 1,000 present values give 95 per cent completeness." caption="Measure separate failure modes before deciding whether data is fit for a consumer." />

A 1,000-row example with 50 nulls in a required field gives: completeness is (1,000 − 50)/1,000 = **95%**. If the contract requires 99%, the batch fails. That result is meaningful only if we say which field and population were measured. A table-wide "95% quality" score would hide whether the missing values are in a critical label, an optional note or one data source that affects a vulnerable customer group.

:::note Added for this site

The course introduces dimensions, scorecards and assertions. The sections below add denominator design, failure routing, delayed labels, the difference between schema checks and real-world truth checks, and a measured test of what each rule type catches.

:::

The default lab reproduces this example: 50 null values among 1,000 rows give **95.0%** completeness and fail a **99%** requirement. Move either slider to see that a metric and a threshold together produce an action.

<DataQualityRulesLab />

**What each control does.**

- **null values** sets how many of the 1,000 rows have a missing required value, 0 to 150.
- **required completeness** sets the threshold in per cent, 90 to 100.

**Try it yourself.**

1. Defaults: 50 nulls gives 95.0% completeness, which fails the 99% requirement.
2. Set nulls to 10. Completeness is exactly 99.0%, and the rule passes because the test is "at least".
3. Set nulls to 150 and the requirement to 90. Completeness is 85.0%, so the rule still fails. Lowering a threshold changes the decision only if the data is close to it.

## Worked example, step by step

Twenty thousand orders, and we deliberately damage 200 of them in each of five ways. The question is how well each rule finds its own damage.

1. Completeness first. If 50 of 1,000 required values are null, completeness is 950 / 1,000 = 95%, below a 99% requirement, so the batch fails.
2. A `not null` rule on `amount` flags every null amount. All 200 flagged rows are the injected nulls, so precision is 200 / 200 = 1.0 and recall is 200 / 200 = 1.0.
3. A range rule `0 <= amount <= 1000` is meant to catch rows whose amount was multiplied by 100 (pounds stored as pence). A 8.50 pound order becomes 850, which is still under 1,000, so the rule misses it. A 12.00 pound order becomes 1,200 and is caught. If 180 of 200 such rows exceed 1,000, precision is 180 / 180 = 1.0 and recall is 180 / 200 = 0.9.
4. A uniqueness rule on `order_id` flags both copies of a duplicated key, because it cannot tell which is the original. With 200 duplicated keys that is about 400 rows flagged for 200 real extra rows, so precision is near 200 / 400 = 0.5.
5. Add the precision and recall columns for every rule, and a rule that looks perfect on one defect type can still miss another type entirely.

In words: precision says how many alarms are false, recall says how many real problems slip through, and a rule is only as good as the defect it was written for. The experiment below computes steps 2 to 4 on the 20,000 rows.

## How it works

### Six dimensions of quality

- **Accuracy · Completeness**; Matches reality; no missing values.
- **Consistency · Timeliness**; No contradictions; fresh enough.
- **Validity · Uniqueness**; Conforms to rules; no duplicates.

:::tip

**Worked.** 1000 rows, 50 nulls → completeness = 950/1000 = 0.95 (fails a 99% rule).

:::

### Rules & scorecards

Assertions (not-null, range, regex, referential integrity, uniqueness) run in the pipeline, producing a per-dimension scorecard; failures alert or quarantine; shifting quality left.


## A real system that works this way

**dbt data tests** provide a concrete implementation pattern. Its built-in tests include `not_null`, `unique`, `accepted_values` and `relationships`. A test queries for violating rows; an empty result passes. These checks are useful at the transformation boundary because they make assumptions about IDs, required fields and references executable. They do not prove that the source described the real world correctly. An order with a unique, non-null customer ID can still be assigned to the wrong person.

Consider a daily customer-risk feature table. A transformation joins transactions, accounts and a label table by customer ID. An `accepted_values` check can reject an unexpected account status, and a `relationships` check can detect transactions pointing to accounts that do not exist in the reference table. A freshness check can ask whether yesterday's partition arrived by a specified time. An accuracy audit must compare sampled records with the system of record or a trusted human review. Different questions need different evidence.

**Great Expectations** documents similar declarative checks, including uniqueness. The point is not that one library is mandatory. A plain SQL assertion or a small Python check may be enough for a local pipeline. The important property is that failures carry enough context to decide whether to reject a batch, quarantine specific rows, warn a consumer or investigate a source. A dashboard that reports "pass" without showing the population, rule version and exceptions is easy to misread.

## Code you can run

Start with the arithmetic. Count rows in the denominator and non-null values in the numerator. A threshold uses `>=` so exactly 99% passes a 99% rule.

```python
rows = 1000
nulls = 50
required = 0.99
complete = rows - nulls
score = complete / rows
passes = score >= required
print(f"complete={complete}/{rows} = {score:.0%}; pass={passes}")
assert complete == 950
assert score == 0.95
assert not passes
```

The next example keeps rule results separate so that a passing validity check cannot hide a failing uniqueness check. The data is synthetic. An empty batch is reported as "not evaluated" rather than given a misleading 100% score.

```python
records = [
    {"customer_id": "C1", "status": "active", "balance": 20},
    {"customer_id": "C2", "status": "closed", "balance": 0},
    {"customer_id": "C2", "status": "active", "balance": 15},
    {"customer_id": "C4", "status": "other", "balance": 5},
]

def check_batch(rows):
    if not rows:
        return {"evaluated": False, "reason": "empty batch"}
    ids = [row["customer_id"] for row in rows]
    return {
        "evaluated": True,
        "id_completeness": sum(bool(item) for item in ids) / len(rows),
        "id_unique": len(ids) == len(set(ids)),
        "status_valid": all(row["status"] in {"active", "closed"} for row in rows),
        "balance_valid": all(row["balance"] >= 0 for row in rows),
    }

result = check_batch(records)
print(result)
assert result["id_completeness"] == 1.0
assert not result["id_unique"]
assert not result["status_valid"]
assert result["balance_valid"]
assert not check_batch([])["evaluated"]
```

This output is a small scorecard, not a complete quality certificate. The duplicate C2 might be a true duplicate or two legitimate versions. The contract must say which one. `other` might be invalid today or a new status awaiting a schema change. Investigate before silently deleting rows.

### Experiment: what each rule catches

The experiment generates 20,000 synthetic orders, damages 200 rows in each of five ways, validates them with pandera 0.34.0 and scores every rule against the truth it knows. It then shows what happens when a schema coerces types instead of rejecting them. Run it with pandas 2.3.3 and NumPy 2.5.3. No external data is used.

```python
import numpy as np
import pandas as pd
import pandera.pandas as pa

rng = np.random.default_rng(7)
n = 20_000
clean = pd.DataFrame({
    "order_id": np.arange(n),
    "country": rng.choice(["GB", "IN", "US", "DE"], n),
    "amount": rng.gamma(2.0, 20.0, n).round(2),
    "age": rng.integers(18, 90, n),
})
data = clean.copy()
names = ["null_amount", "negative_amount", "bad_country", "duplicate_id", "pence_not_pounds"]
truth = {name: np.zeros(n, bool) for name in names}
for name, block in zip(names, np.array_split(rng.permutation(n)[:1000], 5)):
    truth[name][block] = True
data.loc[truth["null_amount"], "amount"] = np.nan
data.loc[truth["negative_amount"], "amount"] *= -1
data.loc[truth["bad_country"], "country"] = "UK"
data.loc[truth["duplicate_id"], "order_id"] -= 1
data.loc[truth["pence_not_pounds"], "amount"] *= 100

schema = pa.DataFrameSchema({
    "order_id": pa.Column(int, unique=True),
    "country": pa.Column(str, pa.Check.isin(["GB", "IN", "US", "DE"])),
    "amount": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1000)], nullable=False),
})
try:
    schema.validate(data, lazy=True)
except pa.errors.SchemaErrors as err:
    cases = err.failure_cases.dropna(subset=["index"])
rows_for = lambda column, check: set(cases.loc[(cases["column"] == column) & (cases["check"] == check), "index"].astype(int))
rules = {
    "not_nullable": (rows_for("amount", "not_nullable"), ["null_amount"]),
    "ge(0)": (rows_for("amount", "greater_than_or_equal_to(0)"), ["negative_amount"]),
    "le(1000)": (rows_for("amount", "less_than_or_equal_to(1000)"), ["pence_not_pounds"]),
    "isin": (rows_for("country", "isin(['GB', 'IN', 'US', 'DE'])"), ["bad_country"]),
    "unique": (rows_for("order_id", "field_uniqueness"), ["duplicate_id"]),
}
print(f"{'rule':14s} {'flagged':>7s} {'precision':>9s} {'recall':>7s}")
for rule, (flagged, targets) in rules.items():
    wanted = set(np.flatnonzero(np.any([truth[t] for t in targets], axis=0)))
    hit = len(flagged & wanted)
    print(f"{rule:14s} {len(flagged):7d} {hit / max(len(flagged), 1):9.3f} {hit / len(wanted):7.3f}")

pence = truth["pence_not_pounds"]
log_amount = np.log(data["amount"].clip(lower=0.01))
reference = np.log(clean["amount"])
outlier = (log_amount - reference.median()).abs() > 4 * reference.std()
print(f"log-outlier rule: catches {int((outlier & pence).sum())} of {int(pence.sum())} pence rows, {int((outlier & ~pence).sum())} false alarms")

truncated = pa.DataFrameSchema({"amount": pa.Column(int, coerce=True)}).validate(clean[["amount"]])
print(f"coerce to int: mean amount {clean['amount'].mean():.3f} becomes {truncated['amount'].mean():.3f}, {(truncated['amount'] != clean['amount']).mean():.1%} of rows changed")
text = clean["amount"].astype(str).where(rng.random(n) > 0.03, clean["amount"].astype(str).str.replace(".", ",", regex=False))
parsed = pd.to_numeric(text, errors="coerce")
print(f"to_numeric(errors='coerce'): {parsed.isna().mean():.1%} become NaN, sum {parsed.sum():,.0f} against {clean['amount'].sum():,.0f}; fillna(0) hides it")
try:
    pa.DataFrameSchema({"amount": pa.Column(float, coerce=True)}).validate(pd.DataFrame({"amount": text}))
except pa.errors.SchemaErrors as err:
    print("pandera float coerce on the same text raises:", type(err).__name__)
```

**Reading the output.** The table has one row per rule. `flagged` is how many rows the rule marked, `precision` is the share of those that are real injected defects of the type the rule targets, and `recall` is the share of that type the rule found. Then come the extra distribution rule, and three lines about coercion.

**Line by line.**

- `schema.validate(data, lazy=True)` collects every failure instead of stopping at the first. Without `lazy=True` only the first failing check would appear in the exception.
- `err.failure_cases` is a table of failing values with the row `index` and the `check` name, which is what lets the code score each rule separately.
- `pa.Column(int, coerce=True)` is the dangerous line. It converts the column to integers before checking anything.
- `to_numeric(..., errors='coerce')` turns text it cannot read into NaN, and the later `fillna(0)` idea would hide those NaNs entirely.

The printed output:

```text
rule           flagged precision  recall
not_nullable       200     1.000   1.000
ge(0)              200     1.000   1.000
le(1000)           180     1.000   0.900
isin               200     1.000   1.000
unique             396     0.500   0.990
log-outlier rule: catches 186 of 200 pence rows, 240 false alarms
coerce to int: mean amount 40.061 becomes 39.567, 99.1% of rows changed
to_numeric(errors='coerce'): 3.1% become NaN, sum 777,725 against 801,225; fillna(0) hides it
pandera float coerce on the same text raises: SchemaErrors
```

### Reading the experiment

Four of the five rules are perfect on their own defect: null, negative and bad-country checks all show precision and recall of 1.000. The surprise is the range rule. It was written for the pounds-as-pence error and catches only 0.900 of it, because a cheap order multiplied by 100 still falls inside the allowed range. A rule that checks a legal range cannot see a wrong value that happens to be legal. This is the validity against accuracy distinction from earlier in the chapter, measured.

The uniqueness rule has recall 0.990 but precision 0.500: it flagged 396 rows for 200 injected duplicates, since pandera marks every row sharing a repeated key. Precision here is a property of what you count as a defect, so decide whether the pipeline should quarantine both rows or keep the first. A distribution rule on log amounts recovers 186 of 200 pence rows, but at the price of 240 false alarms, a precision of 186 / 426 = 0.44.

Coercion is the quiet danger. Coercing the amounts to integers moved the mean from 40.061 to 39.567 and altered 99.1% of the rows, with no error. Parsing text with `errors='coerce'` turned 3.1% of values into NaN and, once those become 0, understated the sum by 23,500 (777,725 against 801,225). The strict float coercion in pandera raised instead. Limits: synthetic defects, one seed, rules tuned by hand, and a single table.

<Infographic src="/img/dm-enrich/quality-rule-recall.svg" alt="Bars show precision and recall of five validation rules against injected defects, with cards for the range rule's missed pence rows, the uniqueness rule's two flagged copies and the cost of silent coercion." caption="Look first at the le(1000) bar: a range rule catches 0.900 of the pounds-as-pence defects, not all of them." />

## Designing with it

### Define the population and the denominator

Completeness for a field is the share of rows where that field is present under a stated null policy. Empty strings, whitespace and sentinel values such as `-1` may be missing in practice even though they are not SQL `NULL`. Define the missingness predicate, the partition or time window and the entities included. For an ML label that arrives a week after an event, today's unlabeled records should not automatically count as a quality failure; compare records old enough to have a label.

Segment important checks. A global 99.5% completeness score can conceal 50% missingness in a small source or market. Slice by source, event type and time where there is a plausible failure mode. Avoid indiscriminate slices with tiny sample sizes: a one-record subgroup produces a volatile percentage. Carry both numerator and denominator next to the percentage, then add a minimum count or confidence interval when decisions depend on it.

### Make each dimension operational

| Dimension | Example rule | What the rule cannot prove |
| --- | --- | --- |
| Accuracy | Reconcile sampled balances to a trusted ledger | An untested record is correct |
| Completeness | Required ID present for at least 99% of eligible rows | Optional fields are useful |
| Consistency | Account state agrees across two systems at the same cutoff | Either system is accurate |
| Timeliness | Latest accepted event is under 15 minutes old | Events cover the whole population |
| Validity | Status belongs to the current allowed set | The status reflects reality |
| Uniqueness | One current record per customer key | A key maps to the right person |

Write checks close to the boundary where an error first becomes visible. A parse error belongs at ingestion; an impossible feature value belongs before training and serving; a train/serve mismatch belongs at model integration. Repeating a cheap critical check at more than one boundary is sensible because a later transformation can introduce a new error.

### Choose the failure action

For each rule, decide whether to block the whole batch, quarantine bad rows, send a warning or continue with a documented fallback. A required label missing in half the training examples may justify a stop. A rare malformed optional note may be quarantined without delaying a daily aggregate. The action should match the consumer's loss, not a blanket rule that all invalid rows are dropped. Keep quarantined records with source IDs and reasons so a correction can be replayed and measured.

An alert needs an owner and a decision clock. "Freshness below 99%" is less useful than "the payments partition for 2026-02-01 has not arrived by 08:00 UTC; the risk feature table is stale; use the last approved snapshot for at most one hour". State how a consumer learns that data was degraded. A successful pipeline task can still have bad data if it merely moved all files without checking them.

## Build a quality contract for one feature table

Suppose a lending model uses `recent_missed_payments`. First identify the source and the event that counts as a miss. A payment scheduled for Friday but settled on Monday may be late under one policy and on time under another. The business definition belongs in the feature contract, alongside the entity key, lookback window, time zone and expected publication delay. If policy changes, version the definition and track which model versions consumed it.

Next choose quality checks from the failure modes. Require account IDs and event times; test that account IDs link to a known account snapshot; reject events with impossible negative amounts if the event type forbids them; compare the daily event count with a recent baseline; check the last source watermark. None of these alone establishes accuracy. Sampled reconciliation against the payment ledger is the evidence for whether the event states reflect the underlying transactions. Keep the sample method and exceptions so the audit can be repeated.

Treat a quality score as an observation, not a substitute for a release decision. A 95% completeness number says little without knowing which 5% are missing and what the model does when values are absent. If missingness is concentrated among new customers, a model may behave poorly precisely where product risk is highest. Monitor quality by relevant cohorts and compare predictions for missing versus complete records. A single green tile can hide a systematic gap.

When a rule fails, preserve causal evidence. Store the source object or event IDs, ingestion run, rule version, sample of failed records and the downstream tables affected. If an upstream team fixes the issue, recompute the affected partitions and record whether the resulting training data changes. Data quality is not complete at detection; it includes repair, communication and prevention. A post-fix test should prove the expected records now pass and that the fix did not create duplicates elsewhere.

Finally, distinguish drift from quality failure. A customer population may change legitimately while every record remains valid. A feature distribution shift can be a warning for model performance, but blocking all novel values may erase precisely the new behaviour the model needs to see. Chapter 16 returns to drift and observability. Here the rule is to write down which departures mean malformed data, which mean a new valid pattern and who decides when the schema or model must change.

### Avoid false reassurance from a single score

Suppose a scorecard averages five checks, each weighted equally. Four pass completely and one critical label check fails completely, producing an 80% total. Another batch might score 80% because each check has a small, diffuse failure. Those batches need different actions. Keep the individual results and their affected row IDs; use an overall score only as a navigational summary. Weighting a score requires a stated consumer cost, not a cosmetic desire for one number.

Quality rules can also create a selection effect. If a pipeline drops every record with a missing income field, the resulting training set may overrepresent customers who completed one form. The model could appear accurate in validation but fail on applicants who use a different channel. Record how many rows each rule removes by cohort and compare the retained sample with the population the model will serve. Sometimes the right response is a missingness indicator or a separate review path rather than deletion.

Make remediation measurable. For every recurring failure, assign a source owner, expected fix time and a regression check. Track both the number of bad records and the time until consumers receive corrected data. If a batch is quarantined, define whether the next successful run automatically replays it or whether a person must approve replay. Otherwise an alert can be resolved while the historical gap remains in the feature table. Record that decision beside the failed batch so the next operator can act consistently.

## Where this stands in 2026

:::info Industry view

- Declarative data tests are part of common analytics workflows; dbt's built-in tests cover nulls, uniqueness, accepted values and relationships.
- Expectation-based tools such as Great Expectations can report violating records and help turn assumptions into visible rules.
- A quality threshold is consumer-specific. ML teams should pair field checks with freshness, lineage and downstream model monitoring.

:::

## Common mistakes

1. **Trusting a range rule to catch a unit error.** A legal range feels like a safety net. Pence stored as pounds still passed for 20 of 200 rows. Add a distribution check against a reference, or reconcile with a trusted source.
2. **Using `coerce=True` to make a schema pass.** It feels tidy because validation stops failing. It truncated 19.99 to 19 for 99.1% of rows. Coerce only when you have checked what the conversion destroys, and prefer a strict check.
3. **Parsing text with `errors='coerce'` and filling the gaps.** The pipeline stays green. 3.1% of values vanished and the sum fell by 23,500. Count the failed parses and fail the batch above a stated share.
4. **Reading a uniqueness failure as one bad row.** The rule flags every copy. Decide in the contract which copy survives.
5. **Reporting one overall score.** Four perfect rules and one weak rule average to a comfortable number. Keep each rule's precision, recall and affected rows.

## Practice questions

<details>
<summary><strong>Q1.</strong> What is data quality and why measure it?</summary>

The degree to which data is fit for its intended use; it must be measured because poor quality silently degrades models, biases predictions and wastes effort.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Name the six dimensions of data quality.</summary>

Accuracy, completeness, consistency, timeliness, validity, uniqueness.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A 1000-row column has 50 nulls. Compute completeness.</summary>

(1000−50)/1000 = 950/1000 = 0.95 (95%); fails a 99% rule.<br /><em>Session 2 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How is data quality enforced in a pipeline?</summary>

With data quality rules/assertions (not-null, range, regex, referential integrity, uniqueness) that produce a scorecard; failures alert or quarantine data (shift left).<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Distinguish validity from accuracy.</summary>

Validity = conforms to format/range/rules (e.g. a date is well-formed); accuracy = matches the real-world truth (the date is the correct one).<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q6. (Medium)</strong> A range rule flagged 180 rows, all of them real unit errors, but 200 unit errors existed. Give its precision and recall and say what the missing 20 look like.</summary>

Precision is 180 / 180 = 1.0 and recall is 180 / 200 = 0.9. The missing 20 are cheap orders whose amount multiplied by 100 is still inside the allowed range, such as 8.50 becoming 850. The rule checks legality, not correctness.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> Coercing amounts to integers changed the mean from 40.061 to 39.567 without any error. Why is that a quality failure and how would you catch it?</summary>

The conversion truncates every fractional amount, so values are systematically lower, and a downstream total or a feature mean is biased. No rule failed because the type check now passes. Catch it by keeping the column as float with a strict type check, or by comparing the column's mean and sum before and after the conversion and failing when they move.

</details>

## Go deeper

- [dbt data tests](https://docs.getdbt.com/docs/build/data-tests?version=1.12) documents the four built-in checks and their violating-row behaviour.
- [Great Expectations uniqueness guide](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/uniqueness/) gives examples of key and compound-key checks.
- [pandera DataFrame schemas](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html) (opened 2026-10-09) documents `coerce=True`, which coerces a column to the declared dtype before checks run, notes that integer columns cannot hold NaN, and describes `lazy=True` collection of errors.
- Library versions run for the experiment: pandera 0.34.0, pandas 2.3.3, NumPy 2.5.3, Python 3.14.6.
- Built from the course lecture "dm-s2-principles" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can name the six dimensions and explain why validity and accuracy differ.
- [ ] I can reproduce 950/1,000 = 95% and compare it with a 99% requirement.
- [ ] I can state the denominator, population and missingness predicate behind a quality score.
- [ ] I can route a failed rule to a proportionate action with an owner and evidence.
- [ ] I can compute precision and recall for a validation rule against known defects.
- [ ] I can explain why a legal-range rule misses a unit error that lands inside the range.
- [ ] I can show what silent type coercion changes in a column and how to make it fail loudly.

## Where to go next

Next is [warehouses, lakes and lakehouses](/docs/mlops/data/warehouses-lakes-and-lakehouses), where these rules sit between bronze and silver layers. Chapter 8, [profiling, validation and drift](/docs/mlops/data/profiling-validation-drift), measures drift rules the same way.
