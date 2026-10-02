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

## The idea in plain words

A model trained on wrong, missing or delayed records can produce plausible predictions for the wrong population. Data quality is therefore **fitness for a particular use**, not a single property that a dataset either has or lacks. A postcode that is absent may be tolerable in one analysis and disqualifying in a delivery task. A purchase event arriving five minutes late may be fine for monthly reporting and too late for a real-time fraud decision.

The lecture names six dimensions: **accuracy, completeness, consistency, timeliness, validity and uniqueness**. Each asks a different question. Validity asks whether a value obeys a declared rule, such as a date being parsable; accuracy asks whether that date matches what happened. A valid but fabricated date is inaccurate. Completeness asks whether required values exist, while uniqueness asks whether records that should have one identity are duplicated. A row can pass one dimension and fail another.

<Infographic src="/img/dm/quality-dimensions.svg" alt="Six data-quality dimensions ask about truth, missingness, agreement, freshness, rules and duplicate keys; 950 of 1,000 present values give 95 per cent completeness." caption="Measure separate failure modes before deciding whether data is fit for a consumer." />

The lecture's 1,000-row example has 50 nulls in a required field. Completeness is (1,000 − 50)/1,000 = **95%**. If the contract requires 99%, the batch fails. That result is meaningful only if we say which field and population were measured. A table-wide "95% quality" score would hide whether the missing values are in a critical label, an optional note or one data source that affects a vulnerable customer group.

:::note Beyond the lecture

The lecture introduces dimensions, scorecards and assertions. The sections below add denominator design, failure routing, delayed labels and the difference between schema checks and real-world truth checks.

:::

The default lab reproduces the lecture: 50 null values among 1,000 rows give **95.0%** completeness and fail a **99%** requirement. Move either slider to see that a metric and a threshold together produce an action.

<DataQualityRulesLab />

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

Start with the lecture's arithmetic. Count rows in the denominator and non-null values in the numerator. A threshold uses `>=` so exactly 99% passes a 99% rule.

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

## Go deeper

- [dbt data tests](https://docs.getdbt.com/docs/build/data-tests?version=1.12) documents the four built-in checks and their violating-row behaviour.
- [Great Expectations uniqueness guide](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/uniqueness/) gives examples of key and compound-key checks.
- Built from the course lecture "dm-s2-principles" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can name the six dimensions and explain why validity and accuracy differ.
- [ ] I can reproduce 950/1,000 = 95% and compare it with a 99% requirement.
- [ ] I can state the denominator, population and missingness predicate behind a quality score.
- [ ] I can route a failed rule to a proportionate action with an owner and evidence.
