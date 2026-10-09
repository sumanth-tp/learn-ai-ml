---
id: dm-dataops-and-reliability
title: "Data Management · Session 5 — DataOps and Reliability"
sidebar_label: "5 · DataOps"
sidebar_position: 2
slug: /mlops/data/dataops-reliability
description: "Run reproducible data infrastructure, test pipelines and interpret availability targets as consumer promises."
tags: [data-management, dataops, reliability, infrastructure-as-code]
---

import Infographic from '@site/src/components/Infographic';
import AvailabilityBudgetLab from '@site/src/components/viz/AvailabilityBudgetLab';

**In one line.** A data platform is dependable when changes are reproducible and failures are visible to its consumers.


:::tip Before you start

**You should already know**

- What a pipeline run and a retry are: [building reliable data pipelines](/docs/mlops/data/reliable-pipelines).
- Percentages of time: 1% of a 30-day month is 7.2 hours.
- What a freshness measure is: how old the newest data is.

**Reading time.** About 45 minutes, plus a second to run the simulation.

**After this chapter you can**

- turn an availability or freshness target into an error budget in minutes,
- explain what a burn rate is and why alert rules differ in noise and in what they miss,
- choose an alert rule for a step failure and a different one for a slow leak.

:::

## In 30 seconds

A smoke alarm that rings whenever someone makes toast gets its battery removed. One that only rings when the kitchen is actually on fire, but takes ten minutes to notice, is also a problem. A data feed has the same trade-off. An alert on every slow minute is noisy, an alert on a long outage is quiet, and neither notices a feed that is late 10% of the time for two days. An error budget says how much badness you can afford, and a burn rate says how fast you are spending it.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| SLI | A measured signal about service | Minutes where lag is under 15 |
| SLO | A target for that signal over a window | 99% of minutes over 30 days |
| SLA | A promise to a customer, with consequences | A credit if SLO is missed |
| Error budget | The bad amount the SLO allows | 1% of 43,200 minutes = 432 |
| Burn rate | How fast the budget is being spent, relative to even spending | 10% bad when 1% is allowed = 10 |
| Freshness | Age of the newest approved data | 5 minutes behind the source |
| Alert noise | Pages that did not need a person | A single slow minute |
| Detection delay | Time from the start of a fault to the page | 4 minutes |

## The idea in plain words

A data pipeline that ran correctly once is a demonstration. A data service that can be changed, recovered and explained repeatedly is an operation. DataOps applies version control, testing, deployment discipline and monitoring to data products. It also adds concerns that ordinary software deployments can miss: the exact data version, schema meaning, late corrections and the possibility that a task succeeds while its output is wrong.

Cloud infrastructure offers elastic compute, object storage, managed analytical stores and orchestration. **Infrastructure as Code** (IaC) describes resources in reviewable configuration so an environment can be recreated and changed deliberately. Containers package a job's runtime. Neither technique guarantees reproducible predictions on its own; the job must also identify its input snapshot, code, configuration, random seed and external dependencies.

<Infographic src="/img/dm/dataops-reliability.svg" alt="DataOps defines infrastructure and contracts, verifies data and recovery, then responds with owners and backfills; 99.9 per cent over 365 days permits 8.76 hours of idealised downtime." caption="Reliability combines controlled changes, data checks and consumer-visible recovery." />

The availability calculation is precise under a particular assumption. In a non-leap 365-day year there are 8,760 hours. At 99.9% availability, 0.1% unavailable time is **8.76 hours**. At 99.99%, 0.01% is **0.876 hours**, or 52.56 minutes. The allowed fraction is ten times smaller; the cost of meeting it depends on the architecture and failure modes, so it cannot be calculated from those percentages alone.

:::note Added for this site

The course lists IaC, containers, DataOps and SLA arithmetic. This chapter adds the distinction between SLI, SLO and SLA, the meaning of an error budget, data-version manifests, operational response and a simulated comparison of alert rules on a lagging feed.

:::

At the default 99.9% target over 365 days, the lab gives **8.760 hours** of equivalent complete downtime. Move the target to 99.99% to see **0.876 hours**. The calculation is an illustration; an actual agreement specifies its measurement window, exclusions and remedy.

<AvailabilityBudgetLab />


**What each control does.**

- **target availability** sets the percentage, from 99.00% to 99.99%.
- **period** sets the number of days over which the allowed downtime is computed.

**Try it yourself.**

1. Leave the defaults: 99.90% over 365 days gives 8.760 hours, the first calculation above.
2. Set 99.99%. The allowance falls tenfold to 0.876 hours, 52.6 minutes.
3. Set 99.00% over 30 days. You get 7.2 hours, which is the same 1% that gave the 432-minute budget in the experiment (432 / 60 = 7.2).

## Worked example, step by step

A feed should be fresh (lag under 15 minutes) for 99% of the minutes in a 30-day month.

1. The month has 30 × 24 × 60 = 43,200 minutes. The error budget is 1% of that: 43,200 × 0.01 = 432 bad minutes.
2. A single outage of 120 minutes spends 120 / 432 = 28% of the month's budget.
3. Burn rate is the bad fraction divided by the allowed fraction. If 5% of the last hour was bad, the burn rate is 0.05 / 0.01 = 5. A burn rate of 1 would spend the whole budget exactly at the end of the month.
4. A rule "page if the burn rate over the last hour is above 14.4 and over the last 5 minutes is above 14.4" needs 14.4% of the last hour bad, which is 0.144 × 60 = 8.64 bad minutes, and the last 5 minutes also bad.
5. A single slow minute is 1 bad minute in 60, a burn rate of 1 / 60 / 0.01 = 1.7, nowhere near the page threshold. A rule that pages on any bad minute would page on it anyway.
6. Availability arithmetic from earlier still applies: 99.9% over 365 days allows 8.76 hours, and as a count of daily deliveries it allows 365 × 0.001 = 0.365 missed deliveries.

In words: a budget turns "how reliable" into minutes you can spend, and a burn rate says whether you are spending them slowly or fast. The simulation below compares three alert rules on a feed with known faults.

## How it works

### Cloud, IaC, containers

Elastic compute + object storage + managed warehouse/lakehouse + orchestration; IaC (Terraform) for reproducibility, containers (Docker/K8s) for portability.

### Operate reliably

Version control (code + data), CI/CD for pipelines, automated testing, monitoring. Reliability stated as an SLA.

:::tip

**Worked.** 99.9% → (1−0.999)×365×24 = 8.76 h/yr; 99.99% → 0.876 h/yr.

:::


## A real system that works this way

**Terraform** is a concrete IaC example. Its official workflow is write, plan and apply. A plan shows the proposed changes before the infrastructure is modified; an apply performs them. For a data platform, this can describe storage, compute and access resources consistently across environments. It does not remove the need to review destructive changes or to check that secrets and permissions are appropriate. A perfectly reproducible misconfiguration remains a misconfiguration.

Imagine a daily revenue table used by a finance report and a demand model. The platform team changes the job's container image and its SQL transformation. A safe release records the code version, data source snapshot, schema change and infrastructure plan. It runs tests on realistic data, checks that yesterday's partition and a backfill still work, and publishes a consumer-facing status if the new job misses its delivery time. Rolling back the container without restoring or rebuilding affected data may leave the table in a mixed state.

The **Google SRE workbook** is a source for error-budget thinking. It distinguishes a target such as 99.9% from the operational question of how much unreliability consumers can tolerate in a defined period. For a data product, a useful measure may be "daily approved partition available by 08:00 UTC" or "fresh risk features under five minutes old" rather than generic host uptime. Choose a signal tied to the actual consumer harm.

## Code you can run

The first block reproduces both availability numbers. It treats unavailable time as if it could be represented by one complete outage, which is a useful budget illustration but not necessarily how a real SLA measures failures.

```python
hours_per_year = 365 * 24
for target in (0.999, 0.9999):
    allowed_hours = (1 - target) * hours_per_year
    print(f"{target:.2%}: {allowed_hours:.3f} hours, {allowed_hours * 60:.2f} minutes")

assert round((1 - 0.999) * hours_per_year, 3) == 8.760
assert round((1 - 0.9999) * hours_per_year, 3) == 0.876
```

The second block creates a small manifest for a reproducible input batch. The digest detects that an input changed; it does not prove that the original data was accurate or authorised for use. In production, store the manifest with the run and record the transformation and environment versions too.

```python
import hashlib
import json

rows = [
    {"event_id": "E1", "amount": 20},
    {"event_id": "E2", "amount": 30},
]

def digest(records):
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

manifest = {"source": "payments-2026-02-01", "rows": len(rows), "sha256": digest(rows)}
print(manifest["source"], manifest["rows"], manifest["sha256"][:12])
assert manifest["rows"] == 2
assert digest(rows) == manifest["sha256"]
corrected = [rows[0], {"event_id": "E2", "amount": 35}]
assert digest(corrected) != manifest["sha256"]
```

The hash uses a canonical JSON rendering of this ordered list. A real manifest must define row ordering, serialisation, partition boundaries and schema so that two systems calculate the same identity for the same logical dataset.

### Experiment: alert noise on a lagging feed

The simulation generates 20 months of per-minute lag for a feed. Normal lag is about 5 minutes. Brief slow batches (1 to 3 minutes of lag 20) occur at random, and each month has 4 outages of 20 to 239 minutes at lag 45 and one two-day creeping fault in which about 10% of minutes are slow. The SLO is 99% of minutes under 15 minutes of lag. Three alert rules are compared, and pages are grouped so that a rule that stays on for hours counts once. The two burn-rate windows (1 hour with 5 minutes, and 6 hours with 30 minutes) follow the first two rows of the Google SRE workbook's multiwindow table, opened on 2026-10-09. The data is synthetic. Run with NumPy 2.5.3 and pandas 2.3.3.

```python
import numpy as np
import pandas as pd

MINUTES, SLO, LIMIT, MONTHS = 30 * 24 * 60, 0.99, 15.0, 20

def simulate(seed):
    rng = np.random.default_rng(seed)
    lag = rng.lognormal(np.log(5), 0.25, MINUTES)
    for start in np.flatnonzero(rng.random(MINUTES) < 0.002):
        lag[start : start + rng.integers(1, 4)] = 20.0
    outage, creep = np.zeros(MINUTES, bool), np.zeros(MINUTES, bool)
    for start in rng.integers(0, MINUTES - 300, 4):
        length = int(rng.integers(20, 240))
        lag[start : start + length] = 45.0
        outage[start : start + length] = True
    start = int(rng.integers(0, MINUTES - 3000))
    creep[start : start + 2880] = True
    lag[creep & ~outage & (rng.random(MINUTES) < 0.10)] = 30.0
    return lag, outage, creep

def burn(bad, window):
    return pd.Series(bad.astype(float)).rolling(window, min_periods=1).mean().to_numpy() / (1 - SLO)

def alert_rules(lag):
    bad = lag > LIMIT
    run5 = pd.Series(bad.astype(float)).rolling(5).min().fillna(0).to_numpy() > 0
    fast = (burn(bad, 60) > 14.4) & (burn(bad, 5) > 14.4)
    slow = (burn(bad, 360) > 6) & (burn(bad, 30) > 6)
    return {"any bad minute": bad, "5 bad minutes in a row": run5, "burn rate 14.4 / 6": fast | slow}

def spans(mask):
    idx = np.flatnonzero(mask)
    return [] if idx.size == 0 else np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)

def grouped_pages(flag, quiet=120):
    pages, last = [], -quiet
    for minute in np.flatnonzero(flag):
        if minute - last >= quiet:
            pages.append(minute)
        last = minute
    return pages

stats = {}
for seed in range(MONTHS):
    lag, outage, creep = simulate(seed)
    for name, flag in alert_rules(lag).items():
        starts = grouped_pages(flag)
        record = stats.setdefault(name, {"pages": 0, "idle": 0, "out": 0, "out_hit": 0, "creep_hit": 0, "delay": []})
        record["pages"] += len(starts)
        record["idle"] += sum(not (outage | creep)[s] for s in starts)
        for span in spans(outage):
            hit = np.flatnonzero(flag[span[0] : span[-1] + 1])
            record["out"] += 1
            record["out_hit"] += bool(hit.size)
            record["delay"] += [hit[0]] if hit.size else []
        record["creep_hit"] += bool(flag[creep].any())

print(f"SLO {SLO:.0%} over 30 days allows {(1 - SLO) * MINUTES:.0f} bad minutes; {MONTHS} simulated months, 4 outages and 1 creeping fault each, pages grouped after 120 quiet minutes")
print(f"{'rule':24s} {'pages/month':>11s} {'idle pages':>10s} {'outages caught':>14s} {'median delay':>12s} {'creep caught':>12s}")
for name, r in stats.items():
    print(f"{name:24s} {r['pages'] / MONTHS:11.1f} {r['idle'] / r['pages']:10.0%} {r['out_hit'] / r['out']:14.0%} {np.median(r['delay']):9.0f} min {r['creep_hit'] / MONTHS:12.0%}")
```

**Reading the output.** `pages/month` is the average number of pages. `idle pages` is the share that started when neither an outage nor the creeping fault was under way. `outages caught` and `median delay` describe the four outages. `creep caught` is the share of months in which the rule fired during the two-day creeping fault.

**Line by line.**

- `burn(bad, window)` is the rolling share of bad minutes divided by the allowed 1%, which is exactly the burn rate.
- The `fast | slow` rule fires when both the long and the short window are above the threshold, so a short blip cannot fire it alone and a finished outage clears quickly.
- `grouped_pages` counts a page only when the rule has been quiet for 120 minutes, which mimics an on-call grouping window.
- `spans(outage)` splits the outage mask into separate events so each outage is scored once.

The printed output:

```text
SLO 99% over 30 days allows 432 bad minutes; 20 simulated months, 4 outages and 1 creeping fault each, pages grouped after 120 quiet minutes
rule                     pages/month idle pages outages caught median delay creep caught
any bad minute                  66.3        94%           100%         0 min         100%
5 bad minutes in a row           4.0         1%           100%         4 min          30%
burn rate 14.4 / 6               5.7        17%           100%         8 min         100%
```

### Reading the experiment

Paging on any bad minute caught everything and was useless: 66.3 pages a month, 94% of them idle. Four outages and one creeping fault are five real events a month, so more than 60 of those pages were noise.

Requiring 5 bad minutes in a row cut that to 4.0 pages a month with 1% idle and a 4-minute median delay. It looks like the winner, and for step failures it is. But it caught the creeping fault in only 30% of months, because a feed that is slow on 10% of minutes rarely has five slow minutes in a row, while the fault spends the budget at a burn rate of about 10.

The burn-rate rule caught the creeping fault every month and every outage, at the price of 5.7 pages a month, 17% of them idle, and a median delay of 8 minutes. The idle pages probably come from the long window staying above its threshold after an outage has ended; I did not isolate that cause. The honest comparison is that the simple run rule wins on step failures and the burn-rate rule wins when a fault is partial and slow. Limits: synthetic lag, fixed fault sizes, one SLO, 20 seeds, and pages grouped by my own 120-minute rule.

<Infographic src="/img/dm-enrich/alert-noise.svg" alt="A table compares three alert rules on pages per month, idle pages, outages caught and creeping faults caught, with cards on the 432-minute budget and the trade-off between the run rule and the burn-rate rule." caption="Look first at the creep column: the five-in-a-row rule catches 30% of creeping faults, the burn-rate rule 100%." />

## Designing with it

### Write a consumer-facing reliability objective

An **SLI** is a measured signal, such as the fraction of scheduled partitions approved by 08:00. An **SLO** is the target for that signal over a defined window. An **SLA** is an agreement that may include remedies or exclusions. These words are related but not interchangeable. A 99.9% availability figure is uninterpretable until the service, eligible requests or partitions, time window and exclusions are named. A source can be technically available while returning stale or invalid data, so include freshness and correctness where they drive decisions.

An error budget is the allowed fraction of bad outcomes under an SLO. Google SRE's examples use it to balance reliability work with change pace and to alert on significant budget consumption. For a daily dataset, a single missed partition may use a large share of a monthly budget. Consider both a fast alert for a critical missed delivery and a slower trend alert for repeated near misses. An alert without an owner and response path is just noise.

### Make changes reviewable and reversible

IaC helps describe infrastructure consistently. Review the proposed plan, especially deletions, permission changes and resource replacement. Keep environment-specific values explicit rather than hiding them in manually configured consoles. A container pins much of a job's runtime, but external APIs, source data and secret rotation still affect output. Pin dependency versions and retain the build artefact identifier. Test an upgrade against representative inputs and a prior known-good output before broad deployment.

Data changes need their own rollback strategy. If code version B rewrites a table, switching the worker back to A does not undo the rewritten rows. Publish a new immutable table snapshot or retain the previous approved version, then decide whether consumers should read it while B's data is repaired. Keep the decision and timestamps in a runbook.

### Version enough to explain an outcome

For a model run, record code commit or package version, input dataset snapshot or manifest, feature definitions, label cutoff, configuration, seed, training library versions and output artefact. "Version the data" does not mean putting every giant object into a code repository. It means a stable identifier and retention policy that let a reviewer retrieve the actual inputs. Hashes can detect changes, but a hash without a location and schema cannot reconstruct a dataset.

Test both technical and semantic properties. Unit tests can catch a wrong currency conversion function; schema tests can catch a renamed column; reconciliation can catch missing payments; a canary run can reveal unexpected latency or cost. Observe row counts, null rates, partition age and consumer errors after deployment. A green infrastructure plan and a green container build do not certify a correct data product.

## Investigate a missed data delivery

The revenue table is expected by 08:00, but the latest approved partition is yesterday's. First establish consumer impact: which reports and models read it, and whether they can safely use the previous partition. A stale-data indicator must reach those consumers; a silent fallback can make a report look current when it is not. The on-call engineer checks whether the source export arrived, whether validation rejected rows, whether transformation failed or whether publication succeeded but the catalogue did not update.

Suppose the source export arrived ten minutes after seven and the transform container crashed half an hour after seven, following a dependency upgrade. Reverting the image may restore the job, but only after verifying the partial output cannot be read. If the table uses snapshots, keep the last approved one visible and publish the corrected run atomically. If not, hold the consumer readiness marker until a clean rebuild finishes. Record the affected logical date and input hash so the repair is reproducible.

After restoring service, calculate the SLI honestly. If the partition became available at a quarter to nine, a "by 08:00" delivery SLI failed even though the table was correct later. If the consumer had a documented grace period or exclusion, apply the same rule consistently rather than changing it during an incident. Use the error budget to decide whether additional reliability work is required before more risky releases. The objective is consumer reliability, not a cosmetic uptime number.

Next make recurrence less likely. A predeployment smoke run could import the dependency and validate a sample batch. An integration test could compare the output schema and a few known totals. A release canary could process one date before enabling the daily schedule. Update the runbook with exact rollback and rebuild steps. If the failed dependency was not pinned, pin it or adopt a controlled update cadence. Every incident should improve a specific boundary or detection path.

The time-budget arithmetic itself has limits. A partial degradation may count as a fraction of failed requests, not complete downtime. Some SLAs exclude scheduled maintenance or use a month rather than a year. A leap year has 8,784 hours, changing the simple annual result. Use these numbers for a 365-day, continuously measured, complete-outage illustration and use the actual contract for operational decisions.

### Write a useful runbook

A runbook should tell the next operator how to recognise the failure, which consumers are affected and where the last approved data lives. Include a short decision tree: if the source is late, contact the source owner and mark the output stale; if validation fails, inspect the quarantine and do not publish; if transformation fails, check the image and dependency versions; if publication fails, verify whether the destination commit occurred before retrying. Link each step to a concrete query or dashboard. Avoid instructions that require guessing which account or region is in scope.

The runbook also needs a rollback and replay boundary. A model or dashboard may have cached data from a faulty partition. Rebuilding the warehouse table is not enough if those caches continue to serve the old result. List the downstream products to invalidate and the order in which to resume them. A recovery test should use a representative old date, because a backfill can exercise schema evolution and retention paths that a current-day run never touches.

### Calculate the right budget

If the objective is request availability, count good and total eligible requests. If the objective is a daily partition by 08:00, count on-time approved partitions over the agreed period. Multiplying a percentage by annual hours only suits a time-availability interpretation. For example, 99.9% of 365 daily deliveries allows less than one missed delivery in the mathematical average, but the contract must say how it handles the discrete count. State the denominator and rounding rule. This prevents a team from treating an 8.76-hour outage allowance as permission to miss eight daily deliveries.

High reliability may require redundant workers, isolated queues and faster failover, but cost does not scale mechanically with the number of nines. Some faults are removed cheaply by a validation rule; others need a substantial redesign. Use incident history and a consumer-impact model before buying redundancy. Data correctness can be the dominant risk even when compute uptime is excellent. Budget engineering effort across prevention, detection, repair and communication.

Store a manifest for each accepted run with source IDs, schema version, row counts, hash or snapshot ID, code version and output table version. That bundle lets a reviewer distinguish a changed input from a changed transform. Keep manifests under the same retention policy as the results they justify. If the input has been deleted by policy, say so explicitly; a digest alone cannot recreate it.

## Where this stands in 2026

:::info Industry view

- Terraform's current workflow continues to emphasise a reviewable plan before infrastructure changes are applied.
- SRE practice defines reliability from consumer-visible signals and uses an error budget to guide alerts and change decisions.
- DataOps needs both infrastructure reproducibility and data lineage because code rollback alone does not restore a changed dataset.

:::

## Common mistakes

1. **Paging on every bad sample.** It feels safe because nothing is missed. It gave 66.3 pages a month with 94% idle. Page on sustained or budget-burning behaviour.
2. **Using only a consecutive-minutes rule.** It is quiet and fast on outages. It missed 70% of the creeping faults. Add a burn-rate rule for partial failures.
3. **Copying burn-rate numbers without checking them.** The workbook gives 14.4 and 6 as starting points for a 99.9% SLO, and its budget shares (2% and 5%) assume a 30-day window. This chapter reused the thresholds with a 99% SLO, so tune them against your own objective and fault history.
4. **Reading an uptime percentage as a count of missed deliveries.** 99.9% over 365 days allows 8.76 hours of downtime, but only 0.365 missed daily deliveries. State the denominator.
5. **Ignoring data correctness in the SLI.** A feed can be fresh and wrong. Add a correctness or row-count signal to the objective.

## Practice questions

<details>
<summary><strong>Q1.</strong> What does Infrastructure as Code provide?</summary>

Declarative, version-controlled infrastructure (e.g. Terraform) so environments are reproducible, auditable and repeatable instead of hand-configured.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is DataOps?</summary>

Applying DevOps to data: version control (code + data), CI/CD for pipelines, automated testing, and monitoring for continuous, trustworthy data delivery.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A 99.9% availability SLA allows how much downtime per year?</summary>

(1−0.999)×365×24 = 0.001×8760 = 8.76 hours/year.<br /><em>Session 5 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How much does 99.99% allow, and what's the trade-off?</summary>

0.876 h/yr (≈53 min); the downtime allowance is ten times smaller, while the cost of meeting it depends on the architecture.<br /><em>Session 5 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why version data, not just code?</summary>

Because ML results depend on the exact data; data versioning enables reproducibility, rollback and debugging of pipelines and models.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q6. (Medium)</strong> An SLO says lag under 15 minutes for 99% of minutes in a 30-day month. How many bad minutes does it allow, and how much of it does a 120-minute outage spend?</summary>

The month has 43,200 minutes, so the budget is 432 bad minutes. A 120-minute outage spends 120 / 432 = 28%.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> The five-bad-minutes rule paged 4.0 times a month with 1% idle pages, yet it caught only 30% of creeping faults. Why, and what would you add?</summary>

A creeping fault makes about 10% of minutes slow, scattered rather than consecutive. The chance of five slow minutes in a row is small, so the rule stays silent while the budget burns at roughly 10 times the allowed rate. Add a burn-rate rule over a long window, such as 6 hours, which responds to the share of bad minutes and not to their adjacency.

</details>

## Go deeper

- [Terraform introduction](https://developer.hashicorp.com/terraform/intro) describes its configuration and plan workflow.
- [Google SRE error-budget policy](https://sre.google/workbook/error-budget-policy/) connects SLOs to operating decisions.
- [Google SRE alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) explains alert design around budget consumption.
- [Google SRE workbook, alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) (opened 2026-10-09) defines burn rate as how fast the service consumes the error budget relative to the SLO, and recommends for a 99.9% SLO a page at burn rate 14.4 over 1 hour with a 5-minute short window (2% of budget), a page at 6 over 6 hours with 30 minutes (5%), and a ticket at 1 over 3 days with 6 hours (10%). It presents these as starting points.
- Library versions run for the simulation: NumPy 2.5.3, pandas 2.3.3, Python 3.14.6.
- Built from the course lecture "dm-s5-infra-dataops" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can describe what IaC, containers and a data manifest each make reproducible.
- [ ] I can calculate 8.76 and 0.876 hours under the 365-day assumption.
- [ ] I can define an SLI, SLO and SLA for a specific data consumer.
- [ ] I can recover a failed data release without assuming code rollback repairs data.
- [ ] I can turn a percentage SLO into a budget of bad minutes and say how much one outage spends.
- [ ] I can explain a burn rate and why a long and a short window are combined.
- [ ] I can say which alert rule fits a step failure and which fits a slow leak, using measured pages and misses.

## Where to go next

Next is [data through the ML lifecycle](/docs/mlops/data/ml-lifecycle). For the observability side, see [observing data in production](/docs/mlops/data/data-observability).
