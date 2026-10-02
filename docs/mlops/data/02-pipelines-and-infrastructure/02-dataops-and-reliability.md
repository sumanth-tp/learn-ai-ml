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

## The idea in plain words

A data pipeline that ran correctly once is a demonstration. A data service that can be changed, recovered and explained repeatedly is an operation. DataOps applies version control, testing, deployment discipline and monitoring to data products. It also adds concerns that ordinary software deployments can miss: the exact data version, schema meaning, late corrections and the possibility that a task succeeds while its output is wrong.

The lecture sketches cloud infrastructure: elastic compute, object storage, managed analytical stores and orchestration. **Infrastructure as Code** (IaC) describes resources in reviewable configuration so an environment can be recreated and changed deliberately. Containers package a job's runtime. Neither technique guarantees reproducible predictions on its own; the job must also identify its input snapshot, code, configuration, random seed and external dependencies.

<Infographic src="/img/dm/dataops-reliability.svg" alt="DataOps defines infrastructure and contracts, verifies data and recovery, then responds with owners and backfills; 99.9 per cent over 365 days permits 8.76 hours of idealised downtime." caption="Reliability combines controlled changes, data checks and consumer-visible recovery." />

The lecture's availability calculation is precise under a particular assumption. In a non-leap 365-day year there are 8,760 hours. At 99.9% availability, 0.1% unavailable time is **8.76 hours**. At 99.99%, 0.01% is **0.876 hours**, or 52.56 minutes. The allowed fraction is ten times smaller; the cost of meeting it depends on the architecture and failure modes, so it cannot be calculated from those percentages alone.

:::note Beyond the lecture

The source lists IaC, containers, DataOps and SLA arithmetic. This chapter adds the distinction between SLI, SLO and SLA, the meaning of an error budget, data-version manifests and operational response.

:::

At the default 99.9% target over 365 days, the lab gives **8.760 hours** of equivalent complete downtime. Move the target to 99.99% to see **0.876 hours**. The calculation is an illustration; an actual agreement specifies its measurement window, exclusions and remedy.

<AvailabilityBudgetLab />

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

The first block reproduces both lecture numbers. It treats unavailable time as if it could be represented by one complete outage, which is a useful budget illustration but not necessarily how a real SLA measures failures.

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

At 08:00 the revenue table is expected, but the latest approved partition is yesterday's. First establish consumer impact: which reports and models read it, and whether they can safely use the previous partition. A stale-data indicator must reach those consumers; a silent fallback can make a report look current when it is not. The on-call engineer checks whether the source export arrived, whether validation rejected rows, whether transformation failed or whether publication succeeded but the catalogue did not update.

Suppose the source export arrived at 07:10 and the transform container crashed at 07:30 after a dependency upgrade. Reverting the image may restore the job, but only after verifying the partial output cannot be read. If the table uses snapshots, keep the last approved one visible and publish the corrected run atomically. If not, hold the consumer readiness marker until a clean rebuild finishes. Record the affected logical date and input hash so the repair is reproducible.

After restoring service, calculate the SLI honestly. If the partition became available at 08:45, a "by 08:00" delivery SLI failed even though the table was correct later. If the consumer had a documented grace period or exclusion, apply the same rule consistently rather than changing it during an incident. Use the error budget to decide whether additional reliability work is required before more risky releases. The objective is consumer reliability, not a cosmetic uptime number.

Next make recurrence less likely. A predeployment smoke run could import the dependency and validate a sample batch. An integration test could compare the output schema and a few known totals. A release canary could process one date before enabling the daily schedule. Update the runbook with exact rollback and rebuild steps. If the failed dependency was not pinned, pin it or adopt a controlled update cadence. Every incident should improve a specific boundary or detection path.

The time-budget arithmetic itself has limits. A partial degradation may count as a fraction of failed requests, not complete downtime. Some SLAs exclude scheduled maintenance or use a month rather than a year. A leap year has 8,784 hours, changing the simple annual result. Use the lecture's numbers for a 365-day, continuously measured, complete-outage illustration and use the actual contract for operational decisions.

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

## Go deeper

- [Terraform introduction](https://developer.hashicorp.com/terraform/intro) describes its configuration and plan workflow.
- [Google SRE error-budget policy](https://sre.google/workbook/error-budget-policy/) connects SLOs to operating decisions.
- [Google SRE alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) explains alert design around budget consumption.
- Built from the course lecture "dm-s5-infra-dataops" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can describe what IaC, containers and a data manifest each make reproducible.
- [ ] I can calculate 8.76 and 0.876 hours under the lecture's 365-day assumption.
- [ ] I can define an SLI, SLO and SLA for a specific data consumer.
- [ ] I can recover a failed data release without assuming code rollback repairs data.
