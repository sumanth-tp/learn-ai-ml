---
id: dm-experiments-metadata-and-lineage
title: "Data Management · Lecture 12 — Experiments, Metadata and Lineage"
sidebar_label: "12 · Experiments and lineage"
sidebar_position: 2
slug: /mlops/data/experiments-metadata-lineage
description: "Compare tracked model runs, trace upstream data and make an evidence-based registry decision."
tags: [data-management, experiment-tracking, lineage, mlflow]
---

import Infographic from '@site/src/components/Infographic';
import ExperimentChoiceLab from '@site/src/components/viz/ExperimentChoiceLab';

**In one line.** A metric becomes useful when its data, evaluation method and model artifact can be traced and challenged.

## The idea in plain words

Machine learning is empirical: a model is chosen from experiments whose results must be compared. A notebook number on its own is fragile. Which data snapshot produced it? Was the holdout truly separate? What code and configuration ran? Which artifact was measured? **Experiment tracking** records these relationships so another person can inspect a claim and attempt to repeat it. **Metadata and lineage** connect the run to upstream datasets, features and downstream releases.

The source has three runs with F1 scores **0.71, 0.76 and 0.74**. A tracker can sort them and show that run 2 has the largest reported score. That is only the beginning of a selection decision. The runs must use a comparable evaluation set and metric definition; the measurement needs uncertainty and subgroup checks; deployment may have latency, cost or safety limits. A logged configuration helps reconstruction but does not make a biased evaluation valid. A registry entry should record the evidence and approval decision, not merely promote whichever number is largest.

<Infographic src="/img/dm/experiment-metadata.svg" alt="Three runs report F1 scores of 0.71, 0.76 and 0.74; lineage connects source data, features, run and model version, while release review checks leakage and latency." caption="Tracking makes a reported result inspectable; a release decision still tests the result's meaning." />

Lineage is a graph of what was used and what was produced. A source table feeds a cleaned dataset, which feeds a feature set, which feeds a training run and model version. If a source field changes, the graph helps identify affected descendants. A data catalog adds searchable descriptions, owners, schemas, freshness and quality signals. These tools shorten an investigation, provided the metadata is complete and updated by the actual jobs.

:::note Beyond the lecture

The source states that tracking makes the top run trustworthy and exact reproducibility follows from pinned code, data, configuration and environment. The sections below qualify those claims: evaluation design and deployment constraints determine trust, while hardware, nondeterministic operations and external services can prevent bit-for-bit reproduction.

:::

The lab starts with the lecture's **0.71, 0.76 and 0.74 F1 scores**. With an 80 ms latency budget, run 2 is eligible and has the highest score. Tighten the budget to 60 ms and run 3 becomes the best eligible candidate. The control demonstrates why a metric ranking needs release constraints.

<ExperimentChoiceLab />

## How it works

### Experiments & A/B

Log params, metrics, code + data version, artifacts (MLflow/W&B) so runs compare and the best reproduces; A/B test in production.

:::tip

**Worked.** Runs with F1 0.71/0.76/0.74 → tracker surfaces 0.76 as best, reproducible from its logged config.

:::

### Lineage, catalog, registry

Metadata (schemas, owners, freshness, quality) + lineage power a data catalog and impact analysis. A model registry versions & stages models; reproducibility pins code/data/config/env.


## A real system that works this way

**MLflow Tracking** records experiments and runs, including parameters, metrics, tags and artifacts. **MLflow Model Registry** provides named model versions and workflows for serving and governance. Current MLflow documentation favours aliases and tags for deployment labels; the older fixed stage workflow is deprecated. An alias such as `champion` can point to an approved model version, while a tag can hold an approval state or evaluation cohort. A release process can update an alias after checks and preserve the previous version for rollback.

**OpenLineage** provides an event model connecting jobs, runs and datasets, with facets for additional metadata. A pipeline can emit an event saying that one run consumed a source dataset and produced a feature dataset. A training job can then link that feature snapshot to a model artifact. When a source field is discovered to be wrong, the lineage graph can suggest which runs and models might be affected. The graph is an investigation aid; it may omit an undeclared notebook read or an external feature lookup, so owners should reconcile it with storage access and release records.

A practical system uses tracking and lineage together. The experiment run stores the exact training snapshot ID, code revision, dependency lockfile, preprocessing artifact and evaluation report. The lineage system maps the snapshot back to its source and transform. The registry links the approved artifact to the run and to a deployment alias. This creates a trace from serving model back to data origin and tests, without assuming that metadata alone proves the model is good.

## Code you can run

Start with the lecture's F1 ranking, then apply a serving constraint. The candidate table is small enough to inspect directly.

```python
runs = [
    {"id": "run-1", "f1": 0.71, "latency_ms": 35},
    {"id": "run-2", "f1": 0.76, "latency_ms": 80},
    {"id": "run-3", "f1": 0.74, "latency_ms": 45},
]
reported_best = max(runs, key=lambda run: run["f1"])
eligible = [run for run in runs if run["latency_ms"] <= 60]
release_candidate = max(eligible, key=lambda run: run["f1"])
print(reported_best["id"], release_candidate["id"])
assert reported_best["id"] == "run-2"
assert release_candidate["id"] == "run-3"
```

This manifest hashes the content of a small dataset and links it to a run. A hash can detect a changed byte sequence; it does not say the labels are correct or the evaluation is fair.

```python
import hashlib
import json

records = [{"id": 1, "label": 0}, {"id": 2, "label": 1}]
payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
manifest = {
    "data_sha256": hashlib.sha256(payload).hexdigest(),
    "code_revision": "revision-7",
    "config": {"seed": 17, "split": "holdout-v2"},
}
repeat = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
print(manifest["data_sha256"][:12])
assert hashlib.sha256(repeat).hexdigest() == manifest["data_sha256"]
assert hashlib.sha256(payload + b" ").hexdigest() != manifest["data_sha256"]
```

For a real dataset, identify the immutable snapshot or table version and hash a canonical manifest of its files. A single output hash is useful for detecting drift but may change for harmless ordering differences unless canonicalisation is defined.

## Designing with it

### Define an experiment before logging it

Write the objective, candidate family, target metric, population, holdout construction and release constraints. Two F1 values are comparable only when the label definition, positive class, averaging method, threshold and evaluation population match. A threshold tuned on the holdout turns that holdout into development data. If the same user appears in both train and test through duplicated records, a high score may be leakage. Keep entity and time splits appropriate for the deployment setting, and preserve the exact list of examples used for final evaluation.

Log parameters that change the computation, not just a few headline hyperparameters. Include data snapshot, feature definitions, code revision, dependency versions, preprocessing state, random seeds, model artifact and evaluation report. The report should include denominators, uncertainty, baseline comparison and relevant slices. A metric without the number of positive labels can be misleading; a small change from 0.74 to 0.76 may be within sampling variation. Record who reviewed the result and why a candidate was accepted or rejected.

### Separate reproducibility levels

**Repeatability** means the same setup can produce a result again. **Reproducibility** across machines or environments asks more. Pinning code, data, configuration and dependencies improves both, but exact bytes can still differ with parallel floating-point operations, GPU kernels, nondeterministic libraries and external API responses. Decide what matters: identical artifact checksum, numerically close predictions, or a statistically equivalent metric within a tolerance. State the tolerance and test it. If a hosted feature or embedding service changes, its version or response snapshot must be captured or the experiment is not fully reconstructable.

### Make lineage actionable

Record edges at the granularity needed for change impact. Dataset-level lineage can answer which models consumed a table. Column-level lineage can narrow a field change, but it is harder to capture through arbitrary Python transforms. A catalog entry should state owner, description, grain, schema, update schedule, quality checks and access policy. A lineage edge needs a run or time context: a model trained last month may have consumed an old dataset version even if the table name still exists. Test impact queries against a known run so the graph is not merely decorative.

### Govern the registry decision

Registering an artifact gives it an identity. It does not make it approved. A promotion workflow should link a candidate version to evaluation evidence, security and privacy checks where applicable, latency and resource measurements, and an accountable approver. Use an alias or equivalent deployment pointer to direct traffic to an approved version. Keep rollback instructions and the previous artifact available. If a data issue later invalidates a training snapshot, lineage should find candidate and deployed descendants so the owner can assess exposure.

## Audit the apparent winner

Run 2 reports F1 0.76, above run 3's 0.74. Before calling it the winner, compare the evaluation manifests. Run 2 may have used a random row split while run 3 used a time split. If the production system predicts future events for existing users, the random split may share near-duplicate user records between training and testing. Both metric values were recorded accurately, but they answer different questions. The first audit is to rebuild a common holdout with the target production time horizon and rerun both artifacts without retuning their thresholds.

Next inspect uncertainty. If the holdout has only a handful of positive examples, a two-point F1 difference can result from one changed classification. Report the confusion matrix and a suitable interval or resampling analysis, with care for grouped users and time dependence. Examine performance by important segment. A global F1 gain that comes from one large group may hide a severe regression for a small but high-risk group. The release decision needs a declared policy for those trade-offs.

The serving budget adds another test. Suppose run 2 needs 80 ms at the chosen percentile under realistic load, while the service budget is 60 ms. Run 3 needs 45 ms. If the latency measure is comparable and stable, run 3 may be the only eligible candidate despite its lower F1. A team could optimise run 2, change the serving tier or revise the product budget, but those are new decisions. The tracker can expose both metrics; it cannot choose the business trade-off without policy.

Now inspect provenance. The run should point to an immutable dataset snapshot and exact preprocessing artifact. A code revision alone is insufficient if a mutable external table was queried during training. A saved parameter file alone is insufficient if the feature service used a different transformation at evaluation. Verify that the registry artifact is the one that produced the reported predictions; a packaging step can change preprocessing or thresholds. A smoke test that recomputes predictions on a small locked fixture helps catch such mismatches.

If the candidate passes, attach the evaluation report and approval to a registry version. Advance the approved alias in a controlled deployment, observe live input and output measures, then compare with the prior version. Retain rollback capability. When a source dataset later has a discovered label error, use lineage to find every run that consumed that version. Include both rejected and deployed models in the impact list: rejected experiments may have informed a later design even if they never served traffic.

### Investigate an upstream change

A data owner changes the `account_status` field from two values to three. The catalog should surface the schema change and owner. Dataset lineage identifies transformations that read the table; feature lineage identifies which model features incorporate status; run lineage identifies model versions trained on those features. The result is a candidate impact graph, not proof that every model fails. A transform may map the new value to an unknown bucket safely, or it may drop rows silently. Test the actual path with a sample containing the new value and compare quality checks before and after the change.

If the graph has missing edges, discover why. A one-off notebook may have read the table without emitting lineage events. A scheduled training job may have logged a table name but not a snapshot ID. Improve instrumentation at the point of data read and artifact creation. Periodically select a deployed model and trace it backward to its source versions; if that exercise fails, the metadata system is not ready for incident response. Ownership and documentation need the same maintenance as code.

Metadata can also carry risk. A run artifact may contain sample records or personally identifying fields. Store only what is needed for reconstruction, apply access controls and retention, and use stable snapshot identifiers instead of copying raw sensitive data into every run. The goal is a trustworthy evidence chain, not an unlimited archive of training inputs.

## Where this stands in 2026

:::info Industry view

- MLflow Tracking stores runs and artifacts; its registry documentation uses aliases and tags for current model workflows.
- OpenLineage models jobs, runs and datasets as events with facets, allowing cross-system provenance when emitters are configured.
- A release review needs evaluation validity, operational constraints and ownership in addition to a ranked metric.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What does experiment tracking record and why?</summary>

Each run's parameters, metrics, code version, data version and artifacts, so runs are comparable and the best is reproducible.<br /><em>Lecture 12 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is data lineage and what does it enable?</summary>

A record of how each dataset/feature/model was produced from upstream sources, enabling impact analysis ('what breaks if this changes?') and debugging.<br /><em>Lecture 12 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Three runs report F1 0.71, 0.76, 0.74. Which is chosen and why can it be trusted?</summary>

Run 2 has the highest reported F1 (0.76). Logged data, code and configuration help reconstruct it, but evaluation validity, uncertainty and release constraints must be checked before promotion.<br /><em>Lecture 12 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What does a model registry provide?</summary>

Versioning of trained models and controlled deployment references. Current MLflow workflows use aliases and tags; its older fixed stages are deprecated.<br /><em>Lecture 12 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is required for full reproducibility?</summary>

Pin code, data, configuration and environment, then test a stated reproducibility tolerance; hardware and nondeterministic operations can prevent exact bit-for-bit results.<br /><em>Lecture 12 · conceptual</em>

</details>

## Go deeper

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) describes runs and logged artifacts.
- [MLflow Model Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) describes versions, aliases and tags.
- [OpenLineage object model](https://openlineage.io/docs/spec/object-model/) defines jobs, runs, datasets and facets.
- Built from the course lecture "dm-l12-experimentation-metadata" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can identify run 2 as the highest reported F1 and explain why that alone does not approve it.
- [ ] I can log a run with data, code, configuration, artifact and evaluation identities.
- [ ] I can trace a deployed model back to an upstream dataset version and name missing lineage edges.
- [ ] I can distinguish a registry version, approval state and deployment alias.
