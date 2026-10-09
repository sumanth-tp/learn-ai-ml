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

:::tip Before you start

**You should already know**

- What a pipeline run reads and writes ([Lecture 10, orchestration and recovery](/docs/mlops/data/orchestration-and-recovery)).
- What an F1 score is and why a holdout set must stay separate ([Model evaluation](/docs/theory/ml/model-evaluation)).

**Reading time:** about 40 minutes, plus a minute to run the code.

**After this chapter you can**

- Build a lineage graph from run metadata and ask which models a changed table touches.
- Explain why a plain graph over-reports impact and what a time context fixes.
- Say how much a few missing lineage emitters hurt an impact answer.

:::

## In 30 seconds

Suppose someone tells you a column in the orders table was wrong last month. Which reports and models used it? If every job wrote down what it read and what it wrote, you can follow the arrows from the table to the answer. If you never wrote it down, you start asking people.

Think of a recipe card box where every dish lists its ingredients. When one batch of flour is recalled you look up every dish that used it, and only the dishes cooked after the bad batch arrived.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Experiment tracking | Logging each run's settings, metrics and artifacts | F1 0.76 with seed 17 and snapshot 3 |
| Lineage | The graph of what was read to produce what | `orders` to `features` to `model_7` |
| Upstream, downstream | Earlier and later in that graph | `orders` is upstream of `model_7` |
| Impact analysis | Listing everything downstream of a change | Which models read the changed table? |
| Catalog | A searchable list of datasets with owners and schemas | Owner, grain, update schedule |
| Registry | Named model versions with approval state | `champion` alias points to version 4 |
| Run event | A record that one job run read and wrote datasets | Job `train_7` read `feat_3` on day 70 |
| Time context | The date a read happened | Trained before the bug, so unaffected |


## The idea in plain words

Machine learning is empirical: a model is chosen from experiments whose results must be compared. A notebook number on its own is fragile. Which data snapshot produced it? Was the holdout truly separate? What code and configuration ran? Which artifact was measured? **Experiment tracking** records these relationships so another person can inspect a claim and attempt to repeat it. **Metadata and lineage** connect the run to upstream datasets, features and downstream releases.

Take three runs with F1 scores **0.71, 0.76 and 0.74**. A tracker can sort them and show that run 2 has the largest reported score. That is only the beginning of a selection decision. The runs must use a comparable evaluation set and metric definition; the measurement needs uncertainty and subgroup checks; deployment may have latency, cost or safety limits. A logged configuration helps reconstruction but does not make a biased evaluation valid. A registry entry should record the evidence and approval decision, not merely promote whichever number is largest.

<Infographic src="/img/dm/experiment-metadata.svg" alt="Three runs report F1 scores of 0.71, 0.76 and 0.74; lineage connects source data, features, run and model version, while release review checks leakage and latency." caption="Tracking makes a reported result inspectable; a release decision still tests the result's meaning." />

Lineage is a graph of what was used and what was produced. A source table feeds a cleaned dataset, which feeds a feature set, which feeds a training run and model version. If a source field changes, the graph helps identify affected descendants. A data catalog adds searchable descriptions, owners, schemas, freshness and quality signals. These tools shorten an investigation, provided the metadata is complete and updated by the actual jobs.

:::note Added for this site

The course material says that tracking makes the top run trustworthy and that exact reproducibility follows from pinned code, data, configuration and environment. The sections below qualify those claims: evaluation design and deployment constraints determine trust, while hardware, nondeterministic operations and external services can prevent bit-for-bit reproduction.

:::

The lab starts with the **0.71, 0.76 and 0.74 F1 scores**. With an 80 ms latency budget, run 2 is eligible and has the highest score. Tighten the budget to 60 ms and run 3 becomes the best eligible candidate. The control demonstrates why a metric ranking needs release constraints.

<ExperimentChoiceLab />

## Worked example, step by step

A source table `src` feeds a staging table `stg`, which feeds a feature set `feat`. Two models were trained on `feat`: `model_1` on day 40 and `model_2` on day 70. On day 60 someone finds that `src` was wrong from day 60 onward.

1. **Draw the graph.** Edges: `src` to `stg`, `stg` to `feat`, `feat` to `model_1`, `feat` to `model_2`. The descendants of `src` are `stg`, `feat`, `model_1` and `model_2`.
2. **Plain impact analysis.** Every model below the table is flagged: 2 models.
3. **Add time.** Only runs on or after day 60 can have read the bad data. The `stg` job runs daily, so from day 60 it carries the problem into `feat`. `model_1` was trained on day 40, before the change, so it is unaffected. `model_2` on day 70 is affected. Result: 1 model.
4. **Over-flag factor.** 2 flagged against 1 truly affected is a factor of 2.0.
5. **A missing emitter.** If the `stg` job never emitted lineage events, the graph has no edge from `src` onward. The search finds 0 of the 1 affected models: recall 0.

In words: a lineage graph answers "could this reach that", and a time context answers "did it". A job that does not report cuts the chain. The first block below prints these numbers.

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

Start with the F1 ranking, then apply a serving constraint. The candidate table is small enough to inspect directly.

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

### The worked example in code

This block builds the four-edge graph, runs the plain and time-aware searches and then removes the `stg` job.

```python
import networkx as nx

edges = [("src", "stg"), ("stg", "feat"), ("feat", "model_1"), ("feat", "model_2")]
runs = [(60, "stg", ["src"]), (61, "feat", ["stg"]), (40, "model_1", ["feat"]), (70, "model_2", ["feat"])]
graph = nx.DiGraph(edges)
plain = sorted(n for n in nx.descendants(graph, "src") if n.startswith("model"))

def affected(runs, change_day=60, start="src"):
    tainted = {start}
    for day, output, inputs in sorted(runs):
        if day >= change_day and tainted.intersection(inputs):
            tainted.add(output)
    return sorted(n for n in tainted if n.startswith("model"))

timed = affected(runs)
without_stg = affected([r for r in runs if r[1] != "stg"])
print("plain", plain, "timed", timed, "factor", len(plain) / len(timed))
print("stg emits nothing:", without_stg)
```

**Reading the output.** It prints the plain list with both models, the timed list with `model_2` only, a factor of 2.0, and an empty list when the `stg` job does not report. These are steps 2 to 5.

### An experiment on a lineage graph

Does the graph over-report, and how fragile is it? The block below builds a synthetic estate: 10 source tables, 15 staging tables, 15 marts, 20 feature sets and 60 model training runs, with every table job running daily for 120 days (6,060 run events). It stores them as events, builds a networkx directed graph and asks, for each source, which models are affected if the source was bad from day 60. It compares the plain graph answer with a time-aware answer that follows only runs on or after day 60. Then it silences a random share of jobs, as if their code never emitted events, and measures how many truly affected models are still found.

Versions used: Python 3.14.6, networkx 3.6.1, pandas 2.3.3, NumPy 2.5.3. The estate is random and synthetic. It runs in about a second.

```python
import networkx as nx
import numpy as np
import pandas as pd

rng = np.random.default_rng(12)
days, change_day = 120, 60
sources = [f"src_{i}" for i in range(10)]
layer1 = {f"stg_{i}": list(rng.choice(sources, rng.integers(1, 3), replace=False)) for i in range(15)}
layer2 = {f"mart_{i}": list(rng.choice(list(layer1), rng.integers(1, 3), replace=False)) for i in range(15)}
features = {f"feat_{i}": list(rng.choice(list(layer2) + list(layer1), rng.integers(1, 3), replace=False)) for i in range(20)}
jobs = {**layer1, **layer2, **features}
level = {name: 0 for name in sources} | {name: 1 for name in layer1} | {name: 2 for name in layer2} | {name: 3 for name in features}

events = []
for day in range(days):
    for name, inputs in jobs.items():
        events.append((day, level[name], f"job_{name}", tuple(inputs), name))
for i in range(60):
    day = int(rng.integers(0, days))
    inputs = tuple(rng.choice(list(features), rng.integers(1, 3), replace=False))
    events.append((day, 4, f"train_{i}", inputs, f"model_{i}"))
events.sort()
models = {e[4] for e in events if e[4].startswith("model_")}

graph = nx.DiGraph()
for _, _, job, inputs, output in events:
    graph.add_edges_from((i, output) for i in inputs)

def time_respecting(source, dropped):
    tainted = {source}
    for day, _, job, inputs, output in events:
        if day >= change_day and job not in dropped and tainted.intersection(inputs):
            tainted.add(output)
    return tainted & models

def naive(source):
    return nx.descendants(graph, source) & models if source in graph else set()

job_names = sorted({e[2] for e in events})
print("nodes", graph.number_of_nodes(), "edges", graph.number_of_edges(), "events", len(events), "models", len(models))
rows = []
for source in sources:
    truth, plain = time_respecting(source, set()), naive(source)
    rows.append((source, len(plain), len(truth)))
frame = pd.DataFrame(rows, columns=["source", "graph_flags", "truly_affected"])
print(frame.to_string(index=False))
print("mean flagged", frame.graph_flags.mean(), "mean truly affected", frame.truly_affected.mean(), "over-flag factor", round(frame.graph_flags.sum() / frame.truly_affected.sum(), 2))

print("share of jobs with no lineage events, recall of the affected models")
for share in (0.0, 0.05, 0.10, 0.20):
    recalls = []
    for seed in range(40):
        pick = np.random.default_rng(seed).choice(job_names, int(share * len(job_names)), replace=False)
        dropped = set(pick)
        for source in sources:
            truth = time_respecting(source, set())
            if truth:
                recalls.append(len(time_respecting(source, dropped) & truth) / len(truth))
    print(f"{share:5.2f} {np.mean(recalls):.3f}")
```

The output of the run:

```text
nodes 119 edges 172 events 6060 models 60
source  graph_flags  truly_affected
 src_0            0               0
 src_1           39              20
 src_2           31              12
 src_3           18               6
 src_4           35              15
 src_5            0               0
 src_6           18              11
 src_7           27              12
 src_8           32              18
 src_9           10               4
mean flagged 21.0 mean truly affected 9.8 over-flag factor 2.14
share of jobs with no lineage events, recall of the affected models
 0.00 1.000
 0.05 0.869
 0.10 0.689
 0.20 0.481
```

**Reading the output.** The first line gives the size: 119 tables and models (one source was never read, so it is not a node), 172 edges. Each row is one bad source. `graph_flags` is how many of the 60 models lie below it in the graph, `truly_affected` how many were trained on or after day 60 with a tainted input. The last four lines show recall of the truly affected set as more jobs fail to report.

**Line by line.**

- `events.sort()` orders runs by day, then by level, so a staging job on a given day is processed before the marts and features that read it.
- `time_respecting` marks a table tainted only when a run on or after `change_day` reads a tainted input. That is the time context.
- `dropped` holds job names. A silenced job removes every run of that job, so one missing emitter cuts every path through it.

### What the numbers say

The plain graph flagged 21.0 models on average against 9.8 truly affected, an over-flag factor of 2.14. For `src_1` it named 39 models when 20 were affected. The extra models were trained before the change on data that was still good. A plain graph alone sends the incident team to read twice as many models as needed.

The fragility is the surprise. Silencing 5% of jobs reduced recall to 0.869, 10% to 0.689 and 20% to 0.481. Recall falls much faster than the share of missing jobs, because a path from a source to a model crosses several jobs and one gap cuts the whole path. With a fifth of emitters missing, a bit more than half the affected models are not found, and nothing in the output says so.

Limits: a random synthetic estate with a fixed shape, one change day, jobs silenced entirely at random, and whole-job granularity. Real gaps cluster in notebooks and ad hoc scripts. Measure your own coverage by tracing a deployed model backwards.

<Infographic src="/img/dm-enrich/dm2-lineage-impact.svg" alt="Plain graph flags 21.0 models against 9.8 truly affected, and recall of affected models falls from 1.000 to 0.481 as 20 percent of jobs stop emitting lineage." caption="Look first at the falling bars: a few silent jobs break many paths." />

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Promoting the run with the highest metric | The tracker sorts it first | Compare evaluation sets and latency first. With an 80 ms run at 0.76 and a 60 ms budget, the 0.74 run is the eligible one |
| Reading lineage as proof of impact | The graph says "downstream" | Add a time context. The plain graph flagged 2.14 times as many models as were affected |
| Assuming the graph is complete | A tool is installed, so every job reports | Trace a deployed model backwards on a schedule. Silencing 20% of jobs cut recall to 0.481 |
| Logging a code revision but not a data snapshot | The code is the thing we edit | Log an immutable snapshot ID or a content hash of a canonical manifest |
| Expecting bit-for-bit reruns | Everything is pinned | State a tolerance for the metric or the predictions, and test it |

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

<details>
<summary><strong>Q6.</strong> (Medium) A source table is found to be wrong from day 60. Models were trained on days 40, 55, 70 and 90 from a feature set built from it. How many are affected, and what extra information did you need?</summary>

Two, the models trained on days 70 and 90. You needed the date of each training run, which is the time context on the lineage edge. The plain graph would flag all four.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) In the experiment 10% of jobs were silent and recall was 0.689, not 0.90. Why is recall lower than the share of jobs that report?</summary>

An affected model sits at the end of a path that passes through several jobs, here staging, mart, feature and training. If any one of them is silent the path is cut and the model is missed. With four or five jobs on a path, a 10% chance of silence per job leaves a full path intact only about two times in three. Recall therefore drops faster than the silent share. The remedy is to measure coverage by tracing from a deployed model to its sources, not to trust the graph.

</details>

## Go deeper

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) describes runs and logged artifacts.
- [MLflow Model Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) describes versions, aliases and tags.
- [OpenLineage object model](https://openlineage.io/docs/spec/object-model/) defines jobs, runs, datasets and facets.
- OpenLineage object model (the link above), opened 2026-10-09: a Job is a process that consumes or produces datasets, a Run is one occurrence of a job, and run events carry input and output datasets plus facets such as schema and version.
- MLflow model registry workflow (the link above), opened 2026-10-09: aliases are named references to model versions, tags annotate status, and model stages are deprecated as of MLflow 2.9.0.
- [networkx descendants](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.descendants.html), opened 2026-10-09: all nodes reachable from a source node in a directed graph.
- Built from the course lecture "dm-l12-experimentation-metadata" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can identify run 2 as the highest reported F1 and explain why that alone does not approve it.
- [ ] I can log a run with data, code, configuration, artifact and evaluation identities.
- [ ] I can trace a deployed model back to an upstream dataset version and name missing lineage edges.
- [ ] I can distinguish a registry version, approval state and deployment alias.
- [ ] I can list the models a changed table can reach, and cut the list with a time context.
- [ ] I can explain why 2 flagged models against 1 affected is an over-flag factor of 2.0 and what removes it.
- [ ] I can say why recall of an impact search falls faster than the share of jobs that fail to emit lineage.

## Where to go next

Next: [Lecture 13, distributed processing and skew](/docs/mlops/data/distributed-processing-skew), which looks at how a large job is split across workers. Related: [Lecture 11, features and point-in-time correctness](/docs/mlops/data/features-and-point-in-time), where the same time context decides which feature value a model saw.
