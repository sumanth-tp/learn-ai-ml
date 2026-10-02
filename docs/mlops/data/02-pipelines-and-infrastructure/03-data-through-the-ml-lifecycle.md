---
id: dm-data-through-the-ml-lifecycle
title: "Data Management · Session 6 — Data Through the ML Lifecycle"
sidebar_label: "6 · ML lifecycle"
sidebar_position: 3
slug: /mlops/data/ml-lifecycle
description: "Use CRISP-DM, leakage-safe splits and experiment records to carry data from problem framing to deployment."
tags: [data-management, crisp-dm, data-leakage, experiment-tracking]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** An ML lifecycle is a loop of decisions whose evidence must remain reproducible.

## The idea in plain words

A dataset does not become a model in one straight pass. Teams frame a business decision, inspect available evidence, prepare examples, fit candidates, evaluate them and deploy a chosen version. Results from evaluation often reveal that the label was ambiguous or that an important population was missing, sending the team back to data understanding. That is the value of the lecture's **CRISP-DM** loop: it places modelling inside an iterative investigation.

The six phases are business understanding, data understanding, data preparation, modelling, evaluation and deployment. Each produces an artefact a later reviewer can inspect. A business statement should name the decision and its cost of error. Data understanding should identify the source, grain, time range and quality gaps. Preparation should define labels and features. Modelling should record the candidate and settings. Evaluation should show performance on suitable unseen data. Deployment should monitor real outcomes and feed findings back into the next iteration.

<Infographic src="/img/dm/ml-lifecycle.svg" alt="Six CRISP-DM phases lead from framing through data and modelling to deployment; ten thousand examples split 70, 15 and 15 per cent give 7,000 train, 1,500 validation and 1,500 test." caption="The lifecycle loops, while the final test set remains reserved for one honest evaluation." />

The lecture's numerical example splits 10,000 examples as 70% train, 15% validation and 15% test: **7,000 / 1,500 / 1,500**. Fivefold cross-validation holds out **2,000** examples per fold if all 10,000 participate. Those counts are correct, but a random split is not automatically valid. Time order, repeated customers and related documents can put near-duplicates across partitions. The split must imitate the deployment question.

:::note Beyond the lecture

The lecture names the phases, splits, leakage and experiment registry. The sections below add evaluation design for time and entities, test-set governance, dataset manifests and a correction to the old model-registry stage terminology.

:::

The board shows the lecture's six phases and the 70/15/15 split. The later sections explain why a chronological split or grouped holdout may be necessary even when the arithmetic is correct.

## How it works

### CRISP-DM phases

Business → data understanding → data prep → modeling → evaluation → deployment, looping as you learn. Data prep is the bulk of the work.

### Train/val/test & leakage

Train (fit), validation (tune/select), test (final unbiased). Leakage inflates scores; fit transforms on train only, respect time order.

:::tip

**Worked.** 10,000 @ 70/15/15 → 7,000 / 1,500 / 1,500. 5-fold CV → 2,000 held out per fold.

:::

### Experiments & registry

Track code, data version, config, metrics, artifacts (MLflow, W&B); version models in a registry with stages.


## A real system that works this way

**MLflow** is a concrete system for recording runs, parameters, metrics and model artefacts, then registering model versions. Its current registry guidance uses model-version aliases and tags for workflows. The lecture mentions fixed registry stages; MLflow now documents those stages as deprecated and recommends aliases and tags instead. This is a source correction, not a change to the general need for a governed model lifecycle. See the site's [W&B and MLflow cheatsheet](/docs/cheetsheet/wandb-mlflow-master-cheatsheet) for basic commands, while checking the current MLflow guidance for registry promotion.

Consider a model that predicts whether a customer will miss a payment in the next month. The team first asks whether a prediction will trigger an offer, a human review or an automatic decision. It then identifies historical payment events and defines the label using a fixed future window. Preparation computes features only from events known before the prediction date. Candidate models are compared on validation periods, then the selected model is evaluated once on a later test period. A tracked run records the data snapshot, feature code, cutoff and metrics, so the result can be repeated and audited.

The deployed model is not the end of the loop. New customers may behave differently; late labels may show that validation overestimated benefit. Monitoring can reveal a missing source feed or an input distribution shift. Those observations should update the business and data assumptions before a retrain. Simply automating weekly retraining on whatever data happens to be available repeats an error faster.

## Code you can run

The first block checks the lecture's split arithmetic and fivefold holdout count. It also shows that each fold's held-out count is about validation within a modelling procedure; it is not a replacement for a final independent test set after model selection.

```python
count = 10_000
train = int(count * 0.70)
validation = int(count * 0.15)
test = count - train - validation
fold_held_out = count // 5
print(train, validation, test, fold_held_out)
assert (train, validation, test) == (7_000, 1_500, 1_500)
assert fold_held_out == 2_000
```

The second block demonstrates a preprocessing leak without training a model. Future values are larger than historical values. Fitting a scaler on the whole series changes the mean and standard deviation used for past training points. A valid transform is fitted on the training period and then applied unchanged to validation and test periods.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

history = np.array([10, 12, 11, 13, 14, 12], dtype=float).reshape(-1, 1)
future = np.array([30, 32], dtype=float).reshape(-1, 1)
all_values = np.vstack([history, future])

safe = StandardScaler().fit(history)
leaky = StandardScaler().fit(all_values)
print(f"training-only mean={safe.mean_[0]:.2f}")
print(f"whole-series mean={leaky.mean_[0]:.2f}")
print(f"future first value under training transform={safe.transform(future)[0, 0]:.2f}")
assert round(safe.mean_[0], 2) == 12.00
assert round(leaky.mean_[0], 2) == 16.75
assert safe.transform(future)[0, 0] > leaky.transform(future)[0, 0]
```

The difference is visible before fitting a predictor. In cross-validation, fit each transform separately on each training fold; a pipeline object can enforce this boundary. A holdout that leaks through preprocessing is no longer an honest estimate of future performance.

## Designing with it

### Start from a decision, not an algorithm

Business understanding should say who uses the prediction, when, and what happens after a false positive or false negative. A model with high accuracy may be useless if it arrives after the decision or cannot be acted on. Write a baseline policy and a success measure before exploring complex models. Also define groups and scenarios that need separate evaluation. A single aggregate score can conceal poor performance on new users or rare but costly events.

Data understanding begins with source inventory and the unit of observation. Is one row a person, event, account-day or text passage? Repeated rows for one customer should often stay in the same split or be ordered by time. A label whose definition relies on events after the prediction date is valid as an outcome, but those same future events cannot enter the features. Record the prediction cutoff and the label observation window as separate times.

### Keep each split's role intact

Training data fits parameters and learned preprocessing. Validation data selects models, hyperparameters and thresholds. Test data estimates the selected procedure once, after those choices are fixed. Repeatedly checking the test score while redesigning features turns the test into another validation set. Reserve a new later period or dataset if the final test has been exhausted by iteration. "Unbiased" in the lecture is conditional on this discipline and on the holdout matching deployment.

Use a chronological split for future prediction when behaviour changes over time. Use grouped splits when the same entity contributes related records. For both, keep source preparation and feature generation inside the training fold where they learn any statistics. If a global vocabulary, scaler, imputer or target encoding is fitted before splitting, information about validation or test data can flow into training. The scikit-learn [common pitfalls guide](https://scikit-learn.org/1.5/common_pitfalls.html) specifically warns about this and recommends pipelines.

### Track the experiment as an evidence bundle

Record the objective, source snapshot, sample filters, train/validation/test indices or cutoff, feature definition, code version, environment, model settings, random seeds, metrics and saved artefacts. A model registry associates a reviewed model version with deployment references and status information. With current MLflow, aliases and tags express those references; do not rely on the old fixed stage names in new workflows. A mutable alias such as `champion` points to a version, so record the resolved version at prediction time when auditability matters.

Reproducibility is not simply rerunning one notebook. If a data source has changed, the same code may learn a different model. If a model is promoted without a dataset manifest, later teams cannot tell whether a score change came from code, data or environment. A deployment record should tie the serving artefact back to the tracked run and the exact evaluation that justified it.

## Follow one model from framing to feedback

The customer-payment team first writes a decision memo. It defines the population, prediction time, outcome window and permissible intervention. It specifies that the model is advisory to a reviewer and that a false alarm imposes customer friction. This frame determines which precision, recall, calibration and subgroup checks matter. A model chosen only for overall accuracy could be a poor fit for the decision.

The team then profiles event coverage by month and channel. It discovers that one payment processor began sending a new status code in April. The data preparation step maps the code, verifies its meaning with the source owner and backfills affected dates. It records the corrected source version. If the new code is simply treated as "unknown" without investigation, the model might learn that April customers differ from earlier customers for an artefactual reason.

Features are computed as of each prediction time. For a decision on 1 May, only events available by then are permitted, even if an event with an April event-time stamp was ingested in June. This is the distinction between event time and knowledge time. An offline query that uses today's corrected history can accidentally give an earlier example information that the live model could not have had. Preserve a point-in-time snapshot or explicit as-of join policy.

Several candidate models are fitted on the training period. Validation periods are used to select a threshold and check whether performance is stable across customer groups. Once the design is fixed, the later test period is evaluated once. The team records metrics, confidence intervals where useful, the data manifest and a model artefact. A registry alias may make deployment convenient, but the release note names the immutable version behind it.

After deployment, monitoring checks input schema, freshness, prediction distribution and eventually realised outcomes. If missing labels take a month to mature, live accuracy cannot be computed immediately. Short-term input and operational checks bridge that delay, while mature labels support later performance review. When performance changes, the team returns to business and data understanding: the customer population, source feeds or intervention may have changed. That loop is the practical meaning of CRISP-DM, not a six-box ceremony.

### Make the split reflect the prediction setting

Suppose each customer has twelve monthly rows. A random row split can put January in training and February in test for the same person. The model may learn stable customer traits from training that make the test task easier than predicting for an unseen customer. If the product serves existing customers in future months, a chronological split may be appropriate, with careful handling of repeated entities. If it serves entirely new customers, hold out customer IDs. The right choice depends on the intended generalisation, not a universal rule that group splits are always better.

For a feature with a thirty-day lookback and a label over the following thirty days, adjacent rows can share substantial information. A gap between training and validation periods may be needed to prevent overlapping windows from making the score optimistic. The gap is a modelling decision that should match the outcome horizon and data availability. Record it in the experiment. The arithmetic of 7,000, 1,500 and 1,500 does not reveal this risk; a data timeline does.

Cross-validation repeats the fit-and-score cycle across folds, but preprocessing must be fitted inside each cycle. If a vocabulary is learned from all text before splitting, the validation fold has influenced the feature space. If a target encoder uses labels outside the training fold, the leak is even more direct. Package transformations with the estimator or explicitly fit them on fold training data. When reporting cross-validation, give the range of scores and the splitting scheme, not only a mean.

The final test is a decision record. Freeze the model, threshold and metric definitions before using it. If the result is unacceptable and the team redesigns the system, a later evaluation needs fresh independent evidence. This discipline can feel costly with limited data, but reusing the same test repeatedly turns an apparent generalisation score into another optimised training signal. Keep the test set's access and use visible in the run history.

## Where this stands in 2026

:::info Industry view

- CRISP-DM remains a useful iterative planning model, though a modern deployment adds monitoring and reproducibility details beyond the original six phase names.
- Current MLflow registry guidance favours model-version aliases and tags; fixed registry stages are deprecated.
- Leakage-safe evaluation still depends on training-only preprocessing and splits that reflect future deployment conditions.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Name the CRISP-DM phases.</summary>

Business understanding → data understanding → data preparation → modeling → evaluation → deployment, iterating as you learn.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Why split into train, validation and test?</summary>

Train fits the model, validation tunes hyperparameters/selects models, test gives a final estimate on unseen data if it remains independent and represents deployment.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> 10,000 examples, 70/15/15 split. Give the counts.</summary>

Train = 7,000, validation = 1,500, test = 1,500.<br /><em>Session 6 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is data leakage and how is it prevented?</summary>

Information from validation/test or the future contaminating training, inflating scores. Prevent by fitting all transforms on the training fold only and respecting time order.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why track experiments and use a model registry?</summary>

ML is empirical; tracking code, data version, config, metrics, artifacts makes runs reproducible and comparable, and a registry versions models and records deployment references; current MLflow uses aliases and tags rather than fixed stages.<br /><em>Session 6 · conceptual</em>

</details>

## Go deeper

- [IBM CRISP-DM overview](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview) describes the iterative methodology.
- [scikit-learn common pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html) explains preprocessing leakage.
- [MLflow registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) explains current aliases and tags.
- Built from the course lecture "dm-s6-ml-lifecycle" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check your understanding

- [ ] I can name the six CRISP-DM phases and the evidence each should leave.
- [ ] I can compute the 70/15/15 split and fivefold held-out count.
- [ ] I can choose a split and preprocessing boundary that reflects deployment.
- [ ] I can identify a model version, data snapshot and current registry alias without relying on deprecated stages.
