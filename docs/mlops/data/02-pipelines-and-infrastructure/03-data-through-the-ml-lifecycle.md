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
import SplitGateLab from '@site/src/components/viz/SplitGateLab';

**In one line.** An ML lifecycle is a loop of decisions whose evidence must remain reproducible.


:::tip Before you start

**You should already know**

- What training, validation and test sets are, and what a model's AUC measures.
- What a pipeline run produces: [building reliable data pipelines](/docs/mlops/data/reliable-pipelines).
- Basic scikit-learn: `fit`, `transform` and `Pipeline`.

**Reading time.** About 50 minutes, plus a few seconds to run the experiments.

**After this chapter you can**

- compute split and fold sizes by hand,
- show with numbers that choosing features before splitting can turn pure noise into a 0.946 AUC,
- say why a model can look healthy on AUC while its accuracy collapses after a serving mistake.

:::

## In 30 seconds

Imagine marking your own exam paper after reading the answer sheet. The mark would be high, and it would say nothing about what you know. Data leakage is the same mistake in machine learning: something the model should not know at prediction time slips into training, so the score looks good and the real result disappoints. The lifecycle (frame the problem, understand and prepare data, model, evaluate, deploy) is a loop where every step must leave evidence, and every step is a chance for that leak or for the serving data to differ from the training data.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| CRISP-DM | A six-phase loop from business question to deployment | Framing, data, preparation, modelling, evaluation, deployment |
| Data leakage | Information from outside the training data leaking into it | Choosing features using all the labels |
| Holdout | Data kept aside to estimate real performance | A 15% test set |
| Cross-validation | Repeating fit and score across several folds | 5 folds of 2,000 |
| Train/serve skew | Features at serving differ from training | Centimetres at serving, millimetres in training |
| AUC | Chance that a random positive scores above a random negative | 0.5 is chance, 1.0 is perfect |
| Pipeline (scikit-learn) | Preprocessing and model fitted together on training data only | `make_pipeline(scaler, model)` |
| Model registry | A record of model versions and their references | Alias `champion` points to version 7 |

## The idea in plain words

A dataset does not become a model in one straight pass. Teams frame a business decision, inspect available evidence, prepare examples, fit candidates, evaluate them and deploy a chosen version. Results from evaluation often reveal that the label was ambiguous or that an important population was missing, sending the team back to data understanding. That is the value of the **CRISP-DM** loop: it places modelling inside an iterative investigation.

The six phases are business understanding, data understanding, data preparation, modelling, evaluation and deployment. Each produces an artefact a later reviewer can inspect. A business statement should name the decision and its cost of error. Data understanding should identify the source, grain, time range and quality gaps. Preparation should define labels and features. Modelling should record the candidate and settings. Evaluation should show performance on suitable unseen data. Deployment should monitor real outcomes and feed findings back into the next iteration.

<Infographic src="/img/dm/ml-lifecycle.svg" alt="Six CRISP-DM phases lead from framing through data and modelling to deployment; ten thousand examples split 70, 15 and 15 per cent give 7,000 train, 1,500 validation and 1,500 test." caption="The lifecycle loops, while the final test set remains reserved for one honest evaluation." />

A numerical example splits 10,000 examples as 70% train, 15% validation and 15% test: **7,000 / 1,500 / 1,500**. Fivefold cross-validation holds out **2,000** examples per fold if all 10,000 participate. Those counts are correct, but a random split is not automatically valid. Time order, repeated customers and related documents can put near-duplicates across partitions. The split must imitate the deployment question.

:::note Added for this site

The course names the phases, splits, leakage and experiment registry. The sections below add evaluation design for time and entities, test-set governance, dataset manifests, a correction to the old model-registry stage terminology, and measured experiments on leakage and train/serve skew.

:::

The board shows the six phases and the 70/15/15 split. The later sections explain why a chronological split or grouped holdout may be necessary even when the arithmetic is correct.

## Worked example, step by step

**Split counts.** 10,000 examples with a 70/15/15 split give 7,000, 1,500 and 1,500. Five-fold cross-validation over the same 10,000 holds out 10,000 / 5 = 2,000 per fold and trains on 8,000.

**Skew by hand.** A model trained on `mean radius` (training mean 14.10, standard deviation 3.62) standardises the value with z = (x - mean) / standard deviation.

1. A normal reading of 14.0 gives z = (14.0 - 14.10) / 3.62 = -0.03, close to the average.
2. If serving code sends the same reading in different units, scaled by 0.1 to 1.4, then z = (1.4 - 14.10) / 3.62 = -3.51. The model sees a tiny tumour every time.
3. The model's score shifts for every row, so a threshold of 0.5 now flags the wrong cases even though the model is unchanged.

**Leakage in words.** With 100 rows and 5,000 random features, some features will correlate with a random label by chance. If you pick the 20 best using all labels, those 20 look predictive on the same rows and on any split of them. A fair test picks features inside each training fold only.

In words: leakage is borrowing from the answer sheet and skew is a unit or fill mismatch between training and serving. The experiments below measure both, and the first code block reproduces the split counts.

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

**MLflow** is a concrete system for recording runs, parameters, metrics and model artefacts, then registering model versions. Its current registry guidance uses model-version aliases and tags for workflows. **Correction.** Older tutorials describe fixed registry stages; MLflow now documents those stages as deprecated and recommends aliases and tags instead. This is a correction to older practice, not a change to the general need for a governed model lifecycle. See the site's [W&B and MLflow cheatsheet](/docs/cheetsheet/wandb-mlflow-master-cheatsheet) for basic commands, while checking the current MLflow guidance for registry promotion.

Consider a model that predicts whether a customer will miss a payment in the next month. The team first asks whether a prediction will trigger an offer, a human review or an automatic decision. It then identifies historical payment events and defines the label using a fixed future window. Preparation computes features only from events known before the prediction date. Candidate models are compared on validation periods, then the selected model is evaluated once on a later test period. A tracked run records the data snapshot, feature code, cutoff and metrics, so the result can be repeated and audited.

The deployed model is not the end of the loop. New customers may behave differently; late labels may show that validation overestimated benefit. Monitoring can reveal a missing source feed or an input distribution shift. Those observations should update the business and data assumptions before a retrain. Simply automating weekly retraining on whatever data happens to be available repeats an error faster.

## Code you can run

The first block checks the split arithmetic and fivefold holdout count. It also shows that each fold's held-out count is about validation within a modelling procedure; it is not a replacement for a final independent test set after model selection.

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

### Experiment: leakage and a split by customer

The first experiment has three parts. It selects features on pure noise before and inside cross-validation, then repeats the scaler leak on a real dataset, then compares a random row split with a split by customer on synthetic repeated customers. The real dataset is the Wisconsin diagnostic breast cancer data (569 rows, 30 features) bundled with scikit-learn; the UCI Machine Learning Repository lists it under a CC BY 4.0 licence (opened 2026-10-09). Run with scikit-learn 1.9.1 and NumPy 2.5.3.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

leaky, honest = [], []
for seed in range(10):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, 100)
    picked = SelectKBest(f_classif, k=20).fit_transform(X, y)
    leaky.append(cross_val_score(LogisticRegression(max_iter=1000), picked, y, cv=5, scoring="roc_auc").mean())
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    honest.append(cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean())
print(f"pure noise, 5000 features, labels random: selection on all rows AUC {np.mean(leaky):.3f} (range {min(leaky):.2f} to {max(leaky):.2f}), inside the folds {np.mean(honest):.3f}")

data = load_breast_cancer()
names = list(data.feature_names)
scaler_leak = cross_val_score(LogisticRegression(max_iter=5000), StandardScaler().fit_transform(data.data), data.target, cv=5, scoring="roc_auc").mean()
scaler_ok = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), data.data, data.target, cv=5, scoring="roc_auc").mean()
print(f"breast cancer data, scaler fitted on all rows {scaler_leak:.4f}, inside the folds {scaler_ok:.4f}")

rng = np.random.default_rng(0)
customers, months = 300, 12
customer_of_row = np.repeat(np.arange(customers), months)
traits = rng.normal(size=(customers, 6))
risk = rng.normal(size=customers)
X_rows = np.hstack([traits[customer_of_row] + 0.05 * rng.normal(size=(customers * months, 6)), rng.normal(size=(customers * months, 2))])
y_rows = (risk[customer_of_row] + 0.5 * rng.normal(size=customers * months) > 0).astype(int)
forest = RandomForestClassifier(200, random_state=0, n_jobs=-1)
by_row = cross_val_score(forest, X_rows, y_rows, cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
by_customer = cross_val_score(forest, X_rows, y_rows, cv=GroupKFold(5), groups=customer_of_row, scoring="roc_auc").mean()
print(f"300 customers x 12 months, traits unrelated to risk: random row split AUC {by_row:.3f}, split by customer {by_customer:.3f}")
```

**Reading the output.** The first line gives the AUC for a task with no signal at all. Chance is 0.5. The second line compares scaling on all rows with scaling inside the folds on real data. The third compares splitting rows at random with splitting whole customers.

**Line by line.**

- `SelectKBest(...).fit_transform(X, y)` before `cross_val_score` is the leak: the labels have already chosen the columns the folds are scored on.
- `make_pipeline(SelectKBest(...), LogisticRegression())` repeats the selection inside every fold, using only that fold's training rows.
- `GroupKFold(5)` with `groups=customer_of_row` keeps all rows of one customer in the same fold.
- `traits[customer_of_row]` gives each customer's 12 rows nearly identical features, which is how a random split lets the model recognise a customer it has already seen.

The printed output:

```text
pure noise, 5000 features, labels random: selection on all rows AUC 0.946 (range 0.87 to 0.98), inside the folds 0.526
breast cancer data, scaler fitted on all rows 0.9953, inside the folds 0.9952
300 customers x 12 months, traits unrelated to risk: random row split AUC 0.908, split by customer 0.570
```

### Experiment: train/serve skew

The second experiment trains a logistic regression on the same breast cancer data and then serves test rows with four common mistakes: a unit change, a fill with zero, a frozen lookup and swapped columns. It does this for a model that uses all 30 features and for one that uses only four.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
names = list(data.feature_names)

def skew_report(features):
    cols = [names.index(f) for f in features]
    X_train, X_test, y_train, y_test = train_test_split(data.data[:, cols], data.target, test_size=0.3, random_state=0, stratify=data.target)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_train, y_train)
    a, b = features.index("mean radius"), features.index("mean texture")
    c = features.index("worst concave points")
    served = {"as trained": X_test.copy()}
    served["mean radius in other units (x0.1)"] = X_test.copy(); served["mean radius in other units (x0.1)"][:, a] *= 0.1
    served["worst concave points filled with 0"] = X_test.copy(); served["worst concave points filled with 0"][:, c] = 0
    served["worst concave points frozen at median"] = X_test.copy(); served["worst concave points frozen at median"][:, c] = np.median(X_train[:, c])
    swapped = X_test.copy(); swapped[:, [a, b]] = swapped[:, [b, a]]; served["radius and texture columns swapped"] = swapped
    print(f"model with {len(features)} features")
    for label, rows in served.items():
        p = model.predict_proba(rows)[:, 1]
        print(f"  {label:40s} AUC {roc_auc_score(y_test, p):.4f}  accuracy {accuracy_score(y_test, p > 0.5):.3f}")

skew_report(names)
skew_report(["mean radius", "mean texture", "mean smoothness", "worst concave points"])
```

**Reading the output.** Each line is one way of serving the same 171 held-out rows. `AUC` ranks the cases and `accuracy` applies the 0.5 threshold.

**Line by line.**

- `served[...] = X_test.copy()` makes an independent copy for each mistake, so one does not contaminate another.
- `[:, a] *= 0.1` is the unit change from the worked example.
- `swapped[:, [a, b]] = swapped[:, [b, a]]` is a column-order bug, common when a model is fed a raw array and not named columns.

The printed output:

```text
model with 30 features
  as trained                               AUC 0.9956  accuracy 0.959
  mean radius in other units (x0.1)        AUC 0.9961  accuracy 0.947
  worst concave points filled with 0       AUC 0.9953  accuracy 0.953
  worst concave points frozen at median    AUC 0.9953  accuracy 0.953
  radius and texture columns swapped       AUC 0.9956  accuracy 0.959
model with 4 features
  as trained                               AUC 0.9831  accuracy 0.930
  mean radius in other units (x0.1)        AUC 0.9724  accuracy 0.626
  worst concave points filled with 0       AUC 0.9591  accuracy 0.690
  worst concave points frozen at median    AUC 0.9591  accuracy 0.877
  radius and texture columns swapped       AUC 0.9642  accuracy 0.819
```

### Reading the experiments

Selecting features on all rows turned noise into a 0.946 AUC (ten seeds, from 0.87 to 0.98), when the true value is 0.5. Doing the selection inside each fold gave 0.526. The scikit-learn guide on common pitfalls (version 1.9.1, opened 2026-10-09) shows the same effect with 200 samples and 10,000 random features: 0.76 test accuracy when selecting before splitting, 0.5 when not.

The surprise is that the scaler leak did nothing here: 0.9953 against 0.9952. Preprocessing that never looks at the labels leaks only a little, while selection that uses the labels leaks a lot. Do not conclude that fitting a scaler on all rows is fine; do conclude that the label-aware steps are the dangerous ones.

Splitting rows at random scored 0.908 for features that carry no information about risk, because the forest recognised each customer. Splitting by customer scored 0.570, close to what honest evaluation should give for unseen customers.

For skew, the 30-feature model barely moved: AUC stayed within 0.9953 to 0.9961 and accuracy fell from 0.959 to 0.947 at worst. Redundant correlated features cushion a single bad column. The four-feature model was fragile. A unit mistake on one feature left AUC at 0.9724 (from 0.9831) but cut accuracy from 0.930 to 0.626. AUC alone would have called that nearly healthy. Limits: one dataset, one split, a linear model, hand-made mistakes and a synthetic customer set.

<SplitGateLab />

**What each control does.**

- **examples**, **train share**, **validation share** set the split; the test share is the rest.
- **cross-validation folds** sets how many folds, and so the held-out count per fold.
- **select features on all rows** switches between the two AUCs printed by the noise experiment.

**Try it yourself.**

1. Leave the defaults. You get 7,000, 1,500 and 1,500 and 2,000 held out per fold, then 0.526 AUC on noise.
2. Tick the checkbox. The AUC on pure noise jumps to 0.946, the leakage measured above. The counts do not change.
3. Set folds to 10. The held-out size becomes 1,000, and so each fold is a noisier estimate. Raise examples to 100,000 to see how the same percentages leave a far larger test set.

<Infographic src="/img/dm-enrich/leakage-and-skew.svg" alt="Bars compare AUC for noise features selected on all rows and inside folds, a random against a customer split, and cards show accuracy dropping after serving mistakes in a four-feature model." caption="Look first at the 0.946 bar: the model learned nothing, and the leak made it look excellent." />

## Designing with it

### Start from a decision, not an algorithm

Business understanding should say who uses the prediction, when, and what happens after a false positive or false negative. A model with high accuracy may be useless if it arrives after the decision or cannot be acted on. Write a baseline policy and a success measure before exploring complex models. Also define groups and scenarios that need separate evaluation. A single aggregate score can conceal poor performance on new users or rare but costly events.

Data understanding begins with source inventory and the unit of observation. Is one row a person, event, account-day or text passage? Repeated rows for one customer should often stay in the same split or be ordered by time. A label whose definition relies on events after the prediction date is valid as an outcome, but those same future events cannot enter the features. Record the prediction cutoff and the label observation window as separate times.

### Keep each split's role intact

Training data fits parameters and learned preprocessing. Validation data selects models, hyperparameters and thresholds. Test data estimates the selected procedure once, after those choices are fixed. Repeatedly checking the test score while redesigning features turns the test into another validation set. Reserve a new later period or dataset if the final test has been exhausted by iteration. "Unbiased" is conditional on this discipline and on the holdout matching deployment.

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

## Common mistakes

1. **Selecting features before splitting.** It feels efficient to clean the whole table once. On noise it produced 0.946 AUC. Put every learned step in a pipeline fitted per fold.
2. **Concluding that a scaler leak is harmless.** It changed nothing here (0.9953 against 0.9952). Label-aware steps leak far more. Keep both inside the pipeline anyway, because the habit is cheap.
3. **Splitting rows when the same customer appears many times.** A random split scored 0.908 on features with no information. Split by customer or by time, matching how the model will be used.
4. **Watching only AUC after deployment.** The unit mistake cost 0.0107 AUC but 0.304 accuracy. Track accuracy or the score distribution at the threshold you actually use.
5. **Letting serving code reimplement preprocessing.** Swapped columns and unit changes come from copies of the logic. Reuse the training pipeline object, and compare training and serving feature statistics.

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

<details>
<summary><strong>Q6. (Medium)</strong> A model uses a feature with training mean 14.10 and standard deviation 3.62. Serving sends the value 1.4 where 14.0 was intended. What z-score does the model see, and why can AUC stay high while accuracy drops?</summary>

z = (1.4 - 14.10) / 3.62 = -3.51 instead of -0.03. Every row shifts the same way, so the ranking of rows can stay almost the same, which is what AUC measures, while the fixed 0.5 threshold now falls in the wrong place and accuracy drops. In the four-feature run AUC fell 0.0107 and accuracy fell 0.304.

</details>

<details>
<summary><strong>Q7. (Stretch)</strong> Selecting 20 of 5,000 noise features on all rows gave AUC 0.946, but a scaler fitted on all rows changed AUC by 0.0001. Why the difference?</summary>

Selection uses the labels: it picks the columns that happen to correlate with them, and with 5,000 candidates some always do. A scaler uses only feature means and spreads, which carry no label information, so little leaks. The leak is large when the preprocessing step looks at the target.

</details>

## Go deeper

- [IBM CRISP-DM overview](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview) describes the iterative methodology.
- [scikit-learn common pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html) explains preprocessing leakage.
- [MLflow registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) explains current aliases and tags.
- [scikit-learn common pitfalls, data leakage](https://scikit-learn.org/stable/common_pitfalls.html) (opened 2026-10-09, version 1.9.1) says never to call `fit` on test data and shows feature selection before splitting giving 0.76 test accuracy on random data against 0.5 when done correctly.
- [UCI: Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (opened 2026-10-09) lists 569 instances, 30 features and a CC BY 4.0 licence.
- Library versions run for the experiments: scikit-learn 1.9.1, NumPy 2.5.3, Python 3.14.6.
- Built from the course lecture "dm-s6-ml-lifecycle" (Lecture Library series).

- **[Made With ML](https://madewithml.com/)** `course`
  Goku Mohandas; End-to-end MLOps; data pipelines, testing, deployment and monitoring.
- **[Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)** `docs`
  Google; 43 hard-won rules for building real ML systems and their data.
- **[Apache Airflow docs](https://airflow.apache.org/docs/)** `docs`
  Apache; How production data pipelines are scheduled and orchestrated.

## Check yourself

- [ ] I can name the six CRISP-DM phases and the evidence each should leave.
- [ ] I can compute the 70/15/15 split and fivefold held-out count.
- [ ] I can choose a split and preprocessing boundary that reflects deployment.
- [ ] I can identify a model version, data snapshot and current registry alias without relying on deprecated stages.
- [ ] I can show with a noise experiment that choosing features before splitting inflates AUC.
- [ ] I can explain why a split by customer or time can score far lower than a random row split, and why that is the honest number.
- [ ] I can say why AUC alone can hide a train/serve skew and which second metric to watch.

## Where to go next

Next is [collecting and ingesting data](/docs/mlops/data/collection-and-ingestion). For leakage through time in features, see [features and point-in-time correctness](/docs/mlops/data/features-and-point-in-time).
