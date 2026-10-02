---
id: ml-capstone
title: "Capstone: An End-to-End Tabular Pipeline"
sidebar_label: "Capstone pipeline"
sidebar_position: 3
slug: /theory/ml/capstone-tabular-pipeline
description: "One project from raw table to a saved, checked model: a locked test set, baselines, a scikit-learn pipeline, cross-validation, tuning, calibration, a cost-based threshold, explanation, a model card and automated checks, all on a synthetic churn dataset."
tags: [capstone, pipeline, churn, cross-validation, calibration, threshold, model-card, joblib]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** The method of a tabular project is the order of its steps: decide what you will be judged on, protect the test set, beat a baseline, and only then get clever.

:::note Not from the lecture
This chapter is an **addition**. It ties together the earlier chapters of this subject (preprocessing, supervised models, boosting, [evaluation](/docs/theory/ml/model-evaluation) and [explanation](/docs/theory/ml/explaining-predictions)) into one project you can run offline. The data is synthetic and generated in code; the costs in the decision step are **assumptions** written into the code, not facts about any business.
:::

## The idea in plain words

You are asked to help a retention team. Each month they can phone some customers with an offer, and each phone call costs money. Whom should they call?

Turned into machine learning, the request has four parts.

1. A table, one row per customer, with columns such as contract type, tenure, monthly charge, support calls and a usage trend.
2. A label: did the customer leave in the next period?
3. A score for each current customer: the probability of leaving.
4. A rule that turns scores into calls, which depends on what a call costs and what a saved customer is worth.

Most of the time on such a project goes into everything around the model. The order of operations is where beginners lose weeks and where experienced people save them:

- **Lock the test set before you look at anything.** Whatever you do afterwards uses the rest.
- **Build the dullest baseline first.** If a model cannot beat "predict the average", something upstream is broken.
- **Put every learned step in a pipeline** so cross-validation re-fits it on each fold.
- **Tune with cross-validation, never with the test set.**
- **Check calibration, pick the threshold from costs, then open the test set once.**
- **Explain, document and test** the thing you are about to hand over.

<Infographic src="/img/ml/capstone-tabular-pipeline-flow.svg" alt="A nine-step flow: split into 6400 development and 1600 locked test rows; baselines (majority AUC 0.500, logistic 0.844, boosting 0.840); tuning to cross-validation AUC 0.854; calibration with no gain; threshold 0.18; a single test evaluation with AUC 0.852; permutation importance; model card; saved bundle with six passing checks." caption="The whole project on one board. Every figure is printed by the pipeline block below." />

## How it works

### 1. The data, and what is deliberately imperfect about it

The generator builds 8,000 customers with seven columns: `contract` (monthly, annual, two-year), `tenure_months`, `monthly_charge`, `support_calls`, `usage_trend`, `payment_method` and `region`. About a third churn (0.334). Six per cent of `usage_trend` values are missing, because real tables have holes. The churn rule has features that a plain linear model can only approximate: brand-new customers (under four months) churn much more, high charges bite only above a level, and support calls matter far more for monthly-contract customers whose usage is falling. The `region` column is pure noise: it has **no** effect in the generator, which gives us something to check the explanation against.

### 2. Split first

Twenty per cent of the rows (1,600) are set aside and not touched until step 6. The split is stratified so both parts have the same churn rate. Real churn data usually has a time axis, so in a real project this split would be by date; see the note on time under Designing with it.

### 3. Baselines

A `DummyClassifier` that predicts the training churn rate scores an AUC of exactly 0.500 by construction. A logistic regression is the next honest baseline. A default gradient-boosted model is the first serious contender. All three go through the same `ColumnTransformer`: numeric columns are imputed and scaled for the linear model (trees get them as they are), categories are one-hot encoded, and unseen categories are ignored rather than crashing the service.

### 4. Cross-validation and tuning

Each candidate is scored with stratified five-fold cross-validation on the development rows. The best default model is then tuned with a small randomised search (12 draws over learning rate, depth, leaf count, regularisation and minimum leaf size). The tuned score is optimistic, because we picked the best of 12 on the same folds; the locked test set is what keeps us honest.

### 5. Calibration, then a threshold from the economics

A gradient-boosted model's raw probabilities may be off, so we compare them with isotonic-calibrated ones using **out-of-fold** predictions (each row is scored by a model that never saw it). The decision rule comes from the business: a call costs 10, a called churner is saved 30% of the time, and a saved customer is worth 200. Calling a customer with churn probability `p` is worth it when `p * 0.30 * 200 > 10`, that is when `p > 0.167`. We then confirm on out-of-fold scores by scanning thresholds for the best net benefit.

### 6. One look at the test set

The final model is fitted on all development rows, scored on the locked rows, and the metrics are reported once: ranking quality (AUC, average precision), probability quality (Brier score), and the outcome at the chosen threshold (precision, recall, net benefit against calling everyone).

### 7. Explain, document, save, check

Permutation importance on the raw columns tells the retention team what drives risk; a table of AUC by region shows whether the model treats groups alike; a model card puts intended use, data, decision rule, results and limits in one short document; the model is saved with `joblib` as a bundle (model, threshold, column list, metrics); and a small set of pytest-style checks guards the behaviour we care about.

## A real system that works this way

The shape of this project is the same wherever a model scores customers in bulk. A nightly batch job loads the saved bundle, scores every active customer, applies the threshold and writes a ranked list of customers to contact. Training and scoring use the *same* pipeline object, so the imputation and encoding that were fitted on development rows are applied identically at scoring time. This is the simplest answer to training-serving skew: ship the preprocessing together with the model, as one artefact.

What changes in a real deployment is the data and the risk. The label is delayed (you only know who left weeks later), the split must be by time, the offer itself changes who churns (so the scoring population differs from the training population), and the economics are measured, not assumed. The pipeline above is the part that stays the same.

## Code you can run

Part 1 runs the whole project and saves the bundle. It takes about ten seconds on a laptop CPU. It limits scikit-learn's thread pool to one thread with `threadpool_limits(1)`, which keeps small-data boosting from spending its time on thread coordination.

```python
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

threadpool_limits(1)


def make_churn(n=8000, seed=11):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["monthly", "annual", "two_year"], size=n, p=[0.55, 0.28, 0.17])
    tenure = np.where(contract == "monthly", rng.gamma(1.6, 6.0, n), rng.gamma(3.0, 12.0, n)).round(0)
    monthly_charge = rng.normal(70, 22, n).clip(20, 140).round(2)
    support_calls = rng.poisson(np.where(contract == "monthly", 1.6, 0.9))
    usage_trend = rng.normal(0, 1, n).round(3)
    payment = rng.choice(["card", "bank_transfer", "invoice"], size=n, p=[0.5, 0.35, 0.15])
    region = rng.choice(["north", "south", "east", "west"], size=n)
    logit = (-2.0 + 1.0 * (contract == "monthly") - 0.7 * (contract == "two_year")
             - 0.02 * tenure + 1.4 * (tenure < 4) + 0.045 * np.maximum(monthly_charge - 85, 0)
             + 0.2 * support_calls + 0.7 * support_calls * (contract == "monthly") * (usage_trend < 0)
             - 0.5 * usage_trend + 0.5 * (payment == "invoice"))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame({"contract": contract, "tenure_months": tenure, "monthly_charge": monthly_charge,
                          "support_calls": support_calls, "usage_trend": usage_trend,
                          "payment_method": payment, "region": region})
    frame.loc[rng.random(n) < 0.06, "usage_trend"] = np.nan
    return frame, churn


frame, churn = make_churn()
print(f"rows {len(frame)}, churn rate {churn.mean():.3f}, usage_trend missing {frame['usage_trend'].isna().mean():.3f}")
X_dev, X_test, y_dev, y_test = train_test_split(frame, churn, test_size=0.2, random_state=0, stratify=churn)
print(f"development {len(X_dev)} rows, locked test {len(X_test)} rows")

numeric = ["tenure_months", "monthly_charge", "support_calls", "usage_trend"]
categorical = ["contract", "payment_method", "region"]
linear_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
tree_prep = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
candidates = {
    "majority class": Pipeline([("prep", linear_prep), ("model", DummyClassifier(strategy="prior"))]),
    "logistic regression": Pipeline([("prep", linear_prep), ("model", LogisticRegression(max_iter=1000))]),
    "hist gradient boosting": Pipeline([("prep", tree_prep), ("model", HistGradientBoostingClassifier(random_state=0))]),
}
print(f"\n{'candidate':<24}{'CV AUC':>14}{'CV avg precision':>18}")
for name, pipe in candidates.items():
    auc = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="roc_auc")
    ap = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="average_precision")
    print(f"{name:<24}{auc.mean():>8.3f} +/-{auc.std():.3f}{ap.mean():>12.3f}")

search = RandomizedSearchCV(
    candidates["hist gradient boosting"],
    {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4, None],
     "model__max_leaf_nodes": [8, 15, 31], "model__l2_regularization": [0.0, 1.0, 10.0],
     "model__min_samples_leaf": [20, 50, 100]},
    n_iter=12, scoring="roc_auc", cv=cv, random_state=0, n_jobs=1).fit(X_dev, y_dev)
print(f"\ntuned boosting: CV AUC {search.best_score_:.3f} with {search.best_params_}")

calibrated = CalibratedClassifierCV(search.best_estimator_, method="isotonic", cv=5)
oof = cross_val_predict(calibrated, X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
raw_oof = cross_val_predict(search.best_estimator_, X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
print(f"out-of-fold Brier: raw {brier_score_loss(y_dev, raw_oof):.4f}, isotonic-calibrated {brier_score_loss(y_dev, oof):.4f}")

VALUE_OF_SAVED_CUSTOMER = 200.0
SAVE_RATE_IF_CONTACTED = 0.30
COST_OF_CONTACT = 10.0


def net_benefit(y_true, probability, threshold):
    flagged = probability >= threshold
    saved = y_true[flagged].sum() * SAVE_RATE_IF_CONTACTED * VALUE_OF_SAVED_CUSTOMER
    return saved - flagged.sum() * COST_OF_CONTACT


theory = COST_OF_CONTACT / (SAVE_RATE_IF_CONTACTED * VALUE_OF_SAVED_CUSTOMER)
grid = np.linspace(0.05, 0.60, 56)
benefit = np.array([net_benefit(y_dev, oof, t) for t in grid])
threshold = float(grid[benefit.argmax()])
print(f"\nassumed economics: contact costs {COST_OF_CONTACT:.0f}, saves {SAVE_RATE_IF_CONTACTED:.0%} of churners worth {VALUE_OF_SAVED_CUSTOMER:.0f}")
print(f"threshold from the economics {theory:.3f}; best on out-of-fold scores {threshold:.2f}")
print(f"development net benefit: contact nobody 0, contact everyone {net_benefit(y_dev, np.ones(len(y_dev)), 0.5):.0f}, at threshold {benefit.max():.0f}")

final = CalibratedClassifierCV(search.best_estimator_, method="isotonic", cv=5).fit(X_dev, y_dev)
p = final.predict_proba(X_test)[:, 1]
flagged = p >= threshold
tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()
print("\nlocked test set, touched once")
print(f"AUC {roc_auc_score(y_test, p):.3f}  average precision {average_precision_score(y_test, p):.3f}  Brier {brier_score_loss(y_test, p):.4f}")
print(f"flagged {flagged.sum()} of {len(p)}: TP {tp} FP {fp} FN {fn} TN {tn}  precision {tp / (tp + fp):.3f}  recall {tp / (tp + fn):.3f}")
print(f"net benefit: contact everyone {net_benefit(y_test, np.ones(len(y_test)), 0.5):.0f}, model at threshold {net_benefit(y_test, p, threshold):.0f}")

importance = permutation_importance(final, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0)
print("\npermutation importance on the raw columns (drop in AUC)")
for i in importance.importances_mean.argsort()[::-1]:
    print(f"  {X_test.columns[i]:<16}{importance.importances_mean[i]:.4f}")

print("\nAUC by region (a slice check for the model card)")
by_region = {}
for region in sorted(X_test["region"].unique()):
    mask = (X_test["region"] == region).to_numpy()
    by_region[region] = (int(mask.sum()), float(roc_auc_score(y_test[mask], p[mask])))
    print(f"  {region:<8}n={by_region[region][0]:<5}AUC {by_region[region][1]:.3f}")

artifact = Path(tempfile.gettempdir()) / "churn_model.joblib"
metrics = {"auc": float(roc_auc_score(y_test, p)), "average_precision": float(average_precision_score(y_test, p)),
           "brier": float(brier_score_loss(y_test, p)), "precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)),
           "by_region": by_region, "rows_dev": len(X_dev), "rows_test": len(X_test)}
joblib.dump({"model": final, "threshold": threshold, "columns": list(X_dev.columns), "metrics": metrics}, artifact)
print(f"\nsaved {artifact.name}, {artifact.stat().st_size / 1024:.0f} KB")
```

**What the output says.**

- **Baselines.** The majority-class model gets AUC 0.500 and average precision 0.334 (the churn rate). Logistic regression gets 0.844 and the untuned boosted model 0.840, so the plain linear model is a serious competitor until the boosted model is tuned.
- **Tuning.** The best draw reaches a cross-validation AUC of 0.854. That is the optimistic number; the test AUC of 0.852 confirms it was not luck.
- **Calibration.** Out-of-fold Brier scores are 0.1402 raw and 0.1401 isotonic-calibrated. Calibration made no real difference here, because boosting trained on log loss is already close to calibrated. Check this; do not assume it. The [evaluation chapter](/docs/theory/ml/model-evaluation) shows a model where calibration changes everything.
- **Threshold.** The economics give 0.167; the best threshold on out-of-fold scores is 0.18. They agree because the probabilities are calibrated. Contacting everyone gives a net benefit of 64,280 on the development rows; contacting those above the threshold gives 78,170.
- **Test set, once.** AUC 0.852, average precision 0.769, Brier 0.1419. At the threshold the model flags 903 of 1,600 customers: 468 true churners, 435 false alarms, 66 missed churners and 631 correctly left alone (precision 0.518, recall 0.876). Net benefit is 19,050, against 16,040 for calling everyone, a gain of 3,010 on the test rows under the assumed economics.
- **Explanation.** `contract` (0.1570) and `usage_trend` (0.0675) dominate, then `tenure_months` and `support_calls`. `region` scores 0.0002, which is correct: the generator gave it no effect.
- **Slices.** AUC by region runs from 0.814 to 0.889 on roughly 400 rows each. Since `region` has no effect in the generator, that whole spread is sampling noise. At a few hundred rows per slice, differences of several points need no cause.

Part 2 turns the saved bundle into a model card. Run it after Part 1, since it reads the file that Part 1 saved.

```python
import tempfile
from pathlib import Path

import joblib

bundle = joblib.load(Path(tempfile.gettempdir()) / "churn_model.joblib")
m = bundle["metrics"]
slices = "\n".join(f"| {region} | {n} | {auc:.3f} |" for region, (n, auc) in m["by_region"].items())

card = f"""# Model card: customer churn scorer (synthetic data)

### Intended use
Rank current customers by probability of leaving in the next period so a retention team can decide whom to contact.
Not for pricing, credit or any decision about an individual's access to a service.

### Data
Generated in code (make_churn, seed 11): {m['rows_dev']} development rows and {m['rows_test']} locked test rows.
No real customer is described. The relationships are those the generator was written with.

### Model
Calibrated (isotonic, 5-fold) HistGradientBoostingClassifier behind a ColumnTransformer.
Columns: {", ".join(bundle['columns'])}.

### Decision rule
Contact when predicted probability >= {bundle['threshold']:.2f}. Chosen from assumed economics:
contact cost 10, save rate 30%, saved customer worth 200. Change the economics and the threshold changes.

### Evaluation on the locked test set
AUC {m['auc']:.3f}, average precision {m['average_precision']:.3f}, Brier {m['brier']:.4f}.
At the threshold: precision {m['precision']:.3f}, recall {m['recall']:.3f}.

| region | rows | AUC |
| --- | --- | --- |
{slices}

### Limits
Slice sizes are a few hundred rows, so differences between regions of a few points may be noise.
The model was never evaluated on data later in time than its training data.
Retrain and re-check the threshold when the offer, its cost or the customer mix changes.
"""
print(card)
```

Part 3 is a pytest-style check suite. It also reads the saved bundle, so run it after Part 1. Save it as `test_churn_model.py` and `pytest` will collect the functions; the last lines let it run as a plain script.

```python
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path(tempfile.gettempdir()) / "churn_model.joblib"
bundle = joblib.load(ARTIFACT)
model, threshold, columns = bundle["model"], bundle["threshold"], bundle["columns"]


def customer(**overrides):
    base = {"contract": "monthly", "tenure_months": 12.0, "monthly_charge": 70.0, "support_calls": 1,
            "usage_trend": 0.0, "payment_method": "card", "region": "north"}
    base.update(overrides)
    return pd.DataFrame([base])[columns]


def score(frame):
    return model.predict_proba(frame)[:, 1]


def test_probabilities_are_valid():
    frame = pd.concat([customer(), customer(contract="two_year", tenure_months=60.0)], ignore_index=True)
    p = score(frame)
    assert p.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()


def test_scoring_is_deterministic():
    frame = customer()
    assert np.array_equal(score(frame), score(frame))


def test_missing_value_and_unseen_category_do_not_crash():
    p = score(pd.concat([customer(usage_trend=np.nan), customer(region="atlantis")], ignore_index=True))
    assert np.isfinite(p).all()


def test_new_monthly_customer_is_riskier_than_loyal_two_year_customer():
    risky = score(customer(contract="monthly", tenure_months=1.0))[0]
    safe = score(customer(contract="two_year", tenure_months=60.0))[0]
    assert risky > safe + 0.2


def test_more_support_calls_do_not_lower_risk_on_average():
    calls = np.arange(0, 6)
    p = score(pd.concat([customer(support_calls=int(c)) for c in calls], ignore_index=True))
    assert p[-1] >= p[0]


def test_threshold_is_a_probability():
    assert 0.0 < threshold < 1.0


tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
for check in tests:
    check()
    print(f"pass  {check.__name__}")
print(f"{len(tests)} checks passed")
```

The six checks are small on purpose: valid probabilities, determinism, no crash on a missing value or an unseen category, a behavioural sanity check (a brand-new monthly customer must score at least 0.2 higher than a loyal two-year customer), a monotone-on-average check for support calls, and a threshold in range. Behavioural checks like these catch a retrained model that has quietly changed its mind.

## Designing with it

**What to change when the project is real.**

| Here | In a real project |
| --- | --- |
| Random stratified split | A time-based split: develop on earlier months, test on the latest. Shuffled splits flatter the score |
| Costs are assumptions | Measure the contact cost, the save rate (with an experiment) and customer value, and rerun the threshold step |
| Label arrives instantly | The label arrives late; define the prediction window and the label window and keep them apart |
| One model, one run | Track data version, code version, metrics and threshold with each saved model |
| Checks run once | Run them on every retrain, and add monitoring of input drift and score distribution in production |

**Habits that carry over to any tabular project.**

- Decide the metric and the decision rule before fitting anything.
- Keep a simple model in the comparison until a complex one has clearly beaten it on held-out data.
- Report the tuned score as optimistic and the test score as the estimate.
- Calibrate only if the check shows it is needed; derive the threshold from costs, not from 0.5.
- Never let the model card be the only place where a limitation is written: put the matching check in the test suite.

**Where this simple design breaks.** If treating customers changes their behaviour, a model trained on untreated history does not predict the effect of treatment; you then want an uplift or experiment-based design, not a churn score. If the economics vary by customer (value differs), threshold per segment or rank by expected value. If fairness across groups matters, the slice table must be a gate, not a footnote.

## Where this stands in 2026

:::info Industry view

- A `Pipeline` plus a `ColumnTransformer` remains the standard way to keep preprocessing and model together, and the scikit-learn cross-validation guide shows exactly this pattern for keeping a scaler from seeing the held-out fold.
- scikit-learn 1.9 includes `TunedThresholdClassifierCV` for choosing a decision threshold by cross-validation against a custom scorer, which turns the cost-based step above into a library call.
- Gradient boosting is still the strong default for tabular data (the [gradient boosting chapter](/docs/theory/ml/gradient-boosting-in-practice) covers when it is and is not), but this project shows why a linear baseline is kept in the comparison: here it nearly ties the untuned boosted model.
- Model cards are the common lightweight way to document intended use, data, metrics by slice and limitations; the original proposal dates from 2018.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why is the test set locked before any model is built?</summary>

Anything learned from it (a feature idea, a hyperparameter, a threshold) leaks into the model and makes the final score optimistic. Locking it first means every choice uses only development rows, so one final evaluation is an honest estimate.

</details>

<details>
<summary><strong>Q2.</strong> The tuned model's cross-validation AUC is 0.854 and the test AUC is 0.852. What does each number mean?</summary>

The cross-validation number is the best of 12 candidates scored on the same folds, so it is slightly optimistic. The test AUC comes from rows that took no part in any choice, so it is the estimate to report. Their closeness suggests the search did not overfit the folds.

</details>

<details>
<summary><strong>Q3.</strong> Calibration changed the Brier score from 0.1402 to 0.1401. Should you keep the calibrator?</summary>

It is harmless but unnecessary here. Drop it if you want a simpler artefact, keep it if probabilities will drive pricing and you want a guard against drift. The decision should follow the check, not a habit.

</details>

<details>
<summary><strong>Q4.</strong> A call costs 10 and saves a churner worth 200 with probability 0.3. What is the break-even probability and why?</summary>

0.167. A call on a customer with churn probability `p` earns `p * 0.3 * 200 = 60p` and costs 10, so it pays off when `60p > 10`, that is `p > 1/6`.

</details>

<details>
<summary><strong>Q5.</strong> Region AUCs range from 0.814 to 0.889, yet region has no effect in the generator. Why the spread?</summary>

Each slice has about 400 rows, so the AUC estimate is noisy. Differences of several points between small slices can arise from sampling alone; investigate only if they persist on more data or across retrains.

</details>

<details>
<summary><strong>Q6.</strong> Which part of this project would you change first for real churn data, and why?</summary>

The split: use a time-based split, because shuffled splits let the model see the future and overstate performance. After that, replace the assumed economics with measured ones.

</details>

## Further reading

- [scikit-learn user guide: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): pipelines inside folds, stratification and time-series splits.
- [scikit-learn user guide: tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html): `TunedThresholdClassifierCV` with a custom cost scorer.
- [scikit-learn user guide: probability calibration](https://scikit-learn.org/stable/modules/calibration.html): when to calibrate and with which method.
- [scikit-learn user guide: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html): explaining a fitted pipeline on its raw columns.
- [Mitchell et al. (2018): model cards for model reporting](https://arxiv.org/abs/1810.03993): the proposal behind the card in Part 2.
- [Google Machine Learning Crash Course: classification metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall): a short refresher on the metrics used here.

## Check yourself

- I can lay out the order of a tabular project and say what each step protects against.
- I can build a `Pipeline` with a `ColumnTransformer` that imputes, scales and encodes, and handles unseen categories.
- I can compare a baseline, a linear model and a boosted model with the same cross-validation folds.
- I can tune with a randomised search and explain why the tuned score is optimistic.
- I can check calibration with out-of-fold predictions and decide whether a calibrator is worth keeping.
- I can derive a decision threshold from costs and confirm it on held-out scores.
- I can evaluate once on a locked test set, report precision, recall and net benefit, and explain slice differences honestly.
- I can save a model with its threshold and metrics, write a model card, and write behavioural checks for it.
