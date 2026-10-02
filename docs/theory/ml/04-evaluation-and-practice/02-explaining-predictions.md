---
id: ml-explainability
title: "Explaining Predictions: Importance, Partial Dependence, SHAP and LIME"
sidebar_label: "Explaining predictions"
sidebar_position: 2
slug: /theory/ml/explaining-predictions
description: "How to ask a trained model why it predicts what it does: permutation importance, partial dependence and ICE, SHAP values and a from-scratch LIME surrogate, plus the traps of correlated features and causal over-reading."
tags: [explainability, shap, lime, permutation-importance, partial-dependence, interpretability]
---

import Infographic from '@site/src/components/Infographic';
import ShapWaterfallLab from '@site/src/components/viz/ShapWaterfallLab';

**In one line.** An explanation describes how a model behaves, not how the world works, and each method answers a different question about that behaviour.

:::note Not from the lecture
This chapter is an **addition**. The course lectures stop at evaluation; explaining a trained model is the step that comes after you trust its score. Every number below is printed by code in this chapter.
:::

## The idea in plain words

A good score does not tell you whether a model is right for the right reasons. Two situations make you open the box. The first is **debugging**: a model that has quietly learned a shortcut (a column that leaks the answer) scores well and fails in use. The second is **decisions about people**: someone declined for credit, flagged for churn outreach or sent for a manual check will reasonably ask why.

Every method in this chapter sits on one of two axes.

- **Global or local.** A global method summarises the whole model ("which columns does it lean on?"). A local method explains one prediction ("why did *this* applicant score 0.85?").
- **Built into the model, or applied from outside.** A linear model's coefficients are an explanation by construction. A boosted forest has no such thing, so we probe it by changing inputs and watching outputs. Everything here is of the second kind and works on any fitted model.

We use one running example: a synthetic credit-default dataset with six columns (`income`, `debt_ratio`, `age`, `late_payments`, `utilisation`, `tenure_years`) and a gradient-boosted classifier. The data is generated in code, so the true rule is known and we can check explanations against it.

<Infographic src="/img/ml/explaining-predictions-global-local.svg" alt="Two groups of cards. Global: permutation importance ranks late_payments 0.0940, debt_ratio 0.0727, age 0.0527; partial dependence of default probability on debt_ratio rises from 0.065 to 0.306. Local: ICE curves, a SHAP sum from base value minus 2.008 to plus 1.758 log-odds, and a LIME-style local surrogate." caption="Pick the method from the question. Numbers are printed by the first two code blocks." />

## How it works

### Permutation importance (global)

Score the model on a held-out set. Then shuffle one column, which destroys its relationship with the target while keeping its distribution, and score again. The drop in score is that column's importance. Repeat several times per column for a spread. It works for any model and any metric, and it measures what the *model* uses, not what is predictive in principle. Two cautions from the scikit-learn guide: compute it on held-out data (on training data it also rewards columns the model merely memorised), and know that correlated columns share credit, so each looks less important than it is.

### Partial dependence and ICE (global, and local)

**Partial dependence** answers "what does the model predict, on average, as this one column varies?". For each grid value, set the column to that value for every row, predict, and average. **ICE** (individual conditional expectation) draws one such curve per row instead of averaging. If the ICE curves all rise at the same rate, the average tells the whole story. If they fan out, the average hides an **interaction**: the effect of the column depends on another column. Both methods assume the column can be changed independently of the others, which fails when columns are correlated.

### SHAP values (local, and global by averaging)

SHAP borrows an idea from cooperative game theory. Treat the prediction as a payout and the features as players; a feature's **Shapley value** is its fair share, averaged over every order in which features could join the team. The result has a property that makes it special: the contributions **add up**. For one applicant,

`model output = base value + sum of the SHAP values`

where the base value is the model's average output. For a classifier explained in log-odds, a positive value pushes risk up and a negative one pushes it down, and the sizes are comparable across features. Averaging the absolute SHAP values over many rows gives a global importance. `TreeExplainer` computes these values quickly and exactly for tree ensembles; for other models you pay in computation.

### LIME (local)

LIME fits a small, readable model to the black box *near one point*. Sample points around the applicant, ask the black box to score them, weight each sample by how close it is to the applicant, and fit a weighted linear model. Its coefficients are the explanation: "around here, one standard deviation more `debt_ratio` adds about 0.5 log-odds". Two choices matter and neither has a settled right answer: how wide the neighbourhood is, and how the samples are drawn. We build it from scratch below, because seeing the width choice change the answer teaches more than a library call.

## A real system that works this way

Credit decisioning is the classic place for per-case explanations. Take a lender whose default model is gradient boosting and whose analysts must tell a declined applicant the main reasons. The accurate answer is not a list of the model's globally most important columns, because the main reasons for *this* applicant may be different ones. A per-applicant attribution gives them directly: for the high-risk applicant in the code below, `late_payments` and `debt_ratio` account for the largest push upwards (+1.394 and +1.358 log-odds), while `income` pulls down (-0.412). The same attribution gives a model reviewer a way to check the model against domain sense, such as risk falling as tenure grows.

The pattern is general: wherever a score drives an action on an individual, a global ranking is for the developer and a per-case attribution is for the person affected and the person reviewing the case. Neither replaces testing the model on held-out data, and neither proves a causal claim, as the last block shows.

## Code you can run

The dataset generator is the same in every block, so each block runs on its own. All of them are deterministic.

### Global view: permutation importance, partial dependence, ICE

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def make_credit(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.8, sigma=0.4, size=n)
    debt_ratio = rng.beta(2, 5, size=n)
    age = rng.integers(21, 70, size=n)
    late_payments = rng.poisson(0.6, size=n)
    utilisation = np.clip(0.5 * debt_ratio + rng.normal(0.3, 0.15, size=n), 0, 1)
    tenure_years = rng.gamma(2.0, 2.0, size=n)
    logit = (-3.4 + 3.0 * debt_ratio + 0.7 * late_payments + 1.2 * utilisation
             - 0.9 * (np.log(income) - 10.8) - 0.04 * (age - 40) - 0.05 * tenure_years
             + 1.5 * debt_ratio * (late_payments > 1))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"income": income.round(0), "debt_ratio": debt_ratio.round(3), "age": age,
                      "late_payments": late_payments, "utilisation": utilisation.round(3),
                      "tenure_years": tenure_years.round(1)})
    return X, default


X, y = make_credit()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X_tr, y_tr)
print(f"default rate {y.mean():.3f}, test AUC {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}")

print("\npermutation importance, drop in AUC when a column is shuffled")
print(f"{'feature':<15}{'train':>8}{'test':>8}")
result = {}
for split, (Xs, ys) in {"train": (X_tr, y_tr), "test": (X_te, y_te)}.items():
    result[split] = permutation_importance(model, Xs, ys, scoring="roc_auc", n_repeats=10, random_state=0).importances_mean
for i in np.argsort(-result["test"]):
    print(f"{X.columns[i]:<15}{result['train'][i]:>8.4f}{result['test'][i]:>8.4f}")

curves = partial_dependence(model, X_te, ["debt_ratio"], kind="both", grid_resolution=5, percentiles=(0.05, 0.95))
grid = curves["grid_values"][0]
average = curves["average"][0]
individual = curves["individual"][0]
many_late = (X_te["late_payments"] > 1).to_numpy()
print("\npartial dependence of default probability on debt_ratio")
print(f"{'debt_ratio':<22}" + "".join(f"{g:>8.3f}" for g in grid))
print(f"{'average (PDP)':<22}" + "".join(f"{v:>8.3f}" for v in average))
print(f"{'ICE mean, 2+ late':<22}" + "".join(f"{v:>8.3f}" for v in individual[many_late].mean(axis=0)))
print(f"{'ICE mean, 0-1 late':<22}" + "".join(f"{v:>8.3f}" for v in individual[~many_late].mean(axis=0)))
rise = lambda curve: curve[-1] - curve[0]
print(f"\nrise across the range: all {rise(average):.3f}, 2+ late {rise(individual[many_late].mean(axis=0)):.3f}, 0-1 late {rise(individual[~many_late].mean(axis=0)):.3f}")
```

On held-out data, `late_payments` costs the most AUC when shuffled (0.0940), followed by `debt_ratio` (0.0727) and `age` (0.0527); `utilisation` and `tenure_years` hardly matter. The train column is larger than the test column for every feature, which is the model rewarding columns it has partly memorised. The average partial dependence says default probability rises from 0.065 to 0.306 as `debt_ratio` goes from 0.069 to 0.565, a gain of 0.240. Splitting by `late_payments` shows why the average is not the whole story: for applicants with two or more late payments the rise is 0.609; for the rest it is 0.193. That is an interaction the data was generated with, found by ICE and invisible in the average.

### SHAP: one prediction, then the whole model

```python
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def make_credit(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.8, sigma=0.4, size=n)
    debt_ratio = rng.beta(2, 5, size=n)
    age = rng.integers(21, 70, size=n)
    late_payments = rng.poisson(0.6, size=n)
    utilisation = np.clip(0.5 * debt_ratio + rng.normal(0.3, 0.15, size=n), 0, 1)
    tenure_years = rng.gamma(2.0, 2.0, size=n)
    logit = (-3.4 + 3.0 * debt_ratio + 0.7 * late_payments + 1.2 * utilisation
             - 0.9 * (np.log(income) - 10.8) - 0.04 * (age - 40) - 0.05 * tenure_years
             + 1.5 * debt_ratio * (late_payments > 1))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"income": income.round(0), "debt_ratio": debt_ratio.round(3), "age": age,
                      "late_payments": late_payments, "utilisation": utilisation.round(3),
                      "tenure_years": tenure_years.round(1)})
    return X, default


X, y = make_credit()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X_tr, y_tr)

explainer = shap.TreeExplainer(model)
values = explainer.shap_values(X_te)
base = float(explainer.expected_value[0] if np.ndim(explainer.expected_value) else explainer.expected_value)
log_odds = model.decision_function(X_te)
print(f"base value {base:.3f} log-odds = {1 / (1 + np.exp(-base)):.3f} probability")
print(f"max |base + sum(SHAP) - model output| = {np.abs(base + values.sum(axis=1) - log_odds).max():.2e}")

order = np.argsort(model.predict_proba(X_te)[:, 1])
cases = {"low risk": order[len(order) // 20], "borderline": order[int(len(order) * 0.85)], "high risk": order[-6]}
for label, i in cases.items():
    row = X_te.iloc[i]
    print(f"\n{label}: " + ", ".join(f"{c}={row[c]:g}" for c in X.columns))
    for name, v in sorted(zip(X.columns, values[i]), key=lambda t: -abs(t[1])):
        print(f"  {name:<14}{v:+.3f}")
    print(f"  output {log_odds[i]:+.3f} log-odds = {1 / (1 + np.exp(-log_odds[i])):.3f} probability")

global_importance = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)
print("\nmean |SHAP| (log-odds):")
print(global_importance.round(3).to_string())
```

The additivity line is the check that matters: base value plus the sum of SHAP values reproduces the model's raw output to 5.55e-15, so the values really are an exact decomposition. Three applicants are explained. The base value, -2.008 log-odds, is a probability of 0.118. The high-risk applicant ends at +1.758 log-odds, a probability of 0.853, with `late_payments` (+1.394), `debt_ratio` (+1.358) and `age` (+0.871) doing most of the work. The low-risk applicant ends at -3.560 (probability 0.028), helped most by age 66 (-0.633). Mean absolute SHAP gives a global ranking that differs from permutation importance in detail, with `age` now tied with `late_payments` at 0.479, because the two measure different things: SHAP measures how far a column moves the model's output, permutation importance measures how much accuracy depends on it.

The waterfall lab below holds exactly these three applicants. Its default (high-risk, log-odds, sorted by size) draws the +1.758 above. Switch to probability units and the steps change size with the order, because the sigmoid bends; the log-odds contributions never depend on order.

<ShapWaterfallLab />

:::note TreeExplainer defaults
With no background data, `TreeExplainer` uses the *tree-path-dependent* method, which estimates how a column behaves when "absent" from how training rows flowed down each tree. Supplying background data switches to the interventional method. Both are exact for their definition of "absent"; they can differ when columns are correlated, which is one reason to treat a SHAP value as a statement about the model rather than a measurement of the world.
:::

### LIME, from scratch

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def make_credit(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.8, sigma=0.4, size=n)
    debt_ratio = rng.beta(2, 5, size=n)
    age = rng.integers(21, 70, size=n)
    late_payments = rng.poisson(0.6, size=n)
    utilisation = np.clip(0.5 * debt_ratio + rng.normal(0.3, 0.15, size=n), 0, 1)
    tenure_years = rng.gamma(2.0, 2.0, size=n)
    logit = (-3.4 + 3.0 * debt_ratio + 0.7 * late_payments + 1.2 * utilisation
             - 0.9 * (np.log(income) - 10.8) - 0.04 * (age - 40) - 0.05 * tenure_years
             + 1.5 * debt_ratio * (late_payments > 1))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"income": income.round(0), "debt_ratio": debt_ratio.round(3), "age": age,
                      "late_payments": late_payments, "utilisation": utilisation.round(3),
                      "tenure_years": tenure_years.round(1)})
    return X, default


X, y = make_credit()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X_tr, y_tr)


def local_surrogate(model, reference, row, n_samples=2000, kernel_width=1.0, seed=0):
    rng = np.random.default_rng(seed)
    scale = reference.std().to_numpy()
    z = rng.normal(size=(n_samples, reference.shape[1]))
    z[0] = 0
    neighbours = pd.DataFrame(row.to_numpy() + z * scale, columns=reference.columns)
    distance_sq = (z ** 2).sum(axis=1)
    weights = np.exp(-distance_sq / kernel_width ** 2)
    target = model.decision_function(neighbours)
    surrogate = Ridge(alpha=1.0).fit(z, target, sample_weight=weights)
    return pd.Series(surrogate.coef_, index=reference.columns)


order = np.argsort(-model.predict_proba(X_te)[:, 1])
applicant = X_te.iloc[order[5]]
print("applicant:", ", ".join(f"{c}={applicant[c]:g}" for c in X.columns))
print("\nlog-odds change per one standard deviation, local linear surrogate")
table = pd.DataFrame({
    f"width {w}, seed {s}": local_surrogate(model, X_tr, applicant, kernel_width=w, seed=s)
    for w in (0.5, 1.0, 3.0) for s in (0, 1)
})
print(table.round(2).to_string())
```

Each row is a feature and each column a run of the surrogate. With a wide neighbourhood (width 3.0) and a medium one (1.0) the answer is stable between seeds: `late_payments` +0.64 and +0.60, `debt_ratio` +0.43 and +0.51, `age` and `income` negative. With a narrow one (0.5) the surrogate sees almost nothing and wobbles: `age` is -0.13 on one seed and -0.31 on the other, and `late_payments` is -0.01 or +0.08. Always run an explainer twice with different seeds and several widths before showing anyone its output. Note also that `late_payments` is a count, but the Gaussian samples are not integers, a known awkwardness of this style of sampling.

### Where explanations mislead

<Infographic src="/img/ml/explaining-predictions-pitfalls.svg" alt="Three columns. Baseline: debt_ratio permutation 0.0727 and mean absolute SHAP 0.404. With a near-copy of debt_ratio the credit is split to 0.0317 and 0.0116. With collections_calls, a consequence of default, the new column dominates at 0.3163 and test AUC jumps to 0.970." caption="Two ways an importance table lies, using the numbers the next block prints." />

```python
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def make_credit(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.8, sigma=0.4, size=n)
    debt_ratio = rng.beta(2, 5, size=n)
    age = rng.integers(21, 70, size=n)
    late_payments = rng.poisson(0.6, size=n)
    utilisation = np.clip(0.5 * debt_ratio + rng.normal(0.3, 0.15, size=n), 0, 1)
    tenure_years = rng.gamma(2.0, 2.0, size=n)
    logit = (-3.4 + 3.0 * debt_ratio + 0.7 * late_payments + 1.2 * utilisation
             - 0.9 * (np.log(income) - 10.8) - 0.04 * (age - 40) - 0.05 * tenure_years
             + 1.5 * debt_ratio * (late_payments > 1))
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"income": income.round(0), "debt_ratio": debt_ratio.round(3), "age": age,
                      "late_payments": late_payments, "utilisation": utilisation.round(3),
                      "tenure_years": tenure_years.round(1)})
    return X, default


def report(X, y, label):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X_tr, y_tr)
    perm = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0).importances_mean
    shap_values = shap.TreeExplainer(model).shap_values(X_te)
    print(f"\n{label}   test AUC {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}")
    print(f"  {'feature':<18}{'permutation':>12}{'mean |SHAP|':>13}")
    for i, name in enumerate(X.columns):
        print(f"  {name:<18}{perm[i]:>12.4f}{np.abs(shap_values[:, i]).mean():>13.3f}")


X, y = make_credit()
rng = np.random.default_rng(1)
report(X, y, "baseline")

twin = X.copy()
twin["debt_ratio_copy"] = (X["debt_ratio"] + rng.normal(0, 0.01, len(X))).round(3)
report(twin, y, "debt_ratio duplicated with a little noise")

after_the_fact = X.copy()
after_the_fact["collections_calls"] = rng.poisson(0.2 + 3.0 * y)
report(after_the_fact, y, "collections_calls added (a consequence of default, not a cause)")
```

**Correlated columns split the credit.** Adding a near-copy of `debt_ratio` leaves the AUC unchanged (0.765 to 0.764) but divides its importance: permutation 0.0727 falls to 0.0317 for the original and 0.0116 for the copy, and mean absolute SHAP falls from 0.404 to 0.242 and 0.159. Neither column now looks as important as the one did before. The fix is to treat correlated columns as a group: cluster them, explain the group, or keep one representative.

**Importance is not cause.** `collections_calls` is generated *from* the default label, so it is a consequence of the outcome, not a driver of it. The model finds it anyway: test AUC leaps from 0.765 to 0.970, its permutation importance is 0.3163 against 0.0104 for the next column, and its mean absolute SHAP is 2.055. A model card that listed it as "the top driver of default" would be technically true of the model and useless as advice: telling collections staff to make fewer calls would change nothing. That column would also not exist at the moment a real prediction is needed, which makes it a leak, the same family of fault covered in the [leakage chapter](/docs/theory/ml/features-leakage-and-imbalance).

## Designing with it

| You want to know | Use | Be careful about |
| --- | --- | --- |
| Which columns the model relies on, overall | Permutation importance on held-out data | Correlated columns share credit; score the model first, a poor model gives misleading importances |
| The average shape of an effect | Partial dependence | Assumes columns move independently; hides interactions |
| Whether an effect is the same for everyone | ICE (or centred ICE) | A fan of curves means an interaction |
| Why one prediction came out this way | SHAP waterfall | Attribution of the model's output, not a cause in the world |
| A quick local read when SHAP is too slow | A LIME-style surrogate | Check stability across seeds and widths |

**A short protocol that avoids most mistakes.**

1. Validate the model first. Explanations of a model that does not generalise explain the wrong thing.
2. Explain on held-out data, not on the training set.
3. Check for columns that could not exist at prediction time. If one tops the ranking, suspect a leak before celebrating a discovery.
4. Group correlated columns before reading any ranking.
5. Refit with another seed or data sample and see whether the explanation survives.
6. State what the explanation is: "the model's output moves up by this much because of this column", never "this column causes default".

**Explanation is not recourse.** "Your late payments raised your risk" is a statement about the model. Whether paying a late bill off changes the decision depends on how the model uses that column, which the SHAP value alone does not tell you. The partial dependence and ICE curves are closer to that question, and a counterfactual analysis is closer still.

## Where this stands in 2026

:::info Industry view

- scikit-learn 1.9 keeps permutation importance, partial dependence and ICE in `sklearn.inspection`, with the held-out-data and correlated-column caveats written into the guide; they are the dependable first tools and need no extra library.
- The `shap` library (version 0.52 was used here) explains scikit-learn tree models with exact Tree SHAP by default, and reports a classifier's output in log-odds unless told otherwise, which is why every number above is in log-odds.
- The standard open reference is Christoph Molnar's *Interpretable Machine Learning*, now in a third edition, which states plainly that Shapley values are not causal effects and that LIME needs kernel-width experiments before it can be trusted.
- Model documentation practice, such as model cards, asks for evaluation on slices and for known limitations in plain words, which is where the caveats of this chapter belong.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why compute permutation importance on a held-out set rather than the training set?</summary>

On training data the model is rewarded for columns it has partly memorised, so importances are inflated. In the code above every training importance is larger than the test one (for example `age` 0.0838 against 0.0527). Held-out importance shows what helps generalisation.

</details>

<details>
<summary><strong>Q2.</strong> Two columns are near-copies. What happens to their permutation importances and why?</summary>

Shuffling one leaves the other intact, so the model barely loses accuracy and both look unimportant. In the example `debt_ratio` fell from 0.0727 to 0.0317, with 0.0116 for its copy. Cluster correlated columns and explain the group.

</details>

<details>
<summary><strong>Q3.</strong> What does it mean that SHAP values "add up", and how would you check it?</summary>

For one row, the base value plus the sum of its SHAP values equals the model's output. Check it by comparing that sum with `decision_function` (log-odds for a gradient-boosted classifier). The example prints a maximum error of 5.55e-15.

</details>

<details>
<summary><strong>Q4.</strong> The partial dependence curve for `debt_ratio` rises gently, but ICE curves fan out. What does that tell you?</summary>

The effect of `debt_ratio` depends on another column, here `late_payments`. The average (rise 0.240) mixes a steep group (0.609) with a gentle one (0.193). The average alone would mislead about both groups.

</details>

<details>
<summary><strong>Q5.</strong> A column about collection calls tops every importance ranking. What do you check before reporting it as a driver?</summary>

Whether the column is a consequence of the outcome and unavailable at prediction time. If so it is a leak: it explains the model, not the world, and should be removed from the model, not advertised.

</details>

<details>
<summary><strong>Q6.</strong> Your LIME explanation changes sign for a feature when you change the random seed. What do you do?</summary>

Do not report it. Increase the neighbourhood width and the sample count, run several seeds, and keep only coefficients that are stable. If none are, use a method with a defined answer such as SHAP for trees.

</details>

## Further reading

- [scikit-learn user guide: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html): the algorithm, held-out advice and the correlated-column caveat.
- [scikit-learn user guide: partial dependence and ICE](https://scikit-learn.org/stable/modules/partial_dependence.html): `PartialDependenceDisplay`, centred ICE and the independence assumption.
- [Lundberg and Lee (2017): a unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874): the SHAP paper.
- [SHAP documentation: TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html): exact Tree SHAP, output units and the two perturbation methods.
- [Ribeiro, Singh and Guestrin (2016): "Why should I trust you?"](https://arxiv.org/abs/1602.04938): the LIME paper.
- [Molnar, Interpretable Machine Learning](https://christophmolnar.com/books/interpretable-machine-learning/): free to read online. Its Shapley-values chapter states plainly what SHAP does not tell you, and its LIME chapter covers the kernel-width and instability problems.
- [Mitchell et al. (2018): model cards for model reporting](https://arxiv.org/abs/1810.03993): where explanations and limitations are written down for readers.

## Check yourself

- I can explain the difference between a global and a local explanation, and pick a method for each.
- I can compute permutation importance on held-out data and say why training-set importances are inflated.
- I can read partial dependence and ICE curves, and spot an interaction from a fan of curves.
- I can explain what a SHAP value is, check that SHAP values add up to the model output, and read a waterfall.
- I can build a LIME-style surrogate and show that its answer depends on the neighbourhood width and the seed.
- I can explain why correlated columns split importance and why importance is not causation.
