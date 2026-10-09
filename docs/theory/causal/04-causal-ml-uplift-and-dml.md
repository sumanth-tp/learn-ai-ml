---
id: causal-ml-uplift-dml
title: "Causal Machine Learning in Practice: Uplift Modelling and Double Machine Learning"
sidebar_label: "4 · Causal ML in practice"
sidebar_position: 4
slug: /theory/causal/causal-ml-uplift-and-dml
description: "Who should get the treatment, and how to use flexible machine-learning models for effects without letting them bias the answer: S, T and X learners for uplift, and double machine learning with cross-fitting, checked against a simulated truth."
tags: [causal-inference, uplift-modelling, heterogeneous-treatment-effects, meta-learners, double-machine-learning, cross-fitting, econml]
---

import Infographic from '@site/src/components/Infographic';
import UpliftBudgetLab from '@site/src/components/viz/UpliftBudgetLab';

**In one line.** Once you know the average effect, the useful question is who it works for, and the safe way to use machine-learning models for it is to predict the treatment and the outcome from the other variables first and then study only what those predictions leave unexplained.

:::tip Before you start
- **You should already know** how to estimate an average effect from a randomised experiment or by adjustment ([experiments and adjustment](/docs/theory/causal/experiments-and-adjustment)), and how gradient-boosted trees are fitted and evaluated ([gradient boosting in practice](/docs/theory/ml/gradient-boosting-in-practice)).
- **Reading time:** about 50 minutes. The code takes about three minutes to run on a CPU.
- **After this chapter you can** explain why predicting who buys is not the same as predicting who responds, build and compare S, T and X learners, compute the value of a targeting policy, and run double machine learning with cross-fitting while saying what each step protects against.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every number comes from the code below, on simulated data where the true effect of each customer is known. Environment: Python 3.14, NumPy 2.5.3, scikit-learn 1.9.1, EconML 0.17.0 (the PyPI release dated 31 July 2026). Sources were opened on 8 October 2026.
:::

## In 30 seconds

A shop sends a discount coupon to its customers. Some would have bought anyway, so the coupon only costs money. Some will buy only if they get the coupon. A few are put off by it and buy less. The shop wants to send the coupon to the second group, and a model that predicts who will buy cannot find them, because it ranks the people who would have bought anyway on top.

Uplift modelling predicts the change in behaviour caused by the coupon, customer by customer. The second half of the chapter is about a different risk: when you use a flexible machine-learning model to remove the effect of other variables, its small systematic errors leak into the effect you want. Double machine learning is the recipe that stops the leak.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Heterogeneous effect | The effect differs between people | +2.5 for some, -1.4 for others |
| CATE | Conditional average treatment effect: the average effect for people with given features | Effect for under-40s with low past spend |
| Uplift | The same idea, in marketing language | Extra spend caused by the coupon |
| Persuadable | A person who responds only if treated | Buys with the coupon, not without |
| Sleeping dog | A person the treatment makes worse off | Loyal customer annoyed by a coupon |
| Meta-learner | A recipe that builds an effect model out of ordinary prediction models | S, T and X learners |
| Nuisance model | A model you need on the way but do not care about | Predicting sales from features |
| Cross-fitting | Predict each fold with models trained on the other folds | Five folds |
| Orthogonal | Insensitive to small errors in the nuisance models | The double ML score |

## The idea in plain words

**Why prediction is not enough.** Take 100 customers. A model of purchase finds 30 likely buyers. Most would buy without any coupon. If you give them one, you pay for nothing. What matters is the difference between buying with and without the coupon, and that difference is never observed for one person, the problem from the first chapter. Randomise the coupon and the problem becomes manageable: on average, over people with the same features, the difference between treated and untreated is the effect for those features.

**Three recipes for an effect model**, all built from ordinary regressors:

- **S-learner (single).** Fit one model of spend from the features and the treatment indicator. Predict for each customer with the indicator set to 1 and again with it set to 0, and subtract. Simple, but a regularised model may barely use the indicator and so shrink every effect toward zero.
- **T-learner (two).** Fit one model on treated customers and another on untreated customers. The effect is the difference of their predictions. It cannot ignore the treatment, but it subtracts two noisy models.
- **X-learner (cross).** Fit the two models as above. Impute each treated customer's effect as their actual spend minus the untreated model's prediction, and each untreated customer's effect as the treated model's prediction minus their actual spend. Fit new models to those imputed effects and blend them. It uses the imputed effects as targets, which can help when one arm is much bigger than the other.

**Why a flexible model can mislead on averages.** Suppose sales depend on a price cut and on twenty other variables in a nonlinear way. A random forest can learn the other variables' contribution well, but any model shrinks and smooths. If you subtract its prediction of sales and regress the remainder on the price cut, the price cut's own variation is partly explained by the same variables, and the forest has already given part of the price cut's effect to them. The result is biased, even with a huge data set. This is regularisation bias.

**Double machine learning (DML)** removes it with the idea of residualising both sides. Predict the outcome from the other variables, predict the treatment from the other variables, subtract each prediction, and regress the outcome's leftover on the treatment's leftover. What remains of the treatment is the part the other variables cannot explain, the part that behaves like a coin flip. Small errors in either prediction then have only a second-order effect on the answer, which is what orthogonal means. The second ingredient is **cross-fitting**: predict each row with models trained on other rows, so a model that memorises its training data cannot hide the residual.

<Infographic src="/img/causal/uplift-segments.svg" alt="Four cards: persuadables, 17.6 per cent of customers with effect plus 2.5; older customers, 25.3 per cent with effect plus 0.5; indifferent customers, 44.6 per cent with effect 0; sleeping dogs, 12.5 per cent with effect minus 1.36. A table gives profit per customer: treat nobody 0, treat everyone minus 0.104, random 20 per cent minus 0.021, a perfect ranking of the top 20 per cent 0.352." caption="Read the four cards left to right and subtract the coupon cost of 0.5 from each effect. Only the first is worth treating. The table shows how far a good ranking is from sending to everyone." />

## Worked example, step by step

**Value of a targeting policy.** A coupon costs 0.5. Customers fall into four groups: 17.6 per cent persuadables (effect +2.5), 25.3 per cent older customers (+0.5), 44.6 per cent indifferent (0) and 12.5 per cent sleeping dogs (-1.36 on average).

1. **Net gain per customer treated.** Persuadables: 2.5 - 0.5 = 2.0. Older customers: 0.5 - 0.5 = 0. Indifferent: 0 - 0.5 = -0.5. Sleeping dogs: -1.36 - 0.5 = -1.86.
2. **Treat only the persuadables.** 0.176 x 2.0 = 0.352 per customer in the whole population.
3. **Treat everyone.** 0.176 x 2.0 + 0.253 x 0 + 0.446 x (-0.5) + 0.125 x (-1.86) = 0.352 + 0 - 0.223 - 0.233 = -0.104. A loss.
4. **Treat a random 20 per cent.** 0.2 x the average net gain of -0.104 = -0.021.
5. **Read the lesson.** The average effect is positive, 0.396 per customer, but below the cost of 0.5, so treating everyone loses money. A ranking that finds the persuadables turns a loss into a profit of 0.352.

**Double machine learning with four customers.** Suppose the other variables explain part of the price cut and part of sales. After predicting both, the leftovers are:

| Customer | Price cut minus its prediction | Sales minus its prediction |
| --- | --- | --- |
| 1 | -2 | -1.8 |
| 2 | -1 | -1.1 |
| 3 | +1 | +0.9 |
| 4 | +2 | +2.2 |

6. **Multiply and add.** (-2)(-1.8) + (-1)(-1.1) + (1)(0.9) + (2)(2.2) = 3.6 + 1.1 + 0.9 + 4.4 = 10.0.
7. **Sum the squared price-cut leftovers.** 4 + 1 + 1 + 4 = 10.
8. **Divide.** 10.0 / 10 = 1.0. The effect is 1.0 sales per unit of price cut, estimated from what the other variables could not explain.

<Infographic src="/img/causal/learner-scoreboard.svg" alt="A table comparing S, X and T learners and an oracle on correlation with the true effect, root mean squared error, profit when treating the top 20 per cent, and the mean and spread of correlation over eight datasets: S 0.854 with spread 0.026, X 0.711, T 0.637. Three cards describe each learner." caption="Look at the correlation column first. The simplest learner wins on this data, and the one that subtracts two separate models is the weakest." />

## How it works

### What does an effect model need from the data?

A randomised experiment with the features recorded before treatment. With randomisation, comparing outcomes among customers with equal features is fair, so the models can be fitted without worrying about confounding. With observational data you also need the assumptions from the second chapter (all confounders measured, overlap), and the learners must then be combined with propensity weights or with double machine learning.

### How do I know an uplift model is good when I never see the true effect?

You cannot compute a customer's true effect on real data. You can compare ranking quality using the experiment itself: sort held-out customers by predicted uplift, take the top 20 per cent, and compare treated with untreated customers inside that slice. If the model ranks well, the gap in that slice is large. Block 1 prints this as "gain measured from the experiment", beside the profit computed from the true effects that only a simulation allows. The two agree on the order of the learners: S 0.376, X 0.367, T 0.324.

The standard summary of this idea is the uplift or Qini curve: the cumulative gain from treating the top k per cent. It is noisy on a small test set, so report the spread over resamples.

### Why does cross-fitting matter?

If the nuisance model sees a row when it is fitted, it can fit that row's noise. Its residual for that row is then too small, and so is the residual of the treatment, and the final ratio is biased, in either direction. Cross-fitting predicts each row from a model that never saw it. The cost is five fits instead of one. Block 2 shows that the damage depends on the learner: severe for a boosting model that memorises, mild for a forest.

### What does the final regression assume?

The model Y = θ T + g(X) + noise, with the treatment itself depending on X, is called partially linear: the effect θ is a constant, and everything else is unrestricted. Allow the effect to depend on a few features, and the final step becomes a regression of the outcome's leftover on the treatment's leftover multiplied by those features. Block 3 does this with EconML's `LinearDML`.

<Infographic src="/img/causal/dml-recipe.svg" alt="Five boxes giving the double machine learning recipe: split into five folds, predict sales from the covariates, predict the price cut from the covariates, take residuals, regress one residual on the other. Below, a table of seven estimators with mean and spread over 20 datasets: naive regression 1.961, linear controls 2.095, plug-in forest 0.296, double ML forest cross-fitted 1.161, boosting without cross-fitting 0.633, boosting cross-fitted 0.923; the true effect is 1.000." caption="Follow the recipe along the top, then read the table against the true 1.000. The estimates are scattered above and below it; only the cross-fitted rows come close." />

## Code you can run

All three blocks are self-contained and use only the CPU. Block 1 takes about 90 seconds and block 2 about 80 seconds, because they repeat the experiment on fresh datasets.

### 1. Who responds: S, T and X learners

A coupon experiment on 10,000 customers with spend as the outcome. The coupon helps young, low-spending, non-loyal customers (+2.5), slightly helps customers over 55 (+0.5), and hurts loyal high-spenders (-1.5, or -1.0 when over 55). Treatment is a fair coin.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

def make_customers(rng, n):
    age = rng.uniform(18, 70, n)
    past = rng.gamma(2.0, 20.0, n)
    loyal = rng.binomial(1, 0.3, n)
    base = 5 + 0.15 * past + 6 * loyal
    persuadable = (age < 40) & (past < 40) & (loyal == 0)
    sleeping_dog = (loyal == 1) & (past > 40)
    tau = np.where(persuadable, 2.5, 0.0) + np.where(sleeping_dog, -1.5, 0.0) + np.where(age >= 55, 0.5, 0.0)
    treated = rng.binomial(1, 0.5, n)
    spend = base + tau * treated + rng.normal(0, 4.0, n)
    return np.column_stack([age, past, loyal]), treated, spend, tau, persuadable, sleeping_dog

def fit(X, y):
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(X, y)

def learners(Xtr, ttr, ytr, Xte):
    s_model = fit(np.column_stack([Xtr, ttr]), ytr)
    s = s_model.predict(np.column_stack([Xte, np.ones(len(Xte))])) - s_model.predict(np.column_stack([Xte, np.zeros(len(Xte))]))
    m1, m0 = fit(Xtr[ttr == 1], ytr[ttr == 1]), fit(Xtr[ttr == 0], ytr[ttr == 0])
    x1 = fit(Xtr[ttr == 1], ytr[ttr == 1] - m0.predict(Xtr[ttr == 1]))
    x0 = fit(Xtr[ttr == 0], m1.predict(Xtr[ttr == 0]) - ytr[ttr == 0])
    return {"S-learner": s, "T-learner": m1.predict(Xte) - m0.predict(Xte),
            "X-learner": 0.5 * x1.predict(Xte) + 0.5 * x0.predict(Xte)}

def profit(score, truth, share=0.2, cost=0.5):
    chosen = np.argsort(-score)[: int(share * len(score))]
    return (truth[chosen] - cost).sum() / len(score)

def measured_gain(score, t, y, share=0.2):
    top = np.argsort(-score)[: int(share * len(score))]
    return (y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean()) * share

rng = np.random.default_rng(0)
X, t, y, tau, persuadable, dog = make_customers(rng, 10000)
print(f"true average effect {tau.mean():.3f}; persuadables {persuadable.mean():.3f} of customers (effect 2.5), "
      f"sleeping dogs {dog.mean():.3f} (effect {tau[dog].mean():.3f}); coupon costs 0.5")
Xtr, Xte, ttr, tte, ytr, yte, _, taute = train_test_split(X, t, y, tau, test_size=0.4, random_state=0)
print("learner      corr with truth   rmse   profit, top 20%   gain measured from the experiment")
for name, score in learners(Xtr, ttr, ytr, Xte).items():
    print(f"{name:12s} {np.corrcoef(score, taute)[0, 1]:11.3f} {np.sqrt(np.mean((score - taute) ** 2)):10.3f} "
          f"{profit(score, taute):12.3f} {measured_gain(score, tte, yte):18.3f}")
print(f"{'oracle':12s} {1:11.3f} {0:10.3f} {profit(taute, taute):12.3f}")
print(f"treat everyone: {(taute - 0.5).mean():.3f}   treat nobody: 0.000")

print("\nthe same comparison over 8 fresh datasets (mean and spread):")
runs = {}
for seed in range(1, 9):
    r = np.random.default_rng(seed)
    X, t, y, tau, _, _ = make_customers(r, 10000)
    Xtr, Xte, ttr, tte, ytr, yte, _, taute = train_test_split(X, t, y, tau, test_size=0.4, random_state=seed)
    for name, score in learners(Xtr, ttr, ytr, Xte).items():
        runs.setdefault(name, []).append((np.corrcoef(score, taute)[0, 1], profit(score, taute)))
for name, values in runs.items():
    v = np.array(values)
    print(f"{name:12s} corr {v[:, 0].mean():.3f} +/- {v[:, 0].std():.3f}   profit {v[:, 1].mean():.3f} +/- {v[:, 1].std():.3f}")
```

**Reading the output.** The simulated truth is printed first: an average effect of 0.396, 17.6 per cent persuadables and 12.5 per cent sleeping dogs. A coupon cost of 0.5 therefore exceeds the average effect: treating everyone loses 0.094 per customer.

The S-learner ranks best (correlation with the true effect 0.853, profit 0.308 from treating the top 20 per cent), the X-learner next (0.746, 0.274) and the T-learner last (0.648, 0.210). The oracle, who knows the true effects, earns 0.355, so the S-learner captures 87 per cent of the attainable profit. The measured gain from the experiment orders the learners the same way: 0.376, 0.367, 0.324.

Over eight fresh datasets the ordering holds: correlation 0.854 for S, 0.711 for X and 0.637 for T, with spreads of 0.03 to 0.05, smaller than the gaps between learners.

The honest surprise is the RMSE column. The S-learner has the best ranking but still an RMSE of 0.601 on effects whose typical size is 1, so its estimates of the size of each effect are rough. It ranks well enough to choose whom to treat, which is the decision, and poorly enough that you should not read its numbers as predictions of the extra spend. Ranking and calibration are different jobs.

Neither the textbook warning about the S-learner (it shrinks the effect to zero) nor the one about the T-learner (it is noisy) is a rule. Here the shrinkage costs nothing in ranking because the effect is large where the features that drive the baseline are, and the noise of the two-model difference hurts. On other data the order can change, so run all three.

**Line by line.**

- `persuadable`, `sleeping_dog` and the age term build the heterogeneous effect, so the truth is known for each customer. A real data set has no such column.
- `learners` returns all three effect estimates on the held-out customers. The S-learner predicts twice, once with the treatment column set to ones and once with zeros.
- `x1` and `x0` are the X-learner's second stage: they are fitted to imputed effects, not to spend.
- `profit` adds up (true effect minus cost) for the customers a ranking would treat, and divides by everyone. `measured_gain` is the same slice read from the randomised outcomes only.

### 2. Removing confounding with flexible models, with and without cross-fitting

A price cut raises sales by a true 1.0 per unit. Both the price cut and sales depend on six other variables in nonlinear ways, so the price cut is not randomised. The table compares naive regression, regression with linear controls, a plug-in forest, and double machine learning with two kinds of nuisance model, each with and without cross-fitting.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def make_world(rng, n, theta=1.0):
    X = rng.normal(size=(n, 6))
    price_cut = np.sin(2 * X[:, 0]) + np.abs(X[:, 1]) + 0.5 * X[:, 2] + rng.normal(0, 0.7, n)
    sales = theta * price_cut + 2 * np.sin(2 * X[:, 0]) + 1.5 * np.abs(X[:, 1]) - X[:, 3] ** 2 + X[:, 4] + rng.normal(0, 1, n)
    return X, price_cut, sales

def forest():
    return RandomForestRegressor(n_estimators=60, min_samples_leaf=5, random_state=0, n_jobs=1)

def boosted():
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.3, max_depth=5, random_state=0)

def dml(X, t, y, model, cross_fit=True, folds=5):
    if cross_fit:
        t_res, y_res = np.zeros(len(t)), np.zeros(len(y))
        for train, test in KFold(folds, shuffle=True, random_state=1).split(X):
            t_res[test] = t[test] - model().fit(X[train], t[train]).predict(X[test])
            y_res[test] = y[test] - model().fit(X[train], y[train]).predict(X[test])
    else:
        t_res = t - model().fit(X, t).predict(X)
        y_res = y - model().fit(X, y).predict(X)
    return (t_res @ y_res) / (t_res @ t_res)

def estimates(X, t, y):
    return {
        "naive regression of sales on price cut": LinearRegression().fit(t[:, None], y).coef_[0],
        "linear controls": LinearRegression().fit(np.column_stack([t, X]), y).coef_[0],
        "plug-in forest for sales only": (t @ (y - forest().fit(X, y).predict(X))) / (t @ t),
        "double ML, forest, cross-fitted": dml(X, t, y, forest),
        "double ML, forest, no cross-fitting": dml(X, t, y, forest, cross_fit=False),
        "double ML, boosting, cross-fitted": dml(X, t, y, boosted),
        "double ML, boosting, no cross-fitting": dml(X, t, y, boosted, cross_fit=False),
    }

rng = np.random.default_rng(4)
runs = [estimates(*make_world(rng, 1000)) for _ in range(20)]
print("20 fresh datasets, n = 1000, true effect 1.000")
print(f"{'estimator':40s} {'mean':>7s} {'sd':>7s}")
for name in runs[0]:
    v = np.array([r[name] for r in runs])
    print(f"{name:40s} {v.mean():7.3f} {v.std():7.3f}")
```

**Reading the output.** The naive regression gives 1.961 and linear controls give 2.095: both are nearly double the truth, because the other variables act nonlinearly and a linear control cannot remove them. The plug-in forest, which predicts sales from the other variables and regresses the leftover on the price cut, gives 0.296. It is not noisy (spread 0.016) but it is badly biased: the forest has absorbed most of the price cut's effect into its prediction of sales, because the price cut itself depends on the same variables.

Double machine learning gets close. With a forest and cross-fitting the mean is 1.161 with a spread of 0.073, so there is still a bias of 0.16 at n = 1,000 that comes from the forest's smoothing. With boosting and cross-fitting the mean is 0.923, a bias of -0.08.

The cross-fitting result depends on the learner. For the forest, skipping cross-fitting costs little: 1.110 instead of 1.161, because each tree is built on a bootstrap sample, so a row's in-sample prediction is not a copy of its value. For gradient boosting it is severe: 0.633 instead of 0.923. A boosted model with 200 deep trees nearly memorises its training rows, so the in-sample residuals are tiny and carry little of the real signal.

The surprise is that cross-fitting is not a universal fix to remember to apply; it is insurance whose value depends on how much the learner overfits. It is cheap, so use it always.

**Line by line.**

- `dml` with `cross_fit=True` loops over five folds; each row's two residuals come from models trained on the other four.
- `t_res @ y_res / (t_res @ t_res)` is the final regression with one variable and no intercept.
- The plug-in row skips the treatment's prediction: it divides by `t @ t`, the raw treatment, not its leftover. That omission is the whole difference from double machine learning.

### 3. A heterogeneous effect with a library

EconML's `LinearDML` does cross-fitting, residualising and the final regression, and also reports intervals. We let the true effect depend on one covariate: 1.0 + 0.5 x, and ask for the effect at three values of x.

```python
import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor

def make_world(rng, n):
    X = rng.normal(size=(n, 6))
    price_cut = np.sin(2 * X[:, 0]) + np.abs(X[:, 1]) + 0.5 * X[:, 2] + rng.normal(0, 0.7, n)
    effect = 1.0 + 0.5 * X[:, 5]
    sales = effect * price_cut + 2 * np.sin(2 * X[:, 0]) + 1.5 * np.abs(X[:, 1]) - X[:, 3] ** 2 + X[:, 4] + rng.normal(0, 1, n)
    return X, price_cut, sales

def forest():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=1)

rng = np.random.default_rng(8)
X, t, y = make_world(rng, 4000)
est = LinearDML(model_y=forest(), model_t=forest(), cv=5, random_state=0)
est.fit(y, t, X=X[:, 5:6], W=X[:, :5])
print(f"average effect: {est.ate_inference(X[:, 5:6]).mean_point:.3f}  (true 1.000)")
print(f"intercept {est.intercept_:.3f} (true 1.000), slope on x6 {est.coef_[0]:.3f} (true 0.500)")
print("effect at x6 = -1, 0, +1:")
for v in (-1.0, 0.0, 1.0):
    point = est.effect(np.array([[v]]))[0]
    low, high = est.effect_interval(np.array([[v]]), alpha=0.05)
    print(f"  x6 = {v:+.0f}: {point:.3f}  (95% interval {low[0]:.3f} to {high[0]:.3f})  true {1.0 + 0.5 * v:.3f}")
```

**Reading the output.** The average effect is 1.050 against a true 1.000. The intercept is 1.042 against 1.0 and the slope on the covariate is 0.490 against 0.5. The estimated effect at x = -1, 0 and +1 is 0.552, 1.042 and 1.533, against the true 0.5, 1.0 and 1.5, and each 95 per cent interval contains the truth: 0.468 to 0.636, 0.984 to 1.100 and 1.446 to 1.619.

The small upward bias of about 0.04 to 0.05 echoes block 2: random forests as nuisance models leave a little regularisation bias even after orthogonalisation, and it shrinks with more data. The intervals still contain the truth at all three points, but the interval at x = 0 is narrow enough that a slightly larger bias would have excluded it.

**Line by line.**

- `est.fit(y, t, X=X[:, 5:6], W=X[:, :5])` separates the variables whose effect heterogeneity we model (`X`, one column) from the controls we only need to remove (`W`, five columns).
- `cv=5` is the cross-fitting. Replace `forest()` with any scikit-learn regressor to change the nuisance models.
- `est.effect_interval(..., alpha=0.05)` returns a 95 per cent interval for the effect at the chosen points.

## Try it yourself

The lab applies the exact formulas to the four customer groups of block 1, ranking customers by true effect. Its defaults, a coupon cost of 0.5 and the top 20 per cent, give 0.352, the profit of the worked example. The formulas were checked against a simulation of four million customers in five settings, with differences up to 0.0002.

<UpliftBudgetLab />

**What each control does.**

- **Coupon cost** is what each coupon costs. A customer is worth treating if their effect exceeds it.
- **Share of customers who get a coupon** sets how far down the ranking you go.
- The blue line is the profit of ranking by true effect and the orange line is random targeting. The solid vertical line is your chosen share and the ring is the best share.
- Click **show data** to see the four groups and the exact numbers.

**Try it yourself.**

1. Leave the cost at 0.5 and move the share from 0.2 to 1. Profit rises to 0.352 at a share of 0.176, stays flat until 0.429 (the older customers net zero), then falls and ends at -0.104 when everyone gets a coupon. Why: after the first two groups, each extra customer costs more than they bring.
2. Set the cost to 0. The best share becomes 0.429 and the best profit 0.567. Why: with free coupons, every customer with a positive effect is worth treating, which includes the older customers.
3. Set the cost to 2.0. The best share falls to 0.176 and the best profit to 0.088. Why: only the persuadables, whose effect is 2.5, still exceed the cost, and each earns just 0.5 net.

## Designing with it

1. **Randomise first.** An uplift model needs an experiment with the treatment assigned independently of the features. If the existing treatment was targeted, hold out a small random group.
2. **Define the decision before the model.** The unit is profit per customer, with the cost of treatment, not accuracy. The lab's best share is where the next customer's effect equals the cost.
3. **Compare ranking by experiment, not by model error.** Use the gain among the top slice in held-out randomised data, with a spread over resamples.
4. **Try S, T and X together.** None is always best. Block 1 had S first and T last; the opposite is easy to construct.
5. **For observational effects, use orthogonal estimators with cross-fitting**, report an interval, and check the nuisance models' own fit, since DML does not fix a missing confounder.

## Where this stands in 2026

Orthogonal and doubly robust estimators with machine-learning nuisance models are standard for observational effect estimation, following the double machine learning paper by Chernozhukov and colleagues (arXiv 1608.00060, first submitted in 2016, latest version revised on 3 November 2024). EconML 0.17.0 ships `LinearDML`, forest-based and meta-learner estimators under an MIT licence. Hernán and Robins' 2026 edition includes a section on doubly robust machine-learning estimators. The open questions in practice are validation (how to judge an effect model without the truth) and drift (effects that change as the customer base does).

## Common mistakes

1. **Targeting by predicted purchase.** It feels right because those customers convert. They convert without the coupon too. Rank by uplift.
2. **Trusting the size of a predicted effect.** Block 1's best learner has an RMSE of 0.601 yet ranks well. Use the ranking for decisions and measure the size on a randomised hold-out.
3. **Dropping cross-fitting because the forest looked fine.** Block 2: no loss for the forest, a drop from 0.923 to 0.633 for boosting. Always cross-fit.
4. **Fitting a flexible model of the outcome and calling the leftover coefficient an effect.** The plug-in forest gave 0.296 for a truth of 1.0. Residualise the treatment too.
5. **Calling any estimate causal because it came from a machine-learning pipeline.** DML removes bias from the models' errors, not from an unmeasured confounder.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Why does ranking customers by their probability of buying not find the persuadables?</summary>

Probability of buying includes customers who would buy anyway. A coupon given to them changes nothing, and the sleeping dogs may also rank high. The target is the change in behaviour caused by the coupon, which only an uplift model, or an experiment, can estimate.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> In the worked example, why does treating everyone lose 0.104 per customer when the average effect is positive?</summary>

The average effect is 0.396, below the coupon cost of 0.5, so the average customer loses 0.104. The loss comes from the indifferent customers, who cost 0.5 each for no gain, and the sleeping dogs, who cost 1.86 each.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Compute the double machine learning estimate if the price-cut leftovers are -1, 0, 1 and the sales leftovers are -0.5, 0.1, 1.5.</summary>

Numerator: (-1)(-0.5) + 0 x 0.1 + 1 x 1.5 = 0.5 + 0 + 1.5 = 2.0. Denominator: 1 + 0 + 1 = 2. The estimate is 1.0.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why does the plug-in forest give 0.296, so far below 1.0, and not something noisy around 1.0?</summary>

Its prediction of sales from the other variables already contains the price cut's effect, because the price cut is a function of those variables plus noise. Subtracting it removes that part of the effect, and only the effect carried by the price cut's unexplained noise is left to find. Its small spread, 0.016, shows it is a systematic bias, not noise, and more data would not repair it.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Block 2's boosting model without cross-fitting gave 0.633 and the forest 1.110. Explain the difference.</summary>

Two hundred deep boosted trees fit their own training rows closely, so in-sample predictions of the price cut and of sales are close to the actual values. The residuals are small and mostly not informative: the part of the price cut that the model could not explain was explained by memorising noise. A forest averages trees fitted to bootstrap samples, so in-sample predictions are not copies of the rows. Cross-fitting removes the memorisation by predicting each row from a model that never saw it.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> At a coupon cost of 0.5 the older customers (effect +0.5) are indifferent. If the cost drops to 0.4, what changes in the lab's best share and best profit?</summary>

The best share rises from 0.176 to 0.429, since both persuadables and older customers now exceed the cost, and the best profit becomes 0.176 x 2.1 + 0.253 x 0.1 = 0.370 + 0.025 = 0.395. A small change in cost adds a whole group, which is why the best share is sensitive near the effect of a large group.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W and Robins J, "Double/Debiased Machine Learning for Treatment and Causal Parameters", arXiv 1608.00060, first submitted July 2016, latest version v7 of 3 November 2024. Regularisation bias, orthogonal scores and cross-fitting.
- Hernán MA and Robins JM, *Causal Inference: What If*, edition dated 19 August 2026. Chapter 18, "Variable selection and high-dimensional data", including its sections on causal inference and machine learning and on doubly robust machine-learning estimators; chapter 4 on effect modification. Contents list read. Free to read at the author's page.
- Facure Alves M, *Causal Inference for the Brave and True*, part II: heterogeneous treatment effects and personalisation, evaluating causal models, meta-learners, debiased and orthogonal machine learning. Contents list read.
- [EconML on PyPI](https://pypi.org/project/econml/): release 0.17.0 of 31 July 2026, MIT licence, Python 3.9 to 3.14. Version 0.17.0 was run.
- [scikit-learn user guide](https://scikit-learn.org/stable/modules/ensemble.html): the forests and gradient boosting used in the blocks. Version 1.9.1 was run.

## Check yourself

- I can explain why a purchase model and an uplift model rank customers differently.
- I can build S, T and X learners and compare them on held-out randomised data.
- I can compute the value of a targeting policy and find the best share to treat.
- I can run double machine learning by hand, and say what cross-fitting and orthogonalising each protect against.
- I can say what double machine learning does not fix.

## Where to go next

This is the last chapter of the causal series. A related chapter: [experimentation and A/B testing](/docs/mlops/platform/experimentation-and-ab-testing), which runs the randomised experiment these models need. Another: [explaining predictions](/docs/theory/ml/explaining-predictions), because effect models still need explanations that a business owner can read.
