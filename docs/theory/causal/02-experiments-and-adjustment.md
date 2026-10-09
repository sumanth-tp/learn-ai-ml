---
id: causal-experiments-adjustment
title: "Randomised Experiments and Adjustment: Backdoor, Propensity Scores, Matching and Weighting"
sidebar_label: "2 · Experiments and adjustment"
sidebar_position: 2
slug: /theory/causal/experiments-and-adjustment
description: "How a coin flip removes confounding, and how regression, matching, inverse-probability weighting and doubly robust estimators try to imitate it from observational data, compared over 200 simulated repeats against a known true effect."
tags: [causal-inference, randomisation, propensity-score, ipw, matching, doubly-robust, backdoor-criterion, overlap]
---

import Infographic from '@site/src/components/Infographic';
import IpwLab from '@site/src/components/viz/IpwLab';

**In one line.** Randomising treatment guarantees a fair comparison, and when you cannot randomise you can imitate it by adjusting for what drove treatment, with weights, matches or a regression, each of which fails in its own way.

:::tip Before you start
- **You should already know** the potential-outcome language and the idea of a confounder ([potential outcomes and confounding](/docs/theory/causal/potential-outcomes-and-confounding)), and logistic regression as a model of a yes or no outcome ([classification and logistic regression](/docs/theory/ml/classification-and-logistic-regression)).
- **Reading time:** about 45 minutes, plus about ten seconds to run the code.
- **After this chapter you can** check whether a randomised experiment balanced its groups, compute inverse-probability weights by hand, explain why a weighted estimate can be unbiased and still useless, and say which of three estimators survives which modelling mistake.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. All numbers come from the code below, on simulated data with a known effect of 2.0. Environment: Python 3.14, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1. Sources were opened on 8 October 2026.
:::

## In 30 seconds

Suppose a doctor gives a new drug to the sickest patients. Comparing treated with untreated patients would make the drug look harmful, because the treated were sicker to begin with. A coin flip would have split sick and healthy patients evenly, and the comparison would be fair.

When the coin is not available, you have three options. You can build a model of how the outcome depends on patient features. You can weight patients so the treated and untreated crowds look alike, the way a pollster counts each under-represented voter extra. Or you can pair each treated patient with a similar untreated one. This chapter tries all three on data where we know the answer.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Randomised experiment | Treatment assigned by chance, independent of everything else | A coin flip per user |
| Balance | The groups have similar covariate distributions | Same average age in both |
| Standardised mean difference (SMD) | Gap in a covariate's average, divided by its spread | 0.04 is balanced, 0.67 is not |
| Propensity score | The chance of getting the treatment, given the covariates | 0.8 for engaged users |
| Inverse-probability weight | 1 divided by the chance of the treatment the person got | 1.25 or 5 |
| Matching | Pairing treated and untreated units with similar covariates or scores | Nearest neighbour on the score |
| Overlap (positivity) | Every type of person has some chance of each treatment | No score of exactly 0 or 1 |
| Effective sample size | How many equal-weight users the weighted sample is worth | 320 of 500 |
| Doubly robust | An estimator that is right if either of two models is right | Augmented weighting |

## The idea in plain words

Return to the shop of 1,000 users from the previous chapter. The notification went to 80 per cent of engaged users and 20 per cent of the rest. If instead a coin had decided, 50 per cent of each group would be notified, the groups would look alike, and a plain difference in means would be right. That is why a randomised experiment is the standard of evidence: it makes the treatment independent of everything, including things nobody measured.

Observational data are what you get when someone, or something, chose the treatment. If the choice depended only on things you recorded, there is a way back to a fair comparison. Hernán and Robins call the idea **conditional randomisation**: within each type of user, as the choice was made without reference to anything else, the treated and untreated are comparable. Two estimates then follow.

**Standardisation** compares inside each type of user and averages the answers by how common each type is. **Inverse-probability weighting** (IPW) does something equivalent in another way: it gives each user a weight of one over the probability of the treatment they received. An engaged user who was notified had an 80 per cent chance of it, so counts 1/0.8 = 1.25 times. An engaged user who was not notified had a 20 per cent chance of that, so counts 1/0.2 = 5 times. Every kind of user now appears equally often among the notified and the un-notified: a **pseudo-population** where treatment looks like a coin flip.

With many covariates you cannot compare inside every type of user. The **propensity score**, e(x) = P(treated | covariates x), compresses them into a single number, and Rosenbaum and Rubin's result is that, if the covariates are enough, comparing users with equal scores is enough. In words: it is not necessary to match on everything, only on the chance of treatment.

The chapter's experiment pits these methods against each other with two deliberate mistakes in play: a regression that misses a curve in how covariates affect the outcome, and a propensity model that misses the same curve in how they affect the treatment.

<Infographic src="/img/causal/coin-or-choice.svg" alt="Two panels with bars for the standardised mean difference of three covariates. When a coin assigns treatment the bars are 0.040, 0.011 and zero and the difference in means is 2.145 with an interval containing 2. When covariates assign treatment the bars are 0.666, 0.417 and 0.640 and the difference in means is 5.462 with an interval that misses 2." caption="Compare the bar lengths against the dotted line at 0.1. A coin leaves the covariates balanced and the interval around the true 2.0; a choice made from the covariates leaves large imbalance and an answer that is off by more than 3." />

## Worked example, step by step

The same 1,000 users: 500 engaged, 500 others. Notification probabilities are 0.8 for engaged users and 0.2 for others. Average spend with the notification is 16 for engaged and 12 for others; without it 14 and 10.

1. **Find each user's propensity.** Engaged: 0.8. Others: 0.2.
2. **Give each person a weight.** A notified engaged user: 1/0.8 = 1.25. A not-notified engaged user: 1/(1 - 0.8) = 5. A notified other user: 1/0.2 = 5. A not-notified other user: 1/(1 - 0.2) = 1.25.
3. **Count the weighted crowd.** Engaged, notified: 400 users x 1.25 = 500. Engaged, not notified: 100 x 5 = 500. Others, notified: 100 x 5 = 500. Others, not notified: 400 x 1.25 = 500. Every cell now holds 500: treatment no longer depends on engagement.
4. **Take weighted averages.** Notified: (500 x 16 + 500 x 12) / 1000 = 14.0. Not notified: (500 x 14 + 500 x 10) / 1000 = 12.0.
5. **Subtract.** 14.0 - 12.0 = 2.0. The naive comparison said 4.4.
6. **Count what the weights cost.** The weights are 1.25 and 5, not all the same, so the notified group of 500 is worth fewer than 500 equal-weight users. The effective sample size formula, (sum of weights) squared divided by (sum of squared weights), gives 320 for this group. Weighting buys fairness and pays in precision.

<Infographic src="/img/causal/pseudo-population.svg" alt="Four cards: engaged and notified, 400 users times weight 1.25 equals 500 weighted; engaged not notified, 100 times 5 equals 500; others notified, 100 times 5 equals 500; others not notified, 400 times 1.25 equals 500. Three cards on the right give the weighted means 14.0 and 12.0 and the effect 2.0." caption="Every cell's weighted count is 500, so each type of user appears equally among the notified and the un-notified. Read the right-hand cards for the arithmetic that follows." />

## How it works

### Why does randomisation work, and how do I check it did?

A random assignment is independent of the potential outcomes by design, so the treated and untreated are exchangeable, and the difference in means estimates the ATE without any adjustment. The estimate still has sampling noise, and its interval comes from the usual two-sample formula.

Check that the randomisation worked by comparing covariates measured before treatment across arms. The **standardised mean difference** divides the gap in a covariate's mean by the pooled standard deviation. A common working rule treats an absolute SMD below 0.1 as balanced. In a properly randomised experiment, the SMDs will be small because of chance alone; large ones suggest a broken assignment, such as a bug that gave one arm all the new accounts.

### What does regression adjustment assume?

Fit an outcome model with the treatment and the covariates and read the treatment coefficient. It is unbiased only if the model has the right functional form. A straight line through a curved relationship leaves the curve's effect in the residual, and if the curve is related to treatment, it leaks into the coefficient. Block 3's "wrong form" regression misses a squared term and returns 4.103 for a truth of 2.0.

### What does inverse-probability weighting assume?

It needs a model of treatment, the propensity model, and positivity: no one has a propensity of 0 or 1. Its estimate of the mean outcome under treatment is the average of T times Y divided by e(X):

$$\hat\mu_1 = \frac{1}{n}\sum_i \frac{T_i Y_i}{e(X_i)}.$$

In words: count each treated person as many times as it takes to stand in for everyone who looked like them. The untreated mean is built the same way with 1 - e(X), and the effect is their difference.

The price is variance. A treated user with a propensity of 0.02 receives a weight of 50, and a handful of such users can dominate the estimate.

### How does matching differ?

Matching pairs each treated unit with the untreated unit whose propensity score is closest, and averages the outcome differences. It uses no outcome model and no weights, which makes it easy to explain, but each pair is only approximately alike. The leftover differences within pairs leave a small bias, and matching on a noisy estimated score adds to it.

### What does "doubly robust" mean?

The augmented IPW estimator uses both models. For each user it takes the outcome model's predicted effect, then adds a weighted correction made from that model's prediction errors:

$$\hat\tau = \frac{1}{n}\sum_i \Big[ \hat m_1(X_i) - \hat m_0(X_i) + \frac{T_i (Y_i - \hat m_1(X_i))}{e(X_i)} - \frac{(1 - T_i)(Y_i - \hat m_0(X_i))}{1 - e(X_i)} \Big].$$

Here m1 and m0 are the outcome model's predictions for each user if treated and if not treated. In words: start from the regression answer, then use the weights to clean up wherever the regression got the wrong answer. If the outcome model is right, the correction averages to zero. If the outcome model is wrong but the propensity model is right, the weighted correction removes the bias. Only when both are wrong does it fail, and block 3 shows exactly that.

### What can go wrong with overlap?

If some users almost never get treatment, comparing them to treated users is extrapolation. Weights explode, the effective sample size collapses and one point decides the result. The usual remedy is **trimming**: clip propensities to a range such as 0.05 to 0.95, or drop users outside it. Trimming lowers variance and adds bias, because you have quietly changed whom you are estimating the effect for.

<Infographic src="/img/causal/estimator-scoreboard.svg" alt="A table of eight estimators with mean, standard deviation and root mean squared error over 200 repeats. Regression with the right form and doubly robust with a wrong propensity model both reach 2.004 and 2.003 with spread 0.053. Weighting with the right propensity has spread 0.913. Matching has mean 2.111. The naive difference is 5.736 and regression with the wrong form 4.103." caption="Scan the mean column against the true 2.000, then the spread column. The surprise is in the weighting row: it is on target and 17 times noisier than the regression." />

## Code you can run

Every block is self-contained and CPU only. The simulated world has three covariates. Treatment depends on two of them linearly and on the square of the third, and so does the outcome. The true effect of treatment is exactly 2.0.

### 1. A coin against a choice

The first block builds the world twice, once with the coin and once with confounded assignment, and reports balance and the difference in means.

```python
import numpy as np

def make_world(n, rng, randomised):
    x = rng.normal(size=(n, 3))
    logit = 0.9 * x[:, 0] + 0.6 * x[:, 1] + 0.8 * (x[:, 2] ** 2 - 1)
    p = np.full(n, 0.5) if randomised else 1 / (1 + np.exp(-logit))
    t = rng.binomial(1, p)
    y0 = 3 + 2 * x[:, 0] + 1.5 * x[:, 1] + 2 * x[:, 2] ** 2 + rng.normal(size=n)
    return x, t, y0 + 2.0 * t

def smd(v, t):
    a, b = v[t == 1], v[t == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)

rng = np.random.default_rng(3)
for label, randomised in (("assigned by a coin", True), ("assigned by covariates", False)):
    x, t, y = make_world(2000, rng, randomised)
    gap = y[t == 1].mean() - y[t == 0].mean()
    se = np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum())
    print(f"{label}: treated {t.sum()} of {len(t)}, difference in means {gap:.3f} "
          f"(95% interval {gap - 1.96 * se:.3f} to {gap + 1.96 * se:.3f}), true effect 2.000")
    print(f"   standardised mean differences: x1 {smd(x[:, 0], t):+.3f}, x2 {smd(x[:, 1], t):+.3f}, "
          f"x3 squared {smd(x[:, 2] ** 2, t):+.3f}")
```

**Reading the output.** With a coin, the 984 treated users out of 2,000 are balanced: the SMDs are 0.040, 0.011 and 0.000. The difference in means is 2.145 with an interval 1.806 to 2.484 that contains 2. With assignment by covariates, the imbalance is large (0.666, 0.417, 0.640) and the difference is 5.462, with an interval of 5.165 to 5.759 that is narrow and wrong. A confident interval around a biased estimate is the typical look of confounding.

Notice that the coin's point estimate, 2.145, is 0.145 above the truth. Its interval is wide enough to include it. Randomisation removes bias, not noise.

**Line by line.**

- `np.full(n, 0.5)` replaces the confounded propensity with a coin.
- `smd` divides by the average of the two groups' variances, which keeps the measure symmetric between arms.
- The `x[:, 2] ** 2` column is a non-linear feature. A balance check on the raw covariate would have missed that treatment depends on its square.

### 2. One sample, four ways

Now estimate a propensity score with logistic regression, inspect overlap and balance after weighting, and compute IPW, doubly robust and matching estimates from a single confounded sample.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

def make_world(n, rng):
    x = rng.normal(size=(n, 3))
    logit = 0.9 * x[:, 0] + 0.6 * x[:, 1] + 0.8 * (x[:, 2] ** 2 - 1)
    t = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    y0 = 3 + 2 * x[:, 0] + 1.5 * x[:, 1] + 2 * x[:, 2] ** 2 + rng.normal(size=n)
    return x, t, y0 + 2.0 * t

def smd(v, t, w):
    a = np.average(v[t == 1], weights=w[t == 1])
    b = np.average(v[t == 0], weights=w[t == 0])
    return (a - b) / np.sqrt((v[t == 1].var(ddof=1) + v[t == 0].var(ddof=1)) / 2)

rng = np.random.default_rng(3)
x, t, y = make_world(2000, rng)
f = np.column_stack([x, x[:, 2] ** 2])
ps = LogisticRegression(C=1e6, max_iter=2000).fit(f, t).predict_proba(f)[:, 1]
w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
print(f"propensity range {ps.min():.3f} to {ps.max():.3f}; largest weight {w.max():.1f}")
for g, name in ((1, "treated"), (0, "control")):
    ess = w[t == g].sum() ** 2 / (w[t == g] ** 2).sum()
    print(f"effective sample size, {name}: {ess:.0f} of {(t == g).sum()}")
ones = np.ones(len(t))
for name, v in (("x1", x[:, 0]), ("x2", x[:, 1]), ("x3 squared", x[:, 2] ** 2)):
    print(f"balance on {name:10s} before {smd(v, t, ones):+.3f}   after weighting {smd(v, t, w):+.3f}")

ipw = np.mean(t * y / ps) - np.mean((1 - t) * y / (1 - ps))
m1 = LinearRegression().fit(f[t == 1], y[t == 1]).predict(f)
m0 = LinearRegression().fit(f[t == 0], y[t == 0]).predict(f)
aipw = np.mean(m1 - m0 + t * (y - m1) / ps - (1 - t) * (y - m0) / (1 - ps))
score = np.log(ps / (1 - ps))[:, None]
to_control = NearestNeighbors(n_neighbors=1).fit(score[t == 0]).kneighbors(score[t == 1])[1][:, 0]
to_treated = NearestNeighbors(n_neighbors=1).fit(score[t == 1]).kneighbors(score[t == 0])[1][:, 0]
matching = ((y[t == 1] - y[t == 0][to_control]).sum() + (y[t == 1][to_treated] - y[t == 0]).sum()) / len(y)
print(f"\nnaive {y[t == 1].mean() - y[t == 0].mean():.3f}  ipw {ipw:.3f}  aipw {aipw:.3f}  matching {matching:.3f}  truth 2.000")
```

**Reading the output.** The fitted propensities range from 0.013 to 1.000 and the largest weight is 36.2. The effective sample size is 491 of 917 treated users and 581 of 1,083 untreated: weighting has cost about half of the data's worth of information on each side. In exchange, balance is restored: the SMD for `x1` falls from +0.742 to +0.022, for `x2` from +0.464 to +0.064 and for the squared covariate from +0.659 to +0.048, all below the 0.1 line.

The point estimates on this one sample are IPW 2.370, doubly robust 1.957 and matching 2.011, against the naive 5.858. The IPW estimate is the farthest from 2.0, and a single sample cannot tell you whether that is bias or noise. The next block answers it.

**Line by line.**

- `C=1e6` makes the logistic regression effectively unpenalised, so the propensity model is not shrunk toward 0.5.
- `ess = w.sum() ** 2 / (w ** 2).sum()` is the effective sample size from the lab: it equals the sample size when all weights are equal and falls as they spread.
- The matching lines work on the log-odds of the score, not the raw probability, because distances between probabilities near 0 or 1 are compressed. They match in both directions, treated to untreated and back, to estimate the average over everyone.

### 3. Two hundred repeats, two strengths of confounding

One sample is an anecdote. This block repeats the whole experiment 200 times at each of two strengths of confounding and reports the mean estimate, its spread and its root mean squared error (RMSE). The "wrong" models leave out the squared covariate and the "right" ones include it.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

def make_world(n, rng, strength):
    x = rng.normal(size=(n, 3))
    logit = strength * (0.9 * x[:, 0] + 0.6 * x[:, 1] + 0.8 * (x[:, 2] ** 2 - 1))
    t = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    y0 = 3 + 2 * x[:, 0] + 1.5 * x[:, 1] + 2 * x[:, 2] ** 2 + rng.normal(size=n)
    return x, t, y0 + 2.0 * t

def run(x, t, y):
    wrong, right = x, np.column_stack([x, x[:, 2] ** 2])
    def ps_of(f):
        return np.clip(LogisticRegression(C=1e6, max_iter=2000).fit(f, t).predict_proba(f)[:, 1], 0.001, 0.999)
    def outcome(f):
        return (LinearRegression().fit(f[t == 1], y[t == 1]).predict(f),
                LinearRegression().fit(f[t == 0], y[t == 0]).predict(f))
    def aipw(e, m):
        return np.mean(m[0] - m[1] + t * (y - m[0]) / e - (1 - t) * (y - m[1]) / (1 - e))
    def ipw(e):
        return np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e))
    def match(e):
        score = np.log(e / (1 - e))[:, None]
        to_control = NearestNeighbors(n_neighbors=1).fit(score[t == 0]).kneighbors(score[t == 1])[1][:, 0]
        to_treated = NearestNeighbors(n_neighbors=1).fit(score[t == 1]).kneighbors(score[t == 0])[1][:, 0]
        return ((y[t == 1] - y[t == 0][to_control]).sum() + (y[t == 1][to_treated] - y[t == 0]).sum()) / len(y)
    e_wrong, e_right = ps_of(wrong), ps_of(right)
    m_wrong, m_right = outcome(wrong), outcome(right)
    return {
        "naive difference": y[t == 1].mean() - y[t == 0].mean(),
        "regression, wrong form": LinearRegression().fit(np.column_stack([t, wrong]), y).coef_[0],
        "regression, right form": LinearRegression().fit(np.column_stack([t, right]), y).coef_[0],
        "weighting, wrong propensity": ipw(e_wrong),
        "weighting, right propensity": ipw(e_right),
        "matching, right propensity": match(e_right),
        "weighting, right ps, trimmed": ipw(np.clip(e_right, 0.05, 0.95)),
        "doubly robust, wrong outcome": aipw(e_right, m_wrong),
        "doubly robust, wrong propensity": aipw(e_wrong, m_right),
        "doubly robust, both wrong": aipw(e_wrong, m_wrong),
    }

rng = np.random.default_rng(3)
for strength in (1.0, 2.0):
    runs = [run(*make_world(2000, rng, strength)) for _ in range(200)]
    print(f"\nconfounding strength {strength}: 200 repeats, n = 2000, true effect 2.000")
    print("estimator                          mean    sd    rmse")
    for name in runs[0]:
        v = np.array([r[name] for r in runs])
        print(f"{name:32s} {v.mean():6.3f} {v.std():6.3f} {np.sqrt(((v - 2) ** 2).mean()):6.3f}")
```

**Reading the output.** At strength 1.0 the naive difference averages 5.736. Regression with the wrong form gets 4.103, and weighting with the wrong propensity 4.215, so the right idea with the wrong model does not rescue anything. Regression with the right form is excellent, 2.004 with a spread of 0.053.

Weighting with the right propensity is unbiased (1.978) but its spread is 0.913, seventeen times the regression's. This is the honest surprise of the chapter: a correct, textbook estimator that is nearly useless on a single sample of 2,000 because a few huge weights dominate it. Trimming to 0.05 to 0.95 cuts the spread to 0.213 but moves the mean to 2.461, a bias of 0.46: the RMSE falls from 0.913 to 0.508, so trimming helped on balance but changed what was estimated.

The doubly robust rows show the safety net. With a wrong outcome model and a right propensity model it gets 1.990 with a spread of 0.595, better than weighting alone. With a right outcome model and a wrong propensity model it gets 2.003 and 0.053, as good as the right regression. With both wrong it gets 4.167, as bad as anything. Matching has a spread of only 0.099 but a mean of 2.111.

At strength 2.0, treatment is much more predictable. Weighting's spread rises to 1.147, trimming's bias to 1.24 and matching's mean to 2.327. Regression with the right form and doubly robust with a wrong propensity model are unaffected. The harder the confounding, the more the overlap and the weights matter.

**Line by line.**

- `np.clip(..., 0.001, 0.999)` only protects against dividing by zero. Without it, strength 2.0 produced a propensity of exactly 1.0 and the weights became `nan`, a literal positivity failure.
- `aipw(e_right, m_wrong)` and `aipw(e_wrong, m_right)` pass in one good and one bad model so that each row isolates what happens when only that model is wrong.
- Fitting `LinearRegression` separately on treated and untreated users gives each arm its own outcome model, which the augmented estimator needs.

## Try it yourself

The lab is the worked example with the formulas from this chapter, evaluated exactly rather than by simulation. Its defaults reproduce the 14.0 and 12.0 weighted means, the effect of 2.0 and the effective sample size of 320. The formulas were checked against simulations of six million users in five configurations; effects agreed to 0.002 and effective sample sizes to within 0.2.

<IpwLab />

**What each control does.**

- **Share of users who are engaged** and **Spend gained from engagement** set the size of the head start, as in the previous chapter.
- **Notified, among engaged** and **Notified, among others** set the propensities. The further apart, the more confounded the data and the more uneven the weights.
- **Trim propensities to** clips every propensity into the range from the chosen value to one minus it, then reweights.
- Click **show data** to see the counts, both weighted means, the effective sample sizes and the largest weight.

**Try it yourself.**

1. Leave the defaults. The weighted effect is 2.000 and the effective sample is 320. Now set the two notification rates to 0.95 and 0.05. The weighted effect is still 2.000 but the effective sample size falls to 95 and the largest weight rises to 20. Why: the weights are 1/0.95 and 1/0.05, and the rare cells (5 per cent of the engaged who were not notified) carry the whole comparison.
2. With those rates, set the trim to 0.10. The largest weight halves to 10, the effective sample size more than doubles to 196, and the weighted effect jumps to 3.429. Why: trimming lowers variance, but the rare groups now count for less than they should, so engagement leaks back into the comparison.
3. Set both rates back to 0.8 and 0.2 and set the trim to 0.25. The weights shrink to 1.33 and 4, the effect becomes 2.571 and the effective sample size 377. Why: the stricter the trim, the nearer the answer drifts to the naive 4.4.

## Designing with it

A workable procedure, in the order to do it:

1. **Randomise if you can**, check balance with SMDs, and report the difference in means with its interval. Everything else in this chapter is a second best.
2. **If you cannot, write down why units got treated** and measure those reasons before treatment. A variable measured after treatment may be a mediator or a collider.
3. **Check overlap first**: fit a propensity model, look at its histogram by arm, count units near 0 and 1. If overlap is poor, no estimator will rescue you, and the honest answer is to restrict the population.
4. **Report two estimators that fail differently**, such as regression and doubly robust, and investigate any gap between them. Agreement is mild evidence; disagreement is a signal to look at the models.
5. **Check balance after weighting or matching**, not before. If any SMD is above 0.1, improve the model, not the story.
6. **State the assumption** that no unmeasured variable drives both treatment and outcome. No data can test it, so say what you would need to see to believe it, and try a sensitivity analysis.

## Where this stands in 2026

Doubly robust estimators have become the default for observational effect estimation, because they pair well with flexible machine-learning models for the two nuisance functions, which the fourth chapter of this series develops. Hernán and Robins' 2026 edition has a section on doubly robust machine-learning estimators in its chapter on variable selection and high-dimensional data for exactly this reason. Practical guidance has also moved from "match until the p-value is non-significant" to reporting balance and overlap diagnostics and a sensitivity analysis. The weak point has not changed: the key assumption is untestable.

## Common mistakes

1. **Checking balance with a p-value.** A test for a difference between arms depends on the sample size, so a huge sample flags harmless gaps and a small one hides large ones. Use standardised differences.
2. **Using the propensity model to predict well.** The goal is balance, not accuracy. A model with perfect prediction is a sign of an overlap failure, because the treated and untreated are perfectly separable.
3. **Leaving extreme weights untouched.** Block 3's correct weighting estimator had a spread of 0.913. Look at the weight histogram, the effective sample size and the largest weight before you trust the estimate.
4. **Matching without a caliper.** Pairing a treated unit with the nearest untreated one, however far away, adds hidden bias. Set a maximum distance and report how many units were dropped.
5. **Believing a doubly robust estimator is a guarantee.** It forgives one wrong model, as block 3 shows, and fails if both are wrong. It does nothing about an unmeasured confounder.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Treatment is assigned by a coin and a covariate has an SMD of 0.3 between arms. What should you check?</summary>

A gap that large is unlikely by chance in a sample of any size. Check the assignment: a bug, a rule that sent a sub-population to one arm, or covariates measured after treatment. Check whether the sample is tiny too, as small samples can show a large SMD by chance.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> A treated user had a propensity of 0.1. What is their weight, and what does it mean?</summary>

1 / 0.1 = 10. Such users were rare among the treated, so each one stands in for ten users like them. A weight this large means the estimate leans heavily on one person.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Compute the effective sample size for 500 treated users in the worked example, where 400 have weight 1.25 and 100 have weight 5.</summary>

Sum of weights: 400 x 1.25 + 100 x 5 = 1,000. Sum of squared weights: 400 x 1.5625 + 100 x 25 = 625 + 2,500 = 3,125. Effective sample size: 1,000 squared divided by 3,125 = 320. The 500 users are worth 320 equal-weight users, mostly because the 100 weighted by 5 dominate.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Block 3: why does the doubly robust estimator with a wrong propensity model and a right outcome model (2.003, spread 0.053) beat weighting with the right propensity (1.978, spread 0.913)?</summary>

The outcome model explains almost all of the variation in spend, so what remains for the weighted correction to fix is tiny, and large weights multiply tiny residuals. Weighting alone multiplies the full outcome by the weights, which makes it noisy. The augmented estimator keeps the regression's low variance and uses the weights only as insurance.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> At confounding strength 2.0, trimming to 0.05 to 0.95 changed the mean from 2.453 to 3.240 and the spread from 1.147 to 0.237. Which is better, and what question does each answer?</summary>

The RMSEs are 1.233 and 1.263, so neither is better in accuracy. The untrimmed estimate targets the effect for the whole population and is noisy because it relies on rare users. The trimmed estimate targets the effect for the population that has a real chance of either treatment, which is a different question, and is steadier but biased for the whole population. Choose by the decision: for a policy that will reach everyone, you want the first, and for a decision about users who could plausibly go either way, the second.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> Show that if the true propensity model has e = 0.5 for everyone, IPW reduces to the difference in means.</summary>

Every weight is 1/0.5 = 2. The treated mean estimate is the sum of 2 x Y over treated users divided by n, and when exactly half the users are treated that is the plain mean of the treated arm. The untreated arm works the same way. Weighting changes nothing when assignment was already a fair coin, which is why a randomised experiment needs no propensity model.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Hernán MA and Robins JM, *Causal Inference: What If*, edition dated 19 August 2026, contents list and chapter titles read. Chapter 2 (randomised experiments, conditional randomisation, standardisation, inverse probability weighting), chapter 3 (identifiability conditions), chapters 12 and 13 (IP weighting and standardisation with models), chapter 15 (outcome regression and propensity scores, matching) and chapter 18 (doubly robust machine-learning estimators). Free to read at the author's page.
- Facure Alves M, *Causal Inference for the Brave and True*: the chapters on randomised experiments, matching, propensity score and doubly robust estimation use the same ideas with Python code. Contents list read.
- Rosenbaum PR and Rubin DB (1983), "The central role of the propensity score in observational studies for causal effects", Biometrika. The balancing property of the score is stated here as described in What If chapter 15; the original paper was not opened.
- [scikit-learn user guide, logistic regression and nearest neighbours](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression): the estimators used in the blocks. Version 1.9.1 was run.

## Check yourself

- I can explain why a coin flip removes confounding and how to check that it did, with a standardised mean difference.
- I can compute inverse-probability weights and a weighted mean by hand for two groups.
- I can say what the effective sample size measures and why correct weighting can be too noisy to use.
- I can name what each of regression, weighting, matching and doubly robust estimation needs to be right, and read that off block 3.
- I can explain what trimming trades and how overlap shows up in the weights.

## Where to go next

Next chapter: [quasi-experiments](/docs/theory/causal/quasi-experiments), for the common case where you suspect an unmeasured confounder and cannot randomise, so you look for a natural experiment instead. A related chapter: [model evaluation](/docs/theory/ml/model-evaluation), because the cross-fitting idea in the fourth chapter reuses its train and test discipline.
