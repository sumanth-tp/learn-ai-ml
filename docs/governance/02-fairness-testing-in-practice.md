---
id: gov-fairness
title: "Fairness Testing in Practice"
sidebar_label: "Fairness testing in practice"
sidebar_position: 2
slug: /governance/fairness-testing-in-practice
description: "Measure group fairness with hand-computed and fairlearn metrics on synthetic data, see why the definitions conflict, try pre-, in- and post-processing mitigations, and learn why small slices produce gaps that are not real."
tags: [fairness, demographic-parity, equalized-odds, calibration, fairlearn, bias-mitigation, slicing]
---

import Infographic from '@site/src/components/Infographic';
import FairnessThresholdLab from '@site/src/components/viz/FairnessThresholdLab';

**In one line.** Fairness testing compares a model's errors and decisions across groups; the common definitions cannot all hold at once when the groups differ in base rate, so you pick the one that matches the harm, measure it with an error bar, and document the choice.

:::note Not from a lecture
Written for this site from the sources under Further reading. The data are **synthetic** and made up for the chapter: they illustrate how the metrics behave, and say nothing about any real lender, court or employer.
:::

## The idea in plain words

A loan model is accurate overall and still treats two groups differently. Two questions follow. Different in what way? And does the difference matter? The first is measurement, the second is judgement, and this chapter is about keeping them separate.

Four group metrics cover most audits. Each is a number you compute per group and then compare.

| Metric | Asks | Fairlearn name |
| --- | --- | --- |
| Selection rate (demographic parity) | Is the share receiving the favourable decision the same in every group? | `demographic_parity_difference` |
| True positive rate (equal opportunity) | Among people who deserve the favourable outcome, are they found at the same rate? | part of `equalized_odds_difference` |
| False positive rate (with TPR: equalised odds) | Among people who do not, are they wrongly flagged at the same rate? | part of `equalized_odds_difference` |
| Precision (predictive parity) and calibration | Does a flag, or a score of 0.3, mean the same thing in every group? | `MetricFrame` with `precision_score` |

Fairlearn's documentation describes demographic parity as predictions being independent of group membership, equalised odds as equal true and false positive rates, and equal opportunity as the relaxation that looks only at true positive rates. It also warns that demographic parity throws information away, since it looks only at predictions and never at outcomes.

### Why they conflict

Three papers settle the argument that you cannot have it all. Chouldechova (2017) shows that predictive parity and equal error rates cannot both hold when prevalence differs across groups. Kleinberg, Mullainathan and Raghavan (2016) prove that, except in highly constrained special cases, no method satisfies calibration and balance for both classes at once. Hardt, Price and Srebro (2016) propose equalised odds and equal opportunity as criteria and show how to adjust a trained predictor to meet them. The arithmetic behind the conflict is an identity you can check: the false positive rate equals the base rate odds, times (1 minus precision) over precision, times the true positive rate. Hold precision and TPR equal across groups and a different base rate forces a different false positive rate. The first code block checks this on data.

<Infographic src="/img/gov/fairness-testing-in-practice-metrics.svg" alt="Tables of four group metrics and calibration for two groups from one model, with notes on which parity holds and the identity that links them." caption="One model, one threshold of 0.30: equal opportunity roughly holds, predictive parity and calibration do not. Numbers are printed by block 1." />

## How it works

A fairness audit is a procedure, not a number.

1. **Name the decision and the harm.** A false positive in fraud screening burdens an innocent customer; a false negative in a loan model denies someone credit they deserved. Which error is the harm decides which metric leads.
2. **Choose the groups, including intersections**, from the attributes you are lawfully able to use. Where the attribute is not in the data, you cannot measure the gap; the EU's amended AI Act contains a narrow permission to process special categories of personal data for bias detection and correction, under strict conditions (see [the regulation chapter](/docs/governance/regulation-and-model-documentation)).
3. **Compute every metric per group with an interval**, not one number.
4. **Check calibration within groups**, because a score is often used as a probability.
5. **Mitigate if the gap is real and harmful**, at one of three points: **pre-processing** (reweigh, repair or relabel the data; fairlearn's `CorrelationRemover` removes linear correlation between sensitive and other features), **in-processing** (train with a constraint, for example `ExponentiatedGradient` or `GridSearch`, or an adversarial classifier), or **post-processing** (choose group-specific thresholds, for example `ThresholdOptimizer`).
6. **Document** the metrics, the groups, the thresholds and the trade-off you accepted, so the decision is reviewable. A model card is the usual home (next chapters).

A caution from fairlearn's own documentation belongs here: using the demographic parity **ratio** with an 80% threshold, the "four-fifths rule", is described there as wrong in many scenarios, because it borrows a rule of thumb from employment law and applies it outside its context. Fairness metrics are tools for examining a sociotechnical context, not universal definitions.

## A real system that works this way

**Recidivism prediction instruments** are the case that put the conflict into print. Chouldechova's 2017 paper analyses such instruments and shows that disparate impact can arise when an instrument fails to satisfy error-rate balance, even when it satisfies predictive parity, if recidivism prevalence differs between groups. Hardt and colleagues use FICO credit scores as their empirical case study for equal opportunity in lending. I opened the abstracts of these papers for this chapter and did not re-derive their datasets, so the chapter uses its own synthetic data to show the same mechanism.

## Code you can run

Everything here is CPU only and seeded. The versions used: fairlearn 0.14.0 and scikit-learn 1.9.1. The data are synthetic: two groups, A (60%) and B (40%), with different base rates of the favourable outcome, a model that never sees the group label directly, and a "proxy" feature that is correlated with it, the way a postcode can be.

#### 1. Hand-computed metrics, checked against fairlearn

```python
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, true_positive_rate, false_positive_rate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
n = 20000
group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
is_b = (group == "B").astype(float)
skill = rng.normal(0, 1, n)
proxy = 0.8 * is_b + rng.normal(0, 0.6, n)
noise = rng.normal(0, 1, n)
logit = -1.0 + 1.1 * skill - 0.9 * is_b + 0.3 * noise
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([skill + 0.5 * rng.normal(0, 1, n), proxy, noise])

idx_train, idx_test = train_test_split(np.arange(n), test_size=0.5, random_state=0, stratify=y)
model = LogisticRegression().fit(X[idx_train], y[idx_train])
score = model.predict_proba(X[idx_test])[:, 1]
yt, gt = y[idx_test], group[idx_test]
pred = (score >= 0.3).astype(int)

print("group  n     base rate  selection rate  TPR     FPR     precision")
hand = {}
for g in ("A", "B"):
    m = gt == g
    sel = pred[m].mean()
    tpr = pred[m & (yt == 1)].mean()
    fpr = pred[m & (yt == 0)].mean()
    ppv = yt[m & (pred == 1)].mean()
    hand[g] = (sel, tpr, fpr, ppv)
    print(f"{g}      {m.sum():4d}  {yt[m].mean():.3f}      {sel:.3f}           {tpr:.3f}   {fpr:.3f}   {ppv:.3f}")

dp_gap = abs(hand["A"][0] - hand["B"][0])
eo_gap = max(abs(hand["A"][1] - hand["B"][1]), abs(hand["A"][2] - hand["B"][2]))
print(f"\ndemographic parity difference (hand)  {dp_gap:.4f}")
print(f"equalised odds difference (hand)      {eo_gap:.4f}")

frame = MetricFrame(metrics={"selection": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate,
                             "precision": precision_score},
                    y_true=yt, y_pred=pred, sensitive_features=gt)
print(f"demographic parity difference (fairlearn) {demographic_parity_difference(yt, pred, sensitive_features=gt):.4f}")
print(f"equalised odds difference (fairlearn)     {equalized_odds_difference(yt, pred, sensitive_features=gt):.4f}")
print("MetricFrame by group:")
print(frame.by_group.round(3).to_string())

print("\ncalibration within groups: mean score against observed rate, by score band")
bands = pd.cut(score, [0, 0.2, 0.4, 0.6, 1.0])
table = pd.DataFrame({"group": gt, "band": bands, "score": score, "y": yt})
print(table.groupby(["band", "group"], observed=True).agg(n=("y", "size"), mean_score=("score", "mean"), observed=("y", "mean")).round(3).to_string())

for g in ("A", "B"):
    p = yt[gt == g].mean()
    sel, tpr_g, fpr_g, ppv_g = hand[g]
    implied = p / (1 - p) * (1 - ppv_g) / ppv_g * tpr_g
    print(f"identity check, group {g}: FPR {fpr_g:.4f}, from base rate, precision and TPR {implied:.4f}")

p_a, p_b = yt[gt == "A"].mean(), yt[gt == "B"].mean()
tpr = 0.60
ppv = 0.70
fpr_a = p_a / (1 - p_a) * (1 - ppv) / ppv * tpr
fpr_b = p_b / (1 - p_b) * (1 - ppv) / ppv * tpr
print(f"\nequal precision {ppv} and equal TPR {tpr} with base rates {p_a:.3f} and {p_b:.3f} force FPR {fpr_a:.3f} against {fpr_b:.3f}")
```

```text
group  n     base rate  selection rate  TPR     FPR     precision
A      5956  0.312      0.352           0.579   0.248   0.515
B      4044  0.167      0.275           0.575   0.215   0.349

demographic parity difference (hand)  0.0765
equalised odds difference (hand)      0.0332
demographic parity difference (fairlearn) 0.0765
equalised odds difference (fairlearn)     0.0332
MetricFrame by group:
                     selection    tpr    fpr  precision
sensitive_feature_0                                    
A                        0.352  0.579  0.248      0.515
B                        0.275  0.575  0.215      0.349

calibration within groups: mean score against observed rate, by score band
                     n  mean_score  observed
band       group                            
(0.0, 0.2] A      2522       0.118     0.148
           B      2127       0.110     0.067
(0.2, 0.4] A      2197       0.286     0.343
           B      1338       0.287     0.206
(0.4, 0.6] A       935       0.484     0.540
           B       464       0.479     0.401
(0.6, 1.0] A       302       0.689     0.758
           B       115       0.679     0.617
identity check, group A: FPR 0.2484, from base rate, precision and TPR 0.2484
identity check, group B: FPR 0.2152, from base rate, precision and TPR 0.2152

equal precision 0.7 and equal TPR 0.6 with base rates 0.312 and 0.167 force FPR 0.117 against 0.052
```

The hand-computed gaps match fairlearn exactly, so the library is doing nothing mysterious: it is a `MetricFrame` that slices the same four numbers by group. Group A has a base rate of 0.312 and B of 0.167. At one threshold of 0.30 the selection rates differ (0.352 against 0.275, a demographic parity gap of 0.0765), the true positive rates are almost equal (0.579 against 0.575), the false positive rates are 0.248 against 0.215 (so the equalised odds gap is 0.0332), and precision differs sharply (0.515 against 0.349).

The calibration table is the surprise. The model never sees the group label, yet at the same score of about 0.29, group A's observed rate is 0.343 and group B's is 0.206: the model is **too high for B**. Leaving the sensitive attribute out ("fairness through unawareness") did not make the model equally calibrated, because the proxy feature carries group information into the score. The identity check at the end shows each group's false positive rate recovered from its base rate, precision and TPR (0.2484 and 0.2152), and what equal precision and equal TPR would force (0.117 against 0.052).

The lab holds the same scores. Its defaults are both thresholds at 0.30, giving the numbers above (the lab prints the demographic parity gap as 0.077 where the code prints 0.0765). Use "match selection rate" to equalise decisions for B, then look at what happens to precision and the false positive rate.

<FairnessThresholdLab />

#### 2. Mitigations and what they cost

```python
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(7)
n = 40000
group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
is_b = (group == "B").astype(float)
skill = rng.normal(0, 1, n)
proxy = 0.8 * is_b + rng.normal(0, 0.6, n)
noise = rng.normal(0, 1, n)
logit = -1.0 + 1.1 * skill - 0.9 * is_b + 0.3 * noise
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
X = np.column_stack([skill + 0.5 * rng.normal(0, 1, n), proxy, noise])

train, rest = train_test_split(np.arange(n), test_size=0.5, random_state=0, stratify=y)
fit_post, test = train_test_split(rest, test_size=0.5, random_state=1, stratify=y[rest])
base = LogisticRegression().fit(X[train], y[train])
yt, gt = y[test], group[test]


def row(label, pred):
    sel = {g: pred[gt == g].mean() for g in ("A", "B")}
    ppv = {g: yt[(gt == g) & (pred == 1)].mean() for g in ("A", "B")}
    acc = (pred == yt).mean()
    dp = demographic_parity_difference(yt, pred, sensitive_features=gt)
    eo = equalized_odds_difference(yt, pred, sensitive_features=gt)
    print(f"{label:34s} {acc:.3f}   {sel['A']:.3f} {sel['B']:.3f}   {dp:.3f}   {eo:.3f}   {ppv['A']:.3f} {ppv['B']:.3f}")


print("method                             accuracy  sel A  sel B   DP gap  EO gap  prec A prec B")
score_test = base.predict_proba(X[test])[:, 1]
row("baseline, one threshold 0.50", (score_test >= 0.5).astype(int))
row("baseline, one threshold 0.30", (score_test >= 0.3).astype(int))

score_fit, g_fit = base.predict_proba(X[fit_post])[:, 1], group[fit_post]
target = (score_fit >= 0.3).mean()
thr = {g: np.quantile(score_fit[g_fit == g], 1 - target) for g in ("A", "B")}
by_hand = np.array([score_test[i] >= thr[gt[i]] for i in range(len(yt))]).astype(int)
row(f"per-group thresholds, equal selection", by_hand)
print(f"    thresholds A {thr['A']:.3f}, B {thr['B']:.3f}")

for constraint in ("demographic_parity", "equalized_odds"):
    post = ThresholdOptimizer(estimator=base, constraints=constraint, objective="accuracy_score",
                              prefit=True, predict_method="predict_proba")
    post.fit(X[fit_post], y[fit_post], sensitive_features=g_fit)
    row(f"ThresholdOptimizer, {constraint}", post.predict(X[test], sensitive_features=gt, random_state=0))

reduction = ExponentiatedGradient(LogisticRegression(), constraints=DemographicParity(), eps=0.01)
reduction.fit(X[train], y[train], sensitive_features=group[train])
row("ExponentiatedGradient, parity", reduction.predict(X[test], random_state=0))

aware = LogisticRegression().fit(np.column_stack([X[train], (group[train] == "B")]), y[train])
row("add the group as a feature, 0.30", (aware.predict_proba(np.column_stack([X[test], (gt == "B")]))[:, 1] >= 0.3).astype(int))
```

```text
method                             accuracy  sel A  sel B   DP gap  EO gap  prec A prec B
baseline, one threshold 0.50       0.764   0.117 0.076   0.041   0.050   0.655 0.463
baseline, one threshold 0.30       0.710   0.376 0.271   0.105   0.092   0.500 0.341
per-group thresholds, equal selection 0.701   0.340 0.330   0.010   0.038   0.516 0.313
    thresholds A 0.321, B 0.269
ThresholdOptimizer, demographic_parity 0.759   0.061 0.060   0.001   0.017   0.723 0.473
ThresholdOptimizer, equalized_odds 0.758   0.055 0.041   0.014   0.028   0.735 0.446
ExponentiatedGradient, parity      0.764   0.103 0.089   0.014   0.011   0.674 0.474
add the group as a feature, 0.30   0.719   0.459 0.156   0.302   0.324   0.470 0.428
```

This run uses 40,000 applicants, split into train, a post-processing fit set and a test set of 10,000 rows, so the numbers differ from block 1. Compare rows only with care: accuracy is meaningful between rows with a similar overall selection rate. Against the baseline at threshold 0.50 (accuracy 0.764, demographic parity gap 0.041), the in-processing `ExponentiatedGradient` with a demographic parity constraint keeps accuracy at 0.764 and cuts the gap to 0.014. `ThresholdOptimizer` closes the gap further (0.001 for parity) but selects fewer people overall, so accuracy is 0.759. Hand-built group thresholds that equalise selection at about the 0.30 baseline's volume reduce the gap from 0.105 to 0.010 and cost accuracy 0.710 to 0.701. Note that the equalised-odds optimiser leaves a gap of 0.028 on the test rows: finite-sample noise means a constraint met on the fit set is met only approximately on new data.

The last row is the instructive one. Giving the model the group label makes it better calibrated and sees the true base rates, and it widens the selection gap to 0.302. Whether that is acceptable depends on the setting, and in many jurisdictions on the law about using protected attributes in decisions. That question belongs to your legal and policy colleagues, not to the optimiser.

<Infographic src="/img/gov/fairness-testing-in-practice-mitigation.svg" alt="A table comparing seven mitigation settings by accuracy, selection rates and two fairness gaps, with a note on small-slice noise." caption="Mitigations from block 2 and the small-slice numbers from block 3." />

#### 3. When a gap is just noise

```python
import numpy as np

rng = np.random.default_rng(0)
true_tpr = 0.60
reps = 4000

print("two groups with the SAME true TPR of 0.60: how big a gap do small slices invent?")
print("positives per group   mean |gap|   share of audits with gap above 0.10")
for positives in (20, 50, 200, 1000, 5000):
    a = rng.binomial(positives, true_tpr, reps) / positives
    b = rng.binomial(positives, true_tpr, reps) / positives
    gap = np.abs(a - b)
    print(f"{positives:19d}   {gap.mean():.3f}        {(gap > 0.10).mean():.3f}")

print("\nslices multiply: attributes with 4, 4 and 5 values give this many intersectional slices")
slices = 4 * 4 * 5
print(f"  {slices} slices; at a 5% false-alarm rate each, expect {slices * 0.05:.1f} spurious flags with no real gap")

print("\nbootstrap interval for one audit: 30 of 50 positives caught in group A, 21 of 50 in group B")
caught_a = np.array([1] * 30 + [0] * 20)
caught_b = np.array([1] * 21 + [0] * 29)
boot = []
for _ in range(5000):
    boot.append(rng.choice(caught_a, 50).mean() - rng.choice(caught_b, 50).mean())
low, high = np.percentile(boot, [2.5, 97.5])
print(f"  observed gap {caught_a.mean() - caught_b.mean():.2f}, 95% bootstrap interval {low:.2f} to {high:.2f}")
print(f"  interval contains zero: {low < 0 < high}")
```

```text
two groups with the SAME true TPR of 0.60: how big a gap do small slices invent?
positives per group   mean |gap|   share of audits with gap above 0.10
                 20   0.123        0.456
                 50   0.079        0.283
                200   0.039        0.037
               1000   0.017        0.000
               5000   0.008        0.000

slices multiply: attributes with 4, 4 and 5 values give this many intersectional slices
  80 slices; at a 5% false-alarm rate each, expect 4.0 spurious flags with no real gap

bootstrap interval for one audit: 30 of 50 positives caught in group A, 21 of 50 in group B
  observed gap 0.18, 95% bootstrap interval -0.02 to 0.38
  interval contains zero: True
```

Here the two groups have exactly the same true TPR of 0.60 and the audit still "finds" gaps. With 20 positives per group the average absolute gap is 0.123 and 45.6% of audits show a gap above 0.10. With 200 positives it is 0.039 and 3.7%; with 5,000 it is 0.008 and none. Intersections make it worse: attributes with 4, 4 and 5 values give 80 slices, and at a 5% false-alarm rate each you should expect 4.0 spurious flags with no real gap anywhere. In the bootstrap example an observed gap of 0.18 (30 of 50 against 21 of 50) has an interval of -0.02 to 0.38 that contains zero. Report intervals, set a minimum slice size, and treat small slices as a prompt to collect more data rather than as findings.

## Production snippets (not run here)

The fairlearn calls above already run. The two that would need your own data pipeline are shown as shapes, **Not run in this environment**.

```python
from fairlearn.reductions import EqualizedOdds, GridSearch
from sklearn.linear_model import LogisticRegression

sweep = GridSearch(LogisticRegression(), constraints=EqualizedOdds(), grid_size=20)
sweep.fit(X_train, y_train, sensitive_features=group_train)
candidates = sweep.predictors_
```

`GridSearch` returns a family of predictors along the accuracy-versus-fairness frontier; you pick one with the people who own the decision, rather than letting a script choose.

## Designing with it

- **Pick the metric from the harm.** Screening that burdens innocent people leads with the false positive rate; access to a benefit leads with the true positive rate; a score used as a probability needs calibration within groups.
- **Expect to trade.** In the run above, parity was cheap in accuracy, while equalising every rate was impossible by identity. Show the trade-off table to the decision owner.
- **Do not rely on unawareness.** Dropping the attribute left a proxy in place and a miscalibrated score for group B.
- **Measure with intervals and a floor on slice size.** Small slices invent gaps.
- **Re-audit after drift.** Base rates move, and every metric moves with them.
- **Write it down.** The groups, metrics, thresholds, data and the trade-off accepted go into the model documentation.

See [responsible ML engineering](/docs/theory/seml/responsible-ml) for the wider engineering practice around this.

## Where this stands in 2026

:::info Industry view

- **Fairlearn 0.14.0 (June 2026)** remains the standard open-source toolkit for metrics and the three mitigation families, and its documentation is explicit that the metrics do not settle questions of fairness by themselves.
- **Regulation asks for bias examination and records, not for one metric.** The amended EU AI Act inserts an Article 4a that lets providers of high-risk systems exceptionally process special categories of personal data to the extent strictly necessary for bias detection and correction, subject to listed safeguards, and the same article says it creates no obligation to conduct such detection for other systems. High-risk obligations now apply from 2 December 2027 for the Annex III areas (employment, credit, education and others); see [the regulation chapter](/docs/governance/regulation-and-model-documentation).
- **Documentation is the common thread.** Model cards ask for evaluation broken down by group and intersection, which is exactly what `MetricFrame` produces.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> In block 1 the true positive rates are 0.579 and 0.575 yet precision is 0.515 and 0.349. Is the model fair?</summary>

It satisfies equal opportunity almost exactly and fails predictive parity. Which matters depends on the harm: for access to a benefit the TPR may lead, but a flag means much less in group B, which matters if flagged people are then investigated. The identity linking base rate, precision, TPR and FPR says both cannot be equal when base rates differ (0.312 against 0.167).

</details>

<details>
<summary><strong>Q2.</strong> The model never sees the group label. Why is its score too high for group B?</summary>

The proxy feature carries group information, and the model was trained to fit the pooled data. At a score near 0.29 the observed rate is 0.343 for A and 0.206 for B. Removing an attribute does not remove its influence while correlated features remain.

</details>

<details>
<summary><strong>Q3.</strong> Why did adding the group as a feature produce a demographic parity gap of 0.302?</summary>

The model now sees the true difference in base rates and selects B far less often (0.156 against 0.459). It is better calibrated and less parity-fair, which illustrates that the metrics measure different things.

</details>

<details>
<summary><strong>Q4.</strong> An audit slice has 20 positives per group and shows a TPR gap of 0.15. What do you do?</summary>

Compute an interval and check the slice size. With 20 positives per group, 45.6% of audits show a gap above 0.10 even when the true gap is zero. Collect more data or merge slices before acting.

</details>

<details>
<summary><strong>Q5.</strong> Where in a pipeline can you mitigate, and which one fits when you cannot retrain?</summary>

Before training (reweigh or repair data), during training (constrained learners such as `ExponentiatedGradient`), and after training (group thresholds with `ThresholdOptimizer`). When the model is a fixed, purchased scorer, post-processing is the only option, and it needs the group label at decision time.

</details>

<details>
<summary><strong>Q6.</strong> Why is the "four-fifths rule" a poor default for ML fairness audits?</summary>

Fairlearn's documentation describes applying an 80% threshold to the demographic parity ratio as wrong in many scenarios, because it transplants a rule of thumb from employment law to a different context. Choose thresholds from the harm and the law that applies.

</details>

## Further reading

All opened for this chapter in October 2026.

- Hardt, Price and Srebro, [Equality of opportunity in supervised learning](https://arxiv.org/abs/1610.02413), 2016.
- Chouldechova, [Fair prediction with disparate impact: a study of bias in recidivism prediction instruments](https://arxiv.org/abs/1703.00056), 2017.
- Kleinberg, Mullainathan and Raghavan, [Inherent trade-offs in the fair determination of risk scores](https://arxiv.org/abs/1609.05807), 2016.
- Fairlearn, [common fairness metrics](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html) and [mitigation](https://fairlearn.org/main/user_guide/mitigation/index.html).
- Mitchell and colleagues, [Model cards for model reporting](https://arxiv.org/abs/1810.03993), 2019.

## Check yourself

- I can compute selection rate, TPR, FPR and precision per group by hand and check them against fairlearn.
- I can explain why equal precision and equal error rates cannot both hold when base rates differ.
- I can check calibration within groups and explain why leaving out the sensitive attribute is not enough.
- I can name a pre-, an in- and a post-processing mitigation and say what each costs.
- I can say how large a slice must be before I trust a gap, and report an interval instead of a point.
- I can write down which metric I chose, for which harm, and what I traded away.
