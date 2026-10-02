---
id: ml-classification
title: "Classification and Logistic Regression"
sidebar_label: "Classification and logistic regression"
sidebar_position: 2
slug: /theory/ml/classification-and-logistic-regression
description: "Turn a linear score into a probability with the sigmoid, cut it at a threshold, and judge the result with precision, recall, F1 and the ROC curve instead of accuracy."
tags: [classification, logistic-regression, sigmoid, precision-recall, roc-auc, threshold]
---

import Infographic from '@site/src/components/Infographic';
import LogisticBoundaryLab from '@site/src/components/viz/LogisticBoundaryLab';

**In one line.** A classifier turns a score into a probability and then into a decision, and it must be judged by the mistakes it makes on the class you care about, not by accuracy.

## The idea in plain words

Regression answers "how much?". Classification answers "which one?": spam or not, fraud or not, benign or malignant. For two classes the useful model does not shout an answer, it reports a **probability** that the input belongs to the positive class, and you decide what to do with it.

Logistic regression does this in two moves, both on the previous chapter's machinery.

1. **A linear score.** Compute $z = \theta^\top x$, exactly the weighted sum from regression. It can be any real number, from very negative to very positive.
2. **A squashing function.** Push the score through the *sigmoid* $\sigma(z) = 1/(1+e^{-z})$. Large negative scores become probabilities near 0, large positive ones near 1, and a score of zero becomes exactly 0.5.

Then a **threshold** turns the probability into a class. The usual cut is 0.5, which means "score above zero", and the set of points where the score is exactly zero is the **decision boundary**: a straight line in two dimensions, a flat plane in three, a hyperplane beyond that.

Think of a thermostat's dial. The temperature reading is the score, the sigmoid is the dial that turns it into "how confident am I that it is hot?", and the threshold is where you decide to switch the fan on. Moving the threshold does not change what the thermostat *knows*; it changes how much evidence you demand before acting. That separation, between a model that ranks and a decision rule you set to suit the costs, is the idea the second half of the chapter builds on.

The second half is about judging honestly. If 95 of every 100 emails are genuine, a "model" that never flags anything is 95% accurate and catches no spam at all. So we count four outcomes (caught, missed, false alarm, correctly ignored) and read precision, recall and F1 from them, and we sweep the threshold to draw the **ROC curve**, which scores the ranking itself rather than one cut.

<Infographic src="/img/ml/classification-and-logistic-regression-score-to-class.svg" alt="A pipeline from features to score to sigmoid to probability to class, with worked values and a fitted boundary on 80 points" caption="From score to class. The weights and the 34, 7, 6, 33 counts are printed by block 2 below." />

<Infographic src="/img/ml/classification-and-logistic-regression-accuracy-lies.svg" alt="A two by two table of outcomes with the lecture metrics, beside a table showing that accuracy hides missed positives" caption="Accuracy lies on imbalanced data. The right-hand table is printed by block 3 below." />

## How it works

### What a classifier must do

Assign each input to a discrete class; for binary problems, output P(y=1|x) and threshold it.

### Sigmoid, boundary & decision rule

p̂ = σ(z) = 1/(1+e⁻ᶻ), z = θᵀx. Decision rule: class 1 if p̂ ≥ 0.5 (z ≥ 0). The set z=0 is the boundary.

#### Decision rule

Set the score z and the threshold; see the probability and the predicted class. The lab in the code section below has exactly this control, with the boundary on real data beside it.

:::tip

**Worked.** z=1 → p̂ = σ(1) = **0.731** ≥ 0.5 → class 1. z=−0.5 → 0.378 → class 0. Trained with cross-entropy.

:::

### Beyond accuracy

If a model predicts everything to be the majority class, accuracy can look high while it catches none of the rare class.

:::tip

**Worked.** TP=40, FP=10, FN=5, TN=45 → precision **0.80**, recall **0.889**, F1 **0.842**, accuracy **0.85**. On a 95/5 imbalance, "predict majority" scores 95% accuracy but 0 recall.

:::

### The ROC curve

Sweep the threshold and plot true-positive rate vs false-positive rate. Hug the top-left = great; the diagonal = random.

:::tip

**AUC.** Area under the ROC: 1.0 perfect, 0.5 random. Threshold-independent, so it compares models fairly on imbalanced data.

:::

:::note

**Generative alternative.** A Gaussian mixture models each class as a blob and classifies by which most likely generated the point, the bridge to unsupervised clustering.

:::

### Key takeaways

- **1 · Logistic**: p̂ = σ(θᵀx); threshold at 0.5; hyperplane boundary.
- **2 · Metrics**: Precision, recall, F1; accuracy misleads on imbalance.
- **3 · ROC/AUC**: Threshold-free quality score.

:::note

**The thread.** Classification adds a squashing activation and a threshold to the regression neuron, then insists you evaluate honestly (precision, recall and ROC/AUC) because a single accuracy number hides failure on the class you care about.

:::

## A real system that works this way

**scikit-learn's `LogisticRegression`** is the reference implementation most people meet first. Two details from its user guide matter in practice. It applies an L2 penalty **by default** (the parameter `C` is the inverse of the penalty strength, default 1), so its coefficients are shrunk compared with the textbook formula, which is why the blocks below set `C=1e6` when they want the unpenalised fit. And its `predict` uses the fixed cut of 0.5 on the probability. The same documentation has a page on *tuning the decision threshold* with `TunedThresholdClassifierCV` and `FixedThresholdClassifier`, and motivates it with a medical screening example: a clinician may accept a far lower probability, the page suggests something like 0.02, before flagging a possible tumour, because a missed case costs more than a follow-up test.

**Spam, fraud and screening systems** all share the pattern without needing a name. A model produces a calibrated-enough probability. A separate, business-owned rule decides the action: auto-block above one cut, send to a human reviewer between two cuts, let through below. Re-tuning the cuts as costs change needs no retraining at all.

## Code you can run

Five blocks. Blocks 1 and 3 use the lecture's own numbers and the usual imbalance trap. Blocks 2 and 4 use the lab's data, so they print the exact figures the lab shows by default.

### 1. The lecture's numbers

The sigmoid at the scores the lecture uses, the four-outcome arithmetic, and the "always predict the majority" baseline.

```python
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for z in (1.0, -0.5, 0.0):
    p = sigmoid(z)
    print(f"z = {z:4.1f}  p = {p:.3f}  class = {int(p >= 0.5)}")

tp, fp, fn, tn = 40, 10, 5, 45
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}")
print(f"precision {precision:.2f}   recall {recall:.3f}   F1 {f1:.3f}   accuracy {(tp + tn) / (tp + fp + fn + tn):.2f}")

y = np.array([1] * 50 + [0] * 950)
majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((1000, 1)), y)
pred = majority.predict(np.zeros((1000, 1)))
print(f"\n95/5 imbalance, always predict the majority: accuracy {accuracy_score(y, pred):.2f}, recall {recall_score(y, pred):.2f}")
```

What it prints:

```text
z =  1.0  p = 0.731  class = 1
z = -0.5  p = 0.378  class = 0
z =  0.0  p = 0.500  class = 1

TP=40 FP=10 FN=5 TN=45
precision 0.80   recall 0.889   F1 0.842   accuracy 0.85

95/5 imbalance, always predict the majority: accuracy 0.95, recall 0.00
```

Every lecture figure reproduces: $\sigma(1) = 0.731$ so class 1, $\sigma(-0.5) = 0.378$ so class 0, and precision 0.80, recall 0.889, F1 0.842, accuracy 0.85 from TP 40, FP 10, FN 5, TN 45. The last line is the trap: 95% accuracy with zero recall.

### 2. Logistic regression from scratch, then the ROC curve

The block generates the lab's 80 points with a small seeded generator (so that the browser lab and this code produce the same data), fits the weights by batch gradient descent on the cross-entropy using the update $\theta \leftarrow \theta - \alpha X^\top(\hat p - y)/m$, compares with scikit-learn, sweeps the threshold, and computes the AUC three ways.

```python
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]
    def draw():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return draw

def normal(draw):
    return math.sqrt(-2 * math.log(1 - draw())) * math.cos(2 * math.pi * draw())

draw = mulberry32(7)
rows, labels = [], []
for label, (cx, cy) in enumerate([(-1.0, -0.5), (1.0, 0.5)]):
    for _ in range(40):
        rows.append([cx + 1.1 * normal(draw), cy + 1.1 * normal(draw)])
        labels.append(label)
X, y = np.array(rows), np.array(labels)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

A = np.column_stack([np.ones(len(X)), X])
theta = np.zeros(3)
for _ in range(4000):
    p = sigmoid(A @ theta)
    theta -= 0.5 * A.T @ (p - y) / len(y)

p = sigmoid(A @ theta)
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print("theta (bias, w1, w2) =", np.round(theta, 3), f" cross-entropy = {loss:.4f}")

pred = p >= 0.5
tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
precision, recall = tp / (tp + fp), tp / (tp + fn)
print(f"TP={tp} FP={fp} FN={fn} TN={tn}  precision={precision:.3f} recall={recall:.3f} "
      f"F1={2 * precision * recall / (precision + recall):.3f} accuracy={(tp + tn) / len(y):.3f}")
print(f"AUC = {roc_auc_score(y, p):.4f}")

sk = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
print("sklearn  theta =", np.round(np.r_[sk.intercept_, sk.coef_[0]], 3))
print("max |probability difference| =", f"{np.max(np.abs(sk.predict_proba(X)[:, 1] - p)):.1e}")

thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
print("\nthreshold   TPR     FPR     precision")
for t in thresholds:
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    print(f"   {t:.1f}     {tp / np.sum(y == 1):.3f}   {fp / np.sum(y == 0):.3f}   {tp / max(tp + fp, 1):.3f}")

order = np.argsort(-p)
tpr = np.r_[0, np.cumsum(y[order] == 1) / np.sum(y == 1)]
fpr = np.r_[0, np.cumsum(y[order] == 0) / np.sum(y == 0)]
area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
pos, neg = p[y == 1], p[y == 0]
ranked = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
print(f"\nAUC from the curve {area:.4f}, from sklearn {roc_auc_score(y, p):.4f}, "
      f"from the chance a random positive outscores a random negative {ranked:.4f}")
```

What it prints:

```text
theta (bias, w1, w2) = [-0.133  1.66   0.866]  cross-entropy = 0.3547
TP=34 FP=7 FN=6 TN=33  precision=0.829 recall=0.850 F1=0.840 accuracy=0.838
AUC = 0.9200
sklearn  theta = [-0.133  1.66   0.866]
max |probability difference| = 3.3e-04

threshold   TPR     FPR     precision
   0.9     0.500   0.000   1.000
   0.7     0.725   0.100   0.879
   0.5     0.850   0.175   0.829
   0.3     0.950   0.325   0.745
   0.1     0.975   0.450   0.684

AUC from the curve 0.9200, from sklearn 0.9200, from the chance a random positive outscores a random negative 0.9200
```

<Infographic src="/img/ml/classification-and-logistic-regression-roc-and-threshold.svg" alt="An ROC curve with five threshold points marked and a table of true positive rate, false positive rate and precision at each" caption="The ROC curve for the fitted model. Each point and the AUC of 0.9200 come from block 2." />

The fitted weights are $(-0.133, 1.660, 0.866)$ and the from-scratch loop agrees with scikit-learn to about three decimal places in the probabilities. At the default cut the counts are TP 34, FP 7, FN 6, TN 33. The threshold table shows the trade-off in numbers: at 0.9 you are right every time you flag (precision 1.000) but catch only half (recall 0.500); at 0.1 you catch 97.5% but a third of your flags are wrong. The three AUC computations (area under the curve, scikit-learn, and the chance that a random positive outscores a random negative) all give 0.9200, which is the cleanest meaning of AUC.

### Try it: score, threshold and boundary

The lab holds this exact fitted model. Its defaults (threshold 0.50, score z = 1.0) reproduce block 2: TP 34, FP 7, FN 6, TN 33, AUC 0.9200, and for the lecture's decision-rule widget $\sigma(1.0) = 0.731$, class 1. Move the score to $-0.5$ and you get 0.378, class 0. Drag the threshold and watch the boundary shift, the ringed mistakes change, and the dot slide along a curve that itself never moves. The table view lists the five-row threshold sweep above.

<LogisticBoundaryLab />

### 3. Accuracy lies, and the threshold is yours to move

A 5%-positive problem, a plain logistic regression, the same model with a lower cut, and a class-weighted version.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=8, n_informative=4, weights=[0.95, 0.05],
                           class_sep=1.5, flip_y=0.01, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"positives in test set: {y_test.sum()} of {len(y_test)} ({y_test.mean():.1%})")
print(f"always predict 0: accuracy {1 - y_test.mean():.3f}, recall {recall_score(y_test, [0] * len(y_test)):.3f}\n")

def report(label, model, threshold):
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)
    print(f"{label:34s} thr={threshold:.2f}  accuracy={accuracy_score(y_test, pred):.3f}  "
          f"precision={precision_score(y_test, pred, zero_division=0):.3f}  recall={recall_score(y_test, pred):.3f}")
    return p

plain = LogisticRegression(max_iter=1000).fit(X_train, y_train)
weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
p = report("plain logistic regression", plain, 0.5)
report("plain, threshold moved down", plain, 0.2)
report("class_weight='balanced'", weighted, 0.5)
print(f"\nROC AUC {roc_auc_score(y_test, p):.3f}   average precision (PR AUC) {average_precision_score(y_test, p):.3f}"
      f"   (no-skill average precision = prevalence = {y_test.mean():.3f})")
```

What it prints:

```text
positives in test set: 96 of 1800 (5.3%)
always predict 0: accuracy 0.947, recall 0.000

plain logistic regression          thr=0.50  accuracy=0.958  precision=0.920  recall=0.240
plain, threshold moved down        thr=0.20  accuracy=0.947  precision=0.500  recall=0.448
class_weight='balanced'            thr=0.50  accuracy=0.738  precision=0.135  recall=0.719

ROC AUC 0.799   average precision (PR AUC) 0.467   (no-skill average precision = prevalence = 0.053)
```

Always predicting 0 is 94.7% accurate. The plain model at 0.5 scores 95.8%, barely better, because it is cautious: precision 0.920 but recall only 0.240. Lowering the cut to 0.2 *reduces* accuracy to 94.7% (identical to the do-nothing baseline) while nearly doubling recall to 0.448. `class_weight='balanced'` pushes recall to 0.719 at the cost of precision 0.135 and accuracy 0.738. None of these is "the right one"; each is a different point on the same trade-off, and accuracy alone would have picked the least useful.

### 4. The generative alternative

The lecture ends with a note: instead of drawing a boundary, model each class as a Gaussian blob and classify by which blob more probably produced the point. Here a one-component `GaussianMixture` per class is compared with logistic regression on the lab's data.

```python
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]
    def draw():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return draw

def normal(draw):
    return math.sqrt(-2 * math.log(1 - draw())) * math.cos(2 * math.pi * draw())

draw = mulberry32(7)
rows, labels = [], []
for label, (cx, cy) in enumerate([(-1.0, -0.5), (1.0, 0.5)]):
    for _ in range(40):
        rows.append([cx + 1.1 * normal(draw), cy + 1.1 * normal(draw)])
        labels.append(label)
X, y = np.array(rows), np.array(labels)

blobs = [GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(X[y == c]) for c in (0, 1)]
log_prior = np.log([np.mean(y == 0), np.mean(y == 1)])
scores = np.column_stack([b.score_samples(X) + lp for b, lp in zip(blobs, log_prior)])
generative = scores.argmax(axis=1)

discriminative = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
print("class means found by the blobs:", np.round([b.means_[0] for b in blobs], 2).tolist())
print(f"generative (one Gaussian per class) accuracy : {np.mean(generative == y):.3f}")
print(f"discriminative (logistic regression) accuracy: {discriminative.score(X, y):.3f}")
print(f"the two disagree on {np.sum(generative != discriminative.predict(X))} of {len(y)} points")
```

What it prints:

```text
class means found by the blobs: [[-0.88, -0.49], [1.09, 0.43]]
generative (one Gaussian per class) accuracy : 0.838
discriminative (logistic regression) accuracy: 0.838
the two disagree on 0 of 80 points
```

On this data the two approaches make identical predictions, because two equal-shaped Gaussian blobs produce a linear boundary, which is exactly what logistic regression draws. They diverge when the blobs have different spreads (the generative boundary then curves) or when the Gaussian assumption is wrong, where the discriminative model is usually the safer choice.

### 5. More than two classes

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)

proba = model.predict_proba(X_test[:3])
print("three test flowers, probability of each of the 3 classes:")
print(np.round(proba, 3))
print("each row sums to", np.round(proba.sum(axis=1), 6))
print(f"test accuracy {model.score(X_test, y_test):.3f}")

z = model[-1].decision_function(model[0].transform(X_test[:3]))
softmax = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
print("softmax of the three scores matches predict_proba:", np.allclose(softmax, proba))
```

What it prints:

```text
three test flowers, probability of each of the 3 classes:
[[0.    0.036 0.964]
 [0.001 0.234 0.764]
 [0.971 0.029 0.   ]]
each row sums to [1. 1. 1.]
test accuracy 0.978
softmax of the three scores matches predict_proba: True
```

The softmax of the three class scores reproduces `predict_proba`, and each row sums to 1. This is the multi-class form of exactly the same idea.

## Designing with it

### Choosing the threshold from costs, not habit

The default 0.5 silently assumes a false alarm and a miss cost the same. Write the costs down and the best cut follows: with a cost $c_{FP}$ for a false alarm and $c_{FN}$ for a miss, flag the case when the probability exceeds $c_{FP}/(c_{FP}+c_{FN})$. If a miss costs nine times a false alarm, that cut is 0.1, not 0.5. This holds when the probabilities are well calibrated, which is worth checking (the evaluation chapter later returns to calibration).

| If a miss is very costly (disease, fraud) | If a false alarm is very costly (account lock, arrest) |
| --- | --- |
| Lower the threshold, accept more false alarms | Raise the threshold, accept more misses |
| Watch recall | Watch precision |
| Add a cheap second-stage check for the extra flags | Require human review before acting |

### Practical rules

- **Scale features** before fitting, exactly as in the regression chapter: gradient-based solvers converge faster and the penalty treats all weights fairly.
- **Report precision and recall at the chosen cut, plus a curve.** One number hides the trade-off. If the positive class is rare, prefer the precision-recall view to ROC (see the nuance below).
- **Stratify your splits** so the rare class appears in every fold.
- **Keep the model and the decision rule separate** in code, so a threshold change is a configuration change.

:::note Correction: the tie at exactly 0.5
The lecture's rule is "class 1 if $\hat p \ge 0.5$", which puts the boundary itself in class 1. scikit-learn's documentation instead predicts the positive class when the probability is *above* 0.5 (score above 0). I checked with a model whose weights are all zero: `predict` returns class 0 while `predict_proba` returns 0.5 for both classes. On real data an exact tie almost never happens, but if you hand-code the rule, decide the tie on purpose.
:::

:::note Nuance: AUC and imbalance
The lecture says AUC "compares models fairly on imbalanced data" because it ignores the threshold. That is true in the sense that it does not depend on the cut, but it can flatter a model when positives are rare: false positives are measured against the huge negative class, so thousands of them barely move the false-positive rate. Block 3 shows it. On a 5% positive set the ROC AUC is 0.799 while the average precision (the area under the precision-recall curve) is 0.467, against a no-skill value of 0.053. Report both.
:::

:::note Beyond the lecture: more than two classes
Binary logistic regression extends by replacing the sigmoid with the **softmax**, which turns one score per class into probabilities that sum to 1. Block 5 does this on the iris flowers. Behind `fit` it is still a linear score per class passed through a squashing function, trained with the same cross-entropy loss.
:::

:::note Beyond the lecture: why cross-entropy and not squared error?
The lecture says logistic regression is "trained with cross-entropy". The reason is the gradient. For cross-entropy the gradient with respect to the weights is simply $\frac{1}{m}X^\top(\hat p - y)$, the prediction error times the input, the very same shape as linear regression. Squared error on a sigmoid output has a gradient that vanishes when the model is confidently wrong, which is exactly when you need it to push hardest.
:::

## Where this stands in 2026

:::info Industry view

- Logistic regression remains the default interpretable classifier and the standard baseline. Each coefficient is the change in the log-odds per unit of a feature, which regulators and clinicians can read.
- The ROC curve and AUC are universal, but practitioners facing rare events now routinely report precision-recall curves alongside them, and set the operating threshold from a cost table rather than from 0.5.
- Threshold tuning is a first-class operation in the tooling: scikit-learn ships `TunedThresholdClassifierCV` and `FixedThresholdClassifier` so that the cut is fitted on validation data, not hard-coded.
- The sigmoid-plus-cross-entropy pair is also the last layer of most neural classifiers. What you learn here is the output layer of a deep network.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is the objective of a classification model, and how is binary classification framed?</summary>

To assign each input to a discrete class and generalise. Binary classification models P(y=1|x) and applies a threshold (default 0.5).<br /><em>Module 4 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write logistic regression, its decision rule and its boundary.</summary>

p̂ = σ(z) = 1/(1+e⁻ᶻ), z = θᵀx. Rule: class 1 if p̂ ≥ 0.5 (z ≥ 0), else class 0. The set z = 0 is the decision boundary, a hyperplane.<br /><em>Module 4 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> For z=1.0, compute p̂ and the predicted class (threshold 0.5).</summary>

p̂ = σ(1) = 1/(1+e⁻¹) = 0.731 ≥ 0.5 → predict class 1.<br /><em>Module 4 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why can accuracy be misleading, with a concrete example?</summary>

On imbalanced data a model that predicts the majority class scores high accuracy while catching none of the rare class, e.g. a 95/5 split gives 95% accuracy but 0 recall on the positive class.<br /><em>Module 4 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> From TP=40, FP=10, FN=5, TN=45, compute precision, recall and F1.</summary>

Precision = 40/50 = 0.80; recall = 40/45 = 0.889; F1 = 2(0.8)(0.889)/(0.8+0.889) = 0.842.<br /><em>Module 4 · numeric</em>

</details>

<details>
<summary><strong>Q6.</strong> What does the ROC curve plot, and what does AUC mean?</summary>

It plots true-positive rate vs false-positive rate as the threshold sweeps. AUC is the area under it (1.0 perfect, 0.5 random), a threshold-independent quality score.<br /><em>Module 4 · conceptual</em>

</details>

<details>
<summary><strong>Q7.</strong> How does a Gaussian mixture classify differently from logistic regression?</summary>

Logistic regression is discriminative (draws a boundary); a Gaussian mixture is generative: it models each class as a Gaussian blob and assigns the point to the class most likely to have generated it.<br /><em>Module 4 · conceptual</em>

</details>

## Further reading

- [scikit-learn user guide: Linear models, logistic regression](https://scikit-learn.org/stable/modules/linear_model.html) is the primary reference for the solver and penalty used here.
- [scikit-learn: Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html) explains why 0.5 is a default and not a law, and how to fit the cut.
- [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) defines ROC AUC, average precision and precision-recall curves.
- [Google Machine Learning Crash Course: Logistic regression, the sigmoid function](https://developers.google.com/machine-learning/crash-course/logistic-regression/sigmoid-function) builds the log-odds idea step by step.
- [Google Machine Learning Crash Course: Accuracy, precision and recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) explains why accuracy fails on imbalanced data.
- Built from the course lecture "ml-m4-classification" (Lecture Library series).

- **[An Introduction to Statistical Learning](https://www.statlearning.com/)** `book`
  James, Witten, Hastie & Tibshirani: The friendliest rigorous intro to ML, free PDF plus R/Python labs.
- **[Stanford CS229 (Machine Learning)](https://cs229.stanford.edu/)** `course`
  Andrew Ng, Stanford: The rigorous derivations behind SVMs, GLMs, EM and learning theory.
- **[StatQuest](https://statquest.org/video-index/)** `▶ video`
  Josh Starmer: Short, wonderfully clear videos that build intuition step by step.

## What you should now be able to do

- [ ] I can explain why the sigmoid output can be read as a probability, and why a score of zero is the decision boundary.
- [ ] I can compute $\sigma(1) = 0.731$ by hand and say which class it gives at a threshold of 0.5.
- [ ] I can build a confusion matrix and derive precision, recall, F1 and accuracy from it, and show why accuracy misleads on a 95/5 split.
- [ ] I can choose a threshold from the cost of a false alarm and the cost of a miss instead of defaulting to 0.5.
- [ ] I can say what the ROC curve and AUC measure, and when a precision-recall curve is the better view.
- [ ] I can contrast a discriminative classifier (logistic regression) with a generative one (a Gaussian mixture per class).
