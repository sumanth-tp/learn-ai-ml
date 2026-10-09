---
id: interviews-senior-ai-engineer-additions
title: "Senior AI Engineer: 40 Practical Interview Questions"
sidebar_label: "25 · Senior AI engineer additions"
sidebar_position: 25
slug: /interviews/senior-ai-engineer-additions
description: "Forty worked interview questions for senior AI engineers across classical ML judgement, retrieval and RAG, LLM adaptation and serving, evaluation and governance, system design and engineering craft, each answered with a computed example and a pointer to the chapter that teaches it."
tags: [interview, senior-engineer, ml-judgement, rag, llm-serving, evaluation, governance, system-design, estimation]
---

# Senior AI Engineer: 40 Practical Interview Questions

Forty questions that test judgement rather than recall, each answered with a small experiment whose printed numbers are quoted in the answer.

:::note An addition to the interview bank
This page extends the interview section with questions drawn from the senior chapters on this site. It is not built from candidate reports. Every number below is printed by the code shown with it, run on 8 October 2026 with Python 3.14.6, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1 and fairlearn 0.14.0. Data are synthetic and seeded, and costs and prices are placeholders to replace with your own. Each answer ends with the chapter that teaches the idea in depth.
:::

## How to use this page

Answer aloud first, then open the block. A senior answer names the failure mode, says what to measure, and gives a number or a range rather than an adjective. The six groups are classical ML judgement, retrieval and RAG, LLM adaptation and serving, evaluation and governance, system design, and senior craft.


## 1. Classical ML judgement

Seven questions on metrics, leakage, thresholds, calibration, model choice, explanation and validation.

<details>
<summary><strong>Q1.</strong> A fraud model reports 99% accuracy. Is that good?</summary>

Not yet. Accuracy counts every transaction equally, and 99 of every 100 are legitimate, so a model that never raises an alarm already scores 0.99 while finding no fraud. Ask instead how many frauds sit at the top of the ranking, at the number of alerts a review team can work. The block builds 100,000 synthetic transactions with 1% fraud and measures the ranking.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=100000, n_features=20, n_informative=6, weights=[0.99, 0.01], flip_y=0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
p = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
alerts = np.argsort(-p)[: int(0.01 * len(yte))]
print("positives in the test set:", int(yte.sum()), "of", len(yte))
print("accuracy of always predicting negative:", 1 - yte.mean())
print("model accuracy at threshold 0.5:", round(accuracy_score(yte, p > 0.5), 4))
print("ROC AUC:", round(roc_auc_score(yte, p), 3), " average precision:", round(average_precision_score(yte, p), 3))
print("top 1 percent of scores: precision", round(yte[alerts].mean(), 3), " recall", round(yte[alerts].sum() / yte.sum(), 3))
```

```text
positives in the test set: 300 of 30000
accuracy of always predicting negative: 0.99
model accuracy at threshold 0.5: 0.9901
ROC AUC: 0.852  average precision: 0.213
top 1 percent of scores: precision 0.293  recall 0.293
```

Accuracy of 0.9901 is barely above the 0.99 floor set by predicting negative for everything. ROC AUC of 0.852 sounds respectable, but average precision of 0.213 shows how hard the task is at a 1% base rate. Working the 300 highest-scored cases, which is 1% of the test set, catches 29.3% of the fraud at a precision of 0.293. Precision and recall coincide here because the number of alerts equals the number of positives. Report precision and recall at the alert budget, plus the base rate, and treat accuracy and ROC AUC as context only.

**Chapters that teach this:** [Model evaluation](/docs/theory/ml/model-evaluation); [Features, leakage and imbalance](/docs/theory/ml/features-leakage-and-imbalance).

</details>

<details>
<summary><strong>Q2.</strong> A target-encoded customer ID lifts held-out AUC to 0.84. Do you ship it?</summary>

No. Target encoding replaces each category with the mean label of its rows. If that mean is computed on all rows before splitting, each held-out row's own label has leaked into its feature. The block uses 1,500 customer IDs and coin-flip labels, so the true AUC is 0.5 and any higher figure is leakage.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
n, levels = 4000, 1500
cat = rng.integers(0, levels, n)
y = rng.integers(0, 2, n)
idx = rng.permutation(n)
train, test = idx[:3000], idx[3000:]


def target_mean(cat_rows, y_rows):
    total = np.bincount(cat_rows, weights=y_rows, minlength=levels)
    count = np.bincount(cat_rows, minlength=levels)
    return np.where(count > 0, total / np.maximum(count, 1), y_rows.mean())


leaky = target_mean(cat, y)[cat]
honest = target_mean(cat[train], y[train])[cat[test]]
print("labels are coin flips, so the true AUC is 0.5")
print("encoding fitted on all rows, scored on the held-out rows:", round(roc_auc_score(y[test], leaky[test]), 3))
print("encoding fitted on training rows only:", round(roc_auc_score(y[test], honest), 3))
```

```text
labels are coin flips, so the true AUC is 0.5
encoding fitted on all rows, scored on the held-out rows: 0.841
encoding fitted on training rows only: 0.528
```

Fitting the encoding on all rows scores 0.841 on held-out rows from labels that carry no signal. Fitting it on the training rows only scores 0.528, which is the honest answer. The fix is structural: put the encoder inside the pipeline so it is refitted within each training fold, and treat a suspiciously strong feature as a bug until proven otherwise. High-cardinality IDs are the usual victims because most levels have only a few rows.

**Chapter that teach this:** [Features, leakage and imbalance](/docs/theory/ml/features-leakage-and-imbalance).

</details>

<details>
<summary><strong>Q3.</strong> How do you choose the decision threshold for a classifier?</summary>

From costs, not from 0.5. If a false positive costs 1 unit and a missed positive costs 10, flagging a case is worth it whenever its probability of being positive exceeds 1 / (1 + 10) = 0.0909, provided the probabilities are calibrated. The block scans thresholds on a held-out half of a synthetic problem with 10% positives.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=60000, n_features=15, n_informative=5, weights=[0.9, 0.1], random_state=1)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
p = LogisticRegression(max_iter=2000).fit(Xa, ya).predict_proba(Xb)[:, 1]
cost_fp, cost_fn = 1.0, 10.0


def cost(threshold):
    flagged = p >= threshold
    return (cost_fp * (flagged & (yb == 0)).sum() + cost_fn * (~flagged & (yb == 1)).sum()) / len(yb)


grid = np.linspace(0.01, 0.99, 99)
best = grid[int(np.argmin([cost(t) for t in grid]))]
theory = cost_fp / (cost_fp + cost_fn)
print("threshold from the cost ratio:", round(theory, 4))
print("best threshold on the grid:", round(best, 2))
print("cost per case at 0.5:", round(cost(0.5), 4), " at the theory threshold:", round(cost(theory), 4), " at the grid optimum:", round(cost(best), 4))
```

```text
threshold from the cost ratio: 0.0909
best threshold on the grid: 0.11
cost per case at 0.5: 0.6479  at the theory threshold: 0.3945  at the grid optimum: 0.3878
```

The cost ratio gives 0.0909 and the empirical grid optimum is 0.11, close because the model's probabilities are reasonably calibrated. Cost per case is 0.6479 at the default 0.5 and 0.3878 at the grid optimum, a 40% reduction. The theory threshold reaches 0.3945, within two percent of the best. If the probabilities were miscalibrated the two answers would drift apart, which is why calibration comes before threshold choice.

**Chapter that teach this:** [Model evaluation](/docs/theory/ml/model-evaluation).

</details>

<details>
<summary><strong>Q4.</strong> A random forest's probabilities feed a pricing rule. Do you trust them?</summary>

Check calibration first. A forest averages votes, so its scores rank well but do not read as probabilities. Measure the Brier score (mean squared error of the probability) and the expected calibration error, which compares predicted and observed rates in ten bins, then try isotonic calibration.

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=30000, n_features=20, n_informative=8, flip_y=0.05, random_state=2)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=2)
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(Xtr, ytr)
calibrated = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), method="isotonic", cv=3).fit(Xtr, ytr)


def ece(truth, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & ((prob < edges[i + 1]) if i < bins - 1 else (prob <= edges[i + 1]))
        if mask.any():
            total += mask.sum() * abs(truth[mask].mean() - prob[mask].mean())
    return total / len(truth)


for name, model in (("raw forest", forest), ("isotonic", calibrated)):
    prob = model.predict_proba(Xte)[:, 1]
    print(f"{name:11s} Brier {brier_score_loss(yte, prob):.4f}  ECE {ece(yte, prob):.4f}")
```

```text
raw forest  Brier 0.0741  ECE 0.0660
isotonic    Brier 0.0671  ECE 0.0093
```

Isotonic calibration cuts the expected calibration error from 0.0660 to 0.0093 and the Brier score from 0.0741 to 0.0671. Calibration repairs what a score means, not how well it ranks, so use it when a downstream rule consumes the number, as pricing or thresholds do. Calibrate on data the model did not train on, and recheck after any retraining.

**Chapter that teach this:** [Model evaluation](/docs/theory/ml/model-evaluation).

</details>

<details>
<summary><strong>Q5.</strong> Logistic regression or gradient boosting for a tabular problem?</summary>

Fit the linear model first, because it tells you which world you are in. If the signal is additive, boosting adds nothing. If the signal lives in interactions, a linear model cannot represent it however long you tune it. The block builds one dataset of each kind.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(5)
n = 20000
X = rng.normal(size=(n, 6))
additive = (X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] + rng.normal(size=n) > 0).astype(int)
interaction = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 0.5)).astype(int)
interaction = np.where(rng.random(n) < 0.05, 1 - interaction, interaction)
for name, y in (("additive signal", additive), ("interaction signal", interaction)):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    gbm = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    print(f"{name:19s} logistic AUC {roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]):.3f}  boosting AUC {roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1]):.3f}")
```

```text
additive signal     logistic AUC 0.888  boosting AUC 0.884
interaction signal  logistic AUC 0.498  boosting AUC 0.955
```

On the additive signal the two tie, 0.888 for logistic regression against 0.884 for boosting. On the interaction signal, where the label depends on the product of two features and a threshold on a third, logistic regression scores 0.498 and boosting 0.955. A coin flip is 0.5, so the linear model has learned nothing. Start with the simple baseline, move to boosting when the gap shows up, and keep the baseline as the number the complex model must beat.

**Chapter that teach this:** [Gradient boosting in practice](/docs/theory/ml/gradient-boosting-in-practice).

</details>

<details>
<summary><strong>Q6.</strong> Permutation importance says the key feature halved in importance after you added a near-duplicate. What happened?</summary>

The model spreads credit across correlated features. Permuting one of two near-copies barely hurts, because the other still carries the signal, so each shows a fraction of the importance the single feature had. Nothing about the underlying relationship changed.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(6)
n = 6000
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(scale=0.05, size=n)
x3 = rng.normal(size=n)
y = (x1 + 0.5 * x3 + rng.normal(scale=0.5, size=n) > 0).astype(int)
for label, X in (("features [x1, x3]", np.c_[x1, x3]), ("features [x1, copy of x1, x3]", np.c_[x1, x2, x3])):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    result = permutation_importance(forest, Xte, yte, n_repeats=10, random_state=0, scoring="roc_auc")
    print(f"{label:30s} AUC {roc_auc_score(yte, forest.predict_proba(Xte)[:, 1]):.3f}  permutation importance {np.round(result.importances_mean, 3).tolist()}")
```

```text
features [x1, x3]              AUC 0.928  permutation importance [0.349, 0.103]
features [x1, copy of x1, x3]  AUC 0.931  permutation importance [0.162, 0.051, 0.098]
```

With features x1 and x3 the importances are 0.349 and 0.103. After adding a copy of x1 they become 0.162, 0.051 and 0.098, while AUC barely moves, 0.928 to 0.931. Permuting either copy leaves the other intact, so neither figure shows what the pair contributes together; the two add to 0.213, well below the 0.349 of the single feature. Never read a drop in importance as a drop in relevance when features are correlated. Permute correlated groups together, or compare models trained with and without the group.

**Chapter that teach this:** [Explaining predictions](/docs/theory/ml/explaining-predictions).

</details>

<details>
<summary><strong>Q7.</strong> Cross-validation scores 0.99 AUC on patient visit data. What do you check?</summary>

Whether rows from the same patient sit on both sides of a split. Visits from one patient resemble each other, so a shuffled split lets the model recognise the patient instead of learning the condition. Group-aware splitting keeps each patient wholly in train or wholly in test. The block gives each of 300 patients a coin-flip label and ten near-identical visits.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

rng = np.random.default_rng(7)
patients, visits = 300, 10
groups = np.repeat(np.arange(patients), visits)
X = np.repeat(rng.normal(size=(patients, 40)), visits, axis=0) + rng.normal(scale=0.3, size=(patients * visits, 40))
y = np.repeat(rng.integers(0, 2, patients), visits)
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
shuffled = cross_val_score(forest, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
grouped = cross_val_score(forest, X, y, cv=GroupKFold(5), groups=groups, scoring="roc_auc").mean()
print("the label is a coin flip per patient, so honest AUC is 0.5")
print("shuffled KFold AUC:", round(shuffled, 4))
print("GroupKFold AUC:", round(grouped, 4))
```

```text
the label is a coin flip per patient, so honest AUC is 0.5
shuffled KFold AUC: 0.9999
GroupKFold AUC: 0.533
```

Shuffled five-fold cross-validation scores an AUC of 0.9999 on labels that are pure chance, while GroupKFold scores 0.533, close to the true 0.5. Whenever rows share an entity, a time order or a source document, split on that unit. The same logic applies to near-duplicate documents, to users in recommenders and to sessions in logs.

**Chapter that teach this:** [Model evaluation](/docs/theory/ml/model-evaluation).

</details>

## 2. Retrieval and RAG

Seven questions on fusion, metrics, reranking depth, permissions, index sizing, context and injection.

<details>
<summary><strong>Q8.</strong> Fuse a BM25 ranking and a dense ranking by hand. Why does reciprocal rank fusion work without calibrating scores?</summary>

Reciprocal rank fusion gives a document 1 / (60 + rank) from each list that contains it and adds the results. It uses ranks only, so it never compares a BM25 score with a cosine similarity, which live on unrelated scales. The block fuses a BM25 list d3, d1, d7, d2 with a dense list d1, d9, d3, d4.

```python
bm25 = ["d3", "d1", "d7", "d2"]
dense = ["d1", "d9", "d3", "d4"]
score = {}
for ranking in (bm25, dense):
    for rank, doc in enumerate(ranking, start=1):
        score[doc] = score.get(doc, 0.0) + 1.0 / (60 + rank)
for doc, value in sorted(score.items(), key=lambda item: -item[1]):
    print(doc, round(value, 6))
```

```text
d1 0.032522
d3 0.032266
d9 0.016129
d7 0.015873
d2 0.015625
d4 0.015625
```

d1 is second in BM25 and first in dense: 1/62 + 1/61 = 0.032522. d3 is first and third: 1/61 + 1/63 = 0.032266. d1 wins by a hair although neither list ranks it uniformly high. A document found by only one retriever scores at most 1/61 = 0.0164, so agreement beats a single high rank. d2 and d4 tie at 0.015625, so a real system needs a deterministic tie-break. The constant 60 flattens the gap between rank 1 and rank 10; a smaller constant would favour the top of each list.

**Chapters that teach this:** [Design an enterprise document Q&A system](/docs/senior/design-enterprise-document-qa); [Neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking).

</details>

<details>
<summary><strong>Q9.</strong> Compute precision, recall, MRR, AP and nDCG for a ranking by hand and say what each ignores.</summary>

The ranking places relevant documents at positions 2, 4 and 5 of six results, and four relevant documents exist in total, so one was never retrieved. For nDCG the relevant results have grades 2, 3 and 1 at those positions. Each metric answers a different question, so quote the one that matches the product.

```python
import math

relevance = [0, 1, 0, 1, 1, 0]
grades = [0, 2, 0, 3, 1, 0]
total_relevant = 4


def dcg(values):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(values))


hits = 0
average_precision = 0.0
for rank, rel in enumerate(relevance, start=1):
    if rel:
        hits += 1
        average_precision += hits / rank
first = next(rank for rank, rel in enumerate(relevance, start=1) if rel)
ideal = sorted([3, 2, 1, 0, 0, 0], reverse=True)
print("precision@3:", round(sum(relevance[:3]) / 3, 3), " recall@3:", sum(relevance[:3]) / total_relevant)
print("reciprocal rank:", 1 / first, " average precision:", average_precision / total_relevant)
print("DCG@5:", round(dcg(grades[:5]), 3), " ideal DCG@5:", round(dcg(ideal[:5]), 3), " nDCG@5:", round(dcg(grades[:5]) / dcg(ideal[:5]), 3))
```

```text
precision@3: 0.333  recall@3: 0.25
reciprocal rank: 0.5  average precision: 0.4
DCG@5: 5.294  ideal DCG@5: 9.393  nDCG@5: 0.564
```

Precision at 3 is 1/3 and recall at 3 is 1/4, since one relevant document is among the top three and four exist. The reciprocal rank is 1/2 because the first relevant document is second. Average precision is (1/2 + 2/4 + 3/5) / 4 = 0.4, and the missing fourth document drags it down. nDCG at 5 is 0.564, from a discounted gain of 5.294 against an ideal of 9.393, which rewards putting the grade-3 document first. Precision ignores order and unretrieved documents, MRR ignores everything after the first hit, and recall at k ignores order inside the top k.

**Chapter that teach this:** [Evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval).

</details>

<details>
<summary><strong>Q10.</strong> How deep should a cross-encoder rerank, and what can it never fix?</summary>

The reranker can only reorder what the first stage hands it, so the first stage's recall at depth N is a ceiling, and every extra candidate costs one more model call. A noisy reranker can also lose precision when it sees more distractors. The simulation gives each query one relevant document among 5,000 and a noisy first-stage score, then a sharper second-stage score on the shortlist.

```python
import numpy as np

rng = np.random.default_rng(0)
queries, corpus = 2000, 5000
print("depth  relevant in shortlist  hit@5 after rerank  reranker calls")
for depth in (5, 10, 20, 50, 100, 200):
    present = hit = 0
    for _ in range(queries):
        first_stage = rng.normal(size=corpus)
        first_stage[0] += 4.0
        shortlist = np.argsort(-first_stage)[:depth]
        if 0 in shortlist:
            present += 1
            second = rng.normal(size=depth) + np.where(shortlist == 0, 3.0, 0.0)
            hit += 0 in shortlist[np.argsort(-second)][:5]
    print(f"{depth:5d}  {present / queries:21.3f}  {hit / queries:18.3f}  {depth:14d}")
```

```text
depth  relevant in shortlist  hit@5 after rerank  reranker calls
    5                  0.818               0.818               5
   10                  0.866               0.864              10
   20                  0.914               0.903              20
   50                  0.961               0.908              50
  100                  0.976               0.883             100
  200                  0.988               0.822             200
```

The relevant document reaches the shortlist 0.818 of the time at depth 5 and 0.988 at depth 200, but hit@5 after reranking peaks at 0.908 for depth 50 and falls to 0.822 at depth 200, because 200 candidates give the noisy reranker many more chances to promote a wrong one. Depth 20 already reaches 0.903 at less than half the reranker calls of depth 50. Treat depth as a parameter tuned on your own questions, and fix first-stage recall before buying a bigger reranker. The numbers come from invented score distributions; the shape, not the values, is the lesson.

**Chapter that teach this:** [Contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking).

</details>

<details>
<summary><strong>Q11.</strong> Users see 30% of the corpus. Why does retrieve-then-filter fail, and how much must you over-fetch?</summary>

Post-filtering retrieves the top results and then drops the ones the user may not open, so the survivors are a thinning sample. If each retrieved document is visible with probability 0.3, the number kept follows a binomial distribution. Pre-filtering applies the permission inside the search so the top k are all visible.

```python
from scipy.stats import binom

visible = 0.3
print("retrieved  expected kept  P(at least 5 kept)")
for retrieved in (10, 20, 30, 40):
    print(f"{retrieved:9d}  {retrieved * visible:13.1f}  {1 - binom.cdf(4, retrieved, visible):18.3f}")
needed = next(k for k in range(5, 200) if 1 - binom.cdf(4, k, visible) >= 0.95)
print("smallest number to retrieve for a 95 percent chance of 5 kept:", needed, "which is", needed / 5, "times 5")
```

```text
retrieved  expected kept  P(at least 5 kept)
       10            3.0               0.150
       20            6.0               0.762
       30            9.0               0.970
       40           12.0               0.997
smallest number to retrieve for a 95 percent chance of 5 kept: 28 which is 5.6 times 5
```

Retrieving 10 keeps 3.0 on average, and the chance of keeping at least 5 is only 0.150. Retrieving 20 raises it to 0.762, and 30 to 0.970. To be 95% sure of 5 survivors you must retrieve 28, 5.6 times the target. The enterprise chapter measured 3.7 survivors rather than 3.0 because it forced each user's evidence to be visible. Prefer pre-filtering when the index supports filters, keep access lists fresh, and over-fetch only as a fallback.

**Chapter that teach this:** [Design an enterprise document Q&A system](/docs/senior/design-enterprise-document-qa).

</details>

<details>
<summary><strong>Q12.</strong> Estimate the memory of a vector index for 500,000 documents.</summary>

Count chunks, multiply by dimensions and bytes per value, then add the graph and metadata. Do it in the interview before naming a database. The block uses 20 chunks per document and 768-dimensional vectors.

```python
documents, chunks_per_document, dimensions, links = 500_000, 20, 768, 32
chunks = documents * chunks_per_document
gb = 1e9
print("chunks:", f"{chunks:,}")
print("float32 vectors:", round(chunks * dimensions * 4 / gb, 2), "GB")
print("int8 vectors:", round(chunks * dimensions / gb, 2), "GB")
print("graph links at 32 per vector:", round(chunks * links * 4 / gb, 2), "GB")
print("float32 vectors truncated to 256 dimensions:", round(chunks * 256 * 4 / gb, 2), "GB")
```

```text
chunks: 10,000,000
float32 vectors: 30.72 GB
int8 vectors: 7.68 GB
graph links at 32 per vector: 1.28 GB
float32 vectors truncated to 256 dimensions: 10.24 GB
```

500,000 documents give 10 million chunks. Float32 vectors take 30.72 GB, int8 vectors 7.68 GB, and graph links at 32 per vector another 1.28 GB. Shorter vectors help too: truncating to 256 dimensions would give 10.24 GB, but only if the embedding model keeps its quality at that size, which you must test. Memory, not request rate, usually sets the infrastructure floor, which is why quantisation is a week-one decision rather than a late optimisation.

**Chapter that teach this:** [Design an enterprise document Q&A system](/docs/senior/design-enterprise-document-qa).

</details>

<details>
<summary><strong>Q13.</strong> Why does prefixing each chunk with its document context improve retrieval?</summary>

A chunk that says revenue grew by 12% over the previous quarter fits twenty-four companies and quarters equally well. Stripped of its title, it cannot be told apart from its neighbours. Adding the company and quarter before indexing gives the retriever words to match. The block counts shared query words between a question and 24 near-identical chunks.

```python
import re

companies = ["Brightwell Energy", "Harbour Foods", "Corvane Steel", "Nimbus Cloud", "Alder Rail", "Tessa Pharma"]
quarters = ["Q1", "Q2", "Q3", "Q4"]
query = "How fast did revenue grow at Brightwell Energy in Q3 2025"
words = lambda text: set(re.findall(r"[a-z0-9]+", text.lower()))
for prefixed in (False, True):
    scores = {}
    for company in companies:
        for quarter in quarters:
            text = "Revenue grew by 12% over the previous quarter."
            if prefixed:
                text = f"{company}, {quarter} 2025 report. " + text
            scores[(company, quarter)] = len(words(query) & words(text))
    best = max(scores.values())
    tied = [key for key, value in scores.items() if value == best]
    print("with context prefix" if prefixed else "bare chunks", "| best score", best, "| chunks tied at the top", len(tied), "| chance the right one is first", round(1 / len(tied), 3))
```

```text
bare chunks | best score 1 | chunks tied at the top 24 | chance the right one is first 0.042
with context prefix | best score 5 | chunks tied at the top 1 | chance the right one is first 1.0
```

Without a prefix every chunk shares one word with the query, so all 24 tie and the right one is first with probability 0.042. With the prefix the right chunk shares five words and wins outright. This is a toy built to show the mechanism; on real text the gain depends on how many distinguishing words chunks already carry, so measure it on your corpus. The cost is one model call per chunk at indexing time, paid once.

**Chapter that teach this:** [Contextual retrieval and reranking](/docs/genai/rag-advanced/contextual-retrieval-and-reranking).

</details>

<details>
<summary><strong>Q14.</strong> Your red-team found 3 successful prompt injections in 40 attempts. What do you report, and how clean must a retest be?</summary>

Report the rate with an interval, not as a bare count. With 40 trials the interval is wide. A clean run proves less than it feels like it does, and the upper bound after zero failures shrinks only slowly with the number of trials. Treat retrieved text as data, give the answering step no tools, and keep the gateway, not the prompt, in charge of permissions.

```python
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(centre - half, 3), round(centre + half, 3)


print("3 successful attacks in 40 tries:", wilson(3, 40))
for n in (40, 100, 300, 1000):
    print(f"0 successful attacks in {n} tries: upper bound {wilson(0, n)[1]}")
```

```text
3 successful attacks in 40 tries: (0.026, 0.199)
0 successful attacks in 40 tries: upper bound 0.088
0 successful attacks in 100 tries: upper bound 0.037
0 successful attacks in 300 tries: upper bound 0.013
0 successful attacks in 1000 tries: upper bound 0.004
```

Three of 40 is 7.5%, with a 95% Wilson interval of 0.026 to 0.199. Zero failures in 40 tries still leaves an upper bound of 0.088, and even 100 clean tries leave 0.037. Getting the bound under 0.4% takes 1,000 clean attempts. So say which attack catalogue you ran, report the interval, rerun after every change to prompts, tools or retrieval, and never describe a clean small run as safe.

**Chapters that teach this:** [Red-teaming LLM systems](/docs/governance/red-teaming-llm-systems); [Design an enterprise document Q&A system](/docs/senior/design-enterprise-document-qa).

</details>

## 3. LLM adaptation and serving

Seven questions on adaptation economics, adapter size, KV cache, decode speed, batching, quantisation and speculation.

<details>
<summary><strong>Q15.</strong> Traffic is 400,000 requests a month with a 3,000-token prompt. Do you fine-tune a small model?</summary>

Compute the break-even before you argue. A tuned small model has a tiny per-request cost but a fixed monthly cost for amortised training and the engineer time that keeps it healthy. The long prompt has no fixed cost but pays for 3,000 tokens every call, and provider caching can make most of those tokens ten times cheaper. The prices below are the placeholder price list used in the chapter, not any provider's rates.

```python
PRICE = {"big_in": 3.0, "big_out": 15.0, "small_in": 0.3, "small_out": 1.5}


def per_request(prompt_tokens, output_tokens, price_in, price_out, cached_share=0.0, cache_read=0.1):
    effective = prompt_tokens * (1 - cached_share) + prompt_tokens * cached_share * cache_read
    return (effective * price_in + output_tokens * price_out) / 1_000_000


long_prompt = per_request(3000, 20, PRICE["big_in"], PRICE["big_out"])
cached_prompt = per_request(3000, 20, PRICE["big_in"], PRICE["big_out"], cached_share=0.9)
tuned = per_request(60, 20, PRICE["small_in"], PRICE["small_out"])
fixed_tuned = 3000 / 12 + 8 * 120
print("per request:", long_prompt, cached_prompt, tuned, " fixed monthly cost of the tuned model:", fixed_tuned)
print("break-even against the plain long prompt:", round(fixed_tuned / (long_prompt - tuned)), "requests a month")
print("break-even against the 90 percent cached prompt:", round(fixed_tuned / (cached_prompt - tuned)), "requests a month")
```

```text
per request: 0.0093 0.00201 4.8e-05  fixed monthly cost of the tuned model: 1210.0
break-even against the plain long prompt: 130783 requests a month
break-even against the 90 percent cached prompt: 616718 requests a month
```

A plain long prompt costs 0.0093 per request, a 90% cached one 0.00201, and the tuned model 0.000048 plus 1,210 a month fixed, of which 960 is engineer time. Fine-tuning breaks even at 130,783 requests a month against the plain prompt but only at 616,718 against the cached one. At 400,000 requests a month, caching the prompt is the first move and tuning does not pay yet. Re-run the numbers when prices change, and remember that a tuned model must be re-evaluated whenever the base model or task changes.

**Chapter that teach this:** [Prompt, retrieve or fine-tune?](/docs/llm-engineering/prompt-retrieve-or-fine-tune).

</details>

<details>
<summary><strong>Q16.</strong> How many parameters does a rank-16 LoRA add to an 8B model, and why does it fit on one GPU?</summary>

For each adapted weight of shape in by out, LoRA adds rank x (in + out) parameters. Sum that over the modules and layers. The block uses the published Llama 3.1 8B shapes: 32 layers, hidden size 4096, feed-forward size 14,336, 32 query heads and 8 key-value heads.

```python
hidden, inter, layers, heads, kv_heads, vocab = 4096, 14336, 32, 32, 8, 128256
kv = kv_heads * (hidden // heads)
shapes = {"q": (hidden, hidden), "k": (hidden, kv), "v": (hidden, kv), "o": (hidden, hidden), "gate": (hidden, inter), "up": (hidden, inter), "down": (inter, hidden)}
base = 2 * vocab * hidden + layers * (sum(i * o for i, o in shapes.values()) + 2 * hidden) + hidden
print("base parameters:", f"{base:,}")
for rank in (8, 16, 64):
    qv = layers * sum(rank * sum(shapes[m]) for m in ("q", "v"))
    allin = layers * sum(rank * (i + o) for i, o in shapes.values())
    print(f"rank {rank:2d}: q,v {qv:>12,} ({qv / base:.3%})   all linear {allin:>12,} ({allin / base:.3%})   adapter {allin * 2 / 1e6:.0f} MB in bf16")
allin16 = layers * sum(16 * (i + o) for i, o in shapes.values())
print("rank 16 gradients and two Adam moments in float32:", round(allin16 * 12 / 1e9, 2), "GB")
print("frozen base in bf16:", round(base * 2 / 1e9, 2), "GB; full fine-tuning states with Adam in float32:", round(base * 16 / 1e9, 1), "GB")
```

```text
base parameters: 8,030,261,248
rank  8: q,v    3,407,872 (0.042%)   all linear   20,971,520 (0.261%)   adapter 42 MB in bf16
rank 16: q,v    6,815,744 (0.085%)   all linear   41,943,040 (0.522%)   adapter 84 MB in bf16
rank 64: q,v   27,262,976 (0.340%)   all linear  167,772,160 (2.089%)   adapter 336 MB in bf16
rank 16 gradients and two Adam moments in float32: 0.5 GB
frozen base in bf16: 16.06 GB; full fine-tuning states with Adam in float32: 128.5 GB
```

At rank 16 on all linear layers the adapter has 41,943,040 parameters, 0.522% of the 8,030,261,248 in the base, and 84 MB in bf16. Restricting to query and value gives 6,815,744. Training state for the adapter, gradients plus two Adam moments in float32, is about 0.5 GB, against 128.5 GB of weight, gradient and Adam state for full fine-tuning. The frozen base in bf16 still needs 16.06 GB, and activations are extra. Memory arithmetic like this is an estimate; measure a real run, as the chapter does on a small model.

**Chapter that teach this:** [Supervised fine-tuning with LoRA](/docs/llm-engineering/supervised-fine-tuning-with-lora).

</details>

<details>
<summary><strong>Q17.</strong> How much memory does the KV cache take for Llama 3.1 8B, and how many users does that leave you?</summary>

Each token stores a key and a value for every layer and every key-value head: bytes per token = 2 x layers x KV heads x head dimension x bytes per value. The 8B model has 32 layers, 8 key-value heads (grouped-query attention) and a head dimension of 128.

```python
layers, kv_heads, head_dim, bytes_per_value = 32, 8, 128, 2
per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
print("KV cache per token:", per_token, "bytes =", per_token // 1024, "KiB")
for context in (4096, 32768, 131072):
    print(f"context {context:>7,}: {per_token * context / 1e9:6.2f} GB per sequence")
budget = 24e9
print("sequences that fit in 24 GB at 8,192 tokens:", int(budget // (per_token * 8192)), " at 32,768 tokens:", int(budget // (per_token * 32768)))
print("with an 8-bit cache at 32,768 tokens:", int(budget // (per_token / 2 * 32768)))
```

```text
KV cache per token: 131072 bytes = 128 KiB
context   4,096:   0.54 GB per sequence
context  32,768:   4.29 GB per sequence
context 131,072:  17.18 GB per sequence
sequences that fit in 24 GB at 8,192 tokens: 22  at 32,768 tokens: 5
with an 8-bit cache at 32,768 tokens: 11
```

That is 131,072 bytes, 128 KiB, per token in bf16. One 32,768-token conversation holds 4.29 GB and a 131,072-token one 17.18 GB. A 24 GB cache budget serves 22 sequences at 8,192 tokens but only 5 at 32,768. Storing the cache in 8 bits would raise that to 11, at some accuracy risk that must be measured. The cache grows with every token of every user, so context length and concurrency trade directly against each other.

**Chapter that teach this:** [The KV cache and PagedAttention](/docs/llm-engineering/kv-cache-and-paged-attention).

</details>

<details>
<summary><strong>Q18.</strong> Why is single-user decoding bandwidth-bound, and what does batching change?</summary>

Producing one token needs every weight, so each step streams the whole model from memory. The step cannot be faster than bytes divided by bandwidth. The block uses a hypothetical accelerator with 400 TFLOP/s of compute and 2 TB/s of memory bandwidth, not any real product, and the 8.03 billion parameters of an 8B model.

```python
params = 8_030_261_248
flops, bandwidth = 400e12, 2e12
print("ridge point:", flops / bandwidth, "FLOP per byte")
for name, size in (("bf16", 2), ("int8", 1), ("int4", 0.5)):
    step = params * size / bandwidth
    print(f"{name}: weights {params * size / 1e9:5.1f} GB, one decode step at least {step * 1e3:5.2f} ms, at most {1 / step:5.0f} tokens per second for one user")
print("batch  step ms  total tokens/s  per-user tokens/s  (bf16)")
for batch in (1, 8, 64, 256):
    step = max(2 * params * batch / flops, params * 2 / bandwidth)
    print(f"{batch:5d}  {step * 1e3:7.2f}  {batch / step:14.0f}  {1 / step:17.1f}")
```

```text
ridge point: 200.0 FLOP per byte
bf16: weights  16.1 GB, one decode step at least  8.03 ms, at most   125 tokens per second for one user
int8: weights   8.0 GB, one decode step at least  4.02 ms, at most   249 tokens per second for one user
int4: weights   4.0 GB, one decode step at least  2.01 ms, at most   498 tokens per second for one user
batch  step ms  total tokens/s  per-user tokens/s  (bf16)
    1     8.03             125              124.5
    8     8.03             996              124.5
   64     8.03            7970              124.5
  256    10.28           24906               97.3
```

In bf16 the weights are 16.1 GB, so a step takes at least 8.03 ms and one user can see at most 125 tokens per second. Int8 halves that to 4.02 ms and int4 quarters it to 2.01 ms, because fewer bytes move. Batching is nearly free until the ridge point, 200 FLOP per byte here: batch 64 gives 7,970 tokens per second in total at the same 8.03 ms step, and only at batch 256 does the step grow to 10.28 ms. The model ignores KV cache traffic and attention, which add bytes per step as contexts grow.

**Chapter that teach this:** [Why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound).

</details>

<details>
<summary><strong>Q19.</strong> Why does continuous batching beat static batching?</summary>

A static batch runs until its longest request finishes, so short requests leave idle slots. Continuous batching refills a slot the moment a request ends. The simulation sends 64 requests with log-normal output lengths through 8 slots and counts decode steps.

```python
import heapq

import numpy as np

rng = np.random.default_rng(0)
slots, requests = 8, 64
lengths = np.clip(rng.lognormal(4.0, 0.8, requests).astype(int), 4, 600)
static_steps = sum(int(lengths[i:i + slots].max()) for i in range(0, requests, slots))
free = [0] * slots
for length in lengths:
    heapq.heappush(free, heapq.heappop(free) + int(length))
continuous_steps = max(free)
print("output tokens:", int(lengths.sum()), " mean length:", round(lengths.mean(), 1), " longest:", int(lengths.max()))
print("static batching steps:", static_steps, " tokens per step:", round(lengths.sum() / static_steps, 2))
print("continuous batching steps:", continuous_steps, " tokens per step:", round(lengths.sum() / continuous_steps, 2))
print("speed-up:", round(static_steps / continuous_steps, 2))
```

```text
output tokens: 4764  mean length: 74.4  longest: 261
static batching steps: 1374  tokens per step: 3.47
continuous batching steps: 763  tokens per step: 6.24
speed-up: 1.8
```

The requests produce 4,764 tokens with a mean length of 74.4 and a longest of 261. Static batching needs 1,374 steps, 3.47 tokens per step, and continuous batching 763 steps, 6.24 tokens per step, a 1.8 times speed-up. The gain grows with the spread of lengths. The simulation counts decode steps only: it ignores prefill, memory limits and scheduling overhead, so treat 1.8 as an illustration of the mechanism and measure your engine on your traffic.

**Chapter that teach this:** [Continuous batching and scheduling](/docs/llm-engineering/continuous-batching-and-scheduling).

</details>

<details>
<summary><strong>Q20.</strong> What does int4 quantisation save on an 8B model, and what does the headline number leave out?</summary>

Multiply parameters by bits per weight. Real 4-bit formats also store scales for small groups of weights, so the true size is a little above 4 bits per weight. Smaller weights also speed up decoding where it is bandwidth-bound.

```python
params = 8_030_261_248
print("format                      weights GB")
for name, bits in (("bf16", 16), ("int8", 8), ("int4", 4), ("int4 with fp16 scale per 128", 4 + 16 / 128)):
    print(f"{name:27s} {params * bits / 8 / 1e9:10.2f}")
print("KV cache per token, bf16 versus 8-bit (32 layers, 8 heads, 128 dims):", 2 * 32 * 8 * 128 * 2, "versus", 2 * 32 * 8 * 128, "bytes")
```

```text
format                      weights GB
bf16                             16.06
int8                              8.03
int4                              4.02
int4 with fp16 scale per 128       4.14
KV cache per token, bf16 versus 8-bit (32 layers, 8 heads, 128 dims): 131072 versus 65536 bytes
```

Weights take 16.06 GB in bf16, 8.03 GB in int8 and 4.02 GB in int4. Adding a 16-bit scale for every 128 weights makes int4 4.14 GB. The cache can be quantised too: 8-bit values halve the 131,072 bytes per token to 65,536. The headline leaves out accuracy. Outlier weights and activations make some layers fragile, so measure perplexity and your task metric at each bit width before adopting one, rather than trusting the size reduction.

**Chapter that teach this:** [Quantisation for inference](/docs/llm-engineering/quantisation-for-inference).

</details>

<details>
<summary><strong>Q21.</strong> When does speculative decoding help, and when does adding more draft tokens hurt?</summary>

A small model drafts gamma tokens and the large model verifies them in one pass. If each draft token is accepted with probability alpha, the expected number of tokens per large-model pass is (1 - alpha^(gamma+1)) / (1 - alpha). Drafting costs time too, here 5% of a large-model pass per drafted token.

```python
def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


draft_cost = 0.05
print("alpha  gamma  tokens per target pass  speed-up")
for alpha in (0.6, 0.8, 0.9):
    for gamma in (2, 4, 8):
        tokens = expected_tokens(alpha, gamma)
        print(f"{alpha:5.1f}  {gamma:5d}  {tokens:22.3f}  {tokens / (gamma * draft_cost + 1):8.2f}")
```

```text
alpha  gamma  tokens per target pass  speed-up
  0.6      2                   1.960      1.78
  0.6      4                   2.306      1.92
  0.6      8                   2.475      1.77
  0.8      2                   2.440      2.22
  0.8      4                   3.362      2.80
  0.8      8                   4.329      3.09
  0.9      2                   2.710      2.46
  0.9      4                   4.095      3.41
  0.9      8                   6.126      4.38
```

At alpha 0.8 and gamma 4 each pass yields 3.362 tokens for a speed-up of 2.80. At alpha 0.9 and gamma 8 it reaches 4.38. But at alpha 0.6 the speed-up is 1.92 for gamma 4 and falls to 1.77 for gamma 8, because the extra drafts are mostly rejected and still cost time. Acceptance depends on how well the draft matches the target on your traffic, so measure it. The chapter also explains why the benefit fades at large batch sizes, where the verifier is no longer bandwidth-bound.

**Chapter that teach this:** [Speculative decoding](/docs/llm-engineering/speculative-decoding).

</details>

## 4. Evaluation and governance

Seven questions on significance, judges, regression gates, the EU AI Act, incident clocks, fairness and privacy.

<details>
<summary><strong>Q22.</strong> System A scores 92 of 100 and system B 89 of 100. Is A better?</summary>

Not on this evidence. Each score has a wide interval at n = 100, and because both systems answer the same questions the right test is paired: only the questions on which they disagree carry information. Suppose A is right and B wrong on 7 questions, B right and A wrong on 4, and they agree on the other 89.

```python
import math

from scipy.stats import binomtest, norm


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(centre - half, 3), round(centre + half, 3)


print("system A 92 of 100:", wilson(92, 100), " system B 89 of 100:", wilson(89, 100))
a_only, b_only = 7, 4
print("paired discordant cases A only / B only:", a_only, b_only, " exact McNemar p:", round(binomtest(a_only, a_only + b_only, 0.5).pvalue, 3))


def paired_n(delta, discordant, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return math.ceil(((za * math.sqrt(discordant) + zb * math.sqrt(discordant - delta ** 2)) / delta) ** 2)


print("items for a paired test of a 3-point gap:", paired_n(0.03, 0.11), "if 11 percent disagree,", paired_n(0.03, 0.20), "if 20 percent disagree")
```

```text
system A 92 of 100: (0.85, 0.959)  system B 89 of 100: (0.814, 0.937)
paired discordant cases A only / B only: 7 4  exact McNemar p: 0.549
items for a paired test of a 3-point gap: 957 if 11 percent disagree, 1742 if 20 percent disagree
```

The two Wilson intervals, 0.85 to 0.959 and 0.814 to 0.937, overlap almost entirely. The exact McNemar test on 7 against 4 gives p = 0.549, no evidence of a difference. To detect a true 3-point gap with 80% power you would need 957 questions if the systems disagree on 11% of them, and 1,742 if they disagree on 20%. Report paired differences with intervals, size the evaluation set before you run it, and do not promote a change on a 3-point win over 100 items.

**Chapters that teach this:** [Build vs buy and model selection](/docs/senior/build-vs-buy-and-model-selection); [Project 2: model selection and judge calibration](/docs/llm-evals/project-2-model-selection-and-judge-calibration).

</details>

<details>
<summary><strong>Q23.</strong> An LLM judge agrees with human graders on 91.5% of 200 items. Is it good enough?</summary>

Agreement is inflated when one class dominates. Cohen's kappa subtracts the agreement expected by chance from the same marginals. Also check the class you care about, which is usually the failures. Suppose humans pass 180 items and fail 20, and the judge passes 174 of the 180 and catches 9 of the 20 failures.

```python
true_pass, judge_pass_given_pass = 180, 174
true_fail, judge_fail_given_fail = 20, 9
n = true_pass + true_fail
agree = (judge_pass_given_pass + judge_fail_given_fail) / n
judge_pass_rate = (judge_pass_given_pass + (true_fail - judge_fail_given_fail)) / n
human_pass_rate = true_pass / n
expected = judge_pass_rate * human_pass_rate + (1 - judge_pass_rate) * (1 - human_pass_rate)
print("raw agreement:", agree)
print("agreement expected by chance:", round(expected, 3))
print("Cohen kappa:", round((agree - expected) / (1 - expected), 3))
print("share of true failures the judge catches:", judge_fail_given_fail / true_fail)
```

```text
raw agreement: 0.915
agreement expected by chance: 0.84
Cohen kappa: 0.469
share of true failures the judge catches: 0.45
```

Raw agreement is 0.915, but the agreement expected by chance is 0.84, so kappa is 0.469, moderate at best. The judge catches only 45% of the true failures, so a dashboard built on it will look healthier than the product is. Calibrate the judge against a labelled sample per slice, test for position, length and self-preference bias, and report its recall on failures alongside its agreement.

**Chapter that teach this:** [Project 2: model selection and judge calibration](/docs/llm-evals/project-2-model-selection-and-judge-calibration).

</details>

<details>
<summary><strong>Q24.</strong> How do you set a CI gate on a metric that changes from run to run?</summary>

Measure the noise first. A system with sampling or tool variability gives a different score on each run even if nothing changed, so a gate tighter than the noise fails builds at random. Averaging several runs shrinks the noise. The simulation gives 200 items per-item success probabilities drawn from a Beta distribution with mean 0.79, then compares one-run and three-run gates.

```python
import numpy as np

rng = np.random.default_rng(0)
items = rng.beta(8, 2, size=200)
regressed = np.clip(items - 0.05, 0, 1)


def run_means(probabilities, groups, size):
    draws = rng.random((groups * size, len(probabilities))) < probabilities
    return draws.mean(axis=1).reshape(groups, size).mean(axis=1)


print("true accuracy:", round(items.mean(), 3), " std of one run:", round(run_means(items, 3000, 1).std(), 3), " std of a 3-run mean:", round(run_means(items, 3000, 3).std(), 3))
for size in (1, 3):
    baseline = run_means(items, 3000, size)
    same = run_means(items, 3000, size)
    worse = run_means(regressed, 3000, size)
    for gate in (0.03, 0.05):
        print(f"{size} run(s) each, fail if drop > {gate}: false alarms {np.mean(same - baseline < -gate):.3f}, catches a real 5-point regression {np.mean(worse - baseline < -gate):.3f}")
```

```text
true accuracy: 0.789  std of one run: 0.028  std of a 3-run mean: 0.016
1 run(s) each, fail if drop > 0.03: false alarms 0.231, catches a real 5-point regression 0.716
1 run(s) each, fail if drop > 0.05: false alarms 0.101, catches a real 5-point regression 0.513
3 run(s) each, fail if drop > 0.03: false alarms 0.090, catches a real 5-point regression 0.806
3 run(s) each, fail if drop > 0.05: false alarms 0.012, catches a real 5-point regression 0.501
```

One run has a standard deviation of 0.028, a three-run mean 0.016. A gate that fails on a drop above 3 points with single runs raises false alarms on 23.1% of unchanged builds and catches 71.6% of a real 5-point regression. Using the mean of three runs lowers false alarms to 9.0% and raises the catch rate to 80.6%. A 5-point gate nearly removes false alarms, 1.2% with three runs, but catches only half the real regressions. Choose the threshold from the noise and the cost of each error, and keep per-slice results so a small slice cannot hide a large drop.

**Chapter that teach this:** [Regression testing](/docs/llm-evals/regression-testing).

</details>

<details>
<summary><strong>Q25.</strong> A hiring-screening product is sold in the EU. Which AI Act dates matter on 8 October 2026?</summary>

This is engineering orientation, not legal advice. A CV-screening tool sits in Annex III point 4 (employment), so it is high-risk. Regulation (EU) 2026/1744, the Digital Omnibus on AI, entered into force on 27 July 2026 and replaced the earlier 2 August 2026 application date for those obligations. The block counts days from the reference date.

```python
from datetime import date

today = date(2026, 10, 8)
rows = [
    ("Art 50 transparency and the general date of application", date(2026, 8, 2)),
    ("new Art 5(1)(ba) and (bb) prohibitions; Art 50(2) for older generative systems", date(2026, 12, 2)),
    ("high-risk, Annex III (for example CV screening)", date(2027, 12, 2)),
    ("high-risk, Annex I products", date(2028, 8, 2)),
    ("high-risk systems for public authorities placed earlier", date(2030, 8, 2)),
]
for label, when in rows:
    days = (when - today).days
    print(f"{when}  {'applied ' + str(-days) + ' days ago' if days < 0 else 'in ' + str(days) + ' days'}  {label}")
```

```text
2026-08-02  applied 67 days ago  Art 50 transparency and the general date of application
2026-12-02  in 55 days  new Art 5(1)(ba) and (bb) prohibitions; Art 50(2) for older generative systems
2027-12-02  in 420 days  high-risk, Annex III (for example CV screening)
2028-08-02  in 664 days  high-risk, Annex I products
2030-08-02  in 1394 days  high-risk systems for public authorities placed earlier
```

The transparency rules of Article 50 and the general date of application passed 67 days ago. New prohibitions in Article 5(1)(ba) and (bb), and the marking duty of Article 50(2) for generative systems already on the market, apply in 55 days. The high-risk requirements of Chapter III Sections 1 to 3 for Annex III systems apply in 420 days, on 2 December 2027, and for Annex I products on 2 August 2028. A provider cannot rely on the narrow-task exemption of Article 6(3) if the system profiles natural persons. AI literacy and the prohibitions already apply. Classify at design time, record the date table with its legal source, and take advice from counsel before relying on it.

**Chapter that teach this:** [Regulation and model documentation](/docs/governance/regulation-and-model-documentation).

</details>

<details>
<summary><strong>Q26.</strong> You learn on 8 October 2026 that a high-risk system contributed to a serious incident. What are the reporting clocks?</summary>

Under Article 73 of the AI Act the provider reports serious incidents to the market surveillance authorities of the Member State where the incident occurred, immediately after establishing a causal link or its reasonable likelihood, and within fixed outer limits counted from awareness. An initial, incomplete report is allowed, followed by a complete one. This is orientation, not legal advice.

```python
from datetime import date, timedelta

aware = date(2026, 10, 8)
for label, days in (("serious incident, general rule", 15), ("death of a person", 10), ("widespread infringement or critical infrastructure disruption", 2)):
    print(f"{label:62s} within {days:2d} days: by {aware + timedelta(days=days)}")
```

```text
serious incident, general rule                                 within 15 days: by 2026-10-23
death of a person                                              within 10 days: by 2026-10-18
widespread infringement or critical infrastructure disruption  within  2 days: by 2026-10-10
```

The general outer limit is 15 days, which from 8 October 2026 is 23 October. For the death of a person it is 10 days, 18 October. For a widespread infringement or a serious and irreversible disruption of the management or operation of critical infrastructure it is 2 days, 10 October. Build the clock into the incident process before you need it: severity triage, an owner who can file, a template and a log of when awareness began. Whether these rules reach your system depends on its classification and on the application dates in the previous question.

**Chapters that teach this:** [AI incident response](/docs/governance/ai-incident-response); [Regulation and model documentation](/docs/governance/regulation-and-model-documentation).

</details>

<details>
<summary><strong>Q27.</strong> Two fairness metrics disagree about your model, and one group has only 40 positives. What do you conclude?</summary>

Group fairness metrics measure different things, and with different base rates they cannot all be satisfied together. Compare selection rates for demographic parity and true positive rates for equal opportunity. Then ask whether a gap is larger than sampling noise, because small slices produce gaps from chance alone. The block uses two groups with base rates of 30% and 20%.

```python
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, true_positive_rate

rng = np.random.default_rng(3)
n = 20000
group = rng.choice(["A", "B"], size=n, p=[0.7, 0.3])
y = (rng.random(n) < np.where(group == "A", 0.30, 0.20)).astype(int)
score = rng.normal(size=n) + 1.2 * y - np.where(group == "B", 0.3, 0.0)
pred = (score > 0.6).astype(int)
frame = MetricFrame(metrics={"selection rate": selection_rate, "true positive rate": true_positive_rate}, y_true=y, y_pred=pred, sensitive_features=group)
print(frame.by_group.round(3))
print("demographic parity difference:", round(demographic_parity_difference(y, pred, sensitive_features=group), 3))
print("equalised odds difference:", round(equalized_odds_difference(y, pred, sensitive_features=group), 3))
gaps = np.array([abs((rng.random(100) < 0.8).mean() - (rng.random(40) < 0.8).mean()) for _ in range(5000)])
print("identical true TPR 0.8, 100 versus 40 positives: gap above 0.05 in", round((gaps > 0.05).mean(), 3), "of samples, above 0.10 in", round((gaps > 0.10).mean(), 3))
```

```text
                     selection rate  true positive rate
sensitive_feature_0                                    
A                             0.415               0.725
B                             0.273               0.621
demographic parity difference: 0.142
equalised odds difference: 0.104
identical true TPR 0.8, 100 versus 40 positives: gap above 0.05 in 0.508 of samples, above 0.10 in 0.173
```

Group A has a selection rate of 0.415 and a true positive rate of 0.725, group B 0.273 and 0.621. The demographic parity difference is 0.142 and the equalised odds difference 0.104. Separately, if both groups had an identical true positive rate of 0.8 but 100 and 40 positives, a gap above 0.05 would appear in 50.8% of samples and a gap above 0.10 in 17.3%. Report each gap with an interval, choose the definition that matches the harm in the use case, and do not act on a gap measured on a few dozen cases without more data.

**Chapter that teach this:** [Fairness testing in practice](/docs/governance/fairness-testing-in-practice).

</details>

<details>
<summary><strong>Q28.</strong> How much noise does differential privacy add to a count, and what does epsilon mean in practice?</summary>

For a count, one person changes the answer by at most 1, so the sensitivity is 1. The Laplace mechanism adds noise with scale 1 / epsilon. A smaller epsilon means stronger privacy and more noise, and every further query spends more of the privacy budget.

```python
import math

import numpy as np

rng = np.random.default_rng(3)
sensitivity = 1
for epsilon in (0.1, 0.5, 1.0, 2.0):
    scale = sensitivity / epsilon
    noise = rng.laplace(scale=scale, size=200000)
    print(f"epsilon {epsilon}: noise scale {scale:4.1f}, mean absolute error {np.abs(noise).mean():5.2f}, P(error above 5) simulated {(np.abs(noise) > 5).mean():.4f}, formula {math.exp(-5 / scale):.4f}")
```

```text
epsilon 0.1: noise scale 10.0, mean absolute error 10.00, P(error above 5) simulated 0.6064, formula 0.6065
epsilon 0.5: noise scale  2.0, mean absolute error  2.00, P(error above 5) simulated 0.0817, formula 0.0821
epsilon 1.0: noise scale  1.0, mean absolute error  1.00, P(error above 5) simulated 0.0065, formula 0.0067
epsilon 2.0: noise scale  0.5, mean absolute error  0.50, P(error above 5) simulated 0.0001, formula 0.0000
```

At epsilon 0.5 the noise scale is 2, the mean absolute error 2.00, and an error above 5 happens with probability 0.082, in line with the formula exp(-5 / 2) = 0.0821. At epsilon 0.1 the scale is 10 and the error exceeds 5 in 60.6% of releases, which swamps a small count. At epsilon 2.0 the error is about 0.5. Choose epsilon from the smallest count you must publish accurately, track the cumulative budget across queries, and remember differential privacy bounds one release rather than fixing leaks in logs or prompts.

**Chapter that teach this:** [Privacy, PII and differential privacy](/docs/governance/privacy-pii-and-differential-privacy).

</details>

## 5. System design

Six questions on sizing, latency tails, funnels, interactive latency, agent reliability and shared platforms.

<details>
<summary><strong>Q29.</strong> Size an enterprise document Q&A system for 2 million documents and 20,000 staff.</summary>

Do the arithmetic in three lines before drawing boxes: corpus size, traffic, cost per question. The assumptions are 15 chunks of 350 tokens per document, 1,024-dimensional vectors, 35% of staff active a day with 6 questions each, and placeholder prices.

```python
documents, chunks_per_document, chunk_tokens, dimensions = 2_000_000, 15, 350, 1024
price_in, price_out = 3.0, 15.0
chunks = documents * chunks_per_document
questions_per_day = 20_000 * 0.35 * 6
peak_qps = questions_per_day * 0.15 / 3600 * 3
print("chunks:", f"{chunks:,}", " float32 vectors:", round(chunks * dimensions * 4 / 1e9, 1), "GB", " int8:", round(chunks * dimensions / 1e9, 1), "GB")
print("questions per day:", int(questions_per_day), " peak questions per second:", round(peak_qps, 1))
for k in (3, 6, 10):
    tokens_in = 400 + 60 + k * chunk_tokens
    per_question = (tokens_in * price_in + 350 * price_out) / 1e6
    print(f"{k:2d} chunks: {tokens_in:5d} input tokens, {per_question:.4f} per question, {per_question * questions_per_day * 22:8,.0f} a month (placeholder prices)")
```

```text
chunks: 30,000,000  float32 vectors: 122.9 GB  int8: 30.7 GB
questions per day: 42000  peak questions per second: 5.2
 3 chunks:  1510 input tokens, 0.0098 per question,    9,037 a month (placeholder prices)
 6 chunks:  2560 input tokens, 0.0129 per question,   11,947 a month (placeholder prices)
10 chunks:  3960 input tokens, 0.0171 per question,   15,828 a month (placeholder prices)
```

The index holds 30 million chunks: 122.9 GB as float32 or 30.7 GB as int8. Traffic is 42,000 questions a day and 5.2 per second at the busiest moment, which is small, so memory rather than request rate drives the design. The money is in the prompt: with placeholder prices, 3 chunks cost 9,037 a month, 6 chunks 11,947 and 10 chunks 15,828. Say next how you would handle permissions (inside the search), evaluation (hit rate, faithfulness, citation precision) and rollout in rings.

**Chapter that teach this:** [Design an enterprise document Q&A system](/docs/senior/design-enterprise-document-qa).

</details>

<details>
<summary><strong>Q30.</strong> A fraud decision has a 100 ms budget and fans out to eight feature lookups. Where does the tail come from?</summary>

From taking the maximum of eight parallel calls, not from the average call. The decision waits for the slowest lookup, so its distribution sits well to the right of any single one. The simulation uses log-normal latencies with illustrative parameters, not measurements.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200000
lookups = rng.lognormal(np.log(4.0), 0.6, size=(n, 8))
slowest = lookups.max(axis=1)
network = rng.lognormal(np.log(8.0), 0.4, size=n)
model = rng.lognormal(np.log(6.0), 0.3, size=n)
rules = rng.lognormal(np.log(1.5), 0.3, size=n)
p99 = lambda x: round(float(np.percentile(x, 99)), 1)
print("one lookup: median", round(float(np.median(lookups[:, 0])), 1), "ms, p99", p99(lookups[:, 0]), "ms")
print("slowest of 8 parallel lookups: median", round(float(np.median(slowest)), 1), "ms, p99", p99(slowest), "ms")
total = network + slowest + model + rules
print("stage p99s:", [p99(x) for x in (network, slowest, model, rules)], " their sum:", round(sum(float(np.percentile(x, 99)) for x in (network, slowest, model, rules)), 1), " p99 of the total:", p99(total))
timeout = 15.0
capped = network + np.minimum(slowest, timeout) + model + rules
print("with a 15 ms feature timeout: decisions using default features", round(float((slowest > timeout).mean()), 3), " p99 of the total:", p99(capped))
```

```text
one lookup: median 4.0 ms, p99 16.2 ms
slowest of 8 parallel lookups: median 9.2 ms, p99 24.5 ms
stage p99s: [20.3, 24.5, 12.1, 3.0]  their sum: 59.9  p99 of the total: 44.3
with a 15 ms feature timeout: decisions using default features 0.106  p99 of the total: 40.1
```

One lookup has a median of 4.0 ms and a p99 of 16.2 ms. The slowest of eight has a median of 9.2 ms and a p99 of 24.5 ms. The four stage p99s sum to 59.9 ms, yet the p99 of the whole decision is 44.3 ms, because percentiles do not add; the total must be simulated or measured, not summed. A 15 ms feature timeout makes 10.6% of decisions use default features and trims the total p99 to 40.1 ms. So set timeouts per dependency, decide the degraded behaviour in advance (default features and rules rather than fail open or closed), and log how often it triggers.

**Chapter that teach this:** [Design real-time fraud scoring](/docs/senior/design-real-time-fraud-scoring).

</details>

<details>
<summary><strong>Q31.</strong> Why does a ranking system use a funnel, and where does its cost go?</summary>

Running the best model on every item is impossible, so cheap stages cut the catalogue and expensive stages see only survivors. The cost is dominated by the heavy ranker's depth. The unit costs below are assumptions used in the chapter, not measurements: 3 ms for retrieval, 2 microseconds per item for the light ranker, 60 microseconds for the heavy one.

```python
retrieval, light, heavy = 3e-3, 2e-6, 60e-6
requests_per_second, load = 20_000, 0.5
catalogue = 50_000_000
print("heavy ranker over the whole catalogue:", catalogue * heavy, "CPU seconds per request")
for heavy_depth in (200, 800):
    seconds = retrieval + 1000 * light + heavy_depth * heavy
    print(f"heavy ranker keeps {heavy_depth}: {seconds * 1e3:.1f} ms per request, {seconds * requests_per_second:.0f} busy cores, {seconds * requests_per_second / load:.0f} cores at 50 percent load")
```

```text
heavy ranker over the whole catalogue: 3000.0 CPU seconds per request
heavy ranker keeps 200: 17.0 ms per request, 340 busy cores, 680 cores at 50 percent load
heavy ranker keeps 800: 53.0 ms per request, 1060 busy cores, 2120 cores at 50 percent load
```

Scoring 50 million items with the heavy ranker would take 3,000 CPU seconds per request. The funnel that retrieves 1,000, light-ranks them and heavy-ranks 200 costs 17.0 ms, 340 busy cores at 20,000 requests per second and 680 at half load. Letting the heavy ranker keep 800 raises that to 53.0 ms and 2,120 cores. An item dropped early cannot be recovered, so the first stage's recall caps quality, while the heavy depth sets the bill.

**Chapter that teach this:** [Design search and recommendation ranking](/docs/senior/design-search-and-recommendation-ranking).

</details>

<details>
<summary><strong>Q32.</strong> Why does a code-completion service debounce keystrokes, and what does it cost?</summary>

A completion is useful only if it arrives before the next keystroke. Without a debounce every keystroke sends a request, most of which are stale on arrival. A debounce waits for a pause, sends fewer requests, and each is more likely to be wanted. The simulation uses bursty typing, 85% of gaps around 150 ms and 15% around two seconds, which is an assumption, not a measurement of anyone's users.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200000
burst = rng.random(n) < 0.85
gaps = np.where(burst, rng.exponential(150, n), rng.exponential(2000, n))
next_gap = np.append(gaps[1:], 1e9)
print("model latency  debounce  requests per 100 keys  wanted share per request  wanted per 100 keys")
for latency in (600, 300):
    for debounce in (0, 300):
        fires = next_gap >= debounce
        shown = fires & (next_gap >= debounce + latency)
        print(f"{latency:10d} ms  {debounce:6d} ms  {fires.mean() * 100:21.1f}  {shown.sum() / fires.sum():24.3f}  {shown.mean() * 100:19.1f}")
```

```text
model latency  debounce  requests per 100 keys  wanted share per request  wanted per 100 keys
       600 ms       0 ms                  100.0                     0.126                 12.6
       600 ms     300 ms                   24.3                     0.398                  9.7
       300 ms       0 ms                  100.0                     0.243                 24.3
       300 ms     300 ms                   24.3                     0.518                 12.6
```

With a 600 ms model and no debounce, 100 requests per 100 keystrokes yield 12.6 wanted suggestions, 0.126 per request. A 300 ms debounce cuts requests to 24.3 and raises the wanted share to 0.398, but shows fewer suggestions in total, 9.7. A faster 300 ms model with the same debounce reaches 0.518 per request and 12.6 in total. Model latency and debounce are one decision, and a smaller prompt or model is the cheapest way to improve both.

**Chapter that teach this:** [Design an AI coding assistant](/docs/senior/design-ai-coding-assistant).

</details>

<details>
<summary><strong>Q33.</strong> A support agent resolves 65% of tasks per trial. How reliable is it for a customer, and is automation cheaper than people?</summary>

Per-trial success hides the repeat experience. pass@k asks whether at least one of k attempts succeeds, which suits code generation. A customer needs it right every time, which is pass^k, the chance that all k attempts succeed. Easy tasks and hard tasks differ, so the two metrics diverge. The block assumes 60% easy tasks at 0.95 success and 40% hard tasks at 0.20.

```python
easy_share, p_easy, p_hard = 0.6, 0.95, 0.20
print("per-trial success:", round(easy_share * p_easy + (1 - easy_share) * p_hard, 2))
print("k  pass@k  pass^k")
for k in (1, 2, 4, 8):
    at_k = easy_share * (1 - (1 - p_easy) ** k) + (1 - easy_share) * (1 - (1 - p_hard) ** k)
    all_k = easy_share * p_easy ** k + (1 - easy_share) * p_hard ** k
    print(f"{k}  {at_k:6.3f}  {all_k:6.3f}")
resolved = 0.65
agent, human, wrong = 0.05, 4.0, 15.0
print("cost per ticket, everything automated at 65 percent resolved:", round(agent + (1 - resolved) * (human + wrong), 2), " everything to people:", human)
```

```text
per-trial success: 0.65
k  pass@k  pass^k
1   0.650   0.650
2   0.742   0.557
4   0.836   0.489
8   0.933   0.398
cost per ticket, everything automated at 65 percent resolved: 6.7  everything to people: 4.0
```

Per-trial success is 0.65. At k = 8, pass@8 climbs to 0.933 while pass^8 falls to 0.398. Cost matters as much: with placeholder costs of 0.05 per agent attempt, 4.0 per human ticket and 15.0 extra for a wrong automated answer, automating everything at 65% resolution costs 6.70 per ticket against 4.00 for sending everything to people. Escalate on low confidence and on risky intents, evaluate on repeated trials, and manage cost per correctly resolved ticket instead of containment.

**Chapter that teach this:** [Design a customer support agent platform](/docs/senior/design-customer-support-agent-platform).

</details>

<details>
<summary><strong>Q34.</strong> Two teams share 64 GPUs and 640 CPU cores. How do you divide them fairly?</summary>

Splitting each resource equally goes wrong when teams need different mixes. Dominant resource fairness gives each team the same share of the resource it uses most. Team A runs tasks of 4 GPUs and 16 CPUs, team B tasks of 1 GPU and 40 CPUs, so A is GPU-bound and B is CPU-bound. The block runs progressive filling, always serving the team with the smaller dominant share.

```python
capacity = {"gpu": 64, "cpu": 640}
demand = {"A": {"gpu": 4, "cpu": 16}, "B": {"gpu": 1, "cpu": 40}}
tasks = {"A": 0, "B": 0}
used = {"gpu": 0, "cpu": 0}


def dominant_share(team):
    return tasks[team] * max(demand[team][r] / capacity[r] for r in capacity)


while True:
    for team in sorted(tasks, key=dominant_share):
        if all(used[r] + demand[team][r] <= capacity[r] for r in capacity):
            tasks[team] += 1
            for r in capacity:
                used[r] += demand[team][r]
            break
    else:
        break
print("tasks placed:", tasks, " resources used:", used)
print("dominant shares:", {team: round(dominant_share(team), 3) for team in tasks})
print("equal split of GPUs (32 each) would need", 32 // 4 * 16 + 32 // 1 * 40, "CPUs; the cluster has", capacity["cpu"])
```

```text
tasks placed: {'A': 12, 'B': 11}  resources used: {'gpu': 59, 'cpu': 632}
dominant shares: {'A': 0.75, 'B': 0.688}
equal split of GPUs (32 each) would need 1408 CPUs; the cluster has 640
```

The result is 12 tasks for A and 11 for B, using 59 GPUs and 632 CPUs, with dominant shares of 0.75 and 0.688. An equal split of 32 GPUs each would have A run 8 tasks and B run 32, which together need 1,408 CPUs against the 640 available. Each task adds 0.0625 to its team's dominant share, so the allocation stops when the next task does not fit. Add quotas, preemption rules and chargeback on top; fairness alone does not decide who pays for idle capacity.

**Chapter that teach this:** [Design an ML platform for many teams](/docs/senior/design-ml-platform).

</details>

## 6. Senior craft

Six questions on estimation, buy against host, return on investment, design-doc evidence, coaching and on-call.

<details>
<summary><strong>Q35.</strong> Your team estimates 21 days by adding five likely values. What do you tell the sponsor?</summary>

That 21 days is the sum of the five most likely values, which almost no project achieves, because each task can overrun by more than it can underrun, and some risks hit several tasks at once. Give a range with a probability. The simulation uses triangular distributions over optimistic, likely and pessimistic days, and two data-dependent tasks that take 1.6 times longer when the data is bad, with a 40% chance of that.

```python
import numpy as np

tasks = [("data audit", 2, 3, 8, False), ("baseline", 3, 5, 12, True), ("eval set", 3, 4, 9, True), ("integration", 4, 6, 14, False), ("rollout", 2, 3, 6, False)]
likely = sum(t[2] for t in tasks)
n = 100000


def simulate(shared, bad_probability=0.4, slowdown=1.6):
    rng = np.random.default_rng(2)
    total = np.zeros(n)
    shared_draw = rng.random(n)
    for _, low, mode, high, data_dependent in tasks:
        days = rng.triangular(low, mode, high, n)
        own_draw = rng.random(n)
        bad = (shared_draw if shared else own_draw) < bad_probability
        total += np.where(data_dependent & bad, days * slowdown, days)
    return total


rng = np.random.default_rng(1)
plain = sum(rng.triangular(low, mode, high, n) for _, low, mode, high, _ in tasks)
print("sum of likely values:", likely, "days; share of simulated projects that finish by then:", round(float((plain <= likely).mean()), 3))
print("independent tasks: P50", round(float(np.percentile(plain, 50)), 1), " P85", round(float(np.percentile(plain, 85)), 1))
for label, shared in (("shared data risk", True), ("independent data risk", False)):
    x = simulate(shared)
    print(f"{label:22s} mean {x.mean():.1f}  P50 {np.percentile(x, 50):.1f}  P85 {np.percentile(x, 85):.1f}  P95 {np.percentile(x, 95):.1f}")
```

```text
sum of likely values: 21 days; share of simulated projects that finish by then: 0.015
independent tasks: P50 27.8  P85 31.8
shared data risk       mean 30.9  P50 30.2  P85 36.7  P95 40.7
independent data risk  mean 30.9  P50 30.5  P85 35.9  P95 39.3
```

Only 1.5% of simulated projects finish within 21 days. With independent tasks the median is 27.8 days and the 85th percentile 31.8. Adding the data risk gives a mean of 30.9 days either way, but the shared version has a longer tail: 36.7 days at P85 and 40.7 at P95, against 35.9 and 39.3 when each task draws its own risk. Commit to the P85, plan to the median, and put the data audit first, since it narrows the biggest uncertainty cheapest.

**Chapter that teach this:** [Estimation and planning](/docs/senior/estimation-and-planning).

</details>

<details>
<summary><strong>Q36.</strong> API or self-hosted open weights: how do you decide?</summary>

First remove options with hard gates: data residency, licence, latency and availability. Then compare cost at your volume including the cost of mistakes. Hosting is a staircase, because replicas come in whole units and availability needs at least two, plus the people who run them. The parameters below are placeholders: an API at 0.012 per request and 91% accuracy, replicas at 3,500 a month serving 600,000 requests each, 15,000 a month of operations, 88% accuracy for the hosted model and 0.05 per wrong answer.

```python
import math

api_price, api_accuracy = 0.012, 0.91
gpu_month, replica_capacity, min_replicas, people = 3500.0, 600_000, 2, 15000.0
host_accuracy, error_cost = 0.88, 0.05
print("requests/month   API   self-hosted   API with errors   hosted with errors")
for volume in (100_000, 300_000, 1_000_000, 3_000_000, 6_000_000):
    replicas = max(min_replicas, math.ceil(volume / replica_capacity))
    host = replicas * gpu_month + people
    api = volume * api_price
    print(f"{volume:14,d}  {api:6,.0f}  {host:11,.0f}  {api + volume * (1 - api_accuracy) * error_cost:15,.0f}  {host + volume * (1 - host_accuracy) * error_cost:18,.0f}")
for ignore_errors in (True, False):
    for volume in range(100_000, 8_000_000, 10_000):
        replicas = max(min_replicas, math.ceil(volume / replica_capacity))
        extra_h = 0 if ignore_errors else volume * (1 - host_accuracy) * error_cost
        extra_a = 0 if ignore_errors else volume * (1 - api_accuracy) * error_cost
        if replicas * gpu_month + people + extra_h < volume * api_price + extra_a:
            print("hosting first wins", "ignoring errors" if ignore_errors else "counting errors", "at", f"{volume:,}", "requests a month")
            break
```

```text
requests/month   API   self-hosted   API with errors   hosted with errors
       100,000   1,200       22,000            1,650              22,600
       300,000   3,600       22,000            4,950              23,800
     1,000,000  12,000       22,000           16,500              28,000
     3,000,000  36,000       32,500           49,500              50,500
     6,000,000  72,000       50,000           99,000              86,000
hosting first wins ignoring errors at 2,710,000 requests a month
hosting first wins counting errors at 3,430,000 requests a month
```

At 100,000 requests a month the API costs 1,200 against 22,000 for hosting. Hosting first wins at 2,710,000 requests a month if errors are ignored, and only at 3,430,000 once the hosted model's extra errors are priced. Below those volumes the API is cheaper even though the per-request price looks higher. Decide with your own volumes, write down the exit plan for model retirement, and re-run the comparison when prices or accuracy change.

**Chapter that teach this:** [Build vs buy and model selection](/docs/senior/build-vs-buy-and-model-selection).

</details>

<details>
<summary><strong>Q37.</strong> How do you present the return on an AI feature to a finance lead?</summary>

As a range with a probability, not one number. List the uncertain inputs, give each a distribution, simulate, and report percentiles and the chance of a loss. The invented inputs here are 40,000 to 90,000 tickets a month, 10% to 40% deflected, 3.00 to 5.50 saved per ticket, 18,000 to 45,000 a month to run and 120,000 to 320,000 to build.

```python
import numpy as np

rng = np.random.default_rng(3)
n = 200000
tickets = rng.triangular(40000, 60000, 90000, n)
deflected = rng.triangular(0.10, 0.25, 0.40, n)
saving = rng.triangular(3.0, 4.0, 5.5, n)
run_cost = rng.triangular(18000, 25000, 45000, n)
build = rng.triangular(120000, 180000, 320000, n)
monthly = tickets * deflected * saving - run_cost
net_24 = monthly * 24 - build
point = 60000 * 0.25 * 4.0 - 25000
print("point estimate: monthly net", point, " payback months", round(180000 / point, 1), " 24-month net", point * 24 - 180000)
print("monthly net P10 / P50 / P90:", np.round(np.percentile(monthly, [10, 50, 90])).astype(int).tolist())
print("24-month net P10 / P50 / P90:", np.round(np.percentile(net_24, [10, 50, 90])).astype(int).tolist())
print("chance the 24-month net is positive:", round(float((net_24 > 0).mean()), 3), " chance of payback within 12 months:", round(float(((monthly * 12 - build) > 0).mean()), 3))
```

```text
point estimate: monthly net 35000.0  payback months 5.1  24-month net 660000.0
monthly net P10 / P50 / P90: [9862, 34758, 66256]
24-month net P10 / P50 / P90: [27433, 627359, 1385639]
chance the 24-month net is positive: 0.912  chance of payback within 12 months: 0.803
```

The point estimate says 35,000 a month net, payback in 5.1 months and a 24-month net of 660,000. The simulation gives a monthly net of 9,862 at P10, 34,758 at P50 and 66,256 at P90, a 24-month net between 27,433 and 1,385,639, a 91.2% chance of being positive and an 80.3% chance of paying back within 12 months. The point estimate sat on the median but hid the lower tail: one run in ten nets less than 27,433 over two years. Include people and maintenance, not just tokens, and say which input moves the answer most.

**Chapter that teach this:** [Cost modelling and ROI](/docs/senior/cost-modelling-and-roi).

</details>

<details>
<summary><strong>Q38.</strong> Your design doc says ship if accuracy exceeds 80%. How large must the evaluation set be for that sentence to mean something?</summary>

A threshold on a measured accuracy is only as sharp as the interval around it. A design doc should state the numbers that would prove the design wrong, including how many items back them. The block computes Wilson intervals around an observed 80%.

```python
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


print("items  interval around an observed 80 percent  half-width")
for n in (50, 100, 200, 400, 800):
    low, high = wilson(int(0.8 * n), n)
    print(f"{n:5d}  {low:.3f} to {high:.3f}  {(high - low) / 2:.3f}")
for half_width in (0.10, 0.05, 0.03):
    print("items for a half-width of", half_width, ":", math.ceil(1.96 ** 2 * 0.8 * 0.2 / half_width ** 2))
```

```text
items  interval around an observed 80 percent  half-width
   50  0.670 to 0.888  0.109
  100  0.711 to 0.867  0.078
  200  0.739 to 0.850  0.055
  400  0.758 to 0.836  0.039
  800  0.771 to 0.826  0.028
items for a half-width of 0.1 : 62
items for a half-width of 0.05 : 246
items for a half-width of 0.03 : 683
```

With 50 items the interval is 0.670 to 0.888, a half-width of 0.109. With 200 it is 0.739 to 0.850 (0.055), with 800 0.771 to 0.826 (0.028). A half-width of 5 points needs 246 items and 3 points needs 683. So write the threshold together with the set size and the interval, define done on a fixed evaluation set agreed in advance, and say what result would stop or rescope the project. A reviewer should be able to disagree with a number, not with a mood.

**Chapter that teach this:** [Design docs and reviews](/docs/senior/design-docs-and-reviews).

</details>

<details>
<summary><strong>Q39.</strong> A junior reports a 0.5-point improvement from a new feature. How do you coach them?</summary>

Have them measure the noise before the effect, with code they can rerun. Train the same model with twenty different seeds, and the same model on twenty different splits, and look at the spread. Then compare the claimed gain to it.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, n_features=20, n_informative=6, flip_y=0.1, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=300, random_state=0)
by_seed = [(RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1).fit(Xtr, ytr).predict(Xte) == yte).mean() for s in range(20)]
by_split = []
for s in range(20):
    a, b, c, d = train_test_split(X, y, test_size=300, random_state=s)
    by_split.append((RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(a, c).predict(b) == d).mean())
print("20 model seeds, same split: mean", round(float(np.mean(by_seed)), 3), " std", round(float(np.std(by_seed)), 3), " range", round(min(by_seed), 3), "to", round(max(by_seed), 3))
print("20 data splits, same seed: mean", round(float(np.mean(by_split)), 3), " std", round(float(np.std(by_split)), 3))
mean = float(np.mean(by_seed))
print("binomial standard error of one 300-item test set:", round(float(np.sqrt(mean * (1 - mean) / 300)), 3))
```

```text
20 model seeds, same split: mean 0.843  std 0.008  range 0.83 to 0.853
20 data splits, same seed: mean 0.844  std 0.015
binomial standard error of one 300-item test set: 0.021
```

Twenty seeds on one split give a mean accuracy of 0.843 and a standard deviation of 0.008, ranging from 0.83 to 0.853. Twenty splits with one seed give 0.844 and a standard deviation of 0.015. The binomial standard error of a single 300-item test set is 0.021. A 0.5-point gain is smaller than every one of those, so it is not evidence. Teach the habit: fix the evaluation set, report intervals, compare paired and say what would change your mind. The best coaching shows the number rather than asserting the rule.

**Chapter that teach this:** [Technical leadership and mentoring](/docs/senior/technical-leadership-and-mentoring).

</details>

<details>
<summary><strong>Q40.</strong> How do you alert on answer quality when every request returns 200?</summary>

Grade a sample of live answers, set an error budget on the share judged bad, and choose an alert rule from its false-alarm rate and its power. The setup is 200 graded answers a day, a quality budget of 5% bad over 28 days, and a regression to 12% bad that you want to catch.

```python
from scipy.stats import binom

graded_per_day, budget_rate, regression_rate = 200, 0.05, 0.12
print("28-day budget:", 28 * graded_per_day * budget_rate, "bad answers out of", 28 * graded_per_day, "graded")
print("a regression to", regression_rate, "bad burns the budget", regression_rate / budget_rate, "times faster, so it lasts", round(28 / (regression_rate / budget_rate), 1), "days")
print("alert if daily bad count >=   false alarm per day   days between false alarms   detection at the regression rate")
for threshold in (14, 16, 18, 20):
    false_alarm = 1 - binom.cdf(threshold - 1, graded_per_day, budget_rate)
    detect = 1 - binom.cdf(threshold - 1, graded_per_day, regression_rate)
    print(f"{threshold:26d}   {false_alarm:16.4f}   {1 / false_alarm:24.1f}   {detect:33.3f}")
```

```text
28-day budget: 280.0 bad answers out of 5600 graded
a regression to 0.12 bad burns the budget 2.4 times faster, so it lasts 11.7 days
alert if daily bad count >=   false alarm per day   days between false alarms   detection at the regression rate
                        14             0.1299                        7.7                               0.993
                        16             0.0444                       22.5                               0.973
                        18             0.0121                       82.7                               0.926
                        20             0.0027                      375.3                               0.836
```

The 28-day budget allows 280 bad answers out of 5,600 graded. A regression to 12% bad burns the budget 2.4 times faster, exhausting it in 11.7 days. Alerting when a day has 16 or more bad answers fires falsely on 4.4% of days, about once every 22.5 days, and detects the regression on 97.3% of days. A threshold of 18 cuts false alarms to once every 82.7 days and detects 92.6%; 20 gives once in 375 days and 83.6%. Pick the threshold from the cost of waking someone against the cost of a slow detection, and write the postmortem question as which check would have caught it and how often it would have cried wolf.

**Chapter that teach this:** [Postmortems and on-call for ML](/docs/senior/postmortems-and-on-call-for-ml).

</details>

## What a strong answer has in common

- It names the failure mode before the technique.
- It computes a number, however rough, and says what the number leaves out.
- It reports uncertainty: an interval, a range or a percentile.
- It separates what was measured from what was assumed, and says which assumptions to replace.
- It ends with the next measurement, not with a verdict.
