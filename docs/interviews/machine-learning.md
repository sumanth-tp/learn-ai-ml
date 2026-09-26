---
title: Machine Learning Interview Questions
sidebar_position: 33
sidebar_label: "Reference · Machine Learning"
---

# Machine Learning Interview Questions

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](02-ml-statistics.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

100 essential ML interview questions with in-depth answers and code examples.

---

<details>
<summary><strong>1. What is the bias-variance tradeoff?</strong></summary>

**Answer:**
Bias measures how far predictions are from true values (underfitting); variance measures sensitivity to training data fluctuations (overfitting). The total error = bias² + variance + irreducible noise. Reducing one often increases the other.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X = np.sort(np.random.rand(50, 1) * 4, axis=0)
y = np.sin(X).ravel() + np.random.randn(50) * 0.3

for degree in [1, 4, 15]:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("lr", LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"Degree {degree}: CV MSE = {-scores.mean():.3f} +/- {scores.std():.3f}")
# Degree 1: high bias (underfit); Degree 15: high variance (overfit)
```

**Interview Tip:** Mention that regularization (L1/L2) and ensemble methods (bagging/boosting) help manage this tradeoff.

</details>

<details>
<summary><strong>2. Explain overfitting and how to prevent it.</strong></summary>

**Answer:**
Overfitting occurs when a model learns noise in training data and fails to generalize. Prevention strategies: regularization, dropout, early stopping, more data, simpler models, cross-validation.

```python
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for name, model in [("Linear", LinearRegression()), ("Ridge", Ridge(alpha=1.0)), ("Lasso", Lasso(alpha=1.0))]:
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name}: Train={train_mse:.1f}, Test={test_mse:.1f}")
```

**Interview Tip:** Distinguish overfitting causes: model complexity, insufficient data, data leakage. Mention cross-validation as a diagnosis tool.

</details>

<details>
<summary><strong>3. What is cross-validation and why use it?</strong></summary>

**Answer:**
Cross-validation splits data into k folds, trains on k-1 folds, validates on 1, repeating k times. Provides reliable performance estimate and helps tune hyperparameters without touching the test set.

```python
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(rf, X, y, cv=kf, scoring="accuracy")
print(f"K-Fold CV: {scores.mean():.3f} +/- {scores.std():.3f}")

# Stratified preserves class proportions - use for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_s = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
print(f"Stratified K-Fold: {scores_s.mean():.3f} +/- {scores_s.std():.3f}")
```

**Interview Tip:** Use StratifiedKFold for classification, TimeSeriesSplit for time-series data. Never use test set for model selection.

</details>

<details>
<summary><strong>4. What is regularization? Explain L1 vs L2.</strong></summary>

**Answer:**
Regularization adds a penalty to the loss function to discourage large weights. L1 (Lasso) adds |w| — produces sparse models by zeroing weights. L2 (Ridge) adds w² — shrinks weights evenly. ElasticNet combines both.

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y, true_coef = make_regression(n_samples=200, n_features=30, n_informative=10,
                                   noise=5, coef=True, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1).fit(X_scaled, y)
ridge = Ridge(alpha=1.0).fit(X_scaled, y)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_scaled, y)

print(f"Lasso nonzero coefs: {np.sum(lasso.coef_ != 0)}/30")   # sparse
print(f"Ridge nonzero coefs: {np.sum(ridge.coef_ != 0)}/30")   # all nonzero
print(f"ElasticNet nonzero: {np.sum(enet.coef_ != 0)}/30")
```

**Interview Tip:** L1 for feature selection (sparse), L2 for correlated features, ElasticNet for both. Always scale features before regularization.

</details>

<details>
<summary><strong>5. What is gradient descent? Compare batch, mini-batch, and stochastic.</strong></summary>

**Answer:**
Gradient descent minimizes loss by iteratively stepping in the direction of steepest descent (negative gradient). Variants differ in how many samples are used per update.

```python
import numpy as np

np.random.seed(42)
X = np.random.randn(1000, 5)
y = X @ np.array([2, -1, 0.5, 3, -2]) + np.random.randn(1000) * 0.5
X = np.hstack([np.ones((1000, 1)), X])

def mse_gradient(X, y, w):
    pred = X @ w
    return (2 / len(y)) * X.T @ (pred - y)

w = np.zeros(6)
lr = 0.01
batch_size = 32
losses = []

for epoch in range(100):
    idx = np.random.permutation(len(y))
    for i in range(0, len(y), batch_size):
        batch = idx[i:i+batch_size]
        grad = mse_gradient(X[batch], y[batch], w)
        w -= lr * grad
    loss = np.mean((X @ w - y) ** 2)
    losses.append(loss)

print(f"Final MSE: {losses[-1]:.4f}")
```

| Type | Samples/update | Speed | Noise |
|------|---------------|-------|-------|
| Batch | All | Slow | Low |
| Mini-batch | k | Medium | Medium |
| SGD | 1 | Fast | High |

**Interview Tip:** Mini-batch (32-256) is default for deep learning. Batch GD for convex problems. SGD escapes local minima better.

</details>

<details>
<summary><strong>6. What are precision, recall, F1, and when to use each?</strong></summary>

**Answer:**
- **Precision** = TP/(TP+FP): of predicted positives, how many are real? Use when false positives are costly (spam detection).
- **Recall** = TP/(TP+FN): of actual positives, how many did we catch? Use when false negatives are costly (cancer detection).
- **F1** = 2*(P*R)/(P+R): harmonic mean; use when both matter and classes are imbalanced.

```python
from sklearn.metrics import (precision_score, recall_score, f1_score, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

**Interview Tip:** Accuracy is misleading for imbalanced data. Use ROC-AUC for ranking ability, PR-AUC for imbalanced positives.

</details>

<details>
<summary><strong>7. What is ROC-AUC and how do you interpret it?</strong></summary>

**Answer:**
ROC curve plots True Positive Rate vs False Positive Rate at different thresholds. AUC ranges 0-1; 0.5 = random, 1.0 = perfect. AUC represents the probability that the model ranks a random positive higher than a random negative.

```python
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# Find optimal threshold (Youden's J)
j_scores = tpr - fpr
opt_idx = j_scores.argmax()
print(f"Optimal threshold: {thresholds[opt_idx]:.3f}")
```

**Interview Tip:** ROC-AUC is threshold-agnostic and good for balanced datasets. For imbalanced data, prefer PR-AUC. AUC is not affected by class imbalance.

</details>

<details>
<summary><strong>8. What is feature engineering?</strong></summary>

**Answer:**
Feature engineering transforms raw data into informative inputs for ML models. Includes: encoding categoricals, creating interactions, extracting datetime features, binning, log transforms, polynomial features, text/image embeddings.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": [25, 30, 22, 45, 35],
    "salary": [50000, 80000, 40000, 120000, 95000],
    "department": ["IT", "HR", "IT", "Finance", "HR"],
    "hire_date": pd.to_datetime(["2020-01-15", "2019-06-01", "2021-03-10",
                                  "2015-09-20", "2018-11-05"])
})

df["log_salary"] = np.log1p(df["salary"])
df["salary_per_age"] = df["salary"] / df["age"]
df["years_employed"] = (pd.Timestamp.now() - df["hire_date"]).dt.days / 365
df["hire_month"] = df["hire_date"].dt.month
df["hire_quarter"] = df["hire_date"].dt.quarter

ohe = pd.get_dummies(df["department"], prefix="dept")
df = pd.concat([df, ohe], axis=1)

print(df[["age", "log_salary", "salary_per_age", "years_employed"]].round(2))
```

**Interview Tip:** Domain knowledge drives good feature engineering. Mention target encoding for high-cardinality categoricals, and feature importance for selection.

</details>

<details>
<summary><strong>9. What is the difference between supervised, unsupervised, and reinforcement learning?</strong></summary>

**Answer:**
- **Supervised**: labeled examples → learn mapping (classification, regression)
- **Unsupervised**: no labels → find structure (clustering, dimensionality reduction)
- **Reinforcement**: agent-environment interaction → maximize cumulative reward

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Supervised - classification
clf = RandomForestClassifier().fit(X, y)
print("Supervised prediction:", clf.predict(X[:3]))

# Unsupervised - clustering
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
print("Cluster labels:", km.labels_[:10])

# Unsupervised - dimensionality reduction
pca = PCA(n_components=2).fit(X)
X_2d = pca.transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

**Interview Tip:** Self-supervised learning (BERT, SimCLR) uses unlabeled data to create pseudo-labels. Semi-supervised is another variant combining labeled and unlabeled data.

</details>

<details>
<summary><strong>10. How does a decision tree work?</strong></summary>

**Answer:**
Decision trees split data recursively based on features that best separate classes/reduce error. Split criteria: Gini impurity or entropy (classification), MSE (regression). Greedy top-down algorithm; prone to overfitting without pruning.

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
feat_names = list(load_iris().feature_names)

dt = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
dt.fit(X, y)

tree_text = export_text(dt, feature_names=feat_names)
print(tree_text[:500])

def gini(labels):
    n = len(labels)
    counts = {c: list(labels).count(c) for c in set(labels)}
    return 1 - sum((v/n)**2 for v in counts.values())

print(f"Root Gini (all classes): {gini(y):.3f}")
print(f"Feature importances: {dict(zip(feat_names, dt.feature_importances_.round(3)))}")
```

**Interview Tip:** Gini is faster to compute; entropy provides slightly different splits. max_depth, min_samples_split, min_samples_leaf control complexity.

</details>

<details>
<summary><strong>11. What are Random Forests? How do they work?</strong></summary>

**Answer:**
Random Forests are an ensemble of decision trees built on bootstrap samples (bagging) with random feature subsets at each split. Final prediction is majority vote (classification) or average (regression). Reduces variance without increasing bias.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",    # key for tree diversity
    bootstrap=True,
    oob_score=True,         # free validation estimate
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print(f"Test accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
print(f"OOB score: {rf.oob_score_:.3f}")

feat_names = load_breast_cancer().feature_names
top = np.argsort(rf.feature_importances_)[::-1][:5]
for i in top:
    print(f"  {feat_names[i]}: {rf.feature_importances_[i]:.3f}")
```

**Interview Tip:** OOB score is essentially free cross-validation. n_estimators: more is better but diminishing returns. Parallelizes easily with n_jobs=-1.

</details>

<details>
<summary><strong>12. What is Gradient Boosting? How does it differ from Random Forest?</strong></summary>

**Answer:**
Gradient Boosting builds trees sequentially, each fitting the residuals (pseudo-gradients) of the previous ensemble. Reduces both bias and variance iteratively. More accurate than RF but slower to train and more sensitive to hyperparameters.

```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

for name, model in [
    ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                      max_depth=3, random_state=42))
]:
    t = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: acc={acc:.3f}, time={elapsed:.2f}s")
```

| | Random Forest | Gradient Boosting |
|-|--------------|------------------|
| Trees built | Parallel | Sequential |
| Reduces | Variance | Bias + Variance |
| Speed | Faster | Slower |
| Tuning | Less sensitive | More sensitive |

**Interview Tip:** XGBoost, LightGBM, CatBoost are optimized GB implementations. Use them in practice over sklearn's GradientBoosting.

</details>

<details>
<summary><strong>13. What is XGBoost and why is it popular?</strong></summary>

**Answer:**
XGBoost (Extreme Gradient Boosting) is a regularized GB implementation with second-order gradients, tree pruning, column subsampling, and efficient sparse handling. Faster than sklearn GBM, often best performance on tabular data.

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**Interview Tip:** Key advantages: handles missing values, GPU support, early stopping. LightGBM is faster on large datasets; CatBoost handles categoricals natively.

</details>

<details>
<summary><strong>14. What is the curse of dimensionality?</strong></summary>

**Answer:**
As dimensions increase, data becomes increasingly sparse — distances become meaningless, volume grows exponentially, and more samples are needed. Affects distance-based algorithms (KNN, k-means, SVM) most severely.

```python
import numpy as np

np.random.seed(42)
for d in [2, 10, 50, 100, 500]:
    points = np.random.randn(1000, d)
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    ratio = distances.std() / distances.mean()
    print(f"Dims={d:4d}: mean_dist={distances.mean():.3f}, std/mean={ratio:.3f}")
# std/mean approaches 0: all points equidistant (concentration of measure)
```

**Solutions:**
- Dimensionality reduction: PCA, UMAP, t-SNE
- Feature selection and regularization
- Domain-specific feature engineering

**Interview Tip:** KNN suffers most. Tree methods are more robust. Mention that high dimensions also mean more overfitting risk.

</details>

<details>
<summary><strong>15. What is PCA and how does it work?</strong></summary>

**Answer:**
Principal Component Analysis finds orthogonal directions (principal components) of maximum variance via eigendecomposition of the covariance matrix. Projects data onto k components while preserving maximum variance.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA().fit(X_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 95% variance: {n_95}")

pca_k = PCA(n_components=n_95)
X_pca = pca_k.fit_transform(X_scaled)
print(f"Original: {X_scaled.shape} -> Reduced: {X_pca.shape}")

# Manual PCA
cov = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
top_k_vecs = eigenvectors[:, idx[:2]]
X_manual = X_scaled @ top_k_vecs
print(f"Manual PCA shape: {X_manual.shape}")
```

**Interview Tip:** Always standardize before PCA. PCA is unsupervised — for supervised reduction use LDA. Components are linear combinations of original features and not directly interpretable.

</details>

<details>
<summary><strong>16. What is feature selection and what are the main approaches?</strong></summary>

**Answer:**
Feature selection removes irrelevant/redundant features to improve model performance and interpretability. Three categories: filter (statistical tests), wrapper (model-based search), embedded (regularization selects features during training).

```python
from sklearn.feature_selection import (SelectKBest, f_classif, RFE,
                                        mutual_info_classif)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
feat_names = load_breast_cancer().feature_names

# Filter: ANOVA F-test
selector_f = SelectKBest(f_classif, k=10)
selector_f.fit(X, y)
print("Filter selected:", feat_names[selector_f.get_support()][:5])

# Filter: Mutual Information
mi = mutual_info_classif(X, y, random_state=42)
top_mi = np.argsort(mi)[::-1][:5]
print("Mutual info top 5:", feat_names[top_mi])

# Wrapper: Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
rfe.fit(X, y)
print("RFE selected:", feat_names[rfe.support_][:5])

# Embedded: Lasso
X_scaled = StandardScaler().fit_transform(X)
lasso = Lasso(alpha=0.01).fit(X_scaled, y)
selected = feat_names[lasso.coef_ != 0]
print(f"Lasso selected {len(selected)} features")
```

**Interview Tip:** Filter methods are fast but ignore feature interactions. Wrapper (RFE) is expensive but considers interactions. Embedded (Lasso, tree importance) is a good balance.

</details>

<details>
<summary><strong>17. What is k-Nearest Neighbors (KNN)?</strong></summary>

**Answer:**
KNN classifies new samples by majority vote of the k nearest neighbors in feature space. Non-parametric, lazy learner (no training phase), but slow at prediction time O(n*d).

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

param_grid = {"n_neighbors": range(1, 21), "weights": ["uniform", "distance"]}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
gs.fit(X_train_s, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Test accuracy: {accuracy_score(y_test, gs.predict(X_test_s)):.3f}")

# k=1: overfit, large k: underfit
for k in [1, 5, 15, 50]:
    knn = KNeighborsClassifier(k).fit(X_train_s, y_train)
    print(f"k={k:2d}: {accuracy_score(y_test, knn.predict(X_test_s)):.3f}")
```

**Interview Tip:** Always scale features. Small k = high variance, large k = high bias. Use KD-tree or ball-tree for large datasets.

</details>

<details>
<summary><strong>18. Explain Support Vector Machines (SVM).</strong></summary>

**Answer:**
SVM finds the hyperplane that maximizes the margin between classes. Support vectors are the training points closest to the hyperplane. The kernel trick maps data to higher dimensions for non-linear classification.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm.fit(X_train_s, y_train)
print(classification_report(y_test, svm.predict(X_test_s)))

param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]}
gs = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, n_jobs=-1)
gs.fit(X_train_s, y_train)
print(f"Best: {gs.best_params_}, Score: {gs.best_score_:.3f}")
```

**Kernels:** Linear (fast, good for high-d text), RBF (most popular), Polynomial, Sigmoid.

**Interview Tip:** C controls regularization — high C = less margin, low C = more regularization. SVM scales poorly with n_samples O(n^2 to n^3).

</details>

<details>
<summary><strong>19. What is logistic regression and how does it work?</strong></summary>

**Answer:**
Logistic regression applies the sigmoid function to a linear combination of features to output probabilities. Despite the name, it's a classification algorithm. Trained by maximizing log-likelihood (minimizing cross-entropy loss).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train_s = StandardScaler().fit_transform(X_train)
X_test_s = StandardScaler().fit(X_train).transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

y_prob = lr.predict_proba(X_test_s)[:, 1]
print(f"Accuracy: {lr.score(X_test_s, y_test):.3f}")
print(f"Log loss: {log_loss(y_test, y_prob):.3f}")

def sigmoid(z): return 1 / (1 + np.exp(-z))
z = X_test_s[:3] @ lr.coef_.T + lr.intercept_
print(f"Manual probs: {sigmoid(z)[:3].ravel()}")
```

**Interview Tip:** Outputs calibrated probabilities unlike SVM/RF. Multiclass via OvR or softmax. C = 1/lambda (inverse regularization). Fast and interpretable coefficients.

</details>

<details>
<summary><strong>20. What is k-means clustering?</strong></summary>

**Answer:**
K-means partitions n points into k clusters by alternating between assignment (each point to nearest centroid) and update (recalculate centroids) until convergence. Minimizes within-cluster sum of squares (inertia).

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.8, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

inertias, sil_scores = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = list(K_range)[np.argmax(sil_scores)]
print(f"Best k by silhouette: {best_k}")

km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km_best.fit(X_scaled)
print(f"Silhouette score: {silhouette_score(X_scaled, km_best.labels_):.3f}")
```

**Interview Tip:** K-means assumes spherical clusters of equal size. Use k-means++ initialization. Sensitive to outliers. DBSCAN handles arbitrary shapes.

</details>

<details>
<summary><strong>21. What is the difference between bagging and boosting?</strong></summary>

**Answer:**
**Bagging** trains models in parallel on bootstrap samples, reduces variance. **Boosting** trains models sequentially, each correcting predecessor's errors, reduces bias. Both improve over single models.

```python
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
base = DecisionTreeClassifier(max_depth=3)

models = {
    "Single DT (depth=3)": DecisionTreeClassifier(max_depth=3),
    "Single DT (full)": DecisionTreeClassifier(),
    "Bagging": BaggingClassifier(base, n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(base, n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:25s}: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**Interview Tip:** Random Forest = Bagging + feature randomization. XGBoost/LightGBM = optimized Boosting. Bagging works best with high-variance base learners (deep trees).

</details>

<details>
<summary><strong>22. What is hyperparameter tuning? Grid search vs random search vs Bayesian.</strong></summary>

**Answer:**
Hyperparameter tuning finds optimal model configuration. Grid search: exhaustive. Random search: samples randomly, often finds good solutions in fewer trials. Bayesian: uses surrogate model to guide search.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
from scipy.stats import randint
import time

X, y = load_breast_cancer(return_X_y=True)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None], "max_features": ["sqrt", "log2"]}
t = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
gs.fit(X, y)
print(f"Grid search: {gs.best_score_:.3f} in {time.time()-t:.1f}s")

param_dist = {"n_estimators": randint(50, 300), "max_depth": [3, 5, 7, None], "max_features": ["sqrt", "log2"]}
t = time.time()
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist,
                         n_iter=20, cv=5, n_jobs=-1, random_state=42)
rs.fit(X, y)
print(f"Random search: {rs.best_score_:.3f} in {time.time()-t:.1f}s")
# Optuna or scikit-optimize for Bayesian optimization
```

**Interview Tip:** Random search usually preferred over grid search. Bayesian (Optuna, Hyperopt) best for expensive models. Always use CV score for selection, never test set.

</details>

<details>
<summary><strong>23. What is data leakage and how do you prevent it?</strong></summary>

**Answer:**
Data leakage occurs when information from outside the training set influences the model, causing overly optimistic evaluation. Types: target leakage (future data as features), train-test contamination (preprocessing on full data).

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

X = np.random.randn(1000, 10)
y = (X[:, 0] + np.random.randn(1000) * 0.5 > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# WRONG: scaling entire dataset before split (leakage!)
# scaler = StandardScaler()
# X_all_scaled = scaler.fit_transform(X)  # test stats contaminate training

# CORRECT: use Pipeline - fit only on X_train inside each CV fold
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression())
])
scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"Leak-free CV: {scores.mean():.3f}")

# Common leakage examples:
# - Using "days_to_event" to predict future events
# - Encoding labels before train/test split
# - Feature derived from the target variable
```

**Interview Tip:** Pipeline prevents preprocessing leakage. In time series, always split chronologically. Feature selection inside CV is critical.

</details>

<details>
<summary><strong>24. What is class imbalance and how do you handle it?</strong></summary>

**Answer:**
Class imbalance occurs when one class dominates. Solutions: resampling (oversample minority, undersample majority), class weights, threshold adjustment, synthetic samples (SMOTE).

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print(f"Class distribution: {np.bincount(y_train)}")

# 1. Class weights
lr_w = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
print("With class weights:")
print(classification_report(y_test, lr_w.predict(X_test)))

# 2. Threshold adjustment
lr = LogisticRegression().fit(X_train, y_train)
probs = lr.predict_proba(X_test)[:, 1]
for thresh in [0.3, 0.5, 0.7]:
    preds = (probs >= thresh).astype(int)
    print(f"Threshold {thresh}: F1={f1_score(y_test, preds):.3f}")

# 3. SMOTE: from imblearn.over_sampling import SMOTE
# X_res, y_res = SMOTE().fit_resample(X_train, y_train)
```

**Interview Tip:** class_weight="balanced" is simplest fix. SMOTE creates synthetic minority samples. Always use stratified splits. Evaluate with F1/AUC, not accuracy.

</details>

<details>
<summary><strong>25. What is a confusion matrix?</strong></summary>

**Answer:**
A confusion matrix is a 2D table comparing predicted vs actual class labels, showing TP, FP, TN, FN counts. All classification metrics derive from it.

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
specificity = TN / (TN + FP)
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Specificity={specificity:.3f}")
```

**Interview Tip:** For multiclass, confusion matrix is NxN. Normalize by row (true labels) to get per-class recall. Off-diagonal elements show confusion between classes.

</details>

<details>
<summary><strong>26. What is the difference between parametric and non-parametric models?</strong></summary>

**Answer:**
Parametric models assume a fixed functional form with a set number of parameters (linear regression, logistic regression, naive Bayes). Non-parametric models make fewer assumptions and grow with data (KNN, decision trees, random forests).

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=200, n_features=5, noise=20, random_state=42)

for name, model in [
    ("Linear (parametric)", LinearRegression()),
    ("KNN (non-parametric)", KNeighborsRegressor(n_neighbors=5)),
    ("Decision Tree (non-param)", DecisionTreeRegressor(max_depth=5)),
]:
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:30s}: R2={scores.mean():.3f}")
```

**Interview Tip:** Parametric: faster inference, need less data, more interpretable but constrained. Non-parametric: flexible, need more data, risk overfitting. Neural networks are technically parametric but can have billions of parameters.

</details>

<details>
<summary><strong>27. Explain Naive Bayes classifier.</strong></summary>

**Answer:**
Naive Bayes applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class. Despite the strong assumption, works well for text classification and small datasets.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

texts = [
    "buy cheap medicine now", "limited offer free money",
    "meeting tomorrow at 3pm", "project deadline next week",
    "win lottery prize claim", "urgent bank account verify",
    "lunch at noon today", "code review please check"
]
labels = [1, 1, 0, 0, 1, 1, 0, 0]  # 1=spam, 0=ham

pipe = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("nb", MultinomialNB(alpha=1.0))  # Laplace smoothing
])
scores = cross_val_score(pipe, texts, labels, cv=4, scoring="accuracy")
print(f"Naive Bayes text CV: {scores.mean():.3f}")

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
gnb = GaussianNB()
scores = cross_val_score(gnb, X, y, cv=5)
print(f"Gaussian NB iris: {scores.mean():.3f}")
```

**Interview Tip:** Very fast, works well with high-dimensional sparse data (text). MultinomialNB for word counts, BernoulliNB for binary features, GaussianNB for continuous features.

</details>

<details>
<summary><strong>28. What is the difference between model parameters and hyperparameters?</strong></summary>

**Answer:**
Parameters are learned from training data (weights, biases, tree split thresholds). Hyperparameters are set before training and control the learning process (learning rate, n_estimators, max_depth, regularization strength).

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

lr = LinearRegression().fit(X, y)
print(f"Learned parameters (coef_): {lr.coef_.round(2)}")
print(f"Learned bias (intercept_): {lr.intercept_:.2f}")

ridge = Ridge(alpha=1.0)  # alpha is a hyperparameter
ridge.fit(X, y)
print(f"Ridge coef: {ridge.coef_.round(2)}")

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    score = cross_val_score(Ridge(alpha=alpha), X, y, cv=5).mean()
    print(f"alpha={alpha:6.2f}: CV R2={score:.3f}")
```

**Interview Tip:** Use validation set or cross-validation for hyperparameter selection; test set only for final evaluation.

</details>

<details>
<summary><strong>29. What is transfer learning?</strong></summary>

**Answer:**
Transfer learning uses a model pre-trained on one task as a starting point for a different but related task. Saves training time, requires less data. Especially powerful in NLP (BERT) and vision (ResNet).

```python
# Conceptual - true TL is in deep learning
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_digits(return_X_y=True)

# Source task: digits 0-4
mask_source = y < 5
X_source, y_source = X[mask_source], y[mask_source]
src_model = LogisticRegression(max_iter=1000).fit(X_source, y_source)

# Target task: digits 5-9 (fewer samples)
mask_target = y >= 5
X_target, y_target = X[mask_target], y[mask_target] - 5
X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2)

# PyTorch TL pattern:
# model = ResNet50(pretrained=True)
# model.fc = nn.Linear(2048, n_target_classes)  # replace head
# for param in model.parameters(): param.requires_grad = False  # freeze
# train only model.fc.parameters()

tgt_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"Target accuracy: {tgt_model.score(X_test, y_test):.3f}")
```

**Interview Tip:** Strategies: feature extraction (freeze all, retrain head), fine-tuning (unfreeze some/all layers). Domain similarity determines how much to fine-tune.

</details>

<details>
<summary><strong>30. What is the difference between classification and regression?</strong></summary>

**Answer:**
Classification predicts discrete class labels; regression predicts continuous values. Different loss functions: cross-entropy for classification, MSE/MAE for regression.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X_c, y_c = make_classification(n_samples=500, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
print(f"Classification CV accuracy: {cross_val_score(clf, X_c, y_c, cv=5).mean():.3f}")

X_r, y_r = make_regression(n_samples=500, noise=20, random_state=42)
reg = RandomForestRegressor(n_estimators=50, random_state=42)
r2 = cross_val_score(reg, X_r, y_r, cv=5, scoring="r2")
print(f"Regression CV R2: {r2.mean():.3f}")

print("\nClassification metrics: Accuracy, F1, AUC, Precision, Recall")
print("Regression metrics: MSE, RMSE, MAE, R2, MAPE")

y_true = np.array([1.0, 2.0, 3.0, 100.0])
y_pred = np.array([1.1, 2.1, 3.1, 10.0])  # one outlier
mse = np.mean((y_true - y_pred)**2)
mae = np.mean(np.abs(y_true - y_pred))
print(f"MSE={mse:.1f} (sensitive to outliers), MAE={mae:.1f}")
```

**Interview Tip:** R2 can be negative for bad models. Use RMSE for interpretability (same units as target). Consider Huber loss for regression with outliers.

</details>

<details>
<summary><strong>31. What is the train/validation/test split and why?</strong></summary>

**Answer:**
Training set: fit model parameters. Validation set: tune hyperparameters/model selection. Test set: final unbiased performance estimate. Using test set for model selection leads to optimistic estimates.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

best_lr, best_val = None, 0
for lr in [0.01, 0.05, 0.1, 0.2]:
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, random_state=42)
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    if val_acc > best_val:
        best_val, best_lr = val_acc, lr

final_model = GradientBoostingClassifier(n_estimators=100, learning_rate=best_lr, random_state=42)
final_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
print(f"Best lr={best_lr}, Test accuracy: {accuracy_score(y_test, final_model.predict(X_test)):.3f}")
```

**Interview Tip:** For small datasets, use nested CV instead. Never report test performance during development.

</details>

<details>
<summary><strong>32. What is the EM algorithm?</strong></summary>

**Answer:**
Expectation-Maximization iterates between E-step (compute expected log-likelihood given current parameters) and M-step (maximize to update parameters). Used for latent variable models like Gaussian Mixture Models.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

np.random.seed(42)
X1 = np.random.randn(100, 2) + [2, 2]
X2 = np.random.randn(100, 2) + [-2, -2]
X3 = np.random.randn(100, 2) + [2, -2]
X = np.vstack([X1, X2, X3])

gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
probs = gmm.predict_proba(X)  # soft assignments (E-step output)
print(f"Means:\n{gmm.means_.round(2)}")
print(f"Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")

for k in range(2, 6):
    g = GaussianMixture(n_components=k, random_state=42).fit(X)
    print(f"k={k}: BIC={g.bic(X):.1f}")
```

**Interview Tip:** EM guarantees non-decreasing log-likelihood but converges to local optima — use multiple restarts (n_init). GMM provides soft cluster assignments unlike K-means.

</details>

<details>
<summary><strong>33. What is a learning curve and what does it tell you?</strong></summary>

**Answer:**
Learning curves plot training and validation error vs training set size. Diagnose bias vs variance: high bias = both errors high and converging; high variance = large gap between train and val errors.

```python
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

for name, model in [
    ("Logistic (low variance)", Pipeline([("s", StandardScaler()), ("lr", LogisticRegression())])),
    ("RF (high capacity)", RandomForestClassifier(n_estimators=100, random_state=42)),
]:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy", n_jobs=-1
    )
    print(f"\n{name}:")
    print(f"  Final train: {train_scores[-1].mean():.3f}")
    print(f"  Final val:   {val_scores[-1].mean():.3f}")
    print(f"  Gap: {train_scores[-1].mean() - val_scores[-1].mean():.3f}")
```

**Interview Tip:** Large train-val gap = overfitting -> more data, regularization. Both high and converging = underfitting -> more complexity, better features.

</details>

<details>
<summary><strong>34. What is ensemble learning?</strong></summary>

**Answer:**
Ensemble learning combines multiple models to improve performance over any individual model. Strategies: bagging (parallel, reduces variance), boosting (sequential, reduces bias), stacking (meta-learner on base predictions).

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

estimators = [
    ("lr", Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])),
    ("knn", Pipeline([("s", StandardScaler()), ("knn", KNeighborsClassifier())])),
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
]

voting = VotingClassifier(estimators, voting="soft")
print(f"Voting: {cross_val_score(voting, X, y, cv=5).mean():.3f}")

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
print(f"Stacking: {cross_val_score(stack, X, y, cv=5).mean():.3f}")

for name, model in estimators:
    print(f"{name}: {cross_val_score(model, X, y, cv=5).mean():.3f}")
```

**Interview Tip:** Diversity is key — ensemble different model types. Stacking tends to outperform voting but is more complex.

</details>

<details>
<summary><strong>35. What is a pipeline in ML?</strong></summary>

**Answer:**
A Pipeline chains preprocessing and modeling steps, ensuring transformations are fit only on training data and preventing leakage. Essential for production ML and cross-validation.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200
df = pd.DataFrame({
    "age": np.random.randint(20, 70, n),
    "salary": np.random.randint(30000, 150000, n).astype(float),
    "department": np.random.choice(["IT", "HR", "Finance"], n),
})
df.loc[np.random.choice(n, 20), "salary"] = np.nan
y = (df["salary"].fillna(df["salary"].median()) > 80000).astype(int)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), ["age", "salary"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["department"])
])

pipe = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
scores = cross_val_score(pipe, df, y, cv=5, scoring="accuracy")
print(f"Pipeline CV: {scores.mean():.3f} +/- {scores.std():.3f}")
```

**Interview Tip:** Pipelines are essential for production — single object for fit/transform/predict. Always use pipelines in cross-validation to prevent leakage.

</details>

<details>
<summary><strong>36. What is the difference between generative and discriminative models?</strong></summary>

**Answer:**
**Generative** models learn joint distribution P(X,Y) and can generate samples. **Discriminative** models learn P(Y|X) directly and are typically better at classification.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture
import numpy as np

X, y = load_iris(return_X_y=True)

lr = LogisticRegression(max_iter=1000)
print(f"Logistic (discriminative): {cross_val_score(lr, X, y, cv=5).mean():.3f}")

gnb = GaussianNB()
print(f"Naive Bayes (generative): {cross_val_score(gnb, X, y, cv=5).mean():.3f}")

gnb.fit(X, y)
print("Class priors:", gnb.class_prior_)

# Generative: anomaly detection via density
gmm = GaussianMixture(n_components=3).fit(X)
log_probs = gmm.score_samples(X)
anomalies = X[log_probs < np.percentile(log_probs, 5)]
print(f"Anomalies detected: {len(anomalies)}")
```

**Interview Tip:** Generative: Naive Bayes, GMM, VAE, GAN. Discriminative: Logistic regression, SVM, Neural nets. Discriminative usually better for classification; generative for density estimation.

</details>

<details>
<summary><strong>37. What is dimensionality reduction? PCA vs t-SNE vs UMAP.</strong></summary>

**Answer:**
PCA: linear, fast, preserves global variance. t-SNE: non-linear, preserves local structure, good for visualization, slow. UMAP: faster than t-SNE, preserves more global structure.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

X_pca50 = PCA(n_components=50).fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca50)
print(f"t-SNE shape: {X_tsne.shape}")

for name, X_red in [("Raw", X_scaled), ("PCA-2", X_pca), ("PCA-50", X_pca50)]:
    score = cross_val_score(LogisticRegression(max_iter=1000), X_red, y, cv=5).mean()
    print(f"{name}: {score:.3f}")
```

**Interview Tip:** Use PCA for preprocessing/noise reduction. t-SNE/UMAP for visualization only — don't use as features (non-deterministic). UMAP is faster for larger datasets.

</details>

<details>
<summary><strong>38. What is DBSCAN?</strong></summary>

**Answer:**
DBSCAN groups points in high-density regions, marking low-density points as outliers. Doesn't require specifying k, handles arbitrary shapes, identifies noise.

```python
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
print(f"KMeans silhouette: {silhouette_score(X, km.labels_):.3f}")

print("\nEps sensitivity:")
for eps in [0.1, 0.3, 0.5, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n = len(set(db)) - (1 if -1 in db else 0)
    noise = (db == -1).sum()
    print(f"  eps={eps}: {n} clusters, {noise} noise")
```

**Interview Tip:** Use k-distance plot to choose eps. DBSCAN struggles with varying density. HDBSCAN handles varying density better.

</details>

<details>
<summary><strong>39. What is hierarchical clustering?</strong></summary>

**Answer:**
Hierarchical clustering builds a tree (dendrogram) by successively merging (agglomerative) or splitting (divisive) clusters. No need to specify k upfront.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

for linkage_type in ["ward", "complete", "average", "single"]:
    agg = AgglomerativeClustering(n_clusters=3, linkage=linkage_type)
    labels = agg.fit_predict(X_scaled)
    ari = adjusted_rand_score(y, labels)
    print(f"Linkage={linkage_type}: ARI={ari:.3f}")

Z = linkage(X_scaled, method="ward")
print(f"\nLinkage matrix shape: {Z.shape}")
print(f"Last merge distances: {Z[-5:, 2].round(2)}")
```

**Interview Tip:** Ward minimizes within-cluster variance (most popular). Complete/average are robust to outliers. Single linkage creates elongated clusters. Use dendrogram to choose optimal k.

</details>

<details>
<summary><strong>40. What are evaluation metrics for clustering?</strong></summary>

**Answer:**
With ground truth: ARI, NMI. Without ground truth: Silhouette score, Davies-Bouldin index, Calinski-Harabasz.

```python
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
labels = km.labels_

print(f"ARI:               {adjusted_rand_score(y, labels):.3f}")  # with truth
print(f"NMI:               {normalized_mutual_info_score(y, labels):.3f}")
print(f"Silhouette:        {silhouette_score(X_scaled, labels):.3f}")     # no truth
print(f"Davies-Bouldin:    {davies_bouldin_score(X_scaled, labels):.3f}") # lower better
print(f"Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels):.1f}")

scores = {}
for k in range(2, 8):
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    scores[k] = silhouette_score(X_scaled, km_k.labels_)
print(f"\nBest k by silhouette: {max(scores, key=scores.get)}")
```

**Interview Tip:** ARI is the gold standard when labels are available. Silhouette is the most common unsupervised metric.

</details>

<details>
<summary><strong>41. What is anomaly detection?</strong></summary>

**Answer:**
Anomaly detection identifies rare observations deviating significantly from expected patterns. Applications: fraud detection, equipment failure, network intrusion. Approaches: statistical, isolation-based, density-based, reconstruction-based.

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
import numpy as np

np.random.seed(42)
X_normal = np.random.randn(300, 2)
X_outliers = np.random.uniform(-8, 8, (30, 2))
X = np.vstack([X_normal, X_outliers])
y_true = np.hstack([np.ones(300), -np.ones(30)])

for name, detector in [
    ("Isolation Forest", IsolationForest(contamination=0.1, random_state=42)),
    ("One-Class SVM", OneClassSVM(nu=0.1)),
    ("Elliptic Envelope", EllipticEnvelope(contamination=0.1)),
    ("LOF", LocalOutlierFactor(contamination=0.1)),
]:
    pred = detector.fit_predict(X)
    f1 = f1_score(y_true == -1, pred == -1)
    print(f"{name:20s}: F1={f1:.3f}, detected={(pred==-1).sum()}")
```

**Interview Tip:** Isolation Forest is fast and works in high dimensions. LOF for density-based. Autoencoders for complex/image data. Contamination = expected fraction of outliers.

</details>

<details>
<summary><strong>42. What is model interpretability? SHAP and LIME.</strong></summary>

**Answer:**
Interpretability explains why a model makes predictions. SHAP (Shapley values) provides theoretically grounded attribution. LIME approximates locally with linear models.

```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
feat_names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:50])

mean_shap = np.abs(shap_values[1]).mean(axis=0)
top_features = np.argsort(mean_shap)[::-1][:5]
print("Top 5 SHAP features:")
for i in top_features:
    print(f"  {feat_names[i]}: {mean_shap[i]:.4f}")

sample_shap = shap_values[1][0]
top_local = np.argsort(np.abs(sample_shap))[::-1][:3]
print("\nLocal explanation for sample 0:")
for i in top_local:
    print(f"  {feat_names[i]}: {sample_shap[i]:+.4f}")
```

**Interview Tip:** SHAP is preferred over LIME for consistency and theoretical guarantees. TreeExplainer is fast for tree models.

</details>

<details>
<summary><strong>43. What is the difference between online and batch learning?</strong></summary>

**Answer:**
Batch learning trains on entire dataset at once, requires retraining when new data arrives. Online learning updates incrementally with each sample or mini-batch, adapts to concept drift, suitable for streaming data.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=2000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

sgd = SGDClassifier(loss="log_loss", random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

batch_size = 50
accuracies = []
for i in range(0, len(X_train_s), batch_size):
    X_batch = X_train_s[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    sgd.partial_fit(X_batch, y_batch, classes=[0, 1])  # online update
    if i > 0:
        acc = accuracy_score(y_test, sgd.predict(X_test_s))
        accuracies.append(acc)

print(f"Final online accuracy: {accuracies[-1]:.3f}")
```

**Interview Tip:** SGDClassifier/Regressor support partial_fit for online learning. River library is dedicated to online learning. Concept drift: online learning adapts automatically.

</details>

<details>
<summary><strong>44. What is semi-supervised learning?</strong></summary>

**Answer:**
Semi-supervised learning combines a small labeled dataset with a large unlabeled dataset to improve learning. Assumes unlabeled data reveals useful structure.

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

np.random.seed(42)
labeled_idx = np.random.choice(len(X), size=30, replace=False)
y_partial = np.full_like(y, fill_value=-1)  # -1 = unlabeled
y_partial[labeled_idx] = y[labeled_idx]

lr = LogisticRegression().fit(X[labeled_idx], y[labeled_idx])
print(f"Supervised (30 labels): {accuracy_score(y, lr.predict(X)):.3f}")

lp = LabelPropagation(kernel="rbf", gamma=20, max_iter=1000)
lp.fit(X, y_partial)
print(f"Label Propagation: {accuracy_score(y, lp.predict(X)):.3f}")

lr_all = LogisticRegression().fit(X, y)
print(f"Supervised (all): {accuracy_score(y, lr_all.predict(X)):.3f}")
```

**Interview Tip:** Self-training: train on labeled, predict unlabeled, add confident predictions, repeat. BERT pretraining is semi-supervised.

</details>

<details>
<summary><strong>45. What is multi-label classification?</strong></summary>

**Answer:**
Multi-label classification assigns multiple labels to each instance (unlike multi-class where one label per instance). Example: a news article tagged with both "politics" AND "economy".

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.randn(500, 20)
z = X[:, :4]
y = (z > np.percentile(z, 40, axis=0)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ovr = OneVsRestClassifier(LogisticRegression())
ovr.fit(X_train, y_train)
y_pred = ovr.predict(X_test)

mo = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
mo.fit(X_train, y_train)
y_pred_rf = mo.predict(X_test)

print(f"Hamming Loss (LR): {hamming_loss(y_test, y_pred):.3f}")
print(f"F1 micro (LR): {f1_score(y_test, y_pred, average='micro'):.3f}")
print(f"F1 macro (LR): {f1_score(y_test, y_pred, average='macro'):.3f}")
```

**Interview Tip:** Hamming loss = fraction of wrong labels. Micro-F1 aggregates globally, macro-F1 averages per label. Classifier chains can model label dependencies.

</details>

<details>
<summary><strong>46. What is the difference between OvR and OvO for multiclass?</strong></summary>

**Answer:**
OvR (One-vs-Rest): train k binary classifiers, one per class vs all others. OvO (One-vs-One): train k*(k-1)/2 classifiers, one per pair. OvR is usually preferred (faster). OvO can be better for SVMs.

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)

lr_multi = LogisticRegression(multi_class="multinomial", max_iter=1000)
print(f"Multinomial: {cross_val_score(lr_multi, X, y, cv=5).mean():.3f}")

ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
print(f"OvR: {cross_val_score(ovr, X, y, cv=5).mean():.3f}")
print(f"OvO: {cross_val_score(ovo, X, y, cv=5).mean():.3f}")

print(f"\nFor 10 classes: OvR needs 10, OvO needs {10*9//2} classifiers")
```

**Interview Tip:** Most sklearn classifiers handle multiclass natively. Softmax is the standard for multiclass neural networks.

</details>

<details>
<summary><strong>47. What is model calibration?</strong></summary>

**Answer:**
A calibrated classifier outputs probabilities that match observed frequencies. If it says P=0.8, ~80% of those cases should be positive. Random forests are often poorly calibrated; logistic regression is usually well-calibrated.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=2000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42),
                                 method="isotonic", cv=3)
rf_cal.fit(X_train, y_train)
lr = LogisticRegression().fit(X_train, y_train)

for name, model in [("RF uncalibrated", rf), ("RF calibrated", rf_cal), ("Logistic", lr)]:
    probs = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    cal_error = np.mean(np.abs(frac_pos - mean_pred))
    print(f"{name:20s}: Cal error={cal_error:.3f}")
```

**Interview Tip:** Use Platt scaling (sigmoid) or isotonic regression for calibration. Calibration matters when probabilities drive decisions (medical, finance).

</details>

<details>
<summary><strong>48. What is concept drift?</strong></summary>

**Answer:**
Concept drift occurs when statistical properties of the target variable change over time, degrading model performance. Types: sudden, gradual, recurring. Requires monitoring and model retraining.

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

def generate_data(n, mean_shift=0):
    X = np.random.randn(n, 5) + mean_shift
    y = (X[:, 0] + X[:, 1] > 0 + mean_shift).astype(int)
    return X, y

X_pre, y_pre = generate_data(500, mean_shift=0)
X_post, y_post = generate_data(500, mean_shift=2.0)  # drift!

model = SGDClassifier(loss="log_loss", random_state=42)
model.fit(X_pre, y_pre)

print(f"Pre-drift accuracy:  {accuracy_score(y_pre, model.predict(X_pre)):.3f}")
print(f"Post-drift accuracy: {accuracy_score(y_post, model.predict(X_post)):.3f}")  # degraded

adaptive = SGDClassifier(loss="log_loss", random_state=42)
window_size = 100
for i in range(0, 500, window_size):
    adaptive.partial_fit(X_post[i:i+window_size], y_post[i:i+window_size], classes=[0,1])

print(f"Adaptive post-drift: {accuracy_score(y_post[-100:], adaptive.predict(X_post[-100:])):.3f}")
```

**Interview Tip:** Detect with accuracy drop, population stability index (PSI), or KS test on features. Solutions: periodic retraining, online learning, time-weighted ensemble.

</details>

<details>
<summary><strong>49. What is A/B testing in ML?</strong></summary>

**Answer:**
A/B testing compares two model versions using randomized traffic splitting. Measure business metrics with statistical significance before full rollout.

```python
from scipy import stats
import numpy as np

np.random.seed(42)
n_A, n_B = 1000, 1000

clicks_A = np.random.binomial(1, 0.05, n_A)   # baseline: 5% CTR
clicks_B = np.random.binomial(1, 0.055, n_B)  # new model: 5.5% CTR

from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
count = np.array([clicks_B.sum(), clicks_A.sum()])
nobs = np.array([n_B, n_A])

stat, p_value = proportions_ztest(count, nobs)
print(f"Model A CTR: {clicks_A.mean():.3f}")
print(f"Model B CTR: {clicks_B.mean():.3f}")
print(f"Lift: {(clicks_B.mean() - clicks_A.mean())/clicks_A.mean():.1%}")
print(f"p-value: {p_value:.3f}, Significant: {p_value < 0.05}")

from statsmodels.stats.power import NormalIndPower
effect_size = proportion_effectsize(0.055, 0.05)
n_required = NormalIndPower().solve_power(effect_size, power=0.8, alpha=0.05)
print(f"Required sample size: {int(n_required)} per group")
```

**Interview Tip:** Guard against novelty effect. Use multi-armed bandit for more efficient exploration. Always set sample size before experiment via power analysis.

</details>

<details>
<summary><strong>50. What is AutoML?</strong></summary>

**Answer:**
AutoML automates the ML pipeline: feature engineering, model selection, hyperparameter tuning, and ensembling. Tools: Auto-sklearn, H2O AutoML, TPOT, AutoGluon, Google AutoML.

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
results = {}

models = {
    "LR": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
    "SVM": Pipeline([("s", StandardScaler()), ("m", SVC(probability=True))]),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    score = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
    results[name] = score
    print(f"{name}: AUC={score:.3f}")

best = max(results, key=results.get)
print(f"\nAutoML winner: {best} with AUC={results[best]:.3f}")
```

**Interview Tip:** AutoML democratizes ML but still needs domain expertise for feature engineering and problem framing. Avoid data leakage pitfalls in automated pipelines.

</details>

<details>
<summary><strong>51. What is the No Free Lunch theorem?</strong></summary>

No single algorithm works best for all problems. Every algorithm has inductive biases suited to certain data distributions. Empirically evaluate multiple models on your specific problem.
</details>

<details>
<summary><strong>52. What is Occam's Razor in ML?</strong></summary>

Prefer simpler models that explain the data equally well. Simpler models generalize better and are more interpretable. Use the least complex model that achieves required performance.
</details>

<details>
<summary><strong>53. What is the difference between L1 and L2 loss?</strong></summary>

L1 (MAE): robust to outliers, non-differentiable at 0. L2 (MSE): sensitive to outliers, smooth gradient. Use L1 when outliers are expected; L2 for clean data.
</details>

<details>
<summary><strong>54. What is MAPE?</strong></summary>

MAPE = mean(|actual - predicted| / |actual|) x 100%. Interpretable as percentage error. Undefined when actual=0. Use SMAPE for symmetric version.
</details>

<details>
<summary><strong>55. What is R-squared?</strong></summary>

R2 = 1 - SS_res/SS_tot. Proportion of variance explained. Range (-inf, 1]. R2=1: perfect, R2=0: same as predicting mean. Adjusted R2 penalizes number of features.
</details>

<details>
<summary><strong>56. What is multicollinearity?</strong></summary>

High correlation between features causes unstable coefficient estimates in linear models. Detect with VIF (Variance Inflation Factor). Fix by removing features, PCA, or Ridge regression.
</details>

<details>
<summary><strong>57. What is heteroscedasticity?</strong></summary>

Variance of residuals changes with fitted values (non-constant variance). Detect with residuals-vs-fitted plot. Fix: log/sqrt transform target, weighted regression, or robust standard errors.
</details>

<details>
<summary><strong>58. What is the difference between Type I and Type II errors?</strong></summary>

Type I (false positive): reject H0 when true, probability = alpha. Type II (false negative): fail to reject H0 when false, probability = beta. Power = 1 - beta.
</details>

<details>
<summary><strong>59. What is maximum likelihood estimation (MLE)?</strong></summary>

MLE finds parameters that maximize the probability of observing the training data. For Gaussian noise, MLE reduces to MSE minimization. For classification, MLE with cross-entropy.

```python
import numpy as np
from scipy.optimize import minimize

data = np.random.normal(5, 2, 1000)

def neg_log_likelihood(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    return 0.5 * len(data) * np.log(2*np.pi) + len(data)*log_sigma + \
           0.5 * np.sum((data - mu)**2) / sigma**2

result = minimize(neg_log_likelihood, x0=[0, 0])
mu_mle, sigma_mle = result.x[0], np.exp(result.x[1])
print(f"MLE: mu={mu_mle:.3f}, sigma={sigma_mle:.3f}")
```
</details>

<details>
<summary><strong>60. What is MAP estimation vs MLE?</strong></summary>

Maximum A Posteriori (MAP) = MLE + prior. MAP with Gaussian prior = L2 regularization (Ridge). MAP with Laplace prior = L1 regularization (Lasso). MLE is a special case with uniform prior.
</details>

<details>
<summary><strong>61. What is the kernel trick?</strong></summary>

Implicitly computes inner products in high-dimensional feature space without explicit mapping. Enables SVM to learn non-linear boundaries efficiently using functions like RBF, polynomial.
</details>

<details>
<summary><strong>62. What are support vectors in SVM?</strong></summary>

Support vectors are training points closest to the decision boundary. Only these points affect the decision boundary — the model is sparse in terms of training data used.
</details>

<details>
<summary><strong>63. What is the margin in SVM?</strong></summary>

The margin is the distance between the decision boundary and the closest data points. SVM maximizes this margin. Soft-margin SVM allows misclassifications controlled by C.
</details>

<details>
<summary><strong>64. Hard margin vs soft margin SVM?</strong></summary>

Hard margin: no misclassifications allowed (only for linearly separable data). Soft margin (C-SVM): allows slack variables epsilon >= 0 for misclassifications, controlled by penalty C.
</details>

<details>
<summary><strong>65. What is kernel selection for SVM?</strong></summary>

Linear kernel: high-dimensional data (text), separable data. RBF: default choice, most problems. Polynomial: specific non-linear structures. Use cross-validation to select.
</details>

<details>
<summary><strong>66. What is CART?</strong></summary>

Classification and Regression Trees: binary splits using Gini/entropy (classification) or MSE (regression). Greedy top-down construction, post-pruning (cost-complexity) to generalize.
</details>

<details>
<summary><strong>67. What is information gain and entropy?</strong></summary>

Entropy = -sum(pi * log2(pi)). Information gain = entropy(parent) - weighted_avg_entropy(children). Decision trees maximize information gain at each split.

```python
import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-10))

y = np.array([0,0,1,1,1,0,1,0])
print(f"Entropy: {entropy(y):.3f}")  # 1.0 = maximum uncertainty
```
</details>

<details>
<summary><strong>68. What is pruning in decision trees?</strong></summary>

Pruning removes branches that add little predictive power. Pre-pruning: stop during construction (max_depth, min_samples). Post-pruning: grow full tree then cut back (ccp_alpha in sklearn).
</details>

<details>
<summary><strong>69. What is the difference between AdaBoost and Gradient Boosting?</strong></summary>

AdaBoost: adjusts sample weights for misclassified samples, uses decision stumps. Gradient Boosting: fits trees to pseudo-gradients, more general framework. GBM typically outperforms AdaBoost.
</details>

<details>
<summary><strong>70. What is learning rate in boosting?</strong></summary>

Shrinkage parameter scaling each tree's contribution. Small learning rate + more trees is generally better but slower. Trade-off with n_estimators.
</details>

<details>
<summary><strong>71. What is early stopping in boosting?</strong></summary>

Stop adding trees when validation loss stops improving. Prevents overfitting. Available in XGBoost, LightGBM with early_stopping_rounds parameter.
</details>

<details>
<summary><strong>72. What is feature importance in tree models?</strong></summary>

Impurity-based: sum of impurity decrease weighted by samples (fast, biased toward high-cardinality). Permutation importance: decrease in score when feature shuffled (unbiased). SHAP is most reliable.
</details>

<details>
<summary><strong>73. How does bagging reduce variance?</strong></summary>

Each tree trained on a bootstrap sample (sampling with replacement). Different training sets lead to different trees, and averaging reduces variance. OOB samples provide free validation.
</details>

<details>
<summary><strong>74. Feature importance vs SHAP values?</strong></summary>

Feature importance: global, biased toward high-cardinality, no direction. SHAP: local (per-prediction), signed (direction), unbiased, additive, satisfies consistency axioms.
</details>

<details>
<summary><strong>75. What is a validation curve?</strong></summary>

Plots training and validation score vs a hyperparameter (max_depth, C). Shows underfitting/overfitting relationship, helps choose optimal hyperparameter value.

```python
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
param_range = [1, 2, 3, 5, 7, 10, 15]
train_s, val_s = validation_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y, param_name="max_depth", param_range=param_range, cv=5
)
for d, tr, va in zip(param_range, train_s.mean(axis=1), val_s.mean(axis=1)):
    print(f"depth={d}: train={tr:.3f}, val={va:.3f}")
```
</details>

<details>
<summary><strong>76. What is Shapley value?</strong></summary>

Shapley values fairly distribute a prediction among features using coalition game theory. Each feature's contribution = average marginal contribution across all feature subsets. SHAP implements this efficiently.
</details>

<details>
<summary><strong>77. L-BFGS vs SGD optimizers?</strong></summary>

L-BFGS: second-order quasi-Newton, works well for small datasets, converges faster. SGD: first-order, scales to large data, noisy updates help escape local minima.
</details>

<details>
<summary><strong>78. What is isotonic regression?</strong></summary>

Non-parametric regression fitting a monotonically increasing step function. Used for probability calibration (isotonic calibration). Also for modeling known monotone relationships.
</details>

<details>
<summary><strong>79. What is the Johnson-Lindenstrauss lemma?</strong></summary>

Random projections to O(log n / eps^2) dimensions approximately preserve pairwise distances with distortion eps. Foundation for random projection dimensionality reduction and locality-sensitive hashing.
</details>

<details>
<summary><strong>80. Frequentist vs Bayesian ML?</strong></summary>

Frequentist: parameters are fixed unknowns, estimate via MLE. Bayesian: parameters have prior distributions, update with data to get posterior. Bayesian provides uncertainty estimates but is computationally expensive.
</details>

<details>
<summary><strong>81. What is a Gaussian Process (GP)?</strong></summary>

Non-parametric Bayesian model defining a distribution over functions. Provides uncertainty estimates naturally. Used for surrogate modeling in Bayesian optimization. O(n^3) scaling limits large datasets.
</details>

<details>
<summary><strong>82. What is Bayesian optimization?</strong></summary>

Sequential model-based hyperparameter optimization. Uses GP (or other surrogate) to model objective, acquisition function to balance exploration/exploitation. More efficient than random search for expensive objectives.
</details>

<details>
<summary><strong>83. What is cross-entropy loss?</strong></summary>

CE = -sum(yi * log(p_hat_i)). Binary: BCE = -(y*log(p) + (1-y)*log(1-p)). Equivalent to MLE with Bernoulli/categorical distribution. Standard classification loss in neural networks.
</details>

<details>
<summary><strong>84. What is label smoothing?</strong></summary>

Replaces hard 0/1 labels with epsilon/(K-1) and 1-epsilon, preventing overconfidence. Regularization technique that improves calibration. Common in image classification (epsilon=0.1).
</details>

<details>
<summary><strong>85. What is mixup data augmentation?</strong></summary>

Creates synthetic training samples by linearly interpolating between two random examples and their labels. Lambda ~ Beta(alpha, alpha), x_mix = lambda*x1 + (1-lambda)*x2. Improves generalization for neural networks.
</details>

<details>
<summary><strong>86. Precision@k vs NDCG?</strong></summary>

Precision@k: fraction of top-k results that are relevant (binary relevance). NDCG considers ranking position and graded relevance. NDCG is standard for search/recommendation ranking.
</details>

<details>
<summary><strong>87. What is mean average precision (mAP)?</strong></summary>

mAP = mean of average precision across all queries/classes. Average precision = area under precision-recall curve. Standard metric for object detection (mAP@IoU) and information retrieval.
</details>

<details>
<summary><strong>88. What is micro vs macro averaging?</strong></summary>

Micro: aggregate TP/FP/FN across all classes (dominated by frequent classes). Macro: average metric per class equally (sensitive to rare classes). Weighted: weight by class support.
</details>

<details>
<summary><strong>89. What is the silhouette score?</strong></summary>

s(i) = (b(i) - a(i)) / max(a(i), b(i)) where a = intra-cluster distance, b = nearest-cluster distance. Range [-1, 1]; high = well-clustered.
</details>

<details>
<summary><strong>90. What is the elbow method for k-means?</strong></summary>

Plot inertia vs k. The "elbow" where decrease slows suggests optimal k. Subjective — use with silhouette score for confirmation.
</details>

<details>
<summary><strong>91. What is matrix factorization for recommendations?</strong></summary>

Decompose R ~= UV^T where R is user-item rating matrix, U is user factors, V is item factors. Solve via ALS, SGD, or SVD. K = latent factor dimension (hyperparameter).
</details>

<details>
<summary><strong>92. What are word embeddings?</strong></summary>

Dense vector representations of words learned from context (Word2Vec, GloVe). Similar words have similar vectors, support arithmetic (king - man + woman = queen). Foundation for NLP feature engineering.
</details>

<details>
<summary><strong>93. What is TF-IDF?</strong></summary>

Term Frequency-Inverse Document Frequency: TF(t,d) x IDF(t). IDF = log(N / df(t)) downweights common words, upweights rare informative words. Standard text representation before embeddings.
</details>

<details>
<summary><strong>94. Content-based vs collaborative filtering?</strong></summary>

Content-based: recommend based on item/user attributes. Collaborative: exploit user behavior patterns. Hybrid combines both. Content-based handles cold start better.
</details>

<details>
<summary><strong>95. What is matrix factorization for recommendation?</strong></summary>

Decompose user-item rating matrix R ~= UV^T. Solved via ALS or SGD. K = latent factor dimension. Captures hidden user preferences and item characteristics.
</details>

<details>
<summary><strong>96. Cosine similarity vs Euclidean distance?</strong></summary>

Cosine: angle between vectors, magnitude-invariant (good for text/embeddings). Euclidean: straight-line distance, sensitive to magnitude. For normalized vectors, cosine ~= 1 - Euclidean^2/2.
</details>

<details>
<summary><strong>97. What is out-of-bag (OOB) error in Random Forest?</strong></summary>

For each tree, samples not in its bootstrap sample (~37%) serve as validation. OOB error = average error on these samples. Approximately equals leave-one-out CV without extra computation.
</details>

<details>
<summary><strong>98. What is target encoding?</strong></summary>

Replace categorical value with mean of target for that category. Risk: target leakage if not done carefully — use mean from training fold only (or add noise). Effective for high-cardinality categoricals.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def target_encode_cv(df, cat_col, target_col, n_splits=5):
    df = df.copy()
    df[f"{cat_col}_encoded"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(df):
        encoding = df.iloc[tr_idx].groupby(cat_col)[target_col].mean()
        df.loc[df.index[val_idx], f"{cat_col}_encoded"] = df.iloc[val_idx][cat_col].map(encoding)
    return df

df = pd.DataFrame({"dept": ["IT","HR","IT","Finance","HR","IT"], "target": [1,0,1,1,0,0]})
df = target_encode_cv(df, "dept", "target")
print(df)
```
</details>

<details>
<summary><strong>99. What is feature hashing (hashing trick)?</strong></summary>

Maps high-cardinality or infinite feature spaces to fixed-size vectors using hash functions. Avoids storing vocabulary, handles new features at test time. Memory-efficient for text/sparse features.
</details>

<details>
<summary><strong>100. Model-based vs memory-based collaborative filtering?</strong></summary>

Memory-based (user-user, item-item): uses similarity between users/items directly. Model-based (matrix factorization, neural CF): learns latent representations. Model-based scales better and handles sparsity better.
</details>

---

## Quick Reference

| Algorithm | Type | Key Hyperparameters | Strengths | Weaknesses |
|-----------|------|---------------------|-----------|------------|
| Linear/Logistic Regression | Parametric | C, alpha | Fast, interpretable | Assumes linearity |
| Decision Tree | Non-parametric | max_depth, min_samples | Interpretable | Overfits |
| Random Forest | Ensemble (Bagging) | n_estimators, max_features | Robust, fast | Less interpretable |
| XGBoost/GBM | Ensemble (Boosting) | n_estimators, lr, max_depth | Best accuracy | Many hyperparams |
| SVM | Kernel | C, gamma, kernel | High-dim, small data | Slow on large n |
| KNN | Instance-based | k, metric | Simple, no training | Slow prediction |
| Naive Bayes | Probabilistic | alpha | Fast, text data | Strong assumptions |
| K-Means | Clustering | k, n_init | Fast, scalable | Spherical clusters only |
| DBSCAN | Density-based | eps, min_samples | Arbitrary shapes, detects noise | Sensitive to eps |
| PCA | Linear dim reduction | n_components | Fast, interpretable | Linear only |

## Summary in simple points

- Define the prediction task, baseline and business error costs before selecting an algorithm. Supervised, unsupervised, semi-supervised and reinforcement learning use different feedback.
- Separate training, tuning and final testing with time/entity-aware splits. Keep feature engineering and target encoding inside the training boundary.
- Bias, variance, regularisation, learning curves and validation curves help diagnose generalisation failures; none replace a representative holdout.
- Gradient descent, SGD, mini-batches and quasi-Newton methods have different optimisation and memory behaviour.
- Linear/logistic models, Naive Bayes, nearest neighbours and SVMs rely on different representations and assumptions. Kernels and margin controls change SVM behaviour.
- Trees choose impurity-reducing splits; pruning controls complexity. Bagging, random forests, boosting and stacking combine learners differently.
- Tune hyperparameters with a fixed evaluation protocol. Grid, random and Bayesian searches trade off exploration cost and assumptions.
- Use confusion matrices, precision/recall, PR/ROC curves and macro/micro averaging for the actual class mix. Accuracy alone can hide failures.
- Calibration concerns probability reliability; ranking concerns order. Isotonic or other calibration needs suitable separate data.
- Choose regression loss and metrics for outliers, zeros and asymmetric costs; R-squared is not a universal measure of usefulness.
- PCA, random projections and nonlinear embeddings preserve different properties. Feature selection and feature hashing alter information and interpretability.
- K-means, mixture/EM, density and hierarchical clustering optimise different objectives. Silhouette and elbow plots need domain validation.
- Detect anomalies by relevant segment, distinguish data/concept drift, and use controlled online experiments to measure product effects.
- Interpretation methods, tree importances and Shapley-based explanations describe model behaviour under assumptions; they do not automatically prove causality.
- MLE, MAP, Bayesian models and Gaussian processes express different estimation/uncertainty assumptions. No model wins on every possible problem.
- Embeddings, TF-IDF, cosine distance, recommendation factorisation and collaborative/content approaches need task-specific labels and ranking metrics.
- Transfer learning, augmentation, label smoothing, multi-label/multiclass strategies and online updates require held-out checks for the intended deployment distribution.
