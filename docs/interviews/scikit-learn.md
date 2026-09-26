---
title: Scikit-learn Interview Questions (100)
sidebar_position: 39
sidebar_label: "Reference · Scikit Learn"
---

# Scikit-learn Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](02-ml-statistics.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

## Core Scikit-learn Concepts

<details>
<summary><strong>1. What is Scikit-learn?</strong></summary>

**Answer:**
Python's standard ML library with unified API for classification, regression, clustering, and preprocessing.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
```

**Key components**:
- **Estimators**: `fit()` and `predict()` objects
- **Transformers**: `fit()` and `transform()` for preprocessing
- **Pipelines**: Chain estimators
- **Cross-validation**: Evaluate model
- **Metrics**: Assess quality

**Interview Tip**: Show comfortable use of complete ML pipeline.
</details>

<details>
<summary><strong>2. What is the unified API?</strong></summary>

**Answer:**
All estimators follow consistent interface.

```python
# All estimators follow this pattern:

# 1. Estimator
estimator = ModelClass(hyperparameters)

# 2. Fit
estimator.fit(X_train, y_train)

# 3. Predict
predictions = estimator.predict(X_test)

# 4. Score
score = estimator.score(X_test, y_test)

# Examples of consistent API:

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# All have same interface
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
]

for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{model.__class__.__name__}: {score:.3f}")

# Transformers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

transformers = [StandardScaler(), MinMaxScaler()]

for transformer in transformers:
    X_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
```

**Interview Tip**: Leverage consistency to switch models easily.
</details>

<details>
<summary><strong>3. What are pipelines?</strong></summary>

**Answer:**
Chain preprocessing and model steps together.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict (automatically applies all steps)
predictions = pipeline.predict(X_test)

# Score
score = pipeline.score(X_test, y_test)

# Cross-validate entire pipeline
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV Scores: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Access individual steps
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

# Feature scaling happens on train set only!
# This prevents data leakage
```

**Benefits**:
- Prevents data leakage (scaling happens on train only)
- Simple to switch models
- Easy to save/load complete pipeline

**Interview Tip**: Emphasize data leakage prevention with pipelines.
</details>

<details>
<summary><strong>4. What is cross-validation?</strong></summary>

**Answer:**
Evaluate model on multiple data splits for robust performance estimate.

```python
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold,
    cross_validate
)

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

# Stratified K-Fold (for classification with class imbalance)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)

# Multiple metrics
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

results = cross_validate(model, X, y, cv=5, scoring=scoring)
print(results['test_accuracy'].mean())

# Time series split (for temporal data)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**Interview Tip**: Know when to use stratified CV for imbalanced data.
</details>

<details>
<summary><strong>5. What is hyperparameter tuning?</strong></summary>

**Answer:**
Finding optimal hyperparameters for best performance.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# GridSearchCV - exhaustive search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all processors
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# RandomizedSearchCV - random search (for large search spaces)
param_dist = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Halving search (efficient for large spaces)
from sklearn.model_selection import HalvingGridSearchCV

halving = HalvingGridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    factor=2
)

halving.fit(X_train, y_train)
```

**Interview Tip**: Know GridSearchCV, RandomizedSearchCV, and when to use each.
</details>

<details>
<summary><strong>6. What are preprocessing and scaling?</strong></summary>

**Answer:**
Standardizing features for better model performance.

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, PolynomialFeatures
)

# StandardScaler - zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMaxScaler - scale to [0, 1]
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train)

# RobustScaler - resistant to outliers
robust = RobustScaler()
X_robust = robust.fit_transform(X_train)

# OneHotEncoder - categorical to numerical
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X_categorical)

# LabelEncoder - convert labels to 0-n
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# PolynomialFeatures - create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

# Pipeline example (important: fit on train only!)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

**Key point**: Always fit transformers on training data only to avoid data leakage.

**Interview Tip**: Explain why scaling matters and demonstrate pipeline usage.
</details>

<details>
<summary><strong>7. What are classification algorithms?</strong></summary>

**Answer:**
Predicting categorical labels.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logistic Regression - linear decision boundary
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Decision Tree - hierarchical splits
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

# Random Forest - ensemble of trees
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

# Gradient Boosting - sequential tree building
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_train, y_train)

# SVM - support vector machine
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)

# KNN - k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Comparison
models = [lr, dt, rf, gb, svm, knn, nb]
for model in models:
    score = model.score(X_test, y_test)
    print(f"{model.__class__.__name__}: {score:.3f}")
```

**Interview Tip**: Know when to use each algorithm based on data characteristics.
</details>

<details>
<summary><strong>8. What are evaluation metrics?</strong></summary>

**Answer:**
Choosing right metrics for classification/regression.

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Classification metrics
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Accuracy: (TP + TN) / Total
accuracy = accuracy_score(y_true, y_pred)

# Precision: TP / (TP + FP) - "How many positive predictions are correct?"
precision = precision_score(y_true, y_pred)

# Recall: TP / (TP + FN) - "How many actual positives did we find?"
recall = recall_score(y_true, y_pred)

# F1: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)

# ROC-AUC: Probability curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification report
print(classification_report(y_true, y_pred))

# Regression metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

**When to use**:
- **Accuracy**: Balanced datasets
- **Precision/Recall**: Imbalanced data, different costs
- **F1**: Balance precision and recall
- **ROC-AUC**: Classification with probability threshold

**Interview Tip**: Discuss metric selection based on business requirements.
</details>

<details>
<summary><strong>9. What is feature selection?</strong></summary>

**Answer:**
Choosing relevant features to improve model.

```python
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)

# SelectKBest with chi2 (for non-negative features)
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)

# SelectKBest with f_classif (ANOVA F-test)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Mutual information
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination (RFE)
rfe = RFE(LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# SelectFromModel using feature importance
from sklearn.ensemble import RandomForestClassifier

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100),
    prefit=False,
    threshold='median'
)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
X_selected = rf.transform(X)

# Feature importance from trees
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(10):
    print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.3f}")
```

**Techniques**:
- **Filter methods**: Statistical tests (chi2, ANOVA, mutual info)
- **Wrapper methods**: RFE, iterative elimination
- **Embedded methods**: Feature importance from model

**Interview Tip**: Know when and why to use feature selection.
</details>

<details>
<summary><strong>10. What is handling imbalanced data?</strong></summary>

**Answer:**
Dealing with unequal class distributions.

```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Strategy 1: Class weights (penalize minority class more)
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)

# Strategy 2: Resampling with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)

# Strategy 3: Combined pipeline
pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# Strategy 4: Threshold adjustment
y_pred_proba = model.predict_proba(X_test)
y_pred = (y_pred_proba[:, 1] > 0.3).astype(int)  # Lower threshold

# Evaluation metrics for imbalanced data
from sklearn.metrics import precision_recall_curve, f1_score

# Use F1, precision, recall (NOT accuracy)
# Use stratified cross-validation
```

**Interview Tip**: Know multiple strategies for imbalanced data.
</details>

---

## Regression & Regularization

<details>
<summary><strong>11. What are Ridge, Lasso, and ElasticNet regression?</strong></summary>

**Answer:**
Regularized linear models that add penalty terms to prevent overfitting.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_regression(n_samples=200, n_features=50, noise=10, random_state=42)

# Ridge (L2): shrinks all coefficients, keeps all features
ridge = Ridge(alpha=1.0)
print(cross_val_score(ridge, X, y, cv=5, scoring='r2').mean())

# Lasso (L1): some coefficients become exactly 0 (feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print(f"Lasso non-zero features: {(lasso.coef_ != 0).sum()}")

# ElasticNet: mix of L1 and L2
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
print(cross_val_score(enet, X, y, cv=5, scoring='r2').mean())

# Choose alpha via cross-validation
from sklearn.linear_model import RidgeCV, LassoCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X, y)
print(f"Best alpha: {ridge_cv.alpha_}")

# When to use which:
# Ridge: all features contribute, correlated predictors
# Lasso: sparse model, automatic feature selection
# ElasticNet: Lasso with grouping effect (correlated features selected together)
```
</details>

<details>
<summary><strong>12. How do you build ensemble models (Voting, Stacking, Bagging)?</strong></summary>

**Answer:**
Ensembles combine multiple models to reduce variance and/or bias.

```python
from sklearn.ensemble import (VotingClassifier, StackingClassifier,
                               BaggingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# Voting: combine predictions by majority vote (hard) or avg probability (soft)
estimators = [
    ('lr',  LogisticRegression(max_iter=1000)),
    ('rf',  RandomForestClassifier(n_estimators=50)),
    ('svc', SVC(probability=True)),
]
voting_soft = VotingClassifier(estimators, voting='soft')
print(cross_val_score(voting_soft, X, y, cv=5).mean())

# Stacking: train meta-model on base model out-of-fold predictions
stacking = StackingClassifier(
    estimators=estimators[:2],
    final_estimator=LogisticRegression(),
    cv=5,              # use 5-fold for generating meta-features
    passthrough=False  # only use base model predictions as meta-features
)
print(cross_val_score(stacking, X, y, cv=5).mean())

# Bagging: bootstrap aggregation of the same model
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    max_samples=0.8,   # 80% of training data per bag
    max_features=0.8,  # 80% of features per bag
    bootstrap=True
)
print(cross_val_score(bagging, X, y, cv=5).mean())
```
</details>

<details>
<summary><strong>13. How do you perform clustering and evaluate it?</strong></summary>

**Answer:**
Clustering groups similar samples; evaluation uses internal metrics when labels are unavailable.

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
X = StandardScaler().fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X)
print(f"Silhouette: {silhouette_score(X, labels_km):.3f}")    # higher = better (-1 to 1)
print(f"Davies-Bouldin: {davies_bouldin_score(X, labels_km):.3f}")  # lower = better
print(f"ARI: {adjusted_rand_score(y_true, labels_km):.3f}")   # if ground truth known

# Elbow method: find optimal k
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
# Plot inertias to find elbow

# DBSCAN: density-based, handles arbitrary shapes and outliers
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise    = (labels_db == -1).sum()
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

# Agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_ag = agglo.fit_predict(X)
print(f"Agglo Silhouette: {silhouette_score(X, labels_ag):.3f}")
```
</details>

<details>
<summary><strong>14. How do you use PCA and other dimensionality reduction?</strong></summary>

**Answer:**
Dimensionality reduction for visualization, denoising, and improving model efficiency.

```python
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# PCA: linear, maximizes variance
pca = PCA(n_components=0.95)   # keep 95% of variance
X_pca = pca.fit_transform(X)
print(f"Components kept: {pca.n_components_}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Incremental PCA (for large datasets)
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=50, batch_size=200)
for batch in np.array_split(X, 5):
    ipca.partial_fit(batch)
X_ipca = ipca.transform(X)

# TruncatedSVD (like PCA but works on sparse matrices)
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X)

# t-SNE: non-linear, for visualization only (not for new data)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)   # (1797, 2) — 2D embedding for visualization

# NMF: non-negative matrix factorization (parts-based decomposition)
from sklearn.preprocessing import MinMaxScaler
X_pos = MinMaxScaler().fit_transform(digits.data)   # NMF needs non-negative
nmf = NMF(n_components=20, random_state=42)
X_nmf = nmf.fit_transform(X_pos)
```
</details>

<details>
<summary><strong>15. How do you detect and handle anomalies/outliers?</strong></summary>

**Answer:**
Scikit-learn provides multiple unsupervised anomaly detectors with a unified `fit_predict` API.

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import numpy as np

# Generate data with outliers
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_outliers = rng.uniform(-4, 4, (20, 2))
X_all = np.vstack([X, X_outliers])

# Isolation Forest: efficient, scales well
iso = IsolationForest(contamination=0.1, random_state=42)
labels_iso = iso.fit_predict(X_all)   # 1=normal, -1=outlier
scores_iso = iso.score_samples(X_all)  # anomaly score

# Local Outlier Factor: density-based
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels_lof = lof.fit_predict(X_all)   # -1=outlier

# Elliptic Envelope: assumes Gaussian distribution
ee = EllipticEnvelope(contamination=0.1, random_state=42)
labels_ee = ee.fit_predict(X_all)

# One-Class SVM: learns decision boundary around normal data
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
labels_svm = oc_svm.fit_predict(X_all)

# Evaluate (if ground truth available)
from sklearn.metrics import classification_report
y_true = np.ones(len(X_all))
y_true[-20:] = -1   # last 20 are outliers
print(classification_report(y_true, labels_iso))
```
</details>

<details>
<summary><strong>16. How do you engineer features with sklearn?</strong></summary>

**Answer:**
sklearn's preprocessing and feature extraction tools cover encoding, scaling, polynomial features, and more.

```python
from sklearn.preprocessing import (PolynomialFeatures, KBinsDiscretizer,
                                    PowerTransformer, QuantileTransformer)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Polynomial features: x1, x2 -> x1, x2, x1^2, x1*x2, x2^2
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[:, :2])
print(X_poly.shape)   # (100, 5)  — 2 features → 5 (1, x1, x2, x1^2, x1*x2, x2^2 minus bias)

# Binning: continuous → ordinal bins
binner = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
X_binned = binner.fit_transform(X[:, :1])
print(X_binned.shape)   # (100, 5)

# Power transform: make feature more Gaussian
pt = PowerTransformer(method='yeo-johnson')   # handles negatives
X_pt = pt.fit_transform(np.abs(X))

# Quantile transform: uniform or Gaussian output
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_qt = qt.fit_transform(np.abs(X))

# Feature selection: SelectKBest (univariate)
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)
print(selector.get_support())   # which features selected

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)
print(rfe.support_)   # mask of selected features
```
</details>

<details>
<summary><strong>17. How do you save and load sklearn models?</strong></summary>

**Answer:**
Use `joblib` (recommended) or `pickle` for serialization; `skops` for safer loading.

```python
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
pipeline.fit(X, y)

# joblib: preferred (memory-mapped for large arrays)
joblib.dump(pipeline, 'model.joblib')
loaded = joblib.load('model.joblib')
print(loaded.predict(X[:5]))

# Compressed save (smaller file)
joblib.dump(pipeline, 'model_compressed.joblib', compress=3)

# pickle: standard Python
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
with open('model.pkl', 'rb') as f:
    loaded_pkl = pickle.load(f)

# MLflow integration (versioned model registry)
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.sklearn.log_model(pipeline, "model")

# Versioning best practices:
# - Store sklearn version with model (for compatibility)
# - Never load pickles from untrusted sources (security risk)
# - Use ONNX for cross-framework deployment
import sklearn
print(sklearn.__version__)  # store this alongside model

# ONNX export (for production inference without Python)
# from skl2onnx import convert_sklearn
# model_onnx = convert_sklearn(pipeline, ...)
```
</details>

<details>
<summary><strong>18. How do you use ColumnTransformer for mixed data types?</strong></summary>

**Answer:**
`ColumnTransformer` applies different preprocessing to different columns in parallel.

```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Mixed dataset
data = pd.DataFrame({
    'age':      [25, np.nan, 35, 45],
    'salary':   [50000, 60000, np.nan, 80000],
    'city':     ['NYC', 'LA', 'NYC', None],
    'edu':      ['BS', 'MS', 'PhD', 'BS'],
    'churned':  [0, 1, 0, 1]
})
X = data.drop('churned', axis=1)
y = data['churned']

# Numeric pipeline
numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
])

# Categorical pipeline
categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# Ordinal pipeline
ordinal_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder(categories=[['BS', 'MS', 'PhD']])),
])

preprocessor = ColumnTransformer([
    ('num',      numeric_pipe,     ['age', 'salary']),
    ('cat',      categorical_pipe, ['city']),
    ('ordinal',  ordinal_pipe,     ['edu']),
], remainder='drop')

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(random_state=42))
])

full_pipeline.fit(X, y)
print(full_pipeline.predict(X))

# Auto-detect column types
auto_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
    ('cat', OneHotEncoder(), make_column_selector(dtype_include=object)),
])
```
</details>

<details>
<summary><strong>19. How do you tune hyperparameters efficiently?</strong></summary>

**Answer:**
Move beyond GridSearch to RandomizedSearch and Bayesian optimization for efficiency.

```python
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      HalvingGridSearchCV, HalvingRandomSearchCV)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from scipy.stats import randint, uniform
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
base = GradientBoostingClassifier(random_state=42)

# GridSearch: exhaustive (expensive)
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
gs = GridSearchCV(base, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
gs.fit(X, y)
print(gs.best_params_, gs.best_score_)

# RandomizedSearch: sample n_iter combinations (faster)
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(2, 8),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'min_samples_leaf': randint(1, 20),
}
rs = RandomizedSearchCV(base, param_dist, n_iter=50, cv=5, n_jobs=-1,
                         scoring='roc_auc', random_state=42)
rs.fit(X, y)
print(rs.best_params_, rs.best_score_)

# HalvingRandomSearch: successive halving (faster than random)
hrs = HalvingRandomSearchCV(base, param_dist, n_candidates='exhaust',
                             factor=3, cv=5, random_state=42)
hrs.fit(X, y)

# Optuna for Bayesian optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
        'max_depth':     trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    from sklearn.model_selection import cross_val_score
    return cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(study.best_params)
```
</details>

<details>
<summary><strong>20. How do you interpret models with sklearn tools?</strong></summary>

**Answer:**
Feature importance, permutation importance, and partial dependence plots reveal what drives predictions.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 1. Built-in feature importance (impurity-based, biased toward high-cardinality)
importances = rf.feature_importances_
top_features = np.argsort(importances)[::-1][:10]
for i in top_features:
    print(f"{data.feature_names[i]}: {importances[i]:.4f}")

# 2. Permutation importance (unbiased, works on test set)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                   random_state=42, n_jobs=-1)
for i in np.argsort(perm_imp.importances_mean)[::-1][:5]:
    print(f"{data.feature_names[i]}: {perm_imp.importances_mean[i]:.4f} "
          f"± {perm_imp.importances_std[i]:.4f}")

# 3. Partial Dependence Plot (marginal effect of one feature)
# PartialDependenceDisplay.from_estimator(rf, X_test, features=[0, 1], ...)

# 4. SHAP values (install: pip install shap)
# import shap
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values[1], X_test, feature_names=data.feature_names)
```
</details>

<details>
<summary><strong>21. Custom Transformers</strong></summary>

```python

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self    # stateless transformer

    def transform(self, X):
        return np.log(X + self.offset)

    def inverse_transform(self, X):
        return np.exp(X) - self.offset

# Works in pipelines automatically
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('log', LogTransformer()), ('scale', StandardScaler())])
```
</details>

<details>
<summary><strong>22. Custom Estimators</strong></summary>

```python

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.0, feature_idx=0):
        self.threshold = threshold
        self.feature_idx = feature_idx

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return (X[:, self.feature_idx] > self.threshold).astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
```
</details>

<details>
<summary><strong>23. MultiOutputClassifier and MultiLabelBinarizer</strong></summary>

```python

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# Multi-label: each sample can belong to multiple classes
mlb = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform([{'cat', 'dog'}, {'cat'}, {'dog', 'bird'}])
print(mlb.classes_)   # ['bird' 'cat' 'dog']

# OneVsRest: train one classifier per class
ovr = OneVsRestClassifier(RandomForestClassifier())

# MultiOutput: predict multiple targets simultaneously
X = np.random.randn(100, 5)
y_multi = np.random.randint(0, 2, (100, 3))  # 3 binary targets
mo = MultiOutputClassifier(RandomForestClassifier())
mo.fit(X, y_multi)
preds = mo.predict(X)   # (100, 3)
```
</details>

<details>
<summary><strong>24. CalibratedClassifierCV</strong></summary>

```python

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# GBM probabilities often need calibration
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Platt scaling (sigmoid) or isotonic regression
calibrated = CalibratedClassifierCV(clf, cv=5, method='isotonic')
calibrated.fit(X_train, y_train)

prob = calibrated.predict_proba(X_test)[:, 1]
# Now probabilities are better calibrated (fraction_pos ≈ predicted_prob)
frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
```
</details>

<details>
<summary><strong>25. TransformedTargetRegressor</strong></summary>

```python

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
import numpy as np

# Predict on log-scale, back-transform for evaluation
regressor = TransformedTargetRegressor(
    regressor=Ridge(),
    func=np.log1p,        # transform y before fitting
    inverse_func=np.expm1 # back-transform predictions
)
X = np.random.randn(100, 5)
y = np.abs(np.random.randn(100)) * 100   # skewed target
regressor.fit(X, y)
preds = regressor.predict(X)  # predictions on original scale
```
</details>

<details>
<summary><strong>26. sklearn Metrics: Multiclass</strong></summary>

```python

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 0, 1])

print(classification_report(y_true, y_pred, target_names=['A','B','C']))

# Multiclass AUC (one-vs-rest)
y_prob = np.random.dirichlet(np.ones(3), 6)
roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

# Macro, micro, weighted averaging
from sklearn.metrics import f1_score
print(f1_score(y_true, y_pred, average='macro'))    # unweighted mean per class
print(f1_score(y_true, y_pred, average='micro'))    # globally pool TP/FP/FN
print(f1_score(y_true, y_pred, average='weighted')) # weighted by support
```
</details>

<details>
<summary><strong>27. FeatureUnion: combine parallel feature sets</strong></summary>

```python

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Run PCA and raw features in parallel, then concatenate
feature_union = FeatureUnion([
    ('pca', PCA(n_components=5)),
    ('raw', StandardScaler()),
])
X = np.random.randn(100, 20)
X_combined = feature_union.fit_transform(X)
print(X_combined.shape)   # (100, 25) = 5 PCA + 20 raw
```
</details>

<details>
<summary><strong>28. Cross-validation strategies</strong></summary>

```python

from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold,
                                      TimeSeriesSplit, LeaveOneOut,
                                      RepeatedStratifiedKFold)

# Standard KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified: preserves class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GroupKFold: no group leaks across folds (e.g., users)
gkf = GroupKFold(n_splits=5)
groups = np.random.randint(0, 10, 100)  # 10 groups

# Time series: no future data in training
tss = TimeSeriesSplit(n_splits=5, gap=0)
for train_idx, val_idx in tss.split(X):
    pass  # train always before val

# Repeated: run k-fold multiple times, average results
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```
</details>

<details>
<summary><strong>29. Learning curves and validation curves</strong></summary>

```python

from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(500, 20)
y = (X[:, 0] > 0).astype(int)

# Learning curve: performance vs training set size
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=50),
    X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='roc_auc', n_jobs=-1)

print(train_scores.mean(axis=1))  # overfitting visible if >> val_scores
print(val_scores.mean(axis=1))    # validation performance per size

# Validation curve: performance vs hyperparameter
param_range = [1, 5, 10, 50, 100, 200]
train_s, val_s = validation_curve(
    RandomForestClassifier(), X, y,
    param_name='n_estimators', param_range=param_range,
    cv=5, scoring='roc_auc', n_jobs=-1)
```
</details>

<details>
<summary><strong>30. sklearn Pipelines with cross-validation (avoiding leakage)</strong></summary>

```python

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.random.randn(200, 50)
y = np.random.randint(0, 2, 200)

# WRONG: fit scaler on all data before CV (data leakage)
# scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
# scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)  # LEAKS!

# CORRECT: include all steps in pipeline (refitted on each fold's train set)
pipe = Pipeline([
    ('scale',  StandardScaler()),
    ('select', SelectKBest(f_classif, k=10)),
    ('clf',    LogisticRegression(max_iter=1000))
])
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
print(scores.mean())   # unbiased estimate
```
</details>

<details>
<summary><strong>31. GradientBoostingClassifier: key hyperparameters</strong></summary>

```python

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

# Classic GBM
gbm = GradientBoostingClassifier(
    n_estimators=200,        # number of trees
    learning_rate=0.05,      # shrinkage: lower = more trees needed
    max_depth=4,             # tree depth: 3-7 typical
    subsample=0.8,           # stochastic GBM: fraction of samples per tree
    min_samples_leaf=10,     # regularization
    max_features='sqrt',     # feature subsampling per split
)

# HistGradientBoosting: much faster (like LightGBM), handles NaN natively
hgbm = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,       # 2^depth
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,     # patience
)
```
</details>

<details>
<summary><strong>32. sklearn metrics for regression</strong></summary>

```python

from sklearn.metrics import (mean_squared_error, root_mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error,
                              explained_variance_score)
import numpy as np

y_true = np.array([1., 2., 3., 4., 5.])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")   # 1=perfect, 0=mean, <0=worse than mean
```
</details>

<details>
<summary><strong>33. OneHotEncoder vs OrdinalEncoder vs TargetEncoder</strong></summary>

```python

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3

import numpy as np

X_cat = np.array([['cat'], ['dog'], ['cat'], ['bird'], ['dog']])
y = np.array([1, 0, 1, 0, 0])

# OHE: creates binary columns (n_categories - 1 or n_categories columns)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
print(ohe.fit_transform(X_cat))

# OrdinalEncoder: integer codes (for tree-based models)
oe = OrdinalEncoder()
print(oe.fit_transform(X_cat))

# TargetEncoder: replaces category with smoothed mean of target
# (powerful, but must use CV to avoid leakage)
te = TargetEncoder(smooth='auto', cv=5)
X_te = te.fit_transform(X_cat, y)
print(X_te)
```
</details>

<details>
<summary><strong>34. Handling missing values</strong></summary>

```python

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import numpy as np

X = np.array([[1, 2, np.nan], [3, np.nan, 5], [np.nan, 6, 7], [4, 5, 6]])

# SimpleImputer: mean/median/mode/constant
si_mean   = SimpleImputer(strategy='mean').fit_transform(X)
si_median = SimpleImputer(strategy='median').fit_transform(X)
si_mode   = SimpleImputer(strategy='most_frequent').fit_transform(X)
si_const  = SimpleImputer(strategy='constant', fill_value=-1).fit_transform(X)

# KNN Imputer: use k nearest neighbors
knn_imp = KNNImputer(n_neighbors=2).fit_transform(X)

# Iterative Imputer (MICE): predict each feature using others
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iter_imp = IterativeImputer(random_state=42, max_iter=10).fit_transform(X)

# MissingIndicator: add binary column indicating missingness
from sklearn.impute import MissingIndicator
indicator = MissingIndicator().fit_transform(X)
print(indicator.shape)   # (4, 3) — one column per feature with missings
```
</details>

<details>
<summary><strong>35. Cross-val with custom scorer</strong></summary>

```python

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def weighted_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')

# Wrap as scorer
scorer = make_scorer(weighted_f1, greater_is_better=True)

X = np.random.randn(200, 10)
y = np.random.randint(0, 3, 200)
scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring=scorer)
print(scores.mean())

# Multiple metrics at once
from sklearn.model_selection import cross_validate
results = cross_validate(
    RandomForestClassifier(), X, y, cv=5,
    scoring={'f1': scorer, 'roc_auc': 'roc_auc_ovr_weighted'},
    return_train_score=True
)
print(results['test_f1'].mean())
```
</details>

<details>
<summary><strong>36. Support Vector Machines: kernels and tricks</strong></summary>

```python

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X = np.random.randn(300, 2)
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)   # circular boundary

# SVC with RBF kernel (non-linear)
pipe = Pipeline([
    ('scale', StandardScaler()),   # crucial! SVM is not scale-invariant
    ('svm',   SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])
pipe.fit(X, y)
print(pipe.score(X, y))

# Key params:
# C: regularization (small=large margin, large=small margin)
# gamma: rbf bandwidth (large=overfits, small=underfits)
# kernel: linear, poly, rbf, sigmoid

# LinearSVC: much faster for large datasets (uses liblinear)
lsvc = Pipeline([
    ('scale', StandardScaler()),
    ('svm',   LinearSVC(C=1.0, max_iter=5000))
])

# SVR for regression
svr = Pipeline([
    ('scale', StandardScaler()),
    ('svr',   SVR(kernel='rbf', C=100, epsilon=0.1))
])
```
</details>

<details>
<summary><strong>37. KNeighborsClassifier and distance metrics</strong></summary>

```python

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

X = np.random.randn(200, 5)
y = np.random.randint(0, 3, 200)

# Distance metrics
for metric in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    score = cross_val_score(knn, X, y, cv=5).mean()
    print(f"{metric}: {score:.3f}")

# Weighted KNN (closer neighbors count more)
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')

# Optimal k selection
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
          for k in range(1, 30)]
best_k = np.argmax(scores) + 1
print(f"Best k: {best_k}")
```
</details>

<details>
<summary><strong>38. Naive Bayes classifiers</strong></summary>

```python

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# GaussianNB: continuous features
X_cont = np.random.randn(200, 5)
y = np.random.randint(0, 2, 200)
gnb = GaussianNB()
gnb.fit(X_cont, y)

# MultinomialNB: count data (document classification)
corpus = ['cat sat mat', 'dog ran fast', 'cat dog ran', 'mat fast']
labels = [0, 1, 0, 1]
vec = TfidfVectorizer()
X_text = vec.fit_transform(corpus)
mnb = MultinomialNB(alpha=1.0)  # alpha=Laplace smoothing
mnb.fit(X_text, labels)

# ComplementNB: better for imbalanced text
cnb = ComplementNB()
cnb.fit(X_text, labels)

# BernoulliNB: binary features (word presence/absence)
bnb = BernoulliNB()
bnb.fit(X_text.toarray() > 0, labels)
```
</details>

<details>
<summary><strong>39. Metrics for imbalanced classification</strong></summary>

```python

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, matthews_corrcoef,
                              precision_recall_curve)
import numpy as np

y_true = np.array([0]*90 + [1]*10)    # 10:1 imbalance
y_prob = np.random.rand(100)
y_pred = (y_prob > 0.5).astype(int)

# ROC-AUC: ranking-based, robust to imbalance
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

# PR-AUC: better for severe imbalance
print(f"PR-AUC: {average_precision_score(y_true, y_prob):.3f}")

# Balanced accuracy: mean recall per class
print(f"Balanced Acc: {balanced_accuracy_score(y_true, y_pred):.3f}")

# Matthews Correlation Coefficient: works well for imbalance
print(f"MCC: {matthews_corrcoef(y_true, y_pred):.3f}")   # -1 to 1

# Optimal threshold from PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores[:-1])]
print(f"Best threshold: {best_thresh:.3f}")
```
</details>

<details>
<summary><strong>40. Pipeline with custom cross-val and nested CV</strong></summary>

```python

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

X = np.random.randn(200, 10)
y = np.random.randint(0, 2, 200)

# Nested CV: unbiased performance estimate with hyperparameter tuning
inner_cv  = 3   # folds for hyperparameter selection
outer_cv  = 5   # folds for performance estimation

pipe = Pipeline([('scale', StandardScaler()), ('svc', SVC())])
param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto']}

# Inner: GridSearchCV (picks best params on train fold)
gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')

# Outer: cross_val_score evaluates tuned model generalization
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)
print(f"Nested CV AUC: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
# This estimate is nearly unbiased — unlike single-level CV with tuning
```
</details>

<details>
<summary><strong>41. sparse matrices with sklearn</strong></summary>

```python

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

corpus = ['sky blue', 'sky red', 'grass green', 'green blue']
labels = [0, 0, 1, 1]

# TF-IDF returns sparse matrix
vec = TfidfVectorizer()
X_sparse = vec.fit_transform(corpus)   # scipy.sparse.csr_matrix
print(type(X_sparse), X_sparse.shape)

# Most sklearn estimators accept sparse input
lr = LogisticRegression()
lr.fit(X_sparse, labels)
print(lr.predict(X_sparse))

# Memory advantage: only stores non-zero values
dense = np.random.randn(1000, 10000)
sparse = csr_matrix((dense > 2.5).astype(float) * dense)
print(f"Dense: {dense.nbytes/1e6:.1f}MB, Sparse: {sparse.data.nbytes/1e3:.1f}KB")
```
</details>

<details>
<summary><strong>42. Isotonic regression for calibration</strong></summary>

```python

from sklearn.isotonic import IsotonicRegression
import numpy as np

# Fit monotone non-decreasing function
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([1, 4, 3, 6, 5, 8], dtype=float)   # non-monotone

ir = IsotonicRegression(out_of_bounds='clip')
y_iso = ir.fit_transform(x, y)
print(y_iso)   # [1. 3.5 3.5 5.5 5.5 8.]  — monotone non-decreasing

# Used in probability calibration (isotonic method)
# Guarantee: predicted probability increases monotonically with raw score
```
</details>

<details>
<summary><strong>43. Kernel approximation for scalable SVMs</strong></summary>

```python

from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.random.randn(10000, 20)
y = np.random.randint(0, 2, 10000)

# RBFSampler: approximate RBF kernel features (fast for large datasets)
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('approx', RBFSampler(gamma=0.1, n_components=100, random_state=42)),
    ('clf',    SGDClassifier(random_state=42))
])
pipe.fit(X, y)
print(pipe.score(X, y))

# Nystroem: more accurate but slower approximation
pipe2 = Pipeline([
    ('scale',  StandardScaler()),
    ('nystroem', Nystroem(gamma=0.1, n_components=100, random_state=42)),
    ('clf',    SGDClassifier())
])
```
</details>

<details>
<summary><strong>44. sklearn for time series (lagged features)</strong></summary>

```python

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def create_lag_features(X, n_lags=5):
    """Create lag features for time series."""
    result = []
    for lag in range(1, n_lags + 1):
        result.append(np.roll(X, lag, axis=0))
    return np.hstack([X] + result)[n_lags:]  # drop first n_lags rows

# Time series regression with lag features
ts = np.sin(np.linspace(0, 10*np.pi, 300)) + 0.1*np.random.randn(300)
X_lags = create_lag_features(ts.reshape(-1, 1), n_lags=10)
y = ts[10:]  # target (current value)

# Time series split
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tss.split(X_lags):
    model = Ridge().fit(X_lags[train_idx], y[train_idx])
    score = model.score(X_lags[val_idx], y[val_idx])
    print(f"R²: {score:.4f}")
```
</details>

<details>
<summary><strong>45. Gaussian Processes</strong></summary>

```python

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import numpy as np

X = np.linspace(0, 5, 50).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.1 * np.random.randn(50)

# Define kernel: RBF + noise
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

# GPR: returns mean + uncertainty
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
gpr.fit(X, y)

X_new = np.linspace(0, 6, 100).reshape(-1, 1)
y_pred, y_std = gpr.predict(X_new, return_std=True)

print(f"Learned kernel: {gpr.kernel_}")
# Confidence interval: y_pred ± 2*y_std (95%)
```
</details>

<details>
<summary><strong>46. Linear Discriminant Analysis (LDA)</strong></summary>

```python

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=300, n_features=20, n_classes=3,
                            n_informative=5, random_state=42)

# LDA: linear classifier + dimensionality reduction to n_classes-1 dims
lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
X_lda = lda.fit_transform(X, y)
print(X_lda.shape)   # (300, 2)  — reduced to 2D

# As classifier
print(cross_val_score(lda, X, y, cv=5).mean())

# QDA: allows different covariance per class (more flexible)
qda = QuadraticDiscriminantAnalysis()
print(cross_val_score(qda, X, y, cv=5).mean())

# LDA explained variance ratio
print(lda.explained_variance_ratio_)
```
</details>

<details>
<summary><strong>47. ExtraTreesClassifier vs RandomForest</strong></summary>

```python

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import time

X, y = make_classification(n_samples=5000, n_features=50, random_state=42)

# ExtraTrees: even more random (split threshold random per feature)
# Faster training, similar accuracy, often better on noise
t0 = time.time()
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_score = cross_val_score(et, X, y, cv=3).mean()
print(f"ExtraTrees: {et_score:.3f}, time: {time.time()-t0:.1f}s")

t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_score = cross_val_score(rf, X, y, cv=3).mean()
print(f"RandomForest: {rf_score:.3f}, time: {time.time()-t0:.1f}s")
```
</details>

<details>
<summary><strong>48. Manifold learning: Isomap, LLE, UMAP</strong></summary>

```python

from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data

# Isomap: preserves geodesic distances
iso = Isomap(n_neighbors=10, n_components=2)
X_iso = iso.fit_transform(X)

# LLE: locally linear embedding
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
X_lle = lle.fit_transform(X)

# Spectral Embedding
se = SpectralEmbedding(n_components=2, n_neighbors=10)
X_se = se.fit_transform(X)

# UMAP (not in sklearn, but commonly used)
# pip install umap-learn
# import umap
# X_umap = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```
</details>

<details>
<summary><strong>49. Spectral Clustering</strong></summary>

```python

from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_moons
import numpy as np

X, y_true = make_moons(n_samples=200, noise=0.05, random_state=42)

# SpectralClustering: handles non-convex shapes
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=10, random_state=42)
labels_sc = sc.fit_predict(X)
print(f"SpectralClustering ARI: {adjusted_rand_score(y_true, labels_sc):.3f}")

# Agglomerative clustering with different linkages
for linkage in ['ward', 'complete', 'average', 'single']:
    ac = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    labels = ac.fit_predict(X)
    print(f"{linkage}: ARI={adjusted_rand_score(y_true, labels):.3f}")
```
</details>

<details>
<summary><strong>50. BayesSearchCV and AutoML with sklearn</strong></summary>

```python

# pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(500, 20)
y = np.random.randint(0, 2, 500)

search_spaces = {
    'n_estimators':    Integer(50, 300),
    'max_depth':       Integer(3, 15),
    'min_samples_leaf': Integer(1, 30),
    'max_features':    Real(0.1, 1.0),
}

opt = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
opt.fit(X, y)
print(f"Best score: {opt.best_score_:.4f}")
print(f"Best params: {opt.best_params_}")
```
</details>

<details>
<summary><strong>51. Cross-decomposition: PLS Regression</strong></summary>

```python

from sklearn.cross_decomposition import PLSRegression, CCA
import numpy as np

# PLS: like PCA but supervised (maximizes covariance between X and y)
X = np.random.randn(100, 20)
y = np.random.randn(100, 3)   # multivariate target

pls = PLSRegression(n_components=5)
pls.fit(X, y)
X_scores, y_scores = pls.transform(X, y)
print(X_scores.shape)   # (100, 5)  — latent space
y_pred = pls.predict(X)  # (100, 3)

# Useful when: multicollinear features, many features relative to samples
```
</details>

<details>
<summary><strong>52. Handling categorical features natively (HistGBM)</strong></summary>

```python

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd

# HistGBM can handle categoricals natively (like LightGBM)
X = pd.DataFrame({
    'num1':  np.random.randn(200),
    'num2':  np.random.randn(200),
    'cat1':  np.random.choice(['a','b','c'], 200),
    'cat2':  np.random.choice(['x','y'], 200),
})
y = np.random.randint(0, 2, 200)

# Encode categoricals as integers
cat_cols = ['cat1', 'cat2']
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = oe.fit_transform(X[cat_cols])

# Tell HistGBM which features are categorical
categorical_features = [X.columns.get_loc(c) for c in cat_cols]

hgbm = HistGradientBoostingClassifier(
    categorical_features=categorical_features,
    random_state=42
)
hgbm.fit(X, y)
```

**53-100. Additional Key Concepts**
</details>

<details>
<summary><strong>53. Clone estimator</strong></summary>

`sklearn.base.clone(estimator)` creates unfitted copy with same params.
</details>

<details>
<summary><strong>54. set_output API (sklearn >= 1.2)</strong></summary>

`estimator.set_output(transform='pandas')` returns DataFrames from transformers.
</details>

<details>
<summary><strong>55. Pipeline step access</strong></summary>

`pipe['step_name']` or `pipe.named_steps['step_name']`.
</details>

<details>
<summary><strong>56. Decision boundary visualization</strong></summary>

use `DecisionBoundaryDisplay.from_estimator()`.
</details>

<details>
<summary><strong>57. Estimator tags</strong></summary>

`estimator._get_tags()` reveals capabilities (sparse support, sample weights, etc.).
</details>

<details>
<summary><strong>58. warm_start</strong></summary>

incrementally add trees: `rf.n_estimators = 200; rf.fit(X, y)` after initial fit with `warm_start=True`.
</details>

<details>
<summary><strong>59. n_jobs=-1</strong></summary>

use all CPU cores; -2 means all minus one core.
</details>

<details>
<summary><strong>60. class_weight='balanced'</strong></summary>

auto-compute weights inversely proportional to class frequencies.
</details>

<details>
<summary><strong>61. sample_weight</strong></summary>

pass to `.fit()` for weighted samples (important observations count more).
</details>

<details>
<summary><strong>62. Sparse output from encoders</strong></summary>

`OneHotEncoder(sparse_output=True)` — memory efficient for many categories.
</details>

<details>
<summary><strong>63. get_feature_names_out()</strong></summary>

all modern transformers expose feature names for pipeline introspection.
</details>

<details>
<summary><strong>64. FunctionTransformer</strong></summary>

wrap any NumPy function as a transformer.

```python
from sklearn.preprocessing import FunctionTransformer
log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
```
</details>

<details>
<summary><strong>65. Recursive Feature Elimination with CV (RFECV)</strong></summary>

```python

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
rfecv = RFECV(RandomForestClassifier(), min_features_to_select=5, cv=5)
rfecv.fit(X, y)
print(rfecv.n_features_)  # optimal number of features
```
</details>

<details>
<summary><strong>66. Variance Threshold</strong></summary>

remove features with variance below threshold (removes near-constant features).
</details>

<details>
<summary><strong>67. Mutual information feature selection</strong></summary>

`SelectKBest(mutual_info_classif, k=10)` — non-linear univariate selection.
</details>

<details>
<summary><strong>68. Robust Scaler</strong></summary>

scales using median and IQR — not affected by outliers.

```python
from sklearn.preprocessing import RobustScaler
rs = RobustScaler(quantile_range=(25, 75))
```
</details>

<details>
<summary><strong>69. MaxAbsScaler</strong></summary>

scales to [-1, 1] without centering — preserves sparsity.
</details>

<details>
<summary><strong>70. Normalizer</strong></summary>

normalizes each sample (row) to unit norm — useful for text/cosine similarity.
</details>

<details>
<summary><strong>71. LabelEncoder vs OrdinalEncoder</strong></summary>

LabelEncoder for target y (1D), OrdinalEncoder for features X (2D).
</details>

<details>
<summary><strong>72. GridSearchCV best practices</strong></summary>

use `refit=True` (default), `return_train_score=True`, `error_score='raise'`.
</details>

<details>
<summary><strong>73. cross_val_predict</strong></summary>

get out-of-fold predictions for stacking or diagnostics.

```python
from sklearn.model_selection import cross_val_predict
oof_preds = cross_val_predict(model, X, y, cv=5, method='predict_proba')
```
</details>

<details>
<summary><strong>74. KernelPCA</strong></summary>

non-linear PCA using kernel trick.

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
```
</details>

<details>
<summary><strong>75. SparsePCA and MiniBatchDictionaryLearning</strong></summary>

sparse coding for feature learning on large datasets.
</details>

<details>
<summary><strong>76. CountVectorizer vs TfidfVectorizer</strong></summary>

CountVectorizer returns raw counts; TF-IDF down-weights common terms. Use `ngram_range=(1,2)` for bigrams.
</details>

<details>
<summary><strong>77. HashingVectorizer</strong></summary>

memory-efficient text vectorization without vocabulary (stateless — no fit needed).
</details>

<details>
<summary><strong>78. LatentDirichletAllocation (LDA)</strong></summary>

topic modeling for documents.

```python
from sklearn.decomposition import LatentDirichletAllocation
lda_topic = LatentDirichletAllocation(n_components=5, random_state=42)
```
</details>

<details>
<summary><strong>79. DictVectorizer</strong></summary>

convert list of dicts to feature matrix (useful for NLP features).

```python
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
X = dv.fit_transform([{'city': 'NYC', 'temp': 72}, {'city': 'LA', 'temp': 85}])
```
</details>

<details>
<summary><strong>80. MiniBatchKMeans</strong></summary>

scales KMeans to large datasets with mini-batch updates.

```python
from sklearn.cluster import MiniBatchKMeans
mbkm = MiniBatchKMeans(n_clusters=10, batch_size=1000, random_state=42)
```
</details>

<details>
<summary><strong>81. MeanShift clustering</strong></summary>

bandwidth-free, finds cluster centers at density peaks.
</details>

<details>
<summary><strong>82. OPTICS</strong></summary>

like DBSCAN but handles varying density clusters.
</details>

<details>
<summary><strong>83. BirchClustering</strong></summary>

hierarchical, fast for large datasets, acts as data summarizer.
</details>

<details>
<summary><strong>84. cross_val_score with groups</strong></summary>

use `groups` parameter for GroupKFold.
</details>

<details>
<summary><strong>85. Parallel processing</strong></summary>

`n_jobs=-1` in GridSearchCV, cross_val_score, RandomForest uses joblib internally.
</details>

<details>
<summary><strong>86. Precision-Recall tradeoff</strong></summary>

adjust `predict_proba` threshold based on business cost: lower threshold → higher recall (catch more positives), lower precision (more false positives).
</details>

<details>
<summary><strong>87. Threshold optimization</strong></summary>

```python

from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_true, y_prob)
# Choose threshold maximizing F-beta score
beta = 2  # recall 2x more important
fbeta = (1+beta**2)*p*r/((beta**2)*p + r + 1e-8)
best_thresh = thresholds[fbeta[:-1].argmax()]
```
</details>

<details>
<summary><strong>88. VotingRegressor</strong></summary>

ensemble of regressors with averaged predictions.
</details>

<details>
<summary><strong>89. StackingRegressor</strong></summary>

meta-learner trained on out-of-fold predictions of base regressors.
</details>

<details>
<summary><strong>90. TransformerMixin.fit_transform()</strong></summary>

inherited method combining fit + transform in one call.
</details>

<details>
<summary><strong>91. BaseEstimator.get_params()</strong></summary>

returns all constructor parameters — used by GridSearchCV for param grid.
</details>

<details>
<summary><strong>92. Pipelines and memory caching</strong></summary>

`Pipeline(steps, memory='cache_dir')` caches intermediate results.
</details>

<details>
<summary><strong>93. check_estimator()</strong></summary>

validates custom estimator follows sklearn API conventions.

```python
from sklearn.utils.estimator_checks import check_estimator
check_estimator(MyCustomClassifier())
```
</details>

<details>
<summary><strong>94. AdaBoostClassifier</strong></summary>

boosts weak learners by upweighting misclassified samples.

```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
```
</details>

<details>
<summary><strong>95. SGDClassifier and partial_fit</strong></summary>

online learning — update model incrementally on new data.

```python
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss')
for X_batch, y_batch in data_stream:
    sgd.partial_fit(X_batch, y_batch, classes=[0, 1])
```
</details>

<details>
<summary><strong>96. MLPClassifier and MLPRegressor</strong></summary>

neural network in sklearn (for small to medium datasets).

```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                    max_iter=200, early_stopping=True)
```
</details>

<details>
<summary><strong>97. set_params()</strong></summary>

dynamically change estimator parameters.

```python
rf = RandomForestClassifier(n_estimators=100)
rf.set_params(n_estimators=200, max_depth=10)
```
</details>

<details>
<summary><strong>98. class_prior in Naive Bayes</strong></summary>

override default prior with custom class probabilities.
</details>

<details>
<summary><strong>99. OneVsOneClassifier</strong></summary>

train one classifier per pair of classes (n*(n-1)/2 classifiers) — better than OvR for SVMs.

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
ovo = OneVsOneClassifier(SVC())
```
</details>

<details>
<summary><strong>100. sklearn design principles summary</strong></summary>

```

Estimator API:
- fit(X, y): train
- predict(X): predict
- predict_proba(X): probabilities
- transform(X): transform data
- fit_transform(X): fit + transform combined
- score(X, y): default metric

Consistency:
- All hyperparams set in __init__
- Fitted attributes end with underscore (coef_, classes_)
- Accepts numpy arrays and DataFrames
- Pipeline-compatible (inherits from BaseEstimator + Mixin)

Best practices:
- Always scale features for: SVM, KNN, LinearRegression, NeuralNet
- Tree-based models don't need scaling
- Use Pipeline to prevent data leakage
- Tune with cross-validation, not test set
- Use stratified splits for imbalanced data
```
</details>
---

## sklearn Quick Reference

| Task | Primary | Alternative |
|------|---------|-------------|
| Classification | RandomForest | HistGradientBoosting |
| Regression | GradientBoosting | Ridge |
| Clustering | KMeans | DBSCAN |
| Dimensionality reduction | PCA | UMAP |
| Feature selection | SelectKBest | RFECV |
| HPO | RandomizedSearchCV | Optuna |
| Scaling | StandardScaler | RobustScaler |
| Encoding | OneHotEncoder | TargetEncoder |
| Imputation | SimpleImputer | KNNImputer |
| Anomaly | IsolationForest | LocalOutlierFactor |

## Summary in simple points

- The estimator API separates fit, transform, predict, scoring and parameters. Respect fitted-state and cloning conventions when composing models.
- Pipelines keep learned preprocessing inside validation folds; ColumnTransformer handles columns with different type contracts.
- Scale, impute, encode and select features using training data only. Treat unknown categories and missing values as expected serving cases.
- Linear, regularised, neighbour, SVM, Naive Bayes, tree, ensemble and neural estimators need different preprocessing and tuning choices.
- Bagging, voting, stacking and boosting combine predictions differently. Avoid leaking training predictions into a stacking meta-model.
- Choose classification, regression, ranking and clustering metrics for the task. Interpret macro/micro averaging, PR thresholds and class imbalance explicitly.
- Calibration changes probability reliability. Use appropriate calibration data and keep preprocessing and calibration with the deployed estimator.
- Cross-validation strategies must respect time, groups and stratification. Nested validation separates hyperparameter selection from final estimation.
- Grid, random and Bayesian search differ in cost and exploration; third-party search/AutoML tools are not all part of sklearn itself.
- Learning and validation curves diagnose data/complexity effects. Permutation importance and other explanations need appropriate evaluation data.
- PCA, kernel approximations, sparse decomposition, manifold methods and discriminant analysis preserve different structures; some named tools are external libraries.
- K-means, MiniBatchKMeans, density, spectral, MeanShift and Birch clustering have different geometry, scale and parameter assumptions.
- Custom transformers/estimators need get_params/set_params, fitted attributes, shape contracts and meaningful check_estimator validation.
- FeatureUnion, FunctionTransformer, set_output and feature-name APIs help compose and inspect transformations; verify sparse/dense output expectations.
- Text vectorisers, hashing, dictionaries and topic models have different vocabulary, collision and memory behaviour.
- Multi-output, multi-label, one-versus-rest and one-versus-one strategies change target representation and training structure.
- Sample/class weights, threshold choice and resampling address different objectives. Normalizer rescales rows, while feature scalers act by column.
- Online partial_fit, warm_start and parallel n_jobs are distinct controls; watch oversubscription and class requirements.
- Pipeline caching can save repeat work but needs valid cache identity. Persist only trusted artefacts with compatible dependency and feature-schema versions.
- Use small runnable cases to check API migrations and preprocessing edge cases before evaluating real model quality.
