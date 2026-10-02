---
id: ml-question-bank
title: "Question Bank: Practice Across the Course"
sidebar_label: "Question bank"
sidebar_position: 4
slug: /theory/ml/question-bank
description: "Fifty-five practice questions with worked answers across instance-based learning, SVMs, Bayesian learning, ensembles, unsupervised learning and model evaluation, with every worked number checked in code."
tags: [question-bank, practice, exam, revision]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Answer on paper first, open the answer second, and trust the arithmetic because every worked number below has been recomputed in code.

## The idea in plain words

Reading an answer feels like understanding it. Producing one is the real test. This bank is for the second activity: pick a module, cover the answers, write down what you would say, then compare. The questions come in two kinds. **Conceptual** ones ask you to explain or contrast (bagging against boosting, a generative against a discriminative classifier). **Numeric** ones give you small numbers to work through by hand (a k-NN vote, an SVM margin, a Bayes update, one k-means step, a confusion matrix).

<Infographic src="/img/ml/question-bank-worked-numbers.svg" alt="Six cards of worked results: k-NN distances and the class vote, SVM margin widths, two Bayes updates, a k-means first pass, metrics for TP 40, FP 10, FN 20, TN 30, and the tie in question 36." caption="The numeric questions at a glance. Each figure is printed by the code block below." />

## How it works

The bank merges two practice sets from the same course. The larger set has 50 questions; the shorter "comprehensive" set has 24. They overlap heavily, so the merge keeps one copy of each question:

- **55 questions in all**: the 50 of the larger set, plus 5 from the shorter set that the larger one did not ask (the Euclidean distance, the second SVM margin, the spam update, "why is Naive Bayes a strong baseline" and the cluster centroid).
- **19 duplicates are folded in.** Where the shorter set asked the same thing as a question in the larger set, its wording and answer sit inside that question's answer, under "Also asked as" and "Short form". Nothing was dropped; a few short forms add a point the long answer lacks.
- **Grouped by lecture module**, 6 to 11, because that is the order of the course. Practice questions for modules 1 to 5 sit at the end of their own chapters ([preprocessing](/docs/theory/ml/data-preprocessing), [regression](/docs/theory/ml/regression-and-gradient-descent), [classification](/docs/theory/ml/classification-and-logistic-regression) and [decision trees](/docs/theory/ml/decision-trees)), and the six questions of the evaluation lecture are at the end of the [evaluation chapter](/docs/theory/ml/model-evaluation).
- **Two answers needed a note**, flagged in place: the k-means tie in Q40 and the claim about AUC in Q52.

For module 11, the confusion-matrix calculator in the [evaluation chapter](/docs/theory/ml/model-evaluation) has a preset for the bank's own example (TP 40, FP 10, FN 20, TN 30).

## Code you can run

Every worked number in the bank, recomputed. Compare the printed values with the answers; they agree, and the block also prints the two ways the Q36 tie can be broken.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

print("k-NN: distance between (1,2) and (4,6):", float(np.hypot(4 - 1, 6 - 2)))
points = {"A": ((1, 2), "+"), "B": ((2, 3), "+"), "C": ((6, 6), "-"), "D": ((7, 7), "-")}
query = np.array([3, 4])
dist = {k: float(np.linalg.norm(np.array(v[0]) - query)) for k, v in points.items()}
print("k-NN distances to (3,4):", {k: round(d, 2) for k, d in dist.items()})
for k in (1, 3):
    votes = [points[n][1] for n in sorted(dist, key=dist.get)[:k]]
    print(f"  k={k} votes {votes} -> {max(set(votes), key=votes.count)}")

w = np.array([3, 4])
print("\nSVM margin for w=(3,4):", 2 / np.linalg.norm(w), "  for ||w||=0.5:", 2 / 0.5)
print("SVM w=(2,1), b=-5, x=(2,3): w.x+b =", float(np.dot([2, 1], [2, 3]) - 5))

print("\nBayes, spam: ", round(0.4 * 0.8 / (0.4 * 0.8 + 0.6 * 0.1), 3))
print("Bayes, toothache:", round(0.6 * 0.2 / (0.6 * 0.2 + 0.1 * 0.8), 3))

print("\ncentroid of (1,1),(2,1),(1.5,2):", np.round(np.mean([(1, 1), (2, 1), (1.5, 2)], axis=0), 2))
values = np.array([2, 4, 10, 12, 3, 20, 30, 11, 25])
c1, c2 = 2.0, 4.0
to_c1 = np.abs(values - c1) < np.abs(values - c2)
tie = np.abs(values - c1) == np.abs(values - c2)
print("k-means first assignment, c1 gets:", values[to_c1 | tie].tolist(), " c2 gets:", values[~(to_c1 | tie)].tolist())
print("tie broken towards c1 gives new centroids:", values[to_c1 | tie].mean(), values[~(to_c1 | tie)].mean())
print("tie broken towards c2 gives new centroids:", values[to_c1].mean().round(3), values[~to_c1].mean().round(3))

tp, fp, fn, tn = 40, 10, 20, 30
y_true = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn)
y_pred = np.array([1] * tp + [1] * fp + [0] * fn + [0] * tn)
print("\nconfusion matrix [[TN FP] [FN TP]]:", confusion_matrix(y_true, y_pred).tolist())
print("precision", round(precision_score(y_true, y_pred), 3), "recall", round(recall_score(y_true, y_pred), 3), "F1", round(f1_score(y_true, y_pred), 3))
```

## Practice questions

### Module 6: Instance-based learning

<details>
<summary><strong>Q1.</strong> What is instance-based (lazy) learning, and how does it differ from eager learning?</summary>

Instance-based learners store the training examples and defer all computation to prediction time, they build no explicit global model. To classify a query they find the most similar stored instances and vote/average. This contrasts with eager learners (decision trees, SVMs, neural nets) that build a compact model at training time and discard the data. Lazy learning has near-zero training cost and adapts locally to each query, but prediction is expensive (must compare to many stored points) and it stores the whole dataset.

**Also asked as:** How does k-NN classify, and why is it called 'lazy'?

**Short form:** k-NN stores all training examples and, for a query, finds the k nearest (by a distance metric) and takes a majority vote (classification) or average (regression). It is lazy because there is no training phase, all computation happens at query time.

</details>

<details>
<summary><strong>Q2.</strong> Describe the k-Nearest-Neighbours algorithm for classification.</summary>

Choose k and a distance metric. For a query x: compute the distance from x to every training point, pick the k closest, and return the majority class among them (for regression, the mean of their targets). No training beyond storing the data. k controls smoothing: small k gives a jagged, low-bias/high-variance boundary; large k gives a smoother, higher-bias boundary.

</details>

<details>
<summary><strong>Q3.</strong> Why is feature scaling important for k-NN?</summary>

k-NN relies on distances, and an unscaled feature with a large numeric range dominates the distance, drowning out others. E.g. salary (0–100000) vs age (0–100): Euclidean distance is essentially decided by salary. Standardising (z-score) or min-max scaling puts features on a comparable scale so each contributes fairly. Without scaling the 'nearest' neighbours reflect only the largest-range feature.

</details>

<details>
<summary><strong>Q4.</strong> k-NN numeric: points A(1,2)+, B(2,3)+, C(6,6)-, D(7,7)-. Classify query (3,4) with k=1 and k=3 (Euclidean).</summary>

Distances to (3,4): A=sqrt(4+4)=2.83, B=sqrt(1+1)=1.41, C=sqrt(9+4)=3.61, D=sqrt(16+9)=5.0. k=1: nearest is B(+) -> classify +. k=3: nearest three are B(1.41,+), A(2.83,+), C(3.61,-) -> votes +,+,- -> majority + . So the query is classified positive for both k=1 and k=3.

</details>

<details>
<summary><strong>Q5.</strong> Compute the Euclidean distance between (1,2) and (4,6).</summary>

√((4−1)² + (6−2)²) = √(9+16) = √25 = 5. k-NN would rank neighbours by this distance.

</details>

<details>
<summary><strong>Q6.</strong> How do you choose k? What are the effects of very small or very large k?</summary>

Choose k by cross-validation, picking the value that minimises validation error; odd k avoids ties in binary problems. Very small k (=1) fits noise, low bias, high variance, jagged boundary sensitive to outliers. Very large k oversmooths, high bias, low variance, and as k approaches N it just predicts the majority class. The sweet spot balances the two.

**Also asked as:** What is the effect of k, and why does feature scaling matter for k-NN?

**Short form:** Small k → low bias, high variance (noisy, jagged boundary); large k → smoother but can blur classes. Scaling matters because distances are dominated by large-range features, so unscaled features distort the neighbourhood.

</details>

<details>
<summary><strong>Q7.</strong> What is distance-weighted k-NN and why use it?</summary>

Instead of equal votes, each neighbour votes with weight 1/d^2 (or a kernel of distance), so closer neighbours count more. This makes predictions less sensitive to the exact choice of k, lets you even use all points (k=N) with far ones contributing negligibly, and smooths the influence of borderline neighbours. It helps when neighbours lie at very different distances.

</details>

<details>
<summary><strong>Q8.</strong> What is the curse of dimensionality and how does it affect k-NN?</summary>

In high dimensions, data becomes sparse and distances concentrate, the ratio of nearest to farthest distance approaches 1, so 'nearest neighbour' loses meaning and every point looks roughly equidistant. k-NN degrades badly, needing exponentially more data to keep neighbourhoods dense. Remedies: dimensionality reduction (PCA), feature selection, or learned/weighted metrics.

</details>

<details>
<summary><strong>Q9.</strong> Give one advantage and one disadvantage of k-NN.</summary>

Advantage: simple, no training, naturally handles multi-class and complex non-linear boundaries, and adapts locally. Disadvantage: slow and memory-heavy at prediction (stores and scans all data), sensitive to irrelevant features and feature scaling, and hurt by the curse of dimensionality. Approximate-NN structures (KD-trees, ball trees, LSH) speed up search in low/moderate dimensions.

**Also asked as:** Give one strength and one weakness of instance-based learning.

**Short form:** Strength: no training, adapts instantly to new data, and models complex boundaries. Weakness: slow, memory-heavy predictions (must compare to all points) and sensitive to irrelevant features and the curse of dimensionality.

</details>

### Module 7: Support vector machines

<details>
<summary><strong>Q10.</strong> What is the core idea of a Support Vector Machine?</summary>

An SVM finds the linear decision boundary (hyperplane) that separates the classes with the maximum margin, the largest possible gap between the boundary and the nearest points of each class. Maximising the margin gives the most robust separator and the best expected generalisation. Only the closest points (the support vectors) determine the boundary.

**Also asked as:** What does an SVM optimise, and what is the margin?

**Short form:** It finds the hyperplane that maximises the margin, the distance to the nearest points (support vectors) of each class. A larger margin generalises better. The margin width is 2/||w||.

</details>

<details>
<summary><strong>Q11.</strong> Define the margin and support vectors.</summary>

The margin is the perpendicular distance between the two class-boundary hyperplanes that just touch the nearest training points; the SVM maximises this width. Support vectors are the training points lying exactly on those margin boundaries (or, in soft margin, inside/violating it). They alone define the hyperplane, removing any non-support-vector leaves the solution unchanged.

</details>

<details>
<summary><strong>Q12.</strong> For a hyperplane w·x + b = 0, what is the geometric margin, and what does SVM optimise?</summary>

The margin width is 2/||w||. Maximising it is equivalent to minimising (1/2)||w||^2 subject to y_i(w·x_i + b) >= 1 for all i (the hard-margin constraint). So SVM solves a convex quadratic program: minimise (1/2)||w||^2 s.t. every point is correctly classified with functional margin at least 1.

</details>

<details>
<summary><strong>Q13.</strong> Distinguish hard-margin from soft-margin SVM. What does C control?</summary>

Hard margin requires perfect linear separation with no violations, impossible if classes overlap or data is noisy. Soft margin adds slack variables xi_i >= 0 allowing some points inside/across the margin, minimising (1/2)||w||^2 + C\*sum(xi_i). C trades margin width against violations: large C penalises errors heavily (narrow margin, risk of overfitting), small C allows more violations (wider margin, more regularisation).

**Also asked as:** What are slack variables / the soft margin?

**Short form:** They allow some points to fall inside the margin or be misclassified (penalised by C), so the SVM handles non-separable data. Large C → hard margin (few violations, risk overfit); small C → wider, softer margin.

</details>

<details>
<summary><strong>Q14.</strong> What is the kernel trick and why is it powerful?</summary>

Many problems aren't linearly separable in the input space but are in a higher-dimensional feature space. The kernel trick computes the inner product in that space directly via a kernel K(x,z)=phi(x)·phi(z) without ever forming phi(x). Since the SVM dual depends on data only through inner products, you get non-linear boundaries at the cost of the (cheap) kernel, even for infinite-dimensional spaces (RBF).

**Also asked as:** What is the kernel trick?

**Short form:** Kernels compute inner products in a high-dimensional feature space without explicitly mapping the data (e.g. RBF, polynomial), letting a linear SVM draw nonlinear boundaries efficiently, 'linear in a transformed space'.

</details>

<details>
<summary><strong>Q15.</strong> Name three common kernels and when each is used.</summary>

Linear K=x·z: high-dimensional/sparse data (e.g. text) that is already near-linearly separable. Polynomial K=(x·z+c)^d: interactions up to degree d. RBF/Gaussian K=exp(-gamma||x-z||^2): general non-linear boundaries, the default when unsure; gamma sets the reach of each point (large gamma = very local, risk of overfitting). Choose by cross-validation.

</details>

<details>
<summary><strong>Q16.</strong> SVM numeric: support vectors give w=(2,1) and b=-5. Classify x=(2,3) and give its functional-margin sign.</summary>

Compute w·x + b = 2\*2 + 1\*3 - 5 = 4 + 3 - 5 = 2 > 0 -> predict the positive class. The value 2 is the (signed) functional output; since it exceeds +1, the point lies beyond the positive margin boundary and is correctly, confidently classified (not a support vector).

</details>

<details>
<summary><strong>Q17.</strong> SVM numeric: if ||w|| = 0.5, what is the margin width?</summary>

Margin width = 2/||w|| = 2/0.5 = 4. So the gap between the two class-boundary hyperplanes is 4 units. Smaller ||w|| means a wider margin, which is exactly why minimising ||w|| maximises the margin.

</details>

<details>
<summary><strong>Q18.</strong> For w = (3,4), compute the SVM margin width.</summary>

Margin = 2/||w|| = 2/√(3²+4²) = 2/5 = 0.4. Minimising ||w|| (maximising the margin) is the SVM objective.

</details>

<details>
<summary><strong>Q19.</strong> Why are SVMs effective in high-dimensional spaces?</summary>

Margin maximisation is a form of capacity control (regularisation) that depends on the margin, not directly on the number of features, so SVMs resist overfitting even when features outnumber samples (e.g. text, genomics). The solution depends only on support vectors, and the kernel trick handles non-linearity without explicit high-dimensional computation.

</details>

### Module 8: Bayesian learning

<details>
<summary><strong>Q20.</strong> State Bayes' theorem and name each term.</summary>

P(h|D) = P(D|h)P(h) / P(D). P(h) is the prior (belief before data), P(D|h) the likelihood (how well hypothesis h explains data D), P(D) the evidence (normaliser), and P(h|D) the posterior (updated belief after seeing D). Bayesian learning updates beliefs about hypotheses as data arrives.

**Also asked as:** State Bayes' theorem and the Naive Bayes independence assumption.

**Short form:** P(c|x) = P(x|c)P(c)/P(x). Naive Bayes assumes features are conditionally independent given the class, so P(x|c) = Π P(xᵢ|c): crude but effective, especially for text.

</details>

<details>
<summary><strong>Q21.</strong> What is the MAP hypothesis, and how does it relate to ML (maximum likelihood)?</summary>

The MAP (maximum a posteriori) hypothesis maximises P(D|h)P(h): the posterior up to the constant P(D). The ML hypothesis maximises just the likelihood P(D|h). MAP equals ML when the prior is uniform (all hypotheses equally likely a priori); otherwise the prior shifts the choice. MAP = argmax_h P(D|h)P(h).

</details>

<details>
<summary><strong>Q22.</strong> Describe the Naive Bayes classifier and its key assumption.</summary>

Naive Bayes predicts argmax_c P(c) \* prod_i P(x_i | c): it picks the class maximising the posterior, using the 'naive' assumption that features are conditionally independent given the class. This factorises the joint likelihood into a product of per-feature likelihoods, making estimation trivial and fast. Despite the usually-false independence assumption, it works remarkably well, especially for text.

</details>

<details>
<summary><strong>Q23.</strong> Naive Bayes numeric: P(cavity)=0.2. P(toothache|cavity)=0.6, P(toothache|no cavity)=0.1. Find P(cavity|toothache).</summary>

P(toothache) = 0.6\*0.2 + 0.1\*0.8 = 0.12 + 0.08 = 0.20. P(cavity|toothache) = 0.6\*0.2 / 0.20 = 0.12/0.20 = 0.6. So observing a toothache raises the probability of cavity from 0.2 to 0.6.

</details>

<details>
<summary><strong>Q24.</strong> P(spam)=0.4, P(word|spam)=0.8, P(word|ham)=0.1. Find P(spam|word).</summary>

P(spam|word) = (0.4·0.8)/(0.4·0.8 + 0.6·0.1) = 0.32/0.38 = 0.842. The word strongly raises the spam probability.

</details>

<details>
<summary><strong>Q25.</strong> Why and how is Laplace (add-one) smoothing used in Naive Bayes?</summary>

If a feature value never occurs with a class in training, its estimated likelihood is 0, which zeroes the entire product and blocks that class regardless of other evidence. Laplace smoothing adds a pseudo-count: P(x_i|c) = (count + 1) / (total + V), where V is the number of possible values. This keeps every probability strictly positive and stabilises estimates from sparse data.

**Also asked as:** What is Laplace (add-one) smoothing and why is it needed?

**Short form:** It adds a small count (e.g. 1) to every feature-value/class tally so no probability is zero. Without it, a single unseen feature value makes the whole product P(x|c) zero, wrongly ruling out the class.

</details>

<details>
<summary><strong>Q26.</strong> How does Naive Bayes handle continuous features?</summary>

Gaussian Naive Bayes assumes each feature is normally distributed within each class: estimate the per-class mean and variance, then use the Gaussian density for P(x_i|c). Alternatively, discretise continuous features into bins and treat them as categorical. Gaussian NB is the common default for real-valued inputs.

</details>

<details>
<summary><strong>Q27.</strong> What is a Bayesian (belief) network?</summary>

A directed acyclic graph whose nodes are random variables and whose edges encode direct dependencies; each node carries a conditional probability table given its parents. It compactly represents a full joint distribution as a product of local conditionals, exploiting conditional independence. It supports probabilistic inference, computing any query probability given evidence, far more efficiently than the full joint table.

</details>

<details>
<summary><strong>Q28.</strong> What is the difference between generative and discriminative classifiers? Which is Naive Bayes?</summary>

Generative models learn P(x,c) (via P(c) and P(x|c)) and can generate data and apply Bayes' rule to classify; discriminative models learn P(c|x) or the boundary directly (logistic regression, SVM). Naive Bayes is generative. Generative models work with less data and handle missing features; discriminative models usually give higher accuracy when data is plentiful.

</details>

<details>
<summary><strong>Q29.</strong> Why is Naive Bayes a strong baseline despite its 'naive' assumption?</summary>

It needs little data, trains in one pass, is fast and robust to irrelevant features, and its class-ranking is often correct even when the independence assumption (and the exact probabilities) are wrong.

</details>

### Module 9: Ensemble learning

<details>
<summary><strong>Q30.</strong> What is ensemble learning and why does it improve accuracy?</summary>

An ensemble combines many base learners and aggregates their predictions (voting/averaging). If the learners are individually better than chance and make somewhat independent errors, their mistakes cancel on aggregation, reducing variance (and sometimes bias). The result is usually more accurate and robust than any single model, the 'wisdom of crowds' for models.

**Also asked as:** What is an ensemble, and why can it beat a single model?

**Short form:** It combines many models so their errors partly cancel. If models are accurate and diverse (make different mistakes), the aggregate is more accurate and lower-variance than any single one.

</details>

<details>
<summary><strong>Q31.</strong> Explain bagging and how it reduces variance.</summary>

Bagging (bootstrap aggregating) trains each base learner on a different bootstrap sample (sampled with replacement) of the data and averages/votes their outputs. Because the models see different data, their errors decorrelate, so averaging cuts variance without raising bias, most effective for high-variance learners like deep trees. Random Forests add feature subsampling on top.

</details>

<details>
<summary><strong>Q32.</strong> What is a Random Forest and what extra randomness does it add over bagging?</summary>

A Random Forest is bagging of decision trees plus, at each split, considering only a random subset of features (typically sqrt(p) for classification). This decorrelates the trees further, they can't all rely on the same dominant feature, lowering variance and improving generalisation. It also yields out-of-bag error estimates and feature-importance scores.

**Also asked as:** What extra randomness does a Random Forest add over plain bagging?

**Short form:** At each split it considers only a random subset of features, which decorrelates the trees so their averaged prediction has lower variance than bagged full-feature trees.

</details>

<details>
<summary><strong>Q33.</strong> Explain boosting and how it differs from bagging.</summary>

Boosting builds learners sequentially, each focusing on the examples the previous ones got wrong (by reweighting data or fitting residuals), then combines them as a weighted sum. Unlike bagging (parallel, independent, variance reduction), boosting is sequential and mainly reduces bias, turning weak learners into a strong one. It can overfit noisy data if unregularised.

**Also asked as:** Contrast bagging and boosting.

**Short form:** Bagging trains models in parallel on bootstrap samples and averages/votes, it mainly reduces variance (e.g. Random Forest). Boosting trains models sequentially, each focusing on the previous one's errors, it mainly reduces bias (e.g. AdaBoost, gradient boosting).

</details>

<details>
<summary><strong>Q34.</strong> How does AdaBoost work at a high level?</summary>

Start with equal weights on all training points. Repeat: train a weak learner on the weighted data; compute its weighted error; give it a say (alpha) that grows as error shrinks; increase the weights of misclassified points so the next learner focuses on them. The final classifier is the sign of the weighted vote sum(alpha_t \* h_t(x)). Points that stay hard get progressively more attention.

**Also asked as:** How does AdaBoost focus on hard examples?

**Short form:** It reweights the training data after each round, increasing the weight of misclassified points so the next weak learner concentrates on them; final prediction is a weighted vote of the weak learners.

</details>

<details>
<summary><strong>Q35.</strong> What is gradient boosting?</summary>

Gradient boosting fits each new learner to the negative gradient (for squared loss, the residuals) of the loss with respect to the current model's predictions, then adds it scaled by a learning rate. It is boosting generalised to any differentiable loss via functional gradient descent. Implementations like XGBoost/LightGBM add regularisation, shrinkage and column/row subsampling and are top performers on tabular data.

</details>

<details>
<summary><strong>Q36.</strong> What is stacking (stacked generalisation)?</summary>

Stacking trains several diverse base models, then trains a meta-learner on their out-of-fold predictions to learn the best way to combine them (rather than simple voting). The meta-model discovers which base learner to trust in which region of input space. Using out-of-fold predictions prevents the meta-learner from seeing leaked training labels.

</details>

<details>
<summary><strong>Q37.</strong> Bias-variance view: which ensemble reduces variance and which reduces bias?</summary>

Bagging/Random Forests mainly reduce variance (average many low-bias, high-variance trees) with little effect on bias. Boosting mainly reduces bias (sequentially correcting a high-bias weak learner) but can raise variance/overfit if run too long. Choosing between them follows from whether the base learner underfits (boost) or overfits (bag).

</details>

### Module 10: Unsupervised learning

<details>
<summary><strong>Q38.</strong> What is unsupervised learning? Give two task types.</summary>

Learning structure from unlabelled data, no target output. Main task types: clustering (group similar instances, e.g. k-means, hierarchical, DBSCAN) and dimensionality reduction (compress features while preserving structure, e.g. PCA). Others include density estimation and association-rule mining. The goal is to reveal patterns rather than predict a label.

</details>

<details>
<summary><strong>Q39.</strong> Describe the k-means algorithm.</summary>

Choose k. Initialise k centroids. Repeat until stable: (assignment) assign each point to its nearest centroid; (update) move each centroid to the mean of its assigned points. This iteratively minimises within-cluster sum of squared distances (inertia). It converges to a local optimum, so run several random initialisations (or k-means++) and keep the best.

**Also asked as:** Describe the k-means algorithm.

**Short form:** Choose k; initialise centroids; repeat: assign each point to its nearest centroid, then move each centroid to the mean of its assigned points, until assignments stop changing. It minimises within-cluster sum of squares.

</details>

<details>
<summary><strong>Q40.</strong> k-means numeric: 1-D points \{2,4,10,12,3,20,30,11,25\}, k=2, initial centroids c1=2, c2=4. Give the first assignment.</summary>

Assign each point to the nearer of 2 and 4 (ties by |.|): 2->c1; 3->c1 (|3-2|=1&lt;|3-4|=1? tie, goes c1 or c2, by rounding to c1); 4->c2; 10,11,12,20,25,30 all closer to 4 -> c2. So c1=\{2,3\}, c2=\{4,10,11,12,20,25,30\}. New centroids: c1=mean(2,3)=2.5, c2=mean(4,10,11,12,20,25,30)=16.0. Iteration continues from (2.5, 16.0).

</details>

:::note The tie in Q40
The point 3 is exactly halfway between the centroids 2 and 4, so a tie rule is needed. The answer above sends it to the first centroid, which gives the new centroids 2.5 and 16.0. If the tie went to the second centroid the new centroids would be 2.0 and 14.375. The code in this chapter prints both. Real implementations break ties by index, which is the first-centroid convention.
:::

<details>
<summary><strong>Q41.</strong> Compute the centroid of the cluster \{(1,1), (2,1), (1.5,2)\}.</summary>

Centroid = mean = ((1+2+1.5)/3, (1+1+2)/3) = (1.5, 1.33). k-means would move the centroid here after the assignment step.

</details>

<details>
<summary><strong>Q42.</strong> How do you choose the number of clusters k?</summary>

The elbow method: plot within-cluster SSE (inertia) vs k and pick the k where the curve bends (diminishing returns). The silhouette score measures how well each point fits its cluster vs the next-nearest (higher is better); pick the k maximising mean silhouette. Domain knowledge and the gap statistic also help. There is no single 'correct' k.

**Also asked as:** How do you choose k, and what is a limitation of k-means?

**Short form:** The elbow method (plot within-cluster SSE vs k, pick the elbow) or the silhouette score. Limitations: must pre-specify k, assumes spherical equal-size clusters, and is sensitive to initialisation and outliers.

</details>

<details>
<summary><strong>Q43.</strong> What are the limitations of k-means?</summary>

It needs k in advance, assumes roughly spherical, equal-size clusters (uses Euclidean distance), is sensitive to initialisation and to outliers (means shift), and only finds a local optimum. It struggles with non-convex or varying-density clusters. k-means++ helps initialisation; DBSCAN or spectral clustering handle non-spherical shapes.

</details>

<details>
<summary><strong>Q44.</strong> Contrast agglomerative hierarchical clustering with k-means.</summary>

Agglomerative clustering starts with each point as its own cluster and repeatedly merges the two closest clusters (by a linkage: single/complete/average/Ward), producing a dendrogram you can cut at any level, no need to fix k up front, and it captures nested structure. It is O(n^2) or worse, so it doesn't scale like k-means, but it needs no random init and handles non-spherical shapes better via linkage choice.

**Also asked as:** Contrast k-means with hierarchical clustering.

**Short form:** k-means is flat and needs k up front; hierarchical clustering builds a dendrogram (agglomerative merging or divisive splitting) that reveals structure at all granularities and doesn't require k in advance, but is costlier.

</details>

<details>
<summary><strong>Q45.</strong> How does DBSCAN work and what is its advantage over k-means?</summary>

DBSCAN grows clusters from dense regions: a point is 'core' if it has at least minPts neighbours within radius eps; core points and their reachable neighbours form clusters, and low-density points are labelled noise. Advantages: it finds arbitrary-shaped clusters, doesn't need k, and explicitly flags outliers. Weakness: sensitive to eps/minPts and struggles with clusters of very different densities.

</details>

<details>
<summary><strong>Q46.</strong> What does PCA do, and what does the first principal component maximise?</summary>

PCA finds an orthogonal set of directions (principal components) that capture the most variance in the data, letting you keep the top few and drop the rest for dimensionality reduction. The first principal component is the direction of maximum variance (equivalently, minimum reconstruction error); it is the top eigenvector of the covariance matrix. Components are uncorrelated and ordered by explained variance.

</details>

<details>
<summary><strong>Q47.</strong> What is the difference between clustering and classification?</summary>

Classification is supervised, it learns from labelled examples to assign predefined classes to new inputs. Clustering is unsupervised, it groups unlabelled data by similarity with no predefined classes and no ground-truth labels. Classification is evaluated against known labels; clustering is evaluated by internal measures (silhouette) or external ones only if labels happen to exist.

</details>

### Module 11: Model evaluation

<details>
<summary><strong>Q48.</strong> Define the confusion matrix and its four entries for binary classification.</summary>

A 2x2 table of predicted vs actual: True Positives (predicted +, actually +), True Negatives (predicted -, actually -), False Positives (predicted +, actually -, a 'false alarm'/Type I error), and False Negatives (predicted -, actually +, a 'miss'/Type II error). All standard metrics are computed from these four counts.

</details>

<details>
<summary><strong>Q49.</strong> Define accuracy, precision, recall and F1.</summary>

Accuracy = (TP+TN)/total. Precision = TP/(TP+FP): of predicted positives, how many are correct. Recall = TP/(TP+FN): of actual positives, how many were caught. F1 = 2\*P\*R/(P+R), the harmonic mean of precision and recall, rewarding a balance of the two. Precision matters when false alarms are costly; recall matters when misses are costly.

**Also asked as:** When is F1 preferred over accuracy, and what is it?

**Short form:** On imbalanced data. F1 = 2·precision·recall/(precision+recall), the harmonic mean, which stays low unless both precision and recall are high, unlike accuracy, which a majority-class predictor can inflate.

</details>

<details>
<summary><strong>Q50.</strong> Metric numeric: TP=40, FP=10, FN=20, TN=30. Compute accuracy, precision, recall, F1.</summary>

Accuracy = (40+30)/100 = 0.70. Precision = 40/(40+10) = 0.80. Recall = 40/(40+20) = 0.667. F1 = 2\*0.80\*0.667/(0.80+0.667) = 1.0672/1.467 = 0.727. So 70% accurate, precision 0.80, recall 0.67, F1 0.73.

</details>

<details>
<summary><strong>Q51.</strong> Why can accuracy be misleading? Give an example.</summary>

On imbalanced data, a trivial model can score high accuracy while being useless. If 99% of transactions are legitimate, a model that always predicts 'legit' is 99% accurate but catches zero fraud (recall 0). Precision, recall, F1, ROC-AUC or balanced accuracy reveal this; accuracy alone hides the failure on the minority class that usually matters most.

</details>

<details>
<summary><strong>Q52.</strong> What is the ROC curve and AUC?</summary>

The ROC curve plots true-positive rate (recall) against false-positive rate as the decision threshold varies, showing the full trade-off. AUC (area under the curve) summarises it in one number: the probability that the model ranks a random positive above a random negative. AUC=1 is perfect, 0.5 is random. ROC-AUC is threshold-independent and robust to class imbalance in ranking.

**Also asked as:** What do the ROC curve and AUC measure?

**Short form:** The ROC plots true-positive vs false-positive rate across thresholds; AUC (area under it) is a single threshold-independent score, 1.0 perfect, 0.5 random, good for comparing classifiers on imbalanced data.

</details>

:::note Correction to Q52 and the lecture
The answer says AUC is "robust to class imbalance". It is unchanged by the class mix, but it can still look good when positives are rare and the model is poor on them: with 2.5% positives one example model has ROC AUC 0.838 and average precision 0.575. When positives are rare, report the precision-recall curve too. The [evaluation chapter](/docs/theory/ml/model-evaluation) shows the numbers.
:::

<details>
<summary><strong>Q53.</strong> Explain the bias-variance tradeoff.</summary>

Expected error decomposes into bias^2 + variance + irreducible noise. Bias is error from overly simple assumptions (underfitting); variance is error from sensitivity to the particular training set (overfitting). Increasing model complexity lowers bias but raises variance. The best model minimises their sum, 'as simple as possible, but no simpler'.

**Also asked as:** Define the bias–variance trade-off.

**Short form:** Bias is error from an over-simple model (underfitting); variance is error from over-sensitivity to the training set (overfitting). Total error = bias² + variance + irreducible noise; reducing one often raises the other.

</details>

<details>
<summary><strong>Q54.</strong> What is k-fold cross-validation and why use it?</summary>

Split the data into k folds; train on k-1 and validate on the held-out fold, rotating so every fold is validated once; average the k scores. It uses data efficiently and gives a lower-variance, less lucky estimate of generalisation than a single train/test split, and is the standard way to tune hyperparameters. Stratified k-fold preserves class proportions per fold.

**Also asked as:** Why is a single train/test split not enough, and what does k-fold cross-validation do?

**Short form:** One split gives a high-variance estimate that depends on the lucky/unlucky partition. k-fold CV splits data into k folds, trains on k−1 and tests on the held-out fold k times, and averages, a more reliable, lower-variance estimate.

</details>

<details>
<summary><strong>Q55.</strong> How do you detect and remedy overfitting?</summary>

Detect it when training accuracy is high but validation/test accuracy is much lower, or the gap widens with training. Remedies: get more data, simplify the model, add regularisation (L1/L2, dropout, tree depth limits), use early stopping, apply cross-validation for honest tuning, and use ensembles (bagging). The aim is to close the train-validation gap while keeping validation error low.

</details>

## Further reading

- [An Introduction to Statistical Learning](https://www.statlearning.com/): free PDFs of the R and Python editions, a friendly route through almost every module in this bank.
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html): k-NN, SVMs, naive Bayes, ensembles, clustering and the metrics, each with runnable examples.
- [Google Machine Learning Crash Course: classification metrics](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall): a short refresher on accuracy, precision and recall.
- Built from the course question banks "ml-question-bank" and "ml-comprehensive-question-bank" (Lecture Library series).

## Check yourself

- I can answer the conceptual questions of modules 6 to 11 without looking, and explain the contrasts (lazy against eager, bagging against boosting, k-means against hierarchical clustering).
- I can work a k-NN vote, an SVM margin, a Bayes update, one k-means step and a confusion matrix by hand, and check each in code.
- I can say what changes in an answer when a tie, a skewed class mix or a small sample is involved.
