---
id: ml-regression
title: "Regression and Gradient Descent"
sidebar_label: "Regression and gradient descent"
sidebar_position: 1
slug: /theory/ml/regression-and-gradient-descent
description: "Fit a line two ways, with the normal equation or by walking downhill with gradient descent, then tune the learning rate, scale the features and diagnose under and overfitting."
tags: [regression, linear-regression, gradient-descent, normal-equation, learning-rate, feature-scaling]
---

import Infographic from '@site/src/components/Infographic';
import GradientDescentLab from '@site/src/components/viz/GradientDescentLab';

**In one line.** Fitting a line is an optimisation problem, and the habit of stepping against the gradient is the engine under almost all of machine learning.

## The idea in plain words

A regression model predicts a number. The simplest kind multiplies each input by a weight and adds the results: $\hat y = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d$. Training means choosing the weights $\theta$ so that the predictions land as close as possible to the answers you already have.

"Close" needs a number, so we define a **cost**: the average squared miss, $J(\theta) = \frac{1}{2m}\sum_i (\hat y_i - y_i)^2$. Squaring makes every miss positive and punishes big misses harder than small ones. The half is there only so that the derivative comes out clean. Plot the cost against the weights and you get a smooth bowl with exactly one lowest point. Learning is the search for that point.

There are two ways to find the bottom of a bowl.

- **Solve for it.** The bowl is a quadratic, so calculus hands you the bottom in one formula, the *normal equation*. It is exact on a healthy problem, and the textbook form of it uses a matrix inverse, which becomes expensive when there are many features and breaks when two features are copies of each other. Libraries solve the same least-squares problem with a more stable factorisation instead of forming that inverse, as block 2 shows.
- **Walk down it.** Stand somewhere on the bowl, feel which way is downhill, take a step, repeat. That is *gradient descent*. Picture a hiker in thick fog on a hillside. She cannot see the valley, but she can feel the slope under her boots, so she steps downhill and checks again. The length of her stride is the **learning rate**: too timid and she is still walking at nightfall, too bold and she leaps across the valley floor and lands higher on the far slope than where she started.

Gradient descent matters far beyond straight lines. Swap the single bowl for a bumpy landscape with millions of weights and the very same loop trains a neural network. Regression is where you meet it in a form small enough to compute by hand, and the next two chapters (classification, then trees) reuse the vocabulary you learn here: a model, a cost, and a rule for lowering the cost.

<Infographic src="/img/ml/regression-and-gradient-descent-two-routes.svg" alt="A normal-equation panel beside a gradient descent panel, both reaching theta 0.7857 on the three-point example" caption="Two routes to the same answer. The numbers on the right are reproduced by the first code block below." />

<Infographic src="/img/ml/regression-and-gradient-descent-rate-and-scaling.svg" alt="Three bowls showing a crawling, a converging and a diverging learning rate, beside stretched and round cost contours" caption="Learning rate and feature scaling. Every figure comes from the third and fourth code blocks below." />

## How it works

### The linear model

ŷ = θ₀ + θ₁x₁ + … ; cost J(θ) = (1/2m) Σ (ŷ − y)². The ½m makes the derivative clean.

:::note

**Two routes to the minimum:** a closed-form formula, or iterative descent.

:::

### The normal equation

θ = (XᵀX)⁻¹Xᵀy: exact, no learning rate, no iteration.

:::tip

**Trade-off.** The (XᵀX)⁻¹ inverse costs O(d³), so it's impractical with very many features, where gradient descent wins. In practice, numerical libraries do not form the inverse at all: they solve the same problem with a QR or SVD factorisation.

:::

### Walking downhill

Initialise θ; compute ∂J/∂θ = (1/m)Σ(ŷ−y)x; step θ ← θ − α·∂J/∂θ; repeat until convergence.

:::tip

**Worked first step.** θ=0, α=0.1: ∂J/∂θ = (1/3)[(−1)1+(−2)2+(−2)3] = −3.667 → θ ← **0.367**.

:::

### Scaling, learning rate & diagnosis

- **Learning rate α**: Too large → diverges/oscillates; too small → crawls. Watch J each step to tune it.
- **Feature scaling**: Rounds the stretched cost bowl so descent heads straight to the minimum (normal equation needs no scaling).

:::note

**Diagnose the fit.** Underfit (high bias): poor on train & test → add complexity. Overfit (high variance): great on train, poor on test → add data / simplify / regularise.

:::

### Key takeaways

- **1 · Cost**: J = (1/2m)Σ(ŷ−y)², a convex bowl.
- **2 · Two solvers**: Normal equation (exact) vs gradient descent (scalable).
- **3 · Tuning**: Learning rate + scaling; diagnose under/overfit.

:::note

**The thread.** Regression is the template for supervised learning: a model, a convex cost, and a rule to minimise it. Gradient descent (step against the gradient) is the same engine that trains deep networks.

:::

## A real system that works this way

**scikit-learn's `LinearRegression`** is the everyday case. It does not form $(X^\top X)^{-1}$ at all: its documentation says it fits by least squares through a singular value decomposition of $X$, with a cost that grows with the number of rows times the square of the number of features. That is the lecture's trade-off, already made for you: an exact solve is perfectly good until the data gets large. When it no longer fits comfortably in memory, the same user guide points to `SGDRegressor`, which runs the walk-downhill route one small batch at a time and supports `partial_fit`, so you can stream data through it.

**A pricing or demand baseline** is the pattern that repeats in industry. Before any elaborate forecaster, fit a regression on a handful of features (area, age, day of week, price) and write down its error. Everything built later has to beat that number. It is cheap, its coefficients can be read aloud to a business owner, and when it is badly wrong the residuals usually show which feature is missing.

## Code you can run

Six short blocks, each under a second. Run them in order; the numbers in the prose are the numbers they print.

### 1. The lecture's worked step, then the whole walk

The data are $(1,1)$, $(2,2)$, $(3,2)$ and the model is $\hat y = \theta x$ with no intercept, so there is one weight and the cost is a plain parabola. Starting at $\theta = 0$ with $\alpha = 0.1$, the lecture computes a gradient of $-3.667$ and a new weight of $0.367$. The block prints the lecture's figures next to its own, then keeps stepping.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 2.0])
m = len(x)

def cost(theta):
    return np.sum((theta * x - y) ** 2) / (2 * m)

def gradient(theta):
    return np.sum((theta * x - y) * x) / m

theta, alpha = 0.0, 0.1
g = gradient(theta)
print(f"gradient at theta=0 : {g:.3f}   (lecture: -3.667)")
print(f"theta after step 1  : {theta - alpha * g:.3f}    (lecture: 0.367)")
print(f"cost J(0) = {cost(0.0):.3f}, J(0.367) = {cost(theta - alpha * g):.3f}\n")

for step in range(1, 61):
    theta -= alpha * gradient(theta)
    if step in (1, 2, 5, 10, 20, 60):
        print(f"step {step:2d}: theta = {theta:.4f}   J = {cost(theta):.5f}")

closed_form = (x @ y) / (x @ x)
print(f"\nclosed form  sum(xy)/sum(x^2) = {closed_form:.4f}   J = {cost(closed_form):.5f}")
```

What it prints:

```text
gradient at theta=0 : -3.667   (lecture: -3.667)
theta after step 1  : 0.367    (lecture: 0.367)
cost J(0) = 1.500, J(0.367) = 0.469

step  1: theta = 0.3667   J = 0.46926
step  2: theta = 0.5622   J = 0.17607
step  5: theta = 0.7518   J = 0.06221
step 10: theta = 0.7843   J = 0.05953
step 20: theta = 0.7857   J = 0.05952
step 60: theta = 0.7857   J = 0.05952

closed form  sum(xy)/sum(x^2) = 0.7857   J = 0.05952
```

Both lecture numbers reproduce. By step 10 the weight is 0.7843, by step 20 it sits on the closed-form answer $\sum xy / \sum x^2 = 11/14 = 0.7857$, and the cost has settled at 0.0595. After that the steps are too small to see: the gradient shrinks as you near the bottom, so gradient descent slows itself down without any help.

### Try it: gradient descent, one step at a time

The lab below is the lecture's "run gradient descent step by step and watch the cost fall" widget. Its defaults (learning rate 0.10, one step, start at 0) reproduce block 1: gradient $-3.667$, $\theta = 0.3667$, $J = 0.4693$. Drag the learning rate to 0.45 and the steps to 20 and you get block 3's divergence ($\theta = -4.5002$). Switch the mode to *Two weights, scaling* for the board above in motion: with *standardise features* off it needs 72 steps, with it on, 4 (block 4).

<GradientDescentLab />

### 2. The exact route, and why not to invert

Here the normal equation meets a real dataset: 442 patients, ten features, and a column of ones for the intercept. Three ways of solving it are compared with scikit-learn.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
m, d = X.shape
A = np.column_stack([np.ones(m), X])

theta_inverse = np.linalg.inv(A.T @ A) @ A.T @ y
theta_solve = np.linalg.solve(A.T @ A, A.T @ y)
theta_lstsq, *_ = np.linalg.lstsq(A, y, rcond=None)
sk = LinearRegression().fit(X, y)
theta_sk = np.r_[sk.intercept_, sk.coef_]

print(f"{m} rows, {d} features, design matrix {A.shape}")
for name, t in [("inverse", theta_inverse), ("solve", theta_solve), ("lstsq", theta_lstsq)]:
    print(f"{name:8s} max |difference from sklearn| = {np.max(np.abs(t - theta_sk)):.2e}")
print(f"intercept {theta_sk[0]:.2f}, first three coefficients {np.round(theta_sk[1:4], 1)}")

B = np.column_stack([A, A[:, 1]])
print(f"\nduplicate a column: rank of X'X = {np.linalg.matrix_rank(B.T @ B)} of {B.shape[1]}")
try:
    np.linalg.solve(B.T @ B, B.T @ y)
    print("solve: no error raised")
except np.linalg.LinAlgError as e:
    print("solve raised:", e)
theta_dup, *_ = np.linalg.lstsq(B, y, rcond=None)
print(f"lstsq still returns an answer, residual sum of squares = {np.sum((B @ theta_dup - y) ** 2):,.0f}")
print(f"same as before the duplicate                          = {np.sum((A @ theta_sk - y) ** 2):,.0f}")
```

What it prints:

```text
442 rows, 10 features, design matrix (442, 11)
inverse  max |difference from sklearn| = 1.16e-10
solve    max |difference from sklearn| = 1.41e-10
lstsq    max |difference from sklearn| = 9.09e-13
intercept 152.13, first three coefficients [ -10.  -239.8  519.8]

duplicate a column: rank of X'X = 11 of 12
solve raised: Singular matrix
lstsq still returns an answer, residual sum of squares = 1,263,986
same as before the duplicate                          = 1,263,986
```

The explicit inverse, `solve` and `lstsq` all agree with `LinearRegression` to about $10^{-10}$, so on a healthy problem the choice is about speed and habit. The second half is the lesson. Duplicating a column makes $X^\top X$ rank-deficient (rank 11 out of 12), and the formula has no unique answer. Whether `solve` raises or quietly returns something numerically meaningless depends on rounding, and the exact message can vary by machine. `lstsq` returns the minimum-norm answer and the same residual error as before the duplicate, which is why library solvers are built on it.

### 3. The learning rate

Same data as block 1. For this problem the curvature of the cost is $\sum x^2 / m = 4.667$, and gradient descent is stable only when $\alpha < 2/4.667 = 0.4286$. Four rates, twenty steps each:

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 2.0])
m = len(x)
cost = lambda t: np.sum((t * x - y) ** 2) / (2 * m)
best = (x @ y) / (x @ x)
curvature = np.sum(x * x) / m
print(f"cost curvature sum(x^2)/m = {curvature:.3f}")
print(f"gradient descent is stable only for alpha < 2/curvature = {2 / curvature:.4f}")
print(f"and steps straight in without overshooting for alpha <= 1/curvature = {1 / curvature:.4f}\n")

print(" alpha    after 5 steps   after 20 steps   J after 20   verdict")
for alpha in (0.01, 0.1, 0.4, 0.45):
    theta, trail = 0.0, []
    for _ in range(20):
        theta -= alpha * np.sum((theta * x - y) * x) / m
        trail.append(theta)
    if cost(trail[-1]) > cost(0.0):
        verdict = "diverges"
    elif cost(trail[-1]) - cost(best) > 0.1:
        verdict = "crawls"
    elif max(trail) > best:
        verdict = "overshoots, then settles"
    else:
        verdict = "converges"
    print(f"{alpha:6.2f}   {trail[4]:12.4f}   {trail[-1]:14.4f}   {cost(trail[-1]):10.4f}   {verdict}")

alpha, theta, steps = 0.1, 0.0, 0
while cost(theta) - cost(best) > 1e-6:
    theta -= alpha * np.sum((theta * x - y) * x) / m
    steps += 1
print(f"\nalpha = 0.1 gets within 1e-6 of the minimum cost after {steps} steps")
```

What it prints:

```text
cost curvature sum(x^2)/m = 4.667
gradient descent is stable only for alpha < 2/curvature = 0.4286
and steps straight in without overshooting for alpha <= 1/curvature = 0.2143

 alpha    after 5 steps   after 20 steps   J after 20   verdict
  0.01         0.1670           0.4836       0.2725   crawls
  0.10         0.7518           0.7857       0.0595   converges
  0.40         1.1699           0.7408       0.0642   overshoots, then settles
  0.45         2.0511          -4.5002      65.2544   diverges

alpha = 0.1 gets within 1e-6 of the minimum cost after 12 steps
```

Read the four rows as the lecture's two failure modes and its good case, plus one more it does not draw. 0.01 *crawls*: after 20 steps it has covered only about 60% of the distance (0.4836 of the 0.7857 it needs). 0.10 *converges*. 0.45 sits just outside the limit and *diverges*: every step overshoots by more than it corrected, so the error grows (cost 65 and rising). And 0.40, which is inside the limit but above $1/4.667 = 0.2143$, overshoots to the far side every single step before settling. That sawtooth is what an oscillating loss curve looks like in practice.

### 4. Scaling, measured

Two features on very different scales (area in square metres, age in years), centred so that the intercept is handled. Each run uses its own best simple rate, $\alpha = 1/\lambda_{\max}$, and counts steps until 99.9% of the gap to the minimum is closed.

```python
import numpy as np

X = np.array([[60, 12], [75, 3], [90, 18], [105, 6], [120, 15], [135, 2], [150, 9], [165, 20]], dtype=float)
y = np.array([180, 260, 235, 310, 300, 395, 380, 390], dtype=float)

def gradient_descent(Z, y, tolerance=1e-3):
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    m = len(Z)
    H = Zc.T @ Zc / m
    b = Zc.T @ yc / m
    best = np.linalg.solve(H, b)
    eig = np.linalg.eigvalsh(H)
    alpha = 1 / eig[-1]
    gap = lambda t: 0.5 * (t - best) @ H @ (t - best)
    theta, steps = np.zeros(2), 0
    start = gap(theta)
    while gap(theta) > tolerance * start:
        theta = theta - alpha * (H @ theta - b)
        steps += 1
    return eig, alpha, steps

for name, Z in [("raw features (m2, age)", X), ("standardised", (X - X.mean(axis=0)) / X.std(axis=0))]:
    eig, alpha, steps = gradient_descent(Z, y)
    print(f"{name:24s} eigenvalues {eig[0]:8.2f} {eig[1]:8.2f}  condition {eig[1] / eig[0]:5.2f}"
          f"  largest stable alpha {2 / eig[1]:.4f}  steps to 0.1% of the gap: {steps}")
```

What it prints:

```text
raw features (m2, age)   eigenvalues    38.29  1182.95  condition 30.90  largest stable alpha 0.0017  steps to 0.1% of the gap: 72
standardised             eigenvalues     0.80     1.20  condition  1.51  largest stable alpha 1.6629  steps to 0.1% of the gap: 4
```

The *condition number* (largest curvature over smallest) is the stretch of the bowl. Raw features give 30.90 and need 72 steps; standardised ones give 1.51 and need 4. Note also the largest stable rate: 0.0017 for raw features against 1.6629 once standardised. Unscaled, one learning rate has to suit both a feature measured in the hundreds and one measured in tens, so the biggest feature sets the speed limit.

### 5. Diagnosing the fit

Thirty noisy points from a sine wave, fitted with polynomials of growing flexibility, then scored on 500 fresh points. The noise has variance $0.3^2 = 0.09$, so no model can honestly score a test error much below that.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)
def sample(n):
    x = rng.uniform(0, 1, n)
    return x.reshape(-1, 1), np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)

X_train, y_train = sample(30)
X_test, y_test = sample(500)

print("model                        train MSE   test MSE   reading")
configs = [
    ("degree 1 (a straight line)", LinearRegression(), 1, "underfit: both errors high"),
    ("degree 4", LinearRegression(), 4, "about right"),
    ("degree 15", LinearRegression(), 15, "overfit: train low, test high"),
    ("degree 15 + ridge alpha=0.1", Ridge(alpha=0.1), 15, "regularised: gap shrinks"),
    ("degree 15 + ridge alpha=1", Ridge(alpha=1.0), 15, "too much: underfitting again"),
]
for name, est, degree, reading in configs:
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), est).fit(X_train, y_train)
    tr = mean_squared_error(y_train, model.predict(X_train))
    te = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name:27s} {tr:9.3f} {te:10.3f}   {reading}")
print(f"\nnoise floor (variance of the added noise) = {0.3 ** 2:.3f}")
```

What it prints:

```text
model                        train MSE   test MSE   reading
degree 1 (a straight line)      0.370      0.298   underfit: both errors high
degree 4                        0.097      0.094   about right
degree 15                       0.040      0.138   overfit: train low, test high
degree 15 + ridge alpha=0.1     0.079      0.108   regularised: gap shrinks
degree 15 + ridge alpha=1       0.176      0.149   too much: underfitting again

noise floor (variance of the added noise) = 0.090
```

<Infographic src="/img/ml/regression-and-gradient-descent-fit-diagnosis.svg" alt="Three polynomial fits to noisy sine data, underfit, about right and overfit, with a table of train and test errors" caption="Diagnosing the fit. Every number is printed by block 5." />

The degree-1 line is wrong everywhere (0.370 train, 0.298 test): high bias. Degree 4 sits at 0.097 and 0.094, right at the noise floor. Degree 15 chases the noise: its training error falls to 0.040 while its test error climbs to 0.138, the high-variance signature. Ridge pulls it back (0.108 test), and a ridge penalty that is too heavy (alpha 1) overshoots into underfitting again.

### 6. The walk-downhill route at library scale

`SGDRegressor` is gradient descent in production clothing. It needs scaled inputs, and the two answers should nearly agree.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

exact = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-6, random_state=0)).fit(X_train, y_train)

print(f"exact least squares  test R^2 = {exact.score(X_test, y_test):.4f}")
print(f"SGDRegressor         test R^2 = {sgd.score(X_test, y_test):.4f}")
gap = np.mean(np.abs(exact.predict(X_test) - sgd.predict(X_test)))
print(f"mean absolute difference between the two sets of predictions = {gap:.2f} (target std = {y.std():.1f})")
```

What it prints:

```text
exact least squares  test R^2 = 0.3594
SGDRegressor         test R^2 = 0.3508
mean absolute difference between the two sets of predictions = 1.98 (target std = 77.0)
```

The predictions differ by about 2 on a target whose standard deviation is 77, and the test $R^2$ values are within 0.01 of each other (0.3594 exact, 0.3508 stochastic). The two routes find the same line; one just gets there in small noisy steps.

## Designing with it

### Which solver to reach for

| Situation | Reach for | Why |
| --- | --- | --- |
| Rows and features fit in memory, few thousand features at most | `LinearRegression` (SVD or `lstsq`) | Exact, nothing to tune |
| Correlated features, or more features than rows | `Ridge` (or `Lasso`) | Stabilises the coefficients; plain least squares splits the weight arbitrarily |
| Millions of rows, or data arriving as a stream | `SGDRegressor` on scaled features | One mini-batch at a time, bounded memory |
| A custom loss, or a neural network | Gradient descent in a framework | There is no closed form to solve |

### A checklist before you trust a fit

1. **Scale first** whenever you use gradient descent or a penalty. Without it the learning rate is dictated by your largest feature.
2. **Plot the cost at every step.** A steadily falling curve means the rate is fine. A curve that rises or oscillates means it is too large. A curve that is still falling at the end means you stopped early or it is too small.
3. **Compare training and test error** (the diagnosis table above). Both high is underfitting, a wide gap is overfitting.
4. **Look at the residuals** against the predictions. A curve means a missing non-linear term, a fan shape means the target wants a log or similar transform.
5. **Beat the dumb baseline.** Predicting the training mean gives an $R^2$ of about zero, so any model must clear that, and clear it on held-out data.

:::warning Do not invert $X^\top X$ in production
Forming the inverse squares the sensitivity of the problem to rounding error and fails outright when two features are copies of each other (block 2 shows the failure). Use `numpy.linalg.lstsq`, a QR or SVD route, or just `LinearRegression`.
:::

:::note Beyond the lecture: what scaling really does
The lecture says scaling "rounds" the bowl. It makes it rounder, and the second board shows that precisely: the condition number falls from 30.90 to 1.51 and the step count from 72 to 4. It does not make the bowl a perfect circle here, because area and age are mildly correlated. A perfectly straight run to the minimum would need uncorrelated features as well.
:::

:::note Beyond the lecture: three flavours of gradient descent
The lecture computes the gradient over all rows at every step (*batch* gradient descent). Two cheaper variants dominate in practice.

| Variant | Rows per step | Behaviour |
| --- | --- | --- |
| Batch | All $m$ | Smooth, exact gradient; slow per step on big data |
| Stochastic (SGD) | 1 | Cheap and noisy; the noise can even help escape flat spots |
| Mini-batch | Typically tens to a few thousand | The compromise every deep-learning library uses |

Block 6 runs `SGDRegressor` and lands within a whisker of the exact answer.
:::

:::note Beyond the lecture: why squared error?
If you assume the noise on each target is Gaussian, minimising squared error is the same thing as maximum likelihood. That is the justification. Its weakness is that a single wild point contributes its miss *squared*, so outliers drag the line towards them. When outliers are a fact of life, a robust loss (absolute error or Huber loss) is the usual answer.
:::

:::note Beyond the lecture: regularisation in one paragraph
The lecture's cure for overfitting ends with "regularise". Ridge regression adds $\alpha\sum\theta_j^2$ to the cost, which shrinks all weights and keeps them from chasing noise. Lasso adds $\alpha\sum|\theta_j|$, which also drives some weights exactly to zero. In the diagnosis table, ridge with $\alpha = 0.1$ cuts the degree-15 test error from 0.138 to 0.108, while $\alpha = 1$ is too strong and the model underfits again. $\alpha$ is a hyperparameter you choose on validation data.
:::

## Where this stands in 2026

:::info Industry view

- A linear or ridge model is still the first thing to fit on any tabular problem. It is the baseline that gradient-boosted trees and neural networks have to justify themselves against.
- The training loop in this chapter, with mini-batches and adaptive step sizes, is the loop that trains today's neural networks. Learning-rate schedules and warm-up are the same learning-rate idea, tuned.
- For tabular regression on medium-sized data, benchmark work (Grinsztajn, Oyallon and Varoquaux, 2022) found tree-based models still ahead of deep learning at around ten thousand rows. A tabular foundation model, TabPFN (Nature, January 2025), reports beating tuned baselines on datasets up to 10,000 samples and 500 features. Both are reported results on their own benchmarks, not a guarantee for your data.
- The practical failure is seldom the algebra. It is unscaled features, a leaked target, or a held-out set that looks nothing like production.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Write the linear-regression cost and explain the 1/2m factor.</summary>

J(θ) = (1/2m) Σ (ŷᵢ − yᵢ)². Dividing by m averages over examples; the ½ cancels the 2 from differentiating the square, giving a clean gradient.<br /><em>Module 3 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> State the normal equation and its main limitation.</summary>

θ = (XᵀX)⁻¹Xᵀy: exact and iteration-free, but the inverse costs O(d³), so it is impractical when there are very many features.<br /><em>Module 3 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> List the steps of gradient descent and give the linear-regression gradient.</summary>

Initialise θ; compute the gradient; update θ ← θ − α∇J; repeat until convergence. Gradient: ∂J/∂θⱼ = (1/m) Σ (ŷᵢ − yᵢ) xᵢⱼ.<br /><em>Module 3 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Data (1,1),(2,2),(3,2), ŷ=θx, θ=0, α=0.1. Compute one gradient-descent step.</summary>

∂J/∂θ = (1/3)[(0−1)1 + (0−2)2 + (0−2)3] = (1/3)(−11) = −3.667; θ ← 0 − 0.1(−3.667) = 0.367.<br /><em>Module 3 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What does the learning rate control, and how do you know it's set wrong?</summary>

It sets the step size. Too large → the cost oscillates or diverges; too small → convergence is very slow. Plot J per iteration: a steadily decreasing curve means α is good.<br /><em>Module 3 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> Why does feature scaling speed up gradient descent but not the normal equation?</summary>

Unequal feature ranges make the cost bowl a stretched ellipse, so descent zig-zags; scaling rounds the bowl for a direct path. The normal equation solves it in one shot, so geometry doesn't matter.<br /><em>Module 3 · conceptual</em>

</details>

<details>
<summary><strong>Q7.</strong> How do you diagnose underfitting vs overfitting from errors?</summary>

Underfitting (high bias): high error on both train and test → add complexity. Overfitting (high variance): low train but high test error → add data, simplify, or regularise.<br /><em>Module 3 · conceptual</em>

</details>

## Further reading

- [scikit-learn user guide: Linear models](https://scikit-learn.org/stable/modules/linear_model.html) covers least squares, Ridge, Lasso and SGD, and is the primary reference for the code in this chapter.
- [Google Machine Learning Crash Course: Linear regression](https://developers.google.com/machine-learning/crash-course/linear-regression) is a short, visual module on loss, gradient descent and hyperparameters.
- The linear regression chapter of *An Introduction to Statistical Learning* (linked below) develops the statistics this chapter skips, such as standard errors and hypothesis tests.
- [Why do tree-based models still outperform deep learning on tabular data? (Grinsztajn et al., 2022)](https://arxiv.org/abs/2207.08815) is the benchmark behind the industry note.
- [Accurate predictions on small data with a tabular foundation model (Hollmann et al., Nature, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/) is the open-access TabPFN paper.
- Built from the course lecture "ml-m3-regression" (Lecture Library series).

- **[An Introduction to Statistical Learning](https://www.statlearning.com/)** `book`
  James, Witten, Hastie & Tibshirani: The friendliest rigorous intro to ML, free PDF plus R/Python labs.
- **[Stanford CS229 (Machine Learning)](https://cs229.stanford.edu/)** `course`
  Andrew Ng, Stanford: The rigorous derivations behind SVMs, GLMs, EM and learning theory.
- **[StatQuest](https://statquest.org/video-index/)** `▶ video`
  Josh Starmer: Short, wonderfully clear videos that build intuition step by step.

## What you should now be able to do

- [ ] I can explain why the cost has a factor of one half in front of the mean squared error.
- [ ] I can compute one step of gradient descent by hand and reproduce the gradient of -3.667 and the new weight of 0.367.
- [ ] I can explain why the normal equation needs no learning rate but is a poor choice with very many features, and why I should not invert $X^\top X$ myself.
- [ ] I can read a loss curve and say whether the learning rate is too small, about right or too large, and I know the stability limit for a quadratic cost.
- [ ] I can explain with a condition number why feature scaling speeds up gradient descent but is irrelevant to the normal equation.
- [ ] I can diagnose underfitting and overfitting from training and test error, and name two remedies for each.
