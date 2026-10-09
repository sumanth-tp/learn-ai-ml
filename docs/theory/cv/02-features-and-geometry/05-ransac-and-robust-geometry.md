---
id: cv-ransac-and-robust-geometry
title: "Computer Vision · Session 8; RANSAC and Robust Geometry"
sidebar_label: "5 · RANSAC geometry"
sidebar_position: 5
slug: /theory/cv/ransac-and-robust-geometry
description: "Understand consensus fitting, residual thresholds and the corrected RANSAC iteration count for the lecture exercise."
tags: [computer-vision, ransac, geometry, outliers]
---

import Infographic from '@site/src/components/Infographic';
import RansacIterationsLab from '@site/src/components/viz/RansacIterationsLab';

**In one line.** RANSAC repeatedly fits a model from small samples and keeps the one supported by the largest consistent subset.

:::tip Before you start
**You should already know**

- How SIFT produces candidate matches, some of them wrong: [SIFT keypoints and descriptors](/docs/theory/cv/sift-keypoints-and-descriptors).
- What least-squares line fitting does: [regression and gradient descent](/docs/theory/ml/regression-and-gradient-descent).
- The meaning of a probability of "at least one" success in repeated independent tries.

**Reading time.** About 45 minutes, plus a few seconds to run the code.

**After this chapter you can**

- explain why least squares fails with outliers and what RANSAC does instead,
- compute the number of trials from the inlier fraction, sample size and confidence, and round it correctly,
- say what the 99% in that formula does and does not promise, using measured numbers.
:::

## In 30 seconds

Imagine drawing a straight line through a scatter of points where half of them were typed in wrongly. An average-based fit gets dragged towards the bad points. RANSAC takes a different route: pick two points at random, draw the line through them, and count how many other points agree. Do that many times and keep the line with the most agreement. If half the points are good, a random pair is good one time in four, so you do not need many tries to get at least one good pair. This chapter checks how many tries are enough, and shows that a good pair does not always give a good line.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Outlier | A data point that does not follow the model | A wrong feature match |
| Inlier | A point whose error is below the threshold | Within 3 pixels of the fitted transform |
| Minimal sample | The fewest points that fix the model | 2 for a line, 4 for a homography |
| Residual | How far a point is from the model | 0.4 pixels |
| Threshold | The residual below which a point counts as an inlier | 3 pixels |
| Inlier fraction $w$ | Share of the data that is inlier | 0.5 |
| Consensus | The set of inliers a candidate model gathers | 5 points |
| Homography | A 3 by 3 matrix that maps points between two views of a plane | Four corners of a poster |

## The idea in plain words

:::note Added to the course material

The exhaustive toy fit, probability correction and failure analysis go beyond the course notes. The least-squares motivation, the loop, the examples and the questions remain below.

:::

Feature matches are imperfect. A descriptor can pair two visually similar but unrelated patches, especially in repeated texture or under occlusion. If all matches are fed into ordinary least squares, a few gross errors can pull a fitted line or geometric transform far from the relationship supported by most correct correspondences. Session 8 introduces RANSAC, random sample consensus, as a way to fit a model despite such outliers. The core question is not “which model minimises every error?” but “which model has the strongest set of data points whose errors are acceptably small?”

One RANSAC trial draws a minimal sample, fits a candidate model, computes a residual for every data point, and calls points within a chosen threshold inliers. A line can be determined by two distinct points; a projective homography needs four suitable point correspondences. The algorithm keeps the candidate with the best consensus according to its score. After choosing an inlier set, it commonly refits the model using all those inliers, because the minimal sample gives a noisy estimate. The threshold defines what “consistent” means in the units of the residual, such as pixels of reprojection error.

The sampling step is probabilistic. If an individual match has independent inlier probability $w$ and a minimal model requires $s$ matches, the chance that one sampled set contains only inliers is $w^s$. The chance of failing to draw such a set in $N$ independent trials is $(1-w^s)^N$. To make the chance of at least one all-inlier set at least $p$, solve $1-(1-w^s)^N\ge p$. This gives $N\ge\log(1-p)/\log(1-w^s)$. Since trials are whole numbers, **round the result up**. These probabilities are idealised: real samples are not always independent, the inlier fraction may be unknown and some all-inlier samples can be geometrically degenerate.

For the standard example $w=0.5$, $s=2$, $p=0.99$, the continuous result is about **16.008**, so at least **17 whole trials** are required. The textbook answer rounds to 16, but 16 gives about 98.998% under the stated model, just below the 99% target. At $w=0.3$, the continuous value is about 48.830 and the integer count is **49**. Lower inlier fractions or larger minimal sets make a successful sample rarer, causing the trial budget to rise quickly. This formula answers a confidence-budget question; it does not guarantee the selected model is correct if the residual threshold or geometric model is wrong.

<Infographic src="/img/cv/ransac-consensus.svg" alt="RANSAC samples a minimal model, scores residuals and refits inliers; the lecture's 99 percent example requires 17 whole trials rather than 16." caption="The probability calculation must be rounded up and still rests on sampling assumptions." />

## Worked example, step by step

Seven points: five lie exactly on $y=2x+1$, and two are wrong. The points are $(0,1)$, $(1,3)$, $(2,5)$, $(3,7)$, $(4,9)$, $(1,9)$ and $(3,-3)$.

**Least squares.**

1. The mean of $x$ is 2 and the mean of $y$ is $31/7=4.429$.
2. The slope is $\sum(x-\bar x)(y-\bar y)\,/\,\sum(x-\bar x)^2=8/12=0.667$.
3. The intercept is $4.429-0.667\times2=3.095$.

So least squares reports $y=0.667x+3.095$ when the truth is $y=2x+1$. Two bad points out of seven were enough to ruin it.

**Two RANSAC trials.**

1. Sample $(0,1)$ and $(2,5)$. The line has slope $(5-1)/(2-0)=2$ and intercept 1. Residuals are 0 for the five good points, 6 for $(1,9)$ and 10 for $(3,-3)$. Five inliers.
2. Sample $(1,9)$ and $(3,-3)$. Slope $-12/2=-6$, intercept $9+6=15$. Only those two points lie on the line, so two inliers.
3. Keep the line with five inliers.

**How many trials?** One random pair is all good with probability $w^s=0.5^2=0.25$, so it fails with probability 0.75.

1. Sixteen failures in a row have probability $0.75^{16}=0.01002$, so success is $0.98998$, just under 99%.
2. Seventeen failures have probability $0.75^{17}=0.00752$, so success is $0.99248$, above 99%.

In words: you cannot do 16.008 trials, so you do 17. The 99% describes the chance that at least one sample was all good, not the chance that the final line is good.

## How it works

### Why least squares fails

Matched features contain many wrong pairs (outliers). Least squares minimises total squared error, so a few outliers drag the fit; you need a robust method.

### Random sample consensus

Sample a minimal set (2 for a line, 4 pairs for a homography) → fit → count inliers within a threshold → keep the best → refit on inliers.

:::tip

**Worked.** w=0.5, s=2, p=0.99 → N = log(0.01)/log(0.75) ≈ 16.008, so 17 trials. Drop w to 0.3 → N ≈ 49.

:::

### Key takeaways

- **1 · Outliers**; Least squares is dragged by them.
- **2 · RANSAC**; Sample, fit, count inliers, keep best.
- **3 · Iterations**; N=log(1−p)/log(1−wˢ).

## A real system that works this way

OpenCV's feature-matching and homography tutorial uses SIFT correspondences followed by `findHomography` with a robust-estimation method. It returns a transform and an inlier mask, which can be used to map the corners of a planar reference object into a scene. That is a real instance of the RANSAC pattern: local descriptor matches propose pairs, then global geometry decides which pairs can coexist under one planar mapping. The official tutorial was opened on 2026-10-02 and displayed OpenCV 4.13.0.

The homography model has limits. It can describe two views of one planar surface or camera rotation under certain assumptions, but it cannot align arbitrary non-planar scenes with parallax using one transform. A crowded scene may contain several distinct planes, each with its own consensus. A high inlier count on one repeated texture patch can also be misleading if the desired object is elsewhere. Check the spatial spread of inliers, the mapped corners, reprojection residuals and whether the model is plausible for the intended scene.

The local code uses two-dimensional line fitting instead of downloading images. Five points lie exactly on $y=2x+1$ and two are outliers. It enumerates all valid two-point samples to make the consensus mechanism deterministic and finds the five-point line. Production RANSAC samples rather than exhaustively enumerating all pairs because the number of combinations grows quickly. The toy result proves the arithmetic of this dataset, not a robust-estimator quality metric in arbitrary images.

## Code you can run

The first block verifies both iteration examples with `ceil`. It prints **17** for the 50% inlier case and **49** for the 30% case. It also shows that 16 trials miss the stated 99% success target by a small amount.

```python
from math import ceil, log

def required_trials(inlier_fraction, sample_size, target_success):
    continuous = log(1 - target_success) / log(1 - inlier_fraction ** sample_size)
    return continuous, ceil(continuous)

for inlier_fraction in (0.5, 0.3):
    continuous, whole = required_trials(inlier_fraction, 2, 0.99)
    print(f'w={inlier_fraction:.1f}: continuous={continuous:.3f}, whole={whole}')
success_after_sixteen = 1 - (1 - 0.5 ** 2) ** 16
print(f'Success after 16 trials: {success_after_sixteen:.5%}')
assert required_trials(0.5, 2, 0.99)[1] == 17
assert required_trials(0.3, 2, 0.99)[1] == 49
assert success_after_sixteen < 0.99
```

:::note Correction to the source calculation

The usual answer is about 16 iterations for $w=0.5$, $s=2$ and $p=0.99$. The continuous bound is approximately 16.008; a whole-trial requirement must be rounded **up to 17**. Sixteen trials achieve only about 98.998% under the formula's assumptions. The $w=0.3$ example does round up to 49.

:::

The lab starts at the same 17-trial case. Changing the inlier fraction or sample size shows why robust fitting gets expensive when good correspondences are scarce.

<RansacIterationsLab />

**What each control does.**

- *RANSAC inlier fraction* is $w$, from 0.1 to 0.9.
- *RANSAC minimal sample size* is $s$, from 2 to 5.
- *RANSAC target success probability* is $p$, from 0.8 to 0.999.
- The table shows one-sample success $w^s$, the continuous bound, the whole number of trials and the success achieved.

**Try it yourself.**

1. At the defaults (0.5, 2, 0.99) the table shows 17 trials and an achieved success of 99.248%.
2. Drop the inlier fraction to 0.3. The count rises to 49, because a good pair is now 9% likely instead of 25%.
3. Set the sample size to 4 and the inlier fraction back to 0.5. The count rises to 72, the homography case. Then set the inlier fraction to 0.3: the count jumps to 567. Four points all have to be right.

The second block enumerates every non-degenerate pair in a tiny line dataset, scores inliers with a fixed residual threshold and selects the strongest consensus. It finds **five inliers** on slope **2** and intercept **1**. This is a deterministic model-selection demonstration, not a Monte Carlo timing benchmark.

```python
from itertools import combinations

points = [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9), (1, 9), (3, -3)]
candidates = []
for (x1, y1), (x2, y2) in combinations(points, 2):
    if x1 == x2:
        continue
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    inliers = [point for point in points if abs(point[1] - (slope * point[0] + intercept)) < 0.1]
    candidates.append((len(inliers), slope, intercept, inliers))
best = max(candidates, key=lambda candidate: candidate[0])
print('Consensus size:', best[0])
print('Line slope and intercept:', best[1], best[2])
assert (best[0], best[1], best[2]) == (5, 2.0, 1.0)
```

For noisy data, refit the chosen model on all inliers and evaluate residuals; this construction has exact integer inliers so the minimal pair already recovers the intended line. A homography would need a different model fit, residual and degeneracy check.

The next block reproduces the worked example, so the least-squares line and the two RANSAC samples can be checked against the pencil version.

```python
import numpy as np

x = np.array([0, 1, 2, 3, 4, 1, 3], dtype=float)
y = np.array([1, 3, 5, 7, 9, 9, -3], dtype=float)
slope, intercept = np.polyfit(x, y, 1)
print(f'least squares: slope {slope:.3f}, intercept {intercept:.3f}')
for pair in ((0, 2), (5, 6)):
    i, j = pair
    m = (y[j] - y[i]) / (x[j] - x[i])
    c = y[i] - m * x[i]
    residuals = np.abs(y - (m * x + c))
    print(f'sample {pair}: slope {m:.1f}, intercept {c:.1f}, inliers {int((residuals < 0.1).sum())}')
print(f'0.75 ** 16 = {0.75 ** 16:.5f}, success after 16 trials {1 - 0.75 ** 16:.5f}')
print(f'0.75 ** 17 = {0.75 ** 17:.5f}, success after 17 trials {1 - 0.75 ** 17:.5f}')
```

**Reading the output.** Least squares prints slope 0.667 and intercept 3.095. The first RANSAC sample finds slope 2.0, intercept 1.0 and five inliers. The second finds two. The last two lines give the 16-trial and 17-trial success of 0.98998 and 0.99248.

### Experiment: does 17 trials really find the line, and what happens with a homography?

The first block checks the 17 against a simulation. The data have 1,000 points on $y=2x+1$ with noise of standard deviation 0.4, half of them replaced by random heights. For each number of trials it reports three things over repeated runs: the formula's success, how often a random sample contained only good points (200,000 runs), and how often the final line was within 0.5 of the truth everywhere on $[0,10]$ (2,000 runs, after refitting on the inliers).

```python
from math import ceil, log

import numpy as np

rng = np.random.default_rng(0)
points, slope, intercept, sigma, threshold = 1000, 2.0, 1.0, 0.4, 1.0
grid = np.linspace(0, 10, 50)


def dataset():
    x = rng.uniform(0, 10, points)
    y = slope * x + intercept + rng.normal(0, sigma, points)
    y[: points // 2] = rng.uniform(-10, 30, points // 2)
    return x, y


def sampling_hit(trials, runs):
    draws = rng.integers(0, points, size=(runs, trials, 2))
    distinct = draws[..., 0] != draws[..., 1]
    return ((draws >= points // 2).all(axis=2) & distinct).any(axis=1).mean()


def fit_once(trials):
    x, y = dataset()
    pairs = rng.integers(0, points, size=(trials, 2))
    dx = x[pairs[:, 1]] - x[pairs[:, 0]]
    ok = np.abs(dx) > 1e-9
    m = np.where(ok, (y[pairs[:, 1]] - y[pairs[:, 0]]) / np.where(ok, dx, 1), 0.0)
    c = y[pairs[:, 0]] - m * x[pairs[:, 0]]
    support = np.abs(y[None, :] - (m[:, None] * x[None, :] + c[:, None])) < threshold
    support[~ok] = False
    best = support[support.sum(axis=1).argmax()]
    fitted = np.polyfit(x[best], y[best], 1)
    ls = np.polyfit(x, y, 1)
    truth = slope * grid + intercept
    return np.abs(np.polyval(fitted, grid) - truth).max() < 0.5, np.abs(np.polyval(ls, grid) - truth).max() < 0.5


print('trials  formula  sampling hit (200000 runs)  line recovered (2000 runs)  least squares recovered')
for trials in (8, 16, 17, 34, 68):
    formula = 1 - (1 - 0.5 ** 2) ** trials
    results = np.array([fit_once(trials) for _ in range(2000)])
    print(f'{trials:>6}  {formula:>7.4f}  {sampling_hit(trials, 200000):>26.4f}  {results[:, 0].mean():>26.4f}  {results[:, 1].mean():>23.4f}')
print('trials for 99% at w=0.5, s=2:', ceil(log(0.01) / log(1 - 0.5 ** 2)), 'continuous', round(log(0.01) / log(1 - 0.5 ** 2), 3))
```

**Reading the output.** Each row is a number of trials. "Formula" is $1-0.75^N$. "Sampling hit" is the measured share of runs that drew at least one all-good pair. "Line recovered" is the share of runs whose final line was close to the truth. "Least squares recovered" is the same test for a plain least-squares fit of all points.

**Line by line.**

- `sampling_hit` draws all pairs at once and checks whether both indices are in the good half. It tests only the sampling event the formula talks about.
- `fit_once` scores every candidate line by counting residuals under `threshold`, keeps the best, and refits with `np.polyfit` on its inliers.
- `support[~ok] = False` removes the vertical-pair samples whose slope is undefined.

The second block moves to 2D point correspondences and a homography. Part one compares four estimators as the outlier share grows. Part two uses a four-point RANSAC written out by hand, so the number of trials is under our control.

```python
from math import ceil, log

import cv2
import numpy as np

true_h = np.array([[0.9, -0.2, 40.0], [0.15, 1.05, -25.0], [0.0004, -0.0002, 1.0]])
test_points = np.random.default_rng(99).uniform([0, 0], [640, 480], size=(500, 2)).astype(np.float32)
test_truth = cv2.perspectiveTransform(test_points.reshape(-1, 1, 2), true_h).reshape(-1, 2)


def correspondences(rng, count, outlier_fraction):
    src = rng.uniform([0, 0], [640, 480], size=(count, 2)).astype(np.float32)
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), true_h).reshape(-1, 2) + rng.normal(0, 1.0, (count, 2))
    bad = rng.random(count) < outlier_fraction
    dst[bad] = rng.uniform([0, 0], [640, 480], size=(bad.sum(), 2))
    return src, dst.astype(np.float32), bad


def transfer_error(h):
    if h is None:
        return np.inf
    mapped = cv2.perspectiveTransform(test_points.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.linalg.norm(mapped - test_truth, axis=1).mean())


methods = {'least squares': 0, 'RANSAC': cv2.RANSAC, 'LMEDS': cv2.LMEDS, 'RHO': cv2.RHO}
print('median transfer error in pixels over 30 trials, 300 correspondences, noise sigma 1 px')
print('outliers  ' + '  '.join(f'{name:>13s}' for name in methods))
for fraction in (0.0, 0.1, 0.3, 0.5, 0.7):
    errors = {name: [] for name in methods}
    for seed in range(30):
        src, dst, bad = correspondences(np.random.default_rng(seed), 300, fraction)
        for name, flag in methods.items():
            h, _ = cv2.findHomography(src, dst, flag, 3.0, maxIters=2000, confidence=0.995)
            errors[name].append(transfer_error(h))
    print(f'{fraction:>8.1f}  ' + '  '.join(f'{np.median(errors[name]):>13.2f}' for name in methods))

print('\nformula against a hand-written four-point RANSAC, 100 runs each, success = transfer error under 2 px')
for w in (0.5, 0.3):
    needed = ceil(log(0.01) / log(1 - w ** 4))
    row = []
    for trials in (needed // 4, needed // 2, needed, needed * 2):
        wins = 0
        for seed in range(100):
            rng = np.random.default_rng(1000 + seed)
            src, dst, bad = correspondences(rng, 300, 1 - w)
            best, best_support = 0, None
            for _ in range(trials):
                pick = rng.choice(300, 4, replace=False)
                try:
                    h = cv2.getPerspectiveTransform(src[pick], dst[pick])
                except cv2.error:
                    continue
                mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
                support = np.linalg.norm(mapped - dst, axis=1) < 3.0
                if support.sum() > best:
                    best, best_support = support.sum(), support
            if best_support is not None and best >= 8:
                refit, _ = cv2.findHomography(src[best_support], dst[best_support], 0)
                wins += transfer_error(refit) < 2.0
        row.append(f'{trials} trials: {wins}%')
    print(f'w={w}: formula says {needed}; ' + ' | '.join(row))
```

**Reading the output.** The first table is the median error, in pixels, of the estimated mapping on 500 test points. The second part prints, for each inlier fraction, the trials the formula asks for and the percentage of 100 runs whose final homography was within 2 pixels at a quarter, a half, once and twice that number.

**Line by line.**

- `cv2.getPerspectiveTransform` is the exact four-point homography, the minimal fit.
- Outliers are chosen independently for each point with probability `outlier_fraction`, which matches the independence assumption in the formula.
- `maxIters=2000, confidence=0.995` are the stopping settings handed to OpenCV's own methods.

#### Reading the experiment

The 17 is right. Sixteen trials give a sampling hit of 0.9898 and seventeen give 0.9925, matching the formula's 0.9900 and 0.9925. The correction to 16 is mathematically correct and also tiny: the gap is a quarter of a percentage point. The line-recovery test cannot see it, because 0.947 at 16 trials and 0.953 at 17 differ by less than the sampling noise of 2,000 runs (about 0.005).

The surprise is the third column. The formula promises that at least one sample is clean 99% of the time, and it is. But a clean pair of noisy points can still define a poor line, so at 17 trials the final line was right only 0.953 of the time. About 34 trials were needed to reach 0.999. Least squares recovered the line in none of the runs.

The homography table shows the same gap. With no outliers, least squares is the best method (0.15 pixels against 0.27 for RANSAC). With 10% outliers it is 19.35 pixels off, and with 50% it is 184.09, while RANSAC stays at 0.27 to 0.32. LMEDS is fine to 30% outliers (0.17), degrades at 50% (4.13) and fails at 70% (188.69), consistent with its need for a clear majority of inliers. At the formula's trial count the hand-written four-point RANSAC succeeded 88% of the time at $w=0.5$ (72 trials) and 86% at $w=0.3$ (567 trials). Doubling the trials gave 96% and 97%.

Limits: synthetic data with a known mapping, one noise level, 2 pixel and 0.5 success tests chosen by me, and a hand-written RANSAC without degeneracy checks. Treat the formula as a lower bound on effort.

<Infographic src="/img/cv-enrich/v2-ransac-trials.svg" alt="Left: a table of trials, formula, all-inlier sample rate, line recovered and least squares for a line with half outliers. Right: median homography error in pixels for four methods as outliers grow." caption="Look at the 17 row: the sampling hit is 0.9925 but the line is recovered 0.953 of the time." />

## Designing with it

Choose the geometric model before tuning RANSAC. A line, affine transform, fundamental matrix and homography encode different assumptions. If the scene violates the model, consensus may be large yet systematically wrong. Determine what movement, depth variation and camera calibration imply, then test the chosen model on representative pairs. Do not use a homography for a non-planar scene simply because it is easy to call.

Set the residual threshold in meaningful units. A reprojection error of three pixels may be strict for a high-resolution calibrated marker and loose for a low-resolution or blurry image. A threshold that is too tight fragments true matches; one that is too wide absorbs outliers and biases the refit. Inspect residual distributions and how they change with image scale, lens distortion and keypoint quality. If images are resized, update the pixel threshold accordingly.

Estimate the inlier fraction cautiously. The iteration formula uses $w$, but true $w$ is usually unknown before fitting. An optimistic estimate can stop too early; a pessimistic estimate wastes compute. Adaptive implementations update the required trial count as a better consensus is found, subject to a cap and degeneracy checks. The desired success probability is also a policy choice, not a measured model accuracy. A 99% chance of seeing an all-inlier minimal sample does not mean 99% of final transforms will be useful.

Check inlier distribution and model sanity after selection. All inliers from one narrow strip can fit a homography that extrapolates badly elsewhere. Repeated patterns can create coherent but wrong correspondences. Reject implausible scale, reflection or perspective changes where the task forbids them. Compare projected reference corners with the image, and keep a fallback when too few well-distributed inliers remain.

Finally, make randomness reproducible for experiments. A fixed seed makes a small test stable, but a production run should be evaluated over varied seeds and diverse image pairs. Record iteration cap, sampling policy, threshold, score and refit method. A faster implementation that silently changes any of these may change the failure profile even if the API still says “RANSAC”.

## Where this stands in 2026

:::info Industry view

- RANSAC and robust variants remain common for feature geometry, calibration and mapping. Their usefulness comes from an explicit model and residual test, not from any guarantee against all outliers.
- The OpenCV feature-homography tutorial was checked on 2026-10-02. The chapter examples use Python standard-library maths and exhaustive sampling, so no OpenCV version-dependent result is claimed for the toy fit.
- The 17-trial correction is a mathematical consequence of the example's own inputs; no image dataset or benchmark is involved.

:::

## Common mistakes

1. **Reading 99% as the chance the model is right.** The formula feels like a quality guarantee. It is the chance of one clean sample, and at 17 trials the line was recovered 0.953 of the time. Add trials or refit and check residuals.
2. **Rounding 16.008 down.** Sixteen looks close enough. It gives 0.98998 against a target of 0.99, and the correct whole number is 17. The practical gap is small, but the rule is to round up.
3. **Using a robust method past its limit.** LMEDS worked at 30% outliers (0.17 px) and failed at 70% (188.69 px). Check which breakdown point your method has.
4. **Using RANSAC when there are no outliers.** It feels safer. With clean data least squares was better (0.15 px against 0.27 px). Robust methods give up a little accuracy for protection.
5. **Setting the threshold without the noise level.** With 1 pixel noise per coordinate, about 1.1% of true inliers lie beyond 3 pixels (the Rayleigh tail $e^{-4.5}$). Set the threshold from the measured residual spread, in the image's current units.

## Practice questions

<details>
<summary><strong>Q1.</strong> Why does least squares fail for feature-match model fitting?</summary>

Matches contain many outliers; least squares minimises total squared error, so a few gross errors drag the fit arbitrarily.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Describe the RANSAC algorithm.</summary>

Randomly sample a minimal set → fit the model → count inliers within a threshold → keep the model with the most inliers → refit on all inliers.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> How many point-pairs define a line and a homography for RANSAC?</summary>

A line needs 2 points; a homography needs 4 point correspondences.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Inlier fraction w=0.5, sample size s=2, confidence p=0.99. Compute RANSAC iterations N.</summary>

N = log(1−p)/log(1−wˢ) = log(0.01)/log(1−0.25) = −2/−0.1249 ≈ 16 iterations. Rounded up to a whole number of trials this is 17, as the correction above shows.<br /><em>Session 8 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What happens to N as the inlier fraction drops?</summary>

N rises sharply; e.g. at w=0.3, s=2, p=0.99, N ≈ 49. Larger models (bigger s) also need many more iterations.<br /><em>Session 8 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> Easy. Inlier fraction 0.7, four-point sample, 99% target. How many trials?</summary>

$0.7^4=0.2401$, so a clean sample fails with probability $0.7599$. Then $N=\log(0.01)/\log(0.7599)=16.77$, which rounds up to 17.<br /><em>Easy · numeric</em>

</details>

<details>
<summary><strong>Q7.</strong> Medium. Seven points, five on $y=2x+1$ and two wrong. Why does least squares report slope 0.667?</summary>

The wrong point $(1,9)$ lifts the line on the left and $(3,-3)$ drags it down on the right, which flattens the slope. The sums are $\sum(x-\bar x)(y-\bar y)=8$ and $\sum(x-\bar x)^2=12$, so the slope is $8/12=0.667$. Squared error rewards fitting the bad points as much as the good ones.<br /><em>Medium · numeric</em>

</details>

<details>
<summary><strong>Q8.</strong> Stretch. At 17 trials the sampling hit was 0.9925 but the line was recovered 0.953 of the time. Name two causes and one fix.</summary>

A clean pair of noisy points can define a line that is tilted enough to miss part of the true inlier band, so it gathers fewer inliers than a better pair would. Two pairs with close x values give an unstable slope. Fixes: more trials (34 gave 0.999), a local refit after each promising candidate, or a minimum separation between the sampled points.<br /><em>Stretch · interpretation</em>

</details>

## Further reading

- [OpenCV feature matching with homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) for an image-level robust geometry workflow.
- Built from the course lecture "cv-s8-ransac" (Lecture Library series).

- Fischler and Bolles, "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography", Communications of the ACM 24(6), 381 to 395, 1981 (bibliographic record checked 2026-10-09; paper text not re-read).
- OpenCV 5.0.0 `findHomography`: the docstring bundled with the library (read 2026-10-09) lists least squares (method 0), RANSAC, LMEDS and RHO, and says the robust methods try random four-pair subsets and refine with Levenberg-Marquardt on the inliers.
- Library versions run for the experiment: OpenCV 5.0.0, NumPy 2.5.3.

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## A robust-fit failure review

Imagine a matcher produces 100 candidate pairs and a robust homography reports 60 inliers. That sounds strong until the overlay shows all 60 points in one small repeated logo. The model may fit that patch and extrapolate the rest of the reference outside the image. An inlier count is a measure of agreement under a threshold, not a complete quality judgement. Check how widely inliers cover the reference and destination, whether their residuals are stable, and whether the mapped object corners are geometrically plausible.

Now imagine the inliers are well spread, but most lie on a moving vehicle while the intended model should align a static background. The sampler successfully found consensus for the wrong surface. The remedy is not necessarily more iterations; the model-selection objective or region of interest must reflect the desired scene component. A semantic mask, motion prior or multiple-model analysis may be needed. Robust estimation suppresses numerical outliers relative to its model, not semantic irrelevance.

A third failure comes from the residual threshold. If image coordinates were halved by preprocessing but the threshold stayed at its original pixel value, twice as much physical deviation may now count as inlier. The fit can admit mismatches. If the threshold was scaled in the opposite direction, true matches may be rejected. Coordinate transforms and calibration should be part of the model contract. Print residual units and image scale beside every threshold in an experiment report.

The iteration formula should be read as a probability model. With 50% inliers and two-point samples, each ideal sample has a 25% chance to be all-inlier. Sixteen independent trials fail all sixteen with probability $0.75^{16}$, slightly above 1%. Seventeen lower that failure probability below 1%. A sample that is all-inlier can still be degenerate, and measurement noise can still produce a bad candidate. The bound is a planning aid for the sampling stage, not a guarantee on final application performance.

An estimated inlier fraction can change during fitting. Suppose early descriptor filtering leaves few reliable pairs and the working estimate is 0.3. Planning only 17 two-point trials would be optimistic for 99% all-inlier sampling success; the formula gives 49 under its idealised assumptions. If later a strong consensus supports a larger inlier fraction, an adaptive solver may reduce the remaining budget. It still needs an upper cap for latency and an explicit failure result when no plausible consensus appears. Reporting “no transform” can be safer than returning the best of several bad candidates. A production pipeline should record how often it abstains and whether those cases cluster in low light, repeated texture or scene changes. These cases also test whether the residual threshold and selected geometric model match the task, because additional trials cannot repair a systematically wrong model.

The success percentage describes the sampling event, not the error of the final fitted line. Evaluate held-out correspondences or downstream alignment separately, and include the cases where the method refused to return a model.

## Check yourself

- I can describe sample, fit, score, select and refit in RANSAC.
- I can derive and round up its iteration budget from inlier fraction, sample size and target success.
- I can explain why the 16-trial answer misses 99% and why 17 meets it.
- I can name a wrong-model or bad-threshold failure that more trials would not fix.
- I can fit a line with outliers by least squares and by two RANSAC samples, by hand.
- I can say why the formula's 99% describes the sample and not the final model, using 0.9925 against 0.953.
- I can say where LMEDS and least squares break as outliers grow, and why RANSAC costs a little accuracy on clean data.
