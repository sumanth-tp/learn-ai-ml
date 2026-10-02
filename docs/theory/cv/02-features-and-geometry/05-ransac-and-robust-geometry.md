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

## The idea in plain words

:::note Beyond the lecture

The exhaustive toy fit, probability correction and failure analysis extend the lecture. Its least-squares motivation, loop, examples and questions remain below.

:::

Feature matches are imperfect. A descriptor can pair two visually similar but unrelated patches, especially in repeated texture or under occlusion. If all matches are fed into ordinary least squares, a few gross errors can pull a fitted line or geometric transform far from the relationship supported by most correct correspondences. Session 8 introduces RANSAC, random sample consensus, as a way to fit a model despite such outliers. The core question is not “which model minimises every error?” but “which model has the strongest set of data points whose errors are acceptably small?”

One RANSAC trial draws a minimal sample, fits a candidate model, computes a residual for every data point, and calls points within a chosen threshold inliers. A line can be determined by two distinct points; a projective homography needs four suitable point correspondences. The algorithm keeps the candidate with the best consensus according to its score. After choosing an inlier set, it commonly refits the model using all those inliers, because the minimal sample gives a noisy estimate. The threshold defines what “consistent” means in the units of the residual, such as pixels of reprojection error.

The sampling step is probabilistic. If an individual match has independent inlier probability $w$ and a minimal model requires $s$ matches, the chance that one sampled set contains only inliers is $w^s$. The chance of failing to draw such a set in $N$ independent trials is $(1-w^s)^N$. To make the chance of at least one all-inlier set at least $p$, solve $1-(1-w^s)^N\ge p$. This gives $N\ge\log(1-p)/\log(1-w^s)$. Since trials are whole numbers, **round the result up**. These probabilities are idealised: real samples are not always independent, the inlier fraction may be unknown and some all-inlier samples can be geometrically degenerate.

For the lecture's $w=0.5$, $s=2$, $p=0.99$, the continuous result is about **16.008**, so at least **17 whole trials** are required. The source rounds to 16, but 16 gives about 98.998% under the stated model, just below the 99% target. At $w=0.3$, the continuous value is about 48.830 and the integer count is **49**. Lower inlier fractions or larger minimal sets make a successful sample rarer, causing the trial budget to rise quickly. This formula answers a confidence-budget question; it does not guarantee the selected model is correct if the residual threshold or geometric model is wrong.

<Infographic src="/img/cv/ransac-consensus.svg" alt="RANSAC samples a minimal model, scores residuals and refits inliers; the lecture's 99 percent example requires 17 whole trials rather than 16." caption="The probability calculation must be rounded up and still rests on sampling assumptions." />

## How it works

### Why least squares fails

Matched features contain many wrong pairs (outliers). Least squares minimises total squared error, so a few outliers drag the fit; you need a robust method.

### Random sample consensus

Sample a minimal set (2 for a line, 4 pairs for a homography) → fit → count inliers within a threshold → keep the best → refit on inliers.

:::tip

**Worked.** w=0.5, s=2, p=0.99 → N = log(0.01)/log(0.75) ≈ 16. Drop w to 0.3 → N ≈ 49.

:::

### Key takeaways

- **1 · Outliers**; Least squares is dragged by them.
- **2 · RANSAC**; Sample, fit, count inliers, keep best.
- **3 · Iterations**; N=log(1−p)/log(1−wˢ).

## A real system that works this way

OpenCV's feature-matching and homography tutorial uses SIFT correspondences followed by `findHomography` with a robust-estimation method. It returns a transform and an inlier mask, which can be used to map the corners of a planar reference object into a scene. That is a real instance of the lecture's pattern: local descriptor matches propose pairs, then global geometry decides which pairs can coexist under one planar mapping. The official tutorial was opened on 2026-10-02 and displayed OpenCV 4.13.0.

The homography model has limits. It can describe two views of one planar surface or camera rotation under certain assumptions, but it cannot align arbitrary non-planar scenes with parallax using one transform. A crowded scene may contain several distinct planes, each with its own consensus. A high inlier count on one repeated texture patch can also be misleading if the desired object is elsewhere. Check the spatial spread of inliers, the mapped corners, reprojection residuals and whether the model is plausible for the intended scene.

The local code uses two-dimensional line fitting instead of downloading images. Five points lie exactly on $y=2x+1$ and two are outliers. It enumerates all valid two-point samples to make the consensus mechanism deterministic and finds the five-point line. Production RANSAC samples rather than exhaustively enumerating all pairs because the number of combinations grows quickly. The toy result proves the arithmetic of this dataset, not a robust-estimator quality metric in arbitrary images.

## Code you can run

The first block verifies both lecture iteration examples with `ceil`. It prints **17** for the 50% inlier case and **49** for the 30% case. It also shows that 16 trials miss the stated 99% success target by a small amount.

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

The source reports about 16 iterations for $w=0.5$, $s=2$ and $p=0.99$. The continuous bound is approximately 16.008; a whole-trial requirement must be rounded **up to 17**. Sixteen trials achieve only about 98.998% under the formula's assumptions. The $w=0.3$ example does round up to 49.

:::

The lab starts at the same 17-trial case. Changing the inlier fraction or sample size shows why robust fitting gets expensive when good correspondences are scarce.

<RansacIterationsLab />

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
- The 17-trial correction is a mathematical consequence of the lecture’s own inputs; no image dataset or benchmark is involved.

:::

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

N = log(1−p)/log(1−wˢ) = log(0.01)/log(1−0.25) = −2/−0.1249 ≈ 16 iterations.<br /><em>Session 8 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What happens to N as the inlier fraction drops?</summary>

N rises sharply; e.g. at w=0.3, s=2, p=0.99, N ≈ 49. Larger models (bigger s) also need many more iterations.<br /><em>Session 8 · conceptual</em>

</details>

## Further reading

- [OpenCV feature matching with homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) for an image-level robust geometry workflow.
- Built from the course lecture "cv-s8-ransac" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
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
- I can explain why the lecture's 16-trial answer misses 99% and why 17 meets it.
- I can name a wrong-model or bad-threshold failure that more trials would not fix.
