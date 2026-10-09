---
id: cv-image-gradients-and-edges
title: "Computer Vision · Session 3; Image Gradients and Edges"
sidebar_label: "1 · Gradients and edges"
sidebar_position: 1
slug: /theory/cv/image-gradients-and-edges
description: "Understand step, ramp and roof edges, Sobel gradients, signed responses and noise-sensitive thresholds."
tags: [computer-vision, edges, gradients, sobel]
---

import Infographic from '@site/src/components/Infographic';
import EdgeGradientLab from '@site/src/components/viz/EdgeGradientLab';

**In one line.** An edge detector responds to local intensity change, then needs a task-specific rule to decide which changes matter.

:::tip Before you start

**You should already know:**

- That an image is a grid of numbers and that blurring averages neighbours. See [colour and filtering](/docs/theory/cv/colour-histograms-and-filtering).
- What precision and recall mean: the share of found items that are right, and the share of right items that were found.

**Reading time:** about 30 minutes.

**After this chapter you can:**

- compute a Sobel gradient, its magnitude and its angle by hand;
- score an edge map against a ground-truth boundary with F1;
- explain why smoothing before differentiating matters more than the choice of operator.

:::

## In 30 seconds

An edge is a place where brightness changes fast. To find one, subtract the pixel on the left from the pixel on the right: a big difference means an edge. The trouble is that noise also makes big differences between neighbours, so differencing amplifies noise. Think of listening for a whisper in a noisy room: you lean in and average over a moment before you decide. Smoothing before differencing is that averaging.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Gradient | The direction and rate of fastest brightness increase | Gx = 4, Gy = 3 |
| Sobel, Scharr, Prewitt | Small 3 by 3 grids of weights that estimate the gradient | Sobel weights 1, 2, 1 across the rows |
| Laplacian | A second derivative: how the gradient itself changes | Responds at both sides of a step |
| Ground truth | The correct answer, here the true shape boundary | The outline of a rectangle |
| Precision | Share of found edge pixels that are real | 80 of 100 |
| Recall | Share of real edge pixels that were found | 90 of 100 |
| F1 | 2 × precision × recall / (precision + recall) | 0.847 for 0.8 and 0.9 |
| Noise sd | Standard deviation of random grey-level noise | sd 40 on a 120-level step |


## The idea in plain words

:::note Additions to the course material

The signed-gradient experiment and design discussion extend the course material. Its edge profiles, gradient example, Sobel and Prewitt discussion and all practice questions remain below.

:::

Edges are not objects. They are locations where a measured image value changes quickly over space. A black shape against a white background creates one sort of edge, but a shadow, reflection, texture stripe or compression artifact can also create a strong change. Edge profiles make that distinction visible. A step changes abruptly, a ramp changes over several pixels, and a roof rises then falls like a narrow line. In real images, optics and sampling spread many ideal steps into ramps. The profile shape affects how a derivative responds and how a threshold should be interpreted.

The image gradient packages the local change into two components: $G_x$ for horizontal variation and $G_y$ for vertical variation. The vector points in the direction where intensity rises fastest. Its magnitude is $\sqrt{G_x^2+G_y^2}$, and its orientation is best computed with `atan2(Gy, Gx)` so the correct quadrant is retained. An edge boundary itself runs approximately perpendicular to that vector. The worked values $G_x=4$ and $G_y=3$ give magnitude 5 and angle 36.87°. The angle describes the normal to a local intensity boundary, not the angle of a whole detected object.

Digital images do not provide a continuous derivative. Operators approximate it with neighbouring samples. Sobel and Prewitt use small masks that combine a directional difference with some smoothing across the other direction. This reduces sensitivity to isolated noise compared with a raw two-pixel difference. The Laplacian is a second-derivative operator; a zero crossing can mark a transition after appropriate smoothing, but the response is also sensitive to noise. No operator alone knows whether the transition is a useful region boundary.

Think about the gradient's sign before converting types. A dark-to-bright transition has one sign; bright-to-dark has the opposite sign for the same axis. If a signed derivative is forced directly into an unsigned 8-bit output, negative values may be clipped to zero. The official OpenCV gradient tutorial calls this out explicitly. Work in a signed or floating representation when measuring both directions, then take magnitude or a suitable absolute value for display. This is a common example of a numerically valid function call producing an incomplete result because its output type was chosen incorrectly.

An edge map is usually created by thresholding gradient strength, but a single threshold can leave thick ridges, gaps and unrelated texture. A high threshold suppresses weak true boundaries; a low threshold admits noise. The Canny process in the next chapter adds non-maximum suppression and hysteresis to address these issues. Before adding that complexity, make sure the image can actually resolve the target boundary and that its contrast survives acquisition and preprocessing.

<Infographic src="/img/cv/gradients.svg" alt="A board compares step, ramp and roof edge profiles, Sobel gradient components 4 and 3 giving magnitude 5, and an orientation of 36.87 degrees." caption="Gradient magnitude measures change; its direction points across the edge." />

## Worked example, step by step

**One gradient.** A 3 by 3 patch has three columns of values 10, 10, 50 (a vertical step of contrast 40).

1. Apply the Sobel weights for Gx, with the right column positive: (50 − 10) × 1 + (50 − 10) × 2 + (50 − 10) × 1 = 160.
2. The vertical gradient Gy is 0, because each column is constant.
3. Magnitude = √(160² + 0²) = 160 and the angle is 0 degrees. The gradient points right, across the edge.

**Why noise matters.** Noise of standard deviation sd adds to each of the nine pixels, and each weight multiplies it.

1. The signal part of a step edge grows by the sum of the positive weights: 4 × contrast for Sobel (4 × 40 = 160).
2. The noise part grows by the square root of the sum of squared weights: √(1+4+1+1+4+1) = √12 = 3.46 × sd.
3. The ratio signal to noise is 4 / 3.46 = 1.155 for Sobel. For Prewitt it is 3 / 2.45 = 1.225 and for Scharr 16 / 15.36 = 1.042.
4. On this crude measure Scharr is the noisiest of the three, in spite of being the most accurate derivative for clean images.

**Scoring an edge map.** If 100 pixels are marked, 80 lie near the true boundary, the boundary has 100 pixels and 90 of them have a marked pixel nearby, then precision is 0.8, recall is 0.9, and F1 = 2 × 0.8 × 0.9 / 1.7 = 0.847. The first block below reproduces these numbers.

## How it works

### What is an edge?

A location of rapid intensity change; a region boundary. Profiles: step (jump), ramp (gradual, real edges), roof (thin line).

### Gradients & operators

∇f = (∂f/∂x, ∂f/∂y), approximated by finite differences. Magnitude √(Gx²+Gy²) is large at edges; orientation arctan(Gy/Gx) is perpendicular to the edge.

:::tip

Sobel/Prewitt 3×3 masks estimate Gx, Gy while smoothing; important because differentiation amplifies noise.

:::

:::tip

**Worked.** Gx=4, Gy=3 → magnitude √25 = 5; orientation arctan(3/4) = 36.87°.

:::

### From gradients to edges

Thresholding magnitude gives edge pixels, but one threshold gives broken/thick edges; motivating Canny and Hough next. The Laplacian (2nd derivative, zero crossings) is an alternative.

### Key takeaways

- **1 · Edge**; Rapid intensity change.
- **2 · Gradient**; Mag √(Gx²+Gy²), angle arctan(Gy/Gx).
- **3 · Operators**; Sobel/Prewitt; smooth first.

## A real system that works this way

OpenCV's `Sobel`, `Scharr` and `Laplacian` functions are a practical implementation of the operators in this chapter. The current official Python tutorial demonstrates first derivatives in horizontal and vertical directions and a second-derivative Laplacian. It also warns that an unsigned output hides negative slopes. This makes OpenCV a useful named system for understanding the engineering detail: an edge algorithm is not just a mathematical mask, but also a data type, border policy, kernel size and display mapping.

Imagine a camera reading a printed black rectangle on a light package. The left boundary goes light to dark and the right goes dark to light. A signed horizontal derivative should show opposite responses for those two sides. If the application only preserves positive derivative values, it may detect the left side and miss the right. The toy code below reproduces that failure with a bright vertical band on a dark field. A production system would then combine orientations and signed magnitudes, inspect the line geometry, and confirm that reflections or package seams are not mistaken for the print boundary.

The tutorial was checked on 2026-10-02 as OpenCV documentation version 4.13.0. The local CPU example was executed with `opencv-python-headless` 5.0.0.93. It uses a synthetic image and makes no claim about detection quality on a real camera feed. The local example isolates the mechanism so a reader can test the sign and data type rather than relying on a figure copied from the tutorial.

## Code you can run

The first block checks the worked vector. `atan2` gives **36.87°** for the default `(4,3)` vector and handles negative components correctly if you change the inputs. When both components are zero, orientation has no useful meaning, although magnitude is zero.

```python
from math import atan2, degrees, hypot

gx = 4
gy = 3
magnitude = hypot(gx, gy)
orientation = degrees(atan2(gy, gx))
print(f'Magnitude: {magnitude:.2f}')
print(f'Orientation: {orientation:.2f} degrees')
assert magnitude == 5
assert round(orientation, 2) == 36.87
```

The lab starts with the same gradient and lets you change both components. Its vector drawing is a local derivative illustration, not an outline of an object.

<EdgeGradientLab />

The second block creates a five-row bright band. A signed Sobel derivative produces **ten positive and ten negative** responses, each with magnitude 800 in this constructed case. Clipping the result directly to unsigned bytes removes the negative side. The exact response depends on the kernel and border rule, so the code fixes a 3 by 3 Sobel kernel and this one array.

```python
import cv2
import numpy as np


image = np.zeros((5, 7), dtype=np.uint8)
image[:, 2:5] = 200
signed = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
unsigned = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
positive = int(np.count_nonzero(signed > 0))
negative = int(np.count_nonzero(signed < 0))
print('Signed positive and negative:', positive, negative)
print('Signed extrema:', int(signed.min()), int(signed.max()))
print('Unsigned nonzero count:', int(np.count_nonzero(unsigned)))
assert (positive, negative) == (10, 10)
assert (int(signed.min()), int(signed.max())) == (-800, 800)
assert int(np.count_nonzero(unsigned)) == 10
```

The operator returns measurements, not final edge decisions. A detector must combine direction, magnitude, continuity and task-specific evidence. A signed gradient can also be converted to an absolute magnitude after computation if only strength is needed, but keep the sign while investigating transition direction.

### The worked numbers in code

```python
import numpy as np

patch = np.array([[10, 10, 50], [10, 10, 50], [10, 10, 50]], dtype=float)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
print("Gx on the step patch:", float((patch * sobel_x).sum()), "magnitude:", float(np.hypot((patch * sobel_x).sum(), (patch * sobel_x.T).sum())))
for name, kernel in (("Prewitt", prewitt_x), ("Sobel", sobel_x), ("Scharr", scharr_x)):
    signal = np.abs(kernel).sum() / 2
    noise_gain = np.sqrt((kernel ** 2).sum())
    print(f"{name}: step gain {signal:.0f} x contrast, noise gain {noise_gain:.2f} x sd, ratio {signal / noise_gain:.3f}")
precision, recall = 80 / 100, 90 / 100
print("F1 for precision 0.8 and recall 0.9:", round(2 * precision * recall / (precision + recall), 4))
```

**Reading the output.** The step patch gives Gx = 160 and magnitude 160. The signal-to-noise ratio of the kernel is 1.225 for Prewitt, 1.155 for Sobel and 1.042 for Scharr, and F1 for precision 0.8 and recall 0.9 is 0.8471.

### Experiment: noise sensitivity against a true boundary

The scene is a 160 by 160 image with a rectangle, a circle and a triangle on a background, blurred slightly (sigma 1) so edges are ramps. The true boundary of each shape is known exactly. The step height is 120 grey levels (190 against 70). Gaussian noise of standard deviation 0, 20, 40 and 60 is added. Six detectors produce a strength image. Each strength image is thresholded at 60 levels, and the best F1 over those thresholds is reported, with a pixel counted as correct if it is within 1 pixel of the true boundary.

```python
import cv2
import numpy as np
from scipy import ndimage

def scene():
    mask = np.zeros((160, 160), np.uint8)
    cv2.rectangle(mask, (20, 20), (70, 80), 1, -1)
    cv2.circle(mask, (115, 50), 28, 1, -1)
    cv2.fillPoly(mask, [np.array([[30, 140], [80, 95], [130, 140]])], 1)
    truth = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(np.where(mask > 0, 190.0, 70.0), (0, 0), 1.0), truth.astype(bool)

def best_f1(strength, truth, tolerance=1):
    near_truth = ndimage.distance_transform_edt(~truth) <= tolerance
    best = 0.0
    for t in np.quantile(strength, np.linspace(0.70, 0.995, 60)):
        found = strength >= t
        near_found = ndimage.distance_transform_edt(~found) <= tolerance
        precision = (found & near_truth).sum() / max(found.sum(), 1)
        recall = (truth & near_found).sum() / truth.sum()
        best = max(best, 2 * precision * recall / (precision + recall + 1e-9))
    return best

def gradient(image, kind):
    if kind == "Laplacian":
        return np.abs(cv2.Laplacian(image, cv2.CV_64F, ksize=3))
    if kind == "Scharr":
        return cv2.magnitude(cv2.Scharr(image, cv2.CV_64F, 1, 0), cv2.Scharr(image, cv2.CV_64F, 0, 1))
    size = 5 if kind == "Sobel 5x5" else 3
    return cv2.magnitude(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=size), cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=size))

clean, truth = scene()
print(f"{'best F1 over thresholds':28s}" + "".join(f"  sd={s:<3d}" for s in (0, 20, 40, 60)))
for kind in ("Sobel 3x3", "Scharr", "Sobel 5x5", "Laplacian", "Gaussian 1.5 + Sobel 3x3", "Gaussian 1.5 + Laplacian"):
    row = ""
    for sd in (0, 20, 40, 60):
        noisy = clean + np.random.default_rng(sd).normal(0, sd, clean.shape)
        if kind.startswith("Gaussian"):
            noisy = cv2.GaussianBlur(noisy, (0, 0), 1.5)
        row += f"  {best_f1(gradient(noisy, kind.split('+ ')[-1]), truth):6.3f}"
    print(f"{kind:28s}{row}")
```

**Reading the output.** On this machine, best F1 over thresholds:

| Operator | sd 0 | sd 20 | sd 40 | sd 60 |
| --- | ---: | ---: | ---: | ---: |
| Sobel 3x3 | 0.999 | 0.945 | 0.559 | 0.333 |
| Scharr | 1.000 | 0.937 | 0.507 | 0.287 |
| Sobel 5x5 | 0.997 | 0.967 | 0.880 | 0.687 |
| Laplacian | 0.821 | 0.158 | 0.131 | 0.135 |
| Gaussian 1.5, then Sobel 3x3 | 0.980 | 0.958 | 0.890 | 0.884 |
| Gaussian 1.5, then Laplacian | 0.691 | 0.461 | 0.338 | 0.253 |

**Line by line.**

- `best_f1` measures distance to the true boundary with a distance transform, so thick bands of edge pixels are not penalised more than thin ones. There is no thinning stage here.
- `np.quantile(strength, ...)` places thresholds at the 70th to 99.5th percentile of the strength image, so each operator is swept over its own scale.
- `cv2.CV_64F` keeps negative derivatives. The magnitude is then built from both directions with `cv2.magnitude`.
- Noise is drawn from a generator seeded with `sd`, so every operator sees the same noisy image at a given level.

**What the numbers say.** Without noise every first-derivative operator is near perfect (0.997 to 1.000). The differences appear as noise grows. At sd 40, Sobel 3 by 3 keeps 0.559 and the Laplacian has collapsed to 0.131, because a second derivative amplifies noise more than a first. Even at sd 20 the Laplacian scores 0.158 against 0.945.

Two results run against habit. First, Scharr is slightly worse than Sobel under noise (0.507 against 0.559 at sd 40), which fits the hand calculation: its signal-to-noise ratio is 1.042 against 1.155. The calculation is a white-noise argument and the experiment is not proof of it, but the ordering agrees. Second, smoothing first beats every choice of operator. A Gaussian of sigma 1.5 before Sobel keeps 0.884 at sd 60, where Sobel alone has 0.333. The price is small: 0.980 against 0.999 on noise-free input, because the blur widens the edge.

Limits: one synthetic scene, thresholds chosen per method with knowledge of the answer (an optimistic oracle that real systems do not have), no non-maximum suppression, one tolerance, one noise draw per level. Real edges include texture and shadows that this scene lacks. The reason the Laplacian loses 0.18 even without noise was not isolated.

<Infographic src="/img/cv-enrich/v1-edge-noise.svg" alt="A line chart of best F1 against noise level for six edge detectors: operators without smoothing fall steeply, while a Gaussian before Sobel stays near 0.88." caption="Follow the green line: smoothing before Sobel keeps 0.884 at sd 60 while plain Sobel falls to 0.333." />

## Designing with it

Choose the input scale before the kernel. A three-pixel-wide edge operator on a tiny image and on a high-resolution image responds to different physical structures. If a target defect is only one or two pixels wide, a smoothing stage may erase it; if the image is noisy, skipping smoothing may yield a field of false edges. Compare the operator at the resolution used by the actual product and inspect small-object cases separately. Changing camera distance or resize policy changes the target scale in pixels, so it also changes what a fixed kernel means.

Choose whether polarity matters. For measuring a dark line on a bright background, the two sides may have opposite derivative signs. For generic contours, gradient magnitude discards polarity. For detecting a particular material transition, sign can be informative. A design that uses only unsigned output by accident is neither choice; it silently removes one direction. Keep signed computation and make an explicit decision about what to retain.

Set thresholds on a validation set, not on one attractive preview. A strong edge may be a shadow or specular highlight, and a true object boundary may be weak in low contrast. Test lighting, material variation and sensor noise. Measure the downstream task: a line detector may tolerate extra candidate edge pixels if a robust geometric stage rejects them, while a segmentation boundary may suffer when texture edges intrude. One global edge-count metric cannot tell whether the right boundaries survived.

Remember border handling. Convolution near an image edge needs values outside the array or a truncated support. Different reflection, constant or replication rules produce different border responses. If an object can touch the frame edge, border policy is part of accuracy. Keep it identical in training and serving, and log it with the kernel size, derivative depth and threshold.

Finally, inspect representative errors as images and numbers. Record gradient magnitude distributions for true boundaries and distractors. Visualise both signs and orientations. If all errors cluster in one capture setting, improve acquisition or stratify evaluation before adding a more complex detector. The gradient is a measurement of change, and its utility depends on how that measurement is used.

## Where this stands in 2026

:::info Industry view

- Gradient operators remain useful in camera calibration, document processing, shape measurement and as diagnostics for learned vision systems. Their output is local evidence, not a semantic label.
- The chapter code was run with `opencv-python-headless` 5.0.0.93. Official OpenCV documentation checked on 2026-10-02 displayed 4.13.0; the signed-derivative behaviour was verified in the local environment.
- The synthetic 800 response is an operator result for a specific 200-level band and kernel, not a transferable detection threshold.

:::

## Common mistakes

1. **Choosing an operator before measuring the noise.** Sobel against Scharr moved F1 by 0.05 at sd 40, while smoothing moved it by 0.55 at sd 60. Estimate the noise level of your camera first.
2. **Using the Laplacian on raw pixels.** It looks sharp on clean test images (0.821) and fails at sd 20 (0.158). Smooth first, then look for zero crossings.
3. **Tuning the threshold on the test image.** The F1 values above use the best threshold per method, which an unlabelled deployment cannot do. Choose the threshold on validation images and report the test score with that threshold.
4. **Counting an edge map as a boundary.** A pixel within 1 pixel of the truth is "correct", and this hides thick bands. If you need thin, connected contours, add non-maximum suppression, as the next chapter does.
5. **Forgetting the sign.** Keep derivatives in a signed type, as the chapter's earlier block shows, until you take a magnitude.

## Practice questions

<details>
<summary><strong>Q1.</strong> Define an edge and name three edge profiles.</summary>

An edge is a location of rapid intensity change (region boundary). Profiles: step (jump), ramp (gradual; real edges), roof (thin line).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> How does the image gradient detect edges?</summary>

The gradient ∇f=(∂f/∂x,∂f/∂y) is large where intensity changes fast. Magnitude √(Gx²+Gy²) flags edges; orientation arctan(Gy/Gx) is perpendicular to the edge.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Sobel gives Gx=4, Gy=3 at a pixel. Compute gradient magnitude and orientation.</summary>

Magnitude = √(16+9) = √25 = 5; orientation = arctan(3/4) = 36.87°.<br /><em>Session 3 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why smooth an image before differentiating?</summary>

Differentiation amplifies noise; Sobel/Prewitt masks smooth while estimating Gx,Gy so spurious responses are reduced.<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is a second-derivative edge method?</summary>

The Laplacian: edges appear at zero crossings of the second derivative (responds to intensity curvature).<br /><em>Session 3 · conceptual</em>

</details>

<details>
<summary><strong>Q6 (Easy).</strong> A 3 by 3 patch has columns 20, 20, 80. What are Gx and the gradient magnitude with the Sobel weights?</summary>

Gx = (80 − 20) × (1 + 2 + 1) = 240. Gy = 0, so the magnitude is 240.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> Compute the step gain, noise gain and ratio for the Prewitt kernel. Is Prewitt better or worse than Sobel by this measure?</summary>

The weights are three rows of (−1, 0, 1). Step gain = 6 / 2 = 3, noise gain = √6 = 2.45, ratio 1.225. That is higher than Sobel's 1.155. The measure assumes white noise and a perfect step, and it was not tested against a Prewitt run here, so treat it as a hypothesis for your data.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> The chapter's F1 scores use the best threshold for each method. Why is that optimistic and what should a production system do?</summary>

The best threshold is found using the answer, so each score is an upper bound. A production system chooses the threshold on labelled validation images, or relative to a noise estimate, and reports test performance with that fixed value. The gap between the oracle and the fixed value is itself a measure of how sensitive the method is to its threshold.

</details>

## Further reading

- [OpenCV image gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) for Sobel, Scharr, Laplacian and signed-output guidance.
- [OpenCV image gradients tutorial](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html), listed in the earlier version of this chapter; it could not be opened from the build environment on 2026-10-09 (HTTP 403), so the unsigned-output warning rests on the runs above.
- Versions run: OpenCV 5.0.0, SciPy 1.18.1, NumPy 2.5.3.
- The scene is generated in the code, so there is no image source or licence to state.
- Built from the course lecture "cv-s3-edges" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Diagnosing an edge result

Consider an edge map that lights up every letter printed on a package but misses a faint tear along the label. The operator may be functioning exactly as specified: printed text contains sharp changes and the tear is low contrast. Raising the threshold would remove more text edges only if their magnitude were lower than the tear, which is unlikely in this example. A better route may be to restrict the region of interest, compare expected label geometry, alter lighting to make the tear visible, or use texture information over a larger area. The error is in the definition of useful evidence, not necessarily in the derivative code.

Another map shows only one side of every bright rectangle. That pattern strongly suggests a sign or unsigned-type error. Check the signed derivative before thresholding and inspect both dark-to-light and light-to-dark transitions. The code's ten positive and ten negative samples give a controlled diagnostic. If one side disappears only after `CV_8U` conversion, changing the threshold cannot bring it back. Correct the representation before tuning parameters.

A third map has a thick band around each boundary. The gradient can be high over several adjacent pixels because a real edge is blurred into a ramp. A local maximum stage can thin the ridge by keeping the strongest response along the gradient normal. This motivates Canny in the next chapter. Thinning does not identify whether the band belongs to a target object, and it can break a faint contour if parameters are too aggressive. Separate the goals of finding change, thinning a ridge and recognising a meaningful structure.

The gradient angle must also be interpreted carefully. A horizontal line has a vertical gradient normal; a vertical line has a horizontal normal. An algorithm that confuses the normal with the line direction can mis-bin orientations in HoG or Hough processing. Test a tiny synthetic horizontal and vertical bar with known expected signs and angles before moving to real images. Then keep one coordinate convention throughout the pipeline.

An edge profile also controls localisation. A step blurred into a ramp produces a broad region of derivative response, and its strongest pixel may move if blur or sampling phase changes. A roof profile produces two opposite transitions around a thin bright line; at low resolution those responses can overlap. If the product needs a precise physical boundary, a thresholded edge pixel is only a first estimate. Calibrate the camera, inspect subpixel fitting where appropriate and report uncertainty against a measured reference. For small structures, one pixel of displacement can dominate an overlap metric even when the displayed outline looks acceptable. Signed responses help distinguish the two sides of a roof, while magnitude alone merges them. This is why acquisition, numerical representation and evaluation must be considered together. A unit test on one bright bar checks arithmetic, but a validation set with varying blur, light and target width checks whether that arithmetic serves the product.

Keep the raw image and the gradient map aligned when reviewing mistakes. A change in crop or border padding can move an edge relative to its annotation even when the derivative values are correct. Compare coordinates in one explicit frame of reference.

## Check yourself

- I can compute gradient magnitude and orientation from two components, including their signs.
- I can explain step, ramp and roof profiles and why real edges are often ramps.
- I can explain why unsigned derivative output can erase one transition direction.
- I can distinguish a strong intensity change from a verified object boundary.
- I can compute Gx, Gy, the magnitude and the angle from a 3 by 3 patch.
- I can score an edge map against a ground-truth boundary, and say why the best-threshold F1 is optimistic.
- I can explain why smoothing before a derivative matters more than the Sobel against Scharr choice.
