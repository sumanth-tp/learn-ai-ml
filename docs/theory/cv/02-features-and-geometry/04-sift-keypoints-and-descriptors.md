---
id: cv-sift-keypoints-and-descriptors
title: "Computer Vision · Session 7; SIFT Keypoints and Descriptors"
sidebar_label: "4 · SIFT features"
sidebar_position: 4
slug: /theory/cv/sift-keypoints-and-descriptors
description: "Trace SIFT scale-space detection, orientation assignment, 128-value descriptors and ambiguity-aware matching."
tags: [computer-vision, sift, keypoints, descriptors]
---

import Infographic from '@site/src/components/Infographic';
import SiftDescriptorLab from '@site/src/components/viz/SiftDescriptorLab';

**In one line.** SIFT seeks repeatable keypoints across image scales and orientations, then describes local gradient structure for matching.

:::tip Before you start
**You should already know**

- Why Harris corners are not scale invariant: [Harris corners and HoG](/docs/theory/cv/harris-corners-and-hog).
- What a gradient orientation histogram is: the HoG section of the same chapter.
- How a Euclidean distance between two vectors is computed.

**Reading time.** About 45 minutes, plus a minute to run the code.

**After this chapter you can**

- list the four SIFT stages and derive the 128-value descriptor length,
- apply the ratio test by hand and say what it keeps and what it throws away,
- read a table of match precision against ratio threshold, and explain why a geometry check is still needed.
:::

## In 30 seconds

Take a photograph of a poster from two metres away, then from one metre. The same corner now looks twice as big, so a detector with a fixed window sees a different patch. SIFT searches over many blur levels, so it picks each point at the size where it stands out, and it writes down the edge directions around that point as 128 numbers. To match two photographs you compare those numbers. A match is only trusted when the best candidate is clearly better than the second best, which is the ratio test. This chapter measures how many matches survive that test, and how many of them are right.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Keypoint | A point with a position, a size and a direction | A blob at (120, 80), size 6 pixels |
| Scale space | The same image blurred by more and more | Sigma 1.6, 2.3, 3.2 and so on |
| Difference of Gaussians | One blurred image minus the next blurrier one | A cheap blob detector |
| Descriptor | The 128 numbers that describe the patch around a keypoint | $4\times4$ cells with 8 bins each |
| Nearest neighbour | The candidate with the smallest distance to the query | Distance 0.30 |
| Ratio test | Accept only if nearest distance is less than a fixed fraction of the second nearest | 0.30 / 0.50 = 0.6 passes |
| Precision | Share of accepted matches that are correct | 215 correct of 244 kept is 0.88 |
| Homography | A 3 by 3 matrix that maps points between two views of a plane | Four corners of a poster |

## The idea in plain words

:::note Added to the course material

The runnable extraction, failure analysis and matching design go beyond the course notes. The four stages, the 128-value calculation and the original questions remain below.

:::

A corner detected with one fixed neighbourhood can disappear when the camera zooms. The same physical corner occupies a different number of pixels, so a fixed window sees a different patch. Session 7 introduces SIFT, the Scale-Invariant Feature Transform, to address that scale problem. It detects candidate points over a Gaussian scale space, refines them, assigns a local orientation and describes the surrounding gradient pattern. The result is a keypoint with location, scale and orientation plus a numeric descriptor. The descriptor can be compared between images; it is not an object class label by itself.

The first stage builds progressively smoothed images and differences adjacent Gaussian levels. A candidate is an extremum not only in x and y but also across neighbouring scales. Searching scale matters because a stable physical structure may appear as a strong response at one support size and a weak response at another. Difference of Gaussians is an efficient approximation to a scale-normalised Laplacian response, not an exact proof of scale invariance. Discrete octaves and thresholds limit what changes can be handled.

The second stage refines keypoint location and scale. Low-contrast candidates are unreliable because small noise can move their extrema. Edge-like responses are also rejected: a long edge is poorly localised along itself, so it creates ambiguous matches. This rejection echoes the distinction between edges and corners from the Harris chapter. It does not mean SIFT ignores all edge information; its descriptor is built from gradients around retained points. It means the selected point should be stable enough to locate again.

The third stage assigns a dominant orientation from local gradient directions. The descriptor is expressed relative to that orientation, making a rotated image more likely to produce comparable vectors. The fourth stage divides a neighbourhood into a 4 by 4 grid of spatial cells. Each cell accumulates an eight-bin gradient-orientation histogram. Concatenating these histograms gives $4\times4\times8=128$ values, followed by normalisation. The spatial grid preserves rough arrangement: two patches with the same total orientation counts but different positions can have different descriptors. The calculation explains vector length, not the full weighting, interpolation and clipping details of an implementation.

Matching usually compares a descriptor with candidates in a second image using a distance. The nearest-neighbour ratio test is the usual filter: accept a nearest match if its distance is sufficiently smaller than the second-nearest. For distances 0.30 and 0.50, the ratio is 0.60 and passes a teaching threshold of 0.8; for 0.48 and 0.50, the ratio is 0.96 and fails. A low ratio screens ambiguity among these candidates but does not prove geometric correctness. Repeated textures, viewpoint changes and outliers still require cross-checks or robust geometry.

<Infographic src="/img/cv/sift-features.svg" alt="SIFT detects Difference-of-Gaussians extrema, assigns orientation and describes a 4 by 4 grid with eight bins per cell, yielding 128 values." caption="Scale and orientation handling improve repeatability, but matching still needs ambiguity and geometry checks." />

## Worked example, step by step

**The descriptor length.** SIFT looks at a 16 by 16 pixel window around the keypoint.

1. Split it into a 4 by 4 grid of cells. Each cell is 4 by 4 pixels, so there are 16 cells.
2. In each cell, count edge directions into 8 bins of 45 degrees.
3. Join the 16 histograms: $16\times8=128$ numbers, then rescale to unit length.

In words: the grid remembers roughly where each edge direction sits, and the histograms forget exact pixel positions.

**The ratio test on two-number descriptors.** Real descriptors have 128 numbers, but two are enough to do the arithmetic. The query is $(1, 0)$, and the other image offers three candidates.

1. Clear case, candidates $(0.9, 0.1)$, $(0.5, 0.5)$ and $(0, 1)$. The distances are $\sqrt{0.01+0.01}=0.1414$, $\sqrt{0.25+0.25}=0.7071$ and $\sqrt{2}=1.4142$.
2. The nearest is 0.1414 and the second nearest is 0.7071, so the ratio is $0.1414/0.7071=0.20$. That is below 0.8, so the match is accepted.
3. Ambiguous case, candidates $(0.55, 0.45)$, $(0.5, 0.5)$ and $(0, 1)$. The nearest distance is $\sqrt{0.2025+0.2025}=0.6364$, the second is 0.7071, and the ratio is $0.6364/0.7071=0.90$.
4. 0.90 is above 0.8, so the match is rejected: two candidates look almost equally good, and the best one may be the wrong one.

In words: the ratio test does not ask "is the best candidate close?" It asks "is the best candidate much closer than the runner-up?"

## How it works

### The scale problem

Harris corners break under zoom. SIFT keypoints are invariant to scale and rotation, robust to illumination; enabling stitching, recognition and visual SLAM.

### The four stages

- **1 · DoG scale-space**; Difference-of-Gaussians pyramid; keypoints = extrema across space and scale.
- **2 · Localize**; Refine, reject low-contrast/edge points.
- **3 · Orientation**; Dominant gradient direction → rotation invariance.
- **4 · Descriptor**; Gradient-orientation histograms around the keypoint.

### The 128-D signature

16×16 window → 4×4 sub-regions → 8-bin orientation histogram each → 4×4×8 = 128-D, normalized.

:::tip

**Worked.** 4×4×8 = 128 dimensions. Match by Euclidean distance + Lowe's ratio test (nearest/second-nearest &lt; 0.8).

:::

### Key takeaways

- **1 · Invariance**; Scale + rotation.
- **2 · Stages**; DoG → localize → orient → describe.
- **3 · 128-D**; 4×4×8, ratio-test matching.

## A real system that works this way

The official OpenCV SIFT tutorial describes scale-space extrema, low-contrast and edge-response filtering, orientation assignment and descriptor formation. Its feature-homography tutorial then shows a real matching workflow: detect and describe points in a reference and scene image, compare descriptors, filter matches and estimate a homography to localise a planar object. The homography is a geometric check, not just another descriptor threshold. The current documentation opened on 2026-10-02 displayed OpenCV 4.13.0.

For a panorama, two overlapping photographs contain parts of the same scene from different camera poses. A set of reliable correspondences can support an image alignment model; one attractive nearest-neighbour match cannot. SIFT can help find correspondences despite moderate changes in scale and rotation, but moving objects and non-planar depth create mismatches to a single homography. RANSAC in the next chapter selects a transformation supported by a consistent subset. The complete stitching pipeline must also choose a warp, blend seams and handle exposure differences. A descriptor is one component of that system.

The runnable local extraction uses `cv2.SIFT_create()` in `opencv-python-headless` 5.0.0.93. It creates a synthetic checkerboard, detects points and verifies that returned descriptors have 128 columns. The exact number of detected keypoints is implementation and parameter dependent, so the chapter does not use that count as a general fact. No image is downloaded and no benchmark accuracy is inferred from the synthetic checkerboard.

## Code you can run

The first block verifies the **128**-value layout and two ratio decisions. A 0.8 cutoff is a common policy example; it should be tuned and validated for a particular matching task. It cannot convert a descriptor distance into a probability of correctness.

```python
cells_per_side = 4
orientation_bins = 8
dimensions = cells_per_side ** 2 * orientation_bins
ratio_clear = 0.30 / 0.50
ratio_ambiguous = 0.48 / 0.50
threshold = 0.8
print('Descriptor dimensions:', dimensions)
print(f'Clear ratio: {ratio_clear:.2f}; accepted: {ratio_clear < threshold}')
print(f'Ambiguous ratio: {ratio_ambiguous:.2f}; accepted: {ratio_ambiguous < threshold}')
assert dimensions == 128
assert ratio_clear < threshold < ratio_ambiguous
```

Change the cell-grid side or bin count in the lab. Its default **128 dimensions** matches the first block. The lab visualises layout arithmetic rather than claiming to perform full SIFT extraction.

<SiftDescriptorLab />

**What each control does.**

- *Cells per side* sets the grid size, from 2 to 6.
- *Orientation bins* sets the number of direction slots per cell, from 4 to 12.
- The table shows total cells and descriptor length.

**Try it yourself.**

1. Leave the grid at 4 by 4 with 8 bins. The length is 128, the value printed above.
2. Set the grid to 3 by 3. The length falls to 72 ($9\times8$). A shorter descriptor is cheaper to store and compare but remembers less layout.
3. Set 4 by 4 with 12 bins. The length is 192. Matching cost grows in step with it, so more bins must buy better matches to be worth it.

The second block checks a real local SIFT call on a synthetic 256 by 256 checkerboard. It prints the number of detected keypoints for the installed package and checks that every descriptor has **128 values**. The point count can change with OpenCV version and defaults, so only the descriptor width is asserted.

```python
import cv2
import numpy as np

image = np.zeros((256, 256), dtype=np.uint8)
for row in range(4):
    for column in range(4):
        if (row + column) % 2 == 0:
            image[row * 64:(row + 1) * 64, column * 64:(column + 1) * 64] = 255
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
print('Detected keypoints:', len(keypoints))
print('Descriptor shape:', None if descriptors is None else descriptors.shape)
assert descriptors is not None
assert descriptors.shape == (len(keypoints), 128)
```

This image has artificial high-contrast corners and repeated appearance. Repetition can make matches ambiguous even if many points are detected. A real matching evaluation needs different views, labelled correspondence or geometric consistency checks and a failure analysis over blur, occlusion and viewpoint change.

The next block reproduces the ratio-test arithmetic from the worked example, so the printed ratios 0.20 and 0.90 can be checked against the pencil version.

```python
import numpy as np

query = np.array([1.0, 0.0])
cases = {
    'clear': [(0.9, 0.1), (0.5, 0.5), (0.0, 1.0)],
    'ambiguous': [(0.55, 0.45), (0.5, 0.5), (0.0, 1.0)],
}
for name, candidates in cases.items():
    distances = sorted(np.linalg.norm(query - np.array(candidate)) for candidate in candidates)
    ratio = distances[0] / distances[1]
    print(f'{name:9s} distances {[round(float(d), 4) for d in distances]}  ratio {ratio:.2f}  accepted at 0.8: {ratio < 0.8}')
```

**Reading the output.** The clear case prints distances 0.1414 and 0.7071 and a ratio of 0.20, accepted. The ambiguous case prints 0.6364 and 0.7071, a ratio of 0.90, rejected.

### Experiment: how many matches survive the ratio test, and are they right?

The synthetic checkerboard in the previous block is too regular to learn from. These blocks use real images from scikit-image: the camera photograph (CC0, by its photographer) and a brick-wall texture (CC0 textures, per the scikit-image documentation). Each image is paired with a transformed copy whose exact mapping is known, so every match can be marked correct if it lands within 3 pixels of where the mapping sends it. The transformations are a brightness change, a 30 degree rotation, a 0.5 scale, and all three together.

```python
import cv2
import numpy as np
from skimage import data

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)
ratios = (0.6, 0.7, 0.8, 0.9, 1.0)


def make_pair(image, angle, scale, gain, offset):
    matrix = cv2.getRotationMatrix2D((255.5, 255.5), angle, scale)
    warped = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_LINEAR)
    warped = np.clip(warped.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    return warped, matrix


def evaluate(image, angle, scale, gain, offset, tolerance=3.0):
    warped, matrix = make_pair(image, angle, scale, gain, offset)
    keypoints1, descriptors1 = sift.detectAndCompute(image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(warped, None)
    points1 = np.array([k.pt for k in keypoints1])
    points2 = np.array([k.pt for k in keypoints2])
    projected = points1 @ matrix[:, :2].T + matrix[:, 2]
    pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)
    rows = []
    for ratio in ratios:
        kept = [best for best, second in pairs if best.distance < ratio * second.distance]
        correct = sum(np.linalg.norm(points2[m.trainIdx] - projected[m.queryIdx]) <= tolerance for m in kept)
        rows.append((len(kept), correct))
    return len(keypoints1), len(keypoints2), rows


print('OpenCV', cv2.__version__)
cases = [('camera', data.camera(), 'brightness x0.5 +20', (0, 1.0, 0.5, 20)), ('camera', data.camera(), 'rotate 30', (30, 1.0, 1.0, 0)),
         ('camera', data.camera(), 'scale 0.5', (0, 0.5, 1.0, 0)), ('camera', data.camera(), 'rotate 30, scale 0.7, dim', (30, 0.7, 0.6, 20)),
         ('brick', data.brick(), 'rotate 30, scale 0.7, dim', (30, 0.7, 0.6, 20))]
print('ratio threshold'.ljust(40) + ''.join(f'{ratio:>8}' for ratio in ratios))
for name, image, label, args in cases:
    count1, count2, rows = evaluate(image, *args)
    print(f'{name} {label} ({count1} and {count2} keypoints)')
    print('  matches kept'.ljust(40) + ''.join(f'{kept:>8d}' for kept, _ in rows))
    print('  correct matches'.ljust(40) + ''.join(f'{correct:>8d}' for _, correct in rows))
    print('  precision'.ljust(40) + ''.join(f'{correct / kept:>8.2f}' for kept, correct in rows))
```

**Reading the output.** Each case prints three rows over five ratio thresholds. "Matches kept" is how many query descriptors pass the test. "Correct matches" is how many of those land at the right place. "Precision" divides the second by the first. At 1.0 every nearest neighbour is accepted, which is matching with no ratio test.

**Line by line.**

- `knnMatch(..., k=2)` returns the nearest and second-nearest candidate for each query descriptor, which is exactly what the ratio needs.
- `projected` is where the known transform sends each keypoint of the first image, so a match is judged against ground truth rather than by eye.
- `best.distance < ratio * second.distance` is the test from the worked example.

A good match list is only half the story. The next block takes the combined case, fits a homography with RANSAC at three thresholds, and compares the trials the formula asks for, the inliers and the worst corner error of the fitted transform.

```python
from math import ceil, log

import cv2
import numpy as np
from skimage import data

image = data.camera()
matrix = cv2.getRotationMatrix2D((255.5, 255.5), 30, 0.7)
warped = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_LINEAR)
warped = np.clip(warped.astype(np.float32) * 0.6 + 20, 0, 255).astype(np.uint8)
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image, None)
keypoints2, descriptors2 = sift.detectAndCompute(warped, None)
points1 = np.array([k.pt for k in keypoints1])
points2 = np.array([k.pt for k in keypoints2])
projected = points1 @ matrix[:, :2].T + matrix[:, 2]
pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors1, descriptors2, k=2)
frame = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
true_frame = frame @ matrix[:, :2].T + matrix[:, 2]

print('ratio  kept  correct  inlier fraction  trials for 99%  homography inliers  worst corner error (px)')
for ratio in (0.8, 0.9, 1.0):
    kept = [best for best, second in pairs if best.distance < ratio * second.distance]
    src = np.float32([points1[m.queryIdx] for m in kept])
    dst = np.float32([points2[m.trainIdx] for m in kept])
    correct = int(sum(np.linalg.norm(points2[m.trainIdx] - projected[m.queryIdx]) <= 3 for m in kept))
    fraction = correct / len(kept)
    trials = ceil(log(0.01) / log(1 - fraction ** 4))
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.99)
    mapped = cv2.perspectiveTransform(frame.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = np.abs(mapped - true_frame).max()
    print(f'{ratio:>5}  {len(kept):>4}  {correct:>7}  {fraction:>15.3f}  {trials:>14d}  {int(mask.sum()):>18d}  {error:>23.2f}')
```

**Reading the output.** "Inlier fraction" is the correct share of kept matches. "Trials for 99%" applies the RANSAC formula to that fraction with four points per sample. "Worst corner error" maps the four image corners through the fitted homography and reports the largest distance from where the true transform puts them.

**Line by line.**

- `fraction ** 4` is the chance that four random matches are all correct, the $w^s$ of the RANSAC chapter.
- `cv2.findHomography(..., cv2.RANSAC, 3.0, maxIters=5000, confidence=0.99)` runs the robust fit with a 3 pixel inlier threshold.
- `true_frame` is built from the known matrix, so the corner error is exact.

#### Reading the experiment

The ratio test does what the textbook says. On the combined change, with no test, 791 matches are accepted and only 226 are right, a precision of 0.29. At a ratio of 0.8 there are 244 matches and 215 are right, a precision of 0.88. The test removed 536 of the 565 wrong matches (94.9%) at the price of 11 of the 226 correct ones (4.9%). Lowe reports at this threshold that 90% of false matches go and under 5% of correct ones, measured on a different database, so the two agree.

Texture and scale make it harder. On the brick wall, precision at 0.8 is 0.79 and 285 of 316 correct matches survive, a loss of 9.8%. At a scale of 0.5 on the camera image, precision at 0.8 is 0.81, and only 265 of 791 keypoints survive in the small copy. Halving the contrast and adding an offset also halves the keypoint count (791 to 412) but the precision stays at 0.97, because the lost keypoints are the low-contrast ones.

The surprise is in the second table. RANSAC found the same geometry with and without the ratio test: 215, 221 and 227 inliers, and a worst corner error of 0.42, 0.23 and 0.22 pixels. The unfiltered list gave the most accurate transform, because it kept 11 more correct matches. What the test buys is speed: the formula asks for 5 trials at 0.8 and 689 with no test.

Limits: one pair per case, an exact synthetic transform, a scene with no moving objects or depth, and 3 pixel tolerance. In a real scene outliers can agree with each other, and then a cleaner list matters more.

<Infographic src="/img/cv-enrich/v2-sift-ratio.svg" alt="Left: a table for the combined transform showing matches kept, correct matches, precision and RANSAC trials at ratio thresholds 0.8, 0.9 and 1.0. Right: precision by threshold for five cases." caption="Read the left table downwards: precision rises from 0.29 to 0.88 while RANSAC trials fall from 689 to 5." />

## Designing with it

Choose features for the transformation range. If the camera can rotate and zoom, scale and orientation handling matter. If the object deforms or is seen from a very different viewpoint, a local descriptor may still change too much. Record expected scale ratios, blur, lighting and perspective conditions and test image pairs sampled from them. Do not use “invariant” as an unqualified performance guarantee.

Check keypoint coverage, not just count. A thousand points on a textured background and none on the object of interest can make a matcher look busy while yielding poor object localisation. Visualise points across regions, scales and lighting slices. Set contrast and edge thresholds so that candidates are repeatable rather than merely numerous. If a target is smooth, a keypoint approach may lack stable local structure; a region, contour or learned feature may be more suitable.

Keep descriptor distance and match validity separate. The ratio test rejects cases where the best and second-best candidates are similarly close. It does not catch every false best match, and a strict ratio can discard true correspondences in repetitive scenes. Symmetric matching, spatial constraints and a robust geometric model can add evidence. In a panorama, a homography should be checked for plausible geometry and inlier distribution, not accepted solely because it has many matches near one small image patch.

Be precise about descriptor normalisation and distance metric. SIFT's 128 values are floating-point gradient histograms, commonly compared with Euclidean distance in the official workflow. Binary descriptors such as ORB require a different metric. Mixing descriptor types or changing normalisation between indexing and queries can produce nonsensical rankings while preserving array dimensions. Version the extraction parameters and validate the descriptor data type and range at the matching boundary.

Finally, budget compute and memory. A high-resolution image can yield many keypoints, each with a 128-value descriptor. The pipeline pays to detect, store, search and geometrically verify candidates. Reducing image resolution or capping points changes coverage, especially for small objects. Measure quality and latency together on the target hardware. This chapter's toy checkerboard demonstrates a mechanism, not operational throughput.

## Where this stands in 2026

:::info Industry view

- SIFT is a classical reference for scale-aware local features; modern learned local and global descriptors are also used. The right choice depends on the scene, transformations, quality target and compute budget.
- The local `cv2.SIFT_create()` call ran with `opencv-python-headless` 5.0.0.93. The official OpenCV tutorial checked on 2026-10-02 displayed version 4.13.0; the tested descriptor width is 128.
- A nearest-neighbour ratio below 0.8 is a heuristic matching filter, not a guarantee that a correspondence is correct.

:::

## Common mistakes

1. **Believing the ratio test proves a match.** A ratio below 0.8 feels like a guarantee. At 0.8 on the brick wall, precision was still 0.79, so about one match in five was wrong. Follow it with a geometric check.
2. **Throwing away the data to clean it.** A strict threshold looks safer. The test also dropped 11 correct matches out of 226 here, and the fitted homography ended up less accurate (0.42 px against 0.22 px). Pick the threshold by the downstream geometry, not by precision alone.
3. **Using one threshold for every scene.** 0.8 is a convention. Precision at 0.8 ranged from 0.97 (brightness only) to 0.79 (brick) in the experiment. Tune on pairs from your own cameras.
4. **Treating "scale invariant" as unlimited.** A half-size copy kept only 265 of 791 keypoints and precision at 0.8 fell to 0.81. Test the zoom range you expect.
5. **Counting keypoints as quality.** A dimmer copy had half as many keypoints and equal precision. Quality is how many survive and agree, not how many are detected.

## Practice questions

<details>
<summary><strong>Q1.</strong> What problem does SIFT solve that Harris does not?</summary>

Scale invariance (and rotation invariance); Harris corners are not repeatable under scale change; SIFT detects keypoints across a scale-space.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> List the four stages of SIFT.</summary>

DoG scale-space extrema detection → keypoint localization → orientation assignment → descriptor generation.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> How does SIFT achieve scale invariance?</summary>

By detecting extrema of the Difference-of-Gaussians (DoG) across a Gaussian pyramid; extrema across space AND scale; an efficient approximation to the scale-normalized Laplacian.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Why is the SIFT descriptor 128-dimensional?</summary>

4×4 spatial sub-regions × 8 orientation bins = 128, then normalized to unit length for illumination invariance.<br /><em>Session 7 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What is Lowe's ratio test?</summary>

Accept a match only if the nearest-neighbour distance is much smaller than the second-nearest (ratio &lt; 0.8), rejecting ambiguous matches.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> Easy. A query has nearest distance 0.42 and second-nearest distance 0.50. Does the ratio test at 0.8 accept it?</summary>

The ratio is $0.42/0.50=0.84$, which is above 0.8, so the match is rejected. The best candidate is not clearly better than the runner-up.<br /><em>Easy · numeric</em>

</details>

<details>
<summary><strong>Q7.</strong> Medium. With no ratio test the combined case had 791 matches and 226 correct. RANSAC still fitted the right transform. Why is the ratio test used anyway?</summary>

With 226 correct of 791 the inlier fraction is 0.286, so four-point RANSAC needs $\lceil\log(0.01)/\log(1-0.286^4)\rceil=689$ trials for 99% confidence. At a ratio of 0.8 the fraction is 0.881 and 5 trials suffice. The test saves time. It can also matter more than here when the wrong matches agree with each other.<br /><em>Medium · interpretation</em>

</details>

<details>
<summary><strong>Q8.</strong> Stretch. Precision at ratio 0.8 was 0.88 on the camera image and 0.79 on the brick wall under the same transform. Give two reasons and one test.</summary>

Bricks repeat, so many different keypoints have nearly identical descriptors, and the best and second-best candidates both look right. The wall also has fewer distinctive structures than a photograph. A test: tighten the threshold to 0.7 and compare. Measured above, the camera rises from 0.88 to 0.96 and the brick from 0.79 to 0.88, so tightening helps both and the gap between them stays.<br /><em>Stretch · interpretation</em>

</details>

## Further reading

- [OpenCV SIFT introduction](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) for the four stages.
- [OpenCV feature matching and homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) for a complete planar-matching pattern.
- Built from the course lecture "cv-s7-sift" (Lecture Library series).

- Lowe, "Distinctive image features from scale-invariant keypoints", International Journal of Computer Vision, 2004 (opened 2026-10-09, section 7.1): the 0.8 threshold "eliminates 90% of the false matches while discarding less than 5% of the correct matches" on his test database.
- OpenCV 5.0.0 `SIFT_create`, `BFMatcher` and `findHomography`, run locally; the `findHomography` docstring lists the least-squares, RANSAC, LMEDS and RHO methods. The online reference pages returned an access error on 2026-10-09, so the earlier 2026-10-02 reading stands.
- scikit-image 0.26.0 sample images (camera, CC0; brick, CC0), licences read from the library's own documentation.

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## From local matches to a trusted transform

Suppose two images overlap but one was taken closer to the scene. The first image may show a small corner support region while the second shows a larger version. A scale-space detector seeks a scale at which each local structure is stable, then normalises the descriptor's neighbourhood accordingly. Orientation assignment helps if the camera rotated. These steps increase the chance that corresponding physical patches have similar vectors, but the descriptor still contains only local evidence.

The matcher needs an explicit candidate population. A ratio test compares the best and second-best distances within that population. If the true match is absent because it was occluded or never detected, a low ratio can still arise from an unrelated repeated pattern. Conversely, if several nearly identical windows are genuine correspondences, the ratio can reject all of them. A clean ratio is a useful clue, not a ground-truth label. Inspect match lines and their spatial distribution before fitting geometry.

The geometry model adds a global relationship. For a planar scene, a homography can map points between views, but a minimum of four non-collinear point pairs is only enough to compute a candidate; outliers and noise demand more support. RANSAC samples candidate sets, scores inliers and refits. Even a high inlier count can be misleading if all inliers lie in one tiny region and extrapolation is required elsewhere. Check reprojection residuals across the image and whether the warped corners remain plausible.

SIFT also illustrates a distinction between mathematical invariance and system robustness. The detector is designed for changes in scale and rotation, yet image formation can defeat it with motion blur, saturation, repetitive texture, occlusion or extreme viewpoint change. Robustness is measured over a specified transformation range and task outcome. A statement that a descriptor “survives scale” should be read as a design goal with limits, not a promise that every scaled photograph will match.

The synthetic checkerboard is useful because its structure is predictable, but it is a poor proxy for a diverse matching set. Many corners repeat, so a distance ratio may reject true correspondences as ambiguous; small implementation changes can alter the keypoint count. A credible evaluation pairs images with known correspondences and varying blur, scale, rotation and viewpoint. Measure how often the same physical point is detected, how accurately its location is recovered, and how often a proposed match survives geometric verification. Also count time and memory per image at the intended resolution. If the application needs a panorama, evaluate final alignment and seam quality as well as descriptor matching. These measurements tell a stronger story than the number of keypoints in one demonstration frame. They also reveal when a different representation is needed because the scene lacks stable local texture.

A repeatability score without localisation tolerance can also mislead: finding nearby but shifted points may be enough for rough retrieval and inadequate for precise geometric measurement. State the pixel tolerance and camera scale.

The comparison should include both easy overlaps and the difficult transformations expected after deployment.

## Check yourself

- I can explain why fixed-window Harris corners may fail after zoom and how SIFT searches scale.
- I can name SIFT’s detection, refinement, orientation and descriptor stages.
- I can derive 128 descriptor values from 4 by 4 cells with eight bins each.
- I can explain what a ratio test filters and why a geometry check is still required.
- I can apply the ratio test to two candidate distances by hand.
- I can read a precision-against-threshold table and say what 0.8 keeps and drops.
- I can explain why RANSAC found the same homography with and without the ratio test, and what the test still saves.
