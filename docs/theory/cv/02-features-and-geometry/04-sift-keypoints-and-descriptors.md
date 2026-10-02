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

## The idea in plain words

:::note Beyond the lecture

The runnable extraction, failure analysis and matching design extend the lecture. The source's four stages, 128-value calculation and original questions remain below.

:::

A corner detected with one fixed neighbourhood can disappear when the camera zooms. The same physical corner occupies a different number of pixels, so a fixed window sees a different patch. Session 7 introduces SIFT, the Scale-Invariant Feature Transform, to address that scale problem. It detects candidate points over a Gaussian scale space, refines them, assigns a local orientation and describes the surrounding gradient pattern. The result is a keypoint with location, scale and orientation plus a numeric descriptor. The descriptor can be compared between images; it is not an object class label by itself.

The first stage builds progressively smoothed images and differences adjacent Gaussian levels. A candidate is an extremum not only in x and y but also across neighbouring scales. Searching scale matters because a stable physical structure may appear as a strong response at one support size and a weak response at another. Difference of Gaussians is an efficient approximation to a scale-normalised Laplacian response, not an exact proof of scale invariance. Discrete octaves and thresholds limit what changes can be handled.

The second stage refines keypoint location and scale. Low-contrast candidates are unreliable because small noise can move their extrema. Edge-like responses are also rejected: a long edge is poorly localised along itself, so it creates ambiguous matches. This rejection echoes the distinction between edges and corners from the Harris chapter. It does not mean SIFT ignores all edge information; its descriptor is built from gradients around retained points. It means the selected point should be stable enough to locate again.

The third stage assigns a dominant orientation from local gradient directions. The descriptor is expressed relative to that orientation, making a rotated image more likely to produce comparable vectors. The fourth stage divides a neighbourhood into a 4 by 4 grid of spatial cells. Each cell accumulates an eight-bin gradient-orientation histogram. Concatenating these histograms gives $4	imes4	imes8=128$ values, followed by normalisation. The spatial grid preserves rough arrangement: two patches with the same total orientation counts but different positions can have different descriptors. The calculation explains vector length, not the full weighting, interpolation and clipping details of an implementation.

Matching usually compares a descriptor with candidates in a second image using a distance. The lecture cites the nearest-neighbour ratio test: accept a nearest match if its distance is sufficiently smaller than the second-nearest. For distances 0.30 and 0.50, the ratio is 0.60 and passes a teaching threshold of 0.8; for 0.48 and 0.50, the ratio is 0.96 and fails. A low ratio screens ambiguity among these candidates but does not prove geometric correctness. Repeated textures, viewpoint changes and outliers still require cross-checks or robust geometry.

<Infographic src="/img/cv/sift-features.svg" alt="SIFT detects Difference-of-Gaussians extrema, assigns orientation and describes a 4 by 4 grid with eight bins per cell, yielding 128 values." caption="Scale and orientation handling improve repeatability, but matching still needs ambiguity and geometry checks." />

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

The first block verifies the **128**-value layout and two ratio decisions. A 0.8 cutoff is a policy example from the lecture; it should be tuned and validated for a particular matching task. It cannot convert a descriptor distance into a probability of correctness.

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

## Further reading

- [OpenCV SIFT introduction](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) for the four stages.
- [OpenCV feature matching and homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) for a complete planar-matching pattern.
- Built from the course lecture "cv-s7-sift" (Lecture Library series).

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
