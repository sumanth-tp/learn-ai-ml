---
id: cv-harris-corners-and-hog
title: "Computer Vision · Session 6; Harris Corners and HoG"
sidebar_label: "3 · Harris and HoG"
sidebar_position: 3
slug: /theory/cv/harris-corners-and-hog
description: "Separate distinctive corner detection from local shape description, and verify Harris and HoG worked numbers."
tags: [computer-vision, harris, hog, features]
---

import Infographic from '@site/src/components/Infographic';
import HarrisResponseLab from '@site/src/components/viz/HarrisResponseLab';

**In one line.** Harris finds image locations with change in two directions; HoG represents how local gradient directions are distributed.

## The idea in plain words

:::note Beyond the lecture

The exact descriptor geometry, caveats and runnable checks extend the lecture. Its Harris and HoG sequence and five original questions are retained below.

:::

An edge can often be located accurately across its direction but not along it. A long straight border looks similar at many points on the border. A corner changes in two directions, so its position is more distinctive in a local neighbourhood. The lecture uses flat, edge and corner patches to motivate the Harris detector. In a flat patch, shifts in any small direction change little. Along an edge, shifts across the boundary change the appearance but shifts along it may not. Around a corner, small shifts in both principal directions change the patch. That makes a corner useful for matching images or estimating camera motion, provided it can be detected repeatedly under the actual changes between images.

Harris builds a second-moment or structure tensor $M$ from local image gradients. Its two eigenvalues describe how much the local patch changes along principal directions. Two small eigenvalues suggest flatness, one large and one small an edge, and two large values a corner-like patch. The response $R=\det(M)-k\operatorname{trace}(M)^2$ avoids computing eigenvectors for every pixel. With eigenvalues $(1,1)$ and $k=0.04$, the determinant is 1, trace is 2 and $R=1-0.04(2)^2=0.84$. For $(1,0)$, the determinant is zero and $R=-0.04$. These are exercise-scale eigenvalues, not universal thresholds on real images.

A positive Harris response is only a candidate. Gradients must be computed at a stated scale, a neighbourhood window must be chosen, and local maxima must be selected to avoid many nearby points on the same corner. If the image is zoomed, a fixed window covers a different physical region. Harris is approximately rotation stable for local corner structure, but it is not inherently scale invariant. Illumination, blur, noise, repetitive patterns and viewpoint changes can also move or remove detected points. A useful feature detector should be judged by repeatability and localisation error over the transformations the application expects.

HoG, the histogram of oriented gradients, answers a different question. It describes the distribution of local edge directions across a window rather than picking one keypoint. The lecture's pipeline divides the image window into small cells, accumulates orientation bins in each cell, normalises groups of adjacent cells into blocks, then concatenates the numbers. Local normalisation reduces sensitivity to some brightness and contrast changes; it does not make the descriptor fully invariant to illumination or geometry. The vector is a representation that a classifier or matching stage can use, not itself a pedestrian decision.

The lecture gives **3,780** values for a traditional window layout: nine orientation bins, four cells per block and 105 overlapping blocks. The 105 count is not a general HoG constant. It follows from a 64 by 128 pixel window, 8 by 8 cells, 2 by 2 cells per block and a stride of one cell. Horizontally there are $64/8-1=7$ block positions; vertically $128/8-1=15$. Thus $7\times15=105$ and $105\times4\times9=3,780$. Change the window, cell, block or stride and the vector length changes.

<Infographic src="/img/cv/harris-hog.svg" alt="Harris distinguishes flat, edge and corner structure, with response 0.84 for eigenvalues one and one; HoG has 3780 values for a particular 105-block layout." caption="A corner detector supplies locations; a descriptor supplies values for comparing or classifying regions." />

## How it works

### Why corners?

Good features are distinctive, repeatable, invariant. Flat = no change; edge = change in one direction; corner = change in all directions (localisable in 2D).

### The Harris detector

Structure tensor M from windowed gradients; eigenvalues classify the patch. Response R = det(M) − k·trace(M)² avoids eigen-decomposition.

:::tip

**Worked.** λ1=λ2=1, k=0.04 → R = 1 − 0.04·4 = 0.84 (corner). Edge λ=(1,0) → R = −0.04. Rotation- but not scale-invariant → SIFT next.

:::

### Histogram of Oriented Gradients

Divide into cells → per-cell gradient-orientation histogram (9 bins) → group into blocks → normalize → concatenate. Illumination-invariant shape descriptor (classic pedestrian detection + SVM).

:::tip

**Worked.** 9 bins × 4 cells/block × 105 blocks = 3780-D (Dalal–Triggs).

:::

### Key takeaways

- **1 · Corners**; Distinctive; change in all directions.
- **2 · Harris**; R=det−k·trace²; not scale-invariant.
- **3 · HoG**; Normalized orientation histograms.

## A real system that works this way

OpenCV's Harris tutorial documents `cornerHarris` and the response from local gradients. Its 4.13.0 HOGDescriptor reference specifies the window and block geometry used in the 3,780-value calculation: 64 by 128 window, 16 by 16 block, 8 by 8 block stride and cell, nine bins. The implementation combines those descriptors with a detector for one named object category in its historical example, but this chapter only verifies the representation geometry. The current local `opencv-python-headless` 5.0.0.93 binding does not expose `HOGDescriptor`, so no call to that API is claimed to have run here. The dimension is computed directly from the documented layout in the Python block.

Consider matching two photographs of the same planar sign. Harris can propose distinctive corners, but those locations need a descriptor to compare neighbourhoods across images. Conversely, a dense HoG window can describe a candidate object region, but it does not tell which particular corner corresponds to another view. A robust geometry stage must reject false matches. This division of labour explains why a detector and a descriptor are often discussed together yet evaluated separately. A corner can be stable but indistinctive in a repeated grid; a descriptor can be informative but centred at a point that fails to reappear after scaling.

The source lecture cites classical pedestrian detection with HoG plus a classifier. That is a historical application pattern, not a claim that it is the best current pedestrian detector. Modern learned features may perform better under particular datasets and budgets, but the HoG construction remains a clear way to understand spatial pooling, orientation information and local normalisation. Its exact length can be checked without downloading a model or camera image.

## Code you can run

The first block verifies the lecture's Harris examples. At equal eigenvalues of 1, the response is **0.84**; at eigenvalues 1 and 0, it is **−0.04** for $k=0.04$. The lab begins with the corner-like case and lets you change the eigenvalues and $k$.

```python
def harris_response(first, second, k):
    determinant = first * second
    trace = first + second
    return determinant - k * trace ** 2

corner = harris_response(1, 1, 0.04)
edge = harris_response(1, 0, 0.04)
print(f'Corner response: {corner:.2f}')
print(f'Edge response: {edge:.2f}')
assert round(corner, 2) == 0.84
assert round(edge, 2) == -0.04
```

<HarrisResponseLab />

The second block computes the exact HoG length for the stated window geometry. It makes the **105 block** assumption visible. The count is a dimension check, not an extraction of a descriptor from an image.

```python
window_width, window_height = 64, 128
cell_width = cell_height = 8
block_cells_side = 2
stride_cells = 1
orientation_bins = 9
cells_x = window_width // cell_width
cells_y = window_height // cell_height
blocks_x = (cells_x - block_cells_side) // stride_cells + 1
blocks_y = (cells_y - block_cells_side) // stride_cells + 1
block_count = blocks_x * blocks_y
dimensions = block_count * block_cells_side ** 2 * orientation_bins
print(f'Blocks: {blocks_x} × {blocks_y} = {block_count}')
print(f'HoG dimensions: {dimensions}')
assert (blocks_x, blocks_y, block_count, dimensions) == (7, 15, 105, 3780)
```

:::note Correction to the source wording

The source calls corners “invariant” and HoG “illumination-invariant” without conditions. Harris response and HoG normalisation can tolerate some rotation or local contrast changes, but neither gives a blanket guarantee under scale, perspective, blur, saturation or arbitrary lighting. The 3,780 dimension belongs to the stated window geometry.

:::

## Designing with it

When detecting points, define what makes a good repeat. If the application stitches images with modest rotation but large zoom, a fixed-scale Harris detector may miss correspondences even when it finds strong corners in each individual image. A scale-space detector in the next lecture addresses this more directly. If the camera and target scale are fixed, Harris may be simpler and adequate. Evaluate repeatability across real view pairs, not only the number of corners in one image.

Tune the neighbourhood scale and non-maximum suppression together. A small window can respond to noise and texture, while a large window can merge nearby structures and shift localisation. A response threshold may need normalisation when image contrast changes. Use an image pyramid or explicit scale model if expected objects vary widely. Test low-texture scenes and repeated patterns because both can cause matching failures for different reasons.

For descriptors, keep the geometry explicit. Cell size, block size, block stride, bin count, signed or unsigned orientations and normalisation all affect vector length and behaviour. A classifier trained on one layout cannot consume a vector from another. Even if two layouts happen to have the same length, their entries can represent different spatial regions. Store feature configuration with the classifier and verify it at inference. The 3,780-value example is a contract, not just a classroom multiplication.

Local normalisation is a useful but limited defence against brightness changes. Shadows can alter gradient orientation, saturation can erase detail, and strong perspective changes deform local shape. Compare features across the expected capture conditions. If some group of images has different lighting or camera settings, report performance by that group. A high overall accuracy can hide systematic failure under one acquisition condition.

Finally, separate candidate quality from final model quality. More keypoints can increase potential matches and compute cost but also add ambiguous texture. A HoG classifier can score windows but still needs window selection, scale search, duplicate suppression and an evaluation metric for boxes. Inspect localisation errors, not only classification labels. The broader vision pipeline determines whether the feature representation is useful.

## Where this stands in 2026

:::info Industry view

- Classical corner and gradient descriptors remain useful for interpretation, controlled matching and low-compute baselines. Learned features often replace them in large-scale recognition, but the data and geometry assumptions remain relevant.
- OpenCV 4.13.0 documentation was checked on 2026-10-02. The local OpenCV 5.0.0.93 Python package lacks `HOGDescriptor`, so only the documented geometry was independently calculated and run.
- No pedestrian-detection accuracy is claimed from the 3,780-value calculation; feature length is not an evaluation result.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why are corners preferred over edges/flat regions as features?</summary>

Corners show intensity change in all directions, so they are localisable in 2D, distinctive, repeatable and invariant; flat has no change, an edge changes in only one direction.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the Harris corner response and interpret its sign.</summary>

R = det(M) − k·trace(M)² = λ1λ2 − k(λ1+λ2)². R large positive = corner; R&lt;0 = edge; |R| small = flat (k≈0.04–0.06).<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> λ1=λ2=1, k=0.04. Compute the Harris response.</summary>

det=1, trace=2, R = 1 − 0.04·2² = 1 − 0.16 = 0.84 (>0 → corner).<br /><em>Session 6 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Describe the HoG descriptor pipeline.</summary>

Divide into cells → per-cell gradient-orientation histogram (≈9 bins) → group into overlapping blocks → normalize → concatenate. Illumination-invariant shape descriptor.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Is Harris scale-invariant? What fixes it?</summary>

No; Harris is rotation-invariant but not scale-invariant. SIFT (scale-space extrema) provides scale invariance.<br /><em>Session 6 · conceptual</em>

</details>

## Further reading

- [OpenCV Harris corner tutorial](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) for response and implementation.
- [OpenCV HOGDescriptor reference](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html) for the precise default geometry.
- Built from the course lecture "cv-s6-harris-hog" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## A feature-matching design review

Suppose a robot must locate a printed marker across successive camera frames. First ask whether the marker has distinctive local corners. A plain rectangle has four useful corner positions, but many similar rectangles in the scene can confuse a matcher. A repeated checker pattern yields many high Harris responses that look alike. A detector score alone does not ensure unique correspondence. The pipeline should describe each local patch, compare candidate pairs, reject ambiguous matches and fit a geometric model that is consistent across several points.

If the robot approaches the marker, its apparent scale changes. A fixed Harris window may describe different physical regions before and after the motion. A scale-space method can improve repeatability by selecting a support size tied to image structure. This is the motivation for SIFT in the next chapter. If the marker's physical size and camera distance are tightly controlled, the simpler fixed-scale pipeline may still be reliable. The choice follows the operating envelope, not an absolute ranking of algorithms.

For a pedestrian-like object detector, HoG pools shape information over a fixed window. The calculation of 7 by 15 blocks shows overlapping coverage: neighbouring blocks share cells and provide local contrast normalisation around each position. That overlap contributes to descriptor length and cost. A detector operating at multiple scales repeats the computation or uses a pyramid; a 3,780-value vector from one window is not a complete image detector. When evaluating it, record candidate recall, classifier decision, box localisation and duplicate suppression separately.

The two examples also illustrate why feature invariance must be stated narrowly. Harris's response is based on local intensity structure, so significant scale or viewpoint change can alter it. HoG reduces the effect of uniform local contrast scaling but can still respond differently to shadow, blur or occlusion. Test the transformation distribution expected in deployment. If robustness is claimed, provide measured repeatability or task performance over that distribution, not a single visually convincing match.

Corner and descriptor configuration should be stored with the model artefact. The Harris neighbourhood and threshold determine where points are found; HoG's window, cell, block, stride, orientation and normalisation settings determine what each vector position means. If a training run uses one cell size and serving uses another, the resulting vector may differ in length, or worse, retain a plausible length with different semantics. Use a small fixed image patch as a regression fixture and compare intermediate gradients and final vector shape after dependency upgrades. This is relevant here because the local OpenCV 5 Python binding does not expose the HOGDescriptor API documented for OpenCV 4.13.0; the chapter verifies geometry without claiming the missing binding ran.

A fixed descriptor shape does not imply fixed numerical values. Changes in interpolation, border treatment and gradient normalisation can alter the vector while preserving 3,780 entries. Check values on a stable fixture as well as length.

## Check yourself

- I can explain how the two structure-tensor eigenvalues distinguish flat, edge and corner patches.
- I can reproduce the 0.84 Harris response from the lecture's numbers.
- I can derive 105 HoG blocks and 3,780 values from explicit window and stride settings.
- I can explain why a detector location and a descriptor vector are different outputs.
