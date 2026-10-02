---
id: cv-classical-image-segmentation
title: "Computer Vision · Session 11; Classical Image Segmentation"
sidebar_label: "1 · Classical segmentation"
sidebar_position: 1
slug: /theory/cv/classical-image-segmentation
description: "Partition pixels with thresholds, k-means and spatial methods, and understand when the regions do not match objects."
tags: [computer-vision, segmentation, otsu, k-means]
---

import Infographic from '@site/src/components/Infographic';
import PixelClusterLab from '@site/src/components/viz/PixelClusterLab';

**In one line.** Classical segmentation divides an image into regions using measured properties and spatial constraints, before any region is necessarily a semantic object.

## The idea in plain words

:::note Beyond the lecture

The assumptions, failure analysis, local code and deployment choices extend the lecture. Its classical-method outline, pixel assignment and all five practice questions remain below.

:::

An image-level class answers what appears somewhere in a frame. Segmentation asks which pixels belong together or to a target. This distinction matters when a system must measure an area, isolate a foreground, locate a boundary or pass a region to later processing. Session 11 surveys classical segmentation: intensity thresholding, Otsu's automatic threshold, region growing, k-means, watershed, graph cuts and superpixels. These methods use different evidence. A threshold uses pixel value; region growing adds neighbourhood connectivity; k-means groups values in a feature space; graph-based methods combine similarity and spatial structure.

The simplest threshold labels a pixel foreground when its intensity exceeds $T$. A global threshold can work when foreground and background occupy separable intensity ranges under stable lighting. It fails when a dark corner of the target resembles the background or a bright reflection makes the background resemble the target. Otsu's method chooses a threshold that maximises between-class variance in the intensity histogram. This is a useful automatic criterion when the histogram represents two fairly separable populations. It cannot know which population is the desired object, and a good variance split is not the same as a good semantic mask.

The lecture's k-means example is one assignment step in one-dimensional intensity space. With centres 50 and 200 and pixel intensity 120, the distances are 70 and 80; the pixel joins centre 1. A full k-means loop then recomputes each centre from assigned pixels and repeats until a stopping rule is met. In a colour image, features might be three colour channels or a mixture of colour and position. Adding spatial coordinates can discourage remote regions with the same colour from merging, but the feature scales need normalisation. A raw 0–255 colour difference and a large image-coordinate difference cannot be combined meaningfully without a weighting decision.

Region growing starts from seeds and expands to neighbouring pixels that satisfy a similarity criterion. Seed placement matters: a seed in a reflection or shadow can grow the wrong region. Watershed views an image or gradient map as a landscape of basins; uncontrolled noise can create many tiny regions. Markers and preprocessing often help. Graph cuts formalise a global objective over label choices and neighbour relationships. Superpixels group nearby, similar pixels into small regions that can reduce later computation, but one superpixel is not necessarily one object. These techniques can be automatic, seeded or interactive depending on their formulation.

<Infographic src="/img/cv/classical-segmentation.svg" alt="Classical segmentation board: threshold and Otsu, k-means assigning intensity 120 to centre 50 rather than 200, and spatial methods such as region growing and watershed." caption="Grouping pixels by measurements does not automatically attach a semantic class." />

## How it works

### Thresholding & Otsu

Split by intensity: pixel>T is foreground. Otsu picks T automatically by maximising between-class variance (bimodal histogram). Global T fails under uneven light.

### Region growing & k-means

Region growing appends similar neighbours to seeds. k-means clusters pixels: assign to nearest centre → recompute centres → repeat.

:::tip

**Worked.** c1=50, c2=200, pixel=120 → |120−50|=70 &lt; |120−200|=80 → cluster 1.

:::

### Watershed, graph cuts, superpixels

Watershed floods intensity basins (over-segments); graph cuts and superpixels (SLIC) give precise, controllable boundaries. All classic methods are unsupervised; deep semantic segmentation comes next.

### Key takeaways

- **1 · Pixels**; Segmentation labels pixels into regions.
- **2 · Methods**; Threshold/Otsu, region growing, k-means.
- **3 · Advanced**; Watershed, graph cuts, superpixels.

## A real system that works this way

OpenCV's official thresholding tutorial demonstrates fixed thresholds, adaptive thresholding and Otsu threshold selection. Its watershed tutorial shows a marker-based route to separating touching objects, and its k-means tutorial describes iterative cluster assignment and centre updates. These official OpenCV 4.13.0 pages were checked on 2026-10-02. The chapter code below uses a synthetic array so the decisions are inspectable; it does not claim that any method succeeds on a particular real dataset.

Consider counting pale tablets on a dark inspection belt. If lighting is even and tablets do not touch, one threshold followed by connected-component analysis may be enough. If illumination falls off across the belt, a single global threshold might erase tablets in the dim region; adaptive thresholding can use local context, though it may also amplify texture. If tablets touch, a distance transform and marker-based watershed may separate them, but bad markers can split one tablet or merge two. The measure of success should be count error and boundary quality on real belt images, not whether a demonstration frame looks tidy.

For a different task, segmenting road, car and sky in natural scenes, brightness clusters do not correspond consistently to semantic classes. A blue car and blue sky may have similar colour; a shadowed and sunlit road may have different intensity. A learned semantic segmentation model with labelled examples can address the class problem, subject to its own data and generalisation limits. The next chapter compares semantic and instance outputs and evaluates their masks. Classical region methods can still serve as preprocessing, constraints or interpretable baselines.

## Code you can run

The first block reproduces the lecture's pixel assignment exactly. It checks both distances, then chooses the smaller. A tie would need an explicit convention; this example has no tie.

```python
pixel = 120
centres = [50, 200]
distances = [abs(pixel - centre) for centre in centres]
assignment = min(range(len(centres)), key=lambda index: distances[index])
print('Distances:', distances)
print('Cluster:', assignment + 1)
assert distances == [70, 80]
assert assignment == 0
```

The lab starts at the same pixel and centres. Moving them shows where the nearest-centre decision flips. It displays a tie explicitly and uses centre 1 as its deterministic tie convention. It is an assignment illustration, not a complete k-means run.

<PixelClusterLab />

The second block runs one full assignment-and-update step on a tiny list. Starting at centres 50 and 200, it groups 40, 60 and 120 with centre 1, and 180 and 220 with centre 2. The updated means are **73.333** and **200.000**. These numbers depend on the chosen data and starting centres; they do not prove that the method found semantic objects.

```python
pixels = [40, 60, 120, 180, 220]
centres = [50.0, 200.0]
groups = [[], []]
for pixel in pixels:
    index = min(range(2), key=lambda i: abs(pixel - centres[i]))
    groups[index].append(pixel)
updated = [sum(group) / len(group) for group in groups]
print('Groups:', groups)
print('Updated centres:', [round(value, 3) for value in updated])
assert groups == [[40, 60, 120], [180, 220]]
assert [round(value, 3) for value in updated] == [73.333, 200.0]
```

If a cluster becomes empty, its mean is undefined. Practical implementations have an explicit reinitialisation or stopping policy. For a colour image, the same loop works on feature vectors with a defined distance measure. Nearest-centre assignment ignores spatial continuity unless position is included in the features or a separate spatial rule is applied.

## Designing with it

Define the mask's meaning first. A region may mean “all pixels brighter than the belt”, “connected pixels near this seed”, or “all pixels of a particular object class”. These contracts differ. A k-means output can assign the same cluster to spatially disconnected patches, while a connected component is one contiguous region but may contain several touching objects. Label the output with its actual construction and test it against the downstream action. Do not call a colour cluster a detected object without validation.

Control acquisition when possible. Stable lighting, fixed camera geometry and a plain background can make a threshold pipeline cheaper and easier to maintain than a learned model. Conversely, if the product must work across sunlight, weather and arbitrary object colours, a fixed intensity rule is unlikely to survive. Inspect histograms and sample images across capture conditions before choosing a threshold strategy. A threshold tuned on one camera may fail when exposure or white balance changes.

Decide how spatial evidence enters the method. Pure k-means on intensity knows nothing about neighbours. A small bright highlight and a large bright object may share a cluster. Region growing enforces connectivity but depends on a seed and local acceptance rule. Graph cuts let a smoothness term discourage isolated label flips but require a good data term and edge weights. Watershed may be useful for touching components but is sensitive to markers. Each method can be described by the evidence it trusts and the failure it risks.

Evaluate with masks and object-level measures. Pixel overlap, covered area and boundary distance can expose different errors. A one-pixel shift along a long edge can lower overlap for a thin object even when its visual appearance looks close. If the product counts items, compare count and merge/split errors as well. Inspect images where a global histogram criterion succeeds numerically but the wrong physical region is selected. Segmentation quality cannot be inferred from the Otsu objective alone.

Plan the empty and ambiguous cases. A blank frame may contain no foreground, and a saturated frame may make almost every pixel foreground. Record how the pipeline handles both. For k-means, set the number of clusters, initialisation, stopping tolerance and random seed. For region growing, define seed selection and when growth stops. For watershed, define markers and postprocessing. These are model parameters in the operational sense even when there are no learned neural weights.

Keep transformations aligned. A resized or cropped mask needs to map back to the source image in the same coordinate convention. Nearest-neighbour interpolation is commonly used for discrete labels to avoid inventing mixed classes between pixels. If a product measures area in physical units, camera calibration and perspective matter; a count of pixels alone is not a square-centimetre measurement. Store input and output image dimensions with any segmentation result to make downstream geometry checkable.

## Where this stands in 2026

:::info Industry view

- OpenCV 4.13.0 tutorials checked on 2026-10-02 document fixed/adaptive/Otsu thresholds, k-means and marker-based watershed. The local examples use standard-library Python and have no OpenCV runtime dependency.
- The lecture says all classical methods are unsupervised. Graph cuts and region growing can use user or label-derived seeds and constraints, so “unsupervised” is not a universal property of these method families.
- No tablet-counting performance is claimed. The belt example is a design scenario, not a named company deployment.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> How does segmentation differ from classification?</summary>

Classification gives one label per image; segmentation labels each pixel, partitioning the image into regions.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What does Otsu's method optimise?</summary>

It automatically chooses the threshold T that maximises between-class variance (equivalently minimises within-class variance), assuming a bimodal histogram.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Describe the k-means segmentation loop.</summary>

Assign each pixel to the nearest cluster centre → recompute each centre as its members' mean → repeat until stable.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Centres c1=50, c2=200; pixel value 120. Which cluster?</summary>

|120−50|=70 vs |120−200|=80 → nearer c1, so cluster 1.<br /><em>Session 11 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What is a drawback of watershed, and one controllable alternative?</summary>

Watershed tends to over-segment. Alternatives with more control: graph cuts or superpixels (SLIC).<br /><em>Session 11 · conceptual</em>

</details>

## Further reading

- [OpenCV image thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) for fixed, adaptive and Otsu thresholds.
- [OpenCV k-means clustering](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html) for iterative assignment and centre updates.
- [OpenCV watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) for marker-based separation.
- Built from the course lecture "cv-s11-segmentation" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source claim

The lecture describes all classical segmentation methods as unsupervised. Thresholds and k-means can indeed be run without labelled masks, but a region-growing or graph-cut formulation may require human seeds, scribbles or learned terms. Watershed often relies on selected markers. Whether a workflow is unsupervised depends on its inputs, not only the family name. The source's method list remains above.

:::

## Diagnose a failed partition

If one tablet disappears only in a dim corner, compare local foreground and background intensity distributions. A global threshold may cut through the target's own range. Adjusting the threshold to include that corner may admit background elsewhere. Better lighting, shading correction or a local rule can address the cause. Check whether the new method still preserves tablet boundaries under bright conditions rather than tuning on the dim frame alone.

If a cluster contains both sky and a blue vehicle, the nearest-centre arithmetic may be correct. The failure is that colour alone does not distinguish the classes. Adding coordinates may separate sky from road-level vehicles in a fixed camera view, but it can also encode a shortcut that fails when camera placement changes. A learned representation or explicit object evidence may be required. The feature choice defines what “similar” means, so inspect the chosen feature space before increasing the number of clusters.

If watershed divides a textured tablet into several pieces, inspect the gradient or distance map and the markers that seeded flooding. Noise can create many local basins. Smoothing may remove false basins but can merge touching objects; marker selection can restrict the number of regions. Evaluate both split and merge errors, since one parameter change can exchange one failure for the other. A superpixel method faces a similar trade-off: too few superpixels cross true boundaries, while too many provide little computational simplification.

If a graph-cut mask follows a strong shadow edge instead of the object edge, examine the data term and smoothness term. A strong image edge is evidence of change, but it may be illumination rather than material or class. If a seed or unary likelihood favours the shadow region, the global optimisation can faithfully return the wrong mask. A more sophisticated solver cannot repair an incorrect objective. Compare the output with annotated examples and inspect error cases by lighting and object material.

The tiny k-means code offers a useful diagnostic model. The first assignment explains why 120 goes to centre 50, and the next update explains how the centre moves to 73.333. In a real run, repeating assignment can change membership again. Convergence to stable centres is a numerical condition, not proof of semantic correctness. Initial centres and feature scaling can lead to different local solutions. Record those settings and test stability over several initialisations before relying on a cluster label in a product workflow.

Finally, check coordinate handling. A mask produced on a downsampled image may look visually aligned when overlaid after a browser resize but be shifted by a pixel in source coordinates. For a narrow target, that shift can dominate an overlap measure. Keep a known test pattern through every resize and crop, then compare the mask and image on the same grid. This checks the engineering path separately from the segmentation algorithm.

## Check yourself

- I can distinguish an intensity cluster, a connected region and a semantic class.
- I can compute the lecture's nearest-centre assignment and one centre-update step.
- I can describe Otsu's objective and why it can choose the wrong physical region.
- I can choose an evaluation measure for the actual downstream action.
