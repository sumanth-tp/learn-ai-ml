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

:::tip Before you start
**You should already know**

- What a pixel intensity is and how a histogram counts them: [colour, histograms and filtering](/docs/theory/cv/colour-histograms-and-filtering).
- What an image edge is, because several methods below stop at edges: [image gradients and edges](/docs/theory/cv/image-gradients-and-edges).
- How to take the mean of a list and measure the distance between two numbers.

**Reading time.** About 45 minutes, plus a minute to run the code.

**After this chapter you can**

- pick a threshold, a clustering or a seeded method for a simple image and say what evidence each one trusts,
- compute Otsu's threshold and one k-means step by hand,
- predict which method breaks when the lighting is uneven, and measure it with intersection over union (IoU).
:::

## In 30 seconds

Imagine pale tablets on a dark belt. If you can say "everything brighter than this is a tablet", you have segmented the image with one number. That works until a shadow dims one end of the belt and tablets there look like belt. Classical segmentation is a toolbox of such rules: a global cut-off, a local cut-off, groups of similar pixels, and floods that grow from seeds. None of them knows what a tablet is. They only know brightness, colour and position, so the skill is matching the rule to what the scene guarantees.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Segmentation | Giving every pixel a region label | Tablet or belt for each pixel |
| Threshold | A cut-off brightness | Brighter than 128 means foreground |
| Otsu's method | A rule that picks the threshold automatically | Splits the histogram where the two groups differ most |
| Between-class variance | How far apart the two group means are, weighted by group size | 3600 for means 65 and 185 with equal groups |
| k-means | Group pixels around k centres, then move each centre to its group mean | Centres 50 and 200 |
| Watershed | Flood a landscape from marked points until the floods meet | Separates two touching coins |
| GrabCut | Fits colour models to foreground and background inside a user rectangle | Draw a box, get a cut-out |
| IoU | Overlap divided by union of two masks | 1.0 is perfect, 0 is no overlap |

## The idea in plain words

:::note Beyond the course material

The assumptions, failure analysis, local code and deployment choices extend the course material. Its classical-method outline, pixel assignment and all five practice questions remain below.

:::

An image-level class answers what appears somewhere in a frame. Segmentation asks which pixels belong together or to a target. This distinction matters when a system must measure an area, isolate a foreground, locate a boundary or pass a region to later processing. Session 11 surveys classical segmentation: intensity thresholding, Otsu's automatic threshold, region growing, k-means, watershed, graph cuts and superpixels. These methods use different evidence. A threshold uses pixel value; region growing adds neighbourhood connectivity; k-means groups values in a feature space; graph-based methods combine similarity and spatial structure.

The simplest threshold labels a pixel foreground when its intensity exceeds $T$. A global threshold can work when foreground and background occupy separable intensity ranges under stable lighting. It fails when a dark corner of the target resembles the background or a bright reflection makes the background resemble the target. Otsu's method chooses a threshold that maximises between-class variance in the intensity histogram. This is a useful automatic criterion when the histogram represents two fairly separable populations. It cannot know which population is the desired object, and a good variance split is not the same as a good semantic mask.

The course's k-means example is one assignment step in one-dimensional intensity space. With centres 50 and 200 and pixel intensity 120, the distances are 70 and 80; the pixel joins centre 1. A full k-means loop then recomputes each centre from assigned pixels and repeats until a stopping rule is met. In a colour image, features might be three colour channels or a mixture of colour and position. Adding spatial coordinates can discourage remote regions with the same colour from merging, but the feature scales need normalisation. A raw 0–255 colour difference and a large image-coordinate difference cannot be combined meaningfully without a weighting decision.

Region growing starts from seeds and expands to neighbouring pixels that satisfy a similarity criterion. Seed placement matters: a seed in a reflection or shadow can grow the wrong region. Watershed views an image or gradient map as a landscape of basins; uncontrolled noise can create many tiny regions. Markers and preprocessing often help. Graph cuts formalise a global objective over label choices and neighbour relationships. Superpixels group nearby, similar pixels into small regions that can reduce later computation, but one superpixel is not necessarily one object. These techniques can be automatic, seeded or interactive depending on their formulation.

<Infographic src="/img/cv/classical-segmentation.svg" alt="Classical segmentation board: threshold and Otsu, k-means assigning intensity 120 to centre 50 rather than 200, and spatial methods such as region growing and watershed." caption="Grouping pixels by measurements does not automatically attach a semantic class." />

## Worked example, step by step

Six pixels: three dark ones (60, 70, 65) and three bright ones (180, 190, 185). We find Otsu's threshold, then run one k-means step. The first two blocks under "Code you can run" reproduce these numbers.

1. Try the split after 70, so the dark group is 60, 70, 65 and the bright group is 180, 190, 185.
2. Each group holds half the pixels, so both weights are 0.5. The dark mean is (60 + 70 + 65) / 3 = 65 and the bright mean is (180 + 190 + 185) / 3 = 185.
3. Between-class variance is weight one times weight two times the squared gap between means: 0.5 × 0.5 × (185 − 65)² = 0.25 × 14400 = 3600.
4. Try a worse split after 60: the dark group is just 60 and the bright group has five pixels with mean 138. The weights are 1/6 and 5/6, so the score is (1/6) × (5/6) × 78² = 845.
5. The split after 70 scores highest, so Otsu puts the threshold between 70 and 180.
6. For k-means with centres 50 and 200, a pixel at 120 is 70 from the first and 80 from the second, so it joins the first.
7. After assigning 40, 60 and 120 to centre one and 180 and 220 to centre two, the new centres are the group means: (40 + 60 + 120) / 3 = 73.333 and (180 + 220) / 2 = 200.

In words: Otsu asks "where does a cut make the two groups as different as possible?", and k-means asks "which centre is nearer?" and then moves the centres to where their members are.

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

The first block reproduces the pixel assignment exactly. It checks both distances, then chooses the smaller. A tie would need an explicit convention; this example has no tie.

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

**What each control does.** "Pixel" is the intensity being assigned. "Centre 1" and "Centre 2" are the two cluster centres. The data table shows both distances and the chosen cluster.

**Try it yourself.**

1. Leave the defaults (pixel 120, centres 50 and 200). The distances are 70 and 80, so the pixel joins centre 1. This is step 6 of the worked example.
2. Move the pixel to 125. Both distances are 75, a tie, and the lab applies its convention of choosing centre 1. Move it to 130 and it flips to centre 2, because the boundary between clusters sits halfway between the centres.
3. Set the pixel back to 120 and move centre 2 down to 150. The distances become 70 and 30, so the pixel now joins centre 2 even though nothing about the pixel changed. Centres define the clusters, which is why the experiment's position weight mattered so much.

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


The Otsu arithmetic above, scanned over every possible split of those six pixels, is short enough to run. It prints the score for each split, so you can check 3600.0 and 845.0 against the steps.

```python
import numpy as np

pixels = np.array([60, 70, 65, 180, 190, 185], dtype=float)
best = None
for split in np.unique(pixels)[:-1]:
    low, high = pixels[pixels <= split], pixels[pixels > split]
    w0, w1 = len(low) / len(pixels), len(high) / len(pixels)
    between = w0 * w1 * (low.mean() - high.mean()) ** 2
    print(f'split after {split:5.1f}: weights {w0:.3f} and {w1:.3f}, means {low.mean():.1f} and {high.mean():.1f}, between-class variance {between:.1f}')
    if best is None or between > best[1]:
        best = (split, between)
print('Otsu picks the split after', best[0])
```

**Reading the output.** The split after 70 scores 3600.0 and the split after 60 scores 845.0, matching steps 3 and 4. A wrong score would show a weight that does not sum to one or a mean taken over the wrong group.

If a cluster becomes empty, its mean is undefined. Practical implementations have an explicit reinitialisation or stopping policy. For a colour image, the same loop works on feature vectors with a defined distance measure. Nearest-centre assignment ignores spatial continuity unless position is included in the features or a separate spatial rule is applied.


### Experiment: which method survives uneven lighting?

The question a practitioner asks: if I tune nothing, which classical method keeps working when the belt is dimmer at one end? We generate a 256 by 256 belt with 14 pale discs, two of them touching, add Gaussian noise (standard deviation 10) and optionally dim the light linearly to 45% across the width. The true mask is known, so every method is scored with IoU. Each method gets default-style parameters and no tuning. Five random scenes per condition are averaged. Libraries: OpenCV 5.0.0 (`cv2`), SciPy 1.18.1, scikit-image 0.26.0, all run on this machine's CPU.

One sentence on why the block is long: each method is a few lines, but the scene, the scoring and the watershed splitter must all live in one self-contained block so it runs on its own.

```python
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def make_scene(seed, shading):
    rng = np.random.default_rng(seed)
    offsets = [np.array([21.0, 6.0]), np.array([-5.0, 22.0])] + [None] * 10
    centres = []
    for offset in offsets:
        while True:
            c = rng.uniform(30, 226, 2)
            group = [c] if offset is None else [c, c + offset]
            if all(np.hypot(*(g - o)) > 29 for g in group for o in centres):
                break
        centres += group
    truth = np.zeros((256, 256), np.uint8)
    for x, y in centres:
        cv2.circle(truth, (int(x), int(y)), 13, 1, -1)
    ramp = np.linspace(1.0, 0.45 if shading else 1.0, 256)[None, :]
    noisy = np.where(truth > 0, 170.0, 70.0) * ramp + rng.normal(0, 10, truth.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), truth.astype(bool)

def iou(pred, truth):
    return (pred & truth).sum() / max((pred | truth).sum(), 1)

def clean(mask):
    labels, n = ndi.label(mask)
    sizes = ndi.sum(mask, labels, range(1, n + 1))
    return np.isin(labels, 1 + np.flatnonzero(sizes >= 40))

def split_count(mask):
    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=8, labels=mask.astype(int), exclude_border=False)
    markers = np.zeros(mask.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return int(watershed(-distance, markers, mask=mask).max())

def kmeans_mask(image, xy_weight):
    ys, xs = np.mgrid[0:256, 0:256]
    features = np.stack([image.ravel() / 255.0, xy_weight * xs.ravel() / 256, xy_weight * ys.ravel() / 256], 1).astype(np.float32)
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centres = cv2.kmeans(features, 2, None, stop, 3, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(256, 256) == centres[:, 0].argmax()

def methods(image):
    out = {'fixed T=128': image > 128}
    out['Otsu'] = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    out['adaptive 51'] = cv2.adaptiveThreshold(cv2.GaussianBlur(image, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -8) > 0
    out['k-means gray'] = kmeans_mask(image, 0.0)
    out['k-means gray+xy'] = kmeans_mask(image, 0.5)
    gc = np.zeros((256, 256), np.uint8)
    cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), gc, (4, 4, 248, 248), np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    out['GrabCut rect'] = (gc == 1) | (gc == 3)
    return out

rows = {}
for shading in (False, True):
    for seed in range(5):
        image, truth = make_scene(seed, shading)
        for name, mask in methods(image).items():
            mask = clean(mask)
            row = rows.setdefault((shading, name), [[], [], []])
            row[0].append(iou(mask, truth))
            row[1].append(ndi.label(mask)[1])
            row[2].append(split_count(mask) if mask.any() else 0)
print('true: 14 tablets, components per scene', [ndi.label(make_scene(s, False)[1])[1] for s in range(5)])
for shading in (False, True):
    print('shading 1.0 to 0.45 across the belt' if shading else 'even light')
    for (s, name), (i, c, w) in rows.items():
        if s == shading:
            print(f'  {name:15s} IoU {np.mean(i):.3f}  components {np.mean(c):5.1f}  after watershed {np.mean(w):5.1f}')
```

**Reading the output.** Each row is one method. `IoU` is overlap with the true mask averaged over five scenes. `components` is the number of connected blobs after dropping blobs under 40 pixels. The scene has 14 tablets, two touching pairs, so a perfect mask has 12 blobs, and a perfect splitter returns 14 after watershed.

**Line by line.**

- `make_scene` builds the two touching pairs first, then places ten separate discs that stay at least 29 pixels apart.
- `clean` removes small blobs. Without it, noise speckles inflate every component count, and adaptive thresholding would look far worse than it is.
- `split_count` runs a distance transform on the mask, takes local maxima at least 8 pixels apart as markers, and floods from them. It counts regions, not pixels.
- `kmeans_mask(image, 0.5)` adds x and y to the colour features with weight 0.5. The weight is the whole point of that row.
- `cv2.grabCut` is given a rectangle that covers the image apart from a 4 pixel border, so it knows nothing about the tablets.

The printed output was:

```text
true: 14 tablets, components per scene [12, 12, 12, 12, 12]
even light
  fixed T=128     IoU 1.000  components  12.0  after watershed  14.0
  Otsu            IoU 1.000  components  12.0  after watershed  14.0
  adaptive 51     IoU 0.987  components  12.0  after watershed  14.0
  k-means gray    IoU 1.000  components  12.0  after watershed  14.0
  k-means gray+xy IoU 0.129  components   2.4  after watershed   6.4
  GrabCut rect    IoU 1.000  components  12.0  after watershed  14.0
shading 1.0 to 0.45 across the belt
  fixed T=128     IoU 0.317  components   4.8  after watershed  11.6
  Otsu            IoU 0.941  components  12.0  after watershed  19.0
  adaptive 51     IoU 0.977  components  12.0  after watershed  15.2
  k-means gray    IoU 0.933  components  12.4  after watershed  19.2
  k-means gray+xy IoU 0.143  components   3.6  after watershed  12.2
  GrabCut rect    IoU 1.000  components  12.0  after watershed  14.0
```

**What the numbers say.** With even light, the humblest method wins: a fixed threshold of 128 scores IoU 1.000 and so do Otsu, grey-level k-means and GrabCut. Adaptive thresholding is slightly worse (0.987), probably because it compares each pixel with its neighbourhood and so places some edges a pixel off; I did not test that. Once the light fades, the fixed threshold collapses to 0.317 and finds only 4.8 blobs, because every tablet in the dim half falls below 128. Otsu recovers most of the mask (0.941) since it re-estimates the threshold from the histogram, but the histogram is now smeared, so the cut is a compromise. Adaptive thresholding gives 0.977.

Three results deserve attention. First, adding position to k-means is destructive at weight 0.5: IoU falls to 0.129 on even light. Intensity differs by about 0.39 between tablet and belt after scaling to the range 0 to 1, while position spans 0.5, so the two clusters become left and right halves of the image. Feature scale decides what "similar" means. Second, the watershed splitter does not rescue a bad mask: after Otsu or grey k-means on the dimmed belt it returns 19.0 and 19.2 regions for 14 tablets, probably because ragged mask edges create extra distance peaks; I did not test that. It splits the touching pairs correctly only when the mask is clean (14.0). Third, GrabCut with a whole-image rectangle scored 1.000 under shading. That is the surprise, since it was given no hint about the tablets. Its colour models are mixtures of several Gaussians, which plausibly absorb a smooth brightness ramp on a two-level scene; I did not test that either.

<Infographic src="/img/cv-enrich/v3-classical-segmentation.svg" alt="Bars show mask IoU for six classical methods under even light and under a dimming belt, with cards for the k-means position-weight failure and the watershed over-splitting." caption="Look first at the fixed-threshold bar: perfect under even light, 0.317 once the belt dims." />

Limits: synthetic discs, one noise level, a linear light ramp, five scenes per condition, no parameter tuning and no real photographs. GrabCut would likely do worse on textured tablets or a belt with printed marks. Treat the table as a ranking of failure modes, not as a benchmark of the methods.

## Designing with it

Define the mask's meaning first. A region may mean “all pixels brighter than the belt”, “connected pixels near this seed”, or “all pixels of a particular object class”. These contracts differ. A k-means output can assign the same cluster to spatially disconnected patches, while a connected component is one contiguous region but may contain several touching objects. Label the output with its actual construction and test it against the downstream action. Do not call a colour cluster a detected object without validation.

Control acquisition when possible. Stable lighting, fixed camera geometry and a plain background can make a threshold pipeline cheaper and easier to maintain than a learned model. Conversely, if the product must work across sunlight, weather and arbitrary object colours, a fixed intensity rule is unlikely to survive. Inspect histograms and sample images across capture conditions before choosing a threshold strategy. A threshold tuned on one camera may fail when exposure or white balance changes.

Decide how spatial evidence enters the method. Pure k-means on intensity knows nothing about neighbours. A small bright highlight and a large bright object may share a cluster. Region growing enforces connectivity but depends on a seed and local acceptance rule. Graph cuts let a smoothness term discourage isolated label flips but require a good data term and edge weights. Watershed may be useful for touching components but is sensitive to markers. Each method can be described by the evidence it trusts and the failure it risks.

Evaluate with masks and object-level measures. Pixel overlap, covered area and boundary distance can expose different errors. A one-pixel shift along a long edge can lower overlap for a thin object even when its visual appearance looks close. If the product counts items, compare count and merge/split errors as well. Inspect images where a global histogram criterion succeeds numerically but the wrong physical region is selected. Segmentation quality cannot be inferred from the Otsu objective alone.

Plan the empty and ambiguous cases. A blank frame may contain no foreground, and a saturated frame may make almost every pixel foreground. Record how the pipeline handles both. For k-means, set the number of clusters, initialisation, stopping tolerance and random seed. For region growing, define seed selection and when growth stops. For watershed, define markers and postprocessing. These are model parameters in the operational sense even when there are no learned neural weights.

Keep transformations aligned. A resized or cropped mask needs to map back to the source image in the same coordinate convention. Nearest-neighbour interpolation is commonly used for discrete labels to avoid inventing mixed classes between pixels. If a product measures area in physical units, camera calibration and perspective matter; a count of pixels alone is not a square-centimetre measurement. Store input and output image dimensions with any segmentation result to make downstream geometry checkable.

## Where this stands in 2026

:::info Industry view

- OpenCV 4.13.0 tutorials checked on 2026-10-02 document fixed/adaptive/Otsu thresholds, k-means and marker-based watershed. The first blocks use the standard library only; the experiment uses OpenCV 5.0.0, SciPy and scikit-image.
- Course summaries often say all classical methods are unsupervised. Graph cuts and region growing can use user or label-derived seeds and constraints, so “unsupervised” is not a universal property of these method families.
- No tablet-counting performance is claimed. The belt example is a design scenario, not a named company deployment.

:::

## Common mistakes

- **Tuning one threshold on the best frame.** It feels right because the frame looks clean. A dim corner or a different exposure then fails silently, as the fixed threshold did at 0.317. Test on the worst lighting you expect, and consider Otsu or an adaptive rule.
- **Adding coordinates to k-means without scaling.** Position seems like free extra evidence. At weight 0.5 it overwhelmed intensity and halved the image. Scale each feature so the one you trust has the largest spread, and check by looking at the clusters.
- **Trusting watershed on a rough mask.** It looks like a ready-made splitter for touching objects. On a ragged mask it returned 19 regions for 14 tablets. Clean the mask and choose markers first.
- **Reading the Otsu objective as correctness.** A large between-class variance means the two groups differ, not that the bright group is the object you want. Check masks against labelled examples.
- **Counting pixels as area.** A pixel count is not square centimetres. Convert with the camera calibration before reporting a physical measurement.

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

<details>
<summary><strong>Q6 (Easy).</strong> Six pixels are 60, 70, 65, 180, 190, 185. What is the between-class variance for the split after 70?</summary>

Both groups have weight 0.5, with means 65 and 185. The score is 0.5 × 0.5 × 120² = 3600.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> In the experiment, a fixed threshold of 128 scores 1.000 under even light and 0.317 under dimming. Why does Otsu score 0.941 under dimming, and why not 1.000?</summary>

Otsu recomputes its threshold from the histogram of the image it is given, so it moves down towards the dimmed tablets. But the dim tablets and the bright belt end overlap in intensity, so no single cut separates them perfectly. A local method or a spatial model is needed to remove the remaining error.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> k-means with position weight 0.5 scored 0.129 on even light. Estimate which feature dominates and propose a weight that would let intensity win.</summary>

After scaling to 0 to 1, the tablet and belt differ by roughly (170 − 70) / 255 = 0.39 in intensity, while the x and y features each span 0.5 at weight 0.5. Distances are dominated by position, so the clusters follow space. A weight well below 0.1 shrinks position to a tie-breaker. The real fix is to pick the weight by looking at cluster maps on several scenes, not to trust one value.

</details>

## Further reading

- [OpenCV image thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) for fixed, adaptive and Otsu thresholds.
- [OpenCV k-means clustering](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html) for iterative assignment and centre updates.
- [OpenCV watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) for marker-based separation.
- [OpenCV GrabCut documentation string](https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html): rectangle initialisation and iteration count, read from the installed OpenCV 5.0.0 function help on 2026-10-09 because the web page returned HTTP 403 that day.
- [scikit-learn 1.9.1 Jaccard score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html), opened 2026-10-09, the library version of the IoU used throughout the next chapter.
- Built from the course lecture "cv-s11-segmentation" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source claim

Course summaries often describe all classical segmentation methods as unsupervised. Thresholds and k-means can indeed be run without labelled masks, but a region-growing or graph-cut formulation may require human seeds, scribbles or learned terms. Watershed often relies on selected markers. Whether a workflow is unsupervised depends on its inputs, not only the family name. The method list above stands.

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
- I can compute the worked nearest-centre assignment and one centre-update step.
- I can describe Otsu's objective and why it can choose the wrong physical region.
- I can choose an evaluation measure for the actual downstream action.

- I can compute Otsu's between-class variance for a small list of pixels and explain why the best split scores highest.
- I can explain why a fixed threshold, Otsu, an adaptive rule and GrabCut fail differently when the light fades.
- I can say why unscaled position features break k-means segmentation and how to check for it.

## Where to go next

Next: [semantic segmentation and mask metrics](/docs/theory/cv/semantic-segmentation-and-mask-metrics), which scores masks like the ones above with IoU and Dice. Related: [image gradients and edges](/docs/theory/cv/image-gradients-and-edges) for the edge evidence that watershed and graph cuts use.
