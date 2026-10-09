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

:::tip Before you start
**You should already know**

- What an image gradient is and how a filter measures it: [image gradients and edges](/docs/theory/cv/image-gradients-and-edges).
- How edges are picked out of a gradient image: [Canny edges and Hough lines](/docs/theory/cv/canny-edges-and-hough-lines).
- What a linear classifier does with a vector of features: [classification and logistic regression](/docs/theory/ml/classification-and-logistic-regression).

**Reading time.** About 45 minutes, plus a minute to run the code.

**After this chapter you can**

- tell a flat patch, an edge and a corner apart from two eigenvalues, and compute the Harris response by hand,
- derive the 3,780-value HoG length from window, cell, block and stride,
- say, with measured numbers, what rotation, noise, scale and contrast do to corner repeatability and to a HoG classifier.
:::

## In 30 seconds

Picture a chessboard photograph and slide a small window over it. On a plain square nothing changes. Along a straight edge the window changes only when you cross the edge. On the corner of a square it changes whichever way you move. Harris turns that into one number per pixel, and the pixels with the highest numbers are the corners. HoG does a different job: it describes the shape inside a window by counting which way the edges point in small cells, then rescales each group of cells so a dimmer photo gives nearly the same numbers. Here you will measure how often corners survive rotation, noise and zoom, and how HoG compares with raw pixels.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Corner | A spot where the image changes in two directions at once | The corner of a window frame |
| Structure tensor $M$ | A 2 by 2 table that adds up how the gradients in a window point | Eight units along x, eight along y: $M=\begin{pmatrix}8&0\\0&8\end{pmatrix}$ |
| Eigenvalue | How strongly the window changes along one principal direction | Two large values mean a corner |
| Harris response $R$ | $\det(M)-k\,\operatorname{trace}(M)^2$, one number per pixel | $R=53.76$ for the corner below |
| Repeatability | Share of corners found in one image that are found again in a changed copy | 0.795 after a 5 degree turn |
| Cell | A small square of pixels, 8 by 8 in the classic layout, that gets one orientation histogram | 64 pixels, 9 bins |
| Block | A group of neighbouring cells that are rescaled together | 2 by 2 cells |
| Orientation bin | One slot of a histogram of edge directions | Nine slots of 20 degrees each |
| HoG descriptor | All normalised block histograms joined into one long vector | 3,780 numbers for a 64 by 128 window |

## The idea in plain words

:::note Added to the course material

The exact descriptor geometry, caveats and runnable checks go beyond the course notes. The Harris and HoG sequence and the five original questions are retained below.

:::

An edge can often be located accurately across its direction but not along it. A long straight border looks similar at many points on the border. A corner changes in two directions, so its position is more distinctive in a local neighbourhood. Flat, edge and corner patches motivate the Harris detector. In a flat patch, shifts in any small direction change little. Along an edge, shifts across the boundary change the appearance but shifts along it may not. Around a corner, small shifts in both principal directions change the patch. That makes a corner useful for matching images or estimating camera motion, provided it can be detected repeatedly under the actual changes between images.

Harris builds a second-moment or structure tensor $M$ from local image gradients. Its two eigenvalues describe how much the local patch changes along principal directions. Two small eigenvalues suggest flatness, one large and one small an edge, and two large values a corner-like patch. The response $R=\det(M)-k\operatorname{trace}(M)^2$ avoids computing eigenvectors for every pixel. With eigenvalues $(1,1)$ and $k=0.04$, the determinant is 1, trace is 2 and $R=1-0.04(2)^2=0.84$. For $(1,0)$, the determinant is zero and $R=-0.04$. These are exercise-scale eigenvalues, not universal thresholds on real images.

A positive Harris response is only a candidate. Gradients must be computed at a stated scale, a neighbourhood window must be chosen, and local maxima must be selected to avoid many nearby points on the same corner. If the image is zoomed, a fixed window covers a different physical region. Harris is approximately rotation stable for local corner structure, but it is not inherently scale invariant. Illumination, blur, noise, repetitive patterns and viewpoint changes can also move or remove detected points. A useful feature detector should be judged by repeatability and localisation error over the transformations the application expects.

HoG, the histogram of oriented gradients, answers a different question. It describes the distribution of local edge directions across a window rather than picking one keypoint. The classic pipeline divides the image window into small cells, accumulates orientation bins in each cell, normalises groups of adjacent cells into blocks, then concatenates the numbers. Local normalisation reduces sensitivity to some brightness and contrast changes; it does not make the descriptor fully invariant to illumination or geometry. The vector is a representation that a classifier or matching stage can use, not itself a pedestrian decision.

The classic layout gives **3,780** values for a traditional window layout: nine orientation bins, four cells per block and 105 overlapping blocks. The 105 count is not a general HoG constant. It follows from a 64 by 128 pixel window, 8 by 8 cells, 2 by 2 cells per block and a stride of one cell. Horizontally there are $64/8-1=7$ block positions; vertically $128/8-1=15$. Thus $7\times15=105$ and $105\times4\times9=3,780$. Change the window, cell, block or stride and the vector length changes.

<Infographic src="/img/cv/harris-hog.svg" alt="Harris distinguishes flat, edge and corner structure, with response 0.84 for eigenvalues one and one; HoG has 3780 values for a particular 105-block layout." caption="A corner detector supplies locations; a descriptor supplies values for comparing or classifying regions." />

## Worked example, step by step

Both halves use numbers small enough to follow with a pencil. The code in "Code you can run" prints exactly these values.

**Harris on a four-pixel window.** Each pixel has a gradient $(g_x, g_y)$: the change in brightness along x and along y. The tensor adds up $g_x^2$, $g_xg_y$ and $g_y^2$ over the window, and $k=0.04$.

1. Flat window, all four gradients $(0,0)$. Every sum is 0, so $M$ is all zeros, both eigenvalues are 0 and $R=0$.
2. Edge window, all four gradients $(2,0)$. The sum of $g_x^2$ is $4\times4=16$ and the other sums are 0. So $M=\begin{pmatrix}16&0\\0&0\end{pmatrix}$ with eigenvalues 16 and 0. The determinant is 0 and the trace is 16, so $R=0-0.04\times16^2=-10.24$.
3. Corner window, two gradients $(2,0)$ and two gradients $(0,2)$. Now $g_x^2$ sums to $2\times4=8$ and $g_y^2$ sums to 8, while the cross term is 0. So $M=\begin{pmatrix}8&0\\0&8\end{pmatrix}$, the determinant is 64 and the trace is 16. Then $R=64-0.04\times256=64-10.24=53.76$.

In words: an edge puts all its strength into one direction, so the determinant is zero and the penalty term makes $R$ negative. A corner spreads the same strength over two directions, so the determinant is large and $R$ is positive.

**HoG block normalisation.** Suppose a block's cell values are $(3,4)$.

1. Its length is $\sqrt{3^2+4^2}=5$.
2. Dividing by the length gives $(0.6,0.8)$.
3. Photograph the same scene at 30% of the contrast. The values become $(0.9,1.2)$ and the length becomes 1.5.
4. Dividing again gives $(0.9/1.5,\ 1.2/1.5)=(0.6,0.8)$.

In words: dividing by the block's own length cancels any overall brightness gain. This is why the HoG column "contrast x0.3" in the experiment below does not move at all.

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

Classical pedestrian detection pairs HoG with a classifier. That is a historical application pattern, not a claim that it is the best current pedestrian detector. Modern learned features may perform better under particular datasets and budgets, but the HoG construction remains a clear way to understand spatial pooling, orientation information and local normalisation. Its exact length can be checked without downloading a model or camera image.

## Code you can run

The first block verifies the two Harris examples. At equal eigenvalues of 1, the response is **0.84**; at eigenvalues 1 and 0, it is **−0.04** for $k=0.04$. The lab begins with the corner-like case and lets you change the eigenvalues and $k$.

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

**What each control does.**

- *First structure eigenvalue* and *Second structure eigenvalue* set the two eigenvalues of $M$, from 0 to 2.
- *Harris k* sets the penalty on the trace, from 0.02 to 0.10.
- The table underneath shows the determinant, trace and response, and a label for the patch.

**Try it yourself.**

1. Set the second eigenvalue to 0 and leave the first at 1. The response becomes $-0.04$, the edge value. One strong direction is not enough.
2. Put both eigenvalues back to 1 and raise $k$ from 0.04 to 0.10. The response falls from 0.84 to 0.60, because $k$ multiplies the squared trace. A larger $k$ makes the detector fussier about calling something a corner.
3. Set both eigenvalues to 0.5. The determinant is 0.25, the trace is 1, and the response is 0.21. The patch is still corner-like but weak, which is why a real detector compares responses across the image instead of using a fixed number.

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

Corners are often called “invariant” and HoG “illumination-invariant” without conditions. Harris response and HoG normalisation can tolerate some rotation or local contrast changes, but neither gives a blanket guarantee under scale, perspective, blur, saturation or arbitrary lighting. The 3,780 dimension belongs to the stated window geometry.

:::

### Experiment: how stable are corners, and does HoG beat raw pixels?

Two questions a practitioner asks before using either idea. How often does Harris find the same corners again when the image is rotated, noisy or rescaled? And is HoG a better input than raw pixels for a linear classifier? Both blocks use data that ships with the libraries, so nothing is downloaded. The camera image in scikit-image is released under CC0 by its photographer (scikit-image documentation). The 1,797 digits bundled with scikit-learn are the test split of the UCI optical-digits set, which the UCI page licenses under CC BY 4.0.

Before the real image, the next block reproduces the worked example above: the three Harris responses and the block normalisation.

```python
import numpy as np

patches = {'flat': [(0, 0)] * 4, 'edge': [(2, 0)] * 4, 'corner': [(2, 0), (2, 0), (0, 2), (0, 2)]}
for name, gradients in patches.items():
    gx, gy = np.array(gradients, dtype=float).T
    tensor = np.array([[gx @ gx, gx @ gy], [gx @ gy, gy @ gy]])
    values = np.linalg.eigvalsh(tensor)
    response = np.linalg.det(tensor) - 0.04 * np.trace(tensor) ** 2
    print(f'{name:7s} tensor {tensor.tolist()}  eigenvalues {values.round(2).tolist()}  response {response:.2f}')

cell = np.array([3.0, 4.0])
for gain in (1.0, 0.3):
    scaled = cell * gain
    print(f'contrast x{gain}: block {scaled.round(2).tolist()}  normalised {(scaled / np.linalg.norm(scaled)).round(2).tolist()}')
```

**Reading the output.** The tensor and eigenvalue columns match the hand arithmetic: zeros for the flat patch, 16 along one direction for the edge, 8 and 8 for the corner. The responses are 0.00, -10.24 and 53.76. The last two lines show a block $(3,4)$ and the same block at 30% contrast both normalising to $(0.6, 0.8)$.

**Line by line.**

- `np.linalg.eigvalsh` is the eigenvalue routine for symmetric matrices, which $M$ always is.
- `gx @ gx` is the dot product that adds up $g_x^2$ over the window.
- `np.linalg.norm(scaled)` is the block length that the division removes.

The next block measures repeatability on the real image. It finds the 400 strongest Harris corners inside a central disc, warps the image, maps the corner positions through the same transform, and counts how many have a detected corner within 2 pixels. Restricting to a disc keeps rotated image borders out of the comparison.

```python
import cv2
import numpy as np
import skimage
from scipy.spatial import cKDTree
from skimage import data
from skimage.feature import corner_harris, corner_peaks

image = data.camera().astype(np.float32)
centre = np.array([255.5, 255.5])
radius = 195


def corners(img, count):
    response = corner_harris(img, method='k', k=0.04, sigma=1.5)
    peaks = corner_peaks(response, min_distance=6, threshold_rel=0.0, num_peaks=10000)[:, ::-1].astype(float)
    peaks = peaks[np.linalg.norm(peaks - centre, axis=1) < radius]
    strength = response[peaks[:, 1].astype(int), peaks[:, 0].astype(int)]
    return peaks[np.argsort(-strength)][:count]


def repeatability(angle=0.0, scale=1.0, noise=0.0, seed=0, base=400, tolerance=2.0):
    matrix = cv2.getRotationMatrix2D((255.5, 255.5), angle, scale)
    warped = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_LINEAR)
    warped = warped + np.random.default_rng(seed).normal(0, noise, warped.shape).astype(np.float32)
    mapped = corners(image, base) @ matrix[:, :2].T + matrix[:, 2]
    mapped = mapped[np.linalg.norm(mapped - centre, axis=1) < radius]
    distance, _ = cKDTree(corners(warped, len(mapped))).query(mapped)
    return len(mapped), (distance <= tolerance).mean()


print('scikit-image', skimage.__version__, 'OpenCV', cv2.__version__)
print('rotation (degrees)  repeatability')
for angle in (0, 5, 15, 30, 45, 90):
    print(f'{angle:>10}          {repeatability(angle=angle)[1]:.3f}')
print('noise (grey levels, mean of 3 seeds)')
for sigma in (0, 5, 10, 20):
    mean = np.mean([repeatability(noise=sigma, seed=seed)[1] for seed in range(3)])
    print(f'{sigma:>10}          {mean:.3f}')
print('scale  corners compared  repeatability')
for scale in (1.0, 0.9, 0.7, 0.5, 1.25, 1.5, 2.0):
    count, value = repeatability(scale=scale)
    print(f'{scale:>5}  {count:>16}  {value:.3f}')
```

**Reading the output.** Each number is a share of the reference corners that were found again. The row for 0 degrees is 1.000 by construction, which checks the bookkeeping. The count column for the scale table says how many reference corners stay inside the disc after the zoom.

**Line by line.**

- `corner_peaks` keeps only local maxima of the response with `min_distance=6`, so one physical corner gives one point instead of a cluster.
- `matrix[:, :2].T` and `matrix[:, 2]` apply the 2 by 3 affine matrix to the stored positions, so the ground truth is exact and no human labelling is involved.
- `corners(warped, len(mapped))` asks the detector for as many points as there are reference corners to compare, so the two images are treated fairly.
- `cKDTree(...).query` finds the nearest detection to every mapped corner, and the 2 pixel tolerance turns that into a yes or no.

The last block trains a linear classifier on three feature sets and then tests it on images that were changed after training. The HoG vector length is checked first against the 3,780 count from the chapter.

```python
import numpy as np
import sklearn
import skimage
from scipy.ndimage import shift as shift_image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog

window = np.zeros((128, 64))
length = len(hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys'))
print('skimage', skimage.__version__, 'sklearn', sklearn.__version__)
print('HoG length for a 64 by 128 window:', length)

digits = load_digits()
images, labels = digits.images / 16.0, digits.target
train, test, y_train, y_test = train_test_split(images, labels, test_size=0.4, random_state=0, stratify=labels)
print('train', len(train), 'test', len(test), 'image size', images.shape[1:])


def hog_features(batch, cell):
    return np.array([hog(img, orientations=9, pixels_per_cell=(cell, cell), cells_per_block=(2, 2), block_norm='L2-Hys') for img in batch])


def perturb(batch, dx, noise, gain):
    out = np.array([shift_image(img, (0, dx), order=1, mode='constant') for img in batch]) * gain
    return np.clip(out + np.random.default_rng(1).normal(0, noise, out.shape), 0, 1)


extractors = {'raw pixels': lambda b: b.reshape(len(b), -1), 'HoG cell 2': lambda b: hog_features(b, 2), 'HoG cell 4': lambda b: hog_features(b, 4)}
cases = {'clean': (0, 0.0, 1.0), 'shift 1 px': (1, 0.0, 1.0), 'contrast x0.3': (0, 0.0, 0.3), 'noise 0.1': (0, 0.1, 1.0), 'noise 0.2': (0, 0.2, 1.0)}
print(f'{"features":12s} {"dims":>5s}  ' + '  '.join(f'{name:>13s}' for name in cases))
for name, extract in extractors.items():
    model = LinearSVC(C=1.0, max_iter=20000, random_state=0).fit(extract(train), y_train)
    scores = [model.score(extract(perturb(test, *settings)), y_test) for settings in cases.values()]
    print(f'{name:12s} {extract(train[:1]).shape[1]:>5d}  ' + '  '.join(f'{score:>13.3f}' for score in scores))
```

**Reading the output.** The first printed number is 3,780: an independent implementation, scikit-image 0.26.0, gives the same length as the hand count for a 64 by 128 window. The table has one row per feature set and one column per test condition. A column such as "contrast x0.3" multiplies test images by 0.3 after training. A column such as "noise 0.2" adds Gaussian noise of standard deviation 0.2 on a 0 to 1 scale.

**Line by line.**

- `block_norm='L2-Hys'` is the rescaling step from the worked example, with values clipped at 0.2 and renormalised.
- `pixels_per_cell=(cell, cell)` with `cells_per_block=(2, 2)` sets the geometry. On 8 by 8 digits a cell of 2 gives 324 values and a cell of 4 gives 36.
- `perturb` applies the shift, then the contrast gain, then the noise, and clips to the valid range.
- The classifier is fitted on clean training images only, so every other column measures robustness to a change it never saw.

#### Reading the experiment

Rotation is the mildest change. At 90 degrees the repeatability is 1.000, because a quarter turn only permutes pixels and nothing is interpolated. At 5 degrees it already falls to 0.795, and at 45 degrees to 0.745. The geometry is as rotation-stable as the formula claims, so the loss is the bilinear resampling blurring the image and shuffling weak corners. "Rotation invariant" is true of the formula and only roughly true of the pipeline.

Noise and scale are harsher. Gaussian noise with a standard deviation of 5 grey levels, about 2% of the range, cuts repeatability to 0.559 (mean of three seeds). Ten levels give 0.399 and twenty give 0.258. A zoom of 0.9 gives 0.665, a zoom of 0.5 gives 0.140, and a zoom of 2.0 gives 0.326. That collapse under scale is the fixed-window problem that the next chapter solves.

The digits give the surprising result. On clean images raw pixels score 0.971 and HoG with 2-pixel cells scores 0.950, so HoG does not win. It wins on exactly one change: at 30% contrast it stays at 0.950 while raw pixels fall to 0.690. Under noise the gradients hurt it: at 0.2 HoG drops to 0.734 against 0.908 for raw pixels. A 1-pixel shift hurts everything, with HoG at 0.592 and 0.599 against 0.446 for raw pixels.

Limits: one photograph, one detector setting, a 2 pixel tolerance and a top-400 cut-off; 8 by 8 digits are tiny compared with the 64 by 128 windows HoG was designed for; one train and test split; the classifier's `C` was not tuned. Trust the pattern, not the decimals.

<Infographic src="/img/cv-enrich/v2-harris-hog.svg" alt="Left: bars of Harris corner repeatability under rotation, noise and scale, from 1.000 at 90 degrees down to 0.140 at scale 0.5. Right: a table of digit accuracy for raw pixels and two HoG layouts under shift, contrast and noise." caption="Look first at the red bars: noise and zoom hurt Harris far more than rotation does. On the right, only the contrast column favours HoG." />

## Designing with it

When detecting points, define what makes a good repeat. If the application stitches images with modest rotation but large zoom, a fixed-scale Harris detector may miss correspondences even when it finds strong corners in each individual image. A scale-space detector in the next chapter addresses this more directly. If the camera and target scale are fixed, Harris may be simpler and adequate. Evaluate repeatability across real view pairs, not only the number of corners in one image.

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

## Common mistakes

1. **Calling Harris "rotation invariant" and stopping there.** The formula is, so it feels settled. In the experiment a 5 degree turn still lost a fifth of the corners (0.795) because resampling changed the image. Measure repeatability on the transforms your camera really produces.
2. **Counting corners instead of repeating them.** A detector that finds 2,000 points looks productive. Only points that come back in the second image help matching, and at a zoom of 0.5 just 0.140 did. Report repeatability, not totals.
3. **Using HoG on a noisy image without smoothing.** HoG is built from gradients, and gradients magnify noise. Noise of 0.2 cost HoG 0.216 of accuracy against 0.063 for raw pixels. Denoise first, or pick larger cells.
4. **Assuming HoG always beats raw pixels.** It does not on clean, aligned images: 0.950 against 0.971. It earns its place when contrast or lighting varies, so test the variation you expect.
5. **Matching two HoG layouts because the lengths agree.** Equal length is not equal meaning. Store the window, cell, block, stride and bin settings with the classifier and check them at serving time.

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

<details>
<summary><strong>Q6.</strong> Medium. A window has two gradients (3, 0) and two gradients (0, 3). Compute M and the Harris response with k = 0.04.</summary>

The sum of $g_x^2$ is $2\times9=18$ and the sum of $g_y^2$ is 18, with a zero cross term. So $M$ is diagonal with entries 18 and 18, the determinant is 324 and the trace is 36. Then $R=324-0.04\times36^2=324-51.84=272.16$, a strong corner.<br /><em>Medium · numeric</em>

</details>

<details>
<summary><strong>Q7.</strong> Medium. Corner repeatability was 1.000 at 90 degrees and 0.795 at 5 degrees. What explains the gap?</summary>

A 90 degree turn moves every pixel to an exact new pixel, so the image content is unchanged. A 5 degree turn needs interpolation, which blurs the image slightly and changes the weaker responses. The formula is rotation-stable, but the resampled image is not the same image.<br /><em>Medium · interpretation</em>

</details>

<details>
<summary><strong>Q8.</strong> Stretch. HoG stayed at 0.950 when the contrast was scaled to 0.3 while raw pixels fell from 0.971 to 0.690. Why, and what does this not protect against?</summary>

Block normalisation divides each block by its own length, so an overall gain cancels, as in the $(3,4)$ and $(0.9,1.2)$ example. It does not protect against noise, which changes gradient directions (HoG fell to 0.734 at noise 0.2), against shifts (0.592 at 1 pixel) or against a contrast change that is not a plain gain, such as saturation.<br /><em>Stretch · interpretation</em>

</details>

## Further reading

- [OpenCV Harris corner tutorial](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) for response and implementation.
- [OpenCV HOGDescriptor reference](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html) for the precise default geometry.
- Built from the course lecture "cv-s6-harris-hog" (Lecture Library series).

- [scikit-image `hog` and `corner_harris` documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html) (version 0.26.0, opened 2026-10-09) for the parameters used in the experiment, including `L2-Hys` normalisation.
- Harris and Stephens, "A combined corner and edge detector", Proceedings of the 4th Alvey Vision Conference, 1988, for the response function (bibliographic record checked 2026-10-09; paper text not re-read).
- Dalal and Triggs, "Histograms of oriented gradients for human detection", CVPR 2005, pages 886 to 893, for the HoG pipeline (bibliographic record checked 2026-10-09; paper text not re-read).
- [UCI optical recognition of handwritten digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) (opened 2026-10-09) for the digits licence, CC BY 4.0.
- Library versions run for the experiment: OpenCV 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1.

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
- I can reproduce the 0.84 Harris response from eigenvalues 1 and 1.
- I can derive 105 HoG blocks and 3,780 values from explicit window and stride settings.
- I can explain why a detector location and a descriptor vector are different outputs.
- I can compute a structure tensor and a Harris response from four gradient vectors by hand.
- I can say which changes (rotation, noise, scale) cost corner repeatability most, and why a 5 degree turn is not free.
- I can explain why HoG survives a contrast change but not noise, and why it does not automatically beat raw pixels.
