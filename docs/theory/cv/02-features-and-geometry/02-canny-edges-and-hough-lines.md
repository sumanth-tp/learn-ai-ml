---
id: cv-canny-edges-and-hough-lines
title: "Computer Vision · Session 4; Canny and Hough Lines"
sidebar_label: "2 · Canny and Hough"
sidebar_position: 2
slug: /theory/cv/canny-edges-and-hough-lines
description: "Trace Canny’s smoothing, thinning and hysteresis, then see how edge points vote for polar line parameters."
tags: [computer-vision, canny, hough, lines]
---

import Infographic from '@site/src/components/Infographic';
import HoughVoteLab from '@site/src/components/viz/HoughVoteLab';


**In one line.** Canny makes a thin, connected edge candidate map; Hough asks which parameterised lines those points support together.

:::tip Before you start

**You should already know:**

- What an image gradient is and how Sobel measures it. See [gradients and edges](/docs/theory/cv/image-gradients-and-edges).
- What precision, recall and F1 mean, as used in that chapter.

**Reading time:** about 35 minutes.

**After this chapter you can:**

- name Canny's five stages and say which threshold decides which;
- turn an edge point into a Hough vote by hand;
- choose Hough bin sizes and a vote threshold that belong together.

:::

## In 30 seconds

Canny turns a blurry picture into thin, connected outlines. Hough then asks a different question: which straight lines pass through many of those outline points? Each point votes for every line that could pass through it, like a crowd where each person names every road they could be standing on. The road named by most people is probably real. How finely you tally the roads decides whether the votes pile up or scatter.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Hysteresis | Keep a weak edge only if it connects to a strong one | High threshold 160, low threshold 64 |
| Non-maximum suppression | Keep only the strongest pixel across an edge, so it is one pixel wide | A 5-pixel-wide band becomes 1 pixel |
| Accumulator | A table that counts votes for each line | Rows are rho, columns are theta |
| Rho (ρ) | Distance from the origin to a line | 2.828 for the point (2, 2) at 45 degrees |
| Theta (θ) | Angle of the line's normal | 45 degrees |
| Bin | One cell of the accumulator | Width 1 pixel and 1 degree |
| Vote threshold | Minimum count for a cell to be called a line | 60 votes |
| Spurious line | A detected line that matches no true line | A diagonal across a corner |


## The idea in plain words

:::note Additions to the course material

The accumulator experiment, deployment checks and source cautions extend the course material. The five Canny stages, polar-line worked example and questions are retained.

:::

A gradient magnitude image contains too much information for a simple line decision. Blur spreads a true boundary over several pixels, texture creates many local peaks, and sensor noise creates isolated changes. The Canny process organises these problems into stages. First it smooths to reduce noise. Then it estimates gradient magnitude and direction. Non-maximum suppression keeps local maxima along the gradient normal, thinning a ridge. Double thresholding marks strong and possible weak edges. Hysteresis retains weak pixels only when they connect to a strong edge under the chosen connectivity rule. The output is a candidate edge map, not a semantic explanation of the image.

Canny has five stages: Gaussian smoothing, gradient estimation, non-maximum suppression, double thresholding and hysteresis. The order matters. Suppressing local maxima before computing orientation would be undefined; linking weak edges before defining strong seeds would have no anchor. Thresholds are measured in the gradient scale used by the implementation. OpenCV's `Canny` has an `L2gradient` option: one setting uses Euclidean magnitude, while the default uses a sum of absolute components. Parameters copied between implementations or input scales can therefore behave differently even when they have the same numeric values.

Canny answers “where are likely intensity boundaries?” The Hough line transform asks a different question: “which lines receive support from those candidate points?” A line in image coordinates can be written $\rho=x\cos\theta+y\sin\theta$. Here $\theta$ is the angle of the line's normal, and $\rho$ is its signed perpendicular distance from the origin under a chosen convention. Each edge point can lie on many possible lines. As $\theta$ varies over candidate angle bins, the point votes in the corresponding $(\rho,\theta)$ accumulator cells. Points on the same line tend to add votes to a common cell. A peak is a candidate line whose strength depends on edge support, angular and distance resolution and vote threshold.

The point $(2,2)$ at a normal angle of 45° gives $\rho=2\cos45°+2\sin45°=2\sqrt{2}=2.828$ to three decimals. That is **one vote**, not proof of a line. Another point on the same line, such as $(0,4)$, gives the same $\rho$ at 45°. A line emerges only when enough points support the same parameters within the accumulator's finite bins. The polar form avoids the infinite slope of a vertical line in $y=mx+c$. It does not eliminate discretisation, duplicate peaks or the need to distinguish a finite segment from an infinite mathematical line.

<Infographic src="/img/cv/canny-hough.svg" alt="Canny smooths, finds and thins gradients then links weak edges; Hough turns edge points into polar line votes, including rho 2.828 for point two two at 45 degrees." caption="A strong edge pixel and a line with collective support are different levels of evidence." />

## Worked example, step by step

**Votes split by bin size.** The scene in the experiment is a four-sided shape whose sides are 151.3, 131.5, 153.0 and 141.4 pixels long. After Canny, each side is a one-pixel-wide chain of edge points.

1. In an ideal accumulator every point of one side votes for the same cell, so the cell holds about the side length: 131 to 153 votes.
2. Real pixels sit on a grid, so the points of a straight side wobble by up to about half a pixel. With wide bins the wobble stays inside one bin. With very fine bins the points spread over several neighbouring cells.
3. Suppose a side spreads evenly over 4 cells (rho and theta bins each a quarter of the usual size). A side of 131.5 pixels then gives 131.5 / 4 = 32.9 votes per cell, and the best of the other sides gives about 38.
4. A vote threshold of 60 is 46% of the shortest side. No cell reaches it, so every line is lost, though all four lines are in the image.
5. The code below prints these numbers. The experiment then measures the real peaks.

## How it works

### The five stages of Canny

- **Smooth + gradient**; Gaussian blur, then Sobel magnitude/orientation.
- **Non-max suppression**; Thin ridges to 1-pixel edges along the gradient.
- **Double threshold + hysteresis**; Strong/weak/none; keep weak only if linked to strong.

### The Hough transform

A line is ρ = x cos θ + y sin θ. Each edge point votes for all lines through it; collinear points' votes pile up in one (ρ,θ) cell; a peak.

:::tip

**Worked.** (x,y)=(2,2), θ=45° → ρ = 2cos45 + 2sin45 = 2.828. The (2.828, 45°) cell peaks → line detected.

:::

:::note

**Why polar?** y=mx+c can't represent vertical lines (m→∞); ρ,θ is bounded and handles every orientation, and extends to circles.

:::

### Key takeaways

- **1 · Canny**; Smooth→grad→NMS→double thresh→hysteresis.
- **2 · Hough**; ρ=x cosθ+y sinθ; vote → peaks = lines.
- **3 · Polar**; Handles vertical lines & all angles.

## A real system that works this way

OpenCV's current Canny tutorial documents the staged process and its `cv.Canny` call. Its Hough tutorial defines the polar line parameters and two functions: `HoughLines`, which returns line parameters, and `HoughLinesP`, which returns finite segments. A pipeline can use Canny's binary output as the input to a Hough vote, then apply further checks to candidate lines. The tutorials were opened on 2026-10-02 and displayed OpenCV 4.13.0.

Consider detecting long lane-like markings in a controlled overhead view. Canny may find both sides of a painted stripe, shadow edges and texture. Hough can pool support across broken portions of a roughly straight line, but it may also return two parallel lines for the stripe's two sides or vote for a seam. Cropping to a valid region, restricting orientations and checking line length and position are design choices tied to the application. The output still needs evaluation against annotated target lines or downstream control behaviour; an accumulator peak alone is not a safe driving decision.

The same idea appears in document processing. A scanned page with a tilted table can have many letter edges, but long ruled table lines produce coherent votes. A Hough candidate can estimate page skew or table geometry after text and border distractions are handled. A low-resolution scan or a page with curved boundaries violates the simple straight-line assumption. A probabilistic Hough segment or a different geometric model may fit better. The method is valuable because its assumptions are explicit and testable, not because it detects every line in every scene.

## Code you can run

The first block reproduces the worked **2.828** vote with trigonometric functions. Use degrees only for the readable input; Python's `sin` and `cos` require radians.

```python
from math import cos, radians, sin

x, y = 2, 2
angle_degrees = 45
angle_radians = radians(angle_degrees)
rho = x * cos(angle_radians) + y * sin(angle_radians)
print(f'Point: ({x}, {y}); angle: {angle_degrees} degrees')
print(f'Rho vote: {rho:.3f}')
assert round(rho, 3) == 2.828
```

Move the point or normal angle in the lab. Its default matches the printed vote. The drawn normal illustrates the selected angle; it is not a complete Hough accumulator.

<HoughVoteLab />

The second block demonstrates accumulation for three points on $x+y=2$. All three vote for approximately **1.414** at 45°, while their 0° votes are split among three distance bins. This is a deliberately small angle slice of Hough voting, not an OpenCV detector or a benchmark.

```python
from collections import Counter
from math import cos, radians, sin

points = [(0, 2), (1, 1), (2, 0)]

def votes_at(angle_degrees):
    angle = radians(angle_degrees)
    return Counter(round(x * cos(angle) + y * sin(angle), 3) for x, y in points)

diagonal_votes = votes_at(45)
vertical_normal_votes = votes_at(0)
print('45-degree normal votes:', sorted(diagonal_votes.items()))
print('0-degree normal votes:', sorted(vertical_normal_votes.items()))
assert diagonal_votes == {1.414: 3}
assert vertical_normal_votes == {0.0: 1, 1.0: 1, 2.0: 1}
```

In a full implementation, the input would be edge pixels and votes would be distributed across many angle bins. Rounding to three decimals is enough to show this construction; the result can change with bin width or floating-point policy. Detecting a useful finite segment also requires endpoints or a supporting-pixel trace, which this toy vote does not supply.

### The worked numbers in code

```python
import cv2
import numpy as np

polygon = np.array([[40, 60], [190, 40], [210, 170], [60, 200]])
lengths = np.hypot(*(np.roll(polygon, -1, axis=0) - polygon).T)
print("side lengths in pixels:", lengths.round(1).tolist())
print("votes per bin if one side splits evenly over 4 bins:", (lengths / 4).round(1).tolist())
print("shortest side:", round(float(lengths.min()), 1), " a vote threshold of 60 is", round(60 / float(lengths.min()), 2), "of it")
line = np.zeros((240, 240), np.uint8)
cv2.line(line, (40, 60), (190, 40), 255, 1)
print("pixels drawn for the first side:", int(np.count_nonzero(line)))
for step in (1, 0.25):
    found = cv2.HoughLinesWithAccumulator(line, step, np.radians(step), 5)
    print(f"bins {step} px and {step} degrees: best cell holds", int(max(np.ravel(g)[2] for g in found)), "votes")
```

**Reading the output.** The sides are 151.3, 131.5, 153.0 and 141.4 pixels. An even four-way split gives 37.8, 32.9, 38.2 and 35.4 votes, and the shortest side makes a threshold of 60 equal to 0.46 of its length. A single drawn side of 151 pixels puts 81 votes in its best cell with 1 pixel and 1 degree bins, but only 39 with quarter-size bins. The vote count is a property of the bin size.

### Experiment: Canny thresholds and Hough bin sizes against known lines

A four-sided shape on a darker background is drawn at 240 by 240, blurred slightly and given Gaussian noise. Its sides are four known lines. Part one sweeps Canny's high threshold (20 to 320) and the low-to-high ratio (0.1 to 1.0) at noise sd 25 and scores F1 against the true boundary with 1 pixel of tolerance. Part two runs the Hough transform on a fixed Canny map with six bin sizes, a vote threshold of 60, and two noise levels. A true line counts as found if a detected line passes within 4 pixels (or, in the looser count, 12 pixels) of three points on that side. A detection that is within 12 pixels of no true side is called spurious.

```python
import cv2
import numpy as np
from scipy import ndimage

def shapes(noise, seed):
    mask = np.zeros((240, 240), np.uint8)
    polygon = np.array([[40, 60], [190, 40], [210, 170], [60, 200]])
    cv2.fillPoly(mask, [polygon], 1)
    truth = (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))).astype(bool)
    image = cv2.GaussianBlur(np.where(mask > 0, 190.0, 60.0), (0, 0), 1.0)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)
    sides = [(a, (a + b) / 2, b) for a, b in zip(polygon, np.roll(polygon, -1, 0))]
    return np.clip(image, 0, 255).astype(np.uint8), truth, sides

def canny_f1(edges, truth, tolerance=1):
    near_truth = ndimage.distance_transform_edt(~truth) <= tolerance
    near_found = ndimage.distance_transform_edt(edges == 0) <= tolerance
    precision = ((edges > 0) & near_truth).sum() / max((edges > 0).sum(), 1)
    recall = (truth & near_found).sum() / truth.sum()
    return 2 * precision * recall / (precision + recall + 1e-9)

def miss(line, side):
    return max(abs(p[0] * np.cos(line[1]) + p[1] * np.sin(line[1]) - line[0]) for p in side)

image, truth, sides = shapes(25, 0)
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
print("Canny F1 at noise sd 25: rows high threshold, columns low/high ratio 0.1 0.4 0.8 1.0")
for high in (20, 40, 80, 160, 320):
    print(f"  high={high:3d}", "".join(f"{canny_f1(cv2.Canny(blurred, high * r, high), truth):8.3f}" for r in (0.1, 0.4, 0.8, 1.0)))

print("Hough at vote threshold 60: sides found within 4 px / within 12 px, spurious lines, weakest side votes")
for noise in (0, 25):
    image, truth, sides = shapes(noise, 1)
    edges = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 1.4), 100, 200)
    for rho_step, theta_step in ((1, 1), (0.25, 0.25), (0.5, 0.5), (2, 2), (4, 5), (8, 10)):
        found = [np.ravel(g) for g in cv2.HoughLinesWithAccumulator(edges, rho_step, np.radians(theta_step), 20)]
        strong = [f for f in found if f[2] >= 60]
        tight = sum(any(miss(f, s) <= 4 for f in strong) for s in sides)
        loose = sum(any(miss(f, s) <= 12 for f in strong) for s in sides)
        spurious = sum(not any(miss(f, s) <= 12 for s in sides) for f in strong)
        weakest = min(max([f[2] for f in found if miss(f, s) <= 12] or [0]) for s in sides)
        print(f"  noise {noise:2d}  rho {rho_step:<4} theta {theta_step:<4}: {tight}/4  {loose}/4  spurious {spurious:2d}  weakest {int(weakest):3d}")
```

**Reading the output.** The Canny sweep at noise sd 25 printed (rows are the high threshold, columns the ratio):

| high \ ratio | 0.1 | 0.4 | 0.8 | 1.0 |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.060 | 0.060 | 0.061 | 0.063 |
| 40 | 0.062 | 0.063 | 0.082 | 0.102 |
| 80 | 0.084 | 0.173 | 0.513 | 0.712 |
| 160 | 0.997 | 0.997 | 0.997 | 0.997 |
| 320 | 0.997 | 0.997 | 0.472 | 0.143 |

The Hough part printed, for noise 0 (within 4 px, within 12 px, spurious, weakest side's votes):

| Bins (rho, theta) | within 4 px | within 12 px | spurious | weakest votes |
| --- | ---: | ---: | ---: | ---: |
| 0.25, 0.25 | 0/4 | 0/4 | 0 | 36 |
| 0.5, 0.5 | 4/4 | 4/4 | 0 | 63 |
| 1, 1 | 4/4 | 4/4 | 0 | 74 |
| 2, 2 | 4/4 | 4/4 | 0 | 78 |
| 4, 5 | 4/4 | 4/4 | 0 | 79 |
| 8, 10 | 0/4 | 4/4 | 2 | 102 |

At noise 25 the same six rows gave:

| Bins (rho, theta) | within 4 px | within 12 px | spurious | weakest votes |
| --- | ---: | ---: | ---: | ---: |
| 0.25, 0.25 | 0/4 | 0/4 | 0 | 30 |
| 0.5, 0.5 | 3/4 | 3/4 | 0 | 54 |
| 1, 1 | 4/4 | 4/4 | 0 | 72 |
| 2, 2 | 4/4 | 4/4 | 0 | 78 |
| 4, 5 | 3/4 | 4/4 | 0 | 84 |
| 8, 10 | 0/4 | 4/4 | 2 | 101 |

**Line by line.**

- `cv2.Canny(blurred, high * r, high)` passes the low threshold first, then the high. Both are on the gradient scale that OpenCV uses inside Canny, so thresholds do not transfer from other libraries.
- `miss` measures how far a detected line `(rho, theta)` is from a point, using the line equation, so no angle wrapping is needed.
- `cv2.HoughLinesWithAccumulator` returns the vote count with each line, and a threshold of 20 keeps weak cells so that the weakest peak of each side can be reported.
- `strong` applies the vote threshold of 60 afterwards.

**What the numbers say.** Canny has a window with cliffs at both edges. At high threshold 160, every ratio gives F1 0.997. Halving to 80 drops it to between 0.084 and 0.712. Doubling to 320 with a ratio of 0.8 or 1.0 drops it to 0.472 and 0.143, because the true gradient no longer reaches the high threshold and there are no seeds to grow. At a high threshold of 80 a higher ratio helps. A low threshold of 8 (ratio 0.1) probably lets noise connect to seeds and flood the map, which is consistent with the F1 of 0.084, though the run did not inspect the maps.

For Hough, the surprise is that finer bins are not more accurate. At 0.25 for both, no line is found even without noise, since the best cell for the weakest side holds 36 votes, close to the 32.9 to 38.2 predicted by an even split and to the 39 votes of a single drawn side. At the coarsest setting the peaks are strongest (102) and all four lines are found within 12 pixels, but none within 4, and two false lines appear. Bins of 1 and 1 and of 2 and 2 found all four lines within 4 pixels at both noise levels. The vote threshold is a property of the bin size as much as of the image.

Limits: one four-sided shape, noise sd 0 and 25 only, one noise seed, Canny thresholds for the Hough input fixed at 100 and 200, a single vote threshold of 60. The Canny window moves with the noise level and the contrast, which was not swept. The cause of the two false lines at the coarsest bins was not examined directly.

<Infographic src="/img/cv-enrich/v1-canny-hough-sweeps.svg" alt="Left: a grid of Canny F1 for five high thresholds and four ratios, perfect at 160 and failing at 20 to 80. Right: a table of Hough bin sizes against lines found and weakest votes." caption="Look at the 160 row on the left, then the 0.25 row on the right: one parameter has a wide safe window, the other a narrow one." />

## Designing with it

Set edge thresholds against a measured image range and expected noise. Low thresholds preserve faint boundaries but increase false connections; high thresholds produce clean maps with missing pieces. Double thresholding and hysteresis separate a strong seed from a weaker continuation, but they still depend on connectivity and scale. Test on examples with thin targets, bright clutter, shadows and blur. Use a failure matrix rather than tuning by visual preference on one image.

Choose Hough bins to match the required precision. Coarse angle or distance bins combine different lines; very fine bins split support from one noisy line into several cells. A vote threshold depends on how many edge pixels a true line can contribute, which changes with image size and crop. Line-length and gap settings in a segment detector should be chosen in physical or task terms where possible. A long line across a 4K frame and a short line in a 64-pixel crop should not necessarily share one fixed pixel threshold.

Avoid interpreting the normal angle as the direction along the line. A horizontal image line has a vertical normal, so its Hough angle may be near 90° in the convention used here. A code review that assumes the angle is the line direction can silently filter the wrong candidates. Write a unit test using synthetic horizontal and vertical lines and record the coordinate origin, angle range and sign convention. The OpenCV tutorial notes that orientation conventions can vary.

Check duplicate and broken detections. A thick painted mark has two edges. Parallel seams can share votes with desired lines. A curved boundary may produce many short tangent lines. Connected component or region checks can reduce duplicates, while a geometric fit can refine a line after voting. If the downstream decision needs a physical distance, calibrate the camera and map pixels into world coordinates; a line in image space alone does not provide that measurement.

In production, monitor input quality and the distribution of detected candidates. An exposure change may alter edge counts and hence Hough peaks without any scene geometry changing. Keep examples of false peaks and missed lines, and compare candidate recall and final-decision error separately. Canny and Hough are composable measurements; the useful system is the whole pipeline with its validation and fallback policy.

## Where this stands in 2026

:::info Industry view

- Canny and Hough remain useful for explicit geometric constraints, debugging and controlled scenes. Learned vision models can supply candidates, but a known straight-line constraint can still simplify validation.
- The source equations are stable. OpenCV tutorials checked on 2026-10-02 displayed 4.13.0; the first two blocks use only Python standard-library maths, and the experiment draws its own images with OpenCV 5.0.0, so no image download is needed.
- The toy accumulator demonstrates three collinear votes. It does not claim production recall, precision or a line-length guarantee.

:::

## Common mistakes

1. **Tuning Canny thresholds by eye on one image.** The F1 fell from 0.997 to between 0.084 and 0.712 when the high threshold halved. Pick thresholds from the gradient scale of your noise level, and re-check when the camera or the lighting changes.
2. **Using the finest bins in the hope of more accuracy.** They split votes: at 0.25 the weakest side held 36 votes and no line passed the threshold. Keep bins as fine as the edge scatter allows and set the vote threshold below the votes of the shortest line you need.
3. **Trusting the strongest peak at coarse bins.** The 8 and 10 setting had the highest peaks (102) and the worst precision: no line within 4 pixels and two false lines. Check accuracy as well as strength.
4. **Treating the angle as the line direction.** It is the direction of the normal. A horizontal line has a vertical normal.
5. **Reading a vote peak as an object.** It shows only that many edge points agree with a straight line.

## Practice questions

<details>
<summary><strong>Q1.</strong> List the five stages of the Canny edge detector.</summary>

Gaussian smoothing → gradient (Sobel) → non-maximum suppression → double thresholding → hysteresis edge tracking.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What does non-maximum suppression do in Canny?</summary>

It thins edges to one pixel wide by keeping only local maxima of gradient magnitude along the gradient direction.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> What is hysteresis thresholding?</summary>

Two thresholds split pixels into strong/weak/non-edges; weak edges are kept only if connected to a strong edge, giving connected contours without noise specks.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> An edge point (2,2) votes at θ=45°. Compute the Hough ρ.</summary>

ρ = x cosθ + y sinθ = 2cos45° + 2sin45° = 2(0.7071)+2(0.7071) = 2.828.<br /><em>Session 4 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why does Hough use ρ,θ instead of y=mx+c?</summary>

Slope-intercept can't represent vertical lines (m→∞); the polar form ρ,θ is bounded (θ∈[0,π)) and handles all orientations, and extends to circles/parametric shapes.<br /><em>Session 4 · conceptual</em>

</details>

<details>
<summary><strong>Q6 (Easy).</strong> What is the Hough rho for the point (3, 4) at a normal angle of 90 degrees?</summary>

ρ = 3 cos 90° + 4 sin 90° = 0 + 4 = 4.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> A straight side of 120 edge pixels spreads its votes evenly over 3 accumulator cells. A colleague sets the vote threshold to 60. What happens, and what are two fixes?</summary>

Each cell gets 120 / 3 = 40 votes, below 60, so the line is lost. Either lower the threshold to below 40 (and accept more spurious lines) or use coarser bins so the votes fall into fewer cells. The experiment shows the same mechanism: 36 votes at the finest bins against a threshold of 60.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> At the coarsest bins every true line was found within 12 pixels but none within 4, and two false lines appeared. Give a likely reason, and say what the run does not show.</summary>

A cell 8 pixels wide and 10 degrees wide gathers points that are consistent with a whole family of nearby lines, so the peak is strong but its position is imprecise, and edge points of different sides can pool into extra peaks. The run measured the positions and the counts. It did not inspect where the two false lines lie, so the pooling explanation is unconfirmed.

</details>

## Further reading

- [OpenCV Canny tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) for the staged edge detector and implementation options.
- [OpenCV Hough line tutorial](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html) for polar voting and line versus segment outputs.
- [OpenCV Canny tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) and [OpenCV Hough line tutorial](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html), listed in the earlier version of this chapter. Both returned HTTP 403 from the build environment on 2026-10-09, so the threshold and bin behaviour above comes from the runs, not from the manual.
- Versions run: OpenCV 5.0.0, SciPy 1.18.1, NumPy 2.5.3.
- The scene is generated in the code, so there is no image source or licence to state.
- Built from the course lecture "cv-s4-canny-hough" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Reading an accumulator without overclaiming

An accumulator peak reflects agreement under the chosen model and discretisation. It is not a probability that a semantic object is present. A repeated texture can create many votes, and a strong line can be irrelevant to the task. The vote count also depends on image resolution: upscaling an image can create more edge pixels without adding independent evidence. Interpret peaks relative to expected support and apply geometric and contextual checks before a product action.

The three-point calculation in the code shows how a shared parameter emerges. At 45°, all points lie on $x+y=2$ and the projected normal distance is the same. At 0°, their x positions differ and the votes separate. Real edge points are not exact integers on one line; lens distortion, blur and pixel selection perturb them. A finite rho bin allows nearby values to accumulate, but a bin that is too wide also merges nearby physical lines. There is no resolution-free threshold.

When a Hough result looks wrong, inspect the preceding Canny map before changing the vote threshold. If the desired line has no edge pixels, voting cannot recover it. If the edge map is dominated by text or texture, a crop, scale change or different preprocessing may help more than a lower peak threshold. If the desired edge is present but fragmented, inspect non-maximum suppression, hysteresis connectivity and gap tolerance. A stage-by-stage trace turns an apparently mysterious final line into a diagnosable sequence.

Also consider whether a line is the right model. A road edge under perspective may be curved in the image, a bent part may have multiple segments, and an object boundary may not be geometric at all. Hough can be extended to other parameterised shapes, but parameter-space size grows with model complexity. For an irregular contour, segmentation or local tracking may provide a more faithful output. The model should reflect the task's useful structure rather than the availability of a familiar transform.

An accumulator is tied to an origin and bin convention. If an image is cropped, the same physical line has different image coordinates and may acquire a different rho unless the crop offset is accounted for. If angle bins are coarse, a long shallow line may split votes across neighbouring cells; if rho bins are coarse, two nearby parallel lines may merge. A reproducible detector therefore records the crop, coordinate origin, angle interval, rho interval and peak-selection policy together. When comparing two Hough implementations, test a synthetic horizontal and vertical line with known endpoints before comparing full photographs. This reveals sign and angle-convention mismatches that a realistic image can hide. Then evaluate whether the finite segment overlaps the actual target rather than relying on a peak count alone.

When a detected line controls a physical action, convert its image coordinates through the calibrated camera model and test the full action timing. The vote is an intermediate geometric estimate, not the complete control signal.

## Check yourself

- I can name Canny’s five stages and explain what each changes.
- I can calculate one polar Hough vote from a point and a normal angle.
- I can explain why a vote peak needs support and a downstream validation rule.
- I can distinguish a detected infinite line from a finite segment useful to an application.
- I can explain why a Canny threshold has cliffs at both ends of its safe window.
- I can predict that a vote threshold fails when the bins are fine enough to split a line's votes below it.
- I can choose bin sizes and a vote threshold together, and check line accuracy as well as vote strength.
