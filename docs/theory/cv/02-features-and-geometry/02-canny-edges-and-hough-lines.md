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

## The idea in plain words

:::note Beyond the lecture

The accumulator experiment, deployment checks and source cautions extend the lecture. The five Canny stages, polar-line worked example and source questions are retained.

:::

A gradient magnitude image contains too much information for a simple line decision. Blur spreads a true boundary over several pixels, texture creates many local peaks, and sensor noise creates isolated changes. The Canny process organises these problems into stages. First it smooths to reduce noise. Then it estimates gradient magnitude and direction. Non-maximum suppression keeps local maxima along the gradient normal, thinning a ridge. Double thresholding marks strong and possible weak edges. Hysteresis retains weak pixels only when they connect to a strong edge under the chosen connectivity rule. The output is a candidate edge map, not a semantic explanation of the image.

The lecture presents this as five stages: Gaussian smoothing, gradient estimation, non-maximum suppression, double thresholding and hysteresis. The order matters. Suppressing local maxima before computing orientation would be undefined; linking weak edges before defining strong seeds would have no anchor. Thresholds are measured in the gradient scale used by the implementation. OpenCV's `Canny` has an `L2gradient` option: one setting uses Euclidean magnitude, while the default uses a sum of absolute components. Parameters copied between implementations or input scales can therefore behave differently even when they have the same numeric values.

Canny answers “where are likely intensity boundaries?” The Hough line transform asks a different question: “which lines receive support from those candidate points?” A line in image coordinates can be written $\rho=x\cos\theta+y\sin\theta$. Here $\theta$ is the angle of the line's normal, and $\rho$ is its signed perpendicular distance from the origin under a chosen convention. Each edge point can lie on many possible lines. As $\theta$ varies over candidate angle bins, the point votes in the corresponding $(\rho,\theta)$ accumulator cells. Points on the same line tend to add votes to a common cell. A peak is a candidate line whose strength depends on edge support, angular and distance resolution and vote threshold.

The lecture's point $(2,2)$ at a normal angle of 45° gives $\rho=2\cos45°+2\sin45°=2\sqrt{2}=2.828$ to three decimals. That is **one vote**, not proof of a line. Another point on the same line, such as $(0,4)$, gives the same $\rho$ at 45°. A line emerges only when enough points support the same parameters within the accumulator's finite bins. The polar form avoids the infinite slope of a vertical line in $y=mx+c$. It does not eliminate discretisation, duplicate peaks or the need to distinguish a finite segment from an infinite mathematical line.

<Infographic src="/img/cv/canny-hough.svg" alt="Canny smooths, finds and thins gradients then links weak edges; Hough turns edge points into polar line votes, including rho 2.828 for point two two at 45 degrees." caption="A strong edge pixel and a line with collective support are different levels of evidence." />

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

The first block reproduces the lecture's **2.828** vote with trigonometric functions. Use degrees only for the readable input; Python's `sin` and `cos` require radians.

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

## Designing with it

Set edge thresholds against a measured image range and expected noise. Low thresholds preserve faint boundaries but increase false connections; high thresholds produce clean maps with missing pieces. Double thresholding and hysteresis separate a strong seed from a weaker continuation, but they still depend on connectivity and scale. Test on examples with thin targets, bright clutter, shadows and blur. Use a failure matrix rather than tuning by visual preference on one image.

Choose Hough bins to match the required precision. Coarse angle or distance bins combine different lines; very fine bins split support from one noisy line into several cells. A vote threshold depends on how many edge pixels a true line can contribute, which changes with image size and crop. Line-length and gap settings in a segment detector should be chosen in physical or task terms where possible. A long line across a 4K frame and a short line in a 64-pixel crop should not necessarily share one fixed pixel threshold.

Avoid interpreting the normal angle as the direction along the line. A horizontal image line has a vertical normal, so its Hough angle may be near 90° in the convention used here. A code review that assumes the angle is the line direction can silently filter the wrong candidates. Write a unit test using synthetic horizontal and vertical lines and record the coordinate origin, angle range and sign convention. The source tutorial notes that orientation conventions can vary.

Check duplicate and broken detections. A thick painted mark has two edges. Parallel seams can share votes with desired lines. A curved boundary may produce many short tangent lines. Connected component or region checks can reduce duplicates, while a geometric fit can refine a line after voting. If the downstream decision needs a physical distance, calibrate the camera and map pixels into world coordinates; a line in image space alone does not provide that measurement.

In production, monitor input quality and the distribution of detected candidates. An exposure change may alter edge counts and hence Hough peaks without any scene geometry changing. Keep examples of false peaks and missed lines, and compare candidate recall and final-decision error separately. Canny and Hough are composable measurements; the useful system is the whole pipeline with its validation and fallback policy.

## Where this stands in 2026

:::info Industry view

- Canny and Hough remain useful for explicit geometric constraints, debugging and controlled scenes. Learned vision models can supply candidates, but a known straight-line constraint can still simplify validation.
- The source equations are stable. OpenCV tutorials checked on 2026-10-02 displayed 4.13.0; the runnable blocks use only Python standard-library maths and need no image download.
- The toy accumulator demonstrates three collinear votes. It does not claim production recall, precision or a line-length guarantee.

:::

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

## Further reading

- [OpenCV Canny tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) for the staged edge detector and implementation options.
- [OpenCV Hough line tutorial](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html) for polar voting and line versus segment outputs.
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
