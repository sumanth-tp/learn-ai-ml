---
id: cv-midsem-solved
title: "Computer Vision; 2026 Mid-Semester Worked Solutions"
sidebar_label: "2 · Mid-semester solved"
sidebar_position: 2
slug: /theory/cv/midsem-solved
description: "Worked methods for the available Q4–Q6 transcript, with original diagrams and clearly labelled synthetic numbers where scan values are missing."
tags: [computer-vision, practice, midsem]
---

import Infographic from '@site/src/components/Infographic';
import EdgeGradientLab from '@site/src/components/viz/EdgeGradientLab';
import HoughVoteLab from '@site/src/components/viz/HoughVoteLab';
import HogBinLab from '@site/src/components/viz/HogBinLab';

**In one line.** The available transcript contains Q4 to Q6 on edge operators, Hough voting, HoG and Lucas–Kanade flow; solve their methods without pretending the unavailable scan values are known.

## What the source provides

The converted source has three answered panels labelled Q4, Q5 and Q6. It refers to three paper-scan images, but those image files are not present in the handover. The Q4 transcript gives a 5 by 5 image with a 150/50 step but not every pixel or the exact location requested. Q5 does not transcribe the two point coordinates or the “4-cell” accumulator. Q6 does not transcribe the gradients in its HoG cell. Therefore the source supports the algorithms and answer structure, but an exact numeric answer for the scanned diagrams cannot be recovered from it. The board and runnable grids below are **original synthetic teaching examples**, not claims about the scan.

:::note Missing diagram values

The source says the actual 2026 paper contains diagrams and calls its text a low-resolution transcription. The three scan images are unavailable in the provided files. The 5 by 5 grid, Hough points and HoG gradients below are chosen illustrations. Do not quote their computed numbers as the examination paper's answers without checking the original scan.

:::

<Infographic src="/img/cv/midsem-methods.svg" alt="Original board for the available mid-semester Q4 to Q6: a synthetic 150/50 edge, a synthetic y-equals-x Hough line, and a synthetic dominant HoG bin; it says the scan values are unavailable." caption="The board redraws the reasoning with labelled illustrations rather than reproducing a missing scan." />

## Q4 · Edge operators on a controlled step

Use a constructed five-row image whose first two columns have intensity 150 and last three have intensity 50. Each row is `[150, 150, 50, 50, 50]`. With the four-neighbour Laplacian kernel described in the source, the response at a 150 pixel immediately left of the step is `150 + 50 + 150 + 150 − 4×150 = −100`. At the adjacent 50 pixel immediately right of the step it is `150 + 50 + 50 + 50 − 4×50 = +100`. The sign change marks the step. At image borders, a different padding rule can change results, so these two selected pixels are interior vertically and horizontally.

Sobel estimates a first derivative by combining a directional difference with smoothing in the perpendicular direction. For the same vertical edge, the horizontal derivative responds strongly while the vertical derivative should be zero away from top and bottom borders. A Laplacian response lacks gradient direction, and second derivatives can amplify high-frequency noise. Smoothing before a Laplacian gives the Laplacian of Gaussian; Canny adds smoothing, gradient estimation, non-maximum suppression and hysteresis. None of these is “best” in every task: localisation, continuity, noise and processing cost must be tested on the image distribution.

The source compares Roberts, Prewitt, Sobel, Laplacian and LoG/Canny. Roberts uses a compact 2 by 2 difference and can be sensitive to noise. Prewitt and Sobel use 3 by 3 first-derivative masks; Sobel weights the central row or column more. The Laplacian is a direction-free second derivative, while LoG and Canny include smoothing in different pipelines. Canny is a multi-stage edge detector; LoG is an operator whose zero crossings can be used as edge candidates. Their outputs and parameters are not interchangeable.

<EdgeGradientLab />

## Q5 · The line shared by two point votes

For a source question with two actual coordinates, first find the line through them. If their x coordinates differ, compute `m = (y2−y1)/(x2−x1)` and `c = y1−m×x1`. If x coordinates are equal, the line is vertical and slope-intercept form is undefined. In Hough normal form, use `ρ = x cosθ + y sinθ`; each point votes along a curve over θ, and collinear points share a parameter pair. A peak suggests a line under the chosen angle and rho bins. Two points alone always define a line, but a robust image detector needs enough support and a threshold against accidental alignments.

As an illustration only, points `(1,1)` and `(3,3)` lie on `y=x`. One valid normal angle is 135 degrees and the perpendicular distance ρ is zero. The equivalent representation with a normal 180 degrees away changes the sign of ρ, so an implementation must state its angle and sign convention. This example cannot reproduce the scan's unspecified two points or four-cell accumulator. The interactive lab instead starts with the lecture's separately verified point `(2,2)` at 45 degrees, where ρ is 2.828; change its controls to explore the vote equation.

<HoughVoteLab />

## Q6 · HoG vote and Lucas–Kanade motion

Nine unsigned HoG orientation bins span 0 to 180 degrees, so each is 20 degrees wide. Compute a gradient's magnitude and angle, map the angle modulo 180 degrees and add a magnitude-weighted vote to the appropriate bin. A full HoG implementation commonly interpolates orientation votes and normalises blocks of cells. Without the scanned gradient table, the source's dominant bin cannot be calculated. The lab uses one synthetic 50-degree gradient with magnitude five, hard-assigned to bin index 2, the 40–60-degree range. It demonstrates bin arithmetic, not the missing exam cell.

<HogBinLab />

Lucas–Kanade optical flow assumes that brightness is approximately constant as a patch moves a small amount between frames and that nearby pixels share a flow vector. Linearising brightness constancy gives `Ix u + Iy v + It = 0` for each pixel. One equation has two unknowns, so a window supplies multiple equations and a least-squares fit estimates horizontal and vertical motion. The fit needs gradients in more than one direction; a straight edge alone has the aperture problem and cannot determine motion along itself. Large motion, occlusion, illumination changes and nonrigid movement violate the simple assumptions. A tracker can use flow as one cue rather than treating it as an object identity.

## Code you can run

The first block verifies the constructed Q4 Laplacian responses **−100** and **+100** and the existing edge-lab default `(Gx,Gy)=(4,3)` with magnitude **5** and orientation **36.87°**. The gradient default is a separate lecture example, included so the interactive lab matches a printed calculation.

```python
from math import atan2, degrees, hypot

image = [[150, 150, 50, 50, 50] for _ in range(5)]

def laplacian(row, column):
    neighbours = image[row - 1][column] + image[row + 1][column]
    neighbours += image[row][column - 1] + image[row][column + 1]
    return neighbours - 4 * image[row][column]

left = laplacian(2, 1)
right = laplacian(2, 2)
gradient_magnitude = hypot(4, 3)
gradient_angle = degrees(atan2(3, 4))
print('Synthetic Laplacian:', left, right)
print(f'Gradient magnitude: {gradient_magnitude:.0f}; angle: {gradient_angle:.2f}')
assert (left, right) == (-100, 100)
assert (gradient_magnitude, round(gradient_angle, 2)) == (5, 36.87)
```

The second block verifies the synthetic Q5 line `y=x` and normal angle 135 degrees, then checks the Hough lab's default point `(2,2)` and normal angle 45 degrees. Floating-point rounding gives **ρ=0.000** for the line illustration and **ρ=2.828** for the lab default.

```python
from math import cos, radians, sin

points = [(1, 1), (3, 3)]
rho_on_line = [x * cos(radians(135)) + y * sin(radians(135)) for x, y in points]
rho_default = 2 * cos(radians(45)) + 2 * sin(radians(45))
print('Synthetic line votes:', [round(value, 3) for value in rho_on_line])
print(f'Default point vote: {rho_default:.3f}')
assert all(abs(value) < 1e-12 for value in rho_on_line)
assert round(rho_default, 3) == 2.828
```

The third block checks the HoG lab default. A 50-degree unsigned angle falls in the 40–60-degree bin, index **2**, with magnitude weight **5** under the simple hard-assignment rule. It does not implement interpolation or block normalisation.

```python
angle = 50
magnitude = 5
bins = [0] * 9
index = (angle % 180) // 20
bins[index] += magnitude
print('HoG bin index:', index)
print('Nine synthetic bin weights:', bins)
assert index == 2
assert bins == [0, 0, 5, 0, 0, 0, 0, 0, 0]
```

The fourth block solves a synthetic Lucas–Kanade window with three consistent gradient equations. The first equation fixes `u=2`, the second fixes `v=1`, and the third checks their joint consistency. NumPy least squares prints `[2.0, 1.0]`; it is a method check, not the source paper's unknown motion.

```python
import numpy as np

spatial_gradients = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
negative_temporal_gradient = np.array([2, 1, 3], dtype=float)
flow, residuals, rank, singular_values = np.linalg.lstsq(spatial_gradients, negative_temporal_gradient, rcond=None)
print('Synthetic flow:', [round(float(value), 1) for value in flow])
print('Gradient matrix rank:', rank)
assert np.allclose(flow, [2, 1])
assert rank == 2
```

## Source Q4–Q6 text and answers

The following three panels retain the available source transcript, including its method-level answers. The numeric demonstrations above are separately labelled because the source's referenced scan values are not present.

<details>
<summary><strong>Q4.</strong> Q4 · Edge detection; Laplacian vs Sobel On the given 5×5 image (a 150/50 intensity block forming an edge): (a) using the given kernels, compute the Laplacian at an edge location and compare its performance with Sobel; (b) compare the different edge-detection operators.</summary>

**(a)** Apply the Laplacian kernel (e.g. [[0,1,0],[1,−4,1],[0,1,0]]) by centring it on the edge pixel: response = (sum of 4-neighbours) − 4·(centre). At a 150→50 step the neighbours mix 150 and 50, giving a large non-zero second-derivative response that **changes sign across the edge** (zero-crossing). Sobel instead estimates the *first* derivative Gx,Gy and gives a large |∇| *at* the edge. Compare: Laplacian (2nd-derivative) is isotropic and localises edges via zero-crossings but is **very noise-sensitive**; Sobel (1st-derivative) is more noise-robust and also gives edge *direction*. **(b)** Roberts (2×2, fast, noisy), Prewitt/Sobel (3×3 first-derivative, Sobel weights the centre row/col more so is less noisy), Laplacian (second-derivative, no direction), LoG/Canny (smoothing + derivative → best localisation).

</details>

<details>
<summary><strong>Q5.</strong> Q5 · Hough transform Using the Hough transform, determine the line through two given points and its (m,c) → (ρ,θ) representation; show the parameter-space voting (the 4-cell example).</summary>

Each image point (x,y) maps to a sinusoid ρ = x·cosθ + y·sinθ in (ρ,θ) space; **collinear points' sinusoids intersect at one (ρ,θ)**, which is the line. For the two given points, solve for the common (ρ,θ): the slope-intercept line y=mx+c is rewritten as ρ=x cosθ+y sinθ with θ=atan2 of the normal and ρ the perpendicular distance to the origin. The accumulator cell with the most votes (here the shared intersection of the points' curves) identifies the detected line.

</details>

<details>
<summary><strong>Q6.</strong> Q6 · HOG + Lucas-Kanade (a) Compute a 9-bin Histogram of Oriented Gradients for the given cell: bin gradients by orientation (over 0–180°) weighted by magnitude, and read off the dominant bin. (b) What is the Lucas-Kanade algorithm and where is it used in computer vision?</summary>

**(a) HOG:** for each pixel compute gradient magnitude |∇|=√(Gx²+Gy²) and orientation θ=atan2(Gy,Gx); quantise θ into 9 unsigned bins (0–180°, 20° each) and add each pixel's *magnitude* into its bin (with interpolation); normalise the cell histogram. The bin with the largest accumulated magnitude is the cell's dominant edge orientation. **(b) Lucas-Kanade** is a differential **optical-flow** method: assuming brightness constancy and that flow is roughly constant in a small window, it solves the over-determined system of the optical-flow equation (I_x u + I_y v + I_t = 0) by least squares for the motion (u,v). Used for motion tracking, video stabilisation, and feature tracking (KLT tracker).

</details>

## How to check against the original paper

For Q4, transcribe all 25 intensities, the exact kernel signs and the requested centre pixel before convolving. A reflected, replicated or zero-padded border changes a response near the edge of the 5 by 5 grid. Draw the 3 by 3 neighbourhood and multiply every position by its kernel coefficient. If the output sign differs from this chapter's constructed example, that may reflect a reversed kernel convention rather than a wrong magnitude. Explain both the signed response and the comparison with first-derivative operators.

For Q5, transcribe the two point coordinates and the angle range used by the accumulator. A pair of points has one geometric line, but its normal-form parameters can have equivalent sign and angle representations. If the paper's four cells are a discretised accumulator, identify which angle and distance interval each cell represents before counting votes. A shared vote cell depends on bin widths. State the actual `(m,c)` pair only when the coordinates are visible; a vertical line must be reported without a finite slope.

For Q6, list every provided gradient angle and magnitude. Decide whether the paper specifies hard bin assignment or interpolation, and whether angles are signed or unsigned. Apply the specified rule to all pixels, then find the largest accumulated magnitude. For Lucas–Kanade, state brightness constancy and local constant flow, show the linear equations and explain why multiple gradient directions are needed. A qualitative explanation cannot supply a missing numeric HoG histogram, and a toy histogram cannot substitute for the scanned cell.

This review order keeps the answers reproducible. It separates a mathematical method, an illustrative calculation and the exam's exact data. The three are easy to conflate when a source page refers to images that are no longer in the handover. If those scans become available later, insert their transcribed values into the same four code patterns and compare the printed results to the paper, then update the note with a verified exam answer.

## Further reading

- Built from the course lecture "cv-midsem-2026" (Lecture Library series); available transcript contains Q4–Q6 only.
- [OpenCV image gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) and [Hough lines](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html) for the Q4 and Q5 operations.
- [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) for Lucas–Kanade assumptions and usage.
