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

## The idea in plain words

:::note Beyond the lecture

The signed-gradient experiment and design discussion extend the lecture. Its edge profiles, gradient example, Sobel and Prewitt discussion and all practice questions remain below.

:::

Edges are not objects. They are locations where a measured image value changes quickly over space. A black shape against a white background creates one sort of edge, but a shadow, reflection, texture stripe or compression artifact can also create a strong change. The lecture introduces edge profiles to make that distinction visible. A step changes abruptly, a ramp changes over several pixels, and a roof rises then falls like a narrow line. In real images, optics and sampling spread many ideal steps into ramps. The profile shape affects how a derivative responds and how a threshold should be interpreted.

The image gradient packages the local change into two components: $G_x$ for horizontal variation and $G_y$ for vertical variation. The vector points in the direction where intensity rises fastest. Its magnitude is $\sqrt{G_x^2+G_y^2}$, and its orientation is best computed with `atan2(Gy, Gx)` so the correct quadrant is retained. An edge boundary itself runs approximately perpendicular to that vector. The lecture's worked values $G_x=4$ and $G_y=3$ give magnitude 5 and angle 36.87°. The angle describes the normal to a local intensity boundary, not the angle of a whole detected object.

Digital images do not provide a continuous derivative. Operators approximate it with neighbouring samples. Sobel and Prewitt use small masks that combine a directional difference with some smoothing across the other direction. This reduces sensitivity to isolated noise compared with a raw two-pixel difference. The Laplacian is a second-derivative operator; a zero crossing can mark a transition after appropriate smoothing, but the response is also sensitive to noise. No operator alone knows whether the transition is a useful region boundary.

Think about the gradient's sign before converting types. A dark-to-bright transition has one sign; bright-to-dark has the opposite sign for the same axis. If a signed derivative is forced directly into an unsigned 8-bit output, negative values may be clipped to zero. The official OpenCV gradient tutorial calls this out explicitly. Work in a signed or floating representation when measuring both directions, then take magnitude or a suitable absolute value for display. This is a common example of a numerically valid function call producing an incomplete result because its output type was chosen incorrectly.

An edge map is usually created by thresholding gradient strength, but a single threshold can leave thick ridges, gaps and unrelated texture. A high threshold suppresses weak true boundaries; a low threshold admits noise. The next lecture's Canny process adds non-maximum suppression and hysteresis to address these issues. Before adding that complexity, make sure the image can actually resolve the target boundary and that its contrast survives acquisition and preprocessing.

<Infographic src="/img/cv/gradients.svg" alt="A board compares step, ramp and roof edge profiles, Sobel gradient components 4 and 3 giving magnitude 5, and an orientation of 36.87 degrees." caption="Gradient magnitude measures change; its direction points across the edge." />

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

OpenCV's `Sobel`, `Scharr` and `Laplacian` functions are a practical implementation of the operators in the lecture. The current official Python tutorial demonstrates first derivatives in horizontal and vertical directions and a second-derivative Laplacian. It also warns that an unsigned output hides negative slopes. This makes OpenCV a useful named system for understanding the engineering detail: an edge algorithm is not just a mathematical mask, but also a data type, border policy, kernel size and display mapping.

Imagine a camera reading a printed black rectangle on a light package. The left boundary goes light to dark and the right goes dark to light. A signed horizontal derivative should show opposite responses for those two sides. If the application only preserves positive derivative values, it may detect the left side and miss the right. The toy code below reproduces that failure with a bright vertical band on a dark field. A production system would then combine orientations and signed magnitudes, inspect the line geometry, and confirm that reflections or package seams are not mistaken for the print boundary.

The tutorial was checked on 2026-10-02 as OpenCV documentation version 4.13.0. The local CPU example was executed with `opencv-python-headless` 5.0.0.93. It uses a synthetic image and makes no claim about detection quality on a real camera feed. The local example isolates the mechanism so a reader can test the sign and data type rather than relying on a figure copied from the tutorial.

## Code you can run

The first block checks the lecture's vector. `atan2` gives **36.87°** for the default `(4,3)` vector and handles negative components correctly if you change the inputs. When both components are zero, orientation has no useful meaning, although magnitude is zero.

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

## Further reading

- [OpenCV image gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) for Sobel, Scharr, Laplacian and signed-output guidance.
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
