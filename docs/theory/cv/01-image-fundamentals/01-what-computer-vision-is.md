---
id: cv-what-computer-vision-is
title: "Computer Vision · Session 1; What Computer Vision Is"
sidebar_label: "1 · What vision asks"
sidebar_position: 1
slug: /theory/cv/what-computer-vision-is
description: "Understand vision as inverse graphics, why images are ambiguous, and how low, mid and high-level tasks differ."
tags: [computer-vision, image-formation, inverse-graphics]
---

import Infographic from '@site/src/components/Infographic';
import PerspectiveProjectionLab from '@site/src/components/viz/PerspectiveProjectionLab';

**In one line.** Computer vision works backwards from image measurements to a useful account of the scene, even though several scenes may produce the same pixels.

## The idea in plain words

:::note Beyond the lecture

The projection example, design checks and runnable code below extend the lecture. The lecture's own definitions, task list and practice questions are kept in “How it works” and “Practice questions”.

:::

A rendered image starts with a scene: surfaces have shape and material, lights illuminate them, and a camera converts arriving light into pixel values. Vision starts at the opposite end. It observes the pixels and asks which properties of the world matter for a task. That could mean the object category in an image, the location of every object, the boundary of each region, the path of a tracked object, or the geometry of the camera and scene. The choice of output matters because a system can answer one question correctly while failing another. A classifier that says “car” has not found the car; a detector that draws a box has not segmented its outline.

The lecture calls this **inverse graphics**. In the forward direction, a scene and camera create an image. In the inverse direction, an image constrains possible scenes. The inverse is not unique. If a pinhole camera has focal length $f$, a point $(X,Y,Z)$ with positive depth projects to $(fX/Z, fY/Z)$ in a simplified camera-centred coordinate system. With $f=2$, both $(1,1,2)$ and $(2,2,4)$ project to $(1,1)$. They are at different depths, yet the one-pixel coordinate cannot separate them. Real lenses add distortion and sensors add noise, so a production system also estimates camera parameters and uncertainty.

That ambiguity is why a visual model needs additional evidence. A second view gives parallax; temporal frames give motion; a known object size gives a scale cue; a task-specific prior narrows plausible interpretations. A human also uses context, but context can mislead when an image is unusual. A model can exploit background texture correlated with a label instead of the object itself. Evaluating only ordinary images can hide that failure. Test changes in lighting, viewpoint, background, scale and occlusion because these are distinct sources of variation.

The lecture's low, mid and high levels provide a useful ladder. Low-level work handles pixels, filtering and local edges. Mid-level work groups pixels into regions and descriptors. High-level work assigns meaning such as class, object instance or action. Modern models often train these stages together, but the conceptual ladder still helps debug where an error enters. If a defect is invisible at the recorded resolution, no later classifier can restore it reliably. If a region boundary is wrong, a perfect category label may still be insufficient for a measurement task.

<Infographic src="/img/cv/inverse-graphics.svg" alt="Forward rendering maps scene, light and camera to pixels, while inverse vision infers a scene from pixels; two different 3D points share one projected coordinate." caption="A single image coordinate constrains a ray, not a unique depth." />

## How it works

### What is Computer Vision?

Making machines extract meaning from images/video; recovering scene properties (objects, geometry, motion, semantics) from 2D light measurements.

### Why vision is hard

- **Projection**; 3D→2D loses depth; the mapping is many-to-one and ambiguous.
- **Nuisance factors**; Illumination, viewpoint, scale, occlusion, deformation, clutter, noise.
- **Intra-class variation**; Objects of one class look wildly different.

:::note

**Under-determined.** The same brightness can come from a bright surface in shadow or a dark one in light; we recover the most probable scene, not the only one.

:::

### Levels & approaches

- **Levels**; Low (pixels, edges) → mid (regions, features) → high (objects, scenes).
- **Classic vs modern**; Hand-engineered features (SIFT, HoG) vs learned features (CNNs, ViTs).

### Core tasks & applications

Classification, detection, segmentation, recognition, tracking, 3D reconstruction; applied to medicine, driving, face ID, OCR, inspection, AR/VR, remote sensing.

### Key takeaways

- **1 · Inverse graphics**; Infer a scene model from images.
- **2 · Hard**; Many-to-one projection + nuisance factors.
- **3 · Ladder**; Low → mid → high level.

## A real system that works this way

OpenCV's camera-calibration workflow makes the inverse relationship concrete. A known chessboard pattern provides correspondences between points on a physical plane and points measured in several images. The software estimates camera intrinsics such as focal lengths and principal point, together with lens distortion and each view's pose. It can then undistort later images from the same camera. The official tutorial asks for multiple views because one image rarely constrains every parameter well, and it reports that some pattern images may fail corner detection.

This is a real geometry workflow, not an example of a network recognising “chessboard” as a semantic class. The system knows the pattern's layout and seeks geometric parameters that explain observations. If corner locations are inaccurate or all calibration views have similar pose, the estimate can be weak. Reprojection error checks whether the fitted camera maps known 3D points close to their measured image locations; it does not by itself prove that the camera will work for every later scene. A calibration used for measurement should also be checked against held-out views and the camera's actual focus and zoom setting.

The same distinction appears in other applications. An inspection system might first detect a component, then measure a defect's area, and finally decide whether to reject the item. Classification, localisation, measurement and action have different success criteria. A visually plausible label does not prove that a dimension is accurate. The lecture names medicine, driving, OCR, remote sensing and augmented reality as use cases. Each has a different cost for missed objects, false alarms and geometric errors; the system should be evaluated against the outcome it actually serves.

## Code you can run

The first block reproduces the board's ambiguity with the pinhole formula. It prints **(1.0, 1.0)** for two different 3D points. Changing focal length in the lab moves both projections together because both points remain on one ray.

```python
from math import isclose

focal_length = 2.0
points = [(1.0, 1.0, 2.0), (2.0, 2.0, 4.0)]
projected = [(focal_length * x / z, focal_length * y / z) for x, y, z in points]
print('3D points:', points)
print('Image coordinates:', projected)
assert projected == [(1.0, 1.0), (1.0, 1.0)]
assert not isclose(points[0][2], points[1][2])
```

<PerspectiveProjectionLab />

The second block starts with a synthetic image so it runs without a download or camera. A threshold turns pixels into a candidate region, then the region's bounding box is calculated. The result is **16 selected pixels** and a half-open box from `(2, 2)` to `(6, 6)`. It illustrates low-level measurement and mid-level grouping. It is deliberately **not** a semantic recogniser: an intensity threshold cannot tell whether the bright square is a car, a tumour or noise.

```python
import numpy as np

image = np.zeros((8, 8), dtype=np.uint8)
image[2:6, 2:6] = 200
candidate = image > 100
ys, xs = np.nonzero(candidate)
box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
print('Selected pixels:', int(candidate.sum()))
print('Half-open box:', box)
assert int(candidate.sum()) == 16
assert box == (2, 2, 6, 6)
```

The difference between a mask and a box matters. A box includes background when an object is not rectangular; a mask can describe an irregular boundary but depends on pixel resolution and annotation rules. Later vision chapters define intersection over union and segmentation metrics for these outputs. The first block explains ambiguity in geometric inference; the second explains that an output representation is chosen for a task, not produced by every visual algorithm automatically.

## Designing with it

Start with the decision that the image result will support. For a catalogue search task, a category or embedding may be enough. For counting objects, the system needs one detection per instance and a rule for duplicates. For area measurement, a segmentation mask and calibrated scale may be required. For motion, image-by-image boxes need association across frames. For reconstruction, camera geometry and cross-view correspondences are central. Writing the output contract first prevents a team from evaluating a classifier when the product actually needs boundaries or trajectories.

Specify the imaging conditions. Record sensor resolution, exposure behaviour, lens, viewpoint range, frame rate and compression. Data taken from a fixed laboratory camera may fail when the lens changes or the device is moved. Random train/test splits can leak near-duplicate frames or the same physical object into both sets. Split by capture session, site, device, patient or object where that matches deployment. Hold out realistic lighting and occlusion cases, then inspect performance by subgroup rather than only one mean number. The right split is driven by the future data-generating process, not by a convenient percentage.

Treat **illumination and reflectance** separately in reasoning. A dark pixel might mean a dark material, a shadow, a change in exposure or a blocked sensor. A rule that keys on absolute brightness may work on one line and fail on another. Colour constancy, controlled lighting or relative local measurements can reduce this risk, but each needs validation under the intended conditions. An image is evidence about the scene, not a transparent copy of it.

When debugging, trace errors down the ladder. First ask whether the camera actually recorded the feature. Inspect blur, saturation, noise, colour conversion and resolution. Then ask whether a local operator or learned feature responds to the right structure. Only then inspect the high-level prediction. If the label is correct but the box is loose, classification accuracy will not reveal the localisation error. If the model focuses on a background cue, extra training on similar backgrounds may make the failure more confident. Use visual error slices and counterexamples, and keep the original image available for review under the applicable data policy.

Finally, make uncertainty part of the interface. A single image often cannot resolve depth, hidden surfaces or an occluded object. A system can abstain, ask for another view or show a range. Downstream actions should have a stated threshold and a human review path when errors are costly. The point of the inverse-graphics framing is not to promise full 3D recovery from every photo; it is to ask which scene properties are identifiable from the observations at hand.

## Where this stands in 2026

:::info Industry view

- Classical geometry and learned visual features are complementary. A calibrated camera can support metric measurements; a recogniser can supply category or object cues, but one does not replace the other.
- Deep-vision architectures and training mechanics live in the site’s Deep Neural Networks section. This track follows the lecture’s image, feature, geometry, detection and evaluation spine.
- OpenCV Python tutorials checked on 2026-10-02 identify version 4.13.0. The local CPU examples use NumPy and need no model download or camera; geometry values here are teaching examples, not a benchmark.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Define computer vision and contrast it with computer graphics.</summary>

CV infers a scene model (objects, geometry, motion) from 2D images; graphics renders an image from a model. CV is the inverse problem.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Give four reasons vision is hard.</summary>

Any of: 3D→2D depth loss (many-to-one projection), illumination changes, viewpoint/scale variation, occlusion, deformation, background clutter, intra-class variation, noise.<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Explain the three levels of vision.</summary>

Low-level (pixels: filtering, edges), mid-level (regions/features: segments, corners), high-level (semantics: recognition, detection, scene understanding).<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Contrast classic and modern CV.</summary>

Classic hand-engineers features (SIFT, HoG) + a classifier; modern learns features end-to-end with deep nets (CNNs, ViTs).<br /><em>Session 1 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Name four core CV tasks.</summary>

Classification, object detection, segmentation, recognition, tracking, 3D reconstruction.<br /><em>Session 1 · conceptual</em>

</details>

## Further reading

- [OpenCV camera-calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for intrinsics, distortion and multi-view correspondences.
- [Computer Vision: Algorithms and Applications, second edition](https://szeliski.org/Book/) for the broader inverse-problem framing.
- Built from the course lecture "cv-s1-intro" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Putting the levels together

Imagine a camera above a conveyor that must reject packages with a torn label. A low-level operator can reveal strong intensity changes; this may make a tear easier to see, but it will also respond to printed letters and shadows. A mid-level step can group edge fragments or segment a candidate label region. The high-level decision asks whether this particular region is defective according to an agreed annotation rule. A system that stops at “package present” has answered a different question. A system that finds a tear but cannot locate the package's identity may still be unusable when the rejection actuator must fire at the right moment.

The design should follow an error budget through these stages. If the camera sometimes clips the label, that is an acquisition failure. If the relevant texture is present but the preprocessing smooths it away, that is a representation failure. If the feature survives but the model calls it acceptable, that is a decision failure. Each needs a different remedy. Collecting more labels will not fix a camera aimed at the wrong part of the belt. Increasing image resolution may not fix a decision rule trained on biased examples. The lecture's levels make a practical debugging sequence rather than a rigid three-module architecture.

Projection ambiguity also affects what ground truth means. A photograph can have a reliable 2D box annotation while its object's physical distance remains unknown. A depth label might come from stereo, a range sensor, known geometry or manual measurement, each with its own uncertainty. If a project needs a physical size, calibrating pixel dimensions into world units is part of the data pipeline. An output in pixels should not silently become an output in millimetres. The inverse-graphics formula shows why: changing depth changes apparent size even if the object's actual size stays fixed.

An evaluation set should therefore carry the information needed to test the intended output. Classification needs class labels and a policy for ambiguous cases. Detection needs instance boxes, overlap criteria and a duplicate policy. Segmentation needs pixel-level masks and boundary conventions. Tracking needs identities over time, with rules for occlusion and reappearance. Reconstruction needs geometric reference measurements. Agreement between annotators can limit the achievable score when boundaries are subjective. Report those limits instead of treating the annotation as a perfect observation of the world.

The smallest useful experiment is often a simple, transparent baseline. Try a threshold or hand-designed geometric rule on a representative subset, record where it fails, and compare a learned method against those same cases. A baseline can expose that the requirement is actually a measurement problem, that lighting dominates, or that there are too few independent examples. Its purpose is diagnosis, not nostalgia for classical vision. The later chapters show when filters, descriptors, robust geometry and learned models help, and where their assumptions break.

## Check yourself

- I can explain why two points at different depths can share the same pinhole projection.
- I can distinguish a class label, a box, a mask, a track and a geometric reconstruction.
- I can explain which acquisition, representation and decision failures require different fixes.
- I can state what extra evidence would reduce ambiguity for a measurement task.
