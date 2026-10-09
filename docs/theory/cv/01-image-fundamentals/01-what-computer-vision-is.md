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

:::tip Before you start

**You should already know:**

- What a pixel array is: one number per pixel for a grey image, three for a colour image. The [next chapter](/docs/theory/cv/digital-image-formation-and-sampling) covers it in full.
- What a classifier is: a rule that turns numbers into a label. [Logistic regression](/docs/theory/ml/classification-and-logistic-regression) is the simplest example.

**Reading time:** about 30 minutes.

**After this chapter you can:**

- explain why two different scenes can give the same pixels;
- say which level of vision (pixels, regions, meaning) a given task works at;
- test whether a model learned the object or a shortcut.

:::

## In 30 seconds

A camera turns a three-dimensional scene into a flat grid of numbers. Computer vision works backwards: from the numbers, it guesses what was in front of the camera. Think of a shadow on a wall. You can guess the object that cast it, but many objects cast the same shadow, so the guess needs more evidence. A model can also be right on easy pictures for the wrong reason, which only a test on changed pictures reveals.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Pixel | One number (grey) or three numbers (colour) at one place in the grid | A 32 by 32 colour image holds 3,072 numbers |
| Inverse graphics | Working back from an image to the scene that made it | From a flat photo to "a square at this distance" |
| Ambiguity | Several scenes give identical pixels | With focal length 2, the points (1, 1, 2) and (2, 2, 4) both land on (1, 1) |
| Nuisance factor | A change that should not alter the answer | Lighting, rotation, background |
| Shortcut | An easy cue that predicts the label in the training set but not in the world | Red always meant circle |
| HOG | Histogram of oriented gradients: counts of edge directions in small cells | A square gives strong horizontal and vertical counts |
| Pretrained model | A network already trained on a large dataset and reused | ResNet18 trained on ImageNet |


## The idea in plain words

:::note Additions to the course material

The projection example, design checks and runnable code below extend the course material. Its own definitions, task list and practice questions are kept in “How it works” and “Practice questions”.

:::

A rendered image starts with a scene: surfaces have shape and material, lights illuminate them, and a camera converts arriving light into pixel values. Vision starts at the opposite end. It observes the pixels and asks which properties of the world matter for a task. That could mean the object category in an image, the location of every object, the boundary of each region, the path of a tracked object, or the geometry of the camera and scene. The choice of output matters because a system can answer one question correctly while failing another. A classifier that says “car” has not found the car; a detector that draws a box has not segmented its outline.

This is called **inverse graphics**. In the forward direction, a scene and camera create an image. In the inverse direction, an image constrains possible scenes. The inverse is not unique. If a pinhole camera has focal length $f$, a point $(X,Y,Z)$ with positive depth projects to $(fX/Z, fY/Z)$ in a simplified camera-centred coordinate system. With $f=2$, both $(1,1,2)$ and $(2,2,4)$ project to $(1,1)$. They are at different depths, yet the one-pixel coordinate cannot separate them. Real lenses add distortion and sensors add noise, so a production system also estimates camera parameters and uncertainty.

That ambiguity is why a visual model needs additional evidence. A second view gives parallax; temporal frames give motion; a known object size gives a scale cue; a task-specific prior narrows plausible interpretations. A human also uses context, but context can mislead when an image is unusual. A model can exploit background texture correlated with a label instead of the object itself. Evaluating only ordinary images can hide that failure. Test changes in lighting, viewpoint, background, scale and occlusion because these are distinct sources of variation.

The low, mid and high levels provide a useful ladder. Low-level work handles pixels, filtering and local edges. Mid-level work groups pixels into regions and descriptors. High-level work assigns meaning such as class, object instance or action. Modern models often train these stages together, but the conceptual ladder still helps debug where an error enters. If a defect is invisible at the recorded resolution, no later classifier can restore it reliably. If a region boundary is wrong, a perfect category label may still be insufficient for a measurement task.

<Infographic src="/img/cv/inverse-graphics.svg" alt="Forward rendering maps scene, light and camera to pixels, while inverse vision infers a scene from pixels; two different 3D points share one projected coordinate." caption="A single image coordinate constrains a ray, not a unique depth." />

## Worked example, step by step

Two ideas from this chapter fit in small numbers. The first is geometric ambiguity. The second is a shortcut.

**Ambiguity.** Use focal length $f = 2$ and the projection $(fX/Z, fY/Z)$.

1. The point $(1, 1, 2)$ lands at $(2 \cdot 1/2, \; 2 \cdot 1/2) = (1, 1)$.
2. The point $(2, 2, 4)$ lands at $(2 \cdot 2/4, \; 2 \cdot 2/4) = (1, 1)$.
3. Two different depths, one pixel. A single image cannot tell them apart.

**Shortcut.** Suppose a classifier looks only at colour: red means circle, green means square, blue means triangle. In the training images the colour matches the shape 90% of the time. In the other 10% the colour is drawn at random from the three.

1. When the colour matches (90% of images), the rule is right: 0.90.
2. In the other 10% a random colour matches the shape one time in three: 0.10 × 1/3 = 0.0333.
3. Accuracy on images like the training set: 0.90 + 0.0333 = 0.9333.
4. Now show every shape in the next class's colour. The rule is wrong on every image: accuracy 0.
5. The same rule scores 0.9333 on the bench and 0 in the field. The code below reproduces both numbers.

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

The same distinction appears in other applications. An inspection system might first detect a component, then measure a defect's area, and finally decide whether to reject the item. Classification, localisation, measurement and action have different success criteria. A visually plausible label does not prove that a dimension is accurate. Medicine, driving, OCR, remote sensing and augmented reality are typical use cases. Each has a different cost for missed objects, false alarms and geometric errors; the system should be evaluated against the outcome it actually serves.

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

### The colour rule by simulation

This block reproduces the hand calculation. It draws 30,000 labels, applies the 90% colour rule, then moves every colour to the next class.

```python
import numpy as np

rng = np.random.default_rng(0)
labels = rng.integers(0, 3, 30000)
colour = np.where(rng.random(30000) < 0.9, labels, rng.integers(0, 3, 30000))
swapped = (labels + 1) % 3
print("colour rule, colours as in training:", round(float((colour == labels).mean()), 4))
print("colour rule, colours moved to the next class:", float((swapped == labels).mean()))
print("hand calculation:", round(0.9 + 0.1 / 3, 4))
```

**Reading the output.** The simulated accuracy 0.9344 is the hand value 0.9333 plus sampling noise. Moving the colours drops the rule to exactly 0.0, which is worse than the chance level of 0.333. A rule that is confidently wrong is worse than a coin.

### Experiment: four kinds of features on one small task

The task is to tell circles, squares and triangles apart on 32 by 32 colour images with a noisy grey background. Each method gets only 300 training images (100 per shape). In the first training set the shape colour matches the class 90% of the time, which plants a shortcut. In the second, colours are random. Every method is then tested on four sets: the same colours, colours swapped to the next class, shapes rotated through all angles, and Gaussian noise with a standard deviation of 60 grey levels added. The four methods are a colour histogram, HOG with a linear support vector machine (SVM), a tiny convolutional network trained from scratch, and the features of an ImageNet-trained ResNet18 with logistic regression on top.

```python
import cv2
import numpy as np
import torch
import torchvision
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

COLOURS = np.array([[220, 40, 40], [40, 200, 60], [50, 80, 230]])

def draw(label, colour, angle, rng):
    canvas = rng.normal(110, 25, (32, 32, 3)).clip(0, 255).astype(np.uint8)
    centre, radius = rng.uniform(13, 19, 2), rng.uniform(7, 11)
    colour = tuple(int(c) for c in colour)
    if label == 0:
        cv2.circle(canvas, tuple(int(v) for v in centre), int(radius), colour, -1)
    else:
        sides, start = (4, np.pi / 4) if label == 1 else (3, -np.pi / 2)
        corners = [start + angle + 2 * np.pi * k / sides for k in range(sides)]
        points = [centre + radius * np.array([np.cos(a), np.sin(a)]) for a in corners]
        cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], colour)
    return canvas

def make(seed, colours, rotate=False, noise=0, n=100):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), n)
    pick = {"aligned": lambda y: y if rng.random() < 0.9 else rng.integers(3), "swapped": lambda y: (y + 1) % 3, "random": lambda y: rng.integers(3)}[colours]
    images = [draw(y, COLOURS[pick(y)] + rng.normal(0, 15, 3), rng.uniform(0, 2 * np.pi) if rotate else rng.uniform(-0.17, 0.17), rng) for y in labels]
    noisy = np.stack(images) + rng.normal(0, noise, (len(images), 32, 32, 3))
    return noisy.clip(0, 255).astype(np.uint8), labels

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2) / 255

def colour_features(x):
    return np.stack([np.histogramdd(i.reshape(-1, 3), bins=4, range=[(0, 256)] * 3)[0].ravel() / 1024 for i in x])

def hog_features(x):
    return np.stack([hog(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for i in x])

RESNET = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
RESNET.fc = torch.nn.Identity()

def resnet_features(x):
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        return RESNET((torch.nn.functional.interpolate(to_tensor(x), size=64, mode="bilinear") - mean) / std).numpy()

def tiny_cnn(x, y):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(16 * 64, 3))
    opt, xt, yt = torch.optim.Adam(net.parameters(), 3e-3), to_tensor(x), torch.tensor(y)
    for _ in range(25):
        for idx in torch.randperm(len(yt)).split(32):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt[idx]), yt[idx]).backward()
            opt.step()
    return lambda z: net(to_tensor(z)).argmax(1).numpy()

def fit(name, x, y):
    if name == "colour histogram":
        return colour_features, LogisticRegression(max_iter=2000).fit(colour_features(x), y).predict
    if name == "HOG + linear SVM":
        return hog_features, LinearSVC(dual=False).fit(hog_features(x), y).predict
    if name == "ResNet18 + logistic":
        return resnet_features, LogisticRegression(max_iter=3000).fit(resnet_features(x), y).predict
    return (lambda z: z), tiny_cnn(x, y)

tests = {"same colours": make(1, "aligned"), "swapped": make(2, "swapped"), "rotated": make(3, "aligned", rotate=True), "noise sd 60": make(4, "aligned", noise=60)}
print(f"{'model':22s}" + "".join(f"{k:>14s}" for k in tests))
for mode in ("aligned", "random"):
    x, y = make(0, mode)
    print(f"trained with {mode} colours")
    for name in ("colour histogram", "HOG + linear SVM", "tiny CNN", "ResNet18 + logistic"):
        features, predict = fit(name, x, y)
        print(f"  {name:20s}" + "".join(f"{(predict(features(tx)) == ty).mean():14.3f}" for tx, ty in tests.values()))
```

The ResNet18 weights are downloaded on the first run. The block takes about 25 seconds on a laptop CPU.

**Reading the output.** With colours that match the class during training, the table printed on this machine (single seed) is:

| Model | same colours | swapped | rotated | noise sd 60 |
| --- | ---: | ---: | ---: | ---: |
| colour histogram | 0.883 | 0.047 | 0.893 | 0.637 |
| HOG + linear SVM | 0.803 | 0.803 | 0.550 | 0.343 |
| tiny CNN | 0.940 | 0.013 | 0.927 | 0.947 |
| ResNet18 + logistic | 1.000 | 0.990 | 0.987 | 0.513 |

With random colours in training, the tiny network reads 0.917, 0.930, 0.593, 0.743 and ResNet18 reads 1.000, 1.000, 0.943, 0.510. Chance is 0.333 in every cell.

**Line by line.**

- `make` draws every image from its own seeded generator, so each test set is reproducible. The `colours` argument decides how colour relates to class.
- `hog_features` uses grey images on purpose, so HOG sees shape and no colour.
- `RESNET.fc = torch.nn.Identity()` removes the final layer, so the network returns 512 numbers per image. Those numbers become the input of an ordinary logistic regression.
- `fit` returns a pair: a feature function and a predict function. The tiny network needs no separate feature step, so its feature function returns the images unchanged.

### What the experiment shows

Every method has a condition it cannot survive. The colour histogram collapses to 0.047 when colours are swapped, because it never looked at shape. HOG ignores colour (0.803 in both columns) but falls to 0.550 when shapes rotate, because it counts edge directions in fixed cells. The pretrained ResNet18 is perfect or nearly so on three columns, yet drops to 0.513 under noise it never saw.

The surprising row is the tiny network. It scores 0.940 on familiar images, and 0.013 when colours are swapped. It learned the colour rule from the worked example. Its 0.947 under heavy noise is also an artefact of the shortcut: colour survives noise better than shape does. Trained on random colours, the same network reads the swapped images at 0.930 but loses rotation (0.593). Its earlier 0.927 on rotated images was not evidence of rotation invariance. It was the colour cue again.

Limits: one seed, 300 test images per cell (a value near 0.9 carries a 95% interval of about plus or minus 0.035), synthetic shapes at 32 pixels, and a deliberately planted shortcut. The ranking of methods on real photographs will differ. The lesson that transfers is the habit: test on changed data, and read the table by column.

<Infographic src="/img/cv-enrich/v1-shape-shortcuts.svg" alt="A grid of accuracies for four methods against four test conditions: the colour histogram and tiny network fail when colours are swapped, HOG fails on rotation and ResNet18 fails under heavy noise." caption="Read down the colours-swapped column first: 0.047 and 0.013 show two methods that learned colour, not shape." />

## Designing with it

Start with the decision that the image result will support. For a catalogue search task, a category or embedding may be enough. For counting objects, the system needs one detection per instance and a rule for duplicates. For area measurement, a segmentation mask and calibrated scale may be required. For motion, image-by-image boxes need association across frames. For reconstruction, camera geometry and cross-view correspondences are central. Writing the output contract first prevents a team from evaluating a classifier when the product actually needs boundaries or trajectories.

Specify the imaging conditions. Record sensor resolution, exposure behaviour, lens, viewpoint range, frame rate and compression. Data taken from a fixed laboratory camera may fail when the lens changes or the device is moved. Random train/test splits can leak near-duplicate frames or the same physical object into both sets. Split by capture session, site, device, patient or object where that matches deployment. Hold out realistic lighting and occlusion cases, then inspect performance by subgroup rather than only one mean number. The right split is driven by the future data-generating process, not by a convenient percentage.

Treat **illumination and reflectance** separately in reasoning. A dark pixel might mean a dark material, a shadow, a change in exposure or a blocked sensor. A rule that keys on absolute brightness may work on one line and fail on another. Colour constancy, controlled lighting or relative local measurements can reduce this risk, but each needs validation under the intended conditions. An image is evidence about the scene, not a transparent copy of it.

When debugging, trace errors down the ladder. First ask whether the camera actually recorded the feature. Inspect blur, saturation, noise, colour conversion and resolution. Then ask whether a local operator or learned feature responds to the right structure. Only then inspect the high-level prediction. If the label is correct but the box is loose, classification accuracy will not reveal the localisation error. If the model focuses on a background cue, extra training on similar backgrounds may make the failure more confident. Use visual error slices and counterexamples, and keep the original image available for review under the applicable data policy.

Finally, make uncertainty part of the interface. A single image often cannot resolve depth, hidden surfaces or an occluded object. A system can abstain, ask for another view or show a range. Downstream actions should have a stated threshold and a human review path when errors are costly. The point of the inverse-graphics framing is not to promise full 3D recovery from every photo; it is to ask which scene properties are identifiable from the observations at hand.

## Where this stands in 2026

:::info Industry view

- Classical geometry and learned visual features are complementary. A calibrated camera can support metric measurements; a recogniser can supply category or object cues, but one does not replace the other.
- Deep-vision architectures and training mechanics live in the site’s Deep Neural Networks section. This track follows an image, feature, geometry, detection and evaluation spine.
- OpenCV Python tutorials checked on 2026-10-02 identify version 4.13.0. The local CPU examples use NumPy and need no model download or camera; geometry values here are teaching examples, not a benchmark.

:::

## Common mistakes

1. **Reporting one score on test data that looks like the training data.** It feels right because it is the standard split. The planted colour shortcut scored 0.940 on exactly that split. Build at least one shifted test set (colour, rotation, noise, background) and report a table, not a number.
2. **Reading a perfect score from a pretrained model as understanding.** ResNet18 features reached 1.000 and then 0.513 under noise. Test the conditions your camera will actually produce.
3. **Assuming a deep network will learn shape because shape is the intended cue.** A network learns whatever cue is cheapest. The tiny network chose colour. Randomise or remove cues you do not want used, then check that the score survives.
4. **Skipping the simple baseline.** A colour histogram scored 0.883 in a few lines. That is the number a complex model must beat for the right reasons.

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

<details>
<summary><strong>Q6 (Easy).</strong> In the worked example, what does the colour rule score if training colours match the shapes 100% of the time, and what does it score after the swap?</summary>

It scores 1.0 on images like the training set and still 0 after the swap. Raising the match rate makes the shortcut look better in the lab and does not change what happens when the world changes.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> Why is the colour histogram's score of 0.047 on swapped colours below the chance level of 0.333?</summary>

The swap is built to put each shape in the colour of the next class. A model that relies on colour is then not guessing; it is steered to the wrong answer almost every time. Scores far below chance are a clear sign that a model learned a cue that has been reversed.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> A model for reading chest X-rays scores well on the hospital that supplied the training data and badly at a second hospital. Name two shortcuts that could explain this and one test that would separate them.</summary>

Scanner-specific artefacts and text markers that correlate with the diagnosis at the first site are two. Evaluate on the second site with those markers masked, or split by site, as the design section recommends.

</details>

## Further reading

- [OpenCV camera-calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for intrinsics, distortion and multi-view correspondences.
- [Computer Vision: Algorithms and Applications, second edition](https://szeliski.org/Book/) for the broader inverse-problem framing.
- [torchvision ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), opened on 2026-10-09: the `IMAGENET1K_V1` weights list 69.758 top-1 accuracy on ImageNet and 11,689,512 parameters. The page states no licence for the weights, so check the terms before shipping them.
- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/), opened on 2026-10-09: second edition (2022), free PDF for personal use.
- Versions run for the experiment: OpenCV 5.0.0 (`cv2`), scikit-image 0.26.0, scikit-learn 1.9.1, PyTorch 2.14.1 and torchvision 0.29.1.
- Built from the course lecture "cv-s1-intro" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Putting the levels together

Imagine a camera above a conveyor that must reject packages with a torn label. A low-level operator can reveal strong intensity changes; this may make a tear easier to see, but it will also respond to printed letters and shadows. A mid-level step can group edge fragments or segment a candidate label region. The high-level decision asks whether this particular region is defective according to an agreed annotation rule. A system that stops at “package present” has answered a different question. A system that finds a tear but cannot locate the package's identity may still be unusable when the rejection actuator must fire at the right moment.

The design should follow an error budget through these stages. If the camera sometimes clips the label, that is an acquisition failure. If the relevant texture is present but the preprocessing smooths it away, that is a representation failure. If the feature survives but the model calls it acceptable, that is a decision failure. Each needs a different remedy. Collecting more labels will not fix a camera aimed at the wrong part of the belt. Increasing image resolution may not fix a decision rule trained on biased examples. The levels make a practical debugging sequence rather than a rigid three-module architecture.

Projection ambiguity also affects what ground truth means. A photograph can have a reliable 2D box annotation while its object's physical distance remains unknown. A depth label might come from stereo, a range sensor, known geometry or manual measurement, each with its own uncertainty. If a project needs a physical size, calibrating pixel dimensions into world units is part of the data pipeline. An output in pixels should not silently become an output in millimetres. The inverse-graphics formula shows why: changing depth changes apparent size even if the object's actual size stays fixed.

An evaluation set should therefore carry the information needed to test the intended output. Classification needs class labels and a policy for ambiguous cases. Detection needs instance boxes, overlap criteria and a duplicate policy. Segmentation needs pixel-level masks and boundary conventions. Tracking needs identities over time, with rules for occlusion and reappearance. Reconstruction needs geometric reference measurements. Agreement between annotators can limit the achievable score when boundaries are subjective. Report those limits instead of treating the annotation as a perfect observation of the world.

The smallest useful experiment is often a simple, transparent baseline. Try a threshold or hand-designed geometric rule on a representative subset, record where it fails, and compare a learned method against those same cases. A baseline can expose that the requirement is actually a measurement problem, that lighting dominates, or that there are too few independent examples. Its purpose is diagnosis, not nostalgia for classical vision. The later chapters show when filters, descriptors, robust geometry and learned models help, and where their assumptions break.

## Check yourself

- I can explain why two points at different depths can share the same pinhole projection.
- I can distinguish a class label, a box, a mask, a track and a geometric reconstruction.
- I can explain which acquisition, representation and decision failures require different fixes.
- I can state what extra evidence would reduce ambiguity for a measurement task.
- I can read a table of methods against test conditions and say which cue each method relied on.
- I can explain why a perfect in-distribution score does not show that a model uses the object.
- I can compute the ceiling and the collapse of a colour-only rule by hand.
