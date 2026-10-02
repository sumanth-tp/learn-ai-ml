---
id: cv-digital-image-formation-and-sampling
title: "Computer Vision · Session 2; Image Fundamentals"
sidebar_label: "2 · Digital images"
sidebar_position: 2
slug: /theory/cv/digital-image-formation-and-sampling
description: "Follow light through projection, sampling and quantisation, and verify the storage and aliasing calculations."
tags: [computer-vision, image-formation, sampling, quantisation]
---

import Infographic from '@site/src/components/Infographic';
import SamplingQuantisationLab from '@site/src/components/viz/SamplingQuantisationLab';

**In one line.** A digital image is a sampled spatial grid whose recorded values are quantised measurements of light.

## The idea in plain words

:::note Beyond the lecture

The storage-unit clarification, aliasing experiment and acquisition guidance extend the lecture. The source's formation model, worked storage example and questions are retained below.

:::

The camera does not capture a complete 3D world. Light from scene points passes through optics and lands on a finite sensor. Each sensor site measures a response over a small area and exposure interval. An analogue signal is then converted into digital values. By the time an algorithm sees an array, scene geometry, lens behaviour, sampling, quantisation, colour processing and perhaps compression have already shaped the evidence. A model cannot infer details that were never recorded with enough signal.

Write a grayscale image as $f(x,y)$, the recorded intensity at discrete horizontal and vertical coordinates. In code, array indexing is usually row then column, so `image[y, x]` is the value at $(x,y)$. A colour image has an additional channel axis. An RGB array with shape `(height, width, 3)` gives red, green and blue values for each pixel. The number in a pixel is not necessarily physical luminance. It may be gamma encoded, white balanced or transformed by an imaging pipeline. Interpret it according to the representation and data type before applying arithmetic.

The lecture's simple formation model is $f(x,y)=i(x,y)r(x,y)$: illumination times surface reflectance. It isolates why brightness alone is ambiguous. If illumination falls while reflectance rises, the product can stay similar. The model is a teaching approximation: real sensors integrate over wavelength and time; there are shadows, specular reflections, camera response curves and noise. Still, the separation is useful. A change in a pixel can come from the object, the light or the camera. A threshold trained in one lighting condition should be tested when those conditions change.

**Sampling** fixes locations on a finite grid. A 512 by 512 image contains 262,144 pixel positions. If a fine stripe pattern oscillates faster than the grid can represent, distinct patterns can produce the same samples. This is aliasing. Blurring or low-pass filtering before reducing resolution removes frequencies the smaller grid cannot hold. Merely choosing an interpolation algorithm after an aliased capture cannot recover the original scene. Spatial resolution describes the number and spacing of samples, not how many intensity levels each sample can take.

**Quantisation** maps continuous signal values to a finite set of numbers. With $b$ bits, a channel has $2^b$ representable codes; eight bits give 256 codes, usually 0 through 255 for unsigned values. Too few levels can create visible steps in a smooth gradient, called banding or false contouring. More bits increase possible levels but cannot remove sensor noise or restore clipped highlights. Effective precision depends on the whole imaging chain, not just the file's declared bit depth.

<Infographic src="/img/cv/digital-image.svg" alt="A scene is projected onto a sensor, sampled as a 512 by 512 grid, quantised to 256 levels per channel at eight bits, and stored as 786432 uncompressed RGB bytes." caption="Spatial samples, intensity levels and stored bytes are related but distinct." />

## How it works

### An image is a matrix

f(x,y) is the intensity at pixel (x,y). Grayscale = one 2D array; colour = three channels (R,G,B).

### Image formation

A pinhole/lens projects the 3D scene onto the sensor (inverted image), just like the eye onto the retina.

:::tip

Recorded intensity f(x,y) = illumination i(x,y) × reflectance r(x,y), with 0 ≤ r ≤ 1.

:::

### Sampling & quantization

- **Sampling**; Discretise space into a pixel grid (spatial resolution). Too few → aliasing.
- **Quantization**; Discretise intensity into L = 2^b levels (bit-depth). Too few → false contouring.

:::tip

**Worked.** 512×512×3 at 8 bits = 786,432 bytes = 768 KB; L = 2⁸ = 256 levels/channel.

:::

### Key takeaways

- **1 · Matrix**; f(x,y)=i·r.
- **2 · Digitise**; Sample (space) + quantize (L=2^b).
- **3 · Resolution**; Spatial = pixels; intensity = bit-depth.

## A real system that works this way

OpenCV's `resize` operation is a concrete place where sampling decisions become software choices. The official geometric-transform documentation describes mapping each destination pixel back to a source coordinate and choosing an interpolation rule. It lists nearest-neighbour, linear, cubic, area and Lanczos-style options; the area method is a candidate for reduction because it accounts for a pixel area during decimation. This is not a claim that one interpolation method is best for every image. The correct choice depends on whether the data are photographs, class-ID masks, depth maps or another representation.

For a photograph, blending neighbouring values can be appropriate when shrinking. For a semantic segmentation mask, blending label IDs creates values that are not categories; nearest-neighbour resampling preserves the discrete class set. For a binary measurement mask, resampling can change area and perimeter, so evaluate the effect rather than treating it as display-only. A production pipeline should record the original dimensions, colour order, resize method, crop coordinates and any normalization. Training and serving must perform compatible transformations, or the model sees a different signal from what it learned.

OpenCV's Python tutorials were checked at version 4.13.0 on 2026-10-02. The CPU code in this chapter uses NumPy and the installed `opencv-python-headless` 5.0.0.93 package. It does not rely on a screen, camera or downloaded file. The difference between the documentation version and installed wheel is explicit; the API calls used below were run locally, and behaviour should be rechecked when moving a production pipeline to another version.

## Code you can run

The first block verifies the lecture's storage calculation. A tightly packed 512 × 512 RGB array with one byte per channel occupies **786,432 bytes**, or **768 KiB**. The 8-bit channel holds **256 levels**. A compressed image file may be smaller or larger because headers and coding change the stored size; this is a raw-array calculation.

```python
import numpy as np

height = width = 512
channels = 3
bits_per_channel = 8
image = np.zeros((height, width, channels), dtype=np.uint8)
levels = 2 ** bits_per_channel
print('Shape:', image.shape)
print('Levels per channel:', levels)
print('Raw bytes:', image.nbytes, 'KiB:', image.nbytes / 1024)
assert image.nbytes == 786432
assert image.nbytes / 1024 == 768
assert levels == 256
```

:::note Correction to the source calculation

The source calls 786,432 bytes “768 KB”. That equality uses 1,024-byte units, whose precise symbol is **KiB**. In decimal units the same amount is 786.432 kB. The numeric byte count is correct.

:::

Change resolution, channels or bit depth in the lab. Its default matches the code. For non-byte-aligned bit depths, the lab displays a packed-bit minimum; an array library may allocate a whole byte or word for each sample.

<SamplingQuantisationLab />

The second block shows why direct decimation can be misleading. A 16 by 16 alternating pattern sampled at every second column becomes uniform because only one phase is observed. An area-based resize averages the conflicting values. The pattern is synthetic, so the result is a sampling demonstration rather than a claim about camera image quality.

```python
import cv2
import numpy as np

source = np.tile(np.array([0, 255] * 8, dtype=np.uint8), (16, 1))
naive = source[:, ::2]
area = cv2.resize(source, (8, 16), interpolation=cv2.INTER_AREA)
print('Direct-sample unique values:', np.unique(naive).tolist())
print('Area-resize unique values:', np.unique(area).tolist())
assert np.unique(naive).tolist() == [0]
assert np.unique(area).tolist() in ([128], [127])
```

The exact midpoint integer can depend on rounding conventions, so the assertion accepts either adjacent code. The important distinction is that the naive sample falsely appears all dark, while the area estimate records a mid-level response. Anti-alias filtering before sampling matters whenever detail approaches the grid limit.

## Designing with it

Keep spatial and intensity resolution separate in design reviews. Increasing width and height can reveal smaller spatial structures, but also changes memory, bandwidth and inference cost. Increasing bit depth can preserve a finer tonal gradient, but only if the sensor, file format and later processing retain meaningful precision. Neither change is automatically useful if the task target is larger than the current pixels or the sensor noise dominates one-code differences. Run a controlled comparison on representative data and the actual downstream metric.

Define the coordinate convention. Images often use origin at the top left with x increasing right and y increasing down; world coordinates may use a different origin and axis direction. Bounding boxes may use inclusive or half-open end coordinates. A one-pixel disagreement can alter IoU substantially for tiny objects. If a crop or resize is applied, transform annotations using exactly the same geometry. Keep the original image or a reproducible transform record so a prediction can be mapped back to source coordinates.

Document colour and data type. OpenCV commonly reads colour images in BGR channel order, while many plotting and model libraries expect RGB. A silent swap can change red and blue objects without changing array shape. A float image normalized to 0–1 and a uint8 image coded 0–255 may look similar when displayed, but a filter or model will behave differently if given the wrong scale. Check channel order and range at the input boundary with small known colour patches and simple tests.

Separate acquisition quality from preprocessing. Saturated pixels have lost highlight detail; clipped shadows have lost dark detail. Resizing cannot restore either. A blurry image may have enough pixels in its file yet insufficient optical detail. Compression artifacts can create edges that a detector mistakes for structure. Before tuning a model, inspect representative raw frames and measure how often exposure, focus, motion blur and occlusion violate the assumed capture conditions. These checks often identify a cheaper fix in lighting or camera placement.

For datasets, preserve provenance and split by independent capture units. Images from one video have strong temporal correlation; a random frame-level split can overstate generalisation. Differences in resolution or bit depth can also reveal the source site, becoming a shortcut for the label. Standardise only when the transformation is justified and tested. The pipeline should store the chosen dimensions, interpolation, crop, normalisation and colour mapping as part of a versioned model input contract.

## Where this stands in 2026

:::info Industry view

- Images are now handled by learned systems as well as classical operators, but every method receives sampled and quantised input. Acquisition and preprocessing remain part of the model specification.
- The local example was run with `opencv-python-headless` 5.0.0.93 and NumPy. OpenCV documentation checked on 2026-10-02 presented version 4.13.0; production builds should pin and re-run their own version.
- The chapter treats 786,432 bytes as uncompressed data. It makes no claim about JPEG, PNG or video file size because content and encoder settings determine that size.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is a digital image, mathematically?</summary>

A matrix of intensity values: f(x,y) gives the brightness at pixel (x,y); colour images stack channels (R,G,B).<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the image formation model and define its terms.</summary>

f(x,y) = i(x,y)·r(x,y): illumination i multiplied by reflectance r, with 0 ≤ r ≤ 1.<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Contrast sampling and quantization.</summary>

Sampling discretises space into a pixel grid (spatial resolution; too few → aliasing). Quantization discretises intensity into L=2^b levels (bit-depth; too few → false contouring).<br /><em>Session 2 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> How many bytes does a 512×512 RGB image at 8 bits/channel need?</summary>

512×512×3×(8/8) = 786,432 bytes = 768 KiB; L = 2⁸ = 256 levels/channel.<br /><em>Session 2 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Distinguish spatial resolution from intensity resolution.</summary>

Spatial resolution is the number of pixels (e.g. 1920×1080); intensity resolution is the bit-depth (levels per pixel).<br /><em>Session 2 · conceptual</em>

</details>

## Further reading

- [OpenCV geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) for resize, interpolation and reverse mapping.
- [Szeliski’s textbook site](https://szeliski.org/Book/) for image formation and sensing in the broader vision pipeline.
- Built from the course lecture "cv-s2-image-fundamentals" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## From the camera to a model input

Imagine inspecting a tiny printed code on a moving package. First calculate the physical size of a code element at the expected camera distance. Then estimate how many sensor pixels cover that element under the lens and crop. If the element spans less than a few reliable samples, a perfect classifier trained on enlarged images will still fail at the real camera. Slowing the belt, changing the lens or moving the camera can improve the signal before any model change. The intended model input size is an engineering choice only after the optics and task geometry are understood.

Suppose the original image is 512 pixels square and the model consumes 256 pixels square. A naive every-other-pixel sample throws away one spatial phase. An area-based reduction combines nearby values, but also softens very thin details. If a one-pixel crack at the original scale is the target, the resize may remove the evidence. A team should compare detection on the original and reduced scales, inspect the small-object error slice, and decide whether to preserve a high-resolution crop. A global image metric can hide the exact defect class that drove the requirement.

Bit depth has a similar task-dependent trade-off. A smooth medical or scientific signal may benefit from more than 8 bits if acquisition and processing retain the extra levels. A simple line detector on high-contrast drawings may not. Exporting a high-bit-depth source as an 8-bit preview for annotation can lose subtle structure before training. Conversely, storing nominal 16-bit values when the sensor only produces a few noisy effective bits adds data volume without reliable information. Validate effective precision with calibration targets and real operating conditions.

Finally, write one reproducible transformation function for training, validation and serving. It should state how files are decoded, which channel order is used, how the crop is chosen, when filtering occurs, what interpolation is applied, and how numeric values are scaled. Save the version alongside model evaluation. A model score obtained after one resize rule does not automatically apply to another. This is a general ML reproducibility issue expressed at the pixel level.

When comparing experiments, report both the transformed image size and the original capture size. A larger model input produced by upscaling a small original does not create new scene detail. It can change interpolation artifacts and compute cost, but the lost frequencies remain lost.

## Check yourself

- I can compute raw storage from height, width, channels and bits per channel with the correct unit.
- I can explain the difference between spatial sampling and intensity quantisation.
- I can explain why direct decimation can turn alternating stripes into a false constant image.
- I can state which transformations must be shared between training and serving.
