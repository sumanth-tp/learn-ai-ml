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

:::tip Before you start

**You should already know:**

- What a pixel array is and why a colour image has three channels. [What computer vision asks](/docs/theory/cv/what-computer-vision-is) shows the grid in context.
- What an average and a standard deviation are.

**Reading time:** about 30 minutes.

**After this chapter you can:**

- compute the raw storage of an image and its number of grey levels;
- explain why shrinking an image can invent patterns, and how to prevent it;
- read a PSNR figure and say what it cannot tell you about banding.

:::

## In 30 seconds

A photo is a grid of samples, and each sample is rounded to one of 256 levels. Think of a ceiling fan filmed at 24 frames per second: the blades can seem to turn backwards, because the camera samples too slowly. Images do the same. Stripes finer than the pixel grid come out as coarser stripes that were never there, so you blur before you shrink. The rounding has its own failure: round too coarsely and a smooth sky turns into visible steps.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Sampling | Measuring light at points on a grid | A 512 by 512 grid has 262,144 samples |
| Aliasing | A fine pattern appearing as a coarser, false one | Stripes 0.4 cycles per pixel shown at a lower frequency |
| Nyquist limit | The finest pattern a grid can hold: two samples per cycle | A grid of 128 columns holds at most 64 stripe pairs |
| Low-pass filter | A blur that removes fine detail | A Gaussian with sigma 2.0 |
| Quantisation | Rounding each sample to one of a few levels | 3 bits give 8 levels |
| PSNR | Peak signal-to-noise ratio in decibels (dB): higher means closer to a reference | 40 dB is close, 20 dB is rough |
| Banding | Visible steps in a smooth gradient | A sky in 4 bits |
| Dither | Noise added before rounding to hide banding | Random shifts of up to half a step |


## The idea in plain words

:::note Additions to the course material

The storage-unit clarification, aliasing experiment and acquisition guidance extend the course material. The formation model, worked storage example and questions are retained below.

:::

The camera does not capture a complete 3D world. Light from scene points passes through optics and lands on a finite sensor. Each sensor site measures a response over a small area and exposure interval. An analogue signal is then converted into digital values. By the time an algorithm sees an array, scene geometry, lens behaviour, sampling, quantisation, colour processing and perhaps compression have already shaped the evidence. A model cannot infer details that were never recorded with enough signal.

Write a grayscale image as $f(x,y)$, the recorded intensity at discrete horizontal and vertical coordinates. In code, array indexing is usually row then column, so `image[y, x]` is the value at $(x,y)$. A colour image has an additional channel axis. An RGB array with shape `(height, width, 3)` gives red, green and blue values for each pixel. The number in a pixel is not necessarily physical luminance. It may be gamma encoded, white balanced or transformed by an imaging pipeline. Interpret it according to the representation and data type before applying arithmetic.

The simple formation model is $f(x,y)=i(x,y)r(x,y)$: illumination times surface reflectance. It isolates why brightness alone is ambiguous. If illumination falls while reflectance rises, the product can stay similar. The model is a teaching approximation: real sensors integrate over wavelength and time; there are shadows, specular reflections, camera response curves and noise. Still, the separation is useful. A change in a pixel can come from the object, the light or the camera. A threshold trained in one lighting condition should be tested when those conditions change.

**Sampling** fixes locations on a finite grid. A 512 by 512 image contains 262,144 pixel positions. If a fine stripe pattern oscillates faster than the grid can represent, distinct patterns can produce the same samples. This is aliasing. Blurring or low-pass filtering before reducing resolution removes frequencies the smaller grid cannot hold. Merely choosing an interpolation algorithm after an aliased capture cannot recover the original scene. Spatial resolution describes the number and spacing of samples, not how many intensity levels each sample can take.

**Quantisation** maps continuous signal values to a finite set of numbers. With $b$ bits, a channel has $2^b$ representable codes; eight bits give 256 codes, usually 0 through 255 for unsigned values. Too few levels can create visible steps in a smooth gradient, called banding or false contouring. More bits increase possible levels but cannot remove sensor noise or restore clipped highlights. Effective precision depends on the whole imaging chain, not just the file's declared bit depth.

<Infographic src="/img/cv/digital-image.svg" alt="A scene is projected onto a sensor, sampled as a 512 by 512 grid, quantised to 256 levels per channel at eight bits, and stored as 786432 uncompressed RGB bytes." caption="Spatial samples, intensity levels and stored bytes are related but distinct." />

## Worked example, step by step

**Sampling.** Take eight pixels of alternating black and white stripes: 0, 255, 0, 255, 0, 255, 0, 255.

1. Keep every second sample. All kept samples are 0, so the result is four black pixels. The stripes vanished and a false flat region took their place.
2. Average each pair first: (0 + 255) / 2 = 127.5 four times. The result is a flat mid-grey, which is what a sensor that integrates light over a pixel would record.

**Quantisation.** Round the value 100 to 3 bits.

1. Eight levels spread over 0 to 255 give a step of 255 / 7 = 36.43.
2. 100 / 36.43 = 2.745, which rounds to level 3.
3. Level 3 is 3 × 36.43 = 109.29. The error is 9.29, never more than half a step (18.21).
4. Rounding errors spread evenly across a step have power step squared over 12, which gives PSNR = 10 log10(12 × 7²) = 27.69 dB for 3 bits.

The first block below reproduces all of these numbers.

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

The first block verifies the storage calculation. A tightly packed 512 × 512 RGB array with one byte per channel occupies **786,432 bytes**, or **768 KiB**. The 8-bit channel holds **256 levels**. A compressed image file may be smaller or larger because headers and coding change the stored size; this is a raw-array calculation.

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

The figure 786,432 bytes is often written “768 KB”. That equality uses 1,024-byte units, whose precise symbol is **KiB**. In decimal units the same amount is 786.432 kB. The numeric byte count is correct.

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

### The worked numbers in code

```python
import numpy as np

stripes = np.array([0, 255] * 4, dtype=float)
print("every second sample:", stripes[::2].tolist())
print("average of each pair:", stripes.reshape(-1, 2).mean(axis=1).tolist())
step = 255 / (2 ** 3 - 1)
code = round(100 / step)
print("3-bit step:", round(step, 2), "value 100 becomes", round(code * step, 2), "error", round(abs(100 - code * step), 2))
print("predicted PSNR for 3 bits:", round(10 * np.log10(12 * (2 ** 3 - 1) ** 2), 2))
```

**Reading the output.** The kept samples are all 0.0, the pair averages are all 127.5, the 3-bit value is 109.29 with error 9.29, and the predicted PSNR is 27.69 dB. The same formula is checked against real images in the experiment below.

### Experiment 1: which way of shrinking creates false patterns?

We shrink three 512 by 512 images to 128 by 128: a photograph (`camera`, public domain), a brick texture and a synthetic stripe pattern at 0.4 cycles per pixel, far beyond what 128 columns can hold. The reference is an ideal low-pass filter at the new limit, computed with a Fourier transform and compared at the best half-pixel alignment. Each method gets a PSNR against that reference. The last column is the standard deviation of the shrunk stripe image: a correct result is nearly flat, so a large value is leftover false pattern.

```python
import cv2
import numpy as np
from scipy import ndimage
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

def references(image, factor=4):
    freq = np.fft.fftfreq(image.shape[0])
    keep = (np.abs(freq)[:, None] <= 0.5 / factor) & (np.abs(freq)[None, :] <= 0.5 / factor)
    low = np.real(np.fft.ifft2(np.fft.fft2(image) * keep))
    return [ndimage.shift(low, (-o, -o), order=3, mode="wrap")[::factor, ::factor] for o in np.arange(0, factor, 0.5)]

def score(refs, out):
    return max(psnr(r[8:-8, 8:-8], np.clip(out, 0, 255)[8:-8, 8:-8], data_range=255) for r in refs)

x = np.arange(512)
grating = np.tile(127.5 + 100 * np.sin(2 * np.pi * 0.4 * x), (512, 1)).astype(np.float32)
scenes = {"camera": data.camera().astype(np.float32), "brick": data.brick().astype(np.float32), "grating": grating}
size = (128, 128)
methods = {
    "point sample": lambda s: s[::4, ::4],
    "bilinear": lambda s: cv2.resize(s, size, interpolation=cv2.INTER_LINEAR),
    "bicubic": lambda s: cv2.resize(s, size, interpolation=cv2.INTER_CUBIC),
    "area": lambda s: cv2.resize(s, size, interpolation=cv2.INTER_AREA),
    "gauss 1.0, then point": lambda s: cv2.GaussianBlur(s, (0, 0), 1.0)[::4, ::4],
    "gauss 2.0, then point": lambda s: cv2.GaussianBlur(s, (0, 0), 2.0)[::4, ::4],
}
refs = {name: references(scene) for name, scene in scenes.items()}
print(f"{'PSNR against the ideal':24s}" + "".join(f"{n:>10s}" for n in scenes) + "  grating std")
for name, fn in methods.items():
    row = [score(refs[n], fn(s)) for n, s in scenes.items()]
    print(f"{name:24s}" + "".join(f"{v:10.2f}" for v in row) + f"{fn(grating).std():12.2f}")
```

**Reading the output.** On this machine:

| Method | camera | brick | stripes | stripe std |
| --- | ---: | ---: | ---: | ---: |
| point sample | 26.62 | 29.83 | 11.12 | 70.57 |
| bilinear | 29.69 | 31.27 | 21.39 | 21.96 |
| bicubic | 27.43 | 29.88 | 16.89 | 36.86 |
| area | 34.37 | 33.67 | 23.24 | 17.77 |
| gauss 1.0, then point | 32.85 | 33.48 | 38.40 | 3.48 |
| gauss 2.0, then point | 30.34 | 30.00 | 66.01 | 0.63 |

**Line by line.**

- `references` keeps only frequencies below 0.5 / 4 = 0.125 cycles per pixel, then samples at eight possible half-pixel offsets. Methods that sample cell centres and methods that sample corners are each scored at their own alignment.
- `score` ignores 8 pixels at each border, where the filter wraps around.
- `[::4, ::4]` is point sampling: it keeps one pixel in sixteen and averages nothing.

**What the numbers say.** A sine of amplitude 100 has a standard deviation of 100 / √2 = 70.71. Point sampling returns 70.57, so the false stripes arrive at almost full strength. A correct reduction would leave nearly nothing. Gaussian smoothing with sigma 2.0 leaves 0.63, and sigma 1.0 leaves 3.48.

Two results contradict common advice. First, `INTER_AREA`, the method usually recommended for shrinking, is best on the photographs (34.37 and 33.67 dB) yet still leaves a stripe pattern of 17.77, because averaging over a 4 by 4 box is a weak filter for frequencies far beyond the limit. Second, bicubic scored below bilinear on every column (for instance 27.43 against 29.69 on the photograph), so a "higher-order" interpolation does not mean a better reduction.

The cost of strong smoothing is also visible. Sigma 2.0 is best on the stripes and worse than sigma 1.0 on both photographs (30.34 against 32.85 on the photograph). Choose the blur from the finest detail you must keep.

Limits: three images, one factor of four, one reference filter that rings at sharp edges, so compare columns within the table and do not compare PSNR values with other studies.

<Infographic src="/img/cv-enrich/v1-aliasing-methods.svg" alt="Two bar charts for six ways of shrinking an image: PSNR on a photograph, and the leftover false-stripe standard deviation on a stripe pattern." caption="Compare the right-hand bars first: point sampling keeps 70.57 of false pattern, a Gaussian of sigma 2.0 keeps 0.63." />

### Experiment 2: how many bits before banding, and does PSNR see it?

A photograph and a smooth horizontal ramp are rounded to 7 down to 2 bits. The ramp shows banding cleanly. Besides PSNR, the block measures the error after a Gaussian blur of sigma 6, which keeps the slow errors that make visible steps and removes fine noise. It then repeats the ramp with dither: uniform noise of plus or minus half a step added before rounding.

```python
import cv2
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

def quantise(image, bits, dither=False):
    step = 255 / (2 ** bits - 1)
    if dither:
        image = image + np.random.default_rng(0).uniform(-step / 2, step / 2, image.shape)
    return np.clip(np.round(image / step), 0, 2 ** bits - 1) * step

def soften(image):
    return cv2.GaussianBlur(image, (0, 0), 6)

photo = data.camera().astype(np.float64)
ramp = np.tile(np.linspace(0, 255, 512), (128, 1))
print("bits  levels  photo  ramp  formula  ramp_soft  dithered  dithered_soft")
for bits in (7, 6, 5, 4, 3, 2):
    plain, dithered = quantise(ramp, bits), quantise(ramp, bits, dither=True)
    formula = 10 * np.log10(12 * (2 ** bits - 1) ** 2)
    print(f"{bits:4d}{2 ** bits:8d}{psnr(photo, quantise(photo, bits), data_range=255):8.2f}{psnr(ramp, plain, data_range=255):7.2f}{formula:9.2f}"
          f"{psnr(soften(ramp), soften(plain), data_range=255):11.2f}{psnr(ramp, dithered, data_range=255):10.2f}{psnr(soften(ramp), soften(dithered), data_range=255):14.2f}")
print("distinct grey values in the 3-bit ramp:", np.unique(quantise(ramp, 3)).size)
```

**Reading the output.**

| Bits | photo | ramp | formula | ramp, softened | dithered | dithered, softened |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 52.87 | 52.88 | 52.87 | 65.48 | 49.89 | 75.93 |
| 5 | 40.56 | 40.63 | 40.62 | 60.64 | 37.61 | 63.93 |
| 3 | 27.27 | 27.70 | 27.69 | 30.49 | 24.69 | 50.68 |
| 2 | 19.53 | 20.34 | 20.33 | 21.44 | 17.34 | 43.51 |

The 3-bit ramp contains exactly 8 distinct grey values.

**What the numbers say.** The formula from the worked example predicts the measured ramp PSNR to within 0.01 dB in every row, and the photograph to within 0.1 dB. Each extra bit adds about 6 dB: 27.27, 33.88 and 40.56 dB at 3, 4 and 5 bits.

The surprise is dither. It lowers PSNR by about 3 dB at every depth, because the added noise doubles the error power. Yet at 3 bits the softened error improves from 30.49 to 50.68 dB. The banding steps are slow, structured error that survives smoothing. Dither converts that error into fast, unstructured noise that smoothing, and the eye, average away. PSNR ranked the worse-looking image higher.

Limits: one photograph, one synthetic ramp, one noise seed, and a Gaussian blur as a crude stand-in for the eye. No viewing test was run.

<Infographic src="/img/cv-enrich/v1-quantisation-dither.svg" alt="For 5, 4, 3 and 2 bits, bars show PSNR and smoothed PSNR for a plain and a dithered ramp: dither lowers PSNR by about 3 dB but raises the smoothed score greatly." caption="Compare the lower bars at 3 bits: 30.49 for the plain ramp against 50.68 with dither, although the upper bars say the opposite." />

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

## Common mistakes

1. **Shrinking with point sampling because it is fast.** It keeps one pixel in sixteen and averages nothing, so false patterns survive at full strength (stripe std 70.57). Blur first or use area averaging, and check on a test pattern.
2. **Trusting `INTER_AREA` as full protection.** It was the best photograph method here and still left 17.77 on the stripes. If the input has detail far finer than the output grid, add a Gaussian first.
3. **Judging banding with PSNR.** The dithered ramp scored 3 dB lower and looks smoother. Look at the image, and measure error after smoothing.
4. **Resizing label masks like photographs.** Blending class IDs creates classes that do not exist. Use nearest-neighbour for masks.
5. **Changing the resize rule between training and serving.** The model then sees a different signal. Store the rule with the model.

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

<details>
<summary><strong>Q6 (Easy).</strong> How many bytes does a 1024 by 768 greyscale image occupy if each sample is a 12-bit value stored in a 16-bit word?</summary>

1024 × 768 × 2 bytes = 1,572,864 bytes, which is 1,536 KiB. The word size, not the 12 meaningful bits, sets the array size.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> Use the formula to predict the PSNR of a 6-bit quantised ramp. How does it compare with the printed table?</summary>

10 log10(12 × 63²) = 46.78 dB. The experiment printed 46.78 in the formula column and 46.79 on the ramp, so the prediction holds to 0.01 dB.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> Why can adding noise before quantising make a gradient look better while lowering PSNR?</summary>

Plain rounding makes a repeating, low-frequency error: steps with one fixed level between them. The eye is sensitive to that structure. Dither breaks the repetition into random error. The total error power rises (PSNR drops about 3 dB), but it sits at high frequencies that are blurred away at normal viewing distance. The experiment shows the softened error falling from 30.49 to 50.68 dB at 3 bits.

</details>

## Further reading

- [OpenCV geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) for resize, interpolation and reverse mapping.
- [Szeliski’s textbook site](https://szeliski.org/Book/) for image formation and sensing in the broader vision pipeline.
- [scikit-image data module](https://scikit-image.org/docs/stable/api/skimage.data.html): the licence statements for `camera` (CC0, by the photographer Lav Varshney) and `brick` (CC0, from CC0Textures) were read from the docstrings of the installed package, version 0.26.0, on 2026-10-09.
- Versions run: OpenCV 5.0.0, NumPy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0.
- The OpenCV documentation pages could not be opened from the build environment on 2026-10-09 (HTTP 403), so the resize behaviour above comes from the runs, not from the manual.
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
- I can show by hand that every-second-sample turns black and white stripes into a flat region, and that averaging first does not.
- I can predict the PSNR of a b-bit uniform quantiser from 10 log10(12 (2^b - 1)^2).
- I can explain why dither lowers PSNR and still reduces visible banding.
