---
id: cv-colour-histograms-and-filtering
title: "Computer Vision · Session 5; Colour and Intensity Processing"
sidebar_label: "3 · Colour and filtering"
sidebar_position: 3
slug: /theory/cv/colour-histograms-and-filtering
description: "Work through histograms, equalisation, gamma, smoothing and RGB to HSV conversion with tested examples."
tags: [computer-vision, colour, histograms, filtering]
---

import Infographic from '@site/src/components/Infographic';
import GammaTransformLab from '@site/src/components/viz/GammaTransformLab';

**In one line.** Intensity transforms, neighbourhood filters and colour spaces change different parts of an image representation, so choose them against the visual task.

:::tip Before you start

**You should already know:**

- That a colour image holds three channels per pixel, and that 8-bit values run from 0 to 255. See [digital images](/docs/theory/cv/digital-image-formation-and-sampling).
- What a mean and a median of a list of numbers are.

**Reading time:** about 30 minutes.

**After this chapter you can:**

- compute a histogram-equalisation mapping and a median filter by hand;
- choose between RGB, HSV and Lab for a colour task and name what each one does not protect against;
- pick a filter size from the noise density, and say what a median destroys.

:::

## In 30 seconds

Lighting changes the numbers in a photo without changing the objects. A colour space is a different set of coordinates for the same colour, and one of them (HSV) keeps brightness apart from colour. A speck of pure black or white noise is different: it is a wrong value, not a slightly shifted one. Averaging smears a wrong value into its neighbours. Taking the middle value of a small window, the median, throws it away. Think of a queue sorted by height: one giant does not move the person in the middle, but he does change the average.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Histogram | Count of pixels at each intensity | 12 pixels at level 3 |
| CDF | Running total of the normalised histogram | 0.45 of pixels at level 3 or below |
| Gamma | A power curve that brightens or darkens mid-tones | 0.25 to the power 0.5 is 0.5 |
| HSV | Hue (colour family), saturation (purity), value (brightness) | Pure red is (0, 255, 255) in OpenCV |
| Lab | A colour space where L is lightness and a, b carry colour | Grey has a and b near the middle |
| Median filter | Replace each pixel by the middle value in its window | Window 0, 11, 12, 12, 13, 14, 14, 15, 255 gives 13 |
| Salt-and-pepper noise | Random pixels set to pure black or white | 5% of pixels hit |
| SSIM | Structural similarity, 1 means identical | 0.864 after a good filter |


## The idea in plain words

:::note Additions to the course material

The code, design checks and current OpenCV workflow extend the course material. Its histogram, transform, smoothing and colour-space sequence and all five practice questions are retained.

:::

An image can have the right objects and still be hard for an algorithm to read. A camera may produce a narrow range of grey values, uneven lighting or isolated bright noise. Session 5 treats four ways to work on the pixels before a later feature extractor or model: describe intensities with a histogram, remap intensities, smooth neighbourhood noise, and represent colour in a useful coordinate system. These operations are related but do not solve the same problem. A histogram summarises values without remembering where they occur. A point transform changes each value without consulting neighbours. A spatial filter uses nearby pixels. A colour conversion changes how three-channel values are described.

For a grayscale image with $M N$ pixels, let $h(r_k)$ count pixels at level $r_k$. The normalised histogram $p(r_k)=h(r_k)/(MN)$ gives the fraction of pixels at that level. Summing $p$ from the minimum through a level gives its cumulative distribution function, or CDF. A global histogram can reveal clipping, narrow contrast or a bimodal foreground-background pattern, but it cannot tell whether bright pixels belong to a useful object or glare. Two images can have identical histograms and entirely different spatial layouts. Inspect the image and downstream task alongside the distribution.

Histogram equalisation uses the CDF as a monotone mapping. For $L$ possible output levels, the usual form is $s=\operatorname{round}((L-1)\operatorname{CDF}(r))$. At $L=8$ and CDF at level 3 equal to 0.45, this gives $\operatorname{round}(3.15)=3$. The formula spreads occupied levels in a particular way, but it can also amplify background noise or alter appearance across an image. Local contrast-limited methods address some global limitations by mapping neighbourhoods and clipping large histogram peaks. They still need task-specific evaluation, especially when colour or measured intensity is scientifically meaningful.

Point transforms are direct maps from an input value $r$ to output $s$. A negative reverses ordered intensities: $s=L-1-r$. A logarithm expands relative differences among low input values and compresses high ones after a scale factor. A gamma map on normalised values is $s=r^\gamma$. When $0<\gamma<1$, intermediate values become brighter; when $\gamma>1$, they become darker. For $r=0.25$ and $\gamma=0.5$, the output is 0.5. This depends on the data being in the stated normalised domain. Applying the same exponent directly to 8-bit integers without normalising is a different operation and may overflow or clip.

Smoothing is spatial. A mean filter averages a window, a Gaussian filter weights nearby positions more heavily, and a median filter replaces the centre with an ordered middle value. Median filtering can remove isolated salt-and-pepper impulses without averaging the impulse into neighbouring pixels. None is an automatic improvement. A filter that removes small noise can also erase a small crack or thin vessel. Choose the filter according to the expected noise and the signal that the next stage needs.

<Infographic src="/img/cv/colour-processing.svg" alt="A board compares histogram and gamma intensity maps, mean Gaussian and median neighbourhood filters, and RGB versus HSV colour representation." caption="Changing intensity, removing noise and representing colour address different error sources." />

## Worked example, step by step

**Median against mean.** A 3 by 3 window of a smooth region holds the values 12, 255, 14, 13, 11, 0, 15, 14, 12. Two of them are impulses, 255 and 0.

1. Sort: 0, 11, 12, 12, 13, 14, 14, 15, 255.
2. The median is the fifth value, 13, which is a normal grey for this region.
3. The mean is 346 / 9 = 38.4, which is far from every normal pixel. The impulses have been smeared into the answer.

**Equalisation.** An 8-level image has fractions of pixels per level of 0.05, 0.10, 0.12, 0.18, 0.20, 0.15, 0.12 and 0.08.

1. The running totals give the CDF: 0.05, 0.15, 0.27, 0.45, 0.65, 0.80, 0.92, 1.00.
2. Multiply by 7 and round: 0.35 becomes 0, 1.05 becomes 1, 1.89 becomes 2, 3.15 becomes 3, 4.55 becomes 5, 5.60 becomes 6, 6.44 becomes 6 and 7.00 becomes 7.
3. The new levels are 0, 1, 2, 3, 5, 6, 6, 7. Levels 5 and 6 of the input already sit close together and end up sharing one output level.

**When does a median fail?** It returns an impulse only when most of the window is corrupted. At 40% noise the chance that more than half of a 3 by 3 window is hit is the binomial sum 0.267, and for a 5 by 5 window it is 0.154. The first block below reproduces all of these.

## How it works

### Histogram & equalization

The normalized histogram p(r)=h/(MN) is the intensity distribution. Equalization maps through the CDF s=(L−1)·CDF(r) to spread it and boost contrast.

:::tip

**Worked.** L=8, CDF(3)=0.45 → s=round(7×0.45)=round(3.15)=3.

:::

### Intensity transformations

- **Negative**; s = L−1−r (invert; e.g. mammograms).
- **Log**; s = c·log(1+r): expand dark, compress bright.
- **Gamma**; s = c·r^γ: γ&lt;1 brightens, γ>1 darkens.

### Smoothing & colour spaces

- **Smoothing filters**; Mean (blurs), Gaussian (distance-weighted), median (non-linear; kills salt-and-pepper, keeps edges).
- **RGB vs HSV**; RGB mixes colour+brightness; HSV separates hue, saturation, value; better for colour thresholding under changing light.

### Key takeaways

- **1 · Histogram**; p=h/(MN); equalize via CDF.
- **2 · Transforms**; Negative, log, gamma.
- **3 · Smooth & colour**; Mean/Gaussian/median; HSV.

## A real system that works this way

OpenCV's image-processing tutorials implement both global histogram equalisation and contrast-limited adaptive histogram equalisation. The official tutorial explains why a global mapping can be unhelpful when one part of an image is dark and another bright: one mapping cannot simultaneously place both local regions optimally. Its colour-space tutorial shows conversion between BGR and HSV for colour-based object finding. These are concrete software operations, not evidence that equalisation or HSV will improve every model. Their effect depends on the image pipeline and the task metric.

A practical example is finding an orange safety marker in a controlled camera view. A hue range in HSV can be easier to express than separate B, G and R inequalities, because hue roughly corresponds to colour family while the value channel reflects brightness. Yet hue is unstable when saturation is low; shadows and specular highlights can shift measured colour, and an orange-looking object can be clipped by exposure. A threshold needs representative lighting tests, a saturation and brightness policy, and perhaps spatial checks. OpenCV's default colour input order is BGR, so converting with the wrong constant can produce plausible-looking but incorrect hue values.

An inspection pipeline might smooth isolated sensor noise before colour thresholding. Median filtering can remove single-pixel impulses, but a three-pixel filter can also destroy a tiny defect. The correct sequence is an empirical design decision: take a representative validation set, compare masks before and after filtering, and measure downstream misses and false alarms. A pleasing preview is not the same as a better detector. Preserve the original pixel data for audit where the data policy allows it, and record each transform's parameters with the model version.

## Code you can run

The first block checks the two numerical examples printed on the board. The eight-level CDF mapping returns **3**, while gamma 0.5 maps normalised 0.25 to **0.5**. The histogram array is illustrative; CDF 0.45 is supplied as the exercise premise rather than estimated from a fabricated image.

```python
from math import isclose

levels = 8
cdf_at_three = 0.45
equalised_level = round((levels - 1) * cdf_at_three)
input_intensity = 0.25
gamma = 0.5
gamma_output = input_intensity ** gamma
print('Equalised level:', equalised_level)
print('Gamma output:', gamma_output)
assert equalised_level == 3
assert isclose(gamma_output, 0.5)
```

Move the two sliders in the lab. Its default **0.5 output** matches the gamma calculation. The sampled curve in its data view exposes the full mapping rather than only one point.

<GammaTransformLab />

The second block runs a tiny median-filter and colour-conversion example in the local OpenCV CPU package. A lone white impulse in a black 5 by 5 array disappears under a 3 by 3 median filter. A pure red RGB pixel becomes a high-saturation red HSV pixel after explicitly choosing `COLOR_RGB2HSV`; no file or display is required.

```python
import cv2
import numpy as np


impulse = np.zeros((5, 5), dtype=np.uint8)
impulse[2, 2] = 255
smoothed = cv2.medianBlur(impulse, 3)
red_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
red_hsv = cv2.cvtColor(red_rgb, cv2.COLOR_RGB2HSV)
print('Impulse centre before and after:', int(impulse[2, 2]), int(smoothed[2, 2]))
print('Red in OpenCV HSV:', red_hsv[0, 0].tolist())
assert int(smoothed[2, 2]) == 0
assert red_hsv[0, 0].tolist() == [0, 255, 255]
```

OpenCV's 8-bit HSV encoding uses bounded integer channels, so `red_hsv` here is an implementation value, not an angle in physical units. If a pipeline uses a floating-point image, verify its documented range and conversion behaviour separately. A hue threshold should be tested on real lighting and camera examples rather than inferred from this single pure-colour pixel.

### The worked numbers in code

```python
from math import comb

import numpy as np

window = np.array([12, 255, 14, 13, 11, 0, 15, 14, 12])
print("sorted window:", np.sort(window).tolist())
print("median:", int(np.median(window)), " mean:", round(float(window.mean()), 1))
cdf = np.cumsum(np.array([0.05, 0.10, 0.12, 0.18, 0.20, 0.15, 0.12, 0.08]))
print("CDF:", np.round(cdf, 2).tolist())
print("equalised levels:", np.round(7 * cdf).astype(int).tolist())
for size in (3, 5):
    n = size * size
    majority = sum(comb(n, k) * 0.4 ** k * 0.6 ** (n - k) for k in range(n // 2 + 1, n + 1))
    print(f"windows of {size}x{size} with a majority of impulses at 40% noise:", round(majority, 3))
```

**Reading the output.** The median is 13 and the mean 38.4. The CDF and the equalised levels match the hand calculation. At 40% noise, 26.7% of 3 by 3 windows and 15.4% of 5 by 5 windows are majority impulses.

### Experiment 1: which colour space survives which lighting change?

The task is to name a pixel's colour among seven classes (red, orange, yellow, green, blue, purple and grey). Training pixels come from mild lighting. Test pixels come from four conditions: the same light, a shadow (brightness scaled to 0.4 to 0.6), a warm lamp (red times 1.15, blue times 0.8) and a strong tint (red times 1.35, blue times 0.6). A 5-nearest-neighbour classifier is trained once per colour space. In the second set of rows it is trained on brightness from 0.35 to 1.0, which includes shadows.

```python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

BASE = np.array([[200, 40, 40], [220, 130, 30], [210, 200, 40], [40, 160, 60], [40, 80, 200], [130, 50, 170], [120, 120, 120]], dtype=np.float64)

def pixels(n, seed, gain, tint=(1, 1, 1)):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(BASE), n)
    light = rng.uniform(*gain, n)[:, None] * np.array(tint)
    rgb = (BASE[labels] * light + rng.normal(0, 6, (n, 3))).clip(0, 255).astype(np.uint8)
    return rgb.reshape(-1, 1, 3), labels

def features(rgb, space):
    if space == "RGB":
        return rgb.reshape(-1, 3) / 255.0
    if space.startswith("HSV"):
        h, s, v = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).T.astype(float)
        hue = np.stack([np.cos(h * np.pi / 90), np.sin(h * np.pi / 90), s / 255], 1)
        return hue if space == "HSV hue+sat" else np.column_stack([hue, v / 255])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float) / 255
    return lab if space == "Lab" else lab[:, 1:]

narrow, wide = pixels(4000, 0, (0.85, 1.0)), pixels(4000, 5, (0.35, 1.0))
tests = {
    "same light": pixels(3000, 1, (0.85, 1.0)),
    "shadow": pixels(3000, 2, (0.4, 0.6)),
    "warm lamp": pixels(3000, 3, (0.85, 1.0), (1.15, 1.0, 0.8)),
    "strong tint": pixels(3000, 4, (0.85, 1.0), (1.35, 1.0, 0.6)),
}
print(f"{'space':28s}" + "".join(f"{k:>13s}" for k in tests))
for label, train, spaces in (("narrow light", narrow, ("RGB", "HSV", "HSV hue+sat", "Lab", "Lab a+b")), ("wide light", wide, ("RGB", "HSV hue+sat", "Lab"))):
    for space in spaces:
        knn = KNeighborsClassifier(5).fit(features(train[0], space), train[1])
        print(f"{space + ', ' + label:28s}" + "".join(f"{(knn.predict(features(x, space)) == y).mean():13.3f}" for x, y in tests.values()))
```

**Reading the output.** On this machine:

| Features | same light | shadow | warm lamp | strong tint |
| --- | ---: | ---: | ---: | ---: |
| RGB | 1.000 | 0.676 | 1.000 | 0.713 |
| HSV, all four numbers | 1.000 | 0.995 | 0.995 | 0.589 |
| HSV, hue and saturation | 1.000 | 0.997 | 0.993 | 0.573 |
| Lab | 1.000 | 0.659 | 1.000 | 0.724 |
| Lab, a and b only | 1.000 | 0.884 | 0.999 | 0.581 |
| RGB, trained with shadows | 1.000 | 0.998 | 1.000 | 0.703 |
| Lab, trained with shadows | 1.000 | 0.998 | 1.000 | 0.620 |

**Line by line.**

- Hue is an angle, so `features` encodes it as a cosine and a sine. A plain hue number would put red at 0 and at 179 on opposite ends of the scale.
- OpenCV stores 8-bit hue in 0 to 179, which is why the angle is `h * np.pi / 90`.
- `lab[:, 1:]` drops lightness and keeps the two colour axes.
- `KNeighborsClassifier(5)` is used because it makes no assumption about the shape of a class, so differences come from the features.

**What the numbers say.** Shadow is the case HSV exists for: RGB drops from 1.000 to 0.676 and hue-plus-saturation holds at 0.997. Dropping lightness from Lab also helps (0.659 to 0.884) but not fully, because the colour axes shrink toward grey as light falls.

The result that contradicts the usual advice is the strong tint. HSV is the worst space there (0.589 and 0.573), against 0.713 for RGB and 0.724 for Lab. A colour cast changes the ratios between channels, so hue moves, which is the thing HSV was relied on to keep still. The warm lamp, a milder cast, harmed no space.

Training data matters as much as the space. RGB trained on shadows reaches 0.998 on shadow, as good as HSV, with no change of features. No training on shadows helps with the strong tint (RGB 0.703), because the tint was never seen.

Limits: seven synthetic colours with well-separated centres, single pixels with no neighbours, a brightness scale that stands in for shading, one seed, 3,000 test pixels per column. Real shadows are also bluer than the surrounding light.

:::note Refinement of the usual rule

HSV separates brightness from colour, so it helps against shading. It does not help against a change in the colour of the light. Validate a hue threshold under the casts your cameras will see.

:::

<Infographic src="/img/cv-enrich/v1-colour-space-shift.svg" alt="A grid of accuracy for RGB, HSV and Lab features against same light, shadow, warm lamp and strong tint, showing HSV fixes shadow but is worst under a strong tint." caption="Read the shadow and strong-tint columns together: the spaces that win the first lose the second." />

### Experiment 2: mean, Gaussian or median for salt-and-pepper noise?

Salt-and-pepper noise at 5%, 20% and 40% of the pixels is added to the `camera` photograph (public domain). Four filters are scored by PSNR and SSIM against the clean image, and also on the clean image alone, to see what each filter costs when there is nothing to remove.

```python
import cv2
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def salt_pepper(image, density, seed=0):
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    hit = rng.random(image.shape) < density
    noisy[hit] = rng.choice([0, 255], hit.sum())
    return noisy

filters = {
    "mean 3x3": lambda a: cv2.blur(a, (3, 3)),
    "gaussian 5x5": lambda a: cv2.GaussianBlur(a, (5, 5), 0),
    "median 3x3": lambda a: cv2.medianBlur(a, 3),
    "median 5x5": lambda a: cv2.medianBlur(a, 5),
}
clean = data.camera()
print(f"{'PSNR dB / SSIM':16s}{'no noise':>14s}" + "".join(f"{int(d * 100):>13d}%" for d in (0.05, 0.2, 0.4)))
print(f"{'unfiltered':16s}{'':>14s}" + "".join(f"{psnr(clean, salt_pepper(clean, d), data_range=255):7.2f}/{ssim(clean, salt_pepper(clean, d), data_range=255):.3f}" for d in (0.05, 0.2, 0.4)))
for name, fn in filters.items():
    cells = [f"{psnr(clean, fn(clean), data_range=255):7.2f}/{ssim(clean, fn(clean), data_range=255):.3f}"]
    for density in (0.05, 0.2, 0.4):
        out = fn(salt_pepper(clean, density))
        cells.append(f"{psnr(clean, out, data_range=255):7.2f}/{ssim(clean, out, data_range=255):.3f}")
    print(f"{name:16s}" + "  ".join(f"{c:>12s}" for c in cells))
line = np.zeros((64, 64), np.uint8)
line[:, 32] = 255
print("one-pixel line, brightest value left by median 3x3:", cv2.medianBlur(line, 3).max(), "by mean 3x3:", cv2.blur(line, (3, 3)).max())
```

**Reading the output.**

| PSNR in dB | no noise | 5% | 20% | 40% |
| --- | ---: | ---: | ---: | ---: |
| unfiltered | not computed | 17.80 | 11.75 | 8.76 |
| mean 3x3 | 29.44 | 24.89 | 19.34 | 15.54 |
| gaussian 5x5 | 29.33 | 25.69 | 20.25 | 16.26 |
| median 3x3 | 30.56 | 30.12 | 27.11 | 18.26 |
| median 5x5 | 28.01 | 27.85 | 27.25 | 25.32 |

**Line by line.**

- `rng.choice([0, 255], hit.sum())` picks salt or pepper for each corrupted pixel, so about half are white.
- The first print row calls `salt_pepper` twice with the same seed, so PSNR and SSIM score the same noisy image.
- The no-noise cells of the filter rows score a filtered clean image: the price of the filter itself.

**What the numbers say.** At 5% noise the 3 by 3 median gains 5.2 dB over the mean (30.12 against 24.89) and its SSIM stays at 0.864. The mean and the Gaussian never recover, because each 0 or 255 is averaged into its neighbours instead of removed.

Two things surprise. On the clean photograph, the median 3 by 3 costs less than the mean (30.56 against 29.44 dB), so for this photograph it is the gentler filter, even though it is nonlinear. And at 40% the median 3 by 3 collapses to 18.26 while the 5 by 5 holds 25.32. At 5% the order flips: the 5 by 5 is 2.3 dB worse than the 3 by 3 (27.85 against 30.12). The binomial numbers from the first block explain the collapse: at 40% noise, 26.7% of 3 by 3 windows have a majority of impulses, against 15.4% of 5 by 5 windows.

The cost of the median is thin structure. A one-pixel bright line vanishes completely (brightest value 0 after the median), while the mean leaves a faint trace of 85.

Limits: one photograph, one noise seed, impulses that are independent and uniform, and PSNR and SSIM as the only measures. A detector or classifier downstream may rank the filters differently.

<Infographic src="/img/cv-enrich/v1-salt-pepper-filters.svg" alt="Bars of PSNR for four filters at three noise levels: the 3 by 3 median is best at 5 percent, the 5 by 5 median is best at 40 percent, and the median erases a one-pixel line." caption="Compare the 40% group: the 3 by 3 median is at 18.26 while the 5 by 5 median keeps 25.32." />

## Designing with it

Decide what must be preserved before choosing a transform. If the target is a tiny bright speck, a median filter may erase it. If the target is an object's silhouette, mild smoothing can suppress noise but may move its boundary. If the target is a calibrated intensity measurement, histogram equalisation changes the scale and may invalidate a physical interpretation. A cosmetic enhancement pipeline and a measurement pipeline should have different acceptance tests.

For histograms, inspect both the distribution and spatial examples. A dark image may have useful detail, while a bright image may be saturated beyond recovery. Global equalisation can over-amplify noise in a nearly uniform region. Local equalisation can create artificial contrast around small structures. Compare original and processed image patches and evaluate the downstream classifier or segmentation mask. A visual judgement of contrast alone cannot establish a task improvement. Document the transform's parameter values and whether they were selected on training or validation data.

For gamma, specify the signal domain. Image files often contain gamma-encoded RGB values rather than linear light. Applying another power transform to those values changes display appearance but is not the same as modifying radiance. If the algorithm uses colour differences or physical intensity, a proper decode may be required first. Avoid chaining conversions without recording them. A repeated normalisation or colour conversion can produce a systematic error that looks like a model problem because shapes and dimensions remain valid.

For smoothing, match filter behaviour to the noise. A mean or Gaussian kernel can reduce approximately distributed local noise but blurs high-frequency detail. A median is useful for isolated extreme pixels; it is not a universal denoiser. Inspect how a candidate filter changes edge contrast, small-object recall and mask boundaries. Kernel size should reflect object scale in pixels. A fixed 5 by 5 kernel has very different effects on a 32-pixel crop and a 4,000-pixel image.

For colour, remember that HSV does not make hue invariant to every illumination change. Hue becomes unstable near grey, black or saturated white, and camera white balance can move the colour cluster. Build thresholds with a representative sample and define the handling of low-saturation pixels. When a camera or codec changes, recheck the distribution. RGB may be better for a learned model that has been trained with consistent colour augmentation; HSV can still be useful for a transparent rule or diagnostic plot. The representation should be chosen for the task, not because one space is generically superior.

## Where this stands in 2026

:::info Industry view

- Classical image processing remains useful for controlled acquisition, measurement, debugging and preprocessing. Learned visual systems can absorb some variation, but their input transforms still need a documented contract.
- OpenCV tutorials checked on 2026-10-02 show version 4.13.0. The two local blocks were run using `opencv-python-headless` 5.0.0.93; conversion constants and outputs are reported from that run.
- No accuracy improvement is claimed from histogram equalisation, median filtering or HSV. Any improvement would require a dataset, task metric and held-out evaluation.

:::

## Common mistakes

1. **Assuming HSV is immune to lighting.** It feels right because brightness has its own channel. A colour cast still moves hue, and a strong tint made HSV the worst space (0.589). Test the casts your cameras produce.
2. **Thresholding hue at low saturation.** Near grey, hue is noise. Keep a saturation and value floor and include grey among the classes you test.
3. **Using a mean or Gaussian to remove impulses.** The wrong values are averaged into their neighbours (24.89 dB against 30.12 for the median at 5%). Match the filter to the noise type.
4. **Choosing the median window by habit.** A 3 by 3 window fails at 40% noise and a 5 by 5 window costs 2.3 dB at 5%. Estimate the noise density first.
5. **Running a median over thin structures.** A one-pixel crack disappeared completely. Check small targets before and after any filter.

## Practice questions

<details>
<summary><strong>Q1.</strong> What is a normalized histogram and why use it?</summary>

p(r_k) = h(r_k)/(MN): the fraction of pixels at each intensity; the intensity probability distribution. It is resolution-independent and reveals brightness/contrast.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Give the negative, log and gamma intensity transforms.</summary>

Negative s=L−1−r; log s=c·log(1+r) (expand dark); gamma s=c·r^γ (γ&lt;1 brighten, γ>1 darken).<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> For an 8-level image, the CDF at level 3 is 0.45. Give the equalized output.</summary>

s = round((L−1)·CDF) = round(7×0.45) = round(3.15) = 3.<br /><em>Session 5 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Contrast mean, Gaussian and median smoothing.</summary>

Mean averages a window (blurs edges); Gaussian weights by distance (smoother, better structure); median is non-linear, removing salt-and-pepper noise while preserving edges.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why is HSV often better than RGB for colour tasks?</summary>

HSV separates hue (colour) from saturation and value (brightness), so you can threshold a colour by hue robustly to illumination; RGB entangles colour and brightness.<br /><em>Session 5 · conceptual</em>

</details>

<details>
<summary><strong>Q6 (Easy).</strong> In the worked equalisation, what output level does input level 4 receive?</summary>

The CDF at level 4 is 0.65, and 7 × 0.65 = 4.55 rounds to 5.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> How many pixels of a 3 by 3 window can be impulses before a median is guaranteed to return an impulse?</summary>

The median is the fifth of nine sorted values. Four high impulses leave a normal value in fifth place. Five impulses of the same polarity put an impulse there. With mixed black and white impulses the answer depends on the split, but a majority is the danger sign, which is why 26.7% of windows at 40% noise matter.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> A model reads orange safety vests by an HSV hue threshold. Under a new warm lamp it still works; under a coloured stage light it fails. Explain both with this chapter's results and propose a fix.</summary>

A mild cast changes channel ratios little, so hue stays inside the threshold (warm lamp: 0.993). A strong cast moves hue out of it (strong tint: 0.573). Fixes are to include coloured-light examples in the validation set, to add white-balance correction before the threshold, or to train a classifier on data from those lights. Choosing a different colour space alone does not fix it: Lab scored 0.620 to 0.724.

</details>

## Further reading

- [OpenCV histogram equalisation tutorial](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) for global and local contrast mapping.
- [OpenCV colour-space tutorial](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) for BGR and HSV conversions.
- [scikit-image data module](https://scikit-image.org/docs/stable/api/skimage.data.html): the `camera` image is CC0 (photographer Lav Varshney); the licence statement was read from the installed package docstring, version 0.26.0, on 2026-10-09.
- Versions run: OpenCV 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.1, NumPy 2.5.3.
- The OpenCV tutorial pages named above could not be opened from the build environment on 2026-10-09 (HTTP 403), so the 0 to 179 hue range was checked by converting a grid of 140,608 RGB colours with `cv2.cvtColor` (smallest hue 0, largest 179), not read from the manual.
- Built from the course lecture "cv-s5-color" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## An end-to-end decision example

Suppose an inspection team wants to locate pale scratches on painted panels. The data include panels of several colours, images taken under two lamps and a small class of false alarms caused by reflections. Start by collecting a few examples of each condition and measuring scratch width in pixels. A histogram of the full image may mostly describe paint colour rather than the faint scratch. Equalising every image globally could make scratches more visible in some cases and amplify reflective noise in others. The first useful experiment is therefore a held-out comparison of scratch recall and false alarms with and without the transform, stratified by panel colour and lamp.

A gamma change can lift mid-dark values, but it also lifts dark noise. The lab shows this mechanically for one input intensity, not as a performance prediction. If the scratches are only a few pixels wide, smoothing may blur them away. A median filter could remove isolated hot pixels while preserving a longer line, but kernel size and scratch scale determine that. If the target is a colour marker rather than a texture defect, HSV hue may produce a clearer threshold; low saturation and glare must still be excluded. Each candidate step has a measurable reason and a measurable failure mode.

Keep the data pipeline reversible enough to diagnose a bad decision. Record the original input, chosen colour order, numeric range, histogram or gamma parameters, filter type and kernel size, and any resize. When an error is found, compare the same pixel location before and after each step. A model may be correct given its transformed input while the transform has already deleted the evidence. Conversely, a visible scratch may survive every transform but be missed by a later classifier. The debugging target follows from where the signal disappears.

Finally, avoid fitting preprocessing to the test set. If histogram normalisation uses a reference distribution or a threshold is tuned from examples, derive it from training data and select settings on validation data. Hold the final test set for an honest assessment of generalisation. This discipline is the same as in tabular ML, but the visual nature of images can tempt teams to tweak until a small collection “looks right”. A strong result should survive the intended camera, light and material variation.

## Check yourself

- I can compute a normalised histogram, CDF map and gamma output from explicit inputs.
- I can explain why a histogram lacks spatial information and why equalisation can amplify noise.
- I can choose a mean, Gaussian or median filter against a stated noise and object scale.
- I can explain why HSV hue thresholds still need lighting and saturation checks.
- I can compute a median and a mean for a window with impulses and say which one to trust.
- I can say which lighting change an HSV threshold survives and which it does not.
- I can choose between a 3 by 3 and a 5 by 5 median from the noise density.
