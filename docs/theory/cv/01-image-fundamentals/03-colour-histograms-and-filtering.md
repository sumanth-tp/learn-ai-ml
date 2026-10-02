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

## The idea in plain words

:::note Beyond the lecture

The code, design checks and current OpenCV workflow extend the lecture. Its histogram, transform, smoothing and colour-space sequence and all five practice questions are retained.

:::

An image can have the right objects and still be hard for an algorithm to read. A camera may produce a narrow range of grey values, uneven lighting or isolated bright noise. Session 5 treats four ways to work on the pixels before a later feature extractor or model: describe intensities with a histogram, remap intensities, smooth neighbourhood noise, and represent colour in a useful coordinate system. These operations are related but do not solve the same problem. A histogram summarises values without remembering where they occur. A point transform changes each value without consulting neighbours. A spatial filter uses nearby pixels. A colour conversion changes how three-channel values are described.

For a grayscale image with $M N$ pixels, let $h(r_k)$ count pixels at level $r_k$. The normalised histogram $p(r_k)=h(r_k)/(MN)$ gives the fraction of pixels at that level. Summing $p$ from the minimum through a level gives its cumulative distribution function, or CDF. A global histogram can reveal clipping, narrow contrast or a bimodal foreground-background pattern, but it cannot tell whether bright pixels belong to a useful object or glare. Two images can have identical histograms and entirely different spatial layouts. Inspect the image and downstream task alongside the distribution.

Histogram equalisation uses the CDF as a monotone mapping. For $L$ possible output levels, the lecture writes $s=\operatorname{round}((L-1)\operatorname{CDF}(r))$. At $L=8$ and CDF at level 3 equal to 0.45, this gives $\operatorname{round}(3.15)=3$. The formula spreads occupied levels in a particular way, but it can also amplify background noise or alter appearance across an image. Local contrast-limited methods address some global limitations by mapping neighbourhoods and clipping large histogram peaks. They still need task-specific evaluation, especially when colour or measured intensity is scientifically meaningful.

Point transforms are direct maps from an input value $r$ to output $s$. A negative reverses ordered intensities: $s=L-1-r$. A logarithm expands relative differences among low input values and compresses high ones after a scale factor. A gamma map on normalised values is $s=r^\gamma$. When $0<\gamma<1$, intermediate values become brighter; when $\gamma>1$, they become darker. For $r=0.25$ and $\gamma=0.5$, the output is 0.5. This depends on the data being in the stated normalised domain. Applying the same exponent directly to 8-bit integers without normalising is a different operation and may overflow or clip.

Smoothing is spatial. A mean filter averages a window, a Gaussian filter weights nearby positions more heavily, and a median filter replaces the centre with an ordered middle value. Median filtering can remove isolated salt-and-pepper impulses without averaging the impulse into neighbouring pixels. None is an automatic improvement. A filter that removes small noise can also erase a small crack or thin vessel. Choose the filter according to the expected noise and the signal that the next stage needs.

<Infographic src="/img/cv/colour-processing.svg" alt="A board compares histogram and gamma intensity maps, mean Gaussian and median neighbourhood filters, and RGB versus HSV colour representation." caption="Changing intensity, removing noise and representing colour address different error sources." />

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

The first block checks the two numerical examples printed on the board. The lecture's eight-level CDF mapping returns **3**, while gamma 0.5 maps normalised 0.25 to **0.5**. The histogram array is illustrative; CDF 0.45 is supplied as the exercise premise rather than estimated from a fabricated image.

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

## Further reading

- [OpenCV histogram equalisation tutorial](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) for global and local contrast mapping.
- [OpenCV colour-space tutorial](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) for BGR and HSV conversions.
- Built from the course lecture "cv-s5-color" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
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
