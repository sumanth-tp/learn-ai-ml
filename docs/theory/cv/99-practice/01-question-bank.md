---
id: cv-question-bank
title: "Computer Vision; Question Bank"
sidebar_label: "1 · Question bank"
sidebar_position: 1
slug: /theory/cv/question-bank
description: "Thirty-six source questions across vision, with their answers, tested numeric corrections and deployment caveats."
tags: [computer-vision, practice, question-bank]
---

import Infographic from '@site/src/components/Infographic';
import EdgeWeightBudgetLab from '@site/src/components/viz/EdgeWeightBudgetLab';

**In one line.** Work through all 36 unique source questions from image formation to edge deployment, checking the units and evaluation assumptions behind each answer.

## How to use this bank

Attempt each group before opening its answers. The groups follow the lecture sequence. The comprehensive source bank repeats main-bank Q20 to Q36 verbatim, including answers, so each question appears once here. The two sources together contain 53 displayed panels but only 36 unique questions. Every unique question and its source answer is retained; corrections and qualifications follow as labelled notes rather than silently replacing the source.

For a numerical question, write down the input units and denominator first. A 512 by 512 RGB image is 786,432 raw bytes, but that is 768 **KiB**, not 768 decimal KB. A RANSAC confidence formula produces a continuous bound that must be rounded up to a whole trial. A box IoU, mask Dice or average precision has a defined matching policy, not merely an attractive number. Architecture parameters and quantisation bits describe storage, not guaranteed device latency.

<Infographic src="/img/cv/question-bank-map.svg" alt="The thirty-six computer-vision questions are grouped as pixels and features, recognition and regions, then tracking and edge deployment; the last seventeen are repeats in the comprehensive source." caption="The questions progress from measurements to model outputs and deployment decisions." />

## Questions by topic

## Foundations & Image Fundamentals

<details>
<summary><strong>Q1.</strong> What is computer vision, and why is it hard?</summary>

CV builds algorithms that extract meaning from images/video; the inverse of graphics. It is hard because a 2-D image loses depth, and lighting, viewpoint, scale, occlusion and intra-class variation make the same object look very different.

</details>

<details>
<summary><strong>Q2.</strong> How is a digital image represented, and what are the channels of a colour image?</summary>

As a grid of pixels; a grayscale image is one intensity per pixel (0-255), a colour image has three channels (R, G, B). An H×W×3 array holds it.

</details>

<details>
<summary><strong>Q3.</strong> What is the difference between spatial resolution and intensity (bit) depth?</summary>

Spatial resolution is the number of pixels (H×W); detail in space; bit depth is the number of intensity levels per pixel (e.g. 8-bit = 256 levels); detail in brightness.

</details>

<details>
<summary><strong>Q4.</strong> What is a convolution / filter in image processing?</summary>

Sliding a small kernel over the image and computing a weighted sum at each location. It underlies smoothing (blur), sharpening and edge detection; a Gaussian kernel blurs, a derivative kernel finds edges.

</details>

<details>
<summary><strong>Q5.</strong> How many bytes does a 512×512 RGB image at 8 bits/channel take, and how many intensity levels per channel?</summary>

512×512×3×(8/8) = 786,432 bytes = 768 KB; L = 2⁸ = 256 levels per channel. Halving to 4 bits gives only 16 levels (false contouring).

</details>

<details>
<summary><strong>Q6.</strong> Write the image formation model f(x,y)=i(x,y)·r(x,y) and give the range of each term.</summary>

f = illumination i (0 &lt; i &lt; ∞) × reflectance r (0 ≤ r ≤ 1). A pinhole/lens projects the 3-D scene onto the sensor as an inverted image; the eye works the same way (lens → retina).

</details>

## Edges, Features & Colour

<details>
<summary><strong>Q7.</strong> What is an image gradient and how does it relate to edges?</summary>

The gradient (∂I/∂x, ∂I/∂y) measures intensity change; its magnitude is large at edges. Sobel/Prewitt kernels approximate the derivatives, and the gradient direction is perpendicular to the edge.

</details>

<details>
<summary><strong>Q8.</strong> List the steps of the Canny edge detector.</summary>

1) Gaussian smoothing; 2) compute gradient magnitude and direction; 3) non-maximum suppression (thin edges to 1 pixel); 4) double thresholding; 5) hysteresis edge tracking (link weak edges connected to strong ones).

</details>

<details>
<summary><strong>Q9.</strong> What does the Hough transform do, and how does it detect lines?</summary>

It detects parametric shapes by voting in parameter space. Each edge point votes for all lines through it (in (ρ,θ) space); peaks in the accumulator correspond to lines that many points agree on; robust to gaps and noise.

</details>

<details>
<summary><strong>Q10.</strong> Why use the HSV colour space instead of RGB for some tasks?</summary>

HSV separates colour (hue) from intensity (value) and saturation, so hue-based segmentation is more robust to lighting changes than RGB, where brightness is entangled across all three channels.

</details>

<details>
<summary><strong>Q11.</strong> Sobel gives Gx=4, Gy=3 at a pixel. Compute gradient magnitude and orientation.</summary>

Magnitude = √(4²+3²) = √25 = 5; orientation = arctan(3/4) = 36.87° (the edge runs perpendicular to this).

</details>

<details>
<summary><strong>Q12.</strong> An edge point (2,2) votes at θ=45° in the Hough transform. What ρ does it vote for?</summary>

ρ = x cosθ + y sinθ = 2cos45° + 2sin45° = 2(0.7071)+2(0.7071) = 2.828. Polar form is used because y=mx+c can't represent vertical lines.

</details>

<details>
<summary><strong>Q13.</strong> For an 8-level image the CDF at input level 3 is 0.45. Give the histogram-equalized output.</summary>

s = round((L−1)·CDF) = round(7×0.45) = round(3.15) = 3. Applying the CDF at every level stretches the histogram to boost contrast.

</details>

## Local Features: Harris, HoG & SIFT (S6-7)

<details>
<summary><strong>Q14.</strong> Write the Harris corner response and interpret its sign.</summary>

R = det(M) − k·trace(M)² (k≈0.04–0.06), where M is the gradient structure tensor. Large positive R → corner; R &lt; 0 → edge; |R| small → flat.

</details>

<details>
<summary><strong>Q15.</strong> Compute the Harris response for det(M)=1.2, trace(M)=2.0, k=0.05.</summary>

R = 1.2 − 0.05·(2.0)² = 1.2 − 0.2 = 1.0 > 0 → a corner. Harris is rotation-invariant but not scale-invariant.

</details>

<details>
<summary><strong>Q16.</strong> What invariances does SIFT provide, and what is its descriptor dimension?</summary>

SIFT is scale- and rotation-invariant (robust to illumination and small viewpoint change). Its descriptor is 4×4 cells × 8 orientation bins = 128 dimensions, matched with Lowe's ratio test.

</details>

<details>
<summary><strong>Q17.</strong> Outline the HoG descriptor pipeline.</summary>

Compute gradients → per-cell orientation histograms (9 bins × 20°, magnitude-weighted) → group cells into blocks and normalise for illumination → concatenate into one feature vector.

</details>

## Robust Matching: RANSAC (S8)

<details>
<summary><strong>Q18.</strong> Describe the RANSAC loop and why it beats least squares on matches.</summary>

Repeat: pick a minimal random sample, fit the model, count inliers within a threshold, keep the best; then refit on all inliers. Feature matches contain outliers that ruin least squares, but outliers rarely agree on the same wrong model.

</details>

<details>
<summary><strong>Q19.</strong> Compute the RANSAC iteration count for p=0.99, w=0.5, n=4.</summary>

N = log(1−p)/log(1−wⁿ) = log(0.01)/log(1−0.5⁴) = log(0.01)/log(0.9375) ≈ 72. N rises sharply as the inlier fraction falls or the sample size grows.

</details>

## Image Classification & Attention (S9-10)

<details>
<summary><strong>Q20.</strong> What is the semantic gap in image classification?</summary>

The mismatch between raw pixels and semantic meaning: one object varies with viewpoint, illumination, scale, deformation, occlusion and clutter, yet must map to a single label.

</details>

<details>
<summary><strong>Q21.</strong> Contrast nearest-neighbour, linear+softmax, CNN and ViT classifiers.</summary>

k-NN labels by the closest training image (slow, weak on pixels). Linear+softmax learns s=Wx+b (fast, linear only). CNNs learn hierarchical convolutional features. ViTs split the image into patches and self-attend for long-range context.

</details>

<details>
<summary><strong>Q22.</strong> How many patch tokens does a 224×224 image give a ViT with 16×16 patches?</summary>

(224/16)² = 14² = 196 patch tokens (plus a class token).

</details>

<details>
<summary><strong>Q23.</strong> Compute the softmax of class logits [2, 1, 0].</summary>

e²=7.389, e¹=2.718, e⁰=1, sum=11.107 → probabilities [0.665, 0.245, 0.090]; the top class gets 66.5%.

</details>

<details>
<summary><strong>Q24.</strong> A classifier has TP=40, FP=10, FN=20. Compute precision, recall and F1.</summary>

Precision = 40/50 = 0.80; recall = 40/60 = 0.667; F1 = 2(0.8)(0.667)/(0.8+0.667) = 0.727. Accuracy alone misleads under class imbalance.

</details>

## Segmentation & Metrics (S11-12)

<details>
<summary><strong>Q25.</strong> Contrast image segmentation with semantic segmentation.</summary>

Classical segmentation partitions an image into coherent regions (thresholding/Otsu, region growing, clustering, watershed, graph cuts). Semantic segmentation classifies every pixel into a category with encoder–decoder nets (FCN, U-Net, DeepLab).

</details>

<details>
<summary><strong>Q26.</strong> Write IoU and Dice, and compute them for two 16-pixel regions overlapping in 4 pixels.</summary>

IoU = |A∩B|/|A∪B| = 4/(16+16−4) = 0.143; Dice = 2|A∩B|/(|A|+|B|) = 8/32 = 0.25. Dice ≥ IoU always.

</details>

<details>
<summary><strong>Q27.</strong> What do U-Net skip connections and DeepLab atrous convolutions add?</summary>

U-Net skip connections pass encoder detail to the decoder to recover fine boundaries; DeepLab's atrous (dilated) convolutions enlarge the receptive field without losing resolution.

</details>

## Object Detection (S13-14)

<details>
<summary><strong>Q28.</strong> Trace the R-CNN family and contrast it with YOLO.</summary>

R-CNN (region proposals + per-region CNN) → Fast R-CNN (one CNN pass, RoI pooling) → Faster R-CNN (learned Region Proposal Network, end-to-end). YOLO is single-stage; it predicts boxes and classes in one pass over a grid (much faster, real-time).

</details>

<details>
<summary><strong>Q29.</strong> What is Non-Maximum Suppression, and what does mAP measure?</summary>

NMS keeps the highest-confidence box and removes others overlapping it above an IoU threshold, removing duplicates. mAP (mean Average Precision) is the area under the precision–recall curve averaged over classes and IoU thresholds.

</details>

<details>
<summary><strong>Q30.</strong> A predicted and ground-truth box (each 16 px) overlap in 4 px. Is it a hit at IoU 0.5?</summary>

IoU = 4/(16+16−4) = 0.143 &lt; 0.5, so it is a miss at the usual threshold.

</details>

## Tracking, Bag of Words & Edge CV (S14-16)

<details>
<summary><strong>Q31.</strong> What is tracking-by-detection, and the roles of the Kalman filter and Hungarian algorithm?</summary>

Detect objects each frame, then link detections over time. The Kalman filter predicts each track's next box; the Hungarian algorithm matches detections to tracks to maximise total IoU. SORT = Kalman+IoU; DeepSORT adds appearance.

</details>

<details>
<summary><strong>Q32.</strong> Predicted and detected boxes intersect in 30 px with union 70 px. Compute the IoU for association.</summary>

IoU = 30/70 = 0.429; above the ~0.3 threshold it is linked to the track. MOT is scored by MOTA = 1 − (FN+FP+IDSW)/GT.

</details>

<details>
<summary><strong>Q33.</strong> Describe the bag-of-visual-words pipeline and what sets the descriptor dimension.</summary>

Extract SIFT features, cluster with k-means into a visual vocabulary, assign each feature to its nearest word, and build a histogram of word counts. The dimension equals the vocabulary size (e.g. 500 words → 500-dim).

</details>

<details>
<summary><strong>Q34.</strong> For a 3-word codebook with counts [4,1,3], give the normalised (tf) histogram.</summary>

Total 8 → [4/8, 1/8, 3/8] = [0.5, 0.125, 0.375]. Often tf-idf weighted, as in text retrieval.

</details>

<details>
<summary><strong>Q35.</strong> What makes MobileNet efficient, and compute its parameter reduction vs VGG-16 (138M vs 4.2M).</summary>

Depthwise-separable convolutions. 138/4.2 ≈ 33× fewer parameters; feasible on-device.

</details>

<details>
<summary><strong>Q36.</strong> Name the three model-compression techniques and the saving from 8-bit quantisation.</summary>

Pruning (remove redundant weights), quantisation (lower precision), knowledge distillation (small student mimics teacher). 32→8-bit = 4× smaller and faster.

</details>
## Corrections and qualifications

:::note Q5 · Storage units

The source gives 786,432 bytes as “768 KB”. Dividing by 1,024 gives **768 KiB**; dividing by 1,000 gives **786.432 decimal KB**. The byte count and 256 levels per 8-bit channel are correct. The source answer is retained above so the unit discrepancy remains visible.

:::

:::note Q10, Q15 and Q19 · Conditions behind the answer

HSV can make a hue rule less sensitive to some brightness changes, but hue is unstable at low saturation and under colour shifts. A positive Harris response is useful only with a threshold and local comparison; it does not prove a semantic corner by itself. For Q19, the continuous RANSAC result is approximately **71.355**, and the smallest whole number satisfying 99% under the assumed independent sampling model is **72**. The source's rounded answer of about 72 is correct as a whole-trial count.

:::

:::note Q23, Q28, Q29, Q31 and Q32 · Model and evaluation scope

Softmax 0.665 is a relative model probability, not automatically calibrated confidence. YOLO speed depends on variant, input, device and postprocessing, so “real-time” is not universal. AP and mAP need an explicit precision-recall interpolation, class and IoU policy. The Hungarian algorithm optimises the supplied pair-cost matrix, which may include more than IoU. An IoU of 0.429 above a 0.3 gate makes a track-detection pair eligible; competing pairs can still change the final assignment. The source wording is retained in the questions.

:::

:::note Q35 and Q36 · Parameter and quantisation figures

The source's about-33 ratio applies to the original VGG-16 138M and MobileNet V1 1.0-224 4.2M variants. Moving raw stored values from 32 bits to 8 bits gives a fourfold idealised **weight-byte** reduction. The source's “feasible on-device” and “faster” conclusions require an actual device, operator support, memory and latency test; the raw arithmetic alone cannot establish them.

:::

## Code you can run

This block checks two source questions with explicit units and rounding. For Q5 it prints **786432 bytes**, **768.000 KiB** and **786.432 decimal KB**. For Q19 it prints the continuous iteration bound and the required **72 whole trials**.

```python
from math import ceil, log

raw_bytes = 512 * 512 * 3 * (8 // 8)
kibibytes = raw_bytes / 1024
decimal_kilobytes = raw_bytes / 1000
continuous_trials = log(1 - 0.99) / log(1 - 0.5 ** 4)
whole_trials = ceil(continuous_trials)
print('Raw bytes:', raw_bytes)
print(f'Binary KiB: {kibibytes:.3f}')
print(f'Decimal KB: {decimal_kilobytes:.3f}')
print(f'RANSAC continuous: {continuous_trials:.3f}; whole: {whole_trials}')
assert raw_bytes == 786_432
assert (kibibytes, decimal_kilobytes) == (768.0, 786.432)
assert whole_trials == 72
```

The second block checks Q35 and Q36 for the original model variants. It reports the paper's rounded parameter ratio **32.857** and the idealised 32-to-8-bit raw weight factor **4.0**. It does not load either model.

```python
vgg_millions = 138
mobilenet_millions = 4.2
bits_before, bits_after = 32, 8
parameter_ratio = vgg_millions / mobilenet_millions
raw_weight_ratio = bits_before / bits_after
print(f'Paper parameter ratio: {parameter_ratio:.3f}x')
print(f'Idealised raw-weight reduction: {raw_weight_ratio:.1f}x')
assert round(parameter_ratio, 3) == 32.857
assert raw_weight_ratio == 4
```

The lab defaults to MobileNet V1 at eight bits and reproduces those ratios. Its table view shows the model variant and raw decimal megabytes. Change the bit width or select VGG-16 to compare *estimates*, then return to the lecture's questions and decide which quantities need a physical-device measurement.

<EdgeWeightBudgetLab />

## Revision path

Begin with image formation and sampling. Q1 to Q6 ask what a pixel array represents; if you cannot explain why depth is ambiguous or how bytes are counted, later model outputs will be hard to interpret. Q7 to Q13 move through derivatives, edge thinning, line votes and colour. Draw a tiny intensity grid and compute one derivative before memorising algorithm names. Q14 to Q19 distinguish feature detection, description and robust geometric agreement. A feature match is local evidence, while RANSAC tests a global model under an explicit residual threshold.

Q20 to Q24 ask what an image-level class means and how a score becomes a prediction and dataset metric. Keep softmax, precision, recall and F1 separate: they answer different questions. Q25 to Q30 shift the output from one label to pixels and boxes. State whether an overlap is between masks or boxes and whether the objective is a class map, instance count or object location. Q31 to Q34 add time and retrieval: an eligible pair is not a stable identity, and a visual-word histogram is a count representation that discards location.

Finish with Q35 and Q36 as a design exercise. A parameter ratio can guide a shortlist, but the device must run the complete pipeline under its memory, power and deadline budget. Quantisation can change task quality, particularly for rare small targets. A useful answer names the variants, calculates the raw bound and then lists the evidence still needed before deployment. This final step connects the course's image measurements back to an engineering decision.

## Further reading

- Built from the course lectures "cv-question-bank" and "cv-comprehensive-question-bank" (Lecture Library series).
- [Original MobileNets paper](https://arxiv.org/html/1704.04861) for the source model-variant counts.
- [OpenCV documentation](https://docs.opencv.org/4.x/) for edge, feature and image-processing operations.
