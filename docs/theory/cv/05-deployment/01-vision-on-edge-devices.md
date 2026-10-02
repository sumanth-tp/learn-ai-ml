---
id: cv-vision-on-edge-devices
title: "Computer Vision · Session 16; Vision on Edge Devices"
sidebar_label: "1 · Edge vision"
sidebar_position: 1
slug: /theory/cv/vision-on-edge-devices
description: "Estimate parameter and raw weight savings while planning accuracy, memory, power and latency tests on the target device."
tags: [computer-vision, edge, quantisation, mobilenet]
---

import Infographic from '@site/src/components/Infographic';
import EdgeWeightBudgetLab from '@site/src/components/viz/EdgeWeightBudgetLab';

**In one line.** Edge deployment chooses an architecture and representation that meet a measured device budget while preserving the task outcome.

## The idea in plain words

:::note Beyond the lecture

The raw-weight calculation, hardware qualifications, acceptance test and failure analysis extend the lecture. Its architecture and compression outline, two worked ratios and all five practice questions remain below.

:::

Running vision on a phone, camera or other edge device places the entire pipeline under a finite memory, compute, power and timing budget. The useful question is not only whether a model has fewer parameters. The device must acquire and decode an image, transform it into the expected input, execute supported operations, postprocess outputs and deliver an action before its deadline. A smaller model can help with storage and weight transfer while still failing if activations exceed memory, an operator falls back to a slow CPU path or image capture dominates latency. Define the end-to-end deadline and acceptable quality before compressing anything.

The lecture compares VGG-16 at about 138 million parameters with the original MobileNet V1 1.0-224 at about 4.2 million. The original MobileNets paper gives exactly these rounded counts in one comparison table. Dividing 138 by 4.2 yields approximately **32.857**, so “about 33 times fewer parameters” is a fair description of those variants. It is not a claim that every MobileNet version has 4.2 million parameters: width, classifier head and implementation change the count. The current Keras Applications catalogue, checked on 2026-10-02, lists VGG16 at 138.4 million and a packaged MobileNet at 4.3 million. The slight difference illustrates why the variant and counting convention should accompany a figure.

MobileNet V1 uses depthwise separable convolutions to reduce multiply-add operations compared with a full convolution at the same feature-map shapes. A depthwise step filters each channel separately; a pointwise step mixes channels. The architecture also offers width and resolution choices. A narrower model reduces many channel-dependent parameters and operations; a smaller input reduces spatial computation, but can erase small objects. EfficientNet-style compound scaling is another approach to balancing dimensions of a model, not a promise that a named family is best on every device. The existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is) develops convolution; the existing [pretrained CNN chapter](/docs/theory/dnn/pretrained-cnn-models-and-imagenet) covers backbone use in depth.

Quantisation represents numbers with fewer bits. In a deliberately idealised dense raw-weight calculation, 4.2 million values stored as 32-bit floats use 16.8 million bytes, or **16.8 decimal MB**. Storing each as eight bits uses 4.2 million bytes, or **4.2 decimal MB**, exactly one quarter of the raw weight bytes. This arithmetic does not give an actual package size or runtime speedup. Scale and zero-point metadata, alignment, mixed-precision layers, activations and runtime binaries add costs. Integer operations may be faster on a suitable processor or accelerator, but unsupported operations or conversion overhead can eliminate the gain. The target device decides.

<Infographic src="/img/cv/edge-deployment.svg" alt="Edge vision board comparing the original paper's VGG-16 138 million and MobileNet V1 4.2 million parameters, plus an idealised fourfold raw-weight saving from 32-bit to 8-bit storage." caption="Raw parameter arithmetic is a starting budget; accuracy, latency, power and memory need measurements." />

## How it works

### MobileNet & friends

Depthwise-separable convolutions (MobileNet) and EfficientNet-style scaling cut FLOPs and parameters dramatically for on-device inference.

:::tip

**Worked.** VGG-16 138M vs MobileNet 4.2M → ~33×; 32→8-bit quantisation → 4×.

:::

### Prune, quantise, distil

Pruning removes redundant weights; quantisation lowers precision (32→8-bit = 4×); knowledge distillation trains a small student to mimic a large teacher. Stack them to compound savings.

### Key takeaways

## A real system that works this way

The original MobileNets paper is a real system design example: it introduces an architecture and width and resolution controls to trade representation capacity against operation and parameter counts. Its table lists 1.0 MobileNet-224 at 4.2 million parameters and VGG 16 at 138 million under its evaluation setup. The figures are taken only as historical architecture counts; this chapter does not carry the paper's accuracy or speed results to a current phone. Keras Applications currently lists multiple MobileNet variants, reinforcing that a bare “MobileNet” name does not identify a single parameter count.

Google's official LiteRT post-training quantisation guide, opened on 2026-10-02, separates weight-only dynamic-range quantisation from full integer quantisation. The latter requires calibration data for activation ranges and a compatible integer operator path. The guide also discusses float fallback and device compatibility. ExecuTorch 1.5 documentation identifies another current on-device runtime family. These sources establish available paths, but no runtime conversion was executed for this chapter and no device timing is claimed.

Imagine an offline camera that flags a defect on a moving object. A model can meet a storage limit yet still miss the line-speed deadline. Benchmark the whole path with the actual camera resolution, image format, preprocessing, accelerator and sustained frame rate. Measure accuracy specifically on tiny and low-contrast defects after resizing and quantisation. If the model runs quickly only at a crop size that removes the defect, the apparent speed improvement is unusable. A system-level acceptance test should record both timely decisions and correct decisions under representative heat and power conditions.

## Code you can run

The first block reproduces the lecture's rounded architecture comparison. It prints **32.857**, approximately 33. These are the original paper's VGG-16 and MobileNet V1 1.0-224 variants, not universal family sizes.

```python
vgg_millions = 138.0
mobilenet_millions = 4.2
ratio = vgg_millions / mobilenet_millions
print(f'Parameter ratio: {ratio:.3f}x')
assert round(ratio, 3) == 32.857
```

The second block calculates idealised raw weight storage in decimal megabytes. It assumes that every one of the 4.2 million values is packed at the stated precision with no metadata or padding. It prints **16.8 MB** for 32-bit and **4.2 MB** for 8-bit values, hence a **4.0×** raw-weight reduction.

```python
parameters = 4_200_000
bytes_at_32 = parameters * 32 / 8
bytes_at_8 = parameters * 8 / 8
decimal_mb_32 = bytes_at_32 / 1_000_000
decimal_mb_8 = bytes_at_8 / 1_000_000
print(f'32-bit raw weights: {decimal_mb_32:.1f} MB')
print(f'8-bit raw weights: {decimal_mb_8:.1f} MB')
print(f'Raw-weight reduction: {bytes_at_32 / bytes_at_8:.1f}x')
assert (decimal_mb_32, decimal_mb_8) == (16.8, 4.2)
assert bytes_at_32 / bytes_at_8 == 4
```

The lab defaults to that MobileNet example at eight bits and also shows the 138/4.2 comparison. Switch the paper model or bit width to explore the raw storage bound. Its bars and table deliberately say “raw weights” so they are not mistaken for measured runtime memory or an exported file size.

<EdgeWeightBudgetLab />

Pruning and distillation are different levers. Pruning removes or masks parameters, but a sparse model only runs faster if the runtime and hardware use the sparsity pattern. Distillation trains a student against a teacher's outputs or representations; it can improve a small model's task quality, but success is empirical. Combining methods requires an evaluation after each transformation and after the final export because effects need not simply add.

## Designing with it

Start from the device contract. Record available memory for both weights and activations, input rate, deadline, power envelope, thermal behaviour and whether an accelerator is present. Include cold start if the application opens a camera intermittently. A raw parameter estimate helps shortlist candidates but ignores intermediate feature maps and buffers. Peak memory can occur during preprocessing or a wide intermediate layer rather than when the weight file is loaded.

Choose image size from the task. Reducing resolution lowers operation counts, yet a small defect may disappear or a box may become too coarse for its IoU target. Measure object-size distributions in pixels after the actual resize and crop. If a detector needs a high-resolution tile, account for tile overlap and merging, which can increase total compute. A model comparison at different image sizes can be misleading unless quality is matched.

Quantise with a representative calibration set when the chosen method requires it. Include low light, saturated frames, rare colours and small targets so activation ranges are not determined only by easy images. Compare the uncompressed and compressed outputs on the same held-out examples and inspect per-class and small-object errors. A tiny average accuracy drop can hide a serious loss on the rare event the device is meant to detect. Some operators may remain floating point, changing both package size and execution path; inspect the exported graph.

Benchmark on target hardware. Report image decode, preprocessing, inference, postprocessing and queueing separately, then report end-to-end median and tail latency under sustained load. Repeat after the device warms and thermal limits engage. Measure energy or battery draw if relevant, because a fast but power-hungry path may not meet field requirements. A 4× raw-weight reduction does not imply 4× throughput or 4× battery life.

Treat operator support as a design constraint. A theoretically efficient layer may map poorly to a given accelerator, while a more conventional layer has tuned kernels. A quantised operator can fall back to a slower path or require dequantisation between layers. Check the actual runtime's compatibility list and profile the exported graph, not the framework model before conversion. Keep the preprocessing tensor format and normalisation consistent with training.

Plan a rollback and update path. Save the original weights, conversion settings, calibration sample identity and runtime version. Ship a small set of known images for device-level regression checks. Monitor failed loads, inference deadlines and quality audits after a deployment. A new runtime may alter operator placement; a camera firmware update may change input colours. Versioning the whole inference pipeline makes a regression traceable instead of treating it as unexplained model drift.

## Where this stands in 2026

:::info Industry view

- The 138M and 4.2M rounded counts are verified in the original MobileNets paper; current Keras Applications lists 138.4M VGG16 and 4.3M packaged MobileNet. The variant and source determine the count.
- LiteRT quantisation guidance checked on 2026-10-02 distinguishes weight-only from calibrated full-integer paths. The local examples verify arithmetic only; no model conversion, on-device latency, power or accuracy was measured.
- The lecture claims 8-bit values give faster integer arithmetic. That is conditional on supported operators, hardware and conversion overhead, so the labelled note below narrows it.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What constraints define edge deployment of CV models?</summary>

Tight compute, memory and power budgets, often real-time and offline, favouring on-device inference (low latency, privacy).<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What makes MobileNet efficient?</summary>

Depthwise-separable convolutions, which factor a standard convolution into far fewer operations and parameters.<br /><em>Session 16 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> VGG-16 has ~138M params, MobileNet ~4.2M. Compute the reduction factor.</summary>

138/4.2 ≈ 33× fewer parameters.<br /><em>Session 16 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What size saving does 8-bit quantisation give vs 32-bit floats?</summary>

32/8 = 4× smaller (and faster integer arithmetic).<br /><em>Session 16 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Name the three main model-compression techniques.</summary>

Pruning (remove redundant weights), quantisation (lower precision), and knowledge distillation (small student mimics large teacher).<br /><em>Session 16 · conceptual</em>

</details>

## Further reading

- [Original MobileNets paper](https://arxiv.org/html/1704.04861) for the 138M versus 4.2M comparison and width/resolution choices.
- [Keras Applications catalogue](https://keras.io/api/applications/) for current packaged variants and parameter counts.
- [LiteRT post-training quantisation guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization) for conversion choices and calibration.
- [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html) for a current edge runtime family.
- Built from the course lecture "cv-s16-edge-devices" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source savings

The source's 138M versus 4.2M comparison reproduces for the specific VGG-16 and MobileNet V1 variants in the original paper: the ratio is 32.857, or about 33. Its “32 to 8 bits gives 4×” statement is true for ideally packed raw weight values. It does not establish that an exported model file, peak memory, latency or battery use improves fourfold. “Faster integer arithmetic” depends on supported operators and target hardware; some paths retain floating operations or add conversion costs.

:::

## Review an edge deployment claim

Suppose a proposal says a quantised model is four times smaller. Ask what was measured. If it is the raw weight array, the arithmetic is straightforward. If it is an exported package, compare byte counts before and after export and include metadata and runtime dependencies. If the claim concerns peak memory, profile activations and buffers on the device. If it concerns speed, measure the entire camera-to-decision path under realistic load. These are different outcomes and may move by different factors.

Another proposal says MobileNet is “33 times more efficient than VGG”. The source count is about 33 times fewer parameters for two historical variants. Efficiency could mean parameter storage, operation count, latency, energy per decision or task quality per unit cost. The paper's table contains separate operation and quality measures, and current packaged variants have slightly different counts. Name the quantity and variants explicitly. A parameter ratio alone cannot rank two deployed pipelines.

If quantisation lowers average accuracy only slightly, inspect where errors changed. An edge vision system may focus on rare small defects or a particular lighting condition. A drop concentrated there matters more than an average over many easy backgrounds. Compare confusion counts or mask/detection metrics by object size and capture condition. Keep the test set separate from calibration examples, and repeat on the exported runtime to catch conversion differences.

If the device meets its latency deadline for five minutes but misses it after an hour, thermal throttling or memory pressure may be involved. Profile sustained operation with camera capture, display, networking and other processes active. A model-only benchmark on a cool device cannot demonstrate production readiness. Measure deadline misses and queue growth; even if median inference stays fast, a tail of slow frames can cause stale decisions.

Pruning requires similar care. Removing 50% of scalar weights from a dense tensor may leave the same tensor shape and execution kernels, producing little runtime benefit. Structured pruning that removes channels can change shapes and operations, but may require fine-tuning to restore quality. Distillation can make a smaller student more useful, but it does not reduce the student's parameter count automatically; the student architecture determines that. Evaluate each method as an end-to-end artifact on the target hardware.

Finally, an offline requirement has operational consequences beyond model size. The device must start without a network connection, keep input data locally under the product's policy and handle model updates safely. A fallback for unsupported operations or corrupted model files should be tested. The best architecture on a development workstation may be the wrong one if the field device lacks its kernels. Let the target device and the actual visual task close the design loop.

## Check yourself

- I can calculate the original VGG-16 and MobileNet V1 parameter ratio and name the variants.
- I can distinguish a raw-weight storage factor from an exported-size or latency result.
- I can explain why calibration data and operator support affect quantisation outcomes.
- I can design a device-level test that measures quality, memory, power and end-to-end time.
