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

:::tip Before you start
**You should already know**

- What a convolution is and how many weights a convolutional layer has: [what a convolutional neural network is](/docs/theory/dnn/what-a-convolutional-neural-network-is).
- That pretrained image classifiers exist and what top-1 means: [pretrained CNN models and ImageNet](/docs/theory/dnn/pretrained-cnn-models-and-imagenet).
- A byte is 8 bits, so a 32-bit float takes 4 bytes.

**Reading time.** About 50 minutes, plus about 30 seconds to run the experiment.

**After this chapter you can**

- work out raw weight size and the parameter saving of a depthwise separable convolution by hand,
- quantise a few weights to 8-bit integers and see the rounding error,
- read a size, latency and agreement table and say which conversion helped, which did nothing and which broke the model.
:::

## In 30 seconds

A phone cannot hold or run a model the way a datacentre can, so you shrink the model and the numbers it uses. Storing each weight in 8 bits instead of 32 makes the weights one quarter the size, like saving a photo at lower quality. The photo gets smaller, but nobody promises it looks the same or loads faster. In the same way, a smaller model may or may not run faster and may or may not give the same answers, and the only way to know is to measure on the machine you will ship.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Parameter | One learned number in the model | 11.69 million in ResNet-18 |
| float32 | A number stored in 32 bits | 4 bytes each |
| int8 | A whole number from −128 to 127 stored in 8 bits | 1 byte each |
| Quantisation | Replacing float weights with small integers plus a scale | 0.2 becomes 51 with scale 0.003937 |
| Dynamic quantisation | Weights are integers, activation scales are found at run time | No calibration images needed |
| Static quantisation | Activation scales are fixed from calibration images | 32 sample crops |
| ONNX | A file format for models that other runtimes can load | A ResNet-18 exported to one file |
| Latency | Time for one input | 13 ms |
| Top-1 agreement | Share of inputs where two models give the same top class | 0.984 is 63 of 64 |

## The idea in plain words

:::note Beyond the course material

The raw-weight calculation, hardware qualifications, acceptance test and failure analysis extend the course material. Its architecture and compression outline, two worked ratios and all five practice questions remain below.

:::

Running vision on a phone, camera or other edge device places the entire pipeline under a finite memory, compute, power and timing budget. The useful question is not only whether a model has fewer parameters. The device must acquire and decode an image, transform it into the expected input, execute supported operations, postprocess outputs and deliver an action before its deadline. A smaller model can help with storage and weight transfer while still failing if activations exceed memory, an operator falls back to a slow CPU path or image capture dominates latency. Define the end-to-end deadline and acceptable quality before compressing anything.

A standard comparison sets VGG-16 at about 138 million parameters with the original MobileNet V1 1.0-224 at about 4.2 million. The original MobileNets paper gives exactly these rounded counts in one comparison table. Dividing 138 by 4.2 yields approximately **32.857**, so “about 33 times fewer parameters” is a fair description of those variants. It is not a claim that every MobileNet version has 4.2 million parameters: width, classifier head and implementation change the count. The current Keras Applications catalogue, checked on 2026-10-02, lists VGG16 at 138.4 million and a packaged MobileNet at 4.3 million. The slight difference illustrates why the variant and counting convention should accompany a figure.

MobileNet V1 uses depthwise separable convolutions to reduce multiply-add operations compared with a full convolution at the same feature-map shapes. A depthwise step filters each channel separately; a pointwise step mixes channels. The architecture also offers width and resolution choices. A narrower model reduces many channel-dependent parameters and operations; a smaller input reduces spatial computation, but can erase small objects. EfficientNet-style compound scaling is another approach to balancing dimensions of a model, not a promise that a named family is best on every device. The existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is) develops convolution; the existing [pretrained CNN chapter](/docs/theory/dnn/pretrained-cnn-models-and-imagenet) covers backbone use in depth.

Quantisation represents numbers with fewer bits. In a deliberately idealised dense raw-weight calculation, 4.2 million values stored as 32-bit floats use 16.8 million bytes, or **16.8 decimal MB**. Storing each as eight bits uses 4.2 million bytes, or **4.2 decimal MB**, exactly one quarter of the raw weight bytes. This arithmetic does not give an actual package size or runtime speedup. Scale and zero-point metadata, alignment, mixed-precision layers, activations and runtime binaries add costs. Integer operations may be faster on a suitable processor or accelerator, but unsupported operations or conversion overhead can eliminate the gain. The target device decides.

<Infographic src="/img/cv/edge-deployment.svg" alt="Edge vision board comparing the original paper's VGG-16 138 million and MobileNet V1 4.2 million parameters, plus an idealised fourfold raw-weight saving from 32-bit to 8-bit storage." caption="Raw parameter arithmetic is a starting budget; accuracy, latency, power and memory need measurements." />

## Worked example, step by step

Three small calculations that the first block under the experiment's heading reproduces: raw weight size, a convolution's parameter saving, and quantising three weights.

1. Weight size. ResNet-18 has about 11.69 million parameters. At 4 bytes each that is 11.69 × 4 = 46.76 MB. At 1 byte each it is 11.69 MB, exactly one quarter.
2. Convolution parameters. A 3 by 3 convolution from 32 channels to 64 has 3 × 3 × 32 × 64 = 18,432 weights.
3. A depthwise separable version filters each of the 32 channels alone (3 × 3 × 32 = 288 weights), then mixes channels with a 1 by 1 convolution (32 × 64 = 2,048). The total is 2,336, so 18,432 / 2,336 = 7.89 times fewer.
4. Quantise three weights: 0.2, −0.5 and 0.013. The largest magnitude is 0.5, so the scale is 0.5 / 127 = 0.003937.
5. Divide each weight by the scale and round: 0.2 / 0.003937 = 50.8, which rounds to 51. −0.5 becomes −127. 0.013 becomes 3.3, which rounds to 3.
6. Multiply back to see what the model actually uses: 51 × 0.003937 = 0.2008, −127 × 0.003937 = −0.5, 3 × 0.003937 = 0.0118. The errors are +0.0008, 0 and −0.0012.

In words: 8-bit storage is a quarter of the size, and each weight is off by up to half a scale step. The large weights barely notice, but the small one lost nearly 10% of its value, which is why quantisation can hurt layers whose weights span very different sizes.

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

Google's official LiteRT post-training quantisation guide, opened on 2026-10-02, separates weight-only dynamic-range quantisation from full integer quantisation. The latter requires calibration data for activation ranges and a compatible integer operator path. The guide also discusses float fallback and device compatibility. ExecuTorch 1.5 documentation identifies another current on-device runtime family. These sources establish available paths. The experiment below converts and times models with ONNX Runtime and PyTorch on a laptop CPU, but no LiteRT or ExecuTorch conversion was run and no phone, camera or accelerator was measured.

Imagine an offline camera that flags a defect on a moving object. A model can meet a storage limit yet still miss the line-speed deadline. Benchmark the whole path with the actual camera resolution, image format, preprocessing, accelerator and sustained frame rate. Measure accuracy specifically on tiny and low-contrast defects after resizing and quantisation. If the model runs quickly only at a crop size that removes the defect, the apparent speed improvement is unusable. A system-level acceptance test should record both timely decisions and correct decisions under representative heat and power conditions.

## Code you can run

The first block reproduces the rounded architecture comparison. It prints **32.857**, approximately 33. These are the original paper's VGG-16 and MobileNet V1 1.0-224 variants, not universal family sizes.

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

The three calculations in code. This block only needs NumPy; it prints the sizes, the parameter counts and the quantisation round trip.

```python
import numpy as np

parameters = 11_690_000
print('fp32 weights:', parameters * 4 / 1e6, 'MB   int8 weights:', parameters * 1 / 1e6, 'MB')

in_channels, out_channels, kernel = 32, 64, 3
standard = kernel * kernel * in_channels * out_channels
separable = kernel * kernel * in_channels + in_channels * out_channels
print('standard conv parameters:', standard, ' depthwise + pointwise:', separable, ' ratio:', round(standard / separable, 2))

weights = np.array([0.2, -0.5, 0.013])
scale = np.abs(weights).max() / 127
quantised = np.round(weights / scale).astype(int)
print('scale:', round(float(scale), 6), 'int8 values:', quantised.tolist())
print('restored:', np.round(quantised * scale, 4).tolist(), 'errors:', np.round(quantised * scale - weights, 5).tolist())
```

**Reading the output.** The weights are 46.76 and 11.69 MB, the convolution parameters 18,432 and 2,336 (ratio 7.89), the scale 0.003937 and the integers 51, −127 and 3, with a restored 0.0118 for the weight 0.013, as in steps 1 to 6.

The lab defaults to that MobileNet example at eight bits and also shows the 138/4.2 comparison. Switch the paper model or bit width to explore the raw storage bound. Its bars and table deliberately say “raw weights” so they are not mistaken for measured runtime memory or an exported file size.

<EdgeWeightBudgetLab />

**What each control does.** "Paper model example" picks MobileNet V1 (4.2 million parameters) or VGG-16 (138 million). "Stored weight precision" picks 32, 16, 8 or 4 bits per weight. The bars and table show raw weight storage in decimal megabytes.

**Try it yourself.**

1. At the defaults (MobileNet, 8 bits) the weights are 4.2 MB against 16.8 MB at 32 bits, the second block's result.
2. Switch to VGG-16 at 8 bits. The weights are 138 MB against 552 MB, so even quantised, VGG-16 is 33 times larger than MobileNet at the same precision.
3. Set MobileNet to 4 bits. The raw weights halve again to 2.1 MB. The experiment shows why the arithmetic is only a bound: the files measured were a quarter of the float size at 8 bits, and one of two models stopped giving correct outputs.

Pruning and distillation are different levers. Pruning removes or masks parameters, but a sparse model only runs faster if the runtime and hardware use the sparsity pattern. Distillation trains a student against a teacher's outputs or representations; it can improve a small model's task quality, but success is empirical. Combining methods requires an evaluation after each transformation and after the final export because effects need not simply add.

### Experiment: size, speed and agreement for real models

The question: does 8-bit quantisation or ONNX Runtime make a real vision model smaller, faster and still right, on this CPU? We take two pretrained torchvision classifiers, ResNet-18 (11.69 million parameters) and MobileNetV3-Small (2.54 million), and measure six variants of each: PyTorch float32, PyTorch dynamic int8 (which only covers linear layers), ONNX float32, ONNX int8 on the linear layers only, ONNX dynamic int8 on everything and ONNX static int8 in the QDQ format calibrated on 32 image crops. Size is the file on disk, latency is the median of 40 single-image runs on one CPU thread (Apple M3 Pro; one thread because in a preliminary run PyTorch used four threads and was several times slower on MobileNetV3-Small), and agreement is how often the top-1 class equals float32's on 64 crops of natural images taken from six photographs.

Versions: PyTorch 2.14.1, torchvision 0.29.1, ONNX 1.23.1, ONNX Runtime 1.30.0. The weights are the torchvision ImageNet-1k weights downloaded on first use; their terms follow the same note as the detection chapter, so check them before shipping. The photographs come from scikit-image and scikit-learn's sample data (some are downloaded on first use).

```python
import logging
import os
import tempfile
import time
import warnings

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from skimage import data
from sklearn.datasets import load_sample_images
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
torch.set_num_threads(1)
torch.backends.quantized.engine = 'qnnpack'
work = tempfile.mkdtemp()
images = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket()] + list(load_sample_images().images)
mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)

def crops(count, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        image = images[rng.integers(len(images))]
        side = int(rng.integers(min(image.shape[:2]) // 2, min(image.shape[:2]) + 1))
        y, x = rng.integers(0, image.shape[0] - side + 1), rng.integers(0, image.shape[1] - side + 1)
        tensor = torch.from_numpy(image[y:y + side, x:x + side].astype(np.float32) / 255).permute(2, 0, 1)[None]
        tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode='bilinear', antialias=True)[0]
        out.append(((tensor.permute(1, 2, 0).numpy() - mean) / std).transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)

class Reader(CalibrationDataReader):
    def __init__(self, batch):
        self.items = iter({'input': b[None]} for b in batch)
    def get_next(self):
        return next(self.items, None)

def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])

evaluation, calibration, one = crops(64, 1), crops(32, 2), crops(1, 3)
print(f'{"variant":22s} {"MB":>6s} {"ms":>6s} {"top-1 match":>12s} {"max |dlogit|":>13s}')
for name, builder, weights in (('resnet18', resnet18, ResNet18_Weights.IMAGENET1K_V1), ('mobilenet_v3_small', mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1)):
    model = builder(weights=weights).eval()
    dynamic = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(model.state_dict(), f'{work}/a.pt'); torch.save(dynamic.state_dict(), f'{work}/b.pt')
    torch.onnx.export(model, torch.from_numpy(one), f'{work}/f.onnx', input_names=['input'], dynamo=False)
    quantize_dynamic(f'{work}/f.onnx', f'{work}/g.onnx', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'])
    quantize_dynamic(f'{work}/f.onnx', f'{work}/d.onnx', weight_type=QuantType.QInt8)
    quantize_static(f'{work}/f.onnx', f'{work}/s.onnx', Reader(calibration), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    sessions = {k: session(f'{work}/{k}.onnx') for k in 'fgds'}
    onnx_run = lambda k: (lambda x: sessions[k].run(None, {'input': x})[0])
    variants = [
        ('torch fp32', lambda x: model(torch.from_numpy(x)).numpy(), 'a.pt'),
        ('torch int8 (Linear only)', lambda x: dynamic(torch.from_numpy(x)).numpy(), 'b.pt'),
        ('onnx fp32', onnx_run('f'), 'f.onnx'),
        ('onnx int8 Gemm only', onnx_run('g'), 'g.onnx'),
        ('onnx int8 dynamic', onnx_run('d'), 'd.onnx'),
        ('onnx int8 static QDQ', onnx_run('s'), 's.onnx'),
    ]
    print(f'{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')
    with torch.no_grad():
        reference = np.concatenate([variants[0][1](evaluation[i:i + 1]) for i in range(64)])
        for label, run, file in variants:
            out = np.concatenate([run(evaluation[i:i + 1]) for i in range(64)])
            for _ in range(5):
                run(one)
            times = []
            for _ in range(40):
                start = time.perf_counter(); run(one); times.append(time.perf_counter() - start)
            print(f'  {label:22s} {os.path.getsize(f"{work}/{file}") / 1e6:6.1f} {1000 * np.median(times):6.2f} {(out.argmax(1) == reference.argmax(1)).mean():12.3f} {np.abs(out - reference).max():13.3f}')
```

**Reading the output.** `MB` is file size, `ms` is median latency for one image, `top-1 match` is agreement with float32 on 64 crops (1.000 means all 64), and `max |dlogit|` is the largest change in any output score. There are no labels, so this measures change, not accuracy. Sizes, agreement and score changes are identical on every run; latencies varied by up to about 7% across three runs, so read them to the nearest millisecond.

**Line by line.**

- `torch.backends.quantized.engine = 'qnnpack'` is needed because this build has no default quantised engine; without it PyTorch raises "NoQEngine".
- `quantize_dynamic(..., op_types_to_quantize=['MatMul', 'Gemm'])` is the variant that quantises only linear layers.
- `Reader` feeds the 32 calibration crops to `quantize_static`, which uses them to fix each activation's scale. Calibration crops use a different random seed from the evaluation crops, but both come from the same six photographs, so agreement is a little optimistic.
- `options.intra_op_num_threads = 1` pins ONNX Runtime to one thread so it is compared fairly with PyTorch.

The printed output was:

```text
variant                    MB     ms  top-1 match  max |dlogit|
resnet18: 11.69 M parameters
  torch fp32               46.8  13.21        1.000         0.000
  torch int8 (Linear only)   45.3  13.58        0.984         0.204
  onnx fp32                46.7  43.39        1.000         0.000
  onnx int8 Gemm only      45.2  43.62        0.984         0.212
  onnx int8 dynamic        11.7  15.11        0.906         1.692
  onnx int8 static QDQ     11.7   9.28        0.875         1.924
mobilenet_v3_small: 2.54 M parameters
  torch fp32               10.3   9.84        1.000         0.000
  torch int8 (Linear only)    5.5  10.04        1.000         0.459
  onnx fp32                10.2   4.76        1.000         0.000
  onnx int8 Gemm only       5.4   4.71        1.000         0.579
  onnx int8 dynamic         2.7   5.57        0.000        18.917
  onnx int8 static QDQ      2.7   1.27        0.000        21.137
```

**What the numbers say.** For ResNet-18, full int8 does what the arithmetic promised on size: 46.7 MB falls to 11.7 MB, a quarter. Static int8 was also the fastest variant at 9.3 ms against 13.2 ms for PyTorch float32, about 1.4 times faster. But it cost agreement: the top-1 class changed on 8 of 64 crops (0.875), and a single output score moved by up to 1.9. Whether that is acceptable is a question about your task's labels, which this experiment does not have.

Three results contradict the usual story. First, PyTorch's own dynamic quantisation did almost nothing: it only converts linear layers, and ResNet-18's single linear layer is about 4% of its weights, so the file shrank from 46.8 to 45.3 MB and latency did not move (13.6 ms against 13.2). Second, converting to ONNX is not a speed-up in itself. ONNX Runtime float32 took 43 ms on ResNet-18, more than three times PyTorch, yet on MobileNetV3-Small it took 4.8 ms against 9.8 ms. I did not investigate why; the lesson is to time each runtime on each model. Third, MobileNetV3-Small broke under int8 convolutions: top-1 agreement was 0.000 for both dynamic and static int8, with output scores off by about 19 to 21. Static int8 was 3.7 times faster than ONNX float32 (1.3 ms against 4.8 ms) and completely wrong. Quantising only its linear layers kept agreement at 1.000 and saved about half the file (10.2 to 5.4 MB), because its classifier holds a large share of its weights.

I did not diagnose the MobileNetV3 failure. Its squeeze-and-excite gates, hard-swish activations and depthwise convolutions are plausible suspects, but I did not test any of them. The ONNX Runtime documentation suggests quantisation-aware training when post-training methods miss the accuracy goal; that was not tried here. The ONNX Runtime documentation, opened 2026-10-09, recommends static quantisation for CNNs, says benefits depend on hardware such as x86-64 CPUs with VNNI or Arm processors with dot-product instructions, and warns older hardware may see no gain or a loss.

<Infographic src="/img/cv-enrich/v3-edge-quantisation.svg" alt="Bars compare latency and a table lists size and top-1 agreement for ResNet-18 and MobileNetV3-Small under PyTorch float32, ONNX float32, ONNX static int8 and linear-only int8." caption="Look first at MobileNetV3-Small static int8: the fastest bar and the only row with agreement 0.000." />

Limits: one laptop CPU, not an edge device; one thread; batch size one; weights-only file sizes; no power, memory or thermal measurement; agreement on 64 crops from six photographs and no labels; one calibration set; default settings with no tuning of the quantiser.

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
- LiteRT quantisation guidance checked on 2026-10-02 distinguishes weight-only from calibrated full-integer paths. The arithmetic blocks verify the ratios; the experiment measures ONNX Runtime and PyTorch size, single-thread CPU latency and agreement with float32 on one laptop. No LiteRT conversion, on-device latency or power was measured.
- Course summaries claim 8-bit values give faster integer arithmetic. That is conditional on supported operators, hardware and conversion overhead, so the labelled note below narrows it.

:::

## Common mistakes

- **Equating bit width with speed.** Four times fewer bytes sounds like four times faster. On ResNet-18 static int8 was 1.4 times faster than PyTorch float32, and PyTorch's dynamic int8 was not faster at all. Measure the whole runtime.
- **Believing a smaller file means a working model.** The MobileNetV3-Small int8 files were one quarter of the size and agreed with float32 on 0 of 64 crops. Always compare outputs on held-out inputs after conversion.
- **Switching runtime and assuming a win.** ONNX Runtime float32 was three times slower than PyTorch on ResNet-18 and twice as fast on MobileNetV3-Small. Time each pair of model and runtime.
- **Calibrating on easy images.** It is convenient to calibrate on a handful of typical frames. Activation scales then clip the rare frames that matter. Include dark, saturated and small-target frames, and keep them separate from the test set.
- **Reading a library's "quantise" as covering convolutions.** PyTorch's dynamic quantiser converts only the layers you list, and by default that means linear layers. Check the exported graph for which operators changed.

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

<details>
<summary><strong>Q6 (Easy).</strong> A 3 by 3 convolution maps 32 channels to 64. How many weights does it have, and how many does the depthwise separable version have?</summary>

The standard version has 3 × 3 × 32 × 64 = 18,432. The separable version has 3 × 3 × 32 = 288 for the depthwise step and 32 × 64 = 2,048 for the pointwise step, 2,336 in total, so 7.89 times fewer.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> In the experiment, static int8 ResNet-18 ran in 9.3 ms against 13.2 ms for PyTorch float32 and changed the top-1 class on 8 of 64 crops. What would you do before shipping it?</summary>

Measure accuracy, not agreement, on labelled held-out images of the real task, by class and by hard case. Try per-channel weights, a better or larger calibration set and keeping the most sensitive layers in float. Re-time on the target device, because the speed-up depends on its integer instructions. Ship only if the accuracy loss is within the product's budget.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> Quantising only MobileNetV3-Small's linear layers saved about 4.8 MB and kept agreement at 1.000, while quantising everything gave agreement 0.000. Design a procedure to find a better compromise.</summary>

Quantise one operator type or block at a time and measure agreement and size after each, keeping the layers that break the output in float. The ONNX Runtime documentation points to its debugging tools for comparing float and quantised activations, which show the first layer where the outputs diverge. If too many layers must stay in float, try quantisation-aware training, which fine-tunes the model with rounding in the loop, then repeat the size, latency and agreement table.

</details>

## Further reading

- [Original MobileNets paper](https://arxiv.org/html/1704.04861) for the 138M versus 4.2M comparison and width/resolution choices.
- [Keras Applications catalogue](https://keras.io/api/applications/) for current packaged variants and parameter counts.
- [LiteRT post-training quantisation guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization) for conversion choices and calibration.
- [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html) for a current edge runtime family.
- [ONNX Runtime quantisation guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), opened 2026-10-09: dynamic against static, QDQ against QOperator, per-channel, and hardware that benefits.
- [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models.html), opened 2026-10-09: pretrained weights may carry their own licences derived from the training data.
- Built from the course lecture "cv-s16-edge-devices" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source savings

The 138M versus 4.2M comparison reproduces for the specific VGG-16 and MobileNet V1 variants in the original paper: the ratio is 32.857, or about 33. Its “32 to 8 bits gives 4×” statement is true for ideally packed raw weight values. It does not establish that an exported model file, peak memory, latency or battery use improves fourfold. “Faster integer arithmetic” depends on supported operators and target hardware; some paths retain floating operations or add conversion costs.

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

- I can compute raw weight size, a depthwise separable parameter count and a quantisation round trip by hand.
- I can read a size, latency and agreement table and say which conversion helped, which did nothing and which broke the model.
- I can explain why a runtime change alone can make a model slower, and why a smaller file proves nothing about accuracy.

## Where to go next

Next: [the CV question bank](/docs/theory/cv/question-bank), to practise across the whole course. Related: [object detection and box evaluation](/docs/theory/cv/object-detection-and-box-evaluation), whose detector is a typical candidate for the conversions measured here.
