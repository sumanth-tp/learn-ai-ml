---
id: llme-quantisation
title: "Quantisation for Inference"
sidebar_label: "4 · Quantisation for inference"
sidebar_position: 4
slug: /llm-engineering/quantisation-for-inference
description: "Number formats, absmax and zero-point scaling, groups, outliers, GPTQ and AWQ ideas, GGUF k-quants and KV-cache quantisation, with real SmolLM2 weights quantised at several bit widths and measured by perplexity."
tags: [quantisation, int8, int4, fp8, nf4, gptq, awq, gguf, kv-cache, perplexity, inference]
---

import Infographic from '@site/src/components/Infographic';
import QuantisationLab from '@site/src/components/viz/QuantisationLab';

**In one line.** Quantisation stores each weight in fewer bits, so a memory-bound decode step reads less and runs faster, and the whole craft is choosing where the scales live and protecting the few weights and activations that cannot afford rounding.

:::note Not from a lecture
This chapter was written for this site from the sources under Further reading. Every quality number comes from the code below, run on SmolLM2-135M-Instruct, a small model that quantises worse than the large models the papers study, so read the shapes and not the absolute thresholds.
:::

## The idea in plain words

Chapter 1 showed that a decode step is the time to read the weights, so halving the bytes per weight roughly halves the step. **Quantisation** stores a small integer code and a shared scale instead of a 16-bit number, and rebuilds an approximate weight as code times scale when needed.

Think of rounding every measurement to the nearest mark on a ruler. The **scale** is the ruler's length, set by the largest value it must cover, which makes an outlier expensive: one huge weight stretches the ruler and rounds every ordinary weight to the same few marks. The craft follows from that. Give each small **group** of weights its own ruler so an outlier harms only its neighbours, keep outliers out of the rounding, or move the difficulty somewhere it can be absorbed.

The second theme is that the error that matters is not in the weight but in the layer's output, weighted by the activations, and finally in the model's predictions. GPTQ and AWQ use sample activations to steer the rounding. The experiments show that this helps, and that the proxy can mislead.

<Infographic src="/img/llme/quantisation-for-inference-formats.svg" alt="A table of number formats, three pictures of one scale per tensor, row or group, the error of five quantisation schemes on weights with one large outlier, and the error of three sets of 16 levels on Gaussian weights." caption="Formats, scale granularity and the synthetic experiments. Every figure is printed by block 1." />

<Infographic src="/img/llme/quantisation-for-inference-quality.svg" alt="Perplexity of SmolLM2-135M under round-to-nearest quantisation at five bit widths and four group sizes, a comparison of round to nearest, AWQ-style and GPTQ-style at 4 bits, a calibration trap for GPTQ on one layer type, and dynamic int8 results." caption="What quantising a real 135M model costs, from blocks 2, 3 and 4." />

<Infographic src="/img/llme/quantisation-for-inference-kv.svg" alt="Perplexity of the model with fake-quantised keys and values at 8, 4, 3 and 2 bits for three choices of per-token or per-channel scales." caption="Quantising the KV cache, block 5." />

## How it works

### Formats and scaling

| Format | Bits | Notes |
| --- | --- | --- |
| bf16, fp16 | 16 | the baseline; no scale |
| FP8 E4M3, E5M2 | 8 | 4-bit exponent and 3-bit mantissa, or 5-bit exponent and 2-bit mantissa, as defined by the FP8 paper |
| INT8, INT4 | 8, 4 | an integer code and a scale |
| NF4 | 4 | 16 levels at the quantiles of a normal distribution; the QLoRA paper calls it information-theoretically optimal for normally distributed weights |

**Absmax** (symmetric) scaling uses $s = \max|x| / (2^{b-1}-1)$ and stores $\text{round}(x/s)$. **Zero point** (asymmetric) spends the range on $[\min, \max]$ with $s = (\max - \min)/(2^b - 1)$ and a stored offset.

**Granularity** matters most: one scale per tensor is cheapest and worst, one per row is the usual floor, and **group-wise** scales, one per 32 to 128 weights, cost $16/g$ extra bits per weight with 16-bit scales, so 4-bit codes in groups of 64 take 4.25 bits. GGUF's k-quants apply the idea at scale: Hugging Face's table lists Q4_K as super-blocks of eight 32-weight blocks with 6-bit scales and minima at 4.5 bits per weight, Q5_K at 5.5, Q6_K at 6.5625, Q3_K at 3.4375 and Q2_K at 2.625.

### Outliers and calibration

A few activation channels are far larger than the rest. The LLM.int8() paper blames emergent features that dominate attention and prediction, and keeps about 0.1 per cent of values in 16 bits while the rest go to 8-bit. **SmoothQuant** instead rescales equivalently, moving difficulty from activations into the weights. **AWQ** finds that protecting about 1 per cent of weight channels, chosen by activation magnitude, removes most of the error, and does it by scaling those channels before rounding, with no backpropagation. **GPTQ** quantises a layer column by column and updates the remaining columns to compensate, using second-order statistics of the layer's inputs; its paper reports a 175B model quantised in about four GPU hours at 3 or 4 bits. Block 3 implements simplified versions of both.

### The KV cache and the speed to expect

**KIVI** quantises the cache by choosing an axis per half: keys per channel, because key channels have large outliers, values per token. Weight-only quantisation helps where chapter 1 said the time goes, memory-bound decode: the batch-1 weight-reading bound for Llama 3.1 8B falls from 4.79 ms in bf16 to 1.27 ms with 4-bit codes in groups of 64 (block 1). That is a ceiling. Dequantisation needs kernels fused with the matrix multiplication, and Hugging Face's guide cautions that quantisation can slightly raise latency, naming AWQ and fused AWQ modules as exceptions. At high batch or long prompts the step is compute-bound and fewer bytes help much less.

## A real system that works this way

**GPTQ** reports 3.25 times faster generation on an A100 and 4.5 times on an A6000 against FP16, with a 175B model on one GPU. **AWQ**, an MLSys 2024 best paper, reports more than 3 times speedup over FP16 for its TinyChat framework. Hugging Face's guide gives a size anchor: Mistral 7B takes 13.74 GB in bf16 and 6.87 GB in 8-bit. **llama.cpp**'s GGUF files carry the k-quants above.

## Code you can run

Five blocks. Blocks 2 to 5 load SmolLM2-135M-Instruct and WikiText-2 (CC BY-SA 4.0 per its dataset card) from the Hub, and each takes under a minute on CPU after the downloads. Perplexity is measured on the first 4,096 test tokens in four windows of 1,024, a small sample where differences of a few per cent are noise. "Fake quantisation" means rounding the weights and putting the dequantised values back: it measures quality, not speed.

### 1. Scales, outliers and levels on synthetic weights

Four thousand Gaussian weights with one planted outlier of 40, with the error measured on the ordinary weights only. The second half compares ways of placing 16 levels, including an NF4-style grid built from normal quantiles as the QLoRA paper describes (not claimed to equal the exact bitsandbytes table). The last lines turn bits per weight into a decode-step bound.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
weights = rng.normal(0, 1, 4096)
weights[100] = 40.0


def absmax_quantise(x, bits, group):
    grouped = x.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(grouped).max(axis=1, keepdims=True) / qmax
    codes = np.clip(np.round(grouped / scale), -qmax, qmax)
    return (codes * scale).reshape(-1)


def zero_point_quantise(x, bits, group):
    grouped = x.reshape(-1, group)
    levels = 2**bits - 1
    low, high = grouped.min(axis=1, keepdims=True), grouped.max(axis=1, keepdims=True)
    scale = (high - low) / levels
    zero = np.round(-low / scale)
    codes = np.clip(np.round(grouped / scale) + zero, 0, levels)
    return ((codes - zero) * scale).reshape(-1)


def mse(a, b, mask=None):
    diff = (a - b) ** 2
    return diff.mean() if mask is None else diff[mask].mean()


others = np.ones(4096, dtype=bool)
others[100] = False
print("4096 weights from N(0, 1) with one outlier of 40.0; error measured on the 4095 ordinary weights")
print("bits  scheme                       error on ordinary weights")
for bits in [8, 4]:
    for label, fn, group in [
        ("absmax, one scale per tensor", absmax_quantise, 4096),
        ("absmax, groups of 256", absmax_quantise, 256),
        ("absmax, groups of 64", absmax_quantise, 64),
        ("zero point, groups of 64", zero_point_quantise, 64),
    ]:
        print(f"{bits:4d}  {label:28s} {mse(weights, fn(weights, bits, group), others):.6f}")

gaussian = rng.normal(0, 1, 65536)
block = 64
grouped = gaussian.reshape(-1, block)
scale = np.abs(grouped).max(axis=1, keepdims=True)
normalised = grouped / scale

offset = 0.9677
positive = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
negative = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
levels = np.sort(np.concatenate([positive, [0.0], negative]))
levels = levels / np.abs(levels).max()
uniform = np.linspace(-1, 1, 16)


def snap(values, grid):
    return grid[np.abs(values[..., None] - grid).argmin(axis=-1)]


def raw_error(grid):
    return mse(grouped, snap(normalised, grid) * scale) / gaussian.var()


print()
print(f"4-bit codes on Gaussian weights, blocks of {block}, squared error as a share of the weight variance")
print(f"  15 evenly spaced levels (integer absmax): {raw_error(np.linspace(-1, 1, 15)):.4f}")
print(f"  16 evenly spaced levels:                  {raw_error(np.linspace(-1, 1, 16)):.4f}")
print(f"  16 levels from normal quantiles:          {raw_error(levels):.4f}")
print("  levels from normal quantiles:", np.round(levels, 3).tolist())

parameters, bandwidth = 8.03e9, 3.35e12
print()
print("Llama 3.1 8B, lower bound on a batch-1 decode step on the chapter 1 H100 figures (weights only)")
for label, bits in [("bf16", 16), ("8-bit codes, groups of 64", 8 + 16 / 64), ("4-bit codes, groups of 64", 4 + 16 / 64)]:
    gigabytes = parameters * bits / 8 / 1e9
    print(f"  {label:26s} {bits:5.2f} bits per weight  {gigabytes:6.2f} GB  {gigabytes * 1e9 / bandwidth * 1e3:5.2f} ms")
```

With one scale per tensor, 4-bit quantisation turns the ordinary weights into noise: an error of 0.984 against a unit variance. Groups of 256 bring it to 0.077, groups of 64 to 0.026 and a zero point to 0.019. The levels comparison is the NF4 argument in numbers: for bell-shaped weights, 16 normal-quantile levels leave 0.0085 of the variance against 0.0102 for 16 even levels and 0.0118 for the 15 levels of integer absmax.

<QuantisationLab />

The lab uses a real row: row 322 of layer 2's `gate_proj` has one weight of -3.2188 against a standard deviation of 0.2127. Its defaults, 4 bits and groups of 64, give the relative error 0.1767 that block 2 prints. With one scale for the row, almost every ordinary weight rounds to zero and the error is 0.3202 at both 3 and 4 bits. Then try keeping the outlier in 16 bits.

### 2. Round-to-nearest on a real model, by bits and group size

Every decoder linear layer is fake-quantised with absmax scales at 3 to 8 bits and several group sizes. Groups of 192 stand in for 128, which does not divide the hidden size of 576. The tied embeddings stay in fp32.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tokenizer("\n\n".join(test["text"][:400]), return_tensors="pt").input_ids[:, :4096]
windows = ids.reshape(-1, 1024)


def perplexity():
    total = 0.0
    with torch.inference_mode():
        for window in windows:
            total += model(window[None], labels=window[None]).loss.item()
    return float(torch.exp(torch.tensor(total / len(windows))))


def fake_quantise(weight, bits, group):
    out_features, in_features = weight.shape
    size = in_features if group == "row" else group
    grouped = weight.reshape(-1, size)
    qmax = 2 ** (bits - 1) - 1
    scale = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(grouped / scale).clamp(-qmax, qmax) * scale).reshape(out_features, in_features)


layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers")]
originals = [m.weight.data.clone() for m in layers]
print(f"{len(layers)} decoder linear layers, {sum(w.numel() for w in originals) / 1e6:.1f}M weights; "
      f"evaluation: {ids.shape[1]} WikiText-2 test tokens")
print(f"fp32 perplexity {perplexity():.2f}")

groups = ["row", 192, 64, 32]
perplexities = {}
for bits in [8, 6, 5, 4, 3]:
    for group in groups:
        for module, original in zip(layers, originals):
            module.weight.data = fake_quantise(original, bits, group)
        perplexities[(bits, group)] = perplexity()
for module, original in zip(layers, originals):
    module.weight.data = original

print()
print("perplexity (symmetric absmax, one scale per row or per group of 192, 64 or 32 weights)")
print("bits     row     g192      g64      g32")
for bits in [8, 6, 5, 4, 3]:
    print(f"{bits:4d}  " + "  ".join(f"{perplexities[(bits, g)]:7.2f}" for g in groups))
row = torch.round(dict(model.named_modules())["model.layers.2.mlp.gate_proj"].weight.data[322, :256] * 1e4) / 1e4
print()
print(f"one real row: model.layers.2.mlp.gate_proj row 322, first 256 weights, largest {row.abs().max():.4f} "
      f"against a standard deviation of {row.std():.4f}")
print("bits  one scale   groups of 64   groups of 16   (relative error, norm of the error over norm of the weights)")
for bits in [8, 4, 3]:
    cells = [(fake_quantise(row[None], bits, g)[0] - row).norm() / row.norm() for g in (256, 64, 16)]
    print(f"{bits:4d}  {cells[0]:9.4f}   {cells[1]:12.4f}   {cells[2]:12.4f}")
```

Perplexity is 18.07 in fp32. At 8 bits nothing changes and at 6 bits it rises by 0.2 to 0.8. At 4 bits the group size decides everything: 37.80 with one scale per row, 24.85 with groups of 64 and 23.07 with groups of 32, which costs 4.50 bits per weight. At 3 bits the model is broken under every setting, from 102 to 7,921. The papers report that larger models tolerate low bits better, which I did not test. The last table repeats the experiment on the lab's single real row.

### 3. Calibration: AWQ-style scaling and GPTQ-style compensation

Both methods use activations from 8,192 WikiText-2 training tokens and are evaluated on the test tokens. `awq_style` searches a per-channel scale $s = \text{mean}|x|^{\alpha}$ per layer and keeps the $\alpha$ with the smallest output error. `gptq_style` is column-by-column compensation with damping and a Cholesky factor of the inverse Hessian, in blocks of 128 columns. Both are simplified: no clipping search, no propagation through quantised layers, one pass.

```python
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()
torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
evaluation = tokenizer("\n\n".join(test["text"][:400]), return_tensors="pt").input_ids[:, :4096].reshape(-1, 1024)
calibration_ids = tokenizer("\n\n".join(train["text"][:600]), return_tensors="pt").input_ids[:, :8192]
layers = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers")}
originals = {n: m.weight.data.clone() for n, m in layers.items()}


def perplexity():
    total = 0.0
    with torch.inference_mode():
        for window in evaluation:
            total += model(window[None], labels=window[None]).loss.item()
    return float(torch.exp(torch.tensor(total / len(evaluation))))


def fake_quantise(weight, bits, group):
    grouped = weight.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(grouped / scale).clamp(-qmax, qmax) * scale).reshape(weight.shape)


def collect(window_length, only=""):
    chosen = {n: m for n, m in layers.items() if only in n}
    stats = {n: {"h": torch.zeros(m.in_features, m.in_features, dtype=torch.float64), "abs": torch.zeros(m.in_features),
                 "rows": None, "count": 0} for n, m in chosen.items()}

    def make_hook(key):
        def hook(module, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            entry = stats[key]
            entry["h"] += (x.T @ x).double()
            entry["abs"] += x.abs().sum(dim=0)
            entry["count"] += x.shape[0]
            if entry["rows"] is None:
                entry["rows"] = x[:512].clone()
        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in chosen.items()]
    with torch.inference_mode():
        for batch in calibration_ids.reshape(-1, window_length).split(16):
            model(batch)
    for handle in handles:
        handle.remove()
    return stats


def awq_style(weight, entry, bits, group):
    mean_abs = (entry["abs"] / entry["count"]).clamp_min(1e-5)
    rows = entry["rows"]
    reference = rows @ weight.T
    best_error, best_weight, best_alpha = float("inf"), None, 0.0
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        scale = mean_abs**alpha
        scale = scale / (scale.max() * scale.min()).sqrt()
        candidate = fake_quantise(weight * scale, bits, group) / scale
        error = ((rows @ candidate.T - reference) ** 2).mean().item()
        if error < best_error:
            best_error, best_weight, best_alpha = error, candidate, alpha
    return best_weight, best_alpha


def gptq_style(weight, entry, bits, group, damping=0.01, block=128):
    w = weight.double().clone()
    h = entry["h"].clone()
    n = w.shape[1]
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damping * torch.diag(h).mean() * torch.eye(n, dtype=torch.float64)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    quantised = torch.zeros_like(w)
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = w[:, start:stop].clone()
        errors = torch.zeros_like(chunk)
        local = inverse[start:stop, start:stop]
        for i in range(stop - start):
            if i % group == 0:
                scale = chunk[:, i:i + group].abs().amax(dim=1).clamp_min(1e-12) / qmax
            column = chunk[:, i]
            q = torch.round(column / scale).clamp(-qmax, qmax) * scale
            quantised[:, start + i] = q
            error = (column - q) / local[i, i]
            chunk[:, i:] -= error[:, None] * local[i, i:][None, :]
            errors[:, i] = error
        w[:, stop:] -= errors @ inverse[start:stop, stop:]
    return quantised.float()


def restore():
    for n, m in layers.items():
        m.weight.data = originals[n]


def relative_output_error(weight, new, x):
    return ((x @ new.T - x @ weight.T).pow(2).sum() / (x @ weight.T).pow(2).sum()).item()


print(f"fp32 perplexity {perplexity():.2f}; calibration {calibration_ids.numel()} train tokens, evaluation {evaluation.numel()} test tokens")
print()
print("a trap: GPTQ-style on the v_proj layers only, 4 bits, groups of 64 (round to nearest on the same layers: ", end="")
for n, m in layers.items():
    if "v_proj" in n:
        m.weight.data = fake_quantise(originals[n], 4, 64)
print(f"{perplexity():.2f})")
restore()
for window_length in [1024, 64]:
    stats = collect(window_length, only="v_proj")
    first_token, count = 0.0, 0
    for n, m in layers.items():
        if "v_proj" in n:
            new = gptq_style(originals[n], stats[n], 4, 64)
            m.weight.data = new
            first_token += relative_output_error(originals[n], new, stats[n]["rows"][:1])
            count += 1
    print(f"  calibration windows of {window_length:4d} tokens: perplexity {perplexity():.2f}, "
          f"error on the first token of a window {first_token / count:.2f}")
    restore()

stats = collect(64)
print("all decoder linear layers, 4 bits, groups of 64, calibration windows of 64 tokens")
print("method               perplexity  mean output error  seconds")
for method in ["round to nearest", "AWQ-style scaling", "GPTQ-style"]:
    start = time.perf_counter()
    errors, alphas = [], []
    for n, m in layers.items():
        w = originals[n]
        if method == "round to nearest":
            new = fake_quantise(w, 4, 64)
        elif method == "AWQ-style scaling":
            new, alpha = awq_style(w, stats[n], 4, 64)
            alphas.append(alpha)
        else:
            new = gptq_style(w, stats[n], 4, 64)
        rows = stats[n]["rows"]
        errors.append(relative_output_error(w, new, rows))
        m.weight.data = new
    seconds = time.perf_counter() - start
    note = f"  (mean alpha {sum(alphas) / len(alphas):.2f})" if alphas else ""
    print(f"{method:19s} {perplexity():11.2f}  {sum(errors) / len(errors):17.4f}  {seconds:7.1f}{note}")
    restore()
```

At 4 bits with groups of 64, round to nearest gives 24.85, AWQ-style scaling 21.84 (mean $\alpha$ near 0.34) and GPTQ-style 22.38. Both close much of the gap to 18.07 and neither closes it. GPTQ-style has half of AWQ-style's layer output error (0.0038 against 0.0077) yet the worse perplexity: lowest layer error is not the best model.

The first experiment explains a failure I hit while building it. GPTQ-style on the `v_proj` layers alone, calibrated on 1,024-token windows, gave perplexity 23.17, worse than round to nearest on the same layers (18.85), although its per-layer error on typical tokens was far lower. At the first token of a window its relative output error was 1.70. In a causal model the first token often becomes an attention sink that later tokens read, so an error there travels everywhere, and with long windows that token is one calibration row in a thousand, so the fit gives it up. Windows of 64 tokens raised its share sixteen-fold and gave perplexity 18.92 and a first-token error of 0.34. I tested this one explanation on one model.

### 4. PyTorch dynamic int8 on CPU

`torch.ao.quantization.quantize_dynamic` quantises weights to int8 per tensor and activations per tensor on every call. It needs the `qnnpack` engine on this Apple-silicon CPU, and PyTorch prints a notice that it is deprecated in favour of torchao, which is not installed here.

```python
import copy
import io
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()
torch.manual_seed(0)
torch.backends.quantized.engine = "qnnpack"
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
windows = tokenizer("\n\n".join(test["text"][:400]), return_tensors="pt").input_ids[:, :4096].reshape(-1, 1024)
probe = torch.randint(0, 1000, (1, 128))


def perplexity(m):
    total = 0.0
    with torch.inference_mode():
        for window in windows:
            total += m(window[None], labels=window[None]).loss.item()
    return float(torch.exp(torch.tensor(total / len(windows))))


def forward_ms(m):
    with torch.inference_mode():
        m(probe)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            m(probe)
            times.append(time.perf_counter() - start)
    return min(times) * 1e3


def megabytes(m):
    buffer = io.BytesIO()
    torch.save(m.state_dict(), buffer)
    return buffer.tell() / 1e6


decoder_linears = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers")}
variants = {
    "fp32": model,
    "dynamic int8, decoder layers": torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), decoder_linears, dtype=torch.qint8),
    "dynamic int8, every Linear": torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8),
}
print("variant                         file MB  forward ms (128 tokens)  perplexity")
for label, m in variants.items():
    print(f"{label:30s} {megabytes(m):8.0f}  {forward_ms(m):22.0f}  {perplexity(m):10.2f}")
```

The file shrinks from 538 MB to 221 MB with only decoder layers converted, but perplexity rises from 18.07 to 26.94 and 27.66, and the forward pass is slower (53 ms to about 90 ms in the run shown, which varies). Weight-only per-row 8-bit in block 2 changed perplexity by 0.05. The difference is the activations, whose outlier channels cannot share one scale per tensor: the LLM.int8 and SmoothQuant lesson in one table. It is a result for this CPU path and model, not a verdict on int8.

### 5. Quantising the KV cache: which axis

Keys and values are fake-quantised at the output of every `k_proj` and `v_proj`, so keys are rounded before rotary embedding, unlike a cache quantiser. Per-token scales use each head's 64 values, per-channel scales groups of 32 tokens.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()
torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
windows = tokenizer("\n\n".join(test["text"][:400]), return_tensors="pt").input_ids[:, :4096].reshape(-1, 1024)
heads, head_dim = model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads


def perplexity():
    total = 0.0
    with torch.inference_mode():
        for window in windows:
            total += model(window[None], labels=window[None]).loss.item()
    return float(torch.exp(torch.tensor(total / len(windows))))


def quantise(x, bits, dims):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def per_token(x, bits):
    batch, tokens, width = x.shape
    shaped = x.reshape(batch, tokens, heads, head_dim)
    return quantise(shaped, bits, dims=-1).reshape(batch, tokens, width)


def per_channel(x, bits, span=32):
    batch, tokens, width = x.shape
    shaped = x.reshape(batch, tokens // span, span, width)
    return quantise(shaped, bits, dims=2).reshape(batch, tokens, width)


def install(key_fn, value_fn):
    handles = []
    for layer in model.model.layers:
        if key_fn:
            handles.append(layer.self_attn.k_proj.register_forward_hook(lambda m, a, out: key_fn(out)))
        if value_fn:
            handles.append(layer.self_attn.v_proj.register_forward_hook(lambda m, a, out: value_fn(out)))
    return handles


print(f"fp32 perplexity {perplexity():.2f}; {heads} KV heads of dimension {head_dim}; groups of 32 tokens per channel for 'per channel'")
print("bits  keys per token / values per token   keys per channel / values per token   keys per token / values per channel")
for bits in [8, 4, 3, 2]:
    cells = []
    for key_kind, value_kind in [("token", "token"), ("channel", "token"), ("token", "channel")]:
        key_fn = (lambda out, b=bits: per_token(out, b)) if key_kind == "token" else (lambda out, b=bits: per_channel(out, b))
        value_fn = (lambda out, b=bits: per_token(out, b)) if value_kind == "token" else (lambda out, b=bits: per_channel(out, b))
        handles = install(key_fn, value_fn)
        cells.append(perplexity())
        for handle in handles:
            handle.remove()
    print(f"{bits:4d}  {cells[0]:34.2f}   {cells[1]:38.2f}   {cells[2]:36.2f}")
```

Keys by channel win clearly: 19.19 against 23.07 per token at 4 bits, and 25.57 against 75.70 at 3 bits. Values by channel are no better than per token (23.46 against 23.07 at 4 bits), consistent with KIVI's choice. At 2 bits everything collapses in this bare simulation.

## Designing with it

- **Start with 8-bit weight-only, per row.** It was free in block 2 and halves the bytes. Go lower only when memory or bandwidth forces it, and measure on your task.
- **Use groups once you go below 5 bits.** A row scale was acceptable at 5 bits (21.37) but at 4 bits groups of 32 cut perplexity from 37.80 to 23.07. Compare schemes at equal storage, counting the scales.
- **Protect outliers and check end to end.** Keep them in higher precision or scale them away, calibrate on data like your traffic, and evaluate the model, not the layer.
- **Match the format to the hardware.** Without a fused kernel on your device, memory is saved but latency may rise, as block 4 shows. Quantise cache keys per channel and values per token.

## Where this stands in 2026

:::info Industry view

- Weight-only 4-bit with group scales is the best-known deployment format for memory-bound serving, in the forms of GGUF k-quants, GPTQ and AWQ. Their papers date from 2022 and 2023, and AWQ's arXiv record shows a revision as recent as April 2026.
- PyTorch's eager-mode quantisation tools are being retired in favour of torchao, as the notice printed by block 4 says, and Hugging Face's list of libraries, among them Quanto, AQLM, VPTQ, AWQ and GPT-QModel, shows how fragmented the tooling is.
- Large models tolerate low bits better than the 135M model used here. That is the papers' claim and was not tested in this chapter. FP8 support depends on the engine and the hardware, and I did not review current engine documentation for it.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is the storage cost per weight of 4-bit codes with a 16-bit scale per group of 32, and what does a group of 128 cost?</summary>

$4 + 16/32 = 4.5$ bits per weight, and $4 + 16/128 = 4.125$ bits. Smaller groups buy accuracy and cost storage, as block 2's table shows.

</details>

<details>
<summary><strong>Q2.</strong> Block 1 gives an error of 0.984 for 4-bit absmax with one scale on weights with one outlier of 40. Why is it that large?</summary>

The scale is $40/7 \approx 5.7$, so ordinary weights near 1 in size round to zero. The reconstruction of the 4,095 ordinary weights is mostly zero and the error is about their whole variance.

</details>

<details>
<summary><strong>Q3.</strong> Using the chapter's numbers, what is the batch-1 weight-reading time for the 8B model with 4-bit codes and groups of 64, and what does that estimate leave out?</summary>

1.27 ms against 4.79 ms in bf16. It leaves out dequantisation work, kernel efficiency, KV cache traffic and quality loss, so it is a ceiling.

</details>

<details>
<summary><strong>Q4.</strong> Block 2's per-row int8 left perplexity at 18.12 but block 4's dynamic int8 reached 26.94. What differs?</summary>

Block 2 quantises only weights, a scale per row. Dynamic int8 also quantises activations with one scale per tensor, and a few activation channels are far larger than the rest. This is the outlier-feature problem that LLM.int8 and SmoothQuant address.

</details>

<details>
<summary><strong>Q5.</strong> Which axis do you quantise keys along, and which for values, and what does block 5 show?</summary>

Keys per channel and values per token. At 3 bits per-channel keys gave perplexity 25.57 against 75.70 for per-token keys, while per-channel values were no better than per-token ones.

</details>

## Further reading

- [Frantar et al., "GPTQ" (ICLR 2023)](https://arxiv.org/abs/2210.17323) and [Lin et al., "AWQ" (MLSys 2024)](https://arxiv.org/abs/2306.00978): the calibration-based weight quantisers.
- [Dettmers et al., "LLM.int8()" (NeurIPS 2022)](https://arxiv.org/abs/2208.07339) and [Xiao et al., "SmoothQuant" (ICML 2023)](https://arxiv.org/abs/2211.10438): outlier features and two ways to handle them.
- [Dettmers et al., "QLoRA"](https://arxiv.org/abs/2305.14314) for NF4, and [the FP8 formats paper](https://arxiv.org/abs/2209.05433) for E4M3 and E5M2.
- [Liu et al., "KIVI" (ICML 2024)](https://arxiv.org/abs/2402.02750): per-channel keys and per-token values for a 2-bit cache.
- [Hugging Face Hub, GGUF](https://huggingface.co/docs/hub/gguf): the k-quant table with bits per weight.
- [Hugging Face Transformers, optimizing inference](https://huggingface.co/docs/transformers/main/en/llm_optims): the quantisation section and its latency caveat.
- [WikiText-2 dataset card](https://huggingface.co/datasets/Salesforce/wikitext): the perplexity text used here.
- Related chapters: [why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound), [the KV cache](/docs/llm-engineering/kv-cache-and-paged-attention), [parameter-efficient fine-tuning with LoRA](/docs/theory/dnn/parameter-efficient-fine-tuning-lora) and the next chapter, [speculative decoding](/docs/llm-engineering/speculative-decoding).

## Check yourself

- I can compute the storage per weight of a quantisation scheme with group scales.
- I can explain absmax and zero-point scaling and why an outlier is expensive.
- I can say why group-wise scales beat per-tensor and per-row scales, and what they cost.
- I can describe LLM.int8, SmoothQuant, AWQ and GPTQ in a sentence each, and say why layer error is only a proxy.
- I can say which axis to quantise the keys and values of a KV cache along.
- I can estimate the decode speedup ceiling from weight bytes and say what it ignores.
