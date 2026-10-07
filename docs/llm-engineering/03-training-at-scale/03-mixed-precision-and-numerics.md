---
id: llme-precision
title: "Mixed Precision and Numerics"
sidebar_label: "3 · Mixed precision and numerics"
sidebar_position: 3
slug: /llm-engineering/mixed-precision-and-numerics
description: "How fp32, bf16, fp16 and fp8 store numbers, why training keeps a master copy and scales the loss, and what breaks without them, shown with measured experiments on a real small Llama model and real SmolLM2 activations."
tags: [mixed-precision, bf16, fp16, fp8, loss-scaling, gradscaler, autocast, numerics, quantisation, transformer-engine]
---

import Infographic from '@site/src/components/Infographic';
import PrecisionRangeLab from '@site/src/components/viz/PrecisionRangeLab';

**In one line.** Training stores most numbers in 16 or 8 bits to save memory and time, and stays correct only because a few things (a full-precision copy of the weights, a scaled loss, running sums in 32 bits) protect the numbers that small formats would destroy.

:::tip Before you start
- **You should already know** that a model is a set of numbers called weights and that training nudges them with gradients ([Adam optimiser](/docs/theory/dnn/adam-optimizer)), and where the 2 + 2 + 12 bytes per parameter come from ([DDP, FSDP and ZeRO](/docs/llm-engineering/ddp-fsdp-and-zero)).
- **Reading time:** about 40 minutes, plus about a minute to run the code.
- **After this chapter you can** read a floating-point format's range and step size, explain why bf16 needs no loss scale and fp16 does, predict when an update or a sum is lost to rounding, and say when fp8 needs a scale per block.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every experiment runs in this chapter's environment: Python 3.14, PyTorch 2.14.1 on the CPU, Transformers 5.18.0. NVIDIA Transformer Engine (release 2.20.2 on PyPI on 5 October 2026) needs GPUs, so its snippet is not run. Sources were opened on 7 October 2026.
:::

## In 30 seconds

A computer stores a decimal number like 0.1 with a fixed number of binary digits, so it keeps only an approximation. More digits give a better approximation and cost more memory. Think of a ruler: a ruler marked in millimetres cannot show half a millimetre, and a pocket ruler marked in whole centimetres cannot show a millimetre at all.

Language-model training uses rulers of four sizes: 32 bits, 16 bits (in two layouts) and 8 bits. The smaller rulers are faster and use less memory, but they have two failure modes. A small number can fall between two marks and become zero, and a small number added to a big one can vanish. Mixed precision is the set of tricks that avoids both.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Bit | One binary digit, 0 or 1 | A 16-bit number has 16 of them |
| Exponent | The bits that say how big the number is, as a power of 2 | Large exponent field = large number |
| Mantissa | The bits that say which digits the number has | More mantissa bits = a finer ruler |
| Subnormal | A very small number stored with reduced precision, below the smallest normal number | fp16 below 6.1e-5 |
| Overflow and underflow | A number too big for the format (becomes infinity), or too small (becomes zero) | 70,000 in fp16 is inf |
| Master weights | A 32-bit copy of the weights that the optimiser updates | Next to the 16-bit working copy |
| Loss scaling | Multiplying the loss by a constant so small gradients stay representable | Scale 1,024 |
| Autocast | PyTorch's switch that runs chosen operations in lower precision | `torch.autocast` |
| Quantisation scale | A multiplier that moves a tensor into a small format's range | 448 divided by the largest value |

## The idea in plain words

Start with one number, 0.1. In binary it is 1.6 times 2 to the power minus 4, and 1.6 in binary never ends: 1.1001100110011 and so on. A format with 7 digits after the point (bf16) must cut it: 1.1001101, which is 1.6015625. So bf16 stores 0.1 as 0.10009765625. A format with 10 digits (fp16) stores 0.0999755859375, and one with 23 (fp32) stores 0.10000000149. Each is a nearby number, not the number.

Now take two ideas from that. First, the **step size** between neighbouring numbers depends on how many digits you keep and how big the number is. Near 1 the step in bf16 is 0.0078; near 32 it is 0.25. Second, the **range** (the largest and smallest numbers) depends on the exponent bits. bf16 gives 8 exponent bits, like fp32, so it covers almost the same range. fp16 gives only 5, so its range is tiny: the largest number is 65,504.

Those two facts explain every rule of mixed precision. A tiny gradient can sit below the smallest number a format can hold. A small update added to a big weight can be smaller than half a step, so it rounds to nothing. A running sum can grow until each new item is smaller than half a step, and then it stops growing.

<Infographic src="/img/llme/mixed-precision-and-numerics-formats.svg" alt="Bit layouts of fp32, bf16, fp16, fp8 e4m3 and fp8 e5m2 as sign, exponent and mantissa boxes beside a table of the largest number, smallest normal number, smallest subnormal number and the gap above 1, with three notes on bf16, fp16 and fp8" caption="Compare the orange exponent boxes first: fp32 and bf16 have the same, so they have the same range. Then compare the blue mantissa boxes: they set the step size. Figures are printed by block 1." />

## Worked example, step by step

Three small calculations, each of which the code in block 2 prints.

1. **Store 0.1.** In fp16, 0.1 is 1.6 times 2 to the power minus 4, with 10 digits kept: 0.0999755859375. The error is 2.4e-5. In bf16 only 7 digits are kept, so the value is 0.10009765625 and the error is 9.8e-5, four times larger. fp32 is off by 1.5e-9.
2. **Add 0.01 to 32.** In fp16 the gap between neighbours near 32 is 0.03125. Half a gap is 0.0156, which is bigger than 0.01, so the sum rounds back to 32. Do it 10,000 times and fp16 stops at 32.0 (after 2,798 additions). bf16 has a coarser step and stops at 4.0 (after 350). The true answer is 100. An fp32 accumulator gets 100.0030.
3. **A gradient of 2e-8 in fp16.** The smallest fp16 number is 5.96e-8. Half of that is 2.98e-8, and 2e-8 is below it, so the gradient is stored as zero. Now multiply the loss by 1,024 first. The scaled gradient is 2.05e-5, which fp16 stores as 2.0504e-5. After the backward pass, divide by 1,024: 2.0023e-8. That is within 0.12 per cent of 2e-8. This is **loss scaling**.

<Infographic src="/img/llme/mixed-precision-and-numerics-worked.svg" alt="Three columns: storing 0.1 in fp32, fp16 and bf16 with their errors; adding 0.01 to 32 in fp16 and bf16 stopping at 32 and 4; and a 2e-8 gradient lost in fp16 until the loss is scaled by 1024" caption="Read each column top to bottom: the problem, the size of the damage, the cure. All numbers are printed by block 2." />

## How it works

### What is inside a floating-point number?

A floating-point number has a sign bit, some exponent bits and some mantissa bits. For a normal number the value is the sign, times `1.m`, times 2 to the power `e - bias`. Here `m` is the mantissa digits and `e` the exponent field. In words: the exponent picks a power of two, and the mantissa picks one of `2^k` evenly spaced values between that power and the next, where `k` is the number of mantissa bits.

That spacing is the step size. Below the smallest normal number, the format switches to **subnormal** numbers, which have a fixed step and gradually lose digits until they reach zero. Block 1 prints the numbers for five formats from PyTorch's own `torch.finfo`:

| Format | Exponent / mantissa bits | Largest | Smallest normal | Gap just above 1 |
| --- | --- | --- | --- | --- |
| fp32 | 8 / 23 | 3.40e38 | 1.18e-38 | 1.19e-7 |
| bf16 | 8 / 7 | 3.39e38 | 1.18e-38 | 0.0078 |
| fp16 | 5 / 10 | 65,504 | 6.10e-5 | 0.00098 |
| fp8 e4m3 | 4 / 3 | 448 | 0.0156 | 0.125 |
| fp8 e5m2 | 5 / 2 | 57,344 | 6.10e-5 | 0.25 |

bf16 (brain floating point) keeps fp32's exponent and cuts the mantissa. fp8 comes in two layouts: e4m3 for more digits, e5m2 for more range. The FP8 paper defines both, and notes that e4m3 gives up infinities to extend its range to 448.

### What does mixed precision mean?

The 2018 paper that introduced it names three techniques. Keep a **master copy** of the weights in fp32 and update that; use the 16-bit copy for the forward and backward passes. **Scale the loss** so that small gradients survive. Do arithmetic in 16 bits but **accumulate in fp32**. Block 3 and block 5 test the first two. Accumulation in fp32 is a hardware property of the matrix units, so it is shown here only with the running sum of block 2.

In PyTorch the switch is `torch.autocast`. Its documentation says matrix operations such as `linear`, `matmul` and convolutions run in the lower precision, while reductions and sensitive operations such as `softmax`, `layer_norm`, `sum` and `cross_entropy` run in float32. On the CPU the default autocast dtype is bfloat16; on CUDA it is float16.

### Why does fp16 need a loss scale and bf16 not?

The gradient of a loss can be tiny. fp16's smallest normal number is 6.1e-5 and its smallest subnormal 5.96e-8, so gradients below those lose digits or become zero. The 2018 paper found in one network that about 5 per cent of weight gradients had exponents below -24 and would be lost. bf16's smallest normal number is 1.18e-38, the same as fp32, so a gradient that fits in fp32 almost always fits in bf16. The PyTorch documentation notes that CPU mixed precision with bfloat16 only uses autocast, with no scaler.

**Loss scaling** multiplies the loss by a constant before the backward pass. By the chain rule every gradient is multiplied by the same constant, so tiny gradients move up into the range fp16 can hold. Before the optimiser step the gradients are divided by the constant again. It must come before gradient clipping, so the clipping threshold means what you think it means.

The constant cannot be too big. fp16 overflows above 65,504, and an overflow gives an infinity that poisons the update. So PyTorch's `GradScaler` is dynamic. It starts at 65,536, doubles the scale after 2,000 clean steps, and halves it and skips the step when it finds an infinity or NaN. Block 4 runs it with an overflow injected on purpose.

### What does a master copy buy?

An update of 0.001 added to a weight of 1.0 is smaller than half a step in bf16 (the step near 1 is 0.0078). It is lost. In fp32 (step 1.2e-7) it is kept. Without a master copy, small updates to big weights disappear. With one, the update is added in fp32 and the bf16 copy is re-cast from it, so tiny updates pile up until they cross a step.

The master copy costs memory. The recipe is 2 bytes of working weight, 2 of gradient and 12 of optimiser state (the 4-byte master plus Adam's two 4-byte moments): 16 bytes per parameter. That is the same 16 bytes as plain fp32 (4 + 4 + 8). Mixed precision does not shrink model state. It shrinks activations, memory traffic and compute time, which is where most of the savings are. DeepSeek-V3 goes further: it keeps master weights and gradients in fp32 but stores Adam's moments in bf16, and reports no observable loss in quality.

<Infographic src="/img/llme/mixed-precision-and-numerics-step.svg" alt="Five connected boxes for the fp32 master weights, bf16 working copy, forward pass, backward pass and the update that returns to the master, above a table of bytes per parameter for three recipes and a note that pure bf16 left more weights unchanged" caption="Follow the loop from the blue master box to the purple update box and back. The table at the bottom left is the memory bill for each recipe." />

### What changes with fp8?

With 3 or 2 mantissa bits, fp8 has a median rounding error of about 2 per cent (e4m3) or 4 per cent (e5m2) in block 2, and a very small range. So fp8 needs a **scale** for each tensor, chosen so the largest value lands near the format's largest (448 for e4m3). That works until one huge value sets the scale and everything else is pushed toward zero.

Large language models have such values. The paper "Massive Activations in Large Language Models" reports a few activations hundreds or thousands of times larger than typical ones, and block 6 finds one in a real small model. The fix, as in DeepSeek-V3, is a scale per small group: one per 1 x 128 tile of an activation and one per 128 x 128 block of a weight. DeepSeek-V3 then uses e4m3 for all tensors, promotes partial sums to higher precision every 128 elements, and keeps the embeddings, output head, mixture-of-experts gating, normalisation and attention operators in bf16 or fp32.

NVIDIA's Transformer Engine documentation describes three recipes: delayed scaling (scale from a history of maxima), current scaling (scale from the tensor in hand) and MXFP8 (a scale per 32 values, on Blackwell GPUs). In its hybrid mode, forward uses e4m3 and backward uses e5m2. NVFP4 goes to 4 bits with a scale per 16 values.

## A real system that works this way

**Llama 3** trained in bf16: the "BF16 MFU" column in the last chapter's Table 4 says so. The paper's own inference experiments used the native FP8 support of H100 GPUs, not training.

**DeepSeek-V3** is the public example of fp8 training at scale. Its report describes e4m3 for all three matrix multiplications of each linear layer (forward, activation gradient and weight gradient), the tile and block scales above, and a relative loss error against a bf16 baseline that stays below 0.25 per cent on the two smaller models they trained for about 1 trillion tokens. Those are the authors' measurements on their cluster.

**NVFP4 pretraining** (arXiv 2509.25149, submitted 29 September 2025, revised March 2026) reports a 12-billion-parameter model trained on 10 trillion tokens in 4-bit precision, with loss and downstream accuracy comparable to an fp8 baseline. That is the authors' claim about their recipe; this chapter does not reproduce it.

## Code you can run

Six blocks. Everything runs on the CPU in about a minute. Block 6 uses `SmolLM2-135M-Instruct`, which downloads once.

### 1. The five formats, from PyTorch's own numbers

We read `torch.finfo` for each format, derive the bit counts, and show what each does to 0.1, one third and some out-of-range values.

```python
import math

import torch

FORMATS = [
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
    ("fp8 e4m3", torch.float8_e4m3fn),
    ("fp8 e5m2", torch.float8_e5m2),
]

print("format    bits  exp  mant   largest        smallest normal   smallest subnormal   eps (1 to next)")
for name, dt in FORMATS:
    f = torch.finfo(dt)
    mant = round(-math.log2(f.eps))
    exp = f.bits - 1 - mant
    subnormal = f.smallest_normal * f.eps
    print(f"{name:<9} {f.bits:>4} {exp:>4} {mant:>5}   {f.max:<13.6g}  {f.smallest_normal:<16.6g}  {subnormal:<18.6g}  {f.eps:.6g}")

print()
print("how each format stores 0.1 and 1/3")
for name, dt in FORMATS:
    a = torch.tensor([0.1, 1 / 3], dtype=torch.float64).to(dt).to(torch.float64)
    print(f"{name:<9} 0.1 -> {a[0].item():.12g}   1/3 -> {a[1].item():.12g}")

print()
big = torch.tensor([100.0, 300.0, 500.0, 70000.0, 1e-5, 1e-8])
print("what a cast does to values outside the range (torch", torch.__version__ + ")")
print("value      ", *[f"{v:>10g}" for v in big.tolist()])
for name, dt in FORMATS[1:]:
    print(f"{name:<10} ", *[f"{v:>10.4g}" for v in big.to(dt).float().tolist()])
```

**Reading the output.** The table is the one above. 0.1 comes back as 0.10000000149 (fp32), 0.10009765625 (bf16), 0.0999755859375 (fp16), 0.1015625 (e4m3) and 0.09375 (e5m2). In the range test, 70,000 becomes `inf` in fp16 and e5m2 but 70,144 in bf16. In e4m3, 500 and 70,000 both become 448: on this CPU PyTorch saturates e4m3 instead of producing an infinity or a NaN. Another device or kernel may behave differently.

**Line by line.**

- `mant = round(-math.log2(f.eps))` recovers the mantissa bit count from the gap above 1, which is `2 ** -mant`.
- `f.smallest_normal * f.eps` is the smallest subnormal number.
- Going through `torch.float64` first means the printed digits are the stored value, not a rounded display.

### 2. Rounding, lost sums and a lost gradient

The three worked-example calculations, in code, plus the typical rounding error of each format on random numbers.

```python
import torch

print("step 1: how big is the gap between neighbouring numbers near 32?")
for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
    one = torch.tensor(32.0, dtype=dt)
    gap = (torch.nextafter(one, torch.tensor(1e9, dtype=dt)) - one).item()
    print(f"  {name}: gap {gap:g}, half gap {gap / 2:g}, 32 + 0.01 = {(one + torch.tensor(0.01, dtype=dt)).item():g}")

print()
print("step 2: add 0.01 ten thousand times (the exact answer is 100)")
for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
    total = torch.tensor(0.0, dtype=dt)
    step = torch.tensor(0.01, dtype=dt)
    stuck_at = None
    for i in range(10000):
        new = total + step
        if stuck_at is None and new.item() == total.item():
            stuck_at = (i, total.item())
        total = new
    note = f"stopped growing at {stuck_at[1]:g} after {stuck_at[0]} additions" if stuck_at else "never stuck"
    print(f"  {name} accumulator: {total.item():.4f}  ({note})")
acc = torch.tensor(0.0)
small = torch.tensor(0.01).to(torch.bfloat16)
for _ in range(10000):
    acc = acc + small.float()
print(f"  bf16 inputs, fp32 accumulator: {acc.item():.4f}  (each input is really {small.item():.9f})")

print()
print("step 3: a tiny gradient in fp16, with and without a loss scale")
g = 2e-8
for scale in (1, 1024):
    stored = torch.tensor(g * scale, dtype=torch.float16)
    print(f"  scale {scale:>4}: stored {stored.item():.4e}, after dividing by the scale {stored.item() / scale:.4e}")

print()
print("step 4: typical rounding error on 100,000 random normal numbers")
x = torch.randn(100000, generator=torch.Generator().manual_seed(0))
for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp8 e4m3", torch.float8_e4m3fn), ("fp8 e5m2", torch.float8_e5m2)):
    err = ((x.to(dt).float() - x).abs() / x.abs()).median().item()
    print(f"  {name:<9} median relative error {err:.5f}")
```

**Reading the output.** Near 32 the fp16 gap is 0.03125, so `32 + 0.01 = 32`. After 10,000 additions of 0.01, fp16 sits at 32.0, bf16 at 4.0 and fp32 at 100.0030. With bf16 inputs and an fp32 accumulator the answer is 100.0977, which is off only because each bf16 input is really 0.010009766. The 2e-8 gradient is stored as zero at scale 1 and as 2.0504e-05 at scale 1,024, which divides back to 2.0023e-08. The median rounding errors are 0.00017 (fp16), 0.00135 (bf16), 0.02183 (e4m3) and 0.04338 (e5m2).

**Line by line.**

- `torch.nextafter` gives the next representable number, so the difference is the gap.
- `stuck_at` records the first addition that changed nothing.
- The last loop casts each input to bf16 but adds in fp32: this is the "16-bit inputs, 32-bit accumulator" idea.

### 3. How big must the loss scale be?

First a synthetic test: 100,000 gradients spread evenly in log scale from 1e-10 to 1e-2, stored in fp16 at four scales. Then the real tiny Llama (158,016 parameters, the one from the last chapter) run in half precision.

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
TOKENS = torch.randint(0, 512, (8, 32), generator=torch.Generator().manual_seed(1))


def make():
    torch.manual_seed(0)
    return LlamaForCausalLM(CFG)


def grads(model):
    return torch.cat([p.grad.reshape(-1).float() for p in model.parameters()])


print("synthetic gradients, 100,000 values spread evenly in log scale from 1e-10 to 1e-2, stored in fp16")
g = 10 ** (torch.rand(100000, generator=torch.Generator().manual_seed(0)) * 8 - 10)
for scale in (1, 2**8, 2**16, 2**24):
    s = (g * scale).to(torch.float16)
    print(f"  scale 2^{int(torch.log2(torch.tensor(float(scale)))):>2}: flushed to zero {(s == 0).float().mean().item():6.1%}   overflowed {torch.isinf(s).float().mean().item():6.1%}")

reference = make()
reference(TOKENS, labels=TOKENS).loss.backward()
ref = grads(reference)
small = (ref != 0) & (ref.abs() < 2**-14)
print()
print(f"real tiny Llama, fp32 gradients: {int((ref != 0).sum()):,} nonzero, {int(small.sum()):,} below fp16's smallest normal number")
for scale in (1, 2**8, 2**16, 2**24):
    model = make().half()
    (model(TOKENS, labels=TOKENS).loss * scale).backward()
    g16 = grads(model)
    bad = int((~torch.isfinite(g16)).sum())
    lost = int(((g16 == 0) & small).sum())
    ok = small & torch.isfinite(g16)
    err = ((g16[ok] / scale - ref[ok]).abs() / ref[ok].abs()).median().item()
    print(f"  scale 2^{int(torch.log2(torch.tensor(float(scale)))):>2}: small gradients lost {lost:>2}, non-finite {bad:>7,}, median error on small ones {err:.4f}")
```

**Reading the output.** Without a scale, 30.8 per cent of the synthetic gradients flush to zero. A scale of 2^8 cuts that to 0.9 per cent, and 2^16 removes it. At 2^24, 5.1 per cent overflow. The real model shows the same shape, more mildly. It has 17,527 gradients below fp16's smallest normal number. At scale 1 it loses 10 of them to zero and the median error on them is 0.0129. At 2^8 it loses 1 and the median error falls to 0.0049. At 2^24, 118,720 gradients are infinite or NaN.

**What did not work.** This tiny model barely needs loss scaling: only 10 of 138,240 non-zero gradients are lost at scale 1. The synthetic test is the harsh case. Real large models vary, and the 2018 paper found some networks needed no scaling and others needed one to match fp32.

**Line by line.**

- `10 ** (rand * 8 - 10)` draws exponents uniformly between -10 and -2.
- `small` marks gradients below `2 ** -14`, fp16's smallest normal number.
- `g16 / scale` undoes the scale before comparing with the fp32 reference.

### 4. `GradScaler` growing, and skipping on overflow

A real training loop on the CPU with `torch.autocast` in float16 and `torch.amp.GradScaler`. At step 5 the loss is multiplied by 1e30 to force an overflow.

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
TOKENS = torch.randint(0, 512, (8, 32), generator=torch.Generator().manual_seed(1))

torch.manual_seed(0)
model = LlamaForCausalLM(CFG)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cpu", init_scale=2.0**14, growth_interval=3)
first = next(model.parameters())

print("step  loss    scale before -> after   overflow injected   weights changed")
for step in range(10):
    with torch.autocast("cpu", dtype=torch.float16):
        loss = model(TOKENS, labels=TOKENS).loss
    injected = step == 5
    before_scale = scaler.get_scale()
    snapshot = first.detach().clone()
    scaler.scale(loss * (1e30 if injected else 1.0)).backward()
    scaler.step(optimiser)
    scaler.update()
    optimiser.zero_grad()
    changed = not torch.equal(snapshot, first.detach())
    print(f"{step:>4}  {loss.item():.4f}  {before_scale:>8.0f} -> {scaler.get_scale():<8.0f}  {str(injected):<17}   {changed}")

print()
default = torch.amp.GradScaler("cpu")
print(f"defaults of torch.amp.GradScaler: init_scale {default.get_scale():.0f}, growth_factor {default.get_growth_factor()}, "
      f"backoff_factor {default.get_backoff_factor()}, growth_interval {default.get_growth_interval()}")
print("this run used init_scale 16384 and growth_interval 3 so the doubling shows in ten steps")
```

**Reading the output.** The scale starts at 16,384 and doubles to 32,768 after three clean steps. At step 5 the injected overflow makes the scaler skip the optimiser step (weights changed: False) and halve the scale to 16,384. Step 6 repeats step 5's loss, 5.5883, because no update happened, and training goes on. The library defaults are an initial scale of 65,536, a growth factor of 2.0, a backoff factor of 0.5 and a growth interval of 2,000.

**Line by line.**

- `scaler.scale(loss)` multiplies the loss; `scaler.step(optimiser)` unscales the gradients, checks them, and steps only if they are finite.
- `scaler.update()` is where the scale doubles or halves.
- `torch.equal(snapshot, first.detach())` proves whether the weights really moved.

### 5. Five ways to train the same model

Sixty AdamW steps on one fixed batch, in five numeric modes: fp32, bf16 autocast, weights stored in bf16, weights stored in fp16, and fp16 autocast with a scaler. The last column is the share of weights that did not change at all in the final step. A second part looks at Adam's second moment.

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
TOKENS = torch.randint(0, 512, (8, 32), generator=torch.Generator().manual_seed(1))
STEPS = 60


def run(mode, eps=1e-8, lr=1e-3):
    torch.manual_seed(0)
    model = LlamaForCausalLM(CFG)
    if mode == "pure bf16":
        model = model.to(torch.bfloat16)
    if mode == "pure fp16":
        model = model.to(torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, eps=eps)
    scaler = torch.amp.GradScaler("cpu") if mode == "fp16 autocast + scaler" else None
    losses, unchanged = [], 0.0
    for step in range(STEPS):
        before = [p.detach().clone() for p in model.parameters()] if step == STEPS - 1 else None
        dtype = {"bf16 autocast": torch.bfloat16, "fp16 autocast + scaler": torch.float16}.get(mode)
        with torch.autocast("cpu", dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = model(TOKENS, labels=TOKENS).loss
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if before:
            total = sum(p.numel() for p in model.parameters())
            unchanged = sum((a == p.detach()).sum().item() for a, p in zip(before, model.parameters())) / total
    return losses, unchanged


print("60 AdamW steps on one fixed batch of 8 x 32 tokens, learning rate 1e-3")
print("mode                      loss@0   loss@19  loss@59  weights unchanged in the last step")
for mode in ["fp32", "bf16 autocast", "pure bf16", "pure fp16", "fp16 autocast + scaler"]:
    losses, unchanged = run(mode)
    print(f"{mode:<24}  {losses[0]:7.4f}  {losses[19]:7.4f}  {losses[59]:7.4f}  {unchanged:7.1%}")
print(f"1e-8 stored in fp16 is {torch.tensor(1e-8, dtype=torch.float16).item()}, in bf16 {torch.tensor(1e-8, dtype=torch.bfloat16).item():.4g}")
print()
print("Adam's second moment after 5 steps, eps 1e-4 so the pure fp16 run survives: share of entries that are exactly zero")
for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
    torch.manual_seed(0)
    model = LlamaForCausalLM(CFG).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0, eps=1e-4)
    for _ in range(5):
        model(TOKENS, labels=TOKENS).loss.backward()
        opt.step()
        opt.zero_grad()
    v = torch.cat([s["exp_avg_sq"].reshape(-1).float() for s in opt.state.values()])
    print(f"  {name}: {(v == 0).float().mean().item():.1%}")
```

**Reading the output.** fp32, bf16 autocast and fp16 autocast with a scaler end at loss 1.7358, 1.7386 and 1.7363: indistinguishable. Pure bf16 (weights stored in bf16, no master copy) ends at 2.0277, and 34.7 per cent of its weights did not change in the last step, against 12.5 per cent in fp32. Pure fp16 gives NaN from the second step. The reason is Adam's `eps` of 1e-8, which is 0.0 in fp16, so a zero second moment gives a division by zero. With `eps` of 1e-4 it survives but 79.2 per cent of the second moments are exactly zero, against 12.5 per cent in fp32 and bf16, because squaring a small gradient underflows.

**What this does and does not show.** The 12.5 per cent is not an error: it is the embedding rows for tokens that never appear in the batch, which have zero gradient. The comparison is one seed, one fixed batch and a model with 158,016 parameters, so it shows a mechanism, not a ranking of recipes at scale. bf16 autocast on the CPU shows the numerics, not the speed.

**Line by line.**

- `model.to(torch.bfloat16)` makes the weights themselves bf16 and removes the master copy, which is the point of the "pure" modes.
- `torch.autocast(..., enabled=dtype is not None)` is a no-op for the fp32 and pure modes.
- `s["exp_avg_sq"]` is Adam's running mean of squared gradients; its zeros show where squaring underflowed.

### 6. fp8 with one scale, and with a scale per tile

We run a real sentence through SmolLM2-135M, take the hidden states after blocks 1, 15 and 30, and quantise them to e4m3 and back, once with a single scale and once with a scale per group of 64 values. The same is done for one real weight matrix. (DeepSeek-V3 uses groups of 128; this model's width, 576, is not a multiple of 128.)

```python
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
text = "Mixed precision training keeps a master copy of the weights in float32 and does most arithmetic in fewer bits."
with torch.no_grad():
    states = model(**tokenizer(text, return_tensors="pt"), output_hidden_states=True).hidden_states


def to_fp8(x, scale):
    return (x * scale).to(torch.float8_e4m3fn).float() / scale


def per_tensor(x):
    return to_fp8(x, E4M3_MAX / x.abs().max())


def per_tile(x, tile=64):
    rows, cols = x.shape
    parts = x.reshape(rows, cols // tile, tile)
    scale = E4M3_MAX / parts.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    return to_fp8(parts, scale).reshape(rows, cols)


def report(label, x, y):
    flushed = ((y == 0) & (x != 0)).float().mean().item()
    typical = ((y - x).abs() / x.abs().clamp(min=1e-12)).median().item()
    print(f"  {label:<22} flushed to zero {flushed:7.2%}   median relative error {typical:.4f}")


print("SmolLM2-135M hidden states, one 23-token sentence, quantised to fp8 e4m3 and back")
for layer in (1, 15, 30):
    h = states[layer][0]
    print(f"layer {layer}: largest value {h.abs().max().item():,.1f}, median {h.abs().median().item():.2f}, ratio {h.abs().max().item() / h.abs().median().item():,.0f}")
    report("one scale per tensor", h, per_tensor(h))
    report("one scale per 64 values", h, per_tile(h))

w = model.model.layers[10].mlp.gate_proj.weight.detach()
print(f"layer 10 gate_proj weight {tuple(w.shape)}: largest {w.abs().max().item():.2f}, median {w.abs().median().item():.3f}")
report("one scale per tensor", w, per_tensor(w))
report("one scale per 64 values", w, per_tile(w))
```

**Reading the output.** After block 15 the largest hidden-state value is 24,946.1 against a median of 2.09, a ratio of 11,946. With one scale per tensor, 1.46 per cent of the non-zero values are flushed to zero and the median relative error is 0.0257. With a scale per 64 values nothing is flushed and the error is 0.0214. Blocks 1 and 30 have ratios of 43 and 60, so the two methods barely differ (0.0215 against 0.0211 and 0.0220 against 0.0215). For the weight matrix, whose largest value is only 20 times its median, the two methods give 0.0213 and 0.0214: no gain.

**Check by hand.** Scale 448 / 24,946 = 0.01796. A value smaller than half the smallest e4m3 subnormal (0.00098) after scaling becomes zero. That means any original value under 0.00098 / 0.01796 = 0.054 is lost, which is why 1.46 per cent of this tensor goes to zero.

**Line by line.**

- `E4M3_MAX / x.abs().max()` is the per-tensor scale; the `to(torch.float8_e4m3fn)` cast does the rounding.
- `per_tile` reshapes each row into groups of 64 and finds one maximum per group.
- `report` counts values that were non-zero before and zero after, and the median of the relative error.

### Try it yourself

The lab uses the same rounding rules as PyTorch's casts. Its rounding function was checked against PyTorch 2.14.1 on 2,170 values across all five formats with no differences. The defaults, a value of 2e-8 and a scale of 2^0, reproduce block 2's lost gradient: fp16 stores zero. Click "show data" for the table of stored value, value after unscaling, error and status.

<PrecisionRangeLab />

**What each control does.**

- **value** is the number to store; type your own, such as `1e-5`.
- **preset** fills the value box with a case from this chapter.
- **loss scale = 2 to the power** multiplies the value before storing and divides after, as in loss scaling.
- The five bars show each format's range. The solid line is the value; the dashed red line is the scaled value.

**Try it yourself.**

1. Leave the value at 2e-8 and move the loss scale to 10. fp16 now stores 2.0504e-05 and returns 2.0023e-08, an error of 0.12 per cent. Why: 2e-8 times 1,024 is above fp16's smallest number. bf16 stored 2.0023e-08 all along, because its range already reached.
2. Choose the preset 500. fp8 e4m3 reports overflow and stores 448, an error of 10 per cent, while e5m2 stores 512. Why: e4m3's largest value is 448, e5m2's is 57,344, but e5m2 has one fewer mantissa bit.
3. Choose the preset 70000: fp16 overflows to infinity and bf16 stores 70,144. Then type 2 as the value and set the scale to 16: fp16 overflows again and bf16 does not. Why: 2 times 65,536 is 131,072, above fp16's 65,504. A loss scale that is too large overflows, which is why `GradScaler` halves it.

## Production snippets (not run here)

:::warning Not run in this environment
These need a CUDA GPU (fp16 on the GPU, and Transformer Engine needs an H100 or later). The `torch.amp` calls were checked against the PyTorch 2.14 documentation. The Transformer Engine call is copied from its 2.20.2 documentation.
:::

fp16 on a GPU needs the scaler. bf16 does not:

```python
import torch

scaler = torch.amp.GradScaler("cuda")
for batch in loader:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimiser)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimiser)
    scaler.update()
    optimiser.zero_grad()
```

fp8 with Transformer Engine, using its hybrid recipe (e4m3 forward, e5m2 backward):

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
layer = te.Linear(768, 768)
with te.autocast(enabled=True, recipe=recipe):
    output = layer(input_tensor)
```

## Designing with it

1. **Default to bf16 autocast with an fp32 master copy.** It needs no scaler and matches fp32 in block 5.
2. **Use fp16 only with a dynamic scaler,** and unscale before clipping gradients.
3. **Keep sensitive operations in fp32:** softmax, normalisation, the loss, the optimiser state (or at most bf16 moments, as DeepSeek-V3 does).
4. **Never keep the only copy of the weights in a 16-bit format.** Block 5: a third of the weights stopped moving.
5. **Check Adam's epsilon if you ever store the optimiser in 16 bits.** 1e-8 is zero in fp16.
6. **For fp8, scale per block, not per tensor,** and compare the loss curve with bf16 before trusting it. Measure the ratio of a tensor's maximum to its median first, as block 6 does.
7. **Plan memory with the recipe in mind.** Mixed precision does not change the 16 bytes per parameter of model state; it changes activations and speed.

## Where this stands in 2026

:::info Industry view
- **bf16 is the default for large-model training.** Llama 3's reported MFU is for bf16 training.
- **fp8 training is documented and published.** DeepSeek-V3 reports fp8 training with fine-grained scales, and NVIDIA's Transformer Engine ships delayed, current and MXFP8 recipes (release 2.20.2, 5 October 2026).
- **4-bit training has a published recipe.** The NVFP4 paper (March 2026 revision) reports parity with fp8 on a 12B model over 10T tokens, as the authors' own result.
- **Not verified here:** GPU speed-ups, the accuracy of any fp8 or fp4 recipe on your model, and which labs currently train in fp8. This chapter measured numerics on a CPU.
:::

## Common mistakes

1. **Casting a model to fp16 and training it directly.** It is the easy way to save memory. Block 5 shows NaN from step 2 (Adam's `eps`) and 79 per cent zero second moments. Use autocast with a master copy and a scaler.
2. **Using fp16 without a scaler "because it worked on my small model".** Block 3 shows the tiny model loses almost nothing. A deeper model with smaller gradients may. The safe choice costs little.
3. **Summing in low precision.** Totals, means and losses should be accumulated in fp32. Block 2: fp16 stops at 32.0 on a sum whose answer is 100.
4. **Using one fp8 scale for a whole tensor.** One large activation then crushes the rest. Block 6 shows 1.46 per cent of a tensor flushed to zero.
5. **Expecting mixed precision to halve the model-state memory.** The master copy and the optimiser state remain 32-bit: 16 bytes per parameter either way. The saving is in activations and compute.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> What is the largest fp16 number, and what happens to 70,000 when you cast it?</summary>

65,504. It becomes infinity (`inf`), as block 1 prints. bf16 keeps it, as 70,144, because it has fp32's exponent range.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Why does bf16 training not need a loss scale?</summary>

bf16 has 8 exponent bits, like fp32, so its smallest normal number is 1.18e-38 and its largest is 3.39e38. Gradients that fit in fp32 almost always fit in bf16. fp16 has 5 exponent bits and a smallest normal number of 6.1e-5, so small gradients are lost.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> The scaler's scale is 65,536 and one gradient is 1.5. What happens, and what is the scale next?</summary>

1.5 times 65,536 is 98,304, above fp16's largest number 65,504, so it overflows to infinity. The scaler finds a non-finite gradient, skips the optimiser step and halves the scale to 32,768. At that scale the gradient is 49,152, which fits, so the next step runs. After 2,000 clean steps the scale doubles again.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> A 7-billion-parameter model is trained with bf16 weights, bf16 gradients and bf16 Adam moments (no master copy). How many bytes of model state is that, and why would you not do it?</summary>

2 + 2 + 4 = 8 bytes per parameter, so 56 GB, against 112 GB for the 16-byte recipe. You would not, because an update smaller than half a bf16 step is lost. Block 5 found 34.7 per cent of the weights unchanged in the last step against 12.5 per cent for fp32, and a final loss of 2.03 against 1.74.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> After block 15 of SmolLM2-135M the largest hidden-state value is 24,946. With one e4m3 scale for the tensor, which values are flushed to zero? Check against the printed 1.46 per cent.</summary>

The scale is 448 / 24,946 = 0.01796. e4m3's smallest subnormal number is 2^-9 = 0.00195, and anything under half of that, 0.00098, rounds to zero. So original values with magnitude below 0.00098 / 0.01796 = 0.054 are lost. Given a median of 2.09, a small share of the values falls below 0.054, which is the printed 1.46 per cent. A scale per 64 values lets each group use its own maximum, so nothing is lost.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> Pure fp16 training gave NaN from step 2 with Adam's default settings. Explain the mechanism, and why changing `eps` to 1e-4 is not a real fix.</summary>

Adam divides the first moment by the square root of the second plus `eps`. In fp16, `eps = 1e-8` is stored as 0.0 (smaller than 5.96e-8), and a gradient that squares to less than 6e-8 also gives a zero second moment, so the division is 0/0 or x/0 and the result is NaN. Setting `eps` to 1e-4 avoids the NaN, but 79.2 per cent of second moments are still exactly zero, so Adam takes steps of about `lr` times the first moment divided by `eps`, a different and much larger step than the one Adam is meant to take. The real fix is a master copy and fp32 optimiser state.

</details>

## Go deeper

All sources were opened on 7 October 2026.

- [Mixed Precision Training (arXiv 1710.03740)](https://arxiv.org/abs/1710.03740): the fp32 master copy, loss scaling and fp32 accumulation; the figures about gradients below 2^-24.
- [FP8 Formats for Deep Learning (arXiv 2209.05433)](https://arxiv.org/abs/2209.05433): e4m3 and e5m2, their largest values (448 and 57,344) and the per-tensor scaling idea.
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437): fp8 training, tile and block scaling, bf16 optimiser moments, the 0.25 per cent loss error.
- [PyTorch 2.14 automatic mixed precision](https://docs.pytorch.org/docs/2.14/amp.html): autocast op lists, defaults, `GradScaler` defaults.
- [NVIDIA Transformer Engine: using FP8 and FP4](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html): the delayed, current and MXFP8 recipes, the hybrid format and the API.
- [Pretraining Large Language Models with NVFP4 (arXiv 2509.25149)](https://arxiv.org/abs/2509.25149): abstract read only.
- [Massive Activations in Large Language Models (arXiv 2402.17762)](https://arxiv.org/abs/2402.17762): abstract read only.
- [The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/abs/2407.21783): BF16 MFU in Table 4 and the FP8 inference note.

## Check yourself

- I can say what the exponent and mantissa bits of a format control, and read a format's range and step size.
- I can explain why bf16 does not need a loss scale and fp16 does.
- I can explain what a master copy protects against and what recipe costs 16 bytes per parameter.
- I can predict when an addition is lost to rounding, and why sums are kept in fp32.
- I can describe how a dynamic loss scaler grows, backs off and skips a step.
- I can say why fp8 needs a scale per block, and measure an outlier ratio to decide.

## Where to go next

Next chapter: [mixture of experts](/docs/llm-engineering/mixture-of-experts), where only a few experts run for each token and precision choices such as router stability matter. A related chapter: [quantisation for inference](/docs/llm-engineering/quantisation-for-inference), which applies the same ideas to serving a trained model.
