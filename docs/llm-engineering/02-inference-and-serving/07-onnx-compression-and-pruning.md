---
id: llme-compression
title: "ONNX, Model Compression and Pruning"
sidebar_label: "7 · ONNX, compression and pruning"
sidebar_position: 7
slug: /llm-engineering/onnx-compression-and-pruning
description: "Export a model to ONNX and run it with ONNX Runtime, then prune it three ways and measure what the zeros actually buy: accuracy, size and, honestly, speed. Includes how pruning compares with quantisation, low-rank approximation and distillation."
tags: [onnx, onnxruntime, pruning, sparsity, low-rank, distillation, model-compression, inference]
---

import Infographic from '@site/src/components/Infographic';
import PruningLab from '@site/src/components/viz/PruningLab';

**In one line.** ONNX gives a model a portable, optimisable form that many runtimes can execute, and pruning removes weights to shrink it, but zeros only make inference faster when the hardware or the shape of the matrix lets the computation skip them, which is rarer than the headline sparsity numbers suggest.

:::note Not from a lecture
This chapter is written for this site from the ONNX Runtime, PyTorch, NVIDIA and Hugging Face pages listed under Further reading, opened on 2 October 2026. Every number below is printed by the code in this chapter, run on a CPU with PyTorch 2.14.1, ONNX 1.23.1, ONNX Runtime 1.30.0 and onnxscript 0.7.2.
:::

## The idea in plain words

Two separate jobs share this chapter, and keeping them apart avoids most confusion.

**Job one: make the model portable and let a runtime optimise it.** A PyTorch model is Python code. **ONNX** (Open Neural Network Exchange) is a file format for the computation graph underneath: a list of operations (matrix multiply, add, reshape) with the weights attached. Any runtime that reads ONNX can run it, without Python or PyTorch. **ONNX Runtime** (ORT) is one such runtime. It reads the graph, rewrites it (merging small operations into fused ones, removing redundant work) and runs it on a chosen hardware back end through an *execution provider*.

**Job two: make the model smaller.** Several families of techniques exist, and they compress different things.

| Technique | What it removes | Typical gain | Catch |
| --- | --- | --- | --- |
| [Quantisation](/docs/llm-engineering/quantisation-for-inference) | Precision of each weight | Memory and bandwidth, often speed | Needs kernels for the format |
| **Unstructured pruning** | Individual weights, wherever they are smallest | File size after compression | A pruned matrix is still a dense matrix unless a sparse kernel runs |
| **Structured pruning** | Whole units, rows, heads or layers | Real speed and memory | Bigger accuracy hit, needs fine-tuning |
| **2:4 semi-structured pruning** | Two of every four weights, in a fixed pattern | Up to a stated speedup on specific GPUs | Needs NVIDIA Ampere or newer sparse hardware |
| **Low-rank approximation** | Redundant directions in a weight matrix | Fewer parameters and multiplications | Needs a real low-rank structure |
| **Distillation** | Nothing: trains a smaller model to imitate a larger one | A different, smaller architecture | A training run |

The chapter's running theme is the third column of that table, because the honest answer to "does pruning speed up inference?" is "only some kinds, only on some hardware".

<Infographic src="/img/llme/onnx-compression-and-pruning-pipeline.svg" alt="A pipeline from a PyTorch model through ONNX export and ONNX Runtime graph optimisation to a runtime session, with a warning box about exporting with an example batch of one." caption="Export, optimise, run. The warning box is something the code below hit: a model exported from an example with batch size 1 failed at batch 4, and exporting from batch 2 fixed it." />

<Infographic src="/img/llme/onnx-compression-and-pruning-sparsity.svg" alt="Accuracy against sparsity for global magnitude pruning before and after fine-tuning, with the timings of dense, zero-filled, CSR and row-removed matrix multiplies." caption="What the zeros bought on a digits classifier: accuracy survives 90 per cent sparsity after fine-tuning, but the matmul only got faster when rows were physically removed." />

## How it works

### Exporting to ONNX

`torch.onnx.export` traces the model into a graph and writes it out. In PyTorch 2.14 the default path is the one built on `torch.export` (`dynamo=True`, the default since PyTorch 2.9), which returns an `ONNXProgram` you can save. The older TorchScript-based exporter, selected with `dynamo=False`, is deprecated. Dynamic input sizes are declared with `dynamic_shapes` on the new exporter and `dynamic_axes` on the old one. The new exporter needs the `onnxscript` package, which I installed into the environment for this chapter.

One trap, found by running it rather than reading it: **an example input whose batch size is 1 can silently freeze the batch dimension**, even when you declare it dynamic. The code below exports a four-layer transformer encoder from an example of batch 1 and then runs it at batch 4. ONNX Runtime raises an `InvalidArgument` error. Exporting from an example of batch 2 gives a graph that runs at batches 1, 4 and 8 and at several sequence lengths. This is my observation on these library versions, not a statement from the documentation, so test your own dynamic axes at more than one size before you ship.

### ONNX Runtime graph optimisation

ORT describes three levels of graph optimisation. **Basic** rewrites preserve semantics and remove redundant work: constant folding, removing identity nodes, fusing a convolution with its batch norm. **Extended** adds larger fusions after the graph is split across providers, such as GELU, layer normalisation and "skip layer normalisation" (a fused bias, skip connection and layer norm). **Layout** changes how tensors sit in memory for faster kernels on CPU. You choose the level with `SessionOptions.graph_optimization_level` and can write the optimised graph to disk with `optimized_model_filepath`; the documentation warns that an offline-optimised model should be run on the same hardware and options it was optimised for.

The code prints what the optimiser did to the encoder: 197 nodes become 181, and three new fused operations appear (`SkipLayerNormalization`, `FusedMatMul` and `Split`). The speed gain from the levels themselves turned out small on this CPU; most of the gain over PyTorch eager came from ORT itself.

**Execution providers** are chosen from an ordered list. `['CUDAExecutionProvider', 'CPUExecutionProvider']` means "run each node on CUDA if it can, otherwise on the CPU". ORT's provider page names CUDA, TensorRT, ROCm, OpenVINO, CoreML, DirectML, Qualcomm QNN and many more.

### Pruning, in the three forms

**Unstructured magnitude pruning** zeroes the weights with the smallest absolute values. PyTorch's `torch.nn.utils.prune` implements it as a mask: it keeps the original weights under `weight_orig`, stores a binary `weight_mask`, and computes `weight` as their product; `prune.remove` makes the pruning permanent. `global_unstructured` ranks weights across all layers together, so layers end up with different sparsity.

**Structured pruning** (`ln_structured`) removes whole rows by their norm. To get a speedup you then build a physically smaller layer from the surviving rows, which is what the code does.

**2:4 semi-structured pruning** keeps two nonzero values in every block of four. NVIDIA's description: in each contiguous block of four values, two must be zero, giving 50 per cent sparsity. PyTorch's semi-structured sparsity tutorial states the hardware requirement as an NVIDIA GPU with compute capability 8.0 or newer. On an A100, that tutorial reports a 1.3x speedup over the fp16 baseline, up to 2x with `torch.compile`, for a sparsified BERT, with F1 on SQuAD v1.1 of 86.49 against 86.93 dense after fine-tuning. NVIDIA's own post, from July 2021, reports up to 20 per cent faster inference for a sparse ResNeXt-101 on an A100 at larger batch sizes. Those two real measurements sit far below the "2x" that the pattern suggests on paper, which is the first hint that sparsity and speed are different things.

### Low-rank and distillation

A weight matrix with a hidden low-rank structure can be replaced by two thin matrices with a truncated singular value decomposition (SVD). This is the same idea that LoRA uses to *add* a low-rank update; here it is used to *replace* a matrix. **Distillation** is a different move: instead of editing the model, train a smaller one on the larger model's output distribution, as in Hinton, Vinyals and Dean's 2015 paper.

## A real system that works this way

Two real, documented examples of this chapter's ideas in production tooling. **Hugging Face Optimum** provides an integration with ONNX Runtime so that Transformers models can be exported to ONNX and run through ORT classes, with conceptual guides on quantisation and graph optimisation ([Optimum overview](https://huggingface.co/docs/optimum/onnxruntime/overview)). **Triton Inference Server** (now Dynamo-Triton) lists an ONNX Runtime backend among its official backends, so an ONNX file can be hosted behind a Triton endpoint ([backend guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/README.html)). And for sparsity, NVIDIA's Ampere sparsity post describes the whole workflow this chapter simulates: start from a dense pretrained network, prune to the 2:4 pattern by removing low-magnitude weights, then train to recover accuracy, and reports that networks such as ResNet-50 and BERT-Large kept their baseline accuracy.

## Code you can run

The first block exports a four-layer transformer encoder to ONNX, shows the batch-1 trap, then optimises and times the graph.

```python
import collections
import contextlib
import io
import os
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ort.set_default_logger_severity(4)
torch.manual_seed(0)
torch.set_num_threads(4)

encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, dropout=0.0),
    num_layers=4,
    enable_nested_tensor=False,
).eval()
workdir = tempfile.mkdtemp()


def export(example_batch, name):
    path = os.path.join(workdir, name)
    with contextlib.redirect_stdout(io.StringIO()):
        program = torch.onnx.export(
            encoder,
            (torch.randn(example_batch, 64, 256),),
            dynamo=True,
            dynamic_shapes=({0: "batch", 1: "seq"},),
        )
    program.save(path)
    return path


def session(path, level, optimised_path=None):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    opts.intra_op_num_threads = 4
    if optimised_path:
        opts.optimized_model_filepath = optimised_path
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def run(sess, x):
    return sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]


bad = session(export(1, "from_batch_1.onnx"), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
for batch in (1, 4):
    x = torch.randn(batch, 48, 256)
    try:
        run(bad, x)
        print(f"exported from batch 1, run at batch {batch}: ok")
    except Exception as exc:
        print(f"exported from batch 1, run at batch {batch}: {type(exc).__name__}")

path = export(2, "from_batch_2.onnx")
graph = onnx.load(path).graph
ops = collections.Counter(node.op_type for node in graph.node)
print("exported from batch 2:", len(graph.node), "nodes,", len(ops), "distinct ops;", ops.most_common(3))

opt_path = os.path.join(workdir, "encoder_opt.onnx")
sess_off = session(path, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
sess_all = session(path, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, opt_path)
fused = collections.Counter(node.op_type for node in onnx.load(opt_path).graph.node)
print("after ORT_ENABLE_ALL:", sum(fused.values()), "nodes; new ops:", sorted(set(fused) - set(ops)))

for shape in [(4, 48, 256), (1, 16, 256), (8, 100, 256)]:
    z = torch.randn(*shape)
    with torch.no_grad():
        ref = encoder(z).numpy()
    print("shape", shape, "max abs difference vs torch:", f"{np.abs(ref - run(sess_all, z)).max():.1e}")


def bench(fn, reps=20, rounds=7):
    for _ in range(5):
        fn()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - start) / reps * 1000)
    return best


x = torch.randn(4, 48, 256)
with torch.no_grad():
    t_torch = bench(lambda: encoder(x))
t_off = bench(lambda: run(sess_off, x))
t_all = bench(lambda: run(sess_all, x))
print(f"batch 4 x 48 tokens, best of 7 rounds: torch eager {t_torch:.1f} ms | ort without optimisation {t_off:.1f} ms | ort all optimisations {t_all:.1f} ms")
```

On this machine the batch-1 export failed at batch 4 with `InvalidArgument`; the batch-2 export produced 197 nodes of 15 distinct operations; `ORT_ENABLE_ALL` left 181 nodes and added `FusedMatMul`, `SkipLayerNormalization` and `Split`; the largest difference from PyTorch across three shapes was about 3e-06, which is float rounding, not a different model.

The timing line is the honest surprise. Over five runs of this block as printed, ONNX Runtime was steady: 3.8 to 4.7 ms with all optimisations and 3.9 to 4.2 ms with none. Eager PyTorch was **bimodal**: about 2.1 ms in three runs and 7.5 to 7.9 ms in two, and earlier runs with a simpler timing loop showed the same two speeds (2.3 to 7.7 ms). The laptop was shared with other jobs (load average about 6 on 12 cores). So in some runs ORT was about 1.8 times slower than eager and in others about 1.9 times faster. I do not know why eager PyTorch flips between two speeds here, and I will not claim that ONNX Runtime is faster than PyTorch; the defensible statements are that ORT's time was steady and that its optimisation levels made no consistent difference on this model and CPU. The lesson is the method: benchmark on the target hardware, take the best of several rounds, repeat the whole run, and distrust any single comparison.

The second block trains a small digits classifier (a 64-512-512-10 network), then prunes it three ways, times the matrix multiplies, and compares with simulated int8 and int4 weights and with low-rank replacement. With 540 test images, one image is worth 0.0019 accuracy, so differences below that are noise.

```python
import copy
import time
import warnings

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune

torch.manual_seed(0)
torch.set_num_threads(4)

digits = load_digits()
x_tr, x_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.3, random_state=0, stratify=digits.target)
x_tr, x_te = torch.tensor(x_tr, dtype=torch.float32), torch.tensor(x_te, dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)


def make():
    return nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))


def fit(model, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    for _ in range(epochs):
        order = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()


def accuracy(model):
    with torch.no_grad():
        return (model(x_te).argmax(1) == y_te).float().mean().item()


def weights(model):
    return [(m, "weight") for m in model if isinstance(m, nn.Linear)]


dense = make()
fit(dense, 30)
print(f"dense accuracy {accuracy(dense):.4f}")

print("sparsity  no fine-tune  fine-tuned 8 epochs")
for sparsity in (0.5, 0.8, 0.9, 0.95, 0.98):
    model = copy.deepcopy(dense)
    prune.global_unstructured(weights(model), pruning_method=prune.L1Unstructured, amount=sparsity)
    before = accuracy(model)
    fit(model, 8, lr=5e-4)
    zeros = sum((m.weight == 0).sum().item() for m, _ in weights(model))
    total = sum(m.weight.numel() for m, _ in weights(model))
    layer = "/".join(f"{(m.weight == 0).float().mean():.2f}" for m, _ in weights(model))
    print(f"{zeros / total:>8.3f}  {before:>12.4f}  {accuracy(model):>19.4f}   per-layer zeros {layer}")


def two_four(model):
    out = copy.deepcopy(model)
    for m, _ in weights(out):
        if m.in_features % 4:
            continue
        w = m.weight.data.reshape(-1, 4)
        keep = w.abs().topk(2, dim=1).indices
        mask = torch.zeros_like(w).scatter_(1, keep, 1.0)
        m.weight.data = (w * mask).reshape(m.weight.shape)
    return out


pat = two_four(dense)
before = accuracy(pat)
print(f"2:4 pattern on the dense model: accuracy {before:.4f}, fraction zero {(pat[2].weight == 0).float().mean():.3f}")

structured = copy.deepcopy(dense)
prune.ln_structured(structured[0], "weight", amount=0.5, n=2, dim=0)
prune.remove(structured[0], "weight")
alive = (structured[0].weight.abs().sum(1) > 0).nonzero().squeeze(1)
small = nn.Sequential(nn.Linear(64, len(alive)), nn.ReLU(), nn.Linear(len(alive), 512), nn.ReLU(), nn.Linear(512, 10))
small[0].weight.data = structured[0].weight.data[alive].clone()
small[0].bias.data = structured[0].bias.data[alive].clone()
small[2].weight.data = structured[2].weight.data[:, alive].clone()
small[2].bias.data = structured[2].bias.data.clone()
small[4].load_state_dict(structured[4].state_dict())
acc_cut = accuracy(small)
fit(small, 8, lr=5e-4)
print(f"structured: removed {512 - len(alive)} of 512 first-layer units, physically smaller layer; accuracy {acc_cut:.4f} -> {accuracy(small):.4f} after fine-tuning")


def timed(fn, reps=100, rounds=5):
    for _ in range(20):
        fn()
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - start) / reps * 1000)
    return best


size = 2048
x = torch.randn(32, size)
w = torch.randn(size, size)
t_dense = timed(lambda: x @ w.T)
mask = (torch.rand(size, size, generator=torch.Generator().manual_seed(2)) > 0.9).float()
w90 = w * mask
t_masked = timed(lambda: x @ w90.T)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    csr = w90.to_sparse_csr()
t_csr = timed(lambda: torch.sparse.mm(csr, x.T))
w_half = w[: size // 2]
t_half = timed(lambda: x @ w_half.T)
print(f"matmul 32 x {size} by {size} x {size}: dense {t_dense:.3f} ms | 90% zeros in a dense tensor {t_masked:.3f} ms | 90% zeros as CSR {t_csr:.3f} ms | half the rows removed {t_half:.3f} ms")


def int8_roundtrip(model, bits=8):
    out = copy.deepcopy(model)
    levels = 2 ** (bits - 1) - 1
    for m, _ in weights(out):
        scale = m.weight.data.abs().amax(dim=1, keepdim=True) / levels
        m.weight.data = torch.round(m.weight.data / scale).clamp(-levels, levels) * scale
    return out


print(f"int8 weights (per-row absmax, simulated): accuracy {accuracy(int8_roundtrip(dense)):.4f}; int4: {accuracy(int8_roundtrip(dense, 4)):.4f}; dense {accuracy(dense):.4f}")

u, s, vt = torch.linalg.svd(dense[2].weight.data, full_matrices=False)
print("rank  params of 512x512 layer  relative weight error  accuracy")
for r in (256, 64, 16, 8, 4, 2, 1):
    approx = (u[:, :r] * s[:r]) @ vt[:r]
    lr_model = copy.deepcopy(dense)
    lr_model[2].weight.data = approx
    err = (approx - dense[2].weight.data).norm() / dense[2].weight.data.norm()
    print(f"{r:>4}  {r * 1024:>22}  {err:>21.3f}  {accuracy(lr_model):.4f}")
print("full rank params", 512 * 512)
```

Read the outputs in order.

**Accuracy against sparsity.** The dense network scores 0.9759. At 0.50 sparsity, accuracy is unchanged. At 0.80 it drops to 0.9648 before fine-tuning and recovers to 0.9815. At **0.90 it falls to 0.7685 with no fine-tuning, and 8 epochs of fine-tuning bring it back to 0.9759**, the dense score. At 0.95 it is 0.3778 then 0.9481. At 0.98 it is 0.3148 then 0.2889: broken. The per-layer zeros explain why: global ranking pruned the middle and last layers to 0.99 zero, leaving almost no path through the network, which no short fine-tune repairs. Global magnitude pruning does not respect layer importance.

**The 2:4 pattern and structured pruning.** Applying the 2:4 rule to the dense network with no fine-tuning gave 0.9778, effectively the dense accuracy on this easy task (a difference of one image). Removing half the first-layer units (256 of 512) cost accuracy before fine-tuning (0.9407) and recovered after (0.9796).

**Timing: the honest part.** A 32 by 2048 times 2048 by 2048 matrix multiply took between 0.38 and 0.85 ms dense across six runs on a shared machine, so read the ratios. Setting 90 per cent of the weights to zero **inside a dense tensor made no useful difference** (between 0.94 and 1.2 times the dense time), because the dense kernel multiplies the zeros anyway. Converting the same matrix to a compressed sparse row (CSR) tensor, the generic sparse format, was about 20 to 45 times *slower*, because 90 per cent sparsity is not sparse enough for the generic kernel to beat a tuned dense one on a CPU. Only **removing half the rows** made it faster, to about 0.4 to 0.6 of the dense time, because the matrix is simply smaller. The 2:4 pattern's speedup needs the sparse tensor cores of an Ampere-class GPU, which I cannot run here, so the code shows the accuracy of the pattern and not a speed.

**Neighbouring techniques.** Simulated int8 weights (per-row absmax) kept 0.9759 and int4 kept 0.9778, within one image of dense. Low-rank replacement of the middle layer kept accuracy down to rank 16, a 16-fold cut in that layer's parameters (16,384 against 262,144), even though the relative weight error was 0.569, then fell to 0.9519 at rank 8 and 0.6519 at rank 4. Weight error and task error are not the same number; this layer was heavily over-parameterised for ten classes.

<PruningLab />

The lab's default (0.90 unstructured) reproduces the 0.7685 and 0.9759 above.

## Designing with it

1. **Decide the goal before the technique.** Smaller file, lower memory, lower latency and higher throughput are different goals; the table at the top says which technique serves which.
2. **Prefer quantisation and structured changes for speed.** Zeros in a dense tensor are free storage in a sparse file format and no speed. Removing rows, heads or layers, or lowering precision with a kernel that supports it, is where speedups come from on commodity hardware.
3. **Use 2:4 only if you own the hardware.** It needs NVIDIA Ampere or newer, and the published speedups (1.3x, up to 2x with `torch.compile`, in the PyTorch tutorial; up to 20 per cent in NVIDIA's ResNeXt example) are below the headline figure.
4. **Always fine-tune after pruning** and watch layer-wise sparsity when you prune globally.
5. **Test dynamic shapes at several sizes after an ONNX export,** including a batch of more than one.
6. **Check the parity** between the original and the exported model on real inputs, as the code does, before comparing speed.
7. **Distil when you need a different architecture,** not just a thinner copy of the same one.

## Where this stands in 2026

:::info Industry view
Read the documentation each tool publishes. ONNX Runtime's own guidance recommends dynamic quantisation for transformer-based models and static quantisation for CNNs. The serving engines in the [serving engines](/docs/llm-engineering/serving-engines) chapter document quantisation formats at length, and I did not find unstructured sparsity among the features on the pages I opened for them. 2:4 sparsity is documented as an NVIDIA Ampere-and-newer feature with published speedups well under the theoretical 2x. Distillation changes the architecture, so its speed gain does not depend on a sparse kernel. Treat any claim that a sparse model is "N times faster" as unproven until it names the hardware, the kernel and the batch size.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> You zero 90 per cent of a matrix's weights and the matmul does not get faster. Why?</summary>

A dense matrix multiply does the same work whatever the values are. The code shows a dense tensor with 90 per cent zeros taking the same time as the original. Speed needs a kernel that skips zeros (hardware sparse support for 2:4, or a sparse format at much higher sparsity) or a smaller matrix.

</details>

<details>
<summary><strong>Q2.</strong> Why was a CSR copy of the 90 per cent sparse matrix roughly 20 to 45 times slower than the dense one?</summary>

Generic sparse kernels pay for index lookups and irregular memory access. At 90 per cent sparsity there is still one weight in ten to multiply, and a tuned dense kernel on a CPU wins. Real sparse speedups usually need far higher sparsity or hardware support.

</details>

<details>
<summary><strong>Q3.</strong> Why did 98 per cent global pruning fail even after fine-tuning?</summary>

Global ranking compared weights across layers and removed 99 per cent of the middle and last layers, almost cutting the network in two. Fine-tuning cannot rebuild paths that no longer exist. Prune per layer, or stay at a sparsity that leaves every layer connected.

</details>

<details>
<summary><strong>Q4.</strong> An ONNX model exported with dynamic axes fails at batch 4. What is the first thing to test?</summary>

Whether the example input used for export had a dimension of size 1, which can freeze that dimension. Re-export from a batch of 2 and test several batch sizes and sequence lengths.

</details>

<details>
<summary><strong>Q5.</strong> What does ORT's extended optimisation level add over basic?</summary>

Larger node fusions applied after the graph is partitioned across providers, such as GELU, layer normalisation and skip layer normalisation. In the run above it turned 197 nodes into 181 and introduced fused operations, but gave no measurable timing change on this CPU.

</details>

<details>
<summary><strong>Q6.</strong> When is distillation a better choice than pruning?</summary>

When you want a different, smaller architecture that is fast on any hardware, and you can afford a training run. Pruning edits an existing model and keeps its architecture, so its speed gain depends on the hardware and on the structure of what you remove.

</details>

## Further reading

All opened on 2 October 2026.

- ONNX Runtime: [graph optimisations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html), [execution providers](https://onnxruntime.ai/docs/execution-providers/), [quantisation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html); version on the [package page](https://pypi.org/project/onnxruntime/).
- PyTorch: [`torch.onnx` documentation](https://docs.pytorch.org/docs/2.14/onnx.html), [pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html), [semi-structured sparsity tutorial](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html).
- NVIDIA: [Accelerating inference with sparsity using Ampere and TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) (July 2021).
- Hugging Face: [Optimum ONNX Runtime overview](https://huggingface.co/docs/optimum/onnxruntime/overview).
- Triton: [backend guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend/README.html).
- Hinton, Vinyals and Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015.

## Check yourself

- I can export a PyTorch model to ONNX with dynamic shapes and verify it at more than one input size.
- I can describe what ONNX Runtime's basic, extended and layout optimisations do, and measure whether they help.
- I can explain the difference between unstructured, structured and 2:4 pruning.
- I can show with a measurement why zeros in a dense tensor do not speed up a matmul.
- I can say which hardware 2:4 sparsity needs and what speedups were actually reported.
- I can compare pruning with quantisation, low-rank approximation and distillation, and say which goal each serves.
