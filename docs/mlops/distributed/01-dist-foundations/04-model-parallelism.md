---
id: dist-model-parallel
title: "Model Parallelism"
sidebar_label: "Model parallelism"
sidebar_position: 4
slug: /mlops/distributed/model-parallelism
description: "Split the model, not the data: pipeline parallelism by layers and tensor parallelism inside a layer, the pipeline bubble (S-1)/(m+S-1), and a real two-process pipeline checked against a single process."
tags: [model-parallelism, pipeline-parallelism, tensor-parallelism, pipeline-bubble, micro-batches, gpipe, megatron-lm]
---

import Infographic from '@site/src/components/Infographic';
import PipelineBubbleLab from '@site/src/components/viz/PipelineBubbleLab';

**In one line.** When the model is too big for one device, cut it up, between layers or inside a layer, and keep every device busy with micro-batches so that the pipeline does not sit idle.

Built from the course lecture "dml-s4-model-parallelism" (Lecture Library series), extended with runnable measurements.

## The idea in plain words

Think of a car factory. One worker could build a whole car, but the factory has an assembly line: one station fits the engine, the next the doors, the next the paint. Each station only needs the tools and parts for its own job, so no station needs a warehouse big enough for the whole car. That is the point of **model parallelism**: no device needs to hold the whole model.

The line has a flaw at the start and at the end. When the first car enters, the later stations stand idle waiting for something to work on. When the last car leaves, the early stations are idle again. The remedy is to feed the line many small batches of cars one after another so that, in the middle of the run, every station is busy. The idle time at the ends is the **pipeline bubble**, and the lecture gives its size in one formula.

There is a second way to cut a model. Instead of giving each device whole layers, give each device a slice of every layer. For a big matrix multiplication this is like asking two people to compute half of the columns each and then adding their partial answers. That is **tensor parallelism**, and the arithmetic of why it works is a short numpy check below.

<Infographic src="/img/dist/model-parallelism-splitting.svg" alt="Pipeline parallelism puts layers 1 to 6, 7 to 12, 13 to 18 and 19 to 24 on four stages; tensor parallelism slices one MLP block across two devices and adds the partial outputs; a table shows the split matches the unsplit layer to 1.78e-15" caption="Two ways to cut a model. The checks come from block 2 below." />

<Infographic src="/img/dist/model-parallelism-pipeline-bubble.svg" alt="A grid of four stages by eleven ticks showing micro-batches one to eight flowing through with twelve idle cells, a table of bubble fractions for four and eight stages, and a real two-process pipeline whose gradients match a single process" caption="The pipeline bubble for four stages and eight micro-batches: 3/11 = 0.273. Figures come from blocks 1 and 3 below." />

## How it works

Too big for one GPU? Split the model across devices, and mind the pipeline bubble.

### Split layers or tensors

Different layers, or slices of the tensors, live on different devices, and activations flow between them. A 24 GB model over 4 GPUs is 6 GB each.

### Idle time

Pipelining stages causes a bubble while the pipe fills and drains. The fraction is (S - 1) / (m + S - 1) for S stages and m micro-batches. More micro-batches shrink it.

:::tip

**Worked.** S = 4 and m = 8 give a bubble of 3/11 = 0.273, about 27 percent idle.

:::

### What the lecture leaves implicit

:::note Beyond the lecture
**Where the formula comes from.** Split each batch into m micro-batches and send them through S stages, each taking one tick per micro-batch. Stage s starts micro-batch j at tick s + j. The whole forward pass lasts m + S - 1 ticks, but each stage is busy for only m of them, so a fraction (S - 1) / (m + S - 1) of every stage's time is idle. Block 1 draws the grid and counts: 12 idle cells out of 44. The backward pass has the same shape, so the fraction is unchanged.

**What micro-batches cost.** In the plain schedule where every micro-batch is run forward before any backward, a stage must keep the activations of all m micro-batches until its backward pass arrives, so memory grows with m. That is the lecture's "trade memory for utilisation". Recomputing activations during the backward pass (re-materialisation) trades compute for memory instead.

**Tensor parallelism and why the cut falls where it does.** A transformer's feed-forward block computes GELU(XA) B. Cut A by columns: each device computes its own half of the hidden units, and because the nonlinearity acts on each hidden unit separately, no communication is needed before it. Cut B by rows to match: each device multiplies its hidden half by its half of B, giving a partial output, and one sum (an all-reduce) combines them. If you cut A by rows instead, the nonlinearity would need the summed pre-activation first, and skipping that sum gives a wrong answer (block 2 measures it at 3.97 off). The all-reduce payload is only the activations, batch times model width, independent of the hidden width.

**The two cuts compose.** Tensor parallelism splits inside a layer, so it is best kept on the fastest links available; pipeline parallelism sends only activations across each cut, so it tolerates slower links. The Megatron-LM authors describe their intra-layer approach as orthogonal to pipeline model parallelism, and the two can be combined with data parallelism as well.
:::

## A real system that works this way

**GPipe.** Huang and colleagues partition a network of sequential layers over accelerators and split each mini-batch into micro-batches, reporting almost linear speed-up when a model is partitioned over several accelerators. They train a 557-million-parameter AmoebaNet to 84.4 percent ImageNet top-1 accuracy and a 6-billion-parameter, 128-layer Transformer for multilingual translation. The paper names the idle time "bubble overhead", gives it as O((K - 1) / (M + K - 1)) for K partitions and M micro-batches, and reports that in their experiments the overhead was negligible when M is at least 4 x K, partly because recomputation in the backward pass can be scheduled earlier. At K = 4 that is M = 16, where the formula in block 1 still gives 0.158, so "negligible" is an experimental observation on their hardware, not a property of the formula.

**Megatron-LM.** Shoeybi and colleagues describe a simple intra-layer model-parallel approach that can be implemented by inserting a few communication operations in native PyTorch. They converge transformer models of up to 8.3 billion parameters on 512 GPUs, sustaining 15.1 PetaFLOPs across the whole application with 76 percent scaling efficiency.

## Code you can run

Three blocks. Blocks 1 and 2 use numpy; block 3 runs two real processes over PyTorch's Gloo backend on the CPU (Python 3.14, numpy 2.5.3, torch 2.14.1).

### 1. The pipeline bubble, counted and from the formula

```python
def timeline(stages, micro):
    ticks = micro + stages - 1
    grid = [["." for _ in range(ticks)] for _ in range(stages)]
    for s in range(stages):
        for j in range(micro):
            grid[s][s + j] = str(j + 1) if j < 9 else "+"
    return grid


def bubble(stages, micro):
    return (stages - 1) / (micro + stages - 1)


S, m = 4, 8
grid = timeline(S, m)
print(f"forward pass, {S} stages, {m} micro-batches, one tick per micro-batch per stage")
for s, row in enumerate(grid):
    print(f"stage {s}  " + " ".join(row))
ticks = len(grid[0])
busy = sum(cell != "." for row in grid for cell in row)
print("ticks:", ticks, "busy cells:", busy, "of", S * ticks)
print("idle fraction from the grid:", round(1 - busy / (S * ticks), 4))
print("formula (S-1)/(m+S-1):", round(bubble(S, m), 4), "=", f"{S - 1}/{m + S - 1}")

print()
print("micro-batches  bubble (S=4)  bubble (S=8)  micro-batches in flight on stage 0")
for micro in (1, 2, 4, 8, 16, 32, 64):
    print(f"{micro:>13}  {bubble(4, micro):12.3f}  {bubble(8, micro):12.3f}  {micro:>34}")
print("micro-batches for a bubble under 10% at S=4:", next(k for k in range(1, 200) if bubble(4, k) < 0.10))
print("GPipe rule of thumb m >= 4S at S=4 gives m = 16, bubble", round(bubble(4, 16), 3))
```

The grid for 4 stages and 8 micro-batches has 11 ticks and 32 busy cells of 44, an idle fraction of 0.2727, equal to 3/11 from the formula. The table shows the cost of depth: at 8 stages 8 micro-batches leave 0.467 idle, while 64 micro-batches bring it to 0.099. At 4 stages it takes 28 micro-batches to get below 10 percent, and the last column shows that the price in the plain schedule is m micro-batches of activations on stage 0.

### 2. Tensor parallelism: split a layer and get the same answer

A feed-forward block with 8 inputs, 16 hidden units and 8 outputs, and 6 example rows.

```python
import numpy as np

rng = np.random.default_rng(0)
batch, d_model, d_ff = 6, 8, 16
X = rng.normal(size=(batch, d_model))
A = rng.normal(size=(d_model, d_ff))
B = rng.normal(size=(d_ff, d_model))


def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


full = gelu(X @ A) @ B

A1, A2 = A[:, : d_ff // 2], A[:, d_ff // 2:]
B1, B2 = B[: d_ff // 2], B[d_ff // 2:]
partial1 = gelu(X @ A1) @ B1
partial2 = gelu(X @ A2) @ B2
tensor_parallel = partial1 + partial2
print("full layer parameters:", A.size + B.size, " per device:", A1.size + B1.size)
print("max |column-split A, row-split B, one sum - full|:", float(np.abs(tensor_parallel - full).max()))

Ax, Ay = X[:, : d_model // 2] @ A[: d_model // 2], X[:, d_model // 2:] @ A[d_model // 2:]
row_split_first = (gelu(Ax) + gelu(Ay))
true_hidden = gelu(Ax + Ay)
print("splitting A by rows needs a sum before GELU; skipping it is wrong by:", float(np.abs(row_split_first - true_hidden).max()))

activations_bytes = batch * d_model * 4
print("all-reduce payload per layer pair:", activations_bytes, "bytes (batch x d_model float32), independent of d_ff")
```

Splitting A by columns and B by rows across two devices gives the unsplit answer to 1.78e-15 with half the parameters on each (128 of 256). Cutting the first matrix by rows and skipping the sum before the nonlinearity is wrong by 3.97, so the order matters. The one all-reduce moves 192 bytes (6 rows by 8 features by 4 bytes), whatever the hidden width.

### 3. A real two-process pipeline

Stage 0 (a linear layer and tanh) lives on rank 0, stage 1 (a linear layer and the loss) on rank 1. A batch of 64 is split into 4 micro-batches of 16. Rank 0 sends each activation with a non-blocking send; rank 1 receives it, runs its stage and the loss, back-propagates and sends the gradient of the activation back; rank 0 then continues the backward pass through its stage. The loss of each micro-batch is divided by 4 so that the accumulated gradient equals the full-batch gradient.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh()), nn.Linear(16, 1)


def data():
    g = torch.Generator().manual_seed(42)
    X = torch.randn(64, 8, generator=g)
    y = X[:, :1] * 1.5 - X[:, 1:2] + 0.3 * torch.randn(64, 1, generator=g)
    return X, y


def worker(rank, world, port, micro):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    stage0, stage1 = build()
    X, y = data()
    xs, ys = X.chunk(micro), y.chunk(micro)
    if rank == 0:
        acts, handles = [], []
        for xb in xs:
            a = stage0(xb)
            acts.append(a)
            handles.append(dist.isend(a.detach().contiguous(), dst=1))
        for a in acts:
            g = torch.zeros_like(a)
            dist.recv(g, src=1)
            a.backward(g)
        for h in handles:
            h.wait()
        pipeline = torch.cat([p.grad.flatten() for p in stage0.parameters()])
        ref0, ref1 = build()
        loss = nn.functional.mse_loss(ref1(ref0(X)), y)
        loss.backward()
        reference = torch.cat([p.grad.flatten() for p in ref0.parameters()])
        print("micro-batches:", micro, "activation tensor per micro-batch:", tuple(acts[0].shape))
        print("stage 0 gradient max |pipeline - single process|:", f"{(pipeline - reference).abs().max().item():.2e}")
        stage1_grad = torch.zeros(17)
        dist.recv(stage1_grad, src=1)
        ref_last = torch.cat([p.grad.flatten() for p in ref1.parameters()])
        print("stage 1 gradient max |pipeline - single process|:", f"{(stage1_grad - ref_last).abs().max().item():.2e}")
    else:
        sends = []
        for yb in ys:
            buf = torch.zeros(xs[0].shape[0], 16)
            dist.recv(buf, src=0)
            buf.requires_grad_(True)
            loss = nn.functional.mse_loss(stage1(buf), yb) / micro
            loss.backward()
            sends.append(dist.isend(buf.grad.contiguous(), dst=0))
        for h in sends:
            h.wait()
        dist.send(torch.cat([p.grad.flatten() for p in stage1.parameters()]), dst=0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port(), 4), nprocs=2, join=True)
```

Each micro-batch's activation tensor has shape (16, 16), and both stages' gradients match a single-process run on the whole batch: 5.96e-08 for stage 0 and 2.38e-07 for stage 1, which is float32 rounding. This block is about correctness, not speed: on a CPU with two small stages there is nothing to be gained, and it does not time anything.

### Try it yourself

The lab draws block 1's grid for any number of stages and micro-batches. Its defaults (4 stages, 8 micro-batches) show 11 ticks, 32 busy cells of 44 and a bubble of 0.273. Drag the micro-batch slider up and watch the grey cells at the corners shrink; push the stage count up and watch the bubble return. A marker appears once the micro-batches reach four times the stages, GPipe's reported rule of thumb. The table view lists the bubble for micro-batch counts from 1 to 64 at the current stage count.

<PipelineBubbleLab />

## Production snippets (not run here)

On GPUs, PyTorch can shard a feed-forward block over eight devices with its tensor-parallel API. This follows the example in the PyTorch 2.14 documentation, which marks the API as experimental.

Not run in this environment

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

tp_mesh = init_device_mesh("cuda", (8,))
model = parallelize_module(
    model,
    tp_mesh,
    {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
)
```

`model` is your own module with submodules named `w1` and `w2`. The documentation recommends composing column-wise and row-wise styles for attention and MLP layers. The `torch.distributed.pipelining` package, which partitions a model into stages and schedules micro-batches, is described in the same documentation as being in alpha state.

## Designing with it

| Question | Guidance |
| --- | --- |
| Does the model fit on one device? | If it does, use data parallelism. Model parallelism adds idle time and complexity. |
| Where are the cuts? | Pipeline cuts send only activations, so they tolerate slower links between machines. Tensor cuts need an all-reduce per layer pair, so keep them on fast links. |
| How many micro-batches? | Compute (S - 1) / (m + S - 1) for your numbers. More micro-batches cost activation memory (or recomputation). Check against a profile, not only the formula. |
| Are the stages balanced? | The formula assumes equal stage times. The slowest stage sets the tick length, so an unbalanced split wastes more than the bubble. |
| Is training the only goal? | The PyTorch pipelining documentation lists large-scale training, bandwidth-limited clusters and large model inference as the cases where pipelining can be effective. |

**A habit worth keeping.** Test a split model against the unsplit one on a tiny input, as blocks 2 and 3 do. Wrong cuts such as a missing sum produce a model that trains and gives wrong answers.

## Where this stands in 2026

:::info Industry view
- **Pipeline parallelism ships in PyTorch, as alpha.** The `torch.distributed.pipelining` page (last updated 24 July 2026) says it is in alpha state, was migrated from the PiPPy project, and handles partitioning the execution of a model and scheduling micro-batches.
- **Tensor parallelism ships too, as experimental.** The tensor-parallel page (last updated 12 May 2026) provides column-wise, row-wise and sequence-parallel styles on top of `DTensor`, and warns that the APIs are experimental and subject to change.
- **Sharding is the other way to fit a large model.** FSDP shards parameters, gradients and optimiser state across data-parallel workers instead of cutting the model's computation; the pipelining documentation notes that pipelining helps where the computation per device cannot hide the communication of conventional parallelism, such as the weight all-gather of FSDP.
- **Combine, do not choose.** Data, tensor and pipeline parallelism are building blocks that can be combined; the right mix depends on the model, the interconnect and the batch size, which you must measure.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> When and how do we use model parallelism?</summary>

When a model is too large for one device: split its layers or tensor slices across workers, passing activations between them.

</details>

<details>
<summary><strong>Q2.</strong> What is pipeline parallelism?</summary>

A form of model parallelism where stages of the network run on different devices like an assembly line, with micro-batches in flight to keep the stages busy.

</details>

<details>
<summary><strong>Q3.</strong> A 24 GB model is split evenly over 4 GPUs. What is the memory per GPU?</summary>

24 / 4 = 6 GB per GPU, plus activations.

</details>

<details>
<summary><strong>Q4.</strong> Compute the pipeline bubble fraction for S = 4 stages and m = 8 micro-batches.</summary>

(S - 1) / (m + S - 1) = 3/11 = 0.273, about 27 percent idle.

</details>

<details>
<summary><strong>Q5.</strong> How do you reduce the pipeline bubble?</summary>

Use more micro-batches (as m grows the bubble tends to 0), trading memory for utilisation.

</details>

<details>
<summary><strong>Q6.</strong> For 8 stages, how many micro-batches does it take to bring the bubble under 10 percent? Check against the table in block 1.</summary>

Solve 7 / (m + 7) under 0.1, so m + 7 over 70 and m over 63. The table shows 0.099 at m = 64, the first power of two that qualifies.

</details>

<details>
<summary><strong>Q7.</strong> Why is the first matrix of a transformer feed-forward block split by columns and the second by rows in tensor parallelism?</summary>

Column-splitting the first lets each device compute its own hidden units with no communication, because the nonlinearity acts on each hidden unit separately. Row-splitting the second then consumes exactly those units, and one sum of the partial outputs finishes the block. Splitting the first by rows would need the summed pre-activation before the nonlinearity; block 2 shows that skipping it is wrong by 3.97.

</details>

<details>
<summary><strong>Q8.</strong> GPipe reports negligible bubble overhead when M is at least 4 x K. At K = 4 the formula gives 0.158 for M = 16. Reconcile the two.</summary>

The formula counts idle ticks under equal stage times and no scheduling tricks. The paper's statement is an experimental observation: partly because recomputation in the backward pass can be scheduled earlier, the measured overhead was small. So 0.158 is the idealised idle share, and "negligible" describes their measured runs, which you cannot assume for another model or cluster.

</details>

## Further reading

- [Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"](https://arxiv.org/abs/1811.06965), the bubble overhead and the M at least 4 x K observation (section 2). Opened 2 October 2026.
- [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053), intra-layer model parallelism. Opened 2 October 2026.
- [PyTorch 2.14 pipeline parallelism](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html) and [tensor parallelism](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html). Opened 2 October 2026.
- [Dive into Deep Learning, "Computational Performance"](https://d2l.ai/chapter_computational-performance/index.html), multi-GPU and parallel training explained with runnable code.
- On the site: [data parallelism](/docs/mlops/distributed/data-parallelism) for the other half of the story.

## Check yourself

- I can explain the two ways to split a model and when each needs fast links.
- I can derive the bubble fraction (S - 1) / (m + S - 1) from the grid of stages and ticks and compute it for given S and m.
- I can say what more micro-batches cost, and why the cost is memory.
- I can explain why tensor parallelism splits one matrix by columns and the next by rows, and show the split matches the unsplit layer.
- I can check a split model against an unsplit one by comparing gradients.
- I can say what GPipe measured and why that does not make the bubble negligible for every setup.
