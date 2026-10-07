---
id: llme-fsdp-zero
title: "DDP, FSDP and ZeRO"
sidebar_label: "2 · DDP, FSDP and ZeRO"
sidebar_position: 2
slug: /llm-engineering/ddp-fsdp-and-zero
description: "Why plain data parallelism wastes memory on identical copies, how ZeRO stages 1 to 3 and FSDP remove the copies one piece at a time, with real two-process PyTorch runs that match single-process training."
tags: [ddp, fsdp, fsdp2, zero, deepspeed, data-parallel, sharding, optimizer-state, torch-distributed]
---

import Infographic from '@site/src/components/Infographic';
import ZeroStagesLab from '@site/src/components/viz/ZeroStagesLab';

**In one line.** Plain data parallelism keeps a full copy of the weights, the gradients and the optimiser state on every GPU, and ZeRO and FSDP remove those copies one at a time by giving each GPU a slice and fetching the rest only at the moment it is needed.

:::tip Before you start
- **You should already know** how data parallelism averages gradients ([data parallelism](/docs/mlops/distributed/data-parallelism)), what Adam keeps for each parameter ([Adam optimiser](/docs/theory/dnn/adam-optimizer)), and the five parallelism axes and the 16-bytes-per-parameter sum from the previous chapter ([parallelism strategies for LLMs](/docs/llm-engineering/parallelism-strategies-for-llms)).
- **Reading time:** about 40 minutes, plus a minute or two to run the code.
- **After this chapter you can** say what each ZeRO stage shards and what it costs in traffic, run real sharded training on two processes and check it against a single process, and write the FSDP2 and DeepSpeed settings you would use on GPUs.
:::

:::note Not from a lecture
This chapter was written for this site from the ZeRO paper, the PyTorch 2.14 documentation and the DeepSpeed configuration reference listed under Go deeper. Versions used: Python 3.14, PyTorch 2.14.1 (CPU, gloo backend), Transformers 5.18.0. DeepSpeed 0.19.7 (released 16 September 2026 on PyPI) is not installed here, so its snippet is not run. Sources were opened on 7 October 2026.
:::

## In 30 seconds

Eight friends each want to bake from the same cookbook. If every friend buys a full set of three things, the cookbook, a notebook of corrections and a thick logbook of every past bake, then eight sets of identical books fill eight kitchens. Most of that is waste. Give each friend only one eighth of the logbook, and have them swap pages when they need them.

Data parallel training has exactly this waste. Every GPU holds the same weights, the same gradients and the same optimiser state. ZeRO and FSDP are the ways to stop holding identical copies. They save memory at the cost of a little extra talking between GPUs.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| DDP (DistributedDataParallel) | PyTorch's plain data parallel wrapper: full copy per GPU, gradients averaged | 8 GPUs, 8 identical models |
| Optimiser state | The extra numbers an optimiser keeps for each parameter | Adam keeps two (a running mean and a running mean of squares) |
| Master weights | A full-precision copy of the weights kept for the update | fp32 copy next to a bf16 working copy |
| Shard | One slice of a tensor, kept by one GPU | GPU 1 holds parameters 4 to 7 |
| Reduce-scatter | Add every GPU's tensor, then give each GPU only its own slice of the sum | 8 gradients in, 4 summed numbers out |
| All-gather | Every GPU contributes its slice and every GPU receives the whole | 2 slices of 4 become 1 vector of 8 |
| ZeRO (Zero Redundancy Optimizer) | The DeepSpeed method that removes the duplicate copies in three stages | Stage 1 shards optimiser state |
| FSDP (Fully Sharded Data Parallel) | PyTorch's built-in version of ZeRO stage 3 | `fully_shard(layer)` |
| Bucket | A group of gradients sent together so small messages are not wasteful | 25 MB of gradients |

## The idea in plain words

Start with a model so small you can do it by hand. It has 8 parameters. Two GPUs train it, and each sees half of the examples, so each computes a gradient for all 8 parameters. Plain data parallelism averages those two gradients, so both GPUs hold the same 8 averaged numbers and both apply the same update.

Now ask what each GPU actually needs. To update parameters 0 to 3, a GPU needs only the averaged gradient for 0 to 3 and the optimiser state for 0 to 3. It does not need anything about 4 to 7. So split the job: GPU 0 owns parameters 0 to 3, GPU 1 owns 4 to 7. Each GPU receives only its half of the averaged gradient, updates only its half, and then the two halves are put back together so both GPUs have all 8 new weights for the next forward pass.

That is ZeRO in one paragraph. The two operations in the middle have names. **Reduce-scatter** adds the gradients up and hands each GPU only its slice. **All-gather** collects the slices back into the whole. Everything in this chapter is a variation on those two moves.

The ZeRO paper names three stages by what they stop duplicating:

| Stage | What each GPU keeps only a slice of | In PyTorch |
| --- | --- | --- |
| 0 | Nothing: everything is copied | DDP |
| 1 | The optimiser state | `ZeroRedundancyOptimizer` |
| 2 | Optimiser state and gradients | No separate switch in PyTorch; `fully_shard` with `reshard_after_forward=False` behaves like it during a step |
| 3 | Optimiser state, gradients and weights | FSDP, `fully_shard` |

<Infographic src="/img/llme/ddp-fsdp-and-zero-memory.svg" alt="Four columns for DDP and ZeRO stages 1 to 3, each showing three boxes for weights, gradients and optimiser state with the part a GPU keeps shaded, and the per-GPU memory 120.0, 31.4, 16.6 and 1.9 GB for a 7.5B model on 64 GPUs" caption="Compare the purple optimiser box first: it is by far the biggest, so stage 1 already removes most of the memory. The numbers are the ZeRO paper's example and are reproduced by block 2." />

## Worked example, step by step

**The 8-parameter toy.** Every weight starts at 1.0, the learning rate is 0.1, and the two GPUs computed these gradients on their own halves of the batch. Block 1 runs it with real collectives.

1. GPU 0's gradients are 1, 2, 3, 4, 5, 6, 7, 8. GPU 1's are 3, 2, 1, 0, 1, 2, 3, 4.
2. **Reduce-scatter with averaging.** Add them slot by slot and halve: 2, 2, 2, 2, 3, 4, 5, 6. GPU 0 receives the first four (2, 2, 2, 2). GPU 1 receives the last four (3, 4, 5, 6).
3. **Update only the owned slice.** New weight = 1.0 minus 0.1 times the gradient. GPU 0 gets 0.8 for each of its four. GPU 1 gets 0.7, 0.6, 0.5, 0.4.
4. **All-gather.** Both GPUs now hold 0.8, 0.8, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, the same as plain data parallelism would give.

<Infographic src="/img/llme/ddp-fsdp-and-zero-toy-step.svg" alt="Four stacked bands showing eight gradients on two GPUs, the reduce-scatter to two averaged halves, the update of each owned half, and the all-gather that rebuilds the eight weights" caption="Read top to bottom: gradients, reduce-scatter, update, all-gather. Every number here is printed by block 1." />

**The paper's real-size example.** A model with 7.5 billion parameters and 64 GPUs, using the mixed-precision recipe of 2 + 2 + 12 bytes per parameter.

1. Plain data parallelism: 16 bytes x 7.5 billion = 120 GB per GPU.
2. Stage 1 shards the 12-byte optimiser part: 4 + 12/64 = 4.19 bytes, so 31.4 GB.
3. Stage 2 also shards the 2-byte gradients: 2 + 14/64 = 2.22 bytes, so 16.6 GB.
4. Stage 3 shards everything: 16/64 = 0.25 bytes, so 1.9 GB.

Memory fell from 120 GB to 1.9 GB. Block 2 prints exactly these numbers.

## How it works

### What does plain DDP do?

DDP, PyTorch's DistributedDataParallel, copies the model to every GPU. At construction it broadcasts the state from rank 0 so all replicas start identical. It groups gradients into buckets, roughly in the reverse order of the model's parameters, because that is the order they become ready. When every gradient in a bucket is ready, it starts an asynchronous all-reduce, so communication overlaps with the rest of the backward pass. After the backward pass every replica holds the same averaged gradient, and each applies the same optimiser step. This is described in the PyTorch 2.14 DDP notes.

DDP's weakness is the one in the picture above. It makes no attempt to share anything: three full copies per GPU, so a bigger replica count never makes the model fit.

### Why are stages 1 and 2 almost free?

An all-reduce is the same as a reduce-scatter followed by an all-gather. Plain DDP moves about 2 model-sizes of data per step. ZeRO stage 2 does a reduce-scatter on the gradients (1 model-size) and, after the owned slice is updated, an all-gather of the new weights (1 model-size). The total is the same 2. The ZeRO paper states that stages 1 and 2 add no communication over the baseline. You get a large memory saving for nothing but a re-ordering of the same traffic.

### What does stage 3 cost?

At stage 3 the weights themselves are in slices, so a GPU must all-gather a layer's weights before it can compute with them, and free them afterwards. That gather happens in the forward pass and again in the backward pass, then the gradients are reduce-scattered. Three model-sizes of traffic instead of two: 1.5 times, as the paper computes. The cost is hidden when the gather for layer `i + 1` runs while layer `i` computes.

<Infographic src="/img/llme/ddp-fsdp-and-zero-stage3-timeline.svg" alt="Three lanes for all-gather, compute and reduce-scatter across a forward and backward pass over three layers, with each layer gathered once going forward and again going backward and its gradient scattered afterwards" caption="Follow one layer: it is gathered, used, freed, gathered again for the backward pass, and its gradient is scattered. Only about two layers are whole at any time." />

### What is FSDP, and how is it different from ZeRO?

FSDP is PyTorch's own implementation of the stage 3 idea. The PyTorch 2.14 documentation describes FSDP2 (`fully_shard`) as "a fully sharded data parallelism (FSDP) implementation targeting performant eager-mode while using per-parameter sharding". Each parameter becomes a `DTensor` whose local piece is one slice, which block 5 shows. Four settings matter:

| Setting | What it does |
| --- | --- |
| Which modules you wrap | Each wrapped module is one unit that is gathered and freed together; wrap each transformer block, then the root |
| `reshard_after_forward` | `True` frees the gathered weights after forward and gathers them again in backward (less memory, more traffic). `False` keeps them until backward (the reverse). By default it is `True` for inner modules and `False` for the root |
| `mp_policy` | A `MixedPrecisionPolicy` with `param_dtype` (compute and gather), `reduce_dtype` (gradient reduction) and `output_dtype` |
| `offload_policy` | Whether to push sharded state to the CPU |

Llama 3's training used a FSDP that shards optimiser state and gradients but does not reshard the weights after the forward pass, to avoid the extra gather in backward. That is the `False` setting. The weights stay whole from forward until backward is done with them, so for memory this behaves like stage 2 during a step.

DeepSpeed ZeRO is the original library. Its `zero_optimization.stage` setting takes 0 to 3 with the same meanings, and it can offload optimiser state or parameters to CPU memory with `offload_optimizer` and `offload_param`.

### Which should you use?

Use plain DDP while the model state fits. Stage 1 is the cheapest step up, because the optimiser is most of the memory and the traffic is unchanged. Move to stage 2 or 3 when weights are the problem. At very large scale the ZeRO-3 gathers cross the network, so the Ultra-Scale Playbook found pure data parallelism or ZeRO-3 to become communication-bound at 512 GPUs and beyond. That is why the previous chapter combines them with tensor and pipeline parallelism.

## A real system that works this way

**Llama 3 405B** is the example from the last chapter. The paper reports FSDP with sharded optimiser state and gradients for the data-parallel dimension of its 4D layout, and that it chose not to reshard the weights after forward.

**The ZeRO paper itself** is the origin of the three stages. Its authors report training models of over 100B parameters with super-linear speedup on 400 GPUs at 15 petaflops, an 8x increase in model size over the state of the art they compared against. Those are the authors' own 2019 measurements on their cluster.

## Code you can run

Five blocks. Block 1 is the toy with real collectives. Block 2 is arithmetic on real configs. Blocks 3 to 5 train a small real Llama model, 158,016 parameters, on two processes with PyTorch's gloo backend on the CPU, and compare against one process. Each takes a few seconds.

### 1. The 8-parameter toy, with real collectives

Two gloo processes run steps 1 to 4 of the worked example, using `reduce_scatter_single` and `all_gather_single`, the names current in PyTorch 2.14.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GRADS = [[1., 2., 3., 4., 5., 6., 7., 8.], [3., 2., 1., 0., 1., 2., 3., 4.]]
LR = 0.1


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    weights = torch.ones(8)
    grad = torch.tensor(GRADS[rank])
    mine = torch.zeros(4)
    dist.reduce_scatter_single(mine, grad)
    mine /= world
    shard = weights[rank * 4:(rank + 1) * 4] - LR * mine
    rebuilt = torch.zeros(8)
    dist.all_gather_single(rebuilt, shard)
    report = torch.zeros(16)
    dist.all_gather_single(report, torch.cat([mine, shard]))
    if rank == 0:
        for r in range(world):
            g, s = report[8 * r:8 * r + 4].tolist(), report[8 * r + 4:8 * r + 8].tolist()
            print(f"rank {r}: gradients {GRADS[r]}")
            print(f"        averaged gradient shard {g}, updated shard {[round(v, 2) for v in s]}")
        print(f"every rank after the all-gather: {[round(v, 2) for v in rebuilt.tolist()]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port()), nprocs=2, join=True)
```

**Reading the output.** Rank 0 receives the averaged gradient slice 2, 2, 2, 2 and rank 1 receives 3, 4, 5, 6, as in the hand calculation. Their updated slices are 0.8 four times and 0.7, 0.6, 0.5, 0.4. After the all-gather both ranks hold the same eight weights.

**Line by line.**

- `dist.reduce_scatter_single(mine, grad)` sums `grad` over the ranks and writes this rank's quarter-by-rank slice into `mine`. The sum is divided by the world size to make an average.
- `weights[rank * 4:(rank + 1) * 4]` is the slice this rank owns.
- The `report` tensor gathers every rank's numbers to one place so only rank 0 prints; printing from both ranks would interleave randomly.

### 2. Bytes per parameter for every stage, and where it stops helping

This block reproduces the paper's 7.5B example, then asks, for Llama 3.1 8B read from its real config, the smallest number of GPUs that makes the model state fit in 80 GB at each stage.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download

GB = 1e9


def bytes_per_param(stage, n, recipe=(2, 2, 12)):
    w, g, o = recipe
    if stage == 0:
        return w + g + o
    if stage == 1:
        return w + g + o / n
    if stage == 2:
        return w + (g + o) / n
    return (w + g + o) / n


psi = 7.5e9
print("ZeRO paper example: 7.5B parameters, 64-way data parallel, GB per GPU")
for stage in range(4):
    print(f"  stage {stage}: {bytes_per_param(stage, 64) * psi / GB:7.1f} GB   ({bytes_per_param(stage, 64):.3f} bytes per parameter)")

c = json.load(open(hf_hub_download("NousResearch/Meta-Llama-3.1-8B", "config.json")))
h, ffn, v, layers = c["hidden_size"], c["intermediate_size"], c["vocab_size"], c["num_hidden_layers"]
kv = h * c["num_key_value_heads"] // c["num_attention_heads"]
layer = 2 * h * h + 2 * h * kv + 3 * h * ffn + 2 * h
total = 2 * v * h + layers * layer + h
print()
print(f"Llama 3.1 8B: {total:,} parameters, one layer {layer:,}")
print("smallest data-parallel degree whose model state fits in 80 GB (activations ignored)")
for stage in range(4):
    n = next((k for k in range(1, 4097) if bytes_per_param(stage, k) * total / GB <= 80), None)
    print(f"  stage {stage}: {n}")

print()
print("bf16 weights resident on one GPU at the moment of the largest layer, 8 GPUs, 8B model")
shard = 2 * total / 8
print(f"  all weights kept whole (DDP, stage 0 to 2): {2 * total / GB:5.2f} GB")
print(f"  stage 3, reshard after forward: {(shard + 2 * layer) / GB:5.2f} GB  (shard plus two layers, one prefetched)")
print(f"  stage 3, keep gathered after forward: {2 * total / GB:5.2f} GB  (what Llama 3 chose for its weights)")

volume = {0: 2.0, 1: 2.0, 2: 2.0, 3: 3.0}
print()
print("communication per step per GPU, in multiples of the model size (ZeRO paper, section 7)")
for stage, vol in volume.items():
    print(f"  stage {stage}: {vol:.1f}   relative to DDP: {vol / volume[0]:.2f}")
```

**Reading the output.** The paper's example prints 120.0, 31.4, 16.6 and 1.9 GB, matching the figure. For Llama 3.1 8B (8,030,261,248 parameters) plain DDP never fits, stage 1 fits from 3 GPUs, and stages 2 and 3 from 2. The last section shows what stage 3 means for weights: if the full weights are kept after the forward pass, as in the Llama 3 recipe, the whole 16.06 GB stays resident, against 2.44 GB if they are resharded.

**Line by line.**

- `bytes_per_param` is the table in code form: stage 1 divides `o` by `n`, stage 2 also `g`, stage 3 everything.
- The `next(...)` search finds the smallest `n`; it returns `None` when even `n = 4096` is not enough.
- `shard + 2 * layer` is the sharded weights plus the current layer and one prefetched layer, in bf16.

### 3. DDP on a real model, matched against one process

Two processes start from different random seeds. DDP broadcasts rank 0's weights, each rank trains on half of a batch of 8 sequences, and after 10 AdamW steps we compare the result with one process training on the whole batch. We also count the bytes of weights, gradients and Adam state that one rank really holds.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
STEPS, WORLD = 10, 2


def make(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(CFG)


def batch():
    return torch.randint(0, CFG.vocab_size, (8, 32), generator=torch.Generator().manual_seed(1))


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    ddp = DDP(make(rank))
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    x = batch()[rank * 4:(rank + 1) * 4]
    for _ in range(STEPS):
        loss = ddp(x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    mine = flat(ddp.module)
    if rank == 0:
        ref = make(0)
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)
        full = batch()
        for _ in range(STEPS):
            ref(full, labels=full).loss.backward()
            ref_opt.step()
            ref_opt.zero_grad()
        n = mine.numel()
        state = sum(t.nbytes for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0)
        print(f"parameters: {n:,}   ranks started from different seeds and DDP broadcast rank 0's weights")
        print(f"max |DDP - single process on the whole batch| after {STEPS} steps: {(mine - flat(ref)).abs().max().item():.2e}")
        print(f"per rank in fp32: weights {4 * n:,} B, gradients {4 * n:,} B, Adam state {state:,} B")
        print(f"bytes per parameter: {(4 * n + 4 * n + state) / n:.2f}   all-reduced each step: {4 * n:,} B")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(WORLD, free_port()), nprocs=WORLD, join=True)
```

**Reading the output.** The two-process result agrees with the single process to 2.00e-06 after 10 steps. That is larger than float32 rounding on one step, because Adam divides by a running average and so magnifies tiny differences in the order of additions, but it is far below any real difference in the model. Each rank holds 632,064 bytes of weights, 632,064 of gradients and 1,264,128 of Adam state: exactly 16 bytes per parameter in fp32 (4 + 4 + 8). Each step all-reduces 632,064 bytes, the gradient.

**Line by line.**

- `make(rank)` gives each rank a different seed; the `DDP(...)` constructor then overwrites the weights with rank 0's.
- `x = batch()[rank * 4:(rank + 1) * 4]` is the rank's half of the batch. The halves have equal token counts, so averaging the two mean losses equals the mean loss over the whole batch.
- The state size sums only the `exp_avg` and `exp_avg_sq` tensors, not the scalar step counters.

### 4. ZeRO stage 2 by hand, with real collectives

This block writes the sharded update the way the toy did, but on the real model: flatten the parameters, reduce-scatter the gradients so each rank keeps only its slice, update that slice with its own AdamW, and all-gather the new weights. The result is compared with one process.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
STEPS = 10


def make():
    torch.manual_seed(0)
    return LlamaForCausalLM(CFG)


def batch():
    return torch.randint(0, CFG.vocab_size, (8, 32), generator=torch.Generator().manual_seed(1))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = make()
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    padded = -(-n // world) * world
    size = padded // world
    flat = torch.zeros(padded)
    flat[:n] = parameters_to_vector(params).detach()
    shard = torch.nn.Parameter(flat[rank * size:(rank + 1) * size].clone())
    opt = torch.optim.AdamW([shard], lr=1e-3)
    x = batch()[rank * 4:(rank + 1) * 4]
    for _ in range(STEPS):
        model(x, labels=x).loss.backward()
        grads = torch.zeros(padded)
        grads[:n] = torch.cat([p.grad.reshape(-1) for p in params])
        mine = torch.zeros(size)
        dist.reduce_scatter_single(mine, grads)
        shard.grad = mine / world
        opt.step()
        dist.all_gather_single(flat, shard.detach())
        vector_to_parameters(flat[:n].clone(), params)
        for p in params:
            p.grad = None
    state = sum(t.nbytes for s in opt.state.values() for t in s.values() if t.dim() > 0)
    if rank == 0:
        ref = make()
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)
        full = batch()
        for _ in range(STEPS):
            ref(full, labels=full).loss.backward()
            ref_opt.step()
            ref_opt.zero_grad()
        diff = (flat[:n] - parameters_to_vector(ref.parameters()).detach()).abs().max().item()
        print(f"parameters {n:,}, padded to {padded:,}, each rank owns {size:,}")
        print(f"max |sharded Adam - single process Adam| after {STEPS} steps: {diff:.2e}")
        print(f"Adam state per rank: {state:,} B; unsharded it would be {8 * n:,} B ({state / (8 * n):.2f} of that)")
        print(f"gradient shard per rank: {4 * size:,} B instead of {4 * n:,} B")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port()), nprocs=2, join=True)
```

**Reading the output.** The sharded run ends within 2.00e-06 of single-process Adam, the same agreement DDP had. Each rank holds 632,064 bytes of Adam state, exactly half of the 1,264,128 an unsharded rank holds, and keeps 316,032 bytes of gradient instead of 632,064. The padding line says the parameter count divided evenly here (158,016 over two ranks); when it does not, the flat vector is padded and a few bytes are wasted.

**What did not work, and what this does not show.** The full gradients still exist for a moment on every rank, because this simple version runs the whole backward pass before the reduce-scatter. Production code reduces and frees each bucket during the backward pass so the full gradient never exists. This block measures bytes of tensors, not the peak of an allocator on a GPU, so treat it as accounting, not a memory profile. It also says nothing about speed: two CPU processes on one machine have no real network.

**Line by line.**

- `padded = -(-n // world) * world` rounds the parameter count up to a multiple of the number of ranks.
- `shard.grad = mine / world` turns the summed slice into an average before the update.
- `vector_to_parameters` copies the gathered flat vector back into the model so the next forward pass uses the new weights.

### 5. The PyTorch tools: `ZeroRedundancyOptimizer` and FSDP2 `fully_shard`

The same training run with the two built-in tools. `ZeroRedundancyOptimizer` is stage 1 on top of DDP. `fully_shard` is FSDP2 with every transformer block wrapped and the root wrapped last. We compare final weights with a single process and look at what each rank holds.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaForCausalLM

CFG = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
STEPS = 10


def make():
    torch.manual_seed(0)
    return LlamaForCausalLM(CFG)


def batch():
    return torch.randint(0, CFG.vocab_size, (8, 32), generator=torch.Generator().manual_seed(1))


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def state_bytes(opt):
    return sum(local(t).nbytes for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train(model, opt, x):
    for _ in range(STEPS):
        model(x, labels=x).loss.backward()
        opt.step()
        opt.zero_grad()


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    x = batch()[rank * 4:(rank + 1) * 4]
    ref = make()
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    train(ref, ref_opt, batch())
    ref_flat = torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
    n = ref_flat.numel()

    zero_model = DDP(make())
    zero_opt = ZeroRedundancyOptimizer(zero_model.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-3)
    train(zero_model, zero_opt, x)
    zero_flat = torch.cat([p.detach().reshape(-1) for p in zero_model.parameters()])

    mesh = init_device_mesh("cpu", (world,))
    fsdp = make()
    for layer in fsdp.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(fsdp, mesh=mesh)
    fsdp_opt = torch.optim.AdamW(fsdp.parameters(), lr=1e-3)
    train(fsdp, fsdp_opt, x)
    fsdp_flat = torch.cat([p.full_tensor().detach().reshape(-1) for p in fsdp.parameters()])
    held = sum(local(p).numel() for p in fsdp.parameters())

    if rank == 0:
        print(f"ZeroRedundancyOptimizer: max diff {(zero_flat - ref_flat).abs().max().item():.2e}, Adam state on rank 0 {state_bytes(zero_opt.optim):,} B")
        print(f"FSDP2 fully_shard: max diff {(fsdp_flat - ref_flat).abs().max().item():.2e}, Adam state on rank 0 {state_bytes(fsdp_opt):,} B")
        print(f"FSDP2 rank 0 holds {held:,} of {n:,} parameters ({held / n:.3f}); one DDP rank holds {8 * n:,} B of Adam state")
        print(f"parameter type under FSDP2: {type(next(fsdp.parameters())).__name__}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port()), nprocs=2, join=True)
```

**Reading the output.** Both tools match the single process to 2.00e-06. The FSDP2 rank holds 79,008 of 158,016 parameters (0.500), and its parameters are `DTensor` objects whose local piece is the slice. The Adam state on rank 0 is 632,064 bytes under FSDP2 and 632,320 under `ZeroRedundancyOptimizer`. The second is slightly more than half because that optimiser packs whole parameters greedily by size, which the PyTorch documentation calls a sorted-greedy algorithm, so the split is close to half but not exactly.

**What did not work on the first try.** `fully_shard(module)` without a mesh failed on this Mac with `module 'torch.mps' has no attribute 'is_initialized'`, because the default mesh picked the Apple GPU device. Passing `mesh=init_device_mesh("cpu", (world,))` fixed it. On a CUDA machine the default mesh works.

**Line by line.**

- `for layer in fsdp.model.layers: fully_shard(layer, mesh=mesh)` makes each transformer block its own gather-and-free unit.
- `fully_shard(fsdp, mesh=mesh)` last wraps the root, which then owns the embeddings and the output head.
- `p.full_tensor()` gathers a `DTensor` into a normal tensor, only to compare it with the reference.

### Try it yourself

The lab is block 2's formula with the model, the number of GPUs and the stage exposed. Its defaults (7.5B, 64 GPUs, bf16 recipe, 80 GB card) reproduce the four numbers printed by block 2: 120.0, 31.4, 16.6 and 1.9 GB. Click "show data" to see the table, including the traffic and the smallest GPU count that fits.

<ZeroStagesLab />

**What each control does.**

- **model** picks the parameter count: the paper's 7.5B or the real Llama 3.1 sizes from the configs read in chapter 1.
- **data-parallel GPUs N** sets the number of replicas that share the slices.
- **highlight stage** brightens one bar and sets the readout underneath.
- **bytes per parameter** switches between the bf16 recipe (2 + 2 + 12) and plain fp32 (4 + 4 + 8).
- **GPU memory** moves the red limit line, and the "smallest N that fits" readout follows it.

**Try it yourself.**

1. Set **N** to 8 with the other defaults. Stage 1 now needs 41.3 GB, stage 2 28.1 GB and stage 3 15.0 GB. Why: each stage divides by N, so fewer GPUs means a smaller saving. The 64-GPU numbers were flattering.
2. Set **model** to Llama 3.1 70B and keep N at 64. Only stage 3 fits (17.6 GB), and the data view shows stages 1 and 2 "never" fit at any N. Why: stage 1 cannot go below 4 bytes per parameter and stage 2 cannot go below 2, because the weights are still whole. At 70.55B parameters, 2 bytes is 141 GB.
3. Set **bytes per parameter** to fp32 at the defaults. Stage 1 jumps from 31.4 to 60.9 GB. Why: with fp32 everywhere only 8 of the 16 bytes are optimiser state, so stage 1 has less to shard. The bf16 recipe is part of why ZeRO looks so good.

## Production snippets (not run here)

:::warning Not run in this environment
These need CUDA GPUs and several processes. The imports and the signatures of `fully_shard` and `MixedPrecisionPolicy` were checked against PyTorch 2.14.1 on 7 October 2026. The DeepSpeed key names were checked against its configuration reference; DeepSpeed itself is not installed here.
:::

FSDP2 on GPUs. Wrap each block, then the root. The policy computes and gathers in bf16 and reduces gradients in fp32. `reshard_after_forward=False` is the Llama 3 style choice for the weights:

```python
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

for layer in model.model.layers:
    fully_shard(layer, mp_policy=policy, reshard_after_forward=False)
fully_shard(model, mp_policy=policy)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

DeepSpeed ZeRO stage 3 with bf16 and optimiser offload to the CPU, written as a configuration file, and the training loop it expects:

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

```python
import deepspeed

engine, optimiser, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config="ds_config.json")
for batch in loader:
    loss = engine(**batch).loss
    engine.backward(loss)
    engine.step()
```

## Designing with it

1. **Count before you shard.** Use block 2's formula. If the state already fits with room for activations, plain DDP is the fastest option.
2. **Try stage 1 first.** It removes most of the memory with no extra traffic.
3. **Wrap at the block level.** Gather-and-free units should be transformer blocks, so the gather for the next block can overlap with the current one.
4. **Choose `reshard_after_forward` on purpose.** Memory-tight: `True`. Network-tight: `False`.
5. **Watch the traffic when you cross servers.** Stage 3 moves 1.5 times as much as DDP; if your links are slow, prefer stage 2 or combine with tensor and pipeline parallelism.
6. **Do not compare absolute memory across recipes.** "ZeRO saves 8x" depends on the bytes-per-parameter recipe; the lab's fp32 setting shows the difference.

## Where this stands in 2026

:::info Industry view
- **FSDP2 is documented as the per-parameter design.** The PyTorch 2.14 FSDP2 page describes `fully_shard` with `DTensor`-based per-parameter sharding. The original `FullyShardedDataParallel` class is still documented in 2.14 without a deprecation notice, as of 7 October 2026.
- **DeepSpeed is still maintained.** PyPI listed 0.19.7 as the latest release, uploaded on 16 September 2026.
- **Sharding is one layer of a stack.** The Llama 3 paper and the Ultra-Scale Playbook both combine it with tensor, pipeline and context parallelism rather than using it alone.
- **Not verified here:** GPU throughput, offload speed, or which framework a given lab uses today.
:::

## Common mistakes

1. **Expecting ZeRO to shrink activations.** It shards weights, gradients and optimiser state only. If you run out of memory in activations, use the savings from the previous chapter.
2. **Using stage 1 or 2 to fit a model whose weights alone are too big.** They keep the weights whole. Block 2 shows stage 2 never fits Llama 3.1 70B at any GPU count on 80 GB.
3. **Wrapping the whole model as one FSDP unit.** It feels simple. But then the entire model is gathered at once, and the memory saving disappears. Wrap per block.
4. **Forgetting the recipe.** "ZeRO divides memory by N" is only true for the part it shards. Lab experiment 3 shows how the share changes with the recipe.
5. **Comparing a sharded run to a single process after many steps and expecting zero difference.** Adam magnifies ordering differences in float arithmetic. Compare after a few steps, with a tolerance, as blocks 3 to 5 do.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> A 3-billion-parameter model trains in bf16 with Adam (2 + 2 + 12 bytes). How much model state does plain DDP need per GPU, and how much does stage 3 need on 8 GPUs?</summary>

DDP: 3 billion x 16 bytes = 48 GB. Stage 3 on 8 GPUs: 48 / 8 = 6 GB, plus the transient gathered layer and, of course, activations.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Which two collective operations replace one all-reduce in stage 2, and why is the traffic the same?</summary>

A reduce-scatter of the gradients and an all-gather of the updated weights. An all-reduce is itself a reduce-scatter followed by an all-gather, so the same two moves are just done at different times. The ZeRO paper counts 2 model-sizes either way.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Block 2 says plain DDP never fits Llama 3.1 8B in 80 GB but stage 1 fits from 3 GPUs. Check the number by hand.</summary>

8.03 billion x (4 + 12/N) bytes must be at most 80 GB, so 4 + 12/N must be at most 9.96. N = 2 gives 10, which is 80.3 GB and just over. N = 3 gives 8, which is 64.2 GB. So 3 is the smallest.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why does stage 3 cost 1.5 times the traffic of DDP, and when can that cost be hidden?</summary>

Weights are gathered in the forward pass (1 unit), gathered again in the backward pass (1 unit), and the gradients are reduce-scattered (1 unit): 3 against DDP's 2. It is hidden when the gather for the next layer runs while the current layer computes, which is why the wrapping unit should be a transformer block, and why a slow network makes it visible.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> The lab says Llama 3.1 70B at stage 2 "never" fits an 80 GB card. Explain with numbers, and say what you would do.</summary>

Stage 2 keeps the bf16 weights whole: 2 bytes x 70.55 billion = 141 GB, already more than 80 GB whatever N is. The fix is to shard the weights, either with stage 3 (17.6 GB at N = 64) or by cutting them with tensor and pipeline parallelism as in the last chapter.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> After 10 steps the sharded run differs from the single process by 2.00e-06. Is that a bug?</summary>

No. Both reductions add the same numbers in a different order, so float32 rounding differs by about 1e-7 per operation, and Adam's division by a running average amplifies it. A bug would show up as an error of the size of one update, which is about the learning rate (1e-3 here), or larger. The check that matters is that the error stays tiny and does not grow into the loss.

</details>

## Go deeper

All sources were opened on 7 October 2026.

- [ZeRO: Memory optimizations toward training trillion parameter models (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054): the three stages, the 2 + 2 + K accounting, the 7.5B and 64-GPU example, and the communication analysis (2 against 3 model-sizes).
- [PyTorch 2.14 DDP design notes](https://docs.pytorch.org/docs/2.14/notes/ddp.html): the broadcast, buckets and overlapped all-reduce.
- [PyTorch 2.14 FSDP2 `fully_shard`](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html): `reshard_after_forward`, `MixedPrecisionPolicy`, `OffloadPolicy`.
- [PyTorch 2.14 `ZeroRedundancyOptimizer`](https://docs.pytorch.org/docs/2.14/distributed.optim.html): the sorted-greedy partitioning and its caveats.
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (arXiv 2304.11277)](https://arxiv.org/abs/2304.11277): the design paper for the first FSDP; its title and abstract page were opened, the body was not read for this chapter.
- [DeepSpeed configuration reference](https://www.deepspeed.ai/docs/config-json/) and [getting started](https://www.deepspeed.ai/getting-started/): the `zero_optimization` keys and the initialise, backward and step loop.
- [The Ultra-Scale Playbook (Hugging Face, 2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) and [The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/abs/2407.21783): ZeRO in a full layout.

## Check yourself

- I can say what each ZeRO stage shards and give its bytes per parameter for the 2 + 2 + 12 recipe.
- I can explain why an all-reduce equals a reduce-scatter plus an all-gather and why that makes stages 1 and 2 free in traffic.
- I can explain why stage 3 costs 1.5 times the traffic and how overlap hides it.
- I can run two-process sharded training on the CPU and check it against one process with a sensible tolerance.
- I can write the FSDP2 wrapping and the DeepSpeed configuration for a GPU run, and say what each setting trades.
- I can say when plain DDP is the right answer, and why ZeRO does not help with activations.

## Where to go next

Next chapter: [mixed precision and numerics](/docs/llm-engineering/mixed-precision-and-numerics), which explains where the 2, 2 and 12 bytes come from and what can go wrong with fewer bits. A related chapter: [parallelism strategies for LLMs](/docs/llm-engineering/parallelism-strategies-for-llms), which combines sharding with tensor, pipeline and context parallelism.
