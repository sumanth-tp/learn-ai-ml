---
id: dist-data-parallel
title: "Data Parallelism"
sidebar_label: "Data parallelism"
sidebar_position: 3
slug: /mlops/distributed/data-parallelism
description: "Replicate the model, shard the data, average the gradients every step: why it matches single-process training, the linear scaling rule, and how communication sets the scaling efficiency."
tags: [data-parallelism, ddp, all-reduce, effective-batch, linear-scaling-rule, torch-distributed, gloo]
---

import Infographic from '@site/src/components/Infographic';
import DataParallelScalingLab from '@site/src/components/viz/DataParallelScalingLab';

**In one line.** Put the same model on every worker, give each a different slice of the data, then average the gradients so that every copy takes the same step.

Built from the course lecture "dml-s3-data-parallelism" (Lecture Library series), extended with runnable measurements.

## The idea in plain words

Eight students are marking a stack of exam scripts, and they all use the same mark scheme. Each student marks their own slice of the stack and writes down the mistakes they saw most often. Before anyone starts the next batch, they pool their notes into one agreed list, so all eight students correct the mark scheme in exactly the same way. Because each student started with the same scheme and applies the same correction, they stay in agreement forever.

That is **data parallelism**. The "mark scheme" is the model's weights, the "slice of the stack" is a shard of the batch, and the "pooled notes" are the averaged gradients. The model never needs to be split, so your training code barely changes. What you pay for is the pooling: once per step, every worker must exchange its gradient with the others.

Two questions decide whether it works well, and this chapter answers both with measurements. First, is the pooled step really the same as one worker seeing the whole batch? (Yes, exactly, if you average correctly.) Second, how much does the pooling cost, and what do you do about the bigger effective batch it creates?

<Infographic src="/img/dist/data-parallelism-training-loop.svg" alt="Five steps: replicate, shard, local gradient, all-reduce, update, repeating; tables show averaging eight shard gradients matches the full batch to 2.22e-15 and a real two-process gloo run matches a single process to 1.19e-07" caption="The data-parallel step and the checks that show it is exact. Figures come from blocks 1 and 3 below." />

<Infographic src="/img/dist/data-parallelism-scaling-rule.svg" alt="A table showing batch 256 with learning rate 0.8 reaches test loss 0.2208 against 0.4608 at learning rate 0.1, and a scaling model table with efficiency 0.901 at 8 workers and 0.889 at 256" caption="The linear scaling rule and the cost of communication. Figures come from blocks 2 and 4 below; the scaling model's parameters are illustrative." />

## How it works

Same model everywhere, different data on each worker, then average the gradients.

### Replicate and all-reduce

Each worker holds the full model and a data shard, computes local gradients, and all-reduces them to an average so that all replicas stay identical.

:::tip

**Worked.** K = 8 and B = 32 give a global batch of 256; the linear scaling rule says to multiply the learning rate by 8.

:::

### All-reduce every step

Synchronous data parallelism averages the gradients on every step: consistent, but sensitive to stragglers. Asynchronous training relaxes this. Ring all-reduce keeps the per-worker communication near constant as K grows.

### What the lecture leaves implicit

:::note Beyond the lecture
**Why averaging is exact.** For a loss that is the mean of a per-example loss, the gradient of the full batch is the mean of the gradients of its pieces, provided each piece is weighted by its size. With eight equal shards of 32 a plain mean is enough (block 1 shows the difference is 2.22e-15, rounding error). With unequal shards a plain mean over-weights the small shard and is wrong; weight by shard size. That is why distributed samplers pad or drop examples so that every worker sees the same count.

**The three things a framework adds.** Starting the replicas from identical weights, which DDP does by broadcasting the model state from rank 0. Averaging the gradients, which it does by all-reduce in buckets. And hiding the communication, by starting a bucket's all-reduce while the backward pass is still computing earlier layers (block 4's overlap parameter).

**The linear scaling rule, and where it stops.** Goyal and colleagues' statement is "when the minibatch size is multiplied by k, multiply the learning rate by k". The reasoning: k steps of size η on small batches roughly equal one step of size kη on a batch k times larger, provided the gradient does not change much across those k steps. They pair it with a warm-up, ramping the rate from the small-batch value over the first five epochs, because early in training the weights change quickly and the assumption fails. They also found the approach stops working beyond a minibatch of about 8k on ImageNet, so the rule has a range. Block 2 tests it on a small problem.

**Cost model.** A ring all-reduce moves 2(K - 1)/K times the gradient size per worker. Divide by the link bandwidth to get a communication time, add the compute time, and subtract whatever overlaps. Block 4 and the lab below compute exactly this. It ignores per-message latency, which matters for small gradients.
:::

## A real system that works this way

**PyTorch's DistributedDataParallel.** The DDP design notes say it broadcasts the `state_dict()` from rank 0 so that all replicas start from the exact same state; that a `Reducer` organises gradients into buckets and reduces one bucket at a time; and that when every gradient in a bucket is ready it starts an asynchronous all-reduce that computes the mean across processes. Its performance advantage, in the notes' words, comes from overlapping the all-reduce with the backward computation. After the backward pass, the `grad` of each parameter is the same on every process. Block 3 does by hand what DDP does and compares them.

**The one-hour ImageNet run.** Goyal et al. trained ResNet-50 with a minibatch of 8192 on 256 GPUs in one hour with no loss of accuracy against a small-minibatch baseline, using the linear scaling rule and warm-up. It is the standard evidence that synchronous data parallelism scales far with the right learning rate.

## Code you can run

Four blocks. Blocks 1, 2 and 4 use numpy; block 3 runs two real processes with PyTorch's Gloo backend on the CPU (Python 3.14, numpy 2.5.3, torch 2.14.1).

### 1. Averaged shard gradients equal the full-batch gradient

Linear regression on 256 examples cut into 8 shards of 32.

```python
import numpy as np

rng = np.random.default_rng(0)
K, B, d = 8, 32, 5
X = rng.normal(size=(K * B, d))
w_true = np.array([2.0, -1.0, 0.5, 3.0, -2.0])
y = X @ w_true + 0.1 * rng.normal(size=K * B)


def grad(w, Xs, ys):
    return Xs.T @ (Xs @ w - ys) / len(ys)


w = rng.normal(size=d)
full = grad(w, X, y)
shards = [(X[i * B:(i + 1) * B], y[i * B:(i + 1) * B]) for i in range(K)]
local = [grad(w, Xs, ys) for Xs, ys in shards]
averaged = np.mean(local, axis=0)
print("global batch:", K * B)
print("max |mean of 8 shard gradients - full-batch gradient|:", float(np.abs(averaged - full).max()))

uneven = [(X[:200], y[:200]), (X[200:], y[200:])]
naive = np.mean([grad(w, Xs, ys) for Xs, ys in uneven], axis=0)
weighted = sum(len(ys) * grad(w, Xs, ys) for Xs, ys in uneven) / len(y)
print("uneven shards 200 and 56, plain mean error:", float(np.abs(naive - full).max()))
print("uneven shards, size-weighted mean error:", float(np.abs(weighted - full).max()))

w_single = np.zeros(d)
w_dp = np.zeros(d)
for _ in range(25):
    w_single = w_single - 0.1 * grad(w_single, X, y)
    w_dp = w_dp - 0.1 * np.mean([grad(w_dp, Xs, ys) for Xs, ys in shards], axis=0)
print("after 25 steps, single process vs 8 replicas, max weight difference:", float(np.abs(w_single - w_dp).max()))
print("weights:", np.round(w_dp, 3).tolist())
```

The mean of the eight shard gradients differs from the full-batch gradient by 2.22e-15, which is floating-point rounding. With uneven shards of 200 and 56 a plain mean is wrong by 0.567, while the size-weighted mean is exact to 4.44e-16. Twenty-five gradient steps on one process and on eight replicas end with weights that differ by at most 4.44e-16. Data parallelism changes where the arithmetic runs, not what it computes.

### 2. The linear scaling rule on a small problem

Logistic regression on 3072 training examples, two epochs, averaged over ten seeds. One worker with batch 32 makes 192 updates; eight workers with an effective batch of 256 make only 24.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 4096, 20
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = (X @ w_true + 0.5 * rng.normal(size=n) > 0).astype(float)
Xtr, ytr, Xte, yte = X[:3072], y[:3072], X[3072:], y[3072:]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(batch, lr, epochs, warmup_steps=0, seed=0):
    r = np.random.default_rng(seed)
    w = np.zeros(d)
    step = 0
    for _ in range(epochs):
        order = r.permutation(len(ytr))
        for start in range(0, len(ytr), batch):
            idx = order[start:start + batch]
            rate = lr * min(1.0, (step + 1) / warmup_steps) if warmup_steps else lr
            g = Xtr[idx].T @ (sigmoid(Xtr[idx] @ w) - ytr[idx]) / len(idx)
            w = w - rate * g
            step += 1
    p = sigmoid(Xte @ w)
    loss = -np.mean(yte * np.log(p + 1e-12) + (1 - yte) * np.log(1 - p + 1e-12))
    return loss, np.mean((p > 0.5) == yte), step


print("setting                         updates  test loss  test accuracy")
runs = [
    ("1 worker,  B=32,  lr 0.1", 32, 0.1, 0),
    ("8 workers, B=256, lr 0.1", 256, 0.1, 0),
    ("8 workers, B=256, lr 0.8", 256, 0.8, 0),
    ("8 workers, B=256, lr 0.8 + warmup", 256, 0.8, 6),
]
for name, b, lr, wu in runs:
    losses = []
    accs = []
    for seed in range(10):
        loss, acc, steps = train(b, lr, 2, wu, seed)
        losses.append(loss)
        accs.append(acc)
    print(f"{name:<32}{steps:>5}  {np.mean(losses):9.4f}  {np.mean(accs):13.4f}")
```

Keeping the learning rate at 0.1 while the batch grows eightfold leaves the model undertrained: test loss 0.4608 against 0.2235 for the single worker. Scaling the rate to 0.8 recovers it (0.2208) with accuracy 0.9632 against 0.9634. Warm-up gives 0.2300 here. On this tiny, well-conditioned problem warm-up does not help, because early steps are not dangerous, so do not read it as evidence against warm-up on deep networks, where the paper found it necessary. Nothing here says the rule survives 8192 or 65536 examples per step.

### 3. A real two-process run: manual all-reduce, DDP and a single process

A small network (8 inputs, 16 tanh units, 1 output, 161 parameters) trained for 30 SGD steps on 64 examples. Two processes use the Gloo backend. The ranks start from different random weights, so the code must synchronise them. The manual loop broadcasts rank 0's weights, then averages gradients with `all_reduce`; DDP does both itself. Rank 0 finally trains the same model alone on all 64 examples.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))


def make_data():
    g = torch.Generator().manual_seed(42)
    X = torch.randn(64, 8, generator=g)
    y = (X[:, :1] * 1.5 - X[:, 1:2] + 0.3 * torch.randn(64, 1, generator=g))
    return X, y


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def train_single(steps, lr):
    model = make_model(0)
    X, y = make_data()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(X), y).backward()
        opt.step()
    return flat(model)


def train_manual(rank, world, steps, lr):
    model = make_model(rank)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    X, y = make_data()
    Xs, ys = X[rank::world], y[rank::world]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(Xs), ys).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        opt.step()
    return flat(model)


def train_ddp(rank, world, steps, lr):
    model = DDP(make_model(rank))
    X, y = make_data()
    Xs, ys = X[rank::world], y[rank::world]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(Xs), ys).backward()
        opt.step()
    return flat(model.module)


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    steps, lr = 30, 0.1
    manual = train_manual(rank, world, steps, lr)
    ddp = train_ddp(rank, world, steps, lr)
    other = [torch.zeros_like(manual) for _ in range(world)]
    dist.all_gather(other, manual)
    replicas_match = all(torch.equal(other[0], o) for o in other)
    if rank == 0:
        single = train_single(steps, lr)
        print("parameters:", single.numel())
        print("manual all-reduce replicas identical:", replicas_match)
        print("max |manual - single process|:", f"{(manual - single).abs().max().item():.2e}")
        print("max |DDP - single process|:", f"{(ddp - single).abs().max().item():.2e}")
        print("max |DDP - manual|:", f"{(ddp - manual).abs().max().item():.2e}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port()), nprocs=2, join=True)
```

The replicas stay identical in the manual loop, and both the manual loop and DDP agree with the single-process model to 1.19e-07, which is float32 rounding. DDP and the manual loop agree exactly (0.00e+00). Starting ranks from different weights and finishing identical shows the broadcast doing its job.

### 4. How much does the all-reduce cost? A scaling calculator

The cost model from the note above. The parameters are illustrative, not measured hardware: a local batch of 32, 5 ms of compute per sample, a 100 MB gradient and a 10 GB/s ring.

```python
def scaling(K, B=32, ms_per_sample=5.0, grad_mb=100.0, gb_per_s=10.0, overlap=0.0):
    compute = B * ms_per_sample
    comm = 0.0 if K == 1 else 2 * (K - 1) / K * (grad_mb / 1000.0) / gb_per_s * 1000.0
    step = compute + (1.0 - overlap) * comm
    throughput = K * B / step * 1000.0
    base = B / compute * 1000.0
    return compute, comm, step, throughput / base, throughput / base / K


print("local batch 32, 5 ms per sample, 100 MB gradient, 10 GB/s ring, no overlap")
print("workers  global batch  compute ms  comm ms  step ms  speedup  efficiency")
for K in (1, 2, 4, 8, 16, 64, 256):
    c, m, s, sp, ef = scaling(K)
    print(f"{K:>7}  {K * 32:>12}  {c:10.1f}  {m:7.2f}  {s:7.2f}  {sp:7.3f}  {ef:10.3f}")

print()
print("8 workers, effect of overlap and bandwidth")
for overlap in (0.0, 0.5, 0.9):
    for bw in (1.0, 10.0):
        c, m, s, sp, ef = scaling(8, overlap=overlap, gb_per_s=bw)
        print(f"overlap {overlap:.1f}  {bw:>4.0f} GB/s  step {s:8.2f} ms  speedup {sp:6.3f}  efficiency {ef:.3f}")
```

With 8 workers the step takes 177.50 ms against 160.0 ms of compute, a speed-up of 7.211 and an efficiency of 0.901. Efficiency falls only slowly with more workers, from 0.941 at 2 to 0.889 at 256, because the ring's per-worker traffic saturates at twice the gradient size. The second table shows what matters: dropping the link from 10 GB/s to 1 GB/s cuts the efficiency at 8 workers from 0.901 to 0.478 with no overlap, while overlapping 90 percent of communication with the backward pass restores it to 0.901 on the slow link and 0.989 on the fast one.

### Try it yourself

The lab below is block 4 with every parameter exposed. Its defaults (8 workers, local batch 32, 5 ms per sample, 100 MB, 10 GB/s, no overlap) reproduce the printed row: step 177.50 ms, speed-up 7.211, efficiency 0.901. Change the gradient size to 2000 MB to model a larger network, or the overlap slider to see the value of hiding communication. The readout also shows the global batch and the linear-rule learning-rate multiplier.

<DataParallelScalingLab />

## Production snippets (not run here)

On GPUs the same program is launched once per device with `torchrun` and uses the NCCL backend.

Not run in this environment

```python
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = build_model().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1 * dist.get_world_size())

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.cuda(local_rank)), y.cuda(local_rank))
        loss.backward()
        optimizer.step()
```

Launch it with `torchrun --nproc-per-node=8 train.py`. `build_model`, `dataset`, `loss_fn` and `num_epochs` are your own. The PyTorch documentation says `set_epoch()` must be called at the start of each epoch for shuffling to work across epochs, and that `torchrun` launches the given number of processes per node and sets environment variables including `RANK`, `WORLD_SIZE` and `LOCAL_RANK`.

## Designing with it

| Question | Guidance |
| --- | --- |
| Is the effective batch still sensible? | Compute K x B. Scale the learning rate by K, add warm-up, and check the validation curve. Beyond a few thousand examples per step the rule may fail, as the paper found. |
| Are shards equal? | Use a distributed sampler that pads or drops, or weight by shard size. A plain mean of unequal shards is wrong. |
| Is the step compute-bound or communication-bound? | Compare compute time with gradient bytes divided by bandwidth. If communication is larger, overlap it, compress it, or raise the local batch so compute grows. |
| Do all replicas start identical? | Broadcast from rank 0, or use DDP. Different initial weights stay different. |
| Does the model plus its optimiser state fit on one device? | If not, plain data parallelism cannot help: it keeps a full copy per worker. Shard the state (FSDP) or split the model. |
| Do batch-dependent layers matter? | Batch normalisation statistics are computed per worker unless synchronised, so very small local batches can hurt. |

**A habit worth keeping.** Assert that replicas are identical after a few steps. A silent divergence between replicas produces a model that trains but is wrong, and it is cheap to catch with one `all_gather`, as block 3 does.

## Where this stands in 2026

:::info Industry view
- **DDP is the default for models that fit.** The PyTorch 2.14 notes describe the broadcast, bucketed all-reduce and overlap with the backward pass that block 3 reproduces by hand.
- **FSDP removes the full copy per worker.** `FullyShardedDataParallel` shards parameters, gradients and optimiser state; during forward and backward it unshards parameters before they are used, reshards them afterwards, and synchronises gradients with reduce-scatter. The documentation says it is inspired by ZeRO stage 3. The FSDP2 page (`fully_shard`, last updated 24 April 2026) recommends that FSDP1 users consider migrating. Block 4's model changes accordingly: the traffic is not one all-reduce but gathers and reduce-scatters of parameters.
- **Frameworks schedule it for you.** Ray Train and the Hugging Face Accelerate and DeepSpeed integrations it lists wrap the same collectives behind a trainer.
- **Do not trust the calculator over a profile.** Block 4's parameters are illustrative. Real efficiency depends on the interconnect, the bucket sizes and how well communication overlaps, which only a profile of your own job shows.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Describe the data-parallel training loop.</summary>

Replicate the model on every worker, give each a data shard, compute local gradients, then all-reduce (average) them so that all replicas update identically.

</details>

<details>
<summary><strong>Q2.</strong> For K = 8 workers with local batch 32, what is the effective batch size?</summary>

K x B = 8 x 32 = 256.

</details>

<details>
<summary><strong>Q3.</strong> State the linear scaling rule.</summary>

When the global batch is multiplied by K, multiply the learning rate by K too, with warm-up, to preserve training dynamics.

</details>

<details>
<summary><strong>Q4.</strong> What is the main communication cost in data parallelism?</summary>

The per-step gradient all-reduce; ring all-reduce keeps it efficient as K grows.

</details>

<details>
<summary><strong>Q5.</strong> Contrast synchronous and asynchronous data parallelism.</summary>

Synchronous averages gradients every step: consistent, but a straggler stalls everyone. Asynchronous updates without waiting: faster, but with stale gradients.

</details>

<details>
<summary><strong>Q6.</strong> Two workers hold shards of 200 and 56 examples. Their mean gradients are g1 and g2. What is the correct combined gradient, and what goes wrong with (g1 + g2) / 2?</summary>

The correct gradient is (200 g1 + 56 g2) / 256. The plain mean gives the small shard the same weight as the large one, so it is biased towards the 56 examples; block 1 measures the error at 0.567 against 4.44e-16 for the weighted mean.

</details>

<details>
<summary><strong>Q7.</strong> In block 2, why does batch 256 with learning rate 0.1 have a test loss of 0.4608 while the single worker reaches 0.2235?</summary>

Both see the same number of examples, but the large batch makes eight times fewer updates (24 against 192) at the same step size, so it travels one eighth as far. Multiplying the learning rate by 8, the linear rule, restores the distance (0.2208).

</details>

<details>
<summary><strong>Q8.</strong> Using block 4's model, 8 workers on a 1 GB/s link reach an efficiency of 0.478. Name two changes that raise it and the value each reaches in the block.</summary>

Overlap communication with the backward pass (0.646 at 50 percent, 0.901 at 90 percent on the same link), or use a faster link (0.901 at 10 GB/s with no overlap). A larger local batch also helps because compute grows while the gradient size stays fixed.

</details>

## Further reading

- [PyTorch 2.14 notes on DistributedDataParallel](https://docs.pytorch.org/docs/2.14/notes/ddp.html), the broadcast, buckets and overlap. Opened 2 October 2026.
- [PyTorch 2.14 FullyShardedDataParallel](https://docs.pytorch.org/docs/2.14/fsdp.html) and [FSDP2 (`fully_shard`)](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html). Opened 2 October 2026.
- [PyTorch 2.14 torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html) and [`DistributedSampler`](https://docs.pytorch.org/docs/2.14/data.html). Opened 2 October 2026.
- [Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"](https://arxiv.org/abs/1706.02677), the linear scaling rule and warm-up. Opened 2 October 2026.
- [Dive into Deep Learning, "Training on Multiple GPUs"](https://d2l.ai/chapter_computational-performance/index.html), data parallelism with code.
- On the site: [mini-batch gradient descent](/docs/theory/dnn/gradient-descent-in-neural-networks-batch-sgd-and-mini-batch).

## Check yourself

- I can explain why the average of equal-shard gradients equals the full-batch gradient, and what to do with unequal shards.
- I can compute the effective batch and apply the linear scaling rule, and say where it is known to stop working.
- I can write a manual all-reduce training loop with `torch.distributed`, and explain what DDP adds.
- I can estimate scaling efficiency from compute time, gradient size and bandwidth, and say what overlap changes.
- I can say when data parallelism is not enough and what to reach for instead.
