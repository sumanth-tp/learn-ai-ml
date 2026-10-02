---
id: dist-foundations
title: "Why Distribute Machine Learning"
sidebar_label: "Why distribute"
sidebar_position: 1
slug: /mlops/distributed/why-distribute-machine-learning
description: "Why models and datasets outgrow one machine, how data and model parallelism split the work, how parameter servers and all-reduce synchronise it, and what waiting and stale gradients cost."
tags: [distributed-ml, data-parallelism, model-parallelism, all-reduce, parameter-server, synchronous-sgd, effective-batch]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Spread the work across a cluster to train bigger models on more data, faster, then pay for it by keeping every machine in step.

Built from the course lecture "dml-s1-foundations" (Lecture Library series), extended with runnable measurements.

## The idea in plain words

Picture a restaurant kitchen on the busiest night of the year. One cook cannot make five hundred dishes, so you hire more cooks. There are really only two ways to divide the work. You can give every cook the same recipe book and a different pile of orders, which is **data parallelism**. Or the dish itself is so elaborate that one cook cannot hold all of it in their head, so one cook prepares the sauce, another the base, another the garnish, and the plate moves along the line, which is **model parallelism**.

Either way a new problem appears that a single cook never had: the cooks must agree. If five cooks each season the same sauce differently, the dishes do not match. In machine learning, "agreeing" means that after every training step the workers must end up with the same weights. All the engineering in this part of the site is about doing that agreement cheaply.

There are two reasons to leave one machine at all.

- **It does not fit.** The weights, the gradients and the optimiser state of a large model can exceed the memory of one device.
- **It is too slow.** Even a model that fits may take weeks on one device, and a cluster divides the wall-clock time.

The cost of leaving is communication and waiting. Spreading eight workers over a problem rarely makes it eight times faster, and the rest of this chapter, and the next three, measure why.

<Infographic src="/img/dist/why-distribute-machine-learning-data-vs-model.svg" alt="Data parallelism replicates the model on every worker and averages gradients; model parallelism splits one model over four GPUs; a table shows fp32 Adam training needs 16 bytes per parameter" caption="Two ways to split the work, and the memory arithmetic that forces the second one. Every figure is printed by block 1 below." />

<Infographic src="/img/dist/why-distribute-machine-learning-sync-async-traffic.svg" alt="Three tables: synchronous efficiency falls from 1.000 to 0.572 as workers grow to 64, asynchronous staleness makes the loss diverge, and a single parameter server carries N times the gradient while a ring carries under 2" caption="What synchronising costs. The numbers come from blocks 2 and 3 below." />

## How it works

The lecture's whole argument is one sentence: **split the work, then synchronise.** The sections below follow its order.

### Data vs model

- **Data parallelism.** Replicate the model, shard the data, and synchronise gradients. It is limited by communication.
- **Model parallelism.** Split one model across devices (pipeline or tensor parallelism) when it will not fit on one.

### Parameter server, all-reduce

Parameter servers centralise updates; ring all-reduce sums gradients peer to peer, so there is no central bottleneck. Synchronous training is consistent but bound by the slowest worker (the straggler); asynchronous training is fast but uses stale gradients.

:::tip

**Worked.** A local batch of 32 on 8 workers gives an effective batch of 256, so scale the learning rate (the linear rule).

:::

### The same ideas, one level deeper

:::note Beyond the lecture
The lecture states each point in a line. Here is what each one means in numbers, and block 1 to block 3 below compute all of them.

**Why a model does not fit.** Training with the Adam optimiser in 32-bit floating point keeps four numbers per parameter: the weight, its gradient, and two moving averages. That is 4 + 4 + 4 + 4 = 16 bytes per parameter before a single activation is stored. A 7-billion-parameter model therefore needs 112 GB for state alone, more than most single accelerators hold. Mixed-precision recipes change the bytes per parameter, so treat 16 as the plain fp32 baseline, not a law.

**Effective batch.** Each of K workers processes B examples per step, and the averaged gradient is the gradient of a batch of K x B. With K = 8 and B = 32 that is 256. The optimiser sees a different problem from the one it saw on one worker: fewer, bigger, less noisy steps. The linear scaling rule says to multiply the learning rate by K to compensate. The next chapters test that rule.

**Parameter server against all-reduce.** In a parameter server design, workers push gradients to a server group that holds the shared parameters and pull fresh weights back. If one server receives every gradient, its inbound traffic grows with the number of workers. In a ring all-reduce each worker sends only 2(N-1)/N times the gradient size, which tends to 2 whatever N is. Chapter 2 builds the ring.

**Synchronous against asynchronous.** A synchronous step ends when the slowest worker ends. If step times vary, the cluster runs at the speed of the maximum, not the average. Asynchronous training removes the wait, but each gradient was computed against weights that have since moved on.
:::

## A real system that works this way

**A published one-hour ImageNet run.** Goyal and colleagues trained ResNet-50 on ImageNet with a minibatch of 8192 images on 256 GPUs in one hour, matching the accuracy of a small-minibatch baseline. Their paper's two ingredients are the one the lecture calls the linear scaling rule ("when the minibatch size is multiplied by k, multiply the learning rate by k") and a warm-up phase that ramps the rate up over the first five epochs. They also report that the rule stops holding well beyond a minibatch of about 8k on their setup, so it is a working range, not a law.

**The parameter server.** Li and colleagues at OSDI 2014 describe a system in which data and workload are spread over worker nodes while server nodes hold the globally shared parameters, with asynchronous communication and flexible consistency models. It is the design the lecture contrasts with all-reduce, and the paper reports it running on petabytes of real data with billions of examples and parameters.

**The move to all-reduce.** The Horovod paper (Sergeev and Del Balso, 2018) is explicit about the motivation: it replaces the traditional parameter server approach with ring-allreduce, and says existing multi-GPU TensorFlow methods carried non-negligible communication overhead and needed heavy changes to user code.

## Code you can run

Three blocks, all deterministic. They use only numpy (Python 3.14, numpy 2.5.3).

### 1. Effective batch, learning rate and memory

```python
local_batch = 32
workers = 8
base_lr = 0.1
effective_batch = local_batch * workers
print("effective batch:", effective_batch)
print("linear-rule learning rate:", round(base_lr * workers, 2), "(base", base_lr, "x", workers, ")")

bytes_per_param = {"weights": 4, "gradients": 4, "adam first moment": 4, "adam second moment": 4}
per_param = sum(bytes_per_param.values())
print("bytes per parameter for fp32 training with Adam:", per_param)
for billions in (0.135, 1, 7, 70):
    params = billions * 1e9
    gb = params * per_param / 1e9
    print(f"{billions:>6} B parameters -> {gb:8.1f} GB before activations")

model_gb = 24
gpus = 4
print("24 GB model over 4 GPUs:", model_gb / gpus, "GB each")
```

The effective batch is 256 and the linear rule gives a learning rate of 0.8 from a base of 0.1. The memory table is the arithmetic from the note above: 16 bytes per parameter turns 7 billion parameters into 112.0 GB, and 70 billion into 1120.0 GB, long before activations. The lecture's 24 GB model over 4 GPUs is 6.0 GB each.

### 2. What the slowest worker costs

Each worker's step time is drawn from a log-normal distribution with a median of 100 ms. A synchronous step takes the maximum over workers.

```python
import numpy as np

rng = np.random.default_rng(0)
workers = 8
steps = 2000
times = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=(steps, workers))

sync_step = times.max(axis=1).mean()
mean_step = times.mean()
print(f"mean single-worker step: {mean_step:.1f} ms")
print(f"synchronous step (wait for slowest of {workers}): {sync_step:.1f} ms")
print(f"synchronous throughput relative to ideal: {mean_step / sync_step:.3f}")
print("asynchronous throughput relative to ideal: 1.000, but gradients are stale")

for k in (1, 2, 4, 8, 16, 32, 64):
    t = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=(steps, k))
    print(f"{k:>3} workers: sync efficiency {t.mean() / t.max(axis=1).mean():.3f}")
```

The mean single-worker step is 103.3 ms, but the synchronous step on 8 workers is 144.1 ms, so the cluster runs at 0.717 of its ideal throughput. The efficiency column keeps falling as workers are added: 0.876 at 2 workers, 0.712 at 8, 0.572 at 64. Nothing is wrong with any one worker. The maximum of more random numbers is simply larger. The step-time spread is a modelling choice here, so read the trend rather than the exact values.

### 3. What stale gradients cost, and who carries the traffic

Gradient descent on a least-squares problem, where the gradient used at each step was computed `delay` steps ago. Then the traffic a single server and a ring carry.

```python
import numpy as np

rng = np.random.default_rng(1)
n, d = 400, 20
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = X @ w_true + 0.1 * rng.normal(size=n)
L = np.linalg.eigvalsh(X.T @ X / n).max()
lr = 0.5 / L


def run(delay, steps=25):
    w = np.zeros(d)
    history = [w.copy()]
    for _ in range(steps):
        stale = history[max(0, len(history) - 1 - delay)]
        grad = X.T @ (X @ stale - y) / n
        w = w - lr * grad
        history.append(w.copy())
    return 0.5 * np.mean((X @ w - y) ** 2)


for delay in (0, 1, 2, 3, 4, 5):
    print(f"gradient staleness {delay:>2} steps: loss after 25 steps {run(delay):.6f}")

print()
print("workers  parameter-server bytes in per step  ring bytes sent per worker (gradient = 1.0)")
for k in (2, 4, 8, 16, 64):
    print(f"{k:>7}  {k:>34.1f}  {2 * (k - 1) / k:>38.3f}")
```

With a fresh gradient the loss after 25 steps is 0.004025, and one step of delay is harmless (0.004024). Two steps of delay at the same learning rate already hurts (0.027851), and three or more make the run diverge (4.520328, 29.170128, 100.522979). The threshold depends on the learning rate and the curvature, so this shows the mechanism, not a universal limit: the larger the step you take, the less staleness you can tolerate. The second table is the traffic argument. A single server receiving every gradient sees 2, 4, 8, 16 and 64 times the gradient size as workers grow, while a ring sends 1.000, 1.500, 1.750, 1.875 and 1.969 per worker.

## Designing with it

| Question | Guidance |
| --- | --- |
| Does the model, its gradients and optimiser state fit on one device? | If yes, start with data parallelism: the model code does not change. If not, you need to shard state or split the model (chapters 3 and 4). |
| Is one step too slow, or is one run too long? | Both point to more workers, but the speedup is capped by the communication and waiting you measure, not by the worker count. |
| Synchronous or asynchronous? | Default to synchronous: results are reproducible and match a single-process run. Consider asynchrony only when stragglers dominate and your optimiser tolerates stale gradients. |
| Parameter server or all-reduce? | For dense gradients of a deep network all-reduce is the common choice, as the traffic table shows. |
| What is my effective batch? | Compute K x B first. If it changes, change the learning rate and use warm-up, then check the validation curve. |

**A habit worth keeping.** Before buying more workers, measure one worker's step time and the time spent communicating. The ratio tells you the ceiling before you have spent anything.

## Where this stands in 2026

:::info Industry view
- **The synchronous data-parallel default is built into PyTorch.** The `DistributedDataParallel` notes describe broadcasting the model state from rank 0 so that all replicas start identical, grouping gradients into buckets, and starting an asynchronous all-reduce as soon as a bucket is ready so that communication overlaps the backward pass (PyTorch 2.14 documentation).
- **Sharding is the answer when state does not fit.** PyTorch's `FullyShardedDataParallel` shards parameters, gradients and optimiser state across the data-parallel workers, and the documentation says it draws on the ZeRO stage 3 design. The FSDP2 page (`fully_shard`, last updated 24 April 2026) tells users of the first implementation to consider migrating.
- **You rarely wire it by hand.** Ray Train describes itself as a scalable library for distributed training and fine-tuning that takes PyTorch, PyTorch Lightning, Hugging Face Transformers and Accelerate, DeepSpeed and other frameworks from one machine to a cluster. Spark's MLlib remains the choice when the data already lives in a Spark cluster; its DataFrame-based `spark.ml` API is the primary one and the RDD-based API is in maintenance mode (Spark 4.2.0 guide).
- **Backends.** The PyTorch documentation recommends NCCL for CUDA GPUs and Gloo for CPU training, which is why the two-process examples in chapters 2 to 4 use Gloo on this CPU-only machine.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why distribute machine learning?</summary>

Because models and datasets outgrow a single machine's memory and compute, and training is too slow; a cluster trains larger models on more data, faster.

</details>

<details>
<summary><strong>Q2.</strong> Contrast data and model parallelism.</summary>

Data parallelism replicates the model and shards the data, synchronising gradients; model parallelism splits one model across devices when it will not fit.

</details>

<details>
<summary><strong>Q3.</strong> Contrast the parameter server and all-reduce.</summary>

In a parameter server design, workers push gradients to central servers that update and serve the weights. In all-reduce, peers sum gradients directly with no central bottleneck, which is the usual choice for dense deep-network gradients.

</details>

<details>
<summary><strong>Q4.</strong> Local batch 32 on 8 workers. Give the effective batch and its learning-rate implication.</summary>

Effective batch = 32 x 8 = 256. Scale the learning rate up, roughly 8 times (the linear scaling rule), usually with a warm-up.

</details>

<details>
<summary><strong>Q5.</strong> Contrast synchronous and asynchronous SGD.</summary>

Synchronous waits for all workers: consistent, but stragglers slow every step. Asynchronous never waits: faster, but gradients are computed on stale weights.

</details>

<details>
<summary><strong>Q6.</strong> A 7-billion-parameter model is trained in fp32 with Adam. How much memory do the weights, gradients and optimiser state need, ignoring activations?</summary>

Four numbers per parameter (weight, gradient, two Adam moments) at 4 bytes each is 16 bytes per parameter, so 7 x 10^9 x 16 = 112 GB. It does not fit on one typical accelerator, which is the reason to shard.

</details>

<details>
<summary><strong>Q7.</strong> In block 2, eight workers each average 103.3 ms per step but the synchronous step takes 144.1 ms. Why, and does adding workers help or hurt this ratio?</summary>

A synchronous step lasts as long as its slowest worker, and the maximum of eight random times exceeds their mean. Adding workers makes it worse: the efficiency in the block falls from 0.876 at 2 workers to 0.572 at 64.

</details>

<details>
<summary><strong>Q8.</strong> Why does one step of gradient staleness do no harm in block 3 while three steps diverge?</summary>

The update uses a gradient from older weights, so it can overshoot. A small delay is absorbed when the learning rate is below the stability limit for that delay, and the limit shrinks as delay grows. At the block's learning rate it is exceeded between one and two steps of delay.

</details>

## Further reading

- [Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)](https://arxiv.org/abs/1706.02677), the linear scaling rule and warm-up. Opened 2 October 2026.
- [Li et al., "Scaling Distributed Machine Learning with the Parameter Server" (OSDI 2014)](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu), the parameter server design. Opened 2 October 2026.
- [Sergeev and Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow" (2018)](https://arxiv.org/abs/1802.05799), ring all-reduce replacing the parameter server. Opened 2 October 2026.
- [PyTorch 2.14 notes on DistributedDataParallel](https://docs.pytorch.org/docs/2.14/notes/ddp.html) and the [FSDP2 page](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html).
- [Ray Train overview](https://docs.ray.io/en/latest/train/train.html) and the [Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html).
- [Dive into Deep Learning, "Computational Performance" (version 1.0.3)](https://d2l.ai/chapter_computational-performance/index.html), multi-GPU training and parameter servers with runnable code.
- On the site: [mini-batch gradient descent](/docs/theory/dnn/gradient-descent-in-neural-networks-batch-sgd-and-mini-batch) and the [Adam optimiser](/docs/theory/dnn/adam-optimizer).

## Check yourself

- I can explain why a model that fits on one machine may still need a cluster, and why one that does not fit needs more than data parallelism.
- I can compute the effective batch size and the linear-rule learning rate for K workers.
- I can work out the fp32 Adam memory for a model of a given size.
- I can say why a synchronous step runs at the speed of the slowest worker and show how efficiency falls with worker count.
- I can explain why stale gradients make a given learning rate unstable.
- I can compare a single parameter server with a ring on traffic per worker.
