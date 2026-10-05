---
id: dist-deep-learning
title: "Advanced Distributed Deep Learning"
sidebar_label: "2 · Advanced Distributed Deep Learning"
sidebar_position: 2
slug: /mlops/distributed/advanced-distributed-deep-learning
description: "Distributed deep learning combines stochastic gradients across workers, while compression changes which information travels in each message."
tags: [distributed-ml, optimisation, training]
---

import Infographic from '@site/src/components/Infographic';
import GradientCompressionLab from '@site/src/components/viz/GradientCompressionLab';

**In one line.** Distributed deep learning combines stochastic gradients across workers, while compression changes which information travels in each message.

Built from the course lecture "dml-s11-distributed-dl" (Lecture Library series).

## The idea in plain words

:::note Beyond the lecture
Distributed deep learning combines stochastic gradients across workers, while compression changes which information travels in each message.

Think of several observers estimating the direction of a moving object. Each observation is noisy. Averaging independent observations improves precision, but observations from the same angle with the same obstruction share an error. Increasing the number of observers does not remove that common error. Mini-batch gradient averaging has the same distinction between independent noise and correlated noise.

The lecture introduces batch, stochastic and mini-batch descent, then connects stragglers and staleness to distributed execution. This chapter keeps that progression and adds a concrete message-level question: what happens when the gradient is represented with fewer bits? The first example separates variance from standard deviation. The second keeps a quantisation residual so you can see exactly what was sent and what remains unsent. The third trains a small neural network using real CPU autograd.

A batch size and a message representation are separate decisions. A bigger batch changes the gradient estimator and the number of examples per optimiser step. Fewer message bits change the approximation delivered to other workers. Neither can be evaluated by looking only at the bytes on the wire. You also need to know whether the job reaches the required quality and what computation the transformation adds.

Read [data parallelism](/docs/mlops/distributed/data-parallelism) for the baseline reduction. The new lab deliberately has a fixed small gradient rather than an invisible trained model: every coordinate appears in its data table, so changing bits or disabling residual feedback has an interpretable result.
:::

<Infographic src="/img/dist/advanced-distributed-deep-learning-compression.svg" alt="Advanced Distributed Deep Learning: the mechanism and checked example values" caption="Original board. Values reproduce the independent code blocks below; synthetic costs are labelled." />

## How it works

Distributed training is **distributed SGD** , and mini-batches are why it works.

- **What you'll learn** , Empirical risk, GD variants, variance reduction, cluster SGD.
- **How to use it** , Compute mini-batch variance reduction.
- **The one idea** , Average gradients to cut noise.

### Batch, mini-batch, stochastic

Minimise empirical risk (1/n)Σℓᵢ. Batch = accurate/slow; stochastic = noisy/fast; mini-batch = the practical middle, and it parallelises.

:::tip

**Worked.** B=32 → variance σ²/32; gradient std drops by √32 ≈ 5.7×.

:::

### Stragglers & staleness

Synchronous SGD suffers stragglers; asynchronous suffers stale gradients. Parameter-server vs all-reduce are the two architectures.

:::note Corrections and assumptions
The lecture's variance σ²/B assumes independent gradient samples with the same variance. The standard-deviation reduction is √B, not B. Its synchronous/asynchronous contrast describes typical failure modes, not a guarantee that asynchronous training is faster at a fixed quality. Quantisation and error feedback are additions to this lecture; neither is claimed to have negligible accuracy loss on an untested model.
:::

:::note Beyond the lecture
### Distinguish the objective from its estimator

The empirical risk is the mean loss over the training dataset. A stochastic step estimates its gradient using sampled examples. With uniform sampling and a correctly normalised mean loss, the mini-batch estimator targets the empirical gradient. If the sampler deliberately oversamples a class or a client, it targets a different weighted distribution unless importance weights compensate. Distributed execution does not repair a biased sampling design.

For independent, identically distributed gradient observations with coordinate variance σ², the average of B observations has variance σ²/B. Its standard deviation is σ/√B. The lecture's B=32 gives a standard-deviation reduction factor 5.656854, which rounds to 5.7. Calling this a 32-fold reduction in standard deviation confuses two different quantities.

The independent assumption matters. If every pair of observations has the same correlation ρ, the mean's variance is σ²[ρ+(1−ρ)/B]. At ρ=0.2 and B=32 with σ²=1, that is 0.225000, rather than 0.031250. The code constructs a shared random term and independent terms to reproduce this distinction. The empirical values are finite simulation estimates, not an equality promised for every draw.

Without-replacement sampling from a finite dataset has another qualification: the finite-population correction depends on the population size and the variance convention. Real examples can also have different variances. Use the simple formula as a transparent model of noise averaging, then check what the input pipeline actually samples. A larger batch does not guarantee an improvement in generalisation or a linear improvement in runtime.

### Compression is an operator with state

A quantiser maps a vector to a representation that can be decoded. Here the largest magnitude sets a shared scale, each coordinate is rounded to a signed integer, and decoding multiplies by that scale. For b bits the symmetric magnitude limit is 2^(b−1)−1. This deliberately leaves one signed integer code unused and makes the rounding rule explicit. Different libraries can use different ranges or stochastic rounding.

Our vector has eight coordinates. As raw fp32 values it would occupy 32 bytes. An ideally packed 8-bit coordinate payload occupies eight bytes; one fp32 scale brings the modelled transmission to twelve bytes. The pure payload ratio is four, while the ratio including that scale is 32/12. Real collectives also have framing, padding, bucket sizes and transport costs. The small vector makes metadata visible; it is not a representative network benchmark.

Error feedback stores the difference between what should have been sent and what was decoded. On the next transmission it adds that residual to the new vector before quantising. With a constant learning rate you can store residuals in gradient units, as this example does. If the learning rate changes, state clearly whether the residual represents gradients or parameter updates. Mixing these units can silently change the algorithm.

In the fixed-vector example, corrected = gradient + old residual, new residual = corrected − sent. Rearranging gives sent = gradient + old residual − new residual. Summing across steps cancels intermediate residuals. Starting at zero, cumulative sent plus final residual equals cumulative desired gradient. The code asserts this accounting identity. The final residual norm is not zero, so the identity does not claim lossless delivery within a finite number of transmissions.

### The order of operations is part of the algorithm

Quantising the global average is generally different from averaging individually quantised local gradients. Quantisation is nonlinear because its scale and rounding depend on the input. The lab applies one compressor to a repeated single vector; it does not implement a complete distributed compression collective. A production implementation must specify where compression occurs and whether each rank maintains its own residual.

Summing integers encoded with different scales is incorrect unless they are converted to a common representation. Top-k sparsification adds coordinate indices; those indices are part of its payload. A sparse message cannot simply be passed to the same dense all-reduce and assumed to save bandwidth. Compression must be compatible with the collective or use a different communication protocol.

The [error-feedback paper](https://proceedings.mlr.press/v97/karimireddy19a.html) motivates residual correction for biased compressors. Its optimisation claims depend on assumptions, including suitable compression properties. Our accounting check verifies the residual mechanism; it does not reproduce every theorem or establish that an arbitrary compressor preserves every model's accuracy.
:::

## A real system that works this way

:::note Beyond the lecture
[PyTorch's DDP communication-hook interface](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html) exposes gradient buckets for alternative communication behaviour. This is where a model-specific compression experiment can be attached to a replicated training loop. The hook contract includes the returned tensor and asynchronous completion, so implementing a codec is only one part of implementing a correct hook.

The DDP design notes describe bucket reduction overlapping backward computation. A codec that adds serial work can reduce this overlap, even if its payload is smaller. The net result depends on when the bucket becomes ready, encoding cost, transport time and decoding cost. The CPU neural example below validates an actual differentiable model; the real Gloo regression run in the previous chapter validates actual process communication. Neither is presented as a distributed neural compression benchmark.

These are complementary checks. Test the compressor's arithmetic on a fixed vector, test the uncompressed model's training path, and only then combine them with collective communication. This ordering makes it easier to tell whether a failure comes from the model, codec or process group.
:::

## Code you can run

These independent blocks run on CPU using Python 3.14.6, NumPy 2.5.3 and PyTorch 2.14.1 where imported. Every input is synthetic. No dataset download or accelerator is needed.

### 1. Noise averaging with independent and correlated observations

The theoretical independent factor is 32 and the standard-deviation factor is 5.656854. With the fixed seed, the mean variance is 0.031201. Adding a common term gives theoretical variance 0.225000 and empirical variance 0.225641.

```python
import numpy as np
rng = np.random.default_rng(23)
batch = 32
samples = rng.normal(size=(20000, batch))
print('independent variance factor', batch)
print('independent standard-deviation factor', f'{np.sqrt(batch):.6f}')
print('empirical variance of means', f'{samples.mean(axis=1).var():.6f}')
rho = 0.2
correlated = np.sqrt(rho) * rng.normal(size=(20000, 1)) + np.sqrt(1-rho) * samples
print('correlated theoretical variance', f'{rho+(1-rho)/batch:.6f}')
print('correlated empirical variance', f'{correlated.mean(axis=1).var():.6f}')
```

### 2. Quantisation with an explicit residual

The lab defaults reproduce one-step maximum error 0.007087 and a six-step residual norm of 0.013588. Raw fp32 bytes, packed coordinate bytes and coordinate bytes plus one scale are 32, 8 and 12. The code uses floating arrays to check the codec arithmetic; it does not pack network messages.

```python
import numpy as np
g = np.array([0.12, -0.7, 1.5, -2.0, 0.01, 0.8, -0.4, 0.3])
bits = 8
limit = 2 ** (bits-1) - 1
residual = np.zeros_like(g)
total = np.zeros_like(g)
for step in range(6):
    corrected = g + residual
    scale = np.max(np.abs(corrected)) / limit
    quantised = np.sign(corrected) * np.floor(np.abs(corrected)/scale + 0.5)
    sent = quantised * scale
    residual = corrected - sent
    total += sent
    if step == 0:
        print('one-step maximum error', f'{np.max(np.abs(g-sent)):.6f}')
print('six-step residual norm', f'{np.linalg.norm(residual):.6f}')
print('accounting error', f'{np.max(np.abs(6*g-total-residual)):.12f}')
print('raw bytes', len(g)*4, 'payload bytes', len(g)*bits//8,
      'with one fp32 scale', len(g)*bits//8+4)
assert np.allclose(total+residual, 6*g)
```

### 3. A small neural network that really trains

The CPU model's loss falls from 0.631008 to 0.297477 in thirty optimiser steps on this synthetic dataset. This is a training-path smoke check, not a held-out accuracy result. Every tensor and both network layers are real PyTorch objects.

```python
import torch
torch.set_num_threads(1)
torch.manual_seed(5)
x = torch.randn(64, 3)
y = (x[:, 0] - 0.5*x[:, 1] > 0).float()
model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
optimiser = torch.optim.SGD(model.parameters(), lr=0.2)
def loss():
    return torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), y)
initial = loss().item()
for step in range(30):
    optimiser.zero_grad()
    value = loss()
    value.backward()
    optimiser.step()
print('CPU neural network initial loss', f'{initial:.6f}')
print('CPU neural network final loss', f'{loss().item():.6f}')
assert loss().item() < initial
```


<GradientCompressionLab />

## Designing with it

:::note Beyond the lecture
### Write a baseline you can trust

Use an uncompressed optimiser run with a fixed dataset and validation protocol. Keep the number of processed examples, local batch, global batch and learning-rate schedule visible. Then change one feature at a time. Increasing the worker count while changing batch size and message precision simultaneously makes the result hard to interpret, even when the final loss looks plausible.

For a distributed comparison, start from identical model and optimiser states. Record a selected bucket before reduction and compare the reduced tensor with a central calculation. Include a tensor with zero values, mixed signs and very different magnitudes. Large outliers can determine a shared scale and make small coordinates round to zero. The lab's 0.01 coordinate beside −2.0 is included to make this effect inspectable.

### Count bytes and time at the right boundary

A claimed savings factor should say whether it describes parameter coordinates, the serialised message, per-rank traffic or whole-job traffic. If indices, scales or residual synchronisation are transmitted, include them. If a gradient bucket is converted to a different dtype but still copied through a large dense staging buffer, memory movement may remain expensive. A smaller representation can be useful without achieving its ideal bit-width ratio in elapsed time.

Measure encoding and decoding separately from collective waiting. Include warm-up and steady-state measurements without hiding failures. Where communication overlaps backward computation, consider the critical path rather than adding every component's duration. A ten-millisecond operation can have little effect if fully hidden, or dominate if it blocks the next optimiser step.

### Validate quality under the intended sampling policy

Compression can change a small gradient direction even while the norm error is modest. Watch validation metrics and the progress of rare classes or small clients, not only average training loss. A codec tuned on one tensor distribution may behave differently later in training. Store residual state in checkpoints so a resumed run does not lose accumulated unsent information without an explicit reset policy.

Gradient clipping also has an order. Clipping before residual addition, clipping after it and clipping decoded values are different operations. Choose the order based on the intended algorithm and inspect the effect on the residual. An ever-growing residual can signal that the compressor is continually discarding a direction or that clipping prevents the stored error from being delivered.

### Avoid a false proof of convergence

The telescoping identity is a useful invariant because it catches sign mistakes and forgotten residual updates. It does not tell you how the parameter trajectory changes when gradients themselves depend on the evolving model. In training, yesterday's unsent gradient was evaluated at yesterday's parameters. Convergence analysis needs smoothness, sampling and compressor assumptions beyond arithmetic conservation.

The lab intentionally repeats a fixed gradient so the invariant can be explained in isolation. Disable feedback to see cumulative error accumulate without carrying the remainder. Then lower the bits and inspect the table. This experiment supports a mechanism-level understanding; claims about real task accuracy require a trained distributed experiment and a held-out evaluation.
:::

## Where this stands in 2026

:::info Industry view
On 5 October 2026 the checked API reference is PyTorch 2.14; code runs with 2.14.1 on CPU. Gradient buckets and communication hooks remain practical extension points, but a documented extension point is not evidence that a particular codec improves a particular deployment.

For design decisions, report time to a fixed validation target together with payload and encoding cost. The next chapter changes communication frequency instead of just representation. Those techniques can be combined, but their effects on the update trajectory should be measured separately before combining them.
:::

## Practice questions

Exam-style questions on distributed deep learning.

<details>
<summary><strong>Q1.</strong> What objective does distributed SGD minimise?</summary>

The empirical risk (1/n)Σ ℓᵢ(θ) via stochastic gradient steps.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast batch, mini-batch and stochastic gradient descent.</summary>

Batch uses all data (accurate, slow); stochastic uses one sample (noisy, fast); mini-batch uses a small batch (the practical middle, parallelisable).<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> By what factor does a mini-batch of B=32 reduce gradient variance?</summary>

By B: variance σ²/32; the standard deviation drops by √32 ≈ 5.7×.<br /><em>Session 11 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What degrades synchronous vs asynchronous distributed SGD?</summary>

Stragglers stall synchronous SGD; stale gradients degrade asynchronous SGD.<br /><em>Session 11 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Name the two standard distributed-SGD architectures.</summary>

The parameter server (central) and all-reduce (decentralised).<br /><em>Session 11 · conceptual</em>

</details>

:::note Answer qualifications
The source questions and answers above are retained. Read them with the corrections and assumptions in this chapter; concise source answers are not unconditional guarantees.
:::


<details>
<summary><strong>Q6.</strong> Why can two 8-bit workers not directly add their integer gradients?</summary>

If the scales differ, an integer unit represents a different real number on each worker. Decode or establish a compatible shared scale before adding.

</details>

<details>
<summary><strong>Q7.</strong> Does cumulative sent plus residual equality prove that training is unchanged?</summary>

No. It checks information accounting for the chosen sequence. Changing delivery times changes the evolving model and hence later gradients.

</details>


## Further reading

- [PyTorch 2.14 DDP communication hooks](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html): bucket-level extension interface.
- [Error Feedback Fixes SignSGD](https://proceedings.mlr.press/v97/karimireddy19a.html): residual correction and optimisation assumptions.
- [PyTorch 2.14 DDP design](https://docs.pytorch.org/docs/2.14/notes/ddp.html): bucket scheduling and overlap.
- [Advanced SGD techniques](/docs/mlops/distributed/advanced-sgd-techniques): communication frequency and local updates.


## Check yourself

- I can distinguish variance reduction from standard-deviation reduction.
- I can explain how correlated observations weaken noise averaging.
- I can reconstruct the residual accounting identity and its limits.
- I can count scales and metadata before quoting communication savings.
- I can reproduce the CPU neural training loss without claiming a benchmark.
