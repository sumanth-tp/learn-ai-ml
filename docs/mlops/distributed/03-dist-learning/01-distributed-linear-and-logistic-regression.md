---
id: dist-regression
title: "Distributed Linear and Logistic Regression"
sidebar_label: "1 · Distributed Linear and Logistic Regression"
sidebar_position: 1
slug: /mlops/distributed/distributed-linear-and-logistic-regression
description: "Distributed regression preserves the central objective when workers evaluate the same weights and aggregate gradients with the right sample weights."
tags: [distributed-ml, optimisation, training]
---

import Infographic from '@site/src/components/Infographic';
import StaleGradientLab from '@site/src/components/viz/StaleGradientLab';

**In one line.** Distributed regression preserves the central objective when workers evaluate the same weights and aggregate gradients with the right sample weights.

Built from the course lecture "dml-s10-distributed-regression" (Lecture Library series).

## The idea in plain words

:::note Beyond the lecture
Distributed regression preserves the central objective when workers evaluate the same weights and aggregate gradients with the right sample weights.

Imagine two teachers calculating the average examination score. One has marked three scripts and the other has marked nine. Averaging their two averages gives each teacher equal influence, even though one represents three times as many pupils. The correct answer combines total scores and total script counts. Distributed regression has exactly this trap: the gradient reported by a worker is often already an average over its rows.

The model may be a straight line predicting a continuous value or a sigmoid producing a binary probability. Either way, each example contributes a loss, and differentiation respects addition. This lets workers calculate separate pieces of the same objective without passing every row through a central machine. They exchange a vector with one entry per parameter. The mechanism is simple enough to inspect directly, so this chapter uses it to make the consistency contract explicit.

There are two different ways to break the contract. A worker can receive too much weight because its shard is smaller, or it can evaluate the right shard at an older model version. The first changes whose examples matter. The second changes the point at which the objective was differentiated. Both can produce a smooth-looking training curve, so a falling loss alone does not prove the implementation is the intended algorithm.

Start with the [data-parallel training loop](/docs/mlops/distributed/data-parallelism), then use this chapter to check the loss normalisation, the update version and the numerical reference. These checks apply before any decision about adding machines. A well-tested local gradient is the reference against which the distributed step should be compared.
:::

<Infographic src="/img/dist/distributed-regression-weighted.svg" alt="Distributed Linear and Logistic Regression: the mechanism and checked example values" caption="Original board. Values reproduce the independent code blocks below; synthetic costs are labelled." />

## How it works

The loss is a **sum over rows** , so the gradient is just an average across workers.

- **What you'll learn** , Row partitioning, partial gradients, GD equivalence, skew.
- **How to use it** , Average partial gradients.
- **The one idea** , Sum-of-rows ⇒ average gradients.

### Partition rows

Each worker computes a partial gradient on its rows; averaging them gives exactly the full-data gradient. Linear vs logistic differ only in the loss.

:::tip

**Worked.** [2,4,6,8] → (2+4+6+8)/4 = 5; update w ← w − η·5.

:::

### Skew & convergence

Uneven shards cause stragglers and can bias updates unless weighted by shard size. Partitioning and feature scaling drive convergence.

:::note Corrections and assumptions
The opening lecture says a sum-over-rows loss means workers can simply average partial gradients. A plain average of local **means** is exact only for equal shard sizes, or equal intended weights. For arbitrary shard sizes, use sample weighting or sum gradients and divide by the global count. The lecture's worked mean of [2,4,6,8] is correct under that equal-weight assumption. Stale gradients do not preserve synchronous central-GD equivalence.
:::

:::note Beyond the lecture
### Derive the reduction before choosing the collective

Let worker k own nₖ examples and let its local mean loss be Lₖ(w). With N examples in total, the global empirical loss is L(w)=Σₖ(nₖ/N)Lₖ(w). Its gradient is the same weighted sum of local gradients. This identity assumes the partition represents the intended dataset: duplicated rows, omitted records, different sample weights or different preprocessing change the objective before communication even begins.

For squared-error regression, use half the mean squared residual. The gradient is Xᵀ(Xw−y)/N. For logistic regression, use binary cross-entropy with logits. The gradient is Xᵀ(sigmoid(Xw)−y)/N. An intercept is simply another column of ones. The formulas have different residuals but the reduction contract is identical. In the first block, both objectives use exactly the same uneven partition to isolate that contract.

There are two equivalent implementations. Reduce local gradient sums, reduce the associated sample counts, and divide the global sum by the global count. Alternatively, reduce local mean gradients multiplied by their counts and apply the same division. A collective's name does not select the statistical weighting. `SUM` returns a sum. A framework that averages across ranks needs the local loss scaling to account for unequal numbers of examples.

Regularisation also needs a written convention. If every worker includes the same λw gradient inside its local mean objective, a correctly weighted mean includes λw exactly once because the worker weights sum to one. Summing those gradients without the corresponding normalisation duplicates the penalty. If you reduce only data gradients and add the regulariser afterwards, every replica must add the same term using the same current parameter value.

### Exactness needs a version, not just a formula

Write the global parameter version beside each gradient. In synchronous gradient descent, all workers differentiate wₜ and all replicas apply the same reduced vector to wₜ. In asynchronous training, a gradient can have been evaluated at wₜ₋d but applied to wₜ. Linearity of differentiation does not turn that stale vector into the current full gradient. Delayed updates are an optimisation algorithm with different dynamics.

The staleness example removes all dataset noise. It minimises f(w)=w²/2, whose gradient is w. Each accepted update subtracts η times a historical weight. For the first few updates, unavailable history is explicitly clamped to the initial value. This convention makes the simulation reproducible and prevents an accidental negative index from reading the latest history entry.

At learning rate 0.4, a delay of three accepted updates leaves loss 0.218751 after twenty updates. With no delay, the displayed loss rounds to zero. This example demonstrates overshoot without measuring network latency or proving divergence for every delayed run. A later loss improvement would not invalidate the observation that the trajectory is different. Conversely, this single trajectory cannot establish a universal maximum safe delay.

### Numerical equality is a tolerance-based check

Floating-point addition is order-dependent. A tree reduction and a single matrix multiplication need not produce bit-for-bit equal results, even when they represent the same real-number expression. Compare the largest absolute difference, use a tolerance suited to dtype and scale, and inspect the loss definition if the gap is larger than expected. A generous tolerance should not hide a systematically wrong normalisation.

The two-process block uses float64, sum-reduced gradients and the global count. Rank zero owns one row; rank one owns three. Both ranks compare their reduced result with a central gradient calculated on all four rows. The assertion is local to each rank, and `mp.spawn` propagates a failing child process to the parent. The final output is read from each child's result file, rather than assuming that a parent exit code means the collective did the intended arithmetic.
:::

## A real system that works this way

:::note Beyond the lecture
[PyTorch DDP's design notes](https://docs.pytorch.org/docs/2.14/notes/ddp.html) describe gradient buckets and collective reduction. The runnable example below uses the underlying `torch.distributed` API so that the count normalisation remains visible. It is a real pair of CPU processes using Gloo, with a file rendezvous and a bounded initialisation timeout. It does not simulate two ranks inside a Python list.

DDP is useful when a replicated trainable model and frequent collective updates are the right fit. A linear or logistic model can use the same process-group machinery as a neural network, although a small example may spend more time starting processes than computing gradients. That overhead is why this example is a correctness test, not a speed benchmark.

The [distributed programming models chapter](/docs/mlops/distributed/programming-models) explains how this differs from a parameter server. A parameter server can own the update operation; an all-reduce implementation usually leaves each replica applying its own identical update. Either arrangement still needs a shared objective, a clear weight convention and an agreed model version.
:::

## Code you can run

These independent blocks run on CPU using Python 3.14.6, NumPy 2.5.3 and PyTorch 2.14.1 where imported. Every input is synthetic. No dataset download or accelerator is needed.

### 1. Unequal shard sizes, two losses

The weighted gradients match the central reference to twelve displayed decimal places. A plain mean gives maximum errors 0.411543 for linear regression and 0.040139 for logistic regression. The lecture's four gradients average to 5.0.

```python
import numpy as np
rng = np.random.default_rng(17)
x = np.column_stack([np.ones(12), rng.normal(size=(12, 2))])
y = x @ np.array([0.5, 2.0, -1.0])
labels = (y > 0).astype(float)
w = np.array([0.2, -0.1, 0.3])
shards = [np.arange(3), np.arange(3, 12)]
for name, target in [('linear', y), ('logistic', labels)]:
    def gradient(rows):
        scores = x[rows] @ w
        pred = scores if name == 'linear' else 1 / (1 + np.exp(-scores))
        return x[rows].T @ (pred - target[rows]) / len(rows)
    full = gradient(np.arange(12))
    local = [gradient(rows) for rows in shards]
    weighted = sum(len(rows) * g for rows, g in zip(shards, local)) / 12
    naive = np.mean(local, axis=0)
    print(name, 'weighted error', f'{np.max(np.abs(weighted-full)):.12f}',
          'plain mean error', f'{np.max(np.abs(naive-full)):.6f}')
    assert np.allclose(weighted, full)
print('lecture gradient mean', np.mean([2, 4, 6, 8]))
```

### 2. Delayed deterministic gradient descent

The lab defaults are delay 2, learning rate 0.4 and twenty accepted updates: final weight 0.002573 and loss 0.000003. Move the delay to three to reproduce the much larger loss 0.218751. The fresh reference always uses the selected learning rate and update budget.

```python
import numpy as np
for delay in [0, 1, 2, 3]:
    history = [1.0]
    for step in range(20):
        old = history[max(0, step - delay)]
        history.append(history[-1] - 0.4 * old)
    print('delay', delay, 'weight', f'{history[-1]:.6f}',
          'loss', f'{0.5 * history[-1] ** 2:.6f}')
```

<Infographic src="/img/dist/distributed-regression-staleness.svg" alt="Fresh and delayed gradient updates have different final weights and losses at the same learning rate and update budget" caption="Original staleness board. The values come from the preceding deterministic simulation; delay is counted in accepted updates." />

### 3. A real uneven two-rank collective

Each rank prints a central-gradient error of 0.000000000000. Save this block as a Python file and execute it with the stated interpreter; the main guard is necessary for spawned processes. Loopback is chosen for this single-machine demonstration. Use the appropriate interface and rendezvous configuration when moving to several machines.

```python
import os
import tempfile
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        x = torch.tensor([[1., -1.], [1., 0.], [1., 1.], [1., 2.]], dtype=torch.float64)
        y = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
        rows = slice(0, 1) if rank == 0 else slice(1, 4)
        w = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x[rows] @ w, y[rows], reduction='sum')
        loss.backward()
        dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)
        w.grad /= len(y)
        reference = x.T @ (torch.sigmoid(x @ w.detach()) - y) / len(y)
        difference = (w.grad - reference).abs().max().item()
        assert difference < 1e-12
        Path(result_dir, str(rank)).write_text(f'{difference:.12f}')
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo0' if os.uname().sysname == 'Darwin' else 'lo'
    with tempfile.TemporaryDirectory(prefix='dml-b-gloo-') as folder:
        uri = Path(folder, 'rendezvous').as_uri()
        mp.spawn(worker, args=(uri, folder), nprocs=2, join=True)
        for rank in range(2):
            print('rank', rank, 'full-batch gradient error', Path(folder, str(rank)).read_text())
```


<StaleGradientLab />

## Designing with it

:::note Beyond the lecture
### Define the unit of work

Record whether a step represents a fixed global batch or a fixed local batch. Keeping local batch fixed while adding workers changes the number of examples per update. Keeping global batch fixed changes how much work each worker performs. Neither is inherently wrong, but comparing learning curves by update count without this distinction is misleading. Report examples processed as well as optimiser steps.

Check how the input pipeline handles the last batch. Padding can duplicate observations; dropping can omit them; allowing uneven sizes requires correct weighting and compatible collective participation. If one worker finishes early while another continues calling a collective, the problem is no longer a biased gradient: it can be a blocked job. The count and participation policies belong together.

### Separate statistical skew from runtime skew

Uneven counts create a weighting problem. Different feature or label distributions create a statistical problem. Different compute times create a scheduling problem. These can coincide but require different remedies. Repartitioning rows may improve balance without fixing feature scaling. Correct sample weights may fix the objective without reducing straggler time. Monitor all three rather than treating every slow loss curve as a network problem.

For logistic regression, compare the loss on stable logits rather than forming logs of probabilities that can round to zero. For linear regression, standardise features using statistics compatible with the intended global dataset. Computing a separate mean and variance on every shard can make identical raw rows map to different feature coordinates. A model average is meaningful only when coordinates mean the same thing.

### Debug one accepted update first

Freeze a tiny dataset and the initial parameters. Calculate the central gradient, each local contribution, the reduced vector and the post-update model. Then repeat with unequal counts and a different rank order. If these checks pass, test several steps with identical optimiser state. Momentum and adaptive statistics are state too; matching model weights alone is insufficient after resuming from a checkpoint.

When considering asynchronous updates, record the model version used to compute each gradient, the version that accepts it and the observed delay distribution. Reject or bound stale work only under an explicit policy. An asynchronous process that happens to run faster does not automatically reach the same quality earlier. Compare validation quality against elapsed time and examples, then decide whether the additional consistency machinery is justified.

### Keep recovery part of the algorithm

A restarted worker should obtain the model, optimiser and data-progress state that the surviving workers expect. Reusing its old local gradient after it has rejoined a newer global version breaks the update contract. A collective failure may require restarting the entire process group. Treat the small demonstration's temporary files as rendezvous and assertion outputs, not as a checkpoint format.
:::

## Where this stands in 2026

:::info Industry view
As checked on 5 October 2026, the relevant PyTorch reference is the 2.14 documentation and the tested package is 2.14.1 on CPU. The engineering question remains whether the workload is limited by computation, memory, communication or input preparation. This chapter verifies algebra and process communication; it does not claim a current cluster throughput figure or a best framework for every workload.

The existing [core distributed algorithms](/docs/mlops/distributed/core-distributed-algorithms) use the same principle of combining sufficient summaries. Prefer an exact reduction baseline before adding delayed, compressed or local-update variants. A clear numerical reference makes those later trade-offs measurable.
:::

## Practice questions

Exam-style questions on distributed regression.

<details>
<summary><strong>Q1.</strong> Why does regression distribute cleanly?</summary>

Its loss is a sum over data points, so the gradient is a sum too , partition rows, compute partial gradients, and average.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> How do distributed linear and logistic regression differ?</summary>

Only in the loss (squared error vs log-loss with a sigmoid); the distribution mechanism is identical.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Four workers report gradients [2,4,6,8]. Aggregate them.</summary>

(2+4+6+8)/4 = 5; then w ← w − η·5.<br /><em>Session 10 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why is distributed GD identical to single-machine GD here?</summary>

The full-data gradient equals the average of the workers' gradients for an even split , so the updates match exactly.<br /><em>Session 10 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What problems does data skew cause?</summary>

Uneven shards create stragglers and can bias updates unless weighted by shard size.<br /><em>Session 10 · conceptual</em>

</details>

:::note Answer qualifications
The source questions and answers above are retained. Read them with the corrections and assumptions in this chapter; concise source answers are not unconditional guarantees.
:::


<details>
<summary><strong>Q6.</strong> Why can identical replicas still optimise the wrong objective?</summary>

All ranks can receive the same incorrectly weighted gradient. Agreement checks consistency; a central reference and sample counts check the objective.

</details>

<details>
<summary><strong>Q7.</strong> A regulariser is included on each worker. Must it always be divided by the worker count?</summary>

No. Its scaling depends on whether the local objective and reduction are sums or means. Derive the global objective, then verify that the regulariser contributes exactly once.

</details>


## Further reading

- [PyTorch 2.14 DDP design](https://docs.pytorch.org/docs/2.14/notes/ddp.html): gradient reduction and replicated state.
- [PyTorch 2.14 distributed communication](https://docs.pytorch.org/docs/2.14/distributed.html): process groups, Gloo and collectives.
- [Delayed gradients and the error-feedback framework](https://arxiv.org/abs/1909.05350): optimisation assumptions for delayed updates.
- [Data parallelism on this site](/docs/mlops/distributed/data-parallelism): the replicated training baseline.


## Check yourself

- I can derive the correct reduction for equal and unequal shards.
- I can explain why stale gradients invalidate a synchronous equivalence claim.
- I can reproduce both regression checks and both Gloo rank assertions.
- I can distinguish model agreement, objective correctness and runtime efficiency.
