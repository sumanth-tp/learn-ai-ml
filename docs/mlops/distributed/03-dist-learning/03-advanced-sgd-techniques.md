---
id: dist-advanced-sgd
title: "Advanced SGD Techniques"
sidebar_label: "3 · Advanced SGD Techniques"
sidebar_position: 3
slug: /mlops/distributed/advanced-sgd-techniques
description: "Local SGD reduces communication frequency by letting workers take several separate steps before averaging their models."
tags: [distributed-ml, optimisation, training]
---

import Infographic from '@site/src/components/Infographic';
import LocalSgdLab from '@site/src/components/viz/LocalSgdLab';

**In one line.** Local SGD reduces communication frequency by letting workers take several separate steps before averaging their models.

Built from the course lecture "dml-s12-advanced-sgd" (Lecture Library series).

## The idea in plain words

:::note Beyond the lecture
Local SGD reduces communication frequency by letting workers take several separate steps before averaging their models.

Suppose two people are editing copies of the same document. After every sentence they can compare copies, or they can each write a paragraph before reconciling. Comparing less often saves coordination, but the paragraphs can head in different directions. The amount of reconciliation work depends on how similar their starting material and editorial judgement are. Distributed optimisation has a corresponding trade-off between communication and local drift.

The lecture introduces asynchronous updates, low-precision messages and local steps as ways to reduce waiting and talking. They modify different parts of execution. Asynchrony changes when a worker's update is accepted. Quantisation changes the representation of a message. Local SGD changes how many optimiser steps occur before replicas exchange models. Treating all three as synonyms makes it difficult to reason about correctness or to measure the source of an improvement.

This chapter focuses its interactive experiment on local steps. Two workers minimise different quadratic functions. Their global objective is explicitly the equally weighted mean. They start from the same weight, take the same number of steps, and synchronise at a selectable period. Because the problem has a known central optimum, the result can be checked against a meaningful quality target rather than just against another run.

For the message-level mechanism, see [gradient compression](/docs/mlops/distributed/advanced-distributed-deep-learning). For delayed updates, see [distributed regression](/docs/mlops/distributed/distributed-linear-and-logistic-regression). Both are useful neighbours, but the lab below holds those choices fixed so that communication frequency is the only systems lever under examination.
:::

<Infographic src="/img/dist/advanced-sgd-local-steps.svg" alt="Advanced SGD Techniques: the mechanism and checked example values" caption="Original board. Values reproduce the independent code blocks below; synthetic costs are labelled." />

## How it works

Cut the two costs of distributed SGD: **waiting** and **talking**.

- **What you'll learn** , Sync/async, Hogwild, quantisation, local SGD, Adacomm.
- **How to use it** , Compute quantisation savings.
- **The one idea** , Trade a little accuracy for less communication.

### Hogwild & low precision

Asynchronous SGD (Hogwild, Downpour) drops synchronisation for speed at the cost of stale gradients. Quantisation sends low-precision gradients.

:::tip

**Worked.** 32-bit → 8-bit = 32/8 = 4× less communication per step.

:::

### τ local steps; Adacomm

Local SGD runs τ local steps between syncs, cutting communication rounds by τ×; Adacomm tunes τ adaptively , trading a little convergence for large savings.

:::note Corrections and assumptions
The lecture's 32-to-8-bit factor of four is the coordinate-payload ratio, excluding scales, indices, padding and encoding work. Local-step communication is reduced by exactly τ only for a divisible fixed step budget with one communication per round. Asynchronous execution is not always faster to an accepted quality target. Hogwild! specifically exploits favourable shared-memory sparsity; it should not be treated as a synonym for every networked asynchronous architecture.
:::

:::note Beyond the lecture
### Distinguish three update schedules

Synchronous SGD evaluates a local stochastic gradient at a common model version, combines the contributions and updates every replica. Workers wait for the required collective participation. A delayed worker can therefore delay an entire step, even if the other workers have completed their arithmetic. Techniques such as better partitioning or overlapping communication can address that waiting without changing the mathematical update.

Asynchronous SGD accepts contributions as they arrive. Different contributions can have been calculated at different versions. This can increase hardware activity while making the optimisation path less predictable. A smaller loss per second is the relevant benefit; a higher count of accepted updates alone is not enough. Some workloads tolerate delay, and others become unstable without an adjusted step size or a delay policy.

Local SGD starts each round from a shared model but then lets each worker evolve independently for τ local steps. The server or collective averages the resulting model values at the round boundary. The local gradients are evaluated at different parameter values after the first step. Consequently, averaging the final models is generally not identical to τ central gradient steps, even if the first local gradients aggregate exactly.

### Follow the quadratic example exactly

Worker zero has f₀(w)=(w+1)²/2. Worker one has f₁(w)=4(w−2)²/2. The equally weighted global objective is F(w)=[f₀(w)+f₁(w)]/2. Its derivative is [1(w+1)+4(w−2)]/2, which vanishes at w=1.4. The different curvatures are intentional. If both functions had the same curvature and only different minima, simple deterministic averaging could hide the drift effect this chapter needs to illustrate.

At τ=1 and learning rate 0.1, the mean of the local updates is exactly one central update. At τ=4, each worker evaluates three additional gradients at its own updated weight before averaging. The steeper objective pulls worker one rapidly towards two, while the flatter objective moves worker zero more slowly towards minus one. Averaging those positions follows a different effective recursion.

The budget is twenty-four local steps per worker, not twenty-four total steps across the cluster. Periods one, two, four and eight therefore use twenty-four, twelve, six and three communication rounds respectively. With the selected defaults, six rounds leave the global weight at 1.146146 and excess objective at 0.080552. The central optimum remains 1.4 in every row of the comparison.

Excess objective means F(w)−F(w*). Completing the square yields 1.25(w−1.4)² for this example. The positive constant part of F disappears from the difference, so excess loss is zero at the optimum even though individual clients cannot both have zero local loss there. This distinction prevents an incorrect comparison between a client's preferred model and the globally weighted target.

### Count incomplete rounds

If the local-step budget T is divisible by τ, communication rounds are T/τ and the nominal reduction relative to syncing every step is τ. Otherwise the final round is partial and the count is ceil(T/τ). For twenty-five steps and period four there are seven rounds, so the actual reduction is 25/7=3.571429. Quoting four without stating divisibility overstates the realised savings for that run.

The lab includes a partial final round when the selected budget is not divisible by the selected period. At each boundary it averages the two models and broadcasts that average as the next starting value. The table lists round endpoints, both pre-average local models and the global value. The plotted local paths reset to the averaged start between rounds, rather than continuing as though each worker had ignored synchronisation.

### What Hogwild! and AdaComm contribute

The [Hogwild! paper](https://arxiv.org/abs/1106.5730) studies lock-free shared-memory updates and highlights sparse interactions. It is not an all-purpose claim that arbitrary dense distributed neural updates are safe without coordination. Shared-memory write conflicts and gradients delayed across a network are related sources of inconsistency but have different execution and failure models.

The [AdaComm paper](https://arxiv.org/abs/1810.08313) treats the averaging period as a variable affecting error against elapsed time. Its motivation is to use less frequent averaging early and increase frequency to reduce the eventual error floor. This chapter verifies fixed-period arithmetic and an illustrative cost model. The special-topics chapter adds a clearly labelled decreasing schedule; neither block is represented as a reproduction of the complete AdaComm algorithm or its experiments.

The [local SGD analysis](https://arxiv.org/abs/1805.09767) provides a formal setting in which infrequent communication can retain favourable convergence behaviour. The assumptions matter: a deterministic heterogeneous quadratic illustrates a failure mode but cannot prove a theorem for arbitrary data distributions. Use the example to understand what changes, then inspect the assumptions of any convergence guarantee you intend to rely on.
:::

## A real system that works this way

:::note Beyond the lecture
The real algorithm studied in [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767) averages models periodically. The mechanism in the code is the same periodic local-update pattern, implemented in NumPy so every state is visible. It uses exact local gradients rather than noisy samples; that simplification isolates heterogeneity from random gradient noise.

AdaComm extends the decision from a fixed period to an adaptive communication strategy. Its reported experimental improvement is a paper result under the authors' workloads, not a speedup measured by this page. We deliberately avoid turning that published result into a universal forecast for another model or network.

A production system needs a clear round boundary, a way to obtain the same starting state at each round, and a rule for missing workers. A synchronous local-SGD round still waits when it eventually communicates. Less frequent synchronisation reduces the number of waits; it does not automatically remove slow participants or provide recovery from process failure.
:::

## Code you can run

These independent blocks run on CPU using Python 3.14.6, NumPy 2.5.3 and PyTorch 2.14.1 where imported. Every input is synthetic. No dataset download or accelerator is needed.

### 1. The same step budget, four periods

With period four the default lab reproduces six rounds, global weight 1.146146 and excess loss 0.080552. Period one produces weight 1.398595 and excess loss 0.000002. The printed error is measured against the explicitly derived central optimum, not against the mean of local optima.

```python
import numpy as np
curvature = np.array([1.0, 4.0])
target = np.array([-1.0, 2.0])
optimum = np.sum(curvature*target)/np.sum(curvature)
for period in [1, 2, 4, 8]:
    global_weight = 0.0
    rounds = 0
    for start in range(0, 24, period):
        local = np.full(2, global_weight)
        for step in range(min(period, 24-start)):
            local -= 0.1*curvature*(local-target)
        global_weight = local.mean()
        rounds += 1
    excess = 0.25*np.sum(curvature)*(global_weight-optimum)**2
    print('period', period, 'rounds', rounds, 'weight', f'{global_weight:.6f}',
          'optimum', f'{optimum:.6f}', 'excess loss', f'{excess:.6f}')
```

### 2. An illustrative time and payload budget

The named costs are ten milliseconds per local step and forty milliseconds per communication round. They are inputs, not fetched hardware measurements. The period-four estimate is 480 ms against 1200 ms for period one, but the quality is different. The last line checks the partial-round caveat.

```python
import math
steps, compute_ms, sync_ms = 24, 10, 40
for period in [1, 2, 4, 8]:
    rounds = math.ceil(steps/period)
    total = steps*compute_ms+rounds*sync_ms
    print('period', period, 'rounds', rounds, 'illustrative milliseconds', total)
print('32-to-8-bit ideal payload factor', 32/8)
print('25 steps at period 4, actual round factor', 25/math.ceil(25/4))
```


<LocalSgdLab />

## Designing with it

:::note Beyond the lecture
### Optimise time to a target

Before reducing synchronisation, define an accepted validation metric and threshold. A configuration that completes a fixed step budget sooner can still take longer to reach the target, or fail to reach it. Use the same evaluation data and report how frequently evaluation occurs. Evaluating only at round boundaries gives configurations different opportunities to record progress, so compare elapsed time carefully.

The cost model below is intentionally additive: T times local compute plus the number of rounds times synchronisation cost. It ignores overlap, varying worker speeds, evaluation, encoding, queueing and recovery. It is a calculator for understanding the trade-off. Replace these named parameters with measurements before budgeting a real run. If a larger period raises required steps, recompute the total rather than applying its nominal round savings to the old workload.

### Separate drift from stochastic noise

The demonstration's gradients are deterministic. Changing the period changes its error even without random sampling, so this is evidence of a different update path. In a noisy training run, observe both replica disagreement and central validation quality. Replica agreement immediately after averaging does not reveal the disagreement accumulated within the round; record the pre-average distances as well.

Check the similarity of local objectives. Equal sample counts do not imply equal distributions. Workers partitioned by customer, geography or collection time can have different feature distributions and local curvature. Random shuffling may reduce some differences in a centrally managed dataset, while federated data usually remains tied to clients. That is why the next chapter needs an explicit weighted objective and participation policy.

### State what gets averaged

Plain local SGD averages model parameters. With momentum or an adaptive optimiser, workers also have optimiser state. Keeping that state local, resetting it after a round and averaging it are distinct algorithms. Do not silently introduce one while describing another. The example uses no momentum, making the model value the entire optimisation state.

Parameters must share the same coordinate meaning and initialisation. Averaging neural models independently trained from unrelated random initial states can fail because corresponding hidden units need not represent the same features. Periodic local SGD starts from the same global model; that common starting point is part of its definition. It is not a general recipe for combining arbitrary trained checkpoints.

### Design the round and recovery contract

State whether the round includes every registered worker or a selected subset, what happens on timeout and how weights are renormalised. If a worker completes only some of its required local steps, accepting that model changes the amount of local optimisation represented by its contribution. Some algorithms allow this, but the behaviour needs to be described and measured rather than hidden as fault tolerance.

Checkpoint at an agreed boundary or store enough information to reconstruct in-progress local state. Restoring only the global model after several unrecorded local steps loses work and can repeat examples. A fault-tolerant system should know which round was committed and whether an incoming update belongs to that round. Idempotent acceptance prevents retries from giving a worker extra weight.

### Treat combinations as new experiments

Local updates and low-precision messages can both reduce communication, but multiplying their nominal savings does not establish an end-to-end speedup. A larger local period changes gradient statistics, and those changed statistics can affect quantisation error. Residuals can also interact with the round reset. Validate each mechanism first, then test the combined update against the intended objective and target metric.
:::

## Where this stands in 2026

:::info Industry view
The original Hogwild!, local-SGD and AdaComm papers remain useful mechanism references, opened on 5 October 2026. The executed environment is Python 3.14.6 and NumPy 2.5.3. No new optimiser package, accelerator or external model was installed for this chapter.

Current engineering decisions should be based on measured communication cost and acceptable quality. The example demonstrates that fewer rounds and a lower modelled time can coexist with greater excess loss. It provides an auditable baseline for deciding whether adaptation or a heterogeneity-aware method is needed; it does not establish a preferred synchronisation period for every workload.
:::

## Practice questions

Exam-style questions on advanced SGD techniques.

<details>
<summary><strong>Q1.</strong> Contrast synchronous and asynchronous SGD.</summary>

Synchronous waits for all workers (consistent, straggler-bound); asynchronous (Hogwild!, Downpour) updates without waiting (faster, stale gradients).<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What does Hogwild! exploit?</summary>

Lock-free asynchronous updates to shared parameters, relying on gradient sparsity so conflicts are rare.<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> What communication saving does 8-bit gradient quantisation give vs 32-bit?</summary>

32/8 = 4× less communication per step.<br /><em>Session 12 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How does local-update SGD reduce communication?</summary>

It runs τ local steps between synchronisations, cutting communication rounds by τ× (some convergence cost).<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What does Adacomm do?</summary>

It adaptively tunes the number of local steps τ to balance communication savings against convergence.<br /><em>Session 12 · conceptual</em>

</details>

:::note Answer qualifications
The source questions and answers above are retained. Read them with the corrections and assumptions in this chapter; concise source answers are not unconditional guarantees.
:::


<details>
<summary><strong>Q6.</strong> Twenty-five local steps are run with period four. How many rounds occur?</summary>

Seven including the partial final round. Relative to twenty-five synchronisations, the realised reduction is 25/7=3.571429, not exactly four.

</details>

<details>
<summary><strong>Q7.</strong> Why use different curvatures in the lab?</summary>

With equal deterministic curvature, averaging can commute with several local updates and conceal the drift effect. Different curvatures make the local update paths diverge from central descent.

</details>


## Further reading

- [Large Scale Distributed Deep Networks](https://research.google/pubs/large-scale-distributed-deep-networks/): Downpour and distributed training.
- [Hogwild!: a lock-free approach](https://arxiv.org/abs/1106.5730): sparse shared-memory updates.
- [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767): periodic averaging and its assumptions.
- [Adaptive communication strategies, AdaComm](https://arxiv.org/abs/1810.08313): error-runtime trade-offs.
- [Special topics](/docs/mlops/distributed/special-topics): heterogeneous objectives and an illustrative adaptive schedule.


## Check yourself

- I can distinguish message compression, asynchronous acceptance and local updates.
- I can derive the central optimum and excess loss of the two-client example.
- I can count partial communication rounds correctly.
- I can compare quality at a fixed elapsed-time target rather than claiming savings from rounds alone.
- I can specify model, optimiser and round state before proposing recovery.
