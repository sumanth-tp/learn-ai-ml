---
id: dist-special-topics
title: "Special Topics in Distributed ML"
sidebar_label: "2 · Special Topics in Distributed ML"
sidebar_position: 2
slug: /mlops/distributed/special-topics
description: "Communication frequency is a tuning decision whose value depends on both runtime cost and disagreement between local objectives."
tags: [distributed-ml, optimisation, training]
---

import Infographic from '@site/src/components/Infographic';
import NonIidLab from '@site/src/components/viz/NonIidLab';

**In one line.** Communication frequency is a tuning decision whose value depends on both runtime cost and disagreement between local objectives.

Built from the course lecture "dml-s15-special-topics" (Lecture Library series).

## The idea in plain words

:::note Beyond the lecture
Communication frequency is a tuning decision whose value depends on both runtime cost and disagreement between local objectives.

Picture two teams following different maps to a shared destination. Frequent meetings keep their routes aligned but interrupt walking. Infrequent meetings let them cover more ground while potentially moving away from one another. The right meeting schedule depends on how different the maps are and how costly each meeting is. A single instruction to meet less often cannot settle both concerns.

The lecture collects all-reduce, quantisation, local updates and federated aggregation under a common communication theme. That is a useful overview, but it is not a diagnosis for every job. An input-bound worker, an imbalanced pipeline or a small model with process-start overhead can be limited by something else. Measure the critical path before choosing a technique that primarily changes communication.

This chapter makes objective heterogeneity explicit. Two clients have the same number of examples in the conceptual weighting, but their preferred parameter values and curvatures differ. The lab changes the distance between local optima and the number of steps between averaging. A known central optimum lets us distinguish local agreement after a round from actually reaching the intended shared objective.

We then compare three schedules at an identical local-step budget: synchronise every step, synchronise every four steps, and an illustrative decreasing period. The costs are named synthetic parameters. The decreasing schedule demonstrates the motivation for adaptation; it is not passed off as the full AdaComm implementation or as a replication of its published benchmark.
:::

<Infographic src="/img/dist/special-topics-noniid.svg" alt="Special Topics in Distributed ML: the mechanism and checked example values" caption="Original board. Values reproduce the independent code blocks below; synthetic costs are labelled." />

## How it works

It all comes down to one dial: **how often do the workers talk?**

- **What you'll learn** , Adacomm vs distributed SGD, the comm–convergence trade-off.
- **How to use it** , Compute communication reduction from local steps.
- **The one idea** , Minimise communication.

### Local + adaptive sync

Distributed SGD syncs every step; Adacomm does τ local steps and adapts τ, cutting communication with little accuracy loss.

:::tip

**Worked.** τ=4 → 4× fewer communication rounds (one per 4 steps).

:::

### Minimise communication

Efficient all-reduce, quantisation, local updates, federated aggregation , every technique reduces communication, because moving gradients dominates distributed-ML cost.

:::note Corrections and assumptions
The lecture says communication dominates distributed ML and that adaptive local steps lose little accuracy. These are workload-dependent observations, not universal laws. The code gives a counterexample to an unconditional negligible-loss reading: period four ends with excess objective 0.035284 on the default heterogeneous problem. It confirms four-fold fewer rounds only for the divisible twenty-four-step budget. The decreasing schedule is an addition illustrating the idea, not a reproduced AdaComm algorithm.
:::

:::note Beyond the lecture
### Non-IID is a family of differences

Client datasets can differ in label proportions, feature distributions, sample counts, missingness, collection time or the relationship between inputs and targets. The phrase non-IID does not identify which one applies. These differences can change gradient directions, magnitudes, noise and curvature. A useful experiment specifies the difference it creates and which quantities it keeps fixed.

Our example models objective heterogeneity through two scalar quadratics. The curvature vector is [1,4], and the preferred local values are [−gap,+gap]. Both clients have equal global weight. This is not a class-partitioned dataset or a realistic population model; it is a controlled example in which the central target can be calculated exactly.

The weighted global derivative is [1(w+gap)+4(w−gap)]/2. Setting it to zero gives w*=0.6gap. When gap is zero, the shared initial value of zero is already optimal for both clients. The code therefore reports zero disagreement and zero excess loss for all three tested periods. It would be misleading to call this evidence that local updates have no heterogeneity cost in general: the chosen starting point makes the zero-gap case trivial.

With gap one, period one yields final model 0.599398 after twenty-four steps, very close to optimum 0.6. Period four yields 0.431989 and period eight yields 0.263435. Their excess losses are 0.035284 and 0.141595. The final pre-average client gaps are 0.988154 and 1.448040 respectively. Averaging resets disagreement to zero at every round boundary, but it does not erase the effect of the preceding local paths.

### Why averaging local optima gives the wrong target

The average of the clients' preferred values is zero. The optimum of their average objective is 0.6 because the second objective has four times the curvature. These are different operations. Averaging minimisers generally does not commute with minimising an average. A client that has almost reached its own optimum can therefore send a model that steers the global aggregate away from the chosen shared optimum.

The objective's curvature is 2.5, so its excess loss is 1.25(w−0.6gap)². This expression evaluates every configuration against the same target. A low client disagreement can coexist with high global error, and a nonzero pre-average disagreement can coexist with a reasonable global model. Use both measurements for diagnosis; neither replaces a validation metric relevant to the deployed task.

The lab's budget is twenty-four local steps per client. It includes a partial final round if the period does not divide twenty-four. The table exposes the endpoint of each round, both local models, the global average, client gap and excess objective. Changing the gap leaves the curvature, step size and budget fixed, making the effect of this particular heterogeneity parameter easier to interpret.

### Adaptation changes a schedule, not an objective

The [AdaComm paper](https://arxiv.org/abs/1810.08313) motivates less frequent averaging early and more frequent averaging later. It analyses error against runtime rather than only iterations. A deployment-oriented reading is that communication can be saved while large errors dominate, but tighter agreement can become valuable as optimisation approaches its target.

Our illustrative schedule is 4,4,4,2,2,2,1,1,1,1,1,1 local steps across twelve rounds. It totals twenty-four local steps. At ten milliseconds per local step and forty per communication round, its modelled time is 720 ms. Its excess loss is 0.000285, compared with 0.035284 at a constant period four and a modelled 480 ms. Always synchronising uses 1200 ms and an excess loss that rounds to zero.

Those three rows do not identify the best configuration without a quality target. If the threshold is loose, the constant period may suffice. If the target is tighter, the additional rounds may be worthwhile. A measured adaptive implementation would also pay for its diagnostics and decisions. The simple schedule has no measured scheduler cost and does not react to observed loss; it is labelled illustrative throughout.

### Heterogeneity-aware methods ask different questions

The [FedProx paper](https://arxiv.org/abs/1812.06127) adds a proximal term to local training in a heterogeneous setting. Its motivation includes controlling how far local optimisation moves from a supplied reference. The coefficient affects local progress and agreement, so describing it as a guaranteed cure for any non-IID dataset would be too strong.

The [SCAFFOLD paper](https://arxiv.org/abs/1910.06378) introduces control variates to correct client drift. That changes more than the schedule: it adds state and a corrected update rule. Neither method is implemented by the lab. They are concrete directions to investigate when a measured error comes from objective heterogeneity rather than simply excessive network traffic.

There is no need to add such state before establishing a central baseline, a correct weighted aggregate and the actual participation pattern. An incorrect denominator or an inconsistent model version can imitate drift. Fix those implementation errors first; only then ask whether a more sophisticated optimiser addresses the remaining phenomenon.
:::

## A real system that works this way

:::note Beyond the lecture
AdaComm is the source's named adaptive communication strategy, and its [original paper](https://arxiv.org/abs/1810.08313) was opened on 5 October 2026. The runnable comparison demonstrates its motivation using a transparent schedule, not its full method. No published training-time multiplier is claimed as reproduced here.

[FedProx](https://arxiv.org/abs/1812.06127) and [SCAFFOLD](https://arxiv.org/abs/1910.06378) provide primary references for two different ways to address heterogeneous local optimisation. Their names identify actual algorithms, while the NumPy code identifies exactly the simpler algorithm this page executes. This keeps the theory pointers separate from the verified implementation.

The [federated learning chapter](/docs/mlops/distributed/federated-learning) covers weighted rounds and privacy boundaries. A heterogeneity-aware optimiser does not automatically provide secure aggregation, a privacy guarantee or representative participation. These concerns remain separate even when the optimiser improves a validation curve.
:::

## Code you can run

These independent blocks run on CPU using Python 3.14.6, NumPy 2.5.3 and PyTorch 2.14.1 where imported. Every input is synthetic. No dataset download or accelerator is needed.

### 1. A controlled heterogeneity sweep

The default lab is gap one, period four and learning rate 0.1. It reproduces global weight 0.431989, optimum 0.600000, final client gap 0.988154 and excess loss 0.035284 after six rounds. Doubling gap to two gives excess loss 0.141138 for period four; the larger discrepancy is an objective effect, not a communication measurement.

```python
import numpy as np
for gap in [0, 1, 2]:
    h = np.array([1., 4.])
    a = np.array([-gap, gap], dtype=float)
    optimum = np.sum(h*a)/h.sum()
    for period in [1, 4, 8]:
        w = 0.0
        disagreement = 0.0
        for start in range(0, 24, period):
            local = np.full(2, w)
            for step in range(min(period, 24-start)):
                local -= 0.1*h*(local-a)
            disagreement = abs(local[1]-local[0])
            w = local.mean()
        excess = 0.25*h.sum()*(w-optimum)**2
        print('gap', gap, 'period', period, 'weight', f'{w:.6f}',
              'optimum', f'{optimum:.6f}', 'client gap', f'{disagreement:.6f}',
              'excess loss', f'{excess:.6f}')
```

### 2. Compare schedules at the same local-step budget

All three schedules use twenty-four local steps. The illustrative decreasing schedule uses twelve rounds and yields excess loss 0.000285 in a modelled 720 ms. These costs are calculator parameters, not results from a device, network or cloud provider.

```python
import numpy as np
h = np.array([1., 4.])
a = np.array([-1., 1.])
for name, schedule in [('always 1', [1]*24), ('always 4', [4]*6),
                       ('illustrative decreasing', [4]*3+[2]*3+[1]*6)]:
    w = 0.0
    for period in schedule:
        local = np.full(2, w)
        for step in range(period):
            local -= 0.1*h*(local-a)
        w = local.mean()
    optimum = np.sum(h*a)/h.sum()
    excess = 0.25*h.sum()*(w-optimum)**2
    print(name, 'steps', sum(schedule), 'rounds', len(schedule),
          'excess loss', f'{excess:.6f}', 'illustrative ms', 240+40*len(schedule))
```


<NonIidLab />

## Designing with it

:::note Beyond the lecture
### Identify the limiting resource

Trace local compute, input wait, communication wait and aggregation work separately. If local data decoding is the largest cost, reducing gradient messages may have little effect. If a worker is much slower than its peers, reducing the number of barriers can help but still leave long round tails. If models cannot fit in memory, message compression alone does not solve the full storage problem.

Avoid using cluster utilisation as the final objective. A busy cluster can still optimise the wrong weighted population or take many more updates to achieve its target. Use a quality-and-time curve with a clear evaluation cadence. Include failed and retried rounds when estimating operational cost; counting only successful rounds makes a fragile configuration appear cheaper than it is.

### Measure heterogeneity directly enough to act

Inspect permitted aggregate statistics that indicate why local objectives differ. Compare client counts, label shares, feature scale, local loss and update magnitudes where disclosure is allowed. Do not conclude that a large vector norm means malicious behaviour: it can reflect a legitimate different population. Conversely, benign-looking averages do not prove that every client is well served.

Distinguish participation heterogeneity from objective heterogeneity. A client with intermittent connectivity contributes less often even if its data resembles the others. A consistently available client can dominate the sequence of updates. Weighted averaging within each round does not automatically make a long sequence representative of the intended population. Record accepted-client sets and define the population-level metric explicitly.

### Introduce adaptation with a measurable rule

A real adaptive policy should specify its observed signal, update frequency, permitted period range and fallback when the signal is unreliable. Validation loss, gradient disagreement and measured synchronisation time answer different questions. An adaptive decision based only on instantaneous training loss may be noisy or react to changing cohort composition rather than genuine progress.

Use bounded periods and retain a synchronous reference configuration. When a run regresses, you need an operational way to return to a known update path. Log the selected period and its reason with the model version. Otherwise a difficult loss curve becomes impossible to reconstruct from checkpoints and examples alone.

The schedule in the code is fixed in advance. It deliberately avoids implying that the programme derives its periods from AdaComm's analytical rule. To implement the real algorithm, read its full specification, reproduce its assumptions and measure the policy overhead. A paper-inspired schedule can be useful, but name it accurately.

### Keep optimiser state and privacy state distinct

A drift-correction method can store client-specific vectors or reference values. A privacy protocol can store keys, masks and recovery shares. Those states have different lifetimes and exposure requirements. Checkpointing everything in one unprotected debug artifact is not a sound design. Define what can be persisted centrally and what must stay at the client.

The simple lab has no such state beyond the current scalar and round history. That is why its numerical reference is easy to inspect. It does not demonstrate the storage footprint, lifecycle or secure integration of FedProx, SCAFFOLD or a production aggregation protocol. Do not infer those properties from the arithmetic example.

### Report the result as a conditional decision

A useful design report names the workload, budget, target metric, participating clients, local update count, reduction weighting, precision and measured costs. Then it explains which parameter changed and why that change met the target. A statement such as period four reduces communication is incomplete without whether the target remains reachable and how many steps the run needs.

For our synthetic objective, every row can be checked against w*=0.6gap. In a real model there may be no known optimum, so rely on an agreed held-out metric and a strong central or synchronous reference. Say which controls were tested and which inference remains unverified. That honesty makes a small reproducible example more useful than an unsupported general speed claim.
:::

## Where this stands in 2026

:::info Industry view
On 5 October 2026 the primary papers provide verified mechanism references, while Python 3.14.6 and NumPy 2.5.3 execute the synthetic comparison. No FedProx, SCAFFOLD or AdaComm package was installed or benchmarked. Their descriptions are pointers to the original algorithms, with the implementation boundary stated beside the code.

The central engineering lesson is to minimise time and cost subject to an accepted quality target. Communication is one input to that decision. Heterogeneity, sampling, privacy, recovery and memory can change which optimisation is appropriate. The [question bank](/docs/mlops/distributed/question-bank) collects the course's concise answers with these assumptions restored beside them.
:::

## Practice questions

Exam-style questions on special topics in distributed ML.

<details>
<summary><strong>Q1.</strong> What does Adacomm change relative to standard distributed SGD?</summary>

It performs several local updates between synchronisations and adapts how many, cutting communication instead of syncing every step.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> State the central trade-off in distributed ML.</summary>

The communication–convergence trade-off: syncing often is accurate but expensive; syncing rarely is cheap but risks drift.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> If workers sync every τ=4 local steps, what is the communication reduction?</summary>

4× fewer communication rounds (one per four steps).<br /><em>Session 15 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why does Adacomm shrink τ as training proceeds?</summary>

Early on, large τ is safe (gradients are forgiving); near convergence, smaller τ / more frequent sync keeps workers consistent for accuracy.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What single theme unifies the distributed-ML techniques in this course?</summary>

Minimising communication (all-reduce, quantisation, local updates, federated aggregation) , because moving gradients/data, not computing them, dominates cost.<br /><em>Session 15 · conceptual</em>

</details>

:::note Answer qualifications
The source questions and answers above are retained. Read them with the corrections and assumptions in this chapter; concise source answers are not unconditional guarantees.
:::


<details>
<summary><strong>Q6.</strong> Why is the global optimum 0.6 when the local optima average to zero?</summary>

The global derivative weights each local displacement by its curvature. The steeper positive-target objective has greater derivative magnitude, so minimising the average differs from averaging the minimisers.

</details>

<details>
<summary><strong>Q7.</strong> Is the decreasing schedule in this chapter the AdaComm algorithm?</summary>

No. It is a predetermined illustrative schedule with a fixed local-step budget. AdaComm uses its own adaptive rule and assumptions; those were not implemented here.

</details>


## Further reading

- [AdaComm: adaptive communication strategies](https://arxiv.org/abs/1810.08313): error-runtime motivation.
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127): FedProx.
- [SCAFFOLD](https://arxiv.org/abs/1910.06378): control variates and client drift.
- [Federated learning on this site](/docs/mlops/distributed/federated-learning): weighting, participation and privacy boundaries.


## Check yourself

- I can state which form of heterogeneity an experiment changes.
- I can explain why averaging local minimisers can miss the global optimum.
- I can reproduce the period-four default and interpret its excess loss.
- I can distinguish an illustrative adaptive schedule from an implemented paper algorithm.
- I can evaluate communication savings together with quality and operational cost.
