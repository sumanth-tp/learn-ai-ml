---
id: dist-midsem
title: "Distributed ML Mid-Semester Paper: Worked Solutions"
sidebar_label: "2 · Mid-semester solved"
sidebar_position: 2
slug: /mlops/distributed/midsem-solved
description: "Retained source questions and answers, original architecture and schedule redraws, and computed checks with explicit corrections."
tags: [distributed-ml, practice, exam, pipeline-parallelism]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Explain where state lives, how dependencies create waiting, and why replicas need the same correctly normalised update.

Built from the course solved paper "dml-midsem-2026" (Lecture Library series).

## The idea in plain words

The supplied converted paper contains three closed-book questions, each worth ten marks. They cover parameter servers, pipeline scheduling, and all-reduce data parallelism. Its worked answers are qualitative: there is no numeric exam answer to recompute and declare wrong. This page preserves all three questions and supplied answers, then adds labelled worked explanations and independently computed examples.

:::note Source boundary and redraws
The conversion says the question paper includes all diagrams, but its scan images are not present in the supplied text. No scan image is reproduced here. The boards are original conceptual redraws answering the stated diagram prompts. Every new number is labelled an added illustration, rather than represented as unseen exam data. Three ten-mark questions total thirty marks; that total is verified below.
:::

## How it works

For each answer, first draw the data and model placement. Then show what crosses a boundary and who commits an update. For the schedule question, label forward/backward dependencies and distinguish bubble time from live activation storage. For the collective question, derive the loss normalisation before averaging anything.

The source answers below remain visible for comparison. A labelled correction follows wherever the supplied wording overstates a property. This makes the actual correction reviewable instead of silently replacing an examination solution with a different answer.

## Practice questions and worked solutions

<details>
<summary><strong>Q1.</strong> Q1 · Parameter-Server pattern [10 M] Discuss the Parameter Server pattern with a diagram: roles of parameter server vs worker nodes, synchronous vs asynchronous gradient updates, and what the Multiple Parameter Server approach solves over a single server.</summary>

**Architecture:** one (or more) **parameter servers** hold the global model weights; many **workers** each pull the current weights, compute gradients on their data shard, and push gradients back. **Synchronous** updates: the server waits for *all* workers each step (all-reduce-like) → consistent but as slow as the straggler. **Asynchronous**: each worker updates the server independently → fast, no waiting, but uses *stale* gradients (can hurt convergence). **Multiple parameter servers** shard the weights across several servers, removing the **single-server bottleneck** (network bandwidth and memory) so updates from many workers don't all hit one machine , improving scalability and fault tolerance.


**Worked explanation beyond the supplied answer.** A worker owns a data shard and temporary compute state. A parameter server owns some or all global parameter and optimiser state. The worker pulls a model version, computes a local gradient and pushes the relevant slices. In synchronous operation, the server combines the required contributions before committing a version. In asynchronous operation it accepts versioned contributions independently; the delay between evaluation and acceptance is part of the optimisation behaviour.

A multiple-server design shards parameter coordinates. Each worker sends the appropriate slice to the server that owns it. This can spread memory and inbound traffic, but the worker still needs access to every slice its forward/backward computation requires. Shard boundaries do not by themselves guarantee a consistent whole-model snapshot. Nor do they automatically provide fault tolerance: replication, checkpointing, versioned recovery and retry handling must be designed separately.

**Correction to the supplied answer.** Sharding alleviates a bottleneck; it does not automatically remove all network bottlenecks or improve fault tolerance. A failed required shard can stop the model until recovery. Async updates avoid an all-worker barrier but are not guaranteed to reach a given quality faster.

<Infographic src="/img/dist/midsem-parameter-server-redraw.svg" alt="Three workers send gradient slices to two parameter shards, with synchronous and asynchronous update policies stated separately" caption="Original conceptual redraw for Q1. It recreates the architecture requested by the question; it does not reproduce a scan's undisclosed layout or numerical data." />

An answer should name the trade-off rather than only the arrangement: synchronous updates provide a common reduction point and can wait for slow workers; asynchronous updates increase flexibility while admitting stale gradients. Neither consistency policy proves statistical correctness unless sample weights, model versions and optimiser state are also correct. See [programming models](/docs/mlops/distributed/programming-models) for the complete mechanism.

</details>

<details>
<summary><strong>Q2.</strong> Q2 · Pipeline parallelism & 1F1B [10 M] Explain pipeline parallelism. Why do idle 'GPU bubbles' arise in a naïve pipeline, and how does 1F1B (one-forward-one-backward) mitigate it? How does interleaved 1F1B improve further?</summary>

**Pipeline parallelism** splits a deep model's layers across devices (stage 1 on GPU0, stage 2 on GPU1, …); micro-batches flow through the stages. **Bubbles:** in a naïve schedule each GPU sits idle while waiting for the first micro-batch to reach it (fill) and drain at the end → wasted compute proportional to the number of stages. **1F1B** interleaves a backward pass right after each forward once the pipeline is full, keeping every stage busy and bounding the activations memory (steady state of one forward + one backward per step) → smaller bubble. **Interleaved 1F1B** assigns each device *several non-contiguous* layer chunks, so there are more, smaller micro-stages filling the schedule , shrinking the bubble further at the cost of more inter-device communication.


**Worked explanation beyond the supplied answer.** Pipeline parallelism splits successive model operations into stages. Each micro-batch must complete stage zero's forward before stage one's forward, and stage one's backward before stage zero's backward. At the start, later stages wait for inputs. At the end, earlier stages can wait for backward work. These dependency constraints create fill and drain even when every stage has the same task duration.

**Correction to the supplied answer.** Non-interleaved 1F1B in the flush schedule does not inherently reduce the bubble relative to the all-forward/all-backward baseline. The cited primary paper explicitly describes equal bubble time and fewer outstanding forward activations. A stage first performs its warm-up forwards, then alternates forward and backward, then drains remaining backward work. Its main benefit in that comparison is bounded live activation memory.

<Infographic src="/img/dist/midsem-pipeline-redraw.svg" alt="A forward-only four-stage pipeline takes eleven ticks for eight micro-batches, leaving twelve of forty-four stage-time cells idle" caption="Original redraw of fill and drain, with added illustrative values. This forward-only board does not claim to be a full training or 1F1B schedule." />

<Infographic src="/img/dist/midsem-1f1b-redraw.svg" alt="The complete four-stage one-forward-one-backward flush schedule shows twenty-two ticks, sixty-four busy cells and peak live activation counts four, three, two, one" caption="Original 1F1B redraw. Every task is generated using its forward and backward dependencies by the runnable schedule checker; equal task duration is a synthetic assumption." />

The dependency checker below schedules the next stage-local task only if its prerequisites were completed in an earlier tick. With four stages and eight micro-batches, this equal-duration flush schedule takes twenty-two ticks. Sixty-four stage-time cells do work and twenty-four are idle, so idle fraction is 24/88=0.272727. The baseline takes the same twenty-two ticks. Its peak live activations are eight micro-batches per stage, versus [4,3,2,1] here, counting an activation as live until its backward task completes.

Be precise about the denominator. Idle fraction of the realised duration is 3/11=0.272727; bubble overhead relative to ideal compute time is 3/8=0.375000. Both numbers can describe the same timing model. Calling them contradictory loses the difference between total-duration fraction and overhead over an ideal baseline.

Interleaving assigns several smaller non-contiguous model chunks to each device. The paper shows a reduced bubble under its schedule assumptions, at the cost of increased communication and additional schedule constraints. It is a separate modification from merely switching to non-interleaved 1F1B. The code below verifies only the non-interleaved dependency schedule, not GPU performance or an interleaved runtime.

</details>

<details>
<summary><strong>Q3.</strong> Q3 · Data parallelism & All-Reduce SGD [10 M] Explain data parallelism in distributed ML. With a diagram, how does All-Reduce SGD work when multiple workers train on different data shards? Key benefits and limitations vs single-machine training.</summary>

**Data parallelism:** replicate the *same* model on every worker; split the data into shards. Each step, every worker computes gradients on its shard, then **All-Reduce** sums (and averages) the gradients across all workers so each ends with the identical averaged gradient, and all apply the same update , keeping replicas in sync (ring all-reduce passes partial sums around a ring for bandwidth efficiency). **Benefits:** near-linear speed-up with workers, handles large datasets, simple to reason about. **Limitations:** the whole model must fit on one device (doesn't help with huge models , that needs model/pipeline parallelism), communication overhead grows with workers, and very large effective batch sizes can hurt convergence/generalisation.


**Worked explanation beyond the supplied answer.** Each data-parallel rank has the same starting parameters but different examples. It computes a gradient contribution, participates in the same collective order, obtains the shared reduced gradient and applies the same optimiser update. For equal-sized local mean losses, a mean across ranks matches the global sample mean. Unequal batch sizes need weighted means or reduced sums divided by the global sample count.

<Infographic src="/img/dist/midsem-allreduce-redraw.svg" alt="Unequal shards with gradient sums two and twelve reduce to fourteen and normalise by four examples, producing a global mean three point five and model update zero point six five" caption="Original Q3 redraw. Counts, gradients and learning rate are added synthetic values, verified below; they are not asserted to come from a missing scan." />

In the added illustration, one rank owns one example with mean gradient two. Another owns three examples with mean gradient four. Their gradient sums are two and twelve. The global mean is fourteen divided by four, or 3.5. Starting from model one with learning rate 0.1, both replicas should update to 0.650000. The wrong unweighted mean of local means is three and would update both replicas to 0.700000. Identical replicas can therefore still be identically wrong.

**Qualification to the supplied answer.** Near-linear speedup is a possible workload result, not a general benefit guaranteed by the architecture. The full-model memory limit applies to plain replicated data parallelism; it can be combined with state sharding or model parallelism. In an ideal ring, per-rank sent payload approaches two model copies while the number of communication steps grows with worker count. Traffic, latency and effective batch all need separate analysis.

The previous [regression chapter](/docs/mlops/distributed/distributed-linear-and-logistic-regression) includes a real two-process Gloo sum reduction with unequal shards. This paper's block checks the smaller arithmetic illustration so the answer remains easy to inspect. It does not replace that real process test with a simulated collective.

</details>

## A real system that works this way

:::note Beyond the paper
[PyTorch's distributed documentation](https://docs.pytorch.org/docs/2.14/distributed.html) describes real process-group collectives. The [Megatron-LM pipeline study](https://arxiv.org/html/2104.04473v5) provides the primary comparison between flush schedules, live activation counts and interleaving. Its schedule analysis is used to correct Q2; none of its hardware throughput results are claimed as reproduced.
:::

## Code you can run

### 1. Marks and the added update illustration

Python 3.14.6 and NumPy 2.5.3 reproduce thirty marks, global mean gradient 3.500000 and post-update model 0.650000. The parameter-server and ring byte figures use a deliberately named thirty-two-byte gradient, four workers and two server shards. They are payload counts for the illustration, not measurements of the scan or a network.

```python
import numpy as np
marks = np.array([10, 10, 10])
counts = np.array([1, 3])
local_means = np.array([2., 4.])
local_sums = counts*local_means
mean = local_sums.sum()/counts.sum()
print('provided paper questions', len(marks), 'total marks', marks.sum())
print('illustrative gradient sums', local_sums.tolist(), 'global mean', f'{mean:.6f}')
print('correct post-update model', f'{1-0.1*mean:.6f}')
print('wrong unweighted post-update model', f'{1-0.1*local_means.mean():.6f}')
workers, shards, gradient_bytes = 4, 2, 32
print('illustrative PS inbound bytes per shard', workers*gradient_bytes/shards)
print('ring bytes sent per worker', 2*(workers-1)/workers*gradient_bytes)
print('ring bytes received per worker', 2*(workers-1)/workers*gradient_bytes)
assert mean == 3.5
assert np.isclose(1-0.1*mean, 0.65)
```


### 2. A dependency-valid 1F1B schedule

Forward and backward tasks both occupy one tick in this synthetic checker. The output confirms twenty-two ticks, sixty-four busy cells, twenty-four idle cells and peak live micro-batches [4,3,2,1]. It also computes the all-forward/all-backward baseline duration and its eight live micro-batches per stage. The tasks and dependencies are explicit; a failure to make progress raises an assertion rather than silently skipping work.

```python
import numpy as np

stages, microbatches = 4, 8
queues = []
for stage in range(stages):
    warmup = min(stages-stage-1, microbatches)
    tasks = [('F', j) for j in range(warmup)]
    for j in range(microbatches-warmup):
        tasks.extend([('F', warmup+j), ('B', j)])
    tasks.extend(('B', j) for j in range(microbatches-warmup, microbatches))
    queues.append(tasks)
completed = set()
timeline = [[] for stage in range(stages)]
live = np.zeros(stages, dtype=int)
peak = np.zeros(stages, dtype=int)
while any(queues):
    ready = []
    for stage, queue in enumerate(queues):
        if not queue:
            timeline[stage].append('idle')
            continue
        kind, mb = queue[0]
        dependencies = []
        if kind == 'F' and stage > 0:
            dependencies.append((stage-1, 'F', mb))
        if kind == 'B':
            dependencies.append((stage, 'F', mb))
            if stage < stages-1:
                dependencies.append((stage+1, 'B', mb))
        if all(task in completed for task in dependencies):
            ready.append((stage, kind, mb))
            timeline[stage].append(kind+str(mb))
        else:
            timeline[stage].append('idle')
    assert ready, 'schedule has deadlocked'
    for stage, kind, mb in ready:
        queues[stage].pop(0)
        completed.add((stage, kind, mb))
        live[stage] += 1 if kind == 'F' else -1
        peak[stage] = max(peak[stage], live[stage])
ticks = len(timeline[0])
busy = 2*stages*microbatches
idle = stages*ticks-busy
print('1F1B ticks', ticks, 'busy cells', busy, 'idle cells', idle)
print('idle fraction', f'{idle/(stages*ticks):.6f}')
print('peak live micro-batches by stage', peak.tolist())
print('all-forward/all-backward ticks', 2*(microbatches+stages-1))
print('all-forward/all-backward peak live per stage', microbatches)
print('bubble overhead divided by ideal time', f'{(stages-1)/microbatches:.6f}')
for stage, row in enumerate(timeline):
    print('stage', stage, ' '.join(row))
assert ticks == 2*(microbatches+stages-1)
assert len(completed) == 2*stages*microbatches
assert not np.any(live)
```


## Designing with it

:::note Beyond the paper
Use four questions to review an architecture answer. What objective is being optimised? What state must agree? What boundary is expensive? What failure can interrupt a committed step? A diagram that labels only workers and servers leaves too much of the algorithm unspecified.

For a parameter server, mention gradient slices, versioning and the server's optimiser state. For a pipeline, mention micro-batch dependencies, warm-up, steady state, drain and activation lifetime. For data parallelism, mention the local loss reduction, global count, collective order and matching optimiser state. These details explain the limitations directly rather than listing vague advantages.

Separate a mechanism from a measurement. Sharding can distribute memory and traffic, but needs recovery machinery before it provides fault tolerance. A scheduler can reduce activation storage while leaving the ideal bubble unchanged. A collective can keep replicas identical while normalising their gradient incorrectly. Each claim needs the appropriate reference or invariant.
:::

## Where this stands in 2026

:::info Industry view
Official documentation and primary papers were opened on 5 October 2026. The executed checks are CPU arithmetic and a dependency scheduler, with the real Gloo process example linked above. No GPU pipeline, interleaved runtime, fault-tolerant parameter-server deployment or speed benchmark was run. The supplied Q2 bubble claim needs correction; its absence of numerical answers is reported explicitly.
:::

## Further reading

- [Programming models](/docs/mlops/distributed/programming-models): parameter-server sharding.
- [Model parallelism](/docs/mlops/distributed/model-parallelism): pipeline mechanism.
- [Data parallelism](/docs/mlops/distributed/data-parallelism): replicated updates.
- [Distributed regression](/docs/mlops/distributed/distributed-linear-and-logistic-regression): a real two-rank Gloo check.
- [Efficient Large-Scale Language Model Training, section 2.2](https://arxiv.org/html/2104.04473v5): identical non-interleaved bubble time, fewer live activations and interleaving trade-offs.
- [PyTorch 2.14 distributed communication](https://docs.pytorch.org/docs/2.14/distributed.html): collectives and process groups.

## Check yourself

- I can answer all three supplied questions with an original diagram.
- I can explain why sharding and fault tolerance need separate designs.
- I can distinguish the 1F1B memory benefit from bubble reduction by interleaving.
- I can reproduce both bubble denominators without confusing them.
- I can demonstrate that model agreement alone does not prove correct sample weighting.
- I can distinguish supplied exam information from the added synthetic examples.
