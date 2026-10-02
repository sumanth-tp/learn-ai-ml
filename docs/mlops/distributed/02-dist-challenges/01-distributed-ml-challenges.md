---
id: dist-challenges
title: "Distributed ML Challenges"
sidebar_label: "1 · Challenges at scale"
sidebar_position: 1
slug: /mlops/distributed/distributed-ml-challenges
description: "The four systems problems of distributed training, consistency, fault tolerance, communication and resource management, measured with ring all-reduce, straggler and checkpoint simulations."
tags: [distributed-ml, ring-allreduce, stragglers, checkpointing, consistency]
---

import Infographic from '@site/src/components/Infographic';
import StragglerLab from '@site/src/components/viz/StragglerLab';

**In one line.** Scaling training across machines trades the old problem of too little compute for four systems problems, and communication is the one that bites first.

## The idea in plain words

Picture one chef cooking for a hundred guests and a hundred chefs cooking for a hundred guests. The second kitchen should be a hundred times faster, and for a while it is. Then the problems start that the single chef never had: two chefs salt the same pot, one stove fails halfway through service, the sauce has to be carried across the room, and the whole banquet waits for the slowest plate. Distributed machine learning is that second kitchen. Splitting the work is the easy part. Keeping the workers agreeing, alive, fed and in step is the real job.

The lecture groups the trouble into four challenges.

- **Consistency.** Do all workers see the same model at the same moment? Insisting on it (strong consistency) is correct and slow. Relaxing it (eventual consistency) is fast, but workers compute with parameters that are a little out of date.
- **Fault tolerance.** With enough machines something is always broken. A long run survives by saving its state now and then (checkpointing) and resuming from the last save.
- **Communication overhead.** Moving gradients and parameters across a network costs more time than the arithmetic once the cluster is big. This is the lecture's one idea for the chapter: communication is the bottleneck.
- **Resource management.** Workers are never equally fast. A step that waits for everyone runs at the pace of the slowest, so uneven machines, shared networks and busy neighbours all show up as lost time.

The rest of the chapter takes each one in the order the lecture does, then measures it with a small simulation so that the words "slow", "stale" and "bottleneck" become numbers.

<Infographic
  src="/img/dist/distributed-ml-challenges-ring-allreduce.svg"
  alt="Four workers in a ring each pass one chunk per step in two phases, reduce-scatter then all-gather, so a worker sends 1.5 times the model for N = 4 and 1.969 times for N = 64, while a single reducer must receive N copies; a table of one all-reduce on a 400 MB model gives 0.0813 s for the ring and 5.12 s for the single reducer at 64 workers."
  caption="Ring all-reduce, with the traffic and time figures printed by the first code block."
/>

<Infographic
  src="/img/dist/distributed-ml-challenges-stragglers-checkpoints.svg"
  alt="Four panels: synchronous step time grows from 1.184 with one worker to 4.261 with 256 and shrinks with backup workers; checkpoint overhead is lowest near every 0.5 hours; the Llama 3 405B pre-training run had 419 unexpected interruptions in 54 days; and replicas that sync less often drift further from the optimum."
  caption="Waiting, crashing and drifting, each measured by the code blocks below."
/>

## How it works

### Strong vs eventual; checkpointing

Strong consistency = exact sync (slow); eventual = drift + reconcile (fast, stale). Fault tolerance via periodic checkpointing to resume after node failures.

:::note Beyond the lecture
**What a checkpoint must hold.** Model weights alone are not enough. Optimiser state (momentum buffers, Adam moments), the learning-rate schedule position and the data-loader position all decide the next step, and block 5 shows what happens without the momentum buffer. **How often to save.** Saving costs time, failing costs lost work, and the best interval balances them; block 3 measures it and a first-order rule, $\sqrt{2CM}$, locates it. **The fourth challenge, resource management.** The lecture lists it with the other three but the notes give it no section. Its commonest face is the straggler: workers differ in speed, so a synchronous step lasts as long as the slowest. Block 2 measures that, and backup workers (launch a few extra, take the first N) are one remedy.
:::


### Ring all-reduce

Communication overtakes compute at scale. Ring all-reduce moves ≈2(N−1)/N × model size per worker — near-constant in N, unlike a central server (N×).

:::tip

**Worked.** N=4 → 2·3/4 = 1.5× model size; N→∞ → 2×.

:::

:::note Beyond the lecture
**How the ring achieves it.** Split each worker's vector into N chunks and arrange the workers in a ring. In the reduce-scatter phase, each step every worker sends one chunk to its right-hand neighbour and adds the chunk it receives from the left; after N-1 steps worker $i$ holds one fully summed chunk. In the all-gather phase the finished chunks circulate for another N-1 steps. Every worker sends $2(N-1)$ chunks of size $K/N$, which is $2(N-1)K/N$ values, and every link is busy in every step. A central server would instead receive N full copies and send N back. Block 1 moves real chunks and counts them.
:::



## A real system that works this way

**A 16,000-GPU run that stops every few hours.** The Llama 3 technical report describes the reliability of its largest pre-training run in a section called "Reliability and Operational Challenges". It says the synchronous nature of the training makes it less fault-tolerant, so that a single GPU failure may require a restart of the entire job. Over a 54-day snapshot of pre-training on 16 thousand GPUs there were 466 job interruptions in total: 47 planned (for example firmware upgrades) and 419 unexpected. Roughly 78% of the unexpected ones were attributed to confirmed or suspected hardware problems, with GPU issues the largest single category at 58.7%. The team still reported more than 90% effective training time, and says it got there partly by cutting job start-up and checkpointing time.

Turn those counts into a rate and the lecture's advice stops being abstract. 54 days is 1,296 hours, so 419 unexpected interruptions is one every 3.09 hours, and counting the planned ones too it is one every 2.78 hours. At that rate a job that never checkpointed would almost never finish. The third code block below uses the 3.09-hour figure as the mean time between failures and asks how often to save.

**Backup workers against stragglers.** Chen, Pan, Monga, Bengio and Jozefowicz argued in "Revisiting Distributed Synchronous SGD" that synchronous training could be made competitive with asynchronous training by adding backup workers, so the system avoids asynchronous noise while mitigating the worst stragglers. The second code block simulates that idea.

## Code you can run

Five blocks. They check the lecture's ring all-reduce formula by moving real chunks between simulated workers, measure what stragglers do to a synchronous step, find the checkpoint interval that wastes least time, show how replica drift grows as synchronisation gets rarer, and finish with a real checkpoint-and-resume in PyTorch.

### 1. Ring all-reduce, moved chunk by chunk

Each worker holds a vector of 1,920 gradient values and splits it into N chunks. In the first phase every worker sends one chunk to its right-hand neighbour per step and adds what it receives; after N-1 steps each worker owns one fully summed chunk. In the second phase the finished chunks travel round the ring for another N-1 steps. The code counts every value sent and compares the result with a plain sum.

```python
import numpy as np

def ring_allreduce(grads):
    n = len(grads)
    chunks = [list(np.array_split(g.copy(), n)) for g in grads]
    sent = np.zeros(n)
    for step in range(n - 1):
        out = [(i, (i - step) % n, chunks[i][(i - step) % n].copy()) for i in range(n)]
        for i, idx, payload in out:
            chunks[(i + 1) % n][idx] = chunks[(i + 1) % n][idx] + payload
            sent[i] += payload.size
    for step in range(n - 1):
        out = [(i, (i + 1 - step) % n, chunks[i][(i + 1 - step) % n].copy()) for i in range(n)]
        for i, idx, payload in out:
            chunks[(i + 1) % n][idx] = payload
            sent[i] += payload.size
    return [np.concatenate(c) for c in chunks], sent

K = 1920
rng = np.random.default_rng(0)
print("workers | ring sent per worker (x model) | formula 2(N-1)/N | central server in or out (x model) | result exact")
for n in (2, 4, 8, 16, 64):
    grads = [rng.normal(size=K) for _ in range(n)]
    out, sent = ring_allreduce(grads)
    exact = all(np.allclose(o, np.sum(grads, axis=0)) for o in out)
    print(f"{n:7d} | {sent[0] / K:31.4f} | {2 * (n - 1) / n:16.4f} | {n:35d} | {exact}")

S = 400e6
B = 10e9
alpha = 20e-6
print("\nmodel size 400 MB, link 10 GB/s, per-message latency 20 us (named parameters, not measurements)")
print("workers | ring seconds | single reducer seconds")
for n in (2, 4, 8, 16, 64, 256):
    ring = 2 * (n - 1) * alpha + 2 * (n - 1) / n * S / B
    server = 2 * alpha + 2 * n * S / B
    print(f"{n:7d} | {ring:12.4f} | {server:22.4f}")
```

The measured traffic per worker equals $2(N-1)/N$ times the model exactly: 1.5 for four workers as in the lecture's worked example, 1.969 for 64, and the answer is the exact sum on every worker. A single reducer must take in N full copies, so its load grows with N. The second table turns that into time with named parameters (a 400 MB model, a 10 GB/s link, 20 microseconds of latency per message; none of these is a measurement of any real network). The ring pays a latency term that grows with N, $2(N-1)\alpha$, but it is tiny next to the bandwidth term: 0.0813 s at 64 workers against 5.12 s for the single reducer.

### 2. What a straggler costs, and what backup workers buy

Each worker's step time is $e^{\sigma z}$ with $z$ standard normal, so the median is 1 and $\sigma$ sets how uneven the cluster is. A synchronous step ends when the slowest needed worker finishes. The second table launches extra workers and takes the first 64 to report. The random numbers come from the same small generator the lab uses, so the lab's default shows the same 3.327.

```python
import numpy as np

def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def rnd():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rnd

def normals(rnd, count):
    out = []
    while len(out) < count:
        u1 = max(rnd(), 1e-12)
        u2 = rnd()
        r = np.sqrt(-2.0 * np.log(u1))
        out.append(r * np.cos(2 * np.pi * u2))
        out.append(r * np.sin(2 * np.pi * u2))
    return np.array(out[:count])

STEPS = 400

def step_time(n, backups, sigma):
    rnd = mulberry32(12345)
    total = 0.0
    for _ in range(STEPS):
        times = np.sort(np.exp(sigma * normals(rnd, n + backups)))
        total += times[n - 1]
    return total / STEPS

print("worker time = exp(sigma * z), median 1. mean synchronous step time (waiting for the slowest of N)")
print("workers | sigma 0.1 | sigma 0.3 | sigma 0.5")
for n in (1, 4, 16, 64, 256):
    row = " | ".join(f"{step_time(n, 0, s):9.3f}" for s in (0.1, 0.3, 0.5))
    print(f"{n:7d} | {row}")

print("\n64 workers, sigma 0.5: wait for the fastest 64 of 64 + b launched (backup workers)")
print("backups | step time | extra machines")
for b in (0, 1, 2, 4, 8, 16):
    print(f"{b:7d} | {step_time(64, b, 0.5):9.3f} | {b / 64:.1%}")
```

A single worker averages 1.184 (400 sampled steps; the mean of a lognormal sits above its median). With 64 workers at $\sigma=0.5$ the step takes 3.327, nearly three times the median worker, and at 256 workers 4.261: the cluster gets bigger and each step gets slower. At $\sigma=0.1$ the same 64 workers cost only 1.267, so how much a straggler hurts depends on how uneven the machines are. Eight backup workers (12.5% more machines) cut the 64-worker step from 3.327 to 1.819, and sixteen to 1.510.

### 3. How often to checkpoint

Failures arrive at random with a mean of 3.09 hours between them, which is the Llama 3 rate from the last section. A checkpoint costs 0.05 hours (3 minutes) and a restart costs another 0.05 hours; those two are assumptions chosen for the example. The job needs 400 hours of useful work. When a failure strikes, everything since the last checkpoint is lost. The code averages 40 simulated runs per interval and compares them with a first-order approximation.

```python
import numpy as np

hours = 54 * 24
unexpected = 419
planned = 47
mtbf = hours / unexpected
print(f"54 days = {hours} h, {unexpected} unexpected interruptions: one every {mtbf:.2f} h")
print(f"all {unexpected + planned} interruptions: one every {hours / (unexpected + planned):.2f} h")

C = 0.05
R = 0.05
WORK = 400.0

def run(interval, seed):
    rng = np.random.default_rng(seed)
    clock = 0.0
    done = 0.0
    saved = 0.0
    next_fail = rng.exponential(mtbf)
    while saved < WORK:
        segment = min(interval, WORK - saved)
        cost = segment + (C if saved + segment < WORK else 0.0)
        if clock + cost <= next_fail:
            clock += cost
            saved += segment
        else:
            clock = next_fail + R
            next_fail = clock + rng.exponential(mtbf)
    return clock

print(f"\nuseful work {WORK:.0f} h, checkpoint cost {C} h, restart cost {R} h, one failure every {mtbf:.2f} h")
print("interval h | wall-clock h | overhead | first-order model C/T + T/(2M) + R/M")
best = None
for interval in (0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0):
    wall = np.mean([run(interval, s) for s in range(40)])
    over = wall / WORK - 1
    model = C / interval + interval / (2 * mtbf) + R / mtbf
    best = (over, interval) if best is None or over < best[0] else best
    print(f"{interval:10.2f} | {wall:12.1f} | {over:8.1%} | {model:8.1%}")
print(f"best simulated interval: {best[1]} h; first-order optimum sqrt(2 C M) = {np.sqrt(2 * C * mtbf):.3f} h")
```

Checkpointing too often wastes time saving: every 0.10 hours costs 56.1% extra. Too rarely wastes time redoing lost work: every 8 hours costs 390.6%. The best interval among those tried is 0.5 hours at 22.3% overhead. A back-of-envelope rule, overhead $\approx C/T + T/(2M) + R/M$ for checkpoint cost $C$, interval $T$, mean time between failures $M$ and restart cost $R$, is minimised at $T=\sqrt{2CM}=0.556$ hours, close to the simulated best. The approximation is only good near the optimum: it says 131.6% at 8 hours where the simulation says 390.6%, because it ignores failures that strike twice in one segment.

### 4. Strong against eventual consistency

Four replicas each minimise their own quadratic loss (the stand-in for each worker's data shard), with different curvatures, and the real objective is the average of the four. Replicas are averaged only every k steps, and "never" averages once at the end.

```python
import numpy as np

n_workers, dim, steps, lr = 4, 10, 300, 0.05
rng = np.random.default_rng(3)
targets = rng.normal(size=(n_workers, dim))
curv = rng.uniform(0.2, 3.0, size=(n_workers, 1))
noise = rng.normal(scale=0.3, size=(steps, n_workers, dim))
optimum = (curv * targets).sum(axis=0) / curv.sum()

def global_loss(w):
    return float(np.mean(0.5 * curv * (w - targets) ** 2) * dim)

def run(sync_every):
    w = np.zeros((n_workers, dim))
    worst_gap = 0.0
    for t in range(1, steps + 1):
        w = w - lr * (curv * (w - targets) + noise[t - 1])
        if sync_every and t % sync_every == 0:
            w[:] = w.mean(axis=0)
        gap = max(np.linalg.norm(w[i] - w[j]) for i in range(n_workers) for j in range(i))
        worst_gap = max(worst_gap, gap)
    final = w.mean(axis=0)
    return worst_gap, global_loss(final), float(np.linalg.norm(final - optimum))

print(f"loss at the true optimum: {global_loss(optimum):.4f}")
print("sync every | worst replica gap | loss of averaged model | distance to optimum")
for every in (1, 5, 25, 100, 0):
    gap, loss, dist = run(every)
    label = "never" if every == 0 else str(every)
    print(f"{label:>10} | {gap:17.3f} | {loss:22.4f} | {dist:19.4f}")
```

Syncing every step keeps the replicas identical (gap 0.000) and lands 0.0457 from the true optimum, the gap being the injected gradient noise. Syncing every 5 steps lets replicas wander 2.547 apart; every 25 steps, 6.188; never, 6.728. The averaged model gets worse too, because the shards have different curvature and averaging does not commute with the updates: its distance to the optimum rises from 0.0457 to 0.4632. That is the price of "fast but stale".

### 5. A real checkpoint and resume

Block 5 trains a small PyTorch model for six steps, saves the model and optimiser with `torch.distributed.checkpoint` on a single CPU process, rebuilds everything from scratch, loads the checkpoint and trains the last four steps. It compares the result with a run that was never interrupted.

```python
import tempfile
import warnings

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

warnings.filterwarnings("ignore")

torch.manual_seed(0)
batches = [torch.randn(8, 4) for _ in range(10)]
init = torch.nn.Linear(4, 2).state_dict()

def build():
    model = torch.nn.Linear(4, 2)
    model.load_state_dict(init)
    return model, torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

def train(model, opt, data):
    for x in data:
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

model, opt = build()
train(model, opt, batches)
reference = torch.cat([p.detach().flatten() for p in model.parameters()])

def final(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

model, opt = build()
train(model, opt, batches[:6])
with tempfile.TemporaryDirectory() as path:
    model_sd, optim_sd = get_state_dict(model, opt)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)

    model, opt = build()
    model_sd, optim_sd = get_state_dict(model, opt)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    train(model, opt, batches[6:])
    resumed = final(model)

    model, opt = build()
    model_sd, _ = get_state_dict(model, opt)
    state = {"model": model_sd}
    dcp.load(state, checkpoint_id=path)
    model.load_state_dict(state["model"])
    train(model, opt, batches[6:])
    weights_only = final(model)

print("resume with model and optimiser state, difference from the uninterrupted run:", float((reference - resumed).abs().max()))
print("resume with the weights only (momentum buffer lost), difference:", round(float((reference - weights_only).abs().max()), 6))
```

Resuming with the model and the optimiser state reproduces the uninterrupted run exactly (difference 0.0). Restoring the weights alone, so that SGD's momentum buffer starts empty, ends 0.282329 away. A checkpoint is not just the weights: it is everything the next step reads.

### Try it yourself

The lab runs the straggler model of block 2. With its defaults (64 workers, $\sigma=0.5$, no backups) it shows the mean step time 3.327; set backup workers to 8 and it shows 1.819. Raise the spread and watch the histogram of the 400 step times stretch its tail to the right, which is where the dashed mean line goes.

<StragglerLab />

## Production snippets (not run here)

*Not run in this environment.*

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

model_sd, optim_sd = get_state_dict(model, optimizer)
future = dcp.async_save({"model": model_sd, "optim": optim_sd}, checkpoint_id="/shared/ckpt/step_1000")
train_next_steps()
future.result()
```

The `torch.distributed.checkpoint` documentation for PyTorch 2.14 describes `async_save` as staging the data to CPU before the parallel write, which is what lets training carry on while the files are written. It needs an initialised process group with several ranks, so it is not run here; block 5 shows the single-process version.

## Designing with it

**Which challenge is biting?**

| Symptom | Likely cause | First thing to try |
| --- | --- | --- |
| Adding workers makes steps slower | Communication or stragglers dominate | Measure the all-reduce time and the spread of worker step times separately |
| Step time has a long right tail | Uneven machines, noisy neighbours, data skew | Backup workers, rebalance shards, or drop to asynchronous updates |
| Run loses hours when a node dies | Checkpoints too rare, or too slow to write | Checkpoint near $\sqrt{2CM}$, write asynchronously, keep the optimiser state |
| Loss curve wobbles after scaling out | Stale parameters from relaxed consistency | Synchronise more often, or lower the learning rate for the delay |
| Network at full load, GPUs idle | Gradient traffic too large | Ring or tree collectives, gradient compression, local steps (later chapters) |

**Put numbers on it before choosing.** The three formulas in this chapter are cheap to use: ring traffic $2(N-1)/N$ times the model, expected step time from the measured spread of step times, and checkpoint interval $\sqrt{2CM}$ from the checkpoint cost and the failure rate you have actually seen. Plug in your own measurements. The parameters in the code (400 MB, 10 GB/s, 20 microseconds, 3 minutes) are examples, not recommendations.

:::note Not from the lecture
The lecture states the four challenges and the ring cost. The straggler simulation, the checkpoint-interval experiment and the Llama 3 figures are additions for this site.
:::

## Where this stands in 2026

:::info Industry view

- **Failures are routine at scale.** The Llama 3 report counts 419 unexpected interruptions in 54 days on 16 thousand GPUs, and still reports more than 90% effective training time, which it credits partly to faster start-up and checkpointing.
- **Distributed checkpointing is a library feature.** PyTorch 2.14 ships `torch.distributed.checkpoint` with `save`, `load` and `async_save`; it lets every rank write its own shard in parallel and can reshard on load when the cluster shape changes.
- **Communication hooks target the bottleneck.** PyTorch 2.14 documents DDP communication hooks including `fp16_compress_hook`, `PowerSGD_hook` and `post_localSGD_hook`; the quantisation hooks are marked experimental. The later chapters of this section measure the ideas behind them.
- **Synchronous with backups is a design option, not a default.** The 2016 argument for backup workers still frames the choice between waiting for stragglers, ignoring them and updating without them.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Name the four core challenges of distributed ML.</summary>

Consistency, fault tolerance, communication overhead, and resource management.<br /><em>Sessions 5-6 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast strong and eventual consistency.</summary>

Strong keeps all workers exactly in sync (correct, slow); eventual lets replicas drift and reconcile (fast, but uses stale parameters).<br /><em>Sessions 5-6 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> How is fault tolerance achieved during long training?</summary>

Checkpointing — periodically saving state so a failed run resumes from the last checkpoint instead of restarting.<br /><em>Sessions 5-6 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Per all-reduce, how much does each worker transfer under ring all-reduce with N=4?</summary>

≈2(N−1)/N = 2·3/4 = 1.5× the model size (→2× as N→∞).<br /><em>Sessions 5-6 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why does ring all-reduce scale better than a central parameter server?</summary>

Its per-worker traffic is independent of N (→2×), whereas a central server receives N× and becomes a bottleneck.<br /><em>Sessions 5-6 · conceptual</em>

</details>

**Added questions (not in the lecture).**

<details>
<summary><strong>Q6.</strong> A job checkpoints in 0.05 hours and fails on average every 3.09 hours. Using the first-order rule, how often should it checkpoint?</summary>

$\sqrt{2CM}=\sqrt{2\times0.05\times3.09}=\sqrt{0.309}=0.556$ hours, about 33 minutes. The simulation in block 3 found 0.5 hours best among the intervals it tried.

</details>

<details>
<summary><strong>Q7.</strong> A cluster of 64 workers with spread 0.5 takes 3.327 per synchronous step. Eight backup workers bring it to 1.819. What did that cost and why does it work?</summary>

It costs 8/64 = 12.5% more machines. The step now ends when the fastest 64 of 72 finish, so the 8 slowest are ignored and the tail of the step-time distribution no longer sets the pace.

</details>

<details>
<summary><strong>Q8.</strong> How many times the model size does each worker send in ring all-reduce with 64 workers, and how many times does a single reducer receive?</summary>

$2(63)/64=1.969$ times for the ring. The single reducer receives 64 copies.

</details>

## Further reading

- [Andrew Gibiansky, "Bringing HPC techniques to deep learning" (February 2017)](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/), a worked explanation of ring all-reduce and its $2(N-1)K/N$ per-GPU traffic.
- [Chen, Pan, Monga, Bengio and Jozefowicz, "Revisiting Distributed Synchronous SGD" (2016)](https://arxiv.org/abs/1604.00981), the backup-workers idea.
- [Grattafiori et al., "The Llama 3 Herd of Models" (2024)](https://arxiv.org/abs/2407.21783), section 3.3.4 on reliability and operational challenges.
- [PyTorch documentation: `torch.distributed.checkpoint`](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html), distributed save, load and asynchronous save.
- [PyTorch documentation: DDP communication hooks](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html), built-in gradient compression and local-SGD hooks.
- Built from the course lecture "dml-s5-6-challenges" (Lecture Library series).

- **[D2L — Computational Performance](https://d2l.ai/chapter_computational-performance/index.html)** `book`
  Zhang, Lipton, Li & Smola — Multi-GPU and parallel training explained with runnable code.
- **[Ray documentation](https://docs.ray.io/)** `docs`
  Ray — A practical framework for distributed Python and distributed ML training/serving.
- **[Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html)** `docs`
  Apache Spark — Distributed data processing and ML on Spark — the PySpark backbone.

## Check yourself

- I can name the four challenges of distributed ML and say which one the lecture calls the bottleneck.
- I can contrast strong and eventual consistency and show, with a simulation, what looser syncing does to replica drift and to the averaged model.
- I can compute the traffic of ring all-reduce, $2(N-1)/N$ times the model, and explain why it stays near 2 while a single reducer grows with N.
- I can explain why a synchronous step runs at the pace of the slowest worker and how backup workers shorten it.
- I can choose a checkpoint interval from the checkpoint cost and the failure rate, and say why the optimiser state must be saved too.
