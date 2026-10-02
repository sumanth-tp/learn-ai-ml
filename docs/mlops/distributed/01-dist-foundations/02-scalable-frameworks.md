---
id: dist-frameworks
title: "Scalable Frameworks for ML"
sidebar_label: "Scalable frameworks"
sidebar_position: 2
slug: /mlops/distributed/scalable-frameworks
description: "Amdahl's law and why the serial part caps speed-up, how MapReduce and Spark's in-memory RDDs suit iterative ML, and how ring all-reduce sums gradients in 2(N-1) steps."
tags: [amdahl-law, mapreduce, hadoop, spark, rdd, ring-all-reduce, torch-distributed, gloo]
---

import Infographic from '@site/src/components/Infographic';
import AllReduceLab from '@site/src/components/viz/AllReduceLab';

**In one line.** When one machine is not enough, spread the work, but the part that cannot be spread sets the ceiling, and the cost of combining results is what the frameworks and the all-reduce are built to cut.

Built from the course lecture "dml-s2-frameworks" (Lecture Library series), extended with runnable measurements.

## The idea in plain words

Suppose a job takes ten hours and nine of them can be done by any number of helpers at once, while one hour must be done by a single person, say reading the brief and signing off at the end. With ten helpers the nine hours shrink to under one, so the job takes about 1.9 hours, a gain of roughly five times, not ten. With a million helpers the nine hours become almost nothing, but the one serial hour is still there: the job takes just over an hour and you never beat ten times. This is **Amdahl's law**, and it explains why throwing machines at a problem has a ceiling set by the part that cannot be split.

Two families of frameworks grew up to make the splittable part easy on clusters of ordinary machines. **MapReduce** processes data in two simple steps that a runtime can spread and recover automatically. **Spark** keeps the dataset in memory between steps, which matters for machine learning because training makes many passes over the same data.

Training a deep network adds a third need that neither framework was built for: after every step, thousands of numbers (the gradients) must be combined across all workers. The **ring all-reduce** is the algorithm that does this without any single machine becoming a bottleneck. The second half of this chapter builds it from scratch and then runs the real thing between two processes.

<Infographic src="/img/dist/scalable-frameworks-amdahl-mapreduce-spark.svg" alt="Amdahl speed-up table with ceilings 2, 10, 20 and 100, a MapReduce map-shuffle-reduce flow with 23 versus 14 shuffled records, and Spark figures of 17.60 MB read from disk versus 1.76 MB cached" caption="The serial fraction caps speed-up, and what each framework does about cluster costs. Figures come from blocks 1 to 3 below." />

<Infographic src="/img/dist/scalable-frameworks-ring-all-reduce.svg" alt="Four workers each hold four chunks; after three reduce-scatter and three all-gather steps every worker holds 10, 50, 90 and 130; a table shows steps 2(N-1) and traffic 2(N-1)/N; a two-process gloo run sums to 3, 6, 9" caption="Ring all-reduce, with the numbers printed by blocks 4 and 5 below." />

## How it works

When one machine is not enough, spread the work, but the serial part sets the ceiling.

### Hadoop and Spark

Hadoop MapReduce is fault-tolerant batch processing on commodity nodes. Spark is fast in-memory RDDs, suited to iterative ML. Both handle heterogeneous data and hardware.

### Speed-up limits

S(n) = 1 / ((1 - p) + p / n), where p is the parallel fraction and n the number of workers. The serial part, 1 - p, caps the maximum speed-up at 1 / (1 - p).

:::tip

**Worked.** p = 0.9 and n = 10 give S = 1 / (0.1 + 0.09) = 5.26 times, with a ceiling of 1 / (1 - 0.9) = 10 times.

:::

### What the lecture leaves implicit

:::note Beyond the lecture
**Amdahl's law has a second reading.** Rearranged, it tells you how many workers you need for a target. To reach 5 times at p = 0.9 you need 9 workers, and the 10 times ceiling is never reached by finite hardware: even a million workers give 10.00 (block 1). Raising p, by shrinking the serial part, beats adding workers once you are near the ceiling. Real jobs also pay communication per added worker, which this idealised formula leaves out, so the formula is an upper bound.

**MapReduce in three words.** *Map* turns each input record into key and value pairs, on whichever node holds the data. *Shuffle* moves all pairs with the same key to one place. *Reduce* combines the values for each key. A **combiner** is a local reduce run before the shuffle so fewer records cross the network; block 2 shows 23 records shrinking to 14 for a word count. Because every step is a pure function of its input, a failed task can simply be run again, which is how it tolerates failures on commodity machines.

**Why Spark helps iterative ML.** Gradient descent reads the whole dataset once per iteration. If every iteration is a separate job that rereads the data from disk, ten iterations read it ten times. Spark's Resilient Distributed Dataset (RDD) can be cached in memory, so later iterations skip the read. Fault tolerance does not need a replica: each RDD remembers the chain of transformations that produced it (its **lineage**), so a lost partition is rebuilt by replaying that chain for that partition only. Block 3 shows both.

**Ring all-reduce.** The gradient is cut into N chunks, one per worker. In the *reduce-scatter* phase each worker passes one chunk to its right-hand neighbour, which adds it to its own copy; after N - 1 steps each worker holds the complete sum of exactly one chunk. In the *all-gather* phase the finished chunks travel round the ring for another N - 1 steps, so every worker ends with every sum. Every link is busy at every step, which is why no node is a bottleneck.
:::

## A real system that works this way

**MapReduce at Google.** Dean and Ghemawat's OSDI 2004 paper describes programs written in the map and reduce style being automatically parallelised on a large cluster of commodity machines, with the runtime handling partitioning, scheduling, machine failures and communication. It reports hundreds of programs and upwards of one thousand jobs run on Google's clusters every day, at the time of writing.

**Spark's RDDs.** Zaharia and colleagues (NSDI 2012) describe RDDs as a restricted form of shared memory based on coarse-grained transformations rather than fine-grained updates, which is what makes lineage-based recovery possible. They state that keeping data in memory can improve performance by an order of magnitude and name iterative algorithms and interactive data mining as the target workloads.

**Ring all-reduce in practice.** The Horovod paper describes the algorithm exactly as above: each of N nodes communicates with two peers 2 x (N - 1) times, the first N - 1 iterations adding received values and the second N - 1 replacing them, and it notes that Patarasuk and Yuan suggest the algorithm is bandwidth-optimal. In Horovod's own benchmarks on Inception V3 and ResNet-101 it exceeded 90 percent scaling efficiency with RDMA-capable networking, and VGG-16, with its large number of parameters, was communication-bound.

## Code you can run

Five blocks. The first four use numpy and the standard library; the fifth runs two real processes with PyTorch's Gloo backend on the CPU (Python 3.14, numpy 2.5.3, torch 2.14.1).

### 1. Amdahl's law

```python
def speedup(p, n):
    return 1.0 / ((1.0 - p) + p / n)


print("p = 0.9, n = 10:", round(speedup(0.9, 10), 2), "x   ceiling", round(1 / (1 - 0.9), 1), "x")
print()
print("workers  p=0.50  p=0.90  p=0.95  p=0.99")
for n in (1, 2, 4, 10, 100, 1000, 10**6):
    row = "  ".join(f"{speedup(p, n):6.2f}" for p in (0.5, 0.9, 0.95, 0.99))
    print(f"{n:>7}  {row}")
print("ceiling  " + "  ".join(f"{1 / (1 - p):6.1f}" for p in (0.5, 0.9, 0.95, 0.99)))
print()
print("workers to reach 5x when p = 0.9:", round(0.9 / (1 / 5 - 0.1), 2))
```

The lecture's worked case is reproduced: p = 0.9 and n = 10 give 5.26 times against a ceiling of 10.0 times. With p = 0.99, a thousand workers give 90.99 times, not 1000. Nine workers are enough for 5 times at p = 0.9.

### 2. MapReduce, word count

```python
from collections import Counter, defaultdict

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "a dog and a cat sat",
]
partitions = [documents[:2], documents[2:]]


def map_phase(partition, combine):
    pairs = [(word, 1) for line in partition for word in line.split()]
    if not combine:
        return pairs
    local = Counter(word for word, _ in pairs)
    return list(local.items())


def shuffle(mapped, reducers):
    buckets = [defaultdict(list) for _ in range(reducers)]
    for pairs in mapped:
        for key, value in pairs:
            buckets[hash(key) % reducers][key].append(value)
    return buckets


def reduce_phase(buckets):
    out = {}
    for bucket in buckets:
        for key, values in bucket.items():
            out[key] = sum(values)
    return out


for combine in (False, True):
    mapped = [map_phase(p, combine) for p in partitions]
    records = sum(len(m) for m in mapped)
    result = reduce_phase(shuffle(mapped, reducers=2))
    assert result == Counter(w for d in documents for w in d.split())
    print(f"combiner={combine}: records shuffled {records}, distinct words {len(result)}, the={result['the']}, cat={result['cat']}")
```

The result equals the plain `Counter` (the assertion passes) and the counts are identical either way. Without a combiner 23 records cross the shuffle; with a local combine only 14 do. The saving is exactly the repeated words within each partition.

### 3. Iterative jobs, caching and lineage

```python
import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)
n, d = 20000, 10
X = rng.normal(size=(n, d))
w_true = np.arange(1, d + 1, dtype=float)
y = X @ w_true + 0.1 * rng.normal(size=n)

folder = tempfile.mkdtemp()
path = os.path.join(folder, "data.npz")
np.savez(path, X=X, y=y)
size = os.path.getsize(path)
iterations = 10
lr = 0.1


def descend(load):
    w = np.zeros(d)
    reads = 0
    for _ in range(iterations):
        Xi, yi, loaded = load()
        reads += loaded
        w = w - lr * Xi.T @ (Xi @ w - yi) / len(yi)
    return w, reads


def from_disk():
    data = np.load(path)
    return data["X"], data["y"], size


cache = {}


def from_memory():
    if "data" not in cache:
        data = np.load(path)
        cache["data"] = (data["X"], data["y"])
        return cache["data"][0], cache["data"][1], size
    return cache["data"][0], cache["data"][1], 0


w_disk, bytes_disk = descend(from_disk)
w_mem, bytes_mem = descend(from_memory)
print(f"file size: {size / 1e6:.2f} MB, {iterations} iterations")
print(f"read from disk every iteration: {bytes_disk / 1e6:.2f} MB read")
print(f"cached in memory after first pass: {bytes_mem / 1e6:.2f} MB read")
print("same answer:", bool(np.allclose(w_disk, w_mem)))


class Rdd:
    def __init__(self, compute, parent=None):
        self.compute = compute
        self.parent = parent
        self.store = {}
        self.recomputed = 0

    def partition(self, i):
        if i not in self.store:
            self.recomputed += 1
            source = self.parent.partition(i) if self.parent else None
            self.store[i] = self.compute(i, source)
        return self.store[i]

    def map(self, fn):
        return Rdd(lambda i, src: [fn(v) for v in src], self)


base = Rdd(lambda i, _: list(range(i * 4, i * 4 + 4)))
squared = base.map(lambda v: v * v)
plus_one = squared.map(lambda v: v + 1)
before = [plus_one.partition(i) for i in range(3)]
lost = plus_one.store.pop(1)
squared.store.pop(1)
base.store.pop(1)
total_before = base.recomputed + squared.recomputed + plus_one.recomputed
after = plus_one.partition(1)
total_after = base.recomputed + squared.recomputed + plus_one.recomputed
print("partition 1 before loss:", before[1])
print("partition 1 rebuilt from lineage:", after, "identical:", after == lost)
print("computations:", total_before, "to build 3 partitions at 3 levels,", total_after - total_before, "to rebuild partition 1 (one per level, partitions 0 and 2 untouched)")
```

Ten gradient-descent passes over a 1.76 MB file read 17.60 MB when each pass reloads from disk, and 1.76 MB when the data is cached after the first pass, with the same result. The lineage class rebuilds a lost partition from its parents: partition 1 comes back as `[17, 26, 37, 50]`, identical, at a cost of three computations (one per level) while partitions 0 and 2 are never touched.

### 4. Ring all-reduce from scratch

```python
import numpy as np


def ring_all_reduce(buffers):
    n = len(buffers)
    buf = [b.copy() for b in buffers]
    steps = 0
    sent = 0
    for s in range(n - 1):
        outgoing = [(w, (w - s) % n, buf[w][(w - s) % n].copy()) for w in range(n)]
        for w, c, chunk in outgoing:
            buf[(w + 1) % n][c] += chunk
            sent += chunk.size
        steps += 1
    for s in range(n - 1):
        outgoing = [(w, (w + 1 - s) % n, buf[w][(w + 1 - s) % n].copy()) for w in range(n)]
        for w, c, chunk in outgoing:
            buf[(w + 1) % n][c] = chunk
            sent += chunk.size
        steps += 1
    return buf, steps, sent


n = 4
inputs = [np.array([[(w + 1) + 10 * c] for c in range(n)], dtype=float) for w in range(n)]
out, steps, sent = ring_all_reduce(inputs)
print("worker inputs, one number per chunk:", [b.ravel().tolist() for b in inputs])
print("every worker ends with:", out[0].ravel().tolist())
print("all workers identical:", all(np.array_equal(out[0], o) for o in out))
print("steps:", steps, "= 2(N-1) =", 2 * (n - 1))

rng = np.random.default_rng(0)
print()
print("workers  steps  elements sent per worker / gradient size  2(N-1)/N  correct")
for n in (2, 4, 8, 16, 64):
    size = 1024
    data = [rng.normal(size=(n, size // n)) for _ in range(n)]
    out, steps, sent = ring_all_reduce(data)
    ok = all(np.allclose(o, sum(data)) for o in out)
    print(f"{n:>7}  {steps:>5}  {sent / n / size:>41.4f}  {2 * (n - 1) / n:8.4f}  {ok}")
```

The four workers' inputs are `[1, 11, 21, 31]` to `[4, 14, 24, 34]` and every worker finishes with `[10, 50, 90, 130]` after 6 steps, which is 2(N - 1). The table is the bandwidth argument: the number of elements each worker sends, divided by the gradient size, matches 2(N - 1)/N exactly and tends to 2. The step count grows with N, though (126 steps at 64 workers), and every step pays a message latency, which is why a ring suits large gradients rather than tiny ones.

### 5. The real thing: all-reduce between two processes

`torch.distributed` with the Gloo backend on CPU. Two processes are spawned; each holds a tensor and calls `all_reduce`, then `all_gather`.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    local = torch.tensor([1.0, 2.0, 3.0]) * (rank + 1)
    before = local.tolist()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    summed = local.tolist()
    local /= world_size
    averaged = local.tolist()
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([float(rank)]))
    print(f"rank {rank} of {dist.get_world_size()} backend {dist.get_backend()}: before {before} sum {summed} mean {averaged} gathered ranks {[g.item() for g in gathered]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, free_port()), nprocs=2, join=True)
```

Both ranks print the same sum `[3.0, 6.0, 9.0]` (rank 0 held `[1, 2, 3]`, rank 1 held `[2, 4, 6]`), the mean after dividing by 2 is `[1.5, 3.0, 4.5]`, and `all_gather` returns the ranks `[0.0, 1.0]`. A warning about resolving the hostname may appear on stderr; Gloo falls back to the loopback address, which is what a single machine needs.

### Try it yourself

The lab below is block 4 as an animation. Its default (four workers, step 6) shows the finished state: every worker holds 10, 50, 90 and 130, and it took 2(N - 1) = 6 steps. Drag the step slider back to watch the reduce-scatter and all-gather phases, and change the worker count to see the step count and traffic follow 2(N - 1) and 2(N - 1)/N.

<AllReduceLab />

## Production snippets (not run here)

The same MapReduce idea in Spark's DataFrame API. It needs a Spark installation, which this environment does not have.

Not run in this environment

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

spark = SparkSession.builder.appName("wordcount").getOrCreate()
lines = spark.read.text("hdfs:///data/corpus/*.txt")
words = lines.select(explode(split("value", " ")).alias("word"))
words.groupBy("word").count().orderBy("count", ascending=False).show(10)
```

## Designing with it

| Question | Guidance |
| --- | --- |
| What is my parallel fraction? | Profile one iteration. Data loading, the optimiser step on one rank and checkpointing are often serial. Amdahl's ceiling comes from them. |
| Is the job one pass or many? | One pass over large data: MapReduce-style batch engines are enough. Many passes: cache in memory or keep the data resident on the workers. |
| Is the thing being combined small or large? | Small, often: latency dominates and a ring pays 2(N - 1) latencies. Large and dense, such as gradients: the ring's near-constant per-worker traffic wins. |
| Which framework? | If the data and features already live in Spark, use Spark. If the model is a deep network, use a deep-learning framework's data-parallel training, and a launcher such as Ray Train when you need to schedule it on a cluster. |

**A habit worth keeping.** Write the Amdahl bound for your job before you scale it. If the ceiling is 5 times, a cluster of 64 is a waste, and the money is better spent shrinking the serial part.

## Where this stands in 2026

:::info Industry view
- **PyTorch exposes the primitives you just ran.** The `torch.distributed` documentation (PyTorch 2.14.0) lists the Gloo, NCCL, MPI and XCCL backends, recommends NCCL for CUDA GPUs and Gloo for CPU training, and supports `env://`, `tcp://` and shared-file initialisation. `DistributedDataParallel` uses the same all-reduce to average gradients, bucketed and overlapped with the backward pass.
- **Ray is a general scheduler for this work.** The Ray Train overview calls it a scalable library for distributed training and fine-tuning that moves training code from one machine to a cluster and supports PyTorch, Lightning, Hugging Face Transformers and Accelerate, DeepSpeed, TensorFlow, Keras, Horovod, XGBoost and LightGBM.
- **Spark is still the data-side engine.** The Spark 4.2.0 MLlib guide lists classification, regression, clustering and collaborative filtering, featurisation, pipelines and persistence, with the DataFrame-based API primary and the RDD-based API in maintenance mode with bug fixes only.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast Hadoop MapReduce and Apache Spark for ML.</summary>

Hadoop MapReduce writes to disk between steps (fault-tolerant batch); Spark keeps data in memory via RDDs, making iterative ML with many passes far faster.

</details>

<details>
<summary><strong>Q2.</strong> What is an RDD?</summary>

A Resilient Distributed Dataset: Spark's immutable, partitioned, in-memory collection, recovered after failure by replaying its lineage.

</details>

<details>
<summary><strong>Q3.</strong> State Amdahl's law and compute the speed-up for p = 0.9, n = 10.</summary>

S(n) = 1 / ((1 - p) + p / n) = 1 / (0.1 + 0.09) = 5.26 times.

</details>

<details>
<summary><strong>Q4.</strong> What is the maximum speed-up for p = 0.9, even with infinite workers?</summary>

1 / (1 - p) = 1 / 0.1 = 10 times, set by the serial fraction.

</details>

<details>
<summary><strong>Q5.</strong> What does "data and computation heterogeneity" mean?</summary>

Varied data sources and formats, and uneven hardware across the cluster, which frameworks and algorithms must accommodate.

</details>

<details>
<summary><strong>Q6.</strong> How many workers does it take to reach a 5 times speed-up when p = 0.9? Show the algebra.</summary>

Set 1 / (0.1 + 0.9 / n) = 5, so 0.1 + 0.9 / n = 0.2, so n = 9. (Ten workers give 5.26, as in the lecture.)

</details>

<details>
<summary><strong>Q7.</strong> Eight workers run a ring all-reduce. How many steps does it take, and how much does each worker send as a multiple of the gradient size?</summary>

2(N - 1) = 14 steps, and each worker sends 2(N - 1)/N = 1.75 times the gradient size, as block 4 prints.

</details>

<details>
<summary><strong>Q8.</strong> Ten gradient-descent iterations over a 1.76 MB dataset read 17.60 MB with a disk-based job chain and 1.76 MB with a cache. What does lineage add?</summary>

The cache removes repeated reads; lineage removes the need to replicate the cache. If a node is lost, its partition is recomputed from its parents by replaying the recorded transformations, touching only that partition.

</details>

## Further reading

- [Dean and Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/). Opened 2 October 2026.
- [Zaharia et al., "Resilient Distributed Datasets" (NSDI 2012)](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia). Opened 2 October 2026.
- [Sergeev and Del Balso, "Horovod" (2018)](https://arxiv.org/abs/1802.05799), the ring-allreduce description and the scaling figures quoted above. Opened 2 October 2026.
- [Amdahl's law (reference article)](https://en.wikipedia.org/wiki/Amdahl%27s_law), the formula, the limit and the 1967 AFIPS citation. Opened 2 October 2026.
- [PyTorch 2.14 torch.distributed](https://docs.pytorch.org/docs/2.14/distributed.html) for backends and initialisation methods.
- [Ray Train](https://docs.ray.io/en/latest/train/train.html) and the [Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html).
- On the site: [distributed processing and skew](/docs/mlops/data/distributed-processing-skew) for the data-engineering side of the same ideas.

## Check yourself

- I can compute Amdahl speed-up and ceiling, and work out how many workers a target needs.
- I can explain map, shuffle and reduce, and what a combiner saves.
- I can explain why caching in memory suits iterative ML and what lineage recovers.
- I can run a ring all-reduce by hand for four workers and state its step count and per-worker traffic.
- I can run `all_reduce` between two processes with the Gloo backend and read the result.
- I can say why a ring suits large gradients but not tiny messages.
