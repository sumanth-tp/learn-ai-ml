---
id: dist-programming-models
title: "Programming Models for Distributed ML"
sidebar_label: "2 · Programming models"
sidebar_position: 2
slug: /mlops/distributed/programming-models
description: "MapReduce, Spark RDDs and data locality for distributed machine learning, with a combiner, caching and lineage measured in code, plus a sharded parameter server."
tags: [distributed-ml, mapreduce, spark, rdd, parameter-server]
---

import Infographic from '@site/src/components/Infographic';
import ParameterServerLab from '@site/src/components/viz/ParameterServerLab';

**In one line.** Write a map function and a reduce function, move the computation to the data, and let the framework handle the cluster.

## The idea in plain words

Suppose a hundred librarians must count how often every word appears in a hundred thousand shelves of books. Nobody wants to write a scheduling system for that. So you give each librarian a stack and one rule, "tally the words in your stack", and a second rule for the tallies, "add up the counts for each word". The two rules are the whole program. Everything else, who gets which stack, what happens when a librarian faints, how the tallies reach the person adding them, is the job of the building. That is the **map/reduce** idea: write two small functions and let the framework run them on a cluster.

The lecture's single idea for this chapter is **move the computation to the data**. A stack of books is heavy and the instruction "count the words" is light, so you send the instruction to wherever the books already are, rather than hauling the books to one desk. On a cluster the same logic applies to terabytes held on many disks: ship a few kilobytes of code, not the data.

The chapter covers three things in order.

- **MapReduce**: a parallel map followed by a reduce. Simple and fault tolerant, but it writes to disk between steps.
- **Spark and its RDDs**: the same map and reduce style, with datasets that stay in memory between steps and a scheduler that remembers how to rebuild them. This is far better for machine learning, which loops over the same data many times.
- **Data locality**, the rule that decides where each task runs.

An addition at the end of the chapter covers a third model that the lecture does not, the **parameter server**, because it is how many systems split not the data but the model.

<Infographic
  src="/img/dist/programming-models-mapreduce-spark.svg"
  alt="MapReduce word count over 100,000 words and 8 map tasks sends 100,000 pairs through the shuffle without a combiner and 3,979 with one; twenty iterations of gradient descent read 14.16 MB when data is re-read each time and 0.71 MB when cached; a Spark-style lineage rebuilds one lost partition with 5 computations instead of 8."
  caption="MapReduce, caching and lineage, with the numbers printed by the first three code blocks."
/>

<Infographic
  src="/img/dist/programming-models-parameter-server.svg"
  alt="Four workers push gradient slices to four servers that each own five keys of a 20-weight model and pull fresh weights back; a table shows one server shard receiving 400 MB per step at 4 workers and 1 server, 6400 MB at 64 workers, and 400 MB with 16 shards, against a ring time of about 0.02 seconds."
  caption="A parameter server shards the model by key range; more shards divide the load. Figures from the fourth code block."
/>

## How it works

### Map, reduce, RDDs

MapReduce = parallel map then reduce (fault-tolerant, disk-heavy). Spark RDDs add in-memory transforms/actions and a DAG scheduler — much better for iterative ML.

:::

:::note Beyond the lecture
**Shuffle and combiners.** Between map and reduce the framework moves every intermediate pair to the reducer that owns its key. That network step, the shuffle, is usually the expensive one, so a **combiner** pre-adds pairs inside each map task; block 1 shrinks the shuffle from 100,000 pairs to 3,979. **Lazy transformations, eager actions.** In Spark, `map` and `filter` only record a step; an action such as `reduce` or `count` makes the DAG scheduler run the recorded steps. **Lineage.** Each dataset remembers the steps that built it, so a lost partition is recomputed, not restored from a copy (block 3). **Caching.** Persisting a dataset keeps its partitions in memory across iterations, which is the difference between 14.16 MB and 0.71 MB read in block 2.
:::
tip

**Worked.** 1000 GB / 100 mappers → 10 GB each in parallel.

:::

### Data locality

Embarrassingly parallel work suits MapReduce; iterative optimisation suits Spark RDDs. Always exploit data locality — move computation to the data.

:::note Beyond the lecture
**A third model, the parameter server.** MapReduce and Spark partition the *data*. When the *model* is also too large for one machine, it is split too: server nodes each own a range of parameter keys, and workers push gradient slices to the owning server and pull fresh parameters back. The same locality rule applies in reverse: a worker only exchanges the keys its data actually touches, which for sparse models is a small share of the whole. Block 4 builds a sharded server, checks that it equals ordinary gradient descent and measures how the number of shards changes each server's load. The next chapter's lab turns the same numbers into sliders.
:::



## A real system that works this way

**MapReduce at Google.** The MapReduce paper by Jeffrey Dean and Sanjay Ghemawat (OSDI 2004) describes the model exactly as the lecture does: a user writes a map function that turns key/value pairs into intermediate key/value pairs and a reduce function that merges all intermediate values sharing a key. The runtime takes care of partitioning the input across machines, scheduling the program, recovering from machine failures and managing the communication between machines, so that programmers with no distributed-systems experience can use a large cluster. The authors report hundreds of programs and thousands of jobs running daily on thousands of machines.

**Spark and the RDD.** Zaharia and colleagues introduced the resilient distributed dataset (NSDI 2012) for two kinds of work where MapReduce is weak, iterative algorithms and interactive data mining. They claim that keeping data in memory can improve performance by an order of magnitude. Fault tolerance does not come from replicating the data. It comes from coarse-grained transformations and lineage: if a partition is lost, the system recomputes just that partition from the steps that produced it. The third code block builds a toy version.

**Spark's machine-learning library today.** The Spark 4.2.0 documentation says the DataFrame-based `spark.ml` package is the primary machine-learning API, while the RDD-based `spark.mllib` has been in maintenance mode since Spark 2.0. Its k-means uses a parallelised variant of k-means++ called k-means||.

**The parameter server.** Li and colleagues (OSDI 2014) describe a framework in which workers hold the data and server nodes hold the shared parameters as dense or sparse vectors and matrices, with asynchronous communication, flexible consistency, elasticity and fault tolerance. They report applications such as sparse logistic regression on petabytes of data with billions of examples and parameters.

## Code you can run

Four blocks, all on one machine. They run MapReduce word count from first principles, count what an iterative job reads with and without caching, rebuild a lost partition from lineage, and shard a model over simulated parameter servers.

### 1. MapReduce word count, and what a combiner saves

A hundred thousand words, drawn with Zipf-like frequencies, are split among 8 map tasks. Each map task emits `(word, 1)`. The shuffle groups pairs by word and the reduce adds them. The second run adds a **combiner**, a mini-reduce inside each map task before anything crosses the network.

```python
import numpy as np
from collections import Counter, defaultdict

rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(500)]
ranks = np.arange(1, 501)
probs = (1 / ranks) / (1 / ranks).sum()
words = rng.choice(vocab, size=100_000, p=probs)
mappers = 8
splits = np.array_split(words, mappers)

def map_fn(record):
    return [(w, 1) for w in record]

def combine(pairs):
    out = defaultdict(int)
    for k, v in pairs:
        out[k] += v
    return list(out.items())

def run(use_combiner):
    shuffled = defaultdict(list)
    shuffled_pairs = 0
    for split in splits:
        pairs = map_fn(split)
        if use_combiner:
            pairs = combine(pairs)
        shuffled_pairs += len(pairs)
        for k, v in pairs:
            shuffled[k].append(v)
    return {k: sum(v) for k, v in shuffled.items()}, shuffled_pairs

plain, plain_pairs = run(False)
combined, combined_pairs = run(True)
truth = Counter(words.tolist())
print("map tasks:", mappers, "| records per map task:", len(splits[0]))
print("result equals Counter (no combiner, combiner):", plain == dict(truth), combined == dict(truth))
print("pairs sent through the shuffle without a combiner:", plain_pairs)
print("pairs sent through the shuffle with a combiner:   ", combined_pairs, f"({combined_pairs / plain_pairs:.1%})")
print("most common words:", truth.most_common(3))

data_gb, mappers_n = 1000, 100
print(f"\nlecture example: {data_gb} GB over {mappers_n} mappers = {data_gb / mappers_n:.0f} GB per mapper, all running in parallel")
```

Both runs equal the plain `Counter` answer. Without a combiner every one of the 100,000 pairs crosses the shuffle; with one only 3,979 do (4.0%), because common words collapse to one pair per map task. The last line is the lecture's worked example: 1000 GB over 100 mappers is 10 GB per mapper, all processed at the same time.

### 2. Why loops punish disk-based MapReduce

Logistic regression by gradient descent is a loop: each iteration computes a partial gradient on every partition (map), sums them (reduce) and updates the weights. The block runs 20 iterations three ways: every partition re-read from disk each iteration, partitions cached after the first read, and everything already in memory. It counts the bytes read.

```python
import os
import tempfile

import numpy as np

rng = np.random.default_rng(1)
n, d, parts, iters, lr = 8000, 10, 8, 20, 0.5
x = rng.normal(size=(n, d))
w_true = rng.normal(size=d)
y = (x @ w_true + 0.5 * rng.normal(size=n) > 0).astype(float)
shards = list(zip(np.array_split(x, parts), np.array_split(y, parts)))
shard_bytes = sum(a.nbytes + b.nbytes for a, b in shards)

def partial_gradient(w, xs, ys):
    p = 1 / (1 + np.exp(-xs @ w))
    return xs.T @ (p - ys), len(ys)

def train(load_shard):
    w = np.zeros(d)
    for _ in range(iters):
        total, count = np.zeros(d), 0
        for i in range(parts):
            xs, ys = load_shard(i)
            g, c = partial_gradient(w, xs, ys)
            total += g
            count += c
        w -= lr * total / count
    return w

with tempfile.TemporaryDirectory() as tmp:
    for i, (xs, ys) in enumerate(shards):
        np.savez(os.path.join(tmp, f"part{i}.npz"), x=xs, y=ys)
    bytes_read = {"disk": 0}

    def from_disk(i):
        path = os.path.join(tmp, f"part{i}.npz")
        bytes_read["disk"] += os.path.getsize(path)
        z = np.load(path)
        return z["x"], z["y"]

    w_disk = train(from_disk)
    reread_bytes = bytes_read["disk"]
    cache = {}

    def from_cache(i):
        if i not in cache:
            cache[i] = from_disk(i)
        return cache[i]

    bytes_read["disk"] = 0
    w_cached = train(from_cache)
    cached_bytes = bytes_read["disk"]

w_ref = train(lambda i: shards[i])
print("same answer from disk-per-iteration, cached and in-memory:", np.allclose(w_disk, w_cached), np.allclose(w_cached, w_ref))
print(f"data on disk: {shard_bytes / 1e6:.2f} MB in {parts} partitions")
print(f"bytes read over {iters} iterations, re-read every time: {reread_bytes / 1e6:.2f} MB")
print(f"bytes read over {iters} iterations, cached after the first: {cached_bytes / 1e6:.2f} MB, a ratio of {reread_bytes / cached_bytes:.0f}")

D_gb, mappers, disk_mb_s = 1000, 100, 200
per_node_gb = D_gb / mappers
t_read = per_node_gb * 1000 / disk_mb_s
print(f"\nnamed parameters: {D_gb} GB, {mappers} nodes, {disk_mb_s} MB/s disk per node")
print(f"one pass over a node's {per_node_gb:.0f} GB takes {t_read:.0f} s")
print(f"{iters} iterations that each re-read the data: {iters * t_read:.0f} s of reading; cached after the first pass: {t_read:.0f} s")
```

All three runs give the same weights. Re-reading costs 14.16 MB over 20 iterations against 0.71 MB when cached, a ratio of exactly the number of iterations. The parameter lines scale the same effect up with named assumptions (1000 GB over 100 nodes at 200 MB/s per node, which are not measurements of any real cluster): a node's 10 GB takes 50 seconds to read, so 20 re-reads cost 1000 seconds of reading and a cached run costs 50.

### 3. Lineage: rebuild only what was lost

A toy resilient dataset remembers its parent and the function that produced it. `map` and `filter` are lazy; `reduce` is an action that makes them run. `persist` keeps a computed partition in memory. The block counts partition computations.

```python
import numpy as np

class RDD:
    calls = 0

    def __init__(self, parts=None, parent=None, fn=None):
        self.parent, self.fn, self.base = parent, fn, parts
        self.cache = {}
        self.persisted = False

    def map(self, f):
        return RDD(parent=self, fn=lambda rows: [f(r) for r in rows])

    def filter(self, f):
        return RDD(parent=self, fn=lambda rows: [r for r in rows if f(r)])

    def persist(self):
        self.persisted = True
        return self

    def partition(self, i):
        if i in self.cache:
            return self.cache[i]
        if self.parent is None:
            rows = self.base[i]
        else:
            RDD.calls += 1
            rows = self.fn(self.parent.partition(i))
        if self.persisted:
            self.cache[i] = rows
        return rows

    def num_partitions(self):
        return len(self.base) if self.parent is None else self.parent.num_partitions()

    def reduce(self, f, zero):
        acc = zero
        for i in range(self.num_partitions()):
            for r in self.partition(i):
                acc = f(acc, r)
        return acc

data = np.arange(1, 101)
base = RDD(parts=[list(map(int, p)) for p in np.array_split(data, 4)])
squares = base.map(lambda v: v * v).persist()
evens = squares.filter(lambda v: v % 2 == 0)

total = evens.reduce(lambda a, b: a + b, 0)
print("sum of even squares up to 100:", total, "| check:", int(sum(v * v for v in data if (v * v) % 2 == 0)))
print("partition computations so far:", RDD.calls, "(4 partitions x map + 4 x filter)")

RDD.calls = 0
again = evens.reduce(lambda a, b: a + b, 0)
print("second action, squares cached: computations", RDD.calls, "(only the 4 filters)")

del squares.cache[2]
RDD.calls = 0
healed = evens.reduce(lambda a, b: a + b, 0)
print("one cached partition lost, same answer:", healed == total, "| recomputed", RDD.calls, "computations (4 filters + 1 map rebuilt from lineage)")
```

The first action computes 8 partitions (4 maps and 4 filters). The second, with the map output cached, needs only the 4 filters. Throw away one cached partition and the same answer returns after 5 computations: four filters and one map rebuilt from lineage. Nothing was copied or replicated; the recipe was enough.

### 4. A sharded parameter server

Twenty weights are split by key range over 4 servers, five keys each. Four workers each hold a quarter of the data. Every step each worker pulls the weights from the servers, computes its gradient and pushes each slice to the server that owns it. The code checks the result against ordinary gradient descent and tallies traffic.

```python
import numpy as np

rng = np.random.default_rng(2)
n, d, workers, servers, steps, lr = 4000, 20, 4, 4, 30, 0.5
x = rng.normal(size=(n, d))
y = (x @ rng.normal(size=d) + 0.5 * rng.normal(size=n) > 0).astype(float)
shards = list(zip(np.array_split(x, workers), np.array_split(y, workers)))
key_ranges = np.array_split(np.arange(d), servers)

class Server:
    def __init__(self, keys):
        self.keys = keys
        self.w = np.zeros(len(keys))
        self.received = 0
        self.sent = 0

    def push(self, grad_slice):
        self.received += grad_slice.size
        self.pending = getattr(self, "pending", 0) + grad_slice

    def apply(self, n_workers):
        self.w -= lr * self.pending / n_workers
        self.pending = 0

    def pull(self):
        self.sent += self.w.size
        return self.w.copy()

shards_servers = [Server(k) for k in key_ranges]
for _ in range(steps):
    for xs, ys in shards:
        w = np.concatenate([s.pull() for s in shards_servers])
        p = 1 / (1 + np.exp(-xs @ w))
        g = xs.T @ (p - ys) / len(ys)
        for s in shards_servers:
            s.push(g[s.keys])
    for s in shards_servers:
        s.apply(workers)
w_ps = np.concatenate([s.w for s in shards_servers])

w = np.zeros(d)
for _ in range(steps):
    p = 1 / (1 + np.exp(-x @ w))
    w -= lr * x.T @ (p - y) / n
print("sharded parameter server equals single-machine gradient descent, max difference:", float(np.abs(w_ps - w).max()))
print(f"values received per server over {steps} steps:", [s.received for s in shards_servers], f"= workers x {d // servers} keys x {steps}")
print("values a single server would receive:", workers * d * steps)

S, B = 100, 10_000
print("\nmodel 100 MB, link 10 GB/s (named parameters). server shard load per step (MB) and parameter-server time (s); ring time (s)")
print("workers | servers | shard receives MB | PS time s | ring time s")
for n_w in (4, 16, 64):
    for p in (1, 4, 16):
        load = n_w * S / p
        ps_t = 2 * max(S, load) / B
        ring_t = 2 * (n_w - 1) / n_w * S / B
        print(f"{n_w:7d} | {p:7d} | {load:17.0f} | {ps_t:9.3f} | {ring_t:11.5f}")
```

The sharded system matches single-machine gradient descent to 1.1e-16. Each server receives 600 values over 30 steps (4 workers, 5 keys, 30 steps) while one server holding all 20 keys would receive 2,400. The table uses named parameters, a 100 MB model and a 10 GB/s link. A single server takes in $N$ copies per step, 400 MB at 4 workers and 6,400 MB at 64, so its link sets the step time (0.080 s and 1.280 s). Shards divide that load by $P$: 64 workers on 16 servers see 400 MB per shard and 0.080 s. The ring needs between 0.015 and 0.020 seconds in every row. Servers approach that only when there are enough shards that a worker's own link, 2 x 100 MB per step, becomes the limit; that floor is 0.020 s, reached at 4 workers on 4 servers and at 16 workers on 16.

### Try it yourself

The lab computes the table of block 4 for any cluster. Its defaults are 4 workers, 1 server, a 100 MB model and a 10 GB/s link: the single server receives 400 MB per step, a ring worker sends 150 MB (1.5 times the model, the lecture's number from the previous chapter), and the times are 0.080 s for the parameter server and 0.015 s for the ring. Raise the workers and then the shards to see where the server stops being the bottleneck.

<ParameterServerLab />

## Production snippets (not run here)

*Not run in this environment.*

```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)
predictions = model.transform(dataset)
silhouette = ClusteringEvaluator().evaluate(predictions)
for center in model.clusterCenters():
    print(center)
```

This is the k-means example from the Spark 4.2.0 clustering documentation, shortened. The `spark` session and the sample file come from a Spark installation, which this environment does not have. Because Spark keeps the data it loads in memory across the iterations of k-means, it is the lecture's RDD argument in a single call.

## Designing with it

**Choosing a programming model**

| Your workload | Reach for | Why |
| --- | --- | --- |
| One pass over huge data: counting, filtering, feature extraction | MapReduce style (map, shuffle, reduce) | Embarrassingly parallel, fault tolerance is free, disk cost is paid once |
| Many passes over the same data: gradient descent, k-means, ALS | Spark style, with the working set cached | Disk reads fall from one per iteration to one in total (block 2) |
| A model too big or too sparse for one machine's memory | Parameter server, sharded by key | Servers hold slices of the model; workers hold slices of the data |
| Dense deep-learning gradients on GPUs | Collective all-reduce (previous chapter) | No server to overload, traffic near 2 times the model per worker |

**Habits that pay off in any of them.**

- **Shuffle less.** A combiner (block 1) or a pre-aggregation step is the cheapest optimisation there is. Count the pairs that cross the network before counting anything else.
- **Cache deliberately.** Cache what you loop over, not what you read once. Memory is finite, and a cache that does not fit silently falls back to recomputation.
- **Keep tasks near their data.** Moving 10 GB of code and settings is easy; moving 10 GB per mapper is the cost the lecture wants you to avoid.
- **Pick the shard key to balance load.** In a parameter server, a hot key range makes one server the bottleneck again.

:::note Not from the lecture
The combiner experiment, the lineage toy, the cost tables and the whole parameter-server section are additions for this site. The lecture covers MapReduce, RDDs and locality.
:::

## Where this stands in 2026

:::info Industry view

- **Spark ML is DataFrame-first.** The Spark 4.2.0 guide names `spark.ml` the primary machine-learning API and keeps the RDD-based `spark.mllib` in maintenance mode, with bug fixes only, since Spark 2.0.
- **Collectives dominate dense deep learning.** PyTorch 2.14 documents the default DDP communication hook as an all-reduce that takes the mean of the gradients; the parameter-server design remains part of the literature and of course texts (the D2L computational-performance chapter has a section on it).
- **Ray Train covers the mainstream trainers.** Its documentation lists PyTorch, PyTorch Lightning, Hugging Face Transformers, JAX, TensorFlow, Keras, Horovod, XGBoost and LightGBM as supported frameworks.
- **Lineage and caching are the durable ideas.** Whatever the engine, recomputing lost pieces from a recipe and keeping the hot working set in memory are the two things the RDD paper contributed.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Describe the MapReduce programming model.</summary>

A map phase transforms each record in parallel, then a reduce phase aggregates the results — simple and fault-tolerant but disk-heavy.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Why are Spark RDDs better than MapReduce for ML?</summary>

RDDs keep intermediates in memory and use a DAG scheduler, so iterative algorithms (gradient descent, k-means) avoid repeated disk I/O.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> 1000 GB processed by 100 mappers — data per mapper?</summary>

1000/100 = 10 GB each, in parallel.<br /><em>Session 7 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is data locality and why does it matter?</summary>

Moving computation to the data (not data to computation) — it minimises expensive network transfer.<br /><em>Session 7 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Which algorithm types suit MapReduce vs Spark?</summary>

Embarrassingly parallel (feature extraction) → MapReduce; iterative optimisation → Spark RDDs.<br /><em>Session 7 · conceptual</em>

</details>

**Added questions (not in the lecture).**

<details>
<summary><strong>Q6.</strong> A word count over 100,000 words with 8 map tasks sends 100,000 pairs through the shuffle. A combiner brings it to 3,979. What share is that and why so small?</summary>

3,979 / 100,000 = 4.0%. Each map task sends one pair per distinct word instead of one per occurrence, and in natural text a few words account for most occurrences.

</details>

<details>
<summary><strong>Q7.</strong> 64 workers train through one parameter server with a 100 MB model on a 10 GB/s link. How long is the communication per step, and how many shards bring it to 0.080 s?</summary>

The server receives 64 x 100 MB = 6,400 MB and the time is $2\times6400/10{,}000=1.280$ s. With 16 shards each receives 400 MB, giving $2\times400/10{,}000=0.080$ s.

</details>

<details>
<summary><strong>Q8.</strong> In the lineage example, one cached partition is lost. How many partition computations does the next action need and why not 8?</summary>

5: the four filters plus one map to rebuild the lost partition from its parent. The other three map outputs are still cached, so the first action's other 3 map computations are not repeated.

</details>

## Further reading

- [Dean and Ghemawat, "MapReduce: Simplified Data Processing on Large Clusters" (OSDI 2004)](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/), the original model and runtime.
- [Zaharia et al., "Resilient Distributed Datasets" (NSDI 2012)](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia), RDDs, lineage and in-memory iteration.
- [Li et al., "Scaling Distributed Machine Learning with the Parameter Server" (OSDI 2014)](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu), sharded parameters, asynchronous communication, flexible consistency.
- [Apache Spark 4.2.0: MLlib main guide](https://spark.apache.org/docs/latest/ml-guide.html) and [clustering](https://spark.apache.org/docs/latest/ml-clustering.html), the current ML API and the k-means example.
- [Ray Train documentation](https://docs.ray.io/en/latest/train/train.html), distributed training on Ray.
- [PyTorch documentation: DDP communication hooks](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html), the default all-reduce hook.
- Built from the course lecture "dml-s7-programming-models" (Lecture Library series).

- **[D2L — Computational Performance](https://d2l.ai/chapter_computational-performance/index.html)** `book`
  Zhang, Lipton, Li & Smola — Multi-GPU and parallel training explained with runnable code.
- **[Ray documentation](https://docs.ray.io/)** `docs`
  Ray — A practical framework for distributed Python and distributed ML training/serving.
- **[Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html)** `docs`
  Apache Spark — Distributed data processing and ML on Spark — the PySpark backbone.

## Check yourself

- I can describe MapReduce as a map phase and a reduce phase and say what the framework does for me.
- I can explain why iterative ML is slow on a disk-based MapReduce and fast on cached RDDs, and show it by counting bytes read.
- I can say how a lost RDD partition is rebuilt from lineage and why that needs no replication.
- I can compute the data per mapper for a given dataset and mapper count, and say why locality matters.
- I can explain how a parameter server shards the model and how the number of shards changes the load on each server.
