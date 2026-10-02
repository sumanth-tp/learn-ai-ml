---
id: dist-core-algorithms
title: "Core Distributed Algorithms"
sidebar_label: "3 · Core algorithms"
sidebar_position: 3
slug: /mlops/distributed/core-distributed-algorithms
description: "Distributing k-means with sufficient statistics, pruning association-rule mining locally, and a halo-strip DBSCAN, each checked against a single-machine result."
tags: [distributed-ml, kmeans, sufficient-statistics, association-rules, dbscan]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Distribute a classic algorithm by sharing small sufficient statistics instead of raw data, and the answer stays exactly the same as on one machine.

## The idea in plain words

Ten shops want to know their combined average basket value. Nobody should mail all their till receipts to head office. Each shop adds up its own takings and counts its own baskets, and sends two numbers. Head office adds the ten totals, adds the ten counts, divides, and has the exact answer, the same one it would have got from every receipt on one desk. The two numbers are a **sufficient statistic** for the average: they hold everything the calculation needs, in a size that does not depend on how many receipts there were.

That is the lecture's idea for the chapter, **send summaries, not data**, and it is the reason a surprising number of classic algorithms distribute cleanly. The lecture takes two.

- **k-means.** Each worker assigns its own points to the nearest centroid, then sends, for every cluster, the sum of its points and how many there were. The reduce step adds them up and divides to get the new centroids, which are broadcast back. The message size is the number of clusters times the number of dimensions, and has nothing to do with the number of points.
- **Fast Distributed Mining (FDM) of association rules.** Each site mines its own transactions for itemsets that are frequent locally, and only candidates that survive that local test are checked globally, once per level.

The lecture's title also names DBSCAN. Its notes say nothing more about it, so a short addition below shows how density-based clustering can be distributed with the same spirit: keep the heavy work local and exchange only what crosses a boundary.

<Infographic
  src="/img/dist/core-algorithms-distributed-kmeans.svg"
  alt="Four workers send per-cluster sums and counts, 1,010 values each for k = 10 and d = 100, to a reduce step that produces the new centroids; the result differs from scikit-learn by 1.1e-14, whereas averaging the workers local means is off by 9.93; summary traffic stays at 4,040 values while the raw data grows from 2 million to 2 billion values."
  caption="Distributed k-means exchanges only sufficient statistics. Numbers from the first code block."
/>

<Infographic
  src="/img/dist/core-algorithms-fdm-dbscan.svg"
  alt="A table of an FDM-style run on four sites at 6 percent support, with candidates, locally large counts, globally large itemsets and the values sent against counting everything, and a panel showing DBSCAN split into four strips with an eps halo whose core points and clusters match scikit-learn."
  caption="Local pruning for association rules and halo strips for DBSCAN, from the second and third code blocks."
/>

## How it works

### Partial sums → centroids

Partition data; each worker computes partial per-cluster sums; reduce to new centroids; broadcast back. Only summaries move.

:::note Beyond the lecture
**Why sums and counts, not means.** A mean is not a sufficient statistic on its own: two workers' means cannot be combined without knowing how many points each represents. A sum and a count can be added across workers in any order, and the division happens once at the end. Block 1 shows the mistake: averaging local means is off by 9.93 where sums and counts are exact to 1e-14. The same idea covers any statistic that is a sum over rows: totals, variances (with a sum of squares), and the $X^\top X$ matrix of linear regression.
:::


:::tip

**Worked.** k=10, d=100 → 1000 centroid values/iteration — independent of N samples.

:::

### Exact result; FDM

Global averaging makes distributed k-means exact (same as single-machine). FDM (Fast Distributed Mining) prunes itemsets locally and confirms global support once per level.

:::note Beyond the lecture
**Why local pruning is safe.** If an itemset must appear in a fraction $s$ of all transactions, then at least one site must have it in a fraction $s$ of its own, because otherwise every site is below $s$ and so is the weighted average. A candidate that is locally frequent nowhere can be skipped. Block 2 runs a simplified version that keeps this idea and one global confirmation round per level, and checks it against a single machine. **DBSCAN, not in the lecture notes.** Density-based clustering can be distributed by cutting the space into strips, giving each worker a halo of its neighbours' points within eps so that core points are classified exactly, and merging only the connections that cross strip borders. Block 3 does this and matches scikit-learn.
:::



## A real system that works this way

**A published algorithm, FDM.** "A Fast Distributed Algorithm for Mining Association Rules" by David Cheung, Jiawei Han, Vincent Ng, Ada Fu and Yongjian Fu appeared in the proceedings of the 4th International Conference on Parallel and Distributed Information Systems (December 1996). Its abstract says FDM identifies the relationship between locally large and globally large itemsets, generates fewer candidate sets, and reduces the number of messages compared with applying a sequential algorithm directly in a distributed setting, with refinements giving further gains. The code below implements only the two ideas the lecture names, local pruning and one confirmation round per level. It does not reproduce the paper's further optimisations.

**k-means in Spark MLlib.** The Spark 4.2.0 clustering documentation says its k-means uses a parallelised variant of k-means++ called k-means|| for initialisation, and shows the API call in two lines of Python. The iterations that follow are the partial-sums scheme of this chapter in spirit, but the documentation does not describe the update step, so that part is the lecture's account rather than something checked against Spark's source.

## Code you can run

Three blocks. The first runs distributed k-means and checks it against a single-machine implementation. The second distributes frequent-itemset mining the way the lecture describes. The third splits DBSCAN across strips of the data and compares with scikit-learn.

### 1. Distributed k-means is exact

Twenty thousand points in 100 dimensions, 10 clusters, 4 workers. Each worker returns per-cluster sums and counts; the code adds them and divides. The run uses the same starting centroids as scikit-learn's Lloyd implementation and the same 10 iterations. A second run shows the tempting mistake of averaging the workers' local means.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

k, d, workers = 10, 100, 4
x, _ = make_blobs(n_samples=20000, n_features=d, centers=k, cluster_std=3.0, random_state=0)
init = x[np.random.default_rng(0).choice(len(x), k, replace=False)]
shards = np.array_split(x, workers)

def local_stats(shard, centroids):
    dist = ((shard[:, None, :] - centroids[None]) ** 2).sum(axis=2)
    label = dist.argmin(axis=1)
    sums = np.zeros_like(centroids)
    counts = np.zeros(len(centroids))
    for c in range(len(centroids)):
        members = shard[label == c]
        sums[c] = members.sum(axis=0)
        counts[c] = len(members)
    return sums, counts

def distributed_kmeans(iters, weighted=True):
    centroids = init.copy()
    sent = 0
    for _ in range(iters):
        parts = [local_stats(s, centroids) for s in shards]
        sent += sum(p[0].size + p[1].size for p in parts)
        if weighted:
            total, count = sum(p[0] for p in parts), sum(p[1] for p in parts)
            centroids = total / np.maximum(count, 1)[:, None]
        else:
            local_means = [p[0] / np.maximum(p[1], 1)[:, None] for p in parts]
            centroids = np.mean(local_means, axis=0)
    return centroids, sent

iters = 10
dist_c, sent = distributed_kmeans(iters)
central = KMeans(n_clusters=k, init=init, n_init=1, max_iter=iters, tol=0, algorithm="lloyd").fit(x)
print("distributed vs single-machine k-means, max centroid difference:", float(np.abs(dist_c - central.cluster_centers_).max()))
naive, _ = distributed_kmeans(iters, weighted=False)
print("average the workers' local means instead of sums and counts, max centroid difference:", round(float(np.abs(naive - central.cluster_centers_).max()), 3))

per_worker_iter = sent / (iters * workers)
print(f"\nvalues sent per worker per iteration: {per_worker_iter:.0f} = k x d + k = {k} x {d} + {k}")
print(f"values in one worker's raw shard: {shards[0].size}")
print("\npoints N | raw data values | summary values per iteration | ratio")
for n in (20_000, 200_000, 20_000_000):
    raw = n * d
    summary = workers * (k * d + k)
    print(f"{n:9d} | {raw:15d} | {summary:28d} | {raw / summary:8.0f}")
```

The summed result matches single-machine k-means to 1e-14, so the answer is the same, not close. Averaging the workers' local means gives centroids up to 9.93 away, because a worker with two points in a cluster and a worker with two thousand count equally in that average. Sums and counts weight every point once. Each worker sends $k\times d+k=1{,}010$ values per iteration, the lecture's $k\times d=1{,}000$ plus the $k$ counts. The last table shows why that matters: the raw data grows with the number of points, the summaries do not.

### 2. Frequent itemsets with local pruning

Four sites hold 1,500 market-basket transactions each over 24 items. Three items are popular everywhere, each site has two extra items that are popular only there, and one pair (items 11 and 12) appears together in about 30% of transactions at every site. The goal is every itemset in at least 6% of all transactions. The code runs two distributed versions, both of which count candidates locally. The first makes every site report counts for every candidate. The second does what the lecture describes: a site reports only the candidates that are frequent in its own data, and the sites that did not report a candidate are then polled for its count. Both are compared with a single machine.

```python
from itertools import combinations

import numpy as np

rng = np.random.default_rng(5)
n_items, n_sites, per_site, min_sup = 24, 4, 1500, 0.06
base = rng.uniform(0.01, 0.12, size=(n_sites, n_items))
base[:, [0, 1, 2]] = 0.5
local_boost = {0: [3, 4], 1: [5, 6], 2: [7, 8], 3: [9, 10]}
for s, items in local_boost.items():
    base[s, items] = 0.42
sites = []
for s in range(n_sites):
    t = rng.random((per_site, n_items)) < base[s]
    both = rng.random(per_site) < 0.3
    t[both, 11] = True
    t[both, 12] = True
    sites.append(t)
total = n_sites * per_site

def support(t, itemset):
    return int(t[:, list(itemset)].all(axis=1).sum())

def gen(prev, k):
    prev = sorted(prev)
    prevset = set(prev)
    out = set()
    for a, b in combinations(prev, 2):
        u = tuple(sorted(set(a) | set(b)))
        if len(u) == k and all(sub in prevset for sub in combinations(u, k - 1)):
            out.add(u)
    return sorted(out)

def run(mode):
    large = [(i,) for i in range(n_items) if sum(support(t, (i,)) for t in sites) >= min_sup * total]
    all_large = list(large)
    rows = []
    k = 2
    while large:
        cands = gen(large, k)
        if not cands:
            break
        local_counts = [{c: support(t, c) for c in cands} for t in sites]
        if mode == "count distribution":
            values = n_sites * len(cands)
            counted = cands
            local_large_total = len(cands) * n_sites
        else:
            local_large = [{c for c in cands if local_counts[s][c] >= min_sup * per_site} for s in range(n_sites)]
            counted = sorted(set().union(*local_large))
            polls = sum(n_sites - sum(c in local_large[s] for s in range(n_sites)) for c in counted)
            values = sum(len(ll) for ll in local_large) + 2 * polls
            local_large_total = sum(len(ll) for ll in local_large)
        large = [c for c in counted if sum(local_counts[s][c] for s in range(n_sites)) >= min_sup * total]
        rows.append((k, len(cands), local_large_total, len(counted), len(large), values))
        all_large += large
        k += 1
    return set(all_large), rows

cd, cd_rows = run("count distribution")
fdm, fdm_rows = run("fdm")
central = set()
level = [(i,) for i in range(n_items)]
cur = [c for c in level if support(np.vstack(sites), c) >= min_sup * total]
central |= set(cur)
k = 2
while cur:
    cands = gen(cur, k)
    cur = [c for c in cands if support(np.vstack(sites), c) >= min_sup * total]
    central |= set(cur)
    k += 1

print(f"{n_sites} sites x {per_site} transactions, {n_items} items, minimum support {min_sup:.0%}")
print("same frequent itemsets as a single machine:", fdm == central, cd == central, f"({len(central)} itemsets)")
print("\nlevel | candidates | locally large (sum over sites) | counted globally | globally large | values sent: count-everything vs FDM-style")
for (k, nc, _, _, nl, v_cd), (_, _, ll, cg, _, v_fdm) in zip(cd_rows, fdm_rows):
    print(f"{k:5d} | {nc:10d} | {ll:30d} | {cg:16d} | {nl:14d} | {v_cd:6d} vs {v_fdm:6d}")
print("total values sent:", sum(r[5] for r in cd_rows), "vs", sum(r[5] for r in fdm_rows))
```

Both distributed versions find the same 69 frequent itemsets as the single machine, which is what makes the pruning safe. The reason is a pigeonhole argument: an itemset that is frequent over all 6,000 transactions must reach the 6% threshold in at least one site's 1,500, otherwise the four sites' counts could not add up to the threshold. So an itemset that is locally frequent nowhere can be dropped without counting it. At level 2 that discards 108 of 171 candidates and cuts the values sent from 684 to 403. At level 3 the local boosts make nearly every candidate locally frequent somewhere and the polling costs more than counting everything (278 against 176). Over the whole run the FDM-style version sends 694 values against 880. The saving is real but data dependent. The count includes poll requests as well as replies, a choice made for this example.

### 3. DBSCAN on strips, with a halo

The lecture's title includes DBSCAN but its text does not, so this block is an addition. Points are cut into 4 vertical strips. Each strip also receives a **halo**: copies of the neighbouring points that lie within eps of its borders. With the halo, a worker can count every owned point's eps-neighbours exactly, so core points are classified correctly. Workers then connect core points locally, and only the connections that cross a strip border are merged centrally.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

x, _ = make_moons(n_samples=3000, noise=0.07, random_state=0)
eps, min_samples, workers = 0.08, 6, 4
edges_q = np.quantile(x[:, 0], np.linspace(0, 1, workers + 1))
edges_q[0], edges_q[-1] = -np.inf, np.inf

owner = np.digitize(x[:, 0], edges_q[1:-1])
core = np.zeros(len(x), dtype=bool)
halo_sizes = []
for w in range(workers):
    own = np.where(owner == w)[0]
    lo, hi = edges_q[w], edges_q[w + 1]
    halo = np.where((x[:, 0] >= lo - eps) & (x[:, 0] < hi + eps) & (owner != w))[0]
    halo_sizes.append(len(halo))
    local = np.concatenate([own, halo])
    nn = NearestNeighbors(radius=eps).fit(x[local])
    counts = np.array([len(a) for a in nn.radius_neighbors(x[own], return_distance=False)])
    core[own] = counts >= min_samples

parent = np.arange(len(x))

def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

cross_edges = 0
for w in range(workers):
    own = np.where((owner == w) & core)[0]
    lo, hi = edges_q[w], edges_q[w + 1]
    halo = np.where((x[:, 0] >= lo - eps) & (x[:, 0] < hi + eps) & (owner != w) & core)[0]
    local = np.concatenate([own, halo])
    nn = NearestNeighbors(radius=eps).fit(x[local])
    for i, nbrs in zip(own, nn.radius_neighbors(x[own], return_distance=False)):
        for j in local[nbrs]:
            union(i, j)
            if owner[j] != w:
                cross_edges += 1

core_idx = np.where(core)[0]
labels = np.full(len(x), -1)
roots = {}
for i in core_idx:
    labels[i] = roots.setdefault(find(i), len(roots))
core_nn = NearestNeighbors(n_neighbors=1).fit(x[core_idx])
dist, nearest = core_nn.kneighbors(x[~core])
border = dist[:, 0] <= eps
labels[np.where(~core)[0][border]] = labels[core_idx[nearest[border, 0]]]

ref = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
ref_core = np.zeros(len(x), dtype=bool)
ref_core[ref.core_sample_indices_] = True
print("core points agree with scikit-learn:", bool((core == ref_core).all()), f"({int(core.sum())} core points)")
print("adjusted Rand index on core points:", round(adjusted_rand_score(ref.labels_[core], labels[core]), 4))
print("noise points: distributed", int((labels == -1).sum()), "| scikit-learn", int((ref.labels_ == -1).sum()), "| same set:", bool(((labels == -1) == (ref.labels_ == -1)).all()))
print("clusters:", len(roots), "| scikit-learn:", len(set(ref.labels_) - {-1}))
print(f"points {len(x)}, strips {workers}, halo points per strip {halo_sizes}, edges that crossed a strip border {cross_edges}")
```

The distributed run finds the same 2,929 core points as scikit-learn, an adjusted Rand index of 1.0 on them, the same 2 clusters and the same 9 noise points. Border points, which a density-based method may legitimately assign to either of two touching clusters, are given to the nearest core point's cluster. The crossing traffic is not small: 3,316 core-to-core edges crossed strip borders, because this simple version forwards every edge. Sending one edge per pair of local clusters would carry the same information in far fewer messages.

## Designing with it

**When "send summaries" works, and when it does not**

| Algorithm | Summary to send | Exact? | Watch out for |
| --- | --- | --- | --- |
| k-means | per-cluster sums and counts | Yes, identical to one machine | Averaging local means is wrong (block 1) |
| Mean, variance, linear regression normal equations | counts, sums, sums of squares, $X^\top X$ | Yes | Numerical stability of sums of squares |
| Frequent itemsets (FDM style) | counts of locally frequent candidates | Yes | Savings depend on how different the sites are (block 2) |
| DBSCAN | core flags and border-crossing edges | Yes for core points | Halo size grows with eps; dense data means many crossing edges |
| Algorithms with no small summary (k-NN, full kernel methods) | nothing small exists | n/a | Fall back to approximations or other parallel schemes |

**Questions to ask first.** Can the update be written as a sum over data points? If so, each worker can compute a partial sum and the reduce is exact. If the answer depends on distances between all pairs of points, look for a way to localise it, such as the halo in block 3.

**Count what you send.** The three code blocks print communication in values. Do the same in your own system before optimising anything else, because the whole point of these algorithms is that this number does not grow with the data.

:::note Not from the lecture
The mean-of-means comparison, the market-basket experiment, the pigeonhole argument and the DBSCAN block are additions for this site. The lecture states the k-means summary, its cost and the FDM idea in two sentences.
:::

## Where this stands in 2026

:::info Industry view

- **Spark MLlib k-means.** The 4.2.0 documentation names k-means|| as its initialisation and describes the DataFrame-based `KMeans` class; a user writes `KMeans().setK(2).setSeed(1)` and the cluster handles the partitioning.
- **The summary idea is general.** Any statistic that is a sum over rows can be computed this way, which is why the next chapter's regression gradients, a sum over rows, distribute so cleanly.
- **Association-rule mining was a distributed-database topic first.** The FDM paper dates from 1996, in a parallel and distributed information systems conference, and its core insight, that local frequency bounds global frequency, still frames how counting problems are partitioned.
- **Reference implementations matter.** The checks in this chapter use scikit-learn 1.9.1 as the single-machine ground truth; a distributed version that cannot match a trusted single-machine one on small data should not be trusted on big data.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> How is k-means distributed across a cluster?</summary>

Partition data; each worker computes partial per-cluster sums and counts; a reduce combines them into new centroids, which are broadcast back — iterating to convergence.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Why does distributed k-means scale well?</summary>

Workers exchange only sufficient statistics (k·d values), independent of the number of data points N.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> For k=10 clusters and d=100 dimensions, how many centroid values are communicated per iteration?</summary>

k×d = 10×100 = 1000 values (plus k counts).<br /><em>Session 9 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Is distributed k-means exact or approximate?</summary>

Exact — global averaging of sufficient statistics yields the same centroids as single-machine k-means.<br /><em>Session 9 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What is FDM?</summary>

Fast Distributed Mining — distributed association-rule mining that prunes candidate itemsets locally and confirms global support with one communication round per level.<br /><em>Session 9 · conceptual</em>

</details>

**Added questions (not in the lecture).**

<details>
<summary><strong>Q6.</strong> 8 workers run k-means with k = 20 clusters in d = 50 dimensions. How many values does each worker send per iteration, and how many in total?</summary>

Each worker sends $k\times d+k=20\times50+20=1{,}020$ values (sums plus counts). The total over 8 workers is 8,160 values per iteration, whatever the number of points.

</details>

<details>
<summary><strong>Q7.</strong> Worker A has one point at 0 in a cluster and worker B has nine points at 10 in the same cluster. What is the correct new centroid, and what do you get by averaging the two workers' local means?</summary>

The correct centroid is $(0+90)/10=9$. Averaging local means gives $(0+10)/2=5$, because it ignores that B holds nine times as many points. Sending sums (0 and 90) and counts (1 and 9) gives 9.

</details>

<details>
<summary><strong>Q8.</strong> Four equally sized sites see an itemset in 2%, 3%, 4% and 5% of their transactions. The minimum support is 6%. Can it be globally frequent?</summary>

No. The global support is the average, 3.5%, below 6%. This is also why FDM can drop it: it is not locally frequent at any site, since all four are below 6%.

</details>

## Further reading

- [Cheung, Han, Ng, Fu and Fu, "A Fast Distributed Algorithm for Mining Association Rules" (PDIS 1996)](https://research.polyu.edu.hk/en/publications/fast-distributed-algorithm-for-mining-association-rules/), the FDM paper (abstract page).
- [Apache Spark 4.2.0: Clustering](https://spark.apache.org/docs/latest/ml-clustering.html), k-means and its parallel initialisation.
- [scikit-learn user guide: Clustering](https://scikit-learn.org/stable/modules/clustering.html), k-means and DBSCAN, the single-machine references used here.
- [D2L: Computational Performance](https://d2l.ai/chapter_computational-performance/index.html), multi-GPU and parameter-server training for the deep-learning side of the same ideas.
- Built from the course lecture "dml-s9-core-algorithms" (Lecture Library series).

- **[D2L — Computational Performance](https://d2l.ai/chapter_computational-performance/index.html)** `book`
  Zhang, Lipton, Li & Smola — Multi-GPU and parallel training explained with runnable code.
- **[Ray documentation](https://docs.ray.io/)** `docs`
  Ray — A practical framework for distributed Python and distributed ML training/serving.
- **[Spark MLlib guide](https://spark.apache.org/docs/latest/ml-guide.html)** `docs`
  Apache Spark — Distributed data processing and ML on Spark — the PySpark backbone.

## Check yourself

- I can describe distributed k-means in one sentence: each worker sends per-cluster sums and counts, the reduce adds them and divides, and the new centroids are broadcast.
- I can compute the communication per iteration, k times d values plus k counts, and explain why it does not depend on the number of points.
- I can show why the result is exact, and why averaging the workers' local means is not.
- I can state the property behind FDM, that a globally frequent itemset is frequent at some site, and what pruning it allows.
- I can describe how a halo lets DBSCAN run on partitions without changing which points are core.
