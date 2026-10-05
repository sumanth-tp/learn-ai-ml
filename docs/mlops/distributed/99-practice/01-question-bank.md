---
id: dist-question-bank
title: "Distributed ML Question Bank"
sidebar_label: "1 · Question bank"
sidebar_position: 1
slug: /mlops/distributed/question-bank
description: "All 29 unique source questions, grouped by lecture, with retained answers, explicit qualifications and checked calculations."
tags: [distributed-ml, practice, revision]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Reconstruct each distributed-training answer from its objective, update rule and cost assumptions.

Built from the course question banks "dml-question-bank" and "dml-comprehensive-question-bank" (Lecture Library series).

## The idea in plain words

The first bank contains 29 question-answer pairs. The comprehensive bank repeats its last 15 pairs exactly after ignoring their local numbering and whitespace. This chapter retains all 29 unique pairs and the lecture grouping. The original concise answers remain visible; labelled qualifications restore assumptions that an examination shorthand can leave implicit.

Try a question before opening its answer. For a numerical problem, name the quantity before calculating it: model storage is not total training memory, sent ring payload is not sent-plus-received traffic, and a variance reduction is not a standard-deviation reduction. The checked arithmetic block below covers every numerical source answer in this bank.

<Infographic src="/img/dist/question-bank-revision.svg" alt="Six checked quantities: Amdahl, global batch, bubble fraction, ring payload, quantisation ratio and FedAvg" caption="Original revision board. Every value is calculated by the code below; the two source banks contain 29 unique pairs and 15 duplicate pairs." />

## How it works

Use the lecture sections below as the revision order. The bank's answers are retained as supplied, with punctuation adjusted to the site's house style. A qualification is an addition rather than an unmarked replacement. Questions labelled post-mid remain here because this is a complete course bank, not only the mid-semester syllabus.

## Practice questions

## Foundations & Frameworks (S1-2)

<details>
<summary><strong>Q1.</strong> Distinguish parallel and distributed computing, and give the two forms of parallelism in ML.</summary>

Parallel computing uses multiple processors with shared memory on one machine; distributed computing coordinates separate machines over a network. In ML: data parallelism (replicate the model, shard the data, sync gradients) and model parallelism (split one model across devices).


**Qualification beyond the bank.** Parallel computing also includes distributed-memory machines. The source describes a common shared-memory example, not the definition of all parallel computation.

</details>

<details>
<summary><strong>Q2.</strong> Contrast Hadoop MapReduce and Apache Spark for ML.</summary>

Hadoop MapReduce writes to disk between steps (fault-tolerant batch); Spark keeps data in memory via RDDs and uses a DAG scheduler, making iterative ML (many passes) far faster.


**Qualification beyond the bank.** Spark can persist RDDs in memory or on disk; keeping every intermediate in memory is not automatic. Runtime depends on working-set size, persistence, shuffle and workload. The source does not establish a universal speedup.

</details>

<details>
<summary><strong>Q3.</strong> State Amdahl's law and compute the speed-up for p=0.9, n=10 workers.</summary>

S(n) = 1/((1−p)+p/n) = 1/(0.1+0.09) = 5.26×. The ceiling (n→∞) is 1/(1−p) = 10×, set by the serial fraction.

</details>

<details>
<summary><strong>Q4.</strong> What is data and computation heterogeneity?</summary>

Varied data sources/formats and uneven/mixed hardware across the cluster, which frameworks and algorithms must accommodate for balanced, efficient training.

</details>

## Parallelism: Data & Model (S3-4)

<details>
<summary><strong>Q5.</strong> Describe the data-parallel training loop and the effective batch size.</summary>

Replicate the model on every worker, give each a data shard, compute local gradients, then all-reduce (average) them. With K workers at local batch B the effective (global) batch is K·B; e.g. K=8, B=32 → 256.

</details>

<details>
<summary><strong>Q6.</strong> State the linear scaling rule.</summary>

When the global batch is multiplied by K, multiply the learning rate by K too (with warm-up) to keep training dynamics comparable.


**Qualification beyond the bank.** The linear scaling rule is a heuristic with a regime of validity, usually paired with warm-up. It is not guaranteed to preserve accuracy for arbitrary batch sizes or models.

</details>

<details>
<summary><strong>Q7.</strong> When is model parallelism needed, and what is pipeline parallelism?</summary>

When a model is too large for one device: split its layers/tensor slices across workers. Pipeline parallelism runs stages on different devices like an assembly line, keeping micro-batches in flight.

</details>

<details>
<summary><strong>Q8.</strong> Compute the pipeline bubble fraction for S=4 stages and m=8 micro-batches, and how to shrink it.</summary>

bubble = (S−1)/(m+S−1) = 3/11 = 0.273 (~27% idle). Use more micro-batches (m→∞ ⇒ bubble→0), trading memory for utilisation.


**Qualification beyond the bank.** This is an ideal equal-stage timing model. More micro-batches reduce the fill/drain fraction; the activation-memory trade-off depends on scheduling and recomputation, rather than increasing universally with the micro-batch count.

</details>

<details>
<summary><strong>Q9.</strong> A 24 GB model is split evenly over 4 GPUs. Memory per GPU?</summary>

24/4 = 6 GB per GPU (plus activations) , fitting what no single GPU could hold.


**Qualification beyond the bank.** Six GB counts only the specified evenly split model storage. Activations, gradients, optimiser state and temporary buffers may also need memory. No device capacity was given, so the arithmetic alone does not prove that a single device cannot hold the model.

</details>

## Distributed Challenges & Programming Models (S5-7)

<details>
<summary><strong>Q10.</strong> Name the four core challenges of distributed ML.</summary>

Consistency, fault tolerance, communication overhead, and resource management.

</details>

<details>
<summary><strong>Q11.</strong> Contrast strong and eventual consistency, and how fault tolerance is achieved.</summary>

Strong consistency keeps workers exactly in sync (correct, slow); eventual consistency lets replicas drift and reconcile (fast, stale). Fault tolerance uses checkpointing , periodically saving state to resume after failures.


**Qualification beyond the bank.** Consistency is a state agreement property, not an accuracy guarantee. Checkpoint recovery also needs optimiser state and data/round progress; the cost depends on the protocol.

</details>

<details>
<summary><strong>Q12.</strong> Under ring all-reduce with N=4 workers, how much does each worker transfer per all-reduce, and why does it scale?</summary>

≈2(N−1)/N = 2·3/4 = 1.5× the model size (→2× as N→∞). It scales because per-worker traffic is independent of N, unlike a central parameter server (N×).


**Qualification beyond the bank.** The expression counts bytes sent per rank for the ideal ring payload. Each rank receives the same amount separately. It depends on N but is bounded by two model copies, rather than being exactly independent of N; message latency and the number of phases still grow.

</details>

<details>
<summary><strong>Q13.</strong> Describe the MapReduce programming model and why Spark RDDs suit ML better.</summary>

Map transforms each record in parallel, then reduce aggregates , simple, fault-tolerant, disk-heavy. Spark RDDs keep intermediates in memory with a DAG scheduler, so iterative algorithms avoid repeated disk I/O.


**Qualification beyond the bank.** RDD persistence is an explicit policy and has memory/disk storage levels. A DAG scheduler does not by itself eliminate every disk operation or shuffle.

</details>

<details>
<summary><strong>Q14.</strong> 1000 GB processed by 100 mappers , data per mapper, and what is data locality?</summary>

1000/100 = 10 GB each in parallel. Data locality = moving computation to the data (not data to computation) to minimise network transfer.

</details>

## Post-Mid: Core Algorithms & Distributed Regression (S9-10)

<details>
<summary><strong>Q15.</strong> How is k-means distributed, and why does it scale?</summary>

Partition data; each worker computes partial per-cluster sums/counts; a reduce combines them into new centroids, broadcast back. It scales because only sufficient statistics (k·d values) move, independent of N samples , and the result is exact.


**Qualification beyond the bank.** Exact centroids require global sums and counts for the same assignments and starting centroids, including an empty-cluster policy. This does not establish globally optimal clustering or independence from communication topology.

</details>

<details>
<summary><strong>Q16.</strong> For k=10 clusters and d=100 dimensions, how many centroid values are communicated per iteration?</summary>

k×d = 10×100 = 1000 values (plus k counts), independent of the number of data points.


**Qualification beyond the bank.** One worker contributes k*d sums and k counts. Total worker traffic multiplies by the number of workers and includes returning/broadcasting centroids; 1000 is not the whole-job byte count.

</details>

<details>
<summary><strong>Q17.</strong> What is FDM?</summary>

Fast Distributed Mining , distributed association-rule mining that prunes candidate itemsets locally and confirms global support with one communication round per level.


**Qualification beyond the bank.** The source gives a simplified per-level view. FDM includes candidate exchange, polling/support counts and result propagation; one communication round should not be read as one message or as a universal one-collective implementation.

</details>

<details>
<summary><strong>Q18.</strong> Why does regression distribute cleanly, and how do linear vs logistic differ?</summary>

Its loss is a sum over rows, so the gradient is a sum too: partition rows, compute partial gradients, average, step , identical to single-machine GD. Linear vs logistic differ only in the loss (squared error vs log-loss with sigmoid).


**Qualification beyond the bank.** A plain mean of local mean gradients needs equal sample counts. Otherwise combine gradient sums and divide by total count, or weight local means by shard size; all contributions must use the same model version.

</details>

<details>
<summary><strong>Q19.</strong> Four workers report gradients [2,4,6,8] for a parameter. Aggregate them.</summary>

(2+4+6+8)/4 = 5 (the full-data gradient for an even split); then w ← w − η·5.

</details>

## Post-Mid: Distributed & Advanced SGD (S11-12)

<details>
<summary><strong>Q20.</strong> What objective does distributed SGD minimise, and name the three GD variants?</summary>

The empirical risk (1/n)Σℓᵢ(θ). Batch GD (all data, accurate/slow), stochastic GD (one sample, noisy/fast), and mini-batch GD (small batch, the practical middle, parallelisable).

</details>

<details>
<summary><strong>Q21.</strong> By what factor does a mini-batch of B=32 reduce gradient variance?</summary>

By B: variance σ²/32; the gradient standard deviation drops by √32 ≈ 5.7×. Bigger batches give smoother gradients and parallelise well.


**Qualification beyond the bank.** The variance formula assumes independent equal-variance observations. Its standard-deviation factor is sqrt(32)=5.656854. Correlated observations can provide much less variance reduction.

</details>

<details>
<summary><strong>Q22.</strong> Contrast synchronous and asynchronous distributed SGD, and what Hogwild! exploits.</summary>

Synchronous averages gradients each step (consistent, straggler-bound); asynchronous updates without waiting (faster, stale gradients). Hogwild! does lock-free async updates, relying on gradient sparsity.


**Qualification beyond the bank.** Asynchronous execution need not reach a fixed quality faster. Hogwild! specifically studies favourable sparse shared-memory updates; that mechanism is distinct from a general networked parameter server.

</details>

<details>
<summary><strong>Q23.</strong> What communication saving does 8-bit gradient quantisation give vs 32-bit?</summary>

32/8 = 4× less communication per step, often with negligible accuracy loss via error feedback; combined with sparsification the savings multiply.


**Qualification beyond the bank.** Four is the ideal coordinate-payload ratio. Include scales, sparse indices, alignment and codec cost before quoting actual savings. Error feedback does not guarantee negligible accuracy loss for every setting; sparsity and quantisation ratios do not automatically multiply for the complete message.

</details>

<details>
<summary><strong>Q24.</strong> How does local-update SGD reduce communication, and what does Adacomm add?</summary>

It runs τ local steps between synchronisations, cutting communication rounds by τ× (some convergence cost). Adacomm adaptively tunes τ , large early, small near convergence.


**Qualification beyond the bank.** Exactly tau-fold fewer rounds assumes a divisible fixed local-step budget. A partial final round uses ceil(T/tau). Adaptive schedules can help under stated assumptions but do not guarantee a particular loss or runtime.

</details>

## Post-Mid: Federated Learning & Special Topics (S13-15)

<details>
<summary><strong>Q25.</strong> What is federated learning and why use it?</summary>

Training a shared model across clients without moving their raw data , each trains locally and sends only model updates; it is privacy-preserving and suits data that cannot be centralised.


**Qualification beyond the bank.** Keeping raw records local does not by itself guarantee privacy. Updates, aggregates and the trained model can reveal information; secure aggregation and differential privacy need separate threat models and implementation.

</details>

<details>
<summary><strong>Q26.</strong> Write the FedAvg rule and compute the global model for n=[100,300], w=[0.4,0.8].</summary>

w_global = Σₖ(nₖ/n)wₖ = (100·0.4 + 300·0.8)/400 = (40+240)/400 = 0.7 , client models averaged, weighted by data size.

</details>

<details>
<summary><strong>Q27.</strong> What is secure aggregation, and why is non-IID data a challenge?</summary>

Secure aggregation (via MPC) lets the server learn only the sum of client updates, not any individual one. Non-IID data means client distributions differ, so local updates diverge/conflict, slowing convergence and biasing the global model.


**Qualification beyond the bank.** Secure aggregation hides individual inputs under protocol, cohort, collusion and dropout assumptions. It does not make every aggregate private, or prove that an update is honest. Non-IID is a family of distribution and objective differences.

</details>

<details>
<summary><strong>Q28.</strong> What does Adacomm change relative to standard distributed SGD, and the reduction for τ=4?</summary>

It does several adaptive local updates between synchronisations instead of syncing every step; τ=4 local steps gives 4× fewer communication rounds with little accuracy loss.


**Qualification beyond the bank.** Four-fold fewer rounds applies to a divisible step budget. Little accuracy loss is conditional; the heterogeneous quadratic example in special topics gives excess loss 0.035284 at period four.

</details>

<details>
<summary><strong>Q29.</strong> What single theme unifies the distributed-ML techniques in this course?</summary>

Minimising communication (efficient all-reduce, quantisation, local updates, federated aggregation) , because moving gradients/data, not computing them, dominates distributed-ML cost.


**Qualification beyond the bank.** Communication is a recurring course theme, not a universal bottleneck. Measure compute, memory, input wait, network and imbalance before selecting a remedy.

</details>

## A real system that works this way

:::note Beyond the bank
For the data-parallel answers, [PyTorch's DDP design](https://docs.pytorch.org/docs/2.14/notes/ddp.html) supplies a concrete reduction implementation. For the framework answers, [Spark's RDD guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html) describes persistence and storage levels. These documentation references qualify the bank's simplified descriptions; no Spark or Hadoop job was run here.
:::

## Code you can run

### Every numerical answer, recomputed

Python 3.14.6 computes these values without external dependencies. The source rounded answers reproduce: Amdahl 5.26, pipeline bubble 0.273, model share 6 GB, ring sent payload 1.5 model copies, mapper share 10 GB, 1000 centroid coordinates, gradient mean 5, payload factor 4 and FedAvg 0.7. Standard deviation is reduced by 5.656854 under the stated independence assumption.

```python
import math
checks = [
    ('Q3 Amdahl speedup', 1/((1-0.9)+0.9/10)),
    ('Q3 serial ceiling', 1/(1-0.9)),
    ('Q5 global batch', 8*32),
    ('Q8 pipeline bubble', (4-1)/(8+4-1)),
    ('Q9 evenly split model GB', 24/4),
    ('Q12 ring model copies sent', 2*(4-1)/4),
    ('Q14 GB per mapper', 1000/100),
    ('Q16 centroid coordinates', 10*100),
    ('Q16 coordinates including counts', 10*100+10),
    ('Q19 gradient mean', sum([2,4,6,8])/4),
    ('Q21 independent variance factor', 32),
    ('Q21 standard-deviation factor', math.sqrt(32)),
    ('Q23 ideal payload factor', 32/8),
    ('Q26 FedAvg', (100*0.4+300*0.8)/400),
    ('Q28 divisible 24-step round factor', 24/math.ceil(24/4)),
]
for name,value in checks:
    print(name, f'{value:.6f}')
assert math.isclose(checks[0][1], 100/19)
assert math.isclose(checks[3][1], 3/11)
assert math.isclose(checks[13][1], 0.7)
```


## Designing with it

:::note Beyond the bank
For an architectural answer, draw where the model lives, where rows live and which values cross a machine boundary. Then state who performs the optimiser step and what happens if a participant is slow or absent. This exposes the difference between a sharded parameter server, a collective reduction and pipeline execution without relying on names alone.

For a numerical answer, keep units beside the calculation. A ratio of bits per coordinate does not include scale metadata. A count of centroid coordinates does not include all worker transmissions. A storage calculation should separate parameters, activations, gradients and optimiser state. These distinctions are part of the answer, not optional implementation trivia.

When a source says exact, identify the reference: the same batch, initial parameters, assignments, sample weighting and update version. When it says faster, identify the measurement: elapsed time to a fixed quality, a fixed-step toy cost or a published workload. An equality claim and a speed claim need different evidence.
:::

## Where this stands in 2026

:::info Industry view
Sources were checked on 5 October 2026. This bank teaches mechanisms, rather than a current tool ranking or hardware benchmark. The framework pages opened identify Spark 4.2.0 and Hadoop 3.3.5; neither package was installed or executed for this bank. PyTorch examples in the teaching chapters run with 2.14.1 on CPU.
:::

## Further reading

- [Scalable frameworks](/docs/mlops/distributed/scalable-frameworks): Amdahl and collectives.
- [Data parallelism](/docs/mlops/distributed/data-parallelism): sample weighting and batch size.
- [Model parallelism](/docs/mlops/distributed/model-parallelism): pipeline fill and drain.
- [Programming models](/docs/mlops/distributed/programming-models): MapReduce, Spark and parameter servers.
- [Distributed regression](/docs/mlops/distributed/distributed-linear-and-logistic-regression): exact reduction and staleness.
- [Advanced deep learning](/docs/mlops/distributed/advanced-distributed-deep-learning): variance and message precision.
- [Federated learning](/docs/mlops/distributed/federated-learning): weighted models and privacy boundaries.
- [Spark RDD programming guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html): persistence is a policy.
- [Hadoop MapReduce tutorial](https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html): map, shuffle and reduce.
- [Large-minibatch linear scaling study](https://arxiv.org/abs/1706.02677): a rule with a tested range.
- [FDM original paper](https://hub.hku.hk/bitstream/10722/45576/1/26205.pdf): candidate pruning and support-count exchange.

## Check yourself

- I can answer all 29 unique source questions and explain their assumptions.
- I can reproduce every numerical answer without confusing units.
- I can distinguish a sample-weighted objective from equal client influence.
- I can explain why reduced communication is conditional on quality, overhead and workload.
- I can state which privacy and benchmark claims these calculations do not verify.
