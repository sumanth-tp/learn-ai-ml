---
id: plat-kubernetes
title: "Kubernetes for ML"
sidebar_label: "Kubernetes for ML"
sidebar_position: 2
slug: /mlops/platform/kubernetes-for-ml
description: "How Kubernetes runs ML workloads: pods and Deployments for serving, the scheduler and GPU requests, the autoscaler formula, Jobs for batch work, and the failure modes that strand expensive GPUs."
tags: [kubernetes, gpu-scheduling, hpa, deployments, jobs, taints, ml-platform]
---

import Infographic from '@site/src/components/Infographic';
import ReplicaSchedulerLab from '@site/src/components/viz/ReplicaSchedulerLab';

**In one line.** Kubernetes keeps a declared number of copies of your container running, places each copy on a machine with room for it, and resizes the fleet from a metric, so the real work for ML is declaring accurately what each copy needs, especially the GPU.

:::note Not from a lecture

This chapter is written for this site from the Kubernetes documentation and the platform pages listed under Further reading; it is not built from a course lecture.

:::

## The idea in plain words

You have a model behind an HTTP server in a container. One copy handles a few requests a second. Traffic doubles at lunchtime, a machine dies overnight, and next week you ship a new image. Doing this by hand means logging into machines. Kubernetes replaces that with a **declaration**: "run three copies of this image, each needing 2 CPUs, 8 GiB of memory and one GPU". A set of controllers then compares what is running with what you declared and keeps closing the gap.

Five objects cover most of ML work.

| Object | What it is | ML use |
| --- | --- | --- |
| **Pod** | One or more containers scheduled together on one node | One model server replica |
| **Deployment** | Keeps N identical pods running and rolls out new versions | Online inference |
| **Job** | Runs pods until a number of them finish successfully | Batch scoring, training runs, evaluations |
| **HorizontalPodAutoscaler (HPA)** | Changes a Deployment's replica count from a metric | Following request load |
| **Node** (with labels and taints) | A machine, possibly with GPUs | The place where cost is |

Two ideas matter more than the rest. First, **requests** are a promise to the scheduler: a pod asking for 2 CPUs is only placed on a node with 2 CPUs unclaimed, whatever it really uses. Second, a **GPU is an indivisible, integer resource** here. You cannot ask for half of one by writing `0.5`, and a pod that cannot find a whole free GPU simply waits in the `Pending` state, costing nothing and serving nobody.

<Infographic src="/img/plat/plat-kubernetes-scheduling.svg" alt="A three-node cluster with preprocessing and GPU pods placed with and without a taint on the GPU nodes" caption="Why GPUs sit idle. The placements, Pending counts and idle GPUs are printed by block 2 below." />

<Infographic src="/img/plat/plat-kubernetes-autoscaling.svg" alt="The autoscaler formula applied to a load spike, showing replicas rise to 9 and fall back to 2 after a 20 tick window" caption="Autoscaling in numbers. The trace is printed by block 3 below." />

## How it works

### Deployments: declare, then converge

A Deployment owns a ReplicaSet, which owns the pods. Change the pod template (a new image tag) and the Deployment starts a new ReplicaSet and shifts replicas across gradually. The documentation gives `maxSurge` and `maxUnavailable` both a default of 25%, which set how many extra pods may exist and how many may be missing during the shift, and `kubectl rollout undo` returns to the previous revision. For model serving this is your first release safety net; the traffic-splitting patterns above it are in [serving and release strategies](/docs/theory/seml/serving-and-release-strategies).

### Requests, limits and the GPU rule

Each container states `requests` (what the scheduler reserves) and `limits` (what the runtime enforces). The documentation is explicit that CPU over the limit is throttled while memory over the limit can get the container killed, so a model that loads weights into memory needs a memory limit set honestly. GPUs follow stricter rules, quoted from the scheduling page:

- you request them in `limits` with a device-plugin name such as `nvidia.com/gpu`;
- you may give a limit without a request, or both, but they must be equal, and a request without a limit is invalid;
- values must be whole numbers;
- the node needs the vendor's driver and device plugin installed, and a GPU is not shared between pods.

Block 1 turns those rules into a linter and runs it on three correct manifests and one broken one.

### The scheduler: filter, then score

For each pending pod the scheduler first **filters** nodes down to the feasible ones (enough unclaimed CPU, memory and GPU, selectors and taints satisfied), then **scores** the survivors and picks the best; equal scores are broken at random. If nothing is feasible the pod stays `Pending`. Scoring is where cluster policy lives. The documentation describes a `MostAllocated` strategy that favours busy nodes (so idle ones can be scaled down) and a `RequestedToCapacityRatio` strategy whose shape function can instead prefer emptier nodes.

The expensive failure is **stranding**. A GPU node has CPUs too. If cheap CPU-only pods land on it, they consume the CPUs that a GPU pod also needs, and a GPU that no pod can use sits idle. The cure is a **taint** on GPU nodes (a repelling mark) and a matching **toleration** on pods that should be allowed there, which the documentation names as the standard way to dedicate nodes with special hardware. Block 2 reproduces it. The price of the cure is also visible: with GPU nodes closed to ordinary pods, the CPU pool must be big enough on its own.

### The autoscaler: one formula and some brakes

The HPA's core rule from the documentation is

$$
\text{desired} = \left\lceil \text{current} \times \frac{\text{current metric}}{\text{target metric}} \right\rceil
$$

evaluated every 15 seconds by default. A **tolerance** of 0.1 means no action when the ratio is within 10% of 1. With several metrics, the controller computes a result for each and takes the largest. Behaviour is damped by a **scale-down stabilisation window**, 300 seconds by default, during which the controller uses the highest recent recommendation; the default for scaling up is no window. The default scale-up policy allows doubling or adding 4 pods every 15 seconds, whichever is larger.

:::note A correction from checking the source

An early read of the autoscaling page suggested a 3 minute default stabilisation window for scale-up. Re-reading the section on default behaviour gives `stabilizationWindowSeconds: 0` for scale-up and 300 for scale-down. The chapter uses those.

:::

Two limits matter for ML. A new GPU replica may take minutes to pull an image and load weights, and the autoscaler only counts pods once they are ready, so the fleet reacts late to a spike. And CPU utilisation is a poor signal for a GPU-bound server; the HPA also accepts custom and external metrics such as queue depth or requests per second, which track the real bottleneck better. Block 3 simulates the formula, tolerance, startup delay and window. For the Pod-level view in a larger project, see the autoscaling section of the [AgentOps chapter](/docs/projects/ai-security/agentops).

### Jobs: work that ends

A Job runs pods until `completions` of them succeed, with `parallelism` running at once. `backoffLimit` bounds retries, `restartPolicy` must be `Never` or `OnFailure`, `ttlSecondsAfterFinished` cleans up, and `completionMode: Indexed` gives every pod a stable index, which maps neatly to "pod 3 scores shard 3". That index is what makes a batch job restartable; chapter 5 builds on it. Block 4 simulates retries and shows how a low `backoffLimit` fails a job that a higher one would finish.

## A real system that works this way

**Azure Machine Learning** documents Kubernetes as a compute target. Its overview lists training with PyTorch, TensorFlow and MPI as supported "via Azure Machine Learning Kubernetes, Azure Machine Learning compute clusters, and serverless compute", and its endpoints page says both online and batch deployments run on managed compute or on Kubernetes. The ML service layers experiments, registries and endpoints over the objects in this chapter.

**Amazon SageMaker HyperPod** documents task governance for clusters orchestrated by Amazon EKS. It is described as a system to streamline resource allocation across teams and projects, and it adds cluster observability: capacity, availability, team allocation and run and wait times. That is the platform answer to the scheduling problem in block 2: GPUs are scarce, so who may use them, when, is a managed policy rather than a first-come race.

## Code you can run

Four blocks, executed with Python 3.14 and PyYAML 6.0.3. The scheduler and autoscaler are small simulators of the rules above, not Kubernetes itself; each block says what it simplifies.

### 1. Lint manifests against the GPU and Job rules

```python
import yaml

DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedder
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedder
  template:
    metadata:
      labels:
        app: embedder
    spec:
      containers:
        - name: server
          image: registry.example/embedder:1.4.0
          resources:
            requests:
              cpu: "2"
              memory: 8Gi
            limits:
              memory: 8Gi
              nvidia.com/gpu: 1
"""

HPA = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: embedder
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: embedder
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
"""

JOB = """
apiVersion: batch/v1
kind: Job
metadata:
  name: nightly-scoring
spec:
  completions: 8
  parallelism: 3
  completionMode: Indexed
  backoffLimit: 4
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: scorer
          image: registry.example/scorer:2.0.1
"""

BROKEN = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: broken
spec:
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: server
          image: registry.example/api:1
          resources:
            requests:
              nvidia.com/gpu: 1
            limits:
              nvidia.com/gpu: 0.5
"""

def containers(doc):
    return doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])

def lint(doc):
    problems = []
    kind = doc["kind"]
    if kind == "Deployment":
        wanted = doc["spec"]["selector"]["matchLabels"]
        labels = doc["spec"]["template"]["metadata"]["labels"]
        if any(labels.get(k) != v for k, v in wanted.items()):
            problems.append("selector does not match the pod template labels")
    if kind in ("Deployment", "Job"):
        for c in containers(doc):
            res = c.get("resources", {})
            for name in set(res.get("requests", {})) | set(res.get("limits", {})):
                if "/" not in name:
                    continue
                req = res.get("requests", {}).get(name)
                lim = res.get("limits", {}).get(name)
                if lim is None:
                    problems.append(f"{name}: requested without a limit")
                elif req is not None and req != lim:
                    problems.append(f"{name}: request {req} differs from limit {lim}")
                if lim is not None and (not isinstance(lim, int) or lim < 1):
                    problems.append(f"{name}: limit {lim} is not a whole number")
    if kind == "Job":
        policy = doc["spec"]["template"]["spec"].get("restartPolicy")
        if policy not in ("Never", "OnFailure"):
            problems.append(f"restartPolicy {policy} is not allowed for a Job")
    if kind == "HorizontalPodAutoscaler":
        spec = doc["spec"]
        if spec["minReplicas"] > spec["maxReplicas"]:
            problems.append("minReplicas is above maxReplicas")
    return problems

for label, text in [("Deployment", DEPLOYMENT), ("HPA", HPA), ("Job", JOB), ("Broken", BROKEN)]:
    doc = yaml.safe_load(text)
    issues = lint(doc)
    print(f"{label:<11}{doc['kind']:<26}" + ("clean" if not issues else ""))
    for issue in issues:
        print(f"{'':<11}- {issue}")
```

The three real manifests pass. The broken one fails three checks at once: a selector that does not match its pods, a GPU request that differs from its limit, and a fractional GPU. A linter like this, or the cluster's own admission checks, turns a Pending mystery into a one-line error at review time.

### 2. Scheduling and GPU stranding

The model filters on CPU, GPU and taints and scores by CPU utilisation after placement only, a simplification: real scoring weighs cpu and memory by default and can include extended resources.

```python
from dataclasses import dataclass, field

@dataclass
class Node:
    name: str
    cpu: float
    gpu: int
    tainted: bool = False
    used_cpu: float = 0.0
    used_gpu: int = 0

    def feasible(self, pod):
        room = self.cpu - self.used_cpu >= pod["cpu"] and self.gpu - self.used_gpu >= pod["gpu"]
        allowed = pod.get("tolerates", False) or not self.tainted
        return room and allowed

    def utilisation_after(self, pod):
        return (self.used_cpu + pod["cpu"]) / self.cpu

def schedule(nodes, pod, replicas, strategy):
    placed, pending = {}, 0
    for _ in range(replicas):
        candidates = [n for n in nodes if n.feasible(pod)]
        if not candidates:
            pending += 1
            continue
        sign = -1 if strategy == "pack" else 1
        best = min(candidates, key=lambda n: (sign * n.utilisation_after(pod), n.name))
        best.used_cpu += pod["cpu"]
        best.used_gpu += pod["gpu"]
        placed[best.name] = placed.get(best.name, 0) + 1
    return placed, pending

def cluster(tainted):
    return [Node("cpu-1", 8, 0), Node("gpu-a", 16, 2, tainted), Node("gpu-b", 16, 4, tainted)]

preprocess = {"cpu": 3, "gpu": 0}
trainer = {"cpu": 4, "gpu": 1, "tolerates": True}
print("5 preprocessing pods (3 CPU) arrive first, then 6 GPU pods (4 CPU, 1 GPU). Cluster: cpu-1 (8 CPU), gpu-a (16 CPU, 2 GPU), gpu-b (16 CPU, 4 GPU)")
for tainted in (False, True):
    for strategy in ("spread", "pack"):
        nodes = cluster(tainted)
        p1, pend1 = schedule(nodes, preprocess, 5, strategy)
        p2, pend2 = schedule(nodes, trainer, 6, strategy)
        free_gpus = sum(n.gpu - n.used_gpu for n in nodes)
        print(f"  gpu nodes tainted={str(tainted):<5} {strategy:<7} preprocessing placed {sum(p1.values())} of 5 on {dict(sorted(p1.items()))}, "
              f"GPU pods placed {sum(p2.values())} of 6, Pending {pend2}, GPUs left idle {free_gpus}")
```

Untainted, the five preprocessing pods leak onto GPU nodes and the GPU pods find only 4 or 5 homes, leaving 2 or 1 GPUs idle depending on strategy. With the taint, all 6 GPU pods run, at the cost of 3 preprocessing pods waiting for CPU nodes: the cluster now needs more CPU capacity, which is the right problem to have. The lab below runs this exact model with sliders; its defaults reproduce the first row (4 of 6 placed, 2 Pending, 2 GPUs idle) and an autoscaler readout whose defaults reproduce block 3's jump to 9.

<ReplicaSchedulerLab />

### 3. The autoscaling formula under a spike

One tick stands for one 15 second sync, so the 20-tick window is the default 300 seconds. Each pod requests 1 CPU. New pods become ready 2 ticks after creation and only ready pods supply the utilisation figure. The model keeps the formula, tolerance, bounds, startup delay and the scale-down window, and omits the real controller's other rules for starting pods.

```python
import math

def hpa_step(current, ready, load, target, tolerance, lo, hi):
    util = load / ready
    ratio = util / target
    if abs(ratio - 1) <= tolerance:
        return current, util
    return min(hi, max(lo, math.ceil(ready * ratio))), util

def simulate(load_curve, target=0.70, tolerance=0.10, lo=2, hi=12, startup=2, window=20):
    replicas, ready_at = [2], []
    pods = [(0, 2)]
    history, rows = [], []
    for t, load in enumerate(load_curve):
        ready = sum(c for born, c in pods if born <= t)
        current = sum(c for _, c in pods)
        want, util = hpa_step(current, max(ready, 1), load, target, tolerance, lo, hi)
        history.append(want)
        recent = history[-window:]
        chosen = max(recent) if want < current else want
        if chosen > current:
            pods.append((t + startup, chosen - current))
        elif chosen < current:
            drop = current - chosen
            newest_first = sorted(pods, key=lambda p: -p[0])
            pods = []
            kept = []
            for born, c in newest_first:
                take = min(c, drop)
                drop -= take
                if c - take:
                    kept.append((born, c - take))
            pods = kept
        rows.append((t, load, ready, sum(c for _, c in pods), util))
    return rows

curve = [1.4] * 4 + [6.3] * 14 + [1.4] * 26
rows = simulate(curve)
print("load in CPUs of work; one pod requests 1 CPU; target 70%; new pods are ready 2 ticks after creation")
print(" tick  load  ready  replicas  utilisation")
for t, load, ready, replicas, util in rows:
    if t < 12 or t in (17, 18, 22, 30, 37, 43):
        print(f"{t:>5}  {load:>4.1f}  {ready:>5}  {replicas:>8}  {util:>10.0%}")
peak = max(r[3] for r in rows)
print(f"\npeak replicas {peak}; replicas at the end {rows[-1][3]}; first tick at 2 replicas again: {next(r[0] for r in rows if r[0] > 18 and r[3] == 2)}")
print("formula check: ceil(2 * 1.4/2/0.7) =", math.ceil(2 * (1.4 / 2) / 0.7), "and ceil(2 * 6.3/2/0.7) =", math.ceil(2 * (6.3 / 2) / 0.7))
```

At tick 4 the load jumps and two ready pods read 315% of target, so the formula asks for 9 replicas at once. Those pods are not ready until tick 6, which is the spike-time latency to budget for. When the load falls at tick 18 the instant recommendation is far lower, but the window still contains the earlier 9s, so the fleet stays at 9 until tick 37. That patience costs money and protects against flapping.

### 4. Job retries and backoffLimit

```python
import random

def run_job(completions, parallelism, backoff_limit, fail_rate, seed):
    rng = random.Random(seed)
    todo = list(range(completions))
    done, failures, rounds, log = set(), 0, 0, []
    while todo:
        rounds += 1
        batch = todo[:parallelism]
        todo = todo[parallelism:]
        retry = []
        for index in batch:
            if rng.random() < fail_rate:
                failures += 1
                retry.append(index)
            else:
                done.add(index)
        log.append((rounds, batch, len(done), failures))
        if failures > backoff_limit:
            return "Failed (BackoffLimitExceeded)", log, done
        todo = retry + todo
    return "Complete", log, done

for limit, rate in [(4, 0.20), (1, 0.20), (4, 0.80)]:
    state, log, done = run_job(8, 3, limit, rate, seed=4)
    print(f"completions 8, parallelism 3, backoffLimit {limit}, pod failure rate {rate:.0%}: {state}, {len(done)} of 8 indexes done, {log[-1][3]} failed pods in {len(log)} rounds")
state, log, done = run_job(8, 3, 4, 0.20, seed=4)
print("\nround  indexes run  done so far  failed so far")
for rnd, batch, d, f in log:
    print(f"{rnd:>5}  {str(batch):<12} {d:>11}  {f:>13}")
```

With a 20% pod failure rate the same seeded run completes under `backoffLimit: 4`, with 3 failed pods along the way, and fails under `backoffLimit: 1` after only 3 of 8 indexes are done. At an 80% failure rate no limit helps: the job fails and the fault is in the container, not the cluster. Retries are for transient faults such as a preempted node, not for bugs.

## Production snippets (not run here)

These need a cluster. A GPU node pool is tainted, and a batch scoring Job tolerates the taint and runs sharded.

```bash
kubectl taint nodes gpu-a-1 dedicated=gpu:NoSchedule
kubectl label nodes gpu-a-1 accelerator=example-gpu-x100
kubectl rollout status deployment/embedder
kubectl rollout undo deployment/embedder
```

Not run in this environment.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nightly-scoring-gpu
spec:
  completions: 8
  parallelism: 3
  completionMode: Indexed
  backoffLimit: 4
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        accelerator: example-gpu-x100
      tolerations:
        - key: dedicated
          operator: Equal
          value: gpu
          effect: NoSchedule
      containers:
        - name: scorer
          image: registry.example/scorer:2.0.1
          resources:
            limits:
              nvidia.com/gpu: 1
```

Not run in this environment.

## Designing with it

- **Set requests from measurement.** Over-asking strands capacity; under-asking gets pods throttled or killed. Load-test one replica, then set the request near its steady use.
- **Fence the GPUs.** Taint GPU nodes and tolerate only in GPU pods, and size the CPU pool for the preprocessing and sidecars that no longer fit on GPU nodes.
- **Scale on the real bottleneck.** For GPU servers prefer queue length, in-flight requests or tokens per second over CPU, and budget the minutes a cold replica needs before it counts.
- **Keep a floor.** `minReplicas` of 2, on different nodes, can survive one node loss; scaling to zero saves money but turns the first request into a cold start.
- **Use Jobs for batch work, with idempotent shards.** A retried pod must produce the same output for its index; see [batch inference pipelines](/docs/mlops/platform/batch-inference-pipelines).
- **Give every rollout an undo.** Readiness probes plus `rollout undo` beat any amount of staging confidence.
- **Describe the cluster as code.** Hand-edited node pools drift; the next chapter, [infrastructure as code](/docs/mlops/platform/infrastructure-as-code), is how to stop that. Containers and orchestration fundamentals are in [containers and orchestration](/docs/theory/seml/containers-and-orchestration).

## Where this stands in 2026

:::info Industry view

- **Version.** The Kubernetes releases page lists 1.37.1 as the latest stable release, dated 2026-09-15, with 1.36 and 1.35 also supported at the time of writing.
- **GPU scheduling is stable.** The GPU scheduling page is marked stable since Kubernetes v1.26, but remains device-plugin based, so integer GPUs per pod is still the model.
- **Pod-level resources are newer.** The resource management page describes a `PodLevelResources` feature gate from v1.34. Check the current feature state before relying on it.
- **Managed platforms wrap the same primitives.** The Azure and SageMaker pages above present Kubernetes as a compute option under higher-level ML services, which is why the objects in this chapter remain worth knowing even when you never write them.
- **Not verified here.** Autoscaling to zero for HPA, GPU sharing and partitioning schemes, and queueing systems for batch GPU jobs were not checked for this chapter.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A pod requests `nvidia.com/gpu: 0.5`. What happens, and what do you do if you want to share one GPU between two small models?</summary>

The request is invalid: GPU values must be whole numbers and are not shared between pods. Either serve both models from one process in one pod, so the pod holds a whole GPU, or use a partitioning feature that your GPU vendor documents (not checked for this chapter). The linter in block 1 flags the fractional value.

</details>

<details>
<summary><strong>Q2.</strong> Your cluster has 3 idle GPUs but a GPU pod is Pending. Give two causes.</summary>

CPU or memory requests: the free GPUs sit on nodes whose remaining CPU is too small for the pod, often because CPU-only pods landed there (block 2). Or a node selector or taint excludes the nodes with free GPUs. `kubectl describe pod` shows the scheduler's reason.

</details>

<details>
<summary><strong>Q3.</strong> Two ready replicas read 315% CPU against a 70% target. How many replicas does the HPA ask for, and why might users still see errors for a minute?</summary>

ceil(2 x 315 / 70) = 9. The new pods are not counted until ready, and a model server needs time to pull its image and load weights, so capacity arrives late. Keep headroom, prewarm, or scale on a leading metric such as queue depth.

</details>

<details>
<summary><strong>Q4.</strong> Load drops to a fifth of its peak, yet the fleet stays large for five minutes. Is the autoscaler broken?</summary>

No. The default scale-down stabilisation window is 300 seconds and the controller acts on the highest recommendation inside it, so brief dips do not cause flapping. Shorten it with the behavior field if cost matters more than stability.

</details>

<details>
<summary><strong>Q5.</strong> You set `backoffLimit: 1` on a batch job and it fails when one node is preempted twice. What do you change, and what would tell you the problem is not transient?</summary>

Raise `backoffLimit` and make shards idempotent so a retry is safe. If retries keep failing at a high rate (block 4 at 80%) the fault is in the container or data, not the node, and more retries only burn compute.

</details>

<details>
<summary><strong>Q6.</strong> Why taint GPU nodes instead of just labelling them and using node selectors on GPU pods?</summary>

A selector only attracts GPU pods; it does not keep other pods out. Without the taint, ordinary pods can still land on GPU nodes and consume CPU that GPU pods need. A taint repels every pod that does not tolerate it.

</details>

## Further reading

- [Kubernetes: Schedule GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/): resource naming, the limits-only rule, integer values and device plugins.
- [Kubernetes: Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/): the formula, tolerance, behaviour and default windows.
- [Kubernetes: Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/): completions, parallelism, indexed mode and retries.
- [Kubernetes: Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/): rolling updates and rollback.
- [Kubernetes: Resource management for pods and containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/): requests, limits and quality of service.
- [Kubernetes: Taints and tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/): dedicating nodes.
- [Kubernetes: Resource bin packing](https://kubernetes.io/docs/concepts/scheduling-eviction/resource-bin-packing/): scoring strategies including extended resources.
- [Kubernetes releases](https://kubernetes.io/releases/): supported versions.
- [Azure Machine Learning overview](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning) and [Amazon SageMaker AI features](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis-features.html): managed ML platforms that run on Kubernetes.

## Check yourself

- I can explain the difference between a request and a limit, and why GPUs are limits-only integers.
- I can describe how the scheduler filters and scores nodes, and what a Pending pod means.
- I can explain GPU stranding and fix it with a taint, a toleration and a right-sized CPU pool.
- I can compute the HPA's desired replicas from a metric, and name the tolerance and stabilisation window that damp it.
- I can choose between a Deployment and a Job, and explain what backoffLimit and indexed completion buy me.
- I can say which behaviours I took from the Kubernetes documentation and which I only simulated here.
