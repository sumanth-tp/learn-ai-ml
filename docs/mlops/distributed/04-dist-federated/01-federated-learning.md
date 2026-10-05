---
id: dist-federated
title: "Federated Learning"
sidebar_label: "1 · Federated Learning"
sidebar_position: 1
slug: /mlops/distributed/federated-learning
description: "Federated learning keeps training records at clients and combines their local model updates under an explicit weighting and privacy contract."
tags: [distributed-ml, optimisation, training]
---

import Infographic from '@site/src/components/Infographic';
import FedAvgLab from '@site/src/components/viz/FedAvgLab';

**In one line.** Federated learning keeps training records at clients and combines their local model updates under an explicit weighting and privacy contract.

Built from the course lecture "dml-s13-14-federated" (Lecture Library series).

## The idea in plain words

:::note Beyond the lecture
Federated learning keeps training records at clients and combines their local model updates under an explicit weighting and privacy contract.

Imagine several hospitals trying to improve a shared prediction model while retaining patient records locally. A coordinator sends the same starting model to participating hospitals. Each performs authorised local training and returns a model update. The coordinator combines those updates and sends out a new version. The training records remain at the hospitals, but the updates are still information derived from those records.

The architecture addresses where training happens and which information is transmitted. It does not by itself settle who may participate, whether an update exposes a rare record, whether the coordinator can manipulate rounds, or whether the final model reveals information. Those require separate controls and a stated threat model. The lecture's central message, sending models instead of data, is useful when read with that boundary.

FedAvg adds a weighting rule. A client with more examples can contribute more to a sample-weighted objective, but equal client influence is a different legitimate objective. Decide whose outcomes matter before choosing the weights. An average over all records can perform poorly for a small institution even while its pooled metric improves.

This chapter starts from the lecture's two-client arithmetic, then runs local training with real CPU autograd. A separate mask-cancellation example shows why hiding individual inputs needs a protocol with dropout handling. These examples are deliberately small enough to inspect without collecting any real sensitive data. The [local SGD chapter](/docs/mlops/distributed/advanced-sgd-techniques) supplies the optimisation background.
:::

<Infographic src="/img/dist/federated-learning-fedavg.svg" alt="Federated Learning: the mechanism and checked example values" caption="Original board. Values reproduce the independent code blocks below; synthetic costs are labelled." />

## How it works

Train on everyone's data , **without the data ever leaving their device**.

- **What you'll learn** , FL setup, FedAvg, secure aggregation, non-IID, privacy.
- **How to use it** , Compute one FedAvg round.
- **The one idea** , Send models, not data.

### Weighted averaging

Clients train locally and send updates; the server averages them weighted by data size: w = Σ(nₖ/n)·wₖ.

:::tip

**Worked.** n=[100,300], w=[0.4,0.8] → (40+240)/400 = 0.7.

:::

### Secure aggregation & privacy

Secure aggregation and MPC reveal only the sum of updates; differential privacy adds noise. Non-IID client data and cross-client hyperparameter tuning are the hard parts.

:::note Corrections and assumptions
The lecture calls federated learning privacy-preserving because raw records remain local. Keeping records local reduces one exposure channel but is not a complete privacy guarantee: model updates and aggregates can leak information. Secure aggregation protects individual inputs under its protocol assumptions; differential privacy requires bounded contributions, a specified noise mechanism and privacy accounting. The source's FedAvg value 0.7 is arithmetically correct.
:::

:::note Beyond the lecture
### Define the global objective

Let client k have nₖ local examples and mean local loss Fₖ(w). A sample-weighted objective is F(w)=Σₖ(nₖ/N)Fₖ(w), where N is the total example count in the defined population. FedAvg sends a common model to selected clients, runs local optimiser steps and combines their resulting models. If every selected client takes exactly one full local gradient step from the same starting point, count-weighted model averaging is equivalent to a count-weighted gradient step.

That one-step equivalence generally disappears with several local steps. Local models follow gradients evaluated at their own intermediate states. The server averages endpoints, not gradients all evaluated at the original model. This distinction is why heterogeneous clients and the amount of local work matter. The CPU code uses different curvatures and four local steps so the difference can be observed without stochastic noise.

The original lecture's n=[100,300] and w=[0.4,0.8] produce (40+240)/400=0.7. The plain mean is 0.6, which is correct for equal client influence and incorrect for the stated sample weighting. Neither arithmetic result says which policy is fair or suitable for a deployment. The policy must be named alongside the reported value.

When only a subset participates, the usual round average renormalises weights over that selected subset. This makes a valid round aggregate, but it need not be an unbiased estimate of a desired full-population update. Participation can correlate with connectivity, device resources or local data characteristics. A simple count weight cannot automatically compensate for that selection bias.

### Keep the round state explicit

A practical round identifies the starting model version, selected clients, local training configuration, acceptance deadline and aggregation rule. A returned update must match the requested starting version. Combining endpoints from unrelated versions changes the algorithm. A retransmitted update must not be counted twice; retries need identifiers and an idempotent acceptance policy.

Local epochs and local steps are different budget choices. One local epoch gives larger clients more batches, even if their aggregation weight already reflects their example count. Fixed local steps give the same optimiser-step count but can expose a different fraction of each client's dataset. The code below defines four exact steps on each synthetic objective. It does not silently mix a count-weighted objective with an unspecified local training budget.

Counts themselves are data about a client. A production design should decide whether exact counts may be disclosed, whether clipping or capped weights are required, and how falsified counts are prevented. Those decisions can alter the objective. Do not claim sample weighting based on verified counts when the system merely trusts any count returned by a client.

### Separate secure aggregation from differential privacy

Secure aggregation is a protocol for computing an aggregate without exposing each participant's input in the clear to the coordinator. Its guarantees depend on the adversary model, the cohort, thresholds, cryptographic setup and dropout handling. It does not make the aggregate harmless under every situation. A cohort of one reveals that client's input; an aggregate of a few distinctive clients can still expose information.

The [practical secure aggregation paper](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/) addresses a realistic setting with expensive communication and client failures. The code here only demonstrates pairwise mask cancellation: one endpoint adds a shared mask and the other subtracts it. Summing all masked vectors cancels those masks. This is not a cryptographic implementation; all values and masks are visible to the reader.

If a client drops out, masks associated with its edges no longer cancel in the surviving sum. The example's full sum is [310,20] both before and after masking. Removing client two gives masked sum [295,0] instead of the intended surviving sum [280,5]. The discrepancy is the teaching result. A real protocol needs a safe way to recover from eligible dropouts without revealing surviving clients' individual updates.

Differential privacy addresses an information-release guarantee defined using neighbouring datasets, a privacy budget and an accounting method. Adding unspecified noise is not enough. You need a bounded contribution, an appropriate random mechanism, sampling assumptions and accounting across rounds. The adjacency can be example-level or client-level; those are different protections. This chapter does not compute an epsilon or advertise a private training run.

### Evaluate utility and participation together

The shared model needs a validation protocol consistent with the deployment population. Report pooled quality and relevant client-level distributions where disclosure is authorised. Client counts should not disappear behind one average. A large client's improvement can mask several small clients becoming worse, particularly if local distributions differ.

Hold validation records out from local optimisation. Repeated server decisions based on client validation outputs can themselves release information and overfit the evaluation. Decide what aggregates may be returned, how frequently and under which protection. A secure training aggregate does not protect separate unguarded diagnostic messages.
:::

## A real system that works this way

:::note Beyond the lecture
The original [FedAvg paper](https://proceedings.mlr.press/v54/mcmahan17a.html) studies iterative model averaging on decentralised datasets. It is the source for the local-training and weighted-aggregation mechanism. Its experimental communication reductions are results for the authors' configurations, not a multiplier applied to the toy calculation below.

The [secure aggregation work](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/) is a separate concrete protocol contribution. The distinction is operationally useful: model averaging specifies the learning update; secure aggregation specifies how certain inputs can be combined under a threat model. A deployment can use one without the other, and a complete privacy claim needs to describe both the exposed information and the protections actually implemented.

The real executable mechanism here is PyTorch CPU autograd for local optimisation plus tensor-weighted server aggregation. No devices are contacted, no sensitive records are handled and no cryptographic security is claimed. The arithmetic lab makes the server's weighting rule inspectable.
:::

## Code you can run

These independent blocks run on CPU using Python 3.14.6, NumPy 2.5.3 and PyTorch 2.14.1 where imported. Every input is synthetic. No dataset download or accelerator is needed.

### 1. FedAvg arithmetic and real local optimiser steps

The first output is the source's 0.700000. Six rounds of four local steps then produce global weight 1.650350. The central optimum for these count-weighted quadratic clients is 1.769231. The objective falls from 6.125000 to 1.061427, but the global endpoint is not the central optimum. Distinguish that optimisation experiment from the independent 0.4/0.8 aggregation example.

```python
import torch
torch.set_num_threads(1)
counts = torch.tensor([100., 300.], dtype=torch.float64)
models = torch.tensor([0.4, 0.8], dtype=torch.float64)
print('lecture FedAvg', f'{torch.sum(counts*models)/counts.sum():.6f}')
curvature = [1.0, 4.0]
targets = [-1.0, 2.0]
w = torch.tensor(0.0, dtype=torch.float64)
weights = counts/counts.sum()
optimum = sum(weights[i]*curvature[i]*targets[i] for i in range(2))/sum(weights[i]*curvature[i] for i in range(2))
def objective(value):
    return sum(weights[i]*0.5*curvature[i]*(value-targets[i])**2 for i in range(2))
initial = objective(w).item()
for round_index in range(6):
    trained = []
    for h, a in zip(curvature, targets):
        local = w.detach().clone().requires_grad_(True)
        for local_step in range(4):
            gradient, = torch.autograd.grad(0.5*h*(local-a)**2, local)
            local = (local-0.1*gradient).detach().requires_grad_(True)
        trained.append(local.detach())
    w = torch.sum(weights*torch.stack(trained))
print('trained global weight', f'{w.item():.6f}', 'central optimum', f'{optimum:.6f}')
print('objective initial', f'{initial:.6f}', 'final', f'{objective(w).item():.6f}')
```

The default lab reproduces only the lecture's single aggregation: FedAvg 0.700000 and unweighted mean 0.600000. Adjust counts while keeping both model values fixed to see the objective weighting change.

### 2. Why dropout is a protocol problem

This integer calculation proves cancellation for the complete cohort and deliberately exposes the failure after one dropout. It has no key exchange, finite-field protocol, authenticated messaging or privacy proof.

```python
import numpy as np
updates = np.array([[40, 10], [240, -5], [30, 15]], dtype=np.int64)
masks = np.array([[11, 7], [-4, 3], [19, -8]], dtype=np.int64)
masked = updates.copy()
for (i, j), mask in zip([(0, 1), (0, 2), (1, 2)], masks):
    masked[i] += mask
    masked[j] -= mask
print('unmasked sum', updates.sum(axis=0).tolist())
print('masked sum', masked.sum(axis=0).tolist())
print('drop client 2 masked sum', masked[:2].sum(axis=0).tolist())
print('drop client 2 expected sum', updates[:2].sum(axis=0).tolist())
assert np.array_equal(masked.sum(axis=0), updates.sum(axis=0))
```


<FedAvgLab />

## Designing with it

:::note Beyond the lecture
### Start with the data constraint

Federated learning is appropriate to investigate when a shared model is useful and records must remain under local control. It is not automatically the simplest route whenever a dataset is large. Compare the allowed data-sharing alternatives, the cost of client coordination and the evaluation requirements. A central dataset with appropriate governance may permit simpler exact distributed gradients; legally or operationally separate datasets may not.

State the trust boundary. Can the coordinator inspect individual updates? Can clients observe one another's messages? What collusion is considered? Can a malicious participant submit arbitrary vectors or fabricate counts? Secure aggregation can hide an individual vector while also making some anomaly inspection harder. Robustness, confidentiality and optimisation are related design concerns with different guarantees.

### Specify acceptance and clipping

Give updates a model version and round identifier. Define deadlines and the minimum accepted cohort before aggregation. If the accepted client set changes, recompute the normalising denominator and record the participating counts. Reusing the planned denominator after dropouts shrinks the update improperly. Conversely, normalising by survivors does not establish representativeness for the full population.

If updates are clipped, say whether clipping applies to a model delta, gradient, layer or whole vector, and in which norm. A count-weighted clipped update is different from clipping after weighted averaging. For a privacy mechanism, contribution sensitivity and accounting depend on this order. For optimisation alone, clipping can still be useful, but it should not be described as providing differential privacy by itself.

### Maintain an honest evaluation contract

Choose the intended population and define whether performance is sample-weighted, client-weighted or stratified. Include distributions over client outcomes where allowed, rather than only the mean. A model that meets a pooled target can be unacceptable for a client with rare classes or a different feature range. Evaluate across time because the participating cohort and local data may both change.

Avoid comparing a federated run with a central baseline that has different training examples, augmentation or optimiser budgets. A central reference is especially useful on synthetic data because the global optimum can be calculated. In a real project, the relevant central baseline might be impossible to train on combined records; say so rather than pretending that a toy pooled benchmark resolves the deployment question.

### Design recovery without double counting

Store committed round state and the list of accepted update identifiers. When an aggregation job restarts, it should either reproduce the committed round or restart a clearly identified uncommitted one. Reapplying already committed deltas gives some clients extra influence. Retaining only the global model without round bookkeeping makes recovery hard to audit.

Clients need an explicit policy for local optimiser state across rounds. Resetting, retaining or receiving server-controlled state leads to different training behaviour. The example resets its differentiable local scalar from the global model every round, with no momentum state. Extend that carefully when moving to a richer optimiser.

### Keep privacy claims within verified scope

The masking example's complete sum matching is an arithmetic check. It cannot validate secrecy because the programme deliberately contains all individual inputs and masks. A secure implementation needs established protocol components and an adversary analysis, not a larger version of this list manipulation. Likewise, the local training block is a real optimiser run but not a private run.

Treat model-update logs, debug exports and participation telemetry as part of the release surface. Protecting only the main aggregation path while exposing raw deltas in diagnostics defeats its purpose. Record which outputs are actually released and which guarantee covers each. The chapter's code prints only synthetic values and makes its missing protections explicit.
:::

## Where this stands in 2026

:::info Industry view
Sources were opened on 5 October 2026. The code runs with PyTorch 2.14.1 on CPU and needs no federated framework package. FedAvg and secure aggregation remain distinct mechanisms: local training and weighted averaging can be verified numerically, while privacy requires an additional tested protocol and an explicit release guarantee.

The [special-topics chapter](/docs/mlops/distributed/special-topics) examines objective heterogeneity and communication frequency. It illustrates why an apparently efficient federated round can leave a model biased away from the chosen central objective. No current device benchmark, privacy budget or regulatory-compliance claim is inferred from this page.
:::

## Practice questions

Exam-style questions on federated learning.

<details>
<summary><strong>Q1.</strong> What is federated learning and why use it?</summary>

Training a shared model across clients without moving their raw data , each trains locally and sends only model updates; it is privacy-preserving and suits data that cannot be centralised.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the FedAvg aggregation rule.</summary>

w_global = Σₖ (nₖ/n) wₖ , client models averaged, weighted by each client's data size.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Two clients with n=[100,300] send w=[0.4,0.8]. Compute the FedAvg global model.</summary>

(100·0.4 + 300·0.8)/400 = (40+240)/400 = 0.7.<br /><em>Sessions 13-14 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is secure aggregation?</summary>

A protocol (via MPC) letting the server learn only the sum of client updates, not any individual update , protecting privacy.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Why is non-IID client data a challenge in FL?</summary>

Clients' data distributions differ, so local updates diverge/conflict, slowing convergence and biasing the global model.<br /><em>Sessions 13-14 · conceptual</em>

</details>

:::note Answer qualifications
The source questions and answers above are retained. Read them with the corrections and assumptions in this chapter; concise source answers are not unconditional guarantees.
:::


<details>
<summary><strong>Q6.</strong> Does secure aggregation ensure a good or honest model update?</summary>

No. It protects the protocol-defined input confidentiality, subject to its assumptions. A hidden update can still be malformed or harmful; integrity and robustness need separate mechanisms.

</details>

<details>
<summary><strong>Q7.</strong> Why does normalising over participating clients not fix participation bias?</summary>

It forms the weighted mean of the clients who participated. If participation correlates with their data, that mean can differ systematically from the desired full-population update.

</details>


## Further reading

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html): FedAvg.
- [Practical Secure Aggregation](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/): threat models and dropout handling.
- [Deep Learning with Differential Privacy](https://research.google/pubs/deep-learning-with-differential-privacy/): private optimisation and accounting.
- [Local SGD on this site](/docs/mlops/distributed/advanced-sgd-techniques): endpoint averaging and drift.


## Check yourself

- I can calculate sample-weighted and client-weighted aggregates.
- I can explain why several local steps differ from one central gradient step.
- I can identify the information exposed by a federated round.
- I can distinguish mask cancellation, secure aggregation and differential privacy.
- I can explain why participation and retry policies change the effective objective.
