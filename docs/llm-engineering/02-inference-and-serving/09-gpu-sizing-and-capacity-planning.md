---
id: llme-capacity
title: "GPU Sizing and Capacity Planning"
sidebar_label: "9 · GPU sizing and capacity"
sidebar_position: 9
slug: /llm-engineering/gpu-sizing-and-capacity-planning
description: "How many GPUs does an LLM service need? A memory budget for weights and KV cache, a bandwidth-bound decode estimate, a compute-bound prefill estimate, Little's law for replicas, and cost per million tokens as a formula with named parameters instead of invented prices."
tags: [capacity-planning, gpu-sizing, kv-cache, littles-law, throughput, cost-per-token, slo, inference]
---

import Infographic from '@site/src/components/Infographic';
import CapacityPlannerLab from '@site/src/components/viz/CapacityPlannerLab';

**In one line.** Sizing a GPU fleet is three calculations in a fixed order: does the model plus its KV cache fit in memory, how fast can one replica produce tokens within your latency target, and how many replicas does Little's law say your traffic needs, with the estimate resting on a few named assumptions that you must replace with measurements.

:::note Not from a lecture
This chapter is written for this site from NVIDIA's public product pages, the Hugging Face model configuration files and a standard reference on Little's law, all opened on 2 October 2026. Hardware figures are quoted only as fetched, with the date; every price is left as a parameter. The planner is a first estimate, not a substitute for a load test.
:::

## The idea in plain words

A GPU has two scarce things that matter for serving a language model: **memory capacity** (how much it can hold) and **memory bandwidth** (how fast it can read what it holds). The [memory-bound decoding](/docs/llm-engineering/why-decoding-is-memory-bound) chapter showed that every decode step reads all the weights once, plus the KV cache of every request in the batch. So:

1. **Capacity decides whether it fits and how many requests can be in flight.** Weights take a fixed amount; what remains is the KV cache pool, and each request in flight uses a share of it (see [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention)).
2. **Bandwidth decides how fast each step runs,** and so the time per output token (TPOT) your users see. More requests in the batch means more KV cache read per step, so a latency target caps the batch.
3. **Little's law decides how many replicas.** If requests arrive at a rate and each stays in the system for some time, the average number in flight is their product, and each replica can hold only so many.

Prefill is different: it is compute-heavy, processes the whole prompt at once, and competes with decode for the same GPU. Long prompts shift the binding constraint from decode to prefill, which the worked example shows.

<Infographic src="/img/llme/gpu-sizing-and-capacity-planning-formula.svg" alt="The chain of capacity calculations: memory pool minus weights gives the KV budget, bandwidth gives step time and batch under the latency target, Little's law gives replicas, and replicas times GPUs gives cost per million tokens." caption="The planner as a chain, with the worked example's numbers: an 8B model on one H100 SXM has 53.9 GB of KV budget, 316 sequences by memory, a 31.7 ms step, one decode replica and three prefill replicas." />

<Infographic src="/img/llme/gpu-sizing-and-capacity-planning-example.svg" alt="A table of the worked example across an 8B model on four GPUs and a 70B model at tensor-parallel 2, 4 and 8, showing KV budget, sequences, step time, replicas and GPUs." caption="The worked example from the code: 20 requests per second, 1,000 prompt and 300 output tokens, 40 ms TPOT target. The L4 cannot reach the target and 70B does not fit on two H100s." />

## How it works

### 1. The memory budget

For a model with *P* parameters stored at *b* bytes each, the weights occupy *P* times *b* bytes. A 16-bit model has *b* = 2.

The KV cache stores, for every token of every request in flight, a key and a value vector at each layer:

$$\text{KV bytes per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dimension} \times \text{bytes per value}.$$

The "2" is key plus value. The number of KV heads is where grouped-query attention pays off: the Llama 3.1 8B and 70B configuration files both have 8 KV heads and a head dimension of 128, against 32 and 64 attention heads. The code reads those files from the Hub, so the numbers are not typed in by hand.

The memory pool a GPU offers is its capacity times a utilisation fraction, minus a reserve per GPU for activations, CUDA context and fragmentation. The **KV budget** is the pool minus the weights. Dividing by the KV bytes of one request at its full length (prompt plus output) gives the number of sequences that fit. The 90 per cent utilisation and the 2 GB reserve in the code are **assumptions**, not values taken from any engine's documentation.

### 2. The decode step and the latency target

If one decode step reads the weights plus the KV cache of *B* requests with average context *c*, and the GPU delivers an effective bandwidth *β*, then the step time is

$$t_{\text{step}}(B) = \frac{W + B \cdot c \cdot k}{\beta},$$

where *W* is the weight bytes and *k* the KV bytes per token. That step time is the TPOT. Raising *B* spreads the weight read over more requests (good for throughput) but lengthens each step (bad for TPOT). The latency target therefore gives a **maximum batch**: the largest *B* with step time at or below the target. The effective bandwidth is the datasheet bandwidth times an efficiency, 0.6 in the code. That efficiency is an **assumption**; real engines reach different fractions on different models, and you should replace it with a measurement.

### 3. Prefill

Prefill processes the prompt in parallel and is limited by compute, roughly 2 floating-point operations per parameter per token. With dense throughput *F* and a model-FLOPs utilisation *u*, a replica prefills about *F u / (2P)* tokens per second. The code takes *u* = 0.4, again an assumption. A replica must also decode, so the code lets prefill use only a share of each replica's time, 0.3 by default. Prefill replicas needed are the prompt tokens arriving per second divided by the usable prefill rate.

### 4. Little's law

Little's law states that the long-term average number of items in a system equals the long-term average arrival rate times the average time each item spends in the system: *L* = *λ W*. It holds without assumptions about the arrival or service distributions, as long as the system is stationary; John Little published the proof in *Operations Research* in 1961, after the relation had been used without proof (Morse stated it in 1958). Here *λ* is requests per second and *W* is the time a request spends in the system: its time to first token plus the output tokens times the step time. If *L* requests are in flight and a replica holds at most *B*, you need ceil(*L*/*B*) replicas for decode. The planner takes the larger of that and the prefill replica count.

### 5. Cost per million tokens

Cost has a simple form once the fleet is chosen. With *G* GPUs running at the target load, serving *Q* requests per second of *n* output tokens each:

$$\text{GPU-hours per million output tokens} = \frac{G}{Q \cdot n \cdot 3600} \times 10^{6}.$$

Multiply by your **price per GPU-hour** to get money per million tokens. No price is built into the code, because cloud and purchase prices change and differ by contract. Because the formula assumes the fleet runs at the target load all day, the real cost per token is higher whenever the fleet sits idle.

## A real system that works this way

The honest "real systems" here are the inputs. **NVIDIA's product pages** are where the hardware numbers come from, and they carry a trap: the Tensor Core figures are quoted *with sparsity*. The L4 page says "specifications are one-half lower without sparsity", and the A100 page lists 312 teraFLOPS dense and 624 with sparsity for 16-bit. The H100 page marks its figures "with sparsity" (1,979 teraFLOPS for 16-bit on the SXM card), so the dense figure used here, 989.5, is **derived** by the same halving, not quoted. Use a headline number without noticing the footnote and your prefill estimate is out by a factor of two. The **model configuration files** on the Hugging Face Hub are the other input: layer count, KV heads and head dimension come straight from the files the model is published with, and the parameter count comes from instantiating the architecture on PyTorch's meta device, which allocates no memory.

Datasheet figures used (fetched 2 October 2026):

| GPU | Memory | Bandwidth | 16-bit Tensor Core, dense |
| --- | --- | --- | --- |
| L4 | 24 GB | 300 GB/s | 121 teraFLOPS (242 with sparsity, halved per the page) |
| A100 80GB SXM | 80 GB | 2,039 GB/s | 312 teraFLOPS (quoted) |
| H100 SXM | 80 GB | 3.35 TB/s | 989.5 teraFLOPS (derived from 1,979 with sparsity) |
| H100 NVL | 94 GB | 3.9 TB/s | 835.5 teraFLOPS (derived from 1,671 with sparsity) |

## Code you can run

The block below is the whole planner. It reads real model configurations (cached from the Hub), counts parameters on the meta device, plans several cases, and then varies the assumptions that matter. Every assumption is a named field of `Assumptions` or `Workload`.

```python
import math
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM

GPUS = {
    "L4 24GB": {"memory_gb": 24, "bandwidth_gbs": 300, "dense_tflops": 121.0},
    "A100 80GB SXM": {"memory_gb": 80, "bandwidth_gbs": 2039, "dense_tflops": 312.0},
    "H100 SXM 80GB": {"memory_gb": 80, "bandwidth_gbs": 3350, "dense_tflops": 989.5},
    "H100 NVL 94GB": {"memory_gb": 94, "bandwidth_gbs": 3900, "dense_tflops": 835.5},
}


@dataclass
class Workload:
    qps: float = 20.0
    prompt_tokens: int = 1000
    output_tokens: int = 300
    tpot_slo_ms: float = 40.0


@dataclass
class Assumptions:
    memory_utilisation: float = 0.90
    reserve_gb_per_gpu: float = 2.0
    bandwidth_efficiency: float = 0.6
    prefill_mfu: float = 0.4
    prefill_share: float = 0.3
    weight_bytes: float = 2.0
    kv_bytes: float = 2.0


def model_shape(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        params = sum(p.numel() for p in AutoModelForCausalLM.from_config(cfg).parameters())
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return params, cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim


def plan(model_id, gpu, tp, work, a):
    params, layers, kv_heads, head_dim = model_shape(model_id)
    spec = GPUS[gpu]
    weights = params * a.weight_bytes
    kv_per_token = 2 * layers * kv_heads * head_dim * a.kv_bytes
    pool = tp * (spec["memory_gb"] * a.memory_utilisation - a.reserve_gb_per_gpu) * 1e9
    kv_budget = pool - weights
    if kv_budget <= 0:
        return "weights do not fit"
    ctx = work.prompt_tokens + work.output_tokens / 2
    seqs_memory = int(kv_budget // (kv_per_token * (work.prompt_tokens + work.output_tokens)))
    bw = tp * spec["bandwidth_gbs"] * 1e9 * a.bandwidth_efficiency

    def step(b):
        return (weights + b * ctx * kv_per_token) / bw

    seqs_slo = 0
    while seqs_slo < seqs_memory and step(seqs_slo + 1) <= work.tpot_slo_ms / 1000:
        seqs_slo += 1
    batch = min(seqs_memory, seqs_slo)
    if batch == 0:
        return "TPOT SLO unreachable"
    prefill_rate = tp * spec["dense_tflops"] * 1e12 * a.prefill_mfu / (2 * params)
    ttft = work.prompt_tokens / prefill_rate
    latency = ttft + work.output_tokens * step(batch)
    in_flight = work.qps * latency
    for_decode = math.ceil(in_flight / batch)
    for_prefill = math.ceil(work.qps * work.prompt_tokens / (prefill_rate * a.prefill_share))
    replicas = max(for_decode, for_prefill)
    gpu_hours_per_mtok = replicas * tp / (work.qps * work.output_tokens * 3600) * 1e6
    return dict(params=params, pool_gb=pool / 1e9, weights_gb=weights / 1e9, kv_kb=kv_per_token / 1e3, kv_gb=kv_budget / 1e9,
                seqs_memory=seqs_memory, seqs_slo=seqs_slo, batch=batch, step_ms=step(batch) * 1000,
                latency=latency, ttft=ttft, prefill_rate=prefill_rate, in_flight=in_flight, for_decode=for_decode, for_prefill=for_prefill, replicas=replicas, gpus=replicas * tp,
                gpu_hours_per_mtok=gpu_hours_per_mtok)


work, assume = Workload(), Assumptions()
print(f"workload: {work.qps:.0f} QPS, {work.prompt_tokens} prompt + {work.output_tokens} output tokens, TPOT SLO {work.tpot_slo_ms:.0f} ms")
cases = [
    ("NousResearch/Meta-Llama-3.1-8B", "L4 24GB", 1),
    ("NousResearch/Meta-Llama-3.1-8B", "A100 80GB SXM", 1),
    ("NousResearch/Meta-Llama-3.1-8B", "H100 SXM 80GB", 1),
    ("NousResearch/Meta-Llama-3.1-8B", "H100 NVL 94GB", 1),
    ("NousResearch/Meta-Llama-3.1-70B", "H100 SXM 80GB", 2),
    ("NousResearch/Meta-Llama-3.1-70B", "H100 SXM 80GB", 4),
    ("NousResearch/Meta-Llama-3.1-70B", "H100 SXM 80GB", 8),
]
print("model         gpu             tp  weights GB  KV kB/tok  KV GB  seqs(mem)  seqs(SLO)  step ms  TTFT s  in flight  decode  prefill  GPUs  GPU-h per M tokens")
for model_id, gpu, tp in cases:
    r = plan(model_id, gpu, tp, work, assume)
    name = model_id.split("-")[-1]
    if isinstance(r, str):
        print(f"{name:<13} {gpu:<15} {tp:>2}  {r}")
        continue
    print(f"{name:<13} {gpu:<15} {tp:>2}  {r['weights_gb']:>10.1f}  {r['kv_kb']:>9.1f}  {r['kv_gb']:>5.1f}  {r['seqs_memory']:>9}  {r['seqs_slo']:>9}  "
          f"{r['step_ms']:>7.1f}  {r['ttft']:>6.2f}  {r['in_flight']:>9.1f}  {r['for_decode']:>6}  {r['for_prefill']:>7}  {r['gpus']:>4}  {r['gpu_hours_per_mtok']:>18.3f}")

model8 = "NousResearch/Meta-Llama-3.1-8B"
print("8B on H100 SXM: which side binds as the prompt gets longer (decode replicas / prefill replicas)")
for prompt in (200, 1000, 4000):
    r = plan(model8, "H100 SXM 80GB", 1, Workload(prompt_tokens=prompt), assume)
    print(f"  prompt {prompt:>4} tokens: seqs by memory {r['seqs_memory']:>4}, decode {r['for_decode']}, prefill {r['for_prefill']}, GPUs {r['gpus']}")
print("8B on H100 SXM: sensitivity to the two assumptions that move the answer")
for share, mfu in ((0.3, 0.4), (0.5, 0.4), (0.3, 0.25), (0.3, 0.55)):
    r = plan(model8, "H100 SXM 80GB", 1, work, Assumptions(prefill_share=share, prefill_mfu=mfu))
    print(f"  prefill share {share}, prefill MFU {mfu}: GPUs {r['gpus']}")
print("8B on H100 SXM: the TPOT SLO")
for slo in (10, 15, 25, 40):
    r = plan(model8, "H100 SXM 80GB", 1, Workload(tpot_slo_ms=slo), assume)
    print(f"  SLO {slo} ms: " + (r if isinstance(r, str) else f"batch {r['batch']}, decode replicas {r['for_decode']}, GPUs {r['gpus']}"))

params8, _, _, _ = model_shape(model8)
for gpu in ("L4 24GB", "H100 SXM 80GB"):
    spec = GPUS[gpu]
    pool = (spec["memory_gb"] * assume.memory_utilisation - assume.reserve_gb_per_gpu)
    floor_ms = params8 * assume.weight_bytes / (spec["bandwidth_gbs"] * 1e9 * assume.bandwidth_efficiency) * 1000
    print(f"8B on {gpu}: pool {pool:.1f} GB, KV budget {pool - params8 * assume.weight_bytes / 1e9:.1f} GB, weights alone take {floor_ms:.1f} ms per decode step")
r = plan(model8, "H100 SXM 80GB", 1, work, assume)
print(f"8B on H100 SXM: request time in the system {r['latency']:.2f} s, requests in flight {r['in_flight']:.1f}, prefill rate {r['prefill_rate']:.0f} tokens/s per replica")
```

Reading the main table (workload: 20 QPS, 1,000 prompt and 300 output tokens, 40 ms TPOT target; assumptions: 90 per cent memory use, 2 GB reserve per GPU, 0.6 bandwidth efficiency, prefill MFU 0.4, prefill share 0.3).

| Case | Weights GB | KV kB per token | KV budget GB | Sequences by memory | Sequences within SLO | Step ms | GPUs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B on L4 | 16.1 | 131.1 | 3.5 | n/a | none | 89.2 for the weights alone | TPOT target unreachable |
| 8B on A100 80GB | 16.1 | 131.1 | 53.9 | 316 | 218 | 40.0 | 9 |
| 8B on H100 SXM | 16.1 | 131.1 | 53.9 | 316 | 316 | 31.7 | 3 |
| 8B on H100 NVL | 16.1 | 131.1 | 66.5 | 390 | 390 | 32.0 | 4 |
| 70B on 2 H100 | 141.1 | 327.7 | n/a | n/a | n/a | n/a | weights do not fit |
| 70B on 4 H100 | 141.1 | 327.7 | 138.9 | 326 | 326 | 32.8 | 24 |
| 70B on 8 H100 | 141.1 | 327.7 | 418.9 | 983 | 983 | 31.8 | 24 |

Three things are worth seeing.

**The L4 fits the weights and still fails.** Its pool is 19.6 GB, which holds the 16.1 GB of weights and leaves a KV budget of only 3.5 GB, but merely reading the weights takes 89.2 ms per decode step at the assumed effective bandwidth, over twice the 40 ms target, so no batch size meets it. Capacity said yes; bandwidth said no.

**The A100 is limited by the latency target, not by memory.** It could hold 316 sequences, but only 218 fit inside 40 ms. Its step time at 218 is exactly at the limit. The H100 SXM, with more bandwidth, reaches the memory limit of 316 inside 31.7 ms.

**Prefill, not decode, set the GPU count.** On the H100 SXM, the 70.0 GB pool leaves 53.9 GB of KV budget, a request spends 9.55 seconds in the system (0.04 s of prefill plus 300 steps of 31.7 ms), so Little's law says 190.9 requests are in flight (20 per second times 9.55), which one replica holds (316 slots), so decode needs 1 replica; prefill, at the derived 24,644 tokens per second per replica and a 30 per cent share, needs 3, so the plan is 3 GPUs and **0.139 GPU-hours per million output tokens**. The A100's prefill rate is lower (312 against 989.5 dense teraFLOPS), so it needs 9 replicas and 0.417. The 70B model needs four H100s per replica before there is any KV room (two cannot hold 141.1 GB of weights in a 140 GB pool), and at four or eight GPUs per replica the fleet size is the same 24 GPUs with 1.111 GPU-hours per million tokens in this simple model, which ignores the communication cost of tensor parallelism. That omission is a real limitation: add measured overhead before relying on the 70B rows.

The sensitivity prints show where to measure first. For the 8B model on an H100 SXM:

| Change | Result |
| --- | --- |
| Prompt 200 tokens | 823 sequences by memory, 1 GPU |
| Prompt 1,000 tokens | 316 sequences, 3 GPUs (decode 1, prefill 3) |
| Prompt 4,000 tokens | 95 sequences, 11 GPUs (decode 3, prefill 11) |
| Prefill share 0.3, MFU 0.4 | 3 GPUs |
| Prefill share 0.5, MFU 0.4 | 2 GPUs |
| Prefill share 0.3, MFU 0.25 | 5 GPUs |
| Prefill share 0.3, MFU 0.55 | 2 GPUs |
| TPOT target 10 ms | batch 26, decode needs 3 replicas |
| TPOT target 15 ms | batch 93, decode needs 1 replica |

A fivefold longer prompt (200 to 1,000 tokens) turned a one-GPU plan into a three-GPU plan, and 4,000-token prompts needed eleven. A tight 10 ms target cut the batch to 26 and made decode need 3 replicas on its own. Moving the two prefill assumptions alone swung the answer from 2 to 5 GPUs. So **measure prefill throughput and decode step time on your own model first**: those are the numbers that move the answer.

<CapacityPlannerLab />

The lab's defaults (8B, H100 SXM, tensor parallel 1, 20 QPS, 1,000 and 300 tokens, 40 ms, efficiency 0.6, MFU 0.4, share 0.3) reproduce the H100 row above: 316 sequences by memory, a 31.7 ms step, 1 decode replica, 3 prefill replicas, 3 GPUs and 0.139 GPU-hours per million output tokens.

## Designing with it

1. **Size for the peak, not the mean.** The planner assumes steady load. Replace *λ* with your peak arrival rate over a window as long as a request, and add headroom for a replica failure (N+1).
2. **Measure, then replace the assumptions.** Run the benchmark client from the [serving engines](/docs/llm-engineering/serving-engines) chapter against your engine to get real prefill throughput and step time, then put those into the planner.
3. **Treat prompt length as a first-class input.** The worked example moved from one GPU to eleven as prompts grew. If retrieval stuffs long contexts into prompts, prefix caching ([caching, routing and cost](/docs/llm-engineering/semantic-caching-routing-and-cost)) and chunked prefill ([continuous batching](/docs/llm-engineering/continuous-batching-and-scheduling)) change the prefill side.
4. **Lower the weight bytes before adding GPUs.** Quantised weights shrink *W* and raise the memory left for KV; see [quantisation for inference](/docs/llm-engineering/quantisation-for-inference).
5. **Pick the latency target from users, not from the hardware.** A 10 ms target made the same GPU need three decode replicas.
6. **Keep price as a parameter.** Multiply GPU-hours per million tokens by the price you actually pay, and compare it with hosted per-token prices from the same date.
7. **Re-fetch datasheets and note the sparsity footnote** before copying a TFLOPS figure.

## Where this stands in 2026

:::info Industry view
GPU memory and bandwidth have grown, and the fetched figures show the spread in a single product line: an L4 at 300 GB/s, an A100 80GB SXM at 2,039 GB/s and an H100 SXM at 3.35 TB/s. The calculation did not change; what changes is which of the three constraints binds. Short prompts and small models are decode-bound and bandwidth-limited; long prompts shift cost to prefill compute; very large models are capacity-bound first. Quote vendor figures with their date and footnotes, and expect your own measurements to land below the datasheet.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> An 8B model in 16-bit has 16.1 GB of weights and the GPU has 24 GB. Why can the plan still fail?</summary>

Fitting is necessary, not sufficient. The L4's low bandwidth (300 GB/s) makes each decode step take about 89 ms at the assumed efficiency, which exceeds a 40 ms TPOT target for any batch size. Also, only about 3.5 GB would remain for KV cache.

</details>

<details>
<summary><strong>Q2.</strong> Compute the KV bytes per token for the Llama 3.1 8B configuration.</summary>

2 x 32 layers x 8 KV heads x 128 head dimension x 2 bytes = 131,072 bytes, the 131.1 kB the code prints.

</details>

<details>
<summary><strong>Q3.</strong> Twenty requests arrive per second and each spends 9.55 seconds in the system. How many are in flight, and what does that mean for replicas?</summary>

By Little's law, 20 x 9.55 = 191 requests in flight (the code prints 190.9). If a replica holds 316 sequences, one decode replica suffices.

</details>

<details>
<summary><strong>Q4.</strong> Why did the H100 plan need three GPUs when decode needed only one replica?</summary>

Prefill is compute-bound and the planner lets it use only 30 per cent of a replica's time. At 20 requests per second of 1,000-token prompts, prefill alone needed 3 replicas. The binding side was prefill.

</details>

<details>
<summary><strong>Q5.</strong> A vendor page lists 1,979 teraFLOPS for 16-bit on an H100 SXM. What do you check before using it for prefill?</summary>

Whether the figure is with sparsity. The page marks it so, so the dense figure is half, 989.5, by the same convention the L4 page states explicitly. Using the headline number would double the estimated prefill rate.

</details>

<details>
<summary><strong>Q6.</strong> Which two inputs would you measure first to tighten the estimate?</summary>

Prefill throughput (tokens per second per replica at your prompt lengths) and the decode step time at your batch size, because the sensitivity runs show those assumptions moving the GPU count from 2 to 5.

</details>

<details>
<summary><strong>Q7.</strong> Why is the cost formula an optimistic bound?</summary>

It assumes the fleet runs at the target load all day. Idle capacity, peak-versus-mean traffic, headroom for failures and tensor-parallel communication are not in it, so the real cost per token is higher.

</details>

## Further reading

All opened on 2 October 2026.

- NVIDIA product pages for the [H100](https://www.nvidia.com/en-us/data-center/h100/), [A100](https://www.nvidia.com/en-us/data-center/a100/) and [L4](https://www.nvidia.com/en-us/data-center/l4/) (specification tables and their sparsity footnotes).
- Little's law, [reference article](https://en.wikipedia.org/wiki/Little%27s_law), citing J. D. C. Little, "A Proof for the Queuing Formula: L = λW", *Operations Research* 9(3), 1961.
- Model configuration files for Llama 3.1 8B and 70B on the Hugging Face Hub (read through `AutoConfig`).
- On this site: [memory-bound decoding](/docs/llm-engineering/why-decoding-is-memory-bound), [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention), [continuous batching](/docs/llm-engineering/continuous-batching-and-scheduling), [quantisation](/docs/llm-engineering/quantisation-for-inference), [serving engines](/docs/llm-engineering/serving-engines) and [caching, routing and cost](/docs/llm-engineering/semantic-caching-routing-and-cost).

## Check yourself

- I can compute weights and KV-cache memory for a model from its configuration and say how many sequences fit.
- I can explain why a latency target caps the batch size through the decode step time.
- I can estimate prefill throughput from datasheet TFLOPS, and say why the sparsity footnote matters.
- I can use Little's law to turn an arrival rate and a request duration into requests in flight and replicas.
- I can write cost per million tokens as a formula with a price parameter and say what it leaves out.
- I can name the assumptions in my own plan and the measurement that would replace each one.
