---
id: llme-batching
title: "Continuous Batching and Scheduling"
sidebar_label: "3 · Continuous batching and scheduling"
sidebar_position: 3
slug: /llm-engineering/continuous-batching-and-scheduling
description: "How a serving engine decides what to run at every decode step: static against continuous batching, chunked prefill, scheduling policies, queueing and prefix caching, with a seeded simulator and a real continuous-batching run."
tags: [continuous-batching, chunked-prefill, scheduling, prefix-caching, queueing, vllm, orca, inference]
---

import Infographic from '@site/src/components/Infographic';
import BatchingLab from '@site/src/components/viz/BatchingLab';

**In one line.** A serving engine decides at every decode step which requests to run, and letting new requests join the running batch the moment a slot frees, instead of waiting for a whole batch to finish, turns the idle compute of chapter 1 into throughput without wrecking latency.

:::note Not from a lecture
This chapter was written for this site from the sources under Further reading. The simulator uses the roofline numbers of chapter 1 for its step times, so its results are illustrative of the mechanism and are not measurements of any serving engine.
:::

## The idea in plain words

Chapter 1 ended on a promise: a decode step costs the same whether it carries 1 sequence or 64, so batching is nearly free. This chapter is about how to *form* the batch when requests arrive at random times and want different numbers of tokens.

**Static batching** collects a group of requests, runs them together, and returns the answers only when every one is finished. It is a minibus that leaves when full and cannot come back until its farthest passenger is dropped off. A request that wants 20 tokens sits idle while a neighbour generates 1,000, and a request that arrives just after departure waits for the whole trip.

**Continuous batching**, also called in-flight or iteration-level batching, is a tram. Every step is a stop: finished requests step off and are answered at once, waiting requests step on, and nobody is held for the slowest member.

That freedom brings decisions. A new prompt is a big job, and a big prefill inside a step delays every other user's next token. Waiting requests need an order. The cache can run out. And load follows queueing mathematics, so latency is flat until the system nears capacity and then explodes.

<Infographic src="/img/llme/continuous-batching-and-scheduling-static-vs-continuous.svg" alt="Four slots over time under static batching, with idle gaps until the longest request finishes, beside the same slots under continuous batching refilled at once, and a table of throughput, latency and time to first token for both on a simulated trace." caption="Static against continuous batching: a schematic, then the simulated trace. The table is block 1." />

<Infographic src="/img/llme/continuous-batching-and-scheduling-scheduling.svg" alt="Tables for the arrival-rate sweep, four scheduling policies and the prefix cache hit rate, with a picture of a long prefill stalling decodes and chunked prefill splitting it." caption="Queueing, chunked prefill, policies and prefix caching. The first two tables are block 1 and the cache table is block 2." />

## How it works

### Why static batching wastes the GPU

A static batch runs as long as its longest request. Shorter requests keep their slots, the engine computes padding, and answers are held back until the end. The Orca paper, which introduced the alternative, describes letting early finishers return to the client without waiting for the rest.

### Iteration-level scheduling

Orca runs the scheduler between iterations rather than between requests: after each step it removes finished requests and admits waiting ones. Requests in one step sit at different positions, so attention cannot be one rectangular tensor. The paper's answer is **selective batching**: batch only the operations that allow it, such as the large matrix multiplications, and handle the rest per request. Its abstract reports 36.9 times the throughput of NVIDIA FasterTransformer at the same latency on GPT-3 175B.

### What goes into a step

Every request already decoding contributes one token, and new requests contribute their prompt tokens, because prefill is the same forward pass with more tokens. Step time follows chapter 1: the larger of the weight-reading time and the arithmetic for all the tokens, plus the time to read each running sequence's cache. A mostly-decode step costs about the weight-reading time. A step that swallows a 2,048-token prompt costs 33.24 ms of arithmetic, around seven ordinary steps, during which every other user receives no token.

### Chunked prefill

**Chunked prefill** fixes that stall by cutting a long prompt into pieces and giving each step only as many prompt tokens as a **token budget** allows after the decodes are counted. The Sarathi-Serve paper calls the result stall-free scheduling, and reports capacity gains over vLLM under tail-latency limits of 2.6 times for Mistral-7B on one A100, 3.7 times for Yi-34B on two, and 5.6 times for Falcon-180B with pipeline parallelism. vLLM's V1 documentation states that chunked prefill is enabled by default where possible and that its scheduler batches all pending decode requests before any prefill. The knob is `max_num_batched_tokens`: the documentation says lower values, 2,048 as its example, give better inter-token latency because fewer prefills slow the decodes, and higher values give better time to first token.

### Order, preemption and memory

**First come, first served** is fair and simple. **Shortest prompt first** lowers the average wait but can starve long prompts. Hugging Face's continuous batching offers two scheduler types, `fifo` and `prefill_first`, and a `safety_margin` that stops admitting new prefills when free cache blocks run low so running requests can finish. When the cache runs out an engine must **preempt**, evicting a request's blocks to recompute or reload later. vLLM's tuning page says preemption warnings mean there is not enough KV cache space and suggests lowering `max_num_seqs` or `max_num_batched_tokens`. Here the allocator of the previous chapter meets the scheduler.

### Queueing

Arrivals are random, so even a system with spare capacity sometimes queues, and near capacity the queue grows without bound. **Little's law** says the average number of requests in the system equals the arrival rate times the average time each spends there, which tells you the slots and cache a given rate needs. The simulator below checks it.

### Prefix caching

Many requests share a system prompt or document. If cache blocks are keyed by a hash of the tokens so far, a new request can reuse an earlier one's blocks and skip that part of prefill. vLLM's documentation describes automatic prefix caching and its limit: it only shortens prefill, not the generation of new tokens. Hugging Face's continuous batching shares blocks by default. The key is a chain, each block's hash covering everything before it, so one changed token near the start invalidates every later block.

## A real system that works this way

**Orca** and **Sarathi-Serve** are the research systems. **vLLM** combines paged cache blocks, a decode-first scheduler, chunked prefill and automatic prefix caching, as its documentation quoted above describes. **Hugging Face Transformers** also ships continuous batching: `generate_batch` takes tokenised prompts and schedules them internally, and a `ContinuousBatchingManager` accepts requests as they arrive. Block 3 runs it on CPU.

## Code you can run

Three blocks. Block 1 is the simulator and runs in about a second. Block 2 is a prefix cache. Block 3 is the real Transformers scheduler on SmolLM2-135M and takes under a minute after the download.

### 1. A discrete-event simulator for three batching policies

A seeded trace of 600 requests with random arrivals and lognormal prompt lengths (clipped to 8 to 2,048 tokens) and output lengths (clipped to 8 to 1,024). Step time is chapter 1's roofline for Llama 3.1 8B on the quoted H100 figures: 4.79 ms to read the weights, 0.01623 ms of arithmetic per token in the step and 3.913e-5 ms per cached token read. These constants are illustrative, not measured. `run_static` runs each batch to its longest request. `run_continuous` admits requests every step, optionally with a token budget for chunked prefill and optionally shortest prompt first.

```python
import math

WEIGHT_MS = 4.79
COMPUTE_MS_PER_TOKEN = 0.01623
KV_MS_PER_TOKEN = 3.913e-5


def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]

    def draw():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return draw


def make_trace(n, rate_per_s, seed=7):
    draw = mulberry32(seed)

    def normal():
        return math.sqrt(-2 * math.log(1 - draw())) * math.cos(2 * math.pi * draw())

    clock, trace = 0.0, []
    for i in range(n):
        clock += -math.log(1 - draw()) / rate_per_s * 1000
        prompt = min(2048, max(8, round(math.exp(5.2 + 0.8 * normal()))))
        output = min(1024, max(8, round(math.exp(4.8 + 0.9 * normal()))))
        trace.append({"id": i, "arrival": clock, "prompt": prompt, "output": output})
    return trace


def step_ms(batch_tokens, context_tokens):
    return max(WEIGHT_MS, COMPUTE_MS_PER_TOKEN * batch_tokens) + KV_MS_PER_TOKEN * context_tokens


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)]


def summarise(trace, first_token, finish, gaps, makespan):
    latency = [finish[r["id"]] - r["arrival"] for r in trace]
    ttft = [first_token[r["id"]] - r["arrival"] for r in trace]
    tokens = sum(r["output"] for r in trace)
    return {
        "tokens_per_s": tokens / (makespan / 1000),
        "latency_mean_s": sum(latency) / len(latency) / 1000,
        "latency_p99_s": percentile(latency, 0.99) / 1000,
        "ttft_mean_ms": sum(ttft) / len(ttft),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "gap_p99_ms": percentile(gaps, 0.99),
        "makespan_s": makespan / 1000,
    }


def run_static(trace, max_batch):
    t, i = 0.0, 0
    first_token, finish, gaps = {}, {}, []
    while i < len(trace):
        t = max(t, trace[i]["arrival"])
        batch = []
        while i < len(trace) and len(batch) < max_batch and trace[i]["arrival"] <= t:
            batch.append(trace[i])
            i += 1
        t += step_ms(sum(r["prompt"] for r in batch), 0)
        for r in batch:
            first_token[r["id"]] = t
        longest = max(r["output"] for r in batch)
        for k in range(1, longest):
            context = sum(r["prompt"] + min(k, r["output"]) for r in batch)
            d = step_ms(len(batch), context)
            t += d
            gaps.extend([d] * sum(1 for r in batch if r["output"] > k))
        for r in batch:
            finish[r["id"]] = t
    return summarise(trace, first_token, finish, gaps, t)


def run_continuous(trace, max_batch, token_budget=None, shortest_first=False):
    t, i = 0.0, 0
    waiting, running = [], []
    first_token, finish, gaps = {}, {}, []
    done = 0
    area = 0.0
    while done < len(trace):
        while i < len(trace) and trace[i]["arrival"] <= t:
            waiting.append(trace[i])
            i += 1
        if not running and not waiting:
            t = trace[i]["arrival"]
            continue
        if shortest_first:
            waiting.sort(key=lambda r: (r["prompt"], r["id"]))
        while waiting and len(running) < max_batch:
            r = waiting.pop(0)
            running.append({"req": r, "prefilled": 0, "generated": 0})
        decoding = [s for s in running if s["prefilled"] == s["req"]["prompt"]]
        budget = None if token_budget is None else max(0, token_budget - len(decoding))
        chunk_tokens, chunks = 0, []
        for s in running:
            need = s["req"]["prompt"] - s["prefilled"]
            if need == 0:
                continue
            take = need if budget is None else min(need, budget - chunk_tokens)
            if take <= 0:
                break
            chunks.append((s, take))
            chunk_tokens += take
        context = sum(s["req"]["prompt"] + s["generated"] for s in decoding)
        d = step_ms(len(decoding) + chunk_tokens, context)
        area += (len(running) + len(waiting)) * d
        t += d
        gaps.extend([d] * len(decoding))
        for s in decoding:
            s["generated"] += 1
        for s, take in chunks:
            s["prefilled"] += take
            if s["prefilled"] == s["req"]["prompt"]:
                s["generated"] = 1
                first_token[s["req"]["id"]] = t
        for s in list(running):
            if s["generated"] >= s["req"]["output"]:
                finish[s["req"]["id"]] = t
                running.remove(s)
                done += 1
    result = summarise(trace, first_token, finish, gaps, t)
    result["in_system"] = area / t
    return result


def row(name, r):
    return (f"{name:22s} {r['tokens_per_s']:8.0f} {r['latency_mean_s']:8.2f} {r['latency_p99_s']:8.2f} "
            f"{r['ttft_mean_ms']:10.1f} {r['ttft_p99_ms']:10.1f} {r['gap_p99_ms']:8.2f}")


HEADER = f"{'':22s} {'tok/s':>8s} {'lat_mean':>8s} {'lat_p99':>8s} {'ttft_mean':>10s} {'ttft_p99':>10s} {'gap_p99':>8s}   (latency in s, others in ms)"

for rate in [5, 30]:
    trace = make_trace(600, rate)
    print(f"600 requests at {rate} per second, mean prompt {sum(r['prompt'] for r in trace) / 600:.0f}, "
          f"mean output {sum(r['output'] for r in trace) / 600:.0f}")
    print(HEADER)
    print(row("static, batch 16", run_static(trace, 16)))
    print(row("static, batch 64", run_static(trace, 64)))
    print(row("continuous, batch 64", run_continuous(trace, 64)))
    print()

print("continuous batching as the arrival rate rises (max batch 64)")
print("req/s   tok/s   ttft_mean  ttft_p99   in_system   rate x mean latency")
for rate in [10, 30, 50, 60, 70]:
    trace = make_trace(600, rate)
    r = run_continuous(trace, 64)
    served = 600 / r["makespan_s"]
    print(f"{rate:5d} {r['tokens_per_s']:7.0f} {r['ttft_mean_ms']:10.1f} {r['ttft_p99_ms']:9.1f} "
          f"{r['in_system']:11.1f} {served * r['latency_mean_s']:10.1f}")

print()
trace = make_trace(600, 58)
print("policies at 58 requests per second, max batch 64")
print(HEADER)
print(row("first come first served", run_continuous(trace, 64)))
print(row("shortest prompt first", run_continuous(trace, 64, shortest_first=True)))
print(row("chunked, 512 tokens", run_continuous(trace, 64, token_budget=512)))
print(row("chunked, 256 tokens", run_continuous(trace, 64, token_budget=256)))
```

At 5 requests per second all three policies deliver about 800 tokens per second, because that is all the load there is, but a continuous request takes 0.83 s on average against 5.98 s (batch 16) or 4.62 s (batch 64) for a static one, and its first token arrives in milliseconds instead of seconds. At 30 requests per second static batching cannot keep up: the load is about 5,040 tokens per second, it serves 860 or 1,969, and the queue grows until mean latency is 53.31 s at batch 16. Continuous batching handles it at 0.91 s. Throughput is total tokens over the whole run, tail included, so it understates the sustained rate.

The sweep is the queueing curve. Up to 50 requests per second the time to first token stays near 10 ms, then comes the knee: 127.8 ms at 60 and 532.8 ms at 70, where the 64 slots are full and 66.2 requests are in the system. The last two columns are Little's law, the measured average in the system against rate times mean latency: 23.3 against 23.4 at 30 per second.

The policy table at 58 requests per second shows each trade. Shortest prompt first lowers the mean time to first token from 68.6 to 51.8 ms but raises the p99 from 254.7 to 751.9 ms, because long prompts keep being overtaken. Chunked prefill leaves throughput essentially unchanged and cuts the 99th-percentile gap between a user's tokens from 14.95 ms to 9.38 ms with a 512-token budget and 5.89 ms with 256. The simulator ignores padding, attention cost inside prefill, preemption and scheduler overhead.

<BatchingLab />

The lab runs the same simulator in TypeScript with a mirrored random generator and reproduces every digit above. Its defaults, continuous batching at 30 requests per second with batch 64, give 4,324 tokens per second, mean latency 0.91 s and a p99 inter-token gap of 10.88 ms.

### 2. A prefix cache keyed by chained block hashes

Prompts are a 400-token system prompt plus a unique user part of 20 to 199 tokens. Each 16-token block is hashed together with the hash before it, a lookup counts the leading blocks that are present, and an LRU cache evicts the oldest blocks. Only full blocks can be reused, so the partial tail of every prompt is always computed.

```python
from collections import OrderedDict

import numpy as np

BLOCK = 16


class PrefixCache:
    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.blocks = OrderedDict()

    def lookup_and_insert(self, tokens):
        hashes, previous = [], 0
        for start in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
            previous = hash((previous, tuple(tokens[start:start + BLOCK])))
            hashes.append(previous)
        hits = 0
        for h in hashes:
            if h not in self.blocks:
                break
            hits += 1
        for h in hashes:
            self.blocks[h] = True
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.capacity:
            self.blocks.popitem(last=False)
        return hits * BLOCK


def workload(n_requests, n_system_prompts, tweak_first_token=False, seed=0):
    rng = np.random.default_rng(seed)
    systems = [rng.integers(0, 50000, 400).tolist() for _ in range(n_system_prompts)]
    requests = []
    for i in range(n_requests):
        user = rng.integers(0, 50000, int(rng.integers(20, 200))).tolist()
        system = list(systems[int(rng.integers(0, n_system_prompts))])
        if tweak_first_token:
            system[0] = 100000 + i
        requests.append(system + user)
    return requests


def saved_fraction(requests, capacity_blocks):
    cache = PrefixCache(capacity_blocks)
    total = saved = 0
    for tokens in requests:
        saved += cache.lookup_and_insert(tokens)
        total += len(tokens)
    return saved / total


print("share of prompt tokens served from the prefix cache, 500 requests, 400-token system prompt, 16-token blocks")
print("system prompts  cache 32 blocks  256 blocks  4096 blocks")
for n_system in [1, 4, 16, 64]:
    requests = workload(500, n_system)
    cells = [saved_fraction(requests, capacity) for capacity in (32, 256, 4096)]
    print(f"{n_system:14d}  {cells[0]:15.1%} {cells[1]:11.1%} {cells[2]:12.1%}")
tweaked = workload(500, 1, tweak_first_token=True)
print(f"one system prompt, first token differs per request: {saved_fraction(tweaked, 4096):.1%}")
```

With one system prompt the cache serves 78.4 per cent of prompt tokens, the ceiling here since the system prompt is 400 of an average 510 tokens. A 32-block cache cannot hold the prompt's 25 blocks against the user blocks that evict them and manages 47.1 per cent. More distinct system prompts need more cache: with 16, 256 blocks reach 37.3 per cent and 4,096 blocks reach 76.1 per cent. The last line is the practical warning. If the first token differs for every request, for example a timestamp at the top, the hash chain differs from the first block and the hit rate is 0.0 per cent. Put stable content first and variable content last.

### 3. The real scheduler in Transformers

Sixteen requests with output lengths from 8 to 64 tokens run one at a time with `generate`, then go to `continuous_batching_context_manager` together. One detail makes the comparison fair: the continuous-batching path here does not treat an end-of-text token specially, so the sequential run disables it too with `eos_token_id=-1`. Without that, `generate` could not emit the chat model's stop token as a first token and the outputs differed.

```python
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from transformers.generation import ContinuousBatchingConfig, GenerationConfig

logging.set_verbosity_error()
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()

bases = [
    "The capital of France is",
    "Write a short story about a robot who learns to paint.",
    "List three fruits:",
    "Explain why the sky is blue in two sentences.",
]
lengths = [8, 48, 16, 64, 8, 40, 24, 64, 12, 32, 8, 64, 16, 48, 8, 56]
prompts = [bases[i % 4] for i in range(16)]
inputs = [tokenizer.encode(p) for p in prompts]
config = GenerationConfig(max_new_tokens=64, pad_token_id=tokenizer.eos_token_id, do_sample=False)

sequential = []
start = time.perf_counter()
with torch.inference_mode():
    for ids, length in zip(inputs, lengths):
        out = model.generate(
            torch.tensor([ids]),
            max_new_tokens=length,
            min_new_tokens=length,
            do_sample=False,
            eos_token_id=-1,
            pad_token_id=tokenizer.eos_token_id,
        )
        sequential.append(out[0, len(ids):].tolist())
sequential_s = time.perf_counter() - start

results = {}
start = time.perf_counter()
batching = ContinuousBatchingConfig(page_size=16, max_batch_tokens=256)
with model.continuous_batching_context_manager(generation_config=config, continuous_batching_config=batching) as manager:
    for i, (ids, length) in enumerate(zip(inputs, lengths)):
        manager.add_request(input_ids=ids, request_id=f"r{i}", max_new_tokens=length)
    for result in manager:
        results[result.request_id] = result.generated_tokens
        if len(results) == len(inputs):
            break
batched_s = time.perf_counter() - start

useful = sum(lengths)
same = sum(1 for i in range(16) if results[f"r{i}"] == sequential[i])
print(f"16 requests, {useful} output tokens in total, lengths from {min(lengths)} to {max(lengths)}")
print(f"one at a time with generate:  {sequential_s:6.1f} s  {useful / sequential_s:6.1f} tokens/s")
print(f"generate_batch scheduler:     {batched_s:6.1f} s  {useful / batched_s:6.1f} tokens/s")
print(f"requests whose tokens equal the one-at-a-time greedy output: {same} of 16")
```

The scheduler does the same work in less time and, with the stop token aligned, returns exactly the same greedy tokens for all 16 requests. The speedup on this CPU is modest, about 1.4 times in the run shown, and varies between runs. A 135M model on a CPU is not memory-bound the way an 8B model on a GPU is, and I did not profile the scheduler's own overhead, so read this block as a check that the API works and is exact, not as a speed result. The benefit that matters, a steady stream of mixed-length requests on a memory-bound GPU, is what block 1 models.

## Designing with it

- **Use continuous batching by default.** With variable output lengths it beats static batching on latency and throughput in block 1.
- **Set the token budget from your latency targets.** A small `max_num_batched_tokens` protects inter-token latency, a large one favours time to first token.
- **Watch the knee, not the average.** Sweep the arrival rate in load tests. Capacity is where first-token latency bends, and normal load should sit well below it.
- **Treat preemption as a signal** that the cache budget or concurrency limit is wrong.
- **Design prompts for the prefix cache:** stable system prompt and shared documents first, user text last, nothing volatile at the top.
- **Be careful with shortest-first.** It improves the average and hurts the tail, so add ageing if you use it.

## Where this stands in 2026

:::info Industry view

- Paged cache blocks, chunked prefill and prefix caching are documented features of vLLM, and the Hugging Face Transformers documentation lists continuous batching with paged attention, prefix caching and a choice of scheduler.
- Chunked prefill has moved from research to a default: the Sarathi-Serve paper is from March 2024, and vLLM's current V1 documentation says it is enabled by default whenever possible.
- Transformers' own scheduling API is young. Its documentation says setting the continuous-batching config on the `GenerationConfig` is deprecated, and `block_size` already warns that it is deprecated in favour of `page_size`, so pin the library version. This chapter ran Transformers 5.18.0.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why does static batching have the same throughput as continuous batching at 5 requests per second in block 1 but a much higher latency?</summary>

At light load the system keeps up with arrivals either way, so throughput equals the offered load. Latency differs because a static request waits for the current batch to finish before it starts and is returned only when its whole batch is done, while a continuous request starts at the next step and is returned the moment it finishes.

</details>

<details>
<summary><strong>Q2.</strong> A step contains 40 decoding requests and a 2,048-token prompt. Roughly how long does it take with the chapter's constants, and what does chunked prefill with a 512-token budget change?</summary>

The step has 2,088 tokens, so arithmetic is $0.01623 \times 2088 \approx 33.9$ ms against 4.79 ms of weight reading, plus a little for cache reads, and all 40 decoders wait about 34 ms. With a 512-token budget the prompt is spread across four steps of about 8.6 ms each.

</details>

<details>
<summary><strong>Q3.</strong> Why does shortest prompt first raise the 99th-percentile time to first token in block 1 while lowering the mean?</summary>

Short prompts are always served ahead of long ones, so while the queue is non-empty a long prompt can be overtaken again and again. Most requests wait less, which lowers the mean, and the unlucky long ones wait much longer, which raises the tail from 254.7 to 751.9 ms.

</details>

<details>
<summary><strong>Q4.</strong> Use Little's law to estimate how many requests are in the system at 50 requests per second with a mean latency of 1.0 s, and say what that implies for a 64-slot batch.</summary>

About $50 \times 1.0 = 50$ at the nominal rate. Block 1 measures 37.0 because its served rate over the whole run, tail included, is lower than 50. Either way a 64-slot batch is well used, so bursts queue, which is why first-token latency starts to bend between 50 and 60 requests per second.

</details>

<details>
<summary><strong>Q5.</strong> Why does the prefix cache give 0.0 per cent when the first token of every prompt is unique?</summary>

Each block's hash includes the hash of the block before it, so changing the first token changes every block's hash. No later block matches an earlier request, even though almost all the text is identical. The cache only matches whole prefixes.

</details>

<details>
<summary><strong>Q6.</strong> The sequential and batched runs in block 3 first disagreed on 9 of 16 requests. What was the cause and what does it teach?</summary>

The chat model's end-of-text token was the most likely first token for some prompts. `generate` treats it as a stop token and, with a minimum length set, suppresses it, while the continuous-batching path did not. The two paths had different stopping rules, not numerical noise. When comparing serving paths, align stopping and sampling configuration first.

</details>

## Further reading

- [Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling and selective batching.
- [Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"](https://arxiv.org/abs/2403.02310): chunked prefill and stall-free scheduling.
- [vLLM documentation, optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/): chunked prefill, decode priority, `max_num_batched_tokens`, `max_num_seqs` and preemption.
- [vLLM documentation, automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
- [Hugging Face Transformers, continuous batching](https://huggingface.co/docs/transformers/main/en/continuous_batching): `generate_batch`, the manager, schedulers, prefix caching and paged attention.
- [Kwon et al., "PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180), for the cache blocks that make continuous batching practical.
- Related chapters: [the KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention), [serving engines](/docs/llm-engineering/serving-engines) and [GPU sizing and capacity planning](/docs/llm-engineering/gpu-sizing-and-capacity-planning), which uses Little's law for capacity.

## Check yourself

- I can explain why static batching wastes slots and holds back answers.
- I can describe iteration-level scheduling and selective batching.
- I can compute what a step costs from the number of decode and prefill tokens in it.
- I can explain chunked prefill and what the token budget trades between inter-token latency and time to first token.
- I can compare first come first served with shortest prompt first and name the tail risk.
- I can use Little's law to estimate concurrency and explain why latency has a knee.
- I can say what a prefix cache needs to hit and why a changed first token defeats it.
