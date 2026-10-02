---
id: llme-speculative
title: "Speculative Decoding"
sidebar_label: "5 · Speculative decoding"
sidebar_position: 5
slug: /llm-engineering/speculative-decoding
description: "A small model drafts several tokens and the large model verifies them in one pass: the acceptance rule that keeps the output exact, the expected-tokens-per-step formula, Medusa, EAGLE and prompt lookup, and why it stops helping at high batch."
tags: [speculative-decoding, draft-model, medusa, eagle, prompt-lookup, assisted-generation, latency, inference]
---

import Infographic from '@site/src/components/Infographic';
import SpeculativeLab from '@site/src/components/viz/SpeculativeLab';

**In one line.** Speculative decoding lets a cheap model guess the next few tokens and the large model check all the guesses in a single pass, keeping the output exactly what the large model would have produced, so a memory-bound decode step yields several tokens instead of one.

:::note Not from a lecture
This chapter was written for this site from the sources under Further reading. The formulas are those of Leviathan, Kalman and Matias, which I read in the paper; every measurement comes from the code below and was taken on a CPU, where the result differs instructively from a GPU's.
:::

## The idea in plain words

Chapter 1 said a decode step is dominated by reading the weights, and costs about the same whether it carries one token or several. That leaves compute idle, and speculative decoding spends it. Imagine a junior writer who drafts the next four words quickly and an editor who reads all four at once and stops at the first word they would not have written. If the editor agrees with three, three words arrived for the price of one read, plus the editor's own word at the point of disagreement. If the junior is usually right the writing goes much faster, and because the editor decides, the text is the editor's.

Mechanically: a small **draft model** proposes $\gamma$ tokens one after another, which is cheap because it is small. The large **target model** then processes the original context plus all $\gamma$ proposals in one forward pass, producing its own prediction at every position. The longest prefix of proposals that the target agrees with is kept, and the target supplies one more token at the first disagreement. Every pass therefore yields between 1 and $\gamma + 1$ tokens.

<Infographic src="/img/llme/speculative-decoding-draft-verify.svg" alt="A draft of four tokens of which three are kept and one rejected, plus one extra token from the target, a table comparing the expected-tokens formula with simulation, and a table showing the accepted tokens follow the target distribution." caption="One speculative step, the formula, and the exactness check. The tables are blocks 1 and 2." />

<Infographic src="/img/llme/speculative-decoding-reality.svg" alt="Measured acceptance and cost on a CPU, wall-clock tokens per second for open text and copying with and without speculation, and a roofline table showing the speedup per sequence falling below 1 as batch size grows." caption="When it pays and when it does not. The CPU figures are blocks 3 and 4 (one run); the last table is block 5." />

## How it works

### Keeping the output exact

Let $p$ be the target's distribution over the next token and $q$ the draft's. For a drafted token $x$, accept it with probability $\min(1, p(x)/q(x))$. If it is rejected, draw a replacement from the **residual** distribution proportional to $\max(0, p - q)$. Leviathan et al. show that the token that comes out is distributed exactly as $p$. Chen et al., the DeepMind paper that proposed the same idea independently, describe it as a modified rejection sampling scheme that preserves the target's distribution within hardware numerics. For greedy decoding the rule collapses to "keep the proposal if it equals the target's argmax", which is what block 3 implements. vLLM's documentation states the same two properties: sampling is lossless up to the precision limits of hardware numerics, and greedy decoding with speculation matches greedy without it.

### How many tokens, and how much faster

Call $\alpha$ the probability that a drafted token is accepted. The paper defines it as 1 minus a divergence between $p$ and $q$, which block 2 checks. With $\gamma$ proposals the expected tokens per target pass is

$$E = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

Let $c$ be the cost coefficient, the time of one draft step as a fraction of one target step. The paper's Theorem 3.8 gives the wall-time improvement

$$\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

The $+1$ is the verification pass, assumed to cost one ordinary target step because there is enough compute to run the $\gamma + 1$ positions in parallel. When that fails, replace it with $v$, the measured cost of the verification pass in single steps, and the speedup is $E / (\gamma c + v)$. Two consequences follow. A draft must be cheaper than it is accurate, so an improvement needs $\alpha > c$ (the paper's Corollary 3.9). And there is a best $\gamma$: longer drafts add accepted tokens with geometrically falling probability but cost linearly, so the best length depends on $\alpha$ and $c$, as block 1's table shows.

### Where the drafts come from

| Method | Draft source | Reported by its authors |
| --- | --- | --- |
| Separate draft model | a small model sharing the tokenizer | 2 to 3 times on T5-XXL (Leviathan); 2 to 2.5 times on Chinchilla 70B in a distributed setup (Chen) |
| Medusa | extra decoding heads on the target, verified with tree attention | Medusa-1 over 2.2 times with a frozen backbone; Medusa-2 2.3 to 3.6 times with joint fine-tuning |
| EAGLE | a draft head that predicts the target's second-to-top-layer features | 2.7 to 3.5 times latency speedup on LLaMA2-Chat 70B, throughput doubled |
| Prompt lookup | n-grams copied from the prompt, no model | no figure in its Hugging Face documentation; suited to input-grounded tasks such as summarisation |

vLLM lists draft models, EAGLE, MTP, n-gram and suffix methods among others, and says model-based methods give the best latency reduction at low request rates. These figures are from each paper's abstract or documentation, on their own models and hardware.

### When it does not help

- **Low acceptance.** Open-ended text is hard to guess. Block 4 shows no gain there.
- **An expensive draft.** If $c$ approaches $\alpha$ there is nothing to gain.
- **Verification that is not free.** The single-token decode path can be a faster kernel than the multi-token one, so verifying $\gamma + 1$ tokens costs more than one step. Block 3 measures $v$ on this CPU.
- **Large batches.** Once the batch is big enough to use the compute, verifying $\gamma + 1$ tokens per sequence multiplies the work. Block 5 puts a number on it. This is why speculative decoding is a latency tool for lightly loaded services.

## A real system that works this way

**Hugging Face Transformers** ships it as assisted generation: pass `assistant_model` to `generate`, with the same tokenizer as the target, for greedy or sampled decoding without batched inputs, or set `prompt_lookup_num_tokens` for the model-free variant. **vLLM** supports the methods listed above. The research systems report the speedups in the table. Block 4 runs the Transformers versions.

## Code you can run

Five blocks. Blocks 1, 2 and 5 are arithmetic and simulation. Blocks 3 and 4 load SmolLM2-1.7B-Instruct as the target and SmolLM2-135M-Instruct as the draft, which share a tokenizer, and take under a minute each on CPU after the downloads (the larger model's weights file is 3.2 GB).

### 1. Expected tokens per step: formula against simulation

```python
import numpy as np


def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def simulate(alpha, gamma, steps, rng):
    produced = 0
    for _ in range(steps):
        accepted = 0
        while accepted < gamma and rng.random() < alpha:
            accepted += 1
        produced += accepted + 1
    return produced / steps


rng = np.random.default_rng(0)
print("tokens per target step: formula against simulation of 200,000 steps")
print("alpha  gamma  formula  simulated")
for alpha in [0.5, 0.7, 0.8, 0.9]:
    for gamma in [2, 4, 8]:
        print(f"{alpha:5.1f}  {gamma:5d}  {expected_tokens(alpha, gamma):7.3f}  {simulate(alpha, gamma, 200_000, rng):9.3f}")

print()
print("wall-time speedup (1 - a^(g+1)) / ((1 - a)(g c + 1)); c is draft step time over target step time")
print("alpha  c      " + "  ".join(f"g={g}" for g in range(1, 9)) + "   best g")
for alpha in [0.6, 0.8, 0.9]:
    for c in [0.02, 0.1, 0.3]:
        row = [speedup(alpha, g, c) for g in range(1, 9)]
        print(f"{alpha:5.1f}  {c:4.2f}   " + "  ".join(f"{v:3.2f}" for v in row) + f"   {int(np.argmax(row)) + 1}")
print()
print(f"alpha 0.8, gamma 4, c 0.05: {expected_tokens(0.8, 4):.3f} tokens per step, speedup {speedup(0.8, 4, 0.05):.2f}")
print(f"alpha 0.5 and c 0.5 (draft as costly as the acceptance rate): best speedup over gamma 1 to 8 is {max(speedup(0.5, g, 0.5) for g in range(1, 9)):.2f}")
```

The formula and the simulation agree to the third decimal: at $\alpha = 0.8$ and $\gamma = 4$ a target pass yields 3.362 tokens by formula and 3.366 in simulation. The speedup table has three lessons. With a cheap draft ($c = 0.02$) and $\alpha = 0.9$, longer is always better up to 8, and the speedup reaches 5.28. With a costly draft ($c = 0.3$) and $\alpha = 0.6$, the best draft length is 1 and the speedup is only 1.23. And at $\alpha = 0.5$ with $c = 0.5$ nothing helps: the best is 1.00.

<SpeculativeLab />

The lab is the same formula with sliders and a seeded timeline of accepted and rejected drafts. Its defaults, $\alpha = 0.80$, $\gamma = 4$, $c = 0.05$ and $v = 1$, give the 3.362 tokens per step and speedup 2.80 that block 1 prints. Raise $v$ to 3.3 with $\alpha$ 0.77 and $c$ 0.13 and it shows block 3's result.

### 2. Why the output stays exact

Six tokens, a target distribution $p$ and a different draft distribution $q$. Draft a token from $q$, accept with probability $\min(1, p/q)$, else resample from the residual. Compare what comes out with $p$.

```python
import numpy as np

rng = np.random.default_rng(1)
vocab = 6
p = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
q = np.array([0.15, 0.30, 0.20, 0.05, 0.25, 0.05])
trials = 400_000

drafted = rng.choice(vocab, size=trials, p=q)
accept = rng.random(trials) < np.minimum(1.0, p[drafted] / q[drafted])
residual = np.maximum(p - q, 0)
residual = residual / residual.sum()
fallback = rng.choice(vocab, size=trials, p=residual)
output = np.where(accept, drafted, fallback)

empirical = np.bincount(output, minlength=vocab) / trials
naive = np.bincount(drafted, minlength=vocab) / trials
beta = np.minimum(p, q).sum()

print("token         ", "  ".join(f"{i:6d}" for i in range(vocab)))
print("target p      ", "  ".join(f"{v:6.3f}" for v in p))
print("draft q       ", "  ".join(f"{v:6.3f}" for v in q))
print("speculative   ", "  ".join(f"{v:6.3f}" for v in empirical))
print("draft alone   ", "  ".join(f"{v:6.3f}" for v in naive))
print()
print(f"total variation distance to p: speculative {0.5 * np.abs(empirical - p).sum():.4f}, draft alone {0.5 * np.abs(naive - p).sum():.4f}")
print(f"acceptance rate: measured {accept.mean():.4f}, 1 - TV(p, q) = sum of min(p, q) = {beta:.4f}")
```

Over 400,000 trials the speculative output matches $p$ to within 0.002 in total variation distance, while taking the draft's tokens unchecked is off by 0.300. The acceptance rate is 0.6983 measured against 0.7000 predicted by the sum of $\min(p, q)$, which is $1 - \text{TV}(p, q)$ and the $\alpha$ of this pair.

### 3. A real greedy speculative loop

`speculative` below is the algorithm with KV caches for both models: the draft proposes four tokens, the target verifies five positions in one pass, the accepted prefix is kept and the rejected tail is cropped from both caches. It runs on three prompts of 32 new tokens each and is checked against the target's own greedy output. Timings use the minimum of five runs on cached contexts.

```python
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, logging

logging.set_verbosity_error()
torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
target = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float32).eval()
draft = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).eval()
GAMMA = 4
NEW_TOKENS = 32
prompts = [
    "The history of the bicycle is long and",
    "To make a good cup of tea, first",
    "In 1969, the first humans to walk on the Moon",
]


def next_token(model, tokens, cache):
    logits = model(tokens, past_key_values=cache, use_cache=True).logits
    return logits[0].argmax(dim=-1)


def speculative(ids):
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    pending = next_token(target, ids, target_cache)[-1].item()
    draft(ids, past_key_values=draft_cache, use_cache=True)
    produced, accepted_total, tried_total, forwards = [], 0, 0, 0
    while len(produced) < NEW_TOKENS:
        produced.append(pending)
        proposals, current = [], pending
        for _ in range(GAMMA):
            current = next_token(draft, torch.tensor([[current]]), draft_cache)[-1].item()
            proposals.append(current)
        verify = torch.tensor([[pending] + proposals])
        choices = next_token(target, verify, target_cache).tolist()
        forwards += 1
        accepted = 0
        while accepted < GAMMA and proposals[accepted] == choices[accepted]:
            accepted += 1
        accepted_total += accepted
        tried_total += accepted + (1 if accepted < GAMMA else 0)
        produced.extend(proposals[:accepted])
        pending = choices[accepted]
        if accepted < GAMMA:
            target_cache.crop(-(GAMMA - accepted))
            draft_cache.crop(-(GAMMA - 1 - accepted))
        else:
            next_token(draft, torch.tensor([[proposals[-1]]]), draft_cache)
    return produced[:NEW_TOKENS], accepted_total, tried_total, forwards


def timed_step(model, tokens):
    cache = DynamicCache()
    model(tokenizer("The history of the bicycle is long and", return_tensors="pt").input_ids, past_key_values=cache, use_cache=True)
    best = 1e9
    for _ in range(5):
        start = time.perf_counter()
        model(torch.tensor([[5] * tokens]), past_key_values=cache, use_cache=True)
        best = min(best, time.perf_counter() - start)
        cache.crop(-tokens)
    return best


accepted_all = tried_all = forwards_all = tokens_all = 0
matches = 0
with torch.inference_mode():
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        produced, accepted, tried, forwards = speculative(ids)
        reference = target.generate(ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False)[0, ids.shape[1]:].tolist()
        matches += produced == reference
        accepted_all += accepted
        tried_all += tried
        forwards_all += forwards
        tokens_all += len(produced)
        print(f"{prompt!r}: {tokens_all} tokens so far, identical to target greedy: {produced == reference}")
    alpha = accepted_all / tried_all
    expected = (1 - alpha ** (GAMMA + 1)) / (1 - alpha)
    step_target, step_draft, verify = timed_step(target, 1), timed_step(draft, 1), timed_step(target, GAMMA + 1)

print()
print(f"identical outputs: {matches} of {len(prompts)}")
print(f"measured acceptance alpha = {alpha:.3f}; tokens per target forward: measured {tokens_all / forwards_all:.2f}, formula {expected:.2f}")
c, v = step_draft / step_target, verify / step_target
print(f"one target decode step {step_target * 1e3:.0f} ms, one draft step {step_draft * 1e3:.0f} ms, verifying {GAMMA + 1} tokens {verify * 1e3:.0f} ms")
print(f"cost coefficient c = {c:.3f}; verification costs v = {v:.2f} single steps, not 1")
print(f"predicted speedup with v = 1:      {expected / (GAMMA * c + 1):.2f}")
print(f"predicted speedup with measured v: {expected / (GAMMA * c + v):.2f}")
```

All three outputs are identical to the target's greedy output. The measured acceptance is $\alpha = 0.769$, and the measured 3.20 tokens per target pass agrees with the formula's 3.17. Then the cost side. A draft step takes about 13 per cent of a target step, so the formula with a free verification would predict a speedup of about 2.1. But on this CPU, verifying five tokens costs about 3.3 single steps, not 1, because the single-token step appears to take a different, faster kernel path (chapter 1's block 2 starts at batch 2 to avoid the same effect, and I did not profile the cause). With that $v$ the prediction falls to 0.83: slower than plain decoding despite a good draft. Timings vary between runs and machines, so read the ratios.

### 4. The same idea through Transformers, and a task where it wins

`generate` with `assistant_model` is the draft-model method and `prompt_lookup_num_tokens` is the model-free one. Two tasks: continuing a sentence, and repeating a passage that is in the prompt, where the answer is largely copied.

```python
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()
torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
target = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float32).eval()
draft = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).eval()
NEW_TOKENS = 48

open_prompt = tokenizer("The history of the bicycle is long and", return_tensors="pt").input_ids
passage = (
    "The council approved the new library budget on Tuesday. The budget includes funds for longer opening hours, "
    "a refurbished children's section and twelve new public computers. Work will begin in March."
)
grounded_prompt = tokenizer(f"Text: {passage}\nThe same text, repeated word for word: The council", return_tensors="pt").input_ids


def run(ids, **kwargs):
    start = time.perf_counter()
    with torch.inference_mode():
        out = target.generate(ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False, **kwargs)
    return out[0, ids.shape[1]:].tolist(), time.perf_counter() - start


run(open_prompt, assistant_model=None)
print("task                      method                 seconds  tokens/s  same tokens as plain")
for label, ids in [("open continuation", open_prompt), ("copy from the prompt", grounded_prompt)]:
    plain, plain_s = run(ids)
    print(f"{label:25s} {'plain greedy':22s} {plain_s:7.2f}  {NEW_TOKENS / plain_s:8.1f}  -")
    for method, kwargs in [("draft model, 135M", {"assistant_model": draft}), ("prompt lookup, 10 tokens", {"prompt_lookup_num_tokens": 10})]:
        tokens, seconds = run(ids, **kwargs)
        print(f"{'':25s} {method:22s} {seconds:7.2f}  {NEW_TOKENS / seconds:8.1f}  {tokens == plain}")
print()
print("grounded output:", repr(tokenizer.decode(plain[:30])))
```

Every output is token-for-token identical to plain greedy decoding. On open continuation both speculative methods are slower on this CPU, about 7.5 tokens per second against 10.8, in line with block 3. On the copying task the same machinery is faster: about 15.7 tokens per second with the draft model and 16.7 with prompt lookup against 11.0. The copying task makes acceptance high, and prompt lookup needs no second model, which is why it suits summarisation and extraction. The gains here are modest because of the verification cost above, and on a GPU a draft model on open text would normally do much better. I did not test that.

### 5. Why batching erodes the gain

Using chapter 1's roofline for an 8B target and a hypothetical 0.5B draft, the draft's $\gamma$ steps and the target's $B(\gamma + 1)$-token verification are timed per step and compared with plain decoding at the same batch. The draft's own cache traffic is ignored.

```python
PEAK_FLOPS = 989.5e12
BANDWIDTH = 3.35e12
TARGET_PARAMS = 8.03e9
DRAFT_PARAMS = 0.5e9
BYTES = 2
ALPHA, GAMMA = 0.8, 4


def step_ms(params, tokens):
    memory = params * BYTES / BANDWIDTH
    compute = 2 * params * tokens / PEAK_FLOPS
    return max(memory, compute) * 1e3


expected = (1 - ALPHA ** (GAMMA + 1)) / (1 - ALPHA)
print(f"target 8.03 B and a hypothetical 0.5 B draft in bf16, alpha {ALPHA}, gamma {GAMMA}: {expected:.3f} tokens per verification")
print("batch  plain_ms  draft_ms  verify_ms  speculative_ms  speedup_per_sequence")
for batch in [1, 8, 32, 64, 128, 256, 512]:
    plain = step_ms(TARGET_PARAMS, batch)
    drafting = GAMMA * step_ms(DRAFT_PARAMS, batch)
    verify = step_ms(TARGET_PARAMS, batch * (GAMMA + 1))
    speculative = drafting + verify
    print(f"{batch:5d}  {plain:8.2f}  {drafting:8.2f}  {verify:9.2f}  {speculative:14.2f}  {expected * plain / speculative:20.2f}")

crossover = next(b for b in range(1, 1025) if expected * step_ms(TARGET_PARAMS, b) / (GAMMA * step_ms(DRAFT_PARAMS, b) + step_ms(TARGET_PARAMS, b * (GAMMA + 1))) < 1)
print(f"speculative decoding stops paying at a batch of about {crossover} on these numbers")
```

At small batches the speedup is the full 2.69 times: verification costs the same 4.79 ms as a plain step because the weights read dominates. Verifying five tokens per sequence reaches the ridge at a batch of about 59, so by a batch of 64 the speedup is 2.52, at 128 it is 1.39, and above about 184 it is below 1: speculative decoding then does more work than it saves, since the plain step, at batch 256, is still memory-bound at 4.79 ms while verification is compute-bound at 20.77 ms. This ignores scheduling overheads and the draft's cache.

## Designing with it

- **Use it for latency on lightly loaded services**, and measure at your real batch size. It is a way to spend idle compute, so it stops helping when there is none.
- **Measure $\alpha$ on your own traffic.** Acceptance depends on the task: copying and extraction accept far more than free-form writing.
- **Pick the draft by $c$ and $\alpha$ together.** A smaller draft is cheaper but accepts less. Check $\alpha > c$ and pick $\gamma$ from the speedup curve.
- **Measure $v$ on your kernels.** The formula assumes a free verification pass, which was false on this CPU.
- **Try prompt lookup first for grounded tasks.** It costs no memory and needs no second model.

## Where this stands in 2026

:::info Industry view

- The idea is documented in the major open stacks. Hugging Face Transformers has `assistant_model` and `prompt_lookup_num_tokens`, with the stated limits of greedy or sampled decoding and no batched inputs, and vLLM lists a wide set of methods including EAGLE, draft models, n-gram and multi-token prediction.
- The best-known gains come from methods that train the drafter on the target, Medusa and EAGLE, rather than from an off-the-shelf small model. Their authors report 2.2 to 3.6 times and 2.7 to 3.5 times on their own benchmarks.
- vLLM's own guidance ties the choice to load: model-based methods give the best latency reduction at low request rates, which matches block 5.
- Check each engine's current page for supported methods and constraints, since the lists change quickly. I read vLLM's page on 2 October 2026 and did not test its engine.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> With $\alpha = 0.7$ and $\gamma = 4$, how many tokens does each target pass produce on average, and what speedup do you get with $c = 0.1$?</summary>

$E = (1 - 0.7^5)/(1 - 0.7) = 2.773$, which is block 1's value. The speedup is $2.773/(4 \times 0.1 + 1) = 1.98$.

</details>

<details>
<summary><strong>Q2.</strong> A draft model costs half as much per step as the target ($c = 0.5$) and agrees with it 50 per cent of the time. Is it worth using?</summary>

No. The paper's condition for any improvement is $\alpha > c$ and here they are equal, and block 1 prints a best speedup of 1.00. The draft is too expensive for how often it is right.

</details>

<details>
<summary><strong>Q3.</strong> Why is the output unchanged, and what happens at a rejection?</summary>

A drafted token is accepted with probability $\min(1, p/q)$. On rejection the replacement is drawn from the normalised $\max(0, p - q)$. Together these make the emitted token's distribution exactly $p$, which block 2 confirms with a distance of 0.002. In greedy decoding it reduces to keeping tokens that equal the target's argmax.

</details>

<details>
<summary><strong>Q4.</strong> Block 3 predicts a speedup of 2.1 with a free verification pass and 0.83 with the measured one. Which would you trust, and what would you check on a GPU?</summary>

The measured one, for this machine. On a GPU you would measure the time of a $(\gamma + 1)$-token verification pass against a one-token step at your batch size and sequence length, since that ratio $v$ decides whether the formula's assumption holds.

</details>

<details>
<summary><strong>Q5.</strong> Using block 5's logic, why does speculative decoding stop helping at large batch sizes even with a perfect draft?</summary>

Verifying $\gamma + 1$ tokens per sequence multiplies the arithmetic by $\gamma + 1$. Once the batch is large enough that this is compute-bound, the verification takes much longer than the memory-bound plain step it replaces, so the extra tokens no longer pay for it. On block 5's numbers, the break-even is at a batch of about 184.

</details>

## Further reading

- [Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)](https://arxiv.org/abs/2211.17192): the algorithm, Theorem 3.8 and the proof of exactness.
- [Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318): the DeepMind version of the same idea.
- [Cai et al., "Medusa"](https://arxiv.org/abs/2401.10774) and [Li et al., "EAGLE"](https://arxiv.org/abs/2401.15077): drafting heads and feature-level drafting.
- [vLLM documentation, speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/): the supported methods and when they help.
- [Hugging Face Transformers, optimizing inference](https://huggingface.co/docs/transformers/main/en/llm_optims): `assistant_model` and prompt lookup decoding.
- Related chapters: [why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound), [continuous batching](/docs/llm-engineering/continuous-batching-and-scheduling) and [quantisation](/docs/llm-engineering/quantisation-for-inference), the other ways to cut the cost of a decode step.

## Check yourself

- I can describe the draft, verify and accept loop and say why each pass yields between 1 and $\gamma + 1$ tokens.
- I can state the acceptance rule and the residual distribution and say why the output is unchanged.
- I can compute the expected tokens per step and the speedup from $\alpha$, $\gamma$ and $c$.
- I can explain why $\alpha > c$ is necessary and why there is a best $\gamma$.
- I can name the draft sources, a draft model, Medusa heads, EAGLE and prompt lookup, and say when each fits.
- I can explain why batching and an expensive verification pass erode the gain.
