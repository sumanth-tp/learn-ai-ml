---
id: llme-moe
title: "Mixture of Experts"
sidebar_label: "4 · Mixture of experts"
sidebar_position: 4
slug: /llm-engineering/mixture-of-experts
description: "How a router sends each token to a few expert feed-forward networks, why total and active parameters differ, how load-balancing losses and bias updates keep experts busy, and how expert parallelism moves tokens, checked on real configs and current model cards."
tags: [mixture-of-experts, moe, routing, load-balancing, expert-parallelism, mixtral, deepseek, switch-transformer, all-to-all]
---

import Infographic from '@site/src/components/Infographic';
import MoeRoutingLab from '@site/src/components/viz/MoeRoutingLab';

**In one line.** A mixture-of-experts model stores many feed-forward networks (experts) in each layer but sends each token through only a few of them, so it holds the knowledge of a very large model while paying the arithmetic of a small one, in exchange for keeping every expert in memory and keeping the load between experts balanced.

:::tip Before you start
- **You should already know** what a transformer layer's feed-forward block does ([transformer decoder architecture](/docs/theory/dnn/transformer-decoder-architecture)), the parallelism axes from [parallelism strategies for LLMs](/docs/llm-engineering/parallelism-strategies-for-llms), and why decoding is limited by memory traffic ([why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound)).
- **Reading time:** about 45 minutes, plus a minute or two to run the code.
- **After this chapter you can** compute total and active parameters from a config, say what the load-balancing loss and capacity factor do, explain how expert parallelism moves tokens, and read a current model card's mixture-of-experts numbers critically.
:::

:::note Not from a lecture
This chapter was written for this site from the papers and model cards under Go deeper. Model sizes are counted from real configs read from the Hugging Face Hub by the code below. Versions used: Python 3.14, PyTorch 2.14.1 (CPU), Transformers 5.18.0. Model cards and papers were opened on 7 and 8 October 2026; this field changes monthly, so check the card before you rely on a number.
:::

## In 30 seconds

Imagine a hospital with eight specialist doctors and one receptionist. The receptionist glances at each patient and sends them to the two doctors who seem most useful. The hospital as a whole knows far more than any one doctor, yet each patient only spends time with two. That is a mixture-of-experts layer: the receptionist is the router and the doctors are the experts.

Two warnings from real systems. The "specialists" do not become experts in subjects like law or biology; the Mixtral paper looked and found no obvious topic pattern. And the receptionist can send too many patients to the same doctor, so training adds tricks to spread the load.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Expert | One feed-forward network inside a layer | One of 8 in Mixtral 8x7B |
| Router (gate) | A small linear layer that scores every expert for a token | Eight scores per token |
| Top-k | The k highest-scoring experts, the ones that actually run | Mixtral uses k = 2 |
| Active parameters | The parameters a single token passes through | 12.88 billion of Mixtral's 46.70 billion |
| Load-balancing loss | An extra training term that punishes uneven expert use | Coefficient 0.01 in the Switch paper |
| Capacity factor | How much room each expert has, relative to a perfectly even share | 1.25 means 25 per cent spare room |
| Dropped token | A token an over-full expert refuses; it skips the layer | Passed on through the residual connection |
| Shared expert | An expert every token always uses, beside the routed ones | DeepSeek-V3 has 1 |
| All-to-all | A collective where every rank sends a different piece to every other rank | Tokens going to their experts |

## The idea in plain words

In an ordinary transformer, every token goes through the same feed-forward network in every layer. To make the model know more you make that network bigger, and every token then pays for the extra size. A mixture-of-experts layer breaks the link between "how much the model stores" and "how much each token costs". It keeps several feed-forward networks and a router, and only the top few run for each token.

Take the smallest example. Four experts, eight tokens, and the router gives each token a probability for each expert. Each token goes to its single best expert. If five of the eight tokens pick expert A, then A does most of the work and the others sit idle. That is a **load imbalance**, and it is the main thing that goes wrong with these models. It wastes compute, and with expert parallelism the busiest GPU sets the pace for everyone.

So a mixture-of-experts model has three jobs beyond a dense one: choose the experts, keep them evenly used, and move the tokens to wherever the experts live. The next sections take them in that order.

<Infographic src="/img/llme/mixture-of-experts-layer.svg" alt="A dense feed-forward block beside a mixture-of-experts block in which a token passes through a router, two chosen experts out of eight, and a weighted sum, with Mixtral 8x7B's 46.70 billion stored and 12.88 billion used parameters" caption="Look at the right panel: eight experts are stored, two run. The yellow box at the bottom gives the price and the prize in Mixtral's own numbers." />

## Worked example, step by step

The eight tokens and four experts from the smallest example, with the router's probabilities. Block 1 computes every number below.

1. **Pick the top expert.** Tokens 0 to 4 prefer expert A (probabilities 0.7, 0.6, 0.5, 0.4, 0.4). Tokens 5 and 6 prefer B. Token 7 prefers C. No token prefers D.
2. **Count the share of tokens.** f = A 5/8 = 0.625, B 2/8 = 0.25, C 1/8 = 0.125, D 0.
3. **Average the router probabilities.** P = A 0.3875, B 0.275, C 0.2375, D 0.1.
4. **Compute the balance term.** N times the sum of f times P, with N = 4 experts: 4 x (0.625 x 0.3875 + 0.25 x 0.275 + 0.125 x 0.2375 + 0 x 0.1) = 4 x 0.340625 = 1.3625. A perfectly even router would score exactly 1.0. In words: the term is large when the experts that get most of the tokens are also the ones the router likes most.
5. **Apply a capacity.** With a capacity factor of 1.0, each expert has room for 8 / 4 = 2 tokens. A can keep 2 of its 5, so 3 tokens are dropped, 37.5 per cent. With a capacity factor of 1.5 (room for 3) 2 are dropped. With 2.5 (room for 5) none are dropped, but only 8 of 20 slots are used.

<Infographic src="/img/llme/mixture-of-experts-worked.svg" alt="A table of eight tokens with router probabilities for four experts and each token's chosen expert, beside the shares f, the balance term 1.3625 and a table of capacity factors 1.0, 1.5 and 2.5 with tokens dropped and slots used" caption="Read the left table first, then the orange boxes top to bottom, then the capacity table. All figures are printed by block 1." />

## How it works

### What is inside a mixture-of-experts layer?

The router is one linear layer from the hidden size to the number of experts. It gives each token a score for each expert. The top-k scores pick the experts. Mixtral then applies a softmax to those k scores only to get the weights, and the layer's output is the weighted sum of the chosen experts' outputs. The paper writes it as `y = sum of Softmax(Top2(x Wg))_i times SwiGLU_i(x)`. In words: score the experts, keep the best two, turn their scores into weights, and add up their answers.

The experts are ordinary feed-forward blocks. Mixtral's are three-matrix SwiGLU blocks, so the layer has the same shape as a dense Llama layer, repeated eight times. Newer models change the recipe in three ways, which you can see in their `config.json` files:

| Change | What it is | Where you can see it |
| --- | --- | --- |
| Many small experts | 128 to 896 experts instead of 8, each smaller | DeepSeek-V3 has 256 routed experts, top 8; Kimi K3 has 896, top 16 |
| Shared experts | One or two experts every token always uses, besides the routed ones | DeepSeek-V3 `n_shared_experts` 1; Kimi K3 `num_shared_experts` 2 |
| New scoring functions | Sigmoid or square-root-softplus scores instead of a softmax | DeepSeek-V3 `sigmoid`; DeepSeek-V4-Pro `sqrtsoftplus` |

The DeepSeekMoE paper argues for the first two: finer experts let each token combine a more flexible mix, and shared experts hold knowledge that every token needs, so the routed ones need not duplicate it.

### Total parameters against active parameters

Parameters in experts are stored for all experts but used for only k of E. So `active = total - expert parameters x (1 - k/E)`. For Mixtral 8x7B, block 4 counts 46.70 billion parameters, of which 45.10 billion are in experts. With k = 2 of 8, the active count is 46.70 - 45.10 x 0.75 = 12.88 billion. The paper reports "47B" and "13B".

Counting conventions differ. Qwen's card for Qwen3-30B-A3B gives 3.3 billion active, which matches counting everything (3.35). OpenAI's card for gpt-oss-120b gives 5.1 billion, which matches only if the 0.58 billion input embedding is left out (5.71 - 0.58 = 5.13). When you compare models, check how the card counts.

### Why do experts become unbalanced, and what fixes it?

Early in training a few experts get slightly more tokens, learn faster, look better to the router, and get even more tokens. This is **routing collapse**, and Shazeer and colleagues described it in the paper that introduced the sparsely-gated layer. Three remedies are in use.

**A balancing loss.** The Switch Transformer adds `alpha x N x sum of f_i x P_i` to the training loss, where `f_i` is the share of tokens sent to expert `i` and `P_i` is the router's average probability for it. In words: the product is small only when no expert is both busy and favoured. Block 1 shows it equals 1.0 for a perfectly even router. The Switch paper used `alpha = 0.01`, found by sweeping from 0.1 to 0.00001. The loss reaches the router through `P_i`, which is differentiable; `f_i` is a count and is not.

**A capacity limit.** Each expert gets room for `capacity factor x tokens x k / experts` slots. Extra tokens are dropped: their computation is skipped and they pass through the layer on the residual connection. The Switch paper notes that more capacity wastes compute and memory, and that low drop rates matter for quality.

**A bias instead of a loss.** DeepSeek-V3 argues that a big auxiliary loss hurts quality. It adds a bias to each expert's score, used only to choose the top-k, never in the gate weights. After each step it lowers the bias of overloaded experts and raises it for underloaded ones by a fixed step, 0.001 for most of training. It keeps a tiny sequence-level balance loss, coefficient 0.0001, to stop extreme cases within one sequence. The method comes from an earlier paper on auxiliary-loss-free balancing.

Block 2 runs all three on a toy problem.

### What does expert parallelism do?

With expert parallelism, each GPU holds a few experts, so a token's chosen experts may live on other GPUs. Each step has two **all-to-all** exchanges. Dispatch sends each token to the rank that holds each of its experts. After the experts compute, combine sends the results back, and each token's outputs are added with their gate weights. Block 5 runs this on two real processes and matches a single process.

The weak point is balance. Every rank must wait for the slowest, so a rank that receives 40 token slots while another receives 24 sets the pace, with the other idle 40 per cent of the time. DeepSeek-V3 limits each token to experts on at most 4 nodes to keep traffic on the faster links. Its experts for each layer are spread over 64 GPUs on 8 nodes.

<Infographic src="/img/llme/mixture-of-experts-all-to-all.svg" alt="Two ranks each holding two experts, a dispatch all-to-all in which rank 0 sends 21 slots to itself and 11 to rank 1 while rank 1 sends 19 to rank 0 and 13 to itself, and a combine all-to-all that returns the outputs" caption="Follow a token copy from its own rank to the rank of its expert and back. The red box is the lesson: rank 0 receives 40 slots and rank 1 only 24. Counts are printed by block 5." />

### What does it mean for memory and serving?

Compute follows the active parameters; memory follows the total. Every expert must be resident because the next token may choose any of them. So a mixture-of-experts model is cheap per token but needs many GPUs' worth of memory. The previous inference chapters apply: [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention) still sizes the context, and [GPU sizing](/docs/llm-engineering/gpu-sizing-and-capacity-planning) must use total parameters for weights, not active ones.

### Which models are current?

The table is built from the official model cards and configs, read on 7 and 8 October 2026. Dates are when each repository was created on the Hub, which is not always the announcement date.

| Model | Created on the Hub | Total / active parameters (card) | Experts | Notes from the card or config |
| --- | --- | --- | --- | --- |
| Mixtral 8x7B | December 2023 | 47B / 13B (paper) | 8, top 2 | Apache 2.0 weights |
| gpt-oss-120b | August 2025 | 117B / 5.1B | 128, top 4 | MXFP4-quantised expert weights; fits one 80 GB GPU |
| Llama 4 Scout | April 2025 | 109B / 17B | 16 | Card lists 17B activated |
| Qwen3-30B-A3B | April 2025 | 30.5B / 3.3B | 128, top 8 | |
| DeepSeek-V3 | December 2024 | 671B / 37B | 256 + 1 shared, top 8 | Auxiliary-loss-free balancing |
| Qwen3.6-35B-A3B | April 2026 | 35B / 3B | 256, top 8 + 1 shared | Gated DeltaNet and gated attention layers |
| DeepSeek-V4-Pro and Flash | April 2026 | 1.6T / 49B and 284B / 13B | 384 and 256, top 6 | Card calls it a preview; experts in FP4 |
| Qwen3.8-Flash-Next | August 2026 | 125B / 6B (plus 51B n-gram embedding) | 512, 10 routed + 1 shared | Card figures; read on 8 October 2026 |
| Kimi K3 | June 2026 | 2.8T / 104B | 896 + 2 shared, top 16 | MXFP4 weights, quantisation-aware training |
| Kolibri-1 | October 2026 | 78.1B / 3.46B | 384, 1 shared + 6 routed | Experts in fp8 blocks; router in bf16 |

<Infographic src="/img/llme/mixture-of-experts-models.svg" alt="Horizontal bars on a log axis for nine mixture-of-experts models, each with total parameters in blue and activated parameters in orange, from Mixtral 8x7B at 46.7B total and 12.9B active to Kimi K3 at 2,800B total and 104B active" caption="Compare each blue bar with the orange bar beneath it: the active share is 27.6 per cent for Mixtral and 3 to 5 per cent for most 2026 models. Figures are from the cards and papers in the table." />

Later revisions already exist for some rows: the Hub lists DeepSeek-V4-Pro-0813 (August 2026, same expert layout in its config) and DeepSeek-V4.1-Flash (September 2026, a multimodal model whose card gives 552B backbone parameters). The table keeps the first card of each family that this chapter read in full.

Two trends stand out. Experts keep getting more numerous and smaller, and the active fraction keeps falling: from 27.6 per cent in Mixtral to 3 to 9 per cent in the 2026 models. And expert weights are stored in 4 or 8 bits, which links back to the [previous chapter](/docs/llm-engineering/mixed-precision-and-numerics).

## A real system that works this way

**Mixtral 8x7B** is the clearest open example: a Mistral 7B-style model with the feed-forward block replaced by eight experts and top-2 routing. The paper reports that it matches or outperforms Llama 2 70B and GPT-3.5 across the benchmarks it evaluated, using 13B active parameters. Those are the authors' evaluations, from December 2023.

**DeepSeek-V3** is the best-documented training system. Its report describes the 256 routed experts per layer, the bias-based balancing, the 4-node routing limit and fp8 training, as covered above and in the last chapter.

**The Switch Transformer** (2021) is the origin of the single-expert, balancing-loss and capacity-factor recipe, and reports models of up to a trillion parameters.

## Code you can run

Five blocks. Blocks 1 to 3 and 5 run in a few seconds on the CPU. Block 4 builds large models on the `meta` device, which allocates no memory, and reads configs from the Hub.

### 1. Routing by hand, then a seeded router at scale

The 8-token example in numpy, with its capacity table, then a seeded simulation of 1,024 tokens, 16 experts and top-2 routing with a "skew" that makes some experts favourites. This simulation is the one the lab runs.

```python
import math

import numpy as np

probs = np.array([
    [0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1],
    [0.4, 0.2, 0.3, 0.1], [0.2, 0.5, 0.2, 0.1], [0.2, 0.4, 0.3, 0.1], [0.1, 0.2, 0.6, 0.1],
])
T, N = probs.shape
choice = probs.argmax(axis=1)
counts = np.bincount(choice, minlength=N)
f = counts / T
P = probs.mean(axis=0)
aux = N * float((f * P).sum())
print("the hand example: 8 tokens, 4 experts, each token goes to its top expert")
print("tokens per expert:", counts.tolist(), " f =", f.tolist())
print("mean router probability P =", np.round(P, 4).tolist())
print(f"balance term N * sum(f * P) = {aux:.4f}   (1.0000 when perfectly balanced)")
for cf in (1.0, 1.5, 2.5):
    capacity = math.ceil(cf * T / N)
    kept = np.minimum(counts, capacity)
    print(f"capacity factor {cf}: room for {capacity} tokens per expert, dropped {T - kept.sum()} of {T}, slots used {kept.sum()} of {capacity * N}")


def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]

    def draw():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return draw


def simulate(n_experts, k, tokens, cf, skew, seed=7):
    draw = mulberry32(seed)

    def gauss():
        u1, u2 = max(draw(), 1e-12), draw()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    bias = np.array([gauss() for _ in range(n_experts)])
    logits = np.array([[gauss() + skew * bias[e] for e in range(n_experts)] for _ in range(tokens)])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    top = np.argsort(-logits, axis=1)[:, :k]
    counts = np.bincount(top.ravel(), minlength=n_experts)
    f = counts / (tokens * k)
    aux = n_experts * float((f * p.mean(axis=0)).sum())
    capacity = math.ceil(cf * tokens * k / n_experts)
    kept = np.minimum(counts, capacity)
    return aux, counts.max() / (tokens * k / n_experts), 1 - kept.sum() / (tokens * k), kept.sum() / (capacity * n_experts)


print()
print("1,024 tokens, 16 experts, top-2, capacity factor 1.25, seeded router; skew makes some experts favourites")
print("skew   balance term   busiest expert / average   slots dropped   capacity used")
for skew in (0.0, 0.25, 0.5, 1.0):
    aux, busiest, dropped, used = simulate(16, 2, 1024, 1.25, skew)
    print(f"{skew:5.2f}   {aux:12.4f}   {busiest:24.2f}   {dropped:13.1%}   {used:13.1%}")
print()
print("same router, skew 0.5, different capacity factors")
for cf in (1.0, 1.25, 1.5, 2.0, 3.0):
    aux, busiest, dropped, used = simulate(16, 2, 1024, cf, 0.5)
    print(f"capacity factor {cf:4.2f}: dropped {dropped:6.1%}, capacity used {used:6.1%}")
```

**Reading the output.** The hand example prints counts 5, 2, 1, 0, a balance term of 1.3625 and drops of 3, 2 and 0 tokens at capacity factors 1.0, 1.5 and 2.5. In the simulation, no skew gives a balance term of 1.0019, a busiest expert 1.10 times the average, and 80.0 per cent of capacity used, which is exactly 1 / 1.25. Skew 0.5 gives 1.5836, 3.72 times the average and 28.5 per cent of slots dropped. At that skew, raising the capacity factor from 1.25 to 3.0 cuts drops from 28.5 to 4.5 per cent but leaves only 31.8 per cent of capacity in use.

**Line by line.**

- `mulberry32` is a small seeded random generator; the lab uses the same one so both give the same numbers.
- `skew * bias[e]` gives each expert a fixed advantage for every token, which is how a router develops favourites.
- `kept = np.minimum(counts, capacity)` is the capacity rule: whatever an expert receives beyond its room is dropped.

### 2. Training a toy mixture of experts with three kinds of balancing

Eight small experts, top-2 routing, and data in which cluster 0 is eight times as common as cluster 7. We train with no balancing, with the Switch-style loss at two strengths, and with a DeepSeek-style bias update, and look at the last 100 steps averaged over 3 seeds.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

E, D, K, STEPS, BATCH = 8, 16, 2, 600, 256


def train(method, seed, alpha=0.0, gamma=0.003):
    torch.manual_seed(seed)
    centres = torch.randn(E, D) * 2.0
    maps = torch.randn(E, D, D) / D**0.5
    router = nn.Linear(D, E, bias=False)
    with torch.no_grad():
        router.weight.mul_(0.1)
    experts = nn.ModuleList([nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, D)) for _ in range(E)])
    opt = torch.optim.Adam(list(router.parameters()) + list(experts.parameters()), lr=3e-3)
    freq = 1.0 / torch.arange(1, E + 1).float()
    freq = freq / freq.sum()
    bias = torch.zeros(E)
    data = torch.Generator().manual_seed(seed + 100)
    history = []
    for _ in range(STEPS):
        cluster = torch.multinomial(freq, BATCH, replacement=True, generator=data)
        x = centres[cluster] + 0.5 * torch.randn(BATCH, D, generator=data)
        target = torch.einsum("bd,bde->be", x, maps[cluster])
        logits = router(x)
        chosen = (logits + bias).topk(K, dim=-1).indices
        gate = torch.gather(logits, 1, chosen).softmax(-1)
        out = torch.zeros_like(x)
        for e in range(E):
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            if len(rows):
                out[rows] += gate[rows, slot].unsqueeze(-1) * experts[e](x[rows])
        share = torch.bincount(chosen.reshape(-1), minlength=E).float() / (BATCH * K)
        balance = E * (share * logits.softmax(-1).mean(0)).sum()
        task = F.mse_loss(out, target)
        opt.zero_grad()
        (task + alpha * balance).backward()
        opt.step()
        if method == "bias":
            bias -= gamma * torch.sign(share - 1.0 / E)
        history.append((task.item(), share.max().item(), share.min().item()))
    return np.array(history[-100:]).mean(axis=0)


print("top-2 routing, 8 experts, cluster 0 is 8 times as common as cluster 7; mean of the last 100 steps and 3 seeds")
print("method                  task loss   busiest expert share   quietest expert share")
for label, kw in [("no balancing", dict(method="none")), ("aux loss, alpha 0.01", dict(method="aux", alpha=0.01)),
                  ("aux loss, alpha 0.1", dict(method="aux", alpha=0.1)), ("bias update (loss-free)", dict(method="bias"))]:
    r = np.mean([train(seed=s, **kw) for s in range(3)], axis=0)
    print(f"{label:<24} {r[0]:9.4f}   {r[1]:19.3f}   {r[2]:21.3f}")
print(f"a perfectly even share is {1 / E:.3f} for every expert")
```

**Reading the output.** With no balancing the busiest expert takes 26.8 per cent of the slots, the quietest only 3.0 per cent (an even share is 12.5 per cent), and the task loss is 0.0177. A balance loss of 0.01 improves the spread slightly (20.5 and 6.0 per cent) at the same loss, 0.0180. A coefficient of 0.1 balances better (17.5 and 8.7 per cent) but the loss rises to 0.0290. The bias update gets 18.4 and 9.7 per cent with the lowest loss, 0.0171.

**What this does and does not show.** The pattern agrees with DeepSeek's argument: a strong auxiliary loss trades quality for balance and a bias does not. But this is a toy: 600 steps, 3 seeds, a made-up skewed dataset and one bias step size (0.003; in a separate run, not shown here, steps of 0.01 and 0.03 gave worse losses, 0.0233 and 0.0270). The data is skewed on purpose, because with evenly spread clusters the router balanced itself and no method was needed. It is evidence of the mechanism, not a ranking.

**Line by line.**

- `(logits + bias).topk(K)` picks the experts with the bias; `gate` is computed from the original `logits`, as in DeepSeek-V3.
- `bias -= gamma * torch.sign(share - 1.0 / E)` lowers the bias of experts above an even share and raises those below.
- `balance` is the Switch term, computed from the share of slots and the mean router probability.

### 3. The real library's balancing loss

We build a tiny Mixtral with the real Transformers classes (8 experts, top-2), take its router outputs and compare the library's `load_balancing_loss_func` with the formula from block 1.

```python
import torch
from transformers import MixtralConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

E, K, COEF = 8, 2, 0.02
CFG = MixtralConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, num_local_experts=E,
                    num_experts_per_tok=K, router_aux_loss_coef=COEF)


def switch_balance(router_logits):
    p = torch.cat(router_logits).softmax(-1)
    chosen = p.topk(K, dim=-1).indices
    counts = torch.bincount(chosen.reshape(-1), minlength=E).float()
    share = counts / counts.sum()
    return E * (share * p.mean(0)).sum()


torch.manual_seed(0)
model = MixtralForCausalLM(CFG).eval()
tokens = torch.randint(0, 512, (4, 32), generator=torch.Generator().manual_seed(1))
with torch.no_grad():
    plain = model(tokens, labels=tokens)
    routed = model(tokens, labels=tokens, output_router_logits=True)

lib = load_balancing_loss_func(routed.router_logits, E, K)
mine = switch_balance(routed.router_logits)
print(f"router logits per layer: {len(routed.router_logits)} tensors of shape {tuple(routed.router_logits[0].shape)} (tokens x experts)")
print(f"library balance loss {lib.item():.4f}   Switch formula {mine.item():.4f}   formula x k {K * mine.item():.4f}")
print(f"output aux_loss {routed.aux_loss.item():.4f}   equals the library function: {torch.allclose(routed.aux_loss, lib)}")
print(f"loss without routing outputs {plain.loss.item():.4f}   loss with them {routed.loss.item():.4f}   difference {routed.loss.item() - plain.loss.item():.4f}")
print(f"coefficient x aux loss = {COEF * lib.item():.4f}")

experts = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
total = sum(p.numel() for p in model.parameters())
active = total - experts * (E - K) / E
print()
print(f"tiny Mixtral: {total:,} parameters, {experts:,} of them in experts ({experts / total:.1%})")
print(f"each token uses {K} of {E} experts: {active:,.0f} active parameters ({active / total:.1%})")
```

**Reading the output.** The library's balance loss is 2.0059, exactly twice the Switch formula (1.0030), because the library counts every one of the k selections per token, so a perfectly even router scores k, here 2, not 1. The loss with routing outputs enabled is 6.2624, against 6.2223 without, and the difference of 0.0401 is the coefficient 0.02 times 2.0059. In this tiny model 81.1 per cent of parameters are in experts and 39.2 per cent are active.

**Line by line.**

- `output_router_logits=True` asks each layer to return its router scores, one tensor per layer, tokens by experts.
- `torch.cat(router_logits)` pools all layers' tokens, as the library does.
- `experts` counts parameters whose names contain `.experts.`, which in Transformers 5.18 are the fused expert weight tensors.

### 4. Total and active parameters of real models, from configs

We build four real mixture-of-experts models on the `meta` device from their Hub configs and count parameters. For four newer models whose classes are not in this Transformers release, we compute the routed-expert parameters from `config.json` alone. Card numbers are printed beside ours.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

COUNTED = {
    "mistralai/Mixtral-8x7B-v0.1": (8, 2, "47B / 13B"),
    "openai/gpt-oss-120b": (128, 4, "117B / 5.1B"),
    "deepseek-ai/DeepSeek-V3": (256, 8, "671B / 37B"),
    "Qwen/Qwen3-30B-A3B": (128, 8, "30.5B / 3.3B"),
}
print("counted on the meta device with the real Transformers classes (billions)")
print("model                          experts  top-k   total   in experts   active (all)   active (no input embedding)   card: total / active")
for repo, (n_exp, k, card) in COUNTED.items():
    config = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n and "shared" not in n)
    active = total - experts * (1 - k / n_exp)
    embedding = model.get_input_embeddings().weight.numel()
    print(f"{repo.split('/')[-1]:<30} {n_exp:>7} {k:>6} {total / 1e9:7.2f} {experts / 1e9:12.2f} {active / 1e9:14.2f} {(active - embedding) / 1e9:29.2f}   {card}")

ESTIMATED = {
    "deepseek-ai/DeepSeek-V4-Pro": ("n_routed_experts", "moe_intermediate_size", "hidden_size", 61, "1.6T / 49B"),
    "moonshotai/Kimi-K3": ("num_experts", "moe_intermediate_size", "routed_expert_hidden_size", 92, "2.8T / 104B"),
    "Aleph-Alpha/Kolibri-1": ("num_experts", "moe_intermediate_size", "hidden_size", 50, "78.1B / 3.46B"),
    "Qwen/Qwen3.6-35B-A3B": ("num_experts", "moe_intermediate_size", "hidden_size", 40, "35B / 3B"),
}
print()
print("models whose classes are not in this Transformers release: routed-expert parameters from config.json alone, 3 matrices per expert")
for repo, (e_key, f_key, h_key, layers, card) in ESTIMATED.items():
    c = json.load(open(hf_hub_download(repo, "config.json")))
    c = c.get("text_config", c)
    per_expert = 3 * c[h_key] * c[f_key]
    routed = layers * c[e_key] * per_expert
    print(f"{repo.split('/')[-1]:<30} {c[e_key]:>5} experts of {per_expert / 1e6:6.1f}M in {layers} layers = {routed / 1e9:8.1f}B   card total / active: {card}")
```

**Reading the output.** Counts match the cards: Mixtral 46.70 billion total and 12.88 active (card 47B and 13B), DeepSeek-V3 671.03 and 37.55 (671B and 37B), Qwen3-30B-A3B 30.53 and 3.35 (30.5B and 3.3B). For gpt-oss-120b the total is 116.83 (117B), and the active count is 5.71, or 5.13 without the input embedding, which is how the card's 5.1B arises. In the second table, routed experts alone account for most of each model: 1,547 billion of DeepSeek-V4-Pro's 1.6 trillion, 2,723 billion of Kimi K3's 2.8 trillion, 75.5 of Kolibri-1's 78.1 and 32.2 of Qwen3.6's 35.

**What this does not show.** The second table is an estimate: it assumes every layer is a mixture-of-experts layer (Kimi K3 has one dense layer, so 92 are counted), it uses Kimi K3's latent expert width (3,584) as the expert input size, and it ignores shared experts, attention and embeddings. It checks that the card's total is plausible, not the exact active count.

**Line by line.**

- `AutoModelForCausalLM.from_config` under `torch.device("meta")` builds the architecture with no weights in memory.
- `experts * (1 - k / n_exp)` is the number of expert parameters that a token does not use.
- `get_input_embeddings().weight.numel()` is the embedding size that some cards leave out of "active".

### 5. Expert parallelism on two real processes

Four experts, two per rank, 16 tokens per rank, top-2. Each rank sends token copies to the ranks that hold their experts with `all_to_all_single`, runs its experts, sends the results back, and combines them. The output is compared with one process running everything. We run it with an even router and with one that favours expert 0.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

E, K, D, TOKENS, WORLD = 4, 2, 16, 32, 2


def build(seed):
    torch.manual_seed(seed)
    router = nn.Linear(D, E, bias=False)
    experts = nn.ModuleList([nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, D)) for _ in range(E)])
    return router, experts


def route(router, x, skew):
    logits = router(x) + torch.tensor([skew, 0.0, 0.0, 0.0])
    top = logits.topk(K, dim=-1)
    return top.indices, top.values.softmax(-1)


def reference(router, experts, x, skew):
    chosen, gate = route(router, x, skew)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(K):
            out[t] += gate[t, s] * experts[chosen[t, s]](x[t])
    return out


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port, skew):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    router, experts = build(0)
    x_all = torch.randn(TOKENS, D, generator=torch.Generator().manual_seed(1))
    mine = slice(rank * TOKENS // world, (rank + 1) * TOKENS // world)
    x = x_all[mine]
    with torch.no_grad():
        chosen, gate = route(router, x, skew)
        slots = [(t, s) for t in range(x.shape[0]) for s in range(K)]
        slots.sort(key=lambda ts: int(chosen[ts]) // (E // world))
        dest = torch.tensor([int(chosen[ts]) // (E // world) for ts in slots])
        send_counts = torch.bincount(dest, minlength=world)
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        payload = torch.stack([x[t] for t, _ in slots])
        ids = torch.tensor([int(chosen[ts]) for ts in slots], dtype=torch.float32)[:, None]
        packet = torch.cat([payload, ids], dim=1)
        inbox = torch.zeros(int(recv_counts.sum()), D + 1)
        dist.all_to_all_single(inbox, packet, recv_counts.tolist(), send_counts.tolist())
        done = torch.zeros(len(inbox), D)
        for row, packet_row in enumerate(inbox):
            done[row] = experts[int(packet_row[-1])](packet_row[:-1])
        back = torch.zeros(len(slots), D)
        dist.all_to_all_single(back, done, send_counts.tolist(), recv_counts.tolist())
        out = torch.zeros_like(x)
        for row, (t, s) in enumerate(slots):
            out[t] += gate[t, s] * back[row]
        err = (out - reference(router, experts, x_all, skew)[mine]).abs().max().item()
    stats = torch.tensor([err, float(send_counts[0]), float(send_counts[1]), float(recv_counts.sum())], dtype=torch.float64)
    table = [torch.zeros(4, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(table, stats)
    if rank == 0:
        print(f"router favours expert 0 by {skew}")
        for r, t in enumerate(table):
            print(f"  rank {r}: sends {int(t[1])} to rank 0 and {int(t[2])} to rank 1, receives {int(t[3])}, max error {t[0].item():.2e}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    for skew in (0.0, 2.0):
        mp.spawn(worker, args=(WORLD, free_port(), skew), nprocs=WORLD, join=True)
```

**Reading the output.** The result matches a single process exactly (maximum error 0.00e+00) in both runs. With the favouring router, rank 0 sends 21 of its 32 token copies to itself and 11 to rank 1, and rank 1 sends 19 to rank 0 and 13 to itself, so rank 0 receives 40 and rank 1 only 24. The even mean is 32, so rank 0 is 25 per cent over it, and the step runs at the speed of rank 0, leaving rank 1 idle for 40 per cent of the time. With the even router, the two ranks receive 27 and 37, because 4 experts and 32 slots are too few for a perfectly even split.

**Line by line.**

- `send_counts` and `recv_counts` are exchanged first with a small all-to-all, so each rank knows how big a message to expect.
- `all_to_all_single(inbox, packet, recv_counts, send_counts)` takes the output and input split sizes: how many rows come from each rank, and how many go to each.
- The expert id travels with each token as an extra column, so the receiver knows which expert to run.

### Try it yourself

The lab is block 1's simulation with every setting exposed, and the same generator, so its numbers equal the printed ones. Its defaults (16 experts, top 2, 1,024 tokens, capacity factor 1.25, skew 0.5) reproduce the row 1.5836, 3.72 times the average, 28.5 per cent dropped and 57.2 per cent capacity used. The rounding of the numbers and the routing code were checked against a Python run on six other settings with a maximum difference of 2.2e-16.

<MoeRoutingLab />

**What each control does.**

- **experts** is the number of experts in the layer, and **experts per token (top-k)** is how many each token uses.
- **tokens** is the number of tokens in the batch.
- **capacity factor** sets the room per expert, as a multiple of an even share.
- **router skew** gives experts fixed favourites; 0 is a router with no favourites.
- The dashed line is capacity, the dotted line the even share; orange is what each expert drops.

**Try it yourself.**

1. Set **router skew** to 0. The balance term falls to 1.0019, the busiest expert is 1.10 times the average, nothing is dropped, and capacity used is 80.0 per cent. Why: with no favourites the experts share the load, and a capacity of 1.25 times an even share is 80 per cent full.
2. Go back to skew 0.5 and move **capacity factor** to 3.00. Drops fall from 28.5 to 4.5 per cent but capacity used falls from 57.2 to 31.8 per cent. Why: more room stops the drops but leaves most of the reserved slots empty, which is wasted memory and compute.
3. Set **experts** to 64 with the other defaults. The balance term rises to 1.7509, the busiest expert to 7.84 times the average, and 33.3 per cent of slots are dropped. Why: with more experts each gets fewer tokens, so chance and favourites matter more. This is why balancing methods matter more as expert counts grow.

## Production snippets (not run here)

:::warning Not run in this environment
These need a GPU with about 80 GB of memory for gpt-oss-120b (the model card says it fits on one 80 GB GPU). The snippet follows the model card, with the argument name `dtype` that Transformers 5 uses.
:::

Loading a mixture-of-experts model is no different from loading a dense one; the library handles the routing:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="openai/gpt-oss-120b", dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "Explain quantum mechanics clearly and concisely."}]
outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
```

## Designing with it

1. **Plan memory with total parameters and compute with active ones.** A 2.8-trillion-parameter model with 104 billion active still needs the memory for 2.8 trillion.
2. **Compare models by both numbers, and by how the card counts.** Block 4 shows the embedding convention moving gpt-oss-120b from 5.71 to 5.13 billion.
3. **Watch the load, not only the loss.** Log the share of tokens per expert and the drop rate during training. Block 2 shows the loss can look fine while one expert gets 27 per cent of the traffic.
4. **Prefer gentle balancing.** A large auxiliary loss costs quality (block 2); a bias update or a small coefficient costs less.
5. **Budget capacity on purpose.** Capacity factor 1.25 means 20 per cent of slots empty even when perfectly balanced (block 1).
6. **Keep the router in higher precision.** The Switch paper casts the router to float32 selectively, and the Kolibri-1 card keeps the router in bf16 while the experts are fp8.
7. **Choose the parallel layout for the routing.** Expert parallelism turns every layer into two all-to-alls; keep them on fast links, as DeepSeek-V3's node limit does.

## Where this stands in 2026

:::info Industry view
- **Mixture of experts is the common design in the large open models listed in the table above.** The cards for DeepSeek-V4, Kimi K3, Qwen3.6, gpt-oss and Kolibri-1 all describe mixture-of-experts models; read on 7 and 8 October 2026.
- **The recipe is moving.** More and smaller experts, shared experts, sigmoid or square-root-softplus scoring in several configs, and bias-based balancing. The Kolibri-1 card names its own balancing method (Exact Quantile Balancing); this chapter did not study it beyond what the card states.
- **Low-bit experts are normal.** Cards report FP4, MXFP4 or fp8 experts for DeepSeek-V4, Kimi K3, gpt-oss and Kolibri-1.
- **Not verified here:** benchmark claims in any card, GPU throughput, and the exact layer layouts of the 2026 models beyond what each config.json shows. The 2026 models were not downloaded or run.
:::

## Common mistakes

1. **Reading "8x7B" as 56 billion.** The experts share attention and embeddings, so Mixtral has 46.70 billion. The names are marketing, not arithmetic. Count from the config, as block 4 does.
2. **Sizing the GPU by active parameters.** It feels right because that is what a token uses. But every expert must be resident. Use total parameters for weights.
3. **Assuming experts are subject specialists.** The Mixtral paper found no obvious topic pattern in routing. Routing is more syntactic than semantic.
4. **Dropping the balancing and hoping.** Without it, block 2's quietest expert got 3 per cent of the slots. In a real run that is wasted capacity, and under expert parallelism it is a slow GPU.
5. **Setting the balancing loss very high.** It balances, but block 2 shows the loss rises sharply. Raise it gently or use a bias.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Mixtral 8x7B has 46.70 billion parameters, 45.10 billion of them in experts. Each token uses 2 of 8. How many parameters are active?</summary>

The experts are used at 2/8, so 45.10 x 0.25 = 11.27 billion. The remaining 46.70 - 45.10 = 1.60 billion (attention, embeddings, router) are always used. Active = 11.27 + 1.60 = 12.88 billion, 27.6 per cent of the total.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Only 2 of 8 experts run for a token. Why must all 8 stay in GPU memory?</summary>

The router picks experts per token, and a batch contains many tokens that pick different experts. The next token's choice is unknown until the router runs, so any expert may be needed at any moment. Loading them on demand would be far slower than keeping them resident.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> In block 1's example, what would the balance term be if the tokens were split 2, 2, 2, 2 and the average router probabilities were 0.25 each? What would the capacity factor 1.0 drop?</summary>

f = 0.25 each and P = 0.25 each, so the sum of f x P is 4 x 0.0625 = 0.25, and N x 0.25 = 1.0, the perfectly balanced value. A capacity of 2 per expert holds exactly 2 tokens each, so nothing is dropped and all 8 slots are used.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> With 1,024 tokens, top-2, 16 experts and a capacity factor of 1.25, what is the capacity, and how many slots does the busiest expert drop at skew 0.5 (3.72 times the average)?</summary>

Capacity = 1.25 x 1,024 x 2 / 16 = 160. The even share is 128 slots, and the busiest expert received 3.72 x 128 = 476, so it drops 316 slots, nearly two thirds of what it was sent. This is why one hot expert hurts so much.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> DeepSeek-V3 adds a bias to the scores for choosing experts but not to the gate weights. Why is that important, and how big can the bias get with a step of 0.001?</summary>

If the bias also scaled the outputs, balancing would change what the model computes and compete with the training objective, which is the cost of an auxiliary loss. Using it only to choose experts leaves the output weights as the model learned them. An expert that stays overloaded has its bias lowered by 0.001 every step, so after 1,000 steps it can be up to 1.0 lower, comparable to the spread of router scores. That is how fast the correction can act.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> In block 5 with the favouring router, rank 0 receives 40 slots and rank 1 receives 24. How much of the cluster's capacity is wasted, and what are two ways to fix it?</summary>

The step takes as long as rank 0, which does 40 units of work. The mean is 32, so the efficiency is 32 / 40 = 80 per cent; rank 1 is idle for (40 - 24) / 40 = 40 per cent of the step. Fixes: balance the router (a loss, a bias, or both) so the experts get an even load, or cap each expert with a capacity factor so no rank gets more than a set share, at the price of dropped tokens.

</details>

## Go deeper

All sources were opened on 7 and 8 October 2026.

- [Mixtral of Experts (arXiv 2401.04088)](https://arxiv.org/abs/2401.04088): top-2 routing over 8 SwiGLU experts, 47B total and 13B active, and the routing analysis that found no topic pattern.
- [Switch Transformers (arXiv 2101.03961)](https://arxiv.org/abs/2101.03961): the balancing loss, the coefficient 0.01, capacity factor, dropped tokens and selective float32 precision.
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (arXiv 1701.06538)](https://arxiv.org/abs/1701.06538) and [GShard (arXiv 2006.16668)](https://arxiv.org/abs/2006.16668): the earlier work (titles and abstracts opened only).
- [DeepSeekMoE (arXiv 2401.06066)](https://arxiv.org/abs/2401.06066): fine-grained and shared experts (title and abstract opened only).
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (arXiv 2408.15664)](https://arxiv.org/abs/2408.15664): the bias method (title and abstract opened only).
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437): the bias update, the sequence-level balance loss, node-limited routing and the 64-GPU expert layout.
- Model cards and `config.json` files on the Hugging Face Hub, read for block 4: `mistralai/Mixtral-8x7B-v0.1`, `openai/gpt-oss-120b`, `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V4-Pro`, `moonshotai/Kimi-K3`, `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3.6-35B-A3B`, `Aleph-Alpha/Kolibri-1`, `meta-llama/Llama-4-Scout-17B-16E-Instruct` (card only).
- [The Ultra-Scale Playbook (Hugging Face, 2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): the expert-parallel section and its note on DeepSeek-V3's node limit.

## Check yourself

- I can explain what a router, a top-k choice and an expert are, and why a mixture-of-experts model has more parameters than it uses.
- I can compute total and active parameters from a config and say how a model card counted them.
- I can compute the balance term and a capacity from router probabilities by hand.
- I can describe routing collapse and compare a balancing loss with a bias update and a capacity limit.
- I can explain dispatch and combine in expert parallelism and why the busiest rank sets the speed.
- I can say why memory is sized by total parameters and compute by active ones.

## Where to go next

This is the last chapter of the training-at-scale group. Related chapters: [quantisation for inference](/docs/llm-engineering/quantisation-for-inference), for storing expert weights in few bits when serving, and [continuous batching and scheduling](/docs/llm-engineering/continuous-batching-and-scheduling), where the serving engine batches tokens bound for different experts. To revisit the layout side, return to [parallelism strategies for LLMs](/docs/llm-engineering/parallelism-strategies-for-llms).
