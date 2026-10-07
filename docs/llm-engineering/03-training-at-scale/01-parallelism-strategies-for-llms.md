---
id: llme-parallelism
title: "Parallelism Strategies for LLMs"
sidebar_label: "1 · Parallelism strategies"
sidebar_position: 1
slug: /llm-engineering/parallelism-strategies-for-llms
description: "Data, tensor, pipeline, context and expert parallelism, how they combine into the 3D and 4D layouts that train frontier models, and a per-GPU memory model checked against the real Llama 3.1 config and the published 405B layout."
tags: [parallelism, tensor-parallel, pipeline-parallel, context-parallel, sequence-parallel, 3d-parallelism, llama-3, megatron-lm, memory]
---

import Infographic from '@site/src/components/Infographic';
import ParallelismMemoryLab from '@site/src/components/viz/ParallelismMemoryLab';

**In one line.** A large language model is trained by cutting the job along several axes at once (examples, weight matrices, layers, tokens, experts), and the craft is choosing the cuts so that what must fit in one GPU's memory fits and the chatty communication stays on the fastest links.

:::tip Before you start
- **You should already know** what a training step is and why workers average gradients ([data parallelism](/docs/mlops/distributed/data-parallelism)), how a network can be cut by layer or by matrix ([model parallelism](/docs/mlops/distributed/model-parallelism)), and why plain training costs 16 bytes per parameter ([why distribute machine learning](/docs/mlops/distributed/why-distribute-machine-learning)).
- **You should be able to picture** one decoder layer: attention, then a feed-forward block ([transformer decoder architecture](/docs/theory/dnn/transformer-decoder-architecture)).
- **Reading time:** about 40 minutes, plus a few minutes to run the code.
- **After this chapter you can** work out how much memory one GPU needs for a given layout, say which of the five axes fixes which problem, and read a published training configuration such as Llama 3's.
:::

:::note Not from a lecture
This chapter was written for this site from the sources listed under Go deeper. Model sizes come from real `config.json` files read from the Hugging Face Hub by the code below. Versions used: Python 3.14, PyTorch 2.14.1 (CPU), Transformers 5.18.0. Sources were opened on 7 October 2026.
:::

## In 30 seconds

A modern language model has hundreds of billions of numbers to store, and every number needs several copies while it trains. No single graphics card (GPU) can hold that. So we cut the work up and spread it over thousands of GPUs.

Think of printing a huge encyclopaedia. You can add presses that each print different copies, cut every plate in half so two presses share it, or set up an assembly line where each press does some chapters. Each choice solves one problem and makes a new one. This chapter names the five cuts and counts what each costs.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Parameter | One learned number inside the model | Llama 3.1 405B has 405,853,388,800 |
| HBM (high-bandwidth memory) | The fast memory on the GPU card | 80 GB on the H100 cards Llama 3 used |
| Activation | An intermediate result saved during the forward pass so the backward pass can use it | The output of each layer for each token |
| Rank | The number of one process, usually one per GPU | Rank 0 of 8 |
| All-reduce | Every rank contributes a tensor and every rank receives the sum | Eight gradients become one shared gradient |
| Shard | One piece of a tensor that is split across ranks | Each of 8 GPUs holds one eighth of a matrix |
| Micro-batch | A slice of the batch that is pushed through the model on its own | 1 sequence of 8,192 tokens |
| Bubble | Time a GPU sits idle while a pipeline fills and drains | The last stage waits for the first |
| MFU (model FLOPs utilisation) | Useful arithmetic done, as a share of what the hardware could do | Llama 3 reports 38 to 43 per cent |

## The idea in plain words

Start with the smallest case you can do by hand. A model with 1 billion parameters needs about 16 GB while training with the Adam optimiser: 2 bytes for the weight, 2 for its gradient and 12 for the optimiser's copies (explained below). One 80 GB card holds it easily. Now take a model with 10 billion parameters. It needs 160 GB, so it no longer fits on one card, however fast the card is.

You have two broad remedies. **Split the model** so each GPU holds only part of it. Or **split the examples** so many GPUs work at once, each with a full copy. Splitting the model fixes the memory wall. Splitting the examples fixes the speed wall but not the memory wall, because every copy is still full size.

Real runs need both, and there is more than one way to split a model. You can cut each weight matrix in half, cut the stack of layers into stages, cut a long input into pieces, or give different experts to different GPUs. That makes five axes in all.

| Axis | What it splits | What it sends | Where it lives |
| --- | --- | --- | --- |
| Data parallel (DP) | The batch of examples | Gradients, once per step | Across servers |
| Tensor parallel (TP) | Each weight matrix inside a layer | Partial results, several times in every layer | Inside one server |
| Pipeline parallel (PP) | The stack of layers into stages | Activations between neighbouring stages | Across servers |
| Context parallel (CP) | The tokens of one long sequence | Keys and values | Where inputs are very long |
| Expert parallel (EP) | The experts of a mixture-of-experts layer | Tokens, to the expert that will process them | Only for MoE models |

No axis is free. Data parallelism keeps a full copy of the model on every replica, so on its own it cannot help when the model does not fit. Tensor parallelism makes the model fit but talks inside every layer. Pipeline parallelism talks little but leaves GPUs idle while the line fills. A real run combines them: data, tensor and pipeline together are called 3D parallelism, and adding context parallelism gives 4D.

<Infographic src="/img/llme/parallelism-strategies-for-llms-five-axes.svg" alt="Five columns, one per parallelism axis, each saying what it splits and what it communicates, above three cards that map a memory or speed problem to the axis that fixes it" caption="Look at the bottom row first: it says which wall you are hitting. The columns above say which cut removes it. Expert parallelism returns in chapter 4, and sharded data parallelism (ZeRO and FSDP) is chapter 2." />

## Worked example, step by step

We will fit a 70-billion-parameter model on 64 GPUs, using tensor parallel 8, pipeline parallel 2 and data parallel 4 (8 x 2 x 4 = 64). The code in block 2 prints every number below.

1. **Count the parameters.** The config of Llama 3.1 70B says 80 layers, hidden size 8,192, feed-forward size 28,672, 64 attention heads, 8 key-value heads and a vocabulary of 128,256. One layer holds 855,654,400 parameters. Eighty layers give 68,452,352,000. The input and output embeddings add 2,101,346,304, and the final norm adds 8,192. The total is 70,553,706,496.
2. **Count the bytes.** Plain mixed-precision training stores a 2-byte weight, a 2-byte gradient and 12 bytes of optimiser state per parameter: a 4-byte master copy of the weight plus Adam's two running averages (4 bytes each). That is 16 bytes. 70.55 billion x 16 = 1,128.9 GB.
3. **Cut across the model-parallel GPUs.** Tensor 8 x pipeline 2 means 16 GPUs share one copy of the model. 1,128.9 / 16 = 70.55 GB each. That is the "ZeRO stage 0" row, with no sharding across replicas. It is nearly the whole of an 80 GB card before a single activation.
4. **Shard the optimiser state across the 4 replicas (ZeRO stage 1).** The optimiser share is 12 x 70.55 / 16 = 52.92 GB. Divide by 4 and you get 13.23 GB. The total falls to 8.82 + 8.82 + 13.23 = 30.87 GB.
5. **Shard the gradients too (stage 2).** Gradients drop from 8.82 GB to 2.20 GB. Total 24.25 GB.
6. **Shard the weights too (stage 3).** Weights drop to 8.82 / 4 = 2.21 GB, plus 0.21 GB for the one layer that is gathered while it is being used. Total 17.85 GB.

<Infographic src="/img/llme/parallelism-strategies-for-llms-worked-70b.svg" alt="Four steps that count a 70B model's parameters, multiply by 16 bytes, divide by 16 model-parallel GPUs and then shard across 4 replicas, beside bars of per-GPU memory for ZeRO stages 0 to 3 against an 80 GB marker" caption="Read the left column top to bottom, then watch the purple optimiser bar shrink in the right panel. Activations come on top of these bars." />

Notice what moved. Tensor and pipeline parallelism divide all three parts equally. Sharding across replicas, which chapter 2 develops, divides them one at a time. That is the whole memory story for the model itself. The remaining part, activations, behaves differently, and the next section covers it.

## How it works

### What fills a GPU's memory?

Training memory has four parts. The first three belong to the model: weights, gradients and optimiser state. They scale with the number of parameters, and the 16-bytes-per-parameter figure holds for the common recipe (bf16, a 2-byte format with a wide range, for weights and gradients, and fp32 master weights and Adam moments). The Hugging Face Ultra-Scale Playbook gives the same 2 + 2 + 4 + 8 accounting. The fourth part, activations, scales with the batch and the sequence length instead.

The Megatron-LM team's analysis of activations gives the bytes per layer for sequence length `s`, micro-batch `b`, hidden size `h`, `a` attention heads and tensor-parallel size `t`:

| Setting | Activation bytes per layer |
| --- | --- |
| Nothing saved or shared | `sbh(34 + 5as/h)` |
| Tensor parallel | `sbh(10 + 24/t + 5as/(ht))` |
| Tensor + sequence parallel | `sbh/t (34 + 5as/h)` |
| Tensor + sequence parallel + selective recomputation | `34 sbh / t` |
| Full recomputation | `2 sbh` |

In words: the `5as/h` term is the attention score matrices, which grow with the square of the sequence length. **Selective recomputation** throws those matrices away and recomputes them in the backward pass, because they are cheap to recompute and large to store. **Sequence parallelism** (not the same as context parallelism) splits layer norm and dropout along the tokens, because they work on each token separately and need no copy of the whole sequence on every tensor-parallel GPU.

The first pipeline stage holds the activations of `p` micro-batches in flight, and each stage owns `L/p` layers. So it holds the equivalent of all `L` layers at once, whatever `p` is. That is why block 3 multiplies the per-layer figure by the full layer count.

### What does tensor parallelism cost?

A transformer layer is cut so that each GPU holds a slice of the attention heads and a column slice of each feed-forward matrix. The Megatron-LM paper reports only two all-reduces in the forward pass and two in the backward pass per layer. That sounds small, but it is four collective operations in every layer, thousands of times per step.

Block 4 turns that into bytes for the Llama 3 405B layout. A GPU sends about 237 GB per step for tensor parallelism, against about 12.6 GB for data parallelism and 8.6 GB for pipeline parallelism. That ratio is why the Megatron follow-up paper says to use tensor parallelism up to the number of GPUs in a server and only then add pipeline parallelism. The Ultra-Scale Playbook measured the cliff: it saw "significant drops" going from TP 8 to TP 16, because the second group of eight GPUs sits in another server on a slower link. Each GPU needs at least one attention head to work on, so the tensor-parallel size has an upper limit set by the head count.

Block 5 shows the mechanism on a real `LlamaMLP`. Each rank holds half of the gate and up columns and half of the down rows. Alone, each rank's answer is wrong by about 0.2. After one all-reduce, the sum matches the unsplit layer to 1.34e-07.

### What does pipeline parallelism cost?

Think of a laundrette line: wash, dry, fold. While the first load is in the dryer, the folder has nothing to do. That idle time is the **bubble**. With `p` stages and `m` micro-batches in the plain schedule, the idle fraction is `(p - 1)/(m + p - 1)`. In words: more micro-batches mean the line stays full for longer, so the idle ends are a smaller part of the whole.

With 16 stages and 16 micro-batches, nearly half the time is idle (15/31 = 0.484). With 128 micro-batches it is 0.105. **Interleaved schedules** give each GPU several small, non-adjacent chunks of layers, which cuts the bubble roughly by the number of chunks `V`. Llama 3 uses one and writes the overall ratio as `(PP - 1)/(V x M)`. The paper does not give `V` or `M`, so block 4 shows `V = 4` as an illustration only.

### What does context parallelism do for long inputs?

When inputs reach 128K tokens, even one sequence's activations do not fit. The fix is to split the tokens. Llama 3 uses an all-gather of keys and values: each context-parallel rank keeps its own queries, gathers everyone's keys and values, and computes attention for its own chunk. The paper notes this works because grouped-query attention makes keys and values much smaller than queries.

Causal attention makes the work lopsided, because a late token attends to many more tokens than an early one. So the paper cuts the sequence into `2 x CP` chunks and gives rank `i` chunks `i` and `2 x CP - 1 - i`. Block 6 runs this on two processes and counts the work.

### How do the degrees fit together?

The product of the degrees is the number of GPUs: `TP x CP x PP x DP`. The Llama 3 paper orders the dimensions as [TP, CP, PP, DP], innermost to outermost, so the most talkative axis gets the best network. The Ultra-Scale Playbook turns this into a decision order. First fit one model instance (under 10B parameters, one technique such as ZeRO-3 over 8 GPUs may do), then reach the target batch size with data parallelism, then tune throughput. For 10B to 100B parameters it suggests TP 8 with pipeline parallelism or with ZeRO-3. At 512 GPUs and more, pure data parallelism becomes communication-bound. At 1,024 GPUs and more, it suggests TP 8 with ZeRO-2 and pipeline parallelism. It also says you must run experiments on your own cluster.

Expert parallelism, the fifth axis, only applies to mixture-of-experts models and gets its own chapter: [mixture of experts](/docs/llm-engineering/mixture-of-experts).

## A real system that works this way

Meta's Llama 3 paper (v3, 23 November 2024) describes the largest open recipe. The 405B model has 126 layers, hidden size 16,384, 128 attention heads and 8 key-value heads. It trained on up to 16,384 H100 GPUs with 80 GB of HBM each, using 4D parallelism. Table 4 of the paper gives three configurations, each with 16M tokens per batch:

| GPUs | TP | CP | PP | DP | Sequence length | BF16 MFU |
| --- | --- | --- | --- | --- | --- | --- |
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 38% |

The paper also says its FSDP shards optimiser state and gradients but does not reshard the weights after the forward pass, which avoids an extra all-gather in the backward pass. In ZeRO terms the weights then behave as in stage 2 during a step, because the full weights stay in memory from the forward pass until the backward pass is done with them. It pre-trained at 8K tokens without activation checkpointing, after removing the first and last transformer layer from the first and last pipeline stages to balance the load.

<Infographic src="/img/llme/parallelism-strategies-for-llms-llama3-layout.svg" alt="The 16,384 GPU Llama 3 405B layout as nested data, pipeline and tensor groups, with a table of model state per ZeRO stage and a table of activation memory per saving that sums to 78.6 GB" caption="Start with the green box at the bottom right: states plus activations come to 78.6 GB, just under the 80 GB card. Every figure is printed by blocks 2 and 3." />

## Code you can run

Six blocks. Blocks 1 to 4 read real configs from the Hub and do arithmetic. Blocks 5 and 6 start two real processes with PyTorch's gloo backend on the CPU. Everything runs in about a minute after the config downloads.

### 1. Count parameters from real configs

We read three Llama 3.1 configs from the Hub, count the parameters with a formula, and check the formula against the real `LlamaForCausalLM` built on the `meta` device, which allocates no memory even for 405B. The 405B config comes from the Hermes 3 fine-tune, which keeps the architecture of Llama 3.1 405B, because Meta's own repository is gated.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from huggingface_hub import hf_hub_download
from transformers import LlamaConfig, LlamaForCausalLM

REPOS = {
    "8B": "NousResearch/Meta-Llama-3.1-8B",
    "70B": "NousResearch/Meta-Llama-3.1-70B",
    "405B": "NousResearch/Hermes-3-Llama-3.1-405B",
}


def formula(c):
    kv_dim = c["hidden_size"] * c["num_key_value_heads"] // c["num_attention_heads"]
    attn = 2 * c["hidden_size"] ** 2 + 2 * c["hidden_size"] * kv_dim
    mlp = 3 * c["hidden_size"] * c["intermediate_size"]
    per_layer = attn + mlp + 2 * c["hidden_size"]
    total = 2 * c["vocab_size"] * c["hidden_size"] + c["num_hidden_layers"] * per_layer + c["hidden_size"]
    return total, per_layer


def counted(c):
    cfg = LlamaConfig(**{k: c[k] for k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "tie_word_embeddings")})
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    return sum(p.numel() for p in model.parameters())


print("name   layers  hidden    ffn  heads  kv  vocab    formula (B)  transformers (B)  equal")
configs = {}
for name, repo in REPOS.items():
    c = json.load(open(hf_hub_download(repo, "config.json")))
    configs[name] = c
    f, _ = formula(c)
    n = counted(c)
    print(f"{name:<5} {c['num_hidden_layers']:>6} {c['hidden_size']:>7} {c['intermediate_size']:>6} "
          f"{c['num_attention_heads']:>6} {c['num_key_value_heads']:>3} {c['vocab_size']:>6} "
          f"{f / 1e9:>13.3f} {n / 1e9:>17.3f}  {f == n}")

big = formula(configs["405B"])[0]
print()
print(f"405B parameters: {big:,}")
print(f"405B model states at 16 bytes per parameter: {big * 16 / 1e12:.2f} TB")
print(f"80 GB cards needed just for those states: {int(big * 16 / 80e9) + 1}")
paper_vocab = dict(configs["405B"], vocab_size=128000)
print(f"with the paper's rounded vocabulary of 128,000: {formula(paper_vocab)[0]:,}")
```

**Reading the output.** The formula and the library agree to the last digit on all three sizes. The 405B model has 405,853,388,800 parameters, so 16 bytes each is 6.49 TB. That needs at least 82 cards of 80 GB for model state alone. The paper's Table 3 says the vocabulary is 128,000; the real config says 128,256. That difference moves the total by only 8.4 million parameters, 0.002 per cent.

**Line by line.**

- `with torch.device("meta")` builds the model with shapes but no storage, so counting parameters is instant.
- `kv_dim` is the width of the key and value projections: with 8 key-value heads and 128 query heads it is one sixteenth of `hidden_size`.
- `2 * vocab_size * hidden_size` counts the input embedding and the separate output head, because `tie_word_embeddings` is false.

### 2. Check the layout and the model state per GPU

We check that Table 4's rows multiply out, then compute per-GPU model state for each ZeRO stage, first for the 70B worked example and then for the 405B layout.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download

GB = 1e9


def load(repo):
    return json.load(open(hf_hub_download(repo, "config.json")))


def layer_params(c):
    h = c["hidden_size"]
    kv_dim = h * c["num_key_value_heads"] // c["num_attention_heads"]
    return 2 * h * h + 2 * h * kv_dim + 3 * h * c["intermediate_size"] + 2 * h


def params(c):
    return 2 * c["vocab_size"] * c["hidden_size"] + c["num_hidden_layers"] * layer_params(c) + c["hidden_size"]


def states(c, t, p, d, zero):
    n = params(c)
    weights, grads, opt = 2 * n / (t * p), 2 * n / (t * p), 12 * n / (t * p)
    if zero >= 1:
        opt /= d
    if zero >= 2:
        grads /= d
    if zero >= 3:
        weights = weights / d + 2 * layer_params(c) / t
    return weights / GB, grads / GB, opt / GB


c70 = load("NousResearch/Meta-Llama-3.1-70B")
print(f"70B: {params(c70):,} parameters, {16 * params(c70) / GB:,.1f} GB of states at 16 bytes each")
print("70B on 64 GPUs as TP 8 x PP 2 x DP 4, per-GPU states in GB")
for z in range(4):
    w, g, o = states(c70, 8, 2, 4, z)
    print(f"  ZeRO {z}: weights {w:6.2f}  grads {g:6.2f}  optimiser {o:6.2f}  total {w + g + o:6.2f}")

c405 = load("NousResearch/Hermes-3-Llama-3.1-405B")
rows = [(8192, 8, 1, 16, 64, 32, 8192), (16384, 8, 1, 16, 128, 16, 8192), (16384, 8, 16, 16, 8, 16, 131072)]
print()
print("Llama 3 405B, Table 4 of the paper: do the layouts multiply out?")
print("GPUs   TP  CP  PP   DP  seq      tokens per batch  GPUs ok")
for gpus, t, cp, p, d, bs, s in rows:
    print(f"{gpus:>5}  {t:>2}  {cp:>2}  {p:>2}  {d:>3}  {s:>6}  {d * bs * s:>16,}  {t * cp * p * d == gpus}")

print()
print("405B on 16,384 GPUs as TP 8 x PP 16 x DP 128, per-GPU states in GB")
for z in range(4):
    w, g, o = states(c405, 8, 16, 128, z)
    print(f"  ZeRO {z}: weights {w:6.2f}  grads {g:6.2f}  optimiser {o:6.2f}  total {w + g + o:6.2f}")
```

**Reading the output.** The 70B rows are the worked example: 70.55, 30.87, 24.25 and 17.85 GB. All three published layouts multiply out to their GPU counts and to exactly 16,777,216 tokens per batch, which the paper rounds to 16M. For the 405B layout on 16,384 GPUs, state falls from 50.73 GB with no sharding to 6.69 GB at stage 2, which is what Llama 3 does.

**Line by line.**

- `states` divides all three parts by `t * p`, then divides optimiser, gradients and weights by `d` as the ZeRO stage rises.
- At stage 3 it adds `2 * layer_params / t`: the one layer that is gathered into full size while it computes.
- The embeddings and output head are not spread evenly over pipeline stages in a real run, so this is an average, not a per-stage truth.

### 3. Activations for each saving, and does it fit?

We apply the five activation formulas to the 405B layout at 8,192 tokens, add them to the model state of block 2 and compare with the 80 GB card.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download

GB = 1e9
c = json.load(open(hf_hub_download("NousResearch/Hermes-3-Llama-3.1-405B", "config.json")))
L, H, A = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"]
kv_dim = H * c["num_key_value_heads"] // A
layer = 2 * H * H + 2 * H * kv_dim + 3 * H * c["intermediate_size"] + 2 * H
P = 2 * c["vocab_size"] * H + L * layer + H


def activations(s, b, t, cp, mode):
    sl = s / cp
    sbh = sl * b * H
    attn_scores = 5 * A * sl / H
    per_layer = {
        "nothing saved": sbh * (34 + attn_scores),
        "tensor parallel": sbh * (10 + 24 / t + attn_scores / t),
        "tensor + sequence parallel": sbh / t * (34 + attn_scores),
        "+ selective recomputation": sbh * 34 / t,
        "full recomputation": 2 * sbh,
    }[mode]
    return per_layer * L


modes = ["nothing saved", "tensor parallel", "tensor + sequence parallel", "+ selective recomputation", "full recomputation"]
t, p, d = 8, 16, 128
print("activations on the first pipeline stage, 8,192 tokens, micro-batch 1, TP 8 (GB)")
for mode in modes:
    print(f"  {mode:<28} {activations(8192, 1, t, 1, mode) / GB:9.1f}")

states = 2 * P / (t * p) + 2 * P / (t * p) / d + 12 * P / (t * p) / d
acts = activations(8192, 1, t, 1, "+ selective recomputation")
print()
print(f"ZeRO 2 states {states / GB:.1f} GB + activations {acts / GB:.1f} GB = {(states + acts) / GB:.1f} GB against 80 GB of HBM")
a_long = activations(131072, 1, 8, 16, "+ selective recomputation")
print(f"CP 16 at 131,072 tokens: {a_long / GB:.1f} GB, each rank holds {131072 // 16:,} tokens")
print(f"without context parallelism at 131,072 tokens: {activations(131072, 1, 8, 1, '+ selective recomputation') / GB:.1f} GB")
```

**Reading the output.** With nothing saved, activations would take 5,986.6 GB on the first stage. Tensor and sequence parallelism bring that to 748.3 GB. Only selective recomputation gets it to 71.9 GB. Add 6.7 GB of stage 2 state and the total is 78.6 GB against 80 GB. Without context parallelism the 131,072-token case would need 1,150 GB. With CP 16, each rank holds 8,192 tokens and the term returns to 71.9 GB.

**What did not work.** Treat the 78.6 GB as a sanity check, not a measurement. The Megatron formula was derived for a GPT-style layer with a 4h feed-forward block. Llama uses a three-matrix SwiGLU block and grouped-query attention, so the real byte counts differ. The estimate also leaves out the output logits, communication buffers and fragmentation. The lab below shows the third published layout giving 83.8 GB with this formula, over the card, even though Meta trained it. The formula is a guide to which saving matters, not a promise that a layout fits.

### 4. Bubbles and bytes on the wire

First the pipeline bubble for different micro-batch counts. Then the bytes each GPU sends in one step for tensor, pipeline and data parallelism in the 405B layout, counting bf16 activation-sized messages.

```python
import json
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import hf_hub_download

c = json.load(open(hf_hub_download("NousResearch/Hermes-3-Llama-3.1-405B", "config.json")))
L, H, FFN, VOCAB = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"], c["vocab_size"]
KV = H * c["num_key_value_heads"] // c["num_attention_heads"]
P = 2 * VOCAB * H + L * (2 * H * H + 2 * H * KV + 3 * H * FFN + 2 * H) + H

print("pipeline bubble, p = 16 stages")
print("micro-batches m   idle fraction (p-1)/(m+p-1)   bubble / ideal (p-1)/m   interleaved V=4: bubble / ideal")
for m in (16, 32, 64, 128):
    print(f"{m:>15}   {15 / (m + 15):>27.3f}   {15 / m:>22.3f}   {15 / (4 * m):>31.3f}")

t, p, d, seq, seqs_per_rank = 8, 16, 128, 8192, 16
tokens = seq * seqs_per_rank
message = tokens * H * 2
layers_per_stage = L / p
ring = 2 * (t - 1) / t
tp_bytes = 4 * layers_per_stage * ring * message
pp_bytes = 2 * message
dp_bytes = 2 * (2 * P / (t * p)) * (d - 1) / d
print()
print(f"one activation tensor for a DP rank's {tokens:,} tokens, bf16: {message / 1e9:.2f} GB")
print("bytes a single GPU sends per training step, 405B layout (GB)")
print(f"  tensor parallel   {tp_bytes / 1e9:8.1f}   4 collectives per layer, {layers_per_stage:.3f} layers per stage, ring factor {ring:.2f}")
print(f"  pipeline          {pp_bytes / 1e9:8.1f}   one tensor forward and one backward per stage boundary")
print(f"  data parallel     {dp_bytes / 1e9:8.1f}   gradient reduce-scatter plus weight all-gather of a {2 * P / (t * p) / 1e9:.2f} GB shard")
print(f"tensor parallel moves {tp_bytes / dp_bytes:.0f} times the data parallel volume and {tp_bytes / pp_bytes:.0f} times the pipeline volume")
```

**Reading the output.** The bubble falls from 0.484 idle at 16 micro-batches to 0.105 at 128. A DP rank's 131,072 tokens make a 4.29 GB activation tensor. Tensor parallelism sends 236.8 GB per step, 19 times the data-parallel volume and 28 times the pipeline volume. Treat these as estimates: they count only the main collectives, assume micro-batch size does not change the total, and ignore recomputation traffic, which adds more.

**Line by line.**

- `ring = 2 * (t - 1) / t` is the share of the message each GPU sends in a ring all-reduce: 1.75 for 8 GPUs.
- `4 * layers_per_stage` is two forward and two backward collectives per layer.
- `dp_bytes` is a gradient reduce-scatter plus a weight all-gather of the 6.34 GB shard each GPU holds.

### 5. Tensor parallelism on a real `LlamaMLP`, two processes

Two gloo processes each take half of the gate and up columns and half of the down rows of one real `LlamaMLP`, compute their part, and add the parts with one all-reduce.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

HIDDEN, FFN, TOKENS, WORLD = 64, 160, 12, 2


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    mlp = LlamaMLP(LlamaConfig(hidden_size=HIDDEN, intermediate_size=FFN, mlp_bias=False)).eval()
    x = torch.randn(TOKENS, HIDDEN)
    with torch.no_grad():
        reference = mlp(x)
        shard = FFN // world
        cols = slice(rank * shard, (rank + 1) * shard)
        gate = mlp.gate_proj.weight[cols].T
        up = mlp.up_proj.weight[cols].T
        down = mlp.down_proj.weight[:, cols].T
        partial = (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
        before_sum = (partial - reference).abs().max().item()
        summed = partial.clone()
        dist.all_reduce(summed)
        after_sum = (summed - reference).abs().max().item()
    params_full = sum(p.numel() for p in mlp.parameters())
    params_mine = gate.numel() + up.numel() + down.numel()
    stats = torch.tensor([before_sum, after_sum, params_mine], dtype=torch.float64)
    table = [torch.zeros(3, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(table, stats)
    if rank == 0:
        print(f"MLP parameters in one piece: {params_full:,}")
        for r, row in enumerate(table):
            print(f"rank {r}: holds {int(row[2]):,} parameters, "
                  f"error before the sum {row[0].item():.3f}, after the sum {row[1].item():.2e}")
        print(f"bytes in the one all-reduce: {partial.numel() * 4:,} (tokens x hidden x 4)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(WORLD, free_port()), nprocs=WORLD, join=True)
```

**Reading the output.** Each rank holds 15,360 of the layer's 30,720 parameters. Alone, a rank's output differs from the full layer by 0.232 and 0.216, so a half-sum is wrong. After the all-reduce both ranks match the unsplit layer to 1.34e-07, which is float32 rounding. The message was only 3,072 bytes here (12 tokens x 64 x 4); in the 405B layout it is the 268 MB per micro-batch per collective that block 4 counts.

**Line by line.**

- `mlp.gate_proj.weight[cols].T` takes rows of the stored weight, which are the output columns of the matrix. This is the "column split".
- `mlp.down_proj.weight[:, cols].T` takes the matching input rows of the down projection: the "row split".
- The activation sits between the two splits, so no communication is needed until the end.

### 6. Context parallelism with zigzag chunks, two processes

Two gloo processes each hold the zigzag chunks of a 32-token sequence, all-gather keys and values, and compute causal attention for their own queries. The result is compared with attention on the whole sequence, and the work per rank is counted under the zigzag and a contiguous split.

```python
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SEQ, DIM, WORLD = 32, 8, 2


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def causal_attention(q, k, v, q_pos, k_pos):
    scores = q @ k.T / DIM**0.5
    scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def chunks_for(rank, world, seq):
    size = seq // (2 * world)
    mine = [rank, 2 * world - 1 - rank]
    return torch.cat([torch.arange(c * size, (c + 1) * size) for c in mine])


def pairs(positions):
    return int((positions + 1).sum())


def worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(SEQ, DIM, generator=g) for _ in range(3))
    all_pos = torch.arange(SEQ)
    reference = causal_attention(q, k, v, all_pos, all_pos)

    mine = chunks_for(rank, world, SEQ)
    k_full = [torch.zeros(SEQ // world, DIM) for _ in range(world)]
    v_full = [torch.zeros(SEQ // world, DIM) for _ in range(world)]
    dist.all_gather(k_full, k[mine].contiguous())
    dist.all_gather(v_full, v[mine].contiguous())
    order = torch.cat([chunks_for(r, world, SEQ) for r in range(world)])
    out = causal_attention(q[mine], torch.cat(k_full), torch.cat(v_full), mine, order)

    err = (out - reference[mine]).abs().max().item()
    contiguous = torch.arange(rank * SEQ // world, (rank + 1) * SEQ // world)
    stats = torch.tensor([err, pairs(mine), pairs(contiguous)], dtype=torch.float64)
    table = [torch.zeros(3, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(table, stats)
    if rank == 0:
        for r, row in enumerate(table):
            print(f"rank {r}: max error {row[0].item():.2e}  score pairs zigzag {int(row[1])}  contiguous {int(row[2])}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(WORLD, free_port()), nprocs=WORLD, join=True)
```

**Reading the output.** Both ranks match single-process attention to float32 rounding (at most 5.96e-08). The zigzag split gives each rank exactly 264 of the 528 causal score pairs. A contiguous split would give rank 0 only 136 and rank 1 392, nearly three times the work, and the step runs at the speed of the slower rank.

**Line by line.**

- `chunks_for` gives rank `i` chunks `i` and `2 * world - 1 - i`: one early, one late.
- `pairs` counts query-key pairs under the causal mask: position `p` scores `p + 1` keys.
- The `all_gather` of keys and values is the single communication step, as in the Llama 3 recipe.

### Try it yourself

The lab is block 3 with every degree exposed. Its defaults (405B, TP 8, PP 16, CP 1, DP 128, ZeRO 2, 8,192 tokens, micro-batch 1, tensor and sequence parallelism with selective recomputation, 80 GB) reproduce the printed 78.6 GB. The GPU-memory slider is an input you choose, not a specification.

<ParallelismMemoryLab />

**What each control does.**

- **model** picks the Llama 3.1 8B, 70B or 405B shape, from the real configs of block 1.
- **tensor, pipeline, context, data** set the four degrees; the readout shows their product, the GPU count.
- **ZeRO stage** shards optimiser state (1), then gradients (2), then weights (3) across the data-parallel group.
- **tokens per sequence** and **micro-batch** feed the activation formula.
- **activations** chooses which saving is on: nothing, tensor, tensor plus sequence, plus selective recomputation, or full recomputation.
- **GPU memory** moves the dashed limit. The dashed limit line turns red when the total passes it.

**Try it yourself.**

1. Set **ZeRO stage** to 0. Weights, gradients and optimiser state become 6.34, 6.34 and 38.05 GB, and the total is 122.6 GB. Why it matters: the 128-way data-parallel group was holding 128 identical copies of the optimiser; sharding is what makes the layout fit.
2. Set **activations** to "tensor + sequence parallel". The total jumps to 755.0 GB. Why: the attention score matrices (the `5as/h` term) are back. This one setting is worth more than any change to the degrees.
3. Reproduce the paper's third row: set **tokens per sequence** to 131,072, **context** to 16 and **data** to 8. The total is 83.8 GB, over the 80 GB line, and ZeRO stage 3 brings it to 79.0 GB. Why: with only 8 replicas there is much less to shard across, and this crude formula cannot see the savings Meta's kernels make.

## Production snippets (not run here)

:::warning Not run in this environment
These blocks need CUDA GPUs and several servers. The imports were checked against PyTorch 2.14.1 on 7 October 2026; the code was not executed.
:::

A three-dimensional device mesh, with tensor parallelism applied to the feed-forward block of each layer (`ColwiseParallel` on the up projections and `RowwiseParallel` on the down projection, so the only communication is the all-reduce after the second matrix):

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

mesh = init_device_mesh("cuda", (4, 2, 8), mesh_dim_names=("dp", "pp", "tp"))
tp_mesh = mesh["tp"]

for block in model.model.layers:
    parallelize_module(
        block.mlp,
        tp_mesh,
        {"gate_proj": ColwiseParallel(), "up_proj": ColwiseParallel(), "down_proj": RowwiseParallel()},
    )
```

Starting it across servers uses `torchrun`. The strategy does not change how the job is launched:

```bash
torchrun --nnodes 8 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint head-node:29500 train.py
```

## Designing with it

1. **Start from the memory wall, not the speed wall.** Compute 16 bytes per parameter and the activation formula for your sequence length. Only parallelism that removes a real wall is worth its communication.
2. **Fill the server with tensor parallelism, then stop.** Keep tensor groups inside a server (commonly 8) on the fastest links.
3. **Use pipeline parallelism for what is left.** It needs far less bandwidth, so it can cross servers, but give it enough micro-batches that the bubble stays small, and balance the first and last stages.
4. **Spend the remaining GPUs on data parallelism,** sharded with ZeRO or FSDP when state is the constraint.
5. **Add context parallelism only for long sequences,** and use the zigzag split with causal attention.
6. **Prefer less parallelism.** The Playbook's decision order says to fit one instance first and scale out second. Every axis you add is a failure mode and a debugging cost.
7. **Measure MFU.** Llama 3 reports 38 to 43 per cent. If yours is far lower, look for a talkative axis crossing a slow link.

## Where this stands in 2026

:::info Industry view
- **The vocabulary is stable.** Tensor, pipeline, data (sharded) and context parallelism appear in the Llama 3 paper and in the Ultra-Scale Playbook (2025), which treats expert parallelism as the fifth axis. This chapter's version check: the Playbook page was read on 7 October 2026.
- **PyTorch has first-party pieces for most axes.** PyTorch 2.14 ships `fully_shard` (FSDP2), a tensor-parallel module, a pipelining module and `DeviceMesh` for composing them. Importing each worked in this chapter's environment.
- **The published recipe is a 2024 recipe.** Use the Llama 3 paper as a worked example of how the degrees combine. Newer systems differ in detail, and chapter 4 covers the mixture-of-experts models that now share the frontier.
- **Not verified here:** any GPU timing, any cluster's real MFU, and which exact layout any current lab uses. Those are not published for most models.
:::

## Common mistakes

1. **Spanning a tensor-parallel group across servers.** It feels natural to use all 16 GPUs of two servers as one tensor group. Block 4 shows why not: tensor traffic is an order of magnitude larger than the others. Keep it inside a server and use pipeline parallelism between servers.
2. **Sizing by weights alone.** "70B in bf16 is 140 GB, so 2 cards will do." Training stores gradients, optimiser state and activations too: 16 bytes per parameter, plus activations. Do the whole sum.
3. **Adding data parallelism to fix an out-of-memory error.** More replicas make training faster, not smaller. Unless you shard the replicas (chapter 2), each still holds the full copy.
4. **Using a plain contiguous split for causal context parallelism.** It looks like the obvious cut. Block 6 shows one rank doing nearly three times the work of the other. Use the zigzag assignment.
5. **Trusting a memory formula as proof.** The Megatron formula is for GPT-style layers. Use it to pick which saving matters, then measure the real peak on one node.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> The Llama 3 405B run used TP 8, PP 16 and DP 128. How many GPUs is that, and how many 8,192-token sequences does each data-parallel replica process per step if the batch is 16,777,216 tokens?</summary>

8 x 16 x 128 = 16,384 GPUs. The batch over 128 replicas is 131,072 tokens each, which is 16 sequences of 8,192 tokens. That matches the "batch size per DP" of 16 in Table 4.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> A 13B-parameter model trains with plain bf16 and Adam, with no sharding. How much model state does one GPU need?</summary>

13 billion x 16 bytes = 208 GB. That is more than any single card in this chapter, so the model cannot be trained with data parallelism alone, whatever the number of replicas.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Why do practitioners keep tensor-parallel groups inside one server?</summary>

Tensor parallelism needs two all-reduces in the forward pass and two in the backward pass of every layer. Block 4 puts that at about 237 GB sent per GPU per step for the 405B layout, against 12.6 GB for data parallelism. Inside a server the GPUs share the fastest links, so this is affordable. Across servers the traffic drops to a slower network. The Playbook measured clear drops from TP 8 to TP 16.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> A 16-stage pipeline processes 48 micro-batches per step in the plain schedule. What fraction of time is idle, and what would 4-way interleaving change?</summary>

Idle fraction is (16 - 1) / (48 + 16 - 1) = 15/63 = 0.238. Interleaving with V = 4 cuts the bubble to roughly a quarter of its size, so the bubble-to-ideal ratio goes from 15/48 = 0.31 to 15/(4 x 48) = 0.078. The price is more point-to-point messages between stages.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Training a 70B model with TP 8, PP 2, DP 4 and ZeRO 1, you run out of memory. The lab says activations are 22.8 GB with selective recomputation. Name two changes and what each saves.</summary>

Block 2's state total is 30.87 GB, so the run needs 30.87 + 22.82 = 53.7 GB by this estimate, which should fit on 80 GB. If it does not in practice, suspect the formula's blind spots (logits, buffers, fragmentation). Two real levers: full activation recomputation cuts the activation term to about 10.7 GB at the price of an extra forward pass, and ZeRO stage 2 or 3 cuts the model state to 24.25 or 17.85 GB. Raising pipeline stages does little for the first stage's activations, because under one-forward-one-backward scheduling it still holds `p` micro-batches in flight.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> Explain why 78.6 GB is not proof that Llama 3 fits in 80 GB, and what the lab's 83.8 GB for the third layout tells you.</summary>

The 78.6 GB is an upper-bound style estimate for the first pipeline stage from a formula derived for GPT-style layers. It leaves out the output logits, communication buffers, the gathered working copy of weights and fragmentation. The 83.8 GB for the long-context layout is above the card, yet Meta trained that layout, so the formula is pessimistic or Meta applied savings it does not model. The honest reading is that the budget is tight and every saving in the table is probably needed, not that any one number is exact.

</details>

## Go deeper

All sources were opened on 7 October 2026.

- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face, 2025)](https://huggingface.co/spaces/nanotron/ultrascale-playbook): the 2 + 2 + 4 + 8 memory accounting, the TP 8 to TP 16 drop, the 4,000-experiment benchmark on up to 512 GPUs and the three-step decision order.
- [The Llama 3 Herd of Models (arXiv 2407.21783, v3)](https://arxiv.org/abs/2407.21783): Table 3 hyper-parameters, Table 4 configurations, the FSDP note and the zigzag context parallelism.
- [Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053): the tensor-parallel layer split and its two forward and two backward all-reduces.
- [Efficient large-scale language model training on GPU clusters using Megatron-LM (arXiv 2104.04473)](https://arxiv.org/abs/2104.04473): interleaved pipeline schedule and the takeaways on using tensor parallelism up to the server size.
- [Reducing activation recomputation in large transformer models (arXiv 2205.05198)](https://arxiv.org/abs/2205.05198): the activation formulas, sequence parallelism and selective recomputation.
- [ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054): the 16-bytes-per-parameter accounting that chapter 2 develops.
- [PyTorch 2.14 tensor parallel](https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html) and [FSDP2 `fully_shard`](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html).
- Config files read by block 1: `NousResearch/Meta-Llama-3.1-8B`, `NousResearch/Meta-Llama-3.1-70B` and `NousResearch/Hermes-3-Llama-3.1-405B` on the Hugging Face Hub.

## Check yourself

- I can name the five parallelism axes and say what each splits and what each sends.
- I can compute model-state memory per GPU for a layout and ZeRO stage, and say which part is the largest.
- I can use the activation formulas to explain why selective recomputation matters.
- I can explain why tensor parallelism belongs inside a server, using bytes sent per step.
- I can compute a pipeline's idle fraction and name two ways to reduce it.
- I can explain why causal context parallelism needs a zigzag split and show the work balance.
- I can check that a published layout multiplies out to its GPU count and batch size, and say where my memory estimate is only approximate.

## Where to go next

Next chapter: [DDP, FSDP and ZeRO](/docs/llm-engineering/ddp-fsdp-and-zero), which shards the data-parallel axis step by step. A related chapter: [GPU sizing and capacity planning](/docs/llm-engineering/gpu-sizing-and-capacity-planning) does the same memory arithmetic for serving instead of training.
