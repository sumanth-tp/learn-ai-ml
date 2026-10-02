---
id: llme-kv-cache
title: "The KV Cache and PagedAttention"
sidebar_label: "2 · KV cache and paged attention"
sidebar_position: 2
slug: /llm-engineering/kv-cache-and-paged-attention
description: "Why decoding caches keys and values, the size formula checked against real model configs, how grouped-query and latent attention shrink it, and how PagedAttention stops fragmentation from wasting most of the memory."
tags: [kv-cache, paged-attention, vllm, gqa, mqa, mla, memory-fragmentation, inference]
---

import Infographic from '@site/src/components/Infographic';
import KvCacheLab from '@site/src/components/viz/KvCacheLab';

**In one line.** The KV cache lets a decoder compute each token's keys and values once instead of at every step, but it grows with every token of every sequence, so its size formula, its attention design and the way it is laid out in memory decide how many users one GPU can serve.

:::note Not from a lecture
This chapter was written for this site from the sources under Further reading. Model sizes come from the models' own `config.json` files, read by the code below, and the PagedAttention figures are quoted from the paper.
:::

## The idea in plain words

To pick the next word, attention compares the newest token (the query) with every earlier token (their keys) and then mixes the earlier tokens' values by those scores. Here is the useful fact: **an earlier token's key and value never change** once it has been processed, because a decoder only looks backwards. Recomputing them at every step would repeat the same work again and again, so the model keeps them. That store is the **KV cache**.

With the cache, a decode step computes the query, key and value of one new token, appends the key and value to the store, and reads the whole store back. Without it, step number $t$ pushes all $t$ tokens through the network again. The previous chapter said decoding is memory-bound; the cache is the other thing every step reads.

Cached tokens are not free. Each one leaves a key vector and a value vector in every layer, so the cache grows by the same amount for every token, for every user, for as long as the conversation lasts. Think of a hotel. The cache is the rooms, each guest is a sequence, and a guest checks in without knowing how long they will stay. A hotel that reserves its biggest suite for every arrival, just in case, turns away most guests with the rooms half empty. A hotel that hands out one room at a time as they are needed fits far more. That second hotel is **PagedAttention**.

<Infographic src="/img/llme/kv-cache-and-paged-attention-size.svg" alt="The KV cache size formula worked for Llama 3.1 8B, a bar chart of KiB per token for eight real models, and four cards describing MHA, GQA, MQA and MLA." caption="What one token costs, from the models' own configs. Every figure is printed by block 1." />

<Infographic src="/img/llme/kv-cache-and-paged-attention-paging.svg" alt="A table of sequences that fit with contiguous reservation against paged blocks of several sizes, a block table with three and two blocks, copy-on-write sharing of a prompt, and the figures reported by the PagedAttention paper." caption="Reserving the maximum against paging on demand. The table and the sharing numbers are printed by block 3." />

## How it works

### The size formula

For a model with $L$ layers, $H_{kv}$ key-value heads of dimension $d$, and $b$ bytes per element, one token costs

$$\text{bytes per token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b$$

where the 2 is one key vector plus one value vector. The PagedAttention paper works the same sum for the 13-billion-parameter OPT model: $2 \times 5120 \times 40 \times 2$ bytes is 800 KB per token, so a full 2,048-token sequence needs up to 1.6 GB. Block 1 applies the formula to eight real configs.

### Fewer key-value heads

The formula is dominated by $H_{kv}$, and models differ in how many they keep.

| Design | KV heads | Elements per token (DeepSeek-V2 paper notation) | Idea |
| --- | --- | --- | --- |
| MHA, multi-head | one per query head, $n_h$ | $2\,n_h\,d_h\,l$ | the original; best quality, biggest cache |
| MQA, multi-query | 1 | $2\,d_h\,l$ | Shazeer (2019): keys and values shared across all heads |
| GQA, grouped-query | $n_g$ groups, between 1 and $n_h$ | $2\,n_g\,d_h\,l$ | a middle path; the GQA paper converts MHA checkpoints with 5 per cent of the original pre-training compute |
| MLA, multi-head latent | a compressed latent, no per-head keys | $(d_c + d_h^R)\,l$ | cache a low-rank latent plus a small rotary key |

Block 1 shows what shipped. Mistral 7B, Llama 3.1 8B and Llama 3.1 70B use 8 KV heads, Qwen2.5 7B uses 4, SmolLM2 uses 3 and Falcon 7B is multi-query, while GPT-2 is plain multi-head. Llama 3.1 8B keeps 8 KV heads for its 32 query heads, so its cache is a quarter of what the same model would need with full multi-head attention: 128 KiB per token instead of 512.

MLA is the least familiar. DeepSeek-V2 compresses keys and values into one latent vector of size $d_c$ and, because rotary position embeddings cannot be folded into that compression, also caches a small decoupled rotary key of size $d_h^R$. The paper sets $d_c = 4 d_h$ and $d_h^R = d_h/2$, so the cache is 4.5 $d_h$ per layer, which the authors compare to GQA with 2.25 groups while claiming stronger quality than MHA. The `DeepSeek-V2-Lite` config exposes these two fields, `kv_lora_rank` of 512 and `qk_rope_head_dim` of 64, which is how block 1 sizes it.

:::note Reading the DeepSeek numbers
The paper's headline is a 93.3 per cent smaller cache than DeepSeek 67B, a different and larger model. Block 1's 30.4 KiB for V2-Lite next to 128 KiB for Llama 3.1 8B compares two unrelated models and does not reproduce that claim. Compare per-layer elements, as in the table above.
:::

### How the memory goes wrong

A serving system must decide, when a request arrives, where its cache will live. The simple answer, one contiguous slab per request, fails because the final length is unknown. The PagedAttention paper names three kinds of waste in the systems it compares against: **reserved** slots for tokens not yet generated, **internal fragmentation** from sizing for the maximum length, and **external fragmentation** from the allocator itself. Its profiling found that only 20.4 to 38.2 per cent of the cache memory held actual token states in those systems.

### The block table

PagedAttention borrows from operating systems. The cache of each sequence is cut into fixed-size **blocks** that hold the keys and values of a few tokens, 16 by default. Blocks are handed out from a free pool only when the sequence needs one, and a per-sequence **block table** maps logical position to physical block, exactly as a page table maps virtual to physical pages. Blocks need not be adjacent. Waste is limited to the unfilled tail of a sequence's last block, and external fragmentation vanishes because every block has the same size.

Sharing comes free. Two sequences that share a prompt can point at the same physical blocks, with a **reference count** per block. When one of them must write into a shared block, which can only be the last, partly filled one, the system copies it first and drops the reference. The paper describes this for parallel sampling and notes that beam search can save up to 55 per cent of memory. The attention kernel simply gathers keys and values through the table.

Nothing is free. The paper measures its attention kernel as 20 to 26 per cent slower than the FasterTransformer kernel it compares against, and argues the end-to-end gain far outweighs that. It also reports that block sizes of 16 to 128 worked best on one trace while larger blocks hurt on short sequences, which block 3 reproduces in miniature.

## A real system that works this way

**vLLM** is the system built on PagedAttention. The paper reports that it improves the throughput of popular LLMs by 2 to 4 times over FasterTransformer and Orca at the same latency, with larger gains for longer sequences, larger models and more complex decoding. Its own figure of where the memory goes for a 13B model on a 40 GB A100 is a useful anchor: about 65 per cent weights (26 GB) and close to 30 per cent KV cache, which is why the cache sets the batch size. vLLM's documentation describes the block as holding the keys and values of a fixed number of tokens for one head, with 16 tokens and a head size of 128 as its worked example.

## Code you can run

Four blocks. Block 1 reads real configs from the Hugging Face Hub, block 2 runs SmolLM2-135M, and blocks 3 and 4 are a seeded simulation and a numerical check. Everything runs on CPU in well under a minute after the downloads.

### 1. A cache-size calculator checked against real configs

The function handles four layouts: plain multi-head, grouped-query, multi-query (Falcon's config says `multi_query`), and MLA (the config has `kv_lora_rank`). The last lines check the formula against a real forward pass: SmolLM2-135M in fp32 holds 3 KV heads of dimension 64 in 30 layers.

```python
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

REPOS = [
    "openai-community/gpt2",
    "tiiuae/falcon-7b",
    "mistralai/Mistral-7B-v0.1",
    "NousResearch/Meta-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "NousResearch/Meta-Llama-3.1-70B",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite",
]


def load(repo):
    return json.load(open(hf_hub_download(repo, "config.json")))


def kv_elements_per_token(cfg):
    if "kv_lora_rank" in cfg:
        return (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * cfg["num_hidden_layers"], "MLA"
    layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
    q_heads = cfg.get("num_attention_heads", cfg.get("n_head"))
    width = cfg.get("hidden_size", cfg.get("n_embd"))
    kv_heads = 1 if cfg.get("multi_query") else cfg.get("num_key_value_heads", q_heads)
    kind = "MQA" if kv_heads == 1 else "MHA" if kv_heads == q_heads else "GQA"
    return 2 * layers * kv_heads * (width // q_heads), kind


def kv_bytes_per_token(cfg, dtype_bytes=2):
    return kv_elements_per_token(cfg)[0] * dtype_bytes


print(f"{'model':36s} {'kind':4s} {'KiB/token':>9s} {'GB at 32k':>9s}   (bf16 cache, one sequence)")
for repo in REPOS:
    cfg = load(repo)
    elements, kind = kv_elements_per_token(cfg)
    per_token = elements * 2
    print(f"{repo.split('/')[-1]:36s} {kind:4s} {per_token / 1024:9.1f} {per_token * 32768 / 1e9:9.2f}")

llama = load("NousResearch/Meta-Llama-3.1-8B")
mha_equivalent = dict(llama, num_key_value_heads=llama["num_attention_heads"])
print()
print(f"Llama 3.1 8B with its 8 KV heads: {kv_bytes_per_token(llama)} bytes per token")
print(f"same model if it used all 32 heads: {kv_bytes_per_token(mha_equivalent)} bytes per token, "
      f"{kv_bytes_per_token(mha_equivalent) / kv_bytes_per_token(llama):.0f}x more")

smol = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(smol, dtype=torch.float32).eval()
prompt_tokens = 100
with torch.inference_mode():
    cache = model(torch.randint(0, 1000, (1, prompt_tokens)), use_cache=True).past_key_values
measured = sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers)
predicted = kv_bytes_per_token(load(smol), dtype_bytes=4) * prompt_tokens
print(f"SmolLM2-135M, {prompt_tokens} tokens in fp32: measured {measured} bytes, formula {predicted} bytes")

budget_gb = 40
tokens = 640
bytes_per_token = kv_bytes_per_token(llama)
exact = budget_gb * 1e9 / (tokens * bytes_per_token)
reserved_length = 4096
reserved = budget_gb * 1e9 / (reserved_length * bytes_per_token)
print(f"{budget_gb} GB of cache, Llama 3.1 8B, sequences of {tokens} tokens:")
print(f"  space for exactly what is used: {int(exact)} sequences")
print(f"  reserving {reserved_length} tokens each up front: {int(reserved)} sequences")
```

Read the table by head count. Llama 3.1 8B and Mistral 7B both cost 128 KiB per token, which is 4.29 GB for one 32,000-token conversation, so four such users need more memory than the model's 16 GB of weights. The 70B model costs 320 KiB per token. Falcon 7B's single KV head costs 8 KiB, and DeepSeek-V2-Lite's latent costs 30.4 KiB. The formula is exact: the measured cache of 100 tokens is 4,608,000 bytes and the formula predicts the same. The last lines are the budget question: with 40 GB for cache, sequences of 640 tokens (the average context of the previous chapter) fit 476 times when space is allocated for what is used, and only 74 times when 4,096 tokens are reserved per sequence.

<KvCacheLab />

The lab is the same calculation with a model selector. Its defaults, Llama 3.1 8B in bf16 at 640 tokens with a 40 GB budget, a 4,096-token reservation and 16-token blocks, give the 476 and 74 printed above.

### 2. What the cache buys: `use_cache` on and off

Greedy generation on SmolLM2-135M with the cache enabled and disabled. With `use_cache=False` every step re-encodes the whole sequence, and the output must be identical because the cache is purely an optimisation.

```python
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
prompt = tokenizer("The key and value vectors of earlier tokens never change, so a decoder can", return_tensors="pt")


def generate(new_tokens, use_cache):
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **prompt,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0], time.perf_counter() - start


generate(4, True)
print(f"prompt: {prompt['input_ids'].shape[1]} tokens")
print("new_tokens  with_cache_s  no_cache_s  speedup  same_tokens")
for new_tokens in [32, 64, 128, 192]:
    with_cache, t_cache = generate(new_tokens, True)
    without, t_plain = generate(new_tokens, False)
    print(f"{new_tokens:10d}  {t_cache:12.2f}  {t_plain:10.2f}  {t_plain / t_cache:6.1f}x  {bool((with_cache == without).all())}")
```

The tokens match in every row, so nothing changes except time. The cached run grows roughly in proportion to the number of new tokens, while the uncached run grows faster: across the three runs I made, going from 32 to 192 new tokens (6 times as many) multiplied the cached time by 4 to 7 and the uncached time by 8 to 13. The speedup therefore widens with length, from 2.5 to 2.6 times at 32 tokens to 4.6 to 5.5 times at 192, because the work without a cache grows with the square of the length. Timings change from run to run and machine to machine, so trust the shape.

### 3. A block allocator, and contiguous against paged

`PagedAllocator` is a small working version of the mechanism: a free list, a reference count per block, one block table per sequence, `fork` for sharing, and copy on write when a shared last block is written. The second half fills a fixed 40 GB budget with a seeded mix of 4,000 requests, each caught partway through generation, and compares reservation against several block sizes.

```python
import numpy as np


class PagedAllocator:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(num_blocks - 1, -1, -1))
        self.refs = [0] * num_blocks
        self.tables = {}
        self.lengths = {}

    def blocks_in_use(self):
        return sum(1 for r in self.refs if r > 0)

    def _take(self):
        block = self.free.pop()
        self.refs[block] = 1
        return block

    def add_tokens(self, seq, count):
        table = self.tables.setdefault(seq, [])
        length = self.lengths.get(seq, 0)
        for _ in range(count):
            if length % self.block_size == 0:
                table.append(self._take())
            elif self.refs[table[-1]] > 1:
                self.refs[table[-1]] -= 1
                table[-1] = self._take()
            length += 1
        self.lengths[seq] = length

    def fork(self, parent, child):
        self.tables[child] = list(self.tables[parent])
        self.lengths[child] = self.lengths[parent]
        for block in self.tables[child]:
            self.refs[block] += 1

    def release(self, seq):
        for block in self.tables.pop(seq):
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)
        del self.lengths[seq]


small = PagedAllocator(num_blocks=16, block_size=4)
small.add_tokens("a", 10)
small.add_tokens("b", 5)
print("block size 4: a has 10 tokens, b has 5")
print("  block table a:", small.tables["a"], " block table b:", small.tables["b"])
print("  blocks in use:", small.blocks_in_use(), "for", 10 + 5, "tokens")

shared = PagedAllocator(num_blocks=4096, block_size=16)
shared.add_tokens("prompt", 500)
for i in range(4):
    shared.fork("prompt", f"sample{i}")
shared.release("prompt")
for i in range(4):
    shared.add_tokens(f"sample{i}", 64)
copied = shared.blocks_in_use()
independent = 4 * ((500 + 64 + 15) // 16)
print()
print(f"4 samples sharing a 500-token prompt, 64 new tokens each: {copied} blocks with sharing, {independent} without")

rng = np.random.default_rng(0)
BYTES_PER_TOKEN = 131072
BUDGET = 40e9
MAX_LEN = 4096
slots = int(BUDGET // BYTES_PER_TOKEN)
n = 4000
prompt = np.clip(rng.lognormal(5.2, 0.8, n), 8, MAX_LEN // 2).astype(int)
output = np.clip(rng.lognormal(4.8, 0.9, n), 8, MAX_LEN // 2).astype(int)
progress = rng.uniform(0, 1, n)
used = prompt + (progress * output).astype(int)
print()
print(f"{slots} token slots in {BUDGET / 1e9:.0f} GB of cache. Mean prompt {prompt.mean():.0f}, mean output {output.mean():.0f}, mean live length {used.mean():.0f}")


def admitted(cost):
    return int(np.searchsorted(np.cumsum(cost), slots, side="right"))


contiguous = admitted(np.full(n, MAX_LEN))
print(f"{'allocator':34s} {'sequences':>9s} {'tokens used':>11s} {'waste':>6s}")
print(f"{'contiguous, reserve ' + str(MAX_LEN):34s} {contiguous:9d} {used[:contiguous].sum():11d} {1 - used[:contiguous].sum() / slots:6.1%}")
for block in [1, 8, 16, 32, 128, 512]:
    cost = -(-used // block) * block
    count = admitted(cost)
    print(f"{'paged, block ' + str(block):34s} {count:9d} {used[:count].sum():11d} {1 - used[:count].sum() / slots:6.1%}")
```

The first lines are the mechanism: with block size 4, sequence `a` of 10 tokens takes blocks 0, 1 and 2, sequence `b` of 5 tokens takes blocks 3 and 4, and five blocks hold 15 tokens. Four samples that share a 500-token prompt use 51 blocks with sharing against 144 without. The copy on write is visible in the arithmetic: the prompt's last block is only partly full, so each sample except the last copies it before writing.

The table is the main result. With room for 305,175 token slots, reserving 4,096 tokens per sequence admits 74 of them and wastes 91.8 per cent of the memory, because the live sequences average 338 tokens. Paging with 16-token blocks admits 895 and wastes 2.3 per cent. Smaller blocks waste less still, and large ones bring it back: 128-token blocks waste 15.3 per cent and 512-token blocks 45.1 per cent. A block size is a trade between waste on one side and kernel efficiency and sharing on the other, as the paper notes. This is a snapshot of requests caught mid-generation, so it ignores growth after admission, which a real scheduler handles by preempting requests when the cache fills.

### 4. Attention through a block table gives the same answer

The remaining worry is whether scattering the cache across blocks changes the result. This block stores 100 tokens of keys and values in randomly chosen physical blocks, gathers them through a block table and compares with ordinary attention.

```python
import torch

torch.manual_seed(0)
heads, head_dim, block_size, context = 4, 32, 16, 100

keys = torch.randn(context, heads, head_dim)
values = torch.randn(context, heads, head_dim)
query = torch.randn(heads, head_dim)


def attention(q, k, v):
    scores = torch.einsum("hd,thd->ht", q, k) / head_dim**0.5
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", weights, v)


reference = attention(query, keys, values)

num_blocks = 64
physical_k = torch.zeros(num_blocks, block_size, heads, head_dim)
physical_v = torch.zeros(num_blocks, block_size, heads, head_dim)
needed = -(-context // block_size)
block_table = torch.randperm(num_blocks)[:needed].tolist()
for token in range(context):
    block = block_table[token // block_size]
    slot = token % block_size
    physical_k[block, slot] = keys[token]
    physical_v[block, slot] = values[token]

gathered_k = physical_k[block_table].reshape(-1, heads, head_dim)[:context]
gathered_v = physical_v[block_table].reshape(-1, heads, head_dim)[:context]
paged = attention(query, gathered_k, gathered_v)

print(f"{context} tokens in {needed} blocks, block table {block_table}")
print(f"largest difference from contiguous attention: {(paged - reference).abs().max().item():.2e}")
print(f"slots allocated {needed * block_size}, used {context}, wasted in the last block {needed * block_size - context}")
```

The difference is exactly zero, as it must be: the table changes where values live, not what they are. The 12 wasted slots are the unfilled tail of the seventh block, the whole internal fragmentation of this sequence.

## Designing with it

- **Do the sum before choosing hardware.** Bytes per token times context times concurrent sequences, compared with the memory left after the weights. If it does not fit, no scheduler will rescue it.
- **Treat KV heads as a deployment cost.** Among open models of similar quality, the one with fewer KV heads serves more users per GPU. GQA is the common choice in the configs read here.
- **Do not reserve the maximum.** Block-based allocation is the engine default for a reason: block 3 shows it admitting more than ten times as many sequences.
- **Pick the block size on the workload.** Sixteen is the default the paper chose. Short sequences argue for small blocks, heavy prefix sharing against very small ones.
- **Shrink the cache itself when memory binds.** Quantising keys and values to fewer bits is the next lever, covered in the quantisation chapter, and shared prefixes avoid duplicating the cache at all, covered in the batching chapter.

## Where this stands in 2026

:::info Industry view

- Grouped-query attention is the norm in open models. Every Llama 3.1, Mistral 7B and Qwen2.5 config read in block 1 uses it, and only GPT-2, a 2019 model, keeps one KV head per query head.
- Latent attention is the design that cuts the cache furthest. DeepSeek-V2 reports a 93.3 per cent cache reduction against DeepSeek 67B and a 5.76 times larger maximum generation throughput, as its own measurements.
- Cache quantisation is active research. KIVI (ICML 2024) quantises keys per channel and values per token to 2 bits and reports 2.6 times lower peak memory including weights, 4 times larger batches and 2.35 to 3.47 times more throughput on Llama, Falcon and Mistral models.
- Hugging Face's guide describes a static cache with `torch.compile` for up to a 4 times speedup, at the price of a pre-allocated maximum length: the contiguous trade-off again, chosen for the compiler's sake.
- PyPI listed vLLM 0.30.0 as the latest release on 2 October 2026, and the documentation labels its latest pages as developer previews. Check the current docs before relying on a parameter name.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Compute the KV cache per token for a model with 40 layers, 8 KV heads of dimension 128 and an 8-bit cache.</summary>

$2 \times 40 \times 8 \times 128 \times 1 = 81{,}920$ bytes, which is 80 KiB per token. In bf16 it would be 160 KiB.

</details>

<details>
<summary><strong>Q2.</strong> Why does Llama 3.1 8B have a quarter of the cache of the same model with full multi-head attention?</summary>

It has 32 query heads but only 8 key-value heads, so it stores one quarter of the keys and values. Block 1 prints 131,072 bytes against 524,288 bytes per token.

</details>

<details>
<summary><strong>Q3.</strong> A service reserves 4,096 tokens for every request but the live requests average 338 tokens. Roughly what fraction of the cache is wasted, and which of the paper's kinds is it?</summary>

About 92 per cent, 91.8 per cent in block 3. It is mostly reserved slots and internal fragmentation: space held for tokens that may never come, on behalf of sequences far shorter than the maximum.

</details>

<details>
<summary><strong>Q4.</strong> Eight sequences share a prompt whose last block is half full. What happens when the first of them writes a new token, and when the last one does?</summary>

The first finds a reference count above 1 on the shared block, copies it into a new block, writes there and decrements the count. After seven copies the eighth finds a count of 1 and writes in place. This is the behaviour the paper describes.

</details>

<details>
<summary><strong>Q5.</strong> Why does waste climb for blocks above 32 tokens, and why not simply use 1-token blocks?</summary>

Larger blocks leave more of each sequence's last block empty, so waste rises, to 45.1 per cent at 512. One-token blocks waste almost nothing but make the block table huge and the reads fragmented, and the paper notes that very small blocks may not use the GPU's parallelism well when reading the cache.

</details>

<details>
<summary><strong>Q6.</strong> MLA caches 576 elements per layer for DeepSeek-V2-Lite. Express that in head dimensions and compare it with MQA.</summary>

The head dimension in the paper's notation is 128, so 576 is 4.5 head dimensions per layer. MQA caches $2 d_h$, which is 2 head dimensions, so MLA's cache is 2.25 times larger than MQA's, while the paper claims quality stronger than multi-head attention.

</details>

## Further reading

- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180): the paper behind vLLM; the memory figures, block-size study and copy-on-write description used above.
- [vLLM documentation, paged attention design](https://docs.vllm.ai/en/latest/design/paged_attention/): blocks, block size and the cache layout.
- [Ainslie et al., "GQA" (EMNLP 2023)](https://arxiv.org/abs/2305.13245) and [Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"](https://arxiv.org/abs/1911.02150): grouped-query and multi-query attention.
- [DeepSeek-AI, "DeepSeek-V2"](https://arxiv.org/abs/2405.04434): multi-head latent attention and the cache comparison table.
- [Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (ICML 2024)](https://arxiv.org/abs/2402.02750).
- [Hugging Face Transformers, optimizing inference](https://huggingface.co/docs/transformers/main/en/llm_optims): the static cache and `torch.compile`.
- Related chapters: [why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound), [how a transformer generates text, including KV caching](/docs/theory/dnn/transformer-inference-step-by-step), and the next chapter, [continuous batching and scheduling](/docs/llm-engineering/continuous-batching-and-scheduling).

## Check yourself

- I can write the KV cache size formula and compute it from a model's config.
- I can explain why keys and values can be cached and queries cannot.
- I can say how MQA, GQA and MLA shrink the cache and what each trades away.
- I can explain reserved, internal and external fragmentation and why contiguous reservation wastes most of the memory.
- I can explain a block table, a reference count and copy on write.
- I can say why paged attention returns exactly the same values as contiguous attention.
- I can choose a block size and defend it.
