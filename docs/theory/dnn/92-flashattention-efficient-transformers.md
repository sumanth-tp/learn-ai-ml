---
id: flashattention
title: "FlashAttention and efficient transformers"
sidebar_label: "92 · FlashAttention"
sidebar_position: 92
slug: /theory/dnn/flashattention-efficient-transformers
description: "Why standard attention is memory-bound, how FlashAttention reorders computation to avoid materializing the N×N matrix, and the landscape of efficient attention variants for long contexts."
tags: [flashattention, efficient-attention, long-context, transformers, deep-learning]
---

# FlashAttention and efficient transformers

> **TL;DR.** Standard attention is **memory-bound**, not compute-bound — the N×N attention matrix shuttles back and forth between fast SRAM and slow HBM, and that I/O cost dominates. **FlashAttention** computes the *same exact output*, but tiled: it loads small blocks of Q/K/V into SRAM, runs softmax + matmul without ever writing the N×N matrix to HBM, and uses a clever "online softmax" trick to merge results. Memory drops from O(N²) to O(N); wall-clock speed jumps 2–4×; long-context training becomes feasible. It's now the default in every production transformer (PyTorch's `F.scaled_dot_product_attention`, vLLM, all modern LLM training stacks).

Standard scaled dot-product attention materializes an $n \times n$ attention matrix in GPU memory, where $n$ is the sequence length. For $n=4096$: ~64 MB per head per batch item. For $n=32768$: ~4 GB. This memory cost makes long-context transformers prohibitively expensive. FlashAttention solves this by reordering the attention computation to never materialize the full attention matrix, reducing memory from $O(n^2)$ to $O(n)$ without changing the mathematical output.

## Prerequisites

- [74 — Scaled Dot-Product Attention](./74-scaled-dot-product-attention.md) — the math that FlashAttention reorganizes without changing
- [77 — Multi-Head Attention](./77-multi-head-attention-in-transformers.md) — the operation FlashAttention replaces in production
- [81 — Masked Self-Attention](./81-masked-self-attention-in-the-transformer-decoder.md) — the causal-mask case (FlashAttention has `is_causal=True`)
- [84 — Transformer Inference](./84-transformer-inference-step-by-step.md) — KV caching is the inference-side analogue of FlashAttention's training-side memory work
- [88 — GPT (Decoder-Only)](./88-gpt-decoder-only-causal-lm.md) — GQA, sliding-window attention, and other variants are introduced here

## Try it interactively

- **[FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)** — the official Triton/CUDA implementation; one-liner replacement for nn.functional.scaled_dot_product_attention
- **[PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)** — PyTorch 2+ automatically uses FlashAttention 2 when conditions allow
- **[Tri Dao — FlashAttention talk (YouTube)](https://www.youtube.com/results?search_query=tri+dao+flashattention)** — author's own explanation with diagrams
- **[Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)** — the canonical explanation of memory-bound vs compute-bound that motivates FlashAttention
- **[vLLM](https://github.com/vllm-project/vllm)** — production LLM serving framework built on FlashAttention + PagedAttention

## One-line definition

FlashAttention is an exact attention algorithm that computes the same result as standard attention but avoids materializing the full $N \times N$ attention matrix by using tiled computation with online softmax, reducing memory from $O(N^2)$ to $O(N)$ and significantly improving GPU throughput.

![Full self-attention matrix — every query attends to every key, producing an N×N matrix; FlashAttention computes the same result without ever storing this matrix in GPU HBM](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

FlashAttention is what enables modern LLMs to process long contexts (32k, 128k, 1M tokens). It is now the default attention implementation in PyTorch 2.0+ (`F.scaled_dot_product_attention`), HuggingFace Transformers, and every production LLM framework. Understanding FlashAttention explains why context length has exploded from 2048 tokens (GPT-3) to 128k+ (GPT-4, Claude 3) in just a few years.

## The bottleneck: GPU memory hierarchy

Modern GPUs have two types of memory:
- **HBM (High Bandwidth Memory)**: large (~40–80 GB), slow (~2 TB/s bandwidth)
- **SRAM (on-chip shared memory)**: tiny (~20 MB), very fast (~19 TB/s bandwidth)

Standard attention reads/writes matrices from HBM. The $n \times n$ attention matrix is too large for SRAM at any practical sequence length. Standard attention is therefore **memory-bound** — most time is spent waiting for data transfers to/from HBM, not computing.

```
Standard attention (n=4096, 1 head, float16):
1. Read Q, K from HBM → compute QK^T (n×n) → write to HBM    [expensive]
2. Read QK^T from HBM → apply mask → write to HBM             [expensive]
3. Read QK^T from HBM → compute softmax → write A to HBM      [expensive]
4. Read A, V from HBM → compute AV → write output to HBM      [expensive]
Memory: O(n²) reads + writes = 4096² = 16.7M floats per head
```

## FlashAttention: tiled computation with online softmax

FlashAttention splits Q, K, V into tiles and computes attention block by block entirely within SRAM. The key insight is that softmax can be computed incrementally using the **online softmax** trick.

### The online softmax trick

For a vector $x = [x_1, \ldots, x_n]$, standard softmax requires two passes: one to find $\max(x)$ and one to compute $\sum e^{x_i - \max(x)}$. But softmax can be computed in a single pass by maintaining running statistics:

$$
m_j = \max(m_{j-1}, x_j), \quad s_j = s_{j-1} e^{m_{j-1} - m_j} + e^{x_j - m_j}
$$

After processing all elements: $\text{softmax}(x_j) = e^{x_j - m_n} / s_n$.

FlashAttention uses this to process attention **block by block**: for each query block, iterate over all key-value blocks, update the running max and sum, and accumulate the weighted values — all in SRAM, never writing the full $n \times n$ score matrix to HBM.

```
FlashAttention (n=4096, 1 head):
For each Q_tile in Q:
    running_max = -inf, running_sum = 0, acc = 0
    For each KV_tile in (K, V):
        Load KV_tile from HBM to SRAM           [small block only]
        Compute scores = Q_tile @ KV_tile^T     [in SRAM]
        Update running max, sum (online softmax)
        acc += softmax_weights @ V_tile
    Write acc to HBM                            [one write per Q_tile]
Memory: O(n) — only tiles are in SRAM at any time
```

### Performance comparison

For sequence length $n$ on an A100 GPU:

| $n$ | Standard attention memory | FlashAttention memory | Speedup |
|---|---|---|---|
| 512 | 1 MB | 1 MB | 1.2× |
| 2048 | 16 MB | 4 MB | 2× |
| 4096 | 64 MB | 8 MB | 3–4× |
| 16384 | 1 GB | 32 MB | 6–8× |
| 65536 | 16 GB | 128 MB | OOM → feasible |

![GPU memory hierarchy — SRAM is ~1000× faster than HBM but ~1000× smaller. FlashAttention's central insight: keep the attention matrix in SRAM, never round-trip it through HBM](https://horace.io/img/perf_intro/gpu-memory-hierarchy.png)
*Source: [Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)*

## FlashAttention-2 improvements

FlashAttention-2 (Dao, 2023) adds:
- Better parallelism across sequence dimension (not just batch and head)
- Reduced non-matrix-multiply operations
- ~2× speedup over FlashAttention on A100/H100

FlashAttention-3 (2024) targets Hopper (H100) architecture with specialized WGMMA instructions.

## Using FlashAttention in PyTorch

```python
import torch
import torch.nn.functional as F
import math

# ============================================================
# PyTorch 2.0+: F.scaled_dot_product_attention
# Uses FlashAttention automatically if available on the hardware
# ============================================================

batch, heads, seq_len, d_k = 4, 8, 1024, 64

Q = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)
K = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)
V = torch.randn(batch, heads, seq_len, d_k, device="cuda", dtype=torch.float16)

# PyTorch's fused attention (automatically selects FlashAttention if available)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,          # FlashAttention
    enable_math=False,          # Disable standard math path
    enable_mem_efficient=False, # Disable xFormers memory-efficient path
):
    output_flash = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,   # Set True for decoder/causal attention
    )
print(f"FlashAttention output: {output_flash.shape}")   # (4, 8, 1024, 64)


# Causal attention for decoder (is_causal=True)
output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print(f"Causal attention output: {output_causal.shape}")  # (4, 8, 1024, 64)


# ============================================================
# Standard attention (for comparison on CPU/no Flash support)
# ============================================================
def standard_attention(Q, K, V, causal=False):
    """Reference implementation — materializes full attention matrix."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, H, N, N)
    if causal:
        mask = torch.triu(torch.ones(Q.size(-2), K.size(-2),
                                     device=Q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ V


# Both should give identical results (up to floating point differences)
Q_cpu = Q.float().cpu()
K_cpu = K.float().cpu()
V_cpu = V.float().cpu()

out_standard = standard_attention(Q_cpu, K_cpu, V_cpu)
out_flash_cpu = F.scaled_dot_product_attention(Q_cpu, K_cpu, V_cpu)

max_diff = (out_standard - out_flash_cpu).abs().max()
print(f"\nMax difference between standard and flash: {max_diff:.6f}")
# Should be very small (floating point precision difference only)


# ============================================================
# Memory comparison: standard vs. flash
# ============================================================
import time

def measure_memory(fn, *args, **kwargs):
    """Measure peak GPU memory during a function call."""
    if not torch.cuda.is_available():
        return None, fn(*args, **kwargs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    peak = torch.cuda.max_memory_allocated() / 1024**2   # MB
    return peak, result


# Use the PyTorch built-in which auto-selects the algorithm
for n in [512, 1024, 2048, 4096]:
    Q = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    K = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    V = torch.randn(1, 4, n, 64, device="cpu", dtype=torch.float32)
    # Show theoretical memory: n*n*4*4heads = attention matrix size
    attn_matrix_mb = (n * n * 4 * 4) / 1024**2   # float32, 4 heads
    print(f"n={n:5d}: attention matrix = {attn_matrix_mb:.1f} MB")
```

## Other efficient attention variants

### Sparse attention (Longformer, BigBird)

Instead of all $n^2$ pairs, attend only to a sparse pattern:
- **Local window**: each token attends to $w$ neighboring tokens: $O(n \cdot w)$
- **Global tokens**: a few special tokens attend to the entire sequence
- **Dilated strided**: every $k$-th token in a window

Used in Longformer (4096 tokens), BigBird (4096 tokens).

### Linear attention

Replace the $\text{softmax}(QK^T)$ computation with a kernel function that can be computed in $O(n)$:

$$
\text{Attn}(Q, K, V) \approx \phi(Q) \left(\phi(K)^T V\right)
$$

where $\phi$ is a feature map (e.g., $\phi(x) = \text{elu}(x) + 1$). The key trick: compute $K^T V$ first ($d \times d$ matrix), then multiply by $Q$ — $O(n d^2)$ instead of $O(n^2 d)$. Used in Performer, Linear Transformer.

### Sliding window attention (Mistral)

Each token attends only to the most recent $w$ tokens (e.g., $w=4096$). Combined with a large sliding window and group-query attention, Mistral 7B achieves competitive performance while processing longer sequences efficiently.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

Reduce the number of key-value heads to decrease KV cache memory:
- **MQA**: all query heads share 1 K/V head
- **GQA**: $h$ query heads share $g$ K/V heads ($g < h$, e.g., 8 KV heads for 32 Q heads)

Used in LLaMA 3, Mistral, Gemma. Reduces KV cache by 4–8× without significant quality loss.

## The KV cache memory problem at scale

For a 70B model serving 1000 concurrent users with 32k context:

$$
\text{KV cache} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times \text{seq\_len} \times d_{\text{head}} \times \text{bytes}
$$

LLaMA 3 70B: $2 \times 80 \times 8 \times 32768 \times 128 \times 2\text{ bytes} \approx 8.5\text{ GB per user}$

For 1000 users: 8.5 TB — clearly infeasible. Techniques like paged attention (vLLM), prefix caching, and speculative decoding make large-scale LLM serving practical.

## Interview questions

<details>
<summary>Why is standard attention memory-bound rather than compute-bound?</summary>

Standard attention performs the following HBM reads/writes: read Q, K (compute QK^T) → write QK^T → read QK^T (softmax) → write A → read A, V (compute AV) → write output. The $n \times n$ attention matrix requires $O(n^2)$ HBM reads and writes. For $n=4096$: ~100 MB of HBM traffic per head per forward pass. The actual matrix multiplications are fast (SRAM operations), but the dominant cost is the slow HBM transfers. FlashAttention eliminates the intermediate $n \times n$ writes by keeping all intermediates in fast SRAM.
</details>

<details>
<summary>Does FlashAttention produce the exact same output as standard attention?</summary>

Yes — FlashAttention is mathematically exact (not an approximation). It computes the same result as $\text{softmax}(QK^T/\sqrt{d_k})V$ but via a tiled algorithm that never materializes the full $n \times n$ matrix. The online softmax maintains numerically exact running statistics, and the accumulated output equals the standard attention output up to floating-point precision. This is different from approximate attention methods (sparse attention, linear attention) which trade accuracy for efficiency.
</details>

<details>
<summary>What is grouped-query attention and why does it matter?</summary>

Multi-head attention uses $h$ query heads and $h$ key-value heads. The KV cache stores all $h$ K and V matrices per layer per step — $O(h \times n)$ memory. Grouped-query attention (GQA) reduces to $g < h$ KV heads (each shared by $h/g$ query heads). For LLaMA 3 70B: 64 query heads, 8 KV heads — 8× KV cache reduction. At inference, the KV cache is the main memory bottleneck (not model weights), so this reduction allows 8× more concurrent users or 8× longer context at the same memory cost.
</details>

<details>
<summary>Scenario: you enable FlashAttention but observe no speedup on a 256-token sequence. Why?</summary>

FlashAttention's tiling has overhead — block setup, online softmax bookkeeping, and the fact that small sequences fit comfortably in HBM bandwidth anyway. For very short sequences ($n < 256$ or so), the I/O savings don't outweigh the tiling overhead, and standard attention can actually be faster.

FlashAttention shines when the attention matrix is *too large* to fit in SRAM easily, and HBM bandwidth becomes the bottleneck. That's the regime $n \geq 1024$ for most GPUs.

PyTorch's `F.scaled_dot_product_attention` actually picks the fastest backend per call (FlashAttention, memory-efficient attention via xFormers, or vanilla math). For short sequences it often falls back to math.

Practical implication: don't over-engineer FlashAttention for inference of very short prompts (chat bots with 100-token messages). Reserve it for training and for long-context inference.
</details>

<details>
<summary>How does the "online softmax" trick actually compute the correct result in one pass?</summary>

Classical softmax requires two passes: pass 1 finds $\max(x)$, pass 2 computes $\sum e^{x_i - \max(x)}$ then divides each $e^{x_j - \max(x)}$ by that sum. The subtraction is for numerical stability.

Online softmax processes one element at a time, maintaining a running max $m_j$ and running sum $s_j$:

- When a new element $x_j$ arrives, the new max is $m_j = \max(m_{j-1}, x_j)$.
- The old running sum was scaled to $m_{j-1}$. To rescale to $m_j$: multiply by $e^{m_{j-1} - m_j}$ (a correction factor when the max increases).
- Add the new term: $s_j = s_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}$.

After processing all elements, $m_n$ is the true max and $s_n$ is the correctly-normalized sum.

For FlashAttention, the attention weights and the weighted output are accumulated *together* with the running statistics — so the final output matches standard attention exactly, modulo floating-point rounding (Tri Dao showed the error is bounded and tiny in practice).

The deeper idea: online softmax decouples streaming computation from a global max. Many "streaming" or "tiled" deep learning algorithms use the same trick.
</details>

<details>
<summary>Scenario: a teammate suggests using FlashAttention to train a 100M-parameter model on 256-token sequences. Worth doing?</summary>

Probably not worth the engineering cost. FlashAttention's win is greatest when:

- Sequences are *long* (>1024 tokens, ideally 4096+).
- Memory is the binding constraint (training large models or long contexts).
- You're on modern GPUs (A100, H100) that have favorable SRAM/HBM ratios.

For 100M params at 256 tokens:

- Attention matrix is tiny (256² × 4 bytes × 12 heads × batch ≈ a few MB).
- Memory isn't the bottleneck — model weights and activations dominate.
- The actual speedup might be 1.1× — barely measurable.

Better focus areas for this configuration: mixed precision (FP16/BF16), gradient checkpointing if memory is tight, larger batch size, optimizer (AdamW vs Lion vs AdaFactor).

Rule of thumb: FlashAttention is essential above 4K context, helpful at 1-4K, marginal below 1K. PyTorch's SDPA gives it to you free anyway — but don't optimize specifically *for* it unless your context is long.
</details>

<details>
<summary>Why doesn't FlashAttention work straightforwardly with custom attention biases like ALiBi?</summary>

FlashAttention's tiled computation assumes attention scores are computed as $QK^T / \sqrt{d_k}$ — a simple dot product. Custom biases add a position-dependent term: ALiBi adds $-|i-j| \cdot m$ to the score for query $i$, key $j$, head with slope $m$.

This works in principle (just add the bias during the tile-level score computation), but the original FlashAttention kernel didn't handle it. Later versions (FlashAttention-2, FlexAttention) added support for custom biases at the cost of kernel complexity.

Practical implication: if you're using a model with custom attention biases (Llama with ALiBi-derived RoPE adjustments, Falcon with ALiBi, custom learned biases), you need either a FlashAttention version that explicitly supports them, or fall back to the math backend at some performance cost.

PyTorch 2.5+ introduced FlexAttention which generalizes FlashAttention to arbitrary differentiable score modifications. This is the future-proof solution: writing your custom score function once and getting FlashAttention-speed without writing a CUDA kernel.
</details>

<details>
<summary>Sparse attention (Longformer/BigBird) vs FlashAttention: aren't they solving the same problem?</summary>

Different problems, different trade-offs:

- **FlashAttention** computes the *exact* full-attention output more efficiently. Same model, same math, just better memory layout. Output is identical to standard attention.
- **Sparse attention** (Longformer, BigBird, sliding window) changes the *math*: each token attends to only a subset of positions, not all of them. The model is functionally different and must be trained or fine-tuned with the sparse pattern.

When to use which:

- **Long-context training of a standard transformer**: FlashAttention. Same model, faster.
- **Inference on a model already trained with sparse attention**: sparse implementation (no choice).
- **Extreme context lengths (>100K tokens)**: combine both — FlashAttention's tiling for the dense local windows, sparse skipping for the long-range global tokens.
- **Drop-in extension of an existing model to 32K context**: FlashAttention plus context-window extension techniques (NTK-aware RoPE, YaRN) — preserves model capability while gaining length.

Mistral 7B's sliding-window attention is interesting: it's a *sparse* pattern (local window only), but it's compatible with FlashAttention's tiling because the sparsity is structured. Modern efficient transformer designs increasingly combine both.
</details>

<details>
<summary>How does FlashAttention interact with KV caching at inference time?</summary>

At inference, the query is a single new token (Q has shape `(batch, heads, 1, d_head)`) attending to all previously cached K and V. The attention matrix is `(1 × past_len)` — already small in the query dimension. FlashAttention's tiling helps less here.

The key inference optimization is *PagedAttention* (vLLM): manage KV cache as fixed-size pages so multiple users' caches can share GPU memory efficiently, even with very different sequence lengths. PagedAttention isn't "FlashAttention for inference" — it's complementary, addressing a different bottleneck (memory fragmentation across users).

FlashAttention-Decode (a 2024 variant) specifically targets the inference-time case where Q is short and K/V are long. The tiling is reorganized to parallelize over the cached KV dimension rather than the query dimension. This is what makes 128K-context inference fast for chat models.

For training: FlashAttention dominates. For inference: FlashAttention-Decode + PagedAttention together.
</details>

<details>
<summary>Scenario: switching from standard PyTorch attention to FlashAttention, you observe slight numerical differences in the output. How concerned should you be?</summary>

Not very. FlashAttention is mathematically exact, but the floating-point operation *order* is different. With FP16/BF16, this produces differences on the order of $10^{-3}$ to $10^{-4}$ — undetectable in practice for trained models.

What to verify:

1. **Loss curves match closely** during training. If they diverge significantly, you have a bug, not a precision issue.
2. **Downstream metrics match** within run-to-run variance. A model evaluated with FlashAttention vs standard attention should give within 0.1% accuracy.
3. **Gradient norms are similar** in early training.

When to worry: if you see *systematic* drift (FlashAttention always slightly worse), that suggests a bug — likely incorrect mask handling, incorrect dropout placement, or numerical issues with very small/large attention scores. Most modern FlashAttention implementations have been extensively battle-tested, so the issue is almost always in user-side code (e.g., wrong `is_causal` setting, wrong mask shape).

For inference, you may see slightly different generated tokens due to the floating-point differences — this is expected and harmless.
</details>

<details>
<summary>What's the difference between FlashAttention-2 and FlashAttention-3, and when does it matter?</summary>

**FlashAttention** (2022): the original. Tiled computation, online softmax. 2-4× speedup over standard attention on A100.

**FlashAttention-2** (2023): better parallelism (uses sequence dimension, not just batch and head), fewer non-matmul operations, ~2× speedup over FlashAttention-1. Now the default in PyTorch and HuggingFace.

**FlashAttention-3** (2024): targets Hopper architecture (H100, H200). Uses WGMMA (warp-group matrix multiply-accumulate) instructions, achieves 75% of theoretical peak on H100 for some configurations. 1.5-2× faster than FlashAttention-2 on H100.

When does the version matter?

- **A100 or older**: FlashAttention-2 is sufficient. FlashAttention-3 doesn't apply.
- **H100/H200**: FlashAttention-3 is the right choice for max throughput; ~2× over FA-2.
- **Consumer GPUs (RTX 4090)**: FlashAttention-2 with appropriate tile sizes works well.
- **Older GPUs (V100)**: FlashAttention-1 was the only option; FA-2 added V100 support later.

In practice: install the latest `flash-attn` package and trust the auto-selection. The version differences matter for benchmark teams, not for typical users.
</details>

<details>
<summary>Scenario: a researcher proposes "linear attention" claims O(n) memory like FlashAttention but no quality loss. Should you switch?</summary>

Healthy skepticism warranted. Linear attention approximates $\text{softmax}(QK^T)V$ with a factorized form $\phi(Q) (\phi(K)^T V)$. The math gives $O(n)$ instead of $O(n^2)$.

The "no quality loss" claim is the load-bearing part:

- For *some* tasks (especially short-range dependencies, long sequences with smooth attention patterns), linear attention matches softmax attention. Performer, Linear Transformer, RWKV, and Mamba are recent examples.
- For tasks needing *sharp* attention (precise needle-in-haystack retrieval, copy operations), softmax attention is significantly better. Linear attention's smooth feature maps lose the ability to attend strongly to a specific position.

Practical reality:

- 2020-2022 linear attention papers often overstated their parity with softmax. Replication studies usually showed 1-3% drops on complex benchmarks.
- 2023-2024 RNN-revival models (Mamba, RWKV, RetNet) are linear-time but use *recurrent* state rather than feature-map factorization. They're more competitive than older linear attention.
- Transformers with FlashAttention have *exact* $O(n^2)$ compute but $O(n)$ memory — for many use cases this is sufficient and avoids the quality risk.

When to consider linear attention: extreme context (1M+ tokens) where even FlashAttention's $O(n^2)$ compute becomes prohibitive. For typical 32K-128K context: FlashAttention dominates.
</details>

<details>
<summary>How would you debug a FlashAttention training run that produces NaN losses after 1000 steps?</summary>

Diagnostic checklist in order:

1. **Switch to math backend** and rerun the same data and step. If NaN persists, the issue is in your model/data, not FlashAttention.
2. **Check for inf scores** — extreme attention scores (e.g., from very large QK products) can produce inf after softmax exponentiation. FlashAttention's online softmax handles this with shifts, but extreme cases can still overflow in BF16.
3. **Mixed-precision issues**: if you're training in BF16/FP16, NaN can come from accumulating very small values. Try training the relevant computation in FP32 (loss scaling, layer norm parameters).
4. **Gradient explosion**: check pre-step gradient norms. If they spike before NaN appears, you have a gradient explosion issue. Reduce LR or add gradient clipping.
5. **Mask shape mismatch**: a common bug is mask shape `(batch, seq)` instead of `(batch, seq, seq)`, or vice versa. FlashAttention can silently produce wrong output, and accumulated wrong values become NaN.
6. **Causal flag mismatch**: training with `is_causal=False` then evaluating with `is_causal=True` (or vice versa) gives subtle bugs.

Reproducer recipe: save model + data + RNG state at the step before NaN, then bisect. The fix is rarely "FlashAttention has a bug" — it's almost always in the user-side code or hyperparameters.
</details>

<details>
<summary>FlashAttention reduces memory but not theoretical compute (FLOPs). Why does it improve wall-clock training time so much?</summary>

Wall-clock training time on GPUs is rarely bottlenecked by raw compute. It's usually bottlenecked by memory bandwidth (data movement) or memory capacity (fitting the model + activations in VRAM).

FlashAttention attacks both:

1. **Bandwidth**: standard attention reads/writes the $n \times n$ matrix to HBM repeatedly. FlashAttention keeps it in SRAM — eliminating ~80% of the HBM traffic for the attention layer.
2. **Capacity**: with $O(n)$ memory, you can fit larger batches or longer sequences. Larger batches mean better GPU utilization (FLOPs/sec actually used).

For an A100 doing attention on $n = 4096$, standard attention is at ~30% peak FLOPs (memory-bound), FlashAttention reaches ~70% peak FLOPs (compute-bound). The *same compute* takes ~half the time because the GPU isn't waiting for memory.

This is the core insight Horace He's "Making Deep Learning Go Brrrr" article (a recommended read): on modern hardware, deep learning is almost always memory-bound, not compute-bound. Optimizing memory access patterns (kernel fusion, FlashAttention, FSDP) gives bigger wins than optimizing the math.
</details>

<details>
<summary>Scenario: a teammate proposes training with FlashAttention AND bfloat16 mixed precision. Are there hidden interactions you should worry about?</summary>

This combination is standard and generally works well — but there are real interactions worth knowing.

**Good news**:

- FlashAttention is designed to work in BF16/FP16. Modern implementations have BF16 as the *primary* path; FP32 is fallback.
- BF16's wider exponent range (vs FP16) helps with the exp() in softmax, reducing overflow risk.
- Combined throughput on A100/H100 is dramatically better than FP32 standard attention.

**Hidden interactions**:

1. **Accumulation precision**: FlashAttention internally accumulates the running max, running sum, and output in FP32 *even though* Q/K/V are BF16. This is essential — accumulating in BF16 would lose precision rapidly across long sequences. Most implementations get this right, but custom kernels can mess it up.
2. **Softmax stability**: the online softmax trick requires careful rescaling. In BF16, the exp(score - max) term can underflow for very negative scores. FlashAttention handles this in FP32 accumulation, but you can still see NaN if scores have extreme outliers.
3. **Gradient accumulation**: backpropagating through FlashAttention requires recomputing parts of the forward pass. Gradient values in BF16 are coarser than activations, so vanishing gradients are slightly more common with deep transformers + FlashAttention + BF16.
4. **Loss scaling not needed**: unlike FP16, BF16 doesn't need loss scaling because the exponent range matches FP32. This simplifies the training loop.
5. **Hardware-specific kernels**: FlashAttention-3 uses Hopper-specific BF16 instructions (WGMMA). On older hardware, the BF16 path may not be as fast as FP16.

For practical work: BF16 + FlashAttention-2 is the default modern training recipe. If you see NaN, check (a) extreme attention scores from unusual data, (b) BF16 gradient underflow in late training, (c) implementation bugs in custom code. The combination is robust enough that most teams don't think about it.

Edge case: if you're training in *FP8* (H100 inference, MX formats for training), FlashAttention support is newer and more error-prone. Stick to BF16 for training; experiment with FP8 for inference.
</details>

## Points to remember

- FlashAttention is *exact*, not approximate. Same output as standard attention, computed differently.
- The win comes from never materializing the $n \times n$ attention matrix in HBM — keeps it in fast SRAM via tiling and online softmax.
- $O(n^2)$ memory → $O(n)$ memory. Wall-clock speedup 2-8× depending on sequence length.
- Online softmax is the trick: process attention scores incrementally with running max and sum, no two-pass needed.
- PyTorch 2.0+ uses FlashAttention automatically via `F.scaled_dot_product_attention` when hardware allows.
- Most useful for sequences ≥ 1024 tokens. For very short sequences ($n < 256$), tiling overhead negates savings.
- FlashAttention is for *training and prefill*. For autoregressive decoding (single-token Q), FlashDecode + PagedAttention are the relevant tools.
- GQA / MQA reduce KV cache memory (orthogonal to FlashAttention). Combined, they enable practical long-context serving.
- Sparse attention (Longformer, sliding window) is a different lever: changes the *math* to skip distant tokens. FlashAttention preserves the math.
- Custom attention biases (ALiBi, RoPE adjustments) require special handling. Use FlexAttention (PyTorch 2.5+) for arbitrary differentiable score modifications.
- FlashAttention-3 is H100-specific. FlashAttention-2 is the default for A100 and older.
- Modern deep learning is usually *memory-bound*, not compute-bound. Memory-access optimization (FlashAttention, kernel fusion) is the highest-leverage performance work.

## Further reading

- [arXiv: FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135) — the original paper, with hardware-aware analysis of memory hierarchy
- [arXiv: FlashAttention-2 (Dao 2023)](https://arxiv.org/abs/2307.08691) — better parallelism and partitioning across sequence dimension
- [arXiv: FlashAttention-3 (Shah et al. 2024)](https://arxiv.org/abs/2407.08608) — Hopper-specific kernels using WGMMA, achieves 75% peak on H100
- [Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) — the canonical explanation of memory-bound vs compute-bound deep learning
- [arXiv: PagedAttention (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180) — the vLLM serving paper that complements FlashAttention at inference
- [arXiv: GQA (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention as used in LLaMA 3 and Mistral
- [arXiv: Longformer (Beltagy et al. 2020)](https://arxiv.org/abs/2004.05150) — sliding-window + global tokens approach to long context
- [PyTorch blog — FlexAttention](https://pytorch.org/blog/flexattention/) — programmable attention masks/biases with FlashAttention speed
- [Lightning AI — Understanding FlashAttention](https://lightning.ai/pages/community/tutorial/flash-attention/) — visual walkthrough of tiling and the online softmax algorithm

## Common mistakes

- Not using `is_causal=True` when using `F.scaled_dot_product_attention` for a decoder — produces incorrect outputs without the causal mask
- Assuming FlashAttention is always faster for short sequences — for $n < 256$, standard attention may be faster (FlashAttention has tiling overhead)
- Forgetting that FlashAttention does not support custom attention biases easily — ALiBi-style biases require special handling in the tiled computation

## Final takeaway

FlashAttention eliminates the $O(n^2)$ memory bottleneck of standard attention by tiling the computation and using online softmax, keeping all intermediates in fast SRAM. The result is identical to standard attention but 3–8× faster and linear in memory. FlashAttention-2 is the default in all modern LLM frameworks and is what makes 128k+ context windows practical. Combined with GQA (fewer KV heads) and paged KV cache management, it enables efficient deployment of large transformers at scale.

## References

- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
- Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP.
- Child, R., et al. (2019). Generating Long Sequences with Sparse Transformers (Sparse Attention). OpenAI.
