---
id: transformer-inference
title: "Transformer inference step by step"
sidebar_label: "84 · Transformer Inference"
sidebar_position: 84
slug: /theory/dnn/transformer-inference-step-by-step
description: "How a trained transformer generates text token by token: the autoregressive loop, greedy decoding, beam search, temperature sampling, and KV caching for efficient generation."
tags: [transformer, inference, decoding, kv-cache, beam-search, deep-learning]
---

# Transformer inference step by step

> **TL;DR.** Training looks at the whole answer at once and predicts every position in parallel (teacher forcing). Inference can only see what has been generated so far — it produces text one token at a time, feeding each output back as input. The decoding strategy (greedy, beam search, top-p) picks *which* token to emit; KV caching is what makes the loop fast enough to run in real time.

Training a transformer uses teacher forcing: the decoder sees the ground-truth prefix at every step in parallel. Inference is different — the model must generate tokens one at a time, each step feeding its own previous output back as input. This autoregressive loop is the fundamental regime of all text generation.

## One-line definition

Transformer inference is an autoregressive process: at each step, the model conditions on all previously generated tokens to produce a probability distribution over the vocabulary, picks the next token, appends it, and repeats until the end-of-sequence token is reached.

![Transformer decoding — the model generates one token per step, feeding each output back as input; the encoder's output is fixed and re-used at every decoder step](https://jalammar.github.io/images/t/transformer_decoding_2.gif)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Inference mechanics explain why language models behave the way they do in practice. Greedy decoding produces repetitive text; beam search balances quality and diversity; temperature and top-p sampling control creativity. KV caching is why modern LLMs can generate hundreds of tokens per second despite recomputing attention at every step. Understanding these details is essential for deploying and debugging generation systems.

## Try it interactively

Before reading further, play with these tools to see autoregressive generation in action:

- **[OpenAI Playground](https://platform.openai.com/playground)** — toggle temperature, top-p, max tokens; watch how output changes
- **[HuggingFace Inference Playground](https://huggingface.co/playground)** — same idea, open-source models
- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** — step through a single GPT-2 forward pass and see logits at each position
- **[bbycroft LLM Visualization](https://bbycroft.net/llm)** — animated 3D walkthrough of a real transformer generating one token

A quick trick: in any of these playgrounds, set temperature to 0 and re-run the same prompt twice — the output is identical (greedy). Then set temperature to 1.5 — the same prompt produces different completions every time.

## Training vs. inference: teacher forcing vs. autoregression

During **training**, the decoder processes the full target sequence in parallel using teacher forcing:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

The ground-truth tokens $x_1, \ldots, x_{t-1}$ are fed as input; the model predicts all positions simultaneously using the causal mask.

During **inference**, the model has no ground-truth prefix. It starts with only the start token $\langle \text{BOS} \rangle$ (or a user prompt) and generates one token at a time:

$$
x_{t+1} \sim p_\theta(\cdot \mid x_1, x_2, \ldots, x_t)
$$

Each generated token is appended to the sequence and becomes part of the context for the next step.

```mermaid
flowchart TD
    subgraph "Training (parallel, teacher forcing)"
        T1["[BOS] The cat sat"] --> TM["Decoder (masked)"] --> TL["p(on | ...) loss"]
    end
    subgraph "Inference (sequential, autoregressive)"
        I1["[BOS]"] --> IM1["Step 1"] --> I2["The"]
        I2 --> IM2["Step 2"] --> I3["cat"]
        I3 --> IM3["Step 3"] --> I4["sat"]
        I4 --> IM4["Step 4"] --> I5["[EOS]"]
    end
```

### A useful analogy

Training is like an open-book exam: the model can see every answer while learning to predict each one. Inference is like writing an essay live — every sentence depends on everything you've already written, and a wrong word three sentences ago colors everything that follows. This asymmetry is called **exposure bias**: at training time the model only sees correct prefixes; at inference time it has to live with its own mistakes.

## The autoregressive generation loop

```
Algorithm: Autoregressive generation

Input: encoder output (for seq2seq) or prompt token IDs
Output: generated token sequence

1. x = [BOS_TOKEN_ID]          # start with begin-of-sequence token
2. while len(x) < max_length:
3.     logits = model(x)[-1]   # get logits for last position: (vocab_size,)
4.     next_id = decode(logits) # apply decoding strategy
5.     x.append(next_id)
6.     if next_id == EOS_TOKEN_ID:
7.         break
8. return x[1:]                 # strip BOS, return generated tokens
```

At each step, only the last position's logits matter — the model predicts only $x_{t+1}$, not all positions.

### Worked example: 5-token vocabulary

Imagine a tiny model with vocabulary $\{$BOS, "cat", "sat", "mat", EOS$\}$ that has been trained on the sentence "cat sat mat". Step by step with greedy decoding:

| Step | Input so far | Top-1 predicted next token | Sequence after |
|------|--------------|----------------------------|----------------|
| 1 | `[BOS]` | "cat" | `[BOS, cat]` |
| 2 | `[BOS, cat]` | "sat" | `[BOS, cat, sat]` |
| 3 | `[BOS, cat, sat]` | "mat" | `[BOS, cat, sat, mat]` |
| 4 | `[BOS, cat, sat, mat]` | "EOS" | `[BOS, cat, sat, mat, EOS]` |
| 5 | — | (loop exits) | Final output: `cat sat mat` |

Each row reruns the model on the *full* sequence so far (or a single new token if KV caching is on; see below).

## Seq2seq inference: encoder runs once, decoder loops

For an encoder-decoder transformer (translation, summarization), the two stacks behave very differently at inference. Worked end-to-end on the running translation example: source = `"We are friends"`, target = `"हम दोस्त हैं"`.

### Encoder vs. decoder at inference

| Component | Training | Inference |
|---|---|---|
| Encoder | Parallel self-attention over all source tokens | **Identical** — same parallel pass |
| Decoder | Non-autoregressive (parallel, teacher forcing, with causal mask) | **Autoregressive** — one token per step |

The encoder has *no* mode-switch between training and inference. Only the decoder changes regime.

### The encoder pass — done exactly once

`"We are friends"` is fed through the encoder a single time:

1. Tokenize → `["We", "are", "friends"]`
2. Embed → 3 vectors of dimension 512
3. Add positional encoding
4. Pass through $N$ encoder blocks (self-attention + FFN, with residual + LayerNorm)
5. Result: three contextual vectors $\text{enc}_\text{We}, \text{enc}_\text{are}, \text{enc}_\text{friends}$

These are **stored** and reused at every decoder step — every cross-attention sublayer in every decoder block in every time step uses the same encoder outputs as its K and V source. The encoder never runs again during this query.

### The decoder loop — one step per output token

The decoder is invoked once per output token. At every step, the **growing list of generated tokens** becomes the decoder input.

| Step | Decoder input | Decoder output used for prediction | Predicted token |
|---|---|---|---|
| 1 | `[<SOS>]` | last position → projects to vocab | `हम` |
| 2 | `[<SOS>, हम]` | last position only | `दोस्त` |
| 3 | `[<SOS>, हम, दोस्त]` | last position only | `हैं` |
| 4 | `[<SOS>, हम, दोस्त, हैं]` | last position only | `<EOS>` → stop |

Final translation: `हम दोस्त हैं`.

### What each decoder step actually computes

At step $t$ the decoder runs the same three sublayers per block as during training, but on a $t$-token input:

1. **Embed + positional encoding** all $t$ tokens.
2. **Masked self-attention** (with causal mask — see below for why this is *still* required).
3. **Cross-attention** — Q from the current $t$ decoder positions; K, V from the cached encoder outputs (3 source vectors).
4. **FFN** at each of the $t$ positions.
5. After $N$ blocks, take **only the last position's vector** $y^f_t$, project to vocabulary, softmax, pick the next token.

The outputs for positions $1, \ldots, t-1$ are computed but **thrown away** — they were already used (or will be reproduced) at their own respective steps. Only position $t$ is consumed.

### Why the causal mask is *still* applied at inference

A natural question: at step $t$ we have already generated tokens $1, \ldots, t$ — there are no future tokens — so why mask?

- **Consistency with training.** The model's parameters were learned under the causal-mask constraint. Removing the mask at inference would let position $i < t$ attend to positions $i+1, \ldots, t$ that came *after* it in generation order, producing a distribution the model never saw during training.
- **It doesn't change the predicted token anyway.** Only position $t$'s output is consumed, and position $t$ can already attend to all $\leq t$. The mask is essentially free.
- **It does matter for the intermediate positions** that feed into KV caching — without the mask, cached representations at earlier positions would differ from what those positions produced during their own generation step, breaking the cache's correctness.

### Cost growth across the loop

Without KV caching, decoder input grows: step 1 sees 1 token, step 2 sees 2, ..., step $T$ sees $T$. The decoder runs $T$ times on sequences of length $1, 2, \ldots, T$, giving $O(T^2)$ total work — see the [KV caching](#kv-caching) section below for how this becomes $O(T)$ per step.

```mermaid
flowchart LR
    src["Source: 'We are friends'"] --> enc["Encoder (runs ONCE)"]
    enc --> enc_out["enc_We, enc_are, enc_friends\n(cached, reused every step)"]
    subgraph "Decoder loop (runs once per output token)"
        s1["Step 1: [SOS]"] --> dec1["Decoder"] --> hum["हम"]
        hum --> s2["Step 2: [SOS, हम]"] --> dec2["Decoder"] --> dost["दोस्त"]
        dost --> s3["Step 3: [SOS, हम, दोस्त]"] --> dec3["Decoder"] --> hai["हैं"]
        hai --> s4["Step 4: [SOS, हम, दोस्त, हैं]"] --> dec4["Decoder"] --> eos["EOS → stop"]
    end
    enc_out -.cross-attn K,V.-> dec1 & dec2 & dec3 & dec4
```

## Decoding strategies

### Greedy decoding

Always pick the highest-probability token:

$$
x_{t+1} = \arg\max_v p_\theta(v \mid x_{\leq t})
$$

**Advantage**: fast, deterministic.  
**Disadvantage**: tends to produce repetitive or suboptimal sequences. Locally optimal choices can lead to globally poor outputs ("the cat sat on the cat sat on...").

### Beam search

Maintain the top $B$ candidate sequences (beams) at each step, expand each by one token, keep the top $B$ by cumulative log-probability:

$$
\text{score}(x_{1:t}) = \sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})
$$

**Advantage**: considers multiple hypotheses; usually better output quality than greedy.  
**Disadvantage**: $B$ times more computation; tends to produce generic, safe text; degrades in open-ended generation.

Beam search is standard for translation and summarization ($B = 4$ or $B = 5$); less suitable for creative generation.

### Temperature sampling

Scale logits by temperature $\tau$ before softmax:

$$
p_\tau(v) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}
$$

| τ | Behavior | Best for |
|---|----------|----------|
| 0 | Approaches greedy (deterministic) | Code, math, factual answers |
| 0.2–0.5 | Slightly randomized, mostly safe | Question answering |
| 0.7–1.0 | Balanced quality & creativity | Most chat use cases |
| 1.2–2.0 | Diverse, sometimes incoherent | Brainstorming, creative writing |
| ∞ | Uniform random sampling | Never useful in practice |

### Top-$k$ sampling

Sample only from the $k$ highest-probability tokens (zero out the rest):

$$
\tilde{p}(v) \propto p(v) \cdot \mathbb{1}[v \in \text{top-}k]
$$

Prevents the model from accidentally sampling from the long tail of nonsense tokens. GPT-2 defaults to $k = 50$.

### Top-$p$ (nucleus) sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

$$
\mathcal{V}^{(p)} = \arg\min_{V' \subseteq V} \left\{ \sum_{v \in V'} p(v) \geq p \right\}
$$

The set $\mathcal{V}^{(p)}$ adapts in size: when the model is confident, the nucleus is small (1–3 tokens); when uncertain, the nucleus is large. Top-$p = 0.9$ is a common default that works better than fixed top-$k$.

```mermaid
flowchart LR
    logits["Raw logits\n(vocab_size)"] --> temp["÷ temperature τ"]
    temp --> softmax["Softmax → probabilities"]
    softmax --> topk["Top-k filter\n(keep top k)"]
    topk --> topp["Top-p nucleus\n(keep cumulative prob ≤ p)"]
    topp --> sample["Sample next token"]
```

### Side-by-side: same prompt, different strategies

| Prompt: "Once upon a time" | Output (illustrative) |
|----------------------------|------------------------|
| Greedy | "there was a king who lived in a castle. The king was very rich and very happy. The king was very rich..." |
| Beam search ($B=5$) | "there was a brave knight who saved the kingdom from a fearsome dragon." |
| Top-p = 0.9, τ = 0.7 | "in a small fishing village, a curious girl named Mira found a peculiar shell on the shore." |
| Top-p = 0.9, τ = 1.5 | "the moon decided to grow legs and hop down to argue with a particularly opinionated cactus." |

The first row shows greedy's repetitiveness; the last row shows high-temperature creativity tipping into chaos.

## KV caching

Without caching, generating step $t$ requires recomputing keys and values for all $t-1$ previous tokens — $O(t^2)$ total work over the generation. KV caching stores computed key-value pairs from previous steps and only computes for the new token at each step.

**Without KV cache**: at step $t$, re-run the full forward pass on $x_{1:t}$. Cost per step: $O(t \cdot d)$. Total cost for $T$ tokens: $O(T^2 \cdot d)$.

**With KV cache**: at step $t$, compute $K_t$ and $V_t$ only for the new token, concatenate with cached $K_{1:t-1}$ and $V_{1:t-1}$, run attention. Cost per step: $O(t \cdot d)$ for attention but $O(d)$ for new key-value computation. The cache grows by one entry per step.

```mermaid
flowchart TD
    subgraph "Without KV cache (step t)"
        input_t["x₁, x₂, ..., xₜ"] --> full_fwd["Full encoder pass\ncompute all K, V"] --> logits_t["logits for position t"]
    end
    subgraph "With KV cache (step t)"
        new_tok["xₜ only"] --> new_kv["Compute Kₜ, Vₜ"]
        cache["Cached K₁..Kₜ₋₁,\nV₁..Vₜ₋₁"] --> concat["Concat K,V"]
        new_kv --> concat
        concat --> attn["Attention\nxₜ queries all K,V"] --> logits_t2["logits for position t"]
    end
```

### How the cache shape evolves

For a single layer with batch size 1, 8 heads, and $d_{\text{head}} = 64$:

| Generation step | Q shape (per layer) | K, V cache shape |
|-----------------|---------------------|--------------------|
| 1 | (1, 8, 1, 64) | (1, 8, 1, 64) |
| 2 | (1, 8, 1, 64) | (1, 8, 2, 64) |
| 3 | (1, 8, 1, 64) | (1, 8, 3, 64) |
| ... | ... | ... |
| $t$ | (1, 8, 1, 64) | (1, 8, $t$, 64) |

Q stays small (just the new token); K and V grow linearly. This is the key insight: each step does $O(t)$ work, not $O(t^2)$.

**Memory cost of KV cache**: for each layer, storing all keys and values for a sequence of length $T$ requires:

$$
\text{memory} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times T \times d_{\text{head}} \times \text{bytes\_per\_float}
$$

For a 7B-parameter model with 32 layers, 32 heads, $d_{\text{head}} = 128$, sequence length 4096, float16: **~2 GB per batch item**. This is why context length directly limits batch size during inference, and why techniques like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) — which share K,V across heads — are standard in modern LLMs.

## What if KV caching is off?

Try this experiment with HuggingFace `transformers`:

```python
import time, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2")
m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
ids = tok("Once upon a time", return_tensors="pt").input_ids

for use_cache in [True, False]:
    t0 = time.time()
    out = m.generate(ids, max_new_tokens=200, use_cache=use_cache, do_sample=False)
    print(f"use_cache={use_cache}: {time.time()-t0:.2f}s")
```

You will typically see the no-cache run take **5–10× longer** for 200 tokens, and the gap grows quadratically with sequence length.

## Stopping criteria

Generation stops when:
1. The model produces `[EOS]` (end-of-sequence token)
2. `max_new_tokens` is reached
3. A stop string is matched (e.g., `\n\n`, `</answer>`)
4. (For batched inference) all sequences in the batch have hit one of the above

## Python code

### Complete autoregressive generation from scratch

```python
import torch
import torch.nn.functional as F


def greedy_decode(model, encoder_output, start_id: int, end_id: int,
                  max_len: int = 50, device: str = "cpu") -> list[int]:
    """
    Greedy decoding for a seq2seq transformer.
    Returns list of generated token IDs (excluding BOS, including EOS).
    """
    model.eval()
    generated = [start_id]

    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([generated], device=device)  # (1, current_len)

            # Build causal mask for current target length
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device), diagonal=1
            ).bool()  # True = masked

            # Forward pass through decoder only (encoder output is cached)
            logits = model.decode(tgt, encoder_output, tgt_mask)  # (1, tgt_len, vocab)
            next_logits = logits[0, -1, :]  # logits for last position: (vocab,)

            # Greedy: pick the highest probability token
            next_id = next_logits.argmax().item()
            generated.append(next_id)

            if next_id == end_id:
                break

    return generated[1:]  # strip BOS


def temperature_sample(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = 0, top_p: float = 1.0) -> int:
    """
    Sample next token with temperature, top-k, and top-p filtering.
    Args:
        logits:      (vocab_size,) raw logits
        temperature: scaling factor (1.0 = unmodified)
        top_k:       if > 0, keep only the top k tokens
        top_p:       nucleus sampling threshold (0.0–1.0)
    Returns:
        sampled token ID
    """
    # Apply temperature
    logits = logits / max(temperature, 1e-8)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[-1]
        logits[logits < kth_val] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits[sorted_indices] = sorted_logits

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ============================================================
# Demonstrate the decoding strategies
# ============================================================
torch.manual_seed(42)
vocab_size = 10
logits = torch.tensor([0.5, 1.2, 0.3, 2.1, 0.1, 0.8, 1.5, 0.2, 0.9, 0.4])

print("=== Greedy decoding ===")
greedy_id = logits.argmax().item()
print(f"Next token: {greedy_id}")  # Always token 3 (highest logit 2.1)

print("\n=== Temperature sampling (τ=0.7, τ=1.0, τ=1.5) ===")
for tau in [0.7, 1.0, 1.5]:
    counts = {}
    for _ in range(2000):
        tid = temperature_sample(logits.clone(), temperature=tau)
        counts[tid] = counts.get(tid, 0) + 1
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    print(f"  τ={tau}: most sampled tokens = {top3}")

print("\n=== Top-p sampling (p=0.9) ===")
counts = {}
for _ in range(2000):
    tid = temperature_sample(logits.clone(), temperature=1.0, top_p=0.9)
    counts[tid] = counts.get(tid, 0) + 1
print(f"  Sampled tokens: {sorted(counts.keys())}")
```

### Try it yourself: experiments

| Experiment | Modification | Expected observation |
|------------|--------------|----------------------|
| What does τ → 0 do? | `temperature=0.01` | Almost always picks the same token (greedy) |
| What does τ → ∞ do? | `temperature=100.0` | Distribution becomes ~uniform; all tokens equally likely |
| Effect of top-k = 1 | `top_k=1` | Equivalent to greedy |
| Effect of top-p = 0.5 | `top_p=0.5` | Only the top few high-probability tokens sampled |
| Combine top-k + top-p | `top_k=10, top_p=0.9` | Belt-and-suspenders filter (rarely makes a difference) |

### KV cache in practice (GPT-style decoder-only)

```python
import torch
import torch.nn as nn


class CachedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with KV cache support for inference.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                past_kv: tuple[torch.Tensor, torch.Tensor] = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       (batch, 1, d_model)   — single new token at inference
            past_kv: (past_K, past_V) cached from previous steps
        Returns:
            output, (new_K, new_V)
        """
        batch, seq, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape to (batch, heads, seq, d_k)
        def split(t):
            return t.reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(Q), split(K), split(V)

        # Append new K, V to cache
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)  # (batch, heads, past+1, d_k)
            V = torch.cat([past_V, V], dim=2)

        new_kv = (K, V)

        # Attention: query is only the new token, keys/values span all past tokens
        import math
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, heads, 1, past+1)
        attn = scores.softmax(dim=-1)
        out = attn @ V  # (batch, heads, 1, d_k)

        # Merge heads
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.W_O(out), new_kv


# Simulate 5-step generation with KV cache
d_model, num_heads, vocab_size = 64, 4, 100
attn = CachedMultiHeadAttention(d_model=d_model, num_heads=num_heads)
lm_head = nn.Linear(d_model, vocab_size)

generated = []
past_kv = None

for step in range(5):
    x = torch.randn(1, 1, d_model)   # in practice: embed the last generated token
    out, past_kv = attn(x, past_kv)
    logits = lm_head(out[:, -1, :])  # (1, vocab_size)
    next_id = logits.argmax(dim=-1).item()
    generated.append(next_id)
    K_shape = past_kv[0].shape  # (batch, heads, step+1, d_k)
    print(f"Step {step+1}: token={next_id}, KV cache shape={K_shape}")

print(f"Generated: {generated}")
```

## Comparison of decoding strategies

| Strategy | Deterministic? | Quality | Diversity | Speed | Use case |
|---|---|---|---|---|---|
| Greedy | Yes | Moderate | Low | Fastest | Quick baseline, code |
| Beam search ($B=5$) | Yes | High | Low | $B\times$ slower | Translation, summarization |
| Top-$k$ sampling | No | Good | Medium | Fast | Creative generation |
| Top-$p$ sampling | No | Good | Adaptive | Fast | Open-ended generation |
| Temperature + top-$p$ | No | Tunable | High | Fast | Chatbots, storytelling |

## Cross-references

- **Prerequisite:** [81 — Masked Self-Attention in the Decoder](./81-masked-self-attention-in-the-transformer-decoder.md) — the causal mask used at every step
- **Prerequisite:** [83 — Transformer Decoder Architecture](./83-transformer-decoder-architecture.md) — what the decoder actually computes per step
- **Related:** [85 — Transformer Training Objectives](./85-transformer-training-objectives.md) — why teacher forcing creates the train/inference mismatch
- **Follow-up:** [88 — GPT (Decoder-Only Causal LM)](./88-gpt-decoder-only-causal-lm.md) — the simplest setup where this loop is the entire model

## Interview questions

<details>
<summary>Why can't you use teacher forcing during inference?</summary>

Teacher forcing feeds ground-truth tokens as context during training. During inference, there are no ground-truth tokens — you are generating them. The model must use its own previously generated tokens as context. This train/inference mismatch (called "exposure bias") means errors compound: if the model generates a wrong token, all subsequent tokens are conditioned on that error. Training techniques like scheduled sampling try to reduce this gap.
</details>

<details>
<summary>What is the difference between greedy decoding and beam search?</summary>

Greedy decoding picks the highest-probability token at each step — locally optimal but globally suboptimal. Beam search maintains $B$ candidate sequences and expands all of them at each step, keeping the top $B$ by cumulative log-probability. This explores more of the output space and generally produces higher-quality outputs for tasks with structured, correct answers (translation, summarization). However, beam search is $B$ times slower and tends to produce generic output for creative tasks.
</details>

<details>
<summary>What does KV caching speed up and how does it work?</summary>

Without caching, each new decoding step re-runs the full transformer on the entire sequence history — O(t) work per step, O(T²) total for T tokens. KV caching saves the key and value tensors computed in previous steps. At step t, only the new token's Q, K, V are computed; the new K and V are appended to the cache and full attention is run with the accumulated K, V. This reduces per-step computation for new tokens from O(t) to O(1) for key/value projection (though attention itself is still O(t) because the new query attends to all cached keys).
</details>

<details>
<summary>What is top-p sampling and why is it preferred over top-k?</summary>

Top-k sampling always keeps exactly k tokens regardless of the probability distribution. When the model is confident (one token has very high probability), k=50 includes many improbable tokens. When the model is uncertain (probability spread over hundreds of tokens), k=50 cuts off plausible options. Top-p (nucleus) sampling adapts: it keeps the smallest set of tokens whose cumulative probability reaches p. When the model is confident, the nucleus is small (1–3 tokens); when uncertain, the nucleus is large. This gives better calibration between quality and diversity.
</details>

<details>
<summary>Why does the KV cache use so much memory in long-context inference?</summary>

The cache stores keys and values for every layer, every head, and every past token. Memory grows linearly with sequence length and with model size. For a 7B model at 4K context: ~2GB per batch item. For a 70B model at 32K context: ~50GB per batch item. This is the dominant memory cost during long-context inference and is the reason for grouped-query / multi-query attention (which share K,V across heads) and for techniques like PagedAttention (vLLM) that manage cache fragments efficiently across requests.
</details>

## Common mistakes

- Calling `model(full_sequence)` at every step without KV caching — generates correctly but is $O(T^2)$ slower than necessary.
- Applying softmax twice — `F.cross_entropy` expects raw logits; `torch.multinomial` needs probabilities. Confusing these gives wrong sampling.
- Forgetting to set `model.eval()` during generation — dropout is active in training mode, giving different (stochastic) outputs each run.
- Using beam search for open-ended generation (chatbots, stories) — it degenerates to repetitive, safe text; top-p sampling is better.
- Not passing a causal mask when running the decoder for multi-token prompts — the prompt tokens must not attend to future positions.
- Comparing temperatures across models without normalizing — the same `temperature=0.7` can feel very different on a model with calibrated logits vs. an over-confident one.

## Final takeaway

Transformer inference is token-by-token autoregressive generation. The model predicts a probability distribution at each step, a decoding strategy (greedy, beam search, or sampling) selects the next token, and the loop continues until EOS. KV caching makes this efficient by reusing previously computed key-value pairs. The choice of decoding strategy — not just model quality — determines whether the output is repetitive, generic, or diverse and creative.

## References

- Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
- Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration. ICLR. (introduces nucleus/top-p sampling)
- Pope, R., et al. (2023). Efficiently Scaling Transformer Inference. MLSys.
- Kwon, W., et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention (vLLM). SOSP.
- Brendan Bycroft — [LLM Visualization](https://bbycroft.net/llm).
