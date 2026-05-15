---
id: peft-lora
title: "Parameter-efficient fine-tuning: LoRA, adapters, and prefix tuning"
sidebar_label: "91 · LoRA & PEFT"
sidebar_position: 91
slug: /theory/dnn/parameter-efficient-fine-tuning-lora
description: "How to fine-tune large language models with a fraction of the parameters: LoRA's low-rank decomposition, adapter layers, prefix tuning, and why PEFT methods are essential for LLMs."
tags: [lora, peft, adapters, prefix-tuning, fine-tuning, llm, transformers, deep-learning]
---

# Parameter-efficient fine-tuning: LoRA, adapters, and prefix tuning

> **TL;DR.** Full fine-tuning a 7B-param model needs ~80 GB of GPU memory and saves 7 B parameters per task. **LoRA** is the trick that changed this: freeze the original weights, add a tiny pair of low-rank matrices `B·A` (rank 8–64) alongside each attention projection, train *only* those. You get ~99% of the quality with **less than 1% of the parameters** to train and store — a 7B model's LoRA adapter is often under 50 MB. This is how every modern fine-tune of LLaMA / Mistral / etc. is done in practice.

Full fine-tuning a 7B-parameter LLM requires updating 7 billion parameters and storing 7B gradients — consuming ~80 GB of GPU memory for a single fine-tuning run. Parameter-efficient fine-tuning (PEFT) methods achieve comparable task performance while updating only 0.1–1% of parameters. LoRA is the most widely used PEFT method today: it is the standard approach for instruction tuning, domain adaptation, and task-specific customization of large language models.

## Prerequisites

- [77 — Multi-Head Attention](./77-multi-head-attention-in-transformers.md) — LoRA targets the Q, K, V, O projection matrices inside attention
- [88 — GPT (Decoder-Only)](./88-gpt-decoder-only-causal-lm.md) — the architecture LoRA is most commonly applied to
- [90 — Fine-Tuning Transformers](./90-fine-tuning-transformers.md) — the full-fine-tuning baseline LoRA improves on
- [26 — Weight Decay / Regularization](./26-regularization-weight-decay-l1-and-l2-in-neural-networks.md) — LoRA's low-rank constraint is itself an implicit regularizer
- [30 — Xavier / He Initialization](./30-xavier-glorot-and-he-initialization.md) — why $A$ is initialized non-zero and $B$ is initialized zero

## Try it interactively

- **[Hugging Face PEFT library](https://github.com/huggingface/peft)** — official LoRA implementation; wraps any model with one config object
- **[Unsloth](https://github.com/unslothai/unsloth)** — 2× faster QLoRA fine-tuning on consumer GPUs (free Colab notebooks)
- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — config-driven LoRA / QLoRA for LLaMA-class models
- **[QLoRA paper repo](https://github.com/artidoro/qlora)** — fine-tune a 65B model on a single 48GB GPU
- **[LoRA Land (Predibase)](https://huggingface.co/predibase)** — 25+ specialized LoRA adapters for LLaMA, free to try

## One-line definition

PEFT methods adapt a pre-trained model to a new task by adding a small number of trainable parameters while keeping the original weights frozen — LoRA adds low-rank matrices to weight projections, adapters insert small bottleneck layers, and prefix tuning prepends learnable tokens to the context.

![BERT BASE vs BERT LARGE — fine-tuning a model this size from scratch would require updating hundreds of millions of parameters; LoRA reduces this by 100–1000×](https://jalammar.github.io/images/bert-base-bert-large.png)
*Source: [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)*

## Why this topic matters

LoRA is how most LLM fine-tuning in industry and research is done. It reduces GPU memory requirements by 3–10x compared to full fine-tuning, enables fine-tuning on consumer hardware (single 24 GB GPU), and produces results within 1–2% of full fine-tuning. Understanding LoRA is essential for any practical work with LLMs.

## The core problem: full fine-tuning at scale

For a 7B parameter LLM (e.g., LLaMA 2 7B):

| Stage | GPU memory |
|---|---|
| Model weights (bfloat16) | ~14 GB |
| Gradients | ~14 GB |
| Optimizer states (AdamW: 2 moments) | ~56 GB |
| Activations | Variable |
| **Total (full fine-tuning)** | **~100+ GB** |

This requires multiple high-end GPUs. LoRA reduces the trainable parameters (and thus gradient + optimizer memory) by keeping the base model frozen and adding tiny trainable matrices.

![LoRA decomposition — the original weight matrix W stays frozen; a parallel rank-r update BA is trained from scratch. At deployment, BA can be merged into W for zero inference overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_animated.gif)
*Source: [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/conceptual_guides/lora)*

## LoRA: Low-Rank Adaptation

**Key insight**: the weight update $\Delta W$ that occurs during fine-tuning is intrinsically low-rank. Rather than storing the full $\Delta W \in \mathbb{R}^{d \times k}$ (large matrix), we decompose it as:

$$
\Delta W = BA
$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$.

The modified forward pass:

$$
h = W_0 x + \frac{\alpha}{r} \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

- $W_0$: frozen pre-trained weights
- $B, A$: trainable LoRA matrices
- $r$: rank (typically 4, 8, 16, 32)
- $\alpha$: scaling factor (often set to $r$ so $\alpha/r = 1$)

**Initialization**: $A$ is initialized from $\mathcal{N}(0, \sigma^2)$; $B$ is initialized to 0. So $\Delta W = BA = 0$ at the start — the model begins as the original pre-trained model and learns the adaptation.

### Parameter savings

For a linear layer $W \in \mathbb{R}^{d \times k}$ with $d = k = 4096$ and $r = 8$:

| | Full fine-tuning | LoRA |
|---|---|---|
| Trainable params | $4096 \times 4096 = 16.7M$ | $2 \times 4096 \times 8 = 65.5K$ |
| Reduction | — | **256× fewer** |

For LLaMA 2 7B with LoRA applied to all Q/K/V/O projections, $r=16$:
- Full fine-tuning: 7B trainable parameters
- LoRA: ~21M trainable parameters (~0.3%)

### Where to apply LoRA

LoRA is typically applied to the attention projection matrices and sometimes the FFN:

| Matrix | Apply LoRA? | Notes |
|---|---|---|
| $W^Q$ (query) | Yes | Standard |
| $W^K$ (key) | Yes | Standard |
| $W^V$ (value) | Yes | Standard |
| $W^O$ (output) | Yes | Standard |
| $W_1$ (FFN up) | Optional | More capacity |
| $W_2$ (FFN down) | Optional | More capacity |
| Embedding | No | Not typically |

## Adapter layers

Adapters (Houlsby et al., 2019) insert small bottleneck layers inside each transformer block:

```
Input → Pre-trained layer → Adapter(Down-project → Activation → Up-project) → Add → LayerNorm → Next layer
```

The adapter down-projects from $d_{\text{model}}$ to a small bottleneck dimension $m$ (typically 64 or 128), applies a nonlinearity, then up-projects back:

$$
\text{Adapter}(h) = h + W_{\text{up}} \cdot f(W_{\text{down}} h)
$$

- $W_{\text{down}} \in \mathbb{R}^{m \times d}$, $W_{\text{up}} \in \mathbb{R}^{d \times m}$
- Residual connection: if adapter contribution is small at init, the block is approximately identity

**Comparison with LoRA**: Adapters add inference overhead (two extra linear layers per block). LoRA has zero inference overhead because $\Delta W = BA$ can be merged into $W_0 + \Delta W$ after training.

## Prefix tuning

Prefix tuning (Li & Liang, 2021) prepends learnable "prefix" tokens to the key and value of every attention layer. These are continuous vectors (not real tokens from the vocabulary) that can encode task-specific information:

$$
K = [K_{\text{prefix}}; K_{\text{input}}], \quad V = [V_{\text{prefix}}; V_{\text{input}}]
$$

The model's self-attention now attends to both the original input and the learnable prefix. Only the prefix parameters are trained.

**Problem**: directly optimizing prefix vectors is unstable. In practice, a small MLP reparameterizes the prefix: $\text{Prefix} = \text{MLP}(P)$ where $P$ is the actual trainable parameter.

## Comparison of PEFT methods

| Method | Added params | Inference overhead | Merge into weights? | Best for |
|---|---|---|---|---|
| Full fine-tuning | 100% | None | — | Best performance, large GPU |
| LoRA | 0.1–1% | None (mergeable) | Yes | LLM fine-tuning standard |
| QLoRA | 0.1–1% | None | Yes | 4-bit quantized LLMs, low memory |
| Adapters | 0.5–5% | Yes (extra layers) | No | Multi-task serving |
| Prefix tuning | 0.1–1% | Yes (longer context) | No | Few training examples |
| Prompt tuning | < 0.01% | Yes (extra tokens) | No | Very small models or large models |

## QLoRA: LoRA on quantized models

QLoRA (Dettmers et al., 2023) enables fine-tuning 65B parameter models on a single 48 GB GPU by:
1. Loading the base model in 4-bit NormalFloat (NF4) quantization
2. Applying LoRA adapters in 16-bit (bfloat16)
3. Using paged optimizers to handle memory spikes

Memory for fine-tuning LLaMA 2 7B:

| Method | GPU memory | GPU count |
|---|---|---|
| Full fine-tuning (bfloat16) | ~100 GB | 4× A100 |
| LoRA (bfloat16) | ~24 GB | 1× A100 or 1× RTX 4090 |
| QLoRA (4-bit) | ~10 GB | 1× RTX 3080 |

## Python code: LoRA with HuggingFace PEFT

```python
# pip install transformers peft bitsandbytes accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ============================================================
# Standard LoRA (bfloat16)
# ============================================================

model_name = "gpt2"   # Small model for demo; use "meta-llama/Llama-2-7b-hf" in practice
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # rank — controls capacity vs. efficiency
    lora_alpha=16,                # scaling: alpha/r applied to BA
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],   # GPT-2's attention projections
    bias="none",
)

# Apply LoRA to the model
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()
# Example output: "trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364"


# ============================================================
# Training loop (simplified)
# ============================================================
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

texts = [
    "Transformers are the backbone of modern NLP.",
    "LoRA reduces fine-tuning costs dramatically.",
    "Self-attention allows tokens to interact directly.",
]
encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

optimizer = AdamW(lora_model.parameters(), lr=3e-4)   # LoRA uses higher LR than full fine-tuning
total_steps = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps)

lora_model.train()
for step in range(total_steps):
    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = lora_model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    print(f"Step {step+1}/{total_steps}: loss={loss.item():.4f}")


# ============================================================
# Merge LoRA back into base model (zero inference overhead)
# ============================================================
# After training, merge BA into W0 for deployment
merged_model = lora_model.merge_and_unload()
# merged_model is now a standard model with W0 + BA merged into each weight
print(f"\nMerged model type: {type(merged_model)}")


# ============================================================
# Manual LoRA implementation (to understand the math)
# ============================================================
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA: y = (W0 + BA) x
    W0 is frozen; B and A are trainable.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        import torch.nn as nn
        self.r = r
        self.scale = alpha / r

        # Frozen original weights
        self.W0 = nn.Linear(in_features, out_features, bias=False)
        for param in self.W0.parameters():
            param.requires_grad = False

        # Trainable LoRA matrices
        self.A = nn.Linear(in_features, r, bias=False)    # down-project
        self.B = nn.Linear(r, out_features, bias=False)   # up-project

        # Initialize: A ~ N(0, σ²), B = 0
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W0(x) + self.scale * self.B(self.A(x))


import torch.nn as nn
# Demo
lora_layer = LoRALinear(in_features=512, out_features=512, r=8, alpha=16)

trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
print(f"\nLoRALinear: trainable={trainable:,}, frozen={frozen:,}")
# trainable = 8*512 + 512*8 = 8192  (tiny!)
# frozen    = 512*512 = 262144

x = torch.randn(4, 10, 512)   # (batch, seq, d)
out = lora_layer(x)
print(f"LoRALinear output: {out.shape}")   # (4, 10, 512)
```

## Rank selection guide

| Rank $r$ | Use case | Trainable params | Task performance |
|---|---|---|---|
| 2–4 | Memory-constrained, simple tasks | Very few | Lower bound |
| 8 | Default, most tasks | Standard | Good |
| 16 | Complex tasks, domain shift | Moderate | Better |
| 32–64 | Maximum capacity, near full fine-tuning | Significant | Near full FT |

**Rule of thumb**: start with $r=8$. If performance is insufficient, try $r=16$ or $r=32$. Going above $r=64$ rarely helps and approaches the cost of full fine-tuning.

## Interview questions

<details>
<summary>Why does LoRA work? Why is the weight update intrinsically low-rank?</summary>

Empirical evidence from Aghajanyan et al. (2020) shows that when fine-tuning a pre-trained model on a downstream task, the weight updates $\Delta W$ have low "intrinsic rank" — the task-specific adaptation can be captured in a low-dimensional subspace. Intuitively: the pre-trained model has already learned rich general representations. Fine-tuning on a specific task only needs to shift these representations slightly in a low-dimensional direction, not rewrite the entire weight matrix. The low-rank decomposition $\Delta W = BA$ exploits this structure, using only $2 \times d \times r$ parameters instead of $d^2$.
</details>

<details>
<summary>What is the difference between LoRA and adapters?</summary>

Both add small trainable modules while freezing the base model. LoRA decomposes the weight update as a low-rank product and can be merged into the original weights after training — zero inference overhead. Adapters insert extra feedforward layers in the transformer, which add computation at every forward pass. LoRA has become the dominant method because it can be merged, making it transparent to the inference pipeline. Adapters are preferred in multi-task settings where you want to swap task-specific modules at inference.
</details>

<details>
<summary>Why is the B matrix initialized to zero in LoRA?</summary>

$\Delta W = BA$. If $B$ is initialized to zero, then at the start of training $\Delta W = 0$, so the model outputs exactly the same as the frozen base model. This is an ideal starting point: the adaptation starts at zero and learns incrementally. If both $A$ and $B$ were initialized randomly, the initial adapter would perturb the pre-trained model's behavior before any training has happened, potentially degrading starting performance and making optimization harder.
</details>

<details>
<summary>Scenario: you train LoRA with rank 8 on a 7B model and observe perplexity stops improving after epoch 1. Bumping rank to 32 helps marginally. What's happening?</summary>

Rank 8 is the right *starting* point for instruction-style fine-tuning, but it may be too low for genuinely complex adaptations (e.g., learning a new domain language, code generation in a non-standard style, or many-task multi-skill training). The model has insufficient capacity to capture all the task-specific signal in 8 dimensions per layer.

Diagnosis steps:

1. **Compare to full fine-tuning** on a small subset (1-2 epochs): if FFT improves much more, LoRA capacity is the bottleneck.
2. **Check target_modules**: LoRA only on `q, v` is common but `q, k, v, o` (all attention) plus FFN matrices can help. Most modern recipes include FFN modules.
3. **Try DoRA or rank 64**: DoRA (Weight-Decomposed LoRA) explicitly separates magnitude from direction and sometimes beats LoRA at the same rank. Or just increase rank.
4. **Look at training loss vs val loss**: if both are stuck high, rank is the issue. If train drops but val plateaus, it's overfitting/regularization.

Counter-intuition: increasing rank doesn't always help linearly. Empirically there's often a sharp jump in capacity around rank 16-32, then a plateau where higher rank approaches full fine-tuning cost without improvement. The "right" rank is task-dependent.
</details>

<details>
<summary>What is DoRA and why might it beat plain LoRA?</summary>

DoRA (Liu et al. 2024, "Weight-Decomposed Low-Rank Adaptation") decomposes pretrained weight $W$ into a magnitude vector $m$ and a direction matrix $V = W / \|W\|$. LoRA is applied only to the *direction* component; the *magnitude* is separately trained.

$$W_{DoRA} = m \cdot \frac{V + BA}{\|V + BA\|}$$

The insight: full fine-tuning updates both magnitude and direction non-trivially, but LoRA's low-rank product mostly captures *direction* shifts. Separating magnitude as its own (cheap) parameter lets DoRA approximate FFT more closely.

In practice, DoRA matches FFT performance at lower ranks than LoRA can. The cost: slightly more parameters (~5% more than LoRA) and slightly more complex training. Increasingly common in 2024+ LoRA implementations.

The broader pattern: LoRA started as a single technique, but the field has spawned many variants (rsLoRA, LoRA+, AdaLoRA, DoRA, VeRA, GaLore) each optimizing different aspects. For most production work, LoRA or DoRA at rank 16 is the safe choice.
</details>

<details>
<summary>Scenario: you fine-tune LoRA adapters for 50 different customers (50 different domain tasks) on the same base model. How do you serve this efficiently?</summary>

This is the **multi-tenant LoRA serving** problem and it's a major reason LoRA exists in production.

Three approaches with very different cost profiles:

1. **Merge per request**: when customer X queries, merge their LoRA into base weights, run inference, unmerge. Simple but slow — merging adds latency to every request.
2. **Switch active LoRA**: keep the base model in GPU memory, swap which LoRA is "active" per request. Frameworks like vLLM with LoRA support (S-LoRA, dLoRA) do this with minimal overhead. Adapters stay in CPU memory and are paged in as needed.
3. **Pre-merge into N copies**: deploy 50 separate merged models. Highest throughput per model but 50× memory (50 × 14GB = 700GB).

For 50 customers, option 2 is standard: 1 × 14GB base model + 50 × ~50MB LoRA adapters = 17GB total VRAM, with seamless customer routing. This is precisely what platforms like Predibase, Together AI, and Replicate are built around.

Without LoRA: option 3 is the only choice, and it doesn't scale beyond a handful of customers per GPU. LoRA *enabled* the multi-tenant LLM serving market.
</details>

<details>
<summary>Why is LoRA's learning rate (1e-4 to 5e-4) much higher than full fine-tuning's (2e-5 to 5e-5)?</summary>

Two reasons stacked together:

1. **Far fewer parameters mean larger steps are safer**: with 0.5% of the parameters trainable, each parameter sees more "responsibility" for the loss change. Larger steps don't risk catastrophic forgetting because the base model is frozen — only the small adapter can drift.
2. **B is initialized to zero, so initial gradient signal is small**: in the first few steps, the effective update magnitude is naturally throttled by the zero initialization. Higher LRs are needed to actually move the adapter from zero into useful territory.

The math: full fine-tuning's LR is chosen to *protect* pretrained weights from disruptive updates. LoRA doesn't need that protection — pretrained weights are frozen. So LoRA's LR can be ~10-25× higher.

Practical implication: copying a full-fine-tuning recipe's LR (2e-5) into a LoRA training run will result in very slow learning. LoRA needs its own LR sweep, typically starting at 3e-4 and trying 1e-4 to 1e-3.
</details>

<details>
<summary>Scenario: a teammate asks "why not just train at full precision and skip LoRA?" Walk them through the memory math for a 13B model.</summary>

Llama-2 13B in float16:

- **Model weights**: 13B × 2 bytes = 26GB
- **Gradients** (one per parameter): 26GB
- **Adam optimizer state** (momentum + variance, 4 bytes each = 8 bytes per param): 13B × 8 = 104GB
- **Activations** (depends on batch size and seq len, often 20-40GB): 30GB
- **Total**: ~186GB

That's 3-4 × A100 80GB or 1 × H100 80GB shared via DeepSpeed/FSDP — needs multi-GPU sharding even for fine-tuning. For a small team or research project, this is prohibitive.

With LoRA (rank 16, all attention modules):

- **Model weights**: 26GB (still need to forward through)
- **Trainable params** (~0.5% of 13B): 65M
- **Gradients on trainable**: 65M × 2 bytes = 130MB
- **Optimizer state on trainable**: 65M × 8 bytes = 520MB
- **Activations**: same 30GB
- **Total**: ~57GB

Fits on a single A100 80GB. With QLoRA (4-bit base), the model weights drop to ~7GB, bringing total to ~38GB — fits on a 48GB GPU.

The fundamental trade: full fine-tuning needs memory proportional to *total parameters × optimizer overhead*. LoRA needs memory proportional to *trainable parameters × optimizer overhead*, plus a fixed model forward cost. For LLMs, the savings are enormous.
</details>

<details>
<summary>What happens if you apply LoRA to the embedding layer or the LM head?</summary>

LoRA is *not* typically applied to embeddings or the LM head, and there are specific reasons:

1. **Embedding layer**: input embeddings are conceptually a lookup table — there's no "weight update direction" to factorize. You'd be applying low-rank approximation to a sparse-access table, losing the structural advantage. If you want to adapt embeddings, *unfreeze* them and train directly (which adds ~25M params for typical vocabs).
2. **LM head**: with weight tying (head shares weights with input embedding), there's nothing separate to LoRA. Without weight tying, you could LoRA it, but the LM head sees task-specific signal directly and benefits more from being trained fully (or frozen entirely).

Modern PEFT libraries handle this automatically. If you specify `target_modules=["q_proj", "v_proj"]` in HuggingFace PEFT, embeddings and LM head are left alone.

When does it matter? For domain adaptation where vocabulary use is dramatically different (medical, legal): consider unfreezing the embedding layer entirely. For instruction tuning on a model with already-rich embeddings: LoRA on attention is sufficient.
</details>

<details>
<summary>Scenario: you train LoRA adapters that work great in isolation, but when you compose two adapters (e.g., "medical knowledge" + "polite tone"), output quality drops. Why?</summary>

LoRA adapter composition is harder than it looks. Two common failure modes:

1. **Adapter interference**: each adapter $B_i A_i$ shifts the model in a different direction. When you sum them ($\sum_i B_i A_i$), the directions can cancel or amplify in unexpected ways. The combined effect isn't the "OR" of the individual capabilities — it's the *vector sum* of their weight updates.
2. **Scale mismatch**: if each LoRA was trained with $\alpha = 16$, summing two gives an effective adaptation magnitude of $2 \times$ normal. Output distribution shifts in compounding ways.

Solutions (active research area):

- **Scale down**: when combining $N$ LoRAs, divide each by $\sqrt{N}$ or train them with reduced alpha.
- **Orthogonal LoRAs (OLora)**: explicitly train multiple LoRAs to occupy orthogonal subspaces.
- **LoRA Hub / TIES merging**: more sophisticated merging that preserves the highest-magnitude components per adapter.
- **MoE-LoRA**: train a gating network that decides which adapter to activate per input — avoids forced combination.
- **Train a multi-task LoRA directly**: instead of merging single-task adapters, train one LoRA on multi-task data.

For production with composable adapters: use S-LoRA or similar serving frameworks, route requests to the right single adapter rather than merging. Merging is brittle and rarely works without quality loss.
</details>

<details>
<summary>Why is QLoRA's "4-bit NF4 quantization" better than naive INT8 or INT4?</summary>

Naive quantization (INT8, INT4) uses uniform quantization buckets. But neural network weights are *not* uniformly distributed — they're approximately Gaussian-distributed around zero with long tails.

NF4 (NormalFloat-4) uses non-uniform quantization buckets *chosen so that each bucket contains an equal fraction of a standard normal distribution*. Buckets are dense near zero (where most weights are) and sparse in the tails (where few weights are). This is information-theoretically near-optimal for Gaussian-distributed weights.

Quality difference: NF4 quantized models lose ~0.3-1% on benchmarks compared to bfloat16. Plain INT4 loses 2-5%. NF4 is specifically designed for *quantization-aware* fine-tuning — you can train LoRA on top with minimal accuracy hit.

QLoRA also uses **double quantization** (quantize the quantization constants themselves) and **paged optimizers** (CPU↔GPU paging for AdamW state during memory spikes). Together these enable 65B-model fine-tuning on a single 48GB GPU.

The deeper lesson: quantization isn't just about reducing bits per weight — it's about preserving the *information* in those weights. Distribution-aware quantization (NF4, GPTQ, AWQ) all leverage the same insight.
</details>

<details>
<summary>If LoRA approximates full fine-tuning so well at 0.5% of the parameters, why does anyone still do full fine-tuning?</summary>

LoRA hits 95-98% of FFT quality on most tasks, but the missing 2-5% matters in specific scenarios:

1. **Maximum-quality SOTA benchmarks**: when chasing benchmark records (academic competitions, internal leaderboards), every fraction of a point counts.
2. **Highly specialized domains** where the adaptation isn't low-rank: e.g., learning a new programming language, dramatic stylistic change. LoRA's low-rank assumption may not hold.
3. **Multilingual fine-tuning** where many languages have to share adapter capacity: full fine-tuning distributes capacity better.
4. **When you need to also change embeddings**: LoRA doesn't naturally adapt the embedding layer; FFT does.
5. **Research / probing**: when studying what fine-tuning learns, FFT provides a cleaner signal than LoRA's constrained update.

In practice, ~95% of production fine-tuning is LoRA. The other 5% is FFT for very specific quality-critical use cases. For most teams, the question is "LoRA rank 16 vs LoRA rank 32" not "LoRA vs FFT."

There's also a research argument: as model scale grows (70B, 175B, 405B+), full fine-tuning becomes prohibitively expensive. LoRA may *become* the universal fine-tuning method by necessity at sufficient scale.
</details>

<details>
<summary>Scenario: you have LoRA adapters for instruction following, summarization, and code generation. A user asks a code-related instruction. Which adapter wins, and how should the system decide?</summary>

This is the **adapter routing** problem. Several strategies:

1. **Hard-routed**: a separate classifier predicts task type from the prompt, then selects an adapter. Reliable but rigid.
2. **MoE-LoRA / mixture-of-LoRAs**: a learned gating function selects which adapter (or weighted blend) to use. Trained end-to-end. State-of-the-art for many tasks but more complex.
3. **Hierarchical**: use the most general adapter (instruction following) by default, fall through to specialists only when input is clearly typed.
4. **Compositional with conflict resolution**: try to apply multiple adapters, use confidence scores from the model to decide which output to keep.

For your case: code generation usually wins for code-related instructions because the specialist has stronger task-specific signal. But if the prompt has unusual structure (e.g., "Write me a poem about for-loops"), the instruction adapter handles it better.

Production reality check: most systems do hard routing or use a single "general" adapter (instruction-tuned) and call it done. Adapter composition / MoE-LoRA is mostly research right now, with growing production adoption in 2024-2025.
</details>

<details>
<summary>Why doesn't LoRA work as well for vision transformers as it does for language models?</summary>

LoRA works in vision too, but the empirical performance gap to FFT is typically larger than in language. Reasons:

1. **Vision pretraining gives less generalized features**: pretrained ViTs tend to overfit to ImageNet-style statistics. Adaptation to new visual domains (medical imaging, satellite, etc.) may require larger updates than LoRA's low-rank constraint allows.
2. **Less benefit from rich pretraining**: vision tasks have more variety in low-level statistics (color spaces, edge distributions) than language tasks have in basic features. Adapting low-level statistics is hard with rank-8 updates.
3. **Smaller models** mean less redundancy: LoRA exploits the over-parameterization of large LLMs. A 100M-param ViT has less "spare capacity" to factor out.

That said, LoRA *does* work for ViT — just often requires higher rank (16-32) and more targeted module choices (attention + FFN, not just attention). It's also become standard for foundation vision models (CLIP, DINO, Segment Anything).

For multimodal models (CLIP, BLIP, LLaVA): LoRA on the language part + full fine-tuning on the vision adapter is a common recipe. Different modalities have different fine-tuning regimes.
</details>

<details>
<summary>Scenario: you have a base model + LoRA adapter saved separately. The base model gets updated 6 months later (continued pretraining checkpoint). Does your LoRA adapter still work on the new base?</summary>

Usually yes, sometimes no — depends on what changed.

**LoRA adapters are tied to specific base weights.** The adapter learned $BA$ such that $W_0 + BA$ does what you want. If $W_0$ changes (call it $W_1$), then $W_1 + BA$ may not behave the same way — the adapter was trained relative to the old basis.

Cases:

1. **Base model continued pretraining on similar data**: weights drift slightly. Adapter typically still works at 80-95% of original quality — degraded but usable. Re-finetuning the adapter for 1-2 epochs usually restores full quality.
2. **Base model gets a major version bump** (architecture change, different tokenizer): adapter is incompatible. Cannot port; must retrain on new base.
3. **Base model fine-tuned on different domain**: adapter quality degrades more (~50-80%) because the base now has different specialization that conflicts with the adapter.
4. **Base model quantization changes** (FP16 → INT8 → INT4): adapter typically still works since LoRA is robust to base quantization (QLoRA exists for this reason).

Production patterns to handle base updates:

- **Version-pin everything**: lock base model version, lock LoRA training data. Re-train both together on schedule.
- **Continuous adapter retraining**: when base updates, automatically re-train LoRA on cached task data. Takes hours, not days.
- **Adapter portability testing**: maintain a held-out eval set; before promoting a new base+adapter pair to production, verify quality didn't drop.

This is one reason production LLM teams pin to specific model versions: changing the base means revalidating every downstream adapter. Cloud providers (OpenAI, Anthropic) versioning their models per-month is partly about giving customers stable bases for adapter training.
</details>

## Points to remember

- LoRA freezes base model weights and adds rank-$r$ parallel updates $BA$. Only $BA$ is trained; $B$ starts at zero so the model starts as identical to the base.
- Trainable parameters: 0.1-1% of total. Memory savings: ~75-95% vs full fine-tuning.
- Default config: rank 8-16, $\alpha = 16-32$, target attention projections (Q, K, V, O). Add FFN modules for harder tasks.
- LoRA needs a *higher* learning rate (1e-4 to 5e-4) than full fine-tuning, because base weights are protected by being frozen.
- $\Delta W = BA$ can be merged into $W$ at deployment — zero inference overhead.
- QLoRA = LoRA + 4-bit NF4 quantization. Enables 7B-65B fine-tuning on a single consumer/prosumer GPU.
- Multi-tenant serving (50+ LoRAs on one base model) is the production use case LoRA was made for — frameworks like S-LoRA, vLLM, Predibase, Together AI built around it.
- LoRA composition is harder than it looks: summing two adapters usually doesn't give "OR of capabilities." Use single adapters per request or proper MoE-LoRA.
- Don't apply LoRA to embeddings or LM head — they don't benefit from low-rank decomposition. Unfreeze them entirely if needed.
- Variants worth knowing: DoRA (magnitude + direction decomposition, beats LoRA), AdaLoRA (adaptive rank per layer), rsLoRA (rank-stabilized scaling).
- For research-grade max quality, FFT still wins by 1-3%. For production cost-effectiveness, LoRA wins almost always.

## Further reading

- [arXiv: LoRA (Hu et al. 2022)](https://arxiv.org/abs/2106.09685) — the original paper, still the clearest exposition of the low-rank decomposition
- [arXiv: QLoRA (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314) — 4-bit NF4 quantization and the engineering that makes consumer-GPU fine-tuning practical
- [arXiv: DoRA (Liu et al. 2024)](https://arxiv.org/abs/2402.09353) — weight-decomposed LoRA that often matches full fine-tuning quality at lower rank
- [Hugging Face — PEFT Conceptual Guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — official walkthrough of LoRA mechanics with code
- [Sebastian Raschka — Practical LoRA Insights](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — empirical study of rank, target modules, and α
- [arXiv: S-LoRA (Sheng et al. 2023)](https://arxiv.org/abs/2311.03285) — serving thousands of LoRA adapters from one base model in production
- [Together AI — Fine-tuning LLaMA with LoRA](https://www.together.ai/blog/finetuning) — production-oriented guide to LoRA configs that work
- [Unsloth blog](https://unsloth.ai/blog) — engineering deep-dives on faster LoRA / QLoRA implementations

## Common mistakes

- Using a learning rate too low for LoRA (2e-5) — LoRA benefits from higher LRs (3e-4) since it has far fewer parameters and can afford more aggressive updates
- Forgetting to merge the LoRA weights before deployment — running with separate B and A matrices adds overhead
- Applying LoRA only to Q and V but not K and O — including all attention projections usually improves results
- Not printing `model.print_trainable_parameters()` — easy way to verify the PEFT configuration is correct

## Final takeaway

LoRA is the industry standard for fine-tuning LLMs. It freezes all pre-trained weights and adds tiny low-rank matrices $B$ and $A$ to each target layer. The product $BA$ approximates the weight update with 10–1000× fewer parameters. After training, $BA$ can be merged into the original weights for zero inference overhead. QLoRA combines LoRA with 4-bit quantization, enabling fine-tuning 7B+ models on consumer GPUs. The combination of pre-training + LoRA fine-tuning is the standard workflow for adapting modern LLMs to custom applications.

## References

- Hu, E., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP (Adapters). ICML.
- Li, X., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.
