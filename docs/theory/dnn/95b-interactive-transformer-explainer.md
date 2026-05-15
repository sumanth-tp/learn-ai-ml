---
id: interactive-transformer-explainer
title: "Interactive transformer explainer (embedded)"
sidebar_label: "95b · Interactive Explorer"
sidebar_position: 95.5
slug: /theory/dnn/interactive-transformer-explainer
description: "An embedded, fully interactive transformer explainer running locally — visualize every step of a real GPT-style forward pass: token embeddings, multi-head attention, MLP blocks, and the LM head, with controls to inspect activations at each layer."
tags: [transformer, interactive, visualization, gpt, attention, deep-learning]
hide_table_of_contents: true
---

# Interactive transformer explainer

> **TL;DR.** Everything in this curriculum — tokens → embeddings → attention → FFN → logits → next-token — is something you can now *click through* in the explorer below. It's a self-contained offline version of the popular Transformer Explainer (Polo Club @ Georgia Tech), running entirely in your browser. Hover over any cell to inspect its computation. Change the input prompt and watch the entire forward pass re-execute.

You've worked through 95 notes' worth of theory. This is where it all becomes tactile.

## What's in the explorer

The embedded tool visualizes a real, working **GPT-2 small** (124M parameters) performing one forward pass on an input prompt. Every box you see corresponds to an actual tensor in memory, with the exact values being computed live. The visualization is interactive: you can hover, click, edit the prompt, and inspect intermediate activations at every layer.

What you can examine, mapped to the notes you've read:

| Component in the explorer | Where this was explained |
|---------------------------|--------------------------|
| Token IDs after tokenization | [86 — Tokenization (BPE)](./86-tokenization-bpe-wordpiece-sentencepiece.md) |
| Token + positional embeddings | [78 — Positional Encoding](./78-positional-encoding-in-transformers.md) |
| Q, K, V projections per head | [73 — Self-Attention with Code](./73-self-attention-in-transformers-with-code.md) |
| Multi-head attention pattern | [77 — Multi-Head Attention](./77-multi-head-attention-in-transformers.md) |
| Causal (lower-triangular) mask | [81 — Masked Self-Attention](./81-masked-self-attention-in-the-transformer-decoder.md) |
| Scaled dot-product scores | [74 — Scaled Dot-Product Attention](./74-scaled-dot-product-attention.md) |
| Softmax → attention weights | [74](./74-scaled-dot-product-attention.md), [75 — Geometric Intuition](./75-geometric-intuition-for-self-attention.md) |
| Feed-forward MLP sublayer | [80 — Encoder Architecture](./80-transformer-encoder-architecture.md) |
| LayerNorm / residual additions | [79 — LayerNorm vs BatchNorm](./79-layer-normalization-versus-batch-normalization.md) |
| LM head → vocab logits | [88 — GPT (Decoder-Only)](./88-gpt-decoder-only-causal-lm.md) |
| Temperature, top-k, top-p sampling | [84 — Inference Step-by-Step](./84-transformer-inference-step-by-step.md) |

If something in the explorer surprises you, the linked note has the explanation.

## How to use it

1. **Click in the input box** at the top of the explorer to change the prompt.
2. **Watch the entire forward pass re-execute** — every tensor downstream updates.
3. **Hover over any cell** to see its numerical value and computation.
4. **Click on attention heads** to see which tokens attend to which.
5. **Adjust temperature / top-k / top-p** at the bottom to see how the sampling distribution changes.

The explorer below is interactive. Click anywhere inside it.

<iframe
  src="/html/transformer-explainer.html"
  title="Transformer Explainer"
  width="100%"
  height="900"
  style={{border: '1px solid #ddd', borderRadius: '8px', marginTop: '12px'}}
  loading="lazy"
  allowFullScreen
></iframe>

If the iframe doesn't load (slow connection, blocked by browser policy, etc.), <a href="/html/transformer-explainer.html" target="_blank" rel="noopener noreferrer"><strong>open it in a full tab here →</strong></a>

## A guided walkthrough

If you're seeing this for the first time, follow these steps to anchor the abstract concepts in concrete numbers.

### Step 1: Look at the embeddings

The first column on the left is the **token embedding** lookup. Each input token becomes a 768-dimensional vector (GPT-2 small uses `d_model = 768`). The values come from a learned `(50257 × 768)` embedding table.

Try changing the input from `"The cat sat"` to `"The dog sat"`. Only the embedding for the changed token differs; everything downstream re-computes from there.

### Step 2: Inspect a single attention head

Scroll to the **multi-head attention** panel of the first transformer block. GPT-2 small has 12 heads per layer. Each head independently computes:

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i
$$

Click on different heads — you'll see different attention patterns. Some heads focus on adjacent tokens, others on subject-verb pairs, others on the most recent tokens. This is **emergent specialization**: nothing in the training objective told the model to do this; it just happens to be the most useful way to use 12 heads.

### Step 3: See the causal mask

In any attention head, notice that the score matrix is **lower-triangular** — the upper-right triangle is greyed out / set to negative infinity. This is the causal mask from [note 81](./81-masked-self-attention-in-the-transformer-decoder.md). Token 3 cannot attend to tokens 4, 5, 6 because at training time, those would be the tokens we're trying to predict.

### Step 4: Watch the residual stream

The residual stream is the "highway" running down the right side of each block. Notice how the attention output and the FFN output are *added* (not replaced) to it. This is the residual connection. Without it, deep stacks of transformer blocks wouldn't train — gradients would vanish.

### Step 5: The final logits

The top-right of the explorer shows the **logits** — one number per vocabulary token (50,257 of them in GPT-2). After softmax, these become the probability distribution over the next token. The bar chart shows the top-k most-likely tokens.

Drop temperature to 0.1 (very low) — the distribution collapses to a single dominant token. Raise temperature to 1.5 — the distribution flattens out, more diverse tokens become plausible. This is exactly the temperature math from [note 84](./84-transformer-inference-step-by-step.md).

## What this is and isn't

**What this is**: a real, exact GPT-2 small forward pass, with every tensor value computed live from the actual pretrained weights. The numbers you see are not pedagogical mock-ups; they are what the real model produces.

**What this isn't**:
- Not a *trained-from-scratch* model you control. The weights are fixed (GPT-2 small, downloaded once and embedded).
- Not a *generation* loop — only a single forward pass is visualized. For autoregressive generation, see [bbycroft.net/llm](https://bbycroft.net/llm) or run nanoGPT locally.
- Not an *encoder-only* or *encoder-decoder* visualization. The explorer is GPT-2-specific (decoder-only with causal mask). For BERT-style visualizations, use [BertViz](https://github.com/jessevig/bertviz). For T5-style, the [Tensor2Tensor Colab](https://colab.research.google.com/github/tensorflow/tensor2tensor/) has notebooks.

## Cross-references

- **Prerequisite reading order**: [71 — Intro to Transformers](./71-introduction-to-transformers.md) → [72–77 (Self-Attention)](./72-what-self-attention-is.md) → [78 — Positional Encoding](./78-positional-encoding-in-transformers.md) → [80 — Encoder](./80-transformer-encoder-architecture.md) and [83 — Decoder](./83-transformer-decoder-architecture.md) → [88 — GPT](./88-gpt-decoder-only-causal-lm.md)
- **For the encoder-decoder comparison**: [89b — Architecture Families Compared](./89b-encoder-only-vs-decoder-only-vs-encoder-decoder.md)
- **For the inference loop**: [84 — Transformer Inference Step-by-Step](./84-transformer-inference-step-by-step.md)
- **For the math behind every cell**: [74 — Scaled Dot-Product Attention](./74-scaled-dot-product-attention.md)

## Other interactive tools worth bookmarking

The embedded explorer is one of several great visualizations. Each emphasizes a different aspect of the architecture:

| Tool | Best for |
|------|----------|
| **[Transformer Explainer (poloclub)](https://poloclub.github.io/transformer-explainer/)** | The hosted version of what's embedded above; works offline too via this page |
| **[LLM Visualization (bbycroft.net)](https://bbycroft.net/llm)** | Animated 3D walkthrough with continuous generation loop |
| **[The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)** | Static but unmatched conceptual diagrams; referenced throughout this curriculum |
| **[BertViz](https://github.com/jessevig/bertviz)** | Attention pattern visualization in trained BERT / GPT models |
| **[The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)** | The original Vaswani et al. paper transformed into a runnable Jupyter notebook |
| **[nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)** | Train your own tiny GPT in 30 minutes; inspect every component yourself |

## Final takeaway

The transformer is not a magic box — it's a stack of well-understood operations producing tensors you can hover over with your mouse. This curriculum walked you through the math; the explorer above lets you see the math run. If you've made it this far, you understand the architecture that powers every modern LLM, end to end.

There's no "next note". This is where the journey ends. Go build something.

## Credits and license

The embedded interactive explorer is an offline-packaged version of the **Transformer Explainer**, created by Polo Club of Data Science at Georgia Tech:

- Authors: Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Zijie J. Wang, Seongmin Lee, Benjamin Hoover, Duen Horng (Polo) Chau
- Paper: [Transformer Explainer: Interactive Learning of Text-Generative Models](https://arxiv.org/abs/2408.04619) (IEEE VIS 2024)
- Original source: [github.com/poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer) (MIT License)
- Hosted version: [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer/)

All credit for the visualization itself goes to the original authors. This page embeds their tool with attribution under the project's permissive license.
