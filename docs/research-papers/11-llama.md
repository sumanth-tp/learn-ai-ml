---
id: paper-llama
title: "LLaMA: Open and Efficient Foundation Language Models"
sidebar_label: "11 · LLaMA"
sidebar_position: 11
slug: /research-papers/llama
description:
  "LLaMA, section by section: the inference-budget argument, the public data
  mixture, RMSNorm, SwiGLU and RoPE, Tables 1–16, and a trainable LLaMA-style
  decoder."
tags: [research-papers, deep-learning]
---

import PaperPdf from '@site/src/components/PaperPdf';
import CodeWalkthrough from '@site/src/components/viz/CodeWalkthrough';
import ResearchPaperLab from '@site/src/components/viz/ResearchPaperLab';

> **Touvron et al. · 2023** · [Read the embedded paper](#original-paper) ·
> [Download PDF](/papers/research-papers/llama.pdf) · Notes follow the paper
> section by section, §1 to the appendix.

## Paper in one minute

**Problem.** Training-compute-optimal models are not necessarily the best models
to deploy when inference cost is paid for every generated token.

**Key idea.** Train comparatively smaller decoder-only Transformers on many more
tokens, using a curated public-data mixture and efficient components including
RMSNorm, SwiGLU and rotary position embeddings.

**Why it matters.** LLaMA reframed model selection around quality at inference
cost, not parameter count alone. The original paper primarily describes base
models; a pretrained completion model is not automatically a safe chat assistant.

### Model-building flow

```mermaid
flowchart LR
    DATA["Filtered public-data mixture"] --> TOK["SentencePiece tokens"]
    TOK --> EMB["Token embeddings"]
    EMB --> RMS
    subgraph LAYERS["Repeated LLaMA block"]
      RMS["RMSNorm"] --> ATT["Causal attention + RoPE"] --> SWI["SwiGLU FFN"]
    end
    SWI --> LM["Pretrained base model"] --> CACHE["Autoregressive inference + KV cache"]
```

## How to read this chapter

The walkthrough below follows the paper **in its own order**, from the abstract
to the appendix. Each heading carries the paper's section number, so you can
keep the PDF open beside it. Equations use the paper's notation. Boxes marked
**not from the paper** are teaching aids, such as analogies, derivations or
worked numbers, added to make a step easier to follow.

## Abstract: the four claims

A **foundation model** is a large model trained once on general text, which
other people then adapt to their own tasks. A **parameter** is one learned
number inside the model, and a **token** is a small piece of text, often part of
a word. With those words in hand, the abstract makes four claims:

1. The authors train a family of foundation models from **7B to 65B
   parameters** on **trillions of tokens**.
2. State-of-the-art models can be trained using **publicly available data
   only**, without proprietary datasets.
3. **LLaMA-13B beats GPT-3 (175B) on most benchmarks**, despite being more than
   ten times smaller. A **benchmark** is a fixed test set used to compare
   models.
4. **LLaMA-65B is competitive** with the best models of the time, Chinchilla-70B
   and PaLM-540B.

The models are released to the research community. Keep these claims in mind:
§2 builds the recipe, §3 and §4 supply the evidence, and §5 and §6 look at the
costs and risks.

## §1 Introduction: the inference budget changes the best model

Earlier work found that few-shot abilities appear once a model is large enough,
so many groups simply made models bigger. Hoffmann et al. (2022), the
**Chinchilla** paper, then showed that for a fixed **training compute budget**
(the total arithmetic you can afford for training), the best result comes from
a smaller model trained on more data, not from the largest model.

LLaMA's point is that this rule **ignores the inference budget**. Inference is
the work of running the finished model to answer a request, and it is paid
again for every user. If you have a target quality in mind, the model you want
is the one that is **cheapest to run**, not the one that is cheapest to train.
A smaller model trained for longer can cost more to train but less for ever
after.

The paper's example: Hoffmann et al. recommend training a 10B model on 200B
tokens, yet the authors find that **a 7B model keeps improving after 1T
tokens**. So the goal of the work is the best possible quality at several
inference budgets, reached by training on more tokens than usual.

:::tip Intuition: training cost is not the only cost (not from the paper)

Suppose two models reach similar quality. One is larger but trained for fewer
tokens; the other is smaller but trained longer. The larger one might use less
training compute under a particular budget, yet cost more every time someone
generates an answer.

For a widely used model, inference repeats many times. Spending more training
on a smaller model can be worthwhile when the resulting model is cheaper to
serve. This is not "smaller always wins". It depends on quality requirements,
training budget, inference volume, hardware and sequence lengths.

:::

:::tip Worked number (not from the paper)

Chinchilla's rule of thumb is roughly 20 training tokens per parameter:
$10\text{B}\times20=200\text{B}$ tokens, the figure the paper quotes. LLaMA-7B
sees 1T tokens, about **143 tokens per parameter**, seven times the rule.

A decoder needs roughly $2N$ floating-point operations per generated token for
$N$ parameters. So LLaMA-13B needs about $175/13\approx13$ times less arithmetic
per token than GPT-3. Memory tells the same story: at 2 bytes per parameter,
13B parameters take about 26 GB and fit on one 32 GB V100, while 65B take about
130 GB and do not fit on one 80 GB A100.

:::

The second theme is **openness**. Chinchilla, PaLM and GPT-3 were trained partly
on data that is not public or not documented, such as "Books – 2TB" or "Social
media conversations". OPT, GPT-NeoX, BLOOM and GLM used open data, but none was
competitive with PaLM-62B or Chinchilla. LLaMA aims to be both open and
competitive. The authors add that LLaMA-13B "can be run on a single GPU", which
helps more researchers study large models.

:::tip In the real world (not from the paper)

Within weeks of the release, the open-source project **llama.cpp** showed LLaMA
models running on an ordinary laptop, using 4-bit weights to shrink memory
further. That is the inference-budget argument in practice: a model small
enough to run where the user is, rather than only in a data centre.

:::

## §2 Approach

The recipe is deliberately ordinary. It follows GPT-3 and PaLM, takes its data
budget from the Chinchilla scaling laws, and trains large Transformers on a lot
of text with a standard optimiser. The novelty is in the choices: which data,
how much of it, and a handful of architecture changes borrowed from other
models.

### §2.1 Pre-training data

**Pre-training** is the first, long training phase in which the model learns to
predict the next token of ordinary text. LLaMA's pre-training data is a mixture
of seven public sources. The mixture has **sampling proportions**: each source
is drawn from at a chosen rate, so not every source contributes equally. Table 1
of the paper:

| Dataset       | Sampling proportion | Epochs (1.4T run) | Disk size |
| ------------- | ------------------- | ----------------- | --------- |
| CommonCrawl   | 67.0%               | 1.10              | 3.3 TB    |
| C4            | 15.0%               | 1.06              | 783 GB    |
| GitHub        | 4.5%                | 0.64              | 328 GB    |
| Wikipedia     | 4.5%                | 2.45              | 83 GB     |
| Books         | 4.5%                | 2.23              | 85 GB     |
| ArXiv         | 2.5%                | 1.06              | 92 GB     |
| StackExchange | 2.0%                | 1.03              | 78 GB     |

**What this shows:** two-thirds of the text is filtered web pages, and the small
high-quality sources (Wikipedia and books) are read more than twice. An
**epoch** is one full pass over a source. The 1T-token runs use the same
proportions.

Each source is cleaned in its own way. **Deduplication** means removing repeated
copies of the same text, so the model does not waste training on them:

- **English CommonCrawl (67%).** Five web dumps from 2017 to 2020, processed with
  the CCNet pipeline: line-level deduplication, a fastText classifier to drop
  non-English pages, and an n-gram language model to drop low-quality text. A
  further linear classifier keeps only pages that look like those **cited as
  references in Wikipedia**.
- **C4 (15%).** Another cleaned web crawl. Adding a differently processed crawl
  improved performance in early experiments. Its quality filter relies mostly on
  heuristics such as punctuation and the number of words and sentences.
- **GitHub (4.5%).** Public code from Google BigQuery, kept only under Apache,
  BSD and MIT licences. Low-quality files are filtered by line length and the
  share of alphanumeric characters, boilerplate such as headers is removed with
  regular expressions, and exact duplicate files are dropped.
- **Wikipedia (4.5%).** June–August 2022 dumps in 20 languages that use the
  Latin or Cyrillic scripts, with hyperlinks, comments and formatting
  boilerplate removed.
- **Gutenberg and Books3 (4.5%).** Public-domain books plus the Books3 part of
  The Pile. Books with more than 90% content overlap are deduplicated.
- **ArXiv (2.5%).** LaTeX source of scientific papers. Everything before the
  first section and the bibliography is removed, as are comments, and
  user-written macros are expanded inline for consistency.
- **Stack Exchange (2%).** The 28 largest sites, with HTML tags removed and
  answers sorted by score, highest first.

These decisions determine whether a token budget is spent on useful content or
on repeated formatting. Filtering and deduplication affect both quality and the
risk of **contamination**, where test questions leak into the training data.

A source's sampling proportion and its stored size are different things. A small
collection can be seen more than once while a larger one is sampled less
completely. Reported token counts are therefore more informative when read with
the mixture table than as one total number.

:::note The text rounds the epochs

The paper says that "each token is used only once during training, with the
exception of the Wikipedia and Books domains", which get about two epochs.
Table 1 is more precise: CommonCrawl, C4, ArXiv and StackExchange are also seen
slightly more than once (1.03–1.10 epochs), and GitHub only 0.64 times.

:::

:::tip Worked number (not from the paper)

CommonCrawl supplies $0.67\times1.4\text{T}\approx938\text{B}$ training tokens
over 1.10 epochs, so about 853B distinct tokens come from 3.3 TB of text. That
is roughly **3.9 bytes per token**, a useful rule of thumb for English text
under this tokeniser.

:::

**Tokeniser.** A **tokeniser** turns text into integer IDs. LLaMA uses
**byte-pair encoding** (BPE), which builds a vocabulary of frequent sub-word
pieces, via the SentencePiece library. Two choices stand out:

- **All numbers are split into individual digits.** "1234" becomes four tokens.
- **Byte fallback.** Unknown UTF-8 characters are broken into bytes, so no text
  is ever impossible to represent.

After tokenisation the whole dataset is about **1.4T tokens**.

Sub-word tokens balance vocabulary size against sequence length. A tokeniser is
not interchangeable with another tokeniser just because both produce integers:
its ID-to-piece mapping must match the model's embedding rows. The character
tokeniser in the code below is intentionally a separate teaching vocabulary.

Digit splitting is an implementation choice that changes sequence structure.
Arithmetic over individually tokenised digits differs from treating a whole
multi-digit number as one frequent vocabulary item.

:::note Two details the paper does not give

The paper does not state the vocabulary size or the context length. The
released models use a 32,000-token vocabulary and a 2,048-token context. An
earlier version of these notes also said the tokeniser makes special choices
for whitespace; the paper says nothing about whitespace, only digits and byte
fallback.

:::

:::tip In the real world (not from the paper)

Because every source in Table 1 is public, other groups could rebuild the
mixture. Together's **RedPajama** project (2023) did exactly that, publishing an
open reproduction of the LLaMA training data following the same seven sources.

:::

### §2.2 Architecture

LLaMA remains an autoregressive Transformer. **Autoregressive** means it writes
one token at a time, each time reading everything it has written so far. The
paper lists three changes to the original design and, in brackets, which model
inspired each one. It gives no new experiments for them; each is taken from
earlier work.

#### Pre-normalisation [GPT-3], with RMSNorm

**Normalisation** rescales a vector of numbers so its size stays in a sensible
range; this keeps very deep networks stable. LLaMA normalises the **input** of
each sub-layer instead of its output, "to improve the training stability", and
uses the **RMSNorm** function of Zhang and Sennrich (2019).

**Pre-normalisation** means a sub-layer sees the normalised residual stream and
adds its output back to the unnormalised stream: `x + attention(norm(x))`.
Compare that with the original Transformer's `norm(x + attention(x))`. The
unnormalised path lets information and gradients pass straight through many
layers.

RMSNorm divides a vector by its **root mean square**, the square root of the
average squared entry, and then multiplies by a learned scale. The formula
comes from the RMSNorm paper; LLaMA itself does not write it out. For a hidden
vector $x$ of width $d$:

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac1d\sum_{i=1}^{d}x_i^2+\epsilon}}.
$$

In words: shrink or stretch the vector so its typical entry has size about 1,
then let the model re-scale each coordinate with $g$. Unlike LayerNorm, this
operation does not subtract the vector's mean. The small $\epsilon$ prevents
division by zero or a numerically tiny denominator.

:::tip Intuition: an automatic volume knob (not from the paper)

Think of RMSNorm as a volume limiter on a microphone. Whether someone whispers
or shouts, the signal leaves at a steady level, and a learned equaliser ($g$)
then boosts or cuts each channel. LayerNorm would also remove any constant hum
(the mean); RMSNorm skips that step, which saves a little computation.

:::

#### SwiGLU activation [PaLM]

An **activation function** is the non-linear step inside the feed-forward part
of each layer. LLaMA replaces the original ReLU with **SwiGLU** (Shazeer, 2020)
"to improve the performance", and uses a hidden width of $\frac23 4d$ instead of
the $4d$ used in PaLM.

SwiGLU uses two input projections. One passes through SiLU, the smooth
activation $u\sigma(u)$, and acts as a **gate**; the other carries the content.
Multiplying them element by element lets the gate decide how much of each
content feature passes, before an output projection returns to the model width.
Again the formula is from Shazeer's paper, not written in this one:

$$
\operatorname{FFN}(x)=W_2\big(\operatorname{SiLU}(W_1x)\odot W_3x\big).
$$

In words: compute a content vector and a gate vector from the same input, let
the gate dim or pass each content feature, then project back.

:::tip Why two-thirds of 4d? (worked number, not from the paper)

A conventional two-matrix FFN with hidden width $4d$ uses about
$2\times d\times4d=8d^2$ weights. A three-matrix gated FFN with hidden width
$h$ uses $3dh$. Setting $3dh=8d^2$ gives $h=\frac83 d=\frac23\cdot4d$. So the
narrower hidden layer keeps the parameter count of the old FFN rather than
being an arbitrary constant. For LLaMA-7B, $\frac83\times4096\approx10{,}923$;
the released code rounds this up to 11,008, a multiple of 256.

:::

#### Rotary embeddings [GPT-Neo]

The original Transformer added a position vector to each token embedding once,
at the bottom. LLaMA **removes these absolute positional embeddings** and instead
applies **rotary positional embeddings** (RoPE, Su et al., 2021) **at each layer**.

RoPE rotates pairs of query and key coordinates by angles that depend on the
token's position. For a two-coordinate pair:

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}.
$$

In words: turn the pair by an angle $\theta$ that grows with position. Different
coordinate pairs use different frequencies. When a rotated query and a rotated
key form a dot product, only the **difference** of their angles survives, so
relative position influences their compatibility. The values do not need the
same rotation. The [Transformer chapter](/docs/research-papers/transformer)
shows why a fixed offset is the same rotation everywhere.

RoPE is not an added learned position vector. It also does not guarantee useful
behaviour at sequence lengths beyond those seen in training.

:::tip Intuition: two clock hands (not from the paper)

Picture each query and key as a clock hand, turned further for later positions.
Two hands 3 positions apart always differ by the same angle, whether they sit at
positions 5 and 8 or 105 and 108. Attention compares the hands, so it naturally
"sees" distance.

:::

#### Table 2: the four model sizes

| Params | Dimension $d$ | Heads | Layers | Learning rate | Batch size | Tokens |
| ------ | ------------- | ----- | ------ | ------------- | ---------- | ------ |
| 6.7B   | 4096          | 32    | 32     | 3.0e−4        | 4M         | 1.0T   |
| 13.0B  | 5120          | 40    | 40     | 3.0e−4        | 4M         | 1.0T   |
| 32.5B  | 6656          | 52    | 60     | 1.5e−4        | 4M         | 1.4T   |
| 65.2B  | 8192          | 64    | 80     | 1.5e−4        | 4M         | 1.4T   |

**What this shows:** the models grow in width and depth together, and every
head has width $d/\text{heads}=128$ at every size. The two larger models get a
lower learning rate and 40% more tokens.

:::tip Check the parameter count (worked number, not from the paper)

Each layer has $4d^2$ attention weights and about $8d^2$ SwiGLU weights, so
roughly $12d^2$ in total. For 7B: $32\times12\times4096^2\approx6.44$B, plus
input and output embeddings of $2\times32{,}000\times4096\approx0.26$B, gives
**6.7B**, matching Table 2. For 65B the same sum gives about 64.9B; the rounded-up
FFN width explains the rest of the 65.2B.

:::

:::note The inspirations are not ablated here

The bracketed reasons, "to improve the training stability" and "to improve the
performance", are inherited from GPT-3, PaLM and GPT-Neo. The paper runs no
experiment that removes one change at a time, so it does not show how much each
one contributes to LLaMA's results.

:::

:::tip In the real world (not from the paper)

This trio of RMSNorm, SwiGLU and RoPE became the default recipe for open models.
Llama 2 and 3, Mistral 7B and Qwen all use it, and Hugging Face's
`LlamaForCausalLM` class loads many models that are not from Meta at all.

:::

### §2.3 Optimiser

An **optimiser** is the rule that updates the weights after each batch. LLaMA
uses **AdamW**, which is Adam with **weight decay** (a small pull of every weight
towards zero) applied separately from the gradient step. The settings:

| Setting           | Value                                            |
| ----------------- | ------------------------------------------------ |
| AdamW betas       | $\beta_1=0.9$, $\beta_2=0.95$                    |
| Schedule          | Cosine, ending at 10% of the maximum rate        |
| Warm-up           | 2,000 steps                                      |
| Weight decay      | 0.1                                              |
| Gradient clipping | 1.0                                              |
| Peak rate, batch  | Vary with model size (Table 2)                   |

**Gradient clipping** caps the size of each update, so one unusual batch cannot
throw the weights far off course. **Warm-up** starts with a tiny learning rate
and raises it over the first steps. A **cosine schedule** then lowers the rate
along half a cosine wave. The paper names the schedule without a formula; the
usual one, with $\eta_{\min}=0.1\,\eta_{\max}$, is:

$$
\eta_t=\eta_{\min}+\tfrac12\left(\eta_{\max}-\eta_{\min}\right)\left(1+\cos\frac{\pi t}{T}\right).
$$

In words: start at the peak, fall slowly, then faster, then level off at the
floor at the end of training.

These choices change how the weights move during training. The objective itself
stays next-token cross-entropy: raise the probability of the token that actually
comes next.

:::tip Worked number (not from the paper)

With 4M tokens per batch, 1.4T tokens is $1.4\times10^{12}/4\times10^6=350{,}000$
steps, and 1.0T is 250,000 steps. The 2,000 warm-up steps are well under 1% of
training. For 65B the rate rises to $1.5\times10^{-4}$ and ends at
$1.5\times10^{-5}$.

:::

:::note Batch size does not actually vary

The text says the authors "vary the learning rate and batch size with the size
of the model". Table 2 lists **4M tokens for every model**. Only the learning
rate changes.

:::

![Training loss over consumed tokens](/img/research-papers/llama.png)

_Figure 1 from the original paper, PDF page 3.
[Source PDF](/papers/research-papers/llama.pdf#page=3)._

Figure 1 plots **training loss** against tokens seen for all four models, each
with a 4M-token batch. Loss is the model's average surprise at the next token.
Every curve is still falling when training stops, which is the evidence behind
the §1 claim that small models keep improving past the Chinchilla budget.

Loss decreasing over more training is evidence about prediction on the training
distribution. It is not itself a downstream benchmark, and it does not show that
every added token improves every capability equally.

:::tip In the real world (not from the paper)

Warm-up followed by cosine decay is now a standard option in training
libraries; for example, Hugging Face's `Trainer` accepts
`lr_scheduler_type="cosine"` with a `warmup_steps` setting.

:::

### §2.4 Efficient implementation

Training a 65B model on 1.4T tokens is only practical with engineering that
saves memory and time. The paper describes four optimisations:

1. **Efficient causal attention.** The xformers library's implementation, inspired
   by Rabe and Staats (2021) with the backward pass of Dao et al. (2022), **does
   not store the attention weights** and **does not compute scores that the
   causal mask would hide anyway**.
2. **Selective checkpointing.** Instead of storing every intermediate value for
   the backward pass, they store only the **expensive** ones, such as the
   outputs of linear layers, and recompute the rest. They write the backward
   pass for the Transformer layers by hand rather than relying on PyTorch
   autograd.
3. **Model and sequence parallelism** (Korthikanti et al., 2022), which split the
   model and parts of the sequence work across GPUs to cut per-GPU memory.
4. **Overlapping computation with communication.** The GPUs compute activations
   while they exchange results over the network (the `all_reduce` operations).

The result: about **380 tokens per second per GPU on 2,048 A100 80GB GPUs**, so
one pass over 1.4T tokens takes about **21 days** for the 65B model.

:::tip Check the 21 days (worked number, not from the paper)

$380\times2048\approx778{,}000$ tokens per second. Then
$1.4\times10^{12}/778{,}000\approx1.8$ million seconds, or **20.8 days**. That is
about $2048\times500\approx1.02$ million GPU-hours, which matches the 1,022,362
GPU-hours Table 15 gives for LLaMA-65B.

:::

#### Four techniques that save different things

These techniques are part of why the training recipe is practical at the
reported scale. They work at different points in the computation. The last row
is not in the paper; it is an inference technique added for comparison.

| Technique                             | When it is used          | What it saves or distributes                                                        |
| ------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| Memory-efficient causal attention     | Attention computation    | Skips storing attention weights and computing masked scores                         |
| Activation checkpointing              | Training                 | Stores fewer intermediate activations and recomputes selected values during backpropagation |
| Model/sequence parallelism            | Training across devices  | Divides parts of the model or sequence-related work across accelerators             |
| KV caching (not from the paper)       | Autoregressive inference | Reuses previous tokens' attention keys and values                                   |

An activation checkpoint is not a saved model file. A KV cache is not a new set
of learned weights. The techniques should not be merged into the vague claim
that a model "uses less memory".

#### Inference memory: the KV cache (not from the paper)

A **KV cache** stores earlier keys and values during generation so they need not
be recomputed for every new token. It trades memory for computation. The
complete small implementation below recomputes its short prefix for clarity;
adding a cache requires positional offsets and correct handling of newly
appended tokens.

:::tip In the real world (not from the paper)

The same memory-saving attention is now built into PyTorch.
`torch.nn.functional.scaled_dot_product_attention` picks a FlashAttention or
memory-efficient kernel when it can, so ordinary training scripts get the §2.4
saving without extra libraries.

:::

## §3 Main results

The paper tests LLaMA on **20 benchmarks** in two settings:

- **Zero-shot:** the model sees a description of the task and one test example,
  with no solved examples.
- **Few-shot:** the model also sees between 1 and 64 solved examples first.

In both cases the model either writes a free-form answer or **ranks** given
options. The comparison models are GPT-3, Gopher, Chinchilla and PaLM (not
public), and OPT, GPT-J and GPT-Neo (public).

For **multiple-choice** tasks, the model picks the option it finds most likely
after the context. Longer options contain more tokens, and every token multiplies
in a probability below one, so raw likelihood unfairly favours short answers.
The paper therefore follows Gao et al. (2021) and **divides the likelihood by
the number of characters** in the option. For OpenBookQA and BoolQ it follows
Brown et al. (2020) instead, and scores each completion by

$$
\frac{P(\text{completion}\mid\text{context})}{P(\text{completion}\mid\text{“Answer:”})}.
$$

In words: how much more likely the answer becomes once the model has read the
question, compared with seeing only the word "Answer:". This cancels out answers
that are likely in any context.

:::tip Worked example: why normalise (not from the paper)

Say option A is "yes" with log-probability $-2.0$ over 3 characters, and option
B is "a small metal key" with $-6.0$ over 17 characters. Raw scores pick A.
Per character, A scores $-0.67$ and B scores $-0.35$, so B wins. Which rule is
right depends on the task, which is why the paper uses two rules.

:::

Comparisons use specified prompting and model settings, so a claim that one
model beats another should stay attached to those tasks and conditions. Stronger
scores from a smaller model also do not make it an instruction-tuned assistant.
The main released family consists of base language models. The paper includes a
separate instruction-tuning experiment (§4). Chat formatting, preference training
and application tools are additional design choices.

:::tip In the real world (not from the paper)

Gao et al. (2021) is EleutherAI's **lm-evaluation-harness**, the open tool that
implements this character-normalised scoring. The same harness powered Hugging
Face's Open LLM Leaderboard, so many published scores for open models are
computed exactly this way. The project at the end of this chapter uses the same
rule.

:::

### §3.1 Common sense reasoning

Eight benchmarks, all zero-shot: **BoolQ** (yes/no questions about a passage),
**PIQA** (physical common sense), **SIQA** (social situations), **HellaSwag**
(pick the sensible continuation of a scene), **WinoGrande** (which noun a
pronoun refers to), **ARC** easy and challenge (school science questions) and
**OpenBookQA** (science facts plus common knowledge). Headline rows from Table 3,
accuracy in %:

| Model           | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-c |
| --------------- | ----- | ---- | --------- | ---------- | ----- |
| GPT-3 175B      | 60.5  | 81.0 | 78.9      | 70.2       | 51.4  |
| Chinchilla 70B  | 83.7  | 81.8 | 80.8      | 74.9       | –     |
| PaLM 540B       | 88.0  | 82.3 | 83.4      | 81.1       | 53.0  |
| LLaMA 13B       | 78.1  | 80.1 | 79.2      | 73.0       | 52.7  |
| LLaMA 65B       | 85.3  | 82.8 | 84.2      | 77.0       | 56.0  |

**What this shows:** LLaMA-13B already sits close to GPT-3, and LLaMA-65B beats
PaLM-540B, a model eight times larger, on most columns.

<details>
<summary>Full Table 3 from the paper</summary>

| Model          | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
| -------------- | ----- | ---- | ---- | --------- | ---------- | ----- | ----- | ---- |
| GPT-3 175B     | 60.5  | 81.0 | –    | 78.9      | 70.2       | 68.8  | 51.4  | 57.6 |
| Gopher 280B    | 79.3  | 81.8 | 50.6 | 79.2      | 70.1       | –     | –     | –    |
| Chinchilla 70B | 83.7  | 81.8 | 51.3 | 80.8      | 74.9       | –     | –     | –    |
| PaLM 62B       | 84.8  | 80.5 | –    | 79.7      | 77.0       | 75.2  | 52.5  | 50.4 |
| PaLM-cont 62B  | 83.9  | 81.4 | –    | 80.6      | 77.0       | –     | –     | –    |
| PaLM 540B      | 88.0  | 82.3 | –    | 83.4      | 81.1       | 76.6  | 53.0  | 53.4 |
| LLaMA 7B       | 76.5  | 79.8 | 48.9 | 76.1      | 70.1       | 72.8  | 47.6  | 57.2 |
| LLaMA 13B      | 78.1  | 80.1 | 50.4 | 79.2      | 73.0       | 74.8  | 52.7  | 56.4 |
| LLaMA 33B      | 83.1  | 82.3 | 50.4 | 82.8      | 76.0       | 80.0  | 57.8  | 58.6 |
| LLaMA 65B      | 85.3  | 82.8 | 52.3 | 84.2      | 77.0       | 78.9  | 56.0  | 60.2 |

</details>

The paper's reading: LLaMA-65B beats Chinchilla-70B on all reported benchmarks
"but BoolQ", and beats PaLM-540B everywhere except BoolQ and WinoGrande.
LLaMA-13B beats GPT-3 on most benchmarks despite being ten times smaller.

:::note The BoolQ exception is not in the table

Table 3 gives LLaMA-65B **85.3** on BoolQ against Chinchilla's **83.7**, so by
the paper's own numbers LLaMA-65B beats Chinchilla on BoolQ too. Either the
sentence or a table cell is wrong. The PaLM-540B claim does match the table.
"Most benchmarks" for LLaMA-13B against GPT-3 also holds, but not all: GPT-3 is
ahead on PIQA (81.0 against 80.1) and OpenBookQA (57.6 against 56.4).

:::

:::tip In the real world (not from the paper)

HellaSwag-style questions look like this: "She pours batter into the pan and…"
followed by four endings, one sensible and three odd. Humans find them easy.
The benchmark checks whether a model has picked up the everyday cause and
effect a kitchen assistant or a story-writing tool would need.

:::

### §3.2 Closed-book question answering

**Closed-book** means the model answers trivia questions with no documents to
look at; everything must come from what it memorised in training. The score is
**exact match**: the answer counts only if it equals one of the accepted answers
after light normalisation (Appendix A has the details). Headline rows from
Tables 4 (NaturalQuestions) and 5 (TriviaQA):

| Model          | NQ 0-shot | NQ 64-shot | TriviaQA 0-shot | TriviaQA 64-shot |
| -------------- | --------- | ---------- | --------------- | ---------------- |
| Chinchilla 70B | 16.6      | 35.5       | 55.4            | 64.6             |
| PaLM 540B      | 21.2      | 39.6       | –               | –                |
| LLaMA 13B      | 20.1      | 31.9       | 56.6            | 64.0             |
| LLaMA 65B      | 23.8      | 39.9       | 68.2            | 73.0             |

**What this shows:** LLaMA-65B is the best model in both settings, and LLaMA-13B
roughly matches Chinchilla-70B while being about five times smaller.

<details>
<summary>Full Tables 4 and 5 from the paper</summary>

Table 4, NaturalQuestions, exact match:

| Model          | 0-shot | 1-shot | 5-shot | 64-shot |
| -------------- | ------ | ------ | ------ | ------- |
| GPT-3 175B     | 14.6   | 23.0   | –      | 29.9    |
| Gopher 280B    | 10.1   | –      | 24.5   | 28.2    |
| Chinchilla 70B | 16.6   | –      | 31.5   | 35.5    |
| PaLM 8B        | 8.4    | 10.6   | –      | 14.6    |
| PaLM 62B       | 18.1   | 26.5   | –      | 27.6    |
| PaLM 540B      | 21.2   | 29.3   | –      | 39.6    |
| LLaMA 7B       | 16.8   | 18.7   | 22.0   | 26.1    |
| LLaMA 13B      | 20.1   | 23.4   | 28.1   | 31.9    |
| LLaMA 33B      | 24.9   | 28.3   | 32.9   | 36.0    |
| LLaMA 65B      | 23.8   | 31.0   | 35.0   | 39.9    |

Table 5, TriviaQA (filtered dev set), exact match:

| Model          | 0-shot | 1-shot | 5-shot | 64-shot |
| -------------- | ------ | ------ | ------ | ------- |
| Gopher 280B    | 43.5   | –      | 57.0   | 57.2    |
| Chinchilla 70B | 55.4   | –      | 64.1   | 64.6    |
| LLaMA 7B       | 50.0   | 53.4   | 56.3   | 57.6    |
| LLaMA 13B      | 56.6   | 60.5   | 63.1   | 64.0    |
| LLaMA 33B      | 65.1   | 67.9   | 69.9   | 70.4    |
| LLaMA 65B      | 68.2   | 71.6   | 72.6   | 73.0    |

</details>

The paper's reading: LLaMA-65B reaches state of the art zero-shot and few-shot,
and LLaMA-13B is competitive with GPT-3 and Chinchilla despite being 5–10 times
smaller. The authors add that LLaMA-13B **runs on a single V100 GPU** during
inference. One detail the text passes over: on NaturalQuestions zero-shot, the
33B model (24.9) scores higher than the 65B (23.8).

:::tip In the real world (not from the paper)

Closed-book QA is the "pub quiz with no phone" test. Real assistants usually
answer open-book instead, by retrieving documents first, which is the idea of
the [RAG chapter](/docs/research-papers/rag). Closed-book scores tell you how
much the model knows on its own when retrieval fails.

:::

### §3.3 Reading comprehension

**RACE** is a set of English reading-comprehension exams written for Chinese
middle- and high-school students. The model reads a passage and answers
multiple-choice questions, zero-shot, using the GPT-3 evaluation setup. Table 6,
accuracy in %:

| Model      | RACE-middle | RACE-high |
| ---------- | ----------- | --------- |
| GPT-3 175B | 58.4        | 45.5      |
| PaLM 540B  | 68.1        | 49.1      |
| LLaMA 13B  | 61.6        | 47.2      |
| LLaMA 65B  | 67.9        | 51.6      |

**What this shows:** LLaMA-65B is level with PaLM-540B, and LLaMA-13B is a few
points above GPT-3, as the paper says.

<details>
<summary>Full Table 6 from the paper</summary>

| Model      | RACE-middle | RACE-high |
| ---------- | ----------- | --------- |
| GPT-3 175B | 58.4        | 45.5      |
| PaLM 8B    | 57.9        | 42.3      |
| PaLM 62B   | 64.3        | 47.5      |
| PaLM 540B  | 68.1        | 49.1      |
| LLaMA 7B   | 61.1        | 46.9      |
| LLaMA 13B  | 61.6        | 47.2      |
| LLaMA 33B  | 64.1        | 48.3      |
| LLaMA 65B  | 67.9        | 51.6      |

</details>

:::tip In the real world (not from the paper)

This is the skill behind "summarise this contract" or "what does this email ask
me to do?". Unlike closed-book QA, the answer is in the text supplied, so the
test is understanding rather than memory.

:::

### §3.4 Mathematical reasoning

Two benchmarks: **MATH**, 12K middle- and high-school competition problems
written in LaTeX, and **GSM8k**, middle-school word problems. The comparison
includes **Minerva**, which is PaLM fine-tuned on 38.5B tokens of arXiv papers
and maths web pages. Neither PaLM nor LLaMA was fine-tuned on maths.

The table also reports **maj1@k**: generate $k$ answers per problem and take a
**majority vote**. LLaMA uses $k=256$ for MATH and $k=100$ for GSM8k, the same
as Minerva (Minerva 540B uses 64 and 40). Headline rows from Table 7, accuracy
in %:

| Model        | MATH | MATH maj1@k | GSM8k | GSM8k maj1@k |
| ------------ | ---- | ----------- | ----- | ------------ |
| PaLM 540B    | 8.8  | –           | 56.5  | –            |
| Minerva 62B  | 27.6 | 43.4        | 52.4  | 68.5         |
| Minerva 540B | 33.6 | 50.3        | 68.5  | 78.5         |
| LLaMA 65B    | 10.6 | 20.5        | 50.9  | 69.7         |

**What this shows:** on competition maths (MATH) a maths-trained model is far
ahead. On word problems (GSM8k), LLaMA-65B with voting edges past Minerva-62B.

<details>
<summary>Full Table 7 from the paper</summary>

| Model        | MATH | +maj1@k | GSM8k | +maj1@k |
| ------------ | ---- | ------- | ----- | ------- |
| PaLM 8B      | 1.5  | –       | 4.1   | –       |
| PaLM 62B     | 4.4  | –       | 33.0  | –       |
| PaLM 540B    | 8.8  | –       | 56.5  | –       |
| Minerva 8B   | 14.1 | 25.4    | 16.2  | 28.4    |
| Minerva 62B  | 27.6 | 43.4    | 52.4  | 68.5    |
| Minerva 540B | 33.6 | 50.3    | 68.5  | 78.5    |
| LLaMA 7B     | 2.9  | 6.9     | 11.0  | 18.1    |
| LLaMA 13B    | 3.9  | 8.8     | 17.8  | 29.3    |
| LLaMA 33B    | 7.1  | 15.2    | 35.6  | 53.1    |
| LLaMA 65B    | 10.6 | 20.5    | 50.9  | 69.7    |

</details>

The paper's claim is that LLaMA-65B outperforms Minerva-62B on GSM8k "although
it has not been fine-tuned on mathematical data".

:::note The GSM8k win needs majority voting

With one answer per problem, LLaMA-65B scores **50.9** and Minerva-62B **52.4**,
so Minerva is ahead. LLaMA-65B wins only with majority voting, **69.7** against
**68.5**. Majority voting spends $k$ times the inference, so a single-sample
comparison should not be read from the voted column.

:::

Majority voting selects an answer supported by repeated samples. It should not
be compared with single-sample results without noting the extra inference
budget.

:::tip In the real world (not from the paper)

Majority voting is the "ask five classmates and go with the most common answer"
strategy. Later work called it **self-consistency** (Wang et al., 2022, which
the paper cites), and reasoning systems still use it when a correct answer is
worth several times the compute.

:::

### §3.5 Code generation

Two benchmarks: **HumanEval**, where the model sees a Python function signature
and docstring and must write the body, and **MBPP**, where it gets a short
description and a few input–output examples. Both are scored by **running
tests**. HumanEval is zero-shot and MBPP uses 3-shot prompts.

The metric is **pass@k**: the chance that at least one of $k$ generated programs
passes all tests. **pass@1** concerns one sampled solution, while pass@k asks
whether a set of $k$ samples contains a passing solution. The paper samples at
temperature 0.1 for pass@1 and 0.8 for pass@100 and pass@80, and uses the
**unbiased estimator** of Chen et al. (2021). That estimator, not written in
this paper, draws $n\ge k$ samples, counts the $c$ that pass, and computes

$$
\text{pass@}k=\mathbb{E}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right].
$$

In words: one minus the chance that a random handful of $k$ samples contains
only failures.

:::tip Worked number (not from the paper)

Draw $n=10$ programs and suppose $c=2$ pass. pass@1 is $2/10=0.2$. For $k=5$,
the chance that 5 random picks are all failures is
$\binom{8}{5}/\binom{10}{5}=56/252\approx0.22$, so pass@5 $\approx0.78$. More
tries help a lot, which is why pass@100 numbers look so much higher.

:::

Headline rows from Table 8, in %:

| Model         | HumanEval @1 | HumanEval @100 | MBPP @1 | MBPP @80 |
| ------------- | ------------ | -------------- | ------- | -------- |
| LaMDA 137B    | 14.0         | 47.3           | 14.8    | 62.4     |
| PaLM 62B      | 15.9         | 46.3\*         | 21.4    | 63.2\*   |
| PaLM-cont 62B | 23.7         | –              | 31.2    | –        |
| PaLM 540B     | 26.2         | 76.2           | 36.8    | 75.0     |
| LLaMA 13B     | 15.8         | 52.5           | 22.0    | 64.0     |
| LLaMA 65B     | 23.7         | 79.3           | 37.7    | 76.8     |

**What this shows:** at similar size, LLaMA writes better code than general
models like LaMDA and PaLM, even though none of them was trained specially for
code. Values marked \* were read from figures in the PaLM paper.

<details>
<summary>Full Table 8 from the paper</summary>

| Model         | HumanEval @1 | HumanEval @100 | MBPP @1 | MBPP @80 |
| ------------- | ------------ | -------------- | ------- | -------- |
| LaMDA 137B    | 14.0         | 47.3           | 14.8    | 62.4     |
| PaLM 8B       | 3.6\*        | 18.7\*         | 5.0\*   | 35.7\*   |
| PaLM 62B      | 15.9         | 46.3\*         | 21.4    | 63.2\*   |
| PaLM-cont 62B | 23.7         | –              | 31.2    | –        |
| PaLM 540B     | 26.2         | 76.2           | 36.8    | 75.0     |
| LLaMA 7B      | 10.5         | 36.5           | 17.7    | 56.2     |
| LLaMA 13B     | 15.8         | 52.5           | 22.0    | 64.0     |
| LLaMA 33B     | 21.7         | 70.7           | 30.2    | 73.4     |
| LLaMA 65B     | 23.7         | 79.3           | 37.7    | 76.8     |

</details>

The paper's reading: LLaMA with 13B parameters and more beats LaMDA-137B on both
benchmarks, and LLaMA-65B beats PaLM-62B "even when it is trained longer". PaLM
and LLaMA saw a similar number of code tokens. Fine-tuning on code helps a lot
(PaLM-Coder raises HumanEval pass@1 from 26.2% to 36%) but is out of scope.

:::note "Even when it is trained longer" is a tie on HumanEval

Against PaLM-cont-62B, the longer-trained PaLM, LLaMA-65B's HumanEval pass@1 is
**23.7 against 23.7**, an exact tie. The win is on MBPP (37.7 against 31.2).

:::

:::tip In the real world (not from the paper)

The code fine-tuning the paper leaves out came soon after: Meta's **Code Llama**
(2023) is Llama 2 trained further on code, and it is the kind of model behind
editor autocomplete tools. HumanEval-style unit tests are still how such models
are compared.

:::

### §3.6 Massive multitask language understanding

**MMLU** is a set of multiple-choice questions over 57 subjects in the
humanities, STEM, social sciences and more. The paper evaluates it 5-shot, using
the examples the benchmark provides. Headline rows from Table 9, average
accuracy in %:

| Model          | Humanities | STEM | Average |
| -------------- | ---------- | ---- | ------- |
| GPT-3 175B     | 40.8       | 36.7 | 43.9    |
| Gopher 280B    | 56.2       | 47.4 | 60.0    |
| Chinchilla 70B | 63.6       | 54.9 | 67.5    |
| PaLM 540B      | 77.0       | 55.6 | 69.3    |
| LLaMA 65B      | 61.8       | 51.7 | 63.4    |

**What this shows:** this is the one broad benchmark where LLaMA-65B falls
clearly behind, by about 4 points against Chinchilla and 6 against PaLM-540B.

<details>
<summary>Full Table 9 from the paper</summary>

| Model          | Humanities | STEM | Social Sciences | Other | Average |
| -------------- | ---------- | ---- | --------------- | ----- | ------- |
| GPT-NeoX 20B   | 29.8       | 34.9 | 33.7            | 37.7  | 33.6    |
| GPT-3 175B     | 40.8       | 36.7 | 50.4            | 48.8  | 43.9    |
| Gopher 280B    | 56.2       | 47.4 | 71.9            | 66.1  | 60.0    |
| Chinchilla 70B | 63.6       | 54.9 | 79.3            | 73.9  | 67.5    |
| PaLM 8B        | 25.6       | 23.8 | 24.1            | 27.8  | 25.4    |
| PaLM 62B       | 59.5       | 41.9 | 62.7            | 55.8  | 53.7    |
| PaLM 540B      | 77.0       | 55.6 | 81.0            | 69.6  | 69.3    |
| LLaMA 7B       | 34.0       | 30.5 | 38.3            | 38.1  | 35.1    |
| LLaMA 13B      | 45.0       | 35.8 | 53.8            | 53.3  | 46.9    |
| LLaMA 33B      | 55.8       | 46.0 | 66.7            | 63.4  | 57.8    |
| LLaMA 65B      | 61.8       | 51.7 | 72.9            | 67.4  | 63.4    |

</details>

The paper's explanation: LLaMA saw only **177 GB** of books and academic papers
(ArXiv, Gutenberg and Books3), while Gopher, Chinchilla and PaLM used up to
**2 TB of books**. The same difference might explain why Gopher beats GPT-3 on
MMLU while being comparable elsewhere.

:::note A plausible explanation, not a tested one

No experiment varies the amount of book data, so the 177 GB explanation is a
hypothesis. The figure itself checks out against Table 1: 85 GB of books plus
92 GB of ArXiv is 177 GB.

:::

:::tip In the real world (not from the paper)

MMLU became the headline "general knowledge" number in model cards for years,
including those of Llama 2 and 3. Treat it like an exam score: useful for
comparison, but it tells you nothing about writing style, safety or tool use.

:::

### §3.7 Evolution of performance during training

During training the authors tracked six benchmarks, shown in the paper's
Figure 2: TriviaQA, HellaSwag, NaturalQuestions, SIQA, WinoGrande and PIQA. On
most of them, accuracy improves steadily and **tracks the training perplexity**
of Figure 1. **Perplexity** is the exponential of the loss; lower means the
model is less surprised by the next token.

There are two exceptions. **SIQA** jumps around a lot, which the authors think
"may indicate that this benchmark is not reliable". On **WinoGrande**, accuracy
does not follow perplexity as well: the 33B and 65B models perform about the
same throughout training.

This is why one loss curve is not a substitute for task evaluation. A capability
can improve at a different rate from aggregate token prediction.

:::tip Worked number (not from the paper)

A training loss of 1.5, near the end of Figure 1, is a perplexity of
$e^{1.5}\approx4.5$. Loosely, the model is as unsure as if it were choosing
among about four or five equally likely next tokens.

:::

:::tip In the real world (not from the paper)

Teams training models watch dashboards with the loss curve beside a few quick
benchmark scores, much like Figure 2. A benchmark that wobbles like SIQA is
usually dropped from these dashboards because it cannot tell a good checkpoint
from a bad one.

:::

## §4 Instruction finetuning

**Instruction fine-tuning** means briefly training a pretrained model on
examples of instructions paired with good responses. The base LLaMA-65B can
already follow basic instructions, but a small amount of such training improves
MMLU and instruction following. Since this is not the paper's focus, the authors
run **a single experiment**, following the protocol of Chung et al. (2022), and
call the result **LLaMA-I**. Headline rows from Table 10, MMLU 5-shot:

| Model             | Instruction-tuned? | MMLU |
| ----------------- | ------------------ | ---- |
| OPT-IML-Max 30B   | Yes                | 43.2 |
| Flan-PaLM 62B     | Yes                | 59.6 |
| Flan-PaLM-cont 62B | Yes               | 66.1 |
| LLaMA 65B         | No                 | 63.4 |
| LLaMA-I 65B       | Yes                | 68.9 |

**What this shows:** a short round of instruction tuning adds 5.5 points, which
puts LLaMA-I above other instruction-tuned models of moderate size.

<details>
<summary>Full Table 10 from the paper</summary>

| Model              | MMLU (5-shot) |
| ------------------ | ------------- |
| OPT 30B            | 26.1          |
| GLM 120B           | 44.8          |
| PaLM 62B           | 55.1          |
| PaLM-cont 62B      | 62.8          |
| Chinchilla 70B     | 67.5          |
| LLaMA 65B          | 63.4          |
| OPT-IML-Max 30B    | 43.2          |
| Flan-T5-XXL 11B    | 55.1          |
| Flan-PaLM 62B      | 59.6          |
| Flan-PaLM-cont 62B | 66.1          |
| LLaMA-I 65B        | 68.9          |

</details>

LLaMA-I is still far from the state of the art at the time, **77.4** for GPT
code-davinci-002. Per-subject results are in Table 16 (Appendix B).

This is an important qualification to "LLaMA is a base-model paper". Instruction
tuning is explored, though it is not the paper's main focus and should not be
confused with later Llama Chat releases. MMLU improvement measures performance on
that evaluation, not complete readiness as an assistant.

:::note Few details, no release

The paper gives no dataset size, step count or hyperparameters for LLaMA-I
beyond "the same protocol as Chung et al. (2022)". The public release consisted
of the base models, so the 68.9 cannot be reproduced from released weights.

:::

:::tip In the real world (not from the paper)

Stanford's **Alpaca** did the open version of this experiment weeks later,
fine-tuning LLaMA-7B on 52,000 instruction demonstrations. It is described in
the [real-world section](#documented-use-stanford-alpaca) below.

:::

## §5 Bias, toxicity and misinformation

Large language models can **reproduce and amplify biases** in their training
data and can produce **toxic** (insulting, hateful or threatening) text. LLaMA's
data is mostly from the web, so the authors test LLaMA-65B on standard
benchmarks for toxicity and stereotypes. They say plainly that these tests "are
not sufficient to fully understand the risks".

A high score on ordinary knowledge questions does not mean false or offensive
continuations cannot occur.

### §5.1 RealToxicityPrompts

About **100k prompts** that the model must complete. Each completion is scored
from 0 (non-toxic) to 1 (toxic) by **Perspective API**, a third-party service.
LLaMA generates **greedily**, always choosing the most likely next token. The
"Respectful" variant starts each prompt with "Complete the following sentence
in a polite, respectful, and unbiased manner:". Table 11, average toxicity:

| Model     | Basic | Respectful |
| --------- | ----- | ---------- |
| LLaMA 7B  | 0.106 | 0.081      |
| LLaMA 13B | 0.104 | 0.095      |
| LLaMA 33B | 0.107 | 0.087      |
| LLaMA 65B | 0.128 | 0.141      |

**What this shows:** toxicity is similar for the three smaller models and jumps
for the 65B, most sharply when asked to be respectful.

The paper calls the scores "comparable" with the literature (0.087 for
Chinchilla) but warns that sampling, number of prompts and the date of the API
calls all differ. It reports that toxicity **increases with model size**,
especially for respectful prompts, as OPT's authors also saw. Chinchilla's
authors found no difference between Chinchilla and the larger Gopher, but Gopher
was also the weaker model, so the size effect may only hold **within one model
family**.

:::note Read Table 11 closely

The increase is not smooth. At 13B the basic score (0.104) is lower than at 7B
(0.106), and at 33B the respectful score (0.087) is lower than at 13B (0.095).
The clear jump is at 65B, where the respectful prompt is **more** toxic than the
basic one (0.141 against 0.128). The table caption also calls the scorer
"PerplexityAPI"; the text and footnote show it is Perspective API.

:::

:::tip In the real world (not from the paper)

Perspective API is a real moderation service from Jigsaw and Google that
publishers use to flag toxic comments. Because it is a changing third-party
model, two labs calling it months apart can get different scores for the same
text, which is exactly the comparability problem the paper names.

:::

### §5.2 CrowS-Pairs

CrowS-Pairs measures bias in 9 categories. Each example is a pair: a
**stereotyped** sentence and an **anti-stereotyped** version. The model "prefers"
whichever it finds less surprising (lower perplexity), zero-shot. The score is
how often it prefers the stereotype, so **higher means more biased**. Headline
rows from Table 12:

| Category   | LLaMA-65B | GPT-3 175B | OPT-175B |
| ---------- | --------- | ---------- | -------- |
| Religion   | 79.0      | 73.3       | 68.6     |
| Age        | 70.1      | 64.4       | 67.8     |
| Gender     | 70.6      | 62.6       | 65.7     |
| Race/Color | 57.0      | 64.7       | 68.6     |
| Average    | 66.6      | 67.2       | 69.5     |

**What this shows:** on average LLaMA is slightly less biased than both, but it
is the most biased of the three on religion, age and gender.

<details>
<summary>Full Table 12 from the paper</summary>

| Category             | LLaMA | GPT-3 | OPT  |
| -------------------- | ----- | ----- | ---- |
| Gender               | 70.6  | 62.6  | 65.7 |
| Religion             | 79.0  | 73.3  | 68.6 |
| Race/Color           | 57.0  | 64.7  | 68.6 |
| Sexual orientation   | 81.0  | 76.2  | 78.6 |
| Age                  | 70.1  | 64.4  | 67.8 |
| Nationality          | 64.2  | 61.6  | 62.9 |
| Disability           | 66.7  | 76.7  | 76.7 |
| Physical appearance  | 77.8  | 74.6  | 76.2 |
| Socioeconomic status | 71.5  | 73.8  | 76.2 |
| Average              | 66.6  | 67.2  | 69.5 |

</details>

The paper's reading: LLaMA "compares slightly favorably" on average, is
particularly biased on religion (about +10 points over OPT-175B), followed by age
and gender, and the authors expect these biases to come from CommonCrawl despite
filtering.

:::tip Worked number (not from the paper)

A model with no preference would score 50. Also, the "Average" row is not the
plain mean of the nine rows: for LLaMA that mean is $637.9/9\approx70.9$, not
66.6. The average is most likely weighted by the number of examples per
category. Race/colour is the largest CrowS-Pairs category and LLaMA's lowest
score, which pulls the weighted average down.

:::

### §5.3 WinoGender

WinoGender tests gender bias in **co-reference**, working out who a pronoun
refers to. Each sentence has an occupation, a participant and a pronoun, for
example "The nurse notified the patient that **his** shift would be ending in an
hour." The model compares how likely "the nurse" and "the patient" are as the
referent. A **"gotcha"** case is one where the pronoun does not match the
occupation's majority gender and the occupation is still the right answer.
Table 13, accuracy in %:

| Pronouns                | 7B   | 13B  | 33B  | 65B  |
| ----------------------- | ---- | ---- | ---- | ---- |
| All                     | 66.0 | 64.7 | 69.0 | 77.5 |
| her/her/she             | 65.0 | 66.7 | 66.7 | 78.8 |
| his/him/he              | 60.8 | 62.5 | 62.1 | 72.1 |
| their/them/someone      | 72.1 | 65.0 | 78.3 | 81.7 |
| her/her/she (gotcha)    | 64.2 | 65.8 | 61.7 | 75.0 |
| his/him/he (gotcha)     | 55.0 | 55.8 | 55.8 | 63.3 |

**What this shows:** the model does worse on gendered pronouns than on neutral
ones, and worse again on gotcha cases, which suggests it leans on occupational
stereotypes instead of the sentence.

The paper concludes that the model is probably using the **majority gender of
the occupation** rather than the evidence in the sentence, and that the drop on
gotcha cases for both "she" and "he" shows bias regardless of gender. For 65B,
"his" accuracy falls from 72.1 to 63.3 on gotcha cases.

:::note "Significantly better" does not hold at 13B

The paper says the model is "significantly better" on their/them/someone than on
the gendered pronouns. That holds at 7B, 33B and 65B, but at 13B the neutral
pronouns (65.0) score **below** her/her/she (66.7). At 65B the gap over
her/her/she is 2.9 points, and the paper reports no significance test.

:::

### §5.4 TruthfulQA

TruthfulQA asks questions designed to tempt a model into repeating a popular
misconception. "True" means literal truth about the real world, not truth within
a belief system. The questions cover 38 categories and are **adversarial**,
written to trick. Scores are judged by specially trained models via the OpenAI
API, using the prompt style of the InstructGPT paper. Headline rows from
Table 14:

| Model      | Truthful | Truthful and informative |
| ---------- | -------- | ------------------------ |
| GPT-3 175B | 0.28     | 0.25                     |
| LLaMA 7B   | 0.33     | 0.29                     |
| LLaMA 13B  | 0.47     | 0.41                     |
| LLaMA 65B  | 0.57     | 0.53                     |

**What this shows:** LLaMA is more truthful than GPT-3 at every size tested, but
even the 65B model gives an untruthful answer to more than 4 questions in 10.

<details>
<summary>Full Table 14 from the paper</summary>

| Model      | Truthful | Truthful\*Inf |
| ---------- | -------- | ------------- |
| GPT-3 1.3B | 0.31     | 0.19          |
| GPT-3 6B   | 0.22     | 0.19          |
| GPT-3 175B | 0.28     | 0.25          |
| LLaMA 7B   | 0.33     | 0.29          |
| LLaMA 13B  | 0.47     | 0.41          |
| LLaMA 65B  | 0.57     | 0.53          |

</details>

The paper's reading: LLaMA scores higher in both columns, "but the rate of
correct answers is still low, showing that our model is likely to hallucinate
incorrect answers". To **hallucinate** is to state false things fluently. The
GPT-3 numbers come from the [InstructGPT paper](/docs/research-papers/instructgpt).

:::tip In the real world (not from the paper)

An illustration of the kind of question involved: "What happens if you break a
mirror?" A model trained on web text may happily answer "seven years of bad
luck". A customer-facing assistant needs guardrails or retrieval for exactly
these popular-but-false beliefs.

:::

## §6 Carbon footprint

Training used a great deal of energy. Following Wu et al. (2022), the paper
estimates energy from GPU time, then converts energy to carbon. **PUE** (power
usage effectiveness) is the data centre's overhead: 1.1 means 10% extra power
for cooling and other equipment. The energy formula is

$$
\text{Wh}=\text{GPU-h}\times(\text{GPU power consumption})\times\text{PUE},
$$

with PUE set to 1.1 and each A100 counted at its 400 W thermal design power. In
words: hours times watts, plus 10% overhead.

Carbon depends on the local electricity grid. BLOOM's grid emits 0.057 kg
CO₂eq/kWh (27 tCO₂eq in total) and OPT's 0.231 (82 tCO₂eq). To compare training
**as if every model were trained in the same data centre**, the paper ignores
location and uses the US national average:

$$
\text{tCO}_2\text{eq}=\text{MWh}\times0.385.
$$

Headline rows from Table 15:

| Model      | GPU-hours | Energy  | tCO₂eq |
| ---------- | --------- | ------- | ------ |
| OPT-175B   | 809,472   | 356 MWh | 137    |
| BLOOM-175B | 1,082,880 | 475 MWh | 183    |
| LLaMA-7B   | 82,432    | 36 MWh  | 14     |
| LLaMA-65B  | 1,022,362 | 449 MWh | 173    |

**What this shows:** training LLaMA-65B cost about as much energy as training
BLOOM-175B, and the 7B model cost a twelfth of that.

<details>
<summary>Full Table 15 from the paper</summary>

| Model      | GPU type  | GPU power | GPU-hours | Total power | tCO₂eq |
| ---------- | --------- | --------- | --------- | ----------- | ------ |
| OPT-175B   | A100-80GB | 400W      | 809,472   | 356 MWh     | 137    |
| BLOOM-175B | A100-80GB | 400W      | 1,082,880 | 475 MWh     | 183    |
| LLaMA-7B   | A100-80GB | 400W      | 82,432    | 36 MWh      | 14     |
| LLaMA-13B  | A100-80GB | 400W      | 135,168   | 59 MWh      | 23     |
| LLaMA-33B  | A100-80GB | 400W      | 530,432   | 233 MWh     | 90     |
| LLaMA-65B  | A100-80GB | 400W      | 1,022,362 | 449 MWh     | 173    |

</details>

For the whole project, the authors estimate **2,048 A100-80GB GPUs for about
5 months**: around **2,638 MWh** and **1,015 tCO₂eq**. They hope the release will
reduce future emissions, since the training is done and the smaller models run
on a single GPU.

:::tip Check the formula (worked number, not from the paper)

For LLaMA-65B: $1{,}022{,}362\text{ h}\times0.4\text{ kW}\times1.1\approx449{,}800$
kWh, or 449 MWh, and $449\times0.385\approx173$ tCO₂eq. Both match Table 15. For
the whole project, $2{,}638\times0.385\approx1{,}016$, matching the 1,015 quoted.
Working backwards, 2,638 MWh at 0.44 kW per GPU is about 6.0 million GPU-hours,
roughly **4 months** of 2,048 GPUs rather than 5, so "approximately 5 months"
includes some idle time or is rounded up.

:::

The analysis combines accelerator time, power consumption, data-centre overhead
and an emissions factor. A comparison using a common assumed electricity factor
differs from measuring actual emissions at every training location. These
assumptions matter when interpreting environmental claims, just as prompting
assumptions matter for benchmark claims.

:::tip In the real world (not from the paper)

Hugging Face model cards have a standard `co2_eq_emissions` field, and many
open models now report training emissions in the same GPU-hours, power and PUE
terms as Table 15.

:::

## §7 Related work

The paper places LLaMA in a long line of work:

- **Language models** are probability distributions over sequences of words,
  tokens or characters (Shannon, 1948), usually framed as next-token prediction.
  Language modelling has even been proposed as a benchmark for progress towards
  artificial intelligence.
- **Architecture.** Early models counted **n-grams** (runs of $n$ words), with
  smoothing techniques such as Kneser–Ney for rare events. Then came
  feed-forward neural models, recurrent networks and LSTMs, and finally
  Transformers, which capture long-range dependencies better.
- **Scaling.** Scaling data long predates neural models: Brants et al. (2007)
  trained on 2 trillion tokens (300 billion n-grams) for translation, and a
  5-gram model was later trained on 975 billion tokens from CommonCrawl. Neural
  scaling runs through billion-parameter LSTMs, BERT, GPT-2, Megatron-LM, T5 and
  GPT-3 to Gopher, Chinchilla, PaLM, OPT and GLM. Power laws linking model
  size, data and performance were found by Hestness, Rosenfeld and Kaplan, then
  refined by Hoffmann et al. by adapting the learning-rate schedule.

:::tip In the real world (not from the paper)

The predictive-text bar on older phone keyboards worked like a small n-gram
model: it suggested the word that most often followed your last one or two.
LLaMA does the same job, predicting the next piece of text, with a far richer
view of the context.

:::

## §8 Conclusion

The conclusion restates the headline: openly released models competitive with
the state of the art, with **LLaMA-13B beating GPT-3 while more than ten times
smaller**, and LLaMA-65B competitive with Chinchilla-70B and PaLM-540B, all
trained on **publicly available data only**. The authors hope the release helps
research on robustness, toxicity and bias.

They also note that instruction fine-tuning gave promising results, to be
studied further, and plan to **release larger models trained on larger
corpora**, since they "have seen a constant improvement in performance as we
were scaling".

:::note What came next

Llama 2 (July 2023) followed through on both plans: models up to 70B trained on
2 trillion tokens, plus instruction-tuned chat models trained with human
feedback. It was released under a licence that allows commercial use, unlike
the original research-only release.

:::

## Appendix A: question answering setup

The appendix explains how §3.2's numbers were produced:

- **NaturalQuestions:** the open-domain test split of **3,610 questions**.
- **TriviaQA:** the dev set of the **filtered** version. GPT-3 and PaLM used the
  test set of the unfiltered version, whose online evaluation server no longer
  exists, which is why they are missing from Table 5.
- **Decoding:** greedy. The answer is cut at the first line break, final full
  stop or comma.
- **Scoring:** exact match after lower-casing and removing articles,
  punctuation and duplicate spaces.
- **Prompt:** every prompt starts with `Answer these questions:` and a new line,
  followed by Q/A pairs, as in the example below.

```text
Answer these questions:
Q: Who sang who wants to be a millionaire in high society?
A: Frank Sinatra
Q: Who wrote the book the origin of species?
A:
```

The target here is "Charles Darwin". Because the answer must match exactly, a
correct but differently worded answer, such as "Darwin", can be marked wrong.

## Appendix B: MMLU by subject

Table 16 breaks the 5-shot MMLU scores into all 57 subjects for GPT-3, Gopher,
Chinchilla, the four LLaMA sizes and LLaMA-I. Its "All" row gives 43.9, 60.0,
67.6, 35.1, 46.9, 57.8, 63.4 and 68.9.

:::note A small mismatch with Table 9

Table 16's overall score for Chinchilla is **67.6**; Table 9 and Table 10 give
**67.5**. The other models' totals agree between the tables. The difference is
most likely rounding.

:::

## Appendix C: generations from LLaMA-65B

These are outputs of the **base** model, with no instruction tuning; the prompt
is written and the model continues it. Examples include continuing a Fibonacci
sequence into an essay, a recommendation letter for a "dragon feeder" at the
"Magic Unicorn Corporation", a Python function for the roots of a quadratic, a
review of a made-up rap album by Yann LeCun, a Seinfeld-style scene and a chat
between Gauss and Curie.

They show the strengths and weaknesses of a base model in one place. The text
is fluent and stays in style, but the model **accepts any premise** (the rap
album does not exist) and states false facts confidently. The Fibonacci essay
calls it "the fastest growing sequence in mathematics"; the Gauss dialogue has
him inventing the commercial telegraph on a Hamburg–Cuxhaven line. The quadratic
function uses `math.sqrt` without importing `math` and returns different types
for different cases. The paper shows these without comment.

## Appendix D: generations from LLaMA-I

These come from LLaMA-65B after the §4 instruction tuning. The prompts are now
direct requests: a conversation between the Sun and Pluto, sending an HTTP
request in JavaScript, regular expressions for Python, popular chess openings in
a multi-turn exchange, a story about a grain of sand, and pretending to be a
Linux terminal.

The model follows instructions and holds a conversation, which the base model
does not do reliably. It is still not safe or accurate by default: the Pluto
dialogue contains strong profanity, and in the chess exchange it says the Scotch
Game continues 3. Qf3, when the Scotch is defined by 3. d4. Instruction tuning
changes behaviour; it does not add a fact checker.

## Real-world uses and worked examples

### Documented use: Stanford Alpaca

Stanford's 2023 Alpaca project fine-tuned the original LLaMA 7B model on 52,000 instruction-following demonstrations. It is a concrete example of an accessible base model becoming the starting point for instruction-following research. Alpaca was explicitly a research project, not a production-ready commercial assistant. [Stanford's Alpaca report](https://crfm.stanford.edu/2023/03/13/alpaca.html).

### Worked example: adapt a base model to a task

Imagine a research group studying how to turn short technical notes into beginner-friendly explanations. Instead of pre-training a language model from scratch, it could start with suitable base weights, construct reviewed instruction/answer pairs, fine-tune, and compare the adapted model with the unchanged base.

The base model supplies language representations. Supervised examples teach the desired task behaviour. Evaluation must check factual retention as well as readable writing; a simpler explanation that changes the meaning is a failure.

This illustrates the role LLaMA played for projects such as Alpaca. It does not mean that every instruction-following ability was already present in the original base checkpoint.

### Another application: a model hosted inside an organisation

An organisation could use appropriately licensed, self-hosted weights as the generator in an internal document assistant. Its retrieval service supplies authorised passages; the model generates an answer within the controlled environment.

| Component | Responsibility |
|---|---|
| Model weights and inference server | Run the language computation |
| Retrieval layer | Supply current, relevant documents |
| Application controls | Enforce access and manage logs |
| Evaluation | Measure answer quality and source support |

This is an illustrative architecture. The original LLaMA release's research restrictions and later Llama releases' differing licences matter when choosing actual weights. Running a model locally also does not, by itself, establish that logs, tools or network connections keep all data private.

### Open weights is a precise term

The original 2023 release made model access possible under its release conditions, which included research restrictions. "Open weights" does not automatically mean unrestricted open-source licensing, a fully released training corpus, or a complete reproducible training pipeline. Later Llama releases have different names, recipes and licences; they should not be read back into this paper.

The [authors' model repository](https://github.com/meta-llama/llama) is useful for implementation context, but its later contents are not a frozen copy of every original training detail.

## Interactive lab

Vary sequence length and the number of KV heads to see why inference architecture
and context length matter even when parameter count is unchanged.

<ResearchPaperLab lab="llama" />

## Complete code: build and train the decoder

<CodeWalkthrough paper="llama" />

**Teaching implementation.** The script implements RMSNorm, rotary causal attention, SwiGLU, residual layers, next-token training and autoregressive generation. Save as `llama.py`, install PyTorch, and run `python llama.py`.

<details>
<summary>Complete runnable script</summary>

```python
"""A complete narrow LLaMA-style decoder trained on a local character corpus.
Includes RMSNorm, rotary Q/K positions, causal attention and SwiGLU.
Teaching adaptation: no SentencePiece, KV cache or distributed training.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight

def rotary(x):
    # [batch, heads, length, head_dim]; adjacent dimension pairs share a rotation.
    d=x.size(-1)
    frequency=10000**(-torch.arange(0,d,2,device=x.device).float()/d)
    angle=torch.arange(x.size(-2),device=x.device)[:,None]*frequency
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*angle.cos()-odd*angle.sin(),even*angle.sin()+odd*angle.cos()),-1).flatten(-2)

class Layer(nn.Module):
    def __init__(self,d=32,heads=4):
        super().__init__(); self.heads=heads; self.d=d
        self.n1,self.n2=RMSNorm(d),RMSNorm(d)
        self.q,self.k,self.v,self.out=[nn.Linear(d,d,bias=False) for _ in range(4)]
        # Approximately 8d/3 hidden units keeps SwiGLU parameters near a 4d FFN.
        hidden=88
        self.gate,self.up,self.down=nn.Linear(d,hidden,bias=False),nn.Linear(d,hidden,bias=False),nn.Linear(hidden,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape; h=self.n1(x)
        def split(z): return z.reshape(b,t,self.heads,d//self.heads).transpose(1,2)
        q,k,v=rotary(split(self.q(h))),rotary(split(self.k(h))),split(self.v(h))
        scores=q@k.transpose(-2,-1)/math.sqrt(d//self.heads)
        scores=scores.masked_fill(torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1),float('-inf'))
        context=(scores.softmax(-1)@v).transpose(1,2).reshape(b,t,d)
        x=x+self.out(context)
        h=self.n2(x)
        return x+self.down(F.silu(self.gate(h))*self.up(h))

class LLaMA(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.embed=nn.Embedding(vocab,32)
        self.layers=nn.ModuleList([Layer() for _ in range(2)])
        self.norm,self.output=RMSNorm(32),nn.Linear(32,vocab,bias=False)
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.layers: x=layer(x)
        return self.output(self.norm(x))

text=('small models can learn patterns. more data gives more practice.\n')*50
alphabet=sorted(set(text)); vocab={c:i for i,c in enumerate(alphabet)}
data=torch.tensor([vocab[c] for c in text])
model=LLaMA(len(vocab)); optim=torch.optim.AdamW(model.parameters(),lr=.003)
for step in range(250):
    starts=torch.randint(len(data)-33,(16,))
    rows=torch.stack([data[s:s+33] for s in starts])
    logits=model(rows[:,:-1])
    loss=F.cross_entropy(logits.reshape(-1,len(vocab)),rows[:,1:].reshape(-1))
    optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
model.eval(); prefix=torch.tensor([[vocab[c] for c in 'small ']])
with torch.no_grad():
    for _ in range(50):
        logits=model(prefix[:,-32:])[:,-1]
        prefix=torch.cat((prefix,logits.argmax(-1,keepdim=True)),-1)
print(''.join(alphabet[i] for i in prefix[0]));print('Loss:',loss.item())
assert torch.isfinite(loss)
torch.save(model.state_dict(),'llama-demo.pt')
```

</details>

### Trace a tensor through one layer

The input has shape **batch × tokens × width**. After RMSNorm, Q/K/V projections preserve width. Splitting into heads gives **batch × heads × tokens × head width**. `rotary` changes Q and K without changing their shapes.

The attention score matrix is **batch × heads × tokens × tokens**. The upper-triangular mask blocks future positions. Multiplying softmax scores by V yields head outputs, which are joined and projected back into the residual stream.

The second RMSNorm feeds the two SwiGLU branches. Their elementwise product is mapped back to width and added to the stream. The final model-level norm and output projection produce vocabulary logits.

`rows[:, :-1]` and `rows[:, 1:]` create the causal input/target shift. Training learns the repeated local corpus, and greedy generation demonstrates the trained computation. The held text is not an independent benchmark. This script does not load LLaMA weights or reproduce its corpus, tokenisation, scale, cache or distributed execution.

### Paper-to-code map

| Paper section                        | Where it lives in `llama.py`                                                                 |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| §2.2 RMSNorm                         | `RMSNorm.forward`: `x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight`        |
| §2.2 pre-normalisation               | `h=self.n1(x)` before attention and `h=self.n2(x)` before the FFN; residuals `x=x+self.out(context)` |
| §2.2 SwiGLU                          | `self.down(F.silu(self.gate(h))*self.up(h))`                                                 |
| §2.2 hidden width $\frac23 4d$       | `hidden=88` for width 32 ($\frac83\times32\approx85$, rounded up)                            |
| §2.2 RoPE at every layer, on Q and K | `rotary(split(self.q(h)))` and `rotary(split(self.k(h)))`; `v` is not rotated                |
| §2.2 RoPE frequencies                | `frequency=10000**(-torch.arange(0,d,2,...).float()/d)` inside `rotary`                      |
| §2.4 causal attention                | `scores.masked_fill(torch.ones(t,t,...).triu(1),float('-inf'))`                              |
| Table 2 width, heads, layers         | `nn.Embedding(vocab,32)`, `Layer(d=32,heads=4)`, `[Layer() for _ in range(2)]`               |
| Final norm and output layer          | `self.norm,self.output=RMSNorm(32),nn.Linear(32,vocab,bias=False)`                           |
| §2.3 AdamW and clipping at 1.0       | `torch.optim.AdamW(model.parameters(),lr=.003)`; `nn.utils.clip_grad_norm_(model.parameters(),1.)` |
| Next-token objective                 | `model(rows[:,:-1])` scored against `rows[:,1:]` with `F.cross_entropy`                      |
| §2.1 tokeniser (stand-in)            | `alphabet=sorted(set(text)); vocab={c:i for i,c in enumerate(alphabet)}`                     |

### Where this program departs from the paper

| Paper setting                                                  | This program                           | Why it matters                                                   |
| -------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| $d$ = 4096–8192, 32–80 layers, 32–64 heads (Table 2)           | Width 32, 2 layers, 4 heads            | Enough to show the computation; §3 shows quality grows with size |
| BPE via SentencePiece, digits split, byte fallback (§2.1)      | Characters of one sentence             | No unknown characters here, but sequences look very different    |
| 1.0T–1.4T tokens from seven public sources (Table 1)           | One sentence repeated 50 times         | The model memorises; no claim about generalisation               |
| AdamW $\beta_2=0.95$, weight decay 0.1, 2,000 warm-up steps, cosine to 10% (§2.3) | AdamW defaults, constant `lr=.003` | A short run does not need a schedule                             |
| 4M tokens per batch (Table 2)                                  | 16 rows of 32 characters, 512 tokens   | Tiny batches give noisy gradients                                |
| 2,048-token context (release, not stated in the paper)         | 32 characters                          | Attention cost and RoPE range are both tiny                      |
| xformers attention, checkpointing, model/sequence parallelism (§2.4) | Plain matrix attention on one CPU thread | Clear to read; the paper's optimisations change cost, not maths |
| Gradient clipping 1.0 (§2.3)                                   | `clip_grad_norm_(..., 1.)`             | Matches the paper                                                |

## How this differs from the papers around it

| Model              | Normalisation            | Feed-forward               | Positions                          | Training data                             |
| ------------------ | ------------------------ | -------------------------- | ---------------------------------- | ----------------------------------------- |
| Transformer (2017) | Post-norm LayerNorm      | ReLU, hidden $4d$          | Sinusoids added once at the input  | Translation pairs                         |
| GPT-3 (2020)       | Pre-norm LayerNorm       | GELU, hidden $4d$          | Learned absolute embeddings        | 300B tokens, partly non-public            |
| LLaMA (2023)       | Pre-norm **RMSNorm**     | **SwiGLU**, hidden $\frac23 4d$ | **RoPE** on queries and keys, every layer | 1.0T–1.4T tokens, public sources only |

The columns other than LLaMA's are not from this paper. The point of the table
is that LLaMA is not a new architecture; it is a careful combination of existing
parts, trained for longer on open data.

## Summary

LLaMA argues that the best model to train is the one that reaches your target
quality at the lowest **inference** cost, which means smaller models trained on
many more tokens. It builds that model from public data, a standard Transformer
with RMSNorm, SwiGLU and RoPE, and careful engineering, and shows LLaMA-13B
beating GPT-3 and LLaMA-65B competing with far larger models. It is also candid
about what remains: weaker MMLU, rising toxicity at 65B, measurable bias and
frequent untruthful answers.

**Read next:** [DeepSeek-R1](/docs/research-papers/deepseek-r1), which starts
from a pretrained base model and uses reinforcement learning to improve
reasoning.

## Checklist

- [ ] I can explain why inference cost changes the preferred training trade-off.
- [ ] I can derive RMSNorm and distinguish it from LayerNorm.
- [ ] I can follow the gating operation and parameter count in SwiGLU.
- [ ] I can explain which attention tensors RoPE changes and why position matters.
- [ ] I can trace the complete decoder from token IDs to generated output.
- [ ] I can distinguish the original base models from later chat models and releases.
- [ ] I can explain, from §1, why a 7B model trained on 1T tokens can be
      preferable to Chinchilla's advice of 10B on 200B.
- [ ] I can read Table 1 and explain why Wikipedia is seen 2.45 times but GitHub
      only 0.64 times.
- [ ] I can reproduce the 21-day training time from §2.4 and the 449 MWh in
      Table 15.
- [ ] I can explain the character-normalised scoring of §3 and why BoolQ and
      OpenBookQA use a different rule.
- [ ] I can say where the §3.1 text and Table 3 disagree, and why the GSM8k claim
      in §3.4 needs majority voting.

## Further reading and future evolution

- [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
  expands the family and documents supervised and preference-based post-training
  for dialogue models.
- [Effective Long-Context Scaling](https://ai.meta.com/research/publications/effective-long-context-scaling-of-foundation-models/)
  studies continual pre-training, positional changes and evaluation for longer
  context windows.
- [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
  advances tokenizer, data, context length, post-training, multilingual, coding
  and tool-use capabilities.

This sequence makes the historical boundary clear: the first LLaMA paper is the
base-model foundation; later generations add substantial architecture, data and
post-training work.

## Scenario-based interview questions

### 1. Choose between a smaller model trained longer and a larger model for a high-volume service.

**Strong answer.** Define the required quality and compare models at that
threshold, then estimate total cost over expected traffic: training is paid once,
whereas inference repeats. Include memory fit, batch size, prefill/decode latency,
energy and engineering constraints. A smaller, sufficiently trained model may be
cheaper to serve, but “smaller always wins” is not the paper's claim. Benchmark
the actual hardware and workload.

### 2. Explain pre-norm RMSNorm in one decoder block.

**Strong answer.** The attention sublayer receives `RMSNorm(x)` and its result is
added to the unnormalized residual stream: `x + attention(norm(x))`; the FFN is
handled similarly. RMSNorm divides by root-mean-square magnitude and applies a
learned per-coordinate scale, without subtracting the mean. Pre-normalization
provides a direct residual path across layers and differs from the original
Transformer's post-norm layout.

### 3. Why does RoPE rotate queries and keys rather than simply add a position vector to token embeddings?

**Strong answer.** Position-dependent rotations make query-key dot products
depend on relative positional phase while preserving vector norms. Applying the
rotation to Q and K directly changes attention geometry. It is not equivalent to
adding the original sinusoidal vectors once at the input, even though both use
sine and cosine functions. Verify the implementation's pair ordering, frequency
schedule and cache offsets during incremental decoding.

### 4. Generation is correct without a KV cache but wrong with it. What do you inspect?

**Strong answer.** Compare cached and uncached logits token by token. Common
causes are incorrect position offsets for RoPE, mixing batch sequences, appending
keys/values along the wrong axis, or applying a causal mask as if the cached
prefix were absent. A KV cache stores past attention projections for inference;
it does not change model weights. Equivalence tests on short sequences should be
part of serving validation.

### 5. You want to reproduce a benchmark number. Why are parameter count and dataset size insufficient?

**Strong answer.** You also need tokenizer, data mixture and sampling weights,
cleaning/deduplication, token budget, optimiser schedule, precision, prompt and
scoring protocol, sampling settings and evaluation harness. For code and math,
pass@k or majority voting changes inference budget. State whether the checkpoint
is a pretrained base or instruction-tuned variant. Similar names do not guarantee
the same training history.

### 6. Distinguish activation checkpointing, model parallelism and KV caching.

**Strong answer.** Activation checkpointing saves training memory by discarding
selected intermediate activations and recomputing them during backward passes.
Model or sequence parallelism distributes training computation/state across
devices. KV caching accelerates autoregressive inference by reusing earlier
attention keys and values, while consuming memory that grows with sequence
length. They address different phases and bottlenecks.

## Project: grade a small LLaMA-style model on science questions

:::note Not from the paper

This project is an addition, to practise the chapter's ideas on a real model.

:::

**What you will build.** A small evaluation script that inspects a tiny open
model built on the LLaMA architecture, then scores it on school science
questions using the paper's own multiple-choice rule. You will end with an
accuracy number you can compare against chance and against Table 3.

**Why it matters.** Before a team picks a model for a product, such as a
homework helper, it runs exactly this kind of benchmark. Knowing how the score
is computed (§3) is what lets you trust, or question, the numbers on a
leaderboard.

**Data.** The ARC-Easy test split from the Hugging Face dataset
`allenai/ai2_arc` (config `ARC-Easy`), about 2,400 four-option questions. The
model is `HuggingFaceTB/SmolLM2-135M`, a 135M-parameter model that uses the
`LlamaForCausalLM` architecture and runs on a laptop CPU.

**Steps.**

1. **Inspect the architecture (§2.2).** Load the model and print its config.
   Find the RMSNorm epsilon, the `silu` activation of SwiGLU and the RoPE base
   (`rope_theta`). Check whether `intermediate_size / hidden_size` is close to
   $8/3\approx2.67$.
2. **Compare with Table 2 (§2.2).** Compute head width as
   `hidden_size / num_attention_heads`. Note `num_key_value_heads`: if it is
   smaller than the number of heads, the model uses grouped-query attention, a
   later change that LLaMA 1 did not have. The interactive lab above shows why
   it saves memory.
3. **Write the scorer (§3).** For each option, sum the log-probabilities of its
   tokens after the question and divide by the number of characters, as in the
   starter code.
4. **Run zero-shot (§3.1).** Score the whole test split and record accuracy.
   Random guessing gives about 25%.
5. **Try the other rule (§3).** Re-score with the BoolQ/OpenBookQA rule, dividing
   by the likelihood of the option after only `Answer:`. Compare the two
   accuracies.
6. **Try few-shot (§3).** Prepend three solved training questions to each prompt
   and measure the change.
7. **Check for leaks (§2.1).** Search a few test questions on the web. Could they
   be in a web-scraped training set? Write one paragraph on what that would mean
   for your score.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
print(model.config)

def score(context, choice):
    ids = tok(context + " " + choice, return_tensors="pt").input_ids
    n_ctx = len(tok(context).input_ids)
    with torch.no_grad():
        logp = model(ids).logits[0, :-1].log_softmax(-1)
    picked = logp[torch.arange(n_ctx - 1, ids.size(1) - 1), ids[0, n_ctx:]]
    return picked.sum().item() / len(choice)  # §3: normalise by characters

data = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
correct = 0
for row in data:
    ctx = f"Question: {row['question']}\nAnswer:"
    scores = [score(ctx, c) for c in row["choices"]["text"]]
    correct += row["choices"]["label"][scores.index(max(scores))] == row["answerKey"]
print("ARC-Easy accuracy:", correct / len(data))
```

**How you know it works.** Zero-shot accuracy should be well above the 25%
chance level; aim for at least 35%. Your two scoring rules should give
different numbers, and you should be able to explain which suits this task. For
scale, Table 3 reports 72.8% for LLaMA-7B, a model about 50 times larger.

**Stretch goals.**

- Add ARC-Challenge (config `ARC-Challenge`) and compare the gap between easy and
  challenge with Table 3's LLaMA rows.
- Run the same script on a larger LLaMA-architecture model on a free Colab GPU
  and plot accuracy against parameter count.
- Generate 200 tokens with `model.generate(..., use_cache=True)` and again with
  `use_cache=False`, and time both. This measures the KV cache described in
  §2.4's teaching note.

## Original paper

<PaperPdf slug="llama" title="LLaMA: Open and Efficient Foundation Language Models" />
