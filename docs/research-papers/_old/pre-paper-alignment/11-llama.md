---
id: paper-llama
title: "LLaMA: Open and Efficient Foundation Language Models"
sidebar_label: "11 · LLaMA"
sidebar_position: 11
slug: /research-papers/llama
description: "Data and compute trade-offs, RMSNorm, SwiGLU, RoPE, efficient execution, benchmark interpretation and a complete trainable decoder."
tags: [research-papers, deep-learning]
---

import PaperPdf from '@site/src/components/PaperPdf';
import CodeWalkthrough from '@site/src/components/viz/CodeWalkthrough';
import ResearchPaperLab from '@site/src/components/viz/ResearchPaperLab';

> **Touvron et al. · 2023** · [Read the embedded paper](#original-paper) · [Download PDF](/papers/research-papers/llama.pdf)


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

## The motivation: training cost is not the only cost

Suppose two models reach similar quality. One is larger but trained for fewer tokens; the other is smaller but trained longer. The larger one might use less training compute under a particular budget, yet cost more every time someone generates an answer.

For a widely used model, inference repeats many times. LLaMA emphasises this trade-off: spending more training on a smaller model can be worthwhile when the resulting model is cheaper to serve.

This is not “smaller always wins”. It depends on quality requirements, training budget, inference volume, hardware and sequence lengths.

## Section 2.1: data is part of the model recipe

The paper describes a mixture drawn from sources such as filtered Common Crawl, C4, GitHub, Wikipedia, books, arXiv and Stack Exchange. The mixture has sampling proportions; not every source contributes equally. Filtering and deduplication affect both quality and contamination risk.

The original family contains 7B, 13B, 33B and 65B models. The two smaller variants are trained on about one trillion tokens and the larger two on about 1.4 trillion. The exact configurations and source proportions are in the embedded paper's tables.

### Tokenisation: text must become IDs

The paper uses a SentencePiece byte-pair encoding tokenizer, with byte fallback and choices for whitespace and numbers. Subword tokens balance vocabulary size against sequence length. Byte fallback helps represent text outside common learned pieces.

A tokenizer is not interchangeable with another tokenizer just because both produce integers. Its ID-to-piece mapping must match the model's embedding rows. The character tokenizer in the code below is intentionally a separate teaching vocabulary.

## Section 2.2: what changes inside the decoder?

LLaMA remains an autoregressive Transformer. Three design choices are especially useful to understand: pre-normalisation with RMSNorm, SwiGLU feed-forward layers and rotary positions.

### RMSNorm: control magnitude without centring

For a hidden vector x of width d:

$$
\operatorname{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac1d\sum_{i=1}^{d}x_i^2+\epsilon}}.
$$

The learned vector g scales each coordinate. Unlike LayerNorm, this operation does not subtract the vector's mean. It normalises root-mean-square magnitude. Epsilon prevents division by zero or a numerically tiny denominator.

**Pre-normalisation** means a sublayer sees the normalised residual stream and adds its output back to the unnormalised stream: `x + attention(norm(x))`. Compare that with the original Transformer's `norm(x + attention(x))`.

### SwiGLU: one branch gates another

The feed-forward layer uses:

$$
\operatorname{FFN}(x)=W_2\big(\operatorname{SiLU}(W_1x)\odot W_3x\big).
$$

There are two input projections. One goes through SiLU, the smooth activation $u\sigma(u)$. Its elementwise product with the other branch forms a gated representation, then an output projection returns to the model width.

A conventional two-matrix FFN with hidden width 4d uses about $8d^2$ weights. A three-matrix gated FFN with hidden width approximately $8d/3$ has a similar count. This explains the narrower hidden dimension rather than treating it as an arbitrary constant.

### RoPE: position changes query/key geometry

Rotary position embeddings rotate pairs of query and key coordinates by angles depending on token position. For a two-coordinate pair:

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}.
$$

Different coordinate pairs use different frequencies. When rotated queries and keys form a dot product, relative position influences their compatibility. The values do not need the same rotation in this formulation.

RoPE is not an added learned position vector. It also does not guarantee useful behaviour at arbitrary sequence lengths beyond training.

## Sections 2.3–2.4: optimisation and efficient execution

The training recipe uses AdamW, a learning-rate schedule and gradient clipping. The paper also describes efficient attention and memory-saving implementation choices. These change the cost of executing the model, while the mathematical autoregressive objective remains next-token cross-entropy.

A **KV cache** stores earlier keys and values during generation so they need not be recomputed for every new token. It trades memory for computation. The complete small implementation below recomputes its short prefix for clarity; adding a cache requires positional offsets and correct handling of newly appended tokens.

![Training loss over consumed tokens](/img/research-papers/llama.png)

*Figure 1 from the original paper, PDF page 3. [Source PDF](/papers/research-papers/llama.pdf#page=3).*

This original figure plots training loss against tokens consumed. Loss decreasing over more training is evidence about prediction on the training distribution. It is not itself a downstream benchmark, and it does not show that every added token improves every capability equally.

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

## Sections 3–5: evaluation and limits

The paper evaluates commonsense reasoning, closed-book QA, reading comprehension, mathematics, code and aggregate language-understanding benchmarks. It also examines bias, toxicity and truthfulness. Comparisons use specified prompting and model settings, so a claim that one model outperforms another should remain attached to those tasks and conditions.

For example, stronger scores from a smaller model do not mean it is an instruction-tuned assistant. The main released family consists of base language models. The paper also includes the separate LLaMA-I instruction-tuning experiment explained below. Chat formatting, preference training and application tools are additional design choices.

### Open weights is a precise term

The original 2023 release made model access possible under its release conditions, which included research restrictions. “Open weights” does not automatically mean unrestricted open-source licensing, a fully released training corpus, or a complete reproducible training pipeline. Later Llama releases have different names, recipes and licences; they should not be read back into this paper.

The [authors' model repository](https://github.com/meta-llama/llama) is useful for implementation context, but its later contents are not a frozen copy of every original training detail.

## Efficient implementation, LLaMA-I and the wider evaluation

### Three memory techniques solve different problems

The paper's implementation work includes efficient attention, selective recomputation, parallelism and overlapping communication with computation. These are part of why its training recipe is practical at the reported scale.

| Technique | When it is used | What it saves or distributes |
|---|---|---|
| Memory-efficient attention | Attention computation | Avoid materialising or retaining unnecessarily large intermediate attention tensors |
| Activation checkpointing | Training | Store fewer intermediate activations and recompute selected values during backpropagation |
| Model/sequence parallelism | Training across devices | Divide parts of the model or sequence-related work across accelerators |
| KV caching | Autoregressive inference | Reuse previous tokens' attention keys and values |

An activation checkpoint is not a saved model file. A KV cache is not a new set of learned weights. The techniques operate at different points in the computation and should not be combined into the vague claim that a model “uses less memory”.

### Data preparation changes what a token represents

The paper describes source-specific cleaning: removing web boilerplate, filtering code, processing LaTeX and deduplicating books, among other steps. These decisions determine whether a token budget is spent on useful content or repeated formatting.

A source's sampling proportion and its stored size also differ. A small collection can be seen more than once while a larger one is sampled less completely. Reported token counts are therefore more informative when read with the mixture table rather than as one total number.

The tokenizer's handling of digits and byte fallback are implementation choices that affect sequence structure. Arithmetic expressed as individually tokenised digits differs from treating a whole multi-digit number as one frequent vocabulary item.

### The original paper does include an instruction-tuning experiment

The main model family is pretrained base language models, but Section 4 additionally studies **LLaMA-I**, an instruction-fine-tuned 65B model. The reported five-shot MMLU score rises from 63.4 for the base model to 68.9 for that experiment.

This is an important qualification to “LLaMA is a base-model paper”. Instruction tuning is explored, though it is not the paper's main focus and should not be confused with later Llama Chat releases. MMLU improvement measures performance on that evaluation, not complete readiness as an assistant.

### Read the benchmark families with their own rules

Commonsense tasks often compare candidate continuations. Closed-book QA generates answers without a retrieval system. RACE tests reading comprehension from supplied passages. MATH and GSM8k test mathematical answers; HumanEval and MBPP evaluate generated code using tests. MMLU aggregates subject-specific multiple-choice questions.

For code, **pass@1** concerns one sampled solution's probability of passing, while **pass@k** concerns whether a set of k samples contains a passing solution under its estimator. Majority voting in mathematics instead selects an answer supported by repeated samples. Neither should be compared with single-sample results without noting the extra inference budget.

The paper tracks some capabilities across training, showing why one loss curve is not a substitute for task evaluation. A capability can improve at a different rate from aggregate token prediction.

### Bias, truthfulness and carbon accounting

The study includes toxicity, stereotype/bias and truthfulness evaluations, while explicitly recognising that these benchmarks do not exhaust deployment risks. A high score on ordinary knowledge questions does not mean false or offensive continuations cannot occur.

Its carbon analysis combines accelerator time, power consumption, datacentre overhead and an emissions factor. A comparison using a common assumed electricity factor differs from measuring actual emissions at every training location. These assumptions matter when interpreting environmental claims just as prompting assumptions matter for benchmark claims. [Original paper, Sections 2–6](/papers/research-papers/llama.pdf).

## Summary and self-check

- [ ] I can explain why inference cost changes the preferred training trade-off.
- [ ] I can derive RMSNorm and distinguish it from LayerNorm.
- [ ] I can follow the gating operation and parameter count in SwiGLU.
- [ ] I can explain which attention tensors RoPE changes and why position matters.
- [ ] I can trace the complete decoder from token IDs to generated output.
- [ ] I can distinguish the original base models from later chat models and releases.

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


## Original paper

<PaperPdf slug="llama" title="LLaMA: Open and Efficient Foundation Language Models" />
