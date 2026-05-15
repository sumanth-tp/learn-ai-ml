---
id: t5-encoder-decoder
title: "T5 and encoder-decoder pre-training"
sidebar_label: "89 · T5"
sidebar_position: 89
slug: /theory/dnn/t5-encoder-decoder-pretraining
description: "T5's text-to-text framework, span corruption pre-training, and how encoder-decoder transformers unify NLP tasks — plus BART and the difference from encoder-only and decoder-only models."
tags: [t5, bart, encoder-decoder, seq2seq, span-corruption, transformers, deep-learning]
---

# T5 and encoder-decoder pre-training

> **TL;DR.** T5 reframes every NLP task as **string-in, string-out**. Sentiment classification? `"sst2 sentence: I loved it"` → `"positive"`. Translation? `"translate English to French: hi"` → `"salut"`. Summarization? `"summarize: <long text>"` → `"<summary>"`. The architecture is a vanilla encoder-decoder transformer; the pretraining objective is **span corruption** (mask out chunks of text and ask the decoder to regenerate them). One model, one loss, one fine-tuning recipe — for every task.

T5 (Text-to-Text Transfer Transformer) reformulated every NLP task as a seq2seq problem: classification, translation, summarization, question answering — all become "convert this input text to this output text." This unification with a single model architecture and a single training objective is what made T5 influential.

## Prerequisites

- [80 — Transformer Encoder Architecture](./80-transformer-encoder-architecture.md) — T5's encoder stack
- [82 — Cross-Attention](./82-cross-attention-in-transformers.md) — the bridge between T5's encoder and decoder
- [83 — Transformer Decoder Architecture](./83-transformer-decoder-architecture.md) — T5's 3-sublayer decoder
- [84 — Transformer Inference](./84-transformer-inference-step-by-step.md) — encoder-runs-once-then-decoder-loops applies to T5
- [85 — Transformer Training Objectives](./85-transformer-training-objectives.md) — span corruption defined
- [86 — Tokenization](./86-tokenization-bpe-wordpiece-sentencepiece.md) — T5 uses SentencePiece (different from BERT and GPT)
- [87 — BERT](./87-bert-encoder-pretraining.md) and [88 — GPT](./88-gpt-decoder-only-causal-lm.md) — contrast with the other two paradigms

## Try it interactively

- **[T5 demo on HuggingFace](https://huggingface.co/google-t5/t5-base)** — paste any input with task prefix and see the generated output
- **[FLAN-T5 demo](https://huggingface.co/google/flan-t5-large)** — instruction-tuned variant; works on natural-language prompts
- **[BART summarization demo](https://huggingface.co/facebook/bart-large-cnn)** — encoder-decoder for abstractive summarization
- **[mT5 multilingual demo](https://huggingface.co/google/mt5-base)** — same architecture, 101 languages
- **[Hugging Face fine-tuning T5 tutorial](https://huggingface.co/docs/transformers/model_doc/t5)** — official guide for fine-tuning T5 on a custom seq2seq task

## A real-world analogy

T5 is like a **universal translator** that has been trained to convert between every variety of "language": English-to-French, "raw text" to "summary", "question + passage" to "answer", "sentence" to "sentiment label". Internally, every problem becomes the same shape — read input string, write output string — and the model uses the same encoder-decoder machinery for all of them. The task prefix is the *dialect indicator* that tells the translator which conversion to perform.

## One-line definition

T5 is an encoder-decoder transformer pre-trained with span corruption (replacing random contiguous spans with sentinel tokens and training the decoder to reconstruct them), then fine-tuned for any NLP task by framing it as text generation.

![Encoder-decoder stacked architecture — T5 uses this full stack: the encoder processes the corrupted input, and the decoder generates the reconstructed spans](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Encoder-decoder models dominate seq2seq tasks (translation, summarization, structured prediction). BART powers Facebook's summarization and translation. mT5 is the standard multilingual encoder-decoder. T5-style models are still widely used in production for tasks where the output has a complex, multi-token structure. Understanding T5 completes the trifecta of transformer architectures: BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder).

## The text-to-text framework

T5's central idea: every NLP task is a text-to-text problem.

| Task | Input | Output |
|---|---|---|
| Sentiment classification | `sentiment: The movie was great.` | `positive` |
| Translation | `translate English to French: Hello world` | `Bonjour le monde` |
| Summarization | `summarize: Long article text...` | `Short summary` |
| QA | `question: What is Paris? context: Paris is the capital of France.` | `the capital of France` |
| NER | `recognize entities: Barack Obama was born in Hawaii.` | `person: Barack Obama location: Hawaii` |

A single model, a single objective, a single fine-tuning procedure. The task prefix tells the model what to do.

## Architecture

T5 uses the original encoder-decoder transformer architecture (note 80 + 83) with minor modifications:
- **Relative position biases** instead of absolute or sinusoidal positional encodings — each attention head learns position biases that depend on the relative offset between tokens, not absolute positions
- No bias terms in most layers
- Pre-norm (LayerNorm before each sublayer)

| Model | Parameters | $d_{\text{model}}$ | Layers (enc/dec) | Heads |
|---|---|---|---|---|
| T5-small | 60M | 512 | 6/6 | 8 |
| T5-base | 220M | 768 | 12/12 | 12 |
| T5-large | 770M | 1024 | 24/24 | 16 |
| T5-XL | 3B | 2048 | 24/24 | 32 |
| T5-XXL | 11B | 4096 | 24/24 | 64 |
| T5-v1.1 / FLAN-T5 | — | — | — | Improved versions |

![T5 unifies every NLP task as a text-to-text problem — translation, classification, summarization, QA all use the same model and the same loss](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s640/image3.gif)
*Source: [Google Research — Exploring Transfer Learning with T5](https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html)*

## Pre-training: span corruption

T5's pre-training task replaces random contiguous spans with unique sentinel tokens:

1. Sample spans from the input with mean length 3, targeting ~15% corruption
2. Replace each span with a unique sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...)
3. Train the decoder to generate the original spans, each preceded by its sentinel

**Example**:
- Original: `The quick brown fox jumps over the lazy dog`
- Corrupted input: `The quick <extra_id_0> jumps over <extra_id_1> dog`
- Target: `<extra_id_0> brown fox <extra_id_1> the lazy`

The decoder must produce all corrupted spans in order, separated by their sentinel tokens, and end with `<extra_id_N>` as EOS.

```mermaid
flowchart LR
    orig["Original text\n'the quick brown fox jumps'"]
    corrupt["Corrupted input\n'the <e_0> jumps <e_1>'"]
    enc_out["Encoder output\n(bidirectional, full context)"]
    decoder["Decoder\n(autoregressively generates spans)"]
    target["Target\n'<e_0> quick brown fox <e_1>'"]

    orig --> corrupt --> enc_out --> decoder --> target
```

## BART: denoising with a broader set of corruptions

BART (Lewis et al., 2020) is another encoder-decoder model with a similar denoising approach but uses a wider range of corruption strategies:
- **Token masking**: replace tokens with `[MASK]` (like BERT)
- **Token deletion**: remove tokens (model must determine what's missing)
- **Text infilling**: replace a span of tokens with a single `[MASK]`
- **Sentence permutation**: shuffle sentence order
- **Document rotation**: rotate the document to start at a random token

The decoder always reconstructs the original uncorrupted document (not just the corrupted spans). BART excels at abstractive summarization and text generation tasks.

## Python code: T5 with HuggingFace

```python
# pip install transformers sentencepiece
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.eval()


# ============================================================
# Task 1: Translation (English → French)
# ============================================================
def translate(text: str) -> str:
    input_ids = tokenizer.encode(
        f"translate English to French: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    output_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,           # beam search for translation quality
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


print("=== Translation ===")
print(translate("Hello, how are you today?"))


# ============================================================
# Task 2: Summarization
# ============================================================
def summarize(text: str) -> str:
    input_ids = tokenizer.encode(
        f"summarize: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    output_ids = model.generate(
        input_ids,
        max_length=64,
        min_length=10,
        num_beams=4,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


long_text = (
    "The transformer architecture, introduced in 2017, replaced recurrent neural networks "
    "for sequence modeling. It uses attention mechanisms to process all tokens in parallel, "
    "enabling more efficient training on large datasets. BERT, GPT, and T5 are all based "
    "on transformers."
)
print(f"\n=== Summarization ===")
print(summarize(long_text))


# ============================================================
# Task 3: Classification as text generation
# ============================================================
def classify_sentiment(text: str) -> str:
    input_ids = tokenizer.encode(
        f"sst2 sentence: {text}",   # T5 uses "sst2 sentence:" for SST-2
        return_tensors="pt",
    )
    output_ids = model.generate(input_ids, max_length=10)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


print(f"\n=== Sentiment classification (text-to-text) ===")
print(classify_sentiment("The movie was absolutely fantastic!"))
print(classify_sentiment("I really did not enjoy this experience."))


# ============================================================
# Forward pass: understanding encoder-decoder structure
# ============================================================
model.train()
input_text = "translate English to German: I love transformers."
target_text = "Ich liebe Transformatoren."

input_ids = tokenizer.encode(input_text, return_tensors="pt")
# T5 uses -100 for padding in labels (ignored in loss computation)
labels = tokenizer.encode(target_text, return_tensors="pt")

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"\n=== Forward pass ===")
print(f"Input shape:      {input_ids.shape}")
print(f"Encoder output shape: {outputs.encoder_last_hidden_state.shape}")
print(f"Decoder logits:   {logits.shape}")   # (1, target_len, vocab_size)
print(f"Training loss:    {loss.item():.4f}")


# ============================================================
# BART for summarization (better than T5 for abstractive tasks)
# ============================================================
from transformers import BartForConditionalGeneration, BartTokenizer

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_model.eval()

article = (
    "The transformer architecture has revolutionized artificial intelligence. "
    "First introduced in 2017 by Vaswani et al., it replaced recurrent architectures "
    "with attention mechanisms, enabling parallel training and better long-range modeling."
)

inputs = bart_tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
with torch.no_grad():
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=60,
        min_length=20,
        num_beams=4,
        length_penalty=2.0,
    )

print(f"\n=== BART Summary ===")
print(bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

### Try it yourself: experiments

| Question | Try this |
|----------|----------|
| What if you skip the task prefix? | Send raw text to T5 with no prefix — output is gibberish or echoes input |
| Compare beam search vs greedy | Set `num_beams=1` vs `num_beams=4` — beam search wins on translation by 1–3 BLEU |
| Can a single fine-tuning teach two tasks? | Train T5 on a mixed batch (translate + summarize) — emerges multi-task capability |
| Inspect cross-attention | Pass `output_attentions=True`, plot `cross_attentions[0][0]` heatmap |
| FLAN-T5 vs T5 zero-shot | Same instruction-style prompt to both — FLAN-T5 follows it; vanilla T5 doesn't |

## T5 vs. BERT vs. GPT

| Property | T5 | BERT | GPT |
|---|---|---|---|
| Architecture | Encoder-decoder | Encoder-only | Decoder-only |
| Pre-training | Span corruption | MLM | CLM |
| Context | Encoder: bidirectional; Decoder: causal | Bidirectional | Causal |
| Best for | Seq2seq, structured generation | Classification, NER, embedding | Generation, reasoning, prompting |
| Fine-tuning | Text-to-text format for all tasks | Task-specific head | Fine-tune or prompt |
| Generation | Yes (decoder) | No | Yes |
| Understanding | Yes (encoder) | Yes | Limited (unidirectional) |
| Production use | Summarization, translation | Semantic search, classification | Chat, code, long-form generation |

## FLAN-T5: instruction-tuned T5

FLAN-T5 (Wei et al., 2022) fine-tunes T5 on over 1,000 NLP tasks formatted as natural language instructions:

```
Input: "Please classify the sentiment of the following review as positive or negative:
The acting was superb but the plot was confusing."
Output: "negative"
```

FLAN-T5 follows instructions zero-shot far better than vanilla T5 — the key insight is that fine-tuning on many tasks in instruction format improves generalization to new tasks.

## Cross-references

- **Prerequisite:** [82 — Cross-Attention](./82-cross-attention-in-transformers.md) — what bridges T5's encoder and decoder
- **Prerequisite:** [83 — Decoder Architecture](./83-transformer-decoder-architecture.md) — the 3-sublayer decoder T5 uses
- **Prerequisite:** [85 — Training Objectives (Span Corruption)](./85-transformer-training-objectives.md) — T5's pretraining objective
- **Prerequisite:** [86 — Tokenization](./86-tokenization-bpe-wordpiece-sentencepiece.md) — T5 uses SentencePiece
- **Related:** [87 — BERT (Encoder-Only)](./87-bert-encoder-pretraining.md), [88 — GPT (Decoder-Only)](./88-gpt-decoder-only-causal-lm.md) — the other two paradigms
- **Follow-up:** [90 — Fine-Tuning](./90-fine-tuning-transformers.md) — adapting T5 to new tasks via the text-to-text format

## Interview questions

<details>
<summary>What is the advantage of T5's text-to-text framework?</summary>

Unification: every NLP task uses the same model, the same loss (cross-entropy on the target sequence), and the same fine-tuning procedure. There is no need for task-specific architectures (classification heads vs. span extractors vs. sequence generators). A single model can be fine-tuned for sentiment, translation, summarization, and QA by just changing the prompt prefix. This simplifies the entire pipeline and makes multi-task fine-tuning straightforward.
</details>

<details>
<summary>How does span corruption differ from BERT's MLM?</summary>

MLM masks individual tokens; span corruption masks contiguous spans of multiple tokens. MLM is bidirectional — the masked token prediction is embedded in the encoder's forward pass. Span corruption is seq2seq — the encoder reads the corrupted input, and the decoder generates the original spans autoregressively. Span corruption is more suitable for training an encoder-decoder: the decoder gets experience generating text from encoder context, not just predicting a single token.
</details>

<details>
<summary>Why would you choose T5 over GPT for summarization?</summary>

T5's encoder can process the full source document bidirectionally, building a rich representation that the decoder can attend to via cross-attention. GPT processes everything causally — the model must hold the entire document in its left context and generate the summary afterward. For long documents where the summary depends on understanding the whole document (not just its beginning), the encoder-decoder setup is more naturally suited. In practice, the difference has narrowed as GPT models have gotten much larger context windows.
</details>

<details>
<summary>Scenario: a team fine-tunes T5 for SQL generation. The loss drops quickly but generated SQL is syntactically valid only ~60% of the time. What's likely wrong?</summary>

Cross-entropy loss measures token-level likelihood, not syntactic validity. The model can have low loss while still producing token sequences that the next-token distribution accepts but the SQL parser doesn't (mismatched quotes, missing FROM, hallucinated table names).

Typical fixes:

1. **Constrained decoding**: at generation time, mask out tokens that would lead to invalid syntax (e.g., a grammar-aware decoder using something like Picard or LMQL). The model can only generate syntactically valid completions.
2. **Add a structural loss**: post-process generations and add reward/penalty terms during finetuning (e.g., DPO with parser-validation as the reward signal).
3. **Schema linking**: include the target database schema in the prompt so table/column names come from a bounded set.
4. **Better data**: ensure training examples are syntactically valid; even small amounts of malformed SQL in training pushes the model toward malformed outputs.

The lesson: when output has formal structure (SQL, JSON, regex, code), token-level loss is insufficient. Either constrain decoding, validate at training time, or use a structured-prediction approach.
</details>

<details>
<summary>Why does T5 use relative position biases instead of sinusoidal or learned absolute positions?</summary>

Relative position bias adds a *learned scalar* to the attention score based on the relative offset between query and key positions: $\text{score}_{i,j} = q_i \cdot k_j + b_{i-j}$. Each head learns its own bias table.

Advantages:

1. **Generalizes across positions**: the bias for "10 tokens apart" applies whether the pair is at (3, 13) or (200, 210). Learned absolute embeddings can't share information across absolute positions.
2. **Easier to extrapolate** to longer sequences (with caveats — T5 also has a max offset beyond which biases are bucketed).
3. **No extra positional encoding step** — positions are baked into attention directly.

The trade-off: more attention complexity (small lookup per pair), and the offset-bucketing scheme imposes a soft maximum effective range. T5 chose this design partly to enable longer-context generalization than learned absolute positions allow, which mattered for tasks like summarization with variable input length.

Modern decoder-only LLMs adopted *rotary* position embeddings (RoPE) for similar reasons but with a different mechanism. T5's choice was an early step in the same direction.
</details>

<details>
<summary>Scenario: BART beats T5 on CNN/DailyMail summarization but loses on translation. Why?</summary>

The differences trace to the pretraining objective:

- **BART denoising**: includes sentence permutation and text infilling. The model is heavily trained to *reconstruct fluent natural language* from disordered or partial inputs. This is essentially summarization-by-paraphrase practice.
- **T5 span corruption**: contiguous spans of fixed mean length 3. More structured, less fluent-reconstruction practice.

For **summarization**: BART's pretraining is more directly aligned. The model has done millions of "rewrite a corrupted paragraph as a fluent paragraph" exercises, which is what abstractive summarization requires.

For **translation**: span corruption is closer aligned because translation is a more structured token-by-token mapping with less rewriting. BART's heavier denoising bias makes it slightly more prone to paraphrasing where translation needs literal precision.

Lesson: pretraining objective shapes downstream strengths. Two architecturally similar encoder-decoders can have noticeably different downstream profiles depending on the corruption noise distribution.
</details>

<details>
<summary>What's the difference between FLAN-T5 and vanilla T5, and why does it matter so much?</summary>

Vanilla T5 was pretrained with span corruption only. Out of the box, it follows the original task prefixes (`summarize:`, `translate English to French:`) but fails on natural-language instructions ("Please write a short summary of this article").

FLAN-T5 takes vanilla T5 and *instruction-tunes* it on 1,800+ tasks formatted as natural-language prompts: "Tell me if this review is positive or negative", "Translate to French", "Answer the following question". The model learns a meta-skill: read an instruction and follow it.

The effect is dramatic: FLAN-T5 zero-shots on tasks vanilla T5 cannot do at all, while being the same model architecturally. The training data (instruction format coverage) matters more than parameter count for instruction following — FLAN-T5-base (250M params) can beat vanilla T5-XXL (11B) on instruction-following benchmarks.

This was an early demonstration of what later became the standard recipe for all modern LLMs: pretraining → instruction tuning → RLHF/DPO. The objective trinity that defines ChatGPT-class systems.
</details>

<details>
<summary>Scenario: you must build a system that answers questions over a 1M-token corpus. T5 has a 512-token context. How do you proceed?</summary>

T5's 512-token limit is hard — you cannot just feed the whole corpus. The standard architecture is **retrieval-augmented generation (RAG)**:

1. **Chunk** the corpus into ~200-token passages with overlap.
2. **Embed** each chunk with a sentence encoder (Sentence-BERT, E5, BGE).
3. **At query time**, embed the question and retrieve top-K most similar chunks (typically K=3-10).
4. **Feed retrieved chunks + question** to T5: "Question: ... Context: [chunk 1] [chunk 2] [chunk 3] Answer:"
5. **Decoder generates** the answer conditioned on retrieved evidence.

This is what Fusion-in-Decoder (Izacard & Grave, 2021) does specifically with T5 — pass each retrieved passage through the encoder separately, then concatenate encoder outputs for the decoder's cross-attention. Avoids the 512-token bottleneck on the encoder side.

Modern alternatives: long-context LLMs (Claude 200K, GPT-4 128K) can sometimes skip retrieval, but for any corpus larger than the context window, retrieval is still required. T5-based FiD remains competitive for tasks with diverse, hard-to-rank evidence (open-domain QA, fact-checking).
</details>

<details>
<summary>Why doesn't T5 work well with in-context learning the way GPT does?</summary>

T5 was pretrained on span corruption, not on continuation. Few-shot prompting works for GPT because GPT was trained to continue text — showing it a few examples followed by a query naturally leads to a completion in the same pattern.

T5 was trained to fill in masked spans, not continue prompts. Giving T5 a few-shot prompt produces output that often:

- Tries to "fill in" the prompt rather than continue it
- Repeats the structure literally
- Generates sentinel tokens (`<extra_id_0>`) when no mask exists

FLAN-T5 fixes this by training on instruction format, but vanilla T5's pretraining mismatch with few-shot prompting is structural.

This is why decoder-only models won the LLM race for in-context learning — their pretraining objective is *natively* compatible with few-shot prompting, while encoder-decoders need additional instruction tuning to expose this capability.
</details>

<details>
<summary>Why does T5 use SentencePiece with 32K vocab instead of WordPiece (like BERT) or byte-level BPE (like GPT)?</summary>

T5 explicitly targeted multilingual capability (mT5, with 101 languages). SentencePiece is the standard choice because:

1. **Language-agnostic preprocessing**: SentencePiece treats text as raw character streams without language-specific pre-tokenization. Critical for languages without word boundaries (Chinese, Japanese, Thai) and agglutinative languages (Finnish, Turkish).
2. **Explicit whitespace handling**: spaces are represented as `▁` (a real token character), so detokenization is unambiguous. WordPiece has implicit space handling that can lose information.
3. **Vocab size flexibility**: T5 chose 32K to keep the embedding table small while still covering common subwords in English. mT5 uses 250K to span 101 languages.

For pure English-language work, WordPiece and SentencePiece have similar quality. For multilingual or character-rich languages, SentencePiece (especially with unigram language modeling, the SentencePiece default) tends to win.

The choice also reflects era: T5 (2019) was the first major model to deliberately design for non-English-first deployment, anticipating multilingual mT5.
</details>

<details>
<summary>Scenario: a team uses T5-large for production summarization. Latency is 8 seconds per request. What are the realistic optimizations?</summary>

T5-large is 770M parameters; 8s/request on CPU is plausible but slow. Optimization layers in priority order:

1. **Quantize** (INT8 dynamic or static): 2-4× speedup with minimal quality loss. Run with `torch.quantization` or ONNX runtime INT8.
2. **Switch to T5-base** (220M params): 3-4× speedup. Quality drop for summarization is typically small if you fine-tuned T5-large on your data — just re-fine-tune T5-base instead.
3. **Use DistilBART** or another distilled summarizer: pre-distilled from BART or T5 for summarization specifically.
4. **GPU inference**: a T4 GPU does T5-large in ~200ms; an A100 in ~50ms. 30-100× faster than CPU.
5. **Caching encoder output** if you summarize the same documents repeatedly: encoder runs once, decoder generates new summaries from cache.
6. **Reduce beam size**: greedy or beam=2 instead of beam=4. ~2× speedup with small ROUGE drop on most tasks.

For a high-traffic production system: GPU + INT8 + beam=2 + T5-base brings 8s → 50ms. The trade-offs are real (quality drops noticeable on long, complex articles) but usually acceptable for chatbot-style summarization.
</details>

<details>
<summary>Encoder-decoder vs decoder-only at the same parameter budget: when does encoder-decoder still win?</summary>

For *general-purpose chat and reasoning*: decoder-only wins (this is why GPT-4, Claude, Gemini are all decoder-only).

For these specific cases, encoder-decoder remains competitive:

1. **Translation with parallel corpora**: encoder-decoder is naturally suited to "this exact source → this exact target" mapping. NMT systems still often use encoder-decoder (MarianMT, NLLB).
2. **Long input → short output asymmetry**: summarizing a 4000-token document to 100 tokens, the encoder can process input *in parallel* while the decoder loops only 100 times. A decoder-only model loops 4100 times. The asymmetry favors encoder-decoder.
3. **Structured input → structured output**: code editing (long input + diff output), table-to-text, structured QA — tasks where input is genuinely a separate object from output.
4. **Fusion-in-Decoder style retrieval QA**: process retrieved passages through encoder separately, concatenate in decoder. Hard to do as cleanly in decoder-only.

At small scales (under 1B params), encoder-decoder often beats decoder-only because the inductive bias matches the task shape. At large scales, decoder-only's flexibility and scaling efficiency dominate. The crossover depends on task and data; rule of thumb is "below 1B, try both; above 7B, decoder-only is the safe choice."
</details>

<details>
<summary>What is the "no_repeat_ngram_size" parameter and why is it important for T5 summarization?</summary>

T5 (and any seq2seq model trained on summarization) is prone to repetition — generating the same n-gram multiple times in a single output. This is a known failure mode of beam search and is exacerbated by the cross-entropy training objective, which doesn't directly penalize repetition.

`no_repeat_ngram_size=2` blocks any 2-gram from appearing more than once in the generation. If the model wants to emit "the cat" and "the cat" already appeared, it's forced to choose a different next token.

This is a *post-hoc fix*, not a principled solution. Better alternatives:

- **Top-p sampling** instead of beam search (less prone to repetition).
- **Penalty-based decoding** (CTRL-style frequency penalty).
- **Train with unlikelihood loss** (Welleck et al. 2020) that directly penalizes repetition during training.
- **Reinforcement learning from human feedback** to penalize repetitive generations.

For production summarization, n-gram blocking is the cheapest fix and usually sufficient. For higher-quality long-form generation, train-time interventions are needed.
</details>

<details>
<summary>If T5 unifies all NLP tasks, why didn't it become the dominant architecture? Why did decoder-only win instead?</summary>

T5 was the right idea at the wrong time. Several forces favored decoder-only:

1. **Simpler scaling**: one stack to optimize, one objective, one inference path. Easier to scale to 175B and beyond.
2. **In-context learning** is *native* to CLM but requires explicit instruction tuning for span corruption. GPT-3's zero-shot/few-shot demos (2020) shifted the field's expectations.
3. **Denser training signal**: CLM uses every position; span corruption uses ~15%. More data efficient.
4. **Unification still possible without T5's architecture**: instruction-tuned GPT (InstructGPT, ChatGPT) can do everything T5 could, while also doing chat naturally.
5. **Engineering momentum**: by 2022, the entire LLM ecosystem (RLHF, alignment, agents, tool use, RAG) was built around decoder-only. T5 lost the network effects race.

T5's text-to-text *philosophy* won — it's how every modern LLM interface works. T5's *architecture* lost. The lesson is that good ideas don't always come packaged in the form that survives at scale; the best ideas migrate to whatever architecture wins.

Where T5 still wins: smaller, task-specific seq2seq models in production. T5-base on a fine-tuned summarization task often outperforms a 7B decoder-only model that hasn't been fine-tuned, at 1/30th the cost.
</details>

## Points to remember

- T5 reframes every NLP task as text-to-text. The task prefix (`summarize:`, `translate English to French:`) is the activation signal.
- The architecture is the standard encoder-decoder transformer with three modifications: relative position biases, no bias terms, pre-norm.
- Span corruption pretraining masks contiguous spans (mean length 3, ~15% noise density) and trains the decoder to reconstruct them.
- T5 uses SentencePiece tokenization (vs WordPiece for BERT, byte-level BPE for GPT) for language-agnostic preprocessing.
- BART is a related encoder-decoder with more aggressive corruption (sentence permutation, deletion, infilling); it tends to beat T5 on abstractive summarization.
- FLAN-T5 = T5 + instruction tuning on 1,800 tasks. Dramatic zero-shot improvement at no architectural cost.
- T5 is not designed for in-context learning. Use FLAN-T5 or decoder-only models for few-shot prompting.
- Cross-entropy loss doesn't measure structural validity (SQL, JSON, code). Use constrained decoding or grammar-aware training for structured outputs.
- For corpus larger than 512 tokens, use retrieval-augmented generation (Fusion-in-Decoder is the T5-native pattern).
- Encoder-decoder still wins for tasks with long-input-short-output asymmetry, structured-input/structured-output, and parallel translation corpora.
- Decoder-only won the LLM race for general-purpose chat because of simpler scaling, native in-context learning, and denser training signal — but T5's text-to-text *philosophy* lives on in every instruction-tuned model.

## Common mistakes

- Forgetting the task prefix — T5 requires prefixes like "translate English to German:" or "summarize:" to activate task-specific behavior
- Using T5 with too long inputs without truncation — T5-base has max length 512; exceeding this causes errors
- Setting `labels` to raw token IDs including padding (should be -100 for padding) — padding positions should not contribute to loss
- Using greedy decoding for translation — beam search is standard for seq2seq tasks that require higher quality output

## Final takeaway

T5 unified NLP into a single text-to-text framework: every task is a string-in, string-out problem. Span corruption pre-training teaches the encoder-decoder to understand and generate text simultaneously. BART extends this with broader corruption strategies, excelling at abstractive summarization. Together with BERT (understanding) and GPT (generation), T5 completes the three canonical transformer paradigms that underpin modern NLP infrastructure.

## Further reading

- [arXiv: T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683) — the original paper unifying NLP tasks as text-to-text
- [arXiv: BART (Lewis et al. 2020)](https://arxiv.org/abs/1910.13461) — encoder-decoder with broader denoising; the summarization workhorse
- [arXiv: FLAN-T5 (Wei et al. 2022)](https://arxiv.org/abs/2210.11416) — instruction-tuning T5 on 1,800 tasks
- [arXiv: mT5 (Xue et al. 2021)](https://arxiv.org/abs/2010.11934) — multilingual T5 covering 101 languages
- [arXiv: UL2 (Tay et al. 2022)](https://arxiv.org/abs/2205.05131) — unified pretraining mixing CLM, MLM, and span corruption
- [arXiv: Fusion-in-Decoder (Izacard & Grave 2021)](https://arxiv.org/abs/2007.01282) — T5-style retrieval-augmented QA
- [arXiv: PICARD (Scholak et al. 2021)](https://arxiv.org/abs/2109.05093) — constrained decoding for SQL generation from T5
- [Google Research — Exploring Transfer Learning with T5](https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html) — the official launch blog with diagrams
- [Hugging Face T5 docs](https://huggingface.co/docs/transformers/model_doc/t5) — code examples and fine-tuning tutorials
- [Hugging Face — BART vs T5 comparison](https://huggingface.co/blog/encoder-decoder) — which encoder-decoder model to pick
- [LMSYS — Why Decoder-Only LLMs](https://lmsys.org/blog/2023-05-25-leaderboard/) — discussion of why decoder-only displaced encoder-decoder for general LLMs

## References

- Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
- Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. ACL.
- Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners (FLAN). ICLR.
