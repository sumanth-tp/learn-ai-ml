---
id: tokenization
title: "Tokenization: BPE, WordPiece, and SentencePiece"
sidebar_label: "86 · Tokenization"
sidebar_position: 86
slug: /theory/dnn/tokenization-bpe-wordpiece-sentencepiece
description: "How text is converted to token IDs before entering a transformer: byte-pair encoding, WordPiece, and SentencePiece — with algorithms, examples, and the vocabulary trade-offs."
tags: [tokenization, bpe, wordpiece, sentencepiece, transformers, nlp]
---

# Tokenization: BPE, WordPiece, and SentencePiece

> **TL;DR.** Transformers don't see text — they see integers. The tokenizer is the bridge: it splits text into **subword units** (somewhere between letters and full words) and maps them to vocabulary IDs. The three main flavors are **BPE** (greedy, frequency-based — used in GPT-2/3/4, LLaMA), **WordPiece** (likelihood-based — used in BERT), and **SentencePiece** (language-agnostic framework — used in T5, mT5). The choice of tokenizer determines context-window efficiency, multilingual coverage, and even how models read code.

Before a transformer sees any text, the text must be converted to integers. Tokenization — splitting text into subword units and mapping them to vocabulary IDs — is the first and often underappreciated step in every NLP pipeline. The choice of tokenizer affects vocabulary size, out-of-vocabulary handling, multilingual coverage, and even model performance.

## Prerequisites

- [80 — Transformer Encoder Architecture](./80-transformer-encoder-architecture.md) — token embeddings feed the encoder stack
- [78 — Positional Encoding](./78-positional-encoding-in-transformers.md) — what gets added to token embeddings *after* tokenization
- [85 — Transformer Training Objectives](./85-transformer-training-objectives.md) — MLM, CLM, and span corruption all operate on tokens, not characters

## Try it interactively

- **[OpenAI Tokenizer](https://platform.openai.com/tokenizer)** — paste any text and see how GPT-3.5/4 tokenizes it (live, in the browser)
- **[tiktokenizer.vercel.app](https://tiktokenizer.vercel.app/)** — compare GPT-3.5, GPT-4, Claude, and other tokenizers side by side on the same text
- **[Hugging Face Tokenizers playground](https://huggingface.co/docs/tokenizers)** — official docs with runnable examples
- **[Karpathy — Let's build the GPT Tokenizer (YouTube)](https://www.youtube.com/watch?v=zduSFxRajkE)** — implements BPE from scratch in 2 hours, explaining every quirk
- **[OpenAI tiktoken library](https://github.com/openai/tiktoken)** — Python library for fast tokenization with all GPT tokenizers

## A real-world analogy

Tokenization is like **filing books in a library**. You have three options:

- **Word-level**: one drawer per book title. Beautiful organization, but you need a *vast* cabinet — and any new title (a typo, a foreign word, a brand name) won't fit.
- **Character-level**: one drawer per individual letter. Tiny cabinet, but reading a book takes 80,000 drawer trips.
- **Subword (BPE/WordPiece)**: drawers for *common stems and endings* — "transform", "ization", "running", "##ing". Most words fit in 1–2 drawers; rare words spill into 3–4. The cabinet stays a manageable 30k–100k drawers, and *any* string can be filed (worst case, individual letters/bytes).

That third strategy is what every modern transformer uses.

## One-line definition

Tokenization converts raw text into a sequence of integer token IDs by splitting text into subword units (neither full words nor individual characters) that balance vocabulary size, coverage, and morphological consistency.

![Transformer input pipeline — raw text is tokenized into subword IDs, then each ID is looked up in an embedding table before being fed into the first encoder/decoder block](https://jalammar.github.io/images/t/embeddings.png)
*Source: [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

## Why this topic matters

Tokenization determines the input representation of every transformer. A poor tokenizer leads to:
- Rare words split into many tokens → longer sequences → more compute
- OOV (out-of-vocabulary) characters silently dropped or mangled
- Language-specific biases (e.g., English text tokenized much more efficiently than Chinese)

Industry LLMs (GPT-4, LLaMA, BERT) all use subword tokenizers. Understanding them is essential for debugging models, understanding context window limits, and building multilingual systems.

## Why not word-level or character-level?

| Approach | Vocabulary size | OOV problem | Sequence length | Used in |
|---|---|---|---|---|
| Word-level | Large (50k–500k+) | Yes — unseen words | Shortest | Early NLP |
| Character-level | Tiny (100–300) | None — all chars known | Very long | Some RNNs |
| Subword | Medium (30k–100k) | Rare — most covered | Medium | All modern LLMs |

Subword tokenization is the compromise:
- Small enough vocabulary to train on (embedding table is $\text{vocab\_size} \times d_{\text{model}}$)
- Complete coverage — every string can be encoded using individual bytes/chars as a fallback
- Short enough sequences — common words are one token, rare words are a few

## Algorithm 1: Byte Pair Encoding (BPE)

Used in: **GPT-2, GPT-3, GPT-4, LLaMA, Mistral, RoBERTa**

### Training algorithm

1. Start with character-level vocabulary (including special tokens)
2. Count frequencies of all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat steps 2–3 for $V$ merge operations (where $V$ = desired vocabulary size - initial size)

Each merge rule is stored; at inference, the same merge rules are applied in the same order.

**Example**:

Starting corpus: `"low"×5, "lower"×2, "newest"×6, "widest"×3`

Initial characters: `l o w _ , e r , n w e s t , w i d e s t`

Iteration 1: most frequent pair is `(e, s)` → merge to `es`
Iteration 2: most frequent pair is `(es, t)` → merge to `est`
Iteration 3: most frequent pair is `(l, o)` → merge to `lo`
...

After all merges, `"newest"` might encode as `["new", "est"]` and `"lower"` as `["low", "er"]`.

### Byte-level BPE

GPT-2 introduced byte-level BPE: the initial alphabet is the 256 possible byte values (not Unicode characters). This guarantees that any input can be tokenized — unknown Unicode characters fall back to individual bytes. No `[UNK]` token needed.

![Byte-pair encoding — iteratively merging the most frequent adjacent symbol pair into a new token, building a vocabulary from characters up to common subwords](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course/en/chapter6/BPE.png)
*Source: [Hugging Face — Byte-Pair Encoding tutorial](https://huggingface.co/learn/nlp-course/chapter6/5)*

## Algorithm 2: WordPiece

Used in: **BERT, DistilBERT, ALBERT, Electra**

WordPiece is similar to BPE but uses a **likelihood-based** merge criterion instead of frequency:

$$
\text{score}(a, b) = \frac{P(ab)}{P(a) \cdot P(b)}
$$

It merges the pair that maximizes the language model log-likelihood. In practice, this tends to preserve more morphological structure than BPE.

**Key difference from BPE**: WordPiece only merges pairs that improve the language model likelihood. Continuation pieces are prefixed with `##`:

```
"tokenization" → ["token", "##ization"]
"unbelievable" → ["un", "##believable"]
"GPU"          → ["G", "##P", "##U"]
```

## Algorithm 3: SentencePiece

Used in: **T5, LLaMA (via BPE), ALBERT, XLNet, mT5**

SentencePiece is a **language-agnostic** framework that:
1. Treats the input as a raw stream of Unicode characters (no pre-tokenization)
2. Applies BPE or unigram language model on the raw character stream
3. Represents spaces explicitly as a special character (▁)

**Key advantage**: works for any language without language-specific pre-tokenization. Especially important for Chinese, Japanese, Thai (no spaces between words) and agglutinative languages (Finnish, Turkish).

```
"Hello world" → ["▁Hello", "▁world"]
"world"       → ["▁world"]    # ▁ = beginning of word
"worldwide"   → ["▁world", "wide"]
```

## Tokenizer components

Every tokenizer has four parts:

| Component | Description | Example (BERT) |
|---|---|---|
| Vocabulary | Map from token string → integer ID | `{"[CLS]": 101, "cat": 4937, ...}` |
| Special tokens | Reserved for structural use | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]` |
| Encoding rules | How to split text → tokens | WordPiece merge rules |
| Decoding rules | How to convert tokens back → text | Strip `##`, join pieces |

## Token count examples

```
Sentence: "The transformer architecture revolutionized natural language processing."

BERT (WordPiece, vocab=30522):
["The", "transform", "##er", "architecture", "revolution", "##ized",
 "natural", "language", "processing", "."]
→ 10 tokens

GPT-2 (BPE, vocab=50257):
["The", " transformer", " architecture", " revolution", "ized",
 " natural", " language", " processing", "."]
→ 9 tokens

GPT-4 (cl100k, vocab=100257):
["The", " transformer", " architecture", " revolutionized",
 " natural", " language", " processing", "."]
→ 8 tokens  (larger vocab = fewer tokens per sentence)
```

## Python code: using HuggingFace tokenizers

```python
# pip install transformers
from transformers import (
    BertTokenizer, GPT2Tokenizer, AutoTokenizer
)
import torch


# ============================================================
# BERT WordPiece tokenizer
# ============================================================
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")

text = "The transformer architecture revolutionized NLP."
tokens = bert_tok.tokenize(text)
ids = bert_tok.encode(text, add_special_tokens=True)

print("=== BERT WordPiece ===")
print(f"Tokens:    {tokens}")
print(f"Token IDs: {ids}")
print(f"Decoded:   {bert_tok.decode(ids)}")
# Note: ##-prefixed tokens are continuation pieces

# Batch encoding with padding + attention mask
batch = ["The cat sat on the mat.", "Transformers changed everything."]
encoded = bert_tok(
    batch,
    padding=True,          # pad to same length
    truncation=True,       # truncate to max_length
    max_length=128,
    return_tensors="pt",   # return PyTorch tensors
)
print(f"\nBatch input_ids shape:      {encoded['input_ids'].shape}")   # (2, 9)
print(f"Batch attention_mask shape: {encoded['attention_mask'].shape}") # (2, 9)
# attention_mask = 1 for real tokens, 0 for padding


# ============================================================
# GPT-2 BPE tokenizer
# ============================================================
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token   # GPT-2 has no pad token by default

tokens_gpt2 = gpt2_tok.tokenize(text)
ids_gpt2 = gpt2_tok.encode(text)

print("\n=== GPT-2 BPE ===")
print(f"Tokens:    {tokens_gpt2}")
print(f"Token IDs: {ids_gpt2}")
print(f"Decoded:   {gpt2_tok.decode(ids_gpt2)}")
# Note: spaces are included in the token, no ## notation


# ============================================================
# Inspect the vocabulary trade-offs
# ============================================================
print(f"\n=== Vocabulary sizes ===")
print(f"BERT vocab size:  {bert_tok.vocab_size}")   # 30,522
print(f"GPT-2 vocab size: {gpt2_tok.vocab_size}")   # 50,257

# Token count comparison on the same text
long_text = "The quick brown fox jumps over the lazy dog. " * 20
bert_len = len(bert_tok.encode(long_text))
gpt2_len = len(gpt2_tok.encode(long_text))
print(f"\n=== Token count for 20x the pangram ===")
print(f"BERT:  {bert_len} tokens")
print(f"GPT-2: {gpt2_len} tokens")
# Larger vocab → fewer tokens → shorter sequences


# ============================================================
# Understanding the attention mask
# ============================================================
def show_padding_example():
    """Demonstrate how padding and attention mask work."""
    sentences = ["Hello world", "The quick brown fox jumps over the lazy dog"]
    encoded = bert_tok(sentences, padding=True, return_tensors="pt")

    print("\n=== Padding example ===")
    for i, s in enumerate(sentences):
        ids = encoded["input_ids"][i].tolist()
        mask = encoded["attention_mask"][i].tolist()
        print(f"Sentence {i}: {s}")
        print(f"  IDs:  {ids}")
        print(f"  Mask: {mask}")
        # Shorter sentence has 0s in mask for padding positions

show_padding_example()


# ============================================================
# Special tokens
# ============================================================
print("\n=== Special tokens (BERT) ===")
print(f"[CLS] ID:  {bert_tok.cls_token_id}")   # 101
print(f"[SEP] ID:  {bert_tok.sep_token_id}")   # 102
print(f"[MASK] ID: {bert_tok.mask_token_id}")  # 103
print(f"[PAD] ID:  {bert_tok.pad_token_id}")   # 0
print(f"[UNK] ID:  {bert_tok.unk_token_id}")   # 100
```

### Try it yourself: experiments

| Question | Try this |
|----------|----------|
| Token count by language | Encode the same sentence in English, Chinese, Hindi — Chinese & Hindi often need 2–3× more tokens |
| Effect of vocabulary size | Compare `bert.encode(text)` vs `gpt2.encode(text)` for long text — bigger vocab uses fewer tokens |
| Code vs prose | Tokenize a Python function and a paragraph of equal length — code often uses more tokens |
| Whitespace handling | Compare `gpt2.tokenize("hello")` vs `gpt2.tokenize(" hello")` — leading space matters in BPE |
| Find a token that costs many | Try `gpt2.tokenize("antidisestablishmentarianism")` — should split into many subwords |
| OOV impossibility | Try a random Unicode emoji — byte-level BPE always tokenizes to *some* sequence (no [UNK]) |

## Tokenization trade-offs

| Tokenizer | Vocab size | Multilingual? | Languages | Pros | Cons |
|---|---|---|---|---|---|
| BERT WordPiece | 30,522 | No (base), Yes (multi) | 1/100 | Morphologically clean | English-centric |
| GPT-2 BPE | 50,257 | No | English | Byte-level, no UNK | Space handling is implicit |
| LLaMA SentencePiece BPE | 32,000 | Moderate | Mostly English | Fast, no UNK | Non-Latin scripts get many tokens |
| GPT-4 cl100k | 100,277 | Yes | Many | Large vocab → fewer tokens | Large embedding table |
| mT5 SentencePiece | 250,000 | Yes | 101 languages | Excellent multilingual | Very large vocab |

## How tokenization affects context window

Context window is measured in **tokens**, not words. The same content takes different numbers of tokens depending on:
- **Language**: English is typically 1 word ≈ 1.3 tokens for GPT-4; Chinese is ~2 characters per token
- **Vocabulary size**: GPT-4 (100k vocab) encodes the same text in ~20% fewer tokens than GPT-2 (50k vocab)
- **Code**: code is often tokenized very differently from natural language

For a 128k token context window, this might correspond to ~100 pages of English text or ~50 pages of Chinese text.

## Cross-references

- **Follow-up:** [85 — Training Objectives](./85-transformer-training-objectives.md) — what these tokens are predicted from / into
- **Follow-up:** [87 — BERT](./87-bert-encoder-pretraining.md) — uses WordPiece
- **Follow-up:** [88 — GPT](./88-gpt-decoder-only-causal-lm.md) — uses byte-level BPE
- **Follow-up:** [89 — T5](./89-t5-encoder-decoder-pretraining.md) — uses SentencePiece
- **Related:** [78 — Positional Encoding](./78-positional-encoding-in-transformers.md) — what gets added to token embeddings after tokenization

## Interview questions

<details>
<summary>Why do modern LLMs use subword tokenization rather than word-level or character-level?</summary>

Word-level: the vocabulary would need hundreds of thousands of entries to cover all words across all languages and domains. Rare words and new words are OOV. Character-level: guarantees coverage but sequences become very long (a 100-word sentence might be 500+ characters), making attention quadratically expensive. Subword: the vocabulary of 30k–100k entries covers common words as single tokens and rare words as a few morphological pieces. Complete coverage is guaranteed (falling back to characters or bytes), sequences stay reasonably short, and the model learns morphological structure from the subword patterns.
</details>

<details>
<summary>What is the difference between BPE and WordPiece?</summary>

BPE selects merges greedily by frequency — the pair that occurs most often in the corpus is merged next. WordPiece selects merges by likelihood — the pair that maximizes the language model likelihood is merged next, which corresponds to maximizing $P(ab) / (P(a) \cdot P(b))$. In practice, WordPiece tends to preserve more linguistic structure (morphological boundaries), while byte-level BPE guarantees no OOV tokens. BERT uses WordPiece; GPT family uses byte-level BPE.
</details>

<details>
<summary>What are special tokens and why are they needed?</summary>

Special tokens serve structural roles that ordinary vocabulary tokens cannot: `[CLS]` marks the beginning of a sequence and carries the sentence-level representation (used for classification); `[SEP]` separates segments within a sequence (e.g., two sentences in NSP); `[MASK]` marks masked positions during MLM training; `[PAD]` pads shorter sequences to the same length in a batch; `[UNK]` represents unknown characters (in word-level or early subword tokenizers). GPT-style models use `<|endoftext|>` as both EOS and BOS. These tokens are added to the vocabulary and have learned embeddings like ordinary tokens.
</details>

<details>
<summary>Scenario: your GPT-style model is great at English but produces garbled output in Hindi. Loss looks normal. What's likely wrong?</summary>

A tokenizer trained predominantly on English allocates almost no vocab to Devanagari. A Hindi word that is one "natural" token in English (5 letters) becomes 8-12 byte-level BPE tokens — each *byte* of the UTF-8 sequence. Three problems compound:

1. **Effective context window shrinks** by 5-10× for Hindi prompts.
2. **Per-token entropy is higher** — each token carries less semantic content, so attention has to do more work to assemble meaning.
3. **Training compute was concentrated on English tokens**, so the embeddings for Hindi-byte tokens are undertrained.

The fix is not "train more" — it's to use a tokenizer with sufficient Devanagari coverage (mT5-style SentencePiece, or extend the tokenizer and embedding table, then continue pretraining). This is also why translating from English to a low-resource language often produces worse output than the reverse direction.
</details>

<details>
<summary>Scenario: a junior engineer says "we'll just use BERT's tokenizer but with our GPT model since the vocab sizes are similar." What goes wrong?</summary>

Catastrophically wrong, but in a way that doesn't crash immediately. The tokenizer and the model's *embedding table* are tied — token ID 4937 means "cat" only because the model learned an embedding for ID 4937 that represents "cat" in its training. Swap tokenizers: ID 4937 in BERT's vocab might mean something completely different, so feeding BERT-encoded text to a GPT model would feed the GPT model semantic noise.

Even if vocab sizes match exactly, the tokenizer determines the *meaning* of every ID. The tokenizer and model must always come from the same training run. The only correct way to "swap tokenizers" is tokenizer transplant: build a mapping between the two vocabularies, initialize the new embedding table from the old one (averaging when splits don't align), and continue pretraining. Several recent papers (e.g., FOCUS, LRP transplant) do this.
</details>

<details>
<summary>Why does the same model give different outputs for `"hello"` vs `" hello"` (leading space)? Isn't this a bug?</summary>

Not a bug — it's a direct consequence of byte-level BPE. The tokenizer treats whitespace as part of the token: `"hello"` and `" hello"` are different token IDs (the space-prefixed version is what's tokenized when "hello" appears after another word). At training time the model almost always saw the space-prefixed version, so a prompt with no leading space (e.g., immediately after newline) sits at an out-of-distribution token.

This is also why prompt engineering "tricks" sometimes matter: trailing space, no trailing space, capitalization of the first letter — these change the token sequence and shift the model into different distributions. The OpenAI tokenizer playground is the easiest way to confirm what your prompt actually looks like to the model.
</details>

<details>
<summary>Scenario: you train a domain-specific medical NLP model. Should you retrain the tokenizer, extend it, or keep BERT's tokenizer?</summary>

Three options with distinct trade-offs:

- **Keep BERT's tokenizer**: zero engineering cost. But medical terms like "myocardial" → `["my", "##ocard", "##ial"]` and drug names get split heavily. The model wastes capacity learning these decompositions and uses up context window inefficiently.
- **Extend BERT's tokenizer** with medical terms: add (say) 5,000 new tokens, re-initialize their embeddings (zero or average of subword pieces), and continue pretraining. Good middle ground — keeps general-language coverage while reducing medical token bloat.
- **Retrain from scratch**: best tokenization but requires full pretraining from scratch (or transferring weights via tokenizer transplant). Only justified if your domain is *vastly* underrepresented in the original tokenizer.

For most domain-specific work, option 2 (extend) wins. BloombergGPT, BioBERT, and SciBERT all extended rather than retrained from scratch.
</details>

<details>
<summary>Why does code take more tokens than prose of equivalent character count?</summary>

Code uses many short tokens that BPE doesn't merge into single units: punctuation (`{`, `}`, `(`, `)`, `;`), single-letter variable names (`x`, `i`), and indentation whitespace. A typical English word is one token; a typical Python expression like `for i in range(10):` might be 8-10 tokens despite being short.

Modern code-specialized tokenizers (StarCoder, Code Llama) add tokens for common code patterns (`    ` four-space indent, `def `, `import `), reducing token count significantly. GPT-4's `cl100k` tokenizer was deliberately retrained with substantial code in the corpus and produces 20-30% fewer tokens on code than GPT-2's tokenizer.

Practical consequence: a 4K context window can fit ~3,000 words of prose but only ~1,000 lines of dense code.
</details>

<details>
<summary>What is the "glitch token" phenomenon (e.g., `" SolidGoldMagikarp"`)?</summary>

GPT-2/3's tokenizer was trained on a Reddit-derived corpus that included rare usernames and phrases. These got their own token IDs in the vocab. But the *model* was later trained on a different corpus where those tokens almost never appeared, so the embeddings for those token IDs are essentially untrained.

When you feed one of these untrained-embedding tokens to the model, behavior becomes unpredictable: the model might emit them randomly, refuse to repeat them, hallucinate, or produce garbage. The famous `" SolidGoldMagikarp"` token caused GPT-3 to behave bizarrely when prompted with it.

The deeper lesson: tokenizer training data and model training data should be aligned. Mismatch creates "dead" token IDs in the vocabulary. Modern tokenizers are co-trained or post-filtered to avoid this.
</details>

<details>
<summary>Why doesn't byte-level BPE need an `[UNK]` token, but WordPiece does?</summary>

Byte-level BPE's initial alphabet is the 256 possible byte values, not Unicode characters. Any string can be represented as a UTF-8 byte sequence, and any byte sequence can be tokenized using individual bytes as a worst-case fallback. There is no input that cannot be encoded.

WordPiece's initial alphabet is the characters seen in training. Any character outside that set has no token, so the tokenizer emits `[UNK]` and information is lost. BERT was originally trained on text where it was reasonable to assume Latin + some Cyrillic + a few other scripts; for emojis, rare Unicode, or unusual whitespace, `[UNK]` is the result.

This is why byte-level BPE (GPT-2/3/4) is preferred for general-purpose models — robust to *any* input. WordPiece persists in BERT-family models more out of historical inertia than technical preference.
</details>

<details>
<summary>Scenario: you tokenize `"GPT-4"` and get `["G", "##PT", "-", "4"]` (4 tokens) but `"BERT"` gives `["BERT"]` (1 token). Why the asymmetry?</summary>

Token frequency in the training corpus dictates merge order. "BERT" appeared frequently as a whole word in BERT's training corpus (mostly Wikipedia + BookCorpus, where "BERT" itself was rarely discussed but is a common surname). "GPT-4" didn't exist when BERT was trained — and even today, "GPT" appears less frequently than "BERT" in many corpora, especially in academic text.

This illustrates a structural truth about tokenizers: **they reflect the era and corpus they were trained on.** A 2018 tokenizer treats 2024 vocabulary inefficiently. This is why tokenizer "freshness" matters — modern tokenizers (`cl100k`, LLaMA 3) explicitly include recent corpora and brand names. It's also why training a model with a 5-year-old tokenizer puts it at a structural disadvantage on contemporary text.
</details>

<details>
<summary>Tokens are integers — why does the model's vocabulary size matter to performance, not just to compute?</summary>

Larger vocabulary means shorter sequences (more chars per token on average), which makes attention's $O(T^2)$ cost shrink and effectively expands the usable context window. But the embedding table cost scales linearly: vocab × $d_{\text{model}}$ parameters. For a 50k vs 100k vocab at $d = 4096$: 200M extra embedding parameters. The cost of expanding from 50k → 100k might be a 20% reduction in tokens, but a 25% increase in embedding params.

There's also a *training data* effect: very rare tokens (long tail of the vocab) get few gradient updates and end up undertrained. A vocab that's too large for the corpus produces "dead" tokens. mT5's 250k vocab works only because it's trained on 101 languages with hundreds of GB of text per language.

The optimal vocab size balances three things: sequence length (favors big vocab), embedding-table cost (favors small vocab), and adequate training signal per token (favors small vocab relative to data size).
</details>

<details>
<summary>Why is the BERT `[CLS]` token always at position 0, and not, say, position -1?</summary>

It's a convention, but the choice has real consequences. At position 0, `[CLS]` is the first thing the encoder sees; with bidirectional attention, it has equal access to all positions, so it can build a global summary. Placing it at the end would also work *in BERT* because attention is symmetric — every position attends to every other.

The reason position 0 won is mostly compatibility with downstream code: pooling layers expect `[CLS]` at index 0, fine-tuning datasets are pre-tokenized with `[CLS]` first. Changing this would break a huge ecosystem.

In contrast, GPT-style models would not work well with a `[CLS]` at position 0: causal attention means position 0 sees nothing. The natural summary token for a decoder-only model is at the *end* — and it doesn't need to be a special token, just the last hidden state of the input.
</details>

<details>
<summary>What is the "Llama tokenizer is bad for Japanese" claim about, and how would you fix it?</summary>

LLaMA's SentencePiece tokenizer has ~32k vocab dominated by English subwords. Japanese characters (Hiragana, Katakana, Kanji) get tokenized at near-character level, often via UTF-8 bytes — so a 20-character Japanese sentence might be 60-80 tokens. This (1) makes Japanese context windows 3-4× smaller in characters, and (2) means each token carries very little Japanese semantic content.

Two production fixes are common:
- **Tokenizer transplant**: build a Japanese-augmented tokenizer (~64k vocab), expand the embedding table, initialize new embeddings from byte-piece averages, then continue pretraining (e.g., ELYZA-japanese-Llama).
- **Domain-specific finetune**: finetune the base LLaMA on Japanese text without changing the tokenizer. Cheaper but doesn't fix the structural inefficiency.

This is why most Japanese LLMs (Stockmark, Rinna, ELYZA) start from LLaMA *plus* a transplanted tokenizer rather than vanilla LLaMA.
</details>

## Points to remember

- A tokenizer and a model are paired forever — you cannot swap one without retraining or transplanting embeddings.
- Subword tokenization is the universal compromise: small enough vocab to train, complete coverage, short enough sequences.
- BPE is greedy by frequency, WordPiece is greedy by likelihood, SentencePiece is a language-agnostic framework that wraps either.
- Byte-level BPE (GPT family) eliminates `[UNK]` — every input can always be encoded as bytes.
- Token count ≠ word count. Different languages, code, and tokenizer versions all change the ratio dramatically.
- Leading-space sensitivity is a feature of byte-level BPE, not a bug — `"hello"` and `" hello"` are different tokens.
- Vocabulary size trades sequence length against embedding-table cost; very rare tokens become "dead" (undertrained) tokens.
- Domain adaptation is usually done by *extending* the tokenizer + continuing pretraining, not retraining from scratch.
- Context window is always measured in tokens; English ≈ 1.3 tokens/word for GPT-4, but Chinese, Japanese, Hindi can be 3-10× higher.
- Glitch tokens (`" SolidGoldMagikarp"`) reveal mismatch between tokenizer training data and model training data — a structural risk in any pipeline that builds them separately.

## Common mistakes

- Forgetting to add special tokens (`add_special_tokens=True`) when encoding for BERT — `[CLS]` and `[SEP]` are required for correct classification and QA
- Using a different tokenizer at inference than training — the vocab and merge rules must match exactly
- Not applying an attention mask — models treat padding tokens as real tokens without masking, corrupting representations
- Assuming token count equals word count — especially important for understanding actual context window limits

## Final takeaway

Tokenization is the first step in every transformer pipeline. Subword tokenization via BPE or WordPiece balances vocabulary size, OOV coverage, and sequence length. The tokenizer is tied to the model — you cannot mix tokenizers across models. Understanding how tokenization works explains model quirks: why "GPU" becomes ["G", "##P", "##U"] in BERT, why code in a language model can use up more context window than expected, and why multilingual models need much larger vocabularies.

## References

- Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units (BPE). ACL.
- Wu, Y., et al. (2016). Google's Neural Machine Translation System (WordPiece). arXiv.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer. EMNLP.
