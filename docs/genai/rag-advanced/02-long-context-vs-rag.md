---
id: rag-adv-long-context
title: "Long Context vs RAG"
sidebar_label: "2 · Long context vs RAG"
sidebar_position: 2
slug: /genai/rag-advanced/long-context-vs-rag
description: "When can you skip retrieval and put the whole corpus in the prompt? A cost model with real list prices, a needle test on a small model, and the hybrid strategy that routes between the two."
tags: [long-context, rag, needle-in-a-haystack, lost-in-the-middle, prompt-caching, cost-model]
---

import Infographic from '@site/src/components/Infographic';
import ContextVsRagLab from '@site/src/components/viz/ContextVsRagLab';

**In one line.** A long context window lets you skip retrieval for a small corpus, but you pay for every token on every question and the model reads the middle of a long prompt worse than the ends, so the choice between stuffing the prompt and retrieving is a cost and accuracy calculation, not a fashion.

:::tip Before you start
**You should already know**

- What RAG is and why it exists ([RAG basics](/docs/genai/rag)).
- What a token is and what a context window limits ([intro to LangChain](/docs/genai/intro-to-langchain)).
- That the model keeps a key and value for every token it has read ([the KV cache](/docs/llm-engineering/kv-cache-and-paged-attention)).

**Reading time:** about 30 minutes, plus about 10 minutes to run the code (blocks 3 to 5 run a 360-million-parameter model on CPU).

**After this chapter you can**

- Work out, in dollars, what a long-context prompt costs against RAG for your corpus size and traffic.
- Run a needle test and say what it does and does not prove.
- Design a router that tries cheap RAG first and falls back to the full corpus.
:::

:::note Not from a lecture
Written for this site from the sources under Go deeper. Prices are the Claude Sonnet 5.5 list prices read on 7 October 2026 and are named parameters in the code. The needle test uses a small open model, so its curve shows the shape of the problem, not the quality of any production model.
:::

## In 30 seconds

Imagine a 200-page handbook and a colleague who must answer questions about it all day. You can hand them the full handbook before every single question, or you can keep a good index and hand them the three relevant pages. The first needs no index and misses nothing the colleague can find, but it means re-reading 200 pages every time, which is slow and expensive, and people skim the middle of long documents. The second is cheap and quick but fails if the index sends the wrong pages. Long context against RAG is that choice, and the numbers decide it.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Context window | The most tokens a model can read in one request | 1,000,000 for the larger current Claude models |
| Long-context prompting | Putting the whole corpus in the prompt and asking | A 500,000-token handbook plus one question |
| Needle test | Hide one fact in filler text and ask the model for it | "Ingrid's bicycle lock is set to 4817" |
| Haystack | The filler text around the needle | Wikipedia sentences |
| Lost in the middle | Facts at the start or end of a long prompt are found more often than facts in the middle | Recall high at depth 0 and 1, low at 0.5 |
| Prefill | The model reading the whole prompt before it writes the first word | The wait before the first token |
| Prompt cache | A saved copy of a prompt prefix that makes re-reading cheaper | Cache read costs 0.1 times the normal input price |
| Router | A rule that sends each question to RAG or to the full corpus | Try RAG, fall back if the model says it cannot answer |
| Distractor | Text that looks like the answer but is not | "Tomas keeps a gym locker set to 1111" |

## The idea in plain words

Context windows grew from a few thousand tokens to a million. At 1,000,000 tokens you can fit roughly 2.5 million characters, so many whole corpora fit: a product manual, a contract set, a code base. The tempting conclusion is that RAG is obsolete. The numbers say otherwise, in three places.

1. **Cost.** A model charges for every input token on every request. A 500,000-token corpus read twice costs twice. RAG sends a few thousand tokens each time.
2. **Latency.** The model must read the whole prompt (prefill) before it can answer. On the small model in block 5, reading 4,096 tokens takes about five times longer than reading 512, and you wait for all of it before the first word appears.
3. **Accuracy.** Models find facts best near the start and end of a prompt and worse in the middle, and accuracy falls as distractors and length grow. The needle test in block 3 shows this on a small model.

On the other side, retrieval has its own failures: the right chunk may not come back, or the question may need facts spread across the whole corpus (the problem [GraphRAG](/docs/genai/rag-advanced/graphrag-and-knowledge-graphs) addresses). Long context is the better answer when the corpus is small, the questions are few, or the answer needs everything.

Anthropic's September 2024 write-up on contextual retrieval gives a simple rule of thumb: if your knowledge base is smaller than 200,000 tokens, about 500 pages, you can include all of it in the prompt with no need for RAG. That was written when 200,000 tokens was the window. Today the windows are five times larger, so the threshold has moved, and what remains is the cost.

<Infographic src="/img/rag-adv/long-context-vs-rag-decision.svg" alt="A decision chart with three questions: does the corpus fit the window, is it asked more often than every five minutes so the cache stays warm, and does every question need the whole corpus; with the strategy each answer leads to" caption="Read from the top. Each question narrows the choice. The last box is the usual answer for a corpus that fits but is asked about often: RAG first, long context as the fallback." />

## Worked example, step by step

A support team has a 500,000-token knowledge base. A question and an answer are small: 100 tokens in, 300 tokens out. Prices are Claude Sonnet 5.5 list prices: 2.00 dollars per million input tokens, 10.00 per million output tokens. A cache read costs 0.1 times the input price, a five-minute cache write 1.25 times.

1. **Long context, no cache.** Input is 500,000 + 100 = 500,100 tokens. 500,100 x 2.00 / 1,000,000 = 1.0002 dollars. Output is 300 x 10.00 / 1,000,000 = 0.003. Total 1.0032 dollars per question.
2. **RAG.** Retrieve 4,000 tokens of chunks. Input is 4,100 tokens: 4,100 x 2.00 / 1,000,000 = 0.0082. Add the same 0.003 for output. Total 0.0112 dollars per question.
3. **Ratio.** 1.0032 / 0.0112 is about 90. The long-context prompt costs about 90 times more.
4. **With a warm cache.** Cache reads cost 0.1 x 2.00 = 0.20 per million, so the corpus part is 500,000 x 0.20 / 1,000,000 = 0.10. Add 0.0002 for the question and 0.003 for the answer: 0.1032. That is 9.2 times RAG, which is much closer but still not equal.
5. **Is the cache warm?** The cache lasts five minutes unless it is read again. At one question every 5 minutes or faster it stays warm. At one question per hour, every question pays the 1.25 times write price: 500,000 x 2.50 / 1,000,000 = 1.25 for the corpus alone.

Block 1 reproduces these numbers.

<Infographic src="/img/rag-adv/long-context-vs-rag-worked-example.svg" alt="Cost of one question under four strategies for a 500,000-token corpus: long context 1.0032 dollars, long context with cache hit 0.1032, RAG 0.0112, and the ratio of each to RAG" caption="Look at the bar lengths first: long context is about 90 times RAG per question, and a warm cache brings it to about nine times." />

## How it works

### The cost model

For input tokens `n` at price `p` dollars per million, cost is `n x p / 1,000,000`. In words: you pay for every token you send, every time. A cache changes the price of a repeated prefix: the first write costs more, reads cost much less. So long-context cost has three parts: the corpus (cheap only if cached and the cache stays warm), the question, and the answer.

Three things are easy to forget.

- **Traffic decides whether the cache helps.** If questions arrive slower than every five minutes, the cache expires between them and you pay the write premium each time. Block 1 prints this: at one question per hour, caching is 25 percent more expensive than not caching.
- **Output is the same either way.** The difference between the strategies is entirely in the input.
- **Window size is a hard limit.** A 1,200,000-token corpus does not fit in a 1,000,000-token window, whatever the price.

### The accuracy side: needle tests, and why they flatter models

A needle test hides one sentence in filler text and asks about it. Early versions used a single needle with no distractors and models scored near perfect, which is why they became a marketing chart. Later work made the test harder, and the results fell.

- **Lost in the middle** (Liu and colleagues, arXiv 2307.03172, submitted July 2023, final version November 2023) found that language models perform significantly worse when the relevant information is in the middle of a long input, with the best results at the beginning or end, even for models built for long contexts.
- **RULER** (Hsieh and colleagues, arXiv 2404.06654, April 2024) evaluated 17 long-context models on tasks beyond simple retrieval and reported that only about half of the models claiming 32,000 tokens or more kept satisfactory performance at 32,000.
- **NoLiMa** (Modarressi and colleagues, arXiv 2502.05167, ICML 2025) removed the word overlap between the question and the needle. At 32,000 tokens, 11 of 13 models fell below half of their short-context scores; GPT-4o dropped from 99.3 to 69.7 percent.
- **Context Rot** (Chroma, 14 July 2025) tested 18 models and reported that performance degrades as input length grows, that even a single distractor lowers accuracy and that the effect grows with length.

The shared lesson: the more the question differs from the needle's wording, and the more look-alikes surround it, the faster accuracy falls with length. Block 3 reproduces the effect in miniature; block 4 shows what retrieval does about it.

### Hybrid strategies

The Self-Route study (Li and colleagues, arXiv 2407.16833, EMNLP 2024 industry track) compared RAG with long context across several tasks and found that long context outperformed RAG on average when resourced sufficiently, while RAG kept a large cost advantage. It then proposed routing each query: let the model decide, from the retrieved chunks, whether it can answer; if yes, keep the RAG answer; if not, send the query again with the full context. The paper reports that this cuts computation cost significantly while keeping performance comparable to long context. Block 2 prices such a router: how much of the saving survives depends on how often the model correctly says "I cannot answer from these chunks".

Other combinations in use: retrieve a long chunk (a whole section) instead of a short one, cache the stable part of the prompt and retrieve only the volatile part, or keep a compact summary of the corpus in the cached prefix and retrieve detail on demand.

### What stays the same: the KV cache grows with every token

Whatever you pay, the machine must hold a key and a value vector per layer per token. For the model in block 5 that is a fixed number of bytes per token, so memory grows in a straight line with prompt length. At a million tokens it is huge even for a small model, which is part of why long context is priced as it is. The [KV cache chapter](/docs/llm-engineering/kv-cache-and-paged-attention) covers the formula and the tricks that shrink it.

## A real system that works this way

The 1,000,000-token window is real and current. Anthropic's model overview, read on 7 October 2026, lists 1M tokens for Claude Fable 5.1, Claude Opus 5.5 and Claude Sonnet 5.5, and 200K for Claude Haiku 4.5. Its pricing page states that the full window is billed at the standard rate, with no long-context surcharge (a 900,000-token request costs the same per token as a 9,000-token one), that Claude Sonnet 5.5 costs 2.00 dollars per million input tokens and 10.00 per million output tokens, and that cache reads cost 10 percent of the input price while a five-minute cache write costs 1.25 times it. Those are the exact parameters in block 1. Prices change; read the page again before you rely on them.

## Code you can run

Five blocks. Blocks 1 and 2 are pure arithmetic. Blocks 3 and 4 use `HuggingFaceTB/SmolLM2-360M-Instruct` (Apache 2.0, trained for a context of 8,192 tokens; we stay at or below 4,096). Block 5 measures time and memory on the same model. Libraries: Python 3.14, `torch` 2.14.1, `transformers` 5.18.0, `sentence-transformers` 6.1.0, `datasets`; the filler text is the WikiText-2 test split. With the models cached, run with `HF_HUB_OFFLINE=1`. Blocks 3 and 4 take several minutes each on a laptop CPU. Needle results come from 6 trials per cell, so one trial is a step of 17 points: read the shape, not individual cells.

### 1. The cost model

We price one question under each strategy for several corpus sizes, then ask when a prompt cache pays off at different traffic levels.

```python
PRICE_IN, PRICE_OUT = 2.00, 10.00
CACHE_WRITE, CACHE_READ = 2.50, 0.20
WINDOW = 1_000_000
QUESTION, ANSWER = 100, 300
RAG_CONTEXT = 4_000
CACHE_SECONDS = 300


def money(tokens, price):
    return tokens * price / 1e6


def long_context(corpus, mode):
    if corpus + QUESTION + ANSWER > WINDOW:
        return None
    rate = {"cold": PRICE_IN, "write": CACHE_WRITE, "warm": CACHE_READ}[mode]
    return money(corpus, rate) + money(QUESTION, PRICE_IN) + money(ANSWER, PRICE_OUT)


rag = money(RAG_CONTEXT + QUESTION, PRICE_IN) + money(ANSWER, PRICE_OUT)
print(f"one RAG query: {rag:.4f} dollars")
print(f"{'corpus tokens':>14}{'long context':>14}{'cache hit':>11}{'vs RAG':>9}{'vs RAG (hit)':>14}")
for corpus in (50_000, 200_000, 500_000, 900_000, 1_200_000):
    cold, warm = long_context(corpus, "cold"), long_context(corpus, "warm")
    if cold is None:
        print(f"{corpus:>14,}{'does not fit':>14}")
        continue
    print(f"{corpus:>14,}{cold:14.4f}{warm:11.4f}{cold / rag:8.0f}x{warm / rag:13.1f}x")

corpus = 500_000
print(f"\ncost per hour for a {corpus:,}-token corpus, by traffic")
print(f"{'queries/hour':>13}{'no cache':>10}{'with cache':>12}{'RAG':>9}")
for per_hour in (1, 6, 12, 60, 600):
    gap = 3600 / per_hour
    no_cache = per_hour * long_context(corpus, "cold")
    if gap <= CACHE_SECONDS:
        cached = long_context(corpus, "write") + (per_hour - 1) * long_context(corpus, "warm")
    else:
        cached = per_hour * long_context(corpus, "write")
    print(f"{per_hour:>13}{no_cache:10.2f}{cached:12.2f}{per_hour * rag:9.2f}")
```

**Reading the output.** RAG costs 0.0112 dollars per question. A 500,000-token corpus costs 1.0032 uncached (90 times RAG) and 0.1032 on a cache hit (9.2 times). The 1,200,000-token corpus does not fit at all. In the hourly table, at one or six questions an hour the cache loses (1.25 against 1.00, and 7.52 against 6.02), because the gap between questions is longer than five minutes and every request pays the write premium. At 12 questions an hour the gap is exactly five minutes, the cache stays warm, and the hourly cost drops from 12.04 to 2.39. At 600 an hour long context costs 601.92 uncached and 63.07 cached, against 6.72 for RAG.

**Line by line.**

- `long_context(corpus, mode)` returns `None` if the prompt exceeds the window.
- `mode` picks the corpus rate: normal input price, cache write (1.25 times) or cache read (0.1 times).
- `gap <= CACHE_SECONDS` decides whether the cache is warm: if so, one write and then reads; if not, a write every time.

### 2. A router: RAG first, long context as a fallback

Now the hybrid. RAG answers a share of questions; the rest are sent again with the whole corpus. The cost is the RAG attempt plus the fallback share of the long-context price.

```python
PRICE_IN, PRICE_OUT = 2.00, 10.00
QUESTION, ANSWER, RAG_CONTEXT = 100, 300, 4_000
corpus = 500_000


def money(tokens, price):
    return tokens * price / 1e6


rag = money(RAG_CONTEXT + QUESTION, PRICE_IN) + money(ANSWER, PRICE_OUT)
long_cost = money(corpus + QUESTION, PRICE_IN) + money(ANSWER, PRICE_OUT)
print(f"RAG {rag:.4f}, long context {long_cost:.4f} dollars per query")
print(f"{'RAG answers it':>15}{'fall back':>11}{'cost per query':>16}{'vs always long':>16}")
for answered in (1.0, 0.8, 0.6, 0.4, 0.0):
    hybrid = rag + (1 - answered) * long_cost
    print(f"{answered:>15.0%}{1 - answered:>11.0%}{hybrid:16.4f}{hybrid / long_cost:15.2f}x")
```

**Reading the output.** If RAG answers 80 percent of questions, the average cost is 0.2118 dollars, about a fifth of always using long context. At 60 percent it is 0.4125. If RAG answered none, you would pay for the wasted RAG attempt as well: 1.0144, slightly more than long context alone. The saving depends entirely on the routing share, and on the model recognising when it cannot answer.

**Line by line.**

- `hybrid = rag + (1 - answered) * long_cost` is the whole model: every question pays for RAG, and the fallback share also pays for the full prompt.
- The last row (`0%`) shows the cost of a router that never succeeds.

### 3. A needle in a haystack, on a small model

Now the accuracy side, on a 360-million-parameter model. We take WikiText filler of length 512, 2,048 or 4,096 tokens and insert one needle ("Ingrid keeps her bicycle lock set to" a four-digit code) at the start, the middle or the end, together with eight distractor sentences about other people's locks. We ask for Ingrid's code and check whether it appears in the answer.

```python
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
filler = tok(" ".join(t.strip() for t in wiki["text"] if len(t.strip()) > 200))["input_ids"][:20000]
LITERAL = "\n\nQuestion: What is Ingrid's bicycle lock set to?\nAnswer: Ingrid's bicycle lock is set to"
PARAPHRASE = "\n\nQuestion: Which number opens the cycle padlock that Ingrid owns?\nAnswer: The number is"
PEOPLE = ["Tomas", "Mirela", "Joao", "Hanna", "Kofi", "Lena", "Piotr", "Sana"]
ITEMS = ["gym locker", "suitcase lock", "garage door", "office safe", "shed padlock", "bike lock"]


def build(length, depth, code, seed, ask=LITERAL, decoys=8):
    question = tok(ask)["input_ids"]
    rng = np.random.default_rng(seed)
    needle = tok(f" Ingrid keeps her bicycle lock set to {code}.")["input_ids"]
    extra = [tok(f" {PEOPLE[rng.integers(8)]} keeps a {ITEMS[rng.integers(6)]} set to {rng.integers(1000, 9999)}.")["input_ids"] for _ in range(decoys)]
    room = length - len(needle) - sum(map(len, extra)) - len(question)
    inserts = [(int(depth * room), needle)] + [(int(rng.integers(0, room)), e) for e in extra]
    out, last = [], 0
    for position, piece in sorted(inserts, key=lambda p: p[0]):
        out += filler[last:position] + piece
        last = position
    return out + filler[last:room] + question


def answers(ids, code):
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=6, do_sample=False)
    return code in tok.decode(out[0, len(ids):]).replace(" ", "")


CODES = [str(1000 + 937 * i % 9000) for i in range(1, 7)]
print(f"needle recovered out of {len(CODES)}, with 8 decoy sentences; model {name.split('/')[1]}")
print(f"{'tokens':>7}{'start':>8}{'middle':>8}{'end':>6}")
for length in (512, 2048, 4096):
    cells = [sum(answers(build(length, d, c, 100 + i), c) for i, c in enumerate(CODES)) for d in (0.0, 0.5, 1.0)]
    print(f"{length:>7}{cells[0]:>8}{cells[1]:>8}{cells[2]:>6}")

print("\n4096 tokens, needle in the middle: does the wording of the question matter?")
for label, ask in (("question repeats the needle's words", LITERAL), ("question uses different words", PARAPHRASE)):
    hits = sum(answers(build(4096, 0.5, c, 100 + i, ask), c) for i, c in enumerate(CODES))
    print(f"  {label:<40}{hits}/{len(CODES)}")
```

**Reading the output.** Each cell is 6 trials, so one trial is a step of 17 points. The model finds the needle at the start and at the end almost every time: 6 of 6 at every length. In the middle it drops: 5 of 6 at 512 tokens and 4 of 6 at 2,048 and 4,096. That is the lost-in-the-middle shape, in miniature: the ends are safe and the middle is not.

What did not happen is a collapse with length. Going from 512 to 4,096 tokens cost the middle position one trial, and the ends none. With one needle, a question that repeats the needle's words and eight look-alikes, this 360-million-parameter model copes up to 4,096 tokens.

The last two lines are the NoLiMa idea in miniature. At 4,096 tokens with the needle in the middle, the model answers 4 of 6 when the question repeats the needle's words ("Ingrid's bicycle lock") and 2 of 6 when it asks "Which number opens the cycle padlock that Ingrid owns?". Same needle, same haystack, only the wording of the question changed. Six trials is a small sample, so treat the direction as the finding, not the size

**Line by line.**

- `build` places the needle at `depth` times the room available and scatters eight distractors at random positions.
- `answers` decodes six tokens greedily and checks whether the code appears in them.
- `CODES` gives six different four-digit codes, so each cell is six independent trials.

### 4. The same question through RAG

Now the model reads only three retrieved chunks of 64 tokens instead of the whole haystack. A small sentence model retrieves them from overlapping chunks of the same haystack. We try two orderings: the best chunk last, nearest the question, and first.

```python
import time

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
filler = tok(" ".join(t.strip() for t in wiki["text"] if len(t.strip()) > 200))["input_ids"][:20000]
ASK = "What is Ingrid's bicycle lock set to?"
PEOPLE = ["Tomas", "Mirela", "Joao", "Hanna", "Kofi", "Lena", "Piotr", "Sana"]
ITEMS = ["gym locker", "suitcase lock", "garage door", "office safe", "shed padlock", "bike lock"]


def haystack(length, code, seed, decoys=8):
    rng = np.random.default_rng(seed)
    pieces = [f" Ingrid keeps her bicycle lock set to {code}."] + [f" {PEOPLE[rng.integers(8)]} keeps a {ITEMS[rng.integers(6)]} set to {rng.integers(1000, 9999)}." for _ in range(decoys)]
    ids = filler[:length - 150]
    for piece in pieces:
        at = int(rng.integers(0, len(ids)))
        ids = ids[:at] + tok(piece)["input_ids"] + ids[at:]
    return ids


def rag_answer(ids, code, best_last, size=64, stride=48, top=3):
    spans = [ids[i:i + size] for i in range(0, max(1, len(ids) - size + stride), stride)]
    texts = [tok.decode(s) for s in spans]
    sims = retriever.encode(texts, normalize_embeddings=True) @ retriever.encode([ASK], normalize_embeddings=True)[0]
    best = list(np.argsort(-sims)[:top])
    context = "".join(texts[i] for i in (best[::-1] if best_last else best))
    prompt = tok(context + f"\n\nQuestion: {ASK}\nAnswer: Ingrid's bicycle lock is set to")["input_ids"]
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=6, do_sample=False)
    return code in tok.decode(out[0, len(prompt):]).replace(" ", ""), len(prompt), any(code in texts[i].replace(" ", "") for i in best)


CODES = [str(1000 + 937 * i % 9000) for i in range(1, 7)]
print(f"{'tokens':>7}{'needle in top 3':>17}{'best chunk last':>17}{'best chunk first':>18}{'tokens read':>13}")
for length in (512, 2048, 4096):
    runs = {last: [rag_answer(haystack(length, c, 100 + i), c, last) for i, c in enumerate(CODES)] for last in (True, False)}
    print(f"{length:>7}{sum(r[2] for r in runs[True]):>14}/6{sum(r[0] for r in runs[True]):>14}/6{sum(r[0] for r in runs[False]):>15}/6{np.mean([r[1] for r in runs[True]]):>13.0f}")
```

**Reading the output.** The retriever puts the needle in the top 3 chunks in all 18 runs, so retrieval never failed. The answers are less tidy: the model gets 3 of 6 right at 512 tokens, 5 of 6 at 2,048 and 6 of 6 at 4,096, and the order of the chunks (best last or best first) made no difference in this run. It reads about 215 tokens in every case, against 512, 2,048 or 4,096 for the whole prompt.

Look at the first row. At 512 tokens RAG is worse than reading everything (3 of 6 against 17 of 18 across positions for the whole prompt), even though the needle was in the chunks it was given. A likely reason is that eight look-alike sentences are packed into 512 tokens, so the three chunks, which cover about 40 percent of the haystack, contain several of them and the small model picks the wrong code. At 4,096 tokens the look-alikes are spread thin and the same three chunks are cleaner. I did not test this explanation; it is a hypothesis to check before you rely on it.

The lesson is not that RAG wins. On a short, crowded prompt the whole text was better. RAG's advantage here is cost: about 215 tokens read instead of 4,096

**Line by line.**

- `spans` cuts the haystack into 64-token windows that overlap by 16, so a needle on a window edge is still seen whole by one of them.
- The retrieval query is only the question text, as in a real RAG system.
- `best[::-1]` reverses the order so the most similar chunk sits last, next to the question.

### 5. What a long prompt costs the machine

Last, time and memory. We time one forward pass over prompts of increasing length, and compute the KV cache size from the model's configuration.

```python
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

name = "HuggingFaceTB/SmolLM2-360M-Instruct"
config = json.loads(Path(hf_hub_download(name, "config.json")).read_text())
layers, kv_heads = config["num_hidden_layers"], config["num_key_value_heads"]
head_dim = config["hidden_size"] // config["num_attention_heads"]
per_token = 2 * layers * kv_heads * head_dim * 2
print(f"{name.split('/')[1]}: {layers} layers, {kv_heads} KV heads, head size {head_dim}")
print(f"KV cache per token in 16-bit: 2 x {layers} x {kv_heads} x {head_dim} x 2 bytes = {per_token:,} bytes")

model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
print(f"{'tokens':>7}{'prefill seconds':>17}{'ms per 1k tokens':>18}{'KV at 16-bit (MB)':>19}")
for length in (512, 1024, 2048, 4096):
    ids = torch.randint(100, 30000, (1, length))
    best = 1e9
    for _ in range(2):
        start = time.perf_counter()
        with torch.no_grad():
            model(ids, use_cache=False)
        best = min(best, time.perf_counter() - start)
    print(f"{length:>7}{best:>17.2f}{1000 * best / (length / 1000):>18.0f}{length * per_token / 1e6:>19.1f}")
print(f"a 1,000,000-token prompt would need {1_000_000 * per_token / 1e9:.1f} GB of KV cache for this small model alone")
```

**Reading the output.** Reading 512 tokens takes 0.89 seconds on this CPU in 32-bit floats, and reading 4,096 tokens takes 4.71 seconds: 8 times the text costs about 5.3 times the time. Per thousand tokens the cost falls from 1,733 ms to 1,151 ms, because a short prompt carries a fixed overhead. So on this model, at these lengths, prefill time grows close to a straight line. The attention part, which grows faster than a line, is small compared with the rest at 4,096 tokens, and I did not measure longer prompts.

The memory side is exact arithmetic from the model's configuration: 40,960 bytes of KV cache per token, so 21.0 MB at 512 tokens, 167.8 MB at 4,096 tokens and 41.0 GB for a 1,000,000-token prompt, for a model with only 360 million parameters. Your timings will differ with your hardware; the ratios are what to compare

**Line by line.**

- `per_token = 2 * layers * kv_heads * head_dim * 2` is keys and values, over every layer and KV head, at 2 bytes each in 16-bit precision.
- `use_cache=False` times only the prompt-reading step.
- Each length is timed twice and the faster run is kept, to reduce noise from other processes.

### Try the cost model in the lab

The lab uses the same formulas as block 1. Move the corpus size, the traffic and the prices, and watch which strategy wins and when the cache helps.

<ContextVsRagLab />

**What each control does.**

- **corpus tokens**: size of the knowledge base, 10,000 to 1,200,000. From 1,000,000 upwards it no longer fits.
- **queries per hour**: traffic. Under about 12 an hour the five-minute cache goes cold between questions.
- **RAG context tokens**: how much retrieved text RAG sends.
- **answered by RAG**: the share of questions the hybrid router keeps; the rest fall back to the full corpus.
- **price in / price out**: dollars per million tokens. The defaults are the Sonnet 5.5 list prices.
- **show data**: dollars per query and per hour for each strategy.

**Try it yourself.**

1. Leave the defaults (500,000 tokens, 12 queries an hour). The bars show 12.04 uncached, 2.39 cached and 0.13 for RAG, the same as the 12-an-hour row in block 1. Now change queries per hour to 6: the cached bar becomes more expensive than the uncached one.
2. Set the corpus to 50,000 tokens. Long context with a warm cache is about 1.2 times RAG per query: for a small corpus, the simplicity of skipping retrieval may be worth that small premium.
3. Set the corpus to 1,200,000. Long context disappears: it does not fit. Only RAG remains, which is the case for retrieval that no price cut can change.

<Infographic src="/img/rag-adv/long-context-vs-rag-needle-and-cost.svg" alt="Left: needle recovery out of 6 by prompt length and needle position for the whole-prompt approach and for RAG; right: prefill time per 1,000 tokens and KV cache size by prompt length" caption="Left: how often the small model finds the needle. Right: what each prompt costs the machine. Numbers come from blocks 3 to 5." />

## Designing with it

- **Start from the corpus size.** If it does not fit the window, RAG is not a choice. If it fits with room to spare, compute both strategies at your real traffic.
- **Cache only what repeats.** The cache pays only when the same prefix is read again within five minutes. Put the stable corpus first and the changing question last.
- **Order the retrieved chunks.** Put the most relevant chunk nearest the question. Block 4 compared the two orders and found no difference on this small model, so treat the placement as a cheap default and test it on yours.
- **Test with your own questions.** A needle test is a floor, not a guarantee. Use real questions, with the answer's wording different from the question's, as NoLiMa did.
- **Build the router with an escape hatch.** Let the model say "not answerable from these chunks" and pay for the fallback. Track how often it says so and how often it is right.
- **Re-price on every model change.** Prices, cache rules and windows moved a lot between 2024 and 2026.
- **Remember what retrieval gives you for free.** Citations to a source chunk, per-user access control and a corpus that updates one chunk at a time are easier with RAG.

## Where this stands in 2026

:::info Industry view
One-million-token windows are on sale at flat per-token rates, so the question has moved from "can it fit?" to "what does reading it every time cost, and does the model use all of it?". The research above says accuracy still falls with length when questions and answers do not share words and when distractors are present, though I read the abstracts and Chroma's post, not each model's own long-context results, so I cannot tell you how a specific current model behaves on your data. The practical pattern is the hybrid: retrieval first, the full corpus as a fallback, a cache for the stable part. Price and window numbers change quickly; the formulas in block 1 do not.
:::

## Common mistakes

- **Treating a perfect needle chart as proof.** A single needle with no distractors and shared words is the easiest test, and it feels like evidence. Add look-alike distractors and change the wording, as RULER and NoLiMa did.
- **Assuming the cache always helps.** It feels free. Under about one question per five minutes it costs more than no cache (block 1: 1.25 against 1.00 an hour at one question an hour).
- **Pasting retrieved chunks in retrieval order without checking.** It is the default and it feels neutral. Block 3 shows the middle of a long prompt is the weakest spot, so put the best chunk nearest the question, then measure whether the order matters for your model.
- **Judging on one prompt length.** A model that is fine at 512 tokens may not be at 4,096. Test the length you will ship.
- **Dropping RAG because the corpus fits today.** The corpus grows, traffic grows, and the model changes. Keep the retrieval path working even if you launch with long context.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> A corpus is 300,000 tokens and the window is 1,000,000. Does that mean long context is the right choice?</summary>

No. It means long context is possible. The choice depends on cost per question at your traffic, latency, and accuracy on your questions. Compute both strategies with the block 1 formulas.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Using block 1's prices, why is a cache hit about nine times RAG per question instead of equal to it?</summary>

A cache hit makes the corpus 10 times cheaper, not free. The corpus part is still 500,000 tokens at 0.20 per million, which is 0.10 dollars per question, against about 0.0082 for RAG's 4,100 input tokens.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> At 3 questions an hour, is a five-minute prompt cache worth it? Explain with the numbers from block 1.</summary>

No. Questions arrive 20 minutes apart, so the five-minute cache expires every time and each question pays the 1.25 times write price. At 1 and 6 an hour block 1 shows the cached cost above the uncached cost. The same holds at 3.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> A needle test shows 100 percent recall at every length. List two reasons this may not predict your production accuracy.</summary>

The needle shares words with the question, and there are no look-alike distractors. NoLiMa removed the word overlap and RULER added harder tasks, and accuracy fell with length. Also, a test that asks for one fact tells you nothing about questions that need several facts or a summary of the whole input.

</details>

<details>
<summary><strong>Q5 (Medium).</strong> In block 2 the router at 0 percent answered costs more than long context alone. Why does that matter when designing one?</summary>

Every question pays for the RAG attempt before any fallback. If RAG rarely succeeds, you pay for both. A router is worth it only if the share RAG answers correctly is high enough to cover the cost of the wasted attempts on the rest.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> Why does the memory needed for the KV cache grow in a straight line with prompt length, while the time to read the prompt grows faster?</summary>

The cache stores a fixed number of bytes per token, so memory is proportional to tokens. Reading the prompt also involves attention, where each token looks at all earlier ones, and that part grows faster than a straight line. On this small model at up to 4,096 tokens the straight-line part dominates: 8 times the tokens took about 5.3 times as long in block 5. At much longer prompts the attention part grows in importance.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> Design a router for a 400,000-token legal corpus where 70 percent of questions are single-clause lookups and 30 percent compare clauses across many contracts. What would you route where, and what would you measure?</summary>

Send single-clause lookups to RAG, since the answer sits in one or two chunks. Send comparison questions to long context, or to a graph or summary index, since they need breadth. Classify the question first with a cheap model or rules. Measure the classifier's accuracy, RAG's correctness on lookups, the cost per question of each route, and how often RAG answers a comparison question wrongly with high confidence.

</details>

## Go deeper

All sources opened on 7 October 2026.

- Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), arXiv 2307.03172, TACL, submitted 6 July 2023, final version 20 November 2023.
- Hsieh et al., [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), arXiv 2404.06654, submitted 9 April 2024, revised 6 August 2024.
- Modarressi et al., [NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167), arXiv 2502.05167, ICML 2025, latest version 9 July 2025.
- Chroma, [Context Rot](https://www.trychroma.com/research/context-rot), 14 July 2025.
- Li et al., [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach](https://arxiv.org/abs/2407.16833), arXiv 2407.16833, EMNLP 2024 industry track, submitted 23 July 2024, revised 17 October 2024.
- Anthropic, [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), 19 September 2024 (the 200,000-token rule of thumb).
- Anthropic, [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) and [Pricing](https://platform.claude.com/docs/en/about-claude/pricing), read 7 October 2026.
- [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model card (360M parameters, Apache 2.0).
- On this site: [RAG basics](/docs/genai/rag), [the KV cache](/docs/llm-engineering/kv-cache-and-paged-attention), [the enterprise document-QA design](/docs/senior/design-enterprise-document-qa).

## Check yourself

- I can compute the cost of a long-context question and a RAG question from token counts and prices.
- I can say when a prompt cache helps and when it makes things worse.
- I can explain what a needle test shows and two ways it flatters a model.
- I can describe lost in the middle and where I would place retrieved chunks because of it.
- I can price a RAG-first router and say what share RAG must answer for it to pay.
- I can explain why prompt length costs memory and time on the machine.

## Where to go next

Next: [text-to-SQL and structured RAG](/docs/genai/rag-advanced/text-to-sql-and-structured-rag), for questions whose answers sit in tables. For the previous chapter, see [GraphRAG](/docs/genai/rag-advanced/graphrag-and-knowledge-graphs).
