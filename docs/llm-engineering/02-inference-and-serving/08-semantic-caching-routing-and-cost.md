---
id: llme-cost-latency
title: "Prompt Caching, Semantic Caching, Routing and Cost"
sidebar_label: "8 · Caching, routing and cost"
sidebar_position: 8
slug: /llm-engineering/semantic-caching-routing-and-cost
description: "The four levers that cut LLM spend and latency without touching the model: provider prompt caching, semantic caches and their threshold risk, model routing and cascades, and batch APIs, each measured with seeded simulations and a real embedding model."
tags: [prompt-caching, semantic-cache, routing, cascade, batch-api, cost, latency, token-budget]
---

import Infographic from '@site/src/components/Infographic';
import CacheRoutingLab from '@site/src/components/viz/CacheRoutingLab';

**In one line.** Before you buy more GPUs or a smaller model, four cheaper levers exist (reuse a repeated prefix, reuse a repeated answer, send easy requests to a cheap model, and defer work that can wait), and each one has a failure mode that you can measure before you ship it.

:::note Not from a lecture
This chapter is written for this site from the provider documentation pages, papers and product pages listed under Further reading, opened on 2 October 2026. Prices change often, so the cost numbers here are **multipliers and relative units**, never dollar amounts. Provider details were read from the pages on that date and will drift.
:::

## The idea in plain words

An LLM bill and an LLM latency have the same shape: you pay for tokens, in and out, and you wait for them. Four levers change that shape without changing the model.

| Lever | What it reuses or defers | Who does the work | Main risk |
| --- | --- | --- | --- |
| **Prompt (prefix) caching** | The model's processing of a repeated start of the prompt | The provider or engine | Savings vanish if the prefix changes or the cache expires |
| **Semantic caching** | A whole previous answer, for a *similar* question | You, in front of the model | A wrong answer served confidently |
| **Routing and cascades** | The cheap model's answer, when it is good enough | You, or a router model | Sending a hard request to the cheap model |
| **Batch APIs** | Time: results later, at a discount | The provider | Latency measured in hours |

A fifth, quieter lever is the **token budget**: capping output length, trimming context and keeping system prompts short. It needs no infrastructure and often saves more than any of the four.

<Infographic src="/img/llme/semantic-caching-routing-and-cost-levers.svg" alt="Two panels: the cost of reusing a prompt prefix under a 1.25 times write and 0.1 times read multiplier, and the semantic-cache threshold table with hit rate and wrong answers." caption="The first two levers with the numbers the code below prints: a prefix cache write repays itself from the second request, and a semantic cache stops paying once a wrong answer costs ten model calls." />

<Infographic src="/img/llme/semantic-caching-routing-and-cost-cascade.svg" alt="A cascade where a small model answers first and a confidence gate escalates to a large model, with a table of accuracy and cost against the confidence threshold from a seeded simulation." caption="A two-model cascade in a seeded simulation: a confidence gate at 0.5 reaches the large model's accuracy at about a third of its cost, but only because the simulated confidence is informative." />

## How it works

### Prompt and prefix caching

When a request starts with the same tokens as an earlier one, the work of processing those tokens (the prefill, and the KV cache it produces; see [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention)) can be reused. Hosted providers expose this as a feature with its own rules. What the three documentation pages I opened say:

| | Anthropic | OpenAI | Google Gemini |
| --- | --- | --- | --- |
| How it starts | `cache_control` on the request or on content blocks | Enabled by default for supported models | Implicit caching on by default for 2.5 and newer; explicit caching also exists |
| Lifetime | 5 minutes by default, 1 hour at a higher write price | 30 minutes by default on the newest models; earlier models offer in-memory or 24 hour retention | No default stated in what I read |
| Write price | 1.25 times base input (5 minute), 2 times (1 hour) | Not covered in the part I read | Not covered in the part I read |
| Read price | 0.1 times base input on most models; lower on a few | 0.1 times on most models; lower on one | Savings "passed on" when a request hits |
| Minimum prefix | 512 to 4,096 tokens depending on model; shorter prompts are simply not cached, with no error | 1,024 tokens on the newest models | Differs by model; 2,048 and 4,096 appear |
| Visible in the response | `cache_creation_input_tokens`, `cache_read_input_tokens` | `cached_tokens` | `total_cached_tokens` |

Pages: [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching). Anthropic's page also fixes the **order** of a cacheable prompt: tools, then system, then messages, and a change at any level invalidates that level and everything after it. OpenAI's page says reuse needs the entire rendered prefix to match. The design rule is the same for all three: **put the stable material first and the variable material last**.

The arithmetic matters, because a cache write costs *more* than a plain request. With a write multiplier of 1.25 and a read multiplier of 0.1, the first request costs 1.25 units instead of 1, and every later request that reuses the prefix costs 0.1 instead of 1. That only pays off if requests arrive before the cache expires, which is a question about **traffic**, not about the prompt. The first code block works it out.

For caching at the application-framework level (an exact-match LLM response cache inside LangChain), see [LangChain caching](/docs/genai/langchain-advanced/caching).

### Semantic caching and its threshold

A **semantic cache** sits in front of the model. It embeds each incoming question, finds the nearest stored question, and returns the stored answer if the cosine similarity reaches a **threshold**. Redis's LangCache documentation states the trade plainly: it matches by similarity, so *What are Product A's features?* and *Tell me about Product A's capabilities* can share a cached response, but it "gives up the guarantee that a cache hit is always correct". It defaults to a threshold of 0.85 with a recommended starting range of 0.8 to 0.9, says no single value is right for every use case, and advises starting tight and loosening gradually.

The risk is a **false hit**: two questions with high similarity and different answers, such as "What time does the store open?" and "What time does the store close?". The threshold cannot separate meaning that embeddings place close together. The second code block measures this with a real embedding model.

### Routing and cascades

Two different designs often share the name "routing".

- A **router** decides *before* running anything which model gets the request, from the request alone. Its cost is the router call plus the chosen model.
- A **cascade** runs the cheap model first, judges the answer (by a confidence score or a verifier) and escalates to the large model only if the answer is not trusted. Its cost is the cheap call plus the large call for the escalated share.

Two papers frame the idea. **FrugalGPT** (Chen, Zaharia and Zou, 2023) proposes prompt adaptation, LLM approximation and LLM cascade, and reports that it can match the best individual model "with up to 98% cost reduction" on the tasks studied, or improve accuracy by 4 per cent at the same cost. **RouteLLM** (Ong and colleagues, 2024) trains routers on human-preference data to choose between a strong and a weak model, and reports cost reductions of over 2 times in certain cases without compromising response quality. Both numbers are the authors' results on their own benchmarks; your gain depends on how much of your traffic is easy and how good your confidence signal is.

### Batch APIs

If a request can wait, a batch interface trades latency for price. Anthropic's Message Batches API charges 50 per cent of standard prices, allows 100,000 requests or 256 MB per batch whichever comes first, and says most batches finish within an hour while results are available once all requests finish or after 24 hours, whichever comes first; batches expire if not done within 24 hours. OpenAI's Batch API gives a 50 per cent discount with a 24 hour completion window, up to 50,000 requests and 200 MB per file, and a separate rate-limit pool. Good fits are evaluation runs, labelling, embeddings backfills and nightly summaries ([Anthropic batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing), [OpenAI batch](https://developers.openai.com/api/docs/guides/batch)). Whether a batch discount combines with prefix caching is provider-specific, and I did not verify it.

### Token budgets

Count what you send. Trim retrieved context to what the answer needs, drop stale conversation turns, cap `max_tokens` to the longest answer you accept, and ask for terse output where terse is enough. Every token removed is saved on every request, with no hit rate to tune and no cache to invalidate.

## A real system that works this way

Redis's **LangCache** is a hosted semantic cache whose documentation is unusually candid about the correctness trade. Its flow is the one above: a prompt is checked against a similarity threshold; a match returns the cached response, a miss calls the LLM and stores the prompt and response as a new entry. The documentation warns that a close-enough prompt can match even when it is not equivalent, lets you inspect how close a matched entry really was, and gives the advice a careful engineer would: start with a tighter threshold, loosen gradually while monitoring hit rate, and spot-check matches rather than trying to catch bad matches afterwards. The other real system is the providers' prompt-caching features documented in the table above: Anthropic, OpenAI and Google all expose the cached-token count in the response, which is what makes the savings auditable.

## Code you can run

The first block is the arithmetic of prefix caching. Prices are multipliers of one uncached prefix, taken from the Anthropic page (1.25 for a 5-minute write, 2.0 for a 1-hour write, 0.1 for a read). It then simulates Poisson arrivals with exponential gaps to see what happens when traffic is sparse. Assumption: a cache entry is refreshed when it is read, as OpenAI's page describes ("eligible for reuse for 30 minutes after its most recent write or reuse") and Anthropic's page implies by pricing "hits and refreshes" together.

```python
import numpy as np

WRITE_5M = 1.25
WRITE_1H = 2.0
READ = 0.1


def cost_with_cache(requests, write=WRITE_5M, read=READ):
    return write + (requests - 1) * read


print("requests  uncached  5-minute cache  1-hour cache   (units of one uncached prefix)")
for n in (1, 2, 3, 5, 10, 50):
    print(f"{n:>8}  {n:>8.2f}  {cost_with_cache(n):>14.2f}  {cost_with_cache(n, WRITE_1H):>12.2f}")

for label, write in (("5-minute", WRITE_5M), ("1-hour", WRITE_1H)):
    n = 1
    while cost_with_cache(n, write) >= n:
        n += 1
    print(f"{label} write pays for itself from request number {n}")


def simulate(mean_gap_minutes, ttl_minutes, write, read=READ, requests=20000, seed=0):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_gap_minutes, requests)
    total = write
    hits = 0
    for gap in gaps[1:]:
        if gap <= ttl_minutes:
            total += read
            hits += 1
        else:
            total += write
    return total / requests, hits / (requests - 1)


print("mean gap between requests (min)  hit rate  cost per request, 5-minute TTL  1-hour TTL")
for gap in (0.5, 2, 5, 15, 60, 240):
    c5, h5 = simulate(gap, 5, WRITE_5M)
    c60, _ = simulate(gap, 60, WRITE_1H)
    print(f"{gap:>31}  {h5:>8.2f}  {c5:>30.2f}  {c60:>10.2f}")
```

Reading the output. Five requests sharing a prefix cost 5.00 uncached and 1.65 with a 5-minute cache; fifty cost 50.00 and 6.15. The 5-minute write pays for itself from the second request, the 1-hour write (2.0 up front) from the third. The simulation is the sobering part. When requests arrive about every 2 minutes, the hit rate is 0.92 and a request costs 0.19 of an uncached one. At a mean gap of 15 minutes the 5-minute cache hits only 0.29 of the time and costs 0.92. At a mean gap of 60 minutes, the hit rate is 0.08 and the cost 1.16: **caching made each request more expensive than not caching**, because most requests pay the 1.25 write and never get a read. At 240 minutes the 5-minute cache costs 1.23 and the 1-hour cache 1.58, worse still. The right lifetime follows your inter-arrival time.

The second block measures semantic caching with a real embedding model, `sentence-transformers/all-MiniLM-L6-v2`. It uses 48 hand-written questions: eight intents (store opening, store closing, reset password, change password, cancel order, track order, refund timing, return an item) with six phrasings each, deliberately including four **confusable pairs**. A hit is *correct* only when the cached question has the same intent as the new one. The experiment replays 300 random orders of the 48 questions against an empty cache for each threshold, so every question is seen once per replay and exact repeats are excluded (an exact-match cache would handle those). It is a toy workload: 48 questions are enough to show the shape, not to set your threshold.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

INTENTS = {
    "opens": ["What time does the store open?", "When do you open in the morning?", "What are your opening hours?",
              "At what hour does the shop open?", "How early can I come in?", "Opening time please"],
    "closes": ["What time does the store close?", "When do you close in the evening?", "What is your closing time?",
               "At what hour does the shop shut?", "How late are you open?", "Closing time please"],
    "reset": ["How do I reset my password?", "I forgot my password, how can I reset it?", "Steps to reset a forgotten password",
              "Where do I go to reset my login password?", "Password reset instructions please", "Can't remember my password, help"],
    "change": ["How do I change my password?", "I want to change my password to a new one", "Steps to update my current password",
               "Where can I change my account password?", "Change password instructions please", "How to set a different password"],
    "cancel": ["How do I cancel my order?", "I want to cancel the order I just placed", "Steps to cancel an order",
               "Can I cancel an order before it ships?", "Cancel my order please", "How can I stop my order"],
    "track": ["How do I track my order?", "Where is my order right now?", "Steps to track a shipment",
              "Can I see the delivery status of my order?", "Track my order please", "How can I follow my parcel"],
    "refund": ["How long does a refund take?", "When will I get my refund?", "How many days until the refund arrives?",
               "What is the refund processing time?", "Refund timing please", "How soon is the money returned"],
    "return": ["How do I return an item?", "I want to send a product back", "Steps to return a purchase",
               "Where do I post a return?", "Return instructions please", "How can I give back an item"],
}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
names = list(INTENTS)
texts = [q for n in names for q in INTENTS[n]]
labels = np.array([i for i, n in enumerate(names) for _ in INTENTS[n]])
emb = model.encode(texts, normalize_embeddings=True)
sims = emb @ emb.T
same = labels[:, None] == labels[None, :]
off_diag = ~np.eye(len(texts), dtype=bool)
print("same intent: mean cosine", f"{sims[same & off_diag].mean():.3f}", "min", f"{sims[same & off_diag].min():.3f}")
print("different intent: mean cosine", f"{sims[~same].mean():.3f}", "max", f"{sims[~same].max():.3f}")
for a, b in (("opens", "closes"), ("reset", "change"), ("cancel", "track"), ("refund", "return")):
    ia, ib = names.index(a), names.index(b)
    block = sims[np.ix_(labels == ia, labels == ib)]
    print(f"confusable pair {a}/{b}: mean cosine {block.mean():.3f}, max {block.max():.3f}")

rng = np.random.default_rng(0)
orders = [rng.permutation(len(texts)) for _ in range(300)]

COST_LLM, COST_EMBED = 1.0, 0.02
TAUS = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95)
tally = {}
for tau in TAUS:
    hits = wrong = total = 0
    for order in orders:
        cache = []
        for i in order:
            total += 1
            if cache:
                j = max(cache, key=lambda c: sims[i, c])
                if sims[i, j] >= tau:
                    hits += 1
                    wrong += labels[j] != labels[i]
                    continue
            cache.append(i)
    tally[tau] = (hits / total, wrong / total, wrong / max(hits, 1))

print("threshold  hit rate  wrong share of hits  wrong answers per request  cost if wrong = 10")
for tau, (hit, wrong_all, share) in tally.items():
    print(f"{tau:>9.2f}  {hit:>8.4f}  {share:>19.3f}  {wrong_all:>25.4f}  {COST_EMBED + (1 - hit) * COST_LLM + wrong_all * 10:>18.3f}")

print("cost of a wrong answer (in LLM calls)  cheapest threshold  cost per request  (no cache = 1.000)")
for cost_wrong in (1, 2, 5, 10, 50):
    costs = {t: COST_EMBED + (1 - h) * COST_LLM + w * cost_wrong for t, (h, w, _) in tally.items()}
    t = min(costs, key=costs.get)
    print(f"{cost_wrong:>38}  {t:>17.2f}  {costs[t]:>17.3f}")
```

What it printed. Two phrasings of the *same* intent had a mean cosine of 0.685 but a minimum of only 0.321, so some true paraphrases look unrelated to the embedding model. Phrasings of *different* intents had a mean of just 0.213, but a **maximum of 0.903**, and that maximum is a pair from the opens/closes confusable set (mean 0.580, max 0.903). The reset/change pair reached 0.874. Cancel/track topped out at 0.602 and refund/return at 0.524, so some confusable pairs are safe and some are not, and you cannot tell which in advance without measuring.

The threshold table: at 0.55 the hit rate is 0.7548, but 28 per cent of hits are wrong (0.2115 wrong answers per request). At 0.80 the hit rate is 0.3284 and 16.6 per cent of hits are wrong. At 0.90 it is 0.0896 with 23.3 per cent wrong, and at 0.95 there are no hits at all. The wrong share does not fall smoothly with the threshold (0.274 at 0.85 against 0.166 at 0.80); with 48 questions that is sampling noise, and it is exactly the kind of noise your production traffic will have too.

Then the economics. With a wrong answer costing 1 LLM call, the best threshold is 0.55 at 0.477 per request, less than half the no-cache cost of 1.000. At a wrong-answer cost of 2 calls the best is 0.65 at 0.651; at 5 it is 0.80 at 0.965, barely better than no cache. At 10 or 50 calls, the cheapest option is a threshold of 0.95, where nothing is ever served from the cache and the cost is 1.020: **the cache only adds its embedding cost**. The honest conclusion: whether a semantic cache is worth it depends on what a wrong answer costs you, which is a business number and not a tuning parameter.

The third block simulates a cheap model, an expensive model and two ways of choosing between them. Costs are relative (cheap call 1, large call 15, router call 0.1; these are parameters, not prices). Each of 20,000 requests has a difficulty; the cheap model is more likely correct on easy requests. The **cascade** accepts the cheap answer when a confidence score is at least *t*; the confidence is built to carry real information about whether the cheap answer was right. The **router** predicts difficulty from the request with noise and sends hard-looking requests straight to the large model.

```python
import numpy as np

rng = np.random.default_rng(0)
N = 20000
COST_SMALL, COST_LARGE, COST_ROUTER = 1.0, 15.0, 0.1

difficulty = rng.beta(2, 2, N)
p_small = 1 / (1 + np.exp(8 * (difficulty - 0.7)))
p_large = 1 / (1 + np.exp(8 * (difficulty - 1.0)))
small_ok = rng.random(N) < p_small
large_ok = rng.random(N) < p_large
confidence = np.clip(0.5 * small_ok + 0.5 * (1 - difficulty) + rng.normal(0, 0.12, N), 0, 1)
router_score = np.clip(difficulty + rng.normal(0, 0.15, N), 0, 1)

print(f"small alone: accuracy {small_ok.mean():.3f}, cost {COST_SMALL:.2f}")
print(f"large alone: accuracy {large_ok.mean():.3f}, cost {COST_LARGE:.2f}")
print(f"oracle (large only when small is wrong): accuracy {(small_ok | large_ok).mean():.3f}, "
      f"cost {COST_SMALL + (~small_ok).mean() * COST_LARGE:.2f}")

print("cascade: accept the small answer when confidence >= t")
print("     t  escalated  accuracy  cost per request")
for t in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8):
    accept = confidence >= t
    correct = np.where(accept, small_ok, large_ok)
    cost = COST_SMALL + (~accept).mean() * COST_LARGE
    print(f"{t:>6.1f}  {(~accept).mean():>9.3f}  {correct.mean():>8.3f}  {cost:>16.2f}")

print("router: send to the large model when predicted difficulty >= d, before running anything")
print("     d  to large  accuracy  cost per request")
for d in (0.3, 0.4, 0.5, 0.6, 0.7):
    to_large = router_score >= d
    correct = np.where(to_large, large_ok, small_ok)
    cost = COST_ROUTER + to_large.mean() * COST_LARGE + (~to_large).mean() * COST_SMALL
    print(f"{d:>6.1f}  {to_large.mean():>8.3f}  {correct.mean():>8.3f}  {cost:>16.2f}")
```

Output: the small model alone scores 0.734 at cost 1.00, the large model alone 0.945 at 15.00, and an oracle that calls the large model only when the small one is wrong would reach 0.966 at 4.99. The cascade at confidence 0.5 escalates 0.288 of requests and reaches **accuracy 0.961 at cost 5.33**, above the large model alone at about a third of its cost. At 0.2 it escalates 0.171 and reaches 0.877 at 3.56; at 0.8 it escalates 0.656 and the cost climbs to 10.85 for no accuracy gain (0.949). The router is weaker in this simulation: at *d* = 0.5 it sends 0.505 of requests to the large model for 0.901 accuracy at cost 8.17, and even at *d* = 0.3 it reaches only 0.935 at 11.71. The difference is by construction: the cascade sees the cheap model's actual output while the router only sees a noisy guess of difficulty. **The cascade beats the large model on accuracy only because I gave it an informative confidence signal.** A real signal (log probabilities, a verifier, agreement between samples) is usually weaker, and measuring it on labelled traffic is the whole job.

<CacheRoutingLab />

The lab's default (semantic mode, threshold 0.80, wrong answer costing 10) shows a cost of about 1.24 against the 1.238 printed above (the lab uses the table rounded to four decimals); switch the lever to see the prefix-cache and cascade numbers.

## Designing with it

1. **Cut tokens first.** A shorter prompt and a capped output are free and need no hit rate.
2. **Order prompts stable-first.** System text, tool definitions and long documents come before the user's question, so a prefix cache can match.
3. **Match cache lifetime to your inter-arrival time.** Measure the gap distribution of real traffic; the simulation above shows caching can cost more than not caching when most requests arrive after the entry expired.
4. **Treat a semantic cache as a correctness decision.** Put a number on a wrong answer, label a few hundred real queries including confusable pairs, and choose the threshold from a table like the one above. Scope the cache (per tenant, per intent type, per freshness class); do not cache anything personalised, account-specific or time-sensitive.
5. **Verify cascades on your labels.** Their gain is exactly the quality of the gating signal. Record escalation rate, accuracy and cost per request in production.
6. **Send deferrable work to the batch interface.** Evaluations, backfills and offline enrichment rarely need an answer in seconds.
7. **Log cache hits from the provider's own usage fields** so savings are audited, not assumed.

## Where this stands in 2026

:::info Industry view
All three major hosted providers whose pages I opened now offer prefix caching with a visible cached-token count, and two (Anthropic and OpenAI) offer a 50 per cent batch discount, which makes "cache and batch" part of ordinary cost control rather than an optimisation project. Semantic caching sits mostly in application and gateway layers, with vendors such as Redis documenting both the benefit and the risk of a wrong hit; the practical trend is to apply it to bounded, low-risk question sets such as FAQs. Routing research (FrugalGPT, RouteLLM) reports large savings on the authors' benchmarks, and production teams treat those as a design pattern to validate on their own traffic, not a number to expect.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A 5-minute cache write costs 1.25 and a read 0.1. After how many requests per cache lifetime does the write pay off?</summary>

From the second request: one write plus one read costs 1.35 against 2.00 uncached. The code prints that the 5-minute write pays for itself from request number 2 and the 1-hour write (2.0) from request number 3.

</details>

<details>
<summary><strong>Q2.</strong> At a mean gap of 60 minutes between requests, is a 5-minute prompt cache a good idea?</summary>

No. In the simulation the hit rate was 0.08 and the cost per request 1.16, above the uncached cost of 1.00, because most requests paid the write premium and never got a read. A 1-hour entry cost 0.80 at the same gap.

</details>

<details>
<summary><strong>Q3.</strong> Why did phrasings of different intents reach a cosine of 0.903?</summary>

Embedding models place questions with the same topic and structure close together ("What time does the store open?" and "...close?" differ by one word), but they have different answers. Similarity measures closeness of wording and topic, not equality of answer.

</details>

<details>
<summary><strong>Q4.</strong> When is a threshold of 0.95 the cheapest choice in the experiment, and what does that mean?</summary>

When a wrong answer costs 10 or 50 LLM calls. At 0.95 there are no hits, so the cache is switched off in effect and the cost is 1.020 (1.000 plus the embedding cost). It means the cache is not worth its risk at those stakes on this workload.

</details>

<details>
<summary><strong>Q5.</strong> Why does the cascade beat the large model alone in the simulation, and should you expect that in production?</summary>

Its confidence score was built to carry information about whether the small answer was right, so it escalates mostly the wrong ones. Real confidence signals are noisier, so measure the escalation rate and accuracy on labelled traffic before trusting the saving.

</details>

<details>
<summary><strong>Q6.</strong> What is the difference between a router and a cascade in cost terms?</summary>

A router pays one router call, then one model call chosen from the request alone. A cascade always pays the cheap call and pays the large call as well for the escalated share, but it decides with the cheap model's actual output in hand.

</details>

<details>
<summary><strong>Q7.</strong> Which work belongs in a batch API?</summary>

Work that can wait up to a day: evaluation runs, labelling, embedding backfills and nightly summaries. Both providers' pages give a 50 per cent discount and a 24 hour window.

</details>

## Further reading

All opened on 2 October 2026.

- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and [Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing).
- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) and [Batch API](https://developers.openai.com/api/docs/guides/batch).
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching).
- [Redis LangCache concepts](https://redis.io/docs/latest/develop/ai/context-engine/langcache/concepts/) for semantic caching and the threshold trade-off.
- Chen, Zaharia and Zou, [FrugalGPT](https://arxiv.org/abs/2305.05176), 2023.
- Ong and colleagues, [RouteLLM](https://arxiv.org/abs/2406.18665), 2024.
- On this site: [LangChain caching](/docs/genai/langchain-advanced/caching), [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention) and [continuous batching and scheduling](/docs/llm-engineering/continuous-batching-and-scheduling).

## Check yourself

- I can explain why a prefix cache write costs more than a plain request and when it pays back.
- I can say what changes a prefix cache entry's usefulness: prompt order, exact match and the gap between requests.
- I can explain how a semantic cache can return a wrong answer and why a similarity threshold cannot fully prevent it.
- I can choose a threshold from a measured table and a stated cost of a wrong answer, and I know when the answer is "do not cache".
- I can distinguish a router from a cascade and say what a cascade's saving depends on.
- I can pick the work that belongs in a batch API and name the discount and window the providers state.
