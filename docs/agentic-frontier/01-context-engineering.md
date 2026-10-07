---
id: afr-context-engineering
title: "Context Engineering for Agents"
sidebar_label: "1 · Context engineering"
sidebar_position: 1
slug: /agentic-frontier/context-engineering
description: "Treat the context window as a budget: count what fills it, then trim, mask, compact, remember and lay out the prompt for the cache, with a real tokenizer, a 24-step agent experiment and an honest account of which saving is not a saving."
tags: [context-engineering, context-window, prompt-caching, compaction, agents, tokens, tiktoken]
---

import Infographic from '@site/src/components/Infographic';
import ContextWindowLab from '@site/src/components/viz/ContextWindowLab';

**In one line.** Context engineering is deciding, on every request, which tokens deserve a place in the model's window and in what order, because the window is finite, every token costs money and time, and an agent that keeps everything slowly drowns in its own history.

:::note Not from a lecture
Written for this site from the sources under Go deeper, all opened on 7 October 2026. The agent run in the code is synthetic and the "model" in it is a deterministic stand-in, stated plainly where it matters. Tokens are counted with `tiktoken` 0.14.0 and the `o200k_base` encoding on Python 3.14.6. Other providers use other tokenizers, so treat the counts as a faithful sketch, not as your own bill.
:::

:::tip Before you start
You should already know:

- what a token and a context window are, and why a chat model is stateless ([LLM memory](/docs/agentic-ai/llm-memory));
- what an agent loop is: a model that calls a tool, reads the result and decides again ([What is agentic AI?](/docs/agentic-ai/what-is-agentic-ai));
- roughly what the KV cache is ([KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention)).

Reading time: about 35 minutes with the code.

After this chapter you can:

- count what fills a request and find the biggest fixed cost;
- choose between trimming, masking, compacting and remembering for a long agent run;
- lay out a prompt so the provider's prefix cache keeps working, and say when a "saving" costs more.
:::

## In 30 seconds

A chat model has no memory. Each time an agent takes a step, your program sends the whole story again: the rules, the tool list, every earlier step and every tool result. A coding agent that reads files can add a thousand tokens per step, so by step 24 you are sending 24,000 tokens, and you have paid for the early ones 24 times.

Think of a desk. You can only work with what fits on it. A good assistant keeps the rules and the current task in front of you, files old paperwork in a drawer with a one-line label, and writes down the one fact you will need later on a sticky note. Context engineering is the habit of managing that desk on purpose.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Token | A chunk of text the model reads, often a short word or part of one | "refunds" may be 2 tokens |
| Context window | The most tokens the model can read in one request, input and output together | a 128,000-token window |
| Context engineering | Choosing what goes into each request and in what order | drop old tool output, keep the plan |
| Tool result | Text a tool sends back to the model | 1,200 tokens of file contents |
| Masking | Replacing an old tool result with a short stub | `[output removed, 1,675 tokens]` |
| Compaction | Replacing many old steps with a summary | 20 steps become 3 lines |
| Prefix cache | The provider reusing work on the start of a prompt it has seen before | same first 20,000 tokens, cheap |
| Cache hit | The part of a request found in the cache | 91.6% of tokens in the 24-step experiment |

## The idea in plain words

Start with a person. You ask a colleague to find why refunds are off by one cent. Over an afternoon they open files, run tests and read logs. A good colleague does not photocopy every page they have ever read and spread it on the desk. They keep a short list of what they have learned, and they look things up again when needed.

A model-based agent has the opposite habit by default. After each tool call, the result is appended to the conversation, and the whole conversation goes back to the model on the next step. Nothing is forgotten, so nothing is cheap. Three bad things grow together: the **cost** (you pay for the history again each step), the **delay** (long inputs take longer to read), and the **quality** (more text means more distractors, and the model attends less well to any one fact).

The smallest example makes the first one concrete. Suppose the fixed part of a request (rules and tool list) is 200 tokens, and every step adds 200 more. Request 1 sends 400 tokens, request 2 sends 600, request 3 sends 800. After eight steps you have sent 400 + 600 + ... + 1,800 = 8,800 tokens in total, although the story itself is only 1,800 tokens long. The growth is not a bug. It is how stateless APIs work.

Now generalise. With n steps of size s on a fixed part f, the total sent is about n times f plus s times n(n+1)/2. In words: the total cost grows with the square of the number of steps, so doubling the length of a run roughly quadruples the bill. Every technique in this chapter attacks one of the three bad things.

<Infographic src="/img/afr/context-window-anatomy.svg" alt="A stacked bar showing that for twenty tools tool definitions are 1,060 of 1,208 tokens in an almost empty request, above six cards describing six levers with measured numbers." caption="Look at the bar first: before the conversation starts, tool definitions are already 87.7% of the request. The six cards are the levers this chapter measures." />

## Worked example, step by step

Take eight requests with these rules. The fixed part is 200 tokens. A full step (call, result and a short note) adds 200 tokens. A masked step, where the result has been replaced by a stub, adds only 30. The window is 1,000 tokens. Three strategies:

1. **Keep all.** Context after step n is 200 + 200n. At n = 4 it is exactly 1,000. At n = 5 it is 1,200, over the window. The total over eight requests is 8,800 and the peak is 1,800.
2. **Mask old.** Keep the last 2 results in full, stub the rest. At n = 8 the context is 200 + 6 × 30 + 2 × 200 = 780. The sizes are 400, 600, 630, 660, 690, 720, 750, 780, the total is 5,230 and the peak is 780.
3. **Slide.** Drop whole old steps until the request fits in 1,000. The sizes are 400, 600, 800, 1,000, then 1,000 for the rest. The total is 6,800.

Masking sends the fewest tokens. Now add the provider's prefix cache and price it. Say a cached token costs 0.10 and a freshly written one costs 1.25 (placeholder ratios, discussed below). Request 1 writes everything: 400 × 1.25 = 500 units.

- **Keep all, request 2.** The first 400 tokens match the cache, 200 are new: 400 × 0.10 + 200 × 1.25 = 40 + 250 = 290 units. Each later request costs a little more, and the eight requests add up to 2,950 units.
- **Mask old, request 3.** Step 1 is rewritten as a stub, so the text no longer matches the cache after the fixed 200 tokens. Only 200 tokens hit: 200 × 0.10 + 430 × 1.25 = 20 + 537.5 = 557.5 units. Over eight requests the cost is 4,180 units.

So with this cache price the strategy that sends the most tokens costs the least, but it breaks the window. The strategy that sends the fewest tokens costs 42% more than keeping everything. The first code block reproduces every number above.

<Infographic src="/img/afr/context-window-worked-example.svg" alt="A table of tokens sent in each of eight requests under keep all, mask old and slide, next to the cache arithmetic that makes keep all cost 2,950 units and mask old 4,180." caption="Read the left table first for token counts, then the right-hand cards: the cache turns the ranking upside down." />

## How it works

### What actually fills the window

A request is a stack of parts: the system prompt (the rules), the tool definitions (names, descriptions and JSON schemas of every tool the agent may call), memory the program injects, the conversation history, the tool results, and the newest message. Two of these are fixed for the whole run and are paid on every request. The tool definitions are usually the larger.

In the second code block, four tools cost 362 tokens in all, ten cost 680 and twenty cost 1,208. A tool definition is about 53 tokens, so an agent with 20 tools spends 88% of an almost empty request on the list of things it could do. Giving the agent only the tools the current task needs is the cheapest saving in this chapter.

### Why more context can make answers worse

There are two separate reasons. The first is cost and time, which grow with length as above. The second is quality. Liu and colleagues showed in "Lost in the Middle" that models were often best at using information placed at the start or end of a long input and worse when it sat in the middle. Chroma's July 2025 report "Context Rot" tested 18 models and found that performance became less reliable as input length grew, that a single distractor lowered accuracy, and that lower similarity between a question and its answer made the drop steeper.

Anthropic's engineering post of 29 September 2025 describes the same pressure as an "attention budget": each added token draws on a finite capacity to attend. The practical lesson is to treat context as scarce, not as free storage. These papers describe particular models and tests; they do not give a threshold you can reuse, so measure your own task.

### Trimming a single tool result

Most of the growth in an agent comes from tool results: a file, a search page, a test log. The least risky edit is to cap a result before it enters the context, keeping the head and the tail and marking the gap, because errors and summaries tend to sit at the ends. In the fourth code block one 1,233-token result becomes 508 tokens this way.

### Masking: stubs for old results

Once a result has been read and acted on, the agent rarely needs all of it again. **Masking** replaces it with a stub that says what was there, for example `[output removed, 1,675 tokens]`, and keeps the assistant's own words about it. JetBrains researchers compared this with asking a model to summarise old steps, using the SWE-agent scaffold on the SWE-bench Verified benchmark across five model setups. Their paper (arXiv 2508.21433) reports that simple masking roughly halves cost against the raw agent while matching the solve rate of summarisation, and that a hybrid was 7% and 11% cheaper than masking and summarising alone. That result is about tokens billed. The next section shows why a bill can behave differently.

### Compaction: summarise in batches

**Compaction** replaces many old steps with a short summary. It is more flexible than masking (a summary can carry a decision, not just a stub) but it costs a model call and can lose detail. Anthropic's post describes it as taking a conversation near the limit, summarising it and restarting the window with the summary. The word that matters in the experiment is *batches*: compact when a trigger is crossed, not every step.

### Remember: notes outside the window

Masking and compaction share a danger. They delete detail, and the agent only keeps what it wrote down. The cure is a habit: when a tool result contains a fact you will need, say it in your own next message, or write it to a notes file that is loaded back later. Anthropic calls this structured note-taking and Manus, an agent product, describes using the file system as external memory. In the experiment, the agent restates four of six facts, and exactly those four survive masking and summarising. The two it did not restate are lost. Memory features are covered in [LLM memory](/docs/agentic-ai/llm-memory) and [Memory in LangChain](/docs/genai/langchain-advanced/memory).

### Layout: keep the front of the prompt still

Providers cache the processing of the start of a prompt. Anthropic's prompt caching documentation says cache hits cost 0.1 times the base input price, five-minute writes cost 1.25 times and one-hour writes cost 2 times, and that the cache follows the order tools, then system, then messages, so changing a tool definition invalidates everything after it. OpenAI's documentation likewise recommends putting stable instructions first and reports cached input at a fraction of the normal price. Manus calls the cache hit rate the most important metric for a production agent and warns that even one different token, such as a timestamp in the system prompt, invalidates the cache from that token onward.

This is why edits near the front are expensive. Masking an old result changes text in the middle of the history, so everything after it must be written again. Compacting does the same, but only when it fires. The experiment measures this.

## Code you can run

Four blocks. Each runs on its own. They use `tiktoken` 0.14.0 (the first run downloads the encoding file), Python 3.14.6, and a seeded random generator, so every number below repeats exactly.

### 1. The worked example, reproduced

First we turn the by-hand example into code, so you can check your arithmetic against it. A request is a list of blocks, and the cache hit is the number of tokens in the blocks that match the previous request from the start.

```python
FIXED, FULL_STEP, MASKED_STEP, WINDOW, KEEP = 200, 200, 30, 1000, 2
READ, WRITE = 0.10, 1.25

def request(strategy, n):
    blocks = [("fixed", FIXED)]
    first = 0
    if strategy == "slide":
        while FIXED + (n - first) * FULL_STEP > WINDOW:
            first += 1
    for i in range(first, n):
        masked = strategy == "mask" and i < n - KEEP
        blocks.append((f"step{i + 1}{'m' if masked else ''}", MASKED_STEP if masked else FULL_STEP))
    return blocks

def shared(a, b):
    tokens = 0
    for x, y in zip(a, b):
        if x != y:
            break
        tokens += x[1]
    return tokens

for strategy in ("full", "mask", "slide"):
    sizes, spent, previous = [], 0.0, []
    for n in range(1, 9):
        blocks = request(strategy, n)
        size = sum(tokens for _, tokens in blocks)
        hit = shared(previous, blocks)
        spent += hit * READ + (size - hit) * WRITE
        sizes.append(size)
        previous = blocks
    print(f"{strategy:5s} sizes {sizes} peak {max(sizes)} billed {sum(sizes)} cost {spent:.0f} units")
```

**Reading the output.** `sizes` lists the tokens sent in each of the eight requests. `billed` is their sum, 8,800 for keeping everything, 5,230 for masking and 6,800 for sliding. `cost` prices the run with the cache: 2,950 units for keeping all, 4,180 for masking and 5,510 for sliding. These match the numbers worked by hand.

**Line by line.**

- `request` builds the list of blocks for request `n`. A masked step is a different block (`step3m`), so it does not match the unmasked block that was cached before.
- `shared` walks both lists in order and stops at the first block that differs. That is exactly how a prefix cache behaves: it can only reuse an unbroken start.
- `READ` and `WRITE` are the placeholder price ratios. Real ratios are in the provider documentation.

### 2. What fills a request

Now the same idea with a real tokenizer. We assemble a request with four, ten and twenty tools and count each part.

```python
import json
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
tokens = lambda text: len(enc.encode(text))

SYSTEM = (
    "You are a careful coding assistant working inside a Python repository. "
    "Read files before you edit them. Prefer small, reversible changes. "
    "Run the test suite after every change and report the exact failing test name. "
    "Never print secrets. When a tool output is long, note the one fact you need "
    "in your next message so it survives if the output is later removed."
)
NAMES = ["read_file", "search_code", "run_tests", "git_log", "list_dir", "edit_file", "run_shell",
         "open_issue", "comment_pr", "fetch_url", "query_db", "send_email", "create_branch",
         "diff_files", "format_code", "lint_file", "install_pkg", "read_logs", "set_env", "stop"]

def tool(name):
    return {"name": name, "description": f"Run the {name.replace('_', ' ')} operation and return its result as text.",
            "input_schema": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}

MEMORY = "Project notes: payments service, Python 3.12, tests in tests/, config in config/."
HISTORY = [{"role": "user", "content": "Refunds are off by one cent for some orders. Find out why."},
           {"role": "assistant", "content": "I will start by finding the refund code and the failing test."}]

for n_tools in (4, 10, 20):
    parts = {"system prompt": SYSTEM, "tool definitions": json.dumps([tool(n) for n in NAMES[:n_tools]]),
             "memory": MEMORY, "history": json.dumps(HISTORY), "current message": "Where does the rounding happen?"}
    total = sum(tokens(t) for t in parts.values())
    print(f"{n_tools} tools: total {total} tokens")
    if n_tools == 20:
        for name, text in parts.items():
            print(f"  {name:16s} {tokens(text):5d}  {tokens(text) / total:6.1%}")
```

**Reading the output.** The total grows from 362 to 680 to 1,208 tokens as tools are added, so each tool costs about 53 tokens. In the twenty-tool request, the tool definitions are 1,060 of 1,208 tokens, 87.7%. The conversation so far, memory and the new message together are under 12%.

**Line by line.**

- `tokens` encodes text with `o200k_base` and counts the pieces. Counting characters would be wrong, because tokens are not characters.
- `tool(name)` makes a small JSON schema. Real schemas are usually longer, so real tool lists cost more.
- `json.dumps(...)` turns the tool list into the text that would be sent, which is what the tokenizer should count.

### 3. Twenty-four agent steps, six strategies

The experiment. A synthetic coding agent hunts a refund bug in 24 steps. Each step is a tool call whose result is 374 to 1,675 tokens of text. Six facts are planted in the results. Four are marked `FINDING`, and the agent restates those in its next message. Two are incidental details (a line number, a table version) that the agent does not restate. The stand-in "model" is a rule: it can answer a question about a fact only if the fact's text is still in the last request. That keeps the comparison deterministic, and it measures one thing only, what survives.

```python
import random
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
tok = lambda text: len(enc.encode(text))
rng = random.Random(7)
WORDS = "ledger refund amount cents round tax rate order total currency region config retry limit queue batch".split()
FACTS = {3: "FINDING: the failing test is test_refund_rounding", 6: "FINDING: RETRY_LIMIT is read from config/payments.toml",
         9: "ledger.py line 212 rounds half up", 12: "FINDING: the bug appeared in commit 9f3c2ab",
         15: "tax table version 2024-07", 18: "FINDING: only the EU region is affected"}
KEYS = ["test_refund_rounding", "payments.toml", "line 212", "9f3c2ab", "2024-07", "EU region"]

STEPS = []
for n in range(1, 25):
    lines = [f"{i:4d}  " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))) for i in range(rng.randint(25, 110))]
    if n in FACTS:
        lines.append(FACTS[n])
    learned = FACTS.get(n, "").removeprefix("FINDING: ") if FACTS.get(n, "").startswith("FINDING") else ""
    STEPS.append({"call": f"[call tool step {n}]", "result": "\n".join(lines),
                  "note": f"I learned: {learned}." if learned else f"Step {n} done."})

PREFIX = "SYSTEM: coding assistant rules.\nTOOLS: read_file, search_code, run_tests, git_log\n"
WINDOW, KEEP, TRIGGER = 10_000, 4, 8_000

def render(step, masked):
    result = f"[output removed, {tok(step['result'])} tokens]" if masked else step["result"]
    return f"{step['call']}\n{result}\nassistant: {step['note']}\n"

def compose(steps, first=0, summary="", masked_upto=0):
    body = "".join(render(s, i < masked_upto) for i, s in enumerate(steps) if i >= first)
    return PREFIX + summary + body

def build(strategy, n, state):
    steps = STEPS[:n]
    if strategy == "full":
        return compose(steps)
    if strategy == "mask":
        return compose(steps, masked_upto=n - KEEP)
    if strategy == "slide":
        first = 0
        while tok(compose(steps, first)) > WINDOW:
            first += 1
        return compose(steps, first)
    if strategy == "batch":
        text = compose(steps, masked_upto=state["masked"])
        if tok(text) > TRIGGER:
            state["masked"] = n - KEEP
            text = compose(steps, masked_upto=state["masked"])
        return text
    text = compose(steps, state["first"], state["summary"])
    if tok(text) > TRIGGER:
        state["first"] = n - KEEP
        state["summary"] = "SUMMARY: " + " ".join(s["note"] for s in steps[:n - KEEP] if s["note"].startswith("I learned")) + "\n"
        text = compose(steps, state["first"], state["summary"])
    return text

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

READ, WRITE = 0.10, 1.25
print(f"{'strategy':12s} {'peak':>6s} {'billed':>7s} {'over':>5s} {'facts':>6s} {'hit':>6s} {'cost':>8s} {'per tok':>7s}")
for name, strategy, clock in [("full", "full", False), ("full + clock", "full", True), ("slide", "slide", False),
                              ("mask", "mask", False), ("batch mask", "batch", False), ("summary", "summary", False)]:
    state, prev, sizes, hits, spent, text = {"masked": 0, "first": 0, "summary": ""}, [], [], 0, 0.0, ""
    for n in range(1, 25):
        text = build(strategy, n, state)
        text = (f"Current time: 2026-10-07 09:{n:02d}:00\n" + text) if clock else text
        ids = enc.encode(text)
        hit = common_prefix(prev, ids)
        hits, spent, prev = hits + hit, spent + hit * READ + (len(ids) - hit) * WRITE, ids
        sizes.append(len(ids))
    kept = sum(1 for k in KEYS if k in text)
    print(f"{name:12s} {max(sizes):6d} {sum(sizes):7d} {sum(x > WINDOW for x in sizes):5d} {kept:4d}/6 "
          f"{hits / sum(sizes):6.1%} {spent:8.0f} {spent / sum(sizes):7.2f}")
```

**Reading the output.** Every row is a strategy over the same 24 steps.

- `peak` is the largest single request. Keeping everything reaches 24,383 tokens and exceeds the 10,000-token demo window in 13 requests (`over`). The other strategies stay inside it.
- `billed` is the sum of all 24 request sizes. Keeping everything bills 289,768 tokens, although the longest request is only 24,383.
- `facts` counts how many of the six planted facts are in the final request. Keeping all has 6, masking and summarising have 4, and the sliding window has 1.
- `hit` is the share of billed tokens that matched the previous request from the start. `cost` prices the run at read 0.10 and write 1.25, and `per tok` divides cost by billed tokens, so 1.00 would mean no saving from caching.

**What surprised me.** Three things. First, masking at every step sends the fewest tokens (96,328) but costs 107,548 units, nearly double keeping everything (57,017), because it rewrites the tail of the prompt each step and only 11.6% of tokens hit the cache. Second, masking in batches, a small change, cuts the cost to 59,552 units, within 2% of the summarising run's 58,721. Third, putting a clock at the top of an otherwise identical prompt turns a 91.6% hit rate into 0.1% and the cost from 57,017 to 362,376 units, more than six times, although the tokens billed barely change (290,176).

The result does not say masking is bad. JetBrains' paper measured cost in tokens, and in tokens masking wins here too. It says that when a provider discounts cached input heavily, an edit that breaks the cache can cost more than the tokens it removes, so the ranking depends on the price ratio. The lab below lets you change that ratio.

**Line by line.**

- `render` writes one step, and replaces its result with a stub when `masked` is true. The stub keeps the token count so the agent knows what it dropped.
- `compose` builds a request from a first step, an optional summary and a point before which results are masked. The four strategies are different calls to it.
- `build` holds the strategy logic. The `batch` and summary branches change state only when the request passes `TRIGGER` (8,000), then keep the last `KEEP` (4) steps in full.
- `common_prefix` compares token ids, not characters, because the cache works on tokens.
- The `clock` row prepends a changing time string. Only that line differs from `full`.

**What this does not show.** The facts are planted and the model is a rule, so the experiment says nothing about how well a real model uses a summary. The cache is ideal: no minimum prompt length, no block sizes, no five-minute expiry. The price ratios are placeholders taken from the documented 0.1 and 1.25 for illustration, not a quote for any model.

### 4. Packing a budget

Last, the simplest honest policy for one request: a fixed budget, always keep the rules and memory, take retrieved passages by score up to 40% of the budget, then fill the rest with the most recent history, trimming each tool result.

```python
import random
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
tok = lambda text: len(enc.encode(text))
rng = random.Random(11)
WORDS = "ledger refund amount cents round tax rate order total currency region config retry limit queue batch".split()

def passage(lines):
    return "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))) for _ in range(lines))

def trim(text, limit):
    ids = enc.encode(text)
    if len(ids) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return enc.decode(ids[:head]) + f"\n[... {len(ids) - head - tail} tokens omitted ...]\n" + enc.decode(ids[-tail:])

FIXED = "You are a careful coding assistant. Read before you edit. Run the tests after every change."
MEMORY = "Project notes: payments service, Python 3.12, tests in tests/."
RETRIEVED = [(0.91, passage(30)), (0.84, passage(45)), (0.77, passage(60)), (0.52, passage(40))]
HISTORY = [passage(rng.randint(25, 110)) for _ in range(24)]

def pack(budget):
    used = tok(FIXED) + tok(MEMORY)
    fixed, retrieved, history = used, [], []
    for score, chunk in sorted(RETRIEVED, reverse=True):
        if used + tok(chunk) <= budget * 0.4:
            used += tok(chunk)
            retrieved.append(score)
    for index in range(len(HISTORY) - 1, -1, -1):
        cost = tok(trim(HISTORY[index], 500))
        if used + cost > budget:
            break
        used += cost
        history.append(index)
    return fixed, used, retrieved, history

for budget in (3000, 6000, 12000):
    fixed, used, retrieved, history = pack(budget)
    print(f"budget {budget:6d}: used {used:5d}, fixed {fixed}, retrieved scores {retrieved}, "
          f"history steps kept {len(history)} (oldest kept: step {min(history) + 1})")
longest = max(HISTORY, key=tok)
print("longest tool result:", tok(longest), "tokens; after trim to 500:", tok(trim(longest, 500)))
```

**Reading the output.** At a 3,000-token budget the request uses 2,798 tokens: two retrieved passages (scores 0.91 and 0.84) and the four most recent steps. At 6,000 all four passages fit and eight steps are kept. At 12,000 the retrieval share is full and 21 of 24 steps fit. The last line shows the trimmer: the longest result is 1,233 tokens and becomes 508.

**Line by line.**

- `trim` keeps two thirds of the allowed tokens from the head and one third from the tail and says how many were omitted. The marker costs a few tokens, so the result is slightly over the 500 limit.
- `pack` spends the budget in a fixed order. Retrieval is capped at 40% so that history cannot be crowded out by search results.
- The loop over history runs from newest to oldest and stops at the first item that does not fit, so the kept steps are always the most recent unbroken run.

## The lab

<ContextWindowLab />

The default setting, mask old results in batches with the last 4 steps kept, a 10,000-token window and the placeholder prices of 0.10 and 1.25, reproduces the third code block's row: peak 7,900 tokens, 127,273 billed, 4 of 6 facts, 68.0% cache hits and 59,552 cost units.

**What each control does.**

- **strategy** chooses how the 24-step run is managed. The bars show one request each: blue is the part found in the cache, orange the part written fresh.
- **recent steps kept** is how many latest results stay in full for the masking and summary strategies. It is greyed out for the others.
- **window** is the limit. The dashed line marks it, and "requests over the window" counts bars above it. For the sliding window it also changes how much history is kept.
- **cache read price** and **cache write price** are the placeholder ratios. They change the cost line only, not the bars.
- **show data** lists each request's tokens, cached part and written part.

**Try it yourself.**

1. Set the strategy to "Keep everything, clock at the top". Watch the blue parts of the bars disappear: cache hits fall to 0.1% and the cost rises to 362,376 units, about six times the plain keep-everything run, although the bars barely change height. The tokens are the same, the cache cannot use them.
2. Choose "Sliding window" and step the window through 6,000, 10,000 and 16,000. Facts kept go 0, 1, then 3 of 6. A sliding window forgets by age, not by importance, so the early findings go first.
3. Return to the default and move the cache read price to 0.50. The cost ranking changes: keeping everything now costs 163,171 units, more than masking in batches at 94,174 and more than masking every step at 112,022. When cached tokens are only half price, sending fewer tokens matters more than keeping the prefix still.

<Infographic src="/img/afr/context-window-results.svg" alt="A table of six strategies with peak tokens, tokens billed, requests over the window, facts kept, cache hits and cost units, above a bar chart of cost units." caption="The printed result of code block 3. Compare the first two rows: the same tokens, a different cache hit, a different bill." />

## Designing with it

| If your problem is | Reach for | Watch out for |
| --- | --- | --- |
| Tool list dominates every request | Fewer tools per task, or load tools on demand | Changing the tool list mid-run breaks the cache |
| One huge tool result | Trim to head and tail, or return a handle and a summary | The fact you need may be in the middle |
| A long run near the window | Batch masking, then summarise if still near | Compacting every step, which rewrites the prefix |
| Facts must survive for hours | Notes the agent writes, loaded back each time | Trusting a summary to keep what you never marked important |
| High repeat traffic with a long stable prefix | Stable text first, dynamic text last | Clocks, names and ids at the top of the prompt |
| Provider gives a small cache discount | Cut tokens first | Assuming the cache will save you |

Three habits cover most of it. Count tokens per section on a real run before you optimise. Put everything that never changes first and everything that changes last. And test survival: plant a few facts early, run the long task, and ask about them at the end. If a strategy cannot pass that test it is not a saving.

Worked designs live elsewhere on this site. The [coding assistant case](/docs/senior/design-ai-coding-assistant) measures which code to pack under a token budget, and the [support agent case](/docs/senior/design-customer-support-agent-platform) prices memory strategies with and without a cache. For vendor features that do this for you, see the next section.

## Where this stands in 2026

:::info Industry view
Context engineering is now the common name for this discipline. Anthropic's post defines it as curating and maintaining the best set of tokens during inference, and lists compaction, note-taking, sub-agents with clean windows and just-in-time retrieval (keep file paths or queries, load content when needed) as the main tools. Tool-result clearing is described there as one of the safest, lightest forms of compaction.

Providers have started to ship the mechanics. Anthropic's context editing documentation (checked 7 October 2026) lists a strategy that clears old tool results once a size threshold is passed, with options for how many recent tool uses to keep, which tools never to clear and a minimum amount to clear per pass, which it notes helps prompt caching; it requires a beta header and a separate strategy handles thinking blocks. That is batch masking with the knobs of this chapter's experiment. Both Anthropic and OpenAI document prefix caching with cached input at about a tenth of normal input for current models, so the cache-friendly layout above is mainstream advice, not a trick. Check the pages for the model you use, because ratios, minimum lengths and retention differ.

What is not settled is how well models use very long inputs, and whether a bigger window removes the need for any of this. The evidence cited here (Lost in the Middle, Context Rot) argues no for now, but those are tests of particular models at particular dates.
:::

## Common mistakes

- **Appending everything, because more context feels safer.** It feels safe because nothing is lost. But cost grows with the square of the run length, and long inputs degrade answers. Decide what must survive, write it down, and let the rest go.
- **Compacting or masking on every step.** It feels tidy. But each edit rewrites the middle of the prompt and the cache has to be paid for again: in the experiment 11.6% hits and a higher bill than keeping everything. Trigger in batches.
- **A clock, user name or request id at the top of the system prompt.** It feels natural to put "current time" first. One changing token invalidates the cache after it, so the experiment's hit rate fell to 0.1%. Put changing text at the end, or give it as a tool result.
- **Summarising without saying what must survive.** It feels like the model will keep what matters. A summary keeps what its author thought mattered, and the sliding window kept 1 of 6 facts. Make the agent restate key facts as it goes, and test survival.
- **Counting characters or words.** It feels close enough. Tokens differ from both, and every provider tokenises differently. Count with the tokenizer your model uses.

## Practice questions

<details>
<summary><strong>Easy.</strong> The longest request in the keep-everything run is 24,383 tokens, yet the run bills 289,768. Why the difference?</summary>

Every request resends the whole history, and the run makes 24 requests. The billed figure is the sum of the 24 request sizes, which grow from a few hundred tokens to 24,383. The average request is about 12,000 tokens, and 24 times 12,000 is close to the total. The history itself is never larger than 24,383.

</details>

<details>
<summary><strong>Easy.</strong> In the twenty-tool request, which part is largest, and what is the first thing to try if it is too big?</summary>

The tool definitions: 1,060 of 1,208 tokens, 87.7%. Each tool costs about 53 tokens on every request. The first thing to try is giving the agent only the tools it needs for the current task, which removes tokens without losing anything the task uses. The system prompt is only 71 tokens.

</details>

<details>
<summary><strong>Medium.</strong> Work out the cost of request 4 under masking in the by-hand example (read 0.10, write 1.25).</summary>

Request 3 is the blocks fixed, step 1 stub, step 2, step 3. Request 4 is fixed, step 1 stub, step 2 stub, step 3, step 4. They match through the fixed block and the step 1 stub, which is 200 + 30 = 230 tokens. Request 4 has 200 + 30 + 30 + 200 + 200 = 660 tokens, so 430 are new. The cost is 230 × 0.10 + 430 × 1.25 = 23 + 537.5 = 560.5 units. Step 2 was masked at this request, which is why its block no longer matches.

</details>

<details>
<summary><strong>Medium.</strong> Why does masking in batches cost 59,552 units while masking every step costs 107,548, when the second sends fewer tokens?</summary>

Masking every step changes an old result in the middle of the history on each request, so the cache can reuse only the short stable prefix: 11.6% of tokens hit. The recent results, which are the bulk of the tokens, are rewritten every time at 1.25. Batch masking waits until the context passes 8,000 tokens, then masks many results at once. Between those events the prompt only grows at the end, so 68.0% of tokens hit the cache at 0.10. Paying 1.25 instead of 0.10 on most tokens outweighs the saving from sending fewer of them.

</details>

<details>
<summary><strong>Stretch.</strong> A support agent talks to one customer for 200 turns. In turn 2 the customer says they are allergic to nuts. How do you make sure that fact is used in turn 190?</summary>

Do not rely on the history or on a summary to carry it. Have the agent write the fact to a small notes or memory store with a clear label when it is stated, and load that store into every request near the top, where it is stable and cheap to cache. Then test it: a planted-fact test in which the allergy is stated early and a late question depends on it, run against the real long-session setup. The experiment showed that facts the agent restated survived masking and summarising (4 of 6) and the others did not (the sliding window kept 1 of 6). See [LLM memory](/docs/agentic-ai/llm-memory) and [LangChain memory](/docs/genai/langchain-advanced/memory) for the store.

</details>

<details>
<summary><strong>Stretch.</strong> A teammate proposes adding "Current time: ..." as the first line of the system prompt so the agent knows the date. What happens, and what would you do instead?</summary>

Every request now starts with a different token, so the cached prefix cannot be reused: the experiment's hit rate fell from 91.6% to 0.1% and the cost from 57,017 to 362,376 units, with almost no change in tokens billed. Instead, keep the system prompt fixed and supply the time as part of the newest user message, or as the result of a small `get_time` tool, so that only the tail of the prompt changes. If the model rarely needs the time, do not send it at all.

</details>

## Go deeper

All opened on 7 October 2026.

- Anthropic, "Effective context engineering for AI agents", engineering blog, 29 September 2025. Definition, attention budget, compaction, note-taking, sub-agents, just-in-time retrieval.
- Kelly Hong, Anton Troynikov and Jeff Huber, "Context Rot", Chroma technical report, 14 July 2025. 18 models, input length, distractors and needle-question similarity.
- Nelson F. Liu and colleagues, "Lost in the Middle: How Language Models Use Long Contexts", arXiv 2307.03172, submitted 6 July 2023, revised 20 November 2023, Transactions of the Association for Computational Linguistics.
- Yichao "Peak" Ji, "Context Engineering for AI Agents: Lessons from Building Manus", 18 July 2025. Cache hit rate, stable prefixes, the file system as memory.
- Tobias Lindenbauer and colleagues, "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management", arXiv 2508.21433, submitted 29 August 2025, version 3 of 27 October 2025, presented at the DL4C workshop at NeurIPS 2025.
- Anthropic documentation: prompt caching (multipliers 0.1, 1.25 and 2; order tools, system, messages) and context editing (a tool-result clearing strategy and a thinking-block strategy, with a beta header), checked 7 October 2026. OpenAI documentation: prompt caching guide, checked the same day.
- On this site: [LLM memory](/docs/agentic-ai/llm-memory), [Memory in LangChain](/docs/genai/langchain-advanced/memory), [Middleware](/docs/genai/langchain-advanced/middleware), [KV cache and paged attention](/docs/llm-engineering/kv-cache-and-paged-attention), [semantic caching](/docs/llm-engineering/semantic-caching-routing-and-cost), [AI coding assistant case](/docs/senior/design-ai-coding-assistant), [support agent case](/docs/senior/design-customer-support-agent-platform), [operational evals](/docs/llm-evals/operational-evals).

**Not verified here.** The behaviour of any real model with a masked or summarised history; price ratios for a specific model (the ratios in the code are placeholders); cache block sizes, minimum lengths and expiry; the exact tokeniser your provider uses. I did not run the vendor context-editing features, only read their documentation.

## Check yourself

- I can count what fills a request and name the biggest fixed cost.
- I can explain why a long agent run bills far more tokens than its final length.
- I can choose between trimming, masking, compacting and note-taking, and say what each one loses.
- I can lay out a prompt so the prefix cache keeps working, and explain why a timestamp at the top costs so much.
- I can describe a test that shows whether a context strategy keeps the facts the task needs.

## Where to go next

Next: [Agent interoperability: MCP and A2A](/docs/agentic-frontier/agent-interoperability-mcp-and-a2a), where the tool list that fills your window comes from, and how agents hand work to each other. Related: [LLM memory](/docs/agentic-ai/llm-memory) for the stores that live outside the window.
