---
id: senior-case-coding-assistant
title: "System Design Case: An AI Coding Assistant"
sidebar_label: "4 · AI coding assistant"
sidebar_position: 4
slug: /senior/design-ai-coding-assistant
description: "Design an AI coding assistant for 4,000 developers: context assembly under a token budget, three latency tiers, stale-request control, evaluation by execution and the privacy of source code, with sizing, a measured retrieval experiment and a cost formula."
tags: [system-design, coding-assistant, context-engineering, pass-at-k, latency, privacy]
---

import Infographic from '@site/src/components/Infographic';
import AssistantContextLab from '@site/src/components/viz/AssistantContextLab';

**In one line.** An AI coding assistant is three products sharing one engine (completion, chat and an agent), and the engine's real job is to choose which few thousand tokens of a large codebase the model sees, quickly, without leaking the code.

:::note Not from a lecture
Written for this site from the sources under Further reading. The requirements below are the brief of a design exercise, not facts about any company. Every figure in the chapter is printed by the code or comes from a source named beside it.
:::

## The idea in plain words

**The brief.** A company with 4,000 developers wants an assistant in the editor: ghost-text completion that feels instant, a chat that answers questions about the repository, and an agent that edits several files and runs the tests. Source code must not be used to train anyone's model and secrets must never leave the machine. A junior engineer starts with "which model?". A senior engineer starts with three numbers: how many requests, how many tokens per request, and how long a person will wait.

The same engine serves three different relationships with time.

| Tier | The user is | Waits for | What dominates |
| --- | --- | --- | --- |
| Completion | typing | tens to hundreds of milliseconds | stale requests, small prompt, small model |
| Chat | reading | about a second to the first token | retrieval quality, streaming |
| Agent | away from the keyboard | minutes, many model calls | tool loops, prefix caching, evaluation |

<Infographic src="/img/senior/ai-coding-assistant-architecture.svg" alt="Table of three latency tiers with requests, tokens and peak load, above the pipeline editor, context assembly, privacy gate and model, with the evaluation and total-load boxes." caption="The system on one page. Every number in the table is printed by block 1 below." />

Two ideas carry the design. **Context is a budget**: the model can read only so much, and every token costs latency and money, so the question is which tokens earn their place. **Code can be checked by running it**, so evaluation here can be far more honest than for a chat product.

## How it works

**Sizing (block 1).** At the stated peak, completion is 833 requests a second and chat 13, and the agent 80. Together they ask for about 1,001,000 prefill tokens a second, of which completion alone is 625,000 although its requests are the cheapest. The GPU count is a formula, `ceil(peak prefill tokens per second / tokens one GPU prefills per second)`; the per-GPU rates in the block are hypothetical parameters (this environment has no GPU), giving 101, 34 or 11 GPUs at 10,000, 30,000 or 100,000 tokens a second.

**Data flow for one completion.**

1. The editor sends the cursor position, the text around it and the open files. Fill-in-the-middle training (Bavarian et al., 2022) is what lets a model complete between a prefix and a suffix.
2. A **debounce** waits for a pause in typing; a new keystroke cancels the request in flight.
3. **Context assembly** ranks other code and packs the best under a token budget, stable material first so the provider's prefix cache can reuse it.
4. The **privacy gate** drops excluded paths and redacts secrets.
5. A small model returns a few dozen tokens, shown as ghost text.

### Decision 1: how to find context

| Approach | Strength | Weakness | Cost to run |
| --- | --- | --- | --- |
| Neighbouring code (same file, open tabs) | free, fast, right for local style | blind to the helper in another module | none |
| Keyword search (BM25, grep) | exact identifiers, no model, nothing to go stale | misses synonyms and indirect dependencies | low |
| Embedding index | finds paraphrase and intent | must be built, refreshed and protected; the index is a copy of the code | medium to high |
| Call graph or language-server data | follows real dependencies | needs a parser per language | medium |
| Signature map of the repository | cheap breadth | spends tokens on things the task may not need | low |

Measure rather than argue. Block 2 uses the `httpx` package in this environment (0.28.1, 440 functions) as a real testbed. A completion point is 40% of the way through a function body; the question is whether the definitions the finished function goes on to call are in the context. Its results: plain keyword search is a weak guide (at 4,000 tokens `bm25` reaches 0.433 body recall, neighbouring code 0.350); following call edges changes the picture (`bm25+graph` reaches 0.700 at 2,000 tokens where `bm25` has 0.300); and the signature map is a trade (`map+graph` has lower body recall than `bm25+graph` at every budget from 1,000 tokens up, yet sees 0.933 of signatures at 4,000). For writing a call, a signature is often enough, so which recall matters depends on the task.

<Infographic src="/img/senior/ai-coding-assistant-context-budget.svg" alt="Table of body recall for four context strategies at five token budgets, with notes on how each strategy builds its list." caption="The printed table of block 2, with what each strategy does." />

### Decision 2: latency and stale requests

A completion is useful only if it arrives before the next keystroke. Block 3 types 20,000 simulated keystrokes, sends a request after a debounce, and counts a suggestion as shown only if no further key arrived first. Typing pattern and model latencies are assumptions, not measurements of any product. With a model that answers in a median 600 ms and no debounce, only 8.7 of every 100 keystrokes end in a suggestion still wanted (0.09 per request); a 300 ms debounce cuts requests from 100 to 21.5 per 100 keystrokes and lifts the useful share per request to 0.21, but the wait after a pause becomes 900 ms. A 300 ms model turns the same debounce into 0.33 useful per request. **Model latency and debounce are one decision**, and the cheapest way to feel fast is a smaller prompt and a smaller model, which block 4 prices.

| Decision | Option | When it wins |
| --- | --- | --- |
| Completion model | small hosted model, or a small one you serve | latency dominates and the prompt is short |
| Chat and agent model | the strongest model you can afford | quality dominates and a person waits anyway |
| Prefix caching | instructions and repository map first, volatile text last | agent loops that resend the same prefix |

### Decision 3: privacy of source code

Treat the assistant as a data-egress system. The Copilot content-exclusion documentation says excluded files give no inline suggestions and do not inform suggestions elsewhere or Copilot Chat, and it lists limits: Copilot "may use semantic information from an excluded file if the information is provided by the IDE indirectly", and exclusions do not apply to symbolic links or remote file systems. An exclusion list is a control with known holes.

| Layer | What it stops | What it misses |
| --- | --- | --- |
| Path exclusions (`.env`, `*.pem`, `secrets/`) | whole files never read | secrets inside ordinary files; indirect IDE information |
| Pattern redaction (key formats) | well-formed keys | odd formats |
| Entropy redaction on assigned strings | random-looking tokens | weak or readable secrets; it flags readable labels |
| Contract and retention terms | training on your code, long retention | nothing technical: it is a promise |
| Local indexing | the index leaving the machine | retrieved snippets still go to the model |

Block 5 shows the limits of the cheap layers: the entropy rule caught a random secret (4.58 bits per character) but also redacted a readable label (3.70, a false positive) and missed the weak password `changeme-changeme` (2.91, a false negative). On the 8,828 lines of `httpx` it flagged nothing. It is a teaching sketch; production gates use maintained secret scanners plus the exclusion list.

### Decision 4: evaluation by execution

The Codex paper (Chen et al., 2021) built HumanEval, 164 hand-written problems with unit tests, and scores a model by **pass@k**: generate n samples per problem, count the c that pass, and estimate the chance that at least one of k passes with the unbiased formula `1 - C(n-c, k) / C(n, k)`. The paper notes that `1 - (1 - p)^k` from the pass@1 estimate is biased, that BLEU may not reliably indicate functional correctness, and that it ran generated code in a gVisor sandbox. SWE-bench (Jimenez et al., 2023) applies the same principle to 2,294 real tasks from 12 Python repositories. Block 6 shows why text similarity fails here.

## A real system that works this way

Public descriptions show the same structure with different answers. Aider's documentation describes a repository map of the most important definitions with their signatures, ranked by a graph algorithm over files and their dependencies, sized by a token budget that defaults to 1,000 tokens. Anthropic's context-engineering post argues for treating context as "a finite resource with diminishing marginal returns", loading it just in time through lightweight references such as file paths, and says Claude Code is a hybrid: instruction files up front, `glob` and `grep` for the rest. The Cursor page opened for this chapter describes a local search index it calls Instant Grep and states that Cursor does not upload file paths or code to build it and does not store embeddings of the codebase for search. Copilot documents content exclusion with the caveat above. These are evidence for the decision tables, not a ranking of the products.

## Code you can run

All blocks are CPU only and seeded. Block 3 and block 4 depend on assumptions or timing; the prose says which numbers are measured.

#### 1. Sizing

```python
import math

developers = 4000
active_hours = 5
peak_factor = 3.0
tiers = {
    "completion": dict(per_dev_hour=250, prompt=1500, output=40),
    "chat": dict(per_dev_hour=4, prompt=6000, output=400),
    "agent step": dict(per_dev_hour=24, prompt=20000, output=300),
}
cache_hit = {"completion": 0.5, "chat": 0.3, "agent step": 0.8}
P_IN, P_OUT = 1.0, 4.0

print("tier         req/s avg   req/s peak   prefill tok/s peak   decode tok/s peak   daily cost (input-token units)")
total_cost = 0.0
for name, t in tiers.items():
    per_day = developers * active_hours * t["per_dev_hour"]
    avg = developers * t["per_dev_hour"] / 3600
    peak = avg * peak_factor
    fresh = t["prompt"] * (1 - cache_hit[name])
    cost = per_day * (fresh * P_IN + t["prompt"] * cache_hit[name] * 0.1 * P_IN + t["output"] * P_OUT) / 1e6
    total_cost += cost
    print(f"{name:11s} {avg:9.1f}   {peak:10.1f}   {peak * fresh:18,.0f}   {peak * t['output']:17,.0f}   {cost:12,.0f}")
print(f"total per day: {total_cost:,.0f} input-million-token units, {total_cost / developers:.2f} per developer")

peak_prefill = sum(developers * t["per_dev_hour"] / 3600 * peak_factor * t["prompt"] * (1 - cache_hit[n]) for n, t in tiers.items())
print(f"\npeak prefill demand {peak_prefill:,.0f} tokens/s")
for per_gpu in (10_000, 30_000, 100_000):
    print(f"if one GPU prefills {per_gpu:,} tokens/s: {math.ceil(peak_prefill / per_gpu)} GPUs before headroom")
```

The cost column uses placeholder prices (1 unit per input token, 4 for output, a cached read at one tenth), so it is in input-token units and only the ratios carry meaning: completion is 4,925 of 8,667 daily units, 2.17 million input-token equivalents per developer per day.

#### 2. Context under a token budget

```python
import ast
import math
import pathlib
import random
import re
from collections import Counter, defaultdict

import httpx
import numpy as np
from transformers import AutoTokenizer

root = pathlib.Path(httpx.__file__).parent
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


def ntokens(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


chunks, files = [], {}
for path in sorted(root.rglob("*.py")):
    source = path.read_text()
    lines = source.splitlines()
    rel = str(path.relative_to(root))
    files[rel] = lines

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_start = child.body[0].lineno
                sig = "\n".join(lines[child.lineno - 1 : body_start - 1])
                text = "\n".join(lines[child.lineno - 1 : child.end_lineno])
                calls = []
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                        if name:
                            calls.append((sub.lineno, name))
                chunks.append(dict(file=rel, qual=prefix + child.name, name=child.name, start=child.lineno,
                                   end=child.end_lineno, sig=sig, text=text, calls=calls))

    visit(ast.parse(source))

for c in chunks:
    c["tokens"] = ntokens(c["text"])
    c["sig_tokens"] = ntokens(c["sig"])

by_name = defaultdict(list)
for i, c in enumerate(chunks):
    by_name[c["name"]].append(i)


def resolve(name, own):
    ids = by_name.get(name, [])
    return ids[0] if len(ids) == 1 and ids[0] != own and not name.startswith("__") else None


indegree = Counter()
for i, c in enumerate(chunks):
    for _, name in c["calls"]:
        j = resolve(name, i)
        if j is not None:
            indegree[j] += 1

KEYWORDS = {"self", "return", "if", "else", "for", "in", "not", "and", "or", "is", "none", "def", "class", "the",
            "none", "true", "false", "import", "from", "as", "with", "raise", "elif", "try", "except", "str", "int"}


def terms(text):
    out = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        for part in re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).lower().split():
            if len(part) > 1 and part not in KEYWORDS:
                out.append(part)
    return out


doc_terms = [Counter(terms(c["text"] + " " + (c["file"] + " " + c["qual"] + " ") * 2)) for c in chunks]
doc_len = np.array([sum(d.values()) for d in doc_terms], dtype=float)
df = Counter(t for d in doc_terms for t in d)
N = len(chunks)
idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}


def bm25(query_terms, exclude):
    scores = np.zeros(N)
    for t in set(query_terms):
        if t not in idf:
            continue
        for i, d in enumerate(doc_terms):
            f = d.get(t, 0)
            if f:
                scores[i] += idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * doc_len[i] / doc_len.mean()))
    scores[exclude] = -1
    return scores


tasks = []
for i, c in enumerate(chunks):
    if c["end"] - c["start"] < 8:
        continue
    cursor = c["start"] + round(0.4 * (c["end"] - c["start"]))
    needed = sorted({resolve(n, i) for ln, n in c["calls"] if ln > cursor} - {None})
    seen = sorted({resolve(n, i) for ln, n in c["calls"] if ln <= cursor} - {None})
    if len(needed) >= 1:
        tasks.append(dict(own=i, cursor=cursor, needed=needed, seen=seen))
print(f"{N} functions in httpx {httpx.__version__}, {sum(c['tokens'] for c in chunks)} tokens in total")
print(f"{len(tasks)} eligible completion points; sampling 30 with seed 0")
tasks = random.Random(0).sample(tasks, 30)
print(f"needed definitions per task: mean {np.mean([len(t['needed']) for t in tasks]):.2f}, "
      f"already visible in the prefix: {sum(len(set(t['needed']) & set(t['seen'])) for t in tasks)} "
      f"of {sum(len(t['needed']) for t in tasks)}")


def rankings(task):
    own = task["own"]
    c = chunks[own]
    prefix = files[c["file"]][max(0, task["cursor"] - 20) : task["cursor"]]
    scores = bm25(terms("\n".join(prefix)), own)
    order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
    same = sorted((i for i, o in enumerate(chunks) if o["file"] == c["file"] and i != own),
                  key=lambda i: abs(chunks[i]["start"] - task["cursor"]))
    rest = [i for i in range(N) if chunks[i]["file"] != c["file"]]
    seeds = order[:3] + task["seen"]
    hop = {resolve(n, s) for s in seeds for _, n in chunks[s]["calls"]} - {None, own}
    top = scores.max() or 1.0
    boosted = {i: scores[i] / top + (0.6 if i in task["seen"] else 0) + (0.4 if i in hop else 0) for i in set(order) | hop | set(task["seen"])}
    graph = sorted(boosted, key=lambda i: (-boosted[i], i))
    mapped = [i for i, _ in indegree.most_common(80) if i != own]
    return {
        "neighbour": [(i, "full") for i in same + rest],
        "bm25": [(i, "full") for i in order],
        "bm25+graph": [(i, "full") for i in graph],
        "map+graph": [(i, "sig") for i in mapped] + [(i, "full") for i in graph],
    }


def pack(items, budget, map_share=0.3):
    used, sig_used, got_sig, got_body = 0, 0, set(), set()
    for i, mode in items:
        cost = chunks[i]["sig_tokens"] if mode == "sig" else chunks[i]["tokens"]
        if used + cost > budget or (mode == "sig" and sig_used + cost > map_share * budget):
            continue
        used += cost
        got_sig.add(i)
        if mode == "sig":
            sig_used += cost
        else:
            got_body.add(i)
    return used, got_sig, got_body


BUDGETS = (250, 500, 1000, 2000, 4000)
ranked = [rankings(t) for t in tasks]
print("\nstrategy      budget   signature recall   body recall   tokens used")
for strategy in ranked[0]:
    for budget in BUDGETS:
        sig_r, body_r, used = [], [], []
        for t, r in zip(tasks, ranked):
            u, gs, gb = pack(r[strategy], budget)
            need = set(t["needed"])
            sig_r.append(len(need & gs) / len(need))
            body_r.append(len(need & gb) / len(need))
            used.append(u)
        print(f"{strategy:12s} {budget:6d}   {np.mean(sig_r):16.3f}   {np.mean(body_r):11.3f}   {np.mean(used):11.0f}")
```

The lab replays the same 30 completion points. Its defaults, `bm25+graph` at 1,000 tokens over all tasks, give signature recall 0.500 and body recall 0.500, the values printed above. Pick one task to see the budget as a bar filled with packed items.

<AssistantContextLab />

:::warning Indicative, not general
One repository, 30 points and a mean of 1.2 needed definitions per point, so one completion moves a cell by several hundredths. Ground truth is what the original author called, not the only valid completion. `httpx` also has synchronous and asynchronous twins of many methods, which flatters call-graph expansion. Repeat the measurement on your own code.
:::

#### 3. Stale requests

```python
import numpy as np

rng = np.random.default_rng(0)
gaps = rng.lognormal(np.log(0.17), 0.55, 20000)
pauses = rng.random(20000) < 0.08
gaps = np.where(pauses, rng.gamma(2.0, 0.6, 20000), gaps)
times = np.cumsum(gaps)
next_gap = np.append(gaps[1:], 10.0)
noise = rng.normal(0, 1, 20000)

print("median model latency   debounce   requests per 100 keystrokes   shown per 100 keystrokes   shown per request   wait after a pause")
for median in (0.3, 0.6, 1.2):
    latency = median * np.exp(0.4 * noise)
    for debounce in (0.0, 0.08, 0.15, 0.3):
        fires = next_gap > debounce
        shown = fires & (next_gap > debounce + latency)
        wait = debounce + median
        print(f"{median * 1000:12.0f} ms {debounce * 1000:12.0f} ms {fires.mean() * 100:22.1f} {shown.mean() * 100:27.1f} "
              f"{shown.sum() / fires.sum():20.2f} {wait * 1000:14.0f} ms")
```

#### 4. Prompt length is latency

This times the 135M-parameter SmolLM2 on CPU, so the absolute numbers say nothing about a production GPU; the shape is the point.

```python
import pathlib
import time

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
torch.manual_seed(0)
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
params = sum(p.numel() for p in model.parameters())

text = "\n".join(p.read_text() for p in sorted(pathlib.Path(httpx.__file__).parent.rglob("*.py")))[:40000]
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]


def best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


print(f"{params / 1e6:.1f}M parameters")
print("prompt tokens   prefill ms   ms per 100 tokens   relative to 128   GFLOPs (2 x params x tokens)")
base = None
with torch.no_grad():
    model(ids[:64][None])
    for n in (128, 256, 512, 1024, 2048):
        batch = ids[:n][None]
        ms = best_of(lambda: model(batch)) * 1000
        base = base or ms
        print(f"{n:13d} {ms:12.0f} {ms / n * 100:19.1f} {ms / base:17.1f} {2 * params * n / 1e9:26.0f}")
```

The analytic column is deterministic: prefill work is about 2 x parameters x tokens, so 128 tokens cost 34 GFLOPs and 2,048 tokens cost 551. Measured time grew roughly in proportion in the runs made for this chapter (2,048 tokens, 16 times as many as 128, took between 10 and 18 times as long), but timings moved by more than a factor of two between runs on a busy machine, so quote shape and not milliseconds. For completion, 40 output tokens are cheap next to the prompt, which is why trimming context buys latency.

#### 5. Redaction and exclusion

```python
import fnmatch
import math
import pathlib
import re
from collections import Counter

import httpx

PATTERNS = {
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "vendor token": re.compile(r"\btok_[A-Za-z0-9]{36}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
}
ASSIGNED = re.compile(r"(?i)(\w*(?:secret|token|passw(?:or)?d|api_?key)\w*)\s*[:=]\s*['\"]([^'\"]{16,})['\"]")
EXCLUDE = [".env", ".env.*", "*.pem", "*.key", "secrets/*", "*/secrets/*", "id_rsa*"]


def entropy(s):
    counts = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in counts.values())


def excluded(path):
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE)


def redact(text):
    found = Counter()
    for kind, pattern in PATTERNS.items():
        text, n = pattern.subn(f"[REDACTED:{kind}]", text)
        found[kind] += n

    def swap(m):
        if entropy(m.group(2)) >= 3.5:
            found["high-entropy assignment"] += 1
            return m.group(0).replace(m.group(2), "[REDACTED:high-entropy assignment]")
        return m.group(0)

    return ASSIGNED.sub(swap, text), found


synthetic = f'''
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
gh = "tok_{'aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bD1fH3jL5'}"
client_secret = "{'q7Zr2Lx9TuV4wYb8NcE1sKd6'}"
password = "changeme-changeme"
TIMEOUT_TOKEN = "this-is-just-a-long-readable-label"
-----BEGIN RSA PRIVATE KEY-----
header = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345"
'''
clean, found = redact(synthetic)
print("synthetic file:", dict(found))
print("pattern secrets left after redaction:", sum(1 for p in PATTERNS.values() if p.search(clean)))
for label, value in (("random client secret", "q7Zr2Lx9TuV4wYb8NcE1sKd6"), ("readable label", "this-is-just-a-long-readable-label"),
                     ("weak password", "changeme-changeme")):
    print(f"{label:21s} entropy {entropy(value):.2f} bits/char   redacted: {value not in clean}")

for path in (".env", "config/.env.production", "deploy/key.pem", "secrets/db.txt", "src/app.py", "docs/secrets.md"):
    print(f"{path:24s} excluded={excluded(path)}")

root = pathlib.Path(httpx.__file__).parent
total, hits = 0, Counter()
for p in sorted(root.rglob("*.py")):
    text = p.read_text()
    total += len(text.splitlines())
    _, f = redact(text)
    hits.update(f)
print(f"\nhttpx {httpx.__version__}: {total} lines scanned, flagged {dict(hits)}")
```

#### 6. Evaluation by execution

The samples are hand-written, not model output, so the failures are known. The harness runs each in a subprocess with a two-second timeout, which is enough for trusted toys and unsafe for model output at scale.

```python
import difflib
import subprocess
import sys
import tempfile
from math import comb

problems = {
    "merge_intervals": dict(
        tests="assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\nassert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\nassert merge_intervals([]) == []\nassert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]\nassert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]",
        samples={
            "reference": "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
            "different style": "def merge_intervals(iv):\n    iv = sorted(iv, key=lambda p: p[0])\n    merged = []\n    for start, end in iv:\n        if not merged or merged[-1][1] < start:\n            merged.append([start, end])\n        else:\n            merged[-1][1] = max(merged[-1][1], end)\n    return merged",
            "touching not merged": "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a < out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out",
            "keeps last end": "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n        if out and a <= out[-1][1]:\n            out[-1][1] = b\n        else:\n            out.append([a, b])\n    return out",
        }),
}


def passes(code, tests):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + tests + "\n")
    try:
        return subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=2).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


print("problem               sample               passes   similarity to reference")
rows, c_by = [], {}
for pname, p in problems.items():
    ref = p["samples"]["reference"]
    c_by[pname] = 0
    for sname, code in p["samples"].items():
        ok = passes(code, p["tests"])
        sim = difflib.SequenceMatcher(None, ref, code).ratio()
        c_by[pname] += ok
        rows.append((pname, sname, ok, sim))
        print(f"{pname:20s}  {sname:19s}  {str(ok):6s}   {sim:.2f}")

passed = [s for *_, ok, s in rows if ok and s < 1]
failed = [s for *_, ok, s in rows if not ok]
print(f"\nmost similar failing sample {max(failed):.2f}; the correct sample with a different style scores {min(passed):.2f}")
n = 4
print("\nproblem               n  correct  pass@1  pass@2  naive 1-(1-p)^2")
for pname, c in c_by.items():
    print(f"{pname:20s}  {n}  {c:7d}  {pass_at_k(n, c, 1):6.3f}  {pass_at_k(n, c, 2):6.3f}  {1 - (1 - c / n) ** 2:15.3f}")
```

The failing `touching not merged` is one character from the reference and has similarity 1.00 when rounded; the correct sample in a different style scores 0.12, so a similarity metric would rank the bug above the answer. With 2 of 4 correct, pass@1 is 0.500, pass@2 is 0.833 and the biased shortcut says 0.750. The fifth test (`[[1, 10], [2, 3]]`) is what exposes `keeps last end`: execution evaluation is only as good as its tests.

## Designing with it

**Failure modes and mitigations**

| Failure | Cause | Mitigation |
| --- | --- | --- |
| Suggestion arrives after the user moved on | latency above the typing gap | debounce, cancel, smaller prompt and model |
| Call to a function that does not exist | the real definition was not in context | signature map or call-graph expansion |
| Secret in a prompt | secret inside an ordinary file | exclusions, redaction, retention terms, an audit log of what was sent |
| Agent edits the wrong files and keeps going | no verifiable stop condition | tests as the loop's exit, step and token limits, review of the diff |
| Cost grows with every agent step | each step resends the history | stable prefix first, prefix caching, compaction of old steps |
| Metric improves, developers are unhappy | optimising acceptance alone | pair it with execution-based and survey measures |

**Evaluation and rollout.** Offline: an execution suite from your own repositories (a task, tests, a sandbox), reported as pass@1 and pass@k with n stated. Then shadow mode (generate, do not show) to measure latency and context quality, a small group, a staged rollout by team, and a kill switch. Compare against what developers do today, not against nothing. See the site's [evaluation workflow](/docs/llm-evals/evaluation-workflow) and [regression testing](/docs/llm-evals/regression-testing).

**Cost as a formula.** `daily cost = sum over tiers of developers x active hours x requests per hour x (fresh prompt tokens x P_in + cached tokens x r x P_in + output tokens x P_out)`, where `r` is the cached-read price as a fraction of the input price. Every term is a parameter in block 1; with the stated assumptions the completion tier is 57% of daily units.

**What to build first.** One completion tier for one language, neighbouring code plus keyword search, a debounce, and an execution suite of fifty real tasks. Add call-graph expansion only if measurement says it helps, and the agent only after the suite can tell you whether it helps.

## Where this stands in 2026

:::info Industry view

- **Retrieval has moved toward search the model can drive.** Anthropic's September 2025 post favours just-in-time retrieval with simple tools, and the Cursor page describes a local text-search index rather than stored embeddings. Treat older descriptions of embedding-based indexing as needing a re-check.
- **Execution-based evaluation is the norm for code.** HumanEval-style pass@k and repository-level suites such as SWE-bench are the common currency, and both depend on tests the evaluator must keep honest.
- **Privacy is a configuration surface.** Exclusions, retention terms and local indexing are controls with documented limits; assume some context reaches the model and decide what that may contain.
- **Versions move monthly.** This chapter's code ran with `httpx` 0.28.1, `transformers` 5.18.0 and `torch` 2.14.1; product documentation pages were read on 2 October 2026.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why can a 600 ms completion model with no debounce be worse than one with a 300 ms debounce?</summary>

Without a debounce every keystroke sends a request and most are stale before they return: only 8.7 of 100 keystrokes ended in a suggestion still wanted. A 300 ms debounce sends 21.5 requests per 100 keystrokes and raises the useful share per request to 0.21, at the price of a longer wait after a pause. Choose the pair together.

</details>

<details>
<summary><strong>Q2.</strong> `map+graph` sees 0.933 of signatures at 4,000 tokens but has lower body recall than `bm25+graph`. When do you prefer it?</summary>

When the completion mostly writes calls, because a signature is enough and the map buys it cheaply. Prefer `bm25+graph` when the model must read bodies, for example to extend or refactor a helper.

</details>

<details>
<summary><strong>Q3.</strong> A candidate has text similarity 1.00 to the reference and fails the tests. What does that say?</summary>

Similarity measures how a string looks, not what it does; a one-character change to a boundary is invisible to it. Execute against tests, say how many, and remember a weak suite lets a wrong sample pass.

</details>

<details>
<summary><strong>Q4.</strong> You estimate pass@2 from pass@1 as 1 - (1 - p)^2 with p = 0.5. What is wrong, and what do you use?</summary>

It is biased: it gives 0.750 where the unbiased estimator from the same four samples gives 0.833. Use `1 - C(n-c, k) / C(n, k)`.

</details>

<details>
<summary><strong>Q5.</strong> Content exclusion is on for a repository. Name one way an excluded file can still influence suggestions.</summary>

The Copilot documentation says it may use semantic information from an excluded file when the IDE provides it indirectly, such as type definitions or hover data, and exclusions do not apply to symbolic links or remote file systems. Treat exclusion as one layer.

</details>

<details>
<summary><strong>Q6.</strong> The agent tier sends 20,000 prompt tokens per step. How do you stop its cost growing with every step?</summary>

Keep stable material (instructions, repository map) at the front so the provider's prefix cache can reuse it, compact old steps into a summary, and cap steps and tokens per task. In block 1 the assumed 80% cache hit rate is what keeps the agent tier at 3,264 units a day.

</details>

## Further reading

- Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (29 September 2025).
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (19 December 2024), on coding agents and verifiable results.
- Aider documentation, [Repository map](https://aider.chat/docs/repomap.html).
- Cursor documentation, [Codebase indexing](https://cursor.com/docs/context/codebase-indexing).
- Copilot documentation, content exclusion for Copilot (read on 2 October 2026; check the live page, because the list of limits changes).
- Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021).
- Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770) (2023).
- Bavarian et al., [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255) (2022).
- On this site: [testing ML systems](/docs/theory/seml/testing-ml-systems), [agentic systems](/docs/theory/seml/agentic-systems), [guardrails and LLM security](/docs/projects/ai-security/guardrails), [LLM memory](/docs/agentic-ai/llm-memory).

## Check yourself

- I can size an assistant from developers, requests per hour and tokens per request, and say which tier drives prefill load.
- I can explain why debounce and model latency are one decision.
- I can compare retrieval strategies by recall at a token budget, and say what the experiment does not prove.
- I can explain why execution beats text similarity for code, and compute pass@k without bias.
- I can list the layers that keep source code and secrets from leaving the machine, and the hole in each.
- I can write the assistant's cost as a formula and name the parameter I would test first.
