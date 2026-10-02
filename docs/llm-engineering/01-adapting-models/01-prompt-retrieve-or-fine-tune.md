---
id: llme-prompt-rag-finetune
title: "Prompt, Retrieve or Fine-Tune?"
sidebar_label: "Prompt, retrieve or fine-tune"
sidebar_position: 1
slug: /llm-engineering/prompt-retrieve-or-fine-tune
description: "Diagnose what is actually failing (knowledge, behaviour, reasoning or cost), pick the cheapest lever that fixes it, and compute when a tuned small model starts to pay for itself."
tags: [fine-tuning, rag, prompting, prompt-caching, cost, evaluation, decision-framework]
---

import Infographic from '@site/src/components/Infographic';
import AdaptationDecisionLab from '@site/src/components/viz/AdaptationDecisionLab';

**In one line.** Name the failure first, then reach for the cheapest lever that fixes that failure: a better prompt for behaviour, retrieval for knowledge, fine-tuning only when the prompt can no longer carry the load.

:::note Not from a lecture
This chapter is written for this site from the papers, documentation and model card listed under Further reading. Every number in it is printed by the code in the chapter, and the cost figures are synthetic: they show how to compute a break-even, not what any provider charges.
:::

## The idea in plain words

A team has a model that gives bad answers, and someone says "we should fine-tune it". That sentence skips the step that matters. Bad in which way? The three most common ways need different cures.

- **A knowledge gap.** The model does not know your price list, your policy document or last week's incident. It has never seen the facts, so it invents them. Better wording does not help, because nothing in the prompt contains the answer.
- **A behaviour or format gap.** The model knows enough but does not answer the way you need: wrong tone, wrong structure, chatty where you need strict JSON. The knowledge is there; the habit is wrong.
- **A reasoning gap.** The model has the facts and the format but gets multi-step problems wrong. Neither more documents nor a house style fixes arithmetic done badly.

There is a fourth complaint that is not about quality at all: **latency and cost**. The answers are right, but the prompt is three thousand tokens long and you pay for it on every request.

Each failure points to a lever, and the levers are ordered by how cheap they are to try and how easy they are to undo.

| Lever | Fixes | Cost to try | Undo |
| --- | --- | --- | --- |
| Rewrite the prompt, add examples | behaviour, format | minutes | edit a string |
| Retrieval (RAG) | knowledge | days: index, chunking, retrieval quality | change the index |
| Fine-tune (SFT, preference tuning) | behaviour at scale, cost per request | weeks: data, training, evaluation | retrain and redeploy |
| Stronger model, decomposition | reasoning | hours | change a config |

Two published results explain why fine-tuning is the wrong tool for knowledge. Ovadia and colleagues compared unsupervised fine-tuning with retrieval for injecting knowledge and report that retrieval consistently won, for facts the model had seen in pre-training and for entirely new ones. Gekhman and colleagues went further: in their experiments, models learned unfamiliar facts slowly, and once the model did absorb them, each extra unfamiliar example increased its tendency to hallucinate. Their reading is that models acquire factual knowledge mostly in pre-training, and fine-tuning mostly teaches them to use what they already have. So: **fine-tune for how the model answers, retrieve for what it answers with.**

<Infographic src="/img/llme/prompt-retrieve-or-fine-tune-decision-tree.svg" alt="A decision tree that starts from the failure type and leads to prompting, retrieval, a stronger model, or a tuned small model, with the thresholds used in the chapter code" caption="Diagnose, then choose. The thresholds on this board are the ones in block 1 below." />

<Infographic src="/img/llme/prompt-retrieve-or-fine-tune-break-even.svg" alt="Monthly cost of a long prompt, a cached long prompt, retrieval and a tuned small model at three request volumes, with the four break-even points" caption="Where fine-tuning starts to pay. All figures are printed by block 3, on synthetic prices." />

## How it works

### Step 0: build the measuring stick

Before any lever, write a small held-out set of real inputs with the answer you want, and a grader per failure you care about. Without it you cannot tell whether a change helped, and every lever below needs it twice: once to decide, once to confirm. The harness in block 2 is the smallest useful version. It takes any function from input text to output text, runs a labelled set through it, and reports each grader with an interval, because with 16 cases a score of 15 out of 16 is not as certain as it looks. Intervals are Wilson intervals; they are wide on purpose.

### Step 1: the prompt is the cheapest lever, and it has a price

Block 2 uses a running example that continues through the next three chapters: a **support-ticket router** that must reply with strict JSON such as `{"queue": "billing"}`. The model is `SmolLM2-135M-Instruct`, small enough to run on a laptop CPU and bad enough to make the effect visible. With only an instruction it produces no valid JSON at all. Three worked examples in the prompt fix the format for almost every case, and six fix it for all of them. That is the lever working.

It also shows the price. Each example adds tokens to **every** request, and the prompt grew from 55 to 218 tokens. At small scale that is free. At millions of requests it is the bill, and it is the first thing that makes a fine-tuned model attractive: tuning moves the examples out of the prompt and into the weights.

Notice also what examples did not fix: choosing the right queue. The format became reliable long before the judgement did, which is a hint that these two gaps respond to different treatment.

### Step 2: retrieval when the answer lives in your documents

If the failure is a knowledge gap, put the knowledge in the prompt at request time. This is retrieval-augmented generation, from the 2020 paper of Lewis and colleagues, which paired a generator with an explicit external memory over a Wikipedia index and reported better results on open-domain question answering than parametric-only baselines. The practical advantages are the ones fine-tuning cannot offer: the facts can change tomorrow without retraining, the answer can cite its source, and access control can be applied at retrieval time. Retrieval design is a subject of its own; see the [enterprise RAG project](/docs/projects/enterprise-rag/session-1) and the [evaluation chapters](/docs/llm-evals/rag-evaluation-framework).

### Step 3: fine-tune when behaviour must be reliable and cheap

Fine-tuning earns its place when three things are true together: the behaviour you want is stable (it will not change next month), you have or can build hundreds to thousands of good examples, and either the prompt cannot get the behaviour reliable enough or the prompt is expensive enough that moving it into the weights saves money. The next three chapters show how.

Data volume sets expectations. The LIMA paper trained a 65B model on only 1,000 carefully curated examples and argued that almost all knowledge is learned in pre-training while a limited amount of instruction data is enough to teach style and format. That supports the rule of thumb in block 1 (a few hundred good examples can teach a format) while the quality of those examples matters more than their number. It is also an argument against expecting tuning to teach new facts.

### Step 4: price the choice

Block 3 turns the argument into arithmetic. Four options are costed on the same synthetic price list. The long prompt pays per request for 3,000 prompt tokens. The cached variant pays a tenth of the base input price for the cached part of that prompt, which is how provider prompt caching is priced in the documentation opened for this chapter (reads are billed at 0.1 times the base input price for most models, with some models lower, and cache writes at 1.25 times for the five-minute lifetime). RAG pays for roughly 1,000 prompt tokens and a fixed monthly cost for the retrieval system. The tuned small model pays almost nothing per request but carries a fixed monthly cost for amortised training and for the engineer time that keeps it alive. That last number is the one teams forget: **a tuned model is a thing you maintain.**

### What does not move with volume

The break-evens are the easy part of the decision. The hard parts are the ones the calculator cannot see: the tuned model must be re-evaluated each time the base model, the data or the task changes; retrieval needs an owner for the index; and a long prompt, for all its cost, is a string anyone can read and edit. If the volume sits near a break-even, choose the option that is easiest to change.

## A real system that works this way

**Provider prompt caching** is the lever that most changes the calculation in block 3. Anthropic's prompt-caching documentation (read on 2 October 2026) describes caching a prompt prefix so that later requests reuse it: a cache read costs 0.1 times the base input price for most models, with 0.05 and 0.025 for a few named models, a five-minute cache write costs 1.25 times and a one-hour write 2 times, and there is a minimum cacheable length that depends on the model. The effect is that a long, stable prompt is much cheaper than its token count suggests, which pushes the fine-tuning break-even up (block 3: from about 131,000 requests per month to about 617,000 when 90 percent of the prompt is cached). It also illustrates why quoted prices go stale: the multipliers differ by model and the documentation lists them by model name, so treat the 0.1 in the code as a parameter to replace with your provider's current figure.

**The SmolLM2 family** shows the order of operations a real team follows. Its model card describes a two-stage post-training: supervised fine-tuning on public and curated data, then direct preference optimisation on UltraFeedback. The chapters that follow reproduce both stages in miniature on the 135M model.

## Code you can run

Three blocks, all deterministic. Block 1 is the decision logic. Block 2 is the evaluation harness, run on a real model. Block 3 is the break-even calculator. Block 2 downloads the model on first run (about 20 seconds) and then takes about half a minute on a laptop CPU.

### 1. A decision function with thresholds you can argue with

The function encodes the table above. The thresholds (500 labelled examples before tuning for a format, 5,000 verified traces before tuning for reasoning) are rules of thumb, not laws; they are written as plain numbers so a team can change them and see which scenarios flip.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Situation:
    name: str
    gap: str
    facts_change: bool
    labelled: int
    requests: int
    break_even: int


def advise(s):
    steps = ["write the prompt and a held-out eval set"]
    if s.gap == "knowledge":
        steps.append("retrieval over the source documents (RAG)")
        if s.facts_change:
            steps.append("keep the index fresh; do not train facts in")
        return steps
    if s.gap == "format":
        steps.append("few-shot examples in the prompt")
        if s.labelled >= 500:
            steps.append("supervised fine-tuning with LoRA")
        else:
            steps.append("collect more labelled examples before tuning")
        return steps
    if s.gap == "reasoning":
        steps.append("a stronger model or step-by-step decomposition")
        if s.labelled >= 5000:
            steps.append("fine-tune on verified reasoning traces")
        return steps
    steps.append("shorter prompt and prompt caching")
    if s.requests >= s.break_even and s.labelled >= 500:
        steps.append("tune a smaller model on the big model's outputs")
    else:
        steps.append("stay on the large model: volume is below break-even")
    return steps


CASES = [
    Situation("answers about the 2026 price list", "knowledge", True, 0, 20_000, 0),
    Situation("tone must match our brand", "format", False, 120, 20_000, 0),
    Situation("ticket router must emit strict JSON", "format", False, 2_000, 20_000, 0),
    Situation("multi-step refund calculations are wrong", "reasoning", False, 300, 20_000, 0),
    Situation("bill too high, 3k-token prompt", "latency_cost", False, 2_000, 400_000, 130_783),
    Situation("bill too high, small volume", "latency_cost", False, 2_000, 30_000, 130_783),
]

for c in CASES:
    print(f"{c.name}")
    for i, step in enumerate(advise(c), 1):
        print(f"   {i}. {step}")
```

What it prints:

```text
answers about the 2026 price list
   1. write the prompt and a held-out eval set
   2. retrieval over the source documents (RAG)
   3. keep the index fresh; do not train facts in
tone must match our brand
   1. write the prompt and a held-out eval set
   2. few-shot examples in the prompt
   3. collect more labelled examples before tuning
ticket router must emit strict JSON
   1. write the prompt and a held-out eval set
   2. few-shot examples in the prompt
   3. supervised fine-tuning with LoRA
multi-step refund calculations are wrong
   1. write the prompt and a held-out eval set
   2. a stronger model or step-by-step decomposition
bill too high, 3k-token prompt
   1. write the prompt and a held-out eval set
   2. shorter prompt and prompt caching
   3. tune a smaller model on the big model's outputs
bill too high, small volume
   1. write the prompt and a held-out eval set
   2. shorter prompt and prompt caching
   3. stay on the large model: volume is below break-even
```

Read the last two scenarios together. The same complaint, a bill that is too high, gets "tune a smaller model" at 400,000 requests per month and "stay on the large model" at 30,000, because the second is below the break-even of 130,783 that block 3 derives.

### 2. An evaluation harness, and what a prompt buys

The harness is the part to keep. `evaluate` takes any callable from text to text, a list of cases and a dictionary of graders, and returns a count and a Wilson interval per grader. The three systems below differ only in how many worked examples sit in the prompt.

```python
import json
import math
import os
from dataclasses import dataclass
from typing import Callable

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(4)

QUEUES = ["billing", "technical", "account"]
TEST = [
    ("I was charged twice for my subscription this month.", "billing"),
    ("The app crashes every time I open the dashboard.", "technical"),
    ("I forgot my password and the reset email never arrives.", "account"),
    ("Please send me a receipt for last month's payment.", "billing"),
    ("I get a 500 error when I upload a file.", "technical"),
    ("I need to change the email address on my account.", "account"),
    ("The card on file was declined but my bank says it is fine.", "billing"),
    ("Pages take more than a minute to load.", "technical"),
    ("Please delete my account and all my data.", "account"),
    ("I want a refund for the annual plan I bought by mistake.", "billing"),
    ("The export button does nothing when I click it.", "technical"),
    ("My account was locked after too many login attempts.", "account"),
    ("I was billed after I cancelled.", "billing"),
    ("Notifications stopped arriving on my phone.", "technical"),
    ("My teammate needs access to our workspace.", "account"),
    ("The price on my invoice is higher than the one I was quoted.", "billing"),
]
SHOTS = [
    ("My invoice shows an amount I do not recognise.", "billing"),
    ("The mobile app freezes on the login screen.", "technical"),
    ("I cannot enable two-factor authentication.", "account"),
    ("I want to change the credit card used for payments.", "billing"),
    ("The API returns an empty list for my project.", "technical"),
    ("I want to change the name shown on my profile.", "account"),
]


@dataclass
class Case:
    prompt: str
    expected: str
    slice: str


def valid_json(output, expected):
    try:
        return isinstance(json.loads(output)["queue"], str)
    except (ValueError, KeyError, TypeError):
        return False


def correct(output, expected):
    try:
        return json.loads(output)["queue"] == expected
    except (ValueError, KeyError, TypeError):
        return False


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def evaluate(system: Callable[[str], str], cases, graders):
    outputs = [system(c.prompt) for c in cases]
    report = {}
    for name, grade in graders.items():
        hits = sum(grade(o, c.expected) for o, c in zip(outputs, cases))
        low, high = wilson(hits, len(cases))
        report[name] = (hits, len(cases), low, high)
    return report, outputs


name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
SYSTEM = 'Classify the support ticket. Reply with JSON only, for example {"queue": "billing"}. Queues: billing, technical, account.'


def make_system(n_shots):
    prefix = [{"role": "system", "content": SYSTEM}]
    for ticket, queue in SHOTS[:n_shots]:
        prefix += [{"role": "user", "content": ticket}, {"role": "assistant", "content": json.dumps({"queue": queue})}]

    def system(ticket):
        ids = tokenizer.apply_chat_template(prefix + [{"role": "user", "content": ticket}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=12, do_sample=False)
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    system.prompt_tokens = tokenizer.apply_chat_template(prefix + [{"role": "user", "content": TEST[0][0]}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].shape[1]
    return system


cases = [Case(t, q, q) for t, q in TEST]
graders = {"valid JSON": valid_json, "right queue": correct}
print(f"{len(cases)} held-out tickets, greedy decoding, SmolLM2-135M-Instruct")
print(f"{'prompt':<12}{'tokens':>7}  {'valid JSON':<24}{'right queue':<24}")
for shots in (0, 3, 6):
    system = make_system(shots)
    report, outputs = evaluate(system, cases, graders)
    row = [f"{report[g][0]:>2}/{report[g][1]} [{report[g][2]:.2f}, {report[g][3]:.2f}]" for g in graders]
    print(f"{shots}-shot{'':<6}{system.prompt_tokens:>7}  {row[0]:<24}{row[1]:<24}")
print("last output:", repr(outputs[-1]))
```

What it prints:

```text
16 held-out tickets, greedy decoding, SmolLM2-135M-Instruct
prompt       tokens  valid JSON              right queue             
0-shot           55   0/16 [0.00, 0.19]       0/16 [0.00, 0.19]      
3-shot          134  15/16 [0.72, 0.99]       7/16 [0.23, 0.67]      
6-shot          218  16/16 [0.81, 1.00]      11/16 [0.44, 0.86]      
last output: '{"queue": "technical"}'
```

With no examples the model produced valid JSON on 0 of 16 tickets. Three examples lift that to 15 of 16, six to 16 of 16, while the right-queue score climbs from 0 to 7 to 11 of 16. The intervals overlap for 7 versus 11, so on 16 cases you could not claim that six examples classify better than three. The harness is telling you that the test set is too small to rank those two prompts; a real project would collect more cases. The prompt cost is in the second column: 55, 134 and 218 tokens.

### 3. The break-even calculator

Prices are named parameters in a dictionary and carry no real provider's numbers. Change them and the answer changes.

```python
PRICES = {"big_in": 3.0, "big_out": 15.0, "small_in": 0.3, "small_out": 1.5}
CACHE_READ = 0.1


def per_request(prompt_tokens, output_tokens, price_in, price_out, cached_share=0.0):
    effective_in = prompt_tokens * (1 - cached_share) + prompt_tokens * cached_share * CACHE_READ
    return (effective_in * price_in + output_tokens * price_out) / 1_000_000


OPTIONS = {
    "long prompt": dict(prompt=3000, out=20, big=True, fixed=0.0),
    "long prompt, cached": dict(prompt=3000, out=20, big=True, fixed=0.0, cached=0.9),
    "RAG": dict(prompt=1000, out=20, big=True, fixed=600.0),
    "tuned small model": dict(prompt=60, out=20, big=False, fixed=3000 / 12 + 8 * 120),
}

costs = {}
for name, o in OPTIONS.items():
    pi, po = (PRICES["big_in"], PRICES["big_out"]) if o["big"] else (PRICES["small_in"], PRICES["small_out"])
    var = per_request(o["prompt"], o["out"], pi, po, o.get("cached", 0.0))
    costs[name] = (var, o["fixed"])
    print(f"{name:<22} variable {var:.6f} per request, fixed {o['fixed']:>8.2f} per month")

print()
for volume in (10_000, 100_000, 1_000_000):
    row = {n: v * volume + f for n, (v, f) in costs.items()}
    cheapest = min(row, key=row.get)
    print(f"{volume:>9,} requests/month: " + ", ".join(f"{n} {c:,.0f}" for n, c in row.items()) + f"  -> cheapest: {cheapest}")


def break_even(a, b):
    (va, fa), (vb, fb) = costs[a], costs[b]
    return (fb - fa) / (va - vb)


print()
for a, b in [("long prompt", "tuned small model"), ("long prompt, cached", "tuned small model"), ("RAG", "tuned small model"), ("long prompt", "RAG")]:
    print(f"break-even {a} vs {b}: {break_even(a, b):,.0f} requests per month")
```

What it prints:

```text
long prompt            variable 0.009300 per request, fixed     0.00 per month
long prompt, cached    variable 0.002010 per request, fixed     0.00 per month
RAG                    variable 0.003300 per request, fixed   600.00 per month
tuned small model      variable 0.000048 per request, fixed  1210.00 per month

   10,000 requests/month: long prompt 93, long prompt, cached 20, RAG 633, tuned small model 1,210  -> cheapest: long prompt, cached
  100,000 requests/month: long prompt 930, long prompt, cached 201, RAG 930, tuned small model 1,215  -> cheapest: long prompt, cached
1,000,000 requests/month: long prompt 9,300, long prompt, cached 2,010, RAG 3,900, tuned small model 1,258  -> cheapest: tuned small model

break-even long prompt vs tuned small model: 130,783 requests per month
break-even long prompt, cached vs tuned small model: 616,718 requests per month
break-even RAG vs tuned small model: 187,577 requests per month
break-even long prompt vs RAG: 100,000 requests per month
```

At 10,000 requests per month the cached long prompt is cheapest at 20 units, and the tuned model's fixed 1,210 makes it the worst. At a million requests the tuned model wins at 1,258 against 2,010 for the cached prompt. The break-evens in the last four lines are where the lines cross: 130,783 requests per month for a plain long prompt against the tuned model, 616,718 once 90 percent of the prompt is cached.

### Try it: the decision and the cost curves

The lab runs the same decision function and the same cost model as blocks 1 and 3. Its defaults (100,000 requests per month, a 3,000-token prompt and no caching) reproduce the 100,000-request row of block 3: 930 units for the plain long prompt and for RAG, and 1,215 for the tuned model, with a break-even of 130,783 requests. Set the cached share to 0.9 to see 201 units and the 616,718 break-even. Move the volume slider to 1,000,000 to see the tuned model become cheapest at 1,258, and switch the failure type to see the recommended order change.

<AdaptationDecisionLab />

## Designing with it

1. **Write the failure down in one sentence before choosing a lever.** "Answers about pricing are wrong" and "answers are right but 4 seconds slow" are different projects.
2. **Keep the eval set fixed while you try levers.** If the set changes between attempts, you are comparing the sets, not the levers.
3. **Combine, do not choose.** The usual production system is a tuned model that follows your format, fed retrieved facts by a pipeline, behind a prompt that is short because tuning absorbed the examples. The decision is about order of effort, not exclusion.
4. **Do not fine-tune facts that change.** A price list that changes monthly belongs in an index.
5. **Price the maintenance, not just the training.** The 960 units of engineer time in block 3 dominate the 250 of amortised training. That ratio is typical of the way these costs fall.
6. **Re-run the break-even when prices move.** Provider prices and caching rules change; the calculator takes a minute.

## Where this stands in 2026

:::info Industry view
- **The ordering is stable; the thresholds are not.** Prompt first, retrieval for knowledge, tuning for behaviour at scale is how the documentation and papers cited here frame it. What moves is the break-even: caching multipliers and model prices differ by provider and model and change often.
- **The knowledge-injection results are from 2023 and 2024 models.** Ovadia et al. (revised January 2024) and Gekhman et al. (revised October 2024) used models of that period. Treat the direction as well supported and the magnitudes as dated.
- **Long context did not remove retrieval.** A long window lets you put more in the prompt, but you still pay for those tokens on every request unless they are cached, which is the same calculation as block 3.
- **Small tuned models remain the cost lever.** The SmolLM2 family is an example of how capable a small model has become after supervised and preference tuning, which is why distilling a big model's behaviour into a small one is a standard cost move.
- **I could not verify** any claim about which companies run which combination in production; none is made here.
:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A bank's assistant gives wrong answers about its current mortgage rates. A colleague proposes fine-tuning on last quarter's rate sheet. What do you say?</summary>

This is a knowledge gap with facts that change, so tuning is the wrong tool. The rates will be stale after the next change, and the fine-tuning studies cited above found models learn unfamiliar facts slowly and can hallucinate more once they do. Put the current rate sheet behind retrieval, ground the answer in it, and evaluate on questions whose answers changed this quarter.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q2.</strong> In block 2, six examples take the router to 16 of 16 valid JSON but only 11 of 16 correct queues. What does that split tell you about the next step?</summary>

The format gap is closed and the judgement gap is not. More examples in the prompt raise the token bill on every request, so the next options are a stronger model, better label definitions in the prompt, or supervised fine-tuning on a few hundred labelled tickets if the labels are stable. The point is that the numbers separate two failures that "the model is bad" would blur.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q3.</strong> Block 3 gives a break-even of 130,783 requests per month without caching and 616,718 with 90 percent of the prompt cached. Explain the difference and what it means for the decision.</summary>

Caching cuts the per-request cost of the long prompt, so the saving per request from switching to a small tuned model shrinks. The fixed monthly cost of the tuned model is divided by a smaller saving, so the break-even volume rises. If your provider caches prompt prefixes, test that first; tuning for cost only pays at higher volume.<br /><em>Authored · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Why does the harness report a Wilson interval, and what does an interval of [0.23, 0.67] for 7 of 16 mean?</summary>

With 16 cases the observed rate is a rough estimate. The interval says the true rate for this kind of ticket could plausibly be anywhere from 23 to 67 percent. Comparing two prompts whose intervals overlap heavily, such as 7 of 16 and 11 of 16, is not evidence that one is better. The fix is more cases, not a cleverer reading.<br /><em>Authored · interpretation</em>

</details>

<details>
<summary><strong>Q5.</strong> Your prompt carries 40 worked examples and gets 94 percent on the format check. Fine-tuning would reach 97 percent. Give two reasons to stay with the prompt and one to switch.</summary>

Stay: the prompt is readable and editable by anyone and needs no retraining when the task shifts; and below the break-even volume the extra tokens cost less than the tuned model's fixed maintenance. Switch: at very high volume the repeated examples dominate the bill, or the three-point gap matters because each failure is expensive. Decide with the calculator and the eval set, not by preference.<br /><em>Authored · applied</em>

</details>

<details>
<summary><strong>Q6.</strong> The model does multi-step discount calculations wrongly even with retrieved price lists and a strict format. Which lever is next?</summary>

This is a reasoning gap. Try a stronger model or break the calculation into steps the model or ordinary code performs one at a time, and move the arithmetic into a tool. Tuning on a few hundred examples rarely repairs a reasoning gap; block 1 only recommends it past thousands of verified traces.<br /><em>Authored · applied</em>

</details>

## Further reading

- [Ovadia et al., "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs"](https://arxiv.org/abs/2312.05934): retrieval against unsupervised fine-tuning for new and existing knowledge.
- [Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"](https://arxiv.org/abs/2405.05904): why tuning on unfamiliar facts is risky.
- [Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401): the RAG paper.
- [Zhou et al., "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206): a thousand curated examples and the case for data quality.
- [Anthropic documentation: prompt caching](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching): the pricing multipliers, lifetimes and minimum lengths used above.
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct): the model used in the running example and how it was post-trained.
- [Hugging Face TRL documentation: SFT trainer](https://huggingface.co/docs/trl/sft_trainer): the tool the next chapters use.
- Next in this series: [preparing data for fine-tuning](/docs/llm-engineering/preparing-data-for-fine-tuning), then [LoRA fine-tuning](/docs/llm-engineering/supervised-fine-tuning-with-lora). For prompting theory see [in-context learning and prompting](/docs/theory/dnn/in-context-learning-and-prompting).

## Check yourself

- I can sort a complaint about a model into knowledge, behaviour, reasoning or cost, and name the cheapest lever for each.
- I can explain why fine-tuning is a poor way to add facts that change, citing what the studies found.
- I can build an evaluation harness with graders and intervals, and say when a test set is too small to rank two prompts.
- I can state what a long prompt costs on every request and how prompt caching changes that.
- I can compute the monthly break-even between a long prompt, retrieval and a tuned small model, including the maintenance cost people forget.
- I can say which of those inputs I would have to refresh when provider prices change.
