---
id: senior-design-docs
title: "Design Docs and Reviews for AI Systems"
sidebar_label: "Design docs and reviews"
sidebar_position: 4
slug: /senior/design-docs-and-reviews
description: "How to write and review a design doc for an LLM feature: the sections that matter, an architecture decision record, a decision matrix that is tested against its own weights, and a linter that shows what a vague draft looks like next to a precise one."
tags: [design-docs, architecture-decision-records, decision-matrix, reviews, senior-engineering]
---

import Infographic from '@site/src/components/Infographic';
import DecisionMatrixLab from '@site/src/components/viz/DecisionMatrixLab';

**In one line.** A design doc exists to find the disagreement while it is still cheap: state the context, goals and non-goals, the options and why each lost, and the numbers that would prove you wrong, then review it for decisions rather than prose.

:::note Not from a lecture
This chapter is written for this site from the sources listed under Further reading. The draft documents, the scores in the decision matrix and the linter's word list are **examples made up for the exercise**; they show a method, not a standard.
:::

## The idea in plain words

Malte Ubl's account of design docs at Google gives the clearest reason to write one: early identification of design issues is still cheap. The same document gets consensus, makes sure cross-cutting concerns such as security, privacy and observability are considered, spreads senior engineers' knowledge, and records why a decision was taken. It also says when **not** to write one: when the solution is not ambiguous, when the document would read as an implementation manual with no trade-offs, or when the overhead is not compatible with prototyping. Sizes run from a one-to-three page mini doc for an incremental change to ten to twenty pages for a large project, "sufficiently detailed but short enough to actually be read by busy people".

An AI feature needs the same document with extra sections, because it fails in ways ordinary software does not.

- **An evaluation plan,** since correctness is a statistic: how many items, graded by whom, with what interval.
- **Named failure modes** with a control each: invented facts, prompt injection through user text, privacy leaks, drift when the vendor changes the model.
- **A cost model,** since cost scales with tokens and steps (see [cost modelling and ROI](/docs/senior/cost-modelling-and-roi)).
- **A rollout with a quality gate and a rollback trigger written as numbers.**

Two other artefacts sit beside the doc. An **architecture decision record** (ADR) preserves one decision, in a page, for the person who inherits the system. A **decision matrix** compares options on several criteria, and is only trustworthy if you also test how much its answer depends on the weights you chose.

<Infographic src="/img/senior/design-docs-and-reviews-template.svg" alt="The ten sections of a design doc for an LLM feature, each with a prompt for what to write, next to the output of a linter on a bad and a good draft." caption="The first code block: a linter finds 14 problems in a 78-word draft and none in a 269-word draft that states its numbers, non-goals and rollback trigger." />

<Infographic src="/img/senior/design-docs-and-reviews-matrix.svg" alt="A score matrix for three options by six criteria with weighted scores 3.571, 3.429 and 3.143, and the share of 2,000 random weightings each option wins." caption="The second code block: the leader wins 72.2% of weightings, the runner-up 20.3% and the third 7.5%: a close call, stated as one." />

## How it works

### The sections of an AI design doc

| Section | What it must contain | The smell when it is weak |
| --- | --- | --- |
| Context | Measured facts about today: volume, time, cost, pain | Adjectives instead of numbers |
| Goals | Each goal as a number and a date | "Fast", "robust", "high quality" |
| Non-goals | What a reader might assume you will do and you will not | Missing, so scope creeps in review |
| Evaluation | Set size, source, graders, metric, interval, the smallest gap that matters | "We tested it on some examples" |
| Alternatives | Two or more options and the reason each lost | A single option presented as inevitable |
| Design | The call path, the checks on output, the data flow, what is logged | A diagram with no failure paths |
| Risks | Named failures, each with a control | "Hallucinations are possible" |
| Rollout | Stages and the gate between them | "Ship to everyone once it works" |
| Rollback | Trigger numbers and time to undo | No trigger, or a trigger nobody monitors |
| Cost | Tokens per call, volume, budget, alarm | A price per million tokens and nothing else |

Ubl's own list of core contents (context and scope, goals and non-goals, the design with its trade-offs, alternatives considered, cross-cutting concerns) is the skeleton; the evaluation, rollout, rollback and cost rows are this chapter's additions for AI work. His central point is that the doc is the place to suggest solutions **and show why a particular solution best satisfies the goals**: if it only describes what will be built, it is an implementation manual.

### Architecture decision records

Michael Nygard's 2011 proposal is deliberately small. An ADR has a **title** that is a short noun phrase, a **status** (proposed, accepted, deprecated or superseded, with a pointer to the replacement), the **context** (the forces at play, in value-neutral language), the **decision** (full sentences, active voice, "We will ..."), and the **consequences** (the resulting context, positive, negative and neutral). It is one or two pages, written as a conversation with a future developer, in paragraphs rather than bullets. Numbers are sequential and never reused; a reversed decision is kept and marked superseded, because it is still relevant to know it was once the decision.

Here is one for the summarisation example. It is invented, and its numbers echo the synthetic evaluation of the build-vs-buy chapter:

```markdown
# ADR 007: Summarise tickets with a hosted model, prompt only

## Status
Accepted. Review when volume passes 4,000,000 requests a month or when the
vendor announces a retirement of the model we use.

## Context
Agents read a median of 14 messages before replying. We measured a paired
quality gap of 6.7 points between the hosted prompt-only option and a small
self-hosted model on 300 graded tickets, with an interval that excludes zero.
Retrieval of past tickets scored higher on quality but cost more, and its gain
over prompt-only sat inside the noise of the evaluation set.

## Decision
We will call the hosted model with a redacted ticket and one prompt, check
the output for identifiers that are not in the ticket, and show the summary
to the agent without acting on it.

## Consequences
We depend on the vendor's retirement schedule and keep the model name in
configuration. Cost scales linearly with volume. We do not train or host
anything. If volume passes the break-even in the cost model we will write
ADR 008 and supersede this one.
```

### Narratives and the speed of decisions

Amazon is known for replacing slide decks with narrative memos. As quoted in press coverage of Jeff Bezos's 2018 shareholder letter, the company writes "narratively structured six-page memos" and reads one silently at the start of each meeting, "in a kind of 'study hall'". (I could not retrieve the letter text itself, so this rests on that coverage.) The reason it works carries over to design reviews: a narrative forces the writer to say what matters more than what, and a silent read makes sure everyone has read the same thing before talking.

The 2016 letter adds a rule of thumb for how much review a decision deserves. Many decisions are reversible two-way doors, and "those decisions can use a light-weight process". Changing a prompt template or a threshold is a two-way door; a data contract or a vendor commitment is not. Spend your review budget accordingly.

### Reviews that find decisions

A review is not a proofreading pass. A useful reviewer reads for these questions, in this order.

1. **Is the problem stated with a measured baseline?** If not, nothing downstream can be judged.
2. **Are the goals falsifiable?** Could the team fail them?
3. **Were real alternatives considered, and is the reason each lost still true?**
4. **How will we know it works, and how big a gap can that evaluation detect?** Compare with [build vs buy](/docs/senior/build-vs-buy-and-model-selection).
5. **What happens when it fails, and who is paged?**
6. **What is the cheapest experiment that would change the design?**

A comment label helps authors triage: **blocking** (changes the decision or a number), **question** (author must answer in the doc, not only in the thread), **suggestion** (take it or say why not) and **nit** (ignore freely). This labelling is authored practice, not drawn from the sources.

### The decision matrix, and testing it

Score each option 1 to 5 on each criterion, weight the criteria, and compute the weighted average. Three disciplines make it honest.

- **Gates first.** Anything that is a pass or fail requirement removes options before scoring. An average lets a strong score on speed compensate for failing a privacy requirement, which is exactly the compensation the requirement forbids.
- **Test the weights.** Perturb every weight by a random factor between 0.5 and 1.5, many times, and count how often each option wins. A winner that wins 99% of the time is robust; one that wins 72% is a close call, and the document should say so.
- **Record the losers.** The alternatives section says why each option lost, so a reviewer can disagree with a reason.

**Anti-patterns**

| Anti-pattern | Instead |
| --- | --- |
| One option, justified after the fact | Write the alternatives before the design, and update them as you learn. |
| Weights chosen to produce the answer someone wanted | Fix weights before scoring, and test sensitivity. |
| Requirements averaged into scores | Make them gates. |
| A review thread where the author replies and the doc does not change | Answers go in the doc. |
| A design doc for an unambiguous change | Write a mini doc or none. |
| An ADR that records the outcome and not the context | Include the forces, so the next person can tell when it stops being true. |

## A real system that works this way

**Design docs at Google.** Ubl's essay (2020) is a description of a practice: short enough to read, with context and scope, goals and non-goals, the design and its trade-offs, alternatives considered and cross-cutting concerns, and used to catch issues early, build consensus and preserve decisions. It also lists the cases in which the practice does not apply, which is why the chapter's anti-patterns include writing a doc for a decision nobody disputes.

**Architecture decision records.** Nygard's post (2011) is the origin of the ADR format described above, including the storage convention of one numbered Markdown file per decision and the rule that numbers are never reused.

**Narrative memos at Amazon.** The silent six-page memo is a documented practice at a large company, reported from the 2018 letter; the 2016 letter's distinction between reversible and irreversible decisions explains why not every decision needs one.

## Code you can run

#### 1. A design-doc linter on a bad draft and a good draft

The linter checks structure and vagueness, nothing more. It counts the required sections present, vague phrases, quantities with units, and whether the evaluation section states a size. It cannot tell whether the numbers are right.

```python
import re

REQUIRED = ["context", "goals", "non-goals", "evaluation", "alternatives", "design", "risks", "rollout", "rollback", "cost"]
VAGUE = ["fast", "scalable", "robust", "high quality", "accurate", "good enough", "soon", "significant", "as needed", "etc", "tbd", "best practices", "state of the art"]

BAD = [
    ("Context", "Agents spend too long reading tickets. We want to use AI to fix this."),
    ("Goals", "Build a fast, scalable and robust summariser with high quality output."),
    ("Design", "We will call a state of the art model with the ticket text and return a summary. Prompt will be tuned as needed."),
    ("Risks", "Hallucinations are possible. We will use best practices."),
    ("Rollout", "Ship to everyone once it works."),
]

GOOD = [
    ("Context", "Support agents read a median of 14 messages per ticket before replying; 1,200 tickets a day; handling time is 11 minutes."),
    ("Goals", "Cut median time to first reply by 20% (from 11 to 8.8 minutes) for tickets longer than 8 messages.\n"
              "Summary factual-consistency at least 95% on a 300-ticket graded set."),
    ("Non-goals", "Replying to customers automatically.\n"
                  "Summarising tickets in languages other than English and German in this phase."),
    ("Evaluation", "300 graded tickets, two reviewers, paired comparison against the current template; 95% CI reported.\n"
                   "Weekly sample of 100 production summaries graded by the same rubric."),
    ("Alternatives", "Option A: hosted API, prompt only. Option B: hosted API plus retrieval of past tickets. Option C: small self-hosted model.\n"
                     "Chosen: A with redaction, because B costs more for a gain inside the eval noise and C waits until volume passes the break-even in the cost model."),
    ("Design", "Ticket text goes through a redaction step, then one model call with a 900-token prompt; output is checked for ids that are not in the ticket."),
    ("Risks", "Invented order numbers: blocked by the id check. Prompt injection from ticket text: summary is shown, never executed."),
    ("Rollout", "5% of agents for 1 week, then 25%, then all, gated on the weekly graded sample staying at or above 95%."),
    ("Rollback", "Feature flag off in under 5 minutes; triggered if graded consistency drops below 92% or p95 latency exceeds 6 seconds."),
    ("Cost", "About 1,900 input and 120 output tokens per summary; 1,200 tickets a day; budget 400 per month, alarm at 80%."),
]

def render(sections):
    return "# Ticket summaries with an LLM\n" + "".join(f"## {title}\n{body}\n" for title, body in sections)

def lint(name, text):
    headings = [h.strip().lower() for h in re.findall(r"^## (.+)$", text, re.M)]
    missing = [r for r in REQUIRED if not any(r in h for h in headings)]
    lower = text.lower()
    vague = sorted({w for w in VAGUE if re.search(r"\b" + re.escape(w) + r"\b", lower)})
    numbers = len(re.findall(r"\d[\d,.]*\s?(?:%|minutes|seconds|tickets|tokens|days|week)", lower))
    has_eval_size = bool(re.search(r"\d+\s+(?:graded|tickets|sample)", lower))
    words = len(text.split())
    problems = len(missing) + len(vague) + (0 if has_eval_size else 1) + (0 if numbers >= 6 else 1)
    print(f"{name}: {words} words, {numbers} quantities, {len(headings)} sections")
    print(f"  missing sections: {missing if missing else 'none'}")
    print(f"  vague phrases: {vague if vague else 'none'}")
    print(f"  states an eval size: {has_eval_size}; problems found: {problems}")

lint("bad draft ", render(BAD))
lint("good draft", render(GOOD))
```

The bad draft has 78 words, no quantities, five of the ten sections and seven vague phrases ("fast", "scalable", "robust", "high quality", "state of the art", "best practices" and "as needed"): 14 problems. The good draft has 269 words, 15 quantities, all ten sections, no vague phrases and an evaluation size: none. A linter like this belongs in the repository next to the docs, run in review, because it removes the cheapest class of comment and leaves reviewers to argue about decisions.

#### 2. A decision matrix with a weight test

The three options are the ones from the ADR: a hosted API with a prompt only, a hosted API with retrieval, and a small self-hosted model. The scores are invented. The weight test uses a 32-bit seeded generator (the same one the lab uses) so the browser reproduces it exactly.

```python
CRITERIA = ["answer quality", "time to ship", "run cost at volume", "privacy and residency", "on-call load", "reversibility"]
WEIGHTS = [5, 4, 3, 4, 3, 2]
OPTIONS = {
    "hosted API, prompt only": [3, 5, 3, 2, 5, 4],
    "hosted API plus retrieval": [5, 4, 2, 2, 3, 4],
    "small fine-tuned model, self-hosted": [3, 1, 5, 5, 2, 3],
}

def mulberry32(seed):
    state = seed & 0xFFFFFFFF

    def rnd():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF)
        t = t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rnd

def totals(weights):
    s = sum(weights)
    return {name: sum(w * x for w, x in zip(weights, scores)) / s for name, scores in OPTIONS.items()}

base = totals(WEIGHTS)
print("weighted score out of 5 with weights", WEIGHTS)
for name, value in sorted(base.items(), key=lambda kv: -kv[1]):
    print(f"  {value:.3f}  {name}")

rnd = mulberry32(99)
wins = {name: 0 for name in OPTIONS}
draws = 2000
for _ in range(draws):
    jittered = [w * (0.5 + rnd()) for w in WEIGHTS]
    result = totals(jittered)
    best = max(result, key=lambda k: result[k])
    wins[best] += 1
print(f"\nwin share when every weight is scaled by a random factor between 0.5 and 1.5 ({draws} draws)")
for name, count in sorted(wins.items(), key=lambda kv: -kv[1]):
    print(f"  {count / draws:.3f}  {name}")

privacy = CRITERIA.index("privacy and residency")
survivors = [name for name, scores in OPTIONS.items() if scores[privacy] >= 3]
print(f"\nwith privacy as a gate (score of at least 3 to stay in): {survivors}")
```

With the weights 5, 4, 3, 4, 3, 2, the prompt-only API scores 3.571, retrieval 3.429 and the self-hosted model 3.143. The 0.14 lead looks decisive and is not: across 2,000 random weightings the prompt-only option wins 72.2% of the time, retrieval 20.3% and the self-hosted model 7.5%. The last line is the lesson about gates. If a privacy score of at least 3 is a hard requirement, both API options go and only the self-hosted model survives, though it has the lowest average. (This is a stricter privacy requirement than the ADR above assumed, where redaction lets the API options stand.)

The lab lets you move the weights and the gate. Defaults reproduce the scores 3.571, 3.429 and 3.143 and the win shares 0.722, 0.203 and 0.075.

<DecisionMatrixLab />

## Designing with it

**A sequence for a new design doc**

1. Write the context and the goals as numbers. If you cannot, the first task is to measure.
2. Write the non-goals before the design.
3. List the alternatives, including "do nothing" and "no model".
4. Write the evaluation plan and the smallest gap that matters before running anything.
5. Draft the design with its failure paths, rollout and rollback triggers.
6. Run the linter. Fix what it finds. Review with the six questions.
7. Record the decision as an ADR and link it from the code.

**Failure modes to name**

- *The doc as ceremony:* written after the code, reviewed by nobody who could say no.
- *The silent assumption:* the doc relies on a vendor retiring nothing, or on data quality nobody measured.
- *The never-updated doc:* a decision that was true in March and is wrong in October, with no status field to say so.
- *The matrix as theatre:* weights tuned until the preferred option wins; the weight test is what gives it away.

## Where this stands in 2026

:::info Industry view

- **The core of the practice is old and stable.** The format of Nygard's ADR dates from 2011 and Ubl's description of Google's design docs from 2020; neither is specific to AI work, which only adds sections.
- **What AI adds is the evaluation and failure sections.** A design doc for an LLM feature that has no evaluation plan, no named failure modes and no rollback numbers is the same document with the important parts missing.
- **Vague goals are now cheap to generate.** A model can write a fluent "fast, scalable and robust" paragraph in a second; a linter or a reviewer asking "what number?" is the counterweight.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> The linter reports 14 problems on a draft. Is fixing them enough to approve it?</summary>

No. The linter checks form: sections, vague words and the presence of numbers. It cannot tell whether the numbers are right, whether option A really lost for the reason given or whether the risk list is complete. Passing the linter is the entry condition for a human review, not the result of one.

</details>

<details>
<summary><strong>Q2.</strong> The matrix gives 3.571, 3.429 and 3.143. A colleague says the first option clearly wins. What do you do?</summary>

Run the weight test. The first option wins 72.2% of random weightings and the second 20.3%, so the lead depends on the weights. State that in the document and decide on something the matrix leaves out, such as cost of leaving or risk, or measure one of the uncertain scores more carefully.

</details>

<details>
<summary><strong>Q3.</strong> Why should privacy be a gate and not a weighted criterion?</summary>

A weighted average lets a high score on speed make up for a failure on privacy, but a privacy requirement is pass or fail. With a gate of a score of at least 3, both API options are removed before scoring; with weights alone the self-hosted model is last. The two procedures give different answers, and only the gate respects the requirement.

</details>

<details>
<summary><strong>Q4.</strong> When should you not write a design doc?</summary>

When the solution is not ambiguous, when the document would only describe an implementation with no trade-offs, or when the review overhead does not fit prototyping, as in the Google essay. A mini doc of one to three pages is the right size for a small change.

</details>

<details>
<summary><strong>Q5.</strong> What belongs in the status field of an ADR, and why does it matter?</summary>

Proposed, accepted, deprecated or superseded, with a pointer to the replacement. It matters because decisions stop being true. The old ADR stays, marked superseded, so a later reader can see both what was decided and that it no longer applies.

</details>

<details>
<summary><strong>Q6.</strong> Write a non-goal for the ticket-summary feature that prevents a likely misunderstanding.</summary>

"Replying to customers automatically." Readers see an LLM and assume it will draft or send replies; the non-goal says the output is a summary shown to an agent, never an action. Another: "Summarising languages other than English and German in this phase."

</details>

## Further reading

- [Malte Ubl, "Design Docs at Google" (2020)](https://www.industrialempathy.com/posts/design-docs-at-google/): purposes, length, contents and when not to write one.
- [Michael Nygard, "Documenting Architecture Decisions" (2011)](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions): the ADR format and its conventions.
- [Amazon, 2016 letter to shareholders](https://www.aboutamazon.com/news/company-news/2016-letter-to-shareholders): reversible decisions and light-weight process.
- [Amazon, 2018 letter to shareholders](https://www.aboutamazon.com/news/company-news/2018-letter-to-shareholders): the source of the six-page memo practice; I could not retrieve the passage from this page, so the quotations above come from press coverage of it.
- [Evan Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640): how to size an evaluation for the gap you care about.
- [Zinkevich, "Rules of Machine Learning" (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml): practical design advice for ML systems.

## Check yourself

- I can write the ten sections of an AI design doc and say what each must contain.
- I can write non-goals and an alternatives section that records why each option lost.
- I can write an ADR with context, decision and consequences, and mark a decision superseded.
- I can run a decision matrix, test its weights and say how close the call really is.
- I can turn a pass-or-fail requirement into a gate.
- I can review a design doc with six questions and label my comments so the author can triage.
- I can say when a design doc is the wrong tool.
