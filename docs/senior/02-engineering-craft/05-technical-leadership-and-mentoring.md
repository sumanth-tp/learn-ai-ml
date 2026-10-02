---
id: senior-leadership
title: "Technical Leadership and Mentoring"
sidebar_label: "Leadership and mentoring"
sidebar_position: 5
slug: /senior/technical-leadership-and-mentoring
description: "What technical leadership looks like on an AI team: choosing a role shape, writing a short technical strategy, deciding at the right speed, giving feedback that names behaviour and impact, and mentoring evaluation discipline with simulations people can rerun."
tags: [technical-leadership, mentoring, feedback, strategy, evaluation-discipline, senior-engineering]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** Technical leadership is mostly improving other people's decisions: set direction in a page, decide at the right speed, give feedback that names behaviour and impact, and teach evaluation discipline with numbers a junior can rerun and see for themselves.

:::note Not from a lecture
This chapter is written for this site from the sources listed under Further reading. The example documents (a technical strategy, 1:1 notes, feedback sentences) are **invented illustrations**. The two simulations are seeded and run below.
:::

## The idea in plain words

The move from engineer to senior engineer changes what your output is. A strong individual contributor ships features. A senior engineer's output is the quality of decisions the team makes when they are not in the room: what to build, how to judge whether it works, when to stop, and how to disagree. Code you write helps one project; a habit you teach helps all of them.

On an AI team that shows up in particular places.

- **Evaluation habits.** The commonest way for an AI project to go wrong is not a bug but a number that nobody should have trusted: a five-example check, a prompt chosen as the best of twenty on one small set, a regression nobody measured. Mentoring on these is the highest-leverage teaching available.
- **Looking at data.** Hamel Husain's post "Your AI Product Needs Evals" (March 2024) says unsuccessful AI products tend to share one root cause, a failure to create robust evaluation systems, and that you must remove all friction from looking at data. A lead's job is to make that cheap and normal.
- **Deciding under uncertainty.** Model choice, scope and rollout are often reversible. Leaders who treat all of them as irreversible stall the team; leaders who treat all as reversible get surprised by the ones that were not.
- **Direction.** A team of strong engineers with no shared diagnosis pulls in several directions. A short written strategy is what turns skill into progress.

<Infographic src="/img/senior/technical-leadership-and-mentoring-archetypes.svg" alt="The four staff-plus archetypes, how decision speed depends on whether a door is two-way or one-way, and the three parts of situation-behaviour-impact feedback with an example from an evaluation review." caption="Roles, decision speed and feedback: the three frames used in this chapter, each from a source listed under Further reading." />

<Infographic src="/img/senior/technical-leadership-and-mentoring-eval-discipline.svg" alt="Two tables from the simulations in this chapter: how often a small check misses a real 10-point regression, and how much the best of k prompt variants is inflated on the tuning set." caption="The two simulations behind the mentoring exercises: five items catch a real regression 6.7% of the time, and the best of 20 equally good prompts looks 11.6 points better than it is." />

## How it works

### Choose the shape of your role

Will Larson's archetypes for staff-plus engineers are a useful vocabulary because they name trade-offs. The **Tech Lead** guides the approach and execution of one team and works closely with its manager. The **Architect** is responsible for direction, quality and approach in a critical area. The **Solver** digs into arbitrarily complex problems and moves between priorities as leadership directs. The **Right Hand** extends an executive's attention, borrowing their scope and authority to run a complex organisation, without managerial duties. Tech Leads and Architects work with consistent teams for years and build deep relationships; Solvers and Right Hands move between fires and tend to work in isolation. Tech Lead roles are the most common; Architect and Right Hand roles typically appear as companies get large.

For AI work, the same split looks like: a tech lead of the LLM feature team, an architect of the shared evaluation and serving platform, and a solver on the incident nobody can explain (the next chapter's subject). Knowing which role you hold stops you doing the other three badly.

### Direction in a page: diagnosis, policy, actions

Richard Rumelt's account of strategy, which his McKinsey Quarterly article "The Perils of Bad Strategy" summarises (I confirmed its three-part kernel only through search summaries, since the article page timed out), says a good strategy has a **diagnosis** of the challenge that picks out what is critical, a **guiding policy** that sets an overall approach and constrains action, and **coherent actions** that carry it out. Bad strategy fails to face the problem, mistakes goals for strategy, or is fluff. Here is a technical strategy at that scale, for an invented internal LLM platform:

```text
Technical strategy: LLM features at the company, next two quarters

Diagnosis
  Six teams each built their own prompt store, their own evaluation script and their own
  model client. Three have no regression test. A model retirement last quarter cost two
  teams about a week each. The scarce resource is not model access; it is trustworthy
  evaluation and the time to migrate.

Guiding policy
  1. Every LLM feature has an evaluation set drawn from real traffic and a paired comparison
     before any model or prompt change ships.
  2. Model identifiers are configuration, never code.
  3. We build one shared evaluation harness and prompt registry; teams may fork, but they
     must use the same result format.
  4. We do not build our own serving stack until a feature passes the break-even in its cost model.

Coherent actions
  Q1  evaluation harness v1 with paired intervals and a power calculator (two engineers)
  Q1  migrate the three teams without regression tests, starting with the highest traffic
  Q2  model aliases and a one-command re-run of every evaluation on a candidate model
  Q2  quarterly review of the cost model against actual spend

What we are not doing
  A custom fine-tuned model for every feature. A new orchestration framework.
```

### Decide at the right speed

The 2016 Amazon shareholder letter distinguishes decisions by reversibility: many are reversible, two-way doors, and "those decisions can use a light-weight process". It also says most decisions should probably be made with somewhere around 70% of the information you wish you had, and recommends a phrase for deadlock: "Look, I know we disagree on this but will you gamble with me on it? Disagree and commit?"

Applied to an AI team: a prompt template, a threshold or a model alias is a two-way door; decide in a day, with the evaluation as the safety net. A data contract, a vendor commitment or a public API is a one-way door; write it up (see [design docs and reviews](/docs/senior/design-docs-and-reviews)) and slow down. When a decision is made over your objection, say plainly that you disagree and commit, and then do commit; it is the only way the team learns from the outcome.

### Feedback that can be acted on

The Center for Creative Leadership's Situation-Behaviour-Impact model has three parts: **situation** (when and where), **behaviour** (observable facts, with no opinions or judgements) and **impact** (the result of the behaviour). It can be extended with a fourth step, intent, which asks what the person was trying to do and turns feedback into a conversation.

| | Weak | SBI |
| --- | --- | --- |
| Review | "Your eval is sloppy." | "In Tuesday's review of the support evaluation set (situation), you graded all 12 failing items yourself with no second grader (behaviour). That means the 8-point gain might be your taste in answers and not the model (impact). What were you trying to get done?" |
| Praise | "Nice work on the migration." | "On the retirement migration (situation), you re-ran all four evaluations before and after and posted the paired intervals in the pull request (behaviour). Three reviewers approved without a call (impact)." |
| Process | "You're always late with estimates." | "In the last two planning sessions (situation) the estimates came with one number each and no assumptions (behaviour). We committed to a date that the data audit then invalidated (impact). Try giving a range and the assumption it depends on." |

### Mentoring evaluation discipline with numbers

Telling a junior "five examples is not enough" rarely lands. Letting them run the numbers does. Two exercises, both in the code below.

**Exercise 1: the small check.** Suppose a new prompt really is worse, with a pass rate of 0.70 against 0.80. A reviewer tries it on five items. The simulation shows the new prompt looks the same or better on 50.6% of such five-item checks, and a 95% interval on the difference shows it is worse only 6.7% of the time. Even 100 items catches it only 37.8% of the time; 300 items catches it 81.2%.

**Exercise 2: the winner's curse.** A junior tries twenty prompt variants on a 50-item set and ships the best. If all twenty are truly 0.70, the best one scores 0.816 on average on the tuning set, and 0.701 on 400 fresh items. The inflation of 0.116 is selection, not skill.

The habits that follow: fix the evaluation set before iterating; **count the variants you tried** and treat the best of them as optimistic; confirm on items the search never saw; report a paired interval. These are the same ideas as in the [build vs buy](/docs/senior/build-vs-buy-and-model-selection) chapter, now taught as habits. The [evaluation workflow](/docs/llm-evals/evaluation-workflow) and [regression testing](/docs/llm-evals/regression-testing) chapters cover the machinery.

### A 1:1 template and a growth plan

```text
1:1, every week, thirty minutes, the report's document, the lead's attention

Their agenda first
  What is blocking you? What decision do you want to make that you are not sure you can?

Then, in this order
  One thing that went well, specifically (SBI)
  One thing to change, specifically (SBI), or "none this week"
  Where are you on the growth plan?

Growth plan, reviewed monthly
  Skill:       evaluation design
  Next level:  can size an eval for the gap that matters and report a paired interval
  Evidence so far: ran the small-check simulation; wrote the power section of one design doc
  Next step:   own the evaluation plan for the next feature end to end, with a review before launch
  Support from me: read the plan before the first run; do not write it

Delegation, honestly
  What I am handing over, what "done" means, what I want to be told and when, and what they decide alone.
```

### The environment

Google's Project Aristotle studied 180 teams (115 engineering project teams and 65 sales pods) in its re:Work write-up and found five dynamics of effective teams, in this order: psychological safety, dependability, structure and clarity, meaning, and impact. Psychological safety is defined there as a shared belief held by members of a team that the team is safe for interpersonal risk taking. For an AI team this is concrete: a junior who has to admit that the eval set was graded by one person, or that the best-of-twenty prompt has not been checked on fresh data, needs to believe that saying so is safe. Everything in this chapter depends on it.

**Anti-patterns**

| Anti-pattern | Instead |
| --- | --- |
| The hero who fixes it all themselves | Teach the habit; fix the system so the next person does not need a hero. |
| Reviewing by rewriting | Ask the question that would have found the problem; let them rewrite. |
| "Trust me" decisions on one-way doors | Write them up, with alternatives and consequences. |
| Treating every decision as a one-way door | Name the two-way doors and decide them fast. |
| Feedback saved for the annual review | Short, specific, weekly. |
| Praise that names a trait ("smart") | Praise a behaviour and its impact. |

## A real system that works this way

**Staff archetypes** are Larson's published taxonomy at staffeng.com; the chapter uses the four named roles and his observations about stability and prevalence as they appear on that page.

**Project Aristotle** is Google's own people-analytics study, published on its re:Work site; the numbers used here (180 teams, the five dynamics, the definition of psychological safety) are from that page.

**SBI** is CCL's feedback model, described in its own article, from which the definitions above are drawn.

## Code you can run

Numpy only, seeded. Both exercises take about a second each.

#### 1. The small check

```python
import numpy as np

rng = np.random.default_rng(21)
trials = 20000
old_rate, new_rate = 0.80, 0.70

print(f"the new prompt really is worse: pass rate {new_rate:.2f} against {old_rate:.2f}, {trials} simulated reviews per row")
print("items   new scores the same or better   95% interval shows it is worse")
for n in (5, 10, 20, 50, 100, 300):
    old = rng.random((trials, n)) < old_rate
    new = rng.random((trials, n)) < new_rate
    gap = old.mean(axis=1) - new.mean(axis=1)
    looks_fine = np.mean(gap <= 0)
    se = np.sqrt(old.var(axis=1, ddof=1) / n + new.var(axis=1, ddof=1) / n)
    caught = np.mean(gap - 1.96 * se > 0)
    print(f"{n:5d}   {looks_fine:28.3f}   {caught:30.3f}")
```

Read the table as a teaching tool. With five items, the new, worse prompt scores the same or better 50.6% of the time, so a reviewer who "tried it and it looked fine" learned nothing. With 20 items it is 28.2%. The right-hand column is the share of runs in which a 95% interval on the difference excludes zero: 0.067, 0.112, 0.112, 0.214, 0.378 and 0.812 for 5, 10, 20, 50, 100 and 300 items. The point for a junior is not the exact numbers but the shape: detecting a ten-point loss on a pass rate of 0.8 takes hundreds of items, and "it looked fine" is the expected result of a small check even when the model is worse.

#### 2. The winner's curse

```python
import numpy as np

rng = np.random.default_rng(8)
true_rate = 0.70
variants, tune_items, fresh_items, repeats = 20, 50, 400, 2000

best_tune, best_fresh = [], []
for _ in range(repeats):
    tune = (rng.random((variants, tune_items)) < true_rate).mean(axis=1)
    winner = int(np.argmax(tune))
    fresh = (rng.random(fresh_items) < true_rate).mean()
    best_tune.append(tune[winner])
    best_fresh.append(fresh)

print(f"{variants} prompt variants, every one truly passes {true_rate:.2f} of the time")
print(f"score of the best variant on the {tune_items}-item tuning set: mean {np.mean(best_tune):.3f}, "
      f"90th percentile {np.percentile(best_tune, 90):.3f}")
print(f"the same variant on {fresh_items} fresh items: mean {np.mean(best_fresh):.3f}")
print(f"average inflation from picking the best of {variants}: {np.mean(best_tune) - true_rate:+.3f}")

for k in (1, 3, 5, 10, 20, 50):
    scores = (rng.random((repeats, k, tune_items)) < true_rate).mean(axis=2).max(axis=1)
    print(f"best of {k:2d} variants: expected tuning score {scores.mean():.3f}")
```

Twenty variants that are all truly 0.70: the best on a 50-item set scores 0.816 on average (90th percentile 0.860), then 0.701 on 400 fresh items, an inflation of 0.116. The ladder at the bottom shows how it grows with the number tried: 0.700 for one variant, 0.755 for three, 0.773 for five, 0.796 for ten, 0.817 for twenty, 0.839 for fifty. A team that tries fifty prompt variants and reports the best has measured its own search effort more than its model.

## Designing with it

**A mentoring plan for evaluation discipline, in four steps**

1. Have the mentee run both simulations and explain the tables back to you.
2. Have them size the evaluation for their next feature: the smallest gap that matters, the items needed, the grader plan.
3. Have them write down how many variants they tried at the end of the work, and report the held-out score separately.
4. Review the first three evaluation reports together, with SBI feedback, and then stop reviewing them.

**Failure modes to name**

- *The silent senior:* knows the evaluation is weak and fixes it quietly instead of teaching the check.
- *The overruled and sulking:* disagrees, does not commit, and the outcome teaches nobody anything.
- *The strategy that is a list of goals:* "improve quality, reduce cost" with no diagnosis and no policy that rules anything out.
- *Feedback without impact:* an observation with no consequence, so the person cannot weigh it.

## Where this stands in 2026

:::info Industry view

- **Evaluation discipline is the thing to teach first.** The argument that unsuccessful AI products lack robust evaluation systems was made in 2024; the simulations here show why a small check misleads.
- **The role vocabulary is stable.** The four staff-plus archetypes are a way to talk about scope and stability, and the same person may hold different ones over a career.
- **Safety first.** Google's research puts psychological safety first of five dynamics; teams that cannot admit weak evaluations cannot fix them.
- **Written strategy beats verbal alignment** as the team and the number of models it uses grow, because the diagnosis can be checked and revised.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A colleague says they tried the new prompt on five examples and it looked fine. Using the simulation, what do you say?</summary>

If the new prompt really is ten points worse, a five-item check looks the same or better 50.6% of the time, so "looked fine" is close to a coin flip and carries no information. A 95% interval on the difference would show the loss only 6.7% of the time. To detect that regression with reasonable reliability you need hundreds of items (81.2% at 300).

</details>

<details>
<summary><strong>Q2.</strong> A junior reports that the best of 20 prompt variants scores 0.82 on the tuning set. What do you ask?</summary>

How many variants were tried, and what is the score on items the search never saw. In the simulation, the best of 20 variants that are all equally good scores 0.816 on the tuning set and 0.701 on fresh items. The tuning score is inflated by selection.

</details>

<details>
<summary><strong>Q3.</strong> Which of these is a two-way door: a prompt template, a public API schema, a model alias?</summary>

The prompt template and the model alias are two-way doors: reversible, safe to decide fast with the evaluation as a net. The public API schema is a one-way door, since clients will depend on it; write it up and slow down.

</details>

<details>
<summary><strong>Q4.</strong> Rewrite "your estimates are bad" as SBI feedback.</summary>

"In the last two planning sessions (situation), your estimates came as one number each, with no assumptions (behaviour). We committed to a date that the data audit then invalidated (impact). Next time, give a range and name the assumption it depends on. What got in the way?" It names observable facts and a consequence, and the last question invites the person's intent.

</details>

<details>
<summary><strong>Q5.</strong> What are the three parts of a good strategy, and which part do most "strategies" omit?</summary>

A diagnosis of the challenge, a guiding policy that sets an approach and rules things out, and coherent actions. Most omit the diagnosis and the policy and present goals ("improve quality") as strategy. The example in the chapter fills each part and ends with what the team is not doing.

</details>

<details>
<summary><strong>Q6.</strong> Why is psychological safety relevant to evaluation quality?</summary>

Because weak evaluation is usually known to the person who did it. Admitting that the eval set was graded by one person, or that the winning prompt was one of twenty, takes the belief that saying so is safe. The re:Work research lists psychological safety first of five dynamics for that reason.

</details>

## Further reading

- [Will Larson, "Staff archetypes" (staffeng.com)](https://staffeng.com/guides/staff-archetypes/): the four roles and how they differ in stability, availability and prevalence.
- [Amazon, 2016 letter to shareholders](https://www.aboutamazon.com/news/company-news/2016-letter-to-shareholders): reversible decisions, 70% of the information, and disagree and commit.
- [Center for Creative Leadership, the SBI model](https://www.ccl.org/articles/leading-effectively-articles/closing-the-gap-between-intent-vs-impact-sbii/): situation, behaviour, impact, and the intent extension.
- [Google re:Work, understand team effectiveness (Project Aristotle)](https://rework.withgoogle.com/intl/en/guides/understand-team-effectiveness): the five dynamics and the definition of psychological safety.
- [Rumelt, "The Perils of Bad Strategy" (McKinsey Quarterly)](https://www.mckinsey.com/capabilities/strategy-and-corporate-finance/our-insights/the-perils-of-bad-strategy): the diagnosis, guiding policy and coherent actions kernel.
- [Hamel Husain, "Your AI Product Needs Evals" (2024)](https://hamel.dev/blog/posts/evals/): why evaluation systems and looking at data decide AI products.
- [Evan Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640): the statistics behind the first exercise.

## Check yourself

- I can describe the four staff-plus archetypes and say which one my role is closest to.
- I can write a one-page technical strategy with a diagnosis, a guiding policy, actions and what we are not doing.
- I can tell a two-way door from a one-way door and set the process to match.
- I can give feedback as situation, behaviour and impact, and ask about intent.
- I can teach evaluation discipline with the small-check and winner's-curse simulations.
- I can run a 1:1 and a growth plan that names evidence and my own support.
- I can explain why psychological safety is a prerequisite for honest evaluation.
