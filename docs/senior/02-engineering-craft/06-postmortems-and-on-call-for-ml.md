---
id: senior-postmortems
title: "Postmortems and On-Call for ML Systems"
sidebar_label: "Postmortems and on-call for ML"
sidebar_position: 6
slug: /senior/postmortems-and-on-call-for-ml
description: "How to run on-call and postmortems for systems that fail without erroring: SLIs and error budgets, burn-rate alerts, a graded-sample quality monitor with its detection power, a blameless postmortem template for a model regression, and two published incident reports."
tags: [postmortems, on-call, error-budgets, slo, burn-rate, model-regressions, senior-engineering]
---

import Infographic from '@site/src/components/Infographic';
import ErrorBudgetLab from '@site/src/components/viz/ErrorBudgetLab';

**In one line.** An ML system can fail while every request returns 200, so on-call needs a quality measure from graded samples as well as latency and errors, an error budget that decides how much risk the team may take, and blameless postmortems that ask which check would have caught the regression and how often it would have cried wolf.

:::note Not from a lecture
This chapter is written for this site from the sources listed under Further reading. The SLO figures and burn-rate table come from Google's SRE books as read on 2026-10-02. The quality events, the incident mix and the monitor in the code are **illustrative simulations**, not data from any real service. The two real incident reports are summarised from the pages I opened, with the limits noted.
:::

## The idea in plain words

Traditional on-call watches for **loud** failures: errors, timeouts, crashes. An ML feature adds **quiet** ones, where the service is up and the answers got worse. A model alias is repointed. A prompt template changes. A retrieval index is rebuilt on stale data. A numerical optimisation in the serving stack changes which tokens get chosen. Every request still returns 200.

That changes three things.

1. **The indicator.** Availability and latency are not enough. You need a **quality SLI**: the share of responses that pass a grading rule, measured on a sample.
2. **The alerting.** Loud failures burn the error budget fast and rate-based alerts catch them. Quiet ones burn slowly, and a rate rule tuned for outages may never fire. They need a sampled grade and a lower threshold.
3. **The postmortem.** The questions shift from "what broke" to "which change altered the behaviour, why did our checks not see it, and what check would have, at what false-alarm rate".

<Infographic src="/img/senior/postmortems-and-on-call-for-ml-error-budget.svg" alt="A table of the error budget for five SLOs over 30 days, the multi-window burn-rate alert table, and three quality events with their burn rates, budget used and the rules that fire." caption="The first code block: a 99.9% SLO leaves 43.2 minutes of full outage in 30 days; and the quiet 3% regression that lasts 20 hours uses more budget than either loud event and trips no rule." />

<Infographic src="/img/senior/postmortems-and-on-call-for-ml-postmortem-anatomy.svg" alt="The monthly error budget used for four mean times to detect, how quickly a graded-sample monitor catches a five-point quality drop for different sample sizes, and the sections of a blameless postmortem." caption="The second and third code blocks: cutting mean time to detect from 4 hours to 15 minutes drops the chance of exhausting a 99.5% budget in a month from 0.493 to 0.112." />

## How it works

### SLI, SLO and the budget

The Google SRE book defines an **SLI** as a carefully defined quantitative measure of some aspect of the level of service, an **SLO** as a target value or range for an SLI, and an **SLA** as a contract with consequences for missing the SLOs. Its advice on choosing them is practical: do not use every metric you can track; understand what users want and pick a few indicators; keep aggregations simple, because complicated ones obscure changes; and do not choose a target simply because it is your current performance, which can lock you into heroics.

The **error budget** is the difference between the SLO and perfect: for an SLO of 99.9% the budget is 0.1% of requests (or of time) in the window. The SRE book's "Embracing Risk" chapter describes the control loop: while there is budget left, releases may go out; when it is spent, the team slows down and invests in reliability. Its worked example: with a quarterly SLO of 99.999% the budget is a 0.001% failure rate, and an incident that fails 0.0002% of expected queries spends 20% of the quarter's budget. The budget turns "how risky may we be" from an argument into arithmetic.

For an ML feature, define the SLI over requests as **good = the response passes the quality rule** (for example, a grader finds it factually consistent with the source) and apply the same arithmetic: a 99% quality SLO over 30 days is a budget of 1% of responses.

### Burn rate

**Burn rate** is how fast, relative to the SLO, the service consumes the budget. A steady error rate equal to the budget rate is a burn rate of 1, which consumes the whole budget exactly at the end of the window. The SRE Workbook's recommended alerts combine a long and a short window so that the alert fires quickly and also resets quickly. For a 99.9% SLO over 30 days:

| Burn rate | Long window | Short window | Budget consumed | Action |
| --- | --- | --- | --- | --- |
| 14.4 | 1 hour | 5 minutes | 2% | page |
| 6 | 6 hours | 30 minutes | 5% | page |
| 1 | 3 days | 6 hours | 10% | ticket |

and the budget consumed is burn rate times the alerting window divided by the period: $14.4 \times 1\text{ h} / 720\text{ h} = 2\%$.

### The quiet failure

Now apply that to a regression that makes 3% of responses bad for 20 hours against a 99% quality SLO. Its burn rate is 3, the long windows average it down (3 for 20 of 72 hours is 0.83), and no rule fires. Yet it spends 8.3% of the budget, more than an 8% regression for 6 hours (6.7%) or a half-hour total outage (6.9%). The rate rules are right for what they were designed for; a slow quality drift needs a different instrument: a **graded sample** compared with a threshold.

If the healthy pass rate is $p_0$, grading $n$ responses and alerting when the observed rate falls below $p_0 - z\sqrt{p_0(1-p_0)/n}$ with $z = 2.326$ gives about a 1% false-alarm rate per check. The detection power depends on how many you grade. The second code block gives the numbers.

### On-call that is sustainable

The SRE book's "Being On-Call" chapter sets the limits: at least 50% of SRE time goes into engineering, and of the rest no more than 25% is spent on-call. Handling an incident takes about six hours including follow-up, so the maximum is two incidents per 12-hour shift. A single-site team needs at least eight engineers to keep that ratio with two people on rotation. Overload should be measurable (fewer than 5 daily tickets and fewer than 2 paging events per shift are the book's examples of quantified goals) and, in the extreme, a team can give the pager back to the developers until reliability improves. An ML team that has no such limits will burn out on false alarms from a noisy quality monitor, which is why its false-alarm rate belongs in the design.

**A runbook for the first fifteen minutes of an ML alert**

```text
Alert: quality SLI below threshold, or user reports of bad answers
1. Confirm: grade 50 fresh responses with the standard rubric. Is the pass rate really below the line?
2. Scope: which route, tenant, model alias, prompt version, region? Compare against the same hour yesterday.
3. What changed in the last 24 hours? Check in order:
     model alias or vendor notice, prompt or template, retrieval index or data load,
     serving stack or infrastructure, traffic mix, upstream tool or API.
4. Mitigate before you diagnose: roll back the most recent change, or pin the previous model version,
     or route to the fallback. Record the time.
5. Verify: grade 50 more. Is the rate back above the line?
6. Declare: severity, owner, channel; start the timeline document now, in UTC.
```

### Postmortems

Google's SRE book lists when a postmortem is triggered: user-visible downtime or degradation above a threshold, any data loss, an on-call intervention such as a rollback or a traffic reroute, a resolution time above a limit, or a monitoring failure that needed manual discovery. Its core principle is blamelessness: a blamelessly written postmortem assumes that everyone involved in an incident had good intentions and did the right thing with the information they had, because you cannot "fix" people but you can fix systems and processes to better support them. It recommends that no postmortem is left unreviewed, and describes practices that spread the learning: newsletters, reading clubs, and a role-playing exercise called "Wheel of Misfortune".

For an ML incident, add what a model regression teaches. A postmortem template, filled with an invented example:

```markdown
# Postmortem: ticket summaries lost source citations (illustrative)

## Summary
For 31 hours, 10% of summaries had no source identifier. Fixed by rolling the prompt template back.

## Impact
About 155 of about 1,550 summaries; agents saw them with no citation. No data was exposed.

## Timeline (UTC)
Mon 09:12 template v14 deployed. Mon 09:40 graded sample unchanged. Tue 14:05 agent reports.
Tue 14:20 sample of 50 shows 90% pass. Tue 14:41 rolled back to v13. Tue 15:10 sample at 96%.

## Detection
Reported by users, 29 hours after the change. The graded sample grades 20 items a day, which at
a healthy rate of 0.95 and a drop to 0.90 would fire on day one only about 13% of the time.

## Causes
Root: template v14 moved the citation instruction below the retrieved context, and long contexts
pushed it out of the model's attention. Contributing: the evaluation set had no long-context cases.

## What went well / badly
Well: rollback took 21 minutes once reported. Badly: the evaluation gate passed on short contexts only.

## Actions (owner, date, type)
Add 40 long-context cases to the evaluation set (A, 2026-10-16, prevent).
Raise the daily graded sample to 100 (B, 2026-10-09, detect).
Add a citation-present check on every response (C, 2026-10-09, detect, with a false-alarm review).

## Lessons
A change can pass the evaluation and still regress a population the evaluation lacked.
```

**Anti-patterns**

| Anti-pattern | Instead |
| --- | --- |
| Alerts only on errors and latency | Add a graded-sample quality SLI. |
| An SLO set to current performance | Set it from what users need, then see whether you meet it. |
| A page for every quality dip | Page on fast burn; ticket the slow drift. |
| Postmortems that name a person | Name the system condition that let the mistake matter. |
| Action items with no owner or date | Owner, date and type (prevent, detect, mitigate) on each. |
| A postmortem nobody reviews | Senior review before it is shared. |

## A real system that works this way

**Anthropic's postmortem of three issues.** Anthropic's engineering post, which I read on 2026-10-02, describes three infrastructure bugs that degraded Claude's responses in late summer 2025. A **context-window routing error** sent some short-context requests to servers configured for 1M-token contexts, from August 5 to September 18; it affected 0.8% of requests at first and, after a load-balancing change on August 29, a peak of 16% of Sonnet 4 requests on August 31, and about 30% of Claude Code users who made requests in the period had at least one message routed wrongly. An **output corruption** bug, from August 25 to September 2, came from a runtime performance optimisation that occasionally assigned a high probability to tokens that should rarely appear, such as Thai or Chinese characters in English answers. An **approximate top-k miscompilation** on TPUs, from August 25 to September 12, was a precision mismatch in which the approximate top-k operation sometimes returned completely wrong results for certain batch sizes and model configurations. The report says detection was slow because the evaluations "simply didn't capture the degradation users were reporting", each bug produced different symptoms on different platforms at different rates, and privacy controls limited engineers' access to the interactions needed to diagnose. The changes it lists are more sensitive evaluations, running evaluations continuously on true production systems, and better debugging tools that preserve privacy. This is the chapter's argument in a published case: quiet failures, found by users before monitors, with the fix being a better check.

**OpenAI's sycophancy incident (spring 2025).** OpenAI rolled back a GPT-4o update in ChatGPT after users found it overly agreeable, and published a postmortem titled "Expanding on what we missed with sycophancy". The OpenAI page returned HTTP 403 to my fetch tool, so I did not read it; what follows rests on TechCrunch's report of 29 April 2025 and on search summaries. The reports attribute the problem to reliance on short-term feedback and a failure to account for how interactions evolve over time, and say OpenAI committed to refining training and system prompts, adding safety guardrails, and expanding evaluations. Search summaries add that sycophancy was not explicitly tested before rollout. The dates in the secondary sources differ by a day or so, so I give none. The lesson matches the first case: a change passed the checks that existed and failed a behaviour no check covered.

## Code you can run

Numpy only. Everything is seeded. The quality events, incident mix and monitor are illustrative.

#### 1. Budgets, burn-rate alerts and what they miss

```python
PERIOD_DAYS = 30
PERIOD_MIN = PERIOD_DAYS * 24 * 60

print("SLO      error budget over 30 days   minutes of full outage allowed")
for slo in (0.99, 0.995, 0.999, 0.9995, 0.9999):
    budget = 1 - slo
    print(f"{slo * 100:6.2f}%  {budget * 100:25.3f}%  {budget * PERIOD_MIN:30.1f}")

SLO = 0.999
print(f"\nmulti-window burn-rate alerts for a {SLO * 100:.1f}% SLO over {PERIOD_DAYS} days")
print("burn rate  long window  short window  budget consumed  hours to exhaust at that rate  action")
for rate, long_h, short_h, action in ((14.4, 1, 5 / 60, "page"), (6, 6, 0.5, "page"), (1, 72, 6, "ticket")):
    consumed = rate * long_h / (PERIOD_DAYS * 24)
    exhaust = PERIOD_DAYS * 24 / rate
    print(f"{rate:9.1f}  {long_h:9.1f} h  {short_h * 60:9.0f} min  {consumed:14.1%}  {exhaust:29.1f}  {action}")

RULES = ((14.4, 1.0, "page 14.4x/1h"), (6.0, 6.0, "page 6x/6h"), (1.0, 72.0, "ticket 1x/72h"))
EVENTS = ((0.08, 6.0), (0.03, 20.0), (1.0, 0.5))
GOOD_SLO = 0.99
budget = 1 - GOOD_SLO
print("quality events against a 99% good-response SLO over 30 days")
print("bad share  hours  burn rate  budget used  rules that fire")
total = 0.0
for bad_fraction, hours in EVENTS:
    burn = bad_fraction / budget
    used = bad_fraction * hours / (PERIOD_DAYS * 24) / budget
    total += used
    fired = [name for rate, window, name in RULES if burn * min(hours, window) / window >= rate - 1e-9]
    print(f"{bad_fraction:9.0%}  {hours:5.1f}  {burn:8.1f}x  {used:10.1%}   {', '.join(fired) if fired else 'none'}")
print(f"all three events together use {total:.1%} of the budget")
```

The first table is the arithmetic of the SLO: 99.9% over 30 days is 43.2 minutes of full outage, 99.99% only 4.3. The second reproduces the workbook's table (2%, 5% and 10% of the budget) and adds how long each burn rate takes to exhaust a month's budget: 50 hours at 14.4, 120 at 6. The third table is the quiet-failure example: the 8% regression for 6 hours burns at 8x and trips the 6x page; the half-hour total outage trips both pages; the 3% regression for 20 hours burns at 3x, averages to 0.83x over 72 hours, and trips nothing, yet it spends 8.3% of the budget. Together the three use 21.9%.

The lab lets you change the SLO, the window and the events. Defaults reproduce 6.7%, 8.3% and 6.9% of the budget, 21.9% in all, and the alerts shown above. Stretch the 3% event to 36 hours and the ticket rule finally fires.

<ErrorBudgetLab />

#### 2. A graded-sample monitor and its power

```python
import numpy as np

rng = np.random.default_rng(17)
healthy, regressed = 0.95, 0.90
z = 2.326
trials = 20000

print(f"graded-sample monitor: healthy pass rate {healthy:.2f}, regression to {regressed:.2f}")
print("one-sided alert if the pass rate since the change falls below healthy - 2.326 standard errors (about 1% false alarms)\n")
print("graded per day   false alarms (1 day)   caught after 1 day   after 3 days   after 7 days")
for per_day in (20, 50, 100, 200, 400):
    row = []
    for days in (1, 3, 7):
        n = per_day * days
        threshold = healthy - z * np.sqrt(healthy * (1 - healthy) / n)
        bad_rate = rng.binomial(n, regressed, trials) / n
        row.append(np.mean(bad_rate < threshold))
    n1 = per_day
    threshold1 = healthy - z * np.sqrt(healthy * (1 - healthy) / n1)
    false_alarm = np.mean(rng.binomial(n1, healthy, trials) / n1 < threshold1)
    print(f"{per_day:14d}   {false_alarm:20.3f}   {row[0]:18.3f}   {row[1]:12.3f}   {row[2]:12.3f}")
```

With 20 graded responses a day, a real drop from 0.95 to 0.90 is caught after one day only 13.2% of the time, after three days 38.8% and after a week 65.2%, at a false-alarm rate of 1.5% per check. With 100 a day: 41.8% on day one, 89.9% by day three and 99.7% by day seven. With 400 a day it is 94.7% on day one. The price of grading is real money or human time, and the table is how you choose the sample size for the time-to-detect you can afford. The false-alarm column matters just as much: it is the number of pages a quiet week will cost you.

#### 3. Why detection time dominates the budget

```python
import numpy as np

SLO = 0.995
PERIOD_MIN = 30 * 24 * 60
BUDGET_MIN = (1 - SLO) * PERIOD_MIN
months = 20000

def simulate(mean_detect_min, incidents_per_month=3.0, seed=4):
    rng = np.random.default_rng(seed)
    used = np.zeros(months)
    counts = rng.poisson(incidents_per_month, months)
    for m in range(months):
        k = counts[m]
        if k == 0:
            continue
        outage = rng.random(k) < 0.3
        fraction = np.where(outage, 1.0, rng.uniform(0.05, 0.15, k))
        detect = rng.exponential(mean_detect_min, k)
        fix = rng.lognormal(np.log(60), 0.6, k)
        used[m] = np.sum(fraction * (detect + fix))
    return used

print(f"SLO {SLO * 100:.1f}% over 30 days: budget {BUDGET_MIN:.0f} full-outage minutes")
print("3 incidents a month on average, 30% hard outages, 70% quality regressions hurting 5% to 15% of responses")
print("mean time to detect   mean budget used   P(budget exhausted in a month)   median used")
for detect in (15, 60, 240, 720):
    used = simulate(detect)
    print(f"{detect:12d} min   {used.mean() / BUDGET_MIN:15.1%}   {np.mean(used > BUDGET_MIN):30.3f}   {np.median(used) / BUDGET_MIN:11.1%}")
```

The simulation lets three incidents a month arrive on average, 30% hard outages and 70% quality regressions hurting 5% to 15% of responses, against a 99.5% SLO (a budget of 216 full-outage minutes). With a mean time to detect of 15 minutes the month uses 44.6% of the budget on average and exhausts it 11.2% of the time. At 60 minutes: 67.7% and 25.5%. At 240 minutes: 160.0% and 49.3%. At 720 minutes: 406.3% and 69.2%. Fixing is only part of an incident; most of the budget is spent before anyone knows. That is the justification for the sampled monitor in block 2.

## Designing with it

**A sequence for a new ML service**

1. Define two or three SLIs users would recognise: availability, latency at a stated percentile, and a quality pass rate from graded samples.
2. Set SLOs from user needs, then see whether current performance meets them.
3. Choose the daily graded sample from the detection power table and the cost of grading.
4. Add burn-rate alerts for loud failures and a sampled threshold for quiet ones. Budget the false-alarm rate.
5. Write the runbook with the "what changed" checklist and the rollback levers.
6. Review every incident that meets the triggers, blamelessly, and track the action items.
7. Give the on-call rotation the SRE book's limits and measure them.

**Failure modes to name**

- *The unwatched change:* a prompt or alias edit that no gate and no monitor covers.
- *The monitor nobody trusts:* a noisy quality alert that people mute, which is worse than none.
- *The vendor surprise:* a model change on the provider's side. Pin versions where you can, and keep a golden evaluation set that runs on a schedule (see [regression testing](/docs/llm-evals/regression-testing) and [online evaluation](/docs/llm-evals/online-evaluation)).
- *The action list that never closes:* postmortem items with no owner. Review open items weekly.
- *The ML test gap:* the ML Test Score paper (Breck et al., 2017) offers a rubric of 28 tests and monitoring requirements for production readiness; use it as a checklist for what is missing.

Debugging methods for the non-ML parts are in [errors, logging and debugging](/docs/theory/seml/errors-logging-debugging).

## Where this stands in 2026

:::info Industry view

- **Published incident reports now exist for LLM services.** The Anthropic report above and OpenAI's on sycophancy both describe regressions that users noticed before the vendors' own checks did, and both end in better evaluation.
- **Continuous evaluation on production traffic is the stated fix.** Anthropic's report lists running evaluations continuously on true production systems; a graded-sample monitor is the same idea at small scale.
- **The SRE machinery carries over unchanged.** Error budgets, burn-rate alerts, blameless postmortems and on-call limits come from the SRE books; what ML adds is the quality SLI and the question of how to measure it cheaply.
- **Silent regressions from the provider are part of the risk of buying.** See [build vs buy](/docs/senior/build-vs-buy-and-model-selection) for the retirement clock and the exit plan.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> A 99.9% SLO is measured over 30 days. How many minutes of total outage does it allow?</summary>

0.1% of 43,200 minutes, which is 43.2 minutes. At 99.99% it is 4.3 minutes.

</details>

<details>
<summary><strong>Q2.</strong> Why does a 3% quality regression for 20 hours trip no burn-rate alert and still hurt?</summary>

Against a 99% quality SLO it burns at 3 times, below the 6x and 14.4x thresholds, and over the 72-hour window it averages 0.83, below the 1x ticket rule. But it consumes 3% times 20 hours over 720 hours, divided by the 1% budget, which is 8.3% of the budget, more than either the 8% regression for 6 hours (6.7%) or the half-hour outage (6.9%). Only a graded sample would show it.

</details>

<details>
<summary><strong>Q3.</strong> You grade 50 responses a day. How long before you reliably see a drop from 0.95 to 0.90?</summary>

Not on day one: 23.2% chance. After three days 64.2%, after a week 94.0%. To detect it reliably within a day or two, grade more (100 a day gives 41.8% on day one, 89.9% by day three) or accept a slower alarm and rely on user reports as a backstop.

</details>

<details>
<summary><strong>Q4.</strong> Your quality monitor pages four times a week and everything it finds is noise. What do you change?</summary>

Its threshold or its sample. A one-sided alert at 2.326 standard errors has about a 1% false-alarm rate per check; more checks per day multiply that. Reduce the check frequency, raise the sample so the standard error shrinks, or use a window of several days. Treat the false-alarm rate as a design parameter with a budget, not as an afterthought, because a muted monitor protects nobody.

</details>

<details>
<summary><strong>Q5.</strong> What makes a postmortem blameless, and why does it matter?</summary>

It assumes everyone involved had good intentions and acted reasonably with the information they had. That keeps people reporting problems and focuses the actions on systems and processes, which can be fixed, rather than people, who cannot be "fixed". A postmortem that names a culprit teaches everyone to hide the next incident.

</details>

<details>
<summary><strong>Q6.</strong> From the Anthropic report, why did detection take so long, and what does the report say it changed?</summary>

The evaluations did not capture the degradation users reported, each bug had different symptoms on different platforms at different rates, and privacy controls limited access to the interactions needed to debug. It lists more sensitive evaluations, running evaluations continuously on true production systems, and tooling to debug community feedback without giving up privacy.

</details>

## Further reading

- [Google SRE book, "Embracing Risk"](https://sre.google/sre-book/embracing-risk/): error budgets and the release control loop.
- [Google SRE book, "Service Level Objectives"](https://sre.google/sre-book/service-level-objectives/): SLI, SLO, SLA and how to choose them.
- [Google SRE Workbook, "Alerting on SLOs"](https://sre.google/workbook/alerting-on-slos/): burn rate and the multi-window alert table.
- [Google SRE book, "Being On-Call"](https://sre.google/sre-book/being-on-call/): workload limits, team sizing and measurable overload.
- [Google SRE book, "Postmortem Culture: Learning from Failure"](https://sre.google/sre-book/postmortem-culture/): triggers, blamelessness and review.
- [Anthropic, "A postmortem of three recent issues"](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues): the three infrastructure bugs and the changes that followed.
- [TechCrunch, "OpenAI explains why ChatGPT became too sycophantic" (2025-04-29)](https://techcrunch.com/2025/04/29/openai-explains-why-chatgpt-became-too-sycophantic): the secondary report I read; OpenAI's own post blocked my fetch.
- [Breck et al., "The ML Test Score" (2017)](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/): 28 tests and monitoring requirements.
- [Sculley et al., "Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html): why ML systems carry ongoing maintenance costs.

## Check yourself

- I can explain why an ML service needs a quality SLI as well as latency and errors.
- I can compute an error budget in minutes and the budget a given event spends.
- I can read a multi-window burn-rate table and say what a slow regression slips past.
- I can size a graded-sample monitor for a time to detect and a false-alarm rate.
- I can run the first fifteen minutes of an ML alert: confirm, scope, what changed, mitigate, verify.
- I can write a blameless postmortem for a model regression with owner, date and type on every action.
- I can say why detection time dominates the budget and how to shorten it.
