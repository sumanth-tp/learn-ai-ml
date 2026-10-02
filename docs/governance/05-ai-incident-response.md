---
id: gov-incidents
title: "AI Incident Response"
sidebar_label: "AI incident response"
sidebar_position: 5
slug: /governance/ai-incident-response
description: "Detect, triage, contain, communicate and learn when an AI system misbehaves: a severity rubric, detector trade-offs, kill switches and rollbacks, and a timeline analyser, grounded in three published incident write-ups."
tags: [incident-response, severity, postmortem, kill-switch, rollback, monitoring, ai-incident-database]
---

import Infographic from '@site/src/components/Infographic';
import IncidentTimelineLab from '@site/src/components/viz/IncidentTimelineLab';

**In one line.** An AI incident is run like any production incident, with one difference that matters: the failure is often a plausible-looking answer rather than an error, so detection, severity and containment need AI-specific design before the first one happens.

:::note Not from a lecture
Written for this site from the sources under Further reading. The severity rubric and every simulated number are teaching constructs made for this chapter; the three incidents described are taken from their own published write-ups.
:::

## The idea in plain words

A web service that breaks returns errors, and dashboards turn red. An AI system that breaks usually keeps returning HTTP 200. The chatbot quotes a refund rule that does not exist, the retrieval layer shows one user another's document, the model starts to say things the operator forbade. Nothing crashes. The first signal is often a customer, a journalist or a regulator.

The AI Incident Database, run by the Responsible AI Collaborative, exists to collect such cases: it describes itself as documenting harms and near-harms caused by deployed AI, in the way aviation and cybersecurity keep incident records, so that society can learn from failures. Incident response is how an organisation does that learning internally.

Five steps, in order, with overlap.

1. **Detect.** Monitors on harmful-output rates, user reports, red-team findings, log anomalies.
2. **Triage.** Score the severity, name an incident commander, open one channel.
3. **Contain.** Stop the harm before understanding it: a kill switch, a rollback to the last known-good prompt or model, a narrowed tool scope.
4. **Communicate.** Tell affected users, the regulator where required, and the rest of the company, in parallel with the fix.
5. **Learn.** Find the root cause, hold a blameless review, add a regression test and an owner for each action.

NIST's incident-response guidance (SP 800-61 Revision 3, April 2025, which supersedes Revision 2 of 2012) reframes the same cycle as a community profile of the Cybersecurity Framework 2.0, aimed at helping organisations prepare, reduce the number and impact of incidents, and improve detection, response and recovery. It is written for cybersecurity incidents, and an AI incident is frequently one: a leak, a poisoned input, an abused tool.

<Infographic src="/img/gov/ai-incident-response-lifecycle.svg" alt="Five incident response steps, a severity rubric with its scale, scored examples, and three published incident summaries." caption="The five steps, the chapter's severity rubric with scored examples, and the facts of three published incidents." />

## How it works

### Severity: decide how loud to be

A severity matrix turns a messy situation into a response. The chapter's rubric scores five dimensions and adds them: **scope** (one, some or many users), **data** (none, personal, payment or health), **harm** (none, embarrassment, financial, safety or legal), **reversibility** and **public exposure**. The thresholds map the total to SEV1 to SEV4, each with a stated response. It is a teaching rubric. Yours will weigh what your business fears, and should be agreed in advance so nobody argues about severity at 3 a.m.

The regulatory dimension is part of severity. For high-risk systems under the EU AI Act, the provider must report a **serious incident** to the market surveillance authority: immediately once a causal link to the system, or its reasonable likelihood, is established, and in any event within 15 days of becoming aware; within 10 days where a person has died; and within two days for a widespread infringement or a serious and irreversible disruption of critical infrastructure (Article 73, the obligations applying from 2 December 2027 for Annex III systems, as the [regulation chapter](/docs/governance/regulation-and-model-documentation) explains). The Act defines a serious incident as one that directly or indirectly leads to death or serious harm to health, a serious and irreversible disruption of critical infrastructure, an infringement of obligations under Union law protecting fundamental rights, or serious harm to property or the environment. Article 73 also bars the provider from altering the system in a way that could affect later analysis of the causes before informing the authorities. Preserve evidence when you contain.

### Detect: the trade between delay and false alarms

Every alarm threshold trades speed for noise. Block 2 below measures it on a simulated monitor.

### Contain: the control you must build before you need it

Containment tools, from fastest to slowest: a **kill switch** (a flag that turns the feature off or routes to a safe fallback), a **rollback** to the previous prompt, model or index version, a **scope cut** (remove a tool, tighten the allow-list, switch off retrieval from one source), and a **hotfix**. A kill switch needs a human-readable owner, a tested path and a fallback experience, or switching it off is itself an outage. See [AgentOps](/docs/projects/ai-security/agentops) and [operational evals](/docs/llm-evals/operational-evals) for the monitoring side.

### Communicate and learn

Contain first, explain second, but do not wait for the root cause before telling affected people. Write the review blamelessly, so people report near-misses, and end it with actions that have owners and dates, including a regression test that would have caught the change. The AI-specific addition is to feed the incident into the evaluation suite as a permanent test, the same move as keeping a red-team attack in the harness ([red-teaming](/docs/governance/red-teaming-llm-systems)).

## A real system that works this way

Three published write-ups, each opened for this chapter.

**OpenAI, ChatGPT, March 2023.** In a post dated 24 March 2023, OpenAI wrote that it took ChatGPT offline after a bug in an open-source library, the Redis client redis-py, allowed some users to see titles from another active user's chat history. A change it introduced at 1 a.m. Pacific time on Monday, 20 March caused a spike in Redis request cancellations; if a request was cancelled at the wrong moment the shared connection became corrupted and the next, unrelated request could receive data left behind. Further investigation showed that payment-related information of 1.2% of ChatGPT Plus subscribers active during a specific nine-hour window may have been visible, including name, email address, payment address, card type, last four digits and expiry date; full card numbers were not exposed. Its listed actions include adding redundant checks that cache data matches the requesting user, examining logs programmatically, correlating data sources to identify and notify affected users, improving logging, and making the Redis cluster more robust. The lesson for AI engineers: the incident was in ordinary infrastructure around the model, which is where much AI-system harm comes from.

**Air Canada, chatbot, decided 14 February 2024.** In Moffatt v. Air Canada (2024 BCCRT 149, British Columbia Civil Resolution Tribunal), a customer relied on the airline's chatbot, which suggested he could apply for bereavement fares retroactively; the airline's policy did not permit that. The airline argued, in effect, that the chatbot was a separate legal entity responsible for its own actions, which the tribunal member called "a remarkable submission", finding that the chatbot is part of the airline's website and that the airline did not take reasonable care to ensure it was accurate. The airline was ordered to pay \$812.02 in total, made up of \$650.88 in damages, \$36.14 in pre-judgment interest and \$125 in tribunal fees. The lesson: you are responsible for what your assistant says, which makes answer accuracy a legal and a severity question.

**Microsoft, Tay, March 2016.** Microsoft's own post-mortem said a coordinated attack by a subset of people exploited a vulnerability in Tay, and that although the team had prepared for many types of abuse, it had made a critical oversight for this specific attack. Containment was to take the bot offline. The lesson: abuse is social as well as technical, and the review should change the test suite.

The first example is a privacy incident in a product that happens to be an AI service, the second is a wrong answer with legal consequences, the third is adversarial abuse. They sit at different places on the rubric: 15, 9 and 9 below.

## Code you can run

Everything is CPU only, seeded and fast. The incident facts in block 1 come from the write-ups above; the scoring rubric and the simulations are teaching constructs.

#### 1. A severity scorer

The rubric is a small table of weights. The three published incidents and two invented small ones are scored by it.

```python
SCOPE = {"one user": 1, "some users": 3, "many users": 5}
DATA = {"none": 0, "personal": 3, "payment or health": 5}
HARM = {"none": 0, "embarrassment": 1, "financial": 3, "safety or legal": 5}
REVERSIBLE = {"fully": 0, "with effort": 2, "no": 4}
PUBLIC = {"internal only": 0, "customers": 1, "press or regulator": 3}

LEVELS = [(14, "SEV1"), (9, "SEV2"), (5, "SEV3"), (0, "SEV4")]
RESPONSE = {
    "SEV1": "page the on-call lead and the incident commander now; contain first, explain later; legal and communications join within the hour",
    "SEV2": "page on-call; contain within hours; notify the security and privacy owners the same day",
    "SEV3": "ticket with an owner and a due date; fix in the normal release train, with a regression test",
    "SEV4": "log it; review in the weekly quality meeting",
}


def score(incident):
    parts = {
        "scope": SCOPE[incident["scope"]],
        "data": DATA[incident["data"]],
        "harm": HARM[incident["harm"]],
        "reversible": REVERSIBLE[incident["reversible"]],
        "public": PUBLIC[incident["public"]],
    }
    total = sum(parts.values())
    level = next(name for floor, name in LEVELS if total >= floor)
    return total, level, parts


CASES = {
    "chat titles and payment details shown to other users (OpenAI, March 2023)":
        dict(scope="some users", data="payment or health", harm="none", reversible="no", public="press or regulator"),
    "chatbot invents a refund rule, one customer relies on it (Air Canada, 2024)":
        dict(scope="one user", data="none", harm="financial", reversible="with effort", public="press or regulator"),
    "chatbot taught to post abuse by a coordinated attack (Tay, 2016)":
        dict(scope="many users", data="none", harm="embarrassment", reversible="fully", public="press or regulator"),
    "typo in a canned answer, caught in review":
        dict(scope="some users", data="none", harm="none", reversible="fully", public="internal only"),
    "retrieval index leaks an internal document to one employee":
        dict(scope="one user", data="personal", harm="none", reversible="with effort", public="internal only"),
}

print("incident                                                                      score  level")
for name, incident in CASES.items():
    total, level, parts = score(incident)
    print(f"{name:76s} {total:3d}   {level}")
print()
for name, incident in list(CASES.items())[:2]:
    total, level, parts = score(incident)
    print(name)
    print("  parts:", ", ".join(f"{k} {v}" for k, v in parts.items()))
    print("  response:", RESPONSE[level])

print("\nthresholds: " + ", ".join(f"{name} from {floor}" for floor, name in LEVELS))
worst = dict(scope="many users", data="payment or health", harm="safety or legal", reversible="no", public="press or regulator")
print("highest possible score:", score(worst)[0])
```

```text
incident                                                                      score  level
chat titles and payment details shown to other users (OpenAI, March 2023)     15   SEV1
chatbot invents a refund rule, one customer relies on it (Air Canada, 2024)    9   SEV2
chatbot taught to post abuse by a coordinated attack (Tay, 2016)               9   SEV2
typo in a canned answer, caught in review                                      3   SEV4
retrieval index leaks an internal document to one employee                     6   SEV3

chat titles and payment details shown to other users (OpenAI, March 2023)
  parts: scope 3, data 5, harm 0, reversible 4, public 3
  response: page the on-call lead and the incident commander now; contain first, explain later; legal and communications join within the hour
chatbot invents a refund rule, one customer relies on it (Air Canada, 2024)
  parts: scope 1, data 0, harm 3, reversible 2, public 3
  response: page on-call; contain within hours; notify the security and privacy owners the same day

thresholds: SEV1 from 14, SEV2 from 9, SEV3 from 5, SEV4 from 0
highest possible score: 22
```

OpenAI's incident scores 15 and lands at SEV1 (some users, payment details, not reversible, public). The airline case scores 9: only one customer was involved and no data was exposed, but there is financial harm, it was hard to reverse and it reached a tribunal. Tay also scores 9 (many users, embarrassment, public). The rubric is crude, and that is the point: it makes disagreement explicit. If you think Tay should be SEV1, change a weight and re-run, and you now have a recorded decision about what your organisation fears most.

#### 2. How fast can a monitor see a shift?

A monitor watches the fraction of responses flagged as harmful. It runs at 0.5% and then shifts to 4.0% after request 3,000. Two detectors are compared: a rolling mean over the last 200 requests with a threshold, and a likelihood-ratio CUSUM, a sequential test that accumulates evidence request by request.

```python
import numpy as np

BASE, SHIFTED = 0.005, 0.04
WINDOW = 200
rng_master = np.random.default_rng(21)


def stream(rng, change_at, length):
    p = np.where(np.arange(length) < change_at, BASE, SHIFTED)
    return (rng.random(length) < p).astype(int)


def rolling_alarm(x, threshold, start=0):
    for t in range(max(WINDOW, start), len(x)):
        if x[t - WINDOW + 1:t + 1].mean() >= threshold:
            return t
    return None


HIT = np.log(SHIFTED / BASE)
MISS = np.log((1 - SHIFTED) / (1 - BASE))


def cusum_alarm(x, limit, start=0):
    s = 0.0
    for t in range(start, len(x)):
        s = max(0.0, s + (HIT if x[t] else MISS))
        if s >= limit:
            return t
    return None


def evaluate(detector, parameter, runs=300, change_at=3000, length=6000):
    rng = np.random.default_rng(21)
    delays, false_alarms = [], 0
    for _ in range(runs):
        x = stream(rng, change_at, length)
        alarm = detector(x, parameter)
        if alarm is None:
            delays.append(length - change_at)
        elif alarm < change_at:
            false_alarms += 1
        else:
            delays.append(alarm - change_at)
    return false_alarms / runs, float(np.mean(delays)), float(np.percentile(delays, 95))


print(f"harmful-output rate shifts from {BASE:.1%} to {SHIFTED:.1%} after request 3000; 6000 requests per run, 300 runs")
print(f"{'detector':36s}{'false alarms before shift':>26s}{'mean delay':>12s}{'95th pct':>10s}")
for threshold in (0.02, 0.025, 0.03):
    fa, mean, p95 = evaluate(rolling_alarm, threshold)
    print(f"{f'rolling mean of 200 >= {threshold}':36s}{fa:26.3f}{mean:12.1f}{p95:10.1f}")
for limit in (5.0, 7.0, 9.0):
    fa, mean, p95 = evaluate(cusum_alarm, limit)
    print(f"{f'likelihood-ratio CUSUM, limit {limit:.0f}':36s}{fa:26.3f}{mean:12.1f}{p95:10.1f}")

rate_per_minute = 50
print(f"\nat {rate_per_minute} requests per minute, a delay of 100 requests is {100 / rate_per_minute:.0f} minutes of exposure")
```

```text
harmful-output rate shifts from 0.5% to 4.0% after request 3000; 6000 requests per run, 300 runs
detector                             false alarms before shift  mean delay  95th pct
rolling mean of 200 >= 0.02                              0.477        81.9     194.2
rolling mean of 200 >= 0.025                             0.143       110.6     223.4
rolling mean of 200 >= 0.03                              0.020       143.5     276.1
likelihood-ratio CUSUM, limit 5                          0.213        95.6     232.2
likelihood-ratio CUSUM, limit 7                          0.023       131.6     296.2
likelihood-ratio CUSUM, limit 9                          0.010       173.7     363.2

at 50 requests per minute, a delay of 100 requests is 2 minutes of exposure
```

The first three rows show the dial. A threshold of 0.02 detects the shift in 81.9 requests on average but raises a false alarm before the shift in 47.7% of runs; 0.03 cuts false alarms to 2.0% and takes 143.5 requests. The CUSUM rows show the same trade-off. At a similar false-alarm rate (0.023 against 0.020) the CUSUM has a slightly lower mean delay (131.6 against 143.5 requests) and a longer tail (95th percentile 296.2 against 276.1), so on this simulation neither wins outright. The practical rule: choose the false-alarm rate you can staff, then take the fastest detector at that rate, and watch the tail as well as the mean. At 50 requests per minute, 100 requests of delay is 2 minutes of exposure; slow traffic makes the same delay hours long.

#### 3. A timeline analyser and a kill-switch counterfactual

A worked incident gives eight timestamped events. The analyser computes the time in each phase, the size of the harm window, and what a different containment method would have changed. Then 40 seeded incidents show where time goes in aggregate. The containment durations (kill switch 2 minutes, rollback 15, hotfix 120) are **parameters of the model**, not measurements.

```python
from datetime import datetime, timedelta

import numpy as np

PHASES = ["detect", "acknowledge", "decide", "contain"]
CONTAIN_MINUTES = {"kill switch": 2, "rollback": 15, "hotfix": 120}
HARMFUL_PER_MINUTE = 12


def minutes(a, b):
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60


worked = [
    ("2026-10-02T09:00", "harm_starts", "prompt change reaches 100% of traffic"),
    ("2026-10-02T09:47", "detected", "support ticket: assistant quotes a refund rule that does not exist"),
    ("2026-10-02T09:52", "acknowledged", "on-call engineer opens the incident"),
    ("2026-10-02T10:10", "decided", "incident commander chooses rollback over hotfix"),
    ("2026-10-02T10:25", "contained", "previous prompt version restored"),
    ("2026-10-02T11:00", "communicated", "customers who received the rule are emailed a correction"),
    ("2026-10-03T16:00", "root_cause", "change shipped without the policy regression test"),
    ("2026-10-06T10:00", "postmortem", "blameless review published, three actions with owners"),
]
at = {name: stamp for stamp, name, _ in worked}
gaps = {
    "detect": minutes(at["harm_starts"], at["detected"]),
    "acknowledge": minutes(at["detected"], at["acknowledged"]),
    "decide": minutes(at["acknowledged"], at["decided"]),
    "contain": minutes(at["decided"], at["contained"]),
}
exposure_minutes = sum(gaps.values())
print("worked incident, minutes in each phase:", {k: int(v) for k, v in gaps.items()})
print(f"harm window {exposure_minutes:.0f} minutes, about {HARMFUL_PER_MINUTE * exposure_minutes:.0f} harmful responses at {HARMFUL_PER_MINUTE} per minute")
print(f"communication started {minutes(at['contained'], at['communicated']):.0f} minutes after containment; "
      f"root cause {minutes(at['contained'], at['root_cause']) / 60:.1f} hours after; postmortem on day {(datetime.fromisoformat(at['postmortem']) - datetime.fromisoformat(at['harm_starts'])).days}")

print("\nwhat a kill switch would have changed (same detection, same decision time)")
for method, contain in CONTAIN_MINUTES.items():
    window = gaps["detect"] + gaps["acknowledge"] + gaps["decide"] + contain
    print(f"  contain by {method:11s} {contain:4d} min   harm window {window:5.0f} min   harmful responses {HARMFUL_PER_MINUTE * window:6.0f}")

rng = np.random.default_rng(4)
records = []
for _ in range(40):
    detect = rng.lognormal(np.log(30), 0.9)
    ack = rng.lognormal(np.log(6), 0.6)
    decide = rng.lognormal(np.log(15), 0.7)
    method = rng.choice(list(CONTAIN_MINUTES), p=[0.2, 0.5, 0.3])
    contain = CONTAIN_MINUTES[method] * rng.lognormal(0, 0.4)
    records.append((detect, ack, decide, contain, method))
arr = np.array([r[:4] for r in records])
print("\n40 seeded incidents: minutes per phase")
print("phase         median   90th percentile   share of the mean harm window")
shares = arr.mean(axis=0) / arr.mean(axis=0).sum()
for i, phase in enumerate(PHASES):
    print(f"{phase:12s} {np.median(arr[:, i]):7.1f} {np.percentile(arr[:, i], 90):17.1f} {shares[i]:20.1%}")
total = arr.sum(axis=1)
print(f"\ntotal harm window: median {np.median(total):.0f} min, 90th percentile {np.percentile(total, 90):.0f} min")
for method in CONTAIN_MINUTES:
    sel = [t for t, r in zip(total, records) if r[4] == method]
    print(f"  contained by {method:11s} {len(sel):2d} incidents, median window {np.median(sel):5.0f} min")
```

```text
worked incident, minutes in each phase: {'detect': 47, 'acknowledge': 5, 'decide': 18, 'contain': 15}
harm window 85 minutes, about 1020 harmful responses at 12 per minute
communication started 35 minutes after containment; root cause 29.6 hours after; postmortem on day 4

what a kill switch would have changed (same detection, same decision time)
  contain by kill switch    2 min   harm window    72 min   harmful responses    864
  contain by rollback      15 min   harm window    85 min   harmful responses   1020
  contain by hotfix       120 min   harm window   190 min   harmful responses   2280

40 seeded incidents: minutes per phase
phase         median   90th percentile   share of the mean harm window
detect          29.3             138.6                41.0%
acknowledge      6.0              13.4                 5.3%
decide          16.7              39.7                15.2%
contain         17.3             168.2                38.5%

total harm window: median 111 min, 90th percentile 242 min
  contained by kill switch 10 incidents, median window    75 min
  contained by rollback    17 incidents, median window    84 min
  contained by hotfix      13 incidents, median window   183 min
```

In the worked incident the harm starts at 09:00 and is detected 47 minutes later by a support ticket, acknowledged after 5, decided after 18 and contained by rollback after 15 more: an 85-minute window and about 1,020 harmful responses at 12 per minute. A kill switch would have cut the window to 72 minutes (864 responses), saving 13 minutes; a hotfix instead would have meant 190 minutes and 2,280 responses. Across the 40 seeded incidents, detection accounts for 41.0% of the mean harm window and containment for 38.5%, and the 90th-percentile total is 242 minutes against a median of 111. Faster detection pays as much as faster switching, and the containment tail is long when the method is a hotfix. Communication started 35 minutes after containment in the worked incident; do it in parallel next time.

The lab lets you move each phase and change the containment method. Its defaults are the worked incident: 47, 5, 18 and 15 minutes, an 85-minute window and 1,020 harmful responses.

<IncidentTimelineLab />

<Infographic src="/img/gov/ai-incident-response-minutes.svg" alt="A phase bar for the worked incident, a containment comparison, aggregate phase statistics for 40 seeded incidents and a detector comparison table." caption="Blocks 2 and 3: where the minutes go, and what each detector setting costs." />

## Designing with it

- **Write the severity matrix before launch**, with named owners for each level and a response that fits it.
- **Build the kill switch and test it.** Include the fallback experience and the person allowed to pull it without asking.
- **Instrument for detection.** Log a harmful-output flag per response, alarm on rate shifts, and give users a one-click report. Choose the false-alarm rate your on-call can bear.
- **Keep evidence.** Save prompts, retrieved passages, tool calls and versions for each request, long enough to investigate; the Act's own logging periods for high-risk systems are a floor, not a target.
- **Plan communication in advance**: who tells customers, who tells the regulator, with what template, within what deadline.
- **Close the loop.** Every incident becomes a test in the evaluation suite and a line in the risk register. Review near-misses as well as outages.
- **Track the database.** Searching the AI Incident Database for systems like yours gives you free scenarios for your next tabletop exercise.

## Where this stands in 2026

:::info Industry view

- **Serious-incident reporting is written into EU law, with a date.** Article 73 applies together with the other high-risk obligations, from 2 December 2027 for Annex III systems after the July 2026 Omnibus, and general-purpose model providers with systemic risk must already track, document and report serious incidents under Article 55.
- **Incident records are becoming shared infrastructure.** The AI Incident Database is run by the Responsible AI Collaborative and is open to incident submissions from the public; I did not verify its licence terms or its current count.
- **Incident response guidance is current.** NIST's SP 800-61 Revision 3 dates from April 2025 and ties incident response to CSF 2.0.
- **The pattern across public cases is old-fashioned.** Two of the three incidents above were failures of ordinary engineering and of oversight (a library bug; an unverified answer) rather than exotic attacks.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why does the airline case score only SEV2 on the rubric although it reached a tribunal?</summary>

Scope was one customer and no data was exposed, so the total is 9: financial harm 3, reversible with effort 2, public 3, scope 1. The rubric reflects its weights, and a company that fears legal precedent more than loss would raise the weight of the public dimension and change the result.

</details>

<details>
<summary><strong>Q2.</strong> In the worked incident, a kill switch saved only 13 minutes. What would have saved more?</summary>

Faster detection. Detecting took 47 of the 85 minutes, and across the 40 seeded incidents detection is 41.0% of the mean window against containment's 38.5%. Better monitors, a user report button or a canary rollout shorten the biggest phase.

</details>

<details>
<summary><strong>Q3.</strong> A threshold of 0.02 detects the shift fastest. Why not use it?</summary>

It raises a false alarm before the shift in 47.7% of runs, so on-call would be woken for noise and learn to ignore it. Pick the false-alarm rate you can sustain and then the fastest detector at that rate.

</details>

<details>
<summary><strong>Q4.</strong> A provider of a high-risk system in the EU learns a widespread infringement has occurred. How long does it have to report?</summary>

Two days after becoming aware under Article 73(3), immediately where possible. The general limit is 15 days and the limit for a death is 10 days. These obligations apply from 2 December 2027 for Annex III systems. Preserve the system's state before altering it.

</details>

<details>
<summary><strong>Q5.</strong> Why should communication start before the root cause is known?</summary>

People affected need to act: change a password, check a statement, ignore advice. In the worked incident communication began 35 minutes after containment while the root cause took about 30 hours. Say what is known, what is not, and when you will update.

</details>

<details>
<summary><strong>Q6.</strong> What is the AI-specific final step of the review?</summary>

Convert the incident into a permanent test case in the evaluation suite and the red-team harness, so the same failure is caught before release next time.

</details>

## Further reading

All opened for this chapter in October 2026.

- OpenAI, [March 20 ChatGPT outage: here's what happened](https://openai.com/index/march-20-chatgpt-outage/), 24 March 2023.
- Civil Resolution Tribunal of British Columbia, [Moffatt v. Air Canada, 2024 BCCRT 149](https://decisions.civilresolutionbc.ca/crt/crtd/en/525448/1/document.do), 14 February 2024.
- Microsoft, [Learning from Tay's introduction](https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/), 25 March 2016.
- [AI Incident Database](https://incidentdatabase.ai/), Responsible AI Collaborative.
- NIST, [SP 800-61 Revision 3, Incident response recommendations and considerations for cybersecurity risk management](https://csrc.nist.gov/pubs/sp/800/61/r3/final), April 2025.
- [Regulation (EU) 2024/1689](http://data.europa.eu/eli/reg/2024/1689/oj), Articles 3(49), 55 and 73, as amended by [Regulation (EU) 2026/1744](http://data.europa.eu/eli/reg/2026/1744/oj).

## Check yourself

- I can name the five steps of AI incident response and say which to do first when harm is ongoing.
- I can build a severity rubric, apply it to a real write-up and defend the weights.
- I can explain why an alarm threshold is a trade between delay and false alarms, and read both numbers.
- I can say what a kill switch needs to be useful and when detection matters more than containment.
- I can state the EU reporting deadlines for serious incidents with high-risk systems and the date they apply.
- I can turn an incident into a regression test.
