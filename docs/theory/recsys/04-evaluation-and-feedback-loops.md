---
title: "Recommenders · Evaluation and feedback loops"
sidebar_label: "Evaluation and feedback"
sidebar_position: 4
slug: /theory/recsys/evaluation-and-feedback-loops
description: "Offline ranking metrics, exposure bias, online experiments, diversity and operations."
tags: [recommender-systems, evaluation, experimentation]
---

import Infographic from '@site/src/components/Infographic';
import SlateDiversityLab from '@site/src/components/viz/SlateDiversityLab';

**In one line.** Evaluate a recommender as a displayed list that changes what users see, what they do, and what evidence the next model can learn.

## The idea in plain words

A recommender's prediction is a ranked slate, not an isolated user-item score. The first position gets more attention; a repeated topic can make a slate feel narrow; a high-scoring but unavailable item cannot be used. Offline ranking metrics help compare candidate models, but the logged interactions were produced by an earlier recommendation policy. Online experiments test effects on users under controlled exposure. Production monitoring catches freshness, inventory and feedback-loop problems after release. All three views matter.

The feedback loop is fundamental. A model shows items; users can only interact with items they encounter; those interactions become future training data. A system that repeatedly shows popular items gathers more evidence for them and less evidence for new or niche items. Observed engagement can therefore rise while catalogue coverage and discovery shrink. To understand an outcome, keep the displayed slate, positions, eligibility snapshot and policy version together with the later event.

<Infographic src="/img/recsys/evaluation-loops.svg" alt="Offline replay, online experiments and production feedback monitoring each assess recommendation quality; a score-first A,B slate shares one topic while a 0.2 topic bonus selects A,C." caption="List quality includes relevance, opportunity, diversity and the behaviour created by display." />

## How it works

### Build an honest offline split

Train only on interactions that happened before the evaluation period. Hold out later events and reconstruct which items were eligible at each request. Remove items already consumed where the product would do so. If the model uses user history up to a request, do not let it see the held-out event or a later event from the same user. Evaluate new users and new items separately because random interaction splits predominantly measure existing entities. A candidate source should be assessed for recall before a ranker is credited or blamed for a final result.

Even a chronological split cannot reveal all preferences. A held-out click is a positive signal for an item the old policy showed, but the unshown catalogue lacks labels. An offline metric computed over “all unclicked items are negative” can reward imitation of the old policy. Evaluate on exposed alternatives where possible, document candidate and exposure restrictions, and use online randomisation for claims about user impact. Logging the propensity of a controlled exploration policy can support more advanced counterfactual analysis, but such methods require overlap and careful variance checks. Do not infer causal improvement merely from an offline lift.

### Compute list metrics

Precision@$K$ is the fraction of displayed top-$K$ items labelled relevant; Recall@$K$ is the fraction of all labelled relevant items that appear in those $K$ positions. Precision can be high for a narrow slate while recall is poor. Candidate recall and displayed-list recall use different cutoffs and possibly different candidate universes; name the denominator. Mean reciprocal rank emphasises the first relevant item. Average precision rewards the positions of all relevant hits. Normalised discounted cumulative gain, NDCG, accepts graded relevance and discounts lower ranks. The choice should follow the product: one good answer may be enough for a help query, while a browsing surface benefits from several useful options.

For binary relevance with ranking `[A, B, C, D]`, suppose B and D are relevant. At $K=2$, precision is $1/2$ and recall is $1/2$. The first relevant item is at rank two, so reciprocal rank is $1/2$. For graded labels, DCG uses gains such as $2^{grade}-1$ divided by $\log_2(rank+1)$; NDCG divides by the ideal ordering's DCG. The exact gain convention and treatment of unjudged items must be recorded for comparisons. A metric can change solely because the candidate pool or labelling policy changed.

### Evaluate the slate as a whole

Individual item scores do not capture redundancy. Consider A with topic X and relevance 0.9, B with topic X and 0.8, and C with topic Y and 0.7. Score-first top two is A,B, with one distinct topic. A simple illustrative reranker chooses A first, then adds a 0.2 bonus to a new topic; C's adjusted second-position score becomes 0.9, above B's 0.8, yielding A,C and two topics. This rule trades some base relevance for category coverage. It is not evidence that users prefer A,C. Its effect should be assessed with both relevance and user-outcome measures.

Diversity can be measured in several ways: number of distinct topics, pairwise dissimilarity, exposure distribution across creators, or long-tail catalogue coverage. A superficial “distinct topic count” can be gamed by inconsistent metadata and may not reflect meaningful variety. Set diversity targets with the product context. A medical information tool may require a narrow set of vetted sources; an entertainment feed may benefit from exploration. Freshness likewise should not automatically trump relevance. Reranking constraints should have explicit ownership and audit logs.

### Run an online experiment

Randomly assign users or another appropriate unit to control and treatment. Keep assignment stable across repeated visits when cross-session effects matter. Define a primary outcome and guardrails before examining results, such as task completion, satisfaction, complaints, latency and inventory errors. Make sure the two variants see comparable eligible catalogues, and avoid concurrent experiments that change the same surface without accounting for interaction. A click lift can coexist with worse completion or increased returns; interpret the full set of agreed outcomes.

Measure exposure, not just outcomes. If treatment shows more items or uses a different layout, a raw click count may change for interface reasons. Track per-impression and per-user metrics as appropriate, but do not assume they answer the same question. Short tests can miss long-term fatigue or delayed conversion. A treatment that initially boosts novelty may lose value after repeated sessions, or may help users discover durable interests later. Use a rollout and monitoring plan that matches the expected time scale of benefit and harm.

### Monitor a changing system

After release, monitor request success, latency, index freshness, empty candidate lists, catalogue coverage, filtered-item rate and fallbacks. Track outcome metrics by segment when labels arrive. Compare new users, new items, sparse users and long-tail categories. A surge in clicks with falling distinct-item exposure is a feedback-loop warning. A new ranker version can also change the training distribution for the next version. Preserve policy, model and index versions in logs so a later experiment can reproduce what users saw.

Privacy and user control are part of quality. Avoid exposing sensitive inferred categories, and allow a person to indicate that a suggestion is irrelevant or unwanted where the surface permits. Use that feedback as a distinct signal rather than assuming all non-clicks mean the same thing. Retention and deletion rules should apply to impression and interaction logs as well as model features. A technically accurate ranking can still fail if it violates a user's explicit preference or a product safety rule.

## A real system that works this way

The [Google recommendation course](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) describes reranking for diversity, freshness and fairness after scoring. The [2016 YouTube recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) is a historical example of the candidate/ranker split. A course recommender built on that pattern would log eligible lessons, candidates, scores and final positions; test whether the list helps learners complete appropriate lessons; and guard against repeatedly promoting easy but off-path content. These are design implications of the published architecture, not claims about an unpublished current product.

## Code you can run

```python
ranked = [('A', False), ('B', True), ('C', False), ('D', True)]
k = 2
relevant_total = sum(relevant for _, relevant in ranked)
top = ranked[:k]
hits = sum(relevant for _, relevant in top)
precision = hits / k
recall = hits / relevant_total
first_relevant_rank = next(index for index, (_, relevant) in enumerate(ranked, start=1) if relevant)
print('precision@2:', precision)
print('recall@2:', recall)
print('reciprocal rank:', 1 / first_relevant_rank)
```

The three outputs are `0.5`. They describe a fully labelled four-item illustration, not an unbiased estimate from ordinary impression logs.

<SlateDiversityLab />

Move the new-topic bonus. At zero, the second item is B and the slate contains one topic. At the default 0.2, C's adjusted score is 0.9, so the lab selects A,C and shows two topics. The data table preserves each item's base score and adjusted second-position score.

```python
items = [('A', 'X', 0.9), ('B', 'X', 0.8), ('C', 'Y', 0.7)]
bonus = 0.2
first = items[0]
remaining = items[1:]
second = max(remaining, key=lambda item: item[2] + (bonus if item[1] != first[1] else 0))
slate = [first, second]
print('slate:', [item[0] for item in slate])
print('distinct topics:', len({item[1] for item in slate}))
```

This prints `['A', 'C']` and two topics. It only verifies the deterministic reranking arithmetic; it does not measure user satisfaction.

## Designing with it

### Interpret metrics through the logged opportunity set

Suppose a new ranker raises Recall@10 on held-out clicks. Ask which items were eligible and exposed when those clicks were recorded. If the metric treats every unclicked catalogue item as irrelevant, it may favour popular, often-exposed items and punish novel ones that never had an opportunity. A temporal split prevents future interactions from leaking into training, but it does not solve exposure bias. At minimum, report the candidate pool, displayed positions and label construction. When a controlled exploration policy provides known selection probabilities, off-policy estimators may help, but they can become unstable when some actions had tiny probability or were never taken. An online experiment remains the clearer test of an intended product change.

List metrics also need a relevance horizon. For a course, clicking a lesson today is immediate while completing it may take days. A seven-day completion label cannot be evaluated on requests from yesterday. If unfinished recent requests are treated as failures, a model that sends learners to longer, more valuable lessons may look worse. Mature the labels or model the delay explicitly. Report the number of evaluated requests and any exclusions. Likewise, a purchase can be reversed by a return, and a “no purchase” label may depend on inventory and price at the time of display.

### A diversity example with real trade-offs

The lab's A,B versus A,C example shows only category coverage. It deliberately leaves out how useful B and C are to an actual user. To evaluate a diversity bonus, record both immediate engagement and longer-term outcomes, perhaps repeat visits or exploration of new topics. Inspect whether the rule helps users whose histories are narrow, harms users with a specific task, or changes exposure for creators. A one-size-fits-all bonus can be too strong for a focused search-like surface and too weak for a discovery surface. The acceptable relevance loss should be agreed before tuning the bonus to online data.

Topic metadata may be incomplete or manipulable. If a creator can choose a rare tag solely to receive a bonus, the diversity metric can rise without actual variety. Sample final slates for human review and compare semantic or editorial categories where appropriate. Distinct-item coverage across the catalogue is another signal: a model may show many topics but only the same few blockbuster items within each. Measure per-user variety, aggregate catalogue exposure and user value separately because one number cannot represent all three.

### Design an experiment that survives feedback

Randomisation unit matters. If the same user alternates between control and treatment across visits, their history is affected by both policies, making a clean comparison harder. Stable user assignment often suits a personalised surface; other products may need household, organisation or geographic assignment when users share content and influence one another. Predefine the primary outcome, guardrails and duration, and check that instrumentation records both variants consistently. If a new interface changes the number of visible tiles, normalise outcome metrics thoughtfully and report the actual exposure difference.

Early results may be misleading because users explore a new list differently at first. Some benefits require several sessions; some harms, such as fatigue or repetitive content, accumulate. Measure the time scale of the claimed benefit. Do not infer long-term improvement from a short click-rate lift. Keep a holdback or periodic comparison where practical, and monitor after rollout because the training data for the next model version will be produced under the new policy. Version the policy and preserve experiment assignment in the data used for retraining.

### Operate the whole slate pipeline

An incident can come from any stage. An empty candidate list may be an index outage, a new user's missing features or an over-strict eligibility filter. A sudden collapse in long-tail exposure may come from an ANN index change even when the ranker is unchanged. A click spike can come from a layout change. Dashboard stage-level counts, filtered reasons, latency, index age, fallback rate and outcome metrics together. Sample request traces with consent and appropriate privacy controls. When a defect is found, the logs should let operators replay the candidate, ranker and reranker versions that produced the displayed slate.

Define an evaluation card for each recommendation surface: request context, eligible universe, exposure logging, relevance label, label delay, offline split, baseline, list metrics, online primary metric and guardrails. Keep candidate, ranker and reranker versions in a single trace. When a metric moves, inspect stage-level coverage and the actual lists rather than changing the final score blindly. Check whether a data collection or layout change caused the movement.

For an experiment, decide a minimum meaningful effect and an observation window before starting. Do not stop as soon as a noisy daily plot crosses a desired threshold. Segment analysis is useful for diagnosis, but many unplanned subgroup comparisons raise false-discovery risk. Preserve an overall primary decision rule and treat exploratory slices as hypotheses for follow-up. After rollout, maintain a holdback or periodic control where appropriate so the feedback loop does not erase the comparison.

## Where this stands in 2026

As recommendation models and indexes become more capable, the difficult question remains whether a displayed list helps users and supports a healthy catalogue over time. Offline metrics are fast iteration tools, controlled online tests support causal product decisions, and monitoring catches drift and feedback loops. Diversity and fairness are not single universal scores; they require a documented product goal and relevant slices. The model, interface and data-collection policy together determine observed outcomes.

## Practice questions

<details>
<summary>Why can a model improve offline recall while having no online benefit?</summary>

Offline labels reflect the prior exposure policy and may not represent unseen items or the intended user outcome. The displayed slate, latency or eligibility can also differ. Test the complete product effect online.

</details>

<details>
<summary>What are precision@2 and recall@2 for A,B when B and D are relevant?</summary>

One of two displayed items is relevant and one of two relevant catalogue items is displayed, so both are 0.5.

</details>

<details>
<summary>What does the diversity lab demonstrate and what does it not?</summary>

It demonstrates how a new-topic bonus changes a two-item slate from one topic to two. It does not establish that users prefer the reranked slate.

</details>

<details>
<summary>How can a recommender create a popularity feedback loop?</summary>

It shows already-popular items more often, creating more interactions for them, then trains on those interactions. Less-shown items remain data-poor regardless of their potential value.

</details>

## Further reading

- [Google recommendation overview](https://developers.google.com/machine-learning/recommendation/overview/types) connects the stages of a full system.
- [Google scoring guidance](https://developers.google.com/machine-learning/recommendation/dnn/scoring) discusses common scoring of mixed candidates.
- [Google reranking guidance](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking) treats diversity, freshness and fairness.
- [Historical two-stage recommendation paper](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) documents a published large-scale design.

## Check yourself

- I can calculate precision, recall and reciprocal rank on a labelled slate.
- I can distinguish candidate recall from displayed-list quality.
- I can explain why observed clicks depend on exposure and position.
- I can design an online experiment with a primary outcome and guardrails.
- I can monitor catalogue coverage, fresh items and fallback use after release.
