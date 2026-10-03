---
title: "Recommenders · Feedback and objectives"
sidebar_label: "Feedback and objectives"
sidebar_position: 1
slug: /theory/recsys/feedback-and-objectives
description: "Explicit and implicit feedback, exposure, cold start and the objective behind a recommendation."
tags: [recommender-systems, implicit-feedback, product]
---

import Infographic from '@site/src/components/Infographic';
import FeedbackConfidenceLab from '@site/src/components/viz/FeedbackConfidenceLab';

**In one line.** A recommendation system ranks eligible items for a particular context from incomplete, biased evidence about what a user may value.

## The idea in plain words

A search system often begins with a query typed by a user. A recommender frequently starts with a visit, a current item, a history or a task: “What should this person see next?” The candidate catalogue may contain courses, films, products, jobs or documents. A prediction of preference is only one input. The system must also respect availability, age suitability, prerequisites, duplicates, freshness and the size of the screen. A high-scoring item that cannot be delivered is not a useful recommendation.

Feedback comes in two broad forms. **Explicit feedback** is a stated preference, such as a rating, like or “not interested” control. It has a declared meaning, but only some users supply it and the scale may be used differently by different people. **Implicit feedback** is observed behaviour, such as an impression, click, watch, completion or purchase. It is plentiful but ambiguous. A click can express curiosity, accidental selection or attraction to a title; a long dwell time can mean interest or confusion. An unclicked item may never have been shown. The feedback label is not the user's true preference.

<Infographic src="/img/recsys/feedback-objective.svg" alt="A board separates explicit ratings, implicit interactions and new-user or new-item cold start; two interactions with confidence weight two produce confidence five." caption="Observed events need an interpretation, an exposure record and a product objective." />

## How it works

### State the product objective

Before choosing an algorithm, decide what the list should help a person accomplish. A course site may favour sustained learning and prerequisite fit over a short-lived click. A shopping service may care about useful purchases and returns, not clicks alone. A video product may optimise time spent, but should inspect satisfaction, fatigue and diversity. A workplace knowledge tool may prioritise task completion and trustworthy sources. Different outcomes can disagree, so record a primary objective and guardrails rather than hiding a policy inside a training label.

The display surface changes the task. A “related item” carousel has the current item as a strong query; a homepage has a broad user context; a new-user page may have no personal history. Candidate generation for each surface can differ. The same trained user factor may be relevant in one context and irrelevant in another. Decide whether to recommend from all items, only currently eligible items, or a smaller editorial collection. Ineligible items should be filtered at a defined stage, and the evaluation denominator should reflect the actual opportunity set.

### Read implicit evidence carefully

An interaction matrix $r_{ui}$ may contain counts for user $u$ and item $i$. One simple implicit-feedback construction sets preference $p_{ui}=1$ when $r_{ui}>0$ and $0$ otherwise, with confidence $c_{ui}=1+\alpha r_{ui}$. For count two and $\alpha=2$, preference is one and confidence is five. This is the lab's default and a simplified form of the confidence idea in [Hu, Koren and Volinsky's original work](https://yifanhu.net/PUB/cf.pdf). The formula does not prove that the user likes the item five times as much. It says the optimiser gives a repeated interaction more weight under a chosen scheme. The paper discusses confidence as distinct from binary preference because missing observations are uncertain.

Event weighting needs care. A purchase usually carries more commitment than a click, but a purchase may be for someone else and may not predict another purchase of the same item. Repeated views could indicate strong interest or a video that autoplayed. A negative explicit action should be treated differently from a missing event. Deduplicate noisy events, define session boundaries and preserve the type and timestamp. Do not flatten all behaviour into one positive flag without measuring what is lost.

An **impression** records that an item was displayed to the user. It is not perfect proof that the user noticed it, but it is essential context. A missing interaction for an exposed item differs from a missing interaction for an item never shown. Position matters too: the first item is more likely to be seen than the tenth. If only clicked items are logged, a dataset hides the system's earlier selection policy. A recommender trained on that dataset may learn the previous system's exposure patterns rather than users' full interests.

### Handle cold start as separate cases

A new user has little or no personal history. Possible starting points include stated interests, session context, language, location where appropriate, broad popularity and a diverse starter set. Each is a fallback, not a substitute for learning a preference after valid interactions arrive. A new item has no interaction evidence. Content features such as title, topic, creator, price or media embeddings can make it retrievable before collaborative signals accumulate. A new item with poor metadata still has a content cold start. New users and new items require different tests and may need different serving paths.

Exploration gives less-known items a controlled opportunity to collect evidence. It has an opportunity cost because some exploratory items will perform worse immediately. Keep exposure probability and selection policy in logs so later analysis can distinguish “never tried” from “tried and rejected.” Exploration should be bounded by eligibility and quality rules. A popularity-only fallback can work for a first session but, if left as the permanent policy, can starve the long tail and reinforce the existing catalogue winners.

### Define what a score means

A rating-prediction model estimates a stated numeric rating under a selection process. A click-through model estimates a click conditional on an impression, placement and context. A purchase model estimates a different event over a different horizon. A composite score may blend these with business rules. Do not compare raw scores from two candidate sources unless they are calibrated or passed to a common ranker. A dot product from one embedding model and cosine from another have no shared unit merely because both are numbers.

The task is also constrained by privacy and data governance. An interaction history can reveal sensitive interests. Collect only data needed for the feature and evaluation plan, limit retention, protect identifiers and provide controls to correct or reset a preference where the product calls for them. Avoid treating an inferred profile as a fact about a person. The model is a fallible estimate from behaviour shaped by the interface.

## A real system that works this way

The [Google recommendation course](https://developers.google.com/machine-learning/recommendation/overview/types) describes a candidate-generation, scoring and reranking architecture. A course website could use that pattern: retrieve lessons from a learner's completed topics and current course, rank them for likely educational value, then exclude already-completed lessons and enforce prerequisites. A “clicked lesson” label alone would overvalue catchy titles. The product should observe completion, comprehension or return visits where feasible, and show a transparent path for a learner who wants to choose a different topic. The [Information Retrieval lecture on personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) gives the earlier neighbour-rating example; this track extends it to feedback design and production stages.

## Code you can run

```python
interaction_count = 2
alpha = 2
preference = int(interaction_count > 0)
confidence = 1 + alpha * interaction_count
print('preference:', preference)
print('confidence:', confidence)
```

The output is preference `1` and confidence `5`. These values describe a chosen training representation, not a measured psychological state.

<FeedbackConfidenceLab />

Change the event count and confidence weight. At zero events the binary preference is zero, while the lab labels that case “unknown preference.” The default two interactions and alpha two reproduce the code's confidence five.

```python
events = [
    {'item': 'A', 'impressed': True, 'clicked': False},
    {'item': 'B', 'impressed': True, 'clicked': True},
    {'item': 'C', 'impressed': False, 'clicked': False},
]
for event in events:
    status = 'positive interaction' if event['clicked'] else ('exposed without click' if event['impressed'] else 'unobserved')
    print(event['item'], status)
```

Item A and item C have the same click value but different exposure evidence. Even item A's non-click is not a definitive dislike; placement and attention still matter.

## Designing with it

### An interaction is conditional on opportunity

Consider three learners and one advanced lesson. Learner A saw it in the first position and clicked. Learner B saw it below the fold and did not click. Learner C never had the lesson in their candidate set. Encoding A as one and both B and C as zero discards the opportunity difference. B's non-click is weak evidence because the item may not have been noticed; C's zero is no evidence about reaction to that item. A clean log contains request context, candidate sources, eligible catalogue, shown slate, position, impression timestamp and later action. It also needs a consistent identity policy across devices and sessions so that duplicate views are not mistaken for several independent endorsements.

The same behaviour can mean different things on different surfaces. Repeatedly opening a troubleshooting article may mean it is helpful or that it fails to solve the problem. Watching a tutorial to the end may reflect value, but a required training video can be completed without enthusiasm. Returning an item after purchase changes the interpretation of a purchase label. Choose a label in the context of the user's task, and inspect examples with domain experts before treating it as an objective truth. A more complex model trained on a poor label can optimise the wrong behaviour more efficiently.

### Compare objectives on a concrete slate

Suppose a course homepage can show two lessons. One short, sensational lesson is likely to get a click but rarely leads to completion. A foundational lesson gets fewer immediate clicks but helps learners finish a module. A click model may put the first lesson at the top; a completion model may choose the second. A blended score can represent a product trade-off, but the weights need evidence and ownership. Before experimenting, decide whether completion is the primary outcome and whether click rate is a diagnostic, or vice versa. The measured effect may also depend on learner stage: a new learner needs orientation while an experienced learner may value novelty.

Hard constraints are different from score preferences. A lesson behind a paywall, in a language the user did not select, or requiring an unmet prerequisite may be ineligible even if its predicted click rate is high. Filter or enforce these constraints explicitly. A soft bonus for topic diversity can be tuned; a hard prerequisite should not be traded away because an engagement score is large. Record why an item was filtered so the system can explain a missing recommendation and operators can diagnose unexpected catalogue gaps.

### Cold start is a measurement problem too

A new item cannot collect interactions unless it is shown. A recommender that relies only on historic interaction counts may assign it no score and never show it, then infer from zero interactions that users do not want it. Content-based retrieval can place it in plausible candidate sets; a bounded exploration policy can provide initial exposure. Measure early-item performance by launch cohort and age, not only by all-time totals. Similarly, a new user fallback should be evaluated on actual first sessions rather than on existing users after their histories are artificially hidden, because first-session context and selection can differ.

Popularity is an honest baseline if its limitations are stated. It often works well where many users share interests and labels are scarce. It can also create a feedback loop and a narrow catalogue. Compare it with personalised methods on the same eligible set, and report both immediate engagement and distribution of exposure. If personalisation's gain is small, its additional data collection, privacy cost and serving complexity may not be justified.

### Preserve user agency

An inferred profile should be revisable. A user may buy a gift, research a topic for work or change interests. A “not interested” control is a direct signal that deserves distinct handling, and a reset option can help where long histories trap users in old categories. Show enough context that a person understands why an item appears when the product supports explanation. These controls can improve both experience and data quality, but their events should be logged separately from passive non-clicks. Treating every absence of interaction as rejection undermines that distinction.

Create an event dictionary before training. For each event specify how it is emitted, when it becomes available, what exposure it implies, what it does not imply and which user controls can reverse it. Define a label horizon: a click within ten minutes and a purchase within seven days are not interchangeable targets. De-duplicate bot or retry events, and audit whether the instrumentation changed during the training window. A model trained across a logging migration can mistake the migration for a change in preference.

Separate user cold start, item cold start and both-new cases in evaluation. A random split of existing interactions rarely tests either case. Use a chronological split and a held-out entity slice where needed. Include an eligibility snapshot so a model is not penalised for failing to recommend an item that was unavailable. The evaluation chapter develops ranking metrics and exposure limitations in detail.

Keep a simple baseline, such as eligible popularity conditioned on locale or topic, and measure it on the same surface. The goal is a better experience, not a more complex model. If the baseline wins, examine label quality, exposure bias, stale inventory and whether the intended outcome differs from the optimised event.

## Where this stands in 2026

Modern recommenders can combine behaviour, item content and contextual features, yet explicit and implicit evidence remain different kinds of data. Current production designs still separate broad candidate retrieval from richer scoring and policy-aware reranking. An interaction model has to coexist with privacy choices, new users and newly published items. The most durable design decision is to make the objective, exposure and fallback paths observable.

## Practice questions

<details>
<summary>Why is an unclicked item not automatically a negative example?</summary>

It may never have been displayed. Even when displayed, position, attention and context affect the chance of a click. Record impressions and interpret non-clicks conditionally.

</details>

<details>
<summary>What is the difference between an explicit rating and an implicit watch?</summary>

A rating is a declared judgement on a scale; a watch is behaviour from which interest is inferred. Both are selected observations and need their collection context.

</details>

<details>
<summary>What does confidence five mean in the example?</summary>

It is the weight assigned by $1+\alpha r$ with count two and alpha two. It is not a five-star rating or a probability of liking the item.

</details>

<details>
<summary>How would you recommend a newly published lesson?</summary>

Use topic and prerequisite metadata to retrieve it, filter for eligibility, give it measured exposure, and collect interaction and outcome evidence. A collaborative factor cannot be learned from interactions that have not happened.

</details>

## Further reading

- [Google's recommendation overview](https://developers.google.com/machine-learning/recommendation/overview/types) defines the staged architecture.
- [Collaborative filtering basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics) distinguishes explicit and implicit feedback.
- [Hu, Koren and Volinsky, implicit-feedback collaborative filtering](https://yifanhu.net/PUB/cf.pdf) separates preference from confidence.
- [Recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval) gives this curriculum's introductory worked rating example.

## Check yourself

- I can explain why a missing interaction is not a measured dislike.
- I can compute the preference and confidence of the two-event example.
- I can distinguish user cold start from item cold start and choose a fallback for each.
- I can specify an objective, eligibility rules, exposure logs and a label horizon before model training.
