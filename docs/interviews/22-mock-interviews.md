---
title: Mixed-Topic Mock Interviews and Study Plan
sidebar_label: "22 · Mocks and study plan"
sidebar_position: 22
---

# Mixed-Topic Mock Interviews and Study Plan

Use timed implementation and cross-examination to find what you can explain under pressure.

These are original practice rounds assembled from the [reported task families](24-sources.md) and the new topic banks. They are organised by task, across all target roles. They are not leaked employer rounds or predictions of a particular company's process.

## How to score a response

Score each dimension from 0 to 3: **0** missing/wrong; **1** definition only; **2** correct implementation or concrete design; **3** implementation plus tested edge cases and defended trade-offs.

| Dimension | Evidence for a strong answer |
| --- | --- |
| Problem and contract | Explicit inputs, outputs, authority, constraints and success criteria |
| Mechanism | Correct causal explanation, derivation or data flow |
| Implementation | Runnable core code, real schemas, identity and error handling |
| Measurement | Suitable labels, metrics, slices and uncertainty |
| Failure handling | Reproduction, isolation, recovery and regression tests |
| Trade-offs | Compared alternatives using actual constraints and stated assumptions |

An aggregate score cannot compensate for a severe error such as allowing cross-tenant disclosure, silently omitting failed cases or claiming an untested solution is production-proven. This is a study rubric, not a validated hiring threshold.

## Round A · Build and debug document Q&A, 60 minutes

**Prompt:** A support team has private policies from several tenants. Build a baseline for queries, explain citations and update documents. Ten minutes into the round, the interviewer reveals that exact product IDs retrieve the wrong product. Later, a revoked user still gets cached answers.

| Time | Work |
| --- | --- |
| 0–8 minutes | Clarify tenant identity, corpus, updates, expected answers and latency |
| 8–25 minutes | Implement a baseline using [Lab 1](21-coding-labs.md#lab-1) contracts |
| 25–40 minutes | Diagnose retrieval and cache failures using intermediate artefacts |
| 40–52 minutes | Design labels, ranking metrics, answer/citation checks and negative cases |
| 52–60 minutes | Compare options, summarise limits and propose a measured rollout |

**Expected answer:** Enforce server-side tenant scope, keep document/revision IDs, inspect exact-token versus semantic matching, use an oracle-context test, and bind caches to permission and content revisions. Recheck access before returning citations. Evaluate retrieval and generated claims separately, including abstention coverage.

**Cross-questions with answer direction:** “Cosine is 0.9; why is the answer wrong?” Similarity is not calibrated relevance or entailment. “Can reranking solve missing evidence?” Only if it was in the candidate set. “Why not put the whole corpus in the prompt?” Permissions, cost, freshness and distractors still apply.

**Read after attempting:** RAG01–12, RAG19–20, RAG30–40; QA03, QA29; OPS05; SYS02, SYS17, SYS34.

## Round B · Reliable tool execution, 60 minutes

**Prompt:** An assistant prepares a refund, gets approval and calls a tool. The connection times out. A restarted worker repeats the action. The user then cancels while the provider is still working.

**Expected implementation:** Adapt [Lab 4](21-coding-labs.md#lab-4) with immutable action identity and payload binding. Demonstrate pre-commit rollback, post-commit response loss, restart and conflicting payload rejection. Draw the remote-provider boundary explicitly.

**Expected design:** Separate proposed, approved, submitted, uncertain, committed and compensated states. Approval binds the exact operation; tool execution rechecks current authority and mutable preconditions. Use provider idempotency where supported and reconcile uncertain outcomes. Preserve a receipt and audit history. A cancellation signal is not proof that the effect stopped.

**Cross-questions:** “Can we generate a new ID on retry?” That risks a second effect. “The graph checkpoint says the node ran, are we safe?” Not unless the effect and execution protocol handle crash windows. “Can a database transaction undo an external refund?” No; compensation is a separate domain action.

**Read after attempting:** PY02, PY25, PY29; AG03–05, AG14–16, AG32–34; QA12, QA15; SYS32, SYS37.

## Round C · A model is 3 points better, 50 minutes

**Prompt:** A candidate scores 83% against an 80% baseline. The dashboard excludes timeouts. One rare tenant slice regresses, and the judge model was upgraded between runs. Decide whether to release and implement a comparison.

**Expected answer:** First restore complete case identity and distinguish failures from missing scores. Re-evaluate under the same judge/rubric or establish an overlap study. Compare paired cases, estimate uncertainty at the independent unit, inspect severe and high-volume slices, and use a prespecified margin. Report quality, safety, latency, cost and completion separately.

**Coding:** Implement [Lab 8](21-coding-labs.md#lab-8). Inject a duplicate ID, NaN, missing score and critical failure. None may silently produce a pass.

**Cross-questions:** “Zero attacks succeeded in 100 cases; is risk zero?” No; under independent trials the one-sided 95% upper bound is about 3%. “Why not bootstrap individual messages?” Messages from one conversation may be correlated. “Can longer answers win the judge unfairly?” Yes; calibrate correctness separately from verbosity and order effects.

**Read after attempting:** ML08–09, ML27–32; EV04–06, EV10–20, EV29–40; QA16, QA39.

## Round D · Numerical implementation, 60 minutes

**Prompt:** Implement multi-head causal attention or a convolution reference. The interviewer changes batch size, adds padding and asks for memory complexity. Then derive a logistic-loss gradient and explain why a different loss changes optimisation.

**Expected answer:** Name every tensor axis, project Q/K/V, scale by head width, normalise the key axis and treat fully masked rows explicitly. For convolution, distinguish cross-correlation from flipped mathematical convolution and test asymmetric kernels. Use stable log-loss and finite-difference checks.

**Cross-questions:** “Future-token changes alter the first output; what failed?” The causal mask or positions. “Why does a seed not guarantee identical GPU output?” Runtime, kernels, hardware and nondeterminism also matter. “Does LoRA reduce all memory 128 times?” That ratio can describe trainable matrices only; base weights and activations still exist.

**Read after attempting:** [Labs 2 and 7](21-coding-labs.md); ML03–04, ML13–14; DL01–05, DL13–25, DL30, DL39.

## Round E · A production incident, 45 minutes

**Prompt:** Traffic rises tenfold. Median latency is fine, p99 is poor, token quotas are exhausted and the GPU sometimes runs out of memory. A retry change made it worse.

**Expected answer:** Trace the critical path and distinguish queueing, prefill, decode, database and tool waits. Examine token distributions and concurrent KV memory. Bound admission, queue age and retries; apply fairness across interactive and background traffic. Consider batching, routing and caching only with correctness and quality checks. Define recovery and rollback before tuning further.

**Cross-questions:** “Why not add API workers?” The same downstream bottleneck may remain. “Can we add each stage's p95?” Quantiles of sums depend on the joint distribution. “What must rollback include?” A compatible model, prompt, schema, index and configuration bundle.

**Read after attempting:** PY01, PY09–12, PY39; OPS01–40; SYS05, SYS16, SYS26, SYS38.

## Round F · Historical data and applied modelling, 60 minutes

**Prompt:** Recommend relevant items or forecast regional demand. Validation looks excellent, but new users or new regions perform badly. A feature pipeline backfills historical events and the experiment increases clicks without increasing task success.

**Expected answer:** Define entity/time/spatial splits, build a transparent baseline and audit point-in-time availability. Examine cold start, exposure bias, metric choice and downstream effects. Use the feature join in [Lab 5](21-coding-labs.md#lab-5). Compare a global model with local or hierarchical alternatives using held-out regions and realistic serving constraints.

**Cross-questions:** “An event happened yesterday; can yesterday's model use it?” Only if it was available then. “Does a high silhouette score prove useful segments?” No; downstream usefulness needs measurement. “Can inverse propensity weighting evaluate an unseen action?” Without overlap, the effect is not identified by those logs.

**Read after attempting:** PY04, PY19–24; ML01–02, ML17–19, ML29–35; APP01–02, APP09–20; SYS03, SYS22–24.

## Project defence, 15 minutes after any round

Use one real project, including a personal lab if that is the work you have done. State its real scale and your own contribution.

1. What was the user's task, baseline and success criterion?
2. Which data and decisions did you own?
3. What was the most revealing failure, and how did you reproduce it?
4. What experiment changed your design?
5. What did you measure: quality, coverage, latency, cost and severe errors?
6. What remains untested, and what would you do before a larger rollout?

**Answer structure:** requirement → initial design → observed failure → isolated cause → change → evidence → remaining limitation. Bring a trace, a schema, a small code path and one measured comparison. Explain uncertainty instead of inventing production traffic or percentages.

## Study schedule by topic

| Pass | Work | Exit evidence |
| --- | --- | --- |
| Programming and data (1–5) | Python, algorithms, SQL, NumPy and pandas | Solve small coding and data-transformation exercises without notes |
| Statistics and modelling (6–12) | Probability, ML, scikit-learn, deep learning, LLMs and applied modelling | Derive losses, compare models and defend validation choices |
| Building AI systems (13–16) | APIs/data pipelines, system design foundations, RAG and agents | Draw the flow, implement a core routine and explain each failure boundary |
| Advanced production work (17–20) | Evaluation, QA/security, serving and architecture cases | Show a gate rejecting bad outputs; defend capacity, rollout and recovery |
| Interview rehearsal (21–22) | Coding labs, six mixed-topic mocks and project defence | Repeat weak rounds with new inputs rather than memorised wording |

At the end of a topic, close the page and reconstruct its summary. Mark each question **explain**, **implement**, **test**, **defend**. Study the missing capability rather than rereading only the answer you already recognise.

## Summary in simple points

- Practise mixed-topic tasks because a production problem crosses several disciplines.
- Start with a contract, then implement, measure, test and defend the design.
- Use source labels accurately; these mock scenarios are original exercises.
- The six rounds cover retrieval, actions, evaluation, numerical coding, operations and applied data problems.
- Treat missing data, access violations and duplicated actions as explicit failures.
- Compare options using constraints and evidence rather than memorised preferences.
- Defend only experience and results you can demonstrate.
- Revisit weak mechanisms with new fixtures and repeat the timed round.
