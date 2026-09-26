---
title: Practical AI and ML Interview Preparation
sidebar_label: Start here
sidebar_position: 0
description: Topic-based interview preparation with candidate reports, deep explanations, diagrams, answered follow-ups, and executable code.
---

# Practical AI and ML Interview Preparation

Prepare to explain, implement, debug, evaluate, and operate AI systems under interview pressure.

**Research reviewed: 26 September 2026.** This guide is organised by **topic**, across AI engineering, ML engineering, AI/ML engineering, agentic AI, RAG, LLM evaluation, and LLM/agent QA. Every topic connects implementation, architecture, measurement, and testing. There are no separate role silos.

**400 primary questions across 10 topic banks, 40 per topic**, plus answered cross-questions, executable examples, comparison tables, expandable diagrams and complete chapter summaries. Eight downloadable coding labs and six mixed-topic mock rounds turn revision into implementation practice.

## What counts as an actual interview question?

Every new question carries an evidence label:

- **Reported:** a candidate or interviewer explicitly reported the question or task. Wording here is paraphrased. Public reports are self-reported, not employer-verified.
- **Reported theme → practice scenario:** the source reports the topic; the specific constraints and incident are original exercises.
- **Practice extension:** an original question to close a concept gap or deepen a reported question. It is not presented as something a company asked.

Worked answers, numbers, cross-questions, diagrams, rubrics, and labs are original teaching material unless stated otherwise. Sources establish interview provenance or technical behaviour, not an official hiring answer key. [Read the evidence ledger](98-sources.md), including interview dates, source limitations, and YouTube access notes.

The aim is broad, defensible preparation. A claim that any guide covers **99% of future interview questions** would require evidence we do not have. Use the capability checklist and timed mocks to measure readiness instead.

## Topic map

| Topic | What you must be able to do | Start here |
| --- | --- | --- |
| Python, APIs, SQL, data pipelines | Bound concurrency, recover failures, join historical features, prevent duplicate events | [Software and data](01-python-data.md) |
| ML, probability, statistics | Diagnose leakage, derive losses, choose thresholds, explain uncertainty and experiments | [ML and statistical reasoning](02-ml-statistics.md) |
| Deep learning, transformers, LLM adaptation | Implement attention, calculate memory, debug training, defend adaptation choices | [Deep learning and LLMs](03-deep-learning-llms.md) |
| Retrieval and RAG | Build ingestion, diagnose retrieval, evaluate citations, migrate indexes | [RAG and search](04-rag.md) |
| Agents, tools, state, MCP | Choose orchestration, control effects, resume safely, evaluate trajectories | [Agents and tools](05-agents.md) |
| Evaluation and experimentation | Build datasets, calibrate judges, compare changes, operate evaluation jobs | [Evaluation](06-evaluation.md) |
| Testing and security | Test nondeterminism, tenant isolation, injection, tool misuse, and release gates | [Testing and security](07-testing-security.md) |
| Serving, LLMOps, MLOps | Budget latency and tokens, handle load, version deployments, monitor and roll back | [Serving and operations](08-serving-mlops.md) |
| System design and project defence | Define interfaces, capacity, release plans, and justified decisions | [Architecture cases](09-system-design.md) |
| Recommendations, vision, speech, time series | Apply modelling and evaluation discipline across modalities | [Applied modelling](10-applied-modelling.md) |
| Coding rounds | Implement retrieval metrics, attention, concurrency, feature joins, and quality gates | [Executable labs](11-coding-labs.md) |
| Interview rehearsal | Answer under time pressure and identify remaining gaps | [Mocks and readiness](12-mock-interviews.md) |
| Tools and versions | Explain exact contracts and plan an upgrade | [Version notebook](99-tools-versions.md) |

Each topic includes deep explanations, visualisations, answered follow-ups, practical tests, and a **complete plain-language summary at the end**. Start with the explanations; use the summaries for final revision.

## The answer pattern to practise

1. **Clarify the decision.** User, workload, cost of a wrong answer, freshness, permissions, scale, and latency target.
2. **Give a baseline.** State the simplest viable solution and its known failure modes.
3. **Make the mechanism concrete.** Data shapes, schema, APIs, state transitions, equations, or code.
4. **Measure it.** Dataset, denominator, slices, baseline comparison, uncertainty, and business consequence.
5. **Break it.** Missing data, adversarial input, timeout, retry, concurrency, stale permissions, and distribution shift.
6. **Operate it.** Logs, traces, release gates, rollback, ownership, and escalation.

For a two-minute answer, use one sentence per step. For a forty-five-minute design round, expand each step with numbers and alternatives. Numerical constraints in the new chapters are exercise inputs, not published benchmarks.

## Foundation reference banks

The earlier banks remain available for concept revision. They contain 100 entries each, including overlapping material and illustrative snippets. They predate this source audit; they are **not evidence that those exact questions were asked**, and their snippets are not all standalone programmes. Use the new chapters for practical depth and the [version notebook](99-tools-versions.md) for current API differences.

| Fundamentals | Reference |
| --- | --- |
| Programming and algorithms | [Python](python.md), [DSA](dsa.md), [SQL](sql.md) |
| Numerical and tabular work | [NumPy](numpy.md), [pandas](pandas.md), [scikit-learn](scikit-learn.md) |
| Modelling and mathematics | [Machine learning](machine-learning.md), [Deep learning](deep-learning.md), [Statistics](statistics.md) |
| Architecture vocabulary | [System design foundations](system-design.md) |

## Coverage checklist

Score each capability **0** (cannot explain), **1** (definition only), **2** (can implement a baseline), or **3** (can defend, test, and debug it). A memorised answer earns at most 1.

- [ ] I can write Python, SQL, array operations, and core data structures without relying on a framework.
- [ ] I can derive and interpret losses, gradients, sampling, and uncertainty calculations.
- [ ] I can justify features, splits, baselines, and metrics for classification, ranking, regression, and forecasting.
- [ ] I can explain backpropagation, attention, decoding, adaptation, alignment, and inference memory.
- [ ] I can isolate parsing, retrieval, reranking, context, and generation failures.
- [ ] I can design bounded agents with durable state, access control, and idempotent effects.
- [ ] I can distinguish ground truth, reference-free scores, and business outcomes.
- [ ] I can test behaviour, contracts, security boundaries, and nondeterministic failures.
- [ ] I can design capacity, deployment, observability, cost controls, and rollback.
- [ ] I can defend a project with evidence, including a failed experiment and the decision it changed.

## Summary in simple points

- Study by topic and practise building, measuring, and testing the same system.
- A reported interview question and an original practice question have different labels.
- Explain assumptions and trade-offs; do not recite tool names as an answer.
- Learn the mechanism, run the code, then answer the cross-questions aloud.
- Use the diagrams to explain flow, state, and failure points.
- Read each topic's final summary for revision after studying its detailed answers.
- The existing foundation banks fill prerequisite gaps; the new chapters add practical depth.
- Use the labs and mocks to find weak areas. No fixed question list guarantees an interview result.
