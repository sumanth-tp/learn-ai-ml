---
id: index
slug: /interviews/
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

Worked answers, numbers, cross-questions, diagrams, rubrics, and labs are original teaching material unless stated otherwise. Sources establish interview provenance or technical behaviour, not an official hiring answer key. [Read the evidence ledger](24-sources.md), including interview dates, source limitations, and YouTube access notes.

The aim is broad, defensible preparation. A claim that any guide covers **99% of future interview questions** would require evidence we do not have. Use the capability checklist and timed mocks to measure readiness instead.

## Learning order: foundations to advanced

Follow the numbered sidebar in order. The same sequence controls the previous/next links at the bottom of each page. If you already know a foundation, use its summary to check your understanding before moving on.

1. **Programming and data, 1–5:** Python, algorithms, SQL, NumPy and pandas.
2. **Statistics and modelling, 6–12:** probability, ML, scikit-learn, practical statistical reasoning, deep learning, LLMs and applied modelling.
3. **Building AI systems, 13–16:** APIs and data pipelines, system design foundations, RAG and agents.
4. **Advanced production work, 17–20:** evaluation, testing/security, serving/operations and architecture cases.
5. **Interview rehearsal, 21–22:** coding labs and mixed-topic mocks.

Use the [coding labs](21-coding-labs.md) alongside their linked topics, then repeat them under time limits during rehearsal. Consult [tools and versions](23-tools-versions.md) when running code and [evidence and sources](24-sources.md) when checking interview provenance.

## Topic map

| Order | Topic | What you should be able to do |
| --- | --- | --- |
| 1 | [Python foundations](01-python.md) | Use Python objects, functions, classes and error handling. |
| 2 | [Data structures and algorithms](02-dsa.md) | Choose data structures and reason about complexity. |
| 3 | [SQL foundations](03-sql.md) | Write joins, aggregations, windows and reliable queries. |
| 4 | [NumPy foundations](04-numpy.md) | Understand arrays, shapes, broadcasting and numerical operations. |
| 5 | [pandas foundations](05-pandas.md) | Clean, join, reshape and validate tabular data. |
| 6 | [Statistics and probability](06-statistics.md) | Explain distributions, sampling, estimation and uncertainty. |
| 7 | [ML foundations](07-machine-learning.md) | Understand losses, algorithms, features and validation. |
| 8 | [scikit-learn implementation](08-scikit-learn.md) | Build preprocessing pipelines and compare estimators. |
| 9 | [Practical ML and statistics](09-ml-statistics.md) | Diagnose leakage, choose thresholds and defend experiments. |
| 10 | [Deep learning foundations](10-deep-learning.md) | Explain neural networks, backpropagation and training. |
| 11 | [Transformers and LLMs](11-deep-learning-llms.md) | Implement attention and defend adaptation and inference choices. |
| 12 | [Applied modelling](12-applied-modelling.md) | Apply modelling to ranking, recommendations, vision, speech and forecasting. |
| 13 | [APIs and data pipelines](13-python-data.md) | Bound concurrency, recover failures and build historical features. |
| 14 | [System design foundations](14-system-design.md) | Understand APIs, storage, queues, consistency and capacity. |
| 15 | [Retrieval and RAG](15-rag.md) | Build ingestion and retrieval, preserve permissions and verify evidence. |
| 16 | [Agents and tools](16-agents.md) | Design bounded workflows, durable state, tools and MCP integration. |
| 17 | [Evaluation](17-evaluation.md) | Build datasets, calibrate judges and compare changes reliably. |
| 18 | [Testing and security](18-testing-security.md) | Test nondeterminism, isolation, injection and tool effects. |
| 19 | [Serving and operations](19-serving-mlops.md) | Handle load, latency, deployment, monitoring and rollback. |
| 20 | [Architecture and project defence](20-system-design.md) | Combine the topics into justified, measurable end-to-end designs. |
| 21 | [Coding labs](21-coding-labs.md) | Implement and test the core mechanisms under time limits. |
| 22 | [Mocks and study plan](22-mock-interviews.md) | Practise mixed-topic rounds and defend your project decisions. |

Each practical topic includes explanations, visualisations, answered follow-ups, code and a **complete plain-language summary at the end**. The foundation banks also end with summaries; use their beginner sections first and return to advanced questions as your understanding grows.

## The answer pattern to practise

1. **Clarify the decision.** User, workload, cost of a wrong answer, freshness, permissions, scale, and latency target.
2. **Give a baseline.** State the simplest viable solution and its known failure modes.
3. **Make the mechanism concrete.** Data shapes, schema, APIs, state transitions, equations, or code.
4. **Measure it.** Dataset, denominator, slices, baseline comparison, uncertainty, and business consequence.
5. **Break it.** Missing data, adversarial input, timeout, retry, concurrency, stale permissions, and distribution shift.
6. **Operate it.** Logs, traces, release gates, rollback, ownership, and escalation.

For a two-minute answer, use one sentence per step. For a forty-five-minute design round, expand each step with numbers and alternatives. Numerical constraints in the new chapters are exercise inputs, not published benchmarks.

## About the foundation banks

The ten earlier banks are now placed before the practical topics they support. They contain 100 entries each, including overlapping material and illustrative snippets. They predate this source audit; they are **not evidence that those exact questions were asked**, and their snippets are not all standalone programmes. Learn the prerequisites there, then use the practical chapters for sourced scenarios and implementation depth. The [version notebook](23-tools-versions.md) explains current API differences.

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

- Start with programming and data, then statistics and modelling, AI systems, production work and interview rehearsal.
- Follow the numbered sidebar; consult labs and version notes as you study each topic.
- A reported interview question and an original practice question have different labels.
- Explain assumptions and trade-offs; do not recite tool names as an answer.
- Learn the mechanism, run the code, then answer the cross-questions aloud.
- Use the diagrams to explain flow, state, and failure points.
- Read each topic's final summary for revision after studying its detailed answers.
- The existing foundation banks fill prerequisite gaps; the new chapters add practical depth.
- Use the labs and mocks to find weak areas. No fixed question list guarantees an interview result.
