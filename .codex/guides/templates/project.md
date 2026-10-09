---
id: <topic>-project-<n>-<slug>
title: "Project <n>: <what it builds>"
sidebar_label: "Project <n> · <short label>"
sidebar_position: <N>
slug: /<area>/project-<n>-<slug>
description: "<one sentence: what the reader builds and what it proves>"
tags: [project, <tag>, <tag>]
---

import Infographic from '@site/src/components/Infographic';

**In one line.** <what the reader will have built>

:::note Added for this site

This project applies what the <topic> chapters teach. It is not part of the source course.

:::

## The problem statement

### Background

### Users

### Current pain

### Scope and non-goals

### Constraints

### Success criteria

| Measure | Target |
| --- | --- |
| <measure> | <number> |

### A worked example, end to end

<one input traced through the system by hand, to its output>

## What you will learn

- <skill>

**Chapters this project uses**

| Chapter | What the project uses from it |
| --- | --- |
| [<chapter>](/docs/<slug>) | <the idea or code> |

## Requirements

### Functional requirements

### Non-functional requirements

## Architecture

<Infographic
  src="/img/<track>/<project>-architecture.svg"
  alt="<components and data flow>"
  caption="<where to look first>"
/>

### Design decisions

| Decision | Chosen | Rejected | Why |
| --- | --- | --- | --- |

## Tech stack

| Package | Version | Why |
| --- | --- | --- |

## Repository layout

```text
<name>/
  pyproject.toml  uv.lock  Makefile  README.md  .env.example
  src/<package>/...
  tests/...
```

## How to install

```bash
unzip <name>.zip && cd <name>
uv sync --frozen
make test
```

## How to configure

| Variable | Default | Meaning |
| --- | --- | --- |

## Build it task by task

### Task 1: <goal>

<what this task adds and why>

```python
<full file, generated from the repo>
```

```bash
<command>
```

<output you saw, and what it shows>

**Checkpoint.** <the test that proves the task works>

## Testing

## Evaluation

## Observability

## Security and safety

## Deployment

## Cost and scaling

## Failure modes and runbook

| Symptom | Cause | Fix |
| --- | --- | --- |

## Extensions

## Interview questions

### The 2-minute pitch

### Concepts

<details>
<summary><strong>Q1.</strong> <question></summary>

<model answer>

</details>

### System design

### Debugging and incidents

### Trade-offs

## Checklist

- [ ] I can <capability>

## Download

Download the complete project: [<name>.zip](/examples/projects/<name>.zip)

```bash
unzip <name>.zip && cd <name>
uv sync --frozen
make test
make demo
```
