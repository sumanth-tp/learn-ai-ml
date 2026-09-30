---
id: py-video-guide
title: "Python for AI - Full Beginner Course"
sidebar_label: "Start here · Video guide"
sidebar_position: 1
slug: /code/python/video-guide
description: "A beginner route through Python, with a video coverage map, practical exercises and links into the existing course."
tags: [python, beginner, learning-path, video]
---

> **Source:** Dave Ebbelaar, [Python for AI - Full Beginner Course](https://www.youtube.com/watch?v=ygXn5nV5qFc), 5h 15m. Companion: [course handbook](https://python.datalumina.com/).

Start here to learn the language and the tools needed to run a small Python project yourself.

This route adds a beginner foundation to the existing Python course. Work through the setup first, then the introductory sections in the language chapters, then the two data projects. The deeper explanations, examples, milestones and capstone remain available in their existing chapters.

## How to study

1. Type the small examples, predict their output, then run them.
2. Use the interactive window to inspect one expression at a time.
3. Restart the kernel and run the complete script before considering an exercise finished.
4. Change an input and explain why the output changes.
5. Recreate the project in a new folder without copying your virtual environment.

The first goal is a script you can explain, run again and share with its dependencies. An AI assistant can help explain an error; check its suggestion by reading the traceback and running the smallest example that reproduces the problem.

## Video coverage and reading order

The links below use approximate topic starts checked against the transcript. Some published YouTube chapter labels differ from what is on screen, particularly around strings, control flow and data analysis.

| Video location | What to learn | Read and practise here |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=0s) | Course purpose, handbook and practice habits | This guide |
| [03:58](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=238s) | Python installation on Windows/macOS, VS Code, extensions, workspace, first file | [Setup and first run](./02-setup-and-interactive-python.md) |
| [31:07](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=1867s) | Environments, pip, imports, Anaconda context, interactive Python | [Environment and kernel setup](./02-setup-and-interactive-python.md#one-environment-for-this-project) |
| [51:36](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=3096s) | Programming, syntax, PEP 8, errors, variables and comments | [Your first Python program](./03-first-program.md) |
| [1:10:12](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=4212s) | Numbers, strings, booleans, operators, f-strings and string methods | [Types and data structures](../01-core-language/01-types-and-data-structures.md#start-here-values-expressions-and-containers) |
| [1:36:00](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=5760s) | Conditions, indentation, loops and `range` | [Control flow](../01-core-language/02-control-flow-and-comprehensions.md#start-here-choose-a-branch-then-repeat-an-action) |
| [1:47:45](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=6465s) | Lists, indexing, dictionaries, tuples and sets | [Container practice](../01-core-language/01-types-and-data-structures.md#four-containers-you-can-create-and-change) |
| [2:05:51](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=7551s) | Defining/calling functions, parameters, scope and returned values | [Functions](../01-core-language/03-functions-and-scope.md#start-here-input-work-result) |
| [2:38:15](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=9495s) | Standard library, external packages, import forms and requirements | [Imports](../02-programs/01-modules-and-packages.md#start-here-installing-and-importing-are-different-steps), [setup](./02-setup-and-interactive-python.md#install-import-and-recreate) |
| [2:57:00](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=10620s) | HTTP requests, JSON and a reusable weather function | [First API and weather report](../02-programs/06-first-api-and-weather-report.md) |
| [3:06:20](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=11180s) | Dates, pandas tables, Matplotlib charts, PNG and CSV output | [Weather report lab](../02-programs/06-first-api-and-weather-report.md#from-json-to-a-table-and-chart) |
| [3:20:30](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=12030s) | Project structure, paths, CSV/JSON/Excel and helper modules | [Sales analysis lab](../02-programs/07-sales-analysis.md), [file paths](../02-programs/03-files-and-context-managers.md#start-here-where-is-python-looking) |
| [3:39:39](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=13179s) | Syntax/runtime errors and `try/except` | [Errors and recovery](../02-programs/02-errors-and-exceptions.md#start-here-classify-the-failure) |
| [3:45:31](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=13531s) | Classes, instances, `self`, methods, inheritance and state | [First class](../02-programs/04-oop-and-the-data-model.md#start-here-one-class-two-independent-objects) |
| [4:09:44](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=14984s) | Git, GitHub, authentication, clone, commit, push and VS Code UI | [Complete project workflow](../06-engineering/00-project-workflow.md#git-and-github-step-by-step) |
| [4:44:05](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=17045s) | Environment variables, `.env` and `python-dotenv` | [Configuration and secrets](../06-engineering/00-project-workflow.md#configuration-and-env-files) |
| [4:55:45](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=17745s) | Ruff formatting, linting and import sorting | [Ruff in the editor and terminal](../06-engineering/00-project-workflow.md#format-lint-and-sort-imports) |
| [5:01:10](https://www.youtube.com/watch?v=ygXn5nV5qFc&t=18070s) | uv, dependency management, rebuilding a project and final exercise | [uv workflow and final exercise](../06-engineering/00-project-workflow.md#start-a-project-with-uv) |

## What the review added

The review used the complete English automatic-caption transcript and 84 sampled video frames spanning all published chapters. The frames were used to check editor actions, code layout, output and demonstrations. These notes use original explanations and examples; the sampled images and transcript are not republished here.

The main additions are installation and kernel troubleshooting, basic syntax before advanced idioms, a weather API/report exercise, a modular sales analysis exercise, and the complete Git/secrets/Ruff/uv workflow. Sections marked **Added practice** extend the video with edge cases, expected results and independently runnable examples.

### Clarifications to carry into your own code

| Simplification in the demonstration | Precise rule used in these notes |
| --- | --- |
| Triple-quoted strings described as multiline comments | `#` introduces a comment. Triple quotes create a string; in specific positions that string becomes a docstring. |
| `pass` presented alongside a function that does not return a value | `pass` does nothing. Reaching the end of a function without `return` produces `None`, whether or not `pass` appears. |
| Paths explained from the script's folder | Relative paths use the process's current working directory. The demonstrated editor setting can make that match the script folder; `__file__` lets a script choose an explicit base. |
| JSON described as string-based | JSON has strings, numbers, booleans, null, arrays and objects. Dates usually travel as strings because JSON has no native date type. |
| “Class methods” used for functions inside a class | Most demonstrated methods take `self` and are instance methods. Python's `@classmethod` is a separate feature taking `cls`. |
| Selecting the newest Python installation | Select a version supported by your project's dependencies, and use the same interpreter for installs and execution. |

The weather lab also makes the date range explicit: seven completed calendar days exclude today. When both endpoints are included, “today minus seven days through today” covers eight dates.

## After the beginner route

Continue with the existing [tic-tac-toe milestone](../01-core-language/99-milestone-tic-tac-toe.md), [bank ledger](../02-programs/99-milestone-bank-ledger.md), Pythonic tools, standard library, typing and concurrency. The [Orderflow capstone](../07-capstone/01-capstone-orderflow-service.md) combines the engineering material later in the course.

The video's closing section points learners to separate agent-building material. This Python route prepares you for that next step; the two labs here exercise APIs and data processing without needing an AI service account.

## Ready to move on?

- [ ] I can create a project and explain which interpreter runs it.
- [ ] I can explain a function's inputs, return value and side effects.
- [ ] I can turn API data or a CSV into a saved report.
- [ ] I can recreate the environment and run the project from a fresh terminal.
- [ ] I can review what Git will save and keep local credentials out of it.
