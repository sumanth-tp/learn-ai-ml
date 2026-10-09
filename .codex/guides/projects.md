# Projects for every topic

Every topic on the site ends with projects. A project is where the reader stops following chapters and builds something whole from what the chapters taught, with a repo that runs, tests that pass, and decisions they can defend in an interview.

The model to copy is `docs/agentic-ai/29-31` (the three agentic projects) with their ZIPs in `static/examples/projects/`. Read one fully before writing a project.

## What a topic is, and how many projects it gets

A topic is a folder under `docs/` with its own `_category_.json` that holds chapters or numbered modules: `theory/cv`, `mlops/data`, `llm-engineering`, `mcp`, `genai/rag-advanced`, `code/2.pandas`. Grouping folders (`theory`, `mlops`, `code`, `scaler`, `projects`) are not topics. Module folders (`01-foundations`) are not topics.

See where things stand:

```bash
/usr/bin/python3 scripts/source-import/project_inventory.py
/usr/bin/python3 scripts/source-import/project_inventory.py --uncovered
```

It lists each topic, its chapters, its project pages, and how many chapters no project links to. On 2026-10-10, 35 of 50 topics had no project.

| Topic size | Projects |
| --- | --- |
| 6 chapters or more | Three: guided, applied, portfolio |
| 3 to 5 chapters | Two: guided, applied |
| 1 or 2 chapters | One, or share a project with a neighbouring topic and say so in both places |

| Level | Time | What it is |
| --- | --- | --- |
| Project 1, guided | 2 to 4 hours | Extends the chapters' code into one small tool, step by step. Every step is given. |
| Project 2, applied | 1 to 2 days | A real public dataset or API, a fuller problem, design choices for the reader to make, with the solution given after each task. |
| Project 3, portfolio | 3 to 5 days | Production shape: configuration, tests, evaluation, logging, a container, failure modes, cost. Something worth showing an employer. |

**Every chapter of the topic is used by at least one project.** Each project has a table "Chapters this project uses", linking each chapter (that table is what the inventory reads). If a chapter fits no project, a project gets a task for it, or the gap goes in the progress file.

## Where the files go

| Topic layout | Project pages |
| --- | --- |
| Flat (chapters directly in the topic folder, like `agentic-ai`) | `NN-project-1-<slug>.md` numbered after the last chapter |
| Modules (`01-...`, `02-...` subfolders, like `theory/cv`) | A `90-projects/` folder with its own `_category_.json` (label "Projects", unique site-wide, e.g. "Computer vision projects") |

The repo goes in `static/examples/projects/<name>/` with `<name>.zip` beside it. Names are unique site-wide and say what the project builds (`cv-defect-inspector`, not `cv-project-1`).

You may create a `90-projects/_category_.json` you own. Do not edit `sidebars.ts`, an existing `_category_.json` or `learningPath.ts`. Ask in the progress file if wiring is needed.

## The project page

Frontmatter tags include `project`. That switches the gate to its project checks: an architecture board, the sections below, and a download link to a ZIP that exists.

Copy `templates/project.md`. The sections, in order:

1. `**In one line.**` What the reader will have built.
2. `:::note Added for this site` when the project is not from the source (most are not).
3. `## The problem statement`: background, users, current pain, scope, non-goals, constraints, success criteria with numbers, and a worked example end to end (one input, traced through the system by hand, to its output).
4. `## What you will learn`: the skills, then the table **Chapters this project uses** (chapter link, what the project uses from it).
5. `## Requirements`: functional and non-functional.
6. `## Architecture`: a board (not Mermaid) showing the components and data flow, then design decisions with the alternative you rejected and why.
7. `## Tech stack` with pinned versions, and `## Repository layout`.
8. `## How to install` and `## How to configure`: offline mode by default, then the real mode with environment variables.
9. `## Build it task by task`: each task has a goal, the full code of each file it adds or changes, the command to run, the output you saw, and a checkpoint test. Tasks build in order: after task N the project runs.
10. `## Testing`, `## Evaluation` (where quality is measurable), `## Observability`, `## Security and safety`, `## Deployment`, `## Cost and scaling`.
11. `## Failure modes and runbook`: symptoms, cause, fix, as a table.
12. `## Extensions`: three to five directions, each a sentence on what it adds.
13. `## Interview questions`: the 2-minute pitch, then concepts, system design, debugging, trade-offs, each with a model answer in block-form `<details>`.
14. `## Checklist`: "I can ..." capabilities.
15. `## Download`: the ZIP link, then the exact commands from a clean unzip to a passing test run, with the counts you saw.

Projects 1 and 2 may merge sections 10 and drop Observability and Deployment where they do not apply. Project 3 has them all.

## The repo inside the ZIP

- Python 3.12, `uv`, `pyproject.toml` with `uv.lock` committed. One `Makefile` with `install`, `test`, `lint`, `run`, `demo` (and `eval`, `docker-build` where relevant).
- **Runs offline by default.** No API key and no network for `make test` and `make demo`. LLM projects use a deterministic stand-in model (scripted answers or a tiny local model) behind the same interface as the real one. Real mode is switched on with an environment variable and `.env.example` lists every variable without values.
- **Data**: generated by a seeded script, or downloaded at run time from its source with the licence named in the README. Never bundle data whose licence forbids it.
- **Tests**: `pytest`, offline, under a minute for projects 1 and 2. Project 3 adds an evaluation with a golden set and a regression gate.
- **No comments in code**, including the shell blocks on the page (`.codex/AGENTS.md` rule 3). Explanation lives on the page, in the README, and in clear names.
- **No secrets**, no `.venv`, no `__pycache__`, no `.DS_Store` in the ZIP.
- A CI workflow file may be included in the ZIP, but the page never links to GitHub.

Build the ZIP from the folder:

```bash
cd static/examples/projects && rm -f <name>.zip && zip -r -X <name>.zip <name> -x '*/.venv/*' '*/__pycache__/*' '*/.pytest_cache/*' '*/.ruff_cache/*' '*.DS_Store' '*/.env'
```

## The page's code is the repo's code

A project page holds thousands of lines of code. Typed by hand, it drifts from the repo within a day. Generate the code blocks in **Build it task by task** from the repo files, with a small script in your job folder that reads each file and writes it into the page between markers, as the capstone did (`.lecture-import/capstone/`). Re-run the script after every change to the repo.

## Verify from a clean copy

```bash
T=$(mktemp -d) && cp static/examples/projects/<name>.zip $T && cd $T && unzip -q <name>.zip && cd <name>
uv sync --frozen && make test && make demo
```

The test count and demo output on the page come from this run. Then run the gate on the page and look at the architecture board. Report what was not run, for example a real-model path without a key, a GPU path, or a cloud deployment.

## Projects in a source-based course

- When the video or course builds a project, that project is the source's content: it follows the fidelity rules (`youtube-chapter.md`), in the source's order, with its code made to run.
- The topic projects in this guide are additions on top, marked `:::note Added for this site`.
- A course that is itself one big project (`docs/projects/*`) still gets a guided project for practice: a smaller build of the same idea that the reader does alone.

## Order of work for a topic's projects

1. Run the inventory with `--uncovered` for the topic. Read the topic's chapters' outcomes and **Check yourself** lists.
2. Draft three project ideas, each one line: what it builds, which chapters it uses, what the reader proves. Check that every chapter is covered. Write them in the progress file before building.
3. Build the repo for project 1 first, test it from a clean copy, then write the page from it. Then projects 2 and 3.
4. Add a line to each chapter's **Where to go next** pointing to the project that uses it.
5. Gate, link check, build, browser pass, report.
