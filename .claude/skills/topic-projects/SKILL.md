---
name: topic-projects
description: Plan and build projects for a site topic (guided, applied, portfolio), each with a runnable offline ZIP, a page generated from the tested repo, and links from every chapter it uses. Use when the user asks for projects for a topic, a track or "each topic".
---

# Projects for a topic

Read `.codex/AGENTS.md` section 10 and `.codex/guides/projects.md` fully, then read `docs/agentic-ai/31-project-3-autonomous-data-analyst.md` as the model page and `static/examples/projects/agentic-data-analyst/` as the model repo.

Start with `/usr/bin/python3 scripts/source-import/project_inventory.py --uncovered` to see the topic's chapters and which no project uses. Propose the projects (one line each, with the chapters each uses) before building, and confirm with the user when the scope is large: a portfolio project is several thousand lines of tested code.

Use `.codex/guides/templates/project.md` for the page. Verify every ZIP from a clean unzip in a temp folder (`uv sync --frozen`, `make test`, `make demo`) and quote the counts you saw. The gate's project mode checks the page shape.
