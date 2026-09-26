# Project memory: practical interview revamp

## User contract

The interview section is organised by topic across AI, ML, AI/ML, agentic AI, RAG, LLM evaluation and LLM/agent QA engineering. Each substantive new topic has at least 40 primary questions. Answers need practical scenarios, explanations, answered cross-questions, code, differences, visualisations and a comprehensive final summary. User emphasised real interview evidence and real production examples. No guarantee of 99% interview coverage is made.

## Delivered structure

- `docs/interviews/01-python-data.md` through `10-applied-modelling.md`: 10 banks with exactly 40 unique sequential question IDs each, 400 total. Every question has answered cross-questions. Each bank has Mermaid diagrams, a comparison table and a final summary mapped to all 40 IDs.
- `11-coding-labs.md`: eight downloadable, runnable labs and explicit implementation limits.
- `12-mock-interviews.md`: six mixed-topic timed rounds, answer expectations, follow-ups, rubric and study plan.
- `98-sources.md`: dated interview evidence S1–S9 and creator material V1–V2, with confidence and access limitations.
- `99-tools-versions.md`: observed release metadata separate from the tested environment, migration differences, a pipeline example and reproducibility manifest.
- Earlier 10 x 100 foundation entries and their URLs preserved, ordered after new material, with scope notices and final summaries. Targeted fixes: pandas 3 Copy-on-Write/ffill/bfill/private-attribute examples, sklearn RMSE API, Python mutable-default fix example.
- Category URL `/docs/category/interview` retained through explicit generated-index slug because navbar, footer, homepage and keyboard shortcut link to it. Start-here document remains `/docs/interviews`.
- New architecture doc has explicit ID `practical-system-design` to avoid Docusaurus stripping `09-` and colliding with the old bank.

## Evidence discipline

Research reviewed 26 September 2026. Distinguish reported tasks, reported themes expanded into original scenarios, and practice extensions. Answers, numerical workloads, cross-questions, diagrams and labs are original teaching material; public interview reports are not employer-verified.

YouTube subtitles returned HTTP 429; youtube-transcript-api returned IpBlocked. Creator written question lists, description chapters and author notes were used and the limitation is disclosed. Do not claim videos were watched/transcribed or invent their unseen questions. Technical mechanisms use primary documentation and papers.

## Runnable artefacts

`static/examples/interviews`: retrieval, projected multihead attention, bounded async workers, SQLite idempotent local actions, point-in-time SQL features, topological sort, numerical gradient/convolution references, paired evaluation gate. Includes unittest suite, README, tested requirements, package metadata JSON and a ZIP matching these files.

Test environment: `/tmp/interview-labs-venv`, Python 3.12.8, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1. Exact transitive pins in requirements. No paid/model/network calls or GPU needed to run tests. Observed frameworks in the version table were not installed/tested together.

Limits are explicit: lexical evidence retrieval is not a complete generative RAG service; simulated SQLite effects are not exactly-once remote actions; small synthetic evaluation fixtures do not establish production statistical power.

## Validation

- 32 meaningful lab tests passed, covering masking/reference parity, cancellation/concurrency, crash boundaries/restart, joins, graph cycles, numerical parity, invalid/missing scores and critical gates.
- 388 Python blocks in the 10 new banks and version notebook passed as isolated processes in the tested environment.
- All 3 inline SQL queries passed SQLite fixtures covering late data, missing features, NULL anti-joins and deterministic ties.
- The 3 specifically repaired legacy Python/pandas/sklearn examples were executed successfully. The other 1,000 older entries were not comprehensively revalidated.
- All 10 new banks have exactly 40 primary questions; all Markdown pages end in a simple-point summary; all earlier 1,000 entries remain.
- All new JSON examples parsed; archive integrity and source parity passed; `git diff --check` passed.
- `npm run typecheck` passed.
- Production build passed with no broken links/anchors. One third-party webpack warning remains about dynamic require in `vscode-languageserver-types`; dependencies were not changed.
- Playwright checked all 10 rendered banks: 40 questions each, final summary, 15 rendered topic diagrams, 26 comparison tables, no diagram syntax errors/page errors. Expand, zoom to 125%, and Escape close passed for each bank. Mobile lab page had 8 lab anchors and no horizontal page overflow; ZIP download and preserved category landing passed.

Temporary review artefacts: `/tmp/interview-snippet-results.json`, `/tmp/interview-ui-results.json`, `/tmp/interview-desktop.png`, `/tmp/interview-mobile.png`, build logs in `/tmp/interview-build*.log`. Browser used cached playwright-core and Chromium headless shell; no repository dependencies added.

No subagents, commit, push or deployment. Work remains local for review.
