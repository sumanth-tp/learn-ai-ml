You are continuing the "senior AI engineer" curriculum on the Docusaurus site in /Users/sumanth.tp/Resources/ai-ml/learn-ai-ml.
You are Codex. Claude is the other agent and works in parallel. Priority 0 (P0) of your work, information retrieval (16 chapters) and
data management (16 of 18), is built and verified by Claude. P1 is now open for you.

READ FIRST, in this order
1. .codex/AGENTS.md (house rules; ignore the YouTube sections)
2. .codex/senior-ai-plan.md (sections 5, 6 and 11: sources, gates G1 to G10, visuals)
3. .codex/senior-ai-codex-brief.md (your pipeline, chapter shape, track layouts)
4. .codex/senior-ai-progress.md (live status, the ledger, Claude's verification of your P0 work)
5. .lecture-import/track-b/AGENT-PROMPT.md (lessons from Claude's authors; the venv and runner notes apply to you too)

YOUR TASKS, in this order
A. Finish data management. docs/mlops/data/99-practice/ has only its category file. Write two chapters:
   01-question-bank.md (merge dm-question-bank and dm-comprehensive-question-bank: keep every question and answer, de-duplicate,
   group by lecture, answers in <details>) and 02-midsem-solved.md (dm-midsem-2026: question text plus worked solutions; the paper's
   scan images are not reproduced, so redraw any diagram a question needs as an original board). Verify every numeric answer by computing
   it and report wrong ones as findings. Sources:
     cd .lecture-import && mkdir -p in-dm-extras && cp crawl/pages/dm-question-bank_question-bank.html \
       crawl/pages/dm-comprehensive-question-bank_question-bank.html crawl/pages/dm-midsem-2026_midsem-2026-solved.html in-dm-extras/
     ./venv/bin/python convert.py in-dm-extras converted-dm-extras extras
   Same pattern as docs/theory/ir/99-practice/, which you already did.
B. Computer vision, 17 chapters in docs/theory/cv/: the 15 lectures cv-s1 to cv-s16 as laid out in the brief (groups 01 to 05), plus
   99-practice (question bank from cv-question-bank and cv-comprehensive-question-bank, and cv-midsem-2026 solved). Convert with
   convert.py into your own dirs (in-cv, converted-cv, converted-cv-extras). The deep-learning side of vision already lives in
   docs/theory/dnn: link to it, do not repeat it. Verify the current detection and segmentation families live and record them in the ledger.
C. Time series, 5 chapters in docs/theory/timeseries/ (open sources, layout in the brief), then recommender systems, 4 chapters in
   docs/theory/recsys/ (builds on your ir chapter "recommendation-as-personalised-retrieval").
D. Cross-review of Claude's Machine Learning notes: read the first chapter of each of the four groups in docs/theory/ml/ (01-ml-foundations,
   02-supervised-learning, 03-ensembles-and-unsupervised-learning, 04-evaluation-and-practice), check them against gates G1 to G10 and
   against their sources, run their code, and write findings under "Requests and findings" in .codex/senior-ai-progress.md. Do this when
   tasks A to C are blocked or finished; do not edit those chapters, report instead.

STANDING RULES (from the user; they override anything older)
- Do not link to learning.bansal-ai.in anywhere. Copy the lecture content in and credit in plain text:
  Built from the course lecture "cv-s7-sift" (Lecture Library series).
- Every chapter is visual: board-style SVG infographics (not Mermaid) and interactive labs. At least one board per chapter, a lab for every
  concept that moves and for every widget the lecture had. Write each lab spec in .codex/visuals/<track>.md first. Defaults must reproduce a
  number the chapter's code prints.
- No code comments of any kind. British spelling. No em dashes as sentence connectors. No GitHub references.
- Never invent numbers, versions, benchmarks, prices or "company X does Y". Every named system or figure needs a source you opened, with
  the date, in the ledger. Fast-moving tooling: record the version you checked and ran.
- A chapter built from a lecture keeps the lecture text and its practice questions; correct any lecture claim that does not reproduce, in a
  labelled :::note (Claude's authors found several: say what the lecture claimed, what the code shows).

ENVIRONMENTS AND TOOLS
- .lecture-import/venv/bin/python converts and generates. Chapter code: .lecture-import/venv-llm/bin/python (torch CPU, scikit-learn, pandas,
  duckdb, opencv is NOT installed: pip install opencv-python-headless into venv-llm and list it in the progress file) or venv-ml (numpy, scipy,
  pandas, scikit-learn, matplotlib).
- Run code with .lecture-import/codetest/run_all.py <folder> <python>. It executes every block under "## Code you can run", no skipping.
  (Do not use run_ml.py or run_runnable.py: they skip blocks.) If a Hugging Face download stalls, rerun with HF_HUB_OFFLINE=1 once the model
  is cached and say so.
- Never run .lecture-import/regen.sh. Use your own track directories. Do not touch spec_drl_*, spec_nlp_*, spec_seml_*, spec_ml_*.

FILES YOU MAY AND MAY NOT EDIT
- You own: docs/theory/ir, docs/mlops/data, docs/theory/cv, docs/theory/timeseries, docs/theory/recsys, your scripts/infographics/*.py and
  static/img/<your dirs>, new lab files in src/components/viz/, .codex/visuals/<your tracks>.md, and your own rows in the progress file.
- You must NOT edit: src/** other than new lab files, sidebars.ts, docusaurus.config.ts, package.json, docs/interviews/**, learningPath.ts,
  any _category_.json you did not create, or the folders Claude's authors are writing right now: docs/llm-engineering, docs/governance,
  docs/senior, docs/agentic-frontier, docs/genai/rag-advanced, docs/mlops/platform, docs/mlops/distributed, docs/theory/ml.
- Never save a chapter that imports a lab or image that does not exist yet. It breaks the build for everyone. Create the lab first.
- Doc ids and slugs must be unique site-wide; category labels too (prefix generic ones, for example "CV foundations").

BUILD AND REPORT
- Build in isolation: DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex npx docusaurus build --out-dir .lecture-import/codex-build
  It must pass with no warnings. If it fails because of someone else's in-progress files, say so and verify by building a copy of the tree
  without those folders, as Claude did.
- Typecheck: npx tsc --noEmit must be clean.
- Do not commit or push.
- After each group, update .codex/senior-ai-progress.md: chapter log rows (words, code blocks run, boards, labs, gates), ledger rows, findings.
  Say plainly what you did not verify: site build, browser, screen reader, prices.
- When all of A to D are done, write "Codex: P1 DONE" in the progress file with the final counts, and list any lecture figures that did not
  reproduce.

DEFINITION OF DONE for a chapter: G1 to G10 in .codex/senior-ai-plan.md section 6. In short: frontmatter and build pass; every Python block runs and
the prose numbers equal the printed numbers; no bare {...} in prose and currency written as \$; own words except the lecture text you were told to
keep; not-from-the-lecture sections marked; at least one rendered and inspected board; every lab typechecks, is keyboard operable, has a data table,
works in dark mode and at 390 px, and its defaults reproduce a printed number.
