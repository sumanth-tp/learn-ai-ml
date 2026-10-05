You are Codex, session "B", continuing the "senior AI engineer" curriculum on the Docusaurus site in /Users/sumanth.tp/Resources/ai-ml/learn-ai-ml.
Another Codex session ("A") is working in parallel on different folders (causal, gnn, speech, cross-review, one interview file). Claude's authors are
also writing other folders. Stay strictly inside the ownership below.

READ FIRST
1. .codex/codex-next-prompt.md (standing rules, environments, build and report rules: all apply to you)
2. .lecture-import/track-b/AGENT-PROMPT.md (practice-question block format, no GitHub references, venv-llm, run_all.py, HF_HUB_OFFLINE tip)
3. .lecture-import/track-b/ASSIGNMENTS.md: the sections "D1a", "D1b" and "D1c" (ids, slugs, lab names, source stems)
4. The precedent for how these chapters are written: docs/mlops/distributed/01-dist-foundations/ (written by agent D1a), its report
   .lecture-import/track-b/report-D1a.md, and docs/mlops/distributed/02-dist-challenges/ (written by agent D1b). The bansal lectures are only about 300 words each, so
   the chapters are written directly (lecture text kept in "How it works", credited in plain text, no link to the bansal site; additions under :::note Beyond the lecture).
5. .codex/senior-ai-progress.md (latest state)

YOUR TASK: seven chapters of "Distributed machine learning" built from the bansal DML lectures. Converted lectures: .lecture-import/converted-dml/ . Question banks and the
solved mid-semester paper: .lecture-import/converted-dml-extras/ . Same shape, gates G1 to G10, boards and labs as before. Verify every numeric answer in the solved paper by
computing it and report wrong ones as findings. Real code: numpy simulations, CPU torch, and where the idea needs it a real 2-process torch.distributed (gloo) run like D1a's.

| Folder (under docs/mlops/distributed/) | File | id | slug | Source | Lab |
| --- | --- | --- | --- | --- | --- |
| 03-dist-learning | 01-distributed-linear-and-logistic-regression.md | dist-regression | /mlops/distributed/distributed-linear-and-logistic-regression | dml-s10-distributed-regression | StaleGradientLab |
| 03-dist-learning | 02-advanced-distributed-deep-learning.md | dist-deep-learning | /mlops/distributed/advanced-distributed-deep-learning | dml-s11-distributed-dl | GradientCompressionLab |
| 03-dist-learning | 03-advanced-sgd-techniques.md | dist-advanced-sgd | /mlops/distributed/advanced-sgd-techniques | dml-s12-advanced-sgd | LocalSgdLab |
| 04-dist-federated | 01-federated-learning.md | dist-federated | /mlops/distributed/federated-learning | dml-s13-14-federated | FedAvgLab |
| 04-dist-federated | 02-special-topics.md | dist-special-topics | /mlops/distributed/special-topics | dml-s15-special-topics | NonIidLab |
| 99-practice | 01-question-bank.md | dist-question-bank | /mlops/distributed/question-bank | dml-question-bank + dml-comprehensive-question-bank | (none) |
| 99-practice | 02-midsem-solved.md | dist-midsem | /mlops/distributed/midsem-solved | dml-midsem-2026 | (none) |

Practice chapters: keep every question and answer, de-duplicate, group by lecture, answers in <details> blocks (block form, see AGENT-PROMPT.md). The mid-semester paper's scan
images are not reproduced: redraw any diagram a question depends on as an original board and say it is a redraw.

FILES YOU OWN: docs/mlops/distributed/03-dist-learning, docs/mlops/distributed/04-dist-federated, docs/mlops/distributed/99-practice (the category files already exist; do not
edit them), new board scripts scripts/infographics/dist_4.py (the 03 group) and scripts/infographics/dist_3.py (04 and 99), output static/img/dist/, lab specs .codex/visuals/dist-4.md and
.codex/visuals/dist-3.md, and the seven lab files above in src/components/viz/ (new files only; check each name does not already exist before creating it).
DO NOT TOUCH: docs/mlops/distributed/01-dist-foundations and 02-dist-challenges (read them for style and link to them), docs/mlops/platform, docs/genai/rag-advanced,
docs/agentic-frontier, docs/llm-engineering/03-training-at-scale, docs/theory/causal|gnn|speech and the interview folder (session A and Claude own those), any shared component.

SHARED FILE RULES (two Codex sessions write the same progress file): never rewrite .codex/senior-ai-progress.md wholesale. Re-read it, then append your own dated section at the
end with a small edit, and only edit your own rows. Build in isolation with names that differ from session A:
  DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex-b npx docusaurus build --out-dir .lecture-import/codex-build-b
If that build fails because of other people's in-progress files, say so and verify with a copy of the tree that leaves those folders out, as Claude did.
Save a chapter only after the lab and board files it uses exist. Do not commit or push.

REPORT: after each group append to the progress file (chapter log rows, ledger rows, findings, what you did not verify). When all seven are done append "Codex B: DONE" with counts and
every source or lecture claim you had to correct.
