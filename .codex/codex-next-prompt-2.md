You are Codex, continuing the "senior AI engineer" curriculum on the Docusaurus site in /Users/sumanth.tp/Resources/ai-ml/learn-ai-ml.
Your earlier tracks (IR, DM, CV, time series, recommenders) are done and verified by Claude. This is your second batch.

READ FIRST
1. .codex/codex-next-prompt.md (your previous prompt: all its standing rules, environments, file-ownership and build rules still apply)
2. .codex/senior-ai-progress.md (latest state, Claude's verification notes, what changed since you finished)
3. .lecture-import/track-b/AGENT-PROMPT.md (lessons from Claude's authors: practice-question block format, no GitHub references, HF_HUB_OFFLINE tip,
   venv-llm contents, run_all.py)

YOUR TASKS, in this order
A. Track K3, ten chapters, nobody else owns them. Folders are new; create your own _category_.json files with unique labels (prefix generic ones).
   Use the same chapter shape, gates G1 to G10, boards and labs as before. Record sources in the ledger, record versions you ran.
   - docs/theory/causal/ (4 chapters). Source spine: Hernan and Robins "Causal Inference: What If" and "Causal Inference for the Brave and True"
     (verify both are current and openly readable). 1 Correlation, causation, potential outcomes and confounding (DAGs). 2 Randomised experiments and
     adjustment (backdoor criterion, propensity scores, matching, inverse-probability weighting). 3 Quasi-experiments (difference-in-differences, instrumental
     variables, regression discontinuity). 4 Causal ML in practice (uplift and heterogeneous effects, double machine learning). Synthetic data with a known
     ground-truth effect so every estimate can be checked against the truth. pip install into venv-llm only (for example dowhy or econml, if you use them; list
     them in the progress file).
   - docs/theory/gnn/ (3 chapters). Source spine: Stanford CS224W and the original GCN, GraphSAGE and GAT papers (verify). 1 Graphs and message passing.
     2 GCN, GraphSAGE and GAT from scratch in torch (CPU) and numpy on a small graph. 3 Graph tasks in production (node, link and graph prediction; fraud rings,
     recommendation; over-smoothing; neighbour sampling for scale). Prefer from-scratch torch plus networkx; torch_geometric only if it installs cleanly.
   - docs/theory/speech/ (3 chapters). 1 Audio basics: waveforms, spectrograms, mel scales (numpy/scipy, synthetic tones). 2 Speech recognition: CTC, attention
     models, Whisper; a real run of a small Whisper checkpoint on CPU on audio you synthesise or a short licensed clip (state the licence), WER computed by hand and
     with a library. 3 Speech synthesis and voice agents: neural vocoders, latency budgets, turn-taking; link to docs/agentic-frontier (being written by Claude; link only to
     a slug that exists when you save, check the folder first).
B. Independent cross-review of Claude's Track B chapters. Read the FIRST chapter of each group below, check G1 to G10, run its code, check its sources, and write
   findings under "Requests and findings" in .codex/senior-ai-progress.md. Do not edit those chapters.
   - docs/llm-engineering/01-adapting-models/01-prompt-retrieve-or-fine-tune.md and 03-supervised-fine-tuning-with-lora.md
   - docs/llm-engineering/02-inference-and-serving/02-kv-cache-and-paged-attention.md and 06-serving-engines.md (check the feature matrix cell by cell against the cited pages)
   - docs/governance/04-regulation-and-model-documentation.md. **Highest priority.** It states EU AI Act dates read from Regulation (EU) 2024/1689 and the
     "Digital Omnibus on AI" Regulation (EU) 2026/1744 (high-risk Annex III from 2 December 2027, Annex I from 2 August 2028, and other dates). Verify every date and
     article number against the official Official Journal texts yourself and report agreement or disagreement line by line.
   - docs/senior/01-system-design-cases/01-enterprise-document-qa.md and docs/senior/02-engineering-craft/03-estimation-and-planning.md
   - docs/mlops/distributed/01-dist-foundations/03-data-parallelism.md
C. Interview-bank additions. docs/interviews is normally Claude's, but you may CREATE ONE NEW FILE: docs/interviews/25-senior-ai-engineer-additions.md (slug
   /interviews/senior-ai-engineer-additions, unique id, sidebar_position 25), without editing any existing interview file. Write 40 practical questions with worked
   answers in <details> blocks, grouped by: classical ML judgement, retrieval and RAG, LLM adaptation and serving, evaluation and governance, system design, senior craft.
   Each answer must point to the site chapter that teaches it (use /docs/... links; check each slug exists). Work from the chapters that exist; do not invent facts, and run any code
   you include.

DO NOT TOUCH (Claude's authors are or will be writing these): docs/mlops/platform, docs/mlops/distributed (except reading 01-dist-foundations), docs/genai/rag-advanced,
docs/agentic-frontier, docs/llm-engineering/03-training-at-scale, and any file outside your folders listed above.

REPORT: after each part update .codex/senior-ai-progress.md (chapter log rows, ledger rows, findings). When A, B and C are done write "Codex: BATCH 2 DONE" with counts and
list every lecture or source claim you had to correct. Do not commit or push.

QUALITY (added 2026-10-07, mandatory): read .codex/write-like-claude.md before writing and imitate the chapters it names. Every chapter needs a real-library experiment whose printed numbers you quote. Before you report any chapter done, run python3 .lecture-import/track-c/quality_gate.py <your folder>, fix every FAIL, and paste the final GATE line in your report. A chapter that fails the gate is not done.
