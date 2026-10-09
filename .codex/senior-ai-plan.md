# Senior AI engineer curriculum: master plan

Owner: Claude. Written 2026-10-01. Companion files: `senior-ai-progress.md` (current state, pending work, handoff). Rules for every chapter:
`.codex/AGENTS.md` (`.claude/AGENTS.md` is a symlink to it).

## 1. The bar

A senior AI engineer can, without hand-holding:

1. Choose and justify a model family for a problem, including when classical ML beats a neural network.
2. Build, evaluate, secure and deploy an LLM system (RAG, agents, tools) and say what it costs.
3. Adapt a model (prompt, retrieve, fine-tune, distil) and decide which, with evidence.
4. Serve it: latency, throughput, quantisation, GPU sizing, caching, scaling.
5. Run it: data pipelines, drift, experiments, incidents, governance.
6. Design a whole ML system on a whiteboard and defend the trade-offs.
7. Lead: write the design doc, estimate the work, say no with numbers, grow other engineers.

## 2. What already exists (corrected audit)

My first keyword audit under-counted. These **are covered** and must not be rewritten:

| Topic | Where |
| --- | --- |
| LoRA, QLoRA, PEFT, adapters, merging | `theory/dnn/91` (5.2k words) |
| Fine-tuning transformers | `theory/dnn/90` |
| RLHF, PPO, DPO, reward models | `theory/dnn/95` (6.6k words) |
| FlashAttention, KV cache (concept) | `theory/dnn/92`, `84` |
| Scaling laws, tokenisation, pre-training | `theory/dnn/93`, `86`, `87` to `89b` |
| Fairness, responsible ML | `theory/seml/05-responsible-ml` |
| Reranking, hybrid search (applied) | `projects/enterprise-rag/*`, `llm-evals/20` |
| Caching, advanced retrievers (LangChain) | `genai/langchain-advanced/09`, `10` |
| LLM serving and LLMOps (overview) | `scaler/generative-ai/advanced-systems/03` |
| Drift and monitoring (applied) | `llm-evals/22`, `code/11.fast-api-in-depth/project-03` |

New chapters link to these instead of repeating them.

## 3. Real gaps, grouped into tracks

Source column: **B** = the bansal-ai lecture is the spine (already in the pipeline); **W** = best open material
chosen per topic (section 5). Counts are chapters.

| ID | Track | Folder | Chapters | Source | Owner | Priority |
| --- | --- | --- | ---: | --- | --- | --- |
| A | Classical machine learning | `docs/theory/ml/` | 16 (11 lectures, 4 authored, 1 question bank) | B (ML, 11 lectures) + W | Claude | P0 |
| B1 | Information retrieval | `docs/theory/ir/` | 16 (14 lectures, question bank, mid-semester solved) | B (IR) | **Codex** | P0 |
| B2 | Advanced RAG add-ons (GraphRAG, long-context vs RAG, text-to-SQL, contextual retrieval) | `docs/genai/rag-advanced/` | 4 | W | Claude | P0 |
| C1 | Data management and data observability | `docs/mlops/data/` | 18 (16 lectures + 2 practice) | B (DM) | **Codex** | P0 |
| C2 | Platform ops: A/B testing, Kubernetes for ML, infrastructure as code, cloud ML platforms, batch inference | `docs/mlops/platform/` | 5 | W | Claude | P1 |
| E | Adapting models, hands-on (SFT with TRL, chat templates, DPO/ORPO, distillation, synthetic data, embedding and reranker tuning, tune vs RAG vs prompt) | `docs/llm-engineering/adapting-models/` | 7 | W | Claude | P0 |
| F | Inference and serving (prefill/decode, KV cache and paged attention, continuous batching, quantisation formats, speculative decoding, vLLM/TGI/TensorRT-LLM, ONNX and compression, semantic caching and cost, GPU sizing) | `docs/llm-engineering/inference-and-serving/` | 9 | W | Claude | P0 |
| D1 | Distributed ML | `docs/mlops/distributed/` | 14 (12 lectures + 2 practice) | B (DML) | Claude (or Codex when free) | P1 |
| D2 | Training LLMs at scale (DDP, FSDP/ZeRO, parallelism, mixed precision, MoE) | `docs/llm-engineering/training-at-scale/` | 4 | W | Claude | P1 |
| G | Agent frontier (context engineering, A2A, computer-use agents, voice agents, DSPy) | `docs/agentic-frontier/` | 5 | W | Claude | P1 |
| H | Computer vision | `docs/theory/cv/` | 17 (15 lectures + 2 practice) | B (CV) | **Codex** | P1 |
| K1 | Time series | `docs/theory/timeseries/` | 5 | W | **Codex** | P1 |
| K2 | Recommender systems (builds on IR session 14) | `docs/theory/recsys/` | 4 | B + W | **Codex** | P1 |
| K3 | Causal inference, graph ML, speech and audio | `docs/theory/causal/`, `gnn/`, `speech/` | 10 | W | next free agent | P2 |
| L | Safety and governance (LLM red-teaming, fairness testing in practice, privacy and PII, regulation and model cards, AI incident response) | `docs/governance/` | 5 | W | Claude | P0 |
| M | Senior craft (six ML system-design case studies, build vs buy, cost and ROI, design docs, technical leadership, postmortems) | `docs/senior/` | 12 | W | Claude | P0 |
| N | Optional bansal subjects: Unsupervised DL (16), Video analysis (15), Maths for ML (16), Cyber-security ML (14) | later | 61 | B | decide later | P2 |
| O | Integration and verification | `learningPath.ts`, sidebar, Explore, interview bank | n/a | n/a | Claude builds, Codex verifies | each phase |

Totals: P0 = 86 chapters, P1 = 54, P2 = 10 (150 in all), plus optional track N. Every chapter also carries
at least one infographic board and, where the idea moves, an interactive lab (section 11).
Honest scale: the NLP, DRL and SEML imports (43 notes) took a long session. This is over three times that.
So it ships in phases, each phase leaving the site in a working, built state.

## 4. Split

Principle: Codex gets the tracks that come from a bansal spine through the existing pipeline, or that are
self-contained and checkable by running code. Claude keeps LLM-era topics, senior craft and everything that
touches shared files.

| Codex (34 P0 chapters, 23 P1) | Claude (rest) |
| --- | --- |
| B1 IR (14), C1 DM (16), H CV (15), K1 time series (5), K2 recsys (4), plus independent verification | A (15), B2 (4), C2 (5), E (7), F (9), D2 (4), G (5), L (5), M (12), D1 (12), integration |

Work-stealing: whoever finishes a track first takes the next unowned P1/P2 track and writes their name in
`senior-ai-progress.md`. Claude can fan out sub-agents inside its share (one per chapter group).

**Cross-review:** every track's first three chapters get read by the other agent before the rest are written.
Reviewer reports in the progress file; author fixes. This catches template drift early.

**File ownership (no overlaps):**

- Each track owner owns its folder, its `_category_.json` files and its generator/spec files.
- Only Claude edits: `src/data/learningPath.ts`, `src/lib/docsIndex.ts`, `sidebars.ts`,
  `docusaurus.config.ts`, `src/**`, `package.json`, and the interview bank under `docs/interviews/`.
- Codex needs a new stage unit or section label? Ask in the progress file; Claude wires it.

## 5. Picking the best content online

For every W topic, before writing:

1. Shortlist 3 to 5 candidate sources. Prefer, in this order: the primary paper or official documentation;
   an openly readable university course or textbook; a respected practitioner write-up with working code.
2. Check recency (this is October 2026; anything about LLM tooling older than about 12 months must be
   re-checked against current docs and release notes).
3. Check the licence allows summarising (all notes are in our own words; code is re-run, not pasted).
4. Pick one **primary** and one or two **secondary**. Record URL, why chosen, date checked and licence in the
   source ledger (`senior-ai-progress.md`). Never write a W chapter without a ledger row.
5. Link the sources in "Go deeper" at the end of the chapter.

Starting candidates, **to be verified live at the start of each track** (not yet checked):

| Topic | Likely primary | Likely secondary |
| --- | --- | --- |
| Classical ML depth | scikit-learn user guide | ISLP; Google ML crash course |
| Fine-tuning workflow | Hugging Face TRL and PEFT docs | Raschka's LLM notebooks |
| Preference tuning | DPO and ORPO papers | TRL trainer docs |
| Inference engineering | vLLM docs and PagedAttention paper | Hugging Face LLM inference guide |
| Quantisation | AWQ, GPTQ, GGUF project docs | bitsandbytes docs |
| Training at scale | Hugging Face Ultra-Scale Playbook | PyTorch FSDP docs; ZeRO paper |
| Agent frontier | Anthropic engineering posts on agents and context | A2A spec; DSPy docs |
| Time series | Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* | Darts or sktime docs |
| Recommenders | Google recommendation course | Eugene Yan's recsys write-ups |
| Causal inference | Hernan and Robins, *What If* | *Causal Inference for the Brave and True* |
| Governance | NIST AI RMF; EU AI Act text | Model Cards paper |
| System design cases | public engineering write-ups from the companies concerned | Chip Huyen's blog |

## 6. Chapter template and quality gates

Template: the agreed shape from the earlier bansal imports (see `lecture-import-progress` memory, and
`docs/theory/seml/01-systems-thinking/01-what-changes-with-ml.md` as the exemplar): one-line point, plain
words, diagram, how it works, a real system, runnable Python, design guidance, the 2026 industry view,
practice questions, go deeper with source links, a closing capability checklist. Infographic boards sit with the
plain-words section; labs sit beside the code.

A chapter is done only when all of these hold:

| # | Gate | How it is checked |
| --- | --- | --- |
| G1 | Frontmatter matches repo convention; `slug` clean; position set | `npm run build` |
| G2 | Every Python block runs on CPU, or is clearly labelled GPU-only and not claimed as run | `.lecture-import/codetest/run_runnable.py`, and read the output, not the exit code |
| G3 | No bare `{...}` in prose; currency written `\$`; internal links start `/docs/` | the MDX sweep in AGENTS section 7 |
| G4 | Own words; no transcript or doc text pasted; source line and "Go deeper" present | cross-review |
| G5 | Anything that is not from the source is labelled `:::note Not from the source` | cross-review |
| G6 | British spelling; no em dashes as sentence connectors; no GitHub references; no code comments | grep |
| G7 | Diagrams: original Mermaid, or board SVGs only when the user asks (`infographics-means-images` memory) | review |
| G8 | Covers the capability checklist honestly; no invented numbers or versions | cross-review |
| G9 | At least one infographic board per chapter; every board fact matches the chapter and its code output; SVG rendered and inspected, no clipped or overlapping text | render to PNG and look |
| G10 | Labs: typecheck passes, keyboard operable, table view, dark mode, no overflow at 390 px, defaults reproduce a number the chapter prints | `npm run typecheck`, browser check |

**Builds do not collide.** Codex builds with
`DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex npx docusaurus build --out-dir .lecture-import/codex-build`.
Claude uses plain `npm run build`.

## 7. Phases

| Phase | Content | Exit criterion |
| --- | --- | --- |
| 0 | Plan, brief, ledger; venv and crawl (done); stage scaffolding in `learningPath.ts`; copy that says "eight stages" derived from data | Site builds; stages ready for new folders |
| 1 | Claude: A, then E, then F. Codex: B1, then C1 | P0 data and ML core live; cross-review done |
| 2 | Claude: M, L, B2, C2. Codex: H, K1, K2 | Senior craft and governance live |
| 3 | Claude: D1, D2, G. Next free agent: K3 | P1 complete |
| 4 | Decide on N; interview-bank questions for every new topic; cheatsheets; final cross-links; full browser verification by Codex | P0 to P2 complete |

Each phase ends with: build passes with no warnings, route diff shows only additions, browser checks on new
pages (light, dark, 390 px), and a `project` memory update.

## 8. Stage map after the work (proposal)

Ten main stages, up from eight:
1 Foundations, 2 **Machine learning** (new), 3 Deep learning and vision, 4 Language and retrieval,
5 Building with LLMs, 6 **LLM engineering** (new), 7 Agents and MCP, 8 Evaluate, secure and ship,
9 Real systems and senior craft, 10 Interview-ready.
Elective lanes: Reinforcement learning, **Specialised ML** (time series, recommenders, causal, graph, speech,
and track N if chosen), Reference shelf.

## 9. Risks

- **Thin source, thick note.** bansal ML gives about 700 words per lecture and no code. The notes must be
  authored on top, so this is writing work, not conversion work. Budget accordingly.
- **Staleness.** LLM tooling moves monthly. Every W chapter states the library version it was run with.
- **Template drift between two authors.** Mitigated by the cross-review after three chapters.
- **GPU-only topics** (quantisation, vLLM, FSDP). Teach with measured numbers from small CPU-runnable models
  where honest, and label anything that needs a GPU as not run here.
- **Source licensing and provenance.** Open-source material is written in our own words with links. Bansal lecture
  content is copied across as agreed, credited in plain text, with no links to the bansal site. No extracted frames.

## 10. Decisions (confirmed by the user, 2026-10-01)

1. **Scope:** P0 now (82 chapters), P1 after P0 is built and reviewed.
2. **Sources:** bansal for ML, IR, DM, CV and DML; open material for everything else.
3. **Stages:** add Machine learning and LLM engineering, giving 10 main stages. Stages with no notes yet are
   skipped in the sidebar and shown as "Coming soon" on the path page.
4. **Split:** Codex takes B1 and C1 now (34 chapters, with their boards and labs) and H, K1, K2 in P1; Claude takes the rest.

## 11. Visuals (user requirement, 2026-10-01)

"I need infographics and interactive visualisations to understand the topics." So every chapter is a visual one.

- **Infographics** are board-style SVG images (not Mermaid): original redraws in the existing whiteboard style,
  from the kit in `scripts/infographics/board.py`, one definition script per track, output in
  `static/img/<track>/`, placed with `<Infographic>`. At least one per chapter, two where the chapter holds two
  distinct ideas. Projected volume: about 170 boards for the 150 chapters.
- **Interactive labs** are React components on `VizPanel` in `src/components/viz/`. One for every concept that
  moves, and one for every interactive widget the original lecture had, so a reader never needs the source.
  Projected volume: about 70 labs.
- **Lab design.** Per lab: name, controls and ranges, defaults, what is drawn and the expected numbers
  (taken from the chapter's own verified code); see `.codex/AGENTS.md` section 6.
- **Who draws.** The track owner draws its own boards and builds its own labs. Claude fans these out to
  sub-agents per chapter group. Codex does its own for B1 and C1.
- **Shared files stay read-only**: `board.py`, `VizPanel*`, `palette.ts`, `CourseLab*`, `Infographic/`.
- Track A (classical ML) lab list so far: bias-variance fit, scaling and outliers, gradient-descent
  learning-rate, decision boundary and ROC threshold, entropy and Gini impurity calculator, k-NN and curse of
  dimensionality, SVM margin and kernel, Bayes' rule update, bagging vs boosting, k-means steps, PCA,
  confusion matrix and calibration, class imbalance, SHAP waterfall.
