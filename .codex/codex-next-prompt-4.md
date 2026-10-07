You are Codex, session "C", continuing the "senior AI engineer" curriculum on the Docusaurus site in /Users/sumanth.tp/Resources/ai-ml/learn-ai-ml.
The user has decided that Codex does all remaining chapter writing; Claude validates, improves, checks sources and enriches. Other Codex sessions may run in parallel (A: causal, graph, speech,
cross-review, interview file; B: distributed ML 03/04/99). Stay inside the ownership below.

READ FIRST
1. .codex/codex-next-prompt.md (standing rules, environments, build and report rules: all apply)
2. .codex/beginner-friendly-standard.md (NEW, mandatory: the required chapter shape, writing rules, measured targets and the depth bar)
3. .lecture-import/track-b/AGENT-PROMPT.md (practice-question block format, no GitHub references, venv-llm contents, run_all.py, HF_HUB_OFFLINE tip)
4. .lecture-import/track-b/ASSIGNMENTS.md sections "B2" and "G" (ids, slugs, lab names, topics)
5. Two finished chapters that show the standard of depth you must meet or beat: docs/theory/ml/03-ensembles-and-unsupervised-learning/02-gradient-boosting-in-practice.md and
   docs/llm-engineering/01-adapting-models/03-supervised-fine-tuning-with-lora.md (real libraries, 100+ lines of runnable code, measured results, honest surprises).
6. .codex/senior-ai-progress.md (latest state and ownership; append, never rewrite)

YOUR TASKS, in this order. Eight chapters, with boards (at least 2 each) and working labs, following the beginner-friendly standard exactly.

A. Advanced RAG, docs/genai/rag-advanced/. Chapter 01 (GraphRAG) was written by a Claude author before it was stopped: treat it as a draft, validate it against the standard, run its code, check its sources,
   and improve it where it falls short. Its materials: lab src/components/viz/GraphRetrievalLab.tsx, boards static/img/rag-adv/graphrag-*.svg, script scripts/infographics/ragadv_1.py, spec .codex/visuals/ragadv-1.md,
   working files .lecture-import/track-b/b2/. Then write 02 to 04:
   | File | id | slug | Lab |
   | 02-long-context-vs-rag.md | rag-adv-long-context | /genai/rag-advanced/long-context-vs-rag | ContextVsRagLab |
   | 03-text-to-sql-and-structured-rag.md | rag-adv-text-to-sql | /genai/rag-advanced/text-to-sql-and-structured-rag | SchemaLinkingLab |
   | 04-contextual-retrieval-and-reranking.md | rag-adv-contextual | /genai/rag-advanced/contextual-retrieval-and-reranking | RerankLab |
   Link to docs/theory/ir (esp. neural-retrieval-and-reranking) and docs/senior/01-system-design-cases/01-enterprise-document-qa.md instead of repeating their experiments.
B. Agent frontier, docs/agentic-frontier/ (nothing exists yet; category file exists, do not edit it):
   | File | id | slug | Lab |
   | 01-context-engineering.md | afr-context-engineering | /agentic-frontier/context-engineering | ContextWindowLab |
   | 02-agent-interoperability-mcp-and-a2a.md | afr-interoperability | /agentic-frontier/agent-interoperability-mcp-and-a2a | ProtocolFlowLab |
   | 03-computer-use-and-browser-agents.md | afr-computer-use | /agentic-frontier/computer-use-and-browser-agents | ActionSpaceLab |
   | 04-voice-and-realtime-agents.md | afr-voice | /agentic-frontier/voice-and-realtime-agents | TurnTakingLab |
   | 05-automatic-prompt-optimisation-and-dspy.md | afr-dspy | /agentic-frontier/automatic-prompt-optimisation-and-dspy | PromptSearchLab |
   Read MCP and A2A from their current official specifications, DSPy from its current docs, benchmark names and results from the papers; cite per claim and record versions. Link to docs/agentic-ai, docs/mcp,
   docs/genai/langchain-advanced, docs/llm-evals and docs/senior/01-system-design-cases (coding assistant: context budget experiment; support agent: tool gateway and memory) rather than repeating them.

FILES YOU OWN: docs/genai/rag-advanced, docs/agentic-frontier, scripts/infographics/ragadv_1.py (extend) and afr_1.py (new), static/img/rag-adv and static/img/afr, .codex/visuals/ragadv-1.md and afr-1.md, the lab files named above
(new files; GraphRetrievalLab exists), your own rows in the progress file.
DO NOT TOUCH: anything owned by sessions A, B or D (see the progress file), docs/mlops, docs/theory/*, docs/llm-engineering, docs/senior, docs/governance, shared components, learningPath.ts, any _category_.json you did not create.

BUILD AND REPORT: build with DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex-c npx docusaurus build --out-dir .lecture-import/codex-build-c ; typecheck with npx tsc --noEmit ; run each chapter with
.lecture-import/codetest/run_all.py ; never save a chapter that imports a missing lab or image; no commit. Append your report to the progress file after each chapter (rows, ledger, what you did not verify) and
finish with "Codex C: DONE" and the counts.
