You are Codex, session "D", continuing the "senior AI engineer" curriculum on the Docusaurus site in /Users/sumanth.tp/Resources/ai-ml/learn-ai-ml.
The user has decided that Codex does all remaining chapter writing; Claude validates, improves, checks sources and enriches. Other Codex sessions run in parallel (A: causal, graph, speech, cross-review,
interview file; B: distributed ML 03/04/99; C: advanced RAG and agent frontier). Stay inside the ownership below.

READ FIRST
1. .codex/codex-next-prompt.md (standing rules, environments, build and report rules: all apply)
2. .codex/beginner-friendly-standard.md (NEW, mandatory: required chapter shape, writing rules, measured targets, the depth bar)
3. .lecture-import/track-b/AGENT-PROMPT.md (practice-question block format, no GitHub references, venv-llm contents, run_all.py, HF_HUB_OFFLINE tip)
4. .lecture-import/track-b/ASSIGNMENTS.md section "D2"
5. The distributed-ML foundations chapters you build on: docs/mlops/distributed/01-dist-foundations/ (especially 03-data-parallelism.md and 04-model-parallelism.md and their real 2-process CPU torch.distributed gloo
   pattern; report .lecture-import/track-b/report-D1a.md). Reuse the pattern, link to them, build on them (ZeRO and FSDP shard exactly the pieces the 16-bytes-per-parameter arithmetic counts).
6. Two chapters that show the depth you must meet: docs/llm-engineering/02-inference-and-serving/02-kv-cache-and-paged-attention.md and docs/llm-engineering/01-adapting-models/03-supervised-fine-tuning-with-lora.md.
7. .codex/senior-ai-progress.md (append, never rewrite)

YOUR TASK: four chapters in docs/llm-engineering/03-training-at-scale/ (category file exists; do not edit it), each with at least 2 boards and a working lab, following the beginner-friendly standard exactly.
   | File | id | slug | Lab |
   | 01-parallelism-strategies-for-llms.md | llme-parallelism | /llm-engineering/parallelism-strategies-for-llms | ParallelismMemoryLab |
   | 02-ddp-fsdp-and-zero.md | llme-fsdp-zero | /llm-engineering/ddp-fsdp-and-zero | ZeroStagesLab |
   | 03-mixed-precision-and-numerics.md | llme-precision | /llm-engineering/mixed-precision-and-numerics | PrecisionRangeLab |
   | 04-mixture-of-experts.md | llme-moe | /llm-engineering/mixture-of-experts | MoeRoutingLab |
Topics: data, tensor, pipeline, sequence and expert parallelism and 3D parallelism with per-GPU memory formulas checked against a real model config read from the Hub; DDP, ZeRO stages 1 to 3 and FSDP (bytes per parameter for weights,
gradients and optimiser state; a real 2-process CPU torch.distributed gloo DDP demo that runs; FSDP and DeepSpeed API snippets under "Production snippets (not run here)"); mixed precision (fp32, bf16, fp16, fp8, loss scaling, numerics
experiments in numpy and torch); mixture of experts (routing, load-balancing loss, capacity, expert parallelism; a numpy router; verify which MoE models are current from official model cards). Sources to check first: the Hugging Face
Ultra-Scale Playbook, PyTorch FSDP and DDP docs for the installed torch 2.14, the ZeRO and Megatron-LM papers, the Switch Transformer and Mixtral papers; cite per claim and record versions.

A Claude author began this track and was stopped. Treat what exists as a DRAFT to validate against the standard, not as finished: lab src/components/viz/ParallelismMemoryLab.tsx, script scripts/infographics/llme_5.py, spec .codex/visuals/llme-5.md, working
files .lecture-import/track-b/d2-code/ and d2-src/. No chapter file exists yet. Reuse what is sound, redo what is not.

FILES YOU OWN: docs/llm-engineering/03-training-at-scale, scripts/infographics/llme_5.py, static/img/llme (new files only, prefix with your chapter slugs), .codex/visuals/llme-5.md, the four lab files above, your own rows in the progress file.
DO NOT TOUCH: anything owned by sessions A, B or C, docs/llm-engineering/01-adapting-models and 02-inference-and-serving (read and link to them), docs/mlops, shared components, learningPath.ts, any _category_.json.

BUILD AND REPORT: build with DOCUSAURUS_GENERATED_FILES_DIR_NAME=.docusaurus-codex-d npx docusaurus build --out-dir .lecture-import/codex-build-d ; npx tsc --noEmit ; .lecture-import/codetest/run_all.py on your folder; never save a chapter that imports
a missing lab or image; no commit. Append your report to the progress file (rows, ledger, what you did not verify) and finish with "Codex D: DONE" and the counts.

QUALITY (added 2026-10-07, mandatory): read .codex/write-like-claude.md before writing and imitate the chapters it names. Every chapter needs a real-library experiment whose printed numbers you quote. Before you report any chapter done, run python3 .lecture-import/track-c/quality_gate.py <your folder>, fix every FAIL, and paste the final GATE line in your report. A chapter that fails the gate is not done.
