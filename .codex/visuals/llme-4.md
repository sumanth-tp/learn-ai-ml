# Track B, agent F2: labs for docs/llm-engineering/02-inference-and-serving, chapters 06 to 09

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Numbers come from the chapters' own code (the Python blocks print the same values); where a lab
embeds an experiment's output it embeds the rounded printed values.

## ServingEngineChooserLab (chapter 06)

- Data: the feature matrix verified from each engine's documentation on 2026-10-02 (cells: yes, no, not established, depends on
  backend). Engines: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp, Ollama, Triton. Columns: hardware (one of NVIDIA, AMD, Apple
  silicon, CPU, Intel GPU), OpenAI-compatible API, structured output, multi-LoRA, speculative decoding, prefix reuse, tensor-parallel
  multi-GPU, plus a maintenance-mode flag (TGI).
- Controls: hardware select (default NVIDIA); six checkboxes for the needs (all ticked by default); one checkbox "leave out
  projects in maintenance mode" (ticked by default).
- Rule: an engine is out if a needed cell is "no" or it is in maintenance mode and the box is ticked; it fits if every needed
  cell is "yes"; otherwise it is unverified (some cell is not established or depends on the backend). The lab never ranks.
- Drawn: the matrix as a grid of cells with a symbol in each (tick, cross, question mark, tilde), the needed columns outlined,
  and the verdict beside each engine.
- Default result: NVIDIA, all six needs, maintenance engines left out gives fits 3 (vLLM, SGLang, TensorRT-LLM), unverified 3
  (llama.cpp, Ollama, Triton), out 1 (TGI). Chapter code prints `nvidia fleet: fits 3 ... unverified 3 ... out 1`.
- Table: engine, verdict, reason.

## PruningLab (chapter 07)

- Data (from the chapter's code, digits MLP 64-512-512-10, 540 test images, dense accuracy 0.9759): global L1 magnitude pruning at
  sparsity 0.5, 0.8, 0.9, 0.95, 0.98 with accuracy before and after 8 epochs of fine-tuning, and per-layer zero fractions; the 2:4
  pattern result 0.9778; structured pruning of half the first-layer units 0.9407 then 0.9796; timings of a 32 x 2048 by 2048 x 2048
  matmul (dense 0.39 ms, 90 per cent zeros in a dense tensor 0.47, 90 per cent zeros as CSR 14.75, half the rows removed 0.19; one
  run on a shared machine, so the chapter quotes ratios: zeros about 0.94 to 1.2 times dense, CSR about 20 to 45 times, half the rows
  about 0.4 to 0.6 times).
- Controls: sparsity select (dense, 0.5, 0.8, 0.9, 0.95, 0.98; default 0.9); method select (unstructured magnitude, 2:4, structured
  units; default unstructured).
- Drawn: left, a 16 x 16 block of seeded weights with surviving weights filled and pruned ones hollow (unstructured keeps the
  largest magnitudes overall; 2:4 keeps the top two of every four in a row; structured drops whole rows); right, accuracy
  against sparsity for "no fine-tune" and "fine-tuned" with the selected point marked; below, the four timing bars.
- Default result: 0.9 unstructured shows no fine-tune 0.7685 and fine-tuned 0.9759. Chapter code prints both.
- Table: sparsity, accuracy before, accuracy after, per-layer zeros.

## CacheRoutingLab (chapter 08)

- Three panels in one lab selected by a mode select (prompt cache cost, semantic cache threshold, cascade).
- Prompt cache mode: sliders for the number of requests that reuse a prefix (1 to 50, default 5), and a select for the write
  multiplier (1.25 for a 5-minute write, 2.0 for a 1-hour write); read multiplier fixed at 0.1; drawn as two bars (uncached cost
  against cached cost) in units of one uncached prefix. Default 5 requests: uncached 5.00, 5-minute cache 1.65.
- Semantic cache mode: select of the cost of a wrong answer (1, 2, 5, 10, 50 LLM calls; default 10) and a threshold slider on the
  seven measured thresholds 0.55 to 0.95; drawn as hit rate and wrong share of hits from the chapter's measured table, plus the
  resulting cost per request. Default threshold 0.80 at wrong cost 10: hit rate 0.328, cost 1.238.
- Cascade mode: slider for the confidence threshold over 0.2, 0.4, 0.5, 0.6, 0.7, 0.8; drawn as accuracy and cost against the
  large model alone. Default 0.5: escalated 0.288, accuracy 0.961, cost 5.33 against 0.945 and 15.00.
- Table: the values behind the current mode.

## CapacityPlannerLab (chapter 09)

- Model select: Llama 3.1 8B (8,030,261,248 parameters, 32 layers, 8 KV heads, head dim 128) and Llama 3.1 70B (70,553,706,496
  parameters, 80 layers, 8 KV heads, head dim 128), both from the Hub config files and a meta-device parameter count.
- GPU select (datasheet figures fetched 2026-10-02): L4 24 GB 300 GB/s; A100 80GB SXM 2,039 GB/s; H100 SXM 80 GB 3.35 TB/s;
  H100 NVL 94 GB 3.9 TB/s; dense bf16 TFLOPS 121, 312, 989.5, 835.5. Tensor-parallel select 1, 2, 4, 8.
- Sliders: QPS 1 to 100 (default 20), prompt tokens 100 to 8000 (default 1000), output tokens 50 to 1000 (default 300), TPOT SLO
  10 to 100 ms (default 40), bandwidth efficiency 0.3 to 0.9 (default 0.6), prefill MFU 0.2 to 0.6 (default 0.4), prefill share
  0.1 to 0.8 (default 0.3). Memory utilisation 0.90 and reserve 2 GB per GPU are fixed.
- Drawn: a stacked bar of the memory pool (weights, reserve, KV budget, unused), and two bars for replicas needed from decode
  (Little's law) and from prefill, the larger one binding.
- Default result: 8B on H100 SXM 80GB, tensor parallel 1: weights 16.1 GB, KV 131.1 kB per token, KV budget 53.9 GB, 316
  sequences by memory, step 31.7 ms, decode replicas 1, prefill replicas 3, GPUs 3, 0.139 GPU-hours per million output tokens.
- Table: every derived quantity.
