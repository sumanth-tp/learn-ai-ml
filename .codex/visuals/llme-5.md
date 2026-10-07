# Training at scale: lab and board specs (docs/llm-engineering/03-training-at-scale)

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, native keyboard-operable controls and no
randomness at render time. Each lab mirrors the arithmetic of a Python block that the chapter runs.

## ParallelismMemoryLab (chapter 01)

- Mirrors block 3 (`c1_3.py`). Memory unit is 1 GB = 1e9 bytes. Vocabulary 128,256, as in the real Llama 3.1 `config.json` files
  (the paper's Table 3 rounds it to 128,000, a difference of 8.4 million parameters in 405.85 billion).
- Presets: Llama 3 8B (32 layers, h 4096, ffn 14336, 32 heads, 8 KV heads), 70B (80, 8192, 28672, 64, 8), 405B (126, 16384, 53248, 128, 8).
- Controls: model, tensor t (1 to 16), pipeline p (1 to 32), context c (1 to 16), data d (1 to 512), ZeRO stage (0 to 3), tokens per sequence
  (2,048 to 131,072), micro-batch (1, 2, 4), activation mode (none, tp, tp+sp, tp+sp+selective, full), GPU memory (16 to 192 GB, default 80; an
  input, not a specification).
- Default: 405B, t 8, p 16, c 1, d 128, ZeRO 2, 8,192 tokens, micro-batch 1, tp+sp+selective, 80 GB: weights 6.34, gradients 0.05, optimiser
  0.30, activations 71.87, total 78.56 GB, 16,384 GPUs.
- Experiments in the chapter: ZeRO 0 gives 122.61 GB; activations tp+sp gives 755.02 GB; 131,072 tokens with c 16 and d 8 gives 83.76 GB (ZeRO 3: 79.01 GB).

## ZeroStagesLab (chapter 02)

- Mirrors block 2 (`c2_1.py`). Bytes per parameter from a recipe (2 + 2 + 12 or 4 + 4 + 8); stage 1 divides the optimiser by N, stage 2 also the
  gradients, stage 3 also the weights. Traffic in model sizes: 2, 2, 2, 3.
- Controls: model (7.5B, 8.03B, 70.55B, 405.85B), N (1 to 1,024), highlighted stage, recipe, GPU memory (16 to 192 GB).
- Drawn: four stacked bars (weights, gradients, optimiser) with the card limit; readout of the smallest N that fits for the highlighted stage.
- Default (7.5B, N 64, bf16 recipe): 120.00, 31.41, 16.64 and 1.88 GB, printed by block 2 as 120.0, 31.4, 16.6 and 1.9.
- Experiments: N 8 gives 120.0, 41.3, 28.1 and 15.0 GB; Llama 3.1 70B at N 64 gives 1,128.9, 295.4, 156.5 and 17.6 GB with stages 0 to 2 never
  fitting 80 GB; the fp32 recipe at the defaults gives 120.0, 60.9, 31.4 and 1.9 GB.

## PrecisionRangeLab (chapter 03)

- Math in `precisionMath.ts`: `roundTo(x, format)` rounds to nearest even on the format's grid with subnormals; fp32 (8, 23), bf16 (8, 7),
  fp16 (5, 10), e4m3 (4, 3, saturating as torch 2.14.1 does on the CPU), e5m2 (5, 2). Checked against `torch` casts on 2,170 values
  (34 edge values and 400 log-uniform random values, five formats): 0 differences.
- Controls: value (text, default 2e-8), preset select, loss scale exponent (0 to 24).
- Drawn: log-axis range bars (subnormal range and normal range per format) with markers for the value and the scaled value; table of stored value,
  value after unscaling, relative error and status.
- Default (2e-8, scale 2^0): fp16 underflows to zero. With scale 2^10: fp16 stores 2.0504e-05 and returns 2.0023e-08 (block 2, step 3).

## MoeRoutingLab (chapter 04)

- Math in `moeMath.ts`: `mulberry32` generator, Box-Muller Gaussians, per-expert bias drawn first, logits = noise + skew x bias, softmax for P,
  top-k by logit for slots, f = slots / (tokens x k), balance = N x sum(f x P), capacity = ceil(cf x tokens x k / N), dropped = 1 - kept / slots,
  used = kept / (capacity x N). Mirrors `simulate` in block 1 (`c4_1.py`): six settings agree to 2.2e-16.
- Controls: experts (4 to 64), top-k (1 to 4), tokens (256 to 2,048), capacity factor (1 to 3), router skew (0 to 1.5).
- Default (16 experts, top 2, 1,024 tokens, capacity factor 1.25, skew 0.5): balance 1.5836, busiest 3.72, dropped 28.5%, used 57.2%, capacity 160.
- Experiments: skew 0 gives 1.0019, 1.10, 0.0%, 80.0%; capacity factor 3 gives 4.5% dropped and 31.8% used; 64 experts gives 1.7509, 7.84, 33.3%, 53.4%.

## Boards (scripts/infographics/llme_5.py, output static/img/llme)

Chapter 01: five-axes, worked-70b, llama3-layout. Chapter 02: ddp-fsdp-and-zero-memory, -toy-step, -stage3-timeline. Chapter 03:
mixed-precision-and-numerics-formats, -worked, -step. Chapter 04: mixture-of-experts-layer, -worked, -models, -all-to-all. Every number on a board
is printed by the chapter's code or read from the model card named in the chapter.
