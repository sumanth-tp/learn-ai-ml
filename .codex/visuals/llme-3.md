# Track B, agent F1: labs for docs/llm-engineering/02-inference-and-serving (chapters 01 to 05)

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, native range and select inputs,
no external dependencies and no randomness at render time except seeded generators that are mirrored in Python.

## RooflineLab (chapter 01, why decoding is memory-bound)

- Controls: peak compute in TFLOP/s (range 100 to 2000, step 10, default 989.5, labelled dense bf16); memory bandwidth in TB/s
  (range 0.5 to 8, step 0.05, default 3.35); parameters in billions (range 1 to 70, step 0.01, default 8.03); weight format
  (select bf16 = 2 bytes, int8 = 1 byte, int4 = 0.5 bytes; default bf16); decode batch (select 1, 8, 32, 64, 128, 295, 512;
  default 1); prompt tokens for the prefill marker (select 16, 128, 512, 2048; default 512).
- Model: decode FLOPs per step 2 P B, weight traffic P b bytes, intensity 2 B / b, step time max(FLOPs / peak, bytes / bandwidth),
  tokens per second B / step time, compute utilisation (FLOPs / peak) / step time. Prefill uses the same formula with B replaced
  by the prompt length.
- Drawn: log-log roofline (x: FLOP per byte 0.5 to 4096, y: attainable TFLOP/s), the sloped memory roof and flat compute roof,
  the ridge point, a marker for the decode point and one for the prefill point, both labelled with step time.
- Default result: Llama 3.1 8B (8.03 B parameters), bf16, H100 SXM numbers, batch 1: ridge 295.4 FLOP per byte, step 4.79 ms,
  209 tokens per second, compute utilisation 0.003, memory-bound. Batch 295: 61,533 tokens per second, utilisation 0.999.
  Batch 512: step 8.31 ms, compute-bound. Prefill of 512 tokens: 8.31 ms. Chapter code block 1 prints all of these.
- Table: decode at batch 1, 8, 32, 64, 128, 295, 512 for the current settings.
- Keyboard: native range and select inputs.

## KvCacheLab (chapter 02, KV cache and PagedAttention)

- Models (from the real config.json files the chapter code reads): gpt2 (12 layers, 12 KV heads, head dim 64), falcon-7b (32, 1, 64, MQA),
  Mistral-7B-v0.1 (32, 8, 128), Llama 3.1 8B (32, 8, 128), Qwen2.5-7B-Instruct (28, 4, 128), Llama 3.1 70B (80, 8, 128),
  SmolLM2-135M-Instruct (30, 3, 64), DeepSeek-V2-Lite (MLA: 27 layers, latent 512 plus decoupled key 64).
- Controls: model (select, default Llama 3.1 8B); cache dtype (select bf16 = 2 bytes, 8-bit = 1 byte, 4-bit = 0.5 bytes; default bf16);
  context length per sequence (select 640, 2048, 4096, 8192, 32768, 131072; default 640); cache budget in GB (range 1 to 80, step 1,
  default 40); reserved length for the contiguous allocator (select 2048, 4096, 8192, 32768; default 4096); block size in tokens
  (select 1, 8, 16, 32, 128; default 16).
- Model: elements per token = 2 x layers x KV heads x head dim, or (latent + decoupled key) x layers for MLA; bytes per token = elements x dtype
  bytes; sequences that fit contiguous = floor(budget / (reserved length x bytes per token)); paged = floor(budget / (ceil(context / block) x block
  x bytes per token)); contiguous waste = 1 - context / reserved length.
- Drawn: left, horizontal bars of KiB per token for every model at the chosen dtype with the selected model highlighted; right, two bars of
  sequences that fit (contiguous against paged) and, below, one sequence's memory as a proportional strip (used, rounding waste, reserved but unused).
- Default result: Llama 3.1 8B, bf16, 131,072 bytes (128 KiB) per token; at 640 tokens each and 40 GB the paged allocator fits 476 sequences and the
  4,096-token reservation fits 74. Chapter block 1 prints both; block 3 prints 74 for the reservation under its own length mix.
- Table: for each model, elements per token, KiB per token and GB for one sequence at the chosen context and dtype.
- Keyboard: native range and select inputs.

## BatchingLab (chapter 03, continuous batching and scheduling)

- Data: the seeded request trace of the chapter, generated in TypeScript with the same mulberry32 generator and formulas as the Python
  block (600 requests, seed 7, exponential inter-arrival gaps at the chosen rate, lognormal prompt lengths clipped to 8..2048 and output lengths
  clipped to 8..1024). Step time = max(4.79 ms, 0.01623 ms x tokens in the step) + 3.913e-5 ms x cached tokens read; the three constants are the
  chapter 1 roofline figures for Llama 3.1 8B on the quoted H100 numbers and are illustrative, not measured.
- Controls: policy (select: static batching; continuous; continuous with chunked prefill, 512-token budget; continuous with chunked prefill,
  256-token budget; continuous with shortest prompt first; default continuous); arrival rate in requests per second (range 5 to 70, step 5,
  default 30); maximum batch (select 16, 32, 64; default 64).
- Drawn: left, a waterfall of the first 40 requests of the trace, one thin bar each from arrival to finish, the dark part up to the first
  token and the light part after it; right, tokens per second for all five policies at the same settings with the chosen one highlighted.
- Default result (continuous, 30 requests per second, batch 64): 4,324 tokens per second, mean latency 0.91 s, p99 latency 4.49 s, mean
  time to first token 9.6 ms, p99 24.0 ms, p99 inter-token gap 10.88 ms. Static batching with batch 16 at the same rate: 860 tokens per second,
  mean latency 53.31 s. At 58 requests per second with batch 64: first come first served has p99 inter-token gap 14.95 ms, chunked 512 gives 9.38
  and chunked 256 gives 5.89. The chapter's block 1 prints all of these and TypeScript reproduces Python on every printed digit.
- Table: for every policy, tokens per second, mean latency, p99 latency, mean and p99 time to first token, p99 inter-token gap.
- Keyboard: native range and select inputs.

## QuantisationLab (chapter 04, quantisation for inference)

- Data: one real row of SmolLM2-135M-Instruct, `model.layers.2.mlp.gate_proj` row 322, first 256 weights rounded to 4 decimals (the chapter's
  block 2 builds the same row with `torch.round(w * 1e4) / 1e4`). Largest weight -3.2188 at index 100, standard deviation 0.2127.
- Controls: bits (range 2 to 8, step 1, default 4); group size (select 256, 128, 64, 32, 16; default 64; 256 means one scale for the row);
  scheme (select absmax symmetric or zero-point asymmetric; default absmax symmetric); outlier handling (select quantise everything, or keep the
  largest weight in 16 bits and leave it out of the scale; default quantise everything).
- Model: absmax symmetric uses scale = max|x| / (2^(b-1) - 1) per group and round-half-even to codes clipped to plus or minus 2^(b-1) - 1;
  zero point uses scale = (max - min) / (2^b - 1) and a rounded zero point; with the outlier kept, the largest weight is excluded from its group's
  scale and reconstructed exactly. Relative error = norm of the error over norm of the weights; storage = bits + 16 / group per weight.
- Drawn: top, the 256 weights as stems clipped to plus or minus 0.5 (the outlier marked off-scale with its value) with the dequantised values as
  dots and group boundaries as faint ticks; below, the absolute error per weight on the same x axis.
- Default result (4 bits, group 64, absmax): relative error 0.1767, which the chapter's block 2 prints (0.1767). One scale for the whole row: 0.3202;
  groups of 16: 0.1024; 8 bits with groups of 64: 0.0168; keeping the largest weight, 4 bits with one scale: 0.0380 (checked against Python offline,
  not printed by the chapter). Zero point, 4 bits: 0.2728, 0.1514, 0.0760 for groups of 256, 64, 16 (checked offline).
- Table: relative error for 8 to 2 bits against the five group sizes under the current scheme and outlier choice.
- Keyboard: native range and select inputs.

## SpeculativeLab (chapter 05, speculative decoding)

- Controls: acceptance rate alpha (range 0.30 to 0.95, step 0.01, default 0.80); draft length gamma (range 1 to 8, step 1, default 4); draft cost
  coefficient c, the ratio of one draft step to one target step (range 0.01 to 0.50, step 0.01, default 0.05); verification cost v, the cost of the
  (gamma + 1)-token verification pass in single target steps (range 1.0 to 4.0, step 0.1, default 1.0).
- Model: expected tokens per target step E = (1 - alpha^(gamma + 1)) / (1 - alpha); speedup = E / (gamma c + v), which is Leviathan et al.'s Theorem 3.8
  when v = 1. Best gamma is the argmax over 1 to 8.
- Drawn: left, speedup against gamma from 1 to 8 as bars with the chosen gamma and the best gamma marked and the line at 1.0 (no gain); right, a
  timeline of 8 target steps from a seeded stream (mulberry32, seed 3), each a row of gamma draft tokens, green when accepted, red for the first rejection,
  grey for the discarded rest, followed by the one extra token the target always supplies; below, the measured mean tokens per step of those 8 steps.
- Default result: alpha 0.80, gamma 4, c 0.05, v 1.0 gives E = 3.362 tokens per step and speedup 2.80 (chapter block 1 prints both). With the
  chapter's measured CPU verification cost v = 3.3 and c = 0.13, alpha 0.769: speedup 0.83 (block 3 prints 0.83).
- Table: speedup for alpha 0.5, 0.6, 0.7, 0.8, 0.9 against gamma 1 to 8 at the current c and v.
- Keyboard: native range inputs.
