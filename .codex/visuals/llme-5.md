# Track B, agent D2: labs for docs/llm-engineering/03-training-at-scale

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Numbers that a Python block prints are mirrored by a TypeScript function with the same arithmetic.

## ParallelismMemoryLab (chapter 01, parallelism strategies)

- Mirrors block 2 of the chapter (`c1_2.py`). Memory unit is 1 GB = 1e9 bytes.
- Presets from the Llama 3 paper table of hyper-parameters (vocabulary 128,000): 8B (32 layers, h 4096, ffn 14336, 32 heads, 8 KV
  heads), 70B (80, 8192, 28672, 64, 8), 405B (126, 16384, 53248, 128, 8). Parameter count
  `2*V*h + L*(2h^2 + 2h*kvdim + 3h*ffn + 2h) + h` with `kvdim = h*kv/heads`; 405B gives 405,845,000,192.
- Controls: model (select), tensor parallel t (1, 2, 4, 8, 16), pipeline p (1 to 32), context parallel c (1 to 16), data parallel
  d (1 to 512, powers of two), ZeRO stage (0 to 3), sequence length (2,048 to 131,072), micro-batch (1, 2, 4), activation mode
  (none, tp, tp+sp, tp+sp+selective, full), GPU memory (range 16 to 192 GB, step 1, default 80; an input, not a specification).
- Model: weights `2P/(t p)`, gradients `2P/(t p)`, optimiser `12P/(t p)` (bf16 weights and gradients, fp32 master, m, v). ZeRO 1
  divides the optimiser by d, ZeRO 2 also the gradients, ZeRO 3 also the weights and adds one gathered layer `2*layerParams/t`.
  Activations on the first pipeline stage: `per_layer * L` with `s_local = s/c` and `per_layer` = `sbh(34 + 5as/h)` (none),
  `sbh(10 + 24/t + 5as/(ht))` (tp), `sbh/t (34 + 5as/h)` (tp+sp), `34 sbh/t` (selective), `2sbh` (full).
- Drawn: one horizontal stacked bar (weights, gradients, optimiser, activations) against a marker for GPU memory; readout of
  GPUs = t*p*c*d, total GB and whether it fits.
- Table: the four components in GB and the total.
- Default: 405B, t 8, p 16, c 1, d 128, ZeRO 2, 8,192 tokens, micro-batch 1, tp+sp+selective, 80 GB gives weights 6.34, gradients
  0.05, optimiser 0.30, activations 71.87, total 78.6 GB, 16,384 GPUs. The chapter prints 6.7 GB states + 71.9 GB activations = 78.6 GB.
- Keyboard: native select and range inputs.
