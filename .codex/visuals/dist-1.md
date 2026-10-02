# Track B, agent D1a: labs for docs/mlops/distributed/01-dist-foundations

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Each lab mirrors a Python block in the chapter (`.lecture-import/track-b/d1a-code/`), so the
printed number and the lab default are the same number.

## AllReduceLab (chapter 02, scalable frameworks)

- Controls: workers N (select 2, 3, 4, 5, 6, 8; default 4); step slider 0 to 2(N-1) (default 2(N-1), the finished state).
- Inputs: worker w (0-based) holds N chunks, chunk c has the single number (w + 1) + 10 c.
- Model: ring all-reduce. Steps 1 to N-1 are reduce-scatter: at step s worker w sends chunk (w - s) mod N to worker
  (w + 1) mod N, which adds it. Steps N to 2(N-1) are all-gather: at step s' = s - (N-1) worker w sends chunk
  (w + 1 - s') mod N to worker (w + 1) mod N, which overwrites. Sends in a step are snapshotted before any receive.
- Drawn: an N by N grid, rows are workers, columns are chunks, each cell shows its current value. Cells sent in the step
  just taken are outlined; cells that received are filled. Phase label and step counter above.
- Table: workers 2, 4, 8, 16, 64 with steps 2(N-1), gradient traffic per worker 2(N-1)/N, single parameter server inbound N.
- Default result: N = 4, step 6: every worker holds 10, 50, 90, 130; steps 6 = 2(N-1); traffic 1.500 per worker.
  Chapter code block 4 prints the same.
- Keyboard: native range and select inputs.

## DataParallelScalingLab (chapter 03, data parallelism)

- Controls: workers K (select 1, 2, 4, 8, 16, 64, 256; default 8); local batch B (select 8, 16, 32, 64, 128; default 32);
  milliseconds of compute per sample (range 1 to 20, step 1, default 5); gradient size in MB (select 10, 100, 500, 2000;
  default 100); link bandwidth in GB/s (select 1, 10, 50, 100; default 10); overlap of communication with compute
  (range 0 to 0.95, step 0.05, default 0).
- Model: compute = B x ms; comm = 0 at K = 1 else 2(K-1)/K x (MB / 1000) / (GB/s) x 1000 ms (ring, bandwidth term only);
  step = compute + (1 - overlap) x comm; speedup = K x compute / step; efficiency = speedup / K.
- Drawn: speedup against workers on a log axis with the ideal line, the chosen K marked; readout of global batch K x B and
  the linear-rule learning rate multiplier K.
- Table: K = 1, 2, 4, 8, 16, 64, 256 with compute, comm, step, speedup, efficiency at the current settings.
- Default result: K 8, B 32: compute 160.0 ms, comm 17.50 ms, step 177.50 ms, speedup 7.211, efficiency 0.901 (chapter code
  block 4 prints the same). Illustrative parameters, not measured hardware.
- Keyboard: native range and select inputs.

## PipelineBubbleLab (chapter 04, model parallelism)

- Controls: stages S (range 2 to 8, default 4); micro-batches m (range 1 to 32, default 8).
- Model: forward fill and drain. Stage s runs micro-batch j at tick s + j; total ticks m + S - 1; bubble (S-1)/(m+S-1).
- Drawn: S rows by (m + S - 1) columns; busy cells coloured by micro-batch on the sequential ramp, idle cells grey; a marker
  when m >= 4S (the GPipe rule of thumb).
- Table: m = 1, 2, 4, 8, 16, 32, 64 with bubble at the current S.
- Default result: S 4, m 8: 11 ticks, 32 busy cells of 44, bubble 3/11 = 0.273. Chapter code block 1 prints the same.
- Keyboard: native range inputs.
