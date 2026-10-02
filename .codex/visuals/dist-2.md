# Track B, agent D1b: labs for docs/mlops/distributed/02-dist-challenges and 03-dist-learning

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Randomness comes from `dist2Math.ts` (`mulberry32`, Box-Muller `normals`, a seeded logistic
regression generator); the chapters' Python reimplements the same generator, so the lab default and the printed number are
the same number.

## StragglerLab (chapter 02-dist-challenges/01)

- Model: each worker's step time is `exp(sigma * z)` with z standard normal (median 1). A synchronous step waits for the
  N-th fastest of N + b launched workers (b backup workers). 400 steps, `mulberry32(12345)`, the stream continues across steps.
- Controls: workers N (select 1, 4, 16, 64, 256; default 64); spread sigma (slider 0.1 to 0.8, step 0.05; default 0.5);
  backup workers b (select 0, 1, 2, 4, 8, 16; default 0).
- Drawn: left, bars of mean step time for N = 1, 4, 16, 64, 256 at the chosen sigma and b (selected N highlighted);
  right, histogram of the 400 step times for the chosen N with the mean marked.
- Default result: N 64, sigma 0.5, b 0 gives mean step time 3.327. N 64, sigma 0.5, b 8 gives 1.819.
- Table: mean step time for N = 1, 4, 16, 64, 256 at b = 0 and at the chosen b.

## ParameterServerLab (02-dist-challenges/02)

- Model: model size S MB, N workers, P server shards (key ranges), link bandwidth B GB/s. Per step each worker pushes S and
  pulls S. Each server shard receives N*S/P and sends N*S/P. Ring all-reduce: each worker sends 2(N-1)/N * S.
  Parameter-server communication time = 2 * max(S, N*S/P) / B; ring time = 2(N-1)/N * S / B.
- Controls: workers N (slider 1 to 64, default 4); servers P (select 1, 2, 4, 8, 16; default 1); model size (select 25, 100,
  400, 1600 MB; default 100); bandwidth (select 1, 10, 100 GB/s; default 10). The numbers are named parameters, not measurements.
- Drawn: bars of bytes handled per step by one server shard (in) and one worker (push + pull) and the ring worker.
- Default result: N 4, P 1, 100 MB: server shard receives 400 MB, ring worker sends 150 MB (1.5 x model), parameter server
  0.080 s against ring 0.015 s.
- Table: N = 4, 16, 64 against P = 1, 4, 16, server load in MB and communication time.

## StaleGradientLab (03-dist-learning/01)

- Data: `makeLogistic(200, 5, 7)`; full-batch gradient descent on logistic loss, 200 steps, weights start at zero.
- Delay model: the gradient applied at step t was computed at the weights from step t - tau (zero weights before the start).
- Controls: delay tau (slider 0 to 30; default 0); learning rate (select 0.1, 0.5, 1, 2, 4, 6; default 1); checkbox "divide the
  rate by 1 + tau" (default off).
- Drawn: loss against step for tau = 0 (reference) and the chosen tau.
- Default result: tau 0, rate 1 gives final loss as printed by the chapter (see code block 3); the table lists the final loss
  for tau = 0, 1, 2, 4, 8, 16, 30 at the chosen rate.

## GradientCompressionLab (03-dist-learning/02)

- Data: `makeLogistic(400, 20, 11)`, 4 workers with 100 rows each, 100 synchronous steps, rate 1, weights start at zero.
  Every worker compresses its gradient before the mean is taken.
- Controls: method select (none, uniform quantisation, top-k, top-k with error feedback; default none); bits (select 2, 4, 8;
  quantisation only; default 8); kept fraction (select 0.05, 0.1, 0.25, 0.5; top-k only; default 0.1).
- Drawn: loss against step for the uncompressed run and the chosen method; the bytes sent per worker per step as a share of dense.
- Default result: none gives the final loss printed by the chapter, 100% of bytes.
- Table: final loss and byte share for every method setting.

## LocalSgdLab (03-dist-learning/03)

- Data: `makeLogistic(800, 10, 5)` split in 8 contiguous shards of 100, deliberately sorted by label so shards differ
  (non-identical distributions). Each worker runs tau local full-shard gradient steps with rate 0.5, then the models are averaged.
- Controls: local steps tau (select 1, 2, 5, 10, 20, 50; default 10); communication cost per round c (select 1, 5, 10, 50, in
  units of one local step; default 10); schedule select (fixed tau, AdaComm-style decreasing tau; default fixed).
- Wall-clock model: each local step costs 1, each averaging round costs c.
- Drawn: global loss against wall-clock time for tau = 1 and the chosen setting.
- Default result: values printed by chapter 03-dist-learning/03 code block 2.
- Table: for each tau, rounds used, loss after 200 local steps, wall-clock time at c.
