# Track B, agent C2: labs for docs/mlops/platform

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, native range, select and button
controls, no external dependencies and no randomness at render time. Numbers that Python printed are reproduced either by
the same closed-form formula or by an embedded, rounded constant.

## AbTestPowerLab (chapter 01)

- Controls: baseline conversion rate p0 (range 0.01 to 0.50, step 0.01, default 0.10); minimum detectable effect, absolute
  (range 0.002 to 0.050, step 0.001, default 0.010); significance level alpha (select 0.01, 0.05, 0.10; default 0.05);
  target power (select 0.70, 0.80, 0.90; default 0.80); CUPED pre/post correlation rho (range 0 to 0.95, step 0.05,
  default 0); eligible users per day, both arms together (select 2,000, 5,000, 20,000, 100,000; default 5,000).
- Formula: n = (z(1 - alpha/2) sqrt(2 pbar (1 - pbar)) + z(power) sqrt(p0 (1 - p0) + p1 (1 - p1)))^2 / mde^2 with
  p1 = p0 + mde, pbar = (p0 + p1) / 2. With CUPED, n is multiplied by (1 - rho^2). Power at a given n inverts the same
  expression.
- Drawn: a power curve against users per arm (log-spaced x from 500 to 400,000), the target power as a dashed line, the
  required n as a vertical marker. Below the plot, a status line with n per arm, total users and days to run.
- Table: n per arm at MDE x2, x1, x0.5 for the current settings, with and without CUPED.
- Default result: p0 0.10, MDE 0.010, alpha 0.05, power 0.80 gives 14,751 users per arm, 29,502 in total and 6 days at
  5,000 users a day; MDE 0.020 gives 3,841 and MDE 0.005 gives 57,763. Chapter block 1 prints 14,751, 3,841 and 57,763.
- Keyboard: native range and select inputs.

## ReplicaSchedulerLab (chapter 02)

- Cluster (fixed, same as chapter block 2): `cpu-1` 8 CPU and 0 GPU; `gpu-a` 16 CPU and 2 GPU; `gpu-b` 16 CPU and 4 GPU.
- Controls: preprocessing pods, 3 CPU and no GPU each (range 0 to 8, default 5); GPU pods, 4 CPU and 1 GPU each, with a
  toleration (range 0 to 10, default 6); strategy select "spread (lowest CPU utilisation first)" or "bin-pack (highest first)",
  default spread; select "GPU nodes tainted" yes or no, default no. Second group for the autoscaler readout: ready replicas
  (range 1 to 12, default 2), observed CPU utilisation in percent (range 10 to 400, default 315), target (range 30 to 90,
  default 70), minReplicas 2 and maxReplicas 12 fixed.
- Scheduling: preprocessing pods are placed first, then GPU pods, one at a time. A node is feasible when CPU and GPU requests
  fit and, if tainted, the pod tolerates the taint. Score is CPU utilisation after placement; ties break by node name.
- Autoscaler readout: desired = clamp(ceil(ready x utilisation / target), 2, 12), or unchanged when the ratio is within 10%
  of 1.
- Drawn: three node panels with CPU and GPU capacity bars and one square per pod (blue preprocessing, orange GPU), and a row
  of Pending pods under them. Status line with placed, Pending and idle GPUs.
- Table: node, CPU used of capacity, GPU used of capacity, pod count.
- Default result: spread, untainted, 5 preprocessing and 6 GPU pods places 4 of 6 GPU pods, 2 Pending, 2 GPUs idle (block 2
  prints the same). Tainted places 6 of 6 and 2 of 5 preprocessing pods (3 Pending). Autoscaler default: 2 ready at 315%
  against 70% asks for 9 (block 3 prints 9).

## PlanDiffLab (chapter 03)

- State (fixed, the result of applying the chapter's V1 config): `network.main` cidr 10.0.0.0/16; `bucket.artifacts` name
  ml-artifacts-prod, versioning true; `cluster.gpu` node_count 2, machine_type gpu-small; `endpoint.ranker` replicas 3, image
  ranker:1.0, bucket_name ml-artifacts-prod. Immutable attributes: network.cidr and bucket.name.
- Controls: endpoint replicas (range 1 to 8, default 5); endpoint image (select ranker:1.0 or ranker:1.1, default 1.1);
  bucket name (select ml-artifacts-prod, ml-artifacts-prod-eu, ml-artifacts-prod-us; default -eu); checkbox "add latency alarm"
  (default on); checkbox "prevent_destroy on the bucket" (default off); checkbox "someone set node_count to 4 in the console"
  (default off); checkbox "ignore_changes on node_count" (default off).
- Engine: mirrors the chapter's `plan` function. bucket_name in the endpoint follows the bucket's configured name, so renaming
  the bucket also updates the endpoint. A replace counts as one add and one destroy in the summary line.
- Drawn: one row per planned action with the symbol (+, ~, -, -/+), the address and the attribute changes, coloured by action;
  under it the summary line "Plan: A to add, C to change, D to destroy." or "No changes."; an error banner when prevent_destroy
  blocks a replacement.
- Table: action, address, attribute, from, to.
- Default result: "Plan: 2 to add, 1 to change, 1 to destroy." with bucket.artifacts replaced, endpoint.ranker changed (three
  attributes) and alarm.latency added. Drift on and everything else at the state's values gives "Plan: 0 to add, 1 to change,
  0 to destroy." (cluster.gpu node_count 4 to 2); with ignore_changes also on it gives "No changes.".

## PlatformChooserLab (chapter 04)

- Data: the capability matrix printed by chapter block 2: 14 capabilities by 5 platforms (SageMaker AI, Vertex (Agent
  Platform), Azure ML, Bedrock, Foundry). A cell holds the key of the official documentation page that supports it, or nothing
  when the page read did not show it ("not found on the pages read", never "not offered"). The matrix is generated from the
  Python block into the TypeScript file so the two cannot drift.
- Controls: scenario select ("classical ML team", "GenAI application team", "fine-tune and serve", "custom"; default classical
  ML team) and one native checkbox per capability. Changing a checkbox switches the scenario to "custom".
- Drawn: one horizontal bar per platform showing how many of the required capabilities have a source, labelled "k of n", sorted
  by count then name; under each bar the capabilities not found on the pages read.
- Table: capability by platform, with the source key or "not found".
- Default result: classical ML team (7 required): Azure ML 7 of 7, SageMaker AI 7 of 7, Vertex (Agent Platform) 7 of 7,
  Bedrock 2 of 7, Foundry 1 of 7. GenAI application team: Bedrock 6 of 6, Foundry 4, Vertex 4, Azure ML 3, SageMaker AI 3.
  Fine-tune and serve: Azure ML 4, SageMaker AI 4, Vertex 3, Bedrock 2, Foundry 2. These equal the block 2 output.

## BatchWindowLab (chapter 05)

- Controls: rows per night (select 5 million, 20 million, 50 million, 200 million; default 50 million); rows per second per
  worker (select 500, 2,000, 10,000; default 2,000; labelled an assumed rate, the reader replaces it with a measured one);
  workers (range 1 to 32, default 4); batch window in hours (range 1 to 12, step 0.5, default 2); share of the rows held by the
  largest partition (range 0 to 0.60, step 0.05, default 0.30); chunk size (select none, 20 million, 5 million, 1 million;
  default none).
- Formula: makespan = max(rows / (rate x workers), min(share x rows, chunk) / rate), in seconds, with the chunk limit ignored
  when "none". Workers needed with a perfect split = ceil(rows / (rate x window)).
- Drawn: makespan in hours against the number of workers 1 to 32 (a falling curve that flattens at the largest partition's
  time), the window as a dashed horizontal line, the chosen worker count marked; status line with makespan, fits or not,
  workers needed, and utilisation (perfect-split time divided by makespan).
- Table: workers 1, 2, 4, 8, 16, 32 against makespan as partitioned and chunked at 5 million rows.
- Default result: 50 million rows, 2,000 rows/s/worker, 4 workers, 2 h window, share 0.30, no chunk: makespan 2.08 h, does not
  fit, 4 workers needed with a perfect split, utilisation 83%. With the 5 million chunk: 1.74 h, fits. 8 workers unchunked
  stays at 2.08 h. Chapter block 3 prints these.
