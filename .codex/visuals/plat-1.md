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
