# Data Management lab specifications

## ColumnProjectionLab

- Controls: total columns 10 to 100; columns selected 1 to total.
- Defaults: 50 total, 2 selected; a 10 GB row representation and 2 GB Parquet representation are fixed lecture assumptions.
- Draw: relative bytes read for a full row scan and an ideal column projection; show that 2/50 is 4% and 2 GB × 4% is 0.08 GB. The second number is an idealised lower bound, not a measured file scan.
- Data view: representation, assumed stored GB, selected fraction, ideal bytes read.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## DataQualityRulesLab

- Controls: null values 0 to 150 within 1,000 rows; required completeness threshold 90% to 100%.
- Defaults: 50 nulls and a 99% threshold; completeness 95%, fail.
- Draw: complete and missing records as a proportion bar, numeric score and pass/fail state.
- Data view: row count, non-null count, null count, completeness and threshold.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## PipelineFlowLab

- Controls: arrivals per second 20 to 200; mean processing time 0.1 to 1.0 seconds.
- Defaults: 100 events per second and 0.2 seconds, giving 20 events in flight by Little's Law.
- Draw: arrivals, work in flight and time as a labelled flow and a proportion bar. State that Little's Law uses stable long-run averages and does not predict queue growth when capacity is below arrivals.
- Data view: arrival rate, mean time, mean in-flight count.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## AvailabilityBudgetLab

- Controls: availability target from 99.0% to 99.99%; period in days from 7 to 365.
- Defaults: 99.9% over 365 days, giving 8.76 hours of idealised complete downtime. At 99.99%, 0.876 hours.
- Draw: allowed unavailable time and the available share. State that actual SLAs define measurement windows, exclusions and remedies.
- Data view: target, period hours, allowed unavailable hours and minutes.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## SamplingFractionLab

- Controls: source rows 100,000 to 2,000,000; sample rows 1,000 to 10,000.
- Defaults: 2,000,000 source rows and 5,000 selected, so fraction 0.0025 or 0.25%.
- Draw: sampled share and numeric fraction. State that fraction alone does not make a biased sample representative.
- Data view: source count, sample count, fraction and percentage.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## ProfilingDriftLab

- Controls: nulls 0 to 2,000 among 10,000 rows, required null ceiling 1% to 20%, and current high-bin share 10% to 90% against a 50/50 reference.
- Defaults: 1,500 nulls, 5% ceiling; null rate 15%, fail. Equal 50/50 bins give PSI 0.
- Draw: missingness decision and two-bin PSI with a small epsilon when needed. Explain that PSI describes distribution change, not cause or automatic retraining.
- Data view: row/null counts, threshold, reference and current bin shares, PSI.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## ScdHistoryLab

- Controls: number of changes 0 to 5; observed time from initial state to after the final change.
- Defaults: three changes, yielding four versions of one customer; time 2 selects the third version.
- Draw: version cards with half-open validity windows and active version at chosen time.
- Data view: version number, valid-from, valid-to and active flag.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## FeatureCutoffLab

- Controls: prediction day 1 to 6; feature TTL 1 to 5 days.
- Defaults: feature observations at days 1, 3 and 5 with values 4, 8 and 12; prediction day 4 returns 8 under a three-day TTL, while latest 12 would leak.
- Draw: timeline, selected as-of feature and latest feature; distinguish no match when TTL expires.
- Data view: observation day/value, eligibility at cutoff, selected row.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## RetryBackoffLab

- Controls: base delay 1 to 5 seconds; retry count 1 to 5.
- Defaults: base 2 and three retries, giving waits 2, 4 and 8 seconds, total 14 seconds before task runtime.
- Draw: delay bars and total scheduled wait. State that real orchestrators may add jitter, caps and queue time.
- Data view: retry number, delay and cumulative wait.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## ExperimentChoiceLab

- Controls: maximum allowed inference latency 30 to 100 milliseconds.
- Defaults: run F1 scores 0.71, 0.76 and 0.74 with latencies 35, 80 and 45 ms; 100 ms permits run 2, F1 0.76. A 60 ms limit selects run 3, F1 0.74.
- Draw: one card per run with F1, latency and eligibility. State that tracked metadata alone does not validate the evaluation.
- Data view: run ID, F1, latency and selected status.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## PartitionSkewLab

- Controls: dataset size 1 to 20 GiB, target partition 64 to 256 MiB, largest key share 10% to 80%.
- Defaults: 10 GiB and 128 MiB give 80 ideal size-based partitions; a 50% key holds 5 GiB and can create skew.
- Draw: ideal partition count and largest key mass. State that partitions are tasks, not necessarily concurrent workers, and actual counts depend on file boundaries and shuffle plans.
- Data view: data size, target MiB, ideal partitions, largest key share and GiB.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## CosineRetrievalLab

- Controls: second chunk-vector component 0 to 2; illustrative similarity threshold 0.2 to 0.9.
- Defaults: query [1,0,1,1], chunk [1,1,1,0], dot product 2, both norms square root of 3, cosine 2/3 or 0.667.
- Draw: four dimension cards, score and threshold state. Explain that a score alone does not guarantee selection in a corpus.
- Data view: each vector component and product.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## KAnonymityLab

- Controls: first quasi-identifier group size 1 to 8; other groups hold 5 and 7 records.
- Defaults: sizes 4, 5, 7, giving k = 4. One divided by four is a uniform-guess illustration, not a general identity-risk bound.
- Draw: group-size bars and the minimum. Explain why a unique group needs suppression or generalisation and why sensitive-value inference remains possible.
- Data view: group and record count.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.

## FreshnessBudgetLab

- Controls: age since approved load 0 to 180 minutes; maximum age 15 to 120 minutes.
- Defaults: 90 minutes against a 60-minute limit is a 30-minute breach.
- Draw: age bar and breach state. Explain that a load-time clock is not the source event-time watermark.
- Data view: observed age, maximum age and overage.
- Keyboard: native range controls; dark mode uses palette colours; no overflow at 390 px.
# Data Management practice labs

## RepresentationGapLab

- Controls: training rural share 0–100% and deployment rural share 0–100%, each in 1 percentage-point steps.
- Defaults: 10% training, 40% deployment, matching HealthPredict Q1.
- Drawing: two labelled proportion bars and the absolute percentage-point gap.
- Data view: both shares and the gap in a table through VizPanel.
- Expected number: 30 percentage points, reproduced by the solved paper's Python check.
