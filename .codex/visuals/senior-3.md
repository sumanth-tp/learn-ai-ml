# Track B, agent M3: labs for docs/senior/02-engineering-craft

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, native range and select controls,
no external dependencies and no randomness except the seeded `mulberry32` generator in `craftMath.ts`. The maths lives in
`src/components/viz/craftMath.ts`; each chapter's Python mirrors it, so the printed number and the lab default are the same
number. Money values are named assumptions, never quotes.

## BuildBuyLab (chapter 01)

- Fixed parameters: 1,500 input and 300 output tokens per request; API input price 2.00 per million tokens; replica output
  throughput 1,200 tokens/s; peak to mean 4; at least 2 replicas; 730 GPU hours a month; engineer cost 15,000 a month;
  API error rate 6%; cost of one error 0.10.
- Controls: requests per month (select, 100k to 200M, default 5,000,000); API output price per million tokens (slider 2 to
  16, step 1, default 8); GPU price per hour (slider 1 to 6, step 0.25, default 2.50); platform engineers in FTE (slider 0 to
  2, step 0.25, default 0.5); extra error points for the open-weights model (slider 0 to 10, step 1, default 3).
- Drawn: monthly cost of buying (line) and hosting (stepped line, steps are whole replicas) against volume on a log axis,
  a marker at the chosen volume, and a vertical line at the break-even volume found by bisection (80 iterations between
  1,000 and 500,000,000).
- Default numbers: buy 57,000 and host 56,150 at 5,000,000 requests; break-even 4,645,833 requests per month; without the
  quality gap (0 points) 2,064,815.
- Table: monthly cost of both options at 100k, 1M, 5M, 20M, 100M requests.

## RoiLab (chapter 02)

- Fixed: 120 users, 400 tasks per user per month, loaded rate 30 per hour, fixed run cost 6,250 a month, 24 months, discount
  10% a year. Cost per task follows the cached agent-loop formula of the chapter (prefix 1,154 tokens, each step adds 172,
  answer 55 tokens, input 2.00 and output 8.00 per million, cache read 0.1, cache write 1.25), rounded to 5 decimals.
- Controls: adoption (0.1 to 1, step 0.05, default 0.60); minutes saved per useful task (0.5 to 5, step 0.5, default 2.5);
  rework share (0 to 0.6, step 0.05, default 0.25); realisation (0.2 to 1, step 0.05, default 0.60); build cost (30,000 to
  200,000, step 5,000, default 90,000); agent steps per task (1 to 20, default 6).
- Drawn: cumulative net cash by month 0 to 24 starting at minus build cost, a zero line and a payback marker.
- Default numbers: monthly benefit 16,200, monthly cost 6,514, net 9,686, payback 9.3 months, 24-month NPV 120,810, ROI 1.58.
- Table: cumulative cash at months 0, 3, 6, 9, 12, 18, 24.

## EstimateRangeLab (chapter 03)

- Eight tasks of the chapter (optimistic, likely, pessimistic days; three are data-dependent), 4,000 trials, `mulberry32(2026)`,
  triangular inverse CDF, draw order per trial: one shared draw, then for each task a duration draw and an own draw.
- Controls: probability that data is worse than assumed (0 to 0.8, step 0.05, default 0.35); slowdown when it is (1 to 3, step
  0.25, default 1.5); risk model (select: shared, independent; default shared); percentile to read (50 to 95, step 5, default 85).
- Drawn: histogram of project length in 2-day bins, markers at the sum of likely days (40), the median and the chosen percentile.
- Default numbers: shared, 0.35, 1.5: mean 59.6, P50 58.1, P85 70.9, P95 81.2. Independent: P85 69.0.
- Table: percentiles 10, 25, 50, 70, 85, 95.

## DecisionMatrixLab (chapter 04)

- Three options by six criteria, scores 1 to 5 fixed (the chapter's table); weights default 5, 4, 3, 4, 3, 2.
- Controls: six weight sliders (1 to 10, step 1); select for a hard gate on privacy and residency (none, minimum score 3).
- Drawn: weighted score per option (bars), win share under weight jitter (each weight times 0.5 plus a uniform draw, 2,000
  draws, `mulberry32(99)`), and which options the gate removes.
- Default numbers: scores 3.571, 3.429, 3.143; win shares 0.722, 0.203, 0.075; with the gate only the self-hosted model survives.
- Table: the score matrix with the current weights.

## ErrorBudgetLab (chapter 06)

- Controls: SLO (select 99, 99.5, 99.9, 99.95, 99.99; default 99); window days (select 7, 28, 30; default 30); three events,
  each with a share of responses that are bad (select 1, 3, 8, 15, 100 per cent) and a duration in hours (slider 0.5 to 48,
  step 0.5). Defaults 8% for 6 h, 3% for 20 h, 100% for 0.5 h.
- Drawn: a bar of the budget used by each event (stacked) against 100%, the burn rate of each event, and which burn-rate rules
  fire (14.4x over 1 h, 6x over 6 h, 1x over 72 h; a rule fires when burn rate times min(duration, window) divided by window is at or above the rate).
- Default numbers: 6.7%, 8.3% and 6.9% of the budget, 21.9% together; the 8% event burns at 8x and fires the 6x page; the 3%
  event burns at 3x for 20 h and fires no rule (its 72 h average is 0.83x); the outage fires the 14.4x and 6x pages.
- Table: per event, burn rate, budget used and rules fired.
