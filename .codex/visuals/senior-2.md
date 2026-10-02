# Track B, agent M2: labs for docs/senior/01-system-design-cases (chapters 04 to 06)

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Data produced by Python is embedded rounded; the chapter's code prints the same numbers.

## AssistantContextLab (chapter 04, AI coding assistant)

- Data: 30 completion points sampled with `random.Random(0)` from `httpx` 0.28.1 (440 functions, SmolLM2 token counts). For each
  point the code ranks every other function four ways (neighbour, bm25, bm25+graph, map+graph) and the lab packs the ranked list
  greedily under a token budget. The ranked lists are cut at 100 items; Python verified the cut changes no result.
- Controls: strategy select (4, default bm25+graph); budget range 250 to 4,000 step 250 (default 1,000); task select ("All 30 tasks,
  average" default, or one completion point).
- Packing rule: walk the ranked list, add an item if it still fits; signature-only items (map+graph only) may use at most 30% of the
  budget. Signature recall: share of needed definitions whose signature is visible. Body recall: share whose full body is packed.
- Drawn, all-tasks view: line chart of body recall against budget (250 to 4,000) for the four strategies, the chosen budget marked.
  Drawn, single-task view: the budget as a bar filled left to right with the packed items (needed, signature only, other) and a list of
  the needed definitions with status.
- Default result: bm25+graph at 1,000 tokens over all 30 tasks gives signature recall 0.500 and body recall 0.500. Chapter code prints
  0.500 and 0.500. Other printed values: neighbour 4,000 gives 0.350; map+graph 4,000 gives signature 0.933 and body 0.683.
- Table: strategy by budget, signature and body recall.
- Keyboard: native select and range inputs.

## EscalationPolicyLab (chapter 05, customer support agent platform)

- Data: 2,000 synthetic tickets from a `mulberry32(7)` generator mirrored in TypeScript and Python (four uniform draws per ticket:
  difficulty d, risky flag u < 0.15, confidence noise, outcome luck). Chance the agent is right p = 0.97 - 0.85 d d, times 0.6 for
  risky tickets; confidence = clip(p before the risky penalty + (noise - 0.5) x 0.3), so the model is overconfident on risky intents.
- Controls: confidence threshold range 0 to 1 step 0.05 (default 0.6); checkbox "always send risky intents to a person" (default
  on); human cost per handled ticket range 1 to 10 step 0.5 (default 4); cost of a wrong automated answer range 0 to 40 step 1 (default
  15). The agent's own cost is fixed at 0.05 per ticket. All costs are placeholder currency units.
- Rule: a ticket is automated when confidence >= threshold and not (escalate risky and risky). Wrong automated answers cost a human
  contact plus the wrong-answer cost. Cost per ticket = (2,000 x 0.05 + (escalated + wrong) x human + wrong x wrong cost) / 2,000.
- Drawn: left, contained share and agent-resolved share against threshold; right, cost per ticket against threshold with the
  all-humans line; the chosen threshold marked on both.
- Default result: threshold 0.6, risky escalated: contained 0.557, resolved by agent 0.479, 156 wrong, cost per ticket 3.304. Chapter
  code prints 0.557, 0.479, 156, 3.304. Also prints threshold 0.0 with risky automated: contained 1.000, cost 6.729.
- Table: threshold 0 to 1 step 0.1 at the current settings.
- Keyboard: native range inputs and checkbox.

## PlatformCostLab (chapter 06, ML platform for many teams)

- Data: hourly GPU demand for four teams over one week (168 hours), produced by `numpy.random.default_rng(3)` (daily sine with
  per-team phase, rare burst sweeps, noise, rounded to whole GPUs) and embedded as integers. Quotas: search 24, ads 16, vision 16,
  nlp 8 (sum 64).
- Controls: cluster size range 40 to 120 GPUs step 4 (default 64); allocation method select (usage only; idle uniform; idle by
  usage; idle by quota; default idle by usage); price per GPU-hour range 0.5 to 3 step 0.1 (default 1.0, a placeholder unit).
- Model: each hour, if total demand exceeds the cluster, every team is served in proportion (so the sum equals the cluster);
  otherwise demand is served in full. Used GPU-hours per team = sum of served GPUs. Idle = cluster x 168 - used. Week cost = cluster x
  168 x price. Methods: usage only (idle left unallocated); idle split equally; idle split in proportion to usage; idle split in
  proportion to quota.
- Drawn: left, stacked bars per team (usage, idle share) with the bill; right, the hourly pooled demand line with the cluster size
  line and the "sum of each team's own peak" line (static partitions). Also reports hours with unmet demand.
- Default result: pooled peak 56, sum of peaks 104 (utilisation 0.623 pooled, 0.336 static). Idle by usage at 64 GPUs and price 1.0:
  search 4,123, ads 2,913, vision 1,921, nlp 1,795, total 10,752; idle 4,887 GPU-hours (45.5%). Chapter code prints the same.
- Table: team, quota, peak, GPU-hours used, bill, for the current settings.
- Keyboard: native range and select inputs.
