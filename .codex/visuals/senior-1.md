# Track B, agent M1: labs for docs/senior/01-system-design-cases, cases 1 to 3

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness that differs between renders. Numbers come from the chapter code: either embedded results of that code or a
seeded generator mirrored in Python.

## RetrievalFunnelLab (chapter 01, `senior-case-document-qa`)

- Data: `retrievalFunnelData.ts`, produced by the chapter's SciFact block (100 test queries, 1,500 documents, 101
  relevant documents). Per query it holds the best rank of a relevant document under BM25, dense (all-MiniLM-L6-v2),
  hybrid (reciprocal rank fusion, k = 60), and the rank after a cross-encoder rerank of the hybrid top 10 and top 20 (0 = not in
  the candidate list).
- Controls: retriever select (BM25, dense, hybrid; default hybrid); retrieval depth select (10, 20, 50; default 20); rerank
  checkbox (enabled only for hybrid at depth 10 or 20; default on); chunks sent to the model select (1, 3, 5, 10; default 5, never
  more than the depth); tokens per chunk slider (100 to 800, step 50, default 350).
- Drawn: three horizontal bars for the funnel: all documents (1,500), the retrieved depth with the share of queries whose evidence is
  inside it (the ceiling), the chunks in the prompt with the share of queries whose evidence is inside them. Prompt tokens =
  chunks x tokens per chunk.
- Default result: hybrid, depth 20, rerank on, 5 chunks gives 0.89; hybrid without rerank at 5 gives 0.92; ceiling at depth 20 is 0.98. The chapter prints these.
- Table: share of queries with a relevant document in the top 1, 5, 10, 20, 50 for the three retrievers.
- Keyboard: native select, checkbox and range inputs.

## LatencyBudgetLab (chapter 02, `senior-case-fraud`)

- Model: 20,000 requests. Stages are log-normal given a median and a 99th percentile: network in 3 and 12 ms, model 6 and 14 ms,
  rules 1 and 3 ms, network out 3 and 12 ms, and a feature fetch of n lookups with a chosen median 4 ms and chosen p99. Standard normals come from a seeded
  mulberry32 generator (seed 20261002) with Box-Muller, drawn in the same order in TypeScript and in the chapter's Python.
- Controls: lookups in parallel or in series (default parallel); number of lookups 1 to 20 (default 8); lookup p99 10 to 80 ms
  (default 25); hedge checkbox (second call after the single-lookup p95, first answer wins; default off); fetch timeout select (none, 20,
  30, 40, 60 ms; default none; on timeout the request continues with default features and is counted as degraded); SLO slider 50 to 200 ms (default 100).
- Drawn: histogram of end-to-end latency in 2 ms bins with the SLO line and the p99; below it the percentile table.
- Default result: parallel, 8 lookups, p99 25 ms, no hedge: p50 26.8, p95 44.6, p99 57.9, p99.9 93.2 ms, 0.07% over 100 ms. Series gives p99 104.2 and 1.39% over 100 ms. The chapter prints both.
- Table: p50, p95, p99, p99.9, share over SLO, degraded share, extra calls.

## RankingFunnelLab (chapter 03, `senior-case-ranking`)

- Data: `rankingFunnelData.ts`, produced by the chapter's funnel simulation (8,000 items, 4,000 users, 300 evaluation users, relevant = the 80 items of highest true utility per user). Rows for K1 in 100, 300, 1000, 2000 and K2 in 50, 100, 200
  hold recall after retrieval, recall after the light ranker, precision at 10 and NDCG at 10 after the heavy ranker.
- Controls: retrieval K1 select; light ranker keeps K2 select (only K2 not above K1); cost per item for the light and the heavy ranker in microseconds
  (number inputs, defaults 2 and 60); retrieval cost per request in ms (default 3); requests per second (default 20,000).
- Drawn: four stage bars (catalogue, K1, K2, the 10 shown) with the recall or precision reached at each; CPU milliseconds per request = retrieval + K1 x light cost + K2 x heavy cost,
  and cores = requests per second x CPU seconds / 0.5 utilisation.
- Default result: K1 1000, K2 200 gives precision at 10 of 0.345 and NDCG at 10 of 0.337; 17.0 ms CPU and 680 cores at the default costs. The chapter prints the table and the same formula.
- Table: the whole grid.
