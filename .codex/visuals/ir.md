# Information retrieval labs: first group

| Lab | Controls | Default | Drawing and data view | Expected number |
| --- | --- | --- | --- | --- |
| RetrievalMetricsLab | Relevant shown 0–50; total relevant 1–100, constrained to valid counts | 40 relevant shown, 50 shown, 60 relevant total | Precision, recall and F1 bars; numerator and denominator table | 0.800, 0.667, 0.727 |
| BooleanPostingsLab | Toggle document IDs 1, 2, 4, 5, 11, 31 in each list; AND, OR, A NOT B | A = 1,2,4,11,31; B = 1,2,4,5,31; AND | Sorted postings and result chips; membership table | AND = 1,2,4,31 |
| EditDistanceLab | Two text inputs of up to eight letters | cat and cart | Levenshtein dynamic-programming grid, with each cell labelled by distance; matrix table | distance 1; cat to dog gives 3 |
| GapEncodingLab | First ID 1–30; second gap 1–255; third gap 1–30 | IDs 5, 130, 132; gaps 5, 125, 2 | Absolute IDs, gaps and variable-byte hex, plus a byte-count bar and table | 3 compressed bytes versus 12 raw bytes |
| TfIdfLab | Corpus size 100–2000; document frequency 1–N; term frequency 1–10 | N=1000, df=100, tf=3 | IDF and term-weight bar, plus the lecture's fixed cosine and data table | IDF 1.0, tf-idf 3.0, cosine 0.816 |
| DocumentGroupingLab | Classification or clustering select; two cosine sliders 0–1; one k-means step and reset buttons | Sport cosine 0.7, politics 0.4; four fixed 2D points, initial centres at the first two points | Class similarity bars or 2D point plot with moving centres; data table in both modes | Classify as sport; initial clusters 1, 2, 2, 2 become 1, 1, 2, 2 after a step |
| RankingMetricsLab | Up/down buttons for five results | Relevant at ranks 1, 3 and 5; grades 3, 2 and 1 | Ranked rows, AP, P@3, MRR, NDCG; table of gains and precision | AP 0.756 |
| IndexOverlapLab | Two overlap-probability sliders 0.1–1.0 | p_A=0.4 and p_B=0.5 | Pair of overlap bars and inferred relative index-size ratio, with data table | A/B size ratio 1.25 |
| CrawlerRateLab | Parallel host count 100–1000 and per-host delay 1–5 seconds | 500 hosts, 1 second delay | Aggregate pages-per-second bar and per-host schedule table | 500 pages/s while each host receives at most one request/s |
| PageRankLab | Damping 0–1, step and reset buttons | Three-page graph A→P, B→P, P→A; uniform ranks; d=0.85 | Directed graph and changing rank bars; table of inbound contribution | First P update 0.617 |
| CrossLanguageLab | Strategy select, document count 100–2000, query context select | Query translation, 1000 documents, finance context | Translation/indexing work counts and ambiguity example; table | Query translation 1 unit versus 1000 document translations |
| CrossModalSimilarityLab | Image-vector extra coordinate 0–3 | Text [1,0,1,0], image [1,1,1,0] | Four-dimensional vectors, dot product and cosine bars; table | Dot product 2 and cosine 0.816 |
| NeighbourRatingLab | Two ratings 1–5 and two similarities 0.1–1 | Ratings 4 and 5, similarities 0.8 and 0.6 | Weighted numerator, denominator and predicted rating; table | Predicted rating 4.43 |
| RetrievalPipelineLab | Stage select: BM25, illustrative dense, RRF or rerank | Reranked six-document example | Top-three results with relevance marks, P@2 and AP; table | Reranked top two IDs 0 and 2, P@2=1.0 and AP=1.0 |

All three use `VizPanel`, the shared palette and dark-mode hook. Inputs are labelled, keyboard operable and bounded. The default numbers also appear in the corresponding Python blocks and source worked examples.
