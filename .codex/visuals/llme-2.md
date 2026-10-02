# Track B, agent E2: labs for docs/llm-engineering/01-adapting-models (chapters 05 to 07)

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Native range and select inputs only. Data that Python generated is rounded and embedded; the
chapter code prints the same numbers.

## DistillationTemperatureLab (chapter 05, knowledge distillation)

- Fixed teacher logits for five classes (car, truck, bus, cat, carrot): 9.0, 5.5, 4.6, 1.0, -2.0.
- Fixed student logits: 6.0, 6.2, 3.0, 2.5, 0.0, except the truck logit which is a slider.
- Controls: temperature T (slider 1 to 20, step 0.5, default 4); student truck logit (slider 0 to 10, step 0.1, default 6.2);
  weight on the hard label (slider 0 to 1, step 0.05, default 0; the hard label is "car", cross-entropy at T = 1).
- Drawn: grouped bars of the teacher distribution softmax(z_t / T) and the student distribution softmax(z_s / T) for the five
  classes; beneath, the soft-target loss KL(teacher || student), the gradient norm with respect to the student logits, the
  gradient norm times T squared, the hard cross-entropy at T = 1 and the mixed loss alpha * CE + (1 - alpha) * T^2 * KL.
- Default result (block 1 of the chapter): at T = 4 the teacher distribution is 0.5131, 0.2139, 0.1708, 0.0694, 0.0328, the
  KL is 0.10767, the gradient norm 0.06052 and the gradient norm times T^2 0.9683. At T = 1 the KL is 0.66750 and at T = 20
  it is 0.00468.
- Table: per class the two logits and the two tempered probabilities.

## SyntheticFilterLab (chapter 06, synthetic data generation)

- Data: the 600 candidate instructions of the chapter's block 1 (stub generator, `random.Random(0)`), filtered in generation
  order by (1) the rule filters (5 to 40 words, no refusal or image words: 130 rows always dropped), (2) ROUGE-L against the
  12 seeds and every kept row, (3) the stub judge. The grid below was computed by `.lecture-import/track-b/e2/lab_data_synthetic.py`
  and embedded rounded. Diversity is measured on the kept rows: distinct-2 (share of distinct word bigrams), mean pairwise
  cosine of `all-MiniLM-L6-v2` embeddings, and the share of rows with a neighbour above cosine 0.9.
- Controls: judge threshold (slider 5.0 to 9.0, step 0.5, default 7.0); ROUGE-L duplicate threshold (slider 0.40 to 1.00,
  step 0.05, default 0.70).
- Drawn: a stacked bar of the 600 rows split into dropped by rules, dropped as duplicates, dropped by the judge and kept;
  below, two lines against the ROUGE-L threshold at the chosen judge threshold: kept rows, and the share of kept rows with a
  neighbour above cosine 0.9; the current threshold is marked.
- Default result (block 1): 130 dropped by rules, 281 as duplicates, 88 by the judge, 101 kept; distinct-2 0.187, mean
  pairwise cosine 0.249, share with a neighbour above 0.9 of 0.099.
- Table: the 13 ROUGE-L thresholds at the chosen judge threshold with duplicates, judge drops, kept, distinct-2, mean cosine and
  near-neighbour share.

## ContrastiveLossLab (chapter 07, tuning embedding models and rerankers)

- Data: the 8 x 8 cosine matrix of 8 queries (rows) against their 8 documents (columns) from blocks 1 and 2 of the chapter:
  queries "I cannot log in to the compass", "... mailbox", "... harbour", "... pebble", "... lighthouse", "... meadow",
  "... hammock", "... sunbed", documents about eight invented internal tools. Three matrices, rounded to seven decimals,
  from `.lecture-import/track-b/e2/lab_data_contrastive.py`: the base `all-MiniLM-L6-v2`, the model fine-tuned with in-batch
  negatives, and the model fine-tuned with one mined hard negative per pair.
- Controls: model (select: base, tuned with in-batch negatives, tuned with hard negatives; default base); temperature
  (slider 0.01 to 0.50, step 0.01, default 0.05, the library's scale of 20); number of in-batch negatives per query (slider
  1 to 7, default 7, using the first n + 1 queries and documents).
- Loss: mean over the used rows of cross-entropy of softmax(cosine / temperature) against the diagonal.
- Drawn: a heatmap of the softmax probabilities per row (diagonal outlined), with a bar per row for its loss.
- Default result (block 1): base model, temperature 0.05, 7 negatives gives loss 2.5865 (library loss 2.586478). Temperature
  0.01 gives 8.2145, 0.10 gives 2.2106, 0.50 gives 2.0839. At temperature 0.05 one negative gives 0.6904 and three give 1.9741.
  Block 2: tuned 1.0621, tuned with hard negatives 1.3016.
- Table: per query the cosine to the right document, the highest cosine to a wrong document, the probability on the right
  document and the row loss.
