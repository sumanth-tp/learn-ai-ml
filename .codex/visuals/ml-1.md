# Track A, agent A1: lab specifications

All four labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop for the data view,
native range and select controls (keyboard operable), no external dependencies and no randomness other than a
seeded generator. The seeded generator is Mulberry32 (a 32-bit integer generator) plus Box-Muller normals, written
identically in the chapter's Python and in the lab, so the defaults reproduce the chapter's printed numbers.

## BiasVarianceLab (chapter 01 `what-machine-learning-is`)

Idea: the first thing that goes wrong. A polynomial too stiff to bend (underfit) or too loose (overfit).

- Data: `x_i = (i + u_i) / n` for `i = 0..n-1` (`u_i` uniform), `y_i = cos(1.5 pi x_i) + noise * N(0,1)`.
  Training seed `s`, test set of 200 points from seed `s + 1`, same noise.
- Controls: degree 1..min(15, n-1) default 4; noise sd 0.05..0.60 step 0.05 default 0.30; training rows 10..60
  step 5 default 30; seed select (11, 21, 31, 41) default 11 (button "new sample" cycles it).
- Fit: least squares on the Vandermonde of `2x - 1` by Householder QR.
- Draws: panel one, the true curve (dashed), the training points, the fitted polynomial clipped to the y range.
  Panel two, train and test MSE for every degree on a log axis with the chosen degree marked.
- Table: degree, train MSE, test MSE.
- Defaults (n 30, noise 0.3, seed 11, degree 4): train MSE 0.0979, test MSE 0.1051, noise variance 0.0900.
  Other degrees to cross-check against the chapter table: 1 -> 0.3154 / 0.2823, 15 -> 0.0432 / 0.7246.

## ScalingOutlierLab (chapter 02 `data-preprocessing`)

Idea: the lecture's widget "set a value and see min-max and standardised scaling side by side", plus the outlier
rules.

- Data: `{2,4,5,6,7,8,9,10,12}` plus one editable outlier (default 45), which can be switched off.
- Controls: value x 0..100 step 0.5 default 8; outlier value 10..100 default 45; include outlier checkbox default
  on; sigma select (population, sample) default population.
- Draws: a number line with every point, the IQR fences, the mean +- 3 sigma limits, the query value, flagged
  points ringed (IQR rule) or crossed (3-sigma rule).
- Readouts: standardised, min-max and robust scaling of x; Q1, Q3, IQR, fences; mean, sigma, limits; which
  points each rule flags.
- Table: every point with its three scaled values and its two flags.
- Defaults: z -0.239, min-max 0.140, robust (x - median) / IQR = 0.5 / 4.5 = 0.111; Q1 5.25, Q3 9.75, IQR 4.5,
  upper fence 16.5, 3-sigma upper limit 46.0; IQR flags 45, 3-sigma flags nothing. Sample sigma gives z -0.226.

## LeakageLab (chapter 03 `features-leakage-and-imbalance`)

Idea: choosing features with all the rows, then testing on rows that took part in the choice.

- Data: `n` rows of `p` standard normal features and labels that are fair coin flips (+1 or -1), seed `s`.
- Rule: score = sum over the best `k` features of `sign(corr_j) * x_j`; predict +1 if the score is positive.
  `corr_j = mean(x_j * y)`. The first half of the rows is the training half, the second half the test half.
- Leaky: choose features on all rows, score on the test half. Honest: choose on the training half only.
- Controls: rows 40..400 step 20 default 100; features 100..2000 step 100 default 1000; k 1..100 default 20;
  seed select (5, 6, 7, 8) default 5.
- Draws: bars for leaky test accuracy, honest test accuracy and honest accuracy on its own training half with a
  chance line at 0.5; below, both test accuracies against k with the chosen k marked.
- Table: k, leaky test accuracy, honest test accuracy for a set of k values.
- Defaults (100, 1000, 20, seed 5): leaky 0.840, honest 0.460, honest on training half 0.980.

## ImbalanceThresholdLab (chapter 03)

Idea: accuracy, ROC and precision tell different stories as the positive class gets rare; the threshold is a
business decision.

- Model: negatives score N(0,1), positives N(d, 1). No sampling, so the numbers are exact expectations.
- Controls: prevalence select 0.5%, 2%, 5%, 10%, 20%, 50% (stored as a range index) default 2%; separation d
  0.5..4 step 0.1 default 2.0; threshold -2..6 step 0.05 default 1.0; cost of a missed positive 1..100 default
  10 (a false alarm costs 1); button "jump to cost-optimal threshold".
- Draws: the two score curves scaled by `prevalence` and `1 - prevalence` with the threshold line; the ROC curve
  and the precision-recall curve with the operating point on each.
- Readouts per 10,000 cases: caught, missed, false alarms, correct rejections; recall, false-positive rate,
  precision, accuracy, accuracy of "always negative", F1, ROC-AUC, average precision, expected cost, minimum cost
  and its threshold.
- Table: the confusion counts and rates.
- Defaults (2%, d 2.0, threshold 1.0, cost 10): caught 168.3, missed 31.7, false alarms 1554.8, correct
  rejections 8245.2; recall 0.8413, FPR 0.1587, precision 0.0977, accuracy 0.8413 vs 0.9800 for always negative;
  ROC-AUC 0.921, average precision 0.3753; cost 1872 at threshold 1.0, minimum 1194 at threshold 1.79.
