# Track A, agent A4: labs for docs/theory/ml/03-ensembles-and-unsupervised-learning

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and
no randomness at render time. Data that numpy generated is rounded and embedded; the chapter code rounds the same way,
so the printed number and the lab default are the same number.

## EnsembleVoteLab (chapter 01, the lecture's "Majority-vote accuracy" widget)

- Controls: single-model accuracy p (slider 0.50 to 0.95, step 0.01, default 0.70); number of voters n (select, odd values
  1, 3, 5, 7, 9, 11, 15, 25, 51, 101; default 3); error correlation rho (slider 0 to 1, step 0.05, default 0).
- Model: common-cause mixture. With probability rho every voter copies one shared draw (right with probability p);
  otherwise voters are independent. Accuracy = rho * p + (1 - rho) * P(Binomial(n, p) > n / 2).
- Drawn: left, bars for the number of correct voters k = 0..n (green when k is a majority, grey otherwise; with rho > 0 the
  mixture mass sits on k = 0 and k = n); right, a line of ensemble accuracy against n with the chosen n marked.
- Table: n = 1, 3, 5, 11, 25 at the current p and rho.
- Default result: p 0.70, n 3, rho 0 gives 0.784 (0.441 + 0.343). Chapter code prints 0.784.
- Keyboard: native range and select inputs.

## BaggingBoostingLab (chapter 01)

- Dataset: ten points x = 1..10 with labels + + + - - - + + + -. Stumps are searched over thresholds 0.5, 1.5, ..., 10.5 and
  sides +1 then -1; the first stump with a strictly smaller weighted error (tolerance 1e-12) wins ties.
- Mode select: "Boosting (AdaBoost)" or "Bagging (bootstrap)".
- Boosting controls: round slider 0..5. Draws the ten points as circles whose radius follows the sample weight, the
  chosen stump boundary, the weighted error, alpha, and the running ensemble score with its accuracy.
- Boosting expected numbers: round 1 stump x > 3.5 gives -1, else +1, error 0.300, alpha 0.424, weights of the three
  misclassified points 0.167 each, others 0.071; round 2 error 0.214, alpha 0.650; round 3 error 0.182, alpha 0.752 and the
  ensemble is 100% accurate. Default: round 3.
- Bagging controls: number of bootstrap stumps slider 1..7 (default 7). Uses the seven bootstrap index rows printed by
  `np.random.default_rng(1).integers(0, 10, (7, 10))`. Draws each resample's point counts, its stump, and the vote tally.
- Bagging expected number: seven bagged stumps vote to 70% accuracy, the same ceiling as one stump's bias.
- Table: per-round weights (boosting) or per-stump rule (bagging).

## GradientBoostingLab (chapter 02)

- Dataset: 80 points, x sorted uniform on 0..6 from `default_rng(0)`, y = sin(x) + 0.5 sin(3x) + noise(0.25), rounded to
  3 decimals. Truth curve sin(x) + 0.5 sin(3x) on 200 grid points.
- Controls: boosting rounds slider 0..300 (default 100); learning rate select 1.0, 0.3, 0.1, 0.03 (default 0.1).
- Model: depth-1 regression stumps on residuals, start at the mean of y, prediction += lr * stump. Stump split minimises
  squared error, threshold at the midpoint of neighbouring distinct x.
- Drawn: points, truth (dashed), current fit, the residuals as short vertical ticks. Shows train RMSE and RMSE against
  the truth.
- Default result: 100 rounds at lr 0.1 gives train RMSE 0.2687 and truth RMSE 0.1778. lr 1.0 at 300 rounds gives 0.1154
  and 0.1942 (overfit).
- Table: RMSE at 1, 5, 20, 100, 300 rounds for the chosen lr.

## KMeansLab (chapter 03, the lecture's "k-means, step by step (1-D)" widget)

- Dataset select: "Lecture, 1-D" (points 2, 4, 10, 12, 3, 11, 5, k = 2, start centroids 2 and 10) or "2-D blobs" (60 points
  from `make_blobs(60, centers=[(0,0),(5,1),(2,5)], cluster_std=[0.9,0.8,1.0], random_state=3)`, rounded to 2 decimals, k = 3).
- 2-D start select: three presets taking the starting centroids from point indices (30, 37, 49), (0, 25, 27) or (24, 31, 39).
- Buttons: Step (one assign then update), Run to the end, Reset. Step is a single button press; each press is one Lloyd
  iteration.
- Drawn: points coloured by current cluster, centroids as large markers with the path they have travelled, WCSS (sum of
  squared distances to own centroid) and the iteration number.
- Expected: lecture 1-D ends with clusters {2,3,4,5} and {10,11,12}, centroids 3.5 and 11, WCSS 7 after 2 iterations.
  2-D preset (30, 37, 49): WCSS 95.35 (121.31 after the first update); preset (0, 25, 27): 324.28; preset (24, 31, 39): 404.28.
- Also shows the elbow table WCSS for k = 1..8: 652.44, 345.74, 95.35, 79.43, 65.41, 52.99, 44.2, 36.86 (scikit-learn,
  n_init 10, random_state 0).
- Table: current assignments.

## PcaLab (chapter 03)

- Mode select: "Variance explained" (lecture), "Rotate the axis" (2-D cloud), "Scaling matters" (wine).
- Variance explained: four eigenvalue sliders (0..10, step 0.1, defaults 6.2, 2.4, 1.0, 0.4), components kept slider 1..4.
  Draws a scree bar per component and the cumulative line. Default: keep 2 gives 86%.
- Rotate the axis: 60 points from `default_rng(2).multivariate_normal([0,0], [[3,1.6],[1.6,1.2]], 60)` rounded to 2
  decimals. Angle slider 0..180 degrees (step 1, default 0). Draws the points, the axis, the projections as short segments and
  the variance along the axis, the variance retained and the mean squared projection error. A button snaps the axis to PC1.
  Expected: PC1 at 30.9 degrees, variance 3.468, second eigenvalue 0.278, retained 92.6%.
- Scaling matters: bar chart of explained-variance ratios for the 13 wine features, raw vs standardised, with a toggle.
  Expected: raw PC1 99.81%; standardised PC1 36.2%, first two 55.4%.
- Table per mode.
