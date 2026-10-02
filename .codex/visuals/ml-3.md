# Track A, agent A3: lab specs (instance-based learning, SVM, Bayesian learning)

All three labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, deterministic seeds,
no external dependencies, and no code comments. Every control is a native range, select or button, so Tab, arrow keys
and Enter work. Class identity is never colour alone: "+" is a filled circle, "-" is a square.

## KnnLab (`src/components/viz/KnnLab.tsx`), chapter `ml-instance-based`

The lecture's "k-NN playground" plus the curse of dimensionality.

| Control | Range | Default |
| --- | --- | --- |
| view | neighbours, high dimensions | neighbours |
| data (neighbours view) | lecture's five points, two features on different scales | lecture's five points |
| query x, query y | 0 to 6, step 0.1 (five points); the plot box of the other data set | (2, 3) |
| k | 1 to 5 (five points), 1 to 15 (other data set) | 3 |
| weighting | uniform votes, 1/d^2 weights | uniform |
| standardise features (scales data set only) | checkbox | off |
| dimensions (high dimensions view) | 2 to 300 | 50 |

Drawn: the labelled points, the query as a diamond, a line to each of the k neighbours, and the ellipse that just
contains them (a circle whenever the metric is isotropic). The readout states the vote or weight totals and the
prediction. Arrow keys on the focused plot move the query by 0.1.

Expected numbers at the defaults (these are the numbers `knn_1.py` prints): distances B 1.00, C 1.00, E 1.41, A 2.24,
D 3.61; uniform vote + 2 to - 1, predicts +; switching to 1/d^2 gives weights + 1.50 against - 1.00, share of +
60% against 67% unweighted. On the scales data set the leave-one-out accuracy readout shows the effect of standardising.

High dimensions view: 200 uniform points per draw, 30 queries, seed fixed; shows a histogram of distances from one
query and the mean nearest/farthest ratio. It uses its own seeded generator, so its values differ somewhat from the
numpy figures in the chapter, and says so; the trend (ratio rising towards 1) is the same.

Table view: point, class, distance, in the k nearest, weight (neighbours); dimension and mean ratio (high dimensions).

## SvmMarginLab (`src/components/viz/SvmMarginLab.tsx`), chapter `ml-svm`

The lecture's "Tilt the boundary" widget, the C knob, and the kernel idea.

| Control | Range | Default |
| --- | --- | --- |
| view | margin and C, kernel lift (1-D to 2-D), RBF similarity | margin and C |
| angle of the line | 0 to 359 degrees, step 1 | 45 |
| offset of the line | 0 to 5, step 0.01 | 2.12 |
| add the stray "+" at (1.2, 1.3) | checkbox | off |
| C | 0.01 to 100 (log steps of 10^0.25) | 1 |
| fit for this C | button, exact dual solution | none |
| cut height (lift view) | 0 to 16, step 0.05 | 4.25 |
| gamma (RBF view) | 0.05 to 3, step 0.05 | 0.5 |

Data: the eight points of `svm_1.py` (+ at (2,2), (3,1), (4,3), (3,4); - at (1,1), (2,0), (0,0), (0,1)).
Hand mode: the line is w.x = c with w = (cos a, sin a). The margin is twice the distance to the nearest correctly
classified point; errors are the points on the wrong side. Fit mode: an SMO solver finds the exact soft-margin
solution for the chosen C and the lab shows margin 2/|w|, support vectors, slack points and accuracy.

Expected numbers: at the defaults the line is x1 + x2 = 3 (to rounding of the offset), margin 1.41, 0 errors, support
points (2,2), (3,1), (1,1), (2,0). "Fit for this C" with C = 1 and no stray point gives w = (1, 1), b = -3, margin
1.4142. With the stray point on, fitting at C = 0.01, 0.1, 1, 10, 100 gives margins 22.63, 4.80, 1.41, 0.55, 0.36 and
support vector counts 9, 8, 4, 2, 2 (the table `svm_2.py` prints).

Lift view: 11 one-dimensional points (+ for |x| <= 1.5, - for |x| >= 2.5), lifted to (x, x^2). The horizontal cut at
height h falls back on the line at x = +-sqrt(h). Default h = 4.25 gives cut points +-2.062 and 0 errors (as `svm_3.py`).
RBF view: similarity exp(-gamma d^2) against distance d; marker at d = sqrt(13) for gamma 0.5 reads 0.0015 (as
`svm_3.py`).

Table view: the points with their signed score and slack (margin view); the lifted coordinates (lift view); similarity
at a range of distances (RBF view).

## BayesRuleLab (`src/components/viz/BayesRuleLab.tsx`), chapter `ml-bayesian`

The lecture's "disease-test surprise", the Naive Bayes spam example, and Laplace smoothing.

| Control | Range | Default |
| --- | --- | --- |
| view | disease test, spam filter, smoothing | disease test |
| prevalence (disease view) | 0.01% to 50%, log10 steps of 0.05 | 0.1% |
| sensitivity | 50% to 100%, step 0.5 | 99% |
| false-positive rate | 0% to 50%, step 0.5 | 5% |
| positive tests in a row | 1 to 3 | 1 |
| P(spam), four word likelihoods, free/money present or absent (spam view) | 5% to 95%, step 1% | 0.4; 0.8, 0.6, 0.1, 0.2; both present |
| alpha (smoothing view) | 0 to 2, step 0.1 | 1 |

Drawn: a population of 100,000 as a count bar (true positives against false positives among everyone who tests
positive), the prior and posterior probability, and the posterior as a large figure. The spam view draws the two
unnormalised scores and the posterior. The smoothing view shows the six factors of each class product.

Expected numbers at the defaults: 100 sick, 99 true positives, 4,995 false positives, posterior 1.94% (0.0194).
Two positives in a row 28.2% (0.2818), three 88.6% (0.8860). Spam view: spam score 0.192, ham score 0.012,
P(spam) 94.1%; with only "money" present 30.8%. Smoothing view at alpha = 1: spam 6.150e-05, ham 7.688e-06,
P(spam) 0.8889; at alpha = 0 both scores are 0 and the lab says there is no answer (the table `bayes_4.py` prints).

Table view: the counts or factors behind each readout.
