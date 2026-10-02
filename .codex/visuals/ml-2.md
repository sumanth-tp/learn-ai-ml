# Track A, agent A2: lab specifications

Chapters: `docs/theory/ml/02-supervised-learning/01-regression-and-gradient-descent.md`, `02-classification-and-logistic-regression.md`, `03-decision-trees.md`.
All labs: `VizPanel`, `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies, no randomness at render time, native inputs only (range, select, checkbox, button) so everything is keyboard operable.

## GradientDescentLab (`src/components/viz/GradientDescentLab.tsx`)

Widget in the lecture: "Run gradient descent step by step and watch the cost fall."

Mode select: `One weight` (default) or `Two weights, scaling`.

### Mode One weight
- Data: x = [1, 2, 3], y = [1, 2, 2], model y-hat = theta * x, cost J = (1/2m) sum (theta x - y)^2.
- Controls: learning rate alpha 0.01 to 0.50 step 0.01 (default 0.10); steps taken 0 to 40 (default 1); start theta -1.0 to 2.0 step 0.1 (default 0.0).
- Drawn: left, the cost parabola J(theta) with the path of the steps (points joined by a line, last point emphasised, points off the chart are dropped and a note says so); right, J against step number (clipped at 8).
- Readout: gradient used by the last step, theta, J, stability limit 2 / (sum x^2 / m) = 0.4286 and whether the chosen alpha is inside it.
- Table: step, theta, gradient, J for steps 0 to the chosen step.
- Expected at defaults: gradient at theta 0 is -3.667, theta after step 1 is 0.367, J = 0.469 (chapter block 1). alpha 0.45 for 20 steps ends at theta -4.500, J 65.254; alpha 0.01 for 20 steps ends at 0.4836 (chapter block 3). Fixed point 0.7857, J 0.0595.

### Mode Two weights, scaling
- Data: eight rows (area m2, age years) = (60,12) (75,3) (90,18) (105,6) (120,15) (135,2) (150,9) (165,20); price = 180 260 235 310 300 395 380 390. Features and price are centred (equivalent to fitting the intercept); cost J(theta) = (1/2m) |Zc theta - yc|^2.
- Controls: checkbox "standardise features" (default off); alpha as a fraction of 1 / lambda_max, 0.1 to 1.9 step 0.1 (default 1.0); steps taken 0 to 100 (default 10).
- Drawn: elliptical contours of J (levels at 50%, 20%, 5%, 1% and 0.1% of the starting gap), the minimum, and the path of gradient descent from theta = (0, 0).
- Readout: eigenvalues of the Hessian, condition number, largest stable alpha 2 / lambda_max, alpha used, and the number of steps needed to close 99.9% of the gap (cap 20000, otherwise "does not converge").
- Table: step, theta1, theta2, share of gap remaining.
- Expected at defaults (raw): eigenvalues 38.29 and 1182.95, condition 30.90, largest stable alpha 0.0017, 72 steps. Standardised: 0.80 and 1.20, condition 1.51, 1.6629, 4 steps (chapter block 4).

## LogisticBoundaryLab (`src/components/viz/LogisticBoundaryLab.tsx`)

Widget in the lecture: "Move the score and threshold and watch the predicted class."

- Data: 80 points, 40 per class, from mulberry32(7) and Box-Muller, class 0 centred at (-1.0, -0.5), class 1 at (1.0, 0.5), standard deviation 1.1. Model fitted in the component by 4000 steps of batch gradient descent at 0.5 on cross-entropy from zero weights; no regularisation.
- Controls: decision threshold 0.05 to 0.95 step 0.01 (default 0.50); score z -6 to 6 step 0.5 (default 1.0).
- Drawn: scatter with the boundary theta0 + theta1 x1 + theta2 x2 = logit(threshold) and the predicted-positive side shaded, misclassified points ringed, classes told apart by marker shape as well as colour; ROC curve with the current operating point; sigmoid curve with the threshold and the probe z.
- Readout: TP, FP, FN, TN, precision, recall, F1, accuracy, AUC; sigma(z) and the predicted class for the probe.
- Table: threshold sweep 0.9, 0.7, 0.5, 0.3, 0.1 with TPR, FPR, precision.
- Expected at defaults: theta = (-0.133, 1.660, 0.866); TP 34, FP 7, FN 6, TN 33; precision 0.829, recall 0.850, F1 0.840, accuracy 0.838; AUC 0.9200; sigma(1.0) = 0.731 and class 1; z = -0.5 gives 0.378 and class 0 (chapter blocks 1 and 2).

## ImpurityLab (`src/components/viz/ImpurityLab.tsx`)

Widget in the lecture: "Impurity calculator: set the positive/negative counts and see entropy and Gini."

- Controls, node: positives 0 to 50 (default 9), negatives 0 to 50 (default 5). Controls, split: attribute select (Outlook default, Temperature, Humidity, Wind) from the 14-row play-tennis table; criterion select (entropy default, Gini).
- Drawn: entropy, Gini and misclassification error against the share of positives p, with the node marked; for the split, the child nodes as stacked positive/negative bars with their impurity, then the information gain of all four attributes as bars with the chosen one highlighted.
- Readout: p, entropy, Gini, error; for the split, parent impurity, weighted child impurity, gain.
- Table: entropy, Gini and error for the share of positives p across a grid that includes 9/14.
- Expected at defaults: [9+,5-] entropy 0.940, Gini 0.459; Outlook children [2+,3-] 0.971, [4+,0-] 0.000, [3+,2-] 0.971; weighted 0.694; gain 0.247. Gains by entropy: Outlook 0.247, Humidity 0.152, Wind 0.048, Temperature 0.029; by Gini: 0.116, 0.092, 0.031, 0.019 (chapter blocks 1 and 6).
