# Track A, agent 5: labs for 04-evaluation-and-practice

Shared maths lives in `src/components/viz/evalMath.ts` (new file, pure functions, no React). Every default below
reproduces a number printed by the chapter's own code.

## EvalConfusionMatrixLab (lecture widget: "Confusion-matrix calculator")

- Chapter: `docs/theory/ml/04-evaluation-and-practice/01-model-evaluation.md`
- Controls: four sliders TP, FP, FN, TN (0 to 1000, step 1) with matching number outputs; preset select:
  lecture (40, 10, 5, 45), question bank Q50 (40, 10, 20, 30), "always legit" (0, 0, 10, 990).
- Drawn: a 2 by 2 grid of the four counts (cell tint from the sequential ramp by share of total), and a horizontal
  bar for each of accuracy, precision, recall, F1, specificity. Undefined ratios (zero denominator) show "n/a".
- Default (40, 10, 5, 45): accuracy 0.850, precision 0.800, recall 0.889, F1 0.842, specificity 0.818 (the lecture's
  figures, printed by the first code block).
- Table view: metric, formula, value.

## RocPrLab

- Chapter: same. Model: negatives ~ N(0,1), positives ~ N(d,1), 1000 cases.
- Controls: separation d (0 to 4, step 0.1, default 1.5); prevalence (0.01 to 0.50, step 0.01, default 0.10);
  threshold (-2 to 4, step 0.1, default 1.0).
- Drawn: ROC curve (dot at the threshold, diagonal, AUC in the corner) next to the precision-recall curve (dot at
  the threshold, dashed baseline at the prevalence, area under it in the corner); beneath, expected TP, FN, FP, TN
  and precision at the threshold.
- Default: AUC 0.856, TPR 0.691, FPR 0.159, counts TP 69 / FN 31 / FP 143 / TN 757, precision 0.326, area under PR
  0.478 (printed by the ROC/PR block). Moving prevalence leaves the ROC unchanged and moves the PR curve.
- Table view: threshold sweep rows (threshold, TPR, FPR, precision).

## CalibrationLab

- Chapter: same (Beyond the lecture) and referenced from the capstone.
- Model: a score q is the model's stated probability; reality is sigmoid(slope * logit(q) + shift); outcomes use a
  fixed golden-ratio sequence so no randomness. A calibrator is fitted on the even-indexed half and scored on the
  odd-indexed half.
- Controls: slope (0.3 to 1.8, step 0.05, default 0.6); shift (-1.5 to 1.5, step 0.1, default 0);
  cases n (40 to 2000, step 20, default 500); method select none / Platt / isotonic (default none).
- Drawn: reliability diagram on the test half (ten equal-width bins, dot size by count, diagonal), the calibrator's
  mapping as a faint line, Brier and ECE read-outs for all three methods with the selected one marked.
- Default (n 500, slope 0.6, shift 0): none Brier 0.2109 ECE 0.0780; Platt 0.2039 / 0.0282; isotonic 0.2068 / 0.0219.
  At n 60: none 0.1861 / 0.1450; Platt 0.1846 / 0.0945; isotonic 0.2028 / 0.1372.
- Table view: per-bin count, mean predicted, observed fraction.

## ShapWaterfallLab

- Chapter: `02-explaining-predictions.md`.
- Data: the three applicants and their SHAP values printed by the SHAP code block (GradientBoostingClassifier on the
  synthetic credit data, base value -2.008 log-odds).
- Controls: applicant select (high risk, borderline, low risk; default high risk); units radio (log-odds,
  probability; default log-odds); order radio (by size, as listed; default by size).
- Drawn: a waterfall from the base value through each feature to the model output; bars red for risk-raising and
  green for risk-lowering (also signed in text); feature values on the row labels.
- Default (high risk): base -2.008, late_payments +1.394, debt_ratio +1.358, age +0.871, utilisation +0.526,
  income -0.412, tenure_years +0.029, output +1.758 log-odds = 0.853 probability.
- Table view: feature, value, SHAP.
