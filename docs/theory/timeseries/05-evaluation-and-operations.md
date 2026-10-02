---
title: "Time series · Evaluation and operations"
sidebar_label: "Evaluation and operations"
sidebar_position: 5
slug: /theory/timeseries/evaluation-and-operations
description: "Rolling backtests, scaled errors, intervals, monitoring and anomaly detection."
tags: [time-series, evaluation, mlops]
---

import Infographic from '@site/src/components/Infographic';
import ForecastEvaluationLab from '@site/src/components/viz/ForecastEvaluationLab';

**In one line.** A forecast is useful when it improves a decision over a credible baseline at the required horizon and continues to do so after deployment.

## The idea in plain words

Forecasting evaluation is a replay of decisions made at past origins. At each origin, reconstruct the data and feature versions that were available, issue a forecast, and later compare it with the realised target. A single holdout period is better than a shuffled split, but several rolling origins show how performance changes with season, horizon and regime. The comparison must include a baseline under the same conditions. Without that control, an error of two units has no context: two may be excellent for one series and disastrous for another.

Point accuracy is only one dimension. Inventory planning may care more about underestimation, capacity planning may need a high quantile, and anomaly detection asks whether an observed value is unusual under an expected range. Production adds latency, missing-feature handling, retraining cadence, incident response and auditability. An evaluation report should make those choices explicit rather than compress everything into one leaderboard number.

<Infographic src="/img/timeseries/forecast-evaluation.svg" alt="Rolling forecast origins feed error calculation; a test absolute error of one divided by a training naive scale of five thirds gives MASE 0.6, while an illustrative interval from 11 to 15 is shown separately." caption="Scaled point error and interval coverage answer different questions." />

## How it works

### Replay rolling origins

Choose origins $o_1,\ldots,o_k$ within a validation period. At each, train or update the model only using information available through that origin, predict the specified horizons, and score after outcomes arrive. An expanding training window reflects accumulating history. A sliding window reflects a fixed retention period and may react faster to regime changes. The choice affects both performance and compute cost. Refit at the real operational cadence. A model that is retrained monthly should not receive a fresh fit at each daily validation origin unless the production plan changes.

Use a separate model-selection period and a final untouched test period. If many architectures, lags and hyperparameters are tried against the same final period, that period becomes part of model selection. Preserve predictions by model version, origin, horizon, entity and feature snapshot. These records permit paired comparisons at the same origins and later incident analysis. Stratify results by volume, entity age, geographic region, peak events and missing-data frequency. An average can improve while a high-cost slice worsens.

### Select metrics for the decision

Mean absolute error, $\mathrm{MAE}=n^{-1}\sum_i|y_i-\hat y_i|$, retains the target's units and is interpretable. Root mean squared error penalises large misses more strongly. Mean absolute percentage error divides by the actual value and becomes undefined at zero; it can behave badly for low-volume series. A scaled error can compare series of different magnitudes. For nonseasonal MASE, the denominator is the mean absolute successive difference **on the training series**, not on the test series:

$$\mathrm{MASE}=\frac{\operatorname{mean}_{i\in test}|y_i-\hat y_i|}{\frac{1}{T-1}\sum_{t=2}^{T}|y_t-y_{t-1}|}.$$

For training values $[10,12,11,13]$, the successive absolute differences are $[2,1,2]$, so the scale is $5/3$. If test MAE is one, MASE is $1/(5/3)=0.6$. This is the lab default. MASE below one means the evaluated test error is smaller than the mean in-sample naive difference used as scale; it does **not** prove that a model beat the naive forecast on the same test origins. To make that claim, compute both test errors directly. If the training denominator is zero because the series is constant, MASE is undefined and needs an explicit fallback or a different comparison. A seasonal MASE uses a seasonal naive denominator with the chosen period.

When errors have asymmetric cost, a quantile loss or a simulated business objective may be more relevant than symmetric MAE. For example, understaffing by one person and overstaffing by one person can have different service and wage costs. Define those costs with stakeholders, then calculate them on held-out outcomes. A threshold chosen after looking at the test period needs a new test period. Always report both a technical error and a decision-oriented outcome when the latter can be measured.

### Evaluate uncertainty

A prediction interval states a range intended to contain the future outcome with a specified frequency under its assumptions. An interval $[11,15]$ centred on 13 with radius two is only geometry until a method for choosing the radius and evidence of coverage are supplied. The lab labels this interval **illustrative**. To evaluate a 90% interval, count coverage across held-out predictions at each horizon and inspect whether it is close to 90%, along with interval width. Overly wide intervals cover almost everything but may be useless; narrow intervals can miss too often. Check coverage by group and regime, not just in aggregate.

Distributional models may produce quantiles directly. Residual-based calibration uses errors from a calibration period to set widths, but residual exchangeability can fail after a regime change. Prediction intervals usually widen with horizon because uncertainty accumulates. A time-series interval should be backtested at its actual horizon and issued with the same feature availability and model-update process as serving. A chart of observed coverage against nominal coverage helps identify overconfidence.

### Monitor after release

Forecast outcomes are delayed by the horizon, so immediate monitoring needs both input and system signals. Track missing and late features, schema changes, output volume, range checks, inference failures and latency. Once labels mature, track error, bias and coverage by horizon and slice, compared with both the release baseline and a seasonal naive baseline. A sustained signed bias can be more costly than a similar absolute error with balanced signs. Investigate changes in target definition, data collection, pricing, supply constraints and user behaviour before assuming a model refresh is sufficient.

Residuals can also drive anomaly detection. Given an origin-aware forecast and an uncertainty estimate, flag an observation when its residual exceeds a threshold calibrated on a suitable historical period. A fixed threshold in raw units treats high- and low-volume series differently; a scaled residual can help, but only if scale is stable and nonzero. A forecast anomaly is not automatically a data error or a business incident. Stockouts, promotions and sensor outages can all produce large residuals for different reasons. Combine an alert with context, severity, deduplication and an investigation path. Evaluate alert precision, timeliness and operator burden where labelled incidents exist.

## A real system that works this way

The [Forecasting: Principles and Practice accuracy chapter](https://otexts.com/fpp3/accuracy.html) defines MASE and compares it with other accuracy measures; its [time-series cross-validation chapter](https://otexts.com/fpp3/tscv.html) sets out rolling-origin assessment. A practical supply service can use that evaluation shape to compare a current forecaster against seasonal naive at every daily issue time. It would record predictions before outcomes are known, let the horizon mature, then publish an error and coverage dashboard by product family. An alert on a large residual would route to operations with the recent feature and inventory state, not merely display a red point.

## Code you can run

```python
training = [10, 12, 11, 13]
training_naive_errors = [abs(training[i] - training[i - 1]) for i in range(1, len(training))]
scale = sum(training_naive_errors) / len(training_naive_errors)
test_mae = 1.0
print('training naive errors:', training_naive_errors)
print('training scale:', round(scale, 6))
print('MASE:', round(test_mae / scale, 3))
```

The output is errors `[2, 1, 2]`, scale `1.666667` and MASE `0.6`. It is a scaled-error calculation, not an evaluation of a fitted forecasting model.

<ForecastEvaluationLab />

Move test error, forecast centre and interval radius independently. The default error of one reproduces MASE 0.6. The default centre 13 and radius two give the illustrative interval `[11, 15]`; changing the radius does not manufacture a coverage guarantee.

```python
forecasts = [13, 12, 15, 11]
actuals = [12, 13, 14, 12]
radius = 2
inside = [abs(actual - forecast) <= radius for actual, forecast in zip(actuals, forecasts)]
coverage = sum(inside) / len(inside)
mean_width = 2 * radius
print('covered:', inside)
print('sample coverage:', coverage)
print('mean interval width:', mean_width)
```

This tiny synthetic sample has coverage one and width four. Four examples cannot establish calibrated coverage. In a real report, use many held-out origins and show uncertainty in the coverage estimate, separated by horizon and important segments.

## Designing with it

Before modelling, write down a primary metric, guardrail metrics, baselines, horizons and slices. Specify how zero targets, missing outcomes, censored sales and changed entity identifiers are handled. A stockout can make observed sales lower than unconstrained demand, so scoring against sales may reward a model for missing demand. If the forecast drives an intervention, record that intervention; otherwise a feedback loop can make later labels hard to interpret. Keep raw predictions as well as rounded or clipped decisions so model error can be separated from postprocessing.

For deployment, define fallbacks for missing inputs and timeouts. A seasonal-naive fallback is often easier to audit than a stale cached forecast whose origin is unclear. Version the feature pipeline, training data, model, calibration method and decision rule. Monitor them separately. When an incident occurs, replay the original origin with its original inputs before retraining; otherwise a corrected data snapshot can conceal the failure. Schedule periodic re-evaluation of the baseline and interval calibration even when the average point error appears stable.

## Where this stands in 2026

Forecasting systems increasingly compare classical, tabular and pretrained candidates, but evaluation still depends on chronological replay and operational constraints. A model with the best public benchmark rank may be slower, miscalibrated or weaker on the organisation's rare high-cost cases. Current practice also demands uncertainty and monitoring because forecasts are used to make decisions, not merely to fill a chart. The test of an improvement is a matched-origin result with trustworthy data and a decision outcome, followed by production monitoring.

## Practice questions

<details>
<summary>Why is MASE's denominator calculated from training data?</summary>

It defines a fixed scale that is available before forecasting and avoids using future outcomes to set the scale. Test errors form the numerator.

</details>

<details>
<summary>Does MASE 0.6 prove that the model beats seasonal naive on the test period?</summary>

No. It compares test MAE with the training nonseasonal naive scale. Calculate the seasonal naive test errors at the same origins for that claim.

</details>

<details>
<summary>Why does an interval with 100% observed coverage still need inspection?</summary>

It may be too wide to support decisions, and a small sample can give misleading coverage. Report interval width, number of cases, horizon and slice.

</details>

<details>
<summary>What can be monitored before delayed ground truth arrives?</summary>

Feature freshness and missingness, schema, latency, output range, fallback use and prediction volume. Error and interval coverage follow after the forecast horizon matures.

</details>

## Further reading

- [Forecast accuracy](https://otexts.com/fpp3/accuracy.html) defines scale-based metrics and their limits.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) explains rolling origins.
- [Distributional forecasts and prediction intervals](https://otexts.com/fpp3/prediction-intervals.html) develops uncertainty evaluation.

## Check yourself

- I can reconstruct a forecast using only data available at its origin.
- I can compute the example MASE and explain what it does and does not compare.
- I can distinguish an illustrative interval from a calibrated one.
- I can design point-error, coverage, cost and operational monitoring by horizon.
- I can explain why a large forecast residual is an investigation signal rather than an automatic incident label.
