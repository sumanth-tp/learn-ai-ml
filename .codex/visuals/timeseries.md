# Time-series lab specifications

## TemporalSplitLab

- Controls: training cutoff 4 to 7 on the fixed eight-point synthetic series `[10,12,11,13,10,12,11,13]`; horizon is the remaining 1 to 4 observed points.
- Defaults: cutoff 6; training is the first six observations, validation `[11,13]`; the seasonal period-four baseline predicts `[11,13]`, exactly as chapter code prints.
- Drawing: ordered observations with a vertical cutoff and separate train/test colours. No future values may appear on the train side.
- Data view: time index, value, split and seasonal baseline where the required prior season exists.

## SmoothingLab

- Controls: exponential-smoothing alpha from 0 to 1 in 0.05 steps, fixed series `[10,12,11,13]` and initial level 10.
- Defaults: alpha 0.5; updated levels `[10,11,11,12]` and next forecast 12, matching chapter code.
- Drawing: observed values and successive smoothed levels.
- Data view: index, observation, prior level, updated level and next forecast.

## LagWindowLab

- Controls: target index 3 to 5 on fixed series `[10,12,11,13,10,12]`; window size 2 or 3.
- Defaults: target index 3, window size 3; lag one 11, lag two 12 and prior-three mean 11, matching chapter code.
- Drawing: highlight only inputs strictly earlier than the target and show the target separately.
- Data view: feature time index, value, role, computed lags and rolling mean.

## ForecastPatchLab

- Controls: patch width 1 to 3 and forecast horizon 1 to 4 on context `[10,12,11,13,10,12]`; optional known-future promotion flags are displayed but never used as target values.
- Defaults: patch width 2 gives three context patches, horizon 2 gives one masked target patch, matching chapter code.
- Drawing: separate observed context patches, hidden target horizon and labelled known-future covariate flags.
- Data view: patch ranges and values, context length, patch count, horizon and visibility policy.

## ForecastEvaluationLab

- Controls: test absolute error 0 to 4 in 0.1 steps and forecast centre 8 to 18 in 0.5 steps; reference training naive scale fixed at 5/3 from `[10,12,11,13]`; interval radius 0 to 4.
- Defaults: test error 1 yields MASE 0.600, forecast centre 13 and radius 2 give illustrative interval `[11,15]`, all printed by chapter code.
- Drawing: training scale, test error and their ratio plus an interval bar labelled illustrative rather than calibrated.
- Data view: training successive differences, scale, test MAE, MASE, centre, radius and interval endpoints.
