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

:::tip Before you start
**You should already know**

- Rolling origins and MASE ([Classical forecasting](/docs/theory/timeseries/classical-forecasting)).
- How a forecast model is trained on lag features ([Lagged machine learning](/docs/theory/timeseries/lagged-machine-learning)).

**Reading time:** about 45 minutes, plus the code.

**After this chapter you can**

- measure whether an 80 per cent interval really covers 80 per cent of outcomes, by lead day,
- build split-conformal intervals from past forecast errors and check them,
- set up a CUSUM monitor on forecast residuals and measure its delay and false alarms.

:::

## In 30 seconds

A point forecast says "about 100". A planner needs to know "between 90 and 110, nine times out of ten". That range is only worth something if it is right nine times out of ten, so you count. After release the question changes: has the forecaster started to miss? A monitor keeps a running tally of how far recent errors drifted to one side and raises a flag when the tally gets too big. Think of a bathroom scale that is always off by a bit: one reading means nothing, but ten readings off in the same direction mean the scale is broken.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Prediction interval | A range that should contain the outcome with a stated frequency. | 80 per cent: 8 of 10 outcomes inside. |
| Coverage | The share of outcomes that fell inside the interval. | 0.82 observed against 0.80 nominal. |
| Calibration window | Past forecasts used to set the interval width. | The first 14 origins. |
| Conformal interval | The forecast plus or minus a quantile of past absolute errors. | 13 plus or minus 3. |
| Residual | Actual minus the one-step forecast. | 12 against a forecast of 10 is +2. |
| CUSUM | A running sum of residuals above a small allowance, floored at zero. | 0, 1.0, 1.7, 3.0, 4.5. |
| Detection delay | Days between a real change and the alarm. | Shift on day 60, alarm on day 63, delay 3. |
| False alarm | An alarm when nothing changed. | 5 of 60 runs alarmed before the shift. |

## The idea in plain words

Forecasting evaluation is a replay of decisions made at past origins. At each origin, reconstruct the data and feature versions that were available, issue a forecast, and later compare it with the realised target. A single holdout period is better than a shuffled split, but several rolling origins show how performance changes with season, horizon and regime. The comparison must include a baseline under the same conditions. Without that control, an error of two units has no context: two may be excellent for one series and disastrous for another.

Point accuracy is only one dimension. Inventory planning may care more about underestimation, capacity planning may need a high quantile, and anomaly detection asks whether an observed value is unusual under an expected range. Production adds latency, missing-feature handling, retraining cadence, incident response and auditability. An evaluation report should make those choices explicit rather than compress everything into one leaderboard number.

<Infographic src="/img/timeseries/forecast-evaluation.svg" alt="Four cards describe rolling origins, training naive scale five-thirds and MASE 0.6, an illustrative interval from 11 to 15, and production feedback monitoring." caption="Scaled point error and interval coverage answer different questions." />

## Worked example, step by step

**A conformal interval.** Five past forecast errors are 1, -2, 0.5, 3 and -1. We want an 80 per cent interval around a new forecast of 13.

1. **Absolute errors** are 1, 2, 0.5, 3, 1. Sorted: 0.5, 1, 1, 2, 3.
2. **Rank.** With n = 5 errors and a target of 0.8, take the smallest value whose rank is at least 0.8 x (n + 1) = 4.8, so rank 5. That is 3.
3. **Interval.** 13 plus or minus 3 is 10 to 16.

**A CUSUM monitor.** Standardised residuals arrive as 0.2, 1.5, 1.2, 1.8, 2.0. Use an allowance k = 0.5 and an alarm level h = 4. The sum is updated as the larger of 0 and (previous sum + residual - k).

1. 0 + 0.2 - 0.5 is negative, so the sum is 0.
2. 0 + 1.5 - 0.5 = 1.0.
3. 1.0 + 1.2 - 0.5 = 1.7.
4. 1.7 + 1.8 - 0.5 = 3.0.
5. 3.0 + 2.0 - 0.5 = 4.5, which is above 4. Alarm at step 5.

In words: the conformal interval is as wide as the past errors that 80 per cent of the time stayed below it. The CUSUM ignores small wobbles because of the allowance and flags a run of small errors on one side that a one-off threshold would miss.

```python
import numpy as np

calibration_errors = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
rank = int(np.ceil((len(calibration_errors) + 1) * 0.8))
radius = float(np.sort(np.abs(calibration_errors))[rank - 1])
print('rank used:', rank, ' radius:', radius, ' interval for forecast 13:', (13 - radius, 13 + radius))

z = np.array([0.2, 1.5, 1.2, 1.8, 2.0])
s, path = 0.0, []
for value in z:
    s = max(0.0, s + value - 0.5)
    path.append(round(float(s), 2))
print('CUSUM path:', path, ' alarm at step', next(i + 1 for i, v in enumerate(path) if v > 4))
```

It prints rank 5, radius 3.0, the interval `(10.0, 16.0)`, the sum path `[0.0, 1.0, 1.7, 3.0, 4.5]` and an alarm at step 5, matching the steps above.

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

### Experiment 1: do the intervals cover what they promise?

Eight synthetic daily series of 600 days carry a weekly pattern, a small trend, and noise that is autocorrelated, heavy-tailed (a t distribution with 4 degrees of freedom) and 50 per cent larger on weekends. The block runs 28 rolling origins a week apart with a 7-day horizon. ETS and a fixed ARIMA(1,0,1)(0,1,1) with period 7 give 80 per cent intervals. The first 14 origins calibrate a split-conformal interval: for each lead day, the 80th percentile of absolute errors. The last 14 origins are the test.

```python
import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoETS

warnings.filterwarnings('ignore')
rng = np.random.default_rng(5)
n, horizon, windows, n_series = 600, 7, 28, 8
shape = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
shape -= shape.mean()
frames = []
for k in range(n_series):
    t = np.arange(n)
    scale = 3.0 * (1 + 0.5 * (t % 7 >= 5))
    shocks = rng.standard_t(4, n) * scale / np.sqrt(2)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + shocks[i]
    frames.append(pd.DataFrame({'unique_id': f's{k}', 'ds': pd.date_range('2022-01-01', periods=n, freq='D'), 'y': 100 + 0.03 * t + shape[t % 7] + noise}))
df = pd.concat(frames)
models = [AutoETS(season_length=7, model='ZZA'), ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7, alias='ARIMA')]
cv = StatsForecast(models=models, freq='D', n_jobs=1).cross_validation(df=df, h=horizon, step_size=horizon, n_windows=windows, level=[80]).reset_index()
cv['step'] = (cv['ds'] - cv['cutoff']).dt.days
cuts = sorted(cv['cutoff'].unique())
calib = cv[cv['cutoff'].isin(cuts[:windows // 2])]
test = cv[cv['cutoff'].isin(cuts[windows // 2:])]
for m in ('AutoETS', 'ARIMA'):
    inside = (test['y'] >= test[f'{m}-lo-80']) & (test['y'] <= test[f'{m}-hi-80'])
    width = (test[f'{m}-hi-80'] - test[f'{m}-lo-80']).mean()
    q = calib.assign(e=np.abs(calib['y'] - calib[m])).groupby('step')['e'].quantile(0.8)
    r = test['step'].map(q)
    conf = np.abs(test['y'] - test[m]) <= r
    print(m, 'model interval coverage %.3f width %.2f | conformal coverage %.3f width %.2f' % (inside.mean(), width, conf.mean(), 2 * r.mean()))
    print('  by step', [round(float(x), 2) for x in inside.groupby(test['step']).mean()])
```

**Reading the output.** On the 784 test forecasts per model, ETS intervals cover 0.885 of outcomes against a nominal 0.80, and are 13.69 wide on average. The conformal version covers 0.824 with width 11.57, which is 15 per cent narrower. ARIMA intervals cover 0.823 with width 10.92, and the conformal version 0.811 with width 10.28. The per-day list for ARIMA exposes what the average hides: 0.69 on day 1 and 0.74 on day 2, then 0.83 to 0.89 later.

**Line by line.**

- `level=[80]` asks statsforecast for an 80 per cent interval, and the `-lo-80` and `-hi-80` columns hold its ends.
- `groupby('step')['e'].quantile(0.8)` computes one radius per lead day from the calibration windows only, and `test['step'].map(q)` applies it to the test windows.
- The test windows come strictly after the calibration windows (`cuts[:windows // 2]` against the rest), the same ordering the model will face in production.

**Interpretation.** Averaged over all lead days, ARIMA looks calibrated (0.823 against 0.80) while its day-1 intervals miss one outcome in three. Because the origins are a week apart, each lead day lines up with one weekday, so the pattern probably reflects the larger weekend noise that a constant-variance interval cannot follow (not tested separately). ETS errs the other way, and wide intervals cost decisions as much as narrow ones. Conformal intervals fixed the ETS over-coverage, and left ARIMA essentially unchanged. A caution from a smaller development run (8 calibration windows, 10 series): conformal coverage came out at 0.709 for ETS and 0.741 for ARIMA, because the calibration errors happened to be smaller than the test errors. The method needs enough calibration windows, and it assumes the recent past resembles the near future. The limits: synthetic data, one seed, 14 test windows, and a lead day standing in for a weekday.

<Infographic src="/img/ts-enrich/intervals-cusum.svg" alt="Left: bars of ARIMA 80 per cent interval coverage by lead day, 0.69 and 0.74 on days 1 and 2 and 0.83 to 0.89 after. Right: a table of CUSUM and threshold alarm results for level shifts of 3, 6 and 10 units." caption="On the left, look at days 1 and 2 against the dashed 0.80 line. On the right, compare the delay columns row by row." />

### Experiment 2: a residual monitor and its detection delay

Here a SARIMAX model is fitted once to 365 days of a weekly series, its parameters are frozen, and one-step residuals are standardised by the training residual spread (2.16, against a true innovation spread of 2.00). A level shift of 3, 6 or 10 units is added on day 60 of 180 live days. Two rules watch the residuals: CUSUM with allowance 0.5 and alarm level 5, and a plain rule that alarms when one residual exceeds 3 standard deviations. Each setting is run on 60 simulated series.

```python
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')
shape = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
shape -= shape.mean()
n_train, n_live, shift_day, sims = 365, 180, 60, 60

def make(seed, shift):
    rng = np.random.default_rng(seed)
    n = n_train + n_live
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 2.0)
    t = np.arange(n)
    y = 100 + 0.03 * t + shape[t % 7] + noise
    y[n_train + shift_day:] += shift
    return y

model = SARIMAX(make(0, 0)[:n_train], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
sigma = np.std(model.resid[8:])
print('training one-step residual std %.2f (true innovation std 2.00)' % sigma)

def first_alarm(z, rule):
    s = 0.0
    for i, value in enumerate(z):
        s = max(0.0, s + value - 0.5)
        if (rule == 'cusum' and s > 5.0) or (rule == 'threshold' and abs(value) > 3.0):
            return i
    return None

print('shift  rule       false alarms  caught  median delay (days)')
for shift in (0.0, 3.0, 6.0, 10.0):
    residuals = [model.apply(make(seed, shift)).filter_results.forecasts_error[0][n_train:] / sigma for seed in range(1, sims + 1)]
    if shift == 6.0:
        mean_z = np.mean(residuals, axis=0)
    for rule in ('cusum', 'threshold'):
        alarms = [first_alarm(z, rule) for z in residuals]
        early = sum(a is not None and a < shift_day for a in alarms)
        delays = [a - shift_day for a in alarms if a is not None and a >= shift_day]
        median = f'{np.median(delays):.0f}' if delays and shift else '-'
        print(f'{shift:>5}  {rule:<9} {early:>8}/{sims}  {len(delays) if shift else 0:>4}/{sims}  {median:>8}')

print('mean standardised residual, shift of 6: before the shift %.2f, first day %.2f, days 0 to 6 %.2f, days 14 to 20 %.2f' % (
    mean_z[:shift_day].mean(), mean_z[shift_day], mean_z[shift_day:shift_day + 7].mean(), mean_z[shift_day + 14:shift_day + 21].mean()))
```

**Reading the output.** In the no-shift rows, 5 of 60 runs trigger a CUSUM alarm before day 60 and 13 of 60 trigger the threshold rule: false alarms. For a shift of 3 units (1.4 standard deviations) CUSUM catches 54 of the 55 runs with no early alarm, after a median of 10 days, while the threshold rule catches 18 after a median of 27 days. At 6 units CUSUM takes 3 days and the threshold rule 0 days, but the threshold rule catches 36. At 10 units, CUSUM takes 1 day and the threshold rule 0 days. The last line shows the standardised residual after a 6-unit shift: 0.23 on average before it, 2.79 on the first day, 1.66 over the first week and 0.96 on days 14 to 20.

**Line by line.**

- `model.apply(y)` runs the fitted model over a new series with the parameters frozen, and `forecasts_error` holds the one-step errors, each computed only from earlier data.
- `first_alarm` returns the index of the first alarm, so an alarm before `shift_day` counts as false and one after it as a detection.
- The same 60 seeds are reused for every shift size, so the early false alarms are the same 5 and 13 in every block of rows. They are not fresh evidence each time.

**Interpretation.** The threshold rule is fastest for large shifts, but it fires more often when nothing changed (13 against 5) and misses small shifts. CUSUM trades a few days of delay for sensitivity: it catches a 1.4-sigma shift that the threshold rule mostly sees too late. The residual itself shrinks from 2.79 to about 0.96 two weeks after the shift, probably because the model's seasonal differencing and error term pull its forecasts towards the new level (this run measured the shrinking, not the cause). A monitor on forecast residuals therefore sees a change as a short-lived signal, and a slow alarm may never fire. Pick the allowance and alarm level on the false-alarm rate you can afford. The limits: simulated Gaussian-like noise, one model, one shift day, 60 runs and early alarms excluded from the caught counts.

## Designing with it

### Aggregate errors without hiding weak groups

Suppose a forecast service covers one high-volume product and hundreds of slow-moving ones. A pooled MAE weighted by observations may be dominated by the popular product. An unweighted mean of per-product MASE values gives rare products a stronger voice, but becomes unstable where the training scale is near zero. Report several views: total decision-weighted cost, per-series median and tail, and slices by volume and age. Explain which view is primary. A metric's denominator is part of the claim, and changing the set of eligible series between model versions can produce an apparent improvement without any better predictions.

Hierarchical forecasts add coherence. If store-level predictions sum to 120 but the region-level forecast says 110, an operator cannot use both without a reconciliation rule. Evaluate at the level where decisions occur and at the aggregate levels needed for planning. Reconciliation can improve coherence while slightly worsening one level's point error. Show that trade-off explicitly. For a service forecasting multiple related series, include both per-series error and error on the summed total; correlated mistakes can make the total much less reliable than the individual results suggest.

### Use intervals to make decisions

An interval can help choose a safety buffer. If a planner orders to cover a high demand quantile, the exact cost of overstock and understock determines which quantile is useful. A nominal 90% interval is not a safety policy by itself. The 5th and 95th percentiles may be too wide, too narrow or asymmetric relative to the decision cost. Backtest the complete policy: forecast distribution, chosen order quantity, realised demand, waste and stockouts. If only sales are observed after a stockout, demand is censored, so the outcome data may understate missed demand. Record stockouts and avoid claiming a cost improvement from an incomplete label.

Anomaly alerts also need a decision threshold. A threshold can be set on a scaled residual, a predictive tail probability or an empirical calibration set. Then measure how many alerts operators can inspect and how often they lead to action. Repeated alerts from one broken sensor should be grouped, while a new serious event should not be suppressed by a broad deduplication rule. Include a path for human feedback that distinguishes true event, data incident and expected planned change. This feedback can improve future labels without automatically treating every dismissed alert as a negative training example.

### Separate model drift from system drift

When errors rise, first verify data arrival and feature lineage. A delayed inventory feed can make an otherwise unchanged model fail; a changed product taxonomy can scramble group keys; a software update can shift a timezone boundary. Monitor these alongside statistical residuals. If inputs are sound, inspect whether the relationship between covariates and target changed, whether new products dominate traffic, and whether a policy intervention changed demand. Retraining on more recent data can help some shifts and worsen others. A rollback, fallback baseline or feature repair may be the correct immediate action.

Keep a forecast ledger with origin, horizon, entity, data snapshot, feature publication times, model and calibration versions, prediction, interval and later outcome. This record supports matched comparisons and incidents. It also prevents retrospective data corrections from rewriting what the service actually knew. A backtest generated from today's cleaned dataset is useful for development, but the stored production ledger is stronger evidence about real decision quality once enough outcomes mature. Make label maturity visible so a daily dashboard does not compare a seven-day horizon before seven days have passed.

### A release criterion

A candidate model should beat an appropriate baseline on the primary outcome across enough origins to cover expected cycles, meet interval or risk requirements, and stay within cost and latency limits. Its worst important slices should be reviewed, not silently averaged away. Before release, test missing-feature fallbacks and a replay of known difficult days. After release, monitor both forecast quality and system health and define a rollback trigger. This is a practical definition of a finished forecast: a traceable decision process that continues to work when data arrive late or the world changes.

Before modelling, write down a primary metric, guardrail metrics, baselines, horizons and slices. Specify how zero targets, missing outcomes, censored sales and changed entity identifiers are handled. A stockout can make observed sales lower than unconstrained demand, so scoring against sales may reward a model for missing demand. If the forecast drives an intervention, record that intervention; otherwise a feedback loop can make later labels hard to interpret. Keep raw predictions as well as rounded or clipped decisions so model error can be separated from postprocessing.

For deployment, define fallbacks for missing inputs and timeouts. A seasonal-naive fallback is often easier to audit than a stale cached forecast whose origin is unclear. Version the feature pipeline, training data, model, calibration method and decision rule. Monitor them separately. When an incident occurs, replay the original origin with its original inputs before retraining; otherwise a corrected data snapshot can conceal the failure. Schedule periodic re-evaluation of the baseline and interval calibration even when the average point error appears stable.

## Where this stands in 2026

Forecasting systems increasingly compare classical, tabular and pretrained candidates, but evaluation still depends on chronological replay and operational constraints. A model with the best public benchmark rank may be slower, miscalibrated or weaker on the organisation's rare high-cost cases. Current practice also demands uncertainty and monitoring because forecasts are used to make decisions, not merely to fill a chart. The test of an improvement is a matched-origin result with trustworthy data and a decision outcome, followed by production monitoring.

## Common mistakes

1. **Reporting one coverage number.** It feels complete: 0.823 is close to 0.80. Coverage by lead day was 0.69 on day 1. Check coverage by horizon, by weekday and by group.
2. **Trusting a model's own interval.** It feels safe because the library prints it. The interval assumes Gaussian, constant-variance errors, and the synthetic noise here was heavy-tailed and heteroskedastic. Measure coverage on held-out origins and recalibrate if it is off.
3. **Calibrating conformal intervals on too few windows.** It feels like a distribution-free guarantee. A development run with 8 windows covered only 0.709. Use many calibration windows and re-check coverage after calibrating.
4. **Alarming on a single residual.** It feels simple. Here 13 of 60 runs raised a false alarm and small shifts were missed. Use a running sum or a window, and choose its alarm level from the false-alarm rate.
5. **Treating a residual alarm as a data-quality alarm.** It feels like proof something broke. A large residual can be a promotion, a stockout or a sensor fault. Attach context and route it to an investigation, as described earlier.

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

<details>
<summary><strong>Q5 (Easy).</strong> In the worked example, why is the radius 3 and not 2?</summary>

With five errors and a target of 80 per cent, the rank is the ceiling of 0.8 x 6 = 4.8, which is 5. The fifth smallest absolute error is 3. Taking the fourth (2) would cover only four of the five past errors.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> ARIMA's overall coverage was 0.823 for a nominal 0.80. Why is that not enough to call it calibrated?</summary>

Coverage was 0.69 on day 1 and 0.74 on day 2, offset by over-coverage on later days. A planner who acts on the day-1 interval gets a miss one time in three. Check coverage by lead day and by weekday.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> The threshold rule had 13 false alarms in 60 runs and the CUSUM 5. How would you pick CUSUM parameters for a given false-alarm budget?</summary>

Simulate no-shift series with your residual spread, run the monitor with a grid of allowances and alarm levels, and record the share of runs with an alarm in a fixed period. Choose the most sensitive setting whose rate fits the budget, then measure delay on simulated shifts of the size you care about. The NIST handbook suggests an allowance of half the shift and an alarm level of about 4 or 5.

</details>

## Further reading

- [Forecast accuracy](https://otexts.com/fpp3/accuracy.html) defines scale-based metrics and their limits.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) explains rolling origins.
- [Distributional forecasts and prediction intervals](https://otexts.com/fpp3/prediction-intervals.html) develops uncertainty evaluation.

- [Forecasting: Principles and Practice, prediction intervals](https://otexts.com/fpp3/prediction-intervals.html) explains intervals from normal residuals and from bootstrapping, but does not cover conformal methods (opened 2026-10-08).
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511), by Angelopoulos and Bates, submitted 15 July 2021 and revised 7 December 2022 (opened 2026-10-08).
- [NIST/SEMATECH e-Handbook: CUSUM control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm) defines the tabular CUSUM with allowance k and decision interval h (opened 2026-10-08).
- [StatsForecast documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html) shows prediction intervals through `level` (opened 2026-10-08, statsforecast 2.1.1; statsmodels 0.15.0 for SARIMAX).

## Check yourself

- I can reconstruct a forecast using only data available at its origin.
- I can compute the example MASE and explain what it does and does not compare.
- I can distinguish an illustrative interval from a calibrated one.
- I can design point-error, coverage, cost and operational monitoring by horizon.
- I can explain why a large forecast residual is an investigation signal rather than an automatic incident label.
- I can compute a split-conformal radius by hand and measure its coverage by lead day on held-out origins.
- I can explain why an average coverage near the nominal level can hide a failing lead day.
- I can build a CUSUM monitor on forecast residuals, and measure its false-alarm rate and detection delay against a plain threshold.

## Where to go next

This is the last chapter of the time series group. Return to [lagged machine learning](/docs/theory/timeseries/lagged-machine-learning) to apply these checks to a boosted model, or continue to the [recommender systems](/docs/theory/recsys/feedback-and-objectives) group.
