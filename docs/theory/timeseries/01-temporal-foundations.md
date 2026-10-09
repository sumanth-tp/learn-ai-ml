---
title: "Time series · Temporal foundations"
sidebar_label: "Temporal foundations"
sidebar_position: 1
slug: /theory/timeseries/temporal-foundations
description: "Time order, trend, seasonality, stationarity and leakage-safe forecasting splits."
tags: [time-series, forecasting, evaluation]
---

import Infographic from '@site/src/components/Infographic';
import TemporalSplitLab from '@site/src/components/viz/TemporalSplitLab';

**In one line.** A forecast is a prediction made with only the information available at its stated origin, for a stated future horizon.

:::tip Before you start
**You should already know**

- What a mean and a standard deviation are, and how a train and test split works ([Model evaluation](/docs/theory/ml/model-evaluation)).
- Why an input column must not contain the answer ([Features, leakage and imbalance](/docs/theory/ml/features-leakage-and-imbalance)).

**Reading time:** about 40 minutes, plus the code.

**After this chapter you can**

- split a daily series into trend, weekly and yearly parts and say what each part is,
- run a stationarity test without being fooled by seasonality or a trend,
- measure how much a random split flatters a forecasting model.

:::

## In 30 seconds

Sales rows are not independent. Last Monday tells you about next Monday, and this month's level tells you about next month's. So a forecast has to be judged the way it will be used: stand at a date (the origin), look only at what is already known, and predict a later date (the horizon). Think of a weather forecaster. Yesterday's newspaper is allowed. Tomorrow's newspaper is not. Shuffling the rows hands the model tomorrow's newspaper, and the score it earns is not a forecasting score.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Origin | The moment a forecast is made. | Friday 18:00, when orders close. |
| Horizon | How far ahead the forecast looks. | Seven days. |
| Trend | A slow, sustained change in level. | Up 0.04 units a day. |
| Seasonality | A pattern that repeats after a fixed period. | A weekend peak every 7 days. |
| Stationary | Behaviour that does not depend on the calendar date. | Noise around a fixed mean. |
| Differencing | Subtracting an earlier value from a later one. | $y_t - y_{t-7}$. |
| Leakage | An input that was not known at the origin. | Next week's sales used as a feature. |
| Seasonal naive | Forecast equal to the value one season ago. | Next Friday equals last Friday. |

## The idea in plain words

A table of observations becomes a time series when its order carries information. Daily demand, hourly traffic and monthly revenue have different sampling intervals, but in each case the previous values and the calendar may help predict what comes next. Shuffling such a table destroys the problem: a record from Friday can no longer safely stand in for a record from Monday. The central engineering question is therefore not simply which model fits best. It is **what was known when the prediction had to be made**.

Write the observed value at time $t$ as $y_t$. A forecast issued after $t$ for $h$ steps ahead is $\hat y_{t+h\mid t}$. The vertical bar records the forecast origin. If an input was measured at $t+h$, it cannot be used on the right side of that prediction, even if it is present in the final training table. A calendar label for $t+h$ is known in advance. The realised sales at $t+h$ are not. A planned promotion might be known, but only if the plan really existed before the origin and was not later edited in place.

Three patterns often appear together. **Trend** is a sustained change in level, **seasonality** is a repeated pattern with a known or estimated period, and a **cycle** is a broader fluctuation without a fixed calendar period. These are descriptions of the data, not guarantees that a future period repeats. A weekly seasonal pattern can weaken after a product launch, a price change or a change in customer behaviour. Plotting the series with its timestamp, interval and missing observations is the first diagnostic step.

<Infographic src="/img/timeseries/temporal-foundations.svg" alt="Four cards summarise time-series patterns, a cutoff after six observations, leakage prevention and why random shuffling changes the forecasting task." caption="Forecast inputs are defined by the origin, not by which columns happen to exist in a completed dataset." />

## Worked example, step by step

Take two weeks of daily sales, Monday to Sunday. Week one is 20, 22, 21, 23, 30, 38, 35. Week two is 24, 26, 27, 27, 34, 40, 39. We split each value into a level, a weekday effect and what is left over.

1. **Level of each week.** Week one adds up to 189, so its mean is 27. Week two adds up to 217, so its mean is 31. The level rose by 4.
2. **Remove the level.** Week one becomes -7, -5, -6, -4, 3, 11, 8. Week two becomes -7, -5, -4, -4, 3, 9, 8.
3. **Weekday effect.** Average the two weeks day by day: -7, -5, -5, -4, 3, 10, 8. Saturday sits 10 above the level, Monday 7 below.
4. **Remainder.** Subtract the weekday effect from each deviation. Week one leaves 0, 0, -1, 0, 0, 1, 0. Week two leaves 0, 0, 1, 0, 0, -1, 0.
5. **Check one value.** Wednesday of week two is 31 + (-5) + 1 = 27, which is the number we started with.

In words: the trend says where the level is, the seasonal part says how this weekday differs from the level, and the remainder is what neither explains. STL (seasonal-trend decomposition using LOESS) does the same job, but it smooths the level and the weekday effect instead of using plain averages, so they may drift over time.

```python
import numpy as np

two_weeks = np.array([20, 22, 21, 23, 30, 38, 35, 24, 26, 27, 27, 34, 40, 39], dtype=float)
weeks = two_weeks.reshape(2, 7)
trend = weeks.mean(axis=1)
weekday_effect = (weeks - trend[:, None]).mean(axis=0)
remainder = weeks - trend[:, None] - weekday_effect
print('weekly means (trend):', trend)
print('weekday effects:', np.round(weekday_effect, 2))
print('remainder week 1:', np.round(remainder[0], 2))
print('remainder week 2:', np.round(remainder[1], 2))
```

It prints trend `[27. 31.]`, weekday effects `[-7. -5. -5. -4.  3. 10.  8.]` and the two remainder rows from step 4. The decomposition below uses the same idea on three years of data.

## How it works

### Define the forecasting contract

Record the target, unit, sampling cadence, horizon and decision time before constructing features. “Tomorrow's total demand at the end of today” is different from “demand in the next hour, updated every five minutes.” The first may use today's final aggregate; the second can use only events ingested by the five-minute deadline. If the decision is replenishment, forecast the quantity needed by that decision, including lead time. If the decision is staffing, an hourly distribution may be more useful than a daily point estimate. Multiple horizons can require different features, metrics and business penalties.

An irregular event stream needs an explicit aggregation rule. Decide the timezone, daylight-saving treatment, meaning of an absent row and whether the target is a sum, average or last observed state. Filling a missing measurement with zero says that zero happened. For sales, that may confuse “store closed,” “no sales” and “telemetry missing.” Preserve a missingness indicator when this distinction matters. If a measurement arrived late, distinguish its **event time** from its **availability time**. A backtest should replay the availability known at the origin, not the polished table available today.

### Separate components without treating them as laws

An additive description writes $y_t=T_t+S_t+R_t$, with trend $T_t$, seasonal component $S_t$ and remainder $R_t$. It is a useful diagnostic when seasonal amplitude stays roughly constant. A multiplicative description lets the seasonal amplitude vary with the level; a log transform can sometimes make that structure easier to model when values are strictly positive. Neither form is a promise that a fitted decomposition will extrapolate. A decomposition estimated from the full series can leak future levels into training features. Fit transforms and seasonal estimates inside each training window.

**Stationarity** describes stability of a stochastic process, not the visual flatness of a plot. In weak stationarity, mean and variance are constant over time and covariance depends on lag rather than calendar date. A trending series generally violates this assumption. Differencing, $\nabla y_t=y_t-y_{t-1}$, may remove a changing level; seasonal differencing, $y_t-y_{t-m}$, may remove a repeated pattern of period $m$. Both discard information and can amplify noise. Inspect transformed data, residual dependence and the needs of the chosen model before applying them. A tree with explicit lag and calendar features does not require stationarity in the same way that a classical autoregressive derivation may.

### Split in time

Reserve the newest segment as a final test period. Within the earlier data, use one or more chronological validation origins. For an origin $o$, train on observations available through $o$, issue forecasts for $o+1$ through $o+h$, and compare those forecasts with observations only after they arrive. Move the origin and repeat. An expanding window accumulates history; a sliding window keeps a fixed recent span when older behaviour is less relevant. Neither design permits a random row split to stand in for future performance.

The simplest useful competitors are often **naive** forecasts. The last-value forecast repeats $y_o$; the seasonal naive forecast repeats $y_{o+h-m}$ when a complete prior season exists. They are quick to implement, surprisingly competitive and sensitive to the correct seasonal period. A model that cannot beat a suitable baseline may still be useful if it gives intervals or improves the costly cases, but that claim needs evidence from the same origins and horizon. The fixed lab series is $[10,12,11,13,10,12,11,13]$. At origin six, period-four seasonal naive predicts $[11,13]$ for the two held-out observations.

### Watch for temporal leakage

Leakage is an availability error, not only a train/test overlap. A centred moving average uses future observations. Scaling on the entire dataset imports future distribution statistics. A database join to the latest customer state can rewrite the past. Target encoding computed with later targets can quietly contaminate an earlier example. Even a public holiday flag can leak when the feature is actually a post-event “holiday impact” statistic. Build each feature from a reproducible snapshot at the forecast origin, and test its maximum source timestamp and publication lag.

For grouped series, decide whether a unit in validation also appears in training. Forecasting existing shops next month and forecasting a brand-new shop are different tasks. A split by time tests the first; a held-out-entity split additionally probes the second. If a global model learns across many entities, report both where both matter. Keep group identities, product launches and closures in the evaluation record so averages do not hide failures on young or sparse series.

## A real system that works this way

The [scikit-learn lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) turns a time-ordered demand series into lag and rolling-window predictors. It explicitly compares a shuffled split with time-aware evaluation; the latter better reflects forecasting. Its concrete lesson is about experimental design rather than a particular estimator: the same features can appear excellent under a split that lets neighbouring observations from the future influence training. In production, the corresponding pipeline needs an origin-aware feature job and a record of when each covariate became available.

## Code you can run

This synthetic example prints the lab's default forecast. Each block runs independently.

```python
series = [10, 12, 11, 13, 10, 12, 11, 13]
cutoff = 6
period = 4
train = series[:cutoff]
test = series[cutoff:]
prediction = [series[i - period] for i in range(cutoff, len(series))]
print('train:', train)
print('held out:', test)
print('seasonal naive:', prediction)
print('absolute errors:', [abs(a - b) for a, b in zip(test, prediction)])
```

Expected seasonal-naive output is `[11, 13]` and the two absolute errors are zero. That perfect score belongs only to this deliberately repeating eight-point illustration; it is not an expected production result.

<TemporalSplitLab />

Move the cutoff to see which rows become available. The table identifies every observation's split and only shows a seasonal baseline where a prior-season value exists. At the default cutoff of six, the chart and code both forecast 11 and 13.

```python
observations = [(1, 10), (2, 12), (3, 11), (4, 13), (5, 10), (6, 12)]
origin = 4
for time, value in observations:
    role = 'known at origin' if time <= origin else 'future at origin'
    print(time, value, role)
```

This availability check is intentionally small. In a real feature pipeline, replace `time` with both event timestamp and publication timestamp, then reject rows whose publication timestamp is later than the origin. A feature derived from a future observation remains unavailable even if its output column has been materialised.

### Experiment 1: what decomposition recovers, and what the stationarity test says

The series below has a known structure: a trend of 0.04 a day, a weekly pattern, a yearly sine wave of amplitude 9 and autocorrelated noise. MSTL (STL with several seasonal periods, from statsmodels) should recover the parts, and the augmented Dickey-Fuller (ADF) test should tell us whether differencing is needed.

```python
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
rng = np.random.default_rng(7)
n = 3 * 365
t = np.arange(n)
weekly = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
weekly -= weekly.mean()
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 2.0)
y = pd.Series(100 + 0.04 * t + 9.0 * np.sin(2 * np.pi * t / 365.25) + weekly[t % 7] + noise,
              index=pd.date_range('2021-01-01', periods=n, freq='D'))

fit = MSTL(y, periods=(7, 365), stl_kwargs={'robust': True}).fit()
profile = fit.seasonal['seasonal_7'].groupby(y.index.dayofweek).mean()
print('true weekday profile, Mon to Sun:', np.round(weekly[(np.arange(7) - 4) % 7], 1))
print('MSTL weekday profile, Mon to Sun:', np.round(profile.values, 1))
print('trend slope per day (true 0.04): %.4f' % np.polyfit(t, fit.trend.values, 1)[0])
print('remainder std %.2f, true noise std %.2f' % (fit.resid.std(), noise.std()))

def adf_p(series, regression):
    return round(adfuller(series, regression=regression, autolag='AIC')[1], 4)

deseasonalised = y - fit.seasonal.sum(axis=1)
print('ADF p, raw, constant only       :', adf_p(y, 'c'))
print('ADF p, raw, constant and trend  :', adf_p(y, 'ct'))
print('ADF p, deseasonalised, c and ct :', adf_p(deseasonalised, 'c'), adf_p(deseasonalised, 'ct'))
print('ADF p, first difference         :', adf_p(y.diff().dropna(), 'c'))
print('std: series %.2f, first difference %.2f, remainder %.2f' % (y.std(), y.diff().std(), fit.resid.std()))
```

**Reading the output.** The recovered weekday profile is within 0.2 of the true one (Monday -1.8 against -1.7, Wednesday 8.4 against 8.3), and the fitted slope is 0.0403 against the true 0.04. The remainder standard deviation is 1.55, lower than the true noise standard deviation of 2.24, because the slow part of the noise is absorbed into the trend and yearly curves. A remainder much larger than your noise estimate would mean a pattern is still in it.

The ADF null hypothesis is that the series has a unit root, which means it wanders and is not stationary. A small p-value rejects it. On the raw series p is 0.7356 with a constant and 0.6665 with a constant and trend: no rejection either way. On the deseasonalised series it is 0.929 with a constant only, but 0.0 (below 0.00005) once a linear trend is allowed.

**Line by line.**

- `periods=(7, 365)` names both cycles. the values must be the true periods, so the weekly cycle is 7 and the yearly cycle is 365.
- `stl_kwargs={'robust': True}` down-weights outliers when smoothing, so one bad day does not bend the weekday profile.
- `groupby(y.index.dayofweek).mean()` averages the drifting weekly component into one profile per weekday, so it can be compared with the true values.
- `regression='ct'` adds a linear trend to the test. Without it, a series that rises steadily is judged as if it should have a flat mean.

**Interpretation.** The deseasonalised series was never a random walk. It is a straight trend plus stationary noise, which is called trend-stationary. A constant-only ADF test calls it non-stationary (p of 0.929), and the habit of differencing whenever p is above 0.05 would then be wrong. First differencing does pass the test (p of 0.0), but it pays for it: the standard deviation goes from 1.55 in the remainder to 5.52 after one difference, because subtracting the previous value adds the previous value's noise. The right move here is to model the trend directly and leave the data undifferenced. The limits are plain: the series is synthetic, the noise is a single autocorrelated process with one seed, and a real series can have breaks that decomposition smears.

<Infographic src="/img/ts-enrich/leakage-inflation.svg" alt="Two panels of bars. With the time index as a feature, the seasonal naive error is 2.63, the random split error 2.07 and the honest time split error 3.76. Without it the random split is 2.16 and the time split 2.76." caption="Look first at the green and red bars in the left panel: the same model scores 2.07 on a random split and 3.76 on a time split." />

### Experiment 2: how much does a random split flatter a forecaster?

A gradient-boosted tree predicts the value seven days ahead from lags, a rolling mean, the calendar and, in one variant, the raw time index `t`. We score it on the last 20 per cent of the series (a time split) and by shuffled five-fold cross-validation, and compare with seasonal naive.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

rng = np.random.default_rng(7)
n = 3 * 365
t = np.arange(n)
weekly = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
weekly -= weekly.mean()
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 2.0)
y = pd.Series(100 + 0.04 * t + 9.0 * np.sin(2 * np.pi * t / 365.25) + weekly[t % 7] + noise)

horizon = 7
frame = pd.DataFrame({'target': y.shift(-horizon)})
for lag in (0, 1, 6, 13):
    frame[f'lag_{lag}'] = y.shift(lag)
frame['mean_7'] = y.rolling(7).mean()
frame['dow'] = (t + horizon) % 7
frame['doy'] = (t + horizon) % 365
frame['t'] = t
frame = frame.dropna()
target = frame.pop('target')
cut = int(len(frame) * 0.8)

def fit_score(columns, train_rows, test_rows):
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    model.fit(frame[columns].iloc[train_rows], target.iloc[train_rows])
    return mean_absolute_error(target.iloc[test_rows], model.predict(frame[columns].iloc[test_rows]))

rows = np.arange(len(frame))
test = rows[cut:]
print('test rows:', len(test), ' seasonal naive MAE: %.2f' % mean_absolute_error(target.iloc[test], frame['lag_0'].iloc[test]))
for name, columns in (('with t', list(frame.columns)), ('without t', [c for c in frame.columns if c != 't'])):
    honest = fit_score(columns, rows[:cut - horizon], test)
    shuffled = np.mean([fit_score(columns, a, b) for a, b in KFold(5, shuffle=True, random_state=0).split(rows)])
    print(f'{name:<10} time split MAE {honest:.2f}   random 5-fold MAE {shuffled:.2f}')
```

**Reading the output.** Seasonal naive, which simply repeats the value from the same weekday last week, scores a mean absolute error (MAE) of 2.63 on 215 test rows. With `t` in the features, the shuffled split reports 2.07, which is 21 per cent better than the baseline. The honest time split reports 3.76, which is 43 per cent worse. Without `t`, the two splits give 2.16 and 2.76.

**Line by line.**

- `y.shift(-horizon)` builds the target seven days ahead of the origin row, and `lag_0` is today's value, which is exactly the seasonal-naive forecast for that target.
- `rows[:cut - horizon]` stops the training rows seven days before the test starts, so no training target overlaps the first test origin.
- `KFold(5, shuffle=True)` is the leaky design: neighbouring days land on both sides of the split.

**Interpretation.** The shuffled split would have shipped a model that is 43 per cent worse than a one-line baseline. A likely cause is that a tree cannot extrapolate: it outputs averages of values it saw during training, and on a trending series the test period sits above most training rows. Dropping `t` narrows the gap, which fits that explanation, but this run did not test it directly. Removing `t` narrows the gap but does not close it (2.76 against 2.63), so on this series the boosted model with these features does not beat seasonal naive. That is a result about this setup. One seed, one split of 215 rows, no tuning of the model and a series with a steady trend make the trees look worse than they might on a flat series. The chapter on [lagged machine learning](/docs/theory/timeseries/lagged-machine-learning) shows how to give a tree level-free inputs.

## Designing with it

### Work backwards from a decision

Suppose a shop places tomorrow's order at the 18:00 cut-off. The forecast target is tomorrow's units sold, and the origin is 18:00 today. A daily aggregate labelled “today's sales” may not be final until midnight, so it is not necessarily an allowed input. A system could use sales through 17:45, an estimate of the remaining quarter-hour and the version of the promotion schedule published before 18:00. Every one of those inputs needs a timestamp. If a backtest silently uses the midnight total, it has given the model information that the ordering clerk did not have. The correct comparison is a replay of 18:00 snapshots, even if that makes the historical dataset less tidy.

Now change the decision to a warehouse order with a three-day lead time. The relevant target may be cumulative demand over the next three days, not only demand on day three. Summing three independent daily forecasts does not necessarily give a good interval for total demand because errors across days can be correlated. The warehouse may also care more about a high demand quantile than the mean. This illustrates why target definition, horizon and loss function should come from the decision before a feature list is assembled.

### A split is a claim about deployment

Consider two years of daily data. One final quarter can serve as the untouched test period. Earlier months can supply rolling validation origins spaced a week apart, each with a seven-day horizon. This arrangement tests several weekdays and some calendar variation. It still cannot prove resilience to a once-in-a-decade event. State what the historical period contains and what it does not. A validation result from only summer months is weak evidence about winter performance even if it includes thousands of rows.

An expanding window assumes that old data remain useful; a sliding window assumes recent behaviour is more representative. A changing price regime or a redesigned product can favour the latter. A sparse annual seasonal pattern may favour retaining older years despite drift. Test the choice with matched origins. If the model is retrained each Monday, the backtest should not refit every day. Otherwise the test rewards an update frequency that the production service will not have. Include a model version and training cutoff in each stored forecast record.

### Revisions and hierarchical series

Operational targets are often revised. A dashboard may first report provisional daily transactions and later replace them after refunds or delayed ingestion. Decide whether the model predicts the provisional number visible the next morning or the final settled number used for planning. If it predicts the final number, backtesting should preserve the original provisional inputs and compare with final targets only after they mature. Joining all historical rows to today's corrected table can disguise both data latency and a target-definition change.

Many series form hierarchies: item demand sums to category demand, and shops sum to regions. Separate models can produce totals that do not add up. Reconciliation can enforce consistency, but it may change the error at each level. Start by deciding which level drives the action, then report accuracy and coherence at that level. A category forecast can be accurate while missing individual products badly enough to cause stockouts. The right granularity is therefore part of the forecasting contract, not merely a chart setting.

### A compact readiness review

Before accepting a forecasting table, sample several rows and reconstruct their origins by hand. For each feature ask when its source event happened, when it was published, how it was aggregated and whether a correction could rewrite it later. Move one future target value and confirm that earlier feature rows stay unchanged. Check whether missing timestamps mean zero, closure or data loss. Then compute naive and seasonal-naive predictions on exactly the same origins as the proposed model. These checks are inexpensive and can prevent a sophisticated but invalid result from advancing to production.

Start by writing a forecast contract in one sentence and auditing every input against it. Use a plot for each material group, not only the aggregate. Choose a horizon aligned with an actual decision, and make the validation period long enough to contain the relevant calendar cycles. For a weekly pattern, a two-day holdout cannot demonstrate weekly generalisation. For an annual pattern, one year of data supplies very little evidence about changes from one year to the next. The amount of history should follow the decision, cadence and stability of the process.

When data are sparse, a seasonal baseline may be undefined or unstable. Fall back explicitly to a shorter-period, last-value or pooled baseline and record which cases used which rule. If a product has launched recently, prefer a method that shares information across related products only when similarity is defensible and the evaluation includes cold-start products. Never silently remove the hard cases from a metric denominator.

Backtesting is a simulation of past decisions. It remains imperfect if historical interventions would have changed under a different forecast, or if promotions were planned using earlier forecasts. Document such feedback. Keep an untouched final test window for a last check after model and feature selection. Once that window guides decisions, it is validation data and a new final test period is needed.

## Where this stands in 2026

Classical statistical models, feature-based machine learning and pretrained forecasting models all share the same origin and leakage constraints. More flexible models do not repair an invalid split. Foundation models can provide an additional baseline for series with little training data, but their advertised zero-shot setting still needs an evaluation using the target organisation's cadence, horizon and information policy. The latest provider descriptions are discussed in the pretrained-model chapter. For a new forecasting project, accurate data timestamps and strong naive baselines remain valuable before model selection.

## Common mistakes

1. **Shuffled cross-validation on a lag table.** It feels right because it is the default in every tutorial. Neighbouring rows share most of their inputs, so the model is scored on days it has effectively seen. Split by time and keep the origin explicit. In the experiment above it changed the verdict from 21 per cent better to 43 per cent worse than the baseline.
2. **Differencing because the ADF p-value is above 0.05.** It feels like the textbook step. Seasonality and a trend both raise the p-value, and differencing a trend-stationary series multiplies the noise (1.55 to 5.52 here). Remove the seasonal part first, test with a trend term, and compare the spread before and after.
3. **Giving a tree model the time index.** It feels like a way to hand over the trend. A tree cannot go beyond the range it trained on, so the forecast flattens or jumps (3.76 with `t`, 2.76 without). Give it differences, ratios or a detrended target instead.
4. **Fitting the decomposition on the whole series.** It feels harmless because the components look clean. The smoothed trend at the origin then depends on days after the origin. Refit inside each training window.
5. **Filling a missing day with zero.** It feels neutral. A zero says the shop sold nothing, which is different from closed or not recorded. Keep a missing-value flag.

## Practice questions

<details>
<summary>Why is a random train/test split misleading for a lagged forecasting table?</summary>

Nearby rows share much of their lag history. Random allocation can let training contain observations that occur after a validation origin, so it estimates interpolation on a completed history rather than prediction from the past.

</details>

<details>
<summary>At origin six in the example, what may a period-four baseline use?</summary>

It may use observations at times three and four to predict times seven and eight. It may not use the realised values at seven or eight when issuing the forecast. The predictions are 11 and 13.

</details>

<details>
<summary>Is a planned promotion a safe future covariate?</summary>

Only if the promotion plan was fixed and available at the prediction origin. A retrospectively edited final plan is a different, later signal. Store plan versions or their publication timestamps.

</details>

<details>
<summary>When would you test a held-out entity as well as a held-out time period?</summary>

When the system must forecast new products, users, shops or regions. A chronological split on existing entities alone does not measure that cold-start task.

</details>

<details>
<summary><strong>Q5 (Easy).</strong> The deseasonalised series gives an ADF p-value of 0.929 with a constant and 0.0 with a constant and trend. What does the difference tell you?</summary>

The series is a steady trend plus stationary noise. A constant-only test expects a flat mean, so the rising level looks like a unit root. Adding the trend term removes that false signal. Difference only when the test with the right deterministic terms still fails to reject.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> A colleague reports that a boosted-tree model beats seasonal naive by 21 per cent on shuffled cross-validation. Which three checks would you ask for before believing it?</summary>

First, a chronological split with the last window held out (here it gave 43 per cent worse). Second, the baseline scored on exactly the same test rows. Third, a list of the features with the time of the latest source value for each, to confirm none is later than the origin. A feature such as a raw time index also deserves a check on whether the test period lies outside the training range.

</details>

## Further reading

- [Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/) develops decomposition, stationarity, forecast origins and baseline methods.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) shows rolling-origin evaluation.
- [scikit-learn: lagged features for time-series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) illustrates why time-aware validation changes model assessment.

- [statsmodels MSTL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.MSTL.html) documents the multi-seasonal decomposition used above (opened 2026-10-08, statsmodels 0.15.0 was run).
- [statsmodels adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) states the null hypothesis and the `regression` options (opened 2026-10-08).
- [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) is the tree model used in Experiment 2 (scikit-learn 1.9.1 was run).

## Check yourself

- I can explain why a forecast needs both an origin and a horizon.
- I can distinguish event time from feature availability time.
- I can draw a chronological split and compute a seasonal-naive forecast without looking ahead.
- I can explain when differencing helps and why it is not an automatic preprocessing step.
- I can identify at least three leakage paths in a historical feature table.
- I can decompose a daily series into trend, weekday and remainder by hand and read the same parts from an MSTL fit.
- I can explain why an ADF test with a constant only can call a trend-stationary series non-stationary, and what to do instead of differencing.
- I can show that a random split flatters a lag-feature model and report the size of the gap against a seasonal-naive baseline.

## Where to go next

Continue with [classical forecasting](/docs/theory/timeseries/classical-forecasting), where naive, ETS and ARIMA are compared on rolling origins. For the leakage side of the same story, read [features, leakage and imbalance](/docs/theory/ml/features-leakage-and-imbalance).
