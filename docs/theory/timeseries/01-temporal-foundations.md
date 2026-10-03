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

## The idea in plain words

A table of observations becomes a time series when its order carries information. Daily demand, hourly traffic and monthly revenue have different sampling intervals, but in each case the previous values and the calendar may help predict what comes next. Shuffling such a table destroys the problem: a record from Friday can no longer safely stand in for a record from Monday. The central engineering question is therefore not simply which model fits best. It is **what was known when the prediction had to be made**.

Write the observed value at time $t$ as $y_t$. A forecast issued after $t$ for $h$ steps ahead is $\hat y_{t+h\mid t}$. The vertical bar records the forecast origin. If an input was measured at $t+h$, it cannot be used on the right side of that prediction, even if it is present in the final training table. A calendar label for $t+h$ is known in advance. The realised sales at $t+h$ are not. A planned promotion might be known, but only if the plan really existed before the origin and was not later edited in place.

Three patterns often appear together. **Trend** is a sustained change in level, **seasonality** is a repeated pattern with a known or estimated period, and a **cycle** is a broader fluctuation without a fixed calendar period. These are descriptions of the data, not guarantees that a future period repeats. A weekly seasonal pattern can weaken after a product launch, a price change or a change in customer behaviour. Plotting the series with its timestamp, interval and missing observations is the first diagnostic step.

<Infographic src="/img/timeseries/temporal-foundations.svg" alt="Four cards summarise time-series patterns, a cutoff after six observations, leakage prevention and why random shuffling changes the forecasting task." caption="Forecast inputs are defined by the origin, not by which columns happen to exist in a completed dataset." />

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

## Designing with it

### Work backwards from a decision

Suppose a shop places tomorrow's order at 18:00. The forecast target is tomorrow's units sold, and the origin is 18:00 today. A daily aggregate labelled “today's sales” may not be final until midnight, so it is not necessarily an allowed input. A system could use sales through 17:45, an estimate of the remaining quarter-hour and the version of the promotion schedule published before 18:00. Every one of those inputs needs a timestamp. If a backtest silently uses the midnight total, it has given the model information that the ordering clerk did not have. The correct comparison is a replay of 18:00 snapshots, even if that makes the historical dataset less tidy.

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

## Further reading

- [Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/) develops decomposition, stationarity, forecast origins and baseline methods.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) shows rolling-origin evaluation.
- [scikit-learn: lagged features for time-series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) illustrates why time-aware validation changes model assessment.

## Check yourself

- I can explain why a forecast needs both an origin and a horizon.
- I can distinguish event time from feature availability time.
- I can draw a chronological split and compute a seasonal-naive forecast without looking ahead.
- I can explain when differencing helps and why it is not an automatic preprocessing step.
- I can identify at least three leakage paths in a historical feature table.
