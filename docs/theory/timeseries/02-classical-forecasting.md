---
title: "Time series · Classical forecasting"
sidebar_label: "Classical forecasting"
sidebar_position: 2
slug: /theory/timeseries/classical-forecasting
description: "Naive baselines, exponential smoothing, ARIMA and seasonal models."
tags: [time-series, forecasting, arima]
---

import Infographic from '@site/src/components/Infographic';
import SmoothingLab from '@site/src/components/viz/SmoothingLab';

**In one line.** Classical forecasting models make explicit assumptions about level, trend, seasonality and serial dependence, then extrapolate those structures from observations available at the forecast origin.

## The idea in plain words

“Classical” describes a family of statistical tools, not an obsolete stage that every project must outgrow. A small, regular series with a stable calendar pattern may be easier to forecast and explain with a seasonal naive rule or an exponential smoothing model than with a large neural network. The trade-off is that a model's structure must match the data. A flat-level model cannot represent a persistent trend; a seasonal model needs enough cycles to estimate one; an autoregressive model can follow serial dependence but cannot infer an unobserved intervention by magic.

Begin with baselines. For a forecast issued after $T$, the mean method predicts the training mean, the naive method predicts $y_T$, a drift method extends the change between the first and last observations, and seasonal naive repeats the corresponding value from the previous season. Each gives a different claim about persistence. Compute each at the actual decision horizons before fitting a more elaborate model. A baseline is also a unit test: if a supposedly advanced model underperforms it, inspect leakage, horizon alignment, training windows and the business loss before adding complexity.

<Infographic src="/img/timeseries/classical-forecasts.svg" alt="Four cards compare naive baselines, a smoothed level of 12, structured ETS and ARIMA models, and evaluation at matched forecast origins." caption="Choose the recurrence that matches the series and decision horizon." />

## How it works

### Exponential smoothing updates a state

Simple exponential smoothing maintains a level $\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}$ for $0\leq\alpha\leq1$. With a large $\alpha$, recent observations have more influence. With a small $\alpha$, the level changes slowly. The forecast for every future horizon is $\ell_T$ in this simple form, so it is unsuitable when trend or seasonality is material. Initial level and $\alpha$ can be fitted on training data by minimising one-step residual loss. They are model parameters, not values to tune on a final test window.

For the fixed series $[10,12,11,13]$, initialise the level at 10 and use $\alpha=0.5$. The levels after each observation are $[10,11,11,12]$ and the next forecast is 12. This small example shows exactly what the lab calculates; it does not estimate the best alpha. The recurrence weights older observations geometrically, though the finite initial state still contributes weight. Holt's extension adds a trend state, and Holt-Winters methods add a seasonal state. These extensions can overreact to abrupt changes, and a long-horizon linear trend can become implausible. A damped trend limits extrapolation. An ETS model additionally makes an error assumption explicit so it can produce a predictive distribution rather than only a point.

### Autoregression models dependence

An autoregressive model of order $p$ predicts from $p$ earlier values plus an error term. A moving-average component of order $q$ describes dependence on earlier innovations, not an ordinary average of recent observations. ARIMA($p,d,q$) applies $d$ differences before modelling the resulting process. In seasonal ARIMA, $(P,D,Q)_m$ adds seasonal autoregressive, differencing and moving-average structure with period $m$. The period comes from the data cadence and domain, such as seven for a daily weekly pattern, not from a default setting.

An ARIMA workflow starts with plots, a baseline and a training-only transformation. Inspect autocorrelation and residuals, fit a small set of plausible orders, compare out-of-sample rolling-origin errors and check whether the remaining residual dependence is acceptable. Information criteria can help choose among fitted models, but they do not replace out-of-sample comparison at the horizon that matters. Auto-selection searches orders within constraints; a successful optimiser return is not evidence that a model handles a structural break or that its intervals are calibrated.

ARIMAX or SARIMAX adds external regressors. This is useful only if their future values are available or can themselves be forecast. For example, a published holiday calendar is a valid future input; actual future temperature is not. If the model is trained with realised future temperature and served with a weather forecast, validation should use historical weather forecasts to capture their error. Coefficients are conditional associations, not causal effects. A promotional indicator may correlate with demand precisely because a planner selected promotions when demand was expected to change.

### Compare residuals and intervals

For a one-step fitted forecast, residual $e_t=y_t-\hat y_{t\mid t-1}$ records what the model missed. Plot residuals over time and by season; a visible pattern suggests the model has left predictable structure. A zero mean by itself is weak evidence. Large outliers may reflect special events, recording errors or a changed regime. Avoid deleting them solely to improve a metric. A forecast interval adds a statement about uncertainty under model assumptions. The uncertainty of an $h$-step forecast usually grows with $h$, and a model-based interval can be miscalibrated when the process changes. Backtest empirical coverage by horizon and group.

For intermittent counts, a Gaussian error assumption may permit negative forecasts and a flat level may ignore zeros. Consider a count-aware method or a business rule, then evaluate the entire output distribution or the decision cost. Clipping negative values after training changes the error distribution and should be included in backtesting. The appropriate method is the one that improves the decision while meeting latency, interpretability and maintenance constraints.

## A real system that works this way

[Forecasting: Principles and Practice](https://otexts.com/fpp3/ses.html) uses an economic series to show how a fitted exponential-smoothing level produces a flat forecast when no trend or seasonal state is included. Its [ARIMA modelling chapter](https://otexts.com/fpp3/arima-r.html) shows the corresponding fitting and diagnostic workflow. A production planning service can run those same steps for each sufficiently long product series, retain the baseline alongside the fitted model, and switch to a pooled or simple rule for sparse products. Such a service also records the model version and origin so a later audit can reconstruct a prediction.

## Code you can run

```python
observed = [10, 12, 11, 13]
alpha = 0.5
level = observed[0]
levels = []
for value in observed:
    level = alpha * value + (1 - alpha) * level
    levels.append(level)
print('levels:', levels)
print('next forecast:', level)
```

The levels are `[10.0, 11.0, 11.0, 12.0]`; the next forecast is `12.0`.

<SmoothingLab />

Change alpha to see how much a new observation changes the state. At the default 0.5, the lab table and the code both end at 12. The chart is a calculation aid, not a fitted model comparison.

```python
series = [10, 12, 11, 13, 10, 12]
origin = 4
last_value = series[origin - 1]
seasonal_period = 4
next_two = [series[origin + step - seasonal_period] for step in range(2)]
print('naive:', [last_value, last_value])
print('seasonal naive:', next_two)
print('actual:', series[origin:origin + 2])
```

This prints naive `[13, 13]`, seasonal naive `[10, 12]`, and actual `[10, 12]`. The result illustrates why the correct season can matter; it is not a general performance claim.

## Designing with it

### Build a model ladder on one contract

Suppose an energy team must predict next-day hourly load from an 18:00 origin. First calculate last-hour, same-hour-yesterday and same-hour-last-week baselines. The 24-hour and 168-hour seasonal periods express different cycles; neither should be chosen just because it wins on a few convenient days. Evaluate several weeks of origins that include weekdays, weekends and unusual demand. If the team needs a 24-hour vector, record error at each lead hour as well as a total. A model can look good on average because it predicts overnight hours well while failing at the costly evening peak.

Next, fit a simple smoothing model to each hour or a seasonal model to the full hourly series. Simple exponential smoothing produces one level, so its unchanged multi-step forecast is an intentionally weak competitor when daily and weekly patterns are strong. Holt-Winters or seasonal ETS adds components that can reflect a repeating cycle. Keep the training window and origin aligned with the baselines. If a new tariff starts during validation, a seasonal component learned under the old tariff may systematically miss the new load shape. Diagnose that change before searching more parameters.

An ARIMA candidate asks a different question: after differencing and accounting for lags and errors, what dependence remains? The sample autocorrelation and seasonal structure can suggest small orders, but the final choice must survive rolling-origin comparison. A model with lower information criterion may not have the lowest peak-hour cost. Fit residual diagnostics on training data and inspect out-of-sample error by lead time. If residuals retain a weekly wave, the structure is incomplete; if only one holiday is badly wrong, an omitted calendar regressor or changed regime may be the issue.

### Understand state, updates and intervals

The smoothed level is a compact state of the past. At $\alpha=0$, it never updates after its initial value. At $\alpha=1$, it follows each newest observation and its next forecast becomes the last-value naive forecast. Between those extremes, the latest observation receives weight $\alpha$, the previous level receives $1-\alpha$, and older observations retain decaying influence. This explains both the stability and lag of a small alpha. The best alpha depends on the series and loss, and a value fitted on one regime can be unsuitable after a structural change.

Adding trend and season creates more states and more initialisation choices. A seasonal method needs enough cycles to distinguish a recurring effect from noise. With only one observed Christmas, a model cannot know whether a spike is annual seasonality or an unusual event. Some software will still return parameters, which makes visual and backtest checks essential. An interval from an ETS or ARIMA model combines model assumptions with estimated residual variation. It is a distributional claim that must be checked on held-out origins; a wide interval is not automatically useful, and a narrow one can be dangerously overconfident.

An intervention can break a recurrence. If a shop closes for renovation, the observed zeros during closure are real sales but may not describe demand after reopening. A local model trained through those zeros can drive its level down. Use an event flag, a shorter post-reopening training window or a fallback that borrows comparable products, depending on the decision. Record the reason for exclusions rather than deleting hard periods because they worsen a metric. If demand is censored by stockouts, a sales forecast and a demand forecast are different targets. No statistical order search repairs a misdefined target.

### Know when a covariate helps

A future calendar and a committed promotion plan can be valid SARIMAX inputs. A realised future competitor price, footfall or weather observation cannot. If an exogenous variable is itself forecast, use the historical forecast version in the backtest. Suppose a temperature forecast available on Monday predicts Wednesday poorly. Training a demand model with Wednesday's actual temperature and evaluating with that actual value measures an easier system than the one served on Monday. The production error includes both demand-model error and temperature-forecast error, and their effects may interact.

An external variable can improve correlation without giving a causal estimate. A promotion may be scheduled for periods when planners already expect low demand, so its coefficient can look negative even when the promotion increases sales relative to what would otherwise happen. Forecasting with the plan can still be useful if the planning policy remains stable, but changing that policy invalidates the simple association. Use causal methods when the question is “what should we schedule?” rather than “what happens under the existing schedule?”

### Compare systems, not fitted lines

A fair result table should include each baseline and fitted model on identical origins, horizons, target units and missing-outcome rules. Include fit failures, fallback use, runtime and interval coverage. Report the median and difficult tail of error by series, not only a pooled mean. When a method wins by a small amount, ask whether the extra parameter tuning, per-series model management and incident burden are worth it. Classical methods are attractive partly because they are inspectable and cheap to retrain, but they still need versioned data and careful operations.

Keep the original forecast issued at each origin. Recomputing it after revised data arrive would make the historical score look better than the decision the organisation actually made.

Match the model to a specific forecast contract. For a stable nonseasonal level, start with naive and simple smoothing. For a changing level, test trend methods and check extrapolation. For a clear repeated cycle with several observations per season, compare seasonal naive, seasonal smoothing and a seasonal ARIMA candidate. When explanatory variables are required, version their availability and backtest with their historical forecasts. Use a fallback when fitting fails or a model is trained on too few cycles; the fallback should be specified before testing.

Control model proliferation. Thousands of separate ARIMA fits may be costly to maintain and hard to monitor. A global model can share information across series, but it introduces a different bias and needs separate cold-start evaluation. Classical local models have the advantage that a planner can inspect a particular series' level, trend, seasonal component and residuals. That transparency is valuable only if the assumptions and failure modes are explained alongside the point prediction.

## Where this stands in 2026

Exponential smoothing and ARIMA remain useful benchmarks in current forecasting practice. The [current textbook](https://otexts.com/fpp3/) still treats them alongside distributional forecasts and modern evaluation. A model catalogue can also include lag-feature learners and pretrained models, but each candidate should face identical origins, horizons and data availability. The comparison should include simple baselines and operational costs, not just a single average error.

## Practice questions

<details>
<summary>Why does simple exponential smoothing give a flat multi-step forecast?</summary>

Its only state is the current level. Once no new observation arrives, no update changes that state, so every future horizon uses the last level.

</details>

<details>
<summary>What does the “moving average” in ARIMA mean?</summary>

It is a linear dependence on past forecast innovations. It is not the arithmetic average of the last few target values.

</details>

<details>
<summary>When can SARIMAX use a future regressor safely?</summary>

When that regressor's future value, or the version forecast for it, is available at the origin. Backtests should use the same type of value that the serving system will have.

</details>

<details>
<summary>Does lower training residual error prove a better forecast?</summary>

No. Flexible models can fit history without predicting later observations. Compare rolling-origin errors and diagnose residuals on appropriate held-out windows.

</details>

## Further reading

- [Simple exponential smoothing](https://otexts.com/fpp3/ses.html) gives the recurrence and estimation details.
- [ARIMA modelling](https://otexts.com/fpp3/arima-r.html) shows order selection and diagnostics.
- [Simple forecasting methods](https://otexts.com/fpp3/simple-methods.html) defines the baselines used in fair comparisons.

## Check yourself

- I can explain which patterns a naive, seasonal-naive and simple-smoothing forecast can represent.
- I can calculate the four level updates in the example and identify the forecast origin.
- I can distinguish ARIMA's moving-average error term from a rolling mean feature.
- I can say when an external regressor is genuinely available for a future horizon.
- I can choose a fallback and a validation design before fitting a complex model.
