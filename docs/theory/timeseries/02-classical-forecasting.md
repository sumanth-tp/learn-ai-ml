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

:::tip Before you start
**You should already know**

- What a forecast origin and horizon are, and why splits must follow time ([Temporal foundations](/docs/theory/timeseries/temporal-foundations)).
- What a mean absolute error is ([Model evaluation](/docs/theory/ml/model-evaluation)).

**Reading time:** about 40 minutes, plus the code.

**After this chapter you can**

- compute naive, seasonal naive, smoothing and ARIMA forecasts and say what each assumes,
- score them on the same rolling origins with a scaled error (MASE),
- say when a seasonal-naive baseline is hard to beat and when it is easy.

:::

## In 30 seconds

Before you build a clever forecaster, ask what a lazy one would score. A lazy forecaster says tomorrow equals today (naive), or next Friday equals last Friday (seasonal naive). Smoothing models average many past days instead of copying one, and ARIMA adds a memory of recent errors. Copying one past day also copies that day's noise, so averaging helps when the data are noisy and barely helps when they are clean. Judge each model on the same dates, in the same units, and the cleverness either shows up in the numbers or it does not.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Naive forecast | Repeat the last value. | Last sales were 16, so forecast 16. |
| Seasonal naive | Repeat the value one season ago. | Next Friday equals last Friday. |
| Level | The smoothed "typical value right now". | 12 after smoothing 10, 12, 11, 13. |
| ETS | Exponential smoothing with explicit error, trend and seasonal parts. | Weekly pattern plus a slow level. |
| ARIMA | A model of past values and past errors after differencing. | Today leans on yesterday and on yesterday's miss. |
| Rolling origin | Re-fit and re-forecast at a series of cut-off dates. | Fifteen cut-offs, two weeks apart. |
| MASE | Mean absolute error divided by a baseline's in-sample error. | 1.0 means as accurate as the in-sample baseline. |
| Win rate | Share of origins where one model beats another. | 12 of 15. |

## The idea in plain words

“Classical” describes a family of statistical tools, not an obsolete stage that every project must outgrow. A small, regular series with a stable calendar pattern may be easier to forecast and explain with a seasonal naive rule or an exponential smoothing model than with a large neural network. The trade-off is that a model's structure must match the data. A flat-level model cannot represent a persistent trend; a seasonal model needs enough cycles to estimate one; an autoregressive model can follow serial dependence but cannot infer an unobserved intervention by magic.

Begin with baselines. For a forecast issued after $T$, the mean method predicts the training mean, the naive method predicts $y_T$, a drift method extends the change between the first and last observations, and seasonal naive repeats the corresponding value from the previous season. Each gives a different claim about persistence. Compute each at the actual decision horizons before fitting a more elaborate model. A baseline is also a unit test: if a supposedly advanced model underperforms it, inspect leakage, horizon alignment, training windows and the business loss before adding complexity.

<Infographic src="/img/timeseries/classical-forecasts.svg" alt="Four cards compare naive baselines, a smoothed level of 12, structured ETS and ARIMA models, and evaluation at matched forecast origins." caption="Choose the recurrence that matches the series and decision horizon." />

## Worked example, step by step

A series alternates low and high, and drifts upwards: 10, 14, 11, 15, 12, 16. The season is 2 steps long. We forecast the next two values, which turn out to be 13 and 17.

1. **Naive.** The last value is 16, so the forecast is 16 and 16. Errors: |13 - 16| = 3 and |17 - 16| = 1. Mean absolute error (MAE) is 2.
2. **Seasonal naive.** Each forecast copies the value 2 steps back: 12 for the first step, 16 for the second. Errors: |13 - 12| = 1 and |17 - 16| = 1. MAE is 1.
3. **Scale.** The seasonal naive forecast, run over the training data, misses by |11 - 10|, |15 - 14|, |12 - 11| and |16 - 15|, which is 1 each. The scale is 1.
4. **MASE** is MAE divided by the scale: 2 for naive, 1 for seasonal naive.

In words: MASE of 1 means "as good as the seasonal-naive method was on the training data". Above 1 is worse, below 1 is better. The statsforecast library gives the same forecasts.

```python
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

train = np.array([10, 14, 11, 15, 12, 16], dtype=float)
test = np.array([13, 17], dtype=float)
frame = pd.DataFrame({'unique_id': 'a', 'ds': pd.date_range('2026-01-01', periods=len(train), freq='D'), 'y': train})
forecast = StatsForecast(models=[Naive(), SeasonalNaive(season_length=2)], freq='D').forecast(df=frame, h=2).reset_index()
scale = np.abs(train[2:] - train[:-2]).mean()
for name in ('Naive', 'SeasonalNaive'):
    error = np.abs(test - forecast[name].values).mean()
    print(f'{name:<14} forecast {forecast[name].values}  MAE {error:.1f}  MASE {error / scale:.1f}')
print('seasonal scale:', scale)
print('training data:', train, ' test:', test)
```

It prints forecasts `[16. 16.]` and `[12. 16.]`, MASE 2.0 and 1.0, a seasonal scale of 1.0 and the data echoed back, matching the four steps above. Note that the forecast for the first step is 12, not 11: the series ended `12, 16`, so two steps back from the first forecast is the 12.

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

### Experiment: four models, four series, fifteen origins

The block builds four daily series of two years with a trend of 0.04 a day and autocorrelated noise: three with a weekly pattern and noise of 1, 6 and 12, and one with no weekly pattern and noise of 6. For each it runs a rolling-origin backtest with a 14-day horizon, 15 origins and a step of 14 days, using four models from statsforecast: naive, seasonal naive, `AutoETS` and a fixed `ARIMA(1,0,1)(0,1,1)` with period 7. The ARIMA order is fixed because an automatic order search with a weekly season took about 8 seconds for just four windows of one series on this machine, which does not fit a 60-second block.

```python
import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoETS, Naive, SeasonalNaive

warnings.filterwarnings('ignore')
rng = np.random.default_rng(11)
n, horizon, windows = 730, 14, 15
weekly = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
weekly -= weekly.mean()
settings = {'quiet weekly': (1.0, 1.0), 'noisy weekly': (6.0, 1.0), 'very noisy weekly': (12.0, 1.0), 'no season, noisy': (6.0, 0.0)}
frames = []
for name, (sigma, strength) in settings.items():
    t = np.arange(n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, sigma)
    y = 100 + 0.04 * t + strength * weekly[t % 7] + noise
    frames.append(pd.DataFrame({'unique_id': name, 'ds': pd.date_range('2021-01-01', periods=n, freq='D'), 'y': y}))
df = pd.concat(frames)

models = [Naive(), SeasonalNaive(season_length=7), AutoETS(season_length=7),
          ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=7, alias='ARIMA')]
sf = StatsForecast(models=models, freq='D', n_jobs=1)
cv = sf.cross_validation(df=df, h=horizon, step_size=horizon, n_windows=windows).reset_index()
names = [m.alias for m in models]

rows, scales = [], {}
for uid, g in df.groupby('unique_id'):
    first_cutoff = cv.loc[cv['unique_id'] == uid, 'cutoff'].min()
    train = g.loc[g['ds'] <= first_cutoff, 'y'].values
    scale = np.abs(train[7:] - train[:-7]).mean()
    scales[uid] = scale
    c = cv[cv['unique_id'] == uid]
    row = {'series': uid}
    for m in names:
        row[m] = round(np.abs(c['y'] - c[m]).mean() / scale, 3)
    per_window = c.assign(a=np.abs(c['y'] - c['ARIMA']), s=np.abs(c['y'] - c['SeasonalNaive'])).groupby('cutoff')[['a', 's']].mean()
    row['ARIMA beats snaive in'] = f"{int((per_window['a'] < per_window['s']).sum())} of {windows}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

noisy = cv[cv['unique_id'] == 'noisy weekly'].assign(day=lambda d: (d['ds'] - d['cutoff']).dt.days)
for m in ('SeasonalNaive', 'ARIMA'):
    by_day = np.abs(noisy['y'] - noisy[m]).groupby(noisy['day']).mean() / scales['noisy weekly']
    print(f'{m:<14} MASE at lead day 1, 7, 14:', [round(float(by_day[d]), 3) for d in (1, 7, 14)])
```

**Reading the output.** Each cell is MASE, with the scale taken from seasonal differences of the training data before the first origin. On the quiet weekly series (noise 1) the models sit close together: seasonal naive 1.090, ARIMA 1.073, ETS 0.998, and naive is far behind at 3.568. ARIMA beats seasonal naive at 7 of 15 origins there, which is a coin flip. On the noisy weekly series, ARIMA scores 0.738 against 0.975 for seasonal naive and wins at 12 of 15 origins. On the noisiest it scores 0.686 against 0.913 and wins at 14 of 15. The last two lines split the noisy weekly series by lead day: ARIMA's edge is largest on day 1 (0.585 against 1.09) and smaller on day 14 (0.725 against 0.869), because its memory of recent errors fades with distance.

**Line by line.**

- `cross_validation(... step_size=horizon, n_windows=windows)` re-fits every model at each cut-off and forecasts the next 14 days. `cutoff` records the origin.
- `scale` uses `train[7:] - train[:-7]`, the seasonal-naive error on data before the first origin, so all models share one denominator per series.
- `alias='ARIMA'` names the column. Without it the library builds a long name from the order.
- `scales` keeps each series' denominator so the lead-day split at the end uses the same scale as the table.
- The `per_window` grouping gives a paired comparison: the same origin, two models.

**Interpretation.** The seasonal-naive baseline is hard to beat when the weekly pattern is large and the noise is small, because there is little for an average to remove. When noise is large, copying a single past week copies that week's noise, and ARIMA scores 24 to 25 per cent lower than seasonal naive on the two noisy weekly series while ETS scores 7 to 13 per cent lower. The surprise is on the series with no weekly pattern: seasonal naive (0.910) does worse than plain naive (0.866), and ARIMA, whose seasonal difference also looks one week back, still wins at 0.732 because its moving-average term smooths the noise. The limits: the series are synthetic, the ARIMA order is close to how the data were made, there is one seed, and 15 origins give a win count with wide error. Use the table to learn where to look in your own backtest, not as a ranking of methods.

<Infographic src="/img/ts-enrich/backtest-mase.svg" alt="A table of MASE for naive, seasonal naive, ETS and ARIMA on four synthetic series, with the best model in each row outlined and the number of origins where ARIMA beat seasonal naive." caption="Read down the seasonal naive column first, then look at which model is outlined in each row." />

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

## Common mistakes

1. **Comparing models on different origins.** It feels fair because every model "had the same data". A model scored on easy weeks looks better than one scored on hard weeks. Score all models on the same cut-offs and report paired win counts (12 of 15 means more than a mean difference).
2. **Picking the seasonal period from a default.** It feels safe because 7 is common. A wrong period turns seasonal naive into a copy of a random day. Read the period from the cadence and the domain, and check it on the autocorrelation plot.
3. **Treating seasonal naive as always strong.** It feels like a hard baseline, and on the quiet series it is. On the noisy weekly series ARIMA scored 24 to 25 per cent lower. Test it on your noise level.
4. **Trusting an in-sample fit.** It feels like evidence because the line hugs the data. Flexible models fit history and still miss the future. Always compare out-of-sample on rolling origins.
5. **Reporting one average over series with different scales.** It feels simple. Large series dominate the mean. Use MASE per series, then report the median and the worst tail.

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

<details>
<summary><strong>Q5 (Easy).</strong> In the worked example, why is the seasonal-naive forecast for the first step 12 and not 11?</summary>

The season is 2 steps, so the forecast copies the value 2 steps before the forecast date. The series ends 12, 16, so the first forecast copies 12 and the second copies 16.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> ARIMA scored 0.738 and seasonal naive 0.975 on the noisy weekly series, yet ARIMA beat seasonal naive at only 12 of 15 origins. Why report both?</summary>

The mean says how large the gain is, and the count says how reliable it is. A model can win on average through a few large origins and lose on most. With 15 origins from one seed, neither number is precise, so the pair is more honest than either alone.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> On the series with no weekly pattern, seasonal naive was worse than naive. What does that say about choosing between them without looking at the data?</summary>

Seasonal naive adds noise when there is no season to exploit, because it copies a week-old value that carries no extra information. Choose the baseline from evidence of a season, such as the autocorrelation at lag 7 or the profile in a decomposition, and keep the plain naive forecast in the table either way.

</details>

## Further reading

- [Simple exponential smoothing](https://otexts.com/fpp3/ses.html) gives the recurrence and estimation details.
- [ARIMA modelling](https://otexts.com/fpp3/arima-r.html) shows order selection and diagnostics.
- [Simple forecasting methods](https://otexts.com/fpp3/simple-methods.html) defines the baselines used in fair comparisons.

- [StatsForecast documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html) describes `AutoETS`, `SeasonalNaive` and prediction intervals (opened 2026-10-08, statsforecast 2.1.1 was run).
- [Forecasting: Principles and Practice, accuracy](https://otexts.com/fpp3/accuracy.html) defines MASE and its seasonal form (cited by the chapter on evaluation, not re-opened for this chapter).

## Check yourself

- I can explain which patterns a naive, seasonal-naive and simple-smoothing forecast can represent.
- I can calculate the four level updates in the example and identify the forecast origin.
- I can distinguish ARIMA's moving-average error term from a rolling mean feature.
- I can say when an external regressor is genuinely available for a future horizon.
- I can choose a fallback and a validation design before fitting a complex model.
- I can compute naive and seasonal-naive forecasts and a seasonal MASE by hand on a short series.
- I can run four models on the same rolling origins and report a paired win count as well as an average.
- I can say why seasonal naive is hard to beat on a quiet series and beatable on a noisy one, and show it with numbers.

## Where to go next

Continue with [lagged machine learning](/docs/theory/timeseries/lagged-machine-learning), where a boosted tree gets lags and a known promotion plan, or go back to [temporal foundations](/docs/theory/timeseries/temporal-foundations) for the leakage rules every backtest depends on.
