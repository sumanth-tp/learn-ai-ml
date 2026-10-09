---
title: "Time series · Lagged machine learning"
sidebar_label: "Lagged machine learning"
sidebar_position: 3
slug: /theory/timeseries/lagged-machine-learning
description: "Turn past windows and known future signals into leak-free tabular forecasting features."
tags: [time-series, machine-learning, forecasting]
---

import Infographic from '@site/src/components/Infographic';
import LagWindowLab from '@site/src/components/viz/LagWindowLab';

**In one line.** A feature-based forecaster converts each prediction origin into a training example whose inputs were all known at that origin.

:::tip Before you start
**You should already know**

- Origins, horizons and why a split must follow time ([Temporal foundations](/docs/theory/timeseries/temporal-foundations)).
- How a gradient-boosted tree is trained and used ([Gradient boosting in practice](/docs/theory/ml/gradient-boosting-in-practice)).

**Reading time:** about 45 minutes, plus the code.

**After this chapter you can**

- turn a series into a table of lag, rolling and calendar features whose every value was known at the origin,
- train one global model across many series and compare it with a local model and with ETS,
- show with numbers how an unshifted rolling mean makes a model look better than it will ever be in production.

:::

## In 30 seconds

A tree model wants a table: one row per prediction, one column per fact. For forecasting, each row is a moment in time. The columns are what you knew at that moment, such as today's sales, last week's sales and the average of the last seven days, plus anything already scheduled, such as a promotion. The target is the value some days later. Imagine a shopkeeper filling in a card each evening before ordering: only things seen so far go on the card. If a column secretly contains tomorrow's sales, the card predicts brilliantly and the shop still runs out of stock.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Lag feature | An earlier value of the target as an input. | Sales 7 days ago. |
| Rolling feature | A summary of a recent window. | Mean of the last 7 days. |
| Direct forecast | One model trained for one fixed horizon. | A model that predicts 7 days ahead. |
| Recursive forecast | One-step model whose output is fed back as an input. | Predict day 1, use it to predict day 2. |
| Global model | One model trained on many series together. | 24 shops, one tree ensemble. |
| Known-future covariate | An input already fixed at the origin. | Next week's promotion plan. |
| Normalising | Dividing by a recent level so series of different size look alike. | Sales divided by their 28-day mean. |
| Leaky feature | A column computed with values after the origin. | A rolling mean that includes the target day. |

## The idea in plain words

Many supervised learning methods expect independent rows with named input columns. Forecasting data arrive as ordered observations. A lagged-feature design bridges the two: for target $y_t$, include earlier values such as $y_{t-1}$ and $y_{t-2}$, rolling summaries computed strictly before $t$, and calendar or external values actually known when the forecast is issued. The model can then learn nonlinear relationships across those columns. The apparent simplicity hides the main risk: a single misplaced shift can turn a future target into a feature.

One-step prediction and multi-step prediction also differ. At an origin $o$, a direct horizon-two model can be trained to map information through $o$ to $y_{o+2}$. A recursive one-step model predicts $y_{o+1}$ and uses that prediction as an input for $y_{o+2}$. The recursive path may compound errors. A set of direct models avoids that feedback but costs more fitting and may ignore relationships between horizons. A multi-output model predicts the whole horizon together. Backtest the exact serving strategy, including whether future lags come from observed values or earlier forecasts.

<Infographic src="/img/timeseries/lagged-learning.svg" alt="Four cards show the lag-one value 11, lag-two value 12 and prior-three mean 11, then describe tabular learning, direct or recursive horizons and rolling-window leakage." caption="A feature window must end before its target or before the forecast origin for longer horizons." />

## Worked example, step by step

Take the series 10, 12, 11, 13, 10, 12 (positions 0 to 5). Stand at position 3 and forecast two steps ahead, which is position 5, whose value is 12.

1. **What is known.** Positions 0 to 3: 10, 12, 11, 13.
2. **Level for normalising.** Their mean is (10 + 12 + 11 + 13) / 4 = 11.5.
3. **Features, divided by the level.** `lag_0` is 13 / 11.5 = 1.1304. `lag_1` is 11 / 11.5 = 0.9565.
4. **Target, divided by the same level.** 12 / 11.5 = 1.0435. A model that predicts this ratio multiplies it by 11.5 to get units back.
5. **An honest rolling mean** ends at the origin: positions 1 to 3 are 12, 11, 13, mean 12.0.
6. **A leaky rolling mean** ends at the target: positions 3 to 5 are 13, 10, 12, mean 11.6667. It contains the answer, so a model can lean on it.

In words: features look back from the origin, the target sits `horizon` steps ahead, and dividing by a known level lets one model serve small and large series.

```python
import pandas as pd

y = pd.Series([10, 12, 11, 13, 10, 12], dtype=float)
origin, horizon = 3, 2
base = y.iloc[origin - 3:origin + 1].mean()
row = {
    'base': base,
    'lag_0': y.iloc[origin] / base,
    'lag_1': y.iloc[origin - 1] / base,
    'target': y.iloc[origin + horizon] / base,
    'honest_mean_3': y.iloc[origin - 2:origin + 1].mean(),
    'leaky_mean_3': y.iloc[origin + horizon - 2:origin + horizon + 1].mean(),
}
print({name: round(float(value), 4) for name, value in row.items()})
```

It prints base 11.5, `lag_0` 1.1304, `lag_1` 0.9565, target 1.0435, honest mean 12.0 and leaky mean 11.6667, the same numbers as steps 2 to 6.

## How it works

### Construct a row with a clear clock

For a one-step example with target at index $t$, lag one is $y_{t-1}$, lag two is $y_{t-2}$ and a prior-three mean is $(y_{t-1}+y_{t-2}+y_{t-3})/3$. With the synthetic sequence $[10,12,11,13,10,12]$ and zero-based target index three, the target is 13, lag one is 11, lag two is 12 and the mean of the preceding three values is 11. The table below the lab makes the boundary visible. A rolling mean that includes the target would instead be $(12+11+13)/3=12$, a leak. The numerical closeness of those values would make the error easy to miss without an explicit time-index test.

For horizon $h$, the safe target row depends on the origin. If predicting $y_{o+h}$, the last observable target-derived feature is $y_o$, no matter how many calendar timestamps lie between $o$ and $o+h$. In an offline table indexed by target time, a shift of one is sufficient only for one-step forecasts. Direct horizon-$h$ training must shift target-derived features far enough that the last input matches the origin. Label every training row with both target time and origin time, and assert `max_source_time <= origin_time` for all target-derived features.

Calendar features such as weekday, month and scheduled holiday are often known ahead of time. They can improve a model when demand follows a calendar. Encode cyclicity thoughtfully: a raw hour value jumps from 23 to 0, whereas sine and cosine encoding preserves adjacency. For tree models, raw integer or one-hot calendar features may already work. Do not claim that a calendar label explains a structural event. A change in opening hours or pricing may alter the pattern even if the weekday stays the same.

Rolling statistics need their own alignment. A trailing seven-day mean available at the end of day $o$ uses days $o-6$ through $o$; it cannot use the next seven days. A rolling maximum and standard deviation may capture bursts, but require enough history and a defined missing-data rule. Fit imputers, encoders and scalers inside each training fold. A global scaler fitted over the entire backtest sees future shifts in the distribution. A lag feature can also cross entity boundaries accidentally when a dataframe shift ignores product or site groups. Compute groupwise lags before combining series.

### Choose local or global learning

A local model uses only one series. A global model pools examples from many series and may share patterns across them. Pooling can help sparse groups, but the model needs identifiers or metadata and a validation design that measures both existing-series and new-series performance. A one-hot identifier can memorise a training entity yet give no useful representation to a new one. Related product attributes may transfer better. If a model predicts millions of combinations of groups and horizons, training-row weighting determines which entities dominate the loss. Average error across all rows may mainly reflect high-volume entities.

Tree ensembles are a practical choice for mixed lag, calendar and covariate features. They can represent thresholds and interactions without requiring a stationary series. Linear models provide a simpler baseline and may be easier to extrapolate when trend is encoded explicitly. Neither family can automatically forecast an unseen future covariate. A production feature service must distinguish observed features, planned features and predictions of features. Store feature versions and join using timestamps as well as entity keys.

### Keep evaluation aligned with the final pipeline

A common training shortcut builds all possible lag rows, randomly splits them, then scores a tree. Adjacent rows share most of their input window, so this can be much easier than future forecasting. The [scikit-learn time-series example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) demonstrates why time-aware evaluation changes the estimate. Use rolling origins and retrain at the cadence that the real system will use. If the service retrains weekly, do not evaluate as if it were refitted after every hourly observation. If features are delayed, replay the delay. If hyperparameters are selected from several rolling origins, hold back a later final period.

For a point model trained with squared loss, large errors receive extra weight; absolute loss targets a different central tendency and can be more robust to occasional spikes. Quantile regression can produce conditional quantiles, but separate quantile models may cross, and nominal interval coverage still needs evaluation. Model choice should be guided by the decision. A low-error average forecast can be poor for safety stock if underestimation is much more expensive than overestimation. In that case, predict a suitable quantile or optimise a decision model on backtested outcomes.

## A real system that works this way

The [official scikit-learn example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) constructs lag and rolling-window columns for a demand-like time series and assesses them using a temporal split. It shows a credible template for a batch forecaster: build a row at each allowed origin, train an estimator on past rows, and score later origins. An operational version also needs stable feature definitions, entity grouping, availability timestamps and a way to backfill predictions when a feature job fails. The estimator is only one part of the system.

## Code you can run

```python
series = [10, 12, 11, 13, 10, 12]
target_index = 3
window = series[target_index - 3:target_index]
row = {
    'lag_1': series[target_index - 1],
    'lag_2': series[target_index - 2],
    'prior_3_mean': sum(window) / len(window),
    'target': series[target_index],
}
print(row)
```

The output contains `lag_1: 11`, `lag_2: 12`, `prior_3_mean: 11.0` and `target: 13`. None of the input values come from index three.

<LagWindowLab />

Move the target and window controls. The chart highlights earlier feature points separately from the target, while the data view lists exactly which observations were included. Its defaults reproduce the printed row.

```python
series = [10, 12, 11, 13, 10, 12]
rows = []
for target_index in range(3, len(series)):
    origin_index = target_index - 1
    features = series[target_index - 3:target_index]
    rows.append((origin_index, target_index, tuple(features), series[target_index]))
for row in rows:
    print(row)
```

This prints three one-step rows, with origins two, three and four. A direct two-step design would need a different input boundary: the row targeting index five would have origin index three, and could not use value at index four. Keep origin and target explicit in code reviews.

### Experiment: one global model, a promotion plan, and a leaky feature

The block makes 24 daily series of 600 days. Each has its own level (50 to 200), a weekly pattern, a slight trend, autocorrelated noise and promotion days (8 per cent of days) that lift sales by 25 per cent. The task is to forecast 7 days ahead at 12 origins that are 7 days apart. Every candidate is scored by MASE on 288 forecasts. The boosted model is scikit-learn's `HistGradientBoostingRegressor`, trained once on rows whose targets end before the first test origin, with no refitting during the test. ETS is refitted at every origin by statsforecast, which favours ETS, but it receives no promotion information.

```python
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

warnings.filterwarnings('ignore')
rng = np.random.default_rng(3)
n_series, n, horizon, windows = 24, 600, 7, 12
shape = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
shape = (shape - shape.mean()) / 100
days = pd.date_range('2022-01-01', periods=n, freq='D')
frames = []
for k in range(n_series):
    level = rng.uniform(50, 200)
    promo = (rng.random(n) < 0.08).astype(int)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.4 * noise[i - 1] + rng.normal(0, 0.05 * level)
    t = np.arange(n)
    y = level * (1 + 0.0004 * t + shape[t % 7] * 3 + 0.25 * promo) + noise
    frames.append(pd.DataFrame({'unique_id': f's{k:02d}', 'ds': days, 'y': y, 'promo': promo}))
df = pd.concat(frames, ignore_index=True)

g, p = df.groupby('unique_id')['y'], df.groupby('unique_id')['promo']
base = g.transform(lambda s: s.rolling(28).mean())
feat = pd.DataFrame({'unique_id': df['unique_id'], 'ds': df['ds'], 'base': base})
for lag in (0, 1, 6, 13, 20):
    feat[f'lag_{lag}'] = g.shift(lag) / base
feat['mean_7'] = g.transform(lambda s: s.rolling(7).mean()) / base
feat['promo_target'] = p.shift(-horizon)
feat['promo_last7'] = p.transform(lambda s: s.rolling(7).sum())
feat['dow'] = (df['ds'].dt.dayofweek + horizon) % 7
feat['target'] = g.shift(-horizon) / base
feat['actual'] = g.shift(-horizon)
feat['rolling_mean_3_at_target'] = g.transform(lambda s: s.shift(-horizon).rolling(3).mean()) / base
feat = feat.dropna().reset_index(drop=True)
skip = ('unique_id', 'ds', 'target', 'base', 'actual', 'rolling_mean_3_at_target')
honest = [c for c in feat.columns if c not in skip]
no_promo = [c for c in honest if not c.startswith('promo')]

origins = [days[-1] - pd.Timedelta(days=horizon * (i + 1)) for i in range(windows)]
test = feat[feat['ds'].isin(origins)]
train = feat[feat['ds'] <= min(origins) - pd.Timedelta(days=horizon)]
scale = df.groupby('unique_id')['y'].apply(lambda s: np.abs(s.values[7:] - s.values[:-7]).mean())

def mase(frame, pred):
    return (np.abs(frame['actual'] - pred * frame['base']) / frame['unique_id'].map(scale)).mean()

def fit(columns, rows):
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.1, random_state=0)
    return model.fit(rows[columns], rows['target'])

result = {'seasonal naive': mase(test, test['lag_0'])}
global_model = fit(honest, train)
result['global HGB'] = mase(test, global_model.predict(test[honest]))
result['global HGB without promo columns'] = mase(test, fit(no_promo, train).predict(test[no_promo]))
local = pd.Series(0.0, index=test.index)
for uid, rows in train.groupby('unique_id'):
    mask = test['unique_id'] == uid
    local[mask] = fit(honest, rows).predict(test.loc[mask, honest])
result['local HGB, one model per series'] = mase(test, local)
leaky = honest + ['rolling_mean_3_at_target']
result['global HGB with the leaky rolling mean'] = mase(test, fit(leaky, train).predict(test[leaky]))
held = sorted(feat['unique_id'].unique())[::4]
seen, unseen = train[~train['unique_id'].isin(held)], test[test['unique_id'].isin(held)]
result['global HGB on 6 series it never saw'] = mase(unseen, fit(honest, seen).predict(unseen[honest]))
result['same 6 series, model trained on them'] = mase(unseen, global_model.predict(unseen[honest]))

cv = StatsForecast(models=[AutoETS(season_length=7, model='ZZA')], freq='D', n_jobs=1).cross_validation(
    df=df[['unique_id', 'ds', 'y']], h=horizon, step_size=horizon, n_windows=windows).reset_index()
print('origins match:', set(cv['cutoff']) == set(origins))
cv = cv[cv['ds'] == cv['cutoff'] + pd.Timedelta(days=horizon)]
result['ETS, no promo information'] = (np.abs(cv['y'] - cv['AutoETS']) / cv['unique_id'].map(scale)).mean()
for name, value in result.items():
    print(f'{name:<42} MASE {value:.3f}')
print('train rows', len(train), 'test rows', len(test))
```

**Reading the output.** The run confirms `origins match: True`, so ETS and the boosted models are scored at the same dates. The seasonal-naive baseline scores 1.004. ETS scores 0.695. The global boosted model with every honest feature scores 0.505. Remove the two promotion columns and it scores 0.724, slightly worse than ETS. One model per series scores 0.674. With the leaky three-day mean added, the score is 0.420. On six series the global model never saw in training, it scores 0.460, against 0.442 for the same model on those series when they were in training.

**Line by line.**

- `g.shift(lag) / base` shifts inside each series, because `g` is grouped by `unique_id`. A plain `shift` on the stacked table would hand the first rows of one series the last values of the one above it.
- `p.shift(-horizon)` reads the promotion flag of the target day. This is legal only because the plan is fixed before the origin.
- `rolling_mean_3_at_target` shifts the series forward by the horizon before taking the mean, which is what an unshifted `rolling(3).mean()` computed on a table indexed by target day does.
- `train` stops one horizon before the first test origin, so no training target falls after a test origin.

**Interpretation.** The promotion plan is where the gain comes from. Without it, the boosted model (0.724) does not beat ETS (0.695); with it, the score falls by 30 per cent to 0.505. ETS here has no regressor for the plan, so this is not a like-for-like comparison of model families, and an ETS or ARIMA with the plan as a regressor would narrow the gap. The leaky feature improves the score by another 17 per cent (0.505 to 0.420) and would be unavailable at serving time, which is why a backtest that looks too good deserves an audit of every column. Pooling helped: one global model (0.505) beat 24 local models (0.674). On unseen series the loss was small (0.460 against 0.442, 4 per cent), partly because every series in this synthetic set shares the same shape and normalisation makes sizes comparable. Real products differ more. The limits: one seed, synthetic data, 288 test forecasts, one horizon and no tuning.

<Infographic src="/img/ts-enrich/lag-ladder.svg" alt="Horizontal bars of MASE for seven candidates, from seasonal naive at 1.004 down to the global model with a leaky rolling mean at 0.420." caption="Read from the top: the big drop comes from adding the promotion plan; the last bar is the one that cannot be used in production." />

## Designing with it

### Trace a two-step example by hand

Take the six-value series $[10,12,11,13,10,12]$. At origin index three, the latest observed target is 13. A direct two-step model predicts the value at index five, 12 in this synthetic history, using only information through index three. Lag one relative to the **origin** is 13 and lag two is 11. A careless row builder indexed by the target may call index four's value 10 “lag one” for index five. That value is future information at origin three. The model can learn from historical rows constructed with this error and score well in a backtest that repeats it, but it cannot receive that value at serving time. Naming variables `origin_index`, `target_index` and `horizon` prevents this ambiguity.

A recursive method takes a different route. At origin three it predicts index four, then uses that *prediction* when predicting index five. If evaluation inserts the realised 10 at index four instead, it is teacher-forcing the second step and reporting an unrealistically easy result. A direct method needs separate training targets or a model that takes the horizon as an input. A multi-output method produces both target values together. Each method has a different data construction and error profile; compare them with the same origin and final target outcomes.

### Make feature lineage executable

A useful feature registry can record entity key, event timestamp, availability timestamp, source table, transformation, window length and missing-value policy. For a rolling seven-day mean, the registry should state whether a row at origin $o$ includes $y_o$ and whether that observation was already final at the issue time. A feature assertion can verify that the maximum availability timestamp in every row is no later than the origin. This should run on sampled historical rows and on serving requests. A human-readable feature name such as `last_week_sales` is insufficient evidence: a join might have read a revised record from a later snapshot.

Groupwise operations are another quiet source of error. If rows for product A end just before rows for product B begin, a global `shift(1)` can give B the final sales of A. Sort by entity and time, then shift within each entity. Apply the same rule to rolling means, missingness indicators and target encoders. If a product changes identifier, decide whether that is a genuinely new series or a renamed continuation, and keep that mapping versioned. A mismatch between training and serving entity keys can produce plausible-looking but unrelated lags.

### Choose transforms and loss deliberately

Log or square-root target transforms can reduce the influence of very large values, but inverse-transforming a mean prediction does not generally produce the mean on the original scale. If the business evaluates units sold, compute metrics after inverse transformation and any clipping or rounding. Price, promotion and inventory features can be powerful, yet they can also change due to actions taken using earlier forecasts. A model that treats an inventory shortage as low demand may learn to forecast stockouts rather than customer demand. Decide whether the target is observed sales or unconstrained demand and document the difference.

Feature importance from a tree can explain which columns helped the fitted model on its training distribution. It does not prove a causal effect or that a feature will remain useful after a policy change. Permutation importance on a time-aware validation set is more relevant than importance computed on training rows, but correlated lags can substitute for one another and make individual scores unstable. Use feature removal tests and error slices to decide whether a costly feature is worth its maintenance burden.

### A production row is more than a vector

At request time, produce both the model input and a trace of its provenance: entity, origin, horizon, feature timestamps, imputation flags and model version. If an upstream source is late, choose a documented fallback or refuse the prediction rather than silently mixing stale and fresh data. Store the final prediction before the outcome arrives. This trace makes a later backtest comparable with real serving and lets an operator distinguish a model miss from a feature incident. A well-instrumented simple tree can be more valuable than a tuned model whose inputs cannot be reconstructed.

Audit the feature table at the longest horizon as well as the shortest. A feature that is safe one step ahead can become unavailable when the same column is reused for a week-ahead target.

Define a feature contract that names source, timestamp, publication lag, imputation and aggregation for each column. Unit-test the contract with a tiny hand-labelled sequence before training. A test should deliberately alter a future value and confirm that earlier feature rows remain identical. It should also place two entities next to each other and confirm that a shift never crosses entities. These tests catch more consequential errors than a sophisticated model-tuning search.

Choose direct, recursive or multi-output prediction according to horizon and maintenance cost. Direct models make horizon-specific bias visible. Recursive models are compact but require a faithful simulation of generated lags in validation. Multi-output models can encode cross-horizon behaviour but may be harder to train on sparse series. Keep a suitable naive baseline in every backtest and report error by horizon, entity age, volume and special-event period. A single pooled metric can hide precisely the periods when the forecast drives expensive decisions.

## Where this stands in 2026

Feature-based forecasting remains a strong practical option, especially where calendar, price, inventory and known plans matter. Current tooling makes it easy to create many lags automatically, which increases the need for availability checks. Pretrained sequence models provide a different starting point, but they face the same production contract and should be tested against a well-built tabular learner. The next chapter compares their input representation and transfer assumptions.

## Common mistakes

1. **Rolling means without a shift.** It feels right because the rolling mean is a standard pandas call. On a table indexed by target day it includes the target. Shift first, then roll, and test by changing one future value and checking that earlier rows stay the same. In the experiment the leaky column moved the score from 0.505 to 0.420.
2. **A global `shift` across entities.** It feels harmless. The first row of one series receives the last value of the previous series. Sort by series and time and shift within each group.
3. **Comparing a model that sees a promotion plan with one that does not.** It feels like a model-family result. The 0.724 to 0.505 gain came from information, not from trees. Give every candidate the same inputs, or say which ones it lacks.
4. **Scoring recursive forecasts with true intermediate values.** It feels like standard evaluation. The second step then uses a real observation instead of the model's own first-step output, which production never has. Feed predictions back during the backtest.
5. **Normalising with statistics from the whole series.** It feels like tidy preprocessing. The level then contains future values. Compute the base from the window that ends at the origin, as `base` does here.

## Practice questions

<details>
<summary>What is wrong with a rolling mean computed over a window ending at the target?</summary>

It includes the value being predicted. The feature must end at the forecast origin, which is strictly before the target for one-step forecasting.

</details>

<details>
<summary>Why can a one-step row generator be wrong for a three-step forecast?</summary>

Its final lag may come from a time after the three-step origin. Direct multi-step rows must be aligned by origin and horizon, not merely by target index.

</details>

<details>
<summary>What extra test matters for a global model used on new products?</summary>

Hold out products or product launches as well as later time periods. A score on existing products does not measure transfer to an unseen identifier.

</details>

<details>
<summary>Why might a high-volume product dominate a global model?</summary>

It contributes many rows or larger absolute losses to the training objective. Sampling or weighting choices should reflect the decision and be reported explicitly.

</details>

<details>
<summary><strong>Q5 (Easy).</strong> In the worked example, why is the honest rolling mean 12.0 and the leaky one 11.6667?</summary>

The honest window ends at the origin (positions 1 to 3: 12, 11, 13). The leaky window ends at the target (positions 3 to 5: 13, 10, 12) and contains the value being predicted.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> The global boosted model without promotion columns scored 0.724 and ETS 0.695. What can and cannot be concluded?</summary>

On this synthetic set, trees without the plan did not beat ETS, so the tree's advantage came from the extra information. It cannot be concluded that ETS is better in general: the data were generated with a smooth weekly pattern that ETS models well, and the test has one seed and 288 forecasts.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> Unseen series scored 0.460 against 0.442 for seen ones. Why might a real catalogue show a larger gap?</summary>

Here all series share one weekly shape and differ only in level and noise, and dividing by the recent mean removes most of the level difference. Real products differ in shape, promotion response and history length, so the model has less to transfer. Test with a held-out set of series, and report new and old series separately.

</details>

## Further reading

- [scikit-learn lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) gives a complete tabular workflow.
- [Rolling-origin cross-validation](https://otexts.com/fpp3/tscv.html) explains how to test future predictions rather than shuffled interpolation.
- [Forecast accuracy measures](https://otexts.com/fpp3/accuracy.html) compares errors on a common scale and warns about in-sample assessment.

- [scikit-learn: lagged features for time-series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) builds its rolling features on `shift(1)` and compares a shuffled split with `TimeSeriesSplit` (opened 2026-10-08).
- [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) lists the defaults (`max_iter=100`, `learning_rate=0.1`) that the experiment changes to 120 and 0.1 (opened 2026-10-08, scikit-learn 1.9.1).
- [StatsForecast documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html) covers `AutoETS` and cross-validation (opened 2026-10-08, statsforecast 2.1.1).

## Check yourself

- I can create a lag row and identify its origin, target and horizon.
- I can explain how direct and recursive multi-step forecasts use different input values.
- I can detect a feature that was published after its forecast origin.
- I can construct groupwise rolling features without crossing entity boundaries.
- I can design a backtest that reproduces the retraining and feature-update cadence.
- I can build a lag row with an origin, a horizon and a target, and normalise it with a base computed at the origin.
- I can show that an unshifted rolling mean leaks the target, and quote how much it flatters the score.
- I can compare a global model, per-series models and ETS on identical origins, and say which of the gains came from extra inputs.

## Where to go next

Continue with [deep and pretrained forecasting](/docs/theory/timeseries/pretrained-forecasting), which compares a pretrained model with ETS and seasonal naive on the same kind of data, or revisit [classical forecasting](/docs/theory/timeseries/classical-forecasting) for the baselines this chapter beats.
