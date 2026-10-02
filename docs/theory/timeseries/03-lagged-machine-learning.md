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

## The idea in plain words

Many supervised learning methods expect independent rows with named input columns. Forecasting data arrive as ordered observations. A lagged-feature design bridges the two: for target $y_t$, include earlier values such as $y_{t-1}$ and $y_{t-2}$, rolling summaries computed strictly before $t$, and calendar or external values actually known when the forecast is issued. The model can then learn nonlinear relationships across those columns. The apparent simplicity hides the main risk: a single misplaced shift can turn a future target into a feature.

One-step prediction and multi-step prediction also differ. At an origin $o$, a direct horizon-two model can be trained to map information through $o$ to $y_{o+2}$. A recursive one-step model predicts $y_{o+1}$ and uses that prediction as an input for $y_{o+2}$. The recursive path may compound errors. A set of direct models avoids that feedback but costs more fitting and may ignore relationships between horizons. A multi-output model predicts the whole horizon together. Backtest the exact serving strategy, including whether future lags come from observed values or earlier forecasts.

<Infographic src="/img/timeseries/lagged-learning.svg" alt="A forecasting row uses three earlier observations as lag and rolling features, keeps the current target hidden, and sends rows through chronological train and test windows." caption="A feature window must end before its target or before the forecast origin for longer horizons." />

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

## Designing with it

Define a feature contract that names source, timestamp, publication lag, imputation and aggregation for each column. Unit-test the contract with a tiny hand-labelled sequence before training. A test should deliberately alter a future value and confirm that earlier feature rows remain identical. It should also place two entities next to each other and confirm that a shift never crosses entities. These tests catch more consequential errors than a sophisticated model-tuning search.

Choose direct, recursive or multi-output prediction according to horizon and maintenance cost. Direct models make horizon-specific bias visible. Recursive models are compact but require a faithful simulation of generated lags in validation. Multi-output models can encode cross-horizon behaviour but may be harder to train on sparse series. Keep a suitable naive baseline in every backtest and report error by horizon, entity age, volume and special-event period. A single pooled metric can hide precisely the periods when the forecast drives expensive decisions.

## Where this stands in 2026

Feature-based forecasting remains a strong practical option, especially where calendar, price, inventory and known plans matter. Current tooling makes it easy to create many lags automatically, which increases the need for availability checks. Pretrained sequence models provide a different starting point, but they face the same production contract and should be tested against a well-built tabular learner. The next chapter compares their input representation and transfer assumptions.

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

## Further reading

- [scikit-learn lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) gives a complete tabular workflow.
- [Rolling-origin cross-validation](https://otexts.com/fpp3/tscv.html) explains how to test future predictions rather than shuffled interpolation.
- [Forecast accuracy measures](https://otexts.com/fpp3/accuracy.html) compares errors on a common scale and warns about in-sample assessment.

## Check yourself

- I can create a lag row and identify its origin, target and horizon.
- I can explain how direct and recursive multi-step forecasts use different input values.
- I can detect a feature that was published after its forecast origin.
- I can construct groupwise rolling features without crossing entity boundaries.
- I can design a backtest that reproduces the retraining and feature-update cadence.
