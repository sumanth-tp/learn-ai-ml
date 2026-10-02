---
title: "Time series · Deep and pretrained forecasting"
sidebar_label: "Deep and pretrained models"
sidebar_position: 4
slug: /theory/timeseries/pretrained-forecasting
description: "Context windows, patches, covariates and fair evaluation of pretrained forecasters."
tags: [time-series, deep-learning, forecasting]
---

import Infographic from '@site/src/components/Infographic';
import ForecastPatchLab from '@site/src/components/viz/ForecastPatchLab';

**In one line.** Deep forecasters learn representations from sequences, while pretrained forecasters reuse patterns learned across many series; neither removes the need to define what was visible at the forecast origin.

## The idea in plain words

A lagged tabular model sees a curated set of past values and covariates. A sequence model can instead consume a context window and learn a representation of its local changes, repeated structure and relations to other variables. A recurrent model updates a state as observations arrive; a temporal convolution combines nearby values; attention can connect positions farther apart. These are modelling choices, not automatic improvements. A small dataset can make a high-capacity model unstable, and a long context can increase memory, latency and the amount of irrelevant history.

Pretraining changes the starting point. Rather than fit every target series from scratch, a model is trained over a broad collection of series and then applied or adapted to a new task. In a **zero-shot** use, the target series did not participate in task-specific fitting. The phrase says how the model is used, not how accurate or unbiased it will be for a particular organisation. Domain shift, unusual units, intermittent demand, new intervention policies and the required horizon can all change performance. A strong local baseline remains necessary.

<Infographic src="/img/timeseries/pretrained-forecasts.svg" alt="Six observed values become three two-value context patches; an unseen two-step future patch remains masked while a separately labelled known-future covariate can be visible." caption="Target history, hidden future targets and genuinely known future inputs have different visibility rules." />

## How it works

### Encode a context window

At origin $o$, a model receives a context $y_{o-L+1:o}$ of length $L$ and produces an $H$-step forecast. It may also receive past covariates and known-future covariates. The length $L$ should cover useful cycles without assuming that all earlier history helps. A series with a weekly pattern sampled hourly may need enough days to show that pattern. A sudden regime change may make older history misleading. Scale handling matters: a model trained across tiny and enormous targets needs normalisation or scale-aware tokens, with the inverse transform applied to forecasts. Normalisation parameters derived from future targets would leak.

**Patching** groups adjacent observations into tokens. For the synthetic context $[10,12,11,13,10,12]$, patch width two produces `[[10,12], [11,13], [10,12]]`. A horizon of two corresponds to one future patch in the simplified lab. The future target patch is hidden; it is a placeholder whose values must be predicted. This arithmetic illustrates the visibility boundary, not the internal tensor shape of a particular model. Real systems also need rules for an incomplete final patch, missing values and multiple series with different cadences.

Sequence architectures can produce forecasts autoregressively, predicting one future step or patch and feeding it back, or directly, producing the full horizon in one pass. Autoregression reuses a smaller prediction head but compounds model error and raises latency with horizon. Direct output can avoid that loop, but its output structure and training task must cover the required horizons. Neither approach guarantees coherent sums across related series. A business that forecasts both item and category demand may need reconciliation or a model trained with hierarchy constraints.

### Keep covariate visibility explicit

Past-only covariates include realised foot traffic and measured temperature. Their future values must be masked unless a separate forecast is provided. Known-future covariates can include a fixed holiday calendar or a promotion schedule published before the origin. A weather **forecast** is a known input at the origin; realised future weather is not. Its error should be part of historical evaluation. The same column name can have different status at different lead times, so store the source version and publication time.

Multivariate forecasting uses information from several related target or covariate series. It can help when series co-move, but it can also propagate errors and confuse correlation with intervention effects. The training and serving sets must agree on entity order, missing-series policy, units and timestamps. Joining future values from one target series while forecasting another is leakage if those values were not available simultaneously. A visual grid of series and time can help audit which cells are observed, predicted or known by schedule.

### Read model claims carefully

[Google Research's TimesFM-3 announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) describes a 2026 pretrained model that supports multiple targets, past covariates and known-future covariates, and predicts masked future patches in one pass. The provider also describes its internal patching and attention scheme. That is a concrete model architecture, not a universal definition of a foundation forecaster. The [Chronos-2 model card](https://huggingface.co/autogluon/chronos-2) documents another pretrained family with its own input format and constraints. Provider benchmark claims should be treated as reasons to test a model, not as an expected result on a private series. Confirm the exact model version, licence, context limit, horizon support, covariate semantics, hardware footprint and output type before a comparison.

Fine-tuning can adapt a pretrained model when there is sufficient representative data and the deployment cost is justified. It also raises the risk of overfitting to a narrow backtest. Separate origins for model selection, adaptation and final evaluation. If a model consumes a normalised context, use the exact same transformation in evaluation and serving. If a service must forecast thousands of series at once, measure total batch latency and memory, not only per-series inference time. If forecasts are needed after each event, a compact local model may be operationally preferable even when a large model has a slightly better batch metric.

## A real system that works this way

TimesFM-3's [original project description](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) offers a concrete design: observed patches and masked horizon patches enter a model that can also see covariates known in advance. A retail planner might pass historical sales and a promotion plan to forecast several related products. The validity of that setup depends on the promotion plan version being available when the forecast is issued. A planner should compare the pretrained output with seasonal naive and feature-based alternatives at the same origins, record uncertainty by horizon, and review failures during promotions and stockouts.

## Code you can run

This dependency-free example shows the lab's default patch arithmetic. It does not load or pretend to run a pretrained model.

```python
context = [10, 12, 11, 13, 10, 12]
patch_width = 2
horizon = 2
patches = [context[i:i + patch_width] for i in range(0, len(context), patch_width)]
future_patch_count = (horizon + patch_width - 1) // patch_width
print('context patches:', patches)
print('context patch count:', len(patches))
print('masked future patches:', future_patch_count)
```

It prints three context patches and one masked future patch. The values of that future patch are deliberately absent.

<ForecastPatchLab />

Vary patch width and horizon. The table keeps context values visible and future target values hidden. Flags marked as known-future inputs are separate from target observations. At the default settings it reproduces the code's three context patches and one future patch.

```python
origin = 6
features = [
    ('past sales', 6, 'observed'),
    ('promotion plan', 8, 'published at origin 4'),
    ('realised weather', 8, 'published at origin 8'),
]
for name, event_time, availability in features:
    safe = name != 'realised weather'
    print(name, event_time, availability, 'available' if safe else 'hidden')
```

The explicit labels are part of the example's premise. In a real system, calculate `safe` from recorded publication timestamps, not a hard-coded feature name. A weather forecast published at origin six could be valid, while realised weather at time eight is not.

## Designing with it

Begin with a dataset card: cadence, horizons, missingness, number of series, context limits, future covariate policy and latency budget. Choose a pretrained model only after confirming its licence and API constraints from its own documentation. Reproduce preprocessing and postprocessing exactly; a model score obtained with a different normalisation or data-frequency conversion is not an apples-to-apples comparison. Cache model weights and pin a version so a deployment can be reproduced.

Backtest a pretrained model on untouched origins without tuning it to each test segment. Compare against seasonal naive, classical and tabular candidates on the same raw target values. Report error by horizon and series type, interval coverage where relevant, batch cost and failure rate. Include a cold-start slice if that is the reason for adopting pretraining. Evaluate on the target organisation's data even when a public benchmark suggests strong transfer. The training corpus may overlap public benchmark series or differ sharply from the target domain; neither possibility can be resolved from a headline rank.

## Where this stands in 2026

As of October 2026, published provider materials document both TimesFM-3 and Chronos-2 as available pretrained forecasting families. TimesFM-3 adds native multivariate and known-future-covariate support according to its authors. The model landscape can change quickly, so this chapter names versioned examples rather than a permanent winner. The engineering standard is stable: define an origin, mask unknown future values, compare strong baselines, and verify cost and reliability in the intended deployment.

## Practice questions

<details>
<summary>What does zero-shot mean for a forecasting deployment?</summary>

The model is used on the target task without task-specific fitting on that target dataset. It does not mean the model has never seen similar series, nor that it will beat a local baseline.

</details>

<details>
<summary>Why are future holiday flags and future realised sales treated differently?</summary>

A holiday calendar is known at the origin; future sales are the unknown target. Visibility depends on availability, not on whether a value belongs to a future date.

</details>

<details>
<summary>When can a weather value be a known-future covariate?</summary>

When it is a weather forecast published before the forecasting origin. Backtesting should use the historical forecast version, including its error, rather than the realised weather.

</details>

<details>
<summary>What does the patch lab prove about TimesFM-3 accuracy?</summary>

Nothing. It illustrates context grouping and visibility. Accuracy must be measured on held-out origins for the intended series, horizon and operating conditions.

</details>

## Further reading

- [Google Research: TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) describes a versioned multivariate pretrained model.
- [Chronos-2 model card](https://huggingface.co/autogluon/chronos-2) documents another pretrained forecasting family.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) supplies the evaluation framework needed for both.

## Check yourself

- I can identify which context, target and covariate values a model may see at an origin.
- I can calculate a patch count and explain how a horizon is hidden.
- I can distinguish zero-shot use from a claim of good target-domain accuracy.
- I can specify the baseline, slices, costs and versions needed for a fair model comparison.
