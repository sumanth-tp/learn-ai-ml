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

:::tip Before you start
**You should already know**

- Seasonal naive and ETS baselines, and how a rolling-origin backtest is scored ([Classical forecasting](/docs/theory/timeseries/classical-forecasting)).
- Which inputs are known at the origin and which are not ([Temporal foundations](/docs/theory/timeseries/temporal-foundations)).

**Reading time:** about 40 minutes, plus the code.

**After this chapter you can**

- explain how a pretrained forecaster reads a context window, patches, and known-future inputs,
- run Chronos-2 on a CPU and score it against seasonal naive and ETS on the same origins,
- say when a pretrained model helps and when it only matches a local baseline.

:::

## In 30 seconds

A pretrained forecaster has already looked at a huge number of series, so it arrives knowing what weekly cycles and slow drifts look like. You give it a recent window of your series and it returns a forecast without any fitting. That is called zero-shot use. Think of a doctor who has seen thousands of patients and examines yours for the first time. She may be very good, or she may be no better than the nurse who has followed this one patient for years. Only a test on your own data, at your own horizon, settles which.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Context window | The recent past the model is shown. | The last 540 days. |
| Patch | A few neighbouring values treated as one token. | Width 2: 10, 12 becomes one patch. |
| Zero-shot | Used without fitting to the target series. | Call the model, get a forecast. |
| Known-future covariate | An input fixed before the origin. | A published promotion plan. |
| Past-only covariate | An input seen only up to the origin. | Measured foot traffic. |
| Quantile forecast | Several values, each with a probability of being exceeded. | The 0.5 quantile is the median. |
| Cold start | A series with very little history. | A product launched 28 days ago. |
| Normalisation | Rescaling the context by its own mean and spread. | Subtract 11, divide by 1. |

## The idea in plain words

A lagged tabular model sees a curated set of past values and covariates. A sequence model can instead consume a context window and learn a representation of its local changes, repeated structure and relations to other variables. A recurrent model updates a state as observations arrive; a temporal convolution combines nearby values; attention can connect positions farther apart. These are modelling choices, not automatic improvements. A small dataset can make a high-capacity model unstable, and a long context can increase memory, latency and the amount of irrelevant history.

Pretraining changes the starting point. Rather than fit every target series from scratch, a model is trained over a broad collection of series and then applied or adapted to a new task. In a **zero-shot** use, the target series did not participate in task-specific fitting. The phrase says how the model is used, not how accurate or unbiased it will be for a particular organisation. Domain shift, unusual units, intermittent demand, new intervention policies and the required horizon can all change performance. A strong local baseline remains necessary.

<Infographic src="/img/timeseries/pretrained-forecasts.svg" alt="Four cards distinguish three observed two-value patches, two hidden future targets, known planned events and fair comparison of versioned pretrained models." caption="Target history, hidden future targets and genuinely known future inputs have different visibility rules." />

## Worked example, step by step

A model must handle a series of sales near 11 and another near 11,000 with one set of weights. It first rescales each context using only the context. Take the context 10, 12, 10, 12, 10, 12 and a forecast horizon of two.

1. **Mean of the context.** (10 + 12 + 10 + 12 + 10 + 12) / 6 = 11.
2. **Spread.** Each value is 1 from the mean, so the standard deviation is 1.
3. **Normalise.** Subtract 11 and divide by 1 to get -1, 1, -1, 1, -1, 1.
4. **Patch.** With width 2 the context becomes three patches: (-1, 1), (-1, 1), (-1, 1). The model appends one empty slot for the two values to predict.
5. **Predict in normalised units.** Suppose the model returns (-1, 1).
6. **Undo the scaling.** Multiply by 1 and add 11 to get 10 and 12.

In words: the model works on a scale-free pattern and the rescaling is undone at the end. Using the mean of the whole series, future included, would leak the answer into step 1.

```python
import numpy as np

context = np.array([10, 12, 10, 12, 10, 12], dtype=float)
mean, std = context.mean(), context.std()
normalised = (context - mean) / std
patches = normalised.reshape(-1, 2)
predicted_normalised = np.array([-1.0, 1.0])
forecast = predicted_normalised * std + mean
print('mean', mean, 'std', std)
print('normalised patches:', patches.tolist())
print('forecast in original units:', forecast.tolist())
```

It prints mean 11.0 and standard deviation 1.0, three patches of `[-1.0, 1.0]`, and the forecast `[10.0, 12.0]`, matching the six steps above. The arithmetic illustrates the idea only. It is not the internal tensor layout of Chronos-2 or TimesFM-3.

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

### Which pretrained models are current, and which one runs here

Two model cards were read on 2026-10-08. Parameter counts, context limits and licences below come from those cards, and the page dates from the Hub's metadata for each repository.

| | Chronos-2 | TimesFM-3 |
| --- | --- | --- |
| Repository | `amazon/chronos-2` (also mirrored as `autogluon/chronos-2`) | `google/timesfm-3.0-pytorch` |
| Parameters | 120 million | 330 million |
| Licence | Apache-2.0 | TimesFM Non-Commercial License v1.0 |
| Covariates | past-only and known-future, multivariate | multiple targets, past-only and known-future |
| Limits stated | context up to 8,192 steps, prediction up to 1,024 | context patch 32, horizon patch 64 |
| Last updated on the Hub | 2026-06-05 | 2026-09-02 (announcement dated 2026-08-31) |
| Run in this chapter | yes, `chronos-forecasting` 2.3.2 on CPU | no |

Chronos-2 is the smaller of the two and has a permissive licence, so it is the one run below. TimesFM-3 was not run, so nothing here says how it compares. Both providers claim top ranks on public benchmarks. Those claims are not tested here.

### Experiment: a pretrained model against ETS on held-out series

The block builds ten new series (a seed the other chapters do not use) with a weekly pattern, a slight trend, noise and a promotion plan, and forecasts seven days ahead at eight origins. Four candidates are scored by MASE: seasonal naive, ETS, Chronos-2 on the target alone, and Chronos-2 given the promotion plan as a known-future covariate. Everything is repeated with a short context of only 28 days, which is the cold-start case a pretrained model is supposed to help.

```python
import warnings

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

warnings.filterwarnings('ignore')
torch.manual_seed(0)
rng = np.random.default_rng(21)
n, horizon, windows, per_group = 540, 7, 8, 10
shape = np.array([0.0, 2.0, 3.0, 4.0, 7.0, 14.0, 10.0])
shape = (shape - shape.mean()) / 100
days = pd.date_range('2022-01-01', periods=n, freq='D')
frames = []
for k in range(per_group):
    level = rng.uniform(50, 200)
    promo = (rng.random(n) < 0.08).astype(int)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.4 * noise[i - 1] + rng.normal(0, 0.05 * level)
    t = np.arange(n)
    y = level * (1 + 0.0004 * t + shape[t % 7] * 3 + 0.25 * promo) + noise
    frames.append(pd.DataFrame({'id': f'h{k:02d}', 'timestamp': days, 'target': y, 'promo': promo}))
df = pd.concat(frames, ignore_index=True)
scale = df[df['timestamp'] < days[-horizon * windows]].groupby('id')['target'].apply(lambda s: np.abs(s.values[7:] - s.values[:-7]).mean())
cutoffs = [days[-horizon - 1] - pd.Timedelta(days=horizon * i) for i in range(windows)]
pipe = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')

def chronos(context_len, use_promo):
    out = []
    for cutoff in cutoffs:
        ctx = df[df['timestamp'] <= cutoff].groupby('id').tail(context_len)
        future = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= cutoff + pd.Timedelta(days=horizon))]
        cols = ['id', 'timestamp', 'target'] + (['promo'] if use_promo else [])
        pred = pipe.predict_df(ctx[cols], future_df=future[['id', 'timestamp', 'promo']] if use_promo else None,
                               prediction_length=horizon, quantile_levels=[0.5], id_column='id', timestamp_column='timestamp', target='target')
        out.append(pred.merge(future[['id', 'timestamp', 'target']].rename(columns={'target': 'actual'}), on=['id', 'timestamp']))
    return pd.concat(out)

def statistical(context_len):
    d = df.rename(columns={'id': 'unique_id', 'timestamp': 'ds', 'target': 'y'})[['unique_id', 'ds', 'y']]
    sf = StatsForecast(models=[SeasonalNaive(season_length=7), AutoETS(season_length=7, model='ZZA')], freq='D', n_jobs=1)
    return sf.cross_validation(df=d, h=horizon, step_size=horizon, n_windows=windows, input_size=context_len).reset_index().rename(columns={'unique_id': 'id'})

def mase(frame, truth, pred):
    return (np.abs(frame[truth] - frame[pred]) / frame['id'].map(scale)).mean()

for label, length in (('long context (540 days)', 540), ('short context (28 days)', 28)):
    stat = statistical(length)
    assert set(stat['cutoff']) == set(cutoffs)
    plain = chronos(length, False)
    with_promo = chronos(length, True)
    print(label)
    print('  seasonal naive      %.3f' % mase(stat, 'y', 'SeasonalNaive'))
    print('  ETS                 %.3f' % mase(stat, 'y', 'AutoETS'))
    print('  Chronos-2           %.3f' % mase(plain, 'actual', 'predictions'))
    print('  Chronos-2 + promo   %.3f' % mase(with_promo, 'actual', 'predictions'))
```

**Reading the output.** With 540 days of context, seasonal naive scores 0.855, ETS 0.655 and Chronos-2 0.634. That is a tie in practice: the gap is 3 per cent. With the promotion plan as a known-future input, Chronos-2 scores 0.522, which is 18 per cent lower than without it. With 28 days of context the order changes. ETS scores 0.736, Chronos-2 0.758, and Chronos-2 with the plan 0.772. Seasonal naive is unchanged at 0.855 because it only reads the last week.

**Line by line.**

- `torch.manual_seed(0)` and a fixed data seed make the run repeatable. The `assert` checks that statsforecast and Chronos were scored at the same eight cut-offs.
- `groupby('id').tail(context_len)` cuts the context at the origin and keeps at most `context_len` rows per series, so nothing after the origin is shown.
- `future_df` carries only the promotion column for the seven days to predict. Passing it is what turns that column into a known-future covariate. Without it, the model sees the target alone.
- `input_size=context_len` makes the statistical models see the same window as Chronos.

**Interpretation.** The pretrained model matched a tuned local model on a long, clean series, and did not beat it. Its clear gain came from the covariate, which ETS here cannot use. On the cold-start case, the short context gave no advantage: Chronos-2 was 3 per cent worse than ETS, and the promotion plan did not help, plausibly because 28 days hold only a couple of promotions to learn the effect from (the run did not test that explanation). The limits are large. The data are synthetic and were built with a smooth weekly shape, ETS is well suited to them, there is one seed, and each cell is 560 forecasts from ten series. Real series with trends, outages and intermittent demand may favour a pretrained model more. Treat this as a method for testing a model on your data, not as a ranking.

<Infographic src="/img/ts-enrich/pretrained-honest.svg" alt="Two panels of bars for seasonal naive, ETS, Chronos-2 and Chronos-2 with a promotion plan. With 540 days of context Chronos-2 scores 0.634 and with the plan 0.522. With 28 days ETS scores 0.736 and Chronos-2 0.758." caption="Compare the purple Chronos-2 bar with the blue ETS bar in each panel, then look at what the promotion plan did on the left and on the right." />

## Designing with it

### Compare model families on one held-out ledger

Imagine a service forecasting the next seven days for thousands of shop-product pairs. A seasonal-naive baseline needs only a period and recent observations. A lagged tree may use prices, promotions and rolling sales. A pretrained sequence model may consume a context window and selected covariates. Each candidate must use the same origin timestamps, eligible products and final target definition. If the pretrained model sees a revised promotion plan while the tree sees the original plan, their scores are not comparable. If one produces daily forecasts and another a seven-day total, first align the output to the business decision. Store the raw forecast, interval or quantiles, latency and feature versions for every origin.

Evaluation should separate existing products with long histories, recent products, intermittent products and newly launched products. A foundation model may help a cold-start group while losing on stable seasonal products, or the reverse. One average can hide that structure. Check performance at each lead day because a model that wins at day one may lose at day seven. If probabilistic output is needed, compare coverage and width at each horizon, not only point error. A model's advertised quantiles require calibration checks before they drive safety stock.

### Adaptation has a cost and a boundary

Zero-shot use is attractive when there is little local labelled history or limited training infrastructure. Fine-tuning adds local parameters or updates pretrained ones using the organisation's series. It can improve fit to local patterns, but it also consumes representative data and creates a versioned training process. Split local data so the adaptation procedure never sees final evaluation origins. Tuning context length, normalisation or covariate choices on the test period is still test leakage even if model weights are frozen. Keep a simple no-adaptation baseline and report the incremental gain from tuning.

The choice between one global model and many local models affects operations. A single model can share information across sparse series and simplify deployment. It may also under-serve unusual high-value groups because training loss is dominated by common patterns. Per-series fitting can capture local behaviour but requires many updates, fallbacks and monitoring slices. Hybrid operation is reasonable: use one model as a baseline, route some groups to a specialised model when repeated backtests justify it, and maintain a safe fallback for failures. The routing rule itself must be fixed before final evaluation.

### Covariate masks are a data contract

A multivariate input can contain targets, historical covariates and known-future covariates. Keep a typed schema for each channel. For an energy forecast, realised temperature belongs to history, while the weather forecast available at the issue time can populate a future channel. For a retail forecast, an already published holiday calendar is known, while tomorrow's realised foot traffic is hidden. The model's ability to accept a “future covariate” does not make every future-dated database value valid. Store the publication timestamp and source version of each future input; test this contract with deliberately late data.

Patching adds edge cases. If a context length is not divisible by patch width, the model may pad, truncate or use a shorter patch. If a horizon ends mid-patch, some outputs may be masked or discarded. A library's preprocessing convention must be reproduced exactly when comparing variants. Missing observations may receive a mask token, an imputed value or a skipped timestamp; these imply different information. Confirm whether the model expects a regular cadence, and do not resample irregular data without documenting the aggregation and its effect on targets.

### Plan for deployment and audit

Measure cold start and steady-state latency separately. Loading weights can dominate an occasional forecast request, while batched inference may make per-series cost small. Record memory, processor type, batch size and model version alongside accuracy. Review licence and data-handling requirements before placing private series in an external service. If a provider updates a model behind an endpoint, pin a version or keep a replayable validation set to detect changes. A pretrained model is part of an operational forecasting system; its failure modes include stale covariates, unavailable weights, format mismatches and uncertain outputs as well as statistical error.

When a model returns several quantiles, verify that they are ordered for each horizon and that the requested quantiles are actually supported by that model version. A forecast API can expose similarly named outputs with different semantics, such as samples, quantiles or confidence intervals. Convert them only with a documented rule. For a business decision, test the resulting order or staffing policy on held-out outcomes, not merely the numerical forecast score.

Begin with a dataset card: cadence, horizons, missingness, number of series, context limits, future covariate policy and latency budget. Choose a pretrained model only after confirming its licence and API constraints from its own documentation. Reproduce preprocessing and postprocessing exactly; a model score obtained with a different normalisation or data-frequency conversion is not an apples-to-apples comparison. Cache model weights and pin a version so a deployment can be reproduced.

Backtest a pretrained model on untouched origins without tuning it to each test segment. Compare against seasonal naive, classical and tabular candidates on the same raw target values. Report error by horizon and series type, interval coverage where relevant, batch cost and failure rate. Include a cold-start slice if that is the reason for adopting pretraining. Evaluate on the target organisation's data even when a public benchmark suggests strong transfer. The training corpus may overlap public benchmark series or differ sharply from the target domain; neither possibility can be resolved from a headline rank.

## Where this stands in 2026

As of October 2026, published provider materials document both TimesFM-3 and Chronos-2 as available pretrained forecasting families. TimesFM-3 adds native multivariate and known-future-covariate support according to its authors. The model landscape can change quickly, so this chapter names versioned examples rather than a permanent winner. The engineering standard is stable: define an origin, mask unknown future values, compare strong baselines, and verify cost and reliability in the intended deployment.

## Common mistakes

1. **Reading "zero-shot" as "accurate".** It feels like a promise because the provider's benchmark ranks are high. Zero-shot only says no fitting was done. Here Chronos-2 tied ETS (0.634 against 0.655) and was slightly worse on a short context.
2. **Leaving covariates out, or passing the wrong ones.** It feels optional. The promotion plan was worth 18 per cent. Pass known-future inputs through the future frame, and keep realised future values out of it.
3. **Comparing at different origins or context lengths.** It feels equivalent. The pretrained model sees a window and ETS sees a different one, so the scores differ for a reason that has nothing to do with the model. Use the same cut-offs and cap both at the same window.
4. **Testing on public benchmark series that may be in the pretraining data.** It feels like a neutral test. The pretraining corpora are partly public, so a model may have seen the series. Evaluate on your own held-out origins.
5. **Ignoring licence and weights.** It feels like a legal footnote. TimesFM-3 is released under a non-commercial licence, while Chronos-2 is Apache-2.0. Read the card before you plan production use.

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

<details>
<summary><strong>Q5 (Easy).</strong> In the worked example the forecast came back as 10 and 12. Which two numbers undo the scaling?</summary>

The mean (11) is added and the standard deviation (1) is the multiplier. Both come from the context only, so nothing from the future is used.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> Chronos-2 scored 0.634 and ETS 0.655 with a long context. Is that a win?</summary>

Not on this evidence. The gap is 3 per cent, there is one seed and ten series, and the two models could swap places with another seed. The result that stands out is the 18 per cent gain from the promotion plan, which is a difference in information, not in model family.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> The short-context run showed no benefit from the promotion plan. Name two explanations and one test for each.</summary>

First, too few promotions in 28 days: count them and repeat with a context of 90 days. Second, the model did not learn the covariate's effect from so little history: simulate a series with a larger promotion lift and see whether the score improves. The run did not test either explanation.

</details>

## Further reading

- [Google Research: TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) describes a versioned multivariate pretrained model.
- [Chronos-2 model card](https://huggingface.co/autogluon/chronos-2) documents another pretrained forecasting family.
- [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) supplies the evaluation framework needed for both.

- [Chronos-2 model card](https://huggingface.co/autogluon/chronos-2) states 120M parameters, Apache-2.0, a context up to 8,192 steps and support for past and known-future covariates (opened 2026-10-08; `chronos-forecasting` 2.3.2 was run).
- [TimesFM 3.0 model card](https://huggingface.co/google/timesfm-3.0-pytorch) states the non-commercial licence and the patch lengths (opened 2026-10-08; the model was not run).
- [Google Research: TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) is dated 31 August 2026 and describes masked future patches, a single forward pass and covariate handling (opened 2026-10-08).

## Check yourself

- I can identify which context, target and covariate values a model may see at an origin.
- I can calculate a patch count and explain how a horizon is hidden.
- I can distinguish zero-shot use from a claim of good target-domain accuracy.
- I can specify the baseline, slices, costs and versions needed for a fair model comparison.
- I can normalise a context with its own mean and spread, patch it, and undo the scaling on a forecast.
- I can run Chronos-2 on a CPU with and without a known-future covariate and score it against seasonal naive and ETS on the same origins.
- I can say what the experiment supports (a tie with ETS on long context, a covariate gain) and what it does not (a ranking of pretrained models, a cold-start benefit).

## Where to go next

Continue with [evaluation and operations](/docs/theory/timeseries/evaluation-and-operations), which turns point forecasts into intervals and monitors them after release. For the local baselines used above, return to [classical forecasting](/docs/theory/timeseries/classical-forecasting).
