# Model-selection report: support-ticket triage

Run `run-778c1e63c1` · profile `offline` · splits test, private · dataset `6aa928d82ff97147` · prompts `v3` · judge `fake:judge-large`

## Recommendation

**fake:frontier-large** (runner-up: fake:balanced-mini). Highest weighted score on the Pareto frontier (0.63 vs 0.40 for fake:balanced-mini).

### Caveats

- Position bias: 35% of pairwise verdicts flip when the order is swapped; pairwise results use swap-and-average.
- Verbosity bias: content-free padding raises scores by +0.26 (p=0.000).
- Self-preference suggested but not significant (p=0.112): the judge over-scores its own family's replies by +0.29 relative to humans; treat that family's reply scores as optimistic.
- Possible benchmark contamination: fake:leaky-tuned. Excluded from the recommendation; compare private-split numbers before trusting any public score for them.
- The recommended model shares the judge's family (frontier); re-score replies with a judge from another family before signing off.
- 80 test items can only detect composite differences of about 0.048; about 201 are needed for 0.03.

## Results on the test split (80 items)

| Model | Composite [95% CI] | Accuracy | Macro-F1 [95% CI] | JSON valid | Field acc. | Reply (1–5) | p95 latency | Cost / 1k tickets | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fake:balanced-mini | 0.863 [0.826, 0.896] | 85.0% | 0.852 [0.758, 0.920] | 100.0% | 90.8% | 4.31 | 1,356 ms | $0.1064 | pass |
| fake:frontier-large | 0.931 [0.906, 0.954] | 93.8% | 0.935 [0.870, 0.985] | 100.0% | 91.8% | 4.76 | 2,880 ms | $1.79 | pass |
| fake:leaky-tuned | 0.966 [0.954, 0.976] | 100.0% | 1.000 [1.000, 1.000] | 100.0% | 100.0% | 4.54 | 1,566 ms | $0.2079 | contamination suspected (see Contamination) |
| fake:local-8b | 0.730 [0.685, 0.772] | 70.0% | 0.699 [0.585, 0.784] | 87.5% | 75.8% | 3.93 | 2,573 ms | $0.0000 | JSON validity 87.5% < 95% |
| fake:verbose-mid | 0.868 [0.836, 0.897] | 86.2% | 0.864 [0.771, 0.930] | 98.8% | 88.8% | 4.41 | 3,268 ms | $1.36 | pass |

Pareto frontier among models that pass the gates (quality ↑, cost ↓, p95 latency ↓): fake:balanced-mini, fake:frontier-large, fake:verbose-mid

## Weighted decision matrix

Weights: quality 0.6, cost 0.25, latency 0.15. Scores are min-max normalised across the models that pass the gates.

| Model | Quality | Cost | Latency | Total |
| --- | --- | --- | --- | --- |
| fake:frontier-large | 1.00 | 0.00 | 0.20 | **0.63** |
| fake:balanced-mini | 0.00 | 1.00 | 1.00 | **0.40** |
| fake:verbose-mid | 0.07 | 0.26 | 0.00 | **0.11** |

## Significance against the baseline (fake:balanced-mini)

| Candidate | Composite diff [95% CI] | p (bootstrap) | p (permutation) | McNemar (only cand. / only base) | p (McNemar) | Significant after Holm |
| --- | --- | --- | --- | --- | --- | --- |
| fake:frontier-large | +0.068 [+0.035, +0.101] | 0.000 | 0.000 | 9 / 2 | 0.065 | yes |
| fake:leaky-tuned | +0.102 [+0.069, +0.140] | 0.000 | 0.000 | 12 / 0 | 0.000 | yes |
| fake:local-8b | -0.133 [-0.191, -0.077] | 0.000 | 0.000 | 7 / 19 | 0.031 | yes |
| fake:verbose-mid | +0.005 [-0.038, +0.050] | 0.833 | 0.843 | 10 / 9 | 1.000 | no |

Pairwise reply win rate against the baseline (swap-and-average):

| Candidate | Win rate [95% CI] | Order consistency |
| --- | --- | --- |
| fake:frontier-large | 0.62 [0.56, 0.69] | 38.8% |
| fake:verbose-mid | 0.45 [0.37, 0.53] | 57.5% |
| fake:local-8b | 0.40 [0.32, 0.49] | 73.8% |
| fake:leaky-tuned | 0.55 [0.48, 0.61] | 52.5% |

## Sample size

Between fake:frontier-large and fake:verbose-mid: SD of per-item composite differences 0.152, discordant classification rate 15.0%.
With 80 items the minimum detectable effect is **0.048** (80% power, α = 0.05).
Detecting 0.03 needs about **201** items (accuracy via McNemar: 1306).

## Judge calibration (fake:judge-large, 40 human-labelled pairs)

| Check | Value |
| --- | --- |
| Spearman vs human mean | 0.815 |
| Quadratic-weighted kappa vs human median | 0.768 |
| Fleiss' kappa, humans only (ceiling) | 0.336 |
| Fleiss' kappa, humans + judge | 0.385 |
| Pairwise kappa, single order → swapped | 0.323 → 0.448 |
| Position consistency under swap | 65.0% |
| Verbosity: score change from padding | +0.26 (p=0.000) |
| Self-preference: own-family residual minus others | +0.29 (p=0.112) |
| Meta-judge agreement on worst cases | 87.5% |
| Trusted | yes |

Mitigation ablation (Spearman or kappa with humans):

- anchored+reference (default): 0.815
- no anchors: 0.667
- no reference: 0.684
- bare prompt: 0.509
- pairwise kappa: single order: 0.323
- pairwise kappa: swap-and-average: 0.448

## Contamination

| Model | 8-gram probe (test) | Probe (private) | Test quality | Private quality | Gap | Flagged |
| --- | --- | --- | --- | --- | --- | --- |
| fake:frontier-large | 0.00 | 0.00 | 0.931 | 0.929 | +0.002 | no |
| fake:balanced-mini | 0.00 | 0.00 | 0.863 | 0.860 | +0.004 | no |
| fake:verbose-mid | 0.00 | 0.00 | 0.868 | 0.855 | +0.013 | no |
| fake:local-8b | 0.00 | 0.00 | 0.730 | 0.749 | -0.019 | no |
| fake:leaky-tuned | 1.00 | 0.05 | 0.966 | 0.797 | +0.168 | yes |

## Slices (mean composite)

- **ambiguous**: fake:balanced-mini 0.788; fake:frontier-large 0.937; fake:leaky-tuned 0.966; fake:local-8b 0.706; fake:verbose-mid 0.902
- **no_order_id**: fake:balanced-mini 0.886; fake:frontier-large 0.937; fake:leaky-tuned 0.977; fake:local-8b 0.729; fake:verbose-mid 0.895

---
Billed this run: $2.35 · cache entries: 3231 · duration: 2.85 s. Re-run after adding a model to `data/models.toml`; cached calls are free.