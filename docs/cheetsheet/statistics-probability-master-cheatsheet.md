---
title: Statistics and Probability Master Cheatsheet
sidebar_position: 22
---

# Statistics and Probability Master Cheatsheet

## Probability basics

| Method | Description | Code example |
|---|---|---|
| Mean | Average value. Sensitive to outliers. | `values = np.array([1, 2, 3, 100])`<br/>`mean = values.mean()` |
| Median | Middle value. Robust to outliers. | `median = np.median(values)` |
| Variance | Average squared deviation from the mean. | `var = np.var(values, ddof=1)` |
| Standard deviation | Square root of variance; same unit as data. | `std = np.std(values, ddof=1)` |
| Quantile | Value below which a fraction of observations fall. | `q95 = np.quantile(values, 0.95)` |
| Covariance | Measures joint variation between variables. | `cov = np.cov(x, y, ddof=1)` |
| Correlation | Normalized covariance in range -1 to 1. | `corr = np.corrcoef(x, y)[0, 1]` |

## Distributions

| Method | Description | Code example |
|---|---|---|
| Bernoulli | Binary outcome with probability `p`. | `samples = np.random.binomial(n=1, p=0.7, size=1000)` |
| Binomial | Number of successes in `n` Bernoulli trials. | `samples = np.random.binomial(n=10, p=0.3, size=1000)` |
| Normal | Bell-shaped distribution with mean and standard deviation. | `samples = np.random.normal(loc=0, scale=1, size=1000)` |
| Poisson | Count events in fixed interval. | `samples = np.random.poisson(lam=4, size=1000)` |
| Exponential | Waiting time between Poisson events. | `samples = np.random.exponential(scale=2, size=1000)` |
| Multinomial | Counts across multiple categories. | `samples = np.random.multinomial(n=20, pvals=[0.2, 0.5, 0.3])` |

## Inference and tests

| Method | Description | Code example |
|---|---|---|
| Confidence interval | Estimate range for a population parameter. | `mean = sample.mean()`<br/>`se = sample.std(ddof=1) / np.sqrt(len(sample))`<br/>`ci = (mean - 1.96 * se, mean + 1.96 * se)` |
| One-sample t-test | Tests whether sample mean differs from hypothesized mean. | `from scipy import stats`<br/>`stat, p = stats.ttest_1samp(sample, popmean=0)` |
| Two-sample t-test | Tests whether two groups have different means. | `stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)` |
| Chi-square test | Tests categorical association or goodness of fit. | `stat, p, dof, expected = stats.chi2_contingency(table)` |
| Mann-Whitney U | Non-parametric test for two independent groups. | `stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")` |
| Bootstrap CI | Resamples data to estimate uncertainty. | `boots = [np.mean(np.random.choice(sample, len(sample), replace=True)) for _ in range(5000)]`<br/>`ci = np.quantile(boots, [0.025, 0.975])` |

## Bayesian and ML metrics

| Method | Description | Code example |
|---|---|---|
| Bayes rule | Updates belief using prior, likelihood, and evidence. | `posterior = likelihood * prior / evidence` |
| MLE | Chooses parameter values maximizing likelihood. | `mu_hat = sample.mean()`<br/>`sigma_hat = sample.std(ddof=0)` |
| Entropy | Measures uncertainty of a distribution. | `p = np.array([0.2, 0.8])`<br/>`entropy = -(p * np.log2(p)).sum()` |
| Cross entropy | Measures log loss against predicted probabilities. | `loss = -np.mean(np.log(probs[np.arange(len(y)), y]))` |
| KL divergence | Measures how one distribution differs from another. | `kl = np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))` |
| Perplexity | Exponentiated average negative log likelihood. | `perplexity = np.exp(cross_entropy_loss)` |

## Common patterns

| Method | Description | Code example |
|---|---|---|
| A/B test lift | Compare conversion rates. | `rate_a = conv_a / n_a`<br/>`rate_b = conv_b / n_b`<br/>`lift = rate_b / rate_a - 1` |
| Stratified summary | Summarize by group. | `df.groupby("segment")["revenue"].agg(["count", "mean", "median", "std"])` |
| Outlier robust stats | Use median and IQR. | `q1, q3 = np.quantile(x, [0.25, 0.75])`<br/>`iqr = q3 - q1` |
| Z-score | Standardize feature values. | `z = (x - x.mean()) / x.std(ddof=1)` |
| Train metric CI | Bootstrap model metric. | `scores = [metric(y[idx], pred[idx]) for idx in bootstrap_indices]`<br/>`ci = np.quantile(scores, [0.025, 0.975])` |
| Calibration bins | Check predicted probabilities. | `bins = pd.qcut(probs, q=10, duplicates="drop")`<br/>`df.groupby(bins)["target"].mean()` |
| Class imbalance baseline | Compare against majority or random baseline. | `baseline_acc = np.bincount(y).max() / len(y)` |
| Multiple testing | Control false positives when running many tests. | `reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")` |

## Senior ML statistics

| Method | Description | Code example |
|---|---|---|
| Power analysis | Estimate sample size needed to detect an effect. | `from statsmodels.stats.power import TTestIndPower`<br/>`n = TTestIndPower().solve_power(effect_size=0.2, alpha=0.05, power=0.8)` |
| Effect size | Report practical magnitude, not only p-value. | `cohens_d = (group_b.mean() - group_a.mean()) / pooled_std` |
| Sequential testing risk | Peeking repeatedly inflates false positives unless corrected. | `# Use alpha spending, Bayesian monitoring, or pre-registered stopping rules.` |
| CUPED variance reduction | Use pre-period covariates to reduce A/B test variance. | `theta = np.cov(pre, metric)[0, 1] / np.var(pre)`<br/>`metric_cuped = metric - theta * (pre - pre.mean())` |
| Delta method | Approximate variance of transformed estimates. | `# For ratio x / y, approximate SE with gradient and covariance matrix.` |
| Bayesian beta-binomial | Model conversion uncertainty with conjugate prior. | `alpha_post = alpha_prior + conversions`<br/>`beta_post = beta_prior + visitors - conversions` |
| Hierarchical shrinkage | Stabilize small-segment estimates by borrowing strength. | `segment_rate = (segment_success + global_prior_success) / (segment_count + global_prior_count)` |
| Calibration error | Measure probability calibration quality. | `ece = sum(len(bin) / n * abs(bin.confidence.mean() - bin.accuracy.mean()) for bin in bins)` |

## Data quality and evaluation pitfalls

| Method | Description | Code example |
|---|---|---|
| Leakage check | Detect features that know the future or target. | `suspicious = [col for col in X.columns if col.lower().find("target") >= 0]` |
| Stratified bootstrap | Preserve class balance during uncertainty estimation. | `idx_pos = rng.choice(pos_idx, len(pos_idx), replace=True)`<br/>`idx_neg = rng.choice(neg_idx, len(neg_idx), replace=True)` |
| Grouped split | Avoid user/session leakage across train and test. | `GroupShuffleSplit(test_size=0.2).split(X, y, groups=user_ids)` |
| Time split | Evaluate forecasting models on future data. | `train = df[df.date < cutoff]`<br/>`test = df[df.date >= cutoff]` |
| Slice metrics | Senior evaluations always inspect subgroups. | `df.groupby("segment").apply(lambda g: roc_auc_score(g.y, g.pred))` |
| Confidence interval for metric | Quantify uncertainty around AUC/F1/accuracy. | `scores = [metric(y[idx], pred[idx]) for idx in boot_indices]`<br/>`np.quantile(scores, [0.025, 0.975])` |
| Prior probability shift | Monitor base rate changes. | `base_rate_train = y_train.mean()`<br/>`base_rate_prod = y_prod.mean()` |
| Decision threshold | Optimize threshold against business objective. | `thresholds = np.linspace(0, 1, 101)`<br/>`best = max(thresholds, key=lambda t: utility(y, prob >= t))` |
