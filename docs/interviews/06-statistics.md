---
title: Statistics & Probability Interview Questions
sidebar_position: 6
sidebar_label: "6 · Statistics and probability"
---

# Statistics & Probability Interview Questions

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. After revising these foundations, continue with the [practical topic bank](09-ml-statistics.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](23-tools-versions.md) for current behaviour and [coding labs](21-coding-labs.md) for runnable implementations.
:::

100 essential statistics interview questions for AI/ML roles with code examples.

---

<details>
<summary><strong>1. What is the Central Limit Theorem?</strong></summary>

**Answer:**
The CLT states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the population distribution. Requires samples to be independent and identically distributed (i.i.d.).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Population: heavily skewed exponential distribution
population = np.random.exponential(scale=2, size=100000)
print(f"Population: mean={population.mean():.2f}, skew={stats.skew(population):.2f}")

# Sample means for different sample sizes
for n in [5, 30, 100, 500]:
    sample_means = [np.random.choice(population, n).mean() for _ in range(1000)]
    print(f"n={n:3d}: mean={np.mean(sample_means):.2f}, "
          f"std={np.std(sample_means):.3f} (theory: {2/np.sqrt(n):.3f}), "
          f"skew={stats.skew(sample_means):.3f}")

# Key implications:
# Standard error of the mean = sigma / sqrt(n)
# Enables confidence intervals and hypothesis tests even for non-normal populations
```

**Interview Tip:** CLT justifies most statistical tests (t-test, z-test). The standard error = sigma/sqrt(n). CLT kicks in at n >= 30 for most distributions, but more skewed distributions need larger n.

</details>

<details>
<summary><strong>2. What is p-value and statistical significance?</strong></summary>

**Answer:**
The p-value is the probability of obtaining results at least as extreme as observed, assuming H₀ is true. Small p-value suggests evidence against H₀. A result is "statistically significant" if p < α (typically 0.05).

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Example: testing if mean > 0
group_A = np.random.normal(0, 1, 100)    # control (mean=0)
group_B = np.random.normal(0.3, 1, 100)  # treatment (mean=0.3)

# One-sample t-test: is group_B mean significantly different from 0?
t_stat, p_val = stats.ttest_1samp(group_B, popmean=0)
print(f"One-sample t-test: t={t_stat:.3f}, p={p_val:.4f}")
print(f"Significant at alpha=0.05: {p_val < 0.05}")

# Two-sample t-test: is group_B different from group_A?
t_stat, p_val = stats.ttest_ind(group_A, group_B)
print(f"\nTwo-sample t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Common misconceptions about p-value:
print("\np-value does NOT mean:")
print("  - Probability H0 is true")
print("  - Probability the result is due to chance")
print("  - Effect size or practical significance")
print("\np-value DOES mean:")
print("  - Probability of observing this data (or more extreme) if H0 true")

# Multiple testing: Bonferroni correction
n_tests = 20
alpha = 0.05
bonferroni_alpha = alpha / n_tests
print(f"\nBonferroni correction for {n_tests} tests: alpha = {bonferroni_alpha:.4f}")
```

**Interview Tip:** p < 0.05 doesn't prove anything — it's probabilistic evidence. Distinguish statistical significance (p-value) from practical significance (effect size). Multiple comparisons inflate Type I error — use Bonferroni or FDR correction.

</details>

<details>
<summary><strong>3. What is hypothesis testing? Type I and Type II errors.</strong></summary>

**Answer:**
Hypothesis testing evaluates evidence against a null hypothesis (H₀). Type I error (α): reject H₀ when it's true (false positive). Type II error (β): fail to reject H₀ when it's false (false negative). Power = 1 - β.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulation to understand Type I and II errors
def simulate_errors(true_effect, n=50, alpha=0.05, n_simulations=1000):
    type_I = 0   # false positives (reject H0 when true)
    type_II = 0  # false negatives (fail to reject H0 when false)

    for _ in range(n_simulations):
        # H0 true: no effect (true_effect=0)
        group_A = np.random.normal(0, 1, n)
        group_B = np.random.normal(true_effect, 1, n)

        _, p = stats.ttest_ind(group_A, group_B)
        reject = p < alpha

        if true_effect == 0 and reject:
            type_I += 1  # H0 true, but rejected
        elif true_effect != 0 and not reject:
            type_II += 1  # H0 false, but not rejected

    power = 1 - type_II / n_simulations if true_effect != 0 else None
    return type_I / n_simulations, type_II / n_simulations, power

t1, _, _ = simulate_errors(true_effect=0)
print(f"Type I error rate (alpha=0.05): {t1:.3f}")  # ~0.05

_, t2, power = simulate_errors(true_effect=0.5)
print(f"Type II error rate (effect=0.5, n=50): {t2:.3f}")
print(f"Power: {power:.3f}")

# Power analysis
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
power = analysis.solve_power(effect_size=0.5, nobs1=50, alpha=0.05)
print(f"\nPower for d=0.5, n=50: {power:.3f}")

n_required = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(f"Sample size for 80% power, d=0.5: {int(n_required)}")
```

**Interview Tip:** Trade-off: reducing α (more stringent) increases Type II error. Power increases with: larger sample, larger effect, less variance, higher α. Always compute required sample size BEFORE collecting data.

</details>

<details>
<summary><strong>4. What is Bayes' theorem?</strong></summary>

**Answer:**
Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B). Updates prior belief with new evidence to get posterior belief. Foundational for Bayesian inference, spam filters, medical diagnosis.

```python
import numpy as np

# Medical test example
# Disease prevalence: 1% (prior)
# Test sensitivity (TPR): 95% (P(positive|disease))
# Test specificity: 90% (P(negative|no disease))

P_disease = 0.01               # prior
P_positive_given_disease = 0.95  # sensitivity (recall)
P_positive_given_no_disease = 0.10  # false positive rate = 1 - specificity

# P(positive) = P(positive|disease) * P(disease) + P(positive|no disease) * P(no disease)
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * (1 - P_disease))

# P(disease|positive) = P(positive|disease) * P(disease) / P(positive)
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"P(disease) = {P_disease:.2%}")
print(f"P(positive) = {P_positive:.4f}")
print(f"P(disease|positive) = {P_disease_given_positive:.4f} = {P_disease_given_positive:.2%}")
print("\nSurprising: even with 95% sensitivity, positive test only ~8.7% chance of disease!")

# Iterative Bayesian updating
prior = 0.01
for test_result in ["positive", "positive"]:  # two positive tests
    if test_result == "positive":
        likelihood = P_positive_given_disease
        false_alarm = P_positive_given_no_disease
    else:
        likelihood = 1 - P_positive_given_disease
        false_alarm = 1 - P_positive_given_no_disease

    posterior = (likelihood * prior) / (likelihood * prior + false_alarm * (1 - prior))
    print(f"After {test_result} test: P(disease) = {posterior:.4f}")
    prior = posterior  # posterior becomes new prior
```

**Interview Tip:** Base rate neglect: even accurate tests can have low PPV when disease is rare. Key terms: prior = P(H), likelihood = P(D|H), posterior = P(H|D). Bayesian updating is sequential — posterior from one test is prior for the next.

</details>

<details>
<summary><strong>5. What is the difference between probability and statistics?</strong></summary>

**Answer:**
Probability: forward problem — given known model/parameters, compute probability of outcomes. Statistics: inverse problem — given observed data, infer model/parameters. Probability is deductive (mathematical), statistics is inductive (from evidence).

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Probability: given distribution, compute probabilities
# "A fair coin is flipped 10 times. P(exactly 6 heads)?"
from scipy.stats import binom
p_exactly_6 = binom.pmf(6, n=10, p=0.5)
print(f"P(exactly 6 heads in 10 flips): {p_exactly_6:.4f}")
print(f"P(6 or more heads): {1 - binom.cdf(5, n=10, p=0.5):.4f}")

# Statistics: given observations, infer parameters
# "We observed 7 heads in 10 flips. Is the coin fair?"
observed_heads = 7
n_flips = 10

# MLE estimate
p_mle = observed_heads / n_flips
print(f"\nMLE of p (heads probability): {p_mle}")

# Binomial test: is p=0.5?
p_value = stats.binom_test(observed_heads, n_flips, p=0.5, alternative="two-sided")
print(f"Binomial test p-value: {p_value:.4f}")
print(f"Significant at alpha=0.05: {p_value < 0.05}")  # 7/10 not significant

# Confidence interval for p
ci_low, ci_high = stats.binom.interval(0.95, n_flips, p_mle)
print(f"95% CI for heads count: [{ci_low}, {ci_high}]")
```

**Interview Tip:** Frequentist statistics: parameters are fixed, data is random. Bayesian statistics: parameters are random (have distributions), data is fixed. Both start from probability theory.

</details>

<details>
<summary><strong>6. What are probability distributions important for ML?</strong></summary>

**Answer:**
Key distributions: Normal (CLT, linear models), Bernoulli/Binomial (binary classification), Categorical/Multinomial (multiclass), Poisson (count data), Exponential (time between events), Beta (probabilities), Dirichlet (topic models).

```python
import numpy as np
from scipy import stats

# Normal distribution
normal = stats.norm(loc=0, scale=1)
print(f"Normal: P(-1<x<1) = {normal.cdf(1) - normal.cdf(-1):.4f}")  # 68%
print(f"Normal: P(-2<x<2) = {normal.cdf(2) - normal.cdf(-2):.4f}")  # 95%
print(f"Normal: P(-3<x<3) = {normal.cdf(3) - normal.cdf(-3):.4f}")  # 99.7%

# Binomial
binom = stats.binom(n=100, p=0.3)
print(f"\nBinomial(100, 0.3): mean={binom.mean()}, var={binom.var()}")

# Poisson
poisson = stats.poisson(mu=5)  # avg 5 events/unit time
print(f"\nPoisson(5): P(X=0) = {poisson.pmf(0):.4f}, P(X>10) = {1-poisson.cdf(10):.4f}")

# Beta distribution (for probabilities/proportions)
beta = stats.beta(a=2, b=5)  # prior for coin bias
print(f"\nBeta(2,5): mean={beta.mean():.3f}, mode={(2-1)/(2+5-2):.3f}")

# Exponential (memoryless)
exp = stats.expon(scale=2)  # mean=2 minutes
print(f"\nExponential(scale=2): P(X<1) = {exp.cdf(1):.4f}")

# Generate samples
samples = {
    "Normal": np.random.normal(0, 1, 1000),
    "Binomial": np.random.binomial(10, 0.3, 1000),
    "Poisson": np.random.poisson(5, 1000),
    "Exponential": np.random.exponential(2, 1000),
}
for name, s in samples.items():
    print(f"{name}: mean={s.mean():.2f}, std={s.std():.2f}")
```

**Interview Tip:** Know the mean and variance formulas. Normal: mean=mu, var=sigma^2. Binomial: mean=np, var=np(1-p). Poisson: mean=var=lambda. Exponential: mean=1/lambda, var=1/lambda^2.

</details>

<details>
<summary><strong>7. What is correlation vs causation?</strong></summary>

**Answer:**
Correlation measures linear association between two variables (Pearson r: -1 to 1). Correlation ≠ causation — confounding variables, reverse causation, and spurious correlation all produce correlation without causal link.

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 100

# Example: Spurious correlation via confounder
# Ice cream sales and drowning rates both increase in summer
summer = np.random.normal(0, 1, n)  # confounding variable (temperature/summer)
ice_cream = 2 * summer + np.random.normal(0, 0.5, n)
drowning = 1.5 * summer + np.random.normal(0, 0.5, n)

r_spurious, p = stats.pearsonr(ice_cream, drowning)
print(f"Correlation ice cream vs drowning: r={r_spurious:.3f}, p={p:.4f}")
print("High correlation, but ice cream doesn't cause drowning!")

# Partial correlation (controlling for confounder)
# Residuals after regressing out summer
from numpy.linalg import lstsq
def partial_correlation(x, y, z):
    """Correlation between x and y after controlling for z"""
    def residuals(a, b):
        coef = lstsq(np.column_stack([b, np.ones(len(b))]), a, rcond=None)[0]
        return a - np.column_stack([b, np.ones(len(b))]) @ coef

    res_x = residuals(x, z)
    res_y = residuals(y, z)
    return stats.pearsonr(res_x, res_y)

r_partial, p_partial = partial_correlation(ice_cream, drowning, summer)
print(f"Partial correlation (controlling for summer): r={r_partial:.3f}, p={p_partial:.4f}")

# Correlation types
x = np.random.randn(100)
y_linear = 2*x + np.random.randn(100)*0.5
y_monotone = np.exp(x) + np.random.randn(100)*0.5
y_none = np.random.randn(100)

print(f"\nPearson r (linear):  {stats.pearsonr(x, y_linear)[0]:.3f}")
print(f"Spearman r (monotone): {stats.spearmanr(x, y_monotone)[0]:.3f}")
print(f"Pearson r (no relation): {stats.pearsonr(x, y_none)[0]:.3f}")
```

**Interview Tip:** Spearman correlation is rank-based (handles monotone non-linear relationships). Confounders, mediators, and colliders: know Simpson's Paradox. Causal inference methods: RCT, IV, difference-in-differences, propensity score matching.

</details>

<details>
<summary><strong>8. What is confidence interval?</strong></summary>

**Answer:**
A 95% confidence interval means: if we repeat the experiment many times, 95% of constructed intervals will contain the true parameter. It does NOT mean there's 95% probability the true value is in THIS specific interval.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

true_mean = 5.0
population = np.random.normal(true_mean, 2, 100000)

# Simulate many CIs and check coverage
n_experiments = 1000
n_samples = 30
alpha = 0.05
contains_true = 0

for _ in range(n_experiments):
    sample = np.random.choice(population, n_samples)
    mean = sample.mean()
    se = sample.std() / np.sqrt(n_samples)
    t_crit = stats.t.ppf(1 - alpha/2, df=n_samples-1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    if ci_low <= true_mean <= ci_high:
        contains_true += 1

print(f"Coverage of 95% CI: {contains_true/n_experiments:.3f} (should be ~0.95)")

# Compute CI for a real sample
sample = np.random.choice(population, 50)
mean = sample.mean()
se = sample.std() / np.sqrt(len(sample))
ci = stats.t.interval(0.95, df=len(sample)-1, loc=mean, scale=se)
print(f"\nSample mean: {mean:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"True mean: {true_mean} ({'inside' if ci[0] <= true_mean <= ci[1] else 'outside'} CI)")

# CI width depends on: sample size, confidence level, population variance
for n in [10, 30, 100, 500]:
    se = 2 / np.sqrt(n)  # sigma=2
    t_crit = stats.t.ppf(0.975, df=n-1)
    width = 2 * t_crit * se
    print(f"n={n:3d}: CI width = {width:.3f}")
```

**Interview Tip:** Wider CI = more uncertainty. CI narrows proportionally to 1/sqrt(n). Distinction from credible interval: Bayesian credible interval does give a probability statement about the parameter.

</details>

<details>
<summary><strong>9. What is the law of large numbers?</strong></summary>

**Answer:**
The Law of Large Numbers states that sample statistics converge to population parameters as sample size grows. Weak LLN: convergence in probability. Strong LLN: almost sure convergence. Underpins ML — large training sets provide reliable estimates.

```python
import numpy as np

np.random.seed(42)

# Demonstrate convergence of sample mean to population mean
true_mean = 3.0
true_std = 2.0

sample_sizes = [10, 100, 1000, 10000, 100000]
for n in sample_sizes:
    samples = np.random.normal(true_mean, true_std, n)
    sample_mean = samples.mean()
    error = abs(sample_mean - true_mean)
    print(f"n={n:7d}: sample_mean={sample_mean:.4f}, |error|={error:.4f}")

# Monte Carlo integration (application of LLN)
# Estimate pi using random points in unit square
n_points = 1000000
x = np.random.uniform(0, 1, n_points)
y = np.random.uniform(0, 1, n_points)
inside = (x**2 + y**2) <= 1.0

pi_estimate = 4 * inside.mean()  # LLN: average converges to expected value
print(f"\nMonte Carlo pi estimate: {pi_estimate:.5f} (true: {np.pi:.5f})")

# LLN in ML context: loss averaging
# With more data, empirical risk -> expected risk
import numpy as np
for n_train in [100, 1000, 10000]:
    X = np.random.randn(n_train, 5)
    true_w = np.array([1, -2, 0.5, 3, -1])
    y = X @ true_w + np.random.randn(n_train) * 0.5
    # OLS estimate converges to true coefficients as n -> inf
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"n={n_train}: ||w_hat - w_true|| = {np.linalg.norm(w_hat - true_w):.4f}")
```

**Interview Tip:** LLN doesn't say sample statistics are equal to population parameters for finite n — just that they converge. Combined with CLT: we know not just that they converge, but at what rate and with what distribution.

</details>

<details>
<summary><strong>10. What is maximum likelihood estimation (MLE)?</strong></summary>

**Answer:**
MLE finds parameter values that maximize the probability of observing the training data. For regression with Gaussian noise, MLE reduces to minimizing MSE. For classification, MLE minimizes cross-entropy.

```python
import numpy as np
from scipy.optimize import minimize
from scipy import stats

np.random.seed(42)

# MLE for Gaussian: find mu and sigma that maximize likelihood
data = np.random.normal(loc=5, scale=2, size=100)

def neg_log_likelihood(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # ensure positive
    return -np.sum(stats.norm.logpdf(data, mu, sigma))

result = minimize(neg_log_likelihood, x0=[0, 0], method="L-BFGS-B")
mu_mle, sigma_mle = result.x[0], np.exp(result.x[1])
print(f"MLE: mu={mu_mle:.3f}, sigma={sigma_mle:.3f}")
print(f"Closed form: mu={data.mean():.3f}, sigma={data.std():.3f}")

# MLE for Bernoulli: p = sample proportion
binary = np.random.binomial(1, 0.7, 1000)
p_mle = binary.mean()  # analytical MLE
print(f"\nBernoulli MLE: p={p_mle:.3f} (true: 0.7)")

# MLE for linear regression: equivalent to OLS when errors are Gaussian
X = np.random.randn(200, 3)
true_beta = np.array([2.0, -1.0, 0.5])
y = X @ true_beta + np.random.randn(200) * 1.0  # Gaussian noise

beta_mle = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"\nLinear regression MLE: {beta_mle.round(3)} (true: {true_beta})")

# Why log-likelihood? Products become sums, numerically stable
# L = prod_i P(x_i | theta) -> log L = sum_i log P(x_i | theta)
```

**Interview Tip:** MLE is biased for small samples (e.g., MLE sigma is biased — divide by n-1 for unbiased). MAP = MLE + prior. L2 regularization corresponds to Gaussian prior in MAP framework.

</details>

<details>
<summary><strong>11. What is variance and standard deviation?</strong></summary>

Variance = E[(X - mu)^2] = mean of squared deviations. Standard deviation = sqrt(variance). SD is in same units as the data, making it more interpretable. Sample variance divides by n-1 (Bessel's correction) for unbiased estimation.
</details>

<details>
<summary><strong>12. What is covariance and correlation?</strong></summary>

Covariance = E[(X-muX)(Y-muY)]: measures joint variability (scale-dependent). Correlation = Cov(X,Y) / (sigma_X * sigma_Y): normalized to [-1, 1], scale-independent. r=1: perfect positive linear, r=0: no linear relation.
</details>

<details>
<summary><strong>13. What is Pearson vs Spearman correlation?</strong></summary>

Pearson: linear correlation between continuous variables, sensitive to outliers. Spearman: rank-based correlation, handles monotone non-linear relationships and ordinal data, robust to outliers. Kendall tau: another rank correlation.
</details>

<details>
<summary><strong>14. What is the difference between population and sample statistics?</strong></summary>

Population: all individuals in a group. Sample: subset drawn for study. Population parameters (mu, sigma, rho) are fixed unknowns. Sample statistics (x_bar, s, r) are estimates that vary across samples. Use n-1 in sample variance for unbiased estimation.
</details>

<details>
<summary><strong>15. What is the normal distribution and its properties?</strong></summary>

Bell-shaped, symmetric, fully characterized by mean and variance. 68-95-99.7 rule: 68% within 1 sigma, 95% within 2 sigma, 99.7% within 3 sigma. Sum of independent normals is normal. z = (x-mu)/sigma standardizes to N(0,1).
</details>

<details>
<summary><strong>16. What is a t-distribution and when to use it?</strong></summary>

t-distribution has heavier tails than normal, characterized by degrees of freedom. Use when estimating population mean with small sample (n < 30) and unknown sigma. As df -> inf, t -> normal. t-test uses t-distribution for hypothesis testing.
</details>

<details>
<summary><strong>17. What is chi-squared distribution?</strong></summary>

Chi-squared = sum of squared standard normals. Shape: right-skewed for low df, approaches normal as df increases. Used for: goodness-of-fit test, test of independence in contingency tables, and confidence intervals for variance.
</details>

<details>
<summary><strong>18. What is an F-distribution?</strong></summary>

Ratio of two scaled chi-squared distributions. Used in ANOVA (compare multiple group means), F-test for comparing model variances, regression significance testing. F = (explained variance / df1) / (unexplained variance / df2).
</details>

<details>
<summary><strong>19. What is ANOVA (Analysis of Variance)?</strong></summary>

Tests whether means of 3+ groups are equal. One-way ANOVA: one categorical factor. Two-way: two factors + interaction. H0: all group means equal. F-statistic = between-group variance / within-group variance. Post-hoc tests (Tukey, Bonferroni) for pairwise comparisons.

```python
from scipy import stats
import numpy as np

np.random.seed(42)
group1 = np.random.normal(10, 2, 30)
group2 = np.random.normal(12, 2, 30)
group3 = np.random.normal(11, 2, 30)

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
```
</details>

<details>
<summary><strong>20. What is a non-parametric test?</strong></summary>

Non-parametric tests make no distributional assumptions. Use when: small samples, non-normal data, ordinal data. Mann-Whitney U (vs t-test), Kruskal-Wallis (vs ANOVA), Wilcoxon signed-rank (vs paired t-test), Spearman (vs Pearson).
</details>

<details>
<summary><strong>21. What is the Mann-Whitney U test?</strong></summary>

Non-parametric alternative to independent samples t-test. Tests whether one distribution stochastically dominates another. Works with ordinal data and non-normal distributions. H0: no difference between distributions.
</details>

<details>
<summary><strong>22. What is multiple testing correction?</strong></summary>

Running many tests inflates Type I error (FWER). Solutions: Bonferroni (divide alpha by n tests — conservative), Holm-Bonferroni (stepwise, less conservative), Benjamini-Hochberg FDR control (controls false discovery rate, preferred for many tests).
</details>

<details>
<summary><strong>23. What is effect size?</strong></summary>

Quantifies practical significance independent of sample size. Cohen's d = (mean1 - mean2) / pooled_sd. d=0.2 small, 0.5 medium, 0.8 large. Eta-squared for ANOVA. Helps distinguish statistical significance from practical importance.

```python
import numpy as np
from scipy import stats

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

a = np.random.normal(10, 2, 100)
b = np.random.normal(11, 2, 100)
print(f"Cohen's d: {cohens_d(a, b):.3f}")
```
</details>

<details>
<summary><strong>24. What is a goodness-of-fit test?</strong></summary>

Chi-squared goodness-of-fit: tests whether observed frequencies match expected under a hypothesized distribution. Kolmogorov-Smirnov: compares empirical CDF to theoretical. Anderson-Darling: emphasizes tails.
</details>

<details>
<summary><strong>25. What is the chi-squared test of independence?</strong></summary>

Tests whether two categorical variables are independent. Observed vs expected frequencies under independence. df = (rows-1)(cols-1). H0: the variables are independent. Effect size: Cramer's V.
</details>

<details>
<summary><strong>26. What is a contingency table?</strong></summary>

Cross-tabulation showing joint frequency distribution of two categorical variables. Used with chi-squared test for independence. Marginal totals are row/column sums. Expected cell count = (row_total * col_total) / grand_total.
</details>

<details>
<summary><strong>27. What is bootstrapping?</strong></summary>

Resampling with replacement from observed data to estimate distribution of a statistic. No distributional assumptions. Bootstrap CI: compute statistic on B resamples, use percentiles as CI bounds. Computationally intensive but very general.

```python
import numpy as np

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, alpha=0.05):
    boot_stats = [statistic(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_bootstrap)]
    ci = np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])
    return ci

data = np.random.normal(5, 2, 50)
ci = bootstrap_ci(data, np.median)
print(f"Bootstrap 95% CI for median: [{ci[0]:.3f}, {ci[1]:.3f}]")
```
</details>

<details>
<summary><strong>28. What is the difference between frequentist and Bayesian statistics?</strong></summary>

Frequentist: parameters are fixed unknowns, probability = long-run frequency, uses p-values and CIs. Bayesian: parameters have distributions, probability = degree of belief, uses posterior distributions and credible intervals.
</details>

<details>
<summary><strong>29. What is a prior and posterior distribution in Bayesian stats?</strong></summary>

Prior: P(theta) — belief about parameter before seeing data. Likelihood: P(data|theta). Posterior: P(theta|data) proportional to likelihood * prior. Conjugate priors yield posterior in same family as prior (e.g., Beta-Binomial, Normal-Normal).
</details>

<details>
<summary><strong>30. What is Markov Chain Monte Carlo (MCMC)?</strong></summary>

Sampling method for approximating posterior distributions when analytical solution is intractable. Metropolis-Hastings: accept/reject proposals. NUTS (No-U-Turn Sampler): efficient gradient-based MCMC used in Stan, PyMC. Hamiltonian Monte Carlo.
</details>

<details>
<summary><strong>31. What is the Bernoulli distribution?</strong></summary>

Single binary trial with probability p of success. Mean=p, Var=p(1-p). Foundation for logistic regression. Bernoulli(p) is Binomial(1, p). Used for click/no-click, spam/not-spam, etc.
</details>

<details>
<summary><strong>32. What is the geometric distribution?</strong></summary>

Number of Bernoulli trials until first success. P(X=k) = (1-p)^(k-1) * p. Mean=1/p. Memoryless property: P(X>m+n | X>m) = P(X>n). Same memoryless property as exponential (discrete analog).
</details>

<details>
<summary><strong>33. What is the Poisson distribution and when to use it?</strong></summary>

Models count of events in fixed interval when: events are independent, constant rate lambda, two events can't occur simultaneously. Mean=Var=lambda. Examples: customer arrivals, radioactive decay, rare events.
</details>

<details>
<summary><strong>34. What is the exponential distribution?</strong></summary>

Time between Poisson events. Continuous, memoryless. PDF: f(x) = lambda * exp(-lambda * x). Mean=1/lambda. Used for: failure time, waiting time, inter-arrival time.
</details>

<details>
<summary><strong>35. What is the beta distribution?</strong></summary>

Continuous distribution on [0,1], parametrized by alpha and beta. Mean=alpha/(alpha+beta). Used for modeling probabilities/proportions. Conjugate prior to Binomial. Beta(1,1) = Uniform(0,1).
</details>

<details>
<summary><strong>36. What is the Dirichlet distribution?</strong></summary>

Multivariate generalization of Beta distribution. Distribution over probability vectors (sum=1). Conjugate prior to Categorical/Multinomial. Used in topic modeling (LDA). Dir(1,...,1) = uniform distribution over simplex.
</details>

<details>
<summary><strong>37. What is the gamma distribution?</strong></summary>

Generalizes exponential distribution — sum of k exponentials. Parametrized by shape k and rate lambda. Mean=k/lambda. Chi-squared is a special case: Chi^2(k) = Gamma(k/2, 1/2). Used for time until k-th event.
</details>

<details>
<summary><strong>38. What is Markov property?</strong></summary>

Future state depends only on current state, not past history: P(X_t+1 | X_0,...,X_t) = P(X_t+1 | X_t). Foundation for Markov chains, HMMs, and RL. Memoryless — simplifies many computations.
</details>

<details>
<summary><strong>39. What is conditional probability?</strong></summary>

P(A|B) = P(A and B) / P(B) — probability of A given B has occurred. Requires P(B) > 0. Chain rule: P(A and B) = P(A|B) * P(B). Law of total probability: P(A) = sum_i P(A|B_i) * P(B_i).
</details>

<details>
<summary><strong>40. What is independence vs conditional independence?</strong></summary>

Independent: P(A and B) = P(A)*P(B). Conditionally independent given C: P(A and B | C) = P(A|C)*P(B|C). Can be independent but not conditionally, or vice versa (Naive Bayes assumes conditional independence).
</details>

<details>
<summary><strong>41. What is a random variable?</strong></summary>

A function mapping outcomes from a sample space to real numbers. Discrete: countable values (Bernoulli, Poisson). Continuous: uncountable values (Normal, Exponential). Characterized by PMF/PDF and CDF.
</details>

<details>
<summary><strong>42. What is expected value (expectation)?</strong></summary>

E[X] = sum(x * P(X=x)) for discrete, integral(x * f(x)) for continuous. Linearity: E[aX+bY] = aE[X]+bE[Y]. E[XY] = E[X]*E[Y] only if X,Y independent. Jensen's inequality: for convex f, E[f(X)] >= f(E[X]).
</details>

<details>
<summary><strong>43. What is variance formula and properties?</strong></summary>

Var(X) = E[(X-mu)^2] = E[X^2] - (E[X])^2. Var(aX+b) = a^2 * Var(X). Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y). Independent: Var(X+Y) = Var(X)+Var(Y). Variance is always non-negative.
</details>

<details>
<summary><strong>44. What is the law of total expectation and variance?</strong></summary>

Total expectation: E[X] = E[E[X|Y]]. Total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y]) (= expected conditional variance + variance of conditional mean). Used in mixture models and hierarchical models.
</details>

<details>
<summary><strong>45. What is a moment generating function (MGF)?</strong></summary>

M(t) = E[e^(tX)]. kth derivative at t=0 gives kth moment. Uniquely determines distribution. Sum of independent RVs: MGF is product. Used to prove CLT and derive moments.
</details>

<details>
<summary><strong>46. What is skewness and kurtosis?</strong></summary>

Skewness = E[(X-mu)^3] / sigma^3. Positive: right tail, Negative: left tail. Zero for symmetric distributions. Kurtosis = E[(X-mu)^4] / sigma^4 - 3 (excess kurtosis). Normal kurtosis=0. Leptokurtic (heavy tails) > 0, platykurtic < 0.
</details>

<details>
<summary><strong>47. What is a qq-plot?</strong></summary>

Quantile-Quantile plot: plots sample quantiles against theoretical quantiles. If sample follows the theoretical distribution, points fall on a straight line. Used to assess normality (q-q plot vs normal). Deviations reveal skewness, heavy tails.
</details>

<details>
<summary><strong>48. What is a permutation test?</strong></summary>

Non-parametric test that computes p-value by permuting group labels and computing the test statistic. No distributional assumptions. P-value = proportion of permutations with more extreme statistic than observed. Exact test for any statistic.

```python
import numpy as np

def permutation_test(group1, group2, n_permutations=1000):
    observed_stat = group1.mean() - group2.mean()
    combined = np.concatenate([group1, group2])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_stat = combined[:len(group1)].mean() - combined[len(group1):].mean()
        if abs(perm_stat) >= abs(observed_stat):
            count += 1
    return count / n_permutations

a = np.random.normal(5, 1, 30)
b = np.random.normal(5.5, 1, 30)
p = permutation_test(a, b)
print(f"Permutation test p-value: {p:.3f}")
```
</details>

<details>
<summary><strong>49. What is OLS (Ordinary Least Squares) regression?</strong></summary>

OLS minimizes sum of squared residuals. Closed-form solution: beta = (X^T X)^-1 X^T y. Assumptions (BLUE): linearity, homoscedasticity, no autocorrelation, no perfect multicollinearity, zero mean errors. Gauss-Markov: OLS is BLUE.
</details>

<details>
<summary><strong>50. What are regression diagnostics?</strong></summary>

Residual plots to check OLS assumptions: residuals vs fitted (linearity, homoscedasticity), q-q plot (normality), leverage vs residuals (influential points), scale-location (homoscedasticity). Cook's distance measures influence of each point.
</details>

<details>
<summary><strong>51. What is multicollinearity and how to detect it?</strong></summary>

High correlation between predictors causes unstable coefficient estimates (large SE, wrong signs). Detect: Variance Inflation Factor (VIF) = 1/(1-R^2_j) where R^2_j is R^2 from regressing Xj on all other predictors. VIF > 5-10: problematic.
</details>

<details>
<summary><strong>52. What is heteroscedasticity?</strong></summary>

Non-constant variance of residuals violates OLS assumption. Leads to inefficient estimates and incorrect SEs. Detect: Breusch-Pagan test, White test, residual plot. Fix: weighted regression, log transform, robust SEs.
</details>

<details>
<summary><strong>53. What is autocorrelation in regression?</strong></summary>

Correlation between residuals at different time points (time series). Violates OLS independence assumption. Detect: Durbin-Watson statistic (2=no autocorrelation, less than 2 means positive, greater than 2 means negative), ACF/PACF plots. Fix: GLS, include lagged variables.
</details>

<details>
<summary><strong>54. What is instrumental variable (IV) estimation?</strong></summary>

When X and y have confounding, find an instrument Z that affects X but not y directly. IV estimator: beta_IV = Cov(Z,y)/Cov(Z,X). Two-stage least squares (2SLS) for multiple instruments. Addresses endogeneity/confounding.
</details>

<details>
<summary><strong>55. What is a time series and stationarity?</strong></summary>

Time series: sequence of values over time. Stationary: statistical properties (mean, variance, autocorrelation) are constant over time. Required for many time series models. Augmented Dickey-Fuller test for unit root (non-stationarity).
</details>

<details>
<summary><strong>56. What is autocorrelation function (ACF)?</strong></summary>

ACF(k) = Corr(X_t, X at lag t-k): correlation between time series and its k-lag version. PACF: partial autocorrelation controlling for intermediate lags. Used for ARMA model identification: AR(p) has cutoff PACF at lag p, MA(q) has cutoff ACF at lag q.
</details>

<details>
<summary><strong>57. What is ARIMA?</strong></summary>

Autoregressive Integrated Moving Average. AR(p): regression on own past values. MA(q): regression on past errors. I(d): differencing d times for stationarity. ARIMA(p,d,q). SARIMA adds seasonal components.
</details>

<details>
<summary><strong>58. What is cointegration?</strong></summary>

Two non-stationary time series are cointegrated if a linear combination is stationary. Implies a long-run equilibrium relationship. Engle-Granger test. Important for pairs trading, economic relationships.
</details>

<details>
<summary><strong>59. What is the Kolmogorov-Smirnov test?</strong></summary>

Non-parametric test comparing two distributions. One-sample: compare empirical CDF to theoretical. Two-sample: compare two empirical CDFs. Statistic = max difference between CDFs. Detects location, scale, and shape differences.
</details>

<details>
<summary><strong>60. What is kernel density estimation (KDE)?</strong></summary>

Non-parametric estimation of probability density function. Places a kernel (usually Gaussian) at each data point, sums them up. Bandwidth controls smoothing — too small: noisy, too large: over-smoothed. Alternative to histograms.
</details>

<details>
<summary><strong>61. What is Simpson's paradox?</strong></summary>

An association that appears in several groups reverses when the groups are combined. Caused by confounding variable. Classic example: UC Berkeley admissions — females accepted at higher rate in each department but lower overall (due to applying to competitive depts).
</details>

<details>
<summary><strong>62. What is survival analysis?</strong></summary>

Analysis of time-to-event data with censoring (some subjects haven't experienced event yet). Kaplan-Meier estimator: non-parametric survival function estimate. Cox proportional hazards: semi-parametric regression for hazard rate.
</details>

<details>
<summary><strong>63. What is propensity score matching?</strong></summary>

Estimate causal effect of treatment by matching treated and control units on probability of receiving treatment (propensity score). Balances covariates between groups. Used when randomization isn't possible.
</details>

<details>
<summary><strong>64. What is difference-in-differences (DiD)?</strong></summary>

Causal inference method: compare change over time in treatment group vs control group. Estimate = (treated_after - treated_before) - (control_after - control_before). Assumes parallel trends in absence of treatment.
</details>

<details>
<summary><strong>65. What is regression discontinuity (RD)?</strong></summary>

Exploits sharp threshold for treatment assignment. Units just below and above threshold are comparable. Effect = discontinuous jump in outcome at threshold. Valid causal estimate near the threshold.
</details>

<details>
<summary><strong>66. What is the expectation-maximization link to statistics?</strong></summary>

EM is a statistical algorithm for MLE with latent/missing variables. E-step: compute expected sufficient statistics given current params. M-step: update params using those statistics. Monotone increasing log-likelihood. Used in GMM, HMM, imputation.
</details>

<details>
<summary><strong>67. What is Laplace smoothing in statistics?</strong></summary>

Adds pseudo-counts to avoid zero probability in Bayesian estimation. For Binomial: (x + alpha) / (n + alpha * K) where K = number of categories. Equivalent to MAP estimate with Dirichlet prior. Common in text models to handle unseen words.
</details>

<details>
<summary><strong>68. What is cross-validation's statistical interpretation?</strong></summary>

k-fold CV estimates expected prediction error on new data (generalization error). Leave-one-out CV (LOOCV) has low bias but high variance. 5-fold or 10-fold is standard compromise. CV score is an approximation of the true generalization error.
</details>

<details>
<summary><strong>69. What is the bias-variance decomposition of MSE?</strong></summary>

Expected MSE = Bias^2 + Variance + Irreducible noise. Bias: error from wrong assumptions (underfitting). Variance: sensitivity to training data (overfitting). Irreducible: inherent noise in problem. Cannot be reduced. Total error = all three.
</details>

<details>
<summary><strong>70. What is entropy in information theory?</strong></summary>

H(X) = -sum(p(x) * log2(p(x))): measures average information content or uncertainty. Uniform distribution: maximum entropy. Deterministic: zero entropy. Used in decision trees (information gain) and ML loss functions.
</details>

<details>
<summary><strong>71. What is KL divergence?</strong></summary>

KL(P || Q) = sum(P(x) * log(P(x)/Q(x))): measures how much Q differs from P. Not symmetric: KL(P||Q) != KL(Q||P). Always >= 0, equals 0 iff P=Q. Cross-entropy H(P,Q) = H(P) + KL(P||Q). Used in VAE loss, knowledge distillation.
</details>

<details>
<summary><strong>72. What is mutual information?</strong></summary>

I(X;Y) = KL(P(X,Y) || P(X)*P(Y)): measures reduction in uncertainty about X given Y. I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X). Zero iff independent. Symmetric. Used for feature selection.
</details>

<details>
<summary><strong>73. What is the Gini coefficient in ML?</strong></summary>

Gini impurity = 1 - sum(p_i^2): used in decision trees to measure node impurity. For binary: 2*p*(1-p). Maximum at p=0.5 (most uncertain), minimum at 0 or 1 (pure). Alternative to entropy for splitting criterion.
</details>

<details>
<summary><strong>74. What is odds ratio?</strong></summary>

Odds = P(event) / P(no event). Odds Ratio = odds(group1) / odds(group2). Used in logistic regression: coefficient exp(beta) gives OR. OR=1: no effect, OR &gt; 1: higher odds in group1, OR &lt; 1: lower odds.
</details>

<details>
<summary><strong>75. What is a z-score?</strong></summary>

z = (x - mu) / sigma: standardized distance from mean in units of standard deviation. |z| > 2: outside 95% range. z-test: for large samples or known sigma. If z > 1.96, reject H0 at alpha=0.05 (two-tailed).
</details>

<details>
<summary><strong>76. What is a test statistic?</strong></summary>

A function of sample data used to determine whether to reject H0. t-statistic: (x_bar - mu0) / (s/sqrt(n)). z-statistic: with known sigma. F-statistic: ratio of variances. Chi-squared: sum of squared z-scores.
</details>

<details>
<summary><strong>77. What is the power of a test?</strong></summary>

Power = P(reject H0 | H0 false) = 1 - beta. Increases with: larger sample size, larger effect size, higher alpha, smaller variance, one-sided test. Power analysis determines sample size needed before data collection.
</details>

<details>
<summary><strong>78. What is a one-tailed vs two-tailed test?</strong></summary>

Two-tailed: tests if effect is in either direction (H1: mu != mu0). One-tailed: tests for effect in specific direction (H1: mu > mu0). One-tailed has more power but should be specified before data collection. Most scientific work uses two-tailed.
</details>

<details>
<summary><strong>79. What is the F-test in regression?</strong></summary>

Tests overall significance of regression: H0: all coefficients except intercept are zero. F = (R^2/k) / ((1-R^2)/(n-k-1)). Individual coefficient significance: t-test for each beta. F-test uses chi-squared distribution for numerator/denominator.
</details>

<details>
<summary><strong>80. What is adjusted R-squared?</strong></summary>

R^2_adj = 1 - (1-R^2)(n-1)/(n-k-1). Penalizes adding variables that don't improve fit. Unlike R^2, can decrease when adding irrelevant variables. Use for comparing models with different numbers of predictors.
</details>

<details>
<summary><strong>81. What is Akaike Information Criterion (AIC)?</strong></summary>

AIC = 2k - 2*log(L): penalizes complexity by number of parameters k. Lower AIC is better. Rewards goodness of fit, penalizes overfitting. AICc (corrected) for small samples. BIC uses log(n) penalty instead of 2 (more conservative).
</details>

<details>
<summary><strong>82. What is the Akaike weight?</strong></summary>

Relative likelihood of model i: exp(-0.5*delta_AIC_i) / sum(exp(-0.5*delta_AIC)). Gives probability each model is the best among candidates. Used for model averaging.
</details>

<details>
<summary><strong>83. What is conditional expectation and regression?</strong></summary>

E[Y|X=x]: expected value of Y given X=x. Regression estimates E[Y|X]. For linear regression: E[Y|X] = beta0 + beta1*X. Conditional expectation minimizes MSE — OLS finds the best linear predictor.
</details>

<details>
<summary><strong>84. What is the multivariate normal distribution?</strong></summary>

Generalization of normal to k dimensions. Characterized by mean vector mu and covariance matrix Sigma. Covariance captures pairwise relationships. Sum of independent MVN is MVN. Used in LDA, Gaussian processes, Kalman filter.
</details>

<details>
<summary><strong>85. What is principal component analysis from statistical perspective?</strong></summary>

PCA finds directions of maximum variance (principal components). PC1 maximizes variance, PC2 maximizes remaining variance perpendicular to PC1. Eigendecomposition of covariance matrix. PCs are uncorrelated. Dimensionality reduction by keeping top k components.
</details>

<details>
<summary><strong>86. What is factor analysis vs PCA?</strong></summary>

PCA: captures maximum variance with orthogonal components — descriptive. Factor Analysis: assumes observed variables are linear combinations of latent factors + noise — explanatory. FA models covariance structure; PCA models variance.
</details>

<details>
<summary><strong>87. What is independent component analysis (ICA)?</strong></summary>

Separates multivariate signal into additive independent components. Maximizes non-Gaussianity (since mixture of non-Gaussian is more Gaussian than components). Used for blind source separation (cocktail party problem, EEG).
</details>

<details>
<summary><strong>88. What is hypothesis testing framework?</strong></summary>

1. State H0 and H1. 2. Choose test statistic. 3. Set alpha (significance level). 4. Compute test statistic from data. 5. Compute p-value. 6. Reject H0 if p < alpha. 7. Interpret in context. Don't conflate statistical and practical significance.
</details>

<details>
<summary><strong>89. What are sufficient statistics?</strong></summary>

A statistic T(X) is sufficient for parameter theta if the conditional distribution of data given T(X) doesn't depend on theta. Contains all information about theta. Sample mean is sufficient for Gaussian mean (Gaussian with known variance). Exponential family has natural sufficient statistics.
</details>

<details>
<summary><strong>90. What is the Cramer-Rao lower bound?</strong></summary>

Lower bound on variance of any unbiased estimator: Var(theta_hat) >= 1/I(theta) where I(theta) is Fisher information. Efficient estimator: achieves the bound. MLE is asymptotically efficient.
</details>

<details>
<summary><strong>91. What is the delta method?</strong></summary>

Approximates variance of a function of an estimator: Var(g(X_bar)) ~= (g'(mu))^2 * sigma^2/n. Used to compute SEs for transformed parameters (e.g., for odds ratio from logistic regression coefficients).
</details>

<details>
<summary><strong>92. What is resampling and cross-validation from statistical perspective?</strong></summary>

Resampling methods estimate sampling distribution without parametric assumptions. CV estimates generalization error. Bootstrap estimates any statistic's sampling distribution. Leave-one-out is related to AIC for parametric models.
</details>

<details>
<summary><strong>93. What is Gaussian process regression?</strong></summary>

Non-parametric Bayesian regression: places GP prior over functions, updates with data. Posterior is also GP with exact uncertainty quantification. Prediction = mean function + uncertainty. Kernel determines function properties. O(n^3) training.
</details>

<details>
<summary><strong>94. What is the method of moments?</strong></summary>

Estimate parameters by matching population moments to sample moments. Sample mean = theoretical mean, solve for parameters. Simpler than MLE but less efficient. Useful when MLE is intractable.
</details>

<details>
<summary><strong>95. What is importance sampling?</strong></summary>

Estimate E_p[f(X)] using samples from a different distribution q: E_p[f(X)] = E_q[f(X) * p(X)/q(X)]. Importance weights w = p/q. Useful when sampling from p is difficult. Used in MCMC, reinforcement learning (off-policy evaluation).
</details>

<details>
<summary><strong>96. What is Brier score?</strong></summary>

Measures accuracy of probabilistic predictions: BS = mean((p_i - o_i)^2) where p is predicted probability, o is 0/1 outcome. Range [0,1]: 0=perfect, 1=worst. Like MSE for probabilities. Proper scoring rule — incentivizes honest probabilities.
</details>

<details>
<summary><strong>97. What is log-loss (logarithmic loss)?</strong></summary>

Log-loss = -mean(y*log(p) + (1-y)*log(1-p)). Heavily penalizes confident wrong predictions. Proper scoring rule. Equivalent to cross-entropy and negative log-likelihood of Bernoulli model. Range [0, inf): 0=perfect.
</details>

<details>
<summary><strong>98. What is the difference between point estimate and interval estimate?</strong></summary>

Point estimate: single value (MLE, mean, median). Interval estimate: range of plausible values (CI, credible interval). Intervals convey uncertainty. Bayesian credible interval = posterior probability that parameter is in range. Frequentist CI = coverage probability.
</details>

<details>
<summary><strong>99. What is the Jackknife estimator?</strong></summary>

Leave-one-out resampling: remove one observation at a time, compute statistic on remaining n-1. Use n resamples to estimate bias and variance of the estimator. Simpler than bootstrap, less general. Good for bias correction.
</details>

<details>
<summary><strong>100. What is stochastic dominance?</strong></summary>

Distribution F stochastically dominates G if F(x) &lt;= G(x) for all x (first-order: F gives higher outcomes with probability 1). Second-order: integral of CDF &lt;= integral of CDF for G (F preferred by risk-averse decision makers). Used in decision theory, comparing models.
</details>

---

## Key Formulas Quick Reference

| Concept | Formula |
|---------|---------|
| Mean | E[X] = sum(x * P(x)) |
| Variance | Var(X) = E[X^2] - (E[X])^2 |
| Standard Error | SE = sigma / sqrt(n) |
| z-score | z = (x - mu) / sigma |
| t-statistic | t = (x_bar - mu0) / (s/sqrt(n)) |
| Pearson r | r = Cov(X,Y) / (sigma_X * sigma_Y) |
| Bayes | P(A|B) = P(B|A) * P(A) / P(B) |
| Cohen's d | d = (mu1 - mu2) / sigma_pooled |
| AIC | 2k - 2 log(L) |
| KL Divergence | sum(P * log(P/Q)) |
| Entropy | -sum(p * log2(p)) |

## Summary in simple points

- Define the population, sample, random variable and target estimand. Probability models assumptions; statistics learns from observations.
- Expectations, variances, covariance and conditional laws describe different properties. Independence differs from conditional independence.
- Bayes' rule combines priors with likelihoods; posterior probabilities depend on base rates and model assumptions.
- Bernoulli, geometric, Poisson, exponential, beta, Dirichlet, gamma, normal and multivariate normal distributions fit different support and generative stories.
- The law of large numbers concerns averages; the central limit theorem concerns a suitably scaled sampling distribution under conditions.
- Sample moments, skewness, kurtosis, quantile plots and density estimates describe shape; diagnostic plots need interpretation rather than automatic verdicts.
- Hypothesis tests use a null model, test statistic, significance level and decision rule. A p-value is not the probability the null is true.
- Type I/II errors, power, effect size and one/two-sided choices should be planned before observing results. Multiple testing needs a suitable correction strategy.
- Confidence intervals describe coverage of a procedure; Bayesian credible intervals condition on the model and prior. Point and interval estimates answer different questions.
- Use t, chi-squared, F/ANOVA, rank-based, goodness-of-fit and independence tests only with their relevant assumptions and sampling design.
- Bootstrap, permutation, jackknife and cross-validation resample for different purposes. Resample independent units and preserve the appropriate null or split structure.
- OLS diagnostics include multicollinearity, heteroscedasticity and autocorrelation. Adjusted R-squared and information criteria penalise complexity in different ways.
- Correlation, propensity matching, instrumental variables, difference-in-differences and discontinuity designs identify different quantities under explicit assumptions.
- Time-series stationarity, ACF, ARIMA, cointegration and survival/censoring require specialised data assumptions; random row splits can be misleading.
- MLE, MAP, moments, EM and MCMC are different estimation/computation methods. Importance sampling depends on support overlap and stable weights.
- Entropy, KL divergence, mutual information and impurity measures quantify different distribution properties. Clarify whether Gini means tree impurity or inequality.
- PCA, factor analysis and ICA seek different latent structure. Gaussian processes express covariance-based function priors.
- Sufficiency, Cramer–Rao bounds, the delta method and bias–variance decomposition depend on mathematical regularity conditions.
- Brier score and log-loss assess probability predictions. Odds ratios, z-scores, stochastic dominance and conditional expectation need their correct interpretation.
