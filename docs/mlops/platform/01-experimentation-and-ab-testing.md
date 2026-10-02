---
id: plat-ab-testing
title: "Experimentation and A/B Testing"
sidebar_label: "A/B testing"
sidebar_position: 1
slug: /mlops/platform/experimentation-and-ab-testing
description: "Size an A/B test with power and a minimum detectable effect, stop it safely with sequential methods, cut its variance with CUPED, and catch the traps that make a clean-looking result wrong."
tags: [ab-testing, experimentation, statistical-power, sequential-testing, cuped, guardrail-metrics, interference]
---

import Infographic from '@site/src/components/Infographic';
import AbTestPowerLab from '@site/src/components/viz/AbTestPowerLab';

**In one line.** An A/B test is a measurement you design before you run it: decide the smallest effect worth finding, buy enough users to see it, look at the result only in ways that keep the error rate honest, and check that the experiment itself was not broken.

:::note Not from a lecture

This chapter is written for this site from the sources listed under Further reading; it is not built from a course lecture.

:::

## The idea in plain words

You trained a new ranking model offline and it scores higher on the held-out set. Will it earn more revenue? Offline metrics predict that only loosely, so teams show the new model to a random half of users and compare. That is an **online controlled experiment**. Random assignment is the whole trick: because chance alone decides who sees what, any systematic difference in the outcome can be blamed on the model.

Randomness cuts both ways. A group of users differs from another by luck, so even identical models give different conversion rates. The question is never "is B higher than A?" but "is B higher by more than luck would produce?". Four numbers govern the answer.

| Quantity | Meaning | Typical choice |
| --- | --- | --- |
| **alpha** | Chance of a false alarm when nothing changed | 0.05 |
| **power** | Chance of catching a real effect of the target size | 0.80 |
| **MDE** | The smallest effect you care to detect (minimum detectable effect) | set by the business case, not by the data |
| **n** | Users per arm | the output |

Fix three of them and the fourth follows. The expensive one is the MDE: the sample size scales with one over the MDE squared, so asking to see an effect half as large costs four times the users.

Most production mistakes are not arithmetic. They are procedural. Someone checks the dashboard every morning and stops when the number turns green. The traffic split quietly drifts from 50/50 and nobody notices. A feature that helps buyers in a shared marketplace is measured as if buyers did not compete. The sections below name each failure and the tool that answers it.

<Infographic src="/img/plat/plat-ab-testing-power.svg" alt="Sample size for a 10 percent baseline as the detectable lift shrinks, and the variance saved by CUPED at four correlations" caption="Power in numbers. The sample sizes and the CUPED variance shares are printed by blocks 1 and 3 below." />

<Infographic src="/img/plat/plat-ab-testing-peeking.svg" alt="False positive rate and power of five stopping rules over twenty looks, plus the sample ratio mismatch check and the marketplace interference result" caption="Four ways an experiment goes wrong. Every figure is printed by blocks 2 and 4 below." />

## How it works

### Power and the minimum detectable effect

For a conversion rate, the two arms have rates $p_0$ and $p_1 = p_0 + \delta$ where $\delta$ is the MDE. The standard two-sided z-test needs

$$
n = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{\text{power}}\sqrt{p_0(1-p_0)+p_1(1-p_1)}\right)^2}{\delta^2}
$$

users per arm, where $\bar p$ is the average of the two rates. Read it as a recipe. The first term is the evidence you need to reject "no change" with false-alarm rate alpha. The second is the extra you need so that, if the effect is real, the test catches it with the stated power. Noise (the square roots) sits in the numerator and signal ($\delta$) is squared underneath, which is why small effects are so expensive.

Block 1 applies the formula to a 10% baseline and a one-point lift, then runs 4,000 simulated experiments to confirm that the formula's promise holds: power 0.80 and a false positive rate near 0.05.

### Peeking, and what to do instead

The fixed-horizon test above is valid only if you look once, at the planned sample size. Looking twenty times and stopping at the first p-value under 0.05 is a different procedure with a much higher false alarm rate. Johari, Pekelis and Walsh describe this directly: standard inference "is wholly unreliable" when users choose the sample size by continuously monitoring, and they propose always-valid p-values and confidence intervals that stay correct however you decide to stop.

There are three honest ways to look early.

| Method | Idea | Cost |
| --- | --- | --- |
| **Group sequential**, for example Bonferroni or an O'Brien-Fleming shaped boundary | A fixed number of planned looks, each with a stricter threshold early on | You must plan the looks in advance |
| **Always-valid (mixture sequential probability ratio test)** | A likelihood ratio that you may check after every user and stop when it crosses $1/\alpha$ | Less power than a fixed test at the same maximum n |
| **Don't look** | Wait for the planned n | Slow, but simplest and most powerful per user |

Block 2 measures all of them on the same simulated streams. The naive rule that tests at every look is wrong about a quarter of the time. The corrected rules hold the error near or below 5%, and the O'Brien-Fleming shape keeps nearly all the power of a single final look while still allowing an early stop. The always-valid rule is more conservative here because its prior width `tau` was not tuned to the effect size; that is the price of being valid after every single user.

### CUPED: use what you knew before the experiment

Users differ enormously in how much they spend or search, and that between-user spread is most of the noise. If you recorded the same metric before the experiment started, you can subtract the predictable part. The method is called CUPED, Controlled-experiment Using Pre-Experiment Data, from Deng, Xu, Kohavi and Walker at Microsoft. For a metric $Y$ and a pre-experiment covariate $X$, use

$$
Y_{\text{cuped}} = Y - \theta\,(X - \bar X), \qquad \theta = \frac{\operatorname{cov}(Y, X)}{\operatorname{var}(X)}
$$

This is unbiased because $X$ was measured before assignment and so cannot depend on the treatment. With the best $\theta$ the variance falls to $\operatorname{var}(Y)(1-\rho^2)$, where $\rho$ is the correlation between $Y$ and $X$. A correlation of 0.7 removes about half the variance, which is the same as halving the required traffic. The paper reports that the pre-experiment version of the same metric usually works best and that, on Bing experiments, variance fell by about 50%.

### Guardrails, and the experiment that was broken before it started

A single success metric invites damage elsewhere: a change that lifts clicks can slow the page or raise refunds. **Guardrail metrics** are the ones you promise not to hurt (latency, error rate, unsubscribe rate, complaint rate), tested with a one-sided bound for harm. Decide them before launch, along with a ship rule such as "ship if the success metric is up and no guardrail is significantly down".

The most useful trust check is the **sample ratio mismatch** (SRM) test. If you designed a 50/50 split and the arms hold 50,900 and 49,100 users, a chi-square test on the counts says the gap is far beyond luck. Fabijan and colleagues, in a KDD 2019 paper on diagnosing SRM, call it a symptom of a data quality problem, as a fever is a symptom of illness. An experiment with an SRM has a broken pipeline, a bot filter that treats the arms differently, or a bug in assignment, and its effect estimate cannot be trusted whatever it says. Block 4 shows the test and the thresholds.

### Interference: when users are not independent

The analysis assumes one user's outcome does not depend on who else is in the treatment. Marketplaces, social networks and shared capacity break that. If treated riders take the drivers, control riders get fewer, and the test measures a gap that will not exist when everyone is treated. Block 4 simulates exactly this: shared capacity makes the user-level A/B test report more than twice the true launch effect. The usual remedy is to randomise whole clusters, such as a city, a commuting zone or a friend group, so that most interference stays inside one arm.

## A real system that works this way

**Bing's experimentation platform at Microsoft** is the setting of the CUPED paper. The authors applied the method to real experiments and report that variance fell by about 50%, equivalent to doubling traffic or halving the time to reach the same sensitivity. They also report that using the same metric from the pre-experiment period gave the largest variance reduction, and that a longer pre-period helped.

**Facebook's network-effects experiments** are the setting for the interference problem. A published write-up from Facebook Research describes how a user's response can depend on other users' treatments, and gives a food delivery example in which faster ordering for treated users reduces driver supply for control users and overstates the effect. The write-up runs a Jobs on Facebook test with geographic commuting zones as clusters. Randomising by user suggested a 71.8% increase in applications to jobs with no prior applications, while randomising by cluster put the effect at 49.7%. The same write-up credits regression adjustment with pre-treatment metrics and trigger logging (recording who actually saw the change) with improving precision.

## Code you can run

Four blocks, executed with Python 3.14, NumPy 2.5.3 and SciPy 1.18.1. Each uses a seeded generator, so the printed numbers are the ones quoted in the text and drawn on the boards.

### 1. Sample size and a check by simulation

The function is the formula above. The simulation draws 4,000 pairs of arms at the computed n and counts how often the z-test rejects.

```python
import numpy as np
from scipy.stats import norm

def n_per_arm(p0, mde, alpha=0.05, power=0.80):
    p1 = p0 + mde
    pbar = (p0 + p1) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    top = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return top**2 / mde**2

p0, mde = 0.10, 0.01
n = int(np.ceil(n_per_arm(p0, mde)))
print(f"baseline {p0:.0%}, detect +{mde:.0%} absolute, alpha 0.05, power 0.80")
print(f"users needed per arm: {n:,}   total: {2 * n:,}")

print("\nhalve the effect you want to see:")
for m in (0.02, 0.01, 0.005):
    print(f"  MDE {m:.3f} -> {int(np.ceil(n_per_arm(p0, m))):>9,} per arm")

rng = np.random.default_rng(7)
sims = 4000

def z_test(a, b, n):
    pa, pb = a / n, b / n
    pool = (a + b) / (2 * n)
    se = np.sqrt(2 * pool * (1 - pool) / n)
    return (pb - pa) / se

za = norm.ppf(0.975)
a = rng.binomial(n, p0, sims)
b_real = rng.binomial(n, p0 + mde, sims)
b_null = rng.binomial(n, p0, sims)
power = np.mean(np.abs(z_test(a, b_real, n)) > za)
false_pos = np.mean(np.abs(z_test(a, b_null, n)) > za)
print(f"\nsimulated over {sims} experiments at n={n:,} per arm")
print(f"  power when the true lift is +{mde:.0%}: {power:.3f}")
print(f"  false positive rate when there is no lift: {false_pos:.3f}")

short = n // 2
a = rng.binomial(short, p0, sims)
b_real = rng.binomial(short, p0 + mde, sims)
print(f"  power with half the traffic ({short:,} per arm): {np.mean(np.abs(z_test(a, b_real, short)) > za):.3f}")
```

The two checks line up with the formula: 14,751 users per arm gives power 0.800 and a false positive rate of 0.051, within simulation noise of the intended 0.05. Halving the traffic drops power to about 0.495, which is close to a coin flip.

The lab below is the same formula with controls. Its defaults reproduce block 1: 14,751 per arm, 29,502 in total, six days at 5,000 users a day, with 3,841 and 57,763 at the larger and smaller effects.

<AbTestPowerLab />

### 2. Peeking and the stopping rules

Each simulated experiment streams 2,000 users per arm and the true effect is 0.0886 standard deviations in the power runs, chosen so that a single look at the end has about 80% power. The boundary constant for the O'Brien-Fleming shape is found by bisection on null simulations.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(11)
sims, n_max, looks = 3000, 2000, 20
look_n = np.arange(1, looks + 1) * (n_max // looks)
alpha = 0.05
tau = 0.1

def paths(effect):
    x = rng.normal(0, 1, (sims, n_max)).cumsum(axis=1)
    y = rng.normal(effect, 1, (sims, n_max)).cumsum(axis=1)
    n = np.arange(1, n_max + 1)
    diff = (y - x) / n
    return diff, n

def z_at(diff, n):
    return diff / np.sqrt(2.0 / n)

def msprt_log_lr(diff, n):
    v = 2.0 / n
    return 0.5 * np.log(v / (v + tau**2)) + tau**2 * diff**2 / (2 * v * (v + tau**2))

def evaluate(effect, c_obf=None):
    diff, n = paths(effect)
    z = z_at(diff, n)
    zl = z[:, look_n - 1]
    t = look_n / n_max
    out = {}
    out["fixed horizon, one look at the end"] = (np.abs(zl[:, -1]) > norm.ppf(1 - alpha / 2), np.full(sims, n_max))
    naive = np.abs(zl) > norm.ppf(1 - alpha / 2)
    out["naive, test at every look"] = (naive.any(axis=1), np.where(naive.any(axis=1), look_n[naive.argmax(axis=1)], n_max))
    bonf = np.abs(zl) > norm.ppf(1 - alpha / (2 * looks))
    out["Bonferroni, alpha/20 per look"] = (bonf.any(axis=1), np.where(bonf.any(axis=1), look_n[bonf.argmax(axis=1)], n_max))
    obf = np.abs(zl) > c_obf / np.sqrt(t)
    out["O'Brien-Fleming shape, 20 looks"] = (obf.any(axis=1), np.where(obf.any(axis=1), look_n[obf.argmax(axis=1)], n_max))
    ms = msprt_log_lr(diff, n) >= np.log(1 / alpha)
    out["always-valid mSPRT, every user"] = (ms.any(axis=1), np.where(ms.any(axis=1), n[ms.argmax(axis=1)], n_max))
    return out

def calibrate():
    diff, n = paths(0.0)
    zl = z_at(diff, n)[:, look_n - 1]
    t = look_n / n_max
    lo, hi = 1.5, 4.5
    for _ in range(30):
        mid = (lo + hi) / 2
        rate = (np.abs(zl) > mid / np.sqrt(t)).any(axis=1).mean()
        lo, hi = (mid, hi) if rate > alpha else (lo, mid)
    return (lo + hi) / 2

c = calibrate()
print(f"calibrated O'Brien-Fleming constant c = {c:.3f} (stop when |z| > c / sqrt(t))\n")
null = evaluate(0.0, c)
real = evaluate(0.0886, c)
print(f"{'method':<38}{'false positives':>16}{'power':>8}{'mean n if stopped':>20}")
for name in null:
    fp = null[name][0].mean()
    pw = real[name][0].mean()
    stop = real[name][1][real[name][0]].mean()
    print(f"{name:<38}{fp:>16.3f}{pw:>8.3f}{stop:>20.0f}")
```

The naive column is the lesson: testing at every one of 20 looks gives a false positive rate of 0.255 against the nominal 0.05, and, when the effect is real, it stops at a mean of 733 users per arm: fast, but only because it also crowns a quarter of the do-nothing experiments. Bonferroni is valid but wasteful. The O'Brien-Fleming shape pays almost nothing in power (0.784 against 0.799) and can stop early at a mean of 1,360 users. The mixture test is the only one valid after every single user. Its false positive rate is 0.020, which shows it is conservative at this prior width.

### 3. CUPED on a pre/post metric

Pre-experiment and experiment-period values share a correlation of 0.7. The block compares the plain difference of means to the CUPED-adjusted one across 2,000 experiments.

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(3)
n, rho, lift = 5000, 0.7, 0.10
sims = 2000

def one_experiment():
    pre = rng.normal(10, 3, 2 * n)
    noise = rng.normal(0, 3, 2 * n)
    post = 10 + rho * (pre - 10) + np.sqrt(1 - rho**2) * noise
    treat = np.r_[np.zeros(n), np.ones(n)].astype(bool)
    post = post + lift * treat
    return pre, post, treat

def diff_and_se(y, treat):
    a, b = y[~treat], y[treat]
    return b.mean() - a.mean(), np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)

def cuped(pre, post, treat):
    theta = np.cov(post, pre)[0, 1] / pre.var(ddof=1)
    return post - theta * (pre - pre.mean()), theta

raw_d, raw_se, cup_d, cup_se, thetas = [], [], [], [], []
for _ in range(sims):
    pre, post, treat = one_experiment()
    d, s = diff_and_se(post, treat)
    adj, theta = cuped(pre, post, treat)
    d2, s2 = diff_and_se(adj, treat)
    raw_d.append(d); raw_se.append(s); cup_d.append(d2); cup_se.append(s2); thetas.append(theta)

raw_d, raw_se, cup_d, cup_se = map(np.array, (raw_d, raw_se, cup_d, cup_se))
print(f"true lift {lift}, pre/post correlation {rho}, {n:,} users per arm")
print(f"theta (cov(post,pre)/var(pre)) averages {np.mean(thetas):.3f}, theory {rho:.3f}")
print(f"mean estimate   raw {raw_d.mean():.4f}   CUPED {cup_d.mean():.4f}  (both unbiased)")
print(f"std of estimate raw {raw_d.std():.4f}   CUPED {cup_d.std():.4f}")
print(f"variance ratio CUPED / raw: {cup_d.var() / raw_d.var():.3f}   theory 1 - rho^2 = {1 - rho**2:.3f}")
za = norm.ppf(0.975)
print(f"power at +{lift}: raw {np.mean(np.abs(raw_d / raw_se) > za):.3f}   CUPED {np.mean(np.abs(cup_d / cup_se) > za):.3f}")
print("\nvariance kept for other correlations (1 - rho^2):")
for r in (0.3, 0.5, 0.7, 0.9):
    print(f"  rho {r}: {1 - r**2:.2f} of the variance, so {1 - (1 - r**2):.0%} less traffic needed")
```

Both estimators centre on the true lift of 0.10, so CUPED has not biased anything. Its spread is smaller by exactly the predicted factor, a variance ratio of 0.516 against $1-\rho^2 = 0.510$. At 5,000 users per arm that moves power from 0.383 to 0.643 without a single extra user.

### 4. Sample ratio mismatch and interference

```python
import numpy as np
from scipy.stats import chisquare

def srm_p(a, b, expected_share=0.5):
    total = a + b
    return chisquare([a, b], [total * expected_share, total * (1 - expected_share)]).pvalue

print("sample ratio mismatch, 50/50 split intended")
for a, b in [(50_120, 49_880), (50_900, 49_100), (10_000, 9_650)]:
    p = srm_p(a, b)
    flag = "SRM: do not trust the result" if p < 0.001 else ("borderline: investigate" if p < 0.05 else "ok")
    print(f"  control {a:>6,} treatment {b:>6,}  ratio {b / a:.4f}  p = {p:.2e}  {flag}")

rng = np.random.default_rng(5)
markets, users, capacity_per_market, base, boost = 400, 200, 50, 0.20, 1.6

def bookings(share_treated):
    treated = rng.random((markets, users)) < share_treated
    demand = np.where(treated, base * boost, base)
    wants = rng.random((markets, users)) < demand
    out = np.zeros((markets, users), dtype=bool)
    for m in range(markets):
        idx = np.flatnonzero(wants[m])
        rng.shuffle(idx)
        out[m, idx[:capacity_per_market]] = True
    return treated, out

treated, got = bookings(0.5)
naive = got[treated].mean() - got[~treated].mean()
_, all_control = bookings(0.0)
_, all_treated = bookings(1.0)
truth = all_treated.mean() - all_control.mean()
print("\ninterference: treated users take slots from control users in the same market")
print(f"  naive user-level A/B estimate of the booking lift: {naive:+.4f}")
print(f"  lift if everyone were treated vs nobody:           {truth:+.4f}")
print(f"  the A/B test overstates the launch effect by {naive / truth:.1f}x")
```

The SRM test separates a harmless 0.5% wobble (p = 0.448) from a 1.8-point imbalance (p about 1e-8). The middle band matters too: with p below 0.05 but above 0.001 you do not discard the experiment but you do find out why before believing it. In the marketplace simulation each market has 50 bookable slots and 200 users, and treatment raises each user's booking appetite by 60%. Shared capacity means treated users are served at the control users' expense, so the A/B estimate of +0.1105 is 2.2 times the true +0.0500.

## Designing with it

- **Write the plan before the launch.** Primary metric, MDE, alpha, power, guardrails, the ship rule and the stopping rule go in the experiment document first. Changing them after seeing data is how false positives get published.
- **Choose the MDE from the business, not from the traffic.** If the cheapest detectable effect is larger than anything the change could plausibly deliver, the experiment cannot succeed; spend the effort elsewhere or accept a longer run.
- **Pick one stopping discipline and keep it.** Either run to the planned n and look once, or use a sequential method built for repeated looks. Never run a fixed-horizon test with a daily glance at the p-value.
- **Run SRM first, always.** It is cheap, automatic and catches the failures that otherwise become convincing but wrong results.
- **Use CUPED when you have history.** It is nearly free where users return and the metric is stable. It does nothing for first-time users, who have no pre-period.
- **Randomise at the level interference lives.** If treated and control units compete or talk, assign clusters, accept the loss of power, and report that you did.
- **Run in whole weeks.** User behaviour has a weekly cycle; stopping mid-cycle biases the average.

## Where this stands in 2026

:::info Industry view

- **The core methods are old and settled.** CUPED dates from 2013 and the always-valid inference paper from 2015, with a revision in 2019. They remain the standard references for variance reduction and for continuous monitoring.
- **Sequential monitoring is built into commercial platforms.** The always-valid paper reports that the approach was implemented at scale in commercial A/B testing platforms and analysed hundreds of thousands of experiments.
- **Interference is the open frontier.** The Facebook write-up above shows a large gap between user-level and cluster-level estimates in a real marketplace, which is why cluster randomisation and network-aware designs remain active research.
- **Model experiments inherit all of this.** Comparing two LLM prompts or two retrievers online raises the same questions of MDE, guardrails and SRM; see [online evaluation](/docs/llm-evals/online-evaluation) for the LLM side and [serving and release strategies](/docs/theory/seml/serving-and-release-strategies) for how traffic is split safely.
- **Unverified here.** Which specific vendors offer which tests, and their pricing, changes quickly and was not checked for this chapter.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Your baseline conversion is 10% and the product manager wants to detect a 0.5 point lift instead of a 1 point lift. What happens to the traffic you need, and what do you tell them?</summary>

Sample size scales with one over the MDE squared, so halving the effect roughly quadruples the users: 14,751 per arm becomes 57,763 (block 1). Tell them the cost in days at the real traffic, ask whether a 0.5 point lift is worth that wait, and if not, either accept a longer run, raise the MDE, or use CUPED to buy back part of the sample.

</details>

<details>
<summary><strong>Q2.</strong> An analyst checks the p-value daily for twenty days and ships on the first day it is below 0.05. The change did nothing. How often would this procedure ship anyway, and how do you fix it?</summary>

About a quarter of the time in block 2 (0.255 against the nominal 0.05). Fix it by planning the looks: use a group sequential boundary such as the O'Brien-Fleming shape, or an always-valid sequential test, or simply wait for the planned sample size.

</details>

<details>
<summary><strong>Q3.</strong> A CUPED analysis uses the same metric from the four weeks before the test. The correlation is 0.5. How much traffic does that save, and who does it not help?</summary>

The variance is multiplied by $1-\rho^2 = 0.75$, so about 25% fewer users are needed for the same power. It does not help users with no pre-experiment history, such as new accounts, whose covariate is missing or uninformative.

</details>

<details>
<summary><strong>Q4.</strong> Your 50/50 experiment shows 50,900 users in control and 49,100 in treatment. The treatment effect is positive and significant. Do you ship?</summary>

No. The chi-square test on the counts gives p of about 1e-8, which is an SRM. Something differs between the arms before the outcome is measured: a filter, a logging bug or an assignment bug. Find and fix the cause and rerun; the effect estimate from a mismatched experiment is not trustworthy.

</details>

<details>
<summary><strong>Q5.</strong> A ride-hailing company tests a feature that makes treated riders book faster, randomising by rider. Why might the measured lift overstate the launch effect, and what design would you use?</summary>

Drivers are a shared, limited supply. Treated riders take drivers that control riders would have had, so the gap between arms is larger than the gain when everyone is treated (block 4 shows 2.2 times). Randomise by cluster, for example city or zone, so competing riders are mostly in the same arm, and accept that fewer clusters means less power.

</details>

<details>
<summary><strong>Q6.</strong> Name a guardrail metric for a recommendation change that raises click-through, and say how you would test it differently from the success metric.</summary>

Page latency or the unsubscribe rate. The success metric is tested to show a gain; a guardrail is tested to show no harm beyond an agreed margin, with a one-sided bound and a pre-registered tolerance, because the aim is to rule out damage, not to find a benefit.

</details>

## Further reading

- [Deng, Xu, Kohavi and Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM 2013)](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf): the CUPED paper; the formulas, the Bing results and the advice on covariates.
- [Johari, Pekelis and Walsh, "Always Valid Inference: Bringing Sequential Analysis to A/B Testing"](https://arxiv.org/abs/1512.04922): always-valid p-values and why peeking breaks the standard test.
- [Fabijan et al., "Diagnosing Sample Ratio Mismatch in Online Controlled Experiments" (KDD 2019)](https://www.lukasvermeer.nl/publications/papers/2019/07/25/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments.html): the SRM taxonomy and detection advice.
- [Facebook Research, "Testing product changes with network effects" (2021)](https://research.facebook.com/blog/2021/8/testing-product-changes-with-network-effects/): interference, cluster randomisation and the Jobs commuting-zone example.
- [Kohavi, Tang and Xu, "Trustworthy Online Controlled Experiments" (Cambridge University Press, 2020)](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59): the book-length treatment of metrics, guardrails and trust checks, by authors from Microsoft, Google and LinkedIn.

## Check yourself

- I can compute the users per arm for a given baseline, MDE, alpha and power, and explain why halving the MDE quadruples it.
- I can explain why checking a fixed-horizon test every day inflates false positives, and name two ways to look early safely.
- I can write the CUPED adjustment, state why it is unbiased, and estimate the traffic it saves from a correlation.
- I can run a sample ratio mismatch test and say what I do at p below 0.001 and at p below 0.05.
- I can choose guardrail metrics and a ship rule before launching an experiment.
- I can explain why interference makes a user-level A/B test overstate a marketplace launch, and what to randomise instead.
