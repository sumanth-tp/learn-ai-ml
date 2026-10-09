---
id: causal-quasi-experiments
title: "Quasi-Experiments: Difference-in-Differences, Instrumental Variables and Regression Discontinuity"
sidebar_label: "3 · Quasi-experiments"
sidebar_position: 3
slug: /theory/causal/quasi-experiments
description: "Three designs that find a fair comparison in the world instead of in a randomised trial: difference-in-differences, instrumental variables and regression discontinuity, each run on simulated data with a known effect, with the assumption that breaks it shown in numbers."
tags: [causal-inference, difference-in-differences, instrumental-variables, regression-discontinuity, parallel-trends, weak-instruments, 2sls]
---

import Infographic from '@site/src/components/Infographic';
import DidLab from '@site/src/components/viz/DidLab';
import WaldIvLab from '@site/src/components/viz/WaldIvLab';

**In one line.** When you cannot randomise and cannot measure every confounder, look for a rule, a rollout or an accident that behaves like a coin flip for part of the population, and say in advance which assumption would break it.

:::tip Before you start
- **You should already know** what a confounder is and why a naive comparison is biased ([potential outcomes and confounding](/docs/theory/causal/potential-outcomes-and-confounding)), and how adjustment works when confounders are measured ([experiments and adjustment](/docs/theory/causal/experiments-and-adjustment)).
- **Reading time:** about 50 minutes, plus about ten seconds to run the code.
- **After this chapter you can** compute a difference-in-differences, a Wald instrumental-variable estimate and a regression-discontinuity jump by hand, name the one assumption each design leans on, and run the diagnostic that tests it.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. All numbers come from the code below, on simulated data with known effects. Environment: Python 3.14, NumPy 2.5.3, pandas 2.3.3, statsmodels 0.15.0, linearmodels 7.0, rdrobust 2.1.1. Sources were opened on 8 October 2026.
:::

## In 30 seconds

Adjustment works when you can measure what drove treatment. Often you cannot: people who sign up for a course are more motivated, and motivation is not in your data. Quasi-experiments sidestep the problem by borrowing a comparison that is fair for a specific reason.

If a retailer rolls out a change in some stores, the other stores show what would have happened over time. If a letter encouraging people to join a programme is sent at random, the letter moves enrolment without touching earnings by any other route. If a scholarship goes to everyone who scores 50 or more, students at 49 and 51 are nearly the same. Each trick buys a clean comparison and costs an assumption you must defend.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Difference-in-differences (DiD) | The change in the treated group minus the change in a comparison group | (63 - 56) - (54 - 50) = 3 |
| Parallel trends | Without treatment, both groups would have changed by the same amount | Both would have risen 4 |
| Instrument | A variable that shifts treatment and affects the outcome only through it | A random nudge to enrol |
| First stage | How much the instrument moves treatment | Take-up 0.756 against 0.368 |
| Exclusion restriction | The instrument has no route to the outcome except through treatment | The nudge does not raise pay itself |
| Wald estimator | The outcome gap by instrument divided by the treatment gap by instrument | 0.80 / 0.40 = 2.0 |
| Weak instrument | An instrument whose first stage is small, so the estimate is noisy | First-stage F of 1.6 |
| Regression discontinuity (RD) | Compare units just above and just below a cutoff | Score 49 against 51 |
| Bandwidth | How far from the cutoff the data are allowed to come | Scores from 45 to 55 |
| Local average treatment effect (LATE) | The effect for those whose treatment the instrument or cutoff changes | People who enrol only if nudged |

## The idea in plain words

**Difference-in-differences.** A chain installs a new checkout system in 150 of its 300 stores. Sales rise in those stores. Did the system do it? Sales might have risen anyway, because it was December. The other 150 stores tell you how much sales rise in December without the system. Subtract that rise from the treated stores' rise and what remains is the effect. The key assumption is **parallel trends**: without the rollout, the treated stores would have moved in step with the others. Level differences between the groups do not matter, only how they change.

**Instrumental variables.** A city mails an encouragement letter to a random half of the unemployed to join a training course. Joining is the treatment. Motivated people join anyway, so comparing joiners with non-joiners is biased by motivation, which you cannot observe. The letter is random, so comparing everyone who got a letter with everyone who did not is fair, though it only measures the effect of the letter. Because the letter raised enrolment by some amount and nothing else, the effect of the course is the effect of the letter divided by how much the letter raised enrolment.

$$\text{effect of the course} = \frac{\text{outcome gap between letter and no letter}}{\text{enrolment gap between letter and no letter}}.$$

In words: if the letter moved 40 per cent of people into the course and earnings rose 0.80 on average, each person it moved gained 2.0. This is the **Wald estimator**, and it identifies the effect for those the letter moved, not for everyone.

**Regression discontinuity.** A programme admits everyone with a score of 50 or more. Students at 49.9 and 50.1 differ by a hair in ability and sit on opposite sides of a hard rule. Compare the outcomes just either side of the cutoff and you have something close to a randomised experiment for those students. The price is that it speaks only about people near the cutoff, and a narrow window means few data.

<Infographic src="/img/causal/did-lines.svg" alt="Left: two lines for control and treated stores before and after, with a dashed line showing where treated stores would have gone and a red bar for the effect; the estimate 2.996 with interval 2.679 to 3.314. Right: when treated stores were already speeding up, the estimate is 4.734, the pre-period trend gap is 0.426 with p of 0.0001, and the bias is the extra trend times the number of periods." caption="On the left, the gap between the solid and dashed treated lines is the effect. On the right, the same recipe gives 4.734 for a true 3.0, and the pre-period trend gap flags it." />

## Worked example, step by step

**Difference-in-differences.** Stores average 50 before and 54 after in the control group, and 56 before and 63 after in the treated group.

1. **Change in the control group:** 54 - 50 = 4. This is the time effect: what happens without the rollout.
2. **Change in the treated group:** 63 - 56 = 7.
3. **Subtract:** 7 - 4 = 3. The effect is 3.
4. **Compare with the wrong answers.** Treated minus control after the rollout is 63 - 54 = 9, which includes the 6-point gap that existed before. Treated after minus treated before is 7, which includes the time effect of 4.

**Instrument.** The letter is sent to a random half of 1,000 people. Enrolment is 75 per cent with a letter and 35 per cent without, a first stage of 0.40. Average earnings are 14.80 with a letter and 14.00 without, a gap of 0.80.

1. **Reduced form** (the outcome gap by letter): 14.80 - 14.00 = 0.80.
2. **First stage** (the enrolment gap by letter): 0.75 - 0.35 = 0.40.
3. **Wald estimate:** 0.80 / 0.40 = 2.0.
4. **Compare with the naive answer.** Comparing joiners with non-joiners would credit the course with the motivation of those who joined.

**Regression discontinuity.** Scores run from 30 to 70 and the programme starts at 50. A straight line fitted to the outcomes just below 50 reaches 20.0 at the cutoff. A straight line fitted just above reaches 23.0 at the cutoff.

1. **Fit each side separately** using only data near the cutoff.
2. **Read both lines at 50:** 20.0 and 23.0.
3. **Subtract:** the jump is 23.0 - 20.0 = 3.0.
4. **Why not just compare the two sides' averages?** The outcome rises with the score even without the programme, so the average above 50 is higher for that reason. Block 3 shows that comparison giving 7.648 for a true 3.0.

<Infographic src="/img/causal/iv-path.svg" alt="A diagram with a nudge pointing to take-up with first stage plus 0.388, take-up pointing to earnings with an unknown effect, a hidden ability variable pointing at both, and a red dashed forbidden arrow from nudge straight to earnings. A table gives OLS 3.508, Wald 1.981 and a leaking nudge 3.329 for a true 2.000. A strip below shows the estimate range widening as the first-stage F falls from 381.8 to 13.9 to 1.6." caption="Follow the green arrow, then the blue one: the instrument can only matter through take-up. The red dashed arrow is the assumption you cannot test. The strip at the bottom shows what a weak first stage does to the answer." />

## How it works

### What does DiD assume, and how do I test it?

Write each store's outcome as its own level plus a time effect shared by every store plus the treatment effect after the rollout. DiD removes the level and the shared time effect. What it cannot remove is a time effect that differs between groups, and that is the parallel trends assumption failing.

You cannot observe the treated stores' untreated future, but you can look at the past. If treated and control stores trended differently before the rollout, parallel trends is doubtful. Fit a group-by-period slope on the pre-period data and look at it. A flat result does not prove the assumption, because the future can still diverge, but a clear slope is strong evidence against it.

### What does an instrument have to satisfy?

Hernán and Robins give three conditions. **Relevance**: the instrument changes treatment. **Exclusion**: it affects the outcome only through treatment. **Independence**: it shares no common cause with the outcome, which a randomised letter guarantees. Relevance can be tested, as the first-stage F statistic. The other two cannot be tested from the data, only argued.

With a binary treatment and a varying effect, a fourth condition is needed: either the effect is the same for everyone (homogeneity) or the instrument pushes everyone the same way (monotonicity: nobody does the opposite of what the letter suggests). Under monotonicity the Wald estimate is the **local average treatment effect**, the effect for the people the instrument actually moved, who are called compliers. It can differ from the average effect over everyone.

### Why is a weak instrument dangerous?

The Wald estimate divides by the first stage. If the first stage is small, any noise in the numerator is magnified by its inverse. A first stage of 0.05 multiplies the noise by 20. A weak instrument also pulls the estimate toward the biased naive answer if the exclusion assumption leaks even slightly, because a small leak divided by a small first stage is still large.

A common rule of thumb asks for a first-stage F statistic above 10. The number to read is the F, not the p-value.

### How does regression discontinuity work, and what does bandwidth do?

Fit a line on each side of the cutoff using only data within a distance (the **bandwidth**) and read both lines at the cutoff. Points nearer the cutoff get more weight in a triangular kernel. A wide bandwidth uses more data and reduces noise, but if the outcome curves, a straight line fitted over a wide window misses the curve and biases the jump. A narrow bandwidth cuts that bias and increases noise. Block 3 measures this trade-off directly.

Three diagnostics are standard. Other variables measured before treatment should not jump at the cutoff. The number of units should not pile up on one side, which would suggest people manipulate their score. The estimate should be stable across reasonable bandwidths.

<Infographic src="/img/causal/rdd-cutoff.svg" alt="A line rising with score that jumps by 3.0 at a cutoff of 50, beside a table of mean jump and spread by bandwidth: 2.507 and 0.144 at half-width 20, 2.884 and 0.200 at 10, 2.959 and 0.278 at 5, 3.000 and 0.387 at 2.5, and a checklist of background-variable and density checks." caption="Read the table from top to bottom: as the window narrows, the mean climbs to the true 3.0 and the spread grows. The checklist on the right is what you run before believing the estimate." />

## Code you can run

Each block is self-contained and CPU only. Block 2 uses `linearmodels` for two-stage least squares and block 3 uses `rdrobust`, the package that implements the standard data-driven bandwidth, so the hand-written results can be checked against a library.

### 1. Difference-in-differences, with and without parallel trends

Three hundred stores, eight periods, half the stores treated from period 4. Store levels differ, there is a shared upward trend of 1.0 per period, and the true effect is 3.0. In the second run the treated stores also trend 0.4 faster than the rest.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def make_panel(rng, trend_gap, stores=300, periods=8, effect=3.0):
    rows = []
    for store in range(stores):
        treated = int(store < stores // 2)
        level = rng.normal(50, 5) + 6 * treated
        for t in range(periods):
            post = int(t >= 4)
            y = level + 1.0 * t + trend_gap * treated * t + effect * treated * post + rng.normal(0, 2)
            rows.append((store, t, treated, post, y))
    return pd.DataFrame(rows, columns=["store", "period", "treated", "post", "sales"])

rng = np.random.default_rng(5)
for label, gap in (("parallel trends hold", 0.0), ("treated stores trend 0.4 faster", 0.4)):
    df = make_panel(rng, gap)
    means = df.groupby(["treated", "post"]).sales.mean()
    by_hand = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
    fit = smf.ols("sales ~ treated * post", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.store})
    low, high = fit.conf_int().loc["treated:post"]
    print(f"{label}: difference-in-differences by hand {by_hand:.3f}, regression {fit.params['treated:post']:.3f} "
          f"(95% interval {low:.3f} to {high:.3f}), true effect 3.000")
    pre = df[df.post == 0]
    placebo = smf.ols("sales ~ treated * period", data=pre).fit(cov_type="cluster", cov_kwds={"groups": pre.store})
    print(f"   pre-period trend gap: {placebo.params['treated:period']:.3f} per period, p = {placebo.pvalues['treated:period']:.4f}")
```

**Reading the output.** When parallel trends holds, the hand calculation and the regression with a store-clustered standard error agree on 2.996, with an interval of 2.679 to 3.314 around the true 3.0. The pre-period trend gap is 0.157 per period with p = 0.133: no evidence of a problem, and in truth there is none.

When treated stores trend faster, the same recipe gives 4.734, with an interval of 4.412 to 5.056 that excludes the truth. The bias is the extra trend gained between the pre and post periods, 0.4 per period over about four periods, 1.6, plus noise. This time the pre-period check finds a gap of 0.426 per period with p = 0.0001. The assumption fails, and the data reveal it before the effect is read.

One caution from the first run: p = 0.133 on a pre-period check is absence of evidence, not evidence of absence. With fewer stores the same gap of 0.157 could hide a bigger problem.

**Line by line.**

- `level = rng.normal(50, 5) + 6 * treated` makes treated stores start 6 higher. DiD ignores that level gap, so the estimate is unaffected.
- `trend_gap * treated * t` is the violation of parallel trends: a slope that only treated stores have.
- `cov_type="cluster"` with `groups=df.store` clusters the standard errors by store, because the eight rows of one store are not independent. Unclustered errors would be too small.
- The `smf.ols("sales ~ treated * period", data=pre)` call is the pre-trend test: it uses only pre-period rows and asks whether the treated group's slope differs.

### 2. An instrument, a weak instrument and a leaking one

A random nudge to enrol in training raises take-up. Unobserved ability raises both take-up and earnings. The true effect of take-up is 2.0.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

def make_world(rng, n, strength, leak=0.0):
    ability = rng.normal(size=n)
    nudge = rng.binomial(1, 0.5, n)
    uptake = (strength * nudge + 1.0 * ability + rng.normal(size=n) > 0.5).astype(float)
    earnings = 2.0 * uptake + 1.5 * ability + leak * nudge + rng.normal(size=n)
    return pd.DataFrame({"nudge": nudge, "uptake": uptake, "earnings": earnings})

def wald_and_f(d):
    z, t, y = d.nudge.to_numpy(), d.uptake.to_numpy(), d.earnings.to_numpy()
    estimate = np.cov(z, y)[0, 1] / np.cov(z, t)[0, 1]
    r2 = np.corrcoef(z, t)[0, 1] ** 2
    return estimate, r2 * (len(d) - 2) / (1 - r2)

rng = np.random.default_rng(9)
df = make_world(rng, 20000, strength=1.5)
ols = sm.OLS(df.earnings, sm.add_constant(df.uptake)).fit()
wald = (df.earnings[df.nudge == 1].mean() - df.earnings[df.nudge == 0].mean()) / (
    df.uptake[df.nudge == 1].mean() - df.uptake[df.nudge == 0].mean())
iv = IV2SLS.from_formula("earnings ~ 1 + [uptake ~ nudge]", df).fit(cov_type="robust")
low, high = iv.conf_int().loc["uptake"]
print(f"OLS {ols.params['uptake']:.3f} | Wald by hand {wald:.3f} | 2SLS {iv.params['uptake']:.3f} "
      f"(95% interval {low:.3f} to {high:.3f}) | true 2.000")
print(f"first stage: take-up {df.uptake[df.nudge == 1].mean():.3f} with the nudge, "
      f"{df.uptake[df.nudge == 0].mean():.3f} without; F = {iv.first_stage.diagnostics.loc['uptake', 'f.stat']:.1f}")

print("\nnudge strength   first-stage F   median 2SLS   10th to 90th percentile   (300 repeats, n = 2000)")
for strength in (1.5, 0.3, 0.1):
    runs = [wald_and_f(make_world(rng, 2000, strength)) for _ in range(300)]
    est, fstat = [r[0] for r in runs], [r[1] for r in runs]
    p10, p50, p90 = np.percentile(est, [10, 50, 90])
    print(f"{strength:14.1f} {np.median(fstat):15.1f} {p50:13.3f} {p10:12.3f} to {p90:7.3f}")

leaky = make_world(rng, 20000, strength=1.5, leak=0.5)
r = IV2SLS.from_formula("earnings ~ 1 + [uptake ~ nudge]", leaky).fit()
print(f"\nnudge also raises earnings by 0.5 directly: 2SLS {r.params['uptake']:.3f} (true effect 2.000)")
```

**Reading the output.** Ordinary least squares gives 3.508, which credits training with ability. The Wald estimate computed by hand and two-stage least squares from `linearmodels` agree to the third decimal, 1.981, with an interval of 1.851 to 2.112 around the true 2.0. The first stage is large: take-up is 0.756 with the nudge and 0.368 without, F = 3617.

The repeat table is the lesson. At nudge strength 1.5 (F around 382) the middle 80 per cent of estimates is 1.711 to 2.228, and the median is 1.984. At strength 0.3 (F 13.9) the median is still 1.963 but the range is 0.281 to 3.113, about six times wider. At strength 0.1 (F 1.6) the range is -5.627 to 7.484 and the median is 2.593: the instrument is relevant on paper and useless in practice.

The last line shows the exclusion restriction failing. When the nudge raises earnings by 0.5 directly, the estimate is 3.329. The bias is the direct effect divided by the first stage, 0.5 / 0.388 = 1.29, close to the 1.33 seen. Nothing in the data would warn you.

The honest surprise: the F statistic at strength 0.3 is 13.9, above the rule of thumb of 10, and the estimate still ranges from 0.281 to 3.113. The rule of thumb is a floor, not a certificate.

**Line by line.**

- `wald_and_f` computes the Wald estimate as `cov(z, y) / cov(z, t)` and the first-stage F from the squared correlation. With one instrument and no controls, two-stage least squares and the Wald estimator are the same number, and block 2's first line shows it.
- `IV2SLS.from_formula("earnings ~ 1 + [uptake ~ nudge]", df)` reads as: earnings on a constant and on take-up, with take-up instrumented by the nudge.
- The three-row loop varies only `strength`, so differences between rows come from the instrument alone.

### 3. A cutoff, four bandwidths and three checks

Scores are uniform from 30 to 70 and the programme starts at 50. The outcome has a slope everywhere and extra curvature above the cutoff. The true jump is 3.0.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rdrobust import rdrobust

def make_data(rng, n=4000):
    score = rng.uniform(30, 70, n)
    x = score - 50
    passed = (score >= 50).astype(int)
    curve = 0.15 * x + np.where(x >= 0, 0.012 * x ** 2, 0.0)
    return pd.DataFrame({"score": score, "centred": x, "passed": passed,
                         "outcome": 20 + curve + 3.0 * passed + rng.normal(0, 2, n)})

def local_linear(df, h):
    local = df[df.centred.abs() <= h].copy()
    local["weight"] = 1 - local.centred.abs() / h
    fit = smf.wls("outcome ~ passed * centred", data=local, weights=local.weight).fit(cov_type="HC1")
    return fit.params["passed"], fit.conf_int().loc["passed"].tolist(), len(local)

rng = np.random.default_rng(19)
df = make_data(rng)
print(f"difference in means above and below the cutoff: "
      f"{df[df.passed == 1].outcome.mean() - df[df.passed == 0].outcome.mean():.3f} (true jump 3.000)")
print(f"{'bandwidth':>10} {'rows used':>10} {'jump':>8} {'95% interval':>20}")
for h in (20, 10, 5, 2.5):
    jump, (low, high), rows = local_linear(df, h)
    print(f"{h:10.1f} {rows:10d} {jump:8.3f} {low:10.3f} to {high:7.3f}")
fit = rdrobust(df.outcome.to_numpy(), df.centred.to_numpy(), c=0)
print(f"rdrobust chose bandwidth {fit.bws.iloc[0, 0]:.2f}: jump {fit.coef.iloc[0, 0]:.3f}")

print("\nbandwidth   mean jump   spread   (300 repeats)")
for h in (20, 10, 5, 2.5):
    est = [local_linear(make_data(rng), h)[0] for _ in range(300)]
    print(f"{h:9.1f} {np.mean(est):11.3f} {np.std(est):8.3f}")

check = df[df.centred.abs() <= 5].copy()
check["background"] = 30 + 0.2 * check.centred + rng.normal(0, 3, len(check))
placebo = smf.ols("background ~ passed * centred", data=check).fit(cov_type="HC1")
print(f"\nbackground variable jump at the cutoff: {placebo.params['passed']:.3f} (p = {placebo.pvalues['passed']:.3f})")
print("rows in the four 2.5-point bins around the cutoff:", np.histogram(df.score, bins=[45, 47.5, 50, 52.5, 55])[0].tolist())
```

**Reading the output.** Comparing the two sides' averages gives 7.648, because the outcome rises with the score. The local linear fits on this sample give 2.658, 2.946, 2.963 and 2.843 for half-widths 20, 10, 5 and 2.5, and `rdrobust` chose a bandwidth of 6.10 and reported 2.976. The widest window is the only one whose interval, 2.391 to 2.924, excludes 3.0.

One sample is an anecdote, so the repeats give the pattern. Averaged over 300 samples the estimate is 2.507 at half-width 20, 2.884 at 10, 2.959 at 5 and 3.000 at 2.5, while the spread grows from 0.144 to 0.387. A wide window is biased by the curvature, which a straight line cannot follow. A narrow window is unbiased but noisy: at half-width 2.5 a single estimate can easily land 0.8 away from the truth in either direction, and this sample's interval at that width runs from 2.073 to 3.612.

The surprise is that the narrowest window is not the best. Going from half-width 10 to 2.5 removes a bias of 0.116 and nearly doubles the spread, from 0.200 to 0.387, so the root mean squared error rises from about 0.23 to 0.39. The best bandwidth here is in the middle.

The background-variable check finds a jump of -0.075 with p = 0.839. It is an unrelated variable, so it should not jump, and it does not. The four 2.5-point bins around the cutoff hold 266, 256, 257 and 258 rows, with no pile-up on either side.

**Line by line.**

- `np.where(x >= 0, 0.012 * x ** 2, 0.0)` adds curvature only on the right side, which is what makes a straight-line fit over a wide window biased.
- `local["weight"] = 1 - local.centred.abs() / h` is the triangular kernel: weight 1 at the cutoff, falling to 0 at the edge of the window.
- `passed * centred` lets each side have its own slope, so the coefficient on `passed` is the gap between the two lines at the cutoff.

## Try it yourself

Both labs use exact formulas, so they show the mechanism without sampling noise. The difference-in-differences lab's defaults give exactly 3 (block 1's 2.996 is that number plus noise). The Wald lab's defaults give exactly 2. Both were checked against large simulations (200,000 stores per group and 4 million people) to within 0.005.

<DidLab />

**What each control does.**

- **Change over time in both groups** is the common time effect.
- **Treated minus control before** is the level gap, which DiD ignores.
- **True effect** is what we want back.
- **Extra trend in treated group** breaks parallel trends: the treated group would have drifted that much even without treatment.

**Try it yourself.**

1. Set the extra trend to 1.6. The estimate becomes 4.6, a bias of exactly the extra trend. Block 1's 4.734 is the same idea with 0.4 per period over four periods. Why: DiD cannot tell treatment from a head start in the trend.
2. Set the true effect to 0 and the extra trend to 2. DiD reports 2: an effect where none exists. Why: with no effect, any difference in trends is read as treatment.
3. Set the treated gap to -5 with the other defaults. Treated minus control after the rollout becomes -2, but DiD still reports 3. Why: the level gap is the same before and after, so it cancels.

<WaldIvLab />

**What each control does.**

- **Take-up with the nudge** and **Take-up without the nudge** set the first stage, their difference.
- **True effect of take-up** is what the estimate should recover.
- **Direct effect of nudge on earnings** opens the forbidden path.
- Click **show data** for the first stage, reduced form, bias and noise multiplier.

**Try it yourself.**

1. With the defaults, the Wald estimate is exactly 2.000. Set the direct effect to 0.5: it becomes 3.289. Why: the direct effect is divided by the first stage, 0.5 / 0.388 = 1.29.
2. Set take-up to 0.5 with the nudge and 0.45 without. The first stage is 0.05, the noise multiplier is 20, and the estimate is still 2.000 in the formulas but would be wildly noisy in data. Add a direct effect of 0.1: the bias is 2.0. Why: small first stages amplify every leak.
3. Set take-up with and without equal at 0.5. The Wald estimate is undefined: no first stage, no instrument.

## Designing with it

A usable order of preference when you cannot randomise:

1. **Ask whether the world already ran an experiment.** A lottery, a rule with a hard cutoff, a staggered rollout, a policy change in one region. These are worth more than any model.
2. **Write the identifying assumption in one sentence** (parallel trends, exclusion, no manipulation of the score) and say what data would worry you.
3. **Run the design's own diagnostic**: pre-trends, first-stage F, covariate and density checks at the cutoff.
4. **Say who the estimate is about.** The Wald estimate speaks about compliers, the RD estimate about people near the cutoff, the DiD estimate about the treated group. A policy that reaches other people needs further assumptions.
5. **Keep an eye on noise.** All three designs throw data away, or use only a slice of it, and the result is wider intervals than the naive comparison.

## Where this stands in 2026

These designs are routine in economics and in tech experimentation platforms, where staggered rollouts and eligibility cutoffs are common. The active development is in difference-in-differences with different adoption dates and effects that vary over time. Basic two-way regressions can mislead there, and Facure's book has a later chapter on that subject and one on synthetic difference-in-differences, whose contents lists were read but whose text was not. For regression discontinuity, packages such as `rdrobust` now choose the bandwidth from the data and report bias-corrected intervals, which block 3 uses.

## Common mistakes

1. **Choosing a comparison group because it is convenient.** Any group feels like a control. Parallel trends needs a group that would have moved with the treated group. Check several pre-periods and look at the plot.
2. **Reading a clean pre-trend test as proof.** Block 1's p of 0.133 was right here and could be wrong with less data. Report the estimate and its interval for the pre-trend, not only a p-value.
3. **Using an instrument with F just above 10 and trusting the interval.** Block 2 shows F = 13.9 with a range of 0.281 to 3.113. Treat weak-instrument-robust intervals as the standard, and be sceptical of single estimates.
4. **Defending exclusion by saying "it is obviously unrelated".** The nudge that leaks by 0.5 produces 3.329 and nothing in the data complains. List every route from the instrument to the outcome and argue each away.
5. **Reporting a single regression-discontinuity bandwidth.** One bandwidth hides the bias and noise trade-off. Show several and the data-driven choice.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Control stores go from 50 to 54 and treated stores from 56 to 63. Compute the difference-in-differences estimate and name one wrong answer.</summary>

(63 - 56) - (54 - 50) = 7 - 4 = 3. A wrong answer is 63 - 54 = 9, which includes the pre-existing gap of 6. Another is 63 - 56 = 7, which includes the common time effect of 4.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> Why does comparing joiners with non-joiners fail when training is voluntary?</summary>

People who join are different, for example more motivated, and that difference raises their earnings with or without training. The comparison mixes the effect of training with the effect of motivation. Block 2's OLS result of 3.508 for a true 2.0 shows the size of the problem.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> The letter raises enrolment from 35 to 75 per cent and average earnings from 14.00 to 14.80. What effect does the Wald estimator report, and for whom?</summary>

0.80 / 0.40 = 2.0. Under monotonicity it is the average effect for compliers: the people who enrol because of the letter and would not otherwise. It says nothing certain about always-takers or never-takers.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> In block 2 the first-stage F at strength 0.3 was 13.9, above the rule of thumb of 10. Why was the range of estimates still 0.281 to 3.113?</summary>

The first stage is a take-up gap of about 0.07, so the Wald estimate multiplies noise by roughly 14. Many other data sets of the same size give estimates scattered over that range. F above 10 limits the worst bias of a single 2SLS fit, but it does not make the estimate precise. The correct reading of the table is that this design would not tell 1 from 3.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Show that when the instrument also has a direct effect d on the outcome, the Wald estimate equals the true effect plus d divided by the first stage.</summary>

The outcome gap by instrument is (true effect times first stage) plus d: the effect passes through the take-up gap and the direct effect adds d. Dividing by the first stage gives the true effect plus d over the first stage. With d = 0.5 and a first stage of 0.388, the bias is 1.29, and the lab's Wald estimate is 3.289. Block 2 measured 3.329 on a sample of 20,000.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> In block 3, the estimate at half-width 2.5 was 3.000 on average but its spread was 0.387, against 0.144 at half-width 20. How would you pick a bandwidth for a real data set?</summary>

The usual criterion is mean squared error, which adds squared bias to variance. Here the root mean squared error is about sqrt(0.493 squared + 0.144 squared) = 0.51 at half-width 20, about 0.23 at 10, 0.28 at 5 and 0.39 at 2.5, so a middle bandwidth wins. In practice the bias is unknown, so use a data-driven selector such as `rdrobust`, report several bandwidths, and check that the conclusion does not hinge on one choice.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Hernán MA and Robins JM, *Causal Inference: What If*, edition dated 19 August 2026. Chapter 16, "Instrumental variable estimation", with the three instrumental conditions, the usual IV estimand, homogeneity and monotonicity, was read in the contents list; the book does not cover difference-in-differences or regression discontinuity under those names. Free to read at the author's page.
- Facure Alves M, *Causal Inference for the Brave and True*: chapters on instrumental variables, non-compliance and LATE, difference-in-differences, panel data and fixed effects, synthetic control and regression discontinuity design. Contents list read; no stated licence.
- [linearmodels documentation](https://bashtage.github.io/linearmodels/): `IV2SLS` and its first-stage diagnostics. Version 7.0 was run.
- [rdrobust on PyPI](https://pypi.org/project/rdrobust/): local polynomial regression discontinuity with data-driven bandwidths and robust bias-corrected inference. Version 2.1.1 was run.

## Check yourself

- I can compute a difference-in-differences, a Wald estimate and a regression-discontinuity jump from a small table.
- I can state the parallel-trends, exclusion and no-manipulation assumptions and say which cannot be tested.
- I can explain why a weak first stage makes an estimate noisy and amplifies any exclusion leak, with the numbers from block 2.
- I can describe the bias and noise trade-off in the regression-discontinuity bandwidth and run the standard checks.
- I can say whom each estimate is about.

## Where to go next

Next chapter: [causal machine learning in practice](/docs/theory/causal/causal-ml-uplift-and-dml), where the question changes from "what is the average effect" to "who should we treat", and machine-learning models enter safely through uplift modelling and double machine learning. A related chapter: [time series evaluation and operations](/docs/theory/timeseries/evaluation-and-operations), whose rolling-origin logic is the same pre-period discipline that DiD uses.
