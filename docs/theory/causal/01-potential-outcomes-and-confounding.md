---
id: causal-potential-outcomes
title: "Correlation, Causation, Potential Outcomes and Confounding"
sidebar_label: "1 · Potential outcomes and confounding"
sidebar_position: 1
slug: /theory/causal/potential-outcomes-and-confounding
description: "Why a naive comparison of treated and untreated users is biased, how potential outcomes and causal diagrams define the effect we want, and which variables to adjust for, shown on simulated data where the true effect is known."
tags: [causal-inference, potential-outcomes, confounding, dag, backdoor-criterion, collider, dowhy]
---

import Infographic from '@site/src/components/Infographic';
import ConfoundingLab from '@site/src/components/viz/ConfoundingLab';

**In one line.** A causal effect is the difference between what happened and what would have happened otherwise, and a plain comparison of treated and untreated people recovers it only when the two groups were alike before the treatment.

:::tip Before you start
- **You should already know** what an average and a regression coefficient are ([regression and gradient descent](/docs/theory/ml/regression-and-gradient-descent)) and what correlation measures ([correlation](/docs/theory/statistics/16)).
- **Reading time:** about 40 minutes, plus about ten seconds to run the code.
- **After this chapter you can** write down the effect you want to estimate as a difference in potential outcomes, explain why a naive comparison is biased, read a causal diagram to choose what to adjust for, and name three kinds of variable that adjustment makes worse.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every number is printed by the code in the chapter, on simulated data in which the true effect is built in, so each estimate can be checked against the truth. Environment: Python 3.14, NumPy 2.5.3, pandas 2.3.3, statsmodels 0.15.0, NetworkX 3.6.1, DoWhy 0.8. DoWhy 0.14 (8 November 2025) is the latest on PyPI, but its metadata excludes Python 3.14, so pip installed 0.8 here. Sources were opened on 8 October 2026.
:::

## In 30 seconds

An app sends a notification to some users and not to others. The notified users spend more. Did the notification cause that? Not necessarily: the app may send notifications mostly to users who already love it, and those users would have spent more anyway. Think of two classes of runners where only the fast class gets the new shoes. The shoe-wearers win, but the shoes may have nothing to do with it.

Causal inference is the craft of asking what the fast class would have done without shoes. You cannot watch that, so you build the answer from assumptions you can write down and test where possible. This chapter shows the trap, the language used to describe it, and the first tool for escaping it.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Treatment | The thing whose effect we want, often a yes or no | Notified, or not |
| Potential outcome | What a user's outcome would be under each treatment, written Y(1) and Y(0) | Spend if notified, spend if not |
| Counterfactual | The potential outcome that did not happen | The spend of a notified user had they not been notified |
| Average treatment effect (ATE) | The average of Y(1) minus Y(0) over everyone | 2.0 extra spend per user |
| Confounder | A variable that influences both who is treated and the outcome | Engagement |
| Randomisation | Assigning treatment by a coin, so nothing else can steer it | A/B test |
| Causal diagram (DAG) | A drawing of which variables cause which, with arrows | engaged to notified, engaged to spend |
| Backdoor path | A path from treatment to outcome that starts with an arrow into the treatment | notified, engaged, spend |
| Collider | A variable with two arrows pointing into it | A review written by happy buyers and by treated users |

## The idea in plain words

Start with a single user, Asha. If we notify her she spends 15.81. If we do not, she spends 13.81. The effect of the notification on Asha is the difference, 2.00. The trouble is that we only ever see one of the two numbers. If she is notified, 13.81 is a number about a world that never existed. That gap is the fundamental problem of causal inference: the quantity we want is defined by two outcomes and we observe one.

So we settle for averages. Take all users and write Y(1) for the spend each would have if notified and Y(0) for the spend each would have if not. The **average treatment effect** is

$$\mathrm{ATE} = \mathrm{E}[Y(1) - Y(0)].$$

In words: the average, over everyone, of what the treatment would add.

The tempting estimate is the difference between the average spend of the notified and the average spend of the rest. That compares two different groups of people. If the notified group would have spent more even without notification, the difference mixes the effect with this head start:

$$\underbrace{\mathrm{E}[Y \mid T{=}1] - \mathrm{E}[Y \mid T{=}0]}_{\text{what we see}} = \underbrace{\mathrm{E}[Y(1) - Y(0) \mid T{=}1]}_{\text{effect on the notified}} + \underbrace{\mathrm{E}[Y(0) \mid T{=}1] - \mathrm{E}[Y(0) \mid T{=}0]}_{\text{head start}}.$$

In words: what we see equals the effect plus the head start the notified group had. The head start is the bias. Making it vanish is the whole game.

There are two honest ways to make it vanish. The first is to **randomise**: if a coin decides who is notified, nobody has a head start, on average. The second is to **adjust**: if you know what drove the choice, compare notified and not notified users who look alike on that, and average the comparisons. Adjusting works only if you measured everything that drove the choice. That is an assumption about the world, not something the data can prove, which is why the second half of this chapter is about drawing that assumption as a diagram.

<Infographic src="/img/causal/potential-outcomes.svg" alt="A table of six simulated users with a question mark wherever the outcome of the other treatment would have been, beside three cards: the simulator's view with the true average effect 2.000, the analyst's view with the naive difference 4.369, and the reason for the gap, that 79.9 per cent of notified users were engaged against 20.4 per cent of the others." caption="Look at the question marks first: every row hides one of its two outcomes. The right-hand cards show what you get from knowing both columns, and what you get from the one column you have." />

## Worked example, step by step

A shop of 1,000 users. Half are engaged (they open the app daily), half are not. Engaged users are far more likely to be notified: 400 of 500 engaged users are notified, against 100 of 500 others. Average spend, built in by construction, is 10 for others and 14 for engaged users, and the notification adds exactly 2 to either.

1. **Fill in the four cells.** Engaged and notified: 16. Engaged, not notified: 14. Others notified: 12. Others not notified: 10.
2. **Compare the groups naively.** The notified group is 400 engaged and 100 others: (400 x 16 + 100 x 12) / 500 = 15.2. The not-notified group is 100 engaged and 400 others: (100 x 14 + 400 x 10) / 500 = 10.8. The naive gap is 15.2 - 10.8 = 4.4, more than double the true effect.
3. **Find the source of the bias.** Notified users are 80 per cent engaged; others are 20 per cent engaged. Engagement is worth 4, so the head start is 4 x (0.8 - 0.2) = 2.4. Then 2.0 + 2.4 = 4.4.
4. **Compare inside each row.** Engaged: 16 - 14 = 2. Others: 12 - 10 = 2. Weighted by how many users are in each row (500 each) the adjusted effect is 2.0, the truth.

<Infographic src="/img/causal/worked-example.svg" alt="Three steps for 1,000 users: who gets notified, the average spend in each of four cells, and the arithmetic of the naive comparison (15.2 minus 10.8 equals 4.4) against the adjusted comparison (2 and 2 average to 2.0)." caption="Read the table in the top right first: the gap is 2 in both rows, so the effect is 2. The red card then shows how mixing the rows manufactures 4.4." />

The code in block 1 simulates 20,000 users with exactly these probabilities and these effect sizes, so the numbers will land close to 4.4 and 2.0 but not on them, because samples are random.

## How it works

### What makes a naive comparison valid?

Three conditions, in the language of Hernán and Robins, let you read an effect off data.

- **Exchangeability.** The treated and untreated would have had the same outcomes had they all been untreated. Randomisation guarantees it. Within strata of a confounder, it may hold conditionally: engaged users who were notified are exchangeable with engaged users who were not.
- **Positivity.** Every kind of user has some chance of each treatment. If no engaged user is ever left un-notified, there is no comparison to make for that row.
- **Consistency.** The outcome you observe for a treated user equals their potential outcome under treatment. This sounds trivial, and it fails when "treatment" is vague (what exactly is "exercise"?) or when one user's treatment changes another's outcome, as in a marketplace.

### What does a causal diagram add?

A diagram is a set of arrows you believe, one per direct cause. It does not say how strong an arrow is; it says which arrows exist, and, as important, which do not. From it you can read off which variables to adjust for without fitting a single model.

Three shapes cover most cases.

| Shape | Arrows | Adjust for the middle variable? | Why |
| --- | --- | --- | --- |
| Fork (confounder) | Z to T and Z to Y | Yes | Opens a backdoor path T, Z, Y that carries the head start |
| Chain (mediator) | T to M to Y | Not for the total effect | M carries part of the effect itself |
| Collider | T to C and Y to C | No | Conditioning on C opens a path that was closed |

The **backdoor criterion** says: to estimate the effect of T on Y, find a set of variables that blocks every path from T to Y that begins with an arrow into T, and contains no descendant of T. In the notification story the only such path is notified, engaged, spend, and adjusting for engaged blocks it.

### Why does conditioning on a collider hurt?

Imagine a restaurant that reviews only dishes that are either cheap or excellent. Among reviewed dishes, expensive ones are excellent and cheap ones are mediocre, so price looks like it lowers quality, even if price and quality are unrelated across all dishes. Conditioning on "was reviewed" created a relationship out of nothing. That is collider bias, and block 3 shows it flipping the sign of an effect in a randomised experiment.

<Infographic src="/img/causal/three-roles.svg" alt="Three small diagrams. An outcome-only cause of spend is harmless to adjust for, moving the estimate from 2.017 to 2.004. A mediator, usage, carries part of the effect, and adjusting for it reduces the estimate from 2.000 to 1.013. A collider, reviewed, is caused by both treatment and spend, and adjusting for it flips the estimate to minus 0.257." caption="Compare the bottom numbers in each column. Only the first adjustment is safe. The right column is the surprise: one extra variable turns a correct +2.017 into -0.257." />

## Code you can run

Everything is CPU only and takes seconds. All three blocks are self-contained.

### 1. The two worlds, and the naive answer

The simulator draws engagement, then notification (80 per cent for engaged users, 20 per cent for the rest), then spend. It stores both potential outcomes, which no analyst ever has, so the true effect is the mean of their difference.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(7)
n = 20000
engaged = rng.binomial(1, 0.5, n)
notified = rng.binomial(1, np.where(engaged == 1, 0.8, 0.2))
noise = rng.normal(0, 1, n)
spend0 = 10 + 4 * engaged + noise
spend1 = spend0 + 2.0
spend = np.where(notified == 1, spend1, spend0)

world = pd.DataFrame({"engaged": engaged, "notified": notified,
                      "spend0": spend0, "spend1": spend1, "spend": spend})
print(world.head(6).round(2).to_string())
print()

true_ate = (world.spend1 - world.spend0).mean()
naive = world.loc[world.notified == 1, "spend"].mean() - world.loc[world.notified == 0, "spend"].mean()
print(f"true ATE from both columns : {true_ate:.3f}")
print(f"naive difference in means  : {naive:.3f}")

share_engaged = world.groupby("notified").engaged.mean()
print(f"engaged share among notified {share_engaged[1]:.3f}, among not notified {share_engaged[0]:.3f}")

by_group = world.groupby(["engaged", "notified"]).spend.agg(["mean", "size"]).round(3)
print(by_group)
gaps = world.groupby("engaged").apply(
    lambda g: g.loc[g.notified == 1, "spend"].mean() - g.loc[g.notified == 0, "spend"].mean(),
    include_groups=False)
standardised = (gaps * world.engaged.value_counts(normalize=True).sort_index()).sum()
print(f"gap inside each group: {gaps.round(3).to_dict()}  weighted by group size: {standardised:.3f}")

fit = smf.ols("spend ~ notified + engaged", data=world).fit()
low, high = fit.conf_int().loc["notified"]
print(f"regression adjusting for engagement: {fit.params['notified']:.3f}  (95% interval {low:.3f} to {high:.3f})")
```

**Reading the output.** The first rows show both columns for each user; `spend` equals `spend1` where `notified` is 1 and `spend0` otherwise. The mean of `spend1 - spend0` is exactly 2.000 by construction, the truth. The naive difference is 4.369, more than double, close to the 4.4 of the worked example. Notified users are 79.9 per cent engaged against 20.4 per cent of the rest, the 80 and 20 we built in.

The cell means match the hand table: 9.998, 11.973, 14.044 and 16.005, against 10, 12, 14 and 16. Inside each engagement group the gap is 1.975 and 1.961, and weighting by group size gives 1.968. The regression with `engaged` as a control agrees, 1.968, with a 95 per cent interval of 1.933 to 2.002.

One honest detail: the adjusted estimate is 0.032 below the truth, and the interval only just contains 2.0. That is ordinary sampling noise on one sample of 20,000, not a flaw in adjustment. Nobody could tell 1.97 from 2.00 here.

**Line by line.**

- `np.where(engaged == 1, 0.8, 0.2)` makes the notification probability depend on engagement, which is the confounding.
- `spend1 = spend0 + 2.0` builds a constant effect of exactly 2.0 into the world.
- `world.groupby(["engaged", "notified"])` is the standardisation of the worked example in code: first compare within groups, then average the gaps with group sizes as weights.
- `smf.ols("spend ~ notified + engaged", ...)` is the same adjustment as a regression, and the coefficient on `notified` is the adjusted effect.

### 2. Reading the adjustment set from the diagram

NetworkX checks the backdoor criterion by testing d-separation, and DoWhy automates the whole workflow: write the diagram, identify the effect, estimate it, then try to break the answer with a refutation.

DoWhy 0.8 calls a NetworkX function that NetworkX 3.6.1 renamed, so the single alias line before the import restores it. Without that line DoWhy raises an `AttributeError` here.

```python
import contextlib
import io
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
nx.algorithms.d_separated = nx.is_d_separator
from dowhy import CausalModel

graph = nx.DiGraph([("engaged", "notified"), ("engaged", "spend"), ("notified", "spend")])
backdoor = graph.copy()
backdoor.remove_edge("notified", "spend")
print("back-door paths blocked by nothing   :", nx.is_d_separator(backdoor, {"notified"}, {"spend"}, set()))
print("back-door paths blocked by engaged  :", nx.is_d_separator(backdoor, {"notified"}, {"spend"}, {"engaged"}))

rng = np.random.default_rng(7)
n = 20000
engaged = rng.binomial(1, 0.5, n)
notified = rng.binomial(1, np.where(engaged == 1, 0.8, 0.2))
spend = 10 + 4 * engaged + 2.0 * notified + rng.normal(0, 1, n)
data = pd.DataFrame({"engaged": engaged, "notified": notified, "spend": spend})

gml = "graph [directed 1 node [id 0 label \"engaged\"] node [id 1 label \"notified\"] node [id 2 label \"spend\"] " \
      "edge [source 0 target 1] edge [source 0 target 2] edge [source 1 target 2]]"
model = CausalModel(data=data, treatment="notified", outcome="spend", graph=gml)
estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("backdoor adjustment set:", estimand.get_backdoor_variables())
with contextlib.redirect_stdout(io.StringIO()):
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
print(f"DoWhy linear-regression estimate: {estimate.value:.3f}")
with contextlib.redirect_stdout(io.StringIO()):
    placebo = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter",
                                    placebo_type="permute", num_simulations=20, random_seed=1)
print(f"placebo treatment effect (should be near 0): {placebo.new_effect:.3f}")
```

**Reading the output.** With the arrow from `notified` to `spend` removed, the only paths left between them are backdoor paths. They are not blocked by nothing (`False`) and they are blocked once `engaged` is given (`True`). DoWhy finds the same set, `['engaged']`, from the diagram alone and its linear-regression estimate is 1.968, identical to block 1's. The placebo test replaces the real treatment by a shuffled one and re-estimates; the average effect it finds is -0.003, close to zero as it must be. A placebo effect far from zero would mean the pipeline invents effects.

**Line by line.**

- `nx.algorithms.d_separated = nx.is_d_separator` is the compatibility alias described above.
- `graph.copy()` then `remove_edge` builds the backdoor graph: only the paths that start with an arrow into the treatment remain.
- `redirect_stdout` hides DoWhy's progress messages, which would otherwise print dozens of lines.

### 3. Bad controls: a mediator and a collider

This is a randomised experiment, so adjustment is not even needed. We add control variables of three kinds anyway to see what each does to the estimate.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(11)
n = 50000
treated = rng.binomial(1, 0.5, n)
quality = rng.normal(0, 1, n)
usage = 1.0 * treated + rng.normal(0, 1, n)
spend = 1.0 * treated + 1.0 * usage + 1.0 * quality + rng.normal(0, 1, n)
reviewed = 1.0 * treated + 1.0 * spend + rng.normal(0, 1, n)
df = pd.DataFrame({"treated": treated, "quality": quality, "usage": usage,
                   "spend": spend, "reviewed": reviewed})

print("true total effect of treated on spend: 1.0 direct + 1.0 x 1.0 via usage = 2.0")
for label, formula in [
    ("nothing adjusted (randomised)", "spend ~ treated"),
    ("adjust for quality (harmless)", "spend ~ treated + quality"),
    ("adjust for usage (a mediator)", "spend ~ treated + usage"),
    ("adjust for reviewed (a collider)", "spend ~ treated + reviewed"),
]:
    fit = smf.ols(formula, data=df).fit()
    print(f"{label:34s} effect {fit.params['treated']:+.3f}   se {fit.bse['treated']:.3f}")
```

**Reading the output.** The true total effect is 2.0: 1.0 directly plus 1.0 through `usage`. With no adjustment the randomised estimate is 2.017. Adding `quality`, a cause of spend that has nothing to do with treatment, leaves it at 2.004 but tightens the standard error from 0.015 to 0.013, a small gain. Adding `usage` gives 1.013: it removes the half of the effect that flows through usage, which is correct for the direct effect and wrong for the total. Adding `reviewed` gives -0.257. The estimate is not slightly off; it has the wrong sign.

The surprise to remember is the last row. The treatment was assigned by a coin, which is the strongest design there is, and one extra regression control broke it. Randomisation protects the comparison you make, not the comparison you make after conditioning on something the treatment affects.

**Line by line.**

- `reviewed = treated + spend + noise` makes `reviewed` a collider: it is caused by both treatment and outcome.
- The loop fits four regressions that differ only in their control, so every change in the coefficient comes from that control alone.

## Try it yourself

The lab uses the exact formulas for the worked example, so its defaults reproduce the 4.4 of the worked example. The 4.369 of block 1 is that number plus sampling noise. The formulas were checked against a simulation of four million users in four configurations and agreed to within 0.002.

<ConfoundingLab />

**What each control does.**

- **Share of users who are engaged** is the fraction of the population with the head start.
- **Notified, among engaged** and **Notified, among others** are the two notification rates. When they differ, treatment depends on engagement.
- **Spend gained from engagement** is how much the head start is worth.
- **True effect of notification** is the effect we are trying to recover.

**Try it yourself.**

1. Set both notification rates to 0.5. The naive gap falls to exactly the true effect, 2.0. Why: with equal rates the notified and un-notified groups contain the same share of engaged users, so nobody has a head start. This is what a coin flip does.
2. Return to the defaults and set the spend gained from engagement to 0. The naive gap is 2.0 again, even though notifications still go mostly to engaged users. Why: confounding needs both arrows. A variable that steers treatment but does not affect the outcome is harmless.
3. Set the spend gained from engagement to 8 with the other defaults. The naive gap becomes 6.8, more than three times the truth. Why: the head start is 8 x (0.8 - 0.2) = 4.8, added to 2.

## Designing with it

Treat every analysis as three questions, in this order.

1. **What is the effect I want, and for whom?** Write it as a difference in potential outcomes with a population and a time window. "Does the notification raise next-week spend for active users" is a question. "Do notifications work" is not.
2. **Why did each person get the treatment they got?** List the reasons, ask which also move the outcome, and draw them. Reasons you cannot measure are the weak point; say so in the report.
3. **Which design makes the comparison fair?** If you can randomise, do. If not, the next chapters show adjustment, weighting and quasi-experiments, each with the assumption it needs written beside it.

A rule that saves projects: pick controls from the diagram, not from a significance test or a feature-importance ranking. A predictive model happily uses a collider because it predicts well; a causal analysis must not.

## Where this stands in 2026

The language of potential outcomes (Rubin) and diagrams (Pearl) is now standard in experimentation platforms and in the machine-learning literature on fairness and uncertainty. Hernán and Robins' book, whose current online edition is dated 19 August 2026, reframes observational analysis as emulating a "target trial": write the randomised experiment you wish you had run, then ask which of its parts your data can reproduce. Practical libraries such as DoWhy wrap the identify, estimate and refute steps so that the assumptions are explicit, but no library can supply the diagram. That remains a human judgement, which is why the refutation tests only catch some failures.

## Common mistakes

1. **Treating correlation in a dashboard as an effect.** It feels right because the difference is real and statistically significant. Significance measures noise, not bias. Ask who got the treatment and why.
2. **Adjusting for everything in the table.** More controls feel like more rigour. Block 3 shows a mediator halving the effect and a collider flipping its sign. Choose controls from a diagram.
3. **Believing randomisation makes any later analysis safe.** The coin protects the comparison of whole groups. Subsetting by something the treatment changed (users who left a review, who stayed in the study, who completed onboarding) breaks it.
4. **Calling an estimate causal because the model was flexible.** A gradient-boosted model of spend on treatment and thirty features still cannot see a confounder you did not measure. The identifying assumption lives in the data collection, not the model class.
5. **Skipping positivity.** If a type of user is never untreated, the adjusted comparison silently extrapolates. Count users in each cell before fitting.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> Why can the effect of a notification on one user never be observed directly?</summary>

The effect is the user's spend when notified minus their spend when not notified. The user is either notified or not, so one of the two outcomes is never realised. Only averages over groups can be estimated, and only under assumptions that make the groups comparable.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> In the worked example the naive gap is 4.4 and the true effect is 2.0. Where does the extra 2.4 come from?</summary>

From confounding by engagement. Notified users are 80 per cent engaged, un-notified users 20 per cent, and engagement is worth 4. So the head start is 4 x (0.8 - 0.2) = 2.4, and 2.0 + 2.4 = 4.4.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> A randomised experiment shows treatment raises spend by 2. An analyst then restricts to users who left a review and finds the effect is negative. Explain.</summary>

Reviewing is a collider if it is caused by both treatment and spend. Conditioning on it compares treated and untreated users who both reviewed, and among them a treated user can review with less spend than an untreated user. That selection builds a negative association. Block 3 produced -0.257 this way from a true effect of 2.0. The full randomised sample gives the right answer.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Why did adjusting for `quality` change the point estimate from 2.017 to 2.004 and the standard error from 0.015 to 0.013, but not much more?</summary>

`quality` is a cause of spend only. It does not steer treatment, so it is not a confounder and removing it cannot remove bias. It does absorb some of the outcome noise, so the estimate becomes slightly more precise. Both changes are small because its share of the variance is modest.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> In block 1 the interval for the adjusted effect, 1.933 to 2.002, nearly misses the truth. Should the analyst distrust adjustment?</summary>

No. A 95 per cent interval misses the truth in 5 per cent of samples, and a sample can land near the edge without anything being wrong. To check calibration, repeat the simulation with many seeds and count how often the interval contains 2.0. Distrust adjustment when the assumptions fail (an unmeasured confounder, a collider in the control set), not when one sample is a little low.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> Using the formula for the naive gap, show that confounding vanishes when engagement does not affect spend, even if notification still depends on engagement.</summary>

The naive gap equals the true effect plus gamma x (P(engaged | notified) - P(engaged | not notified)), where gamma is the spend that engagement adds. If gamma is 0 the second term is 0 whatever the two shares are. This is the lab's second experiment. Confounding needs both arrows: one into the treatment and one into the outcome.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Hernán MA and Robins JM, *Causal Inference: What If*. Free to download from the author's page; the edition read here is dated 19 August 2026 and its contents list was read. Chapters 1 to 3 (definition, randomised experiments, observational studies), 6 to 8 (diagrams, confounding, selection bias) and 12 to 16 (weighting, standardisation, propensity scores, instruments) are the spine of this and the next three chapters. Licence: all rights reserved, free to read online.
- Facure Alves M, *Causal Inference for the Brave and True*, an open online book with Python code. Its table of contents (introduction, randomised experiments, graphical causal models, linear regression, instrumental variables, matching, propensity scores, doubly robust estimation, difference-in-differences, regression discontinuity, then heterogeneous effects and debiased machine learning) was read; it carries a 2023 copyright and no stated licence. Used here as the reading order for Python practice.
- [DoWhy documentation](https://www.pywhy.org/dowhy/): the model, identify, estimate and refute workflow. Version 0.8 was run.
- [statsmodels formula API](https://www.statsmodels.org/stable/example_formulas.html): the `ols` regressions used in the blocks. Version 0.15.0 was run.

## Check yourself

- I can write an average treatment effect as a difference in potential outcomes and say which half of it is never observed.
- I can split a naive difference into the true effect and a head start, and compute the head start for a simple case.
- I can read a diagram, find the backdoor paths and say which variables block them.
- I can explain why adjusting for a mediator or a collider makes an estimate worse, with a number from this chapter.
- I can state exchangeability, positivity and consistency, and give a case where each fails.

## Where to go next

Next chapter: [randomised experiments and adjustment](/docs/theory/causal/experiments-and-adjustment), where we estimate the same effect with matching, propensity scores and inverse-probability weighting, and measure which one wins. A related chapter: [experimentation and A/B testing](/docs/mlops/platform/experimentation-and-ab-testing), which is the randomised design in production.
