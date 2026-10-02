---
id: senior-case-ml-platform
title: "System Design Case: An ML Platform for Many Teams"
sidebar_label: "6 · ML platform"
sidebar_position: 6
slug: /senior/design-ml-platform
description: "Design a shared ML platform for eight teams: feature, training, registry and serving layers, multi-tenant GPU scheduling with quotas and borrowing, point-in-time feature joins, promotion gates, cost attribution and golden paths, with seeded simulations."
tags: [system-design, ml-platform, gpu-scheduling, feature-store, model-registry, cost-attribution, golden-path]
---

import Infographic from '@site/src/components/Infographic';
import PlatformCostLab from '@site/src/components/viz/PlatformCostLab';

**In one line.** An ML platform is a small set of shared layers (features, training, registry, serving) plus a supported route through them, and the hard design questions are not technical fashion but who gets the GPUs when everyone wants them, who pays for the idle ones, and how a team ships without relearning every lesson the last team paid for.

:::note Not from a lecture
Written for this site from the sources under Further reading. The brief is an exercise, not a company. The simulations use synthetic jobs and a placeholder price of 1.0 per GPU-hour; the numbers they print show shapes, not benchmarks.
:::

## The idea in plain words

**The brief.** A company has eight ML teams and about 40 engineers, with a platform team of four. Teams train models in bursts, run batch inference and serve online models; they share one 64-GPU training cluster. Today each team keeps its own scripts, features are computed twice (once for training, once for serving), a model reaches production when someone copies a file, and nobody can say what a team's experiments cost. The goal: shared building blocks, fair access to GPUs, safe promotion, visible cost, and a route a new team can follow in its first week.

The warning to start from is old and still true. Sculley and colleagues (NIPS 2015, "Hidden Technical Debt in Machine Learning Systems") argued that it is "common to incur massive ongoing maintenance costs in real-world ML systems", pointing to entangled components, undeclared dependencies, feedback loops and configuration debt. A platform exists to pay that debt once instead of eight times.

<Infographic src="/img/senior/ml-platform-layers.svg" alt="Four shared layers (features, training, registry, serving) with a cost-attribution bar beneath them and a golden-path group with a project template, a promotion gate and off-path teams." caption="The platform on one page. The gate numbers come from block 5." />

| Layer | Shared by | Stays with the team |
| --- | --- | --- |
| Features | definitions, offline and online stores, point-in-time joins | which features a model uses |
| Training | the GPU pool, quotas, scheduler | model code, hyperparameters |
| Registry | versions, aliases, lineage, promotion gate | evaluation sets and the metric that matters |
| Serving | endpoints, release, rollback | latency budget and the thresholds |

## How it works

**Sizing (block 2, last line).** Eight services each making 500 requests a second and reading 40 features per request is 160,000 feature reads a second, which is why the online store is a key-value service sized by lookups per second, not a database sized by rows. GPU sizing is the central exercise and is done with demand traces, below.

**Data flow for one model.**

1. A feature definition is written once and materialised to the offline store (for training) and the online store (for serving).
2. A training job is submitted from the project template; the scheduler admits it against the team's quota, lending idle capacity if the team is over it.
3. The job logs a run with code reference and data snapshot.
4. The registry stores the version and a **promotion gate** checks it before an alias such as `champion` moves.
5. Serving loads the aliased artefact; rollback is moving the alias back.

### Decision 1: how to share GPUs

Block 1 simulates a week on 64 GPUs with four teams (search quota 24, ads 16, vision 16, nlp 8), 927 jobs, bursty arrivals and an offered load of 0.78. Ads and nlp ask for more than their quotas; search and vision for less.

| Policy | Idea | Utilisation | Cost |
| --- | --- | --- | --- |
| Static quotas | each team stays inside its own share | 0.651 | ads waits a median 24.2 hours and nlp 50.2 while search and vision capacity sits idle |
| Shared FIFO pool | one queue, anyone runs anywhere | 0.774 | no guarantee: search's 95th-percentile wait is 4.8 hours, no better than under static quotas |
| Quota with borrowing and reclaim | use idle capacity, give it back by preempting the youngest borrower | 0.790 | 156 preemptions and 163 wasted GPU-hours (1.5% of 10,752) buy search a 95th-percentile wait of 2.0 hours |

Utilisation counts restarted work as busy, and waits are measured to the first start. The result is mixed on purpose: borrowing helps the owner who is entitled to capacity but also moves delay to the borrowers (ads' 95th-percentile wait is 7.4 hours against 5.6 in the plain pool). The trace is synthetic, so treat the ordering, not the digits, as the lesson.

The Kueue documentation shows the same vocabulary in a real scheduler. A `ClusterQueue` has a `nominalQuota` per resource; queues in a cohort can borrow each other's unused quota up to a `borrowingLimit`, owners can cap what they lend with a `lendingLimit`, and `reclaimWithinCohort` and `withinClusterQueue` set who may preempt whom. Its fair-sharing mode gives each queue a numeric share, admits from the queue with the lowest share first and preempts from the highest first. GPUs are not the only resource, which is where **dominant resource fairness** matters. Block 3 runs the algorithm from Ghodsi et al. (NSDI 2011) on the paper's own example, nine CPUs and 18 GB shared by a user whose tasks need 1 CPU and 4 GB and a user whose tasks need 3 CPUs and 1 GB. Each task of the first user takes 1/9 of the CPUs but 2/9 of the memory, so memory is its dominant resource; the second user's is CPU. DRF equalises dominant shares and gives three tasks to the first and two to the second, each with a dominant share of two thirds, exactly the paper's allocation. The second example adds GPUs: once the GPUs are gone, the team that needs none keeps receiving CPU and memory, which the paper describes as the intended behaviour.

### Decision 2: the feature layer and point-in-time correctness

A feature store earns its place by removing two bugs: computing a feature differently in training and serving, and letting training rows see the future. Feast's documentation describes point-in-time joins as reproducing "the state of features at a specific point in the past", and warns that by default only the event timestamp is constrained, so "a value that was backfilled or corrected after an entity dataframe timestamp can still be returned"; its `filter_by_created_timestamp` option adds the creation-time condition.

Block 4 builds the failure with pandas `merge_asof` on 14,251 synthetic churn events. A **latest-value join** attaches the final support-ticket count, which includes tickets raised after the customer had already decided to leave. The model trained on it reports AUC 0.670 offline; trained on the correct point-in-time join the honest figure is 0.597; and the leaky model scored on values as they would be known at prediction time reaches 0.616. The offline number overstated what the model could do by 0.054. The data are synthetic and one seed, so the gap is an illustration, not a general size.

### Decision 3: registry, promotion gate and golden path

MLflow's registry documentation defines a registered model with versions, tags, lineage and a mutable named alias pointing at a version, such as a `champion` alias for production. Block 5 builds the same idea in plain Python and adds the part that is a design choice: **a gate before the alias moves**. Of five versions, two are promoted and three blocked, each for a different reason: incomplete lineage, a regression on the `new_users` slice though the headline metric rose, and a 60 ms p95 against a 50 ms budget. Rollback is one alias move.

A **golden path** is the supported route. Spotify's engineering post defines it as the "opinionated and supported" path to build something, and is explicit that engineers may leave it but then do not get the same support. The same block validates a project against a template: it reports a missing `eval_set` and that eight GPUs need an approval above four. Make the path the easiest option, not a mandate.

| Design choice | Option | Trade-off |
| --- | --- | --- |
| Platform stance | mandate one stack | consistent, but slows the team with a real exception |
| | golden path, others unsupported | fast for most, an honest cost for the rest |
| Promotion | manual review only | slow, uneven |
| | automated gate plus review of exceptions | cheap and auditable; gates need maintenance |
| Capacity | partitions per team | simple, wasteful |
| | pool with quotas and borrowing | efficient, needs preemption-tolerant jobs (checkpoints) |

### Decision 4: cost attribution

Pooling saves hardware, and block 2 measures how much on a synthetic week of hourly demand. Sizing each team's partition for its own peak needs 104 GPUs at utilisation 0.336; one shared pool sized to the peak of the sum needs 56 at 0.623, a saving of 48 GPUs. Peaks are single-hour maxima driven by rare sweeps, so a real team might size to a lower percentile and the saving would shrink.

Then comes the question of who pays. On a 64-GPU cluster the week costs 10,752 units and 4,887 GPU-hours (45.5%) are idle. OpenCost's specification defines idle cost as cluster asset cost minus workload cost and lists three common ways to distribute shared cost: uniform, proportional to consumption, or by a custom metric. The block prints four rules, and they disagree: search pays 2,249 for usage alone, 3,471 with idle split equally, 4,123 with idle by usage and 4,082 by quota. None is neutral. Equal splitting taxes the small team, usage-proportional splitting taxes the busy team most in absolute terms, and quota-proportional splitting charges for the capacity promised rather than used. Pick one deliberately, publish it, and keep quota and price visible. The lab replays the same week.

<PlatformCostLab />

<Infographic src="/img/senior/ml-platform-pooling-and-cost.svg" alt="Tables of demand and pooling savings, three scheduling policies, four idle-cost allocation methods and the effect of a leaky training join." caption="Blocks 1, 2 and 4 together: pooling, the scheduler, the bills and the leak." />

The lab's defaults (64 GPUs, price 1.0, idle by usage) give search 4,123, ads 2,913, vision 1,921 and nlp 1,795, matching the printed table. Shrink the cluster below 56 and unmet demand appears.

## A real system that works this way

Uber's engineering post introducing Michelangelo (5 September 2017) describes a platform built around six steps, "manage data, train models, evaluate models, deploy models, make predictions, and monitor predictions", with a shared Feature Store holding about 10,000 features, dozens of teams using it, and the highest-traffic models serving more than 250,000 predictions a second at a P95 latency under 5 ms (under 10 ms with features from Cassandra). Those are 2017 figures, shown for the shape rather than the size. Google's MLOps guide (last reviewed 28 August 2024) describes the maturity ladder the layers climb: level 0 manual, level 1 automated pipelines with a feature store, metadata and triggers, level 2 CI/CD with a model registry. Spotify's golden-path post (17 August 2020) supplies the stance. None of these is a claim about how any specific team runs today.

## Code you can run

All blocks are CPU only and seeded. The scheduler and demand traces are synthetic and the price is a placeholder.

#### 1. Scheduling policies

```python
import numpy as np

GPUS = 64
QUOTA = {"search": 24, "ads": 16, "vision": 16, "nlp": 8}
LOAD = {"search": 0.6, "ads": 1.3, "vision": 0.5, "nlp": 1.2}
TICK_MIN, TICKS = 10, 6 * 24 * 7
SIZES = np.array([1, 2, 4, 8])


def make_jobs(seed=0):
    rng = np.random.default_rng(seed)
    jobs = []
    for team, load in LOAD.items():
        mean_gpus = SIZES.mean() * 0.8
        mean_ticks = 18
        rate = load * QUOTA[team] / (mean_gpus * mean_ticks)
        burst = np.where(np.arange(TICKS) % (6 * 24) < 6 * 8, 2.0, 0.5)
        for t in range(TICKS):
            for _ in range(rng.poisson(rate * burst[t])):
                gpus = int(rng.choice(SIZES, p=[0.3, 0.3, 0.25, 0.15]))
                ticks = max(1, int(rng.lognormal(np.log(14), 0.7)))
                jobs.append(dict(team=team, arrive=t, gpus=gpus, ticks=ticks))
    jobs.sort(key=lambda j: (j["arrive"], j["team"]))
    return jobs


def simulate(jobs, policy):
    queue, running = [], []
    used_by = {t: 0 for t in QUOTA}
    waits, busy, wasted, preempted = {t: [] for t in QUOTA}, 0, 0, 0
    pending = [dict(j, left=j["ticks"], started=None) for j in jobs]
    i = 0
    for now in range(TICKS):
        while i < len(pending) and pending[i]["arrive"] <= now:
            queue.append(pending[i])
            i += 1
        for j in running[:]:
            j["left"] -= 1
            if j["left"] == 0:
                running.remove(j)
                used_by[j["team"]] -= j["gpus"]
        free = GPUS - sum(used_by.values())

        def start(j):
            nonlocal free
            j["started"] = now
            running.append(j)
            used_by[j["team"]] += j["gpus"]
            free -= j["gpus"]
            if not j.get("counted"):
                j["counted"] = True
                waits[j["team"]].append((now - j["arrive"]) * TICK_MIN / 60)
            queue.remove(j)

        if policy == "pool":
            while queue and queue[0]["gpus"] <= free:
                start(queue[0])
        else:
            for team in QUOTA:
                mine = [j for j in queue if j["team"] == team]
                while mine:
                    head = mine[0]
                    within = used_by[team] + head["gpus"] <= QUOTA[team]
                    if policy == "static" and not within:
                        break
                    if head["gpus"] > free and policy == "borrow" and within:
                        victims = sorted((r for r in running if used_by[r["team"]] > QUOTA[r["team"]] and r["team"] != team),
                                         key=lambda r: -r["started"])
                        for v in victims:
                            if head["gpus"] <= free:
                                break
                            running.remove(v)
                            used_by[v["team"]] -= v["gpus"]
                            free += v["gpus"]
                            wasted += v["gpus"] * (now - v["started"])
                            preempted += 1
                            v["left"], v["started"] = v["ticks"], None
                            queue.insert(0, v)
                    if head["gpus"] > free:
                        break
                    start(head)
                    mine.pop(0)
        busy += sum(used_by.values())
    return dict(util=busy / (GPUS * TICKS), waits=waits, wasted=wasted * TICK_MIN / 60, preempted=preempted)


jobs = make_jobs()
offered = sum(j["gpus"] * j["ticks"] for j in jobs) / (GPUS * TICKS)
print(f"{len(jobs)} jobs over 7 days, offered load {offered:.2f} of a {GPUS}-GPU cluster; quotas {QUOTA}")
print("\npolicy    utilisation   preemptions   wasted GPU-hours   " + "   ".join(f"{t} p50/p95 wait h" for t in QUOTA))
for policy in ("static", "pool", "borrow"):
    r = simulate(jobs, policy)
    cells = "   ".join(f"{np.percentile(r['waits'][t], 50):6.1f}/{np.percentile(r['waits'][t], 95):6.1f}      " for t in QUOTA)
    print(f"{policy:8s}  {r['util']:11.3f}   {r['preempted']:11d}   {r['wasted']:16.0f}   {cells}")
```

#### 2. Pooling, sizing and cost attribution

```python
import numpy as np

HOURS = 168
QUOTA = {"search": 24, "ads": 16, "vision": 16, "nlp": 8}
rng = np.random.default_rng(3)
hour = np.arange(HOURS)
phase = {"search": 0, "ads": 6, "vision": 12, "nlp": 18}
base = {"search": 13, "ads": 9, "vision": 6, "nlp": 5}
demand = {}
for team in QUOTA:
    daily = 1 + 0.6 * np.sin(2 * np.pi * (hour - phase[team]) / 24)
    sweeps = (rng.random(HOURS) < 0.04) * rng.integers(6, 16, HOURS)
    demand[team] = np.maximum(0, np.round(base[team] * daily + sweeps + rng.normal(0, 1.2, HOURS))).astype(int)

total = sum(demand.values())
peaks = {t: int(d.max()) for t, d in demand.items()}
print("team      quota   peak   mean demand   GPU-hours used")
for t, d in demand.items():
    print(f"{t:8s}  {QUOTA[t]:5d}  {peaks[t]:5d}  {d.mean():12.1f}  {d.sum():14d}")
sum_peaks, pooled_peak = sum(peaks.values()), int(total.max())
print(f"\nstatic partitions sized to each team's peak: {sum_peaks} GPUs, utilisation {total.sum() / (sum_peaks * HOURS):.3f}")
print(f"one shared pool sized to the peak of the sum: {pooled_peak} GPUs, utilisation {total.sum() / (pooled_peak * HOURS):.3f}")
print(f"GPUs saved by pooling: {sum_peaks - pooled_peak}")

cluster, price = 64, 1.0
cost = cluster * HOURS * price
used = {t: int(d.sum()) for t, d in demand.items()}
idle = cluster * HOURS - sum(used.values())
print(f"\ncluster {cluster} GPUs, week cost {cost:,.0f} at {price} per GPU-hour; idle {idle:,} GPU-hours ({idle / (cluster * HOURS):.1%})")
methods = {
    "usage only": {t: used[t] * price for t in QUOTA},
    "idle uniform": {t: (used[t] + idle / len(QUOTA)) * price for t in QUOTA},
    "idle by usage": {t: used[t] * cluster * HOURS / sum(used.values()) * price for t in QUOTA},
    "idle by quota": {t: (used[t] + idle * QUOTA[t] / sum(QUOTA.values())) * price for t in QUOTA},
}
print("method          " + "".join(f"{t:>10s}" for t in QUOTA) + "   unallocated")
for name, bill in methods.items():
    print(f"{name:15s} " + "".join(f"{bill[t]:10,.0f}" for t in QUOTA) + f"   {cost - sum(bill.values()):10,.0f}")

services, qps, features = 8, 500, 40
print(f"\nonline serving: {services} services x {qps} requests/s x {features} features = {services * qps * features:,} feature reads per second")
```

#### 3. Dominant resource fairness

```python
def drf(capacity, demands):
    used = [0.0] * len(capacity)
    tasks = {u: 0 for u in demands}
    shares = {u: 0.0 for u in demands}
    blocked = set()
    while len(blocked) < len(demands):
        user = min((u for u in demands if u not in blocked), key=lambda u: shares[u])
        need = demands[user]
        if all(used[i] + need[i] <= capacity[i] + 1e-9 for i in range(len(capacity))):
            used = [used[i] + need[i] for i in range(len(capacity))]
            tasks[user] += 1
            shares[user] = max(tasks[user] * need[i] / capacity[i] for i in range(len(capacity)))
        else:
            blocked.add(user)
    return tasks, shares, used


tasks, shares, used = drf([9, 18], {"A": [1, 4], "B": [3, 1]})
print("two resources (CPU, GB), capacity 9 and 18")
for u in tasks:
    print(f"  user {u}: {tasks[u]} tasks, dominant share {shares[u]:.3f}")
print(f"  used {used[0]:.0f} CPU and {used[1]:.0f} GB")

capacity = [64, 1024, 8192]
tasks, shares, used = drf(capacity, {"training": [8, 64, 256], "etl": [0, 96, 1024], "serving": [1, 4, 32]})
print("\nthree resources (GPU, CPU cores, GB), capacity 64, 1024, 8192")
for u in tasks:
    print(f"  {u:8s}: {tasks[u]:3d} tasks, dominant share {shares[u]:.3f}")
print("  used " + ", ".join(f"{used[i]:.0f} of {capacity[i]}" for i in range(3)))
```

The first example is the paper's: user A gets 3 CPUs and 12 GB, user B gets 6 CPUs and 2 GB, so 9 CPUs and 14 GB are used. In the second, the GPUs run out first and the `etl` team, which needs none, ends with the largest dominant share, 0.750.

#### 4. Point-in-time joins

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
n_users, days = 3000, 120
users = np.arange(n_users)
risk = rng.normal(0, 1, n_users)

rows = []
for d in range(days):
    for u in users[rng.random(n_users) < 0.04]:
        rows.append((u, d))
events = pd.DataFrame(rows, columns=["user", "day"])
events["churn"] = (rng.random(len(events)) < 1 / (1 + np.exp(-(risk[events.user] - 1.5)))).astype(int)

feature_rows = []
for u in users:
    tickets = 0
    churn_day = events.loc[(events.user == u) & (events.churn == 1), "day"].min()
    for d in range(days):
        tickets += rng.poisson(0.05 + 0.04 * max(risk[u], 0) + (0.8 if d >= churn_day else 0))
        feature_rows.append((u, d, tickets))
features = pd.DataFrame(feature_rows, columns=["user", "day", "tickets"])

events = events.sort_values("day").reset_index(drop=True)
features = features.sort_values("day")
correct = pd.merge_asof(events, features, on="day", by="user", allow_exact_matches=False)
latest = features.groupby("user").tickets.last().rename("tickets_latest")
leaky = events.join(latest, on="user")

cols = {"point-in-time join": (correct, "tickets"), "latest value join": (leaky, "tickets_latest")}
print(f"{len(events)} labelled events, churn rate {events.churn.mean():.3f}, {int((events.day >= 80).sum())} in the test period")
print("\njoin used for training         AUC offline (same join)   AUC when served with point-in-time values")
test = correct.day >= 80
for name, (frame, col) in cols.items():
    fit = frame.day < 80
    model = HistGradientBoostingClassifier(random_state=0).fit(frame.loc[fit, [col]], frame.loc[fit, "churn"])
    offline = roc_auc_score(frame.loc[~fit, "churn"], model.predict_proba(frame.loc[~fit, [col]])[:, 1])
    served = roc_auc_score(correct.loc[test, "churn"], model.predict_proba(correct.loc[test, ["tickets"]].set_axis([col], axis=1))[:, 1])
    print(f"{name:28s}  {offline:22.3f}   {served:38.3f}")
```

#### 5. Promotion gate and project template

The rollback line sets the alias by hand to show the operation; a real registry would audit it.

```python
from dataclasses import dataclass, field

REQUIRED_LINEAGE = ("run_id", "code_ref", "data_snapshot")
MARGIN, SLICE_TOLERANCE, LATENCY_BUDGET_MS = 0.002, 0.01, 50


@dataclass
class Version:
    number: int
    metric: float
    slices: dict
    lineage: dict
    card: bool
    p95_ms: float


@dataclass
class Registry:
    versions: list = field(default_factory=list)
    aliases: dict = field(default_factory=dict)

    def register(self, **kw):
        v = Version(len(self.versions) + 1, **kw)
        self.versions.append(v)
        return v

    def checks(self, v):
        champion = self.versions[self.aliases["champion"] - 1] if "champion" in self.aliases else None
        failed = []
        if any(k not in v.lineage for k in REQUIRED_LINEAGE):
            failed.append("lineage incomplete")
        if not v.card:
            failed.append("no model card")
        if v.p95_ms > LATENCY_BUDGET_MS:
            failed.append(f"p95 {v.p95_ms} ms over the {LATENCY_BUDGET_MS} ms budget")
        if champion:
            if v.metric < champion.metric + MARGIN:
                failed.append("does not beat the champion by the margin")
            worse = [s for s, m in v.slices.items() if m < champion.slices[s] - SLICE_TOLERANCE]
            if worse:
                failed.append("slice regression: " + ", ".join(worse))
        return failed

    def promote(self, number):
        failed = self.checks(self.versions[number - 1])
        if not failed:
            self.aliases["champion"] = number
        return failed


lineage = dict(run_id="r-101", code_ref="rev-a", data_snapshot="snap-1")
reg = Registry()
reg.register(metric=0.810, slices=dict(new_users=0.78, eu=0.80), lineage=lineage, card=True, p95_ms=31)
reg.register(metric=0.830, slices=dict(new_users=0.80, eu=0.82), lineage=dict(run_id="r-102"), card=True, p95_ms=33)
reg.register(metric=0.840, slices=dict(new_users=0.70, eu=0.88), lineage=lineage, card=True, p95_ms=35)
reg.register(metric=0.835, slices=dict(new_users=0.80, eu=0.83), lineage=lineage, card=True, p95_ms=60)
reg.register(metric=0.832, slices=dict(new_users=0.79, eu=0.82), lineage=lineage, card=True, p95_ms=36)

for n in range(1, 6):
    failed = reg.promote(n)
    print(f"version {n}: {'promoted, champion is now ' + str(n) if not failed else 'blocked: ' + '; '.join(failed)}")
print("aliases:", reg.aliases)
reg.aliases["champion"] = 1
print("rollback is one alias move, champion is version", reg.aliases["champion"])

TEMPLATE = dict(required=("owner", "cost_centre", "eval_set", "registry_name"), gpu_without_approval=4)
project = dict(owner="team-ads", cost_centre="cc-12", registry_name="ads-ctr", gpus=8, approval=None)
problems = [f"missing {k}" for k in TEMPLATE["required"] if k not in project]
if project["gpus"] > TEMPLATE["gpu_without_approval"] and not project["approval"]:
    problems.append(f"{project['gpus']} GPUs need an approval above {TEMPLATE['gpu_without_approval']}")
print("project check:", problems)
```

## Designing with it

**Failure modes and mitigations**

| Failure | Cause | Mitigation |
| --- | --- | --- |
| One team starves the rest | no quotas, or unlimited borrowing | nominal quotas, a borrowing limit, fair-share admission |
| Preemption destroys long jobs | no checkpoints | require checkpointing on the golden path; preempt the youngest first |
| Offline metric does not survive production | leaky join or training-serving skew | point-in-time joins, one feature definition for both stores |
| A promoted model regresses a segment | headline metric only | slice metrics in the gate, shadow release, one-move rollback |
| The platform becomes a bottleneck | every request goes through the platform team | self-service templates, a gate that runs without a person |
| Nobody trusts the bill | allocation rule hidden or unfair | publish the rule, show usage and idle separately |

**Evaluation and rollout of the platform itself.** Treat the platform as a product with metrics: time from a new team's first commit to a first promoted model, GPU utilisation, 95th-percentile queue wait per team, share of models promoted through the gate, blocked promotions by reason, and cost per team against its quota. Roll out one layer at a time with one friendly team, and measure before and after on the numbers above. Link the practice to the site's [CI/CD and data versioning](/docs/theory/seml/cicd-and-data-versioning) and [event-driven and MLOps](/docs/theory/seml/event-driven-and-mlops) chapters.

**Cost as a formula.** `team bill = GPU-hours used x price + idle cost x share`, where `idle cost = (cluster GPUs x hours - GPU-hours used) x price` and `share` is the published rule (equal, by usage, or by quota). Add serving, storage and platform-team cost as separate lines with their own rules.

**What to build first.** A project template with required labels, a shared pool with per-team quotas and a visible usage report, and a registry with a promotion gate that checks lineage and one slice. Add the feature store when two teams actually share a feature, and borrowing with preemption when the waits in the report justify it.

## Where this stands in 2026

:::info Industry view

- **Quota, borrowing and fair share are standard scheduler vocabulary.** Kueue exposes nominal quota, cohorts, borrowing and lending limits, preemption policies and a fair-sharing mode in its documentation read for this chapter.
- **Cost allocation is a specification, not a guess.** OpenCost documents allocation by `max(request, usage)`, idle cost as a defined quantity and named options for shared cost.
- **Aliases are the registry pointer in the current docs.** The MLflow documentation read here describes an alias as a mutable named reference to a version and uses a `champion` alias as its example; it also still mentions stages, so check which your installed version supports.
- **Platform stance has settled on paved roads.** Spotify's post is from 2020 and the idea has stayed, because it gives teams speed without forbidding exceptions.
- **Versions.** Code ran on Python 3.14 with `numpy` 2.5.3, `pandas` 3.0.6 and `scikit-learn` 1.9.1; sources were opened on 2 October 2026.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Eight teams each want to size their own partition for their peak. What does pooling change in the numbers?</summary>

In block 2 the separate peaks sum to 104 GPUs at utilisation 0.336, while the peak of the pooled demand is 56 GPUs at 0.623, a saving of 48. The saving comes from teams peaking at different hours, and it shrinks if peaks coincide or if you size to a percentile instead of the maximum.

</details>

<details>
<summary><strong>Q2.</strong> Quota with borrowing raised utilisation only from 0.774 to 0.790. Why use it?</summary>

It buys a guarantee: search's 95th-percentile wait fell from 4.8 to 2.0 hours, at the cost of 156 preemptions, 163 wasted GPU-hours and a worse tail for the borrowers. Use it when teams are entitled to capacity they must be able to reclaim, and require checkpoints.

</details>

<details>
<summary><strong>Q3.</strong> Why does dominant resource fairness equalise dominant shares rather than split each resource equally?</summary>

Tasks need different mixes of resources, so an equal split of one resource can starve a team that needs another. In the paper's example, user A is limited by memory and user B by CPU, and equalising each user's largest share (two thirds each) gives them 3 and 2 tasks.

</details>

<details>
<summary><strong>Q4.</strong> A model shows AUC 0.670 offline and disappoints in production. What do you check first?</summary>

The training join. A latest-value join lets a row see information from after the prediction time; in block 4 the same model scored 0.616 on honest values and an honest pipeline reported 0.597. Use point-in-time joins and constrain creation time too, as Feast's `filter_by_created_timestamp` does.

</details>

<details>
<summary><strong>Q5.</strong> The gate blocked version 3 although its headline metric was the highest. Defend the gate.</summary>

Its `new_users` slice fell from 0.78 to 0.70, a regression worth more than the 0.01 tolerance, and a headline metric hides segment harm. The gate makes the trade-off visible and lets a person grant an exception on purpose.

</details>

<details>
<summary><strong>Q6.</strong> Three teams complain about the bill. How do you choose an idle-cost rule?</summary>

Show them the options on the same week: search pays 3,471, 4,123 or 4,082 depending on the rule. Choose by what you want to encourage (equal split taxes small teams, usage-proportional charges busy teams, quota-proportional charges for reservation), publish it and show usage and idle as separate lines so the incentive is visible.

</details>

## Further reading

- Sculley et al., [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (NIPS 2015).
- Uber Engineering, [Meet Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/) (5 September 2017).
- Google Cloud, [MLOps: continuous delivery and automation pipelines in machine learning](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).
- Ghodsi et al., [Dominant Resource Fairness](https://www.usenix.org/legacy/event/nsdi11/tech/full_papers/Ghodsi.pdf) (NSDI 2011).
- Kueue, [ClusterQueue](https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/) and [preemption and fair sharing](https://kueue.sigs.k8s.io/docs/concepts/preemption/).
- Feast, [Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins).
- MLflow, [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/).
- OpenCost, [Specification](https://opencost.io/docs/specification).
- Spotify Engineering, [How we use golden paths to solve fragmentation](https://engineering.atspotify.com/2020/08/how-we-use-golden-paths-to-solve-fragmentation-in-our-software-ecosystem) (17 August 2020).
- On this site: [serving and release strategies](/docs/theory/seml/serving-and-release-strategies), [containers and orchestration](/docs/theory/seml/containers-and-orchestration), [testing ML systems](/docs/theory/seml/testing-ml-systems).

## Check yourself

- I can explain why pooled GPU capacity needs fewer GPUs than partitions, and what could make that saving disappear.
- I can compare static quotas, a shared pool and quota with borrowing by utilisation, waits and wasted work.
- I can explain dominant resource fairness with a two-resource example.
- I can build a point-in-time join and say why a latest-value join inflates an offline metric.
- I can design a promotion gate and say what each check protects against.
- I can choose and defend an idle-cost allocation rule, and say who it burdens.
