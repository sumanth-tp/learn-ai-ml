---
id: gov-privacy
title: "Privacy, PII and Differential Privacy"
sidebar_label: "Privacy, PII and differential privacy"
sidebar_position: 3
slug: /governance/privacy-pii-and-differential-privacy
description: "Detect and redact PII, see why anonymised tables still leak, learn the Laplace mechanism and the privacy budget, and measure memorisation with a membership-inference attack against plain and DP-SGD training."
tags: [privacy, pii, redaction, k-anonymity, differential-privacy, laplace-mechanism, dp-sgd, membership-inference]
---

import Infographic from '@site/src/components/Infographic';
import PrivacyBudgetLab from '@site/src/components/viz/PrivacyBudgetLab';

**In one line.** Privacy work for AI systems has four layers: find and remove identifiers, test whether the "anonymised" data still points at people, bound what any single person's data can change in an output, and measure whether the model memorised its training rows.

:::note Not from a lecture
Written for this site from the sources under Further reading. Every dataset below is synthetic, built for the chapter, and this is engineering education, not legal advice about any privacy law.
:::

## The idea in plain words

Personal data leaks from AI systems in three ordinary ways. It is typed into a prompt or stored in a log. It is released in a table that was "anonymised" by deleting the name column. Or it is memorised by a model and later recited. Each has a different tool.

| Leak | Typical example | Tool in this chapter |
| --- | --- | --- |
| Identifiers in text | an email, a card number or a name in a support transcript | detection and redaction, with honest recall figures |
| Linkable records | a table with zip code, age and sex but no name | k-anonymity checks, and why they are not enough |
| Query answers | a statistic published from a sensitive database | differential privacy: the Laplace mechanism and a privacy budget |
| The model itself | training rows reproduced or confirmed by the model | membership-inference tests, DP-SGD, deduplication |

Two famous results frame the problem. Narayanan and Shmatikov showed that an adversary who knows only a little about a subscriber can identify that subscriber's record in the Netflix Prize dataset, which Netflix had released as anonymous (about 500,000 subscriber records, per the paper's abstract). Removing names did not anonymise high-dimensional data. And Carlini and colleagues showed that querying a language model strategically recovers hundreds of verbatim training sequences from GPT-2, including names, phone numbers and email addresses, and that larger models were more vulnerable than smaller ones.

<Infographic src="/img/gov/privacy-pii-and-differential-privacy-anonymisation.svg" alt="Detector precision and recall tables for regex PII detection and for k-anonymity generalisation, with notes on masking, homogeneity and linkage." caption="Blocks 1 and 2: regex detectors are exact on structured identifiers and blind to plain names; generalisation trades rows for anonymity, and k-anonymity still leaks attributes." />

## How it works

### Detect and redact

Structured identifiers (emails, phone numbers, card numbers) follow patterns, so rules work. Names, addresses and free-text identifiers do not follow patterns, so tools pair rules with trained recognisers. Microsoft's Presidio, for example, describes its analyzer as using named entity recognition, regular expressions, rule-based logic and checksums, and its anonymizer as applying configurable operators. Its documentation is blunt about the limit: because detection is automated, there is no guarantee it finds all sensitive information, so other protections should be added.

**Masking** replaces a value by its type. **Keyed pseudonymisation** replaces it by a keyed hash: the same value always gives the same token, so joins still work, but anyone holding the key can test guesses. Keep the key out of the data store and rotate it.

### Why removing names is not anonymisation

Quasi-identifiers such as zip code, age and sex are individually harmless and jointly unique. A table is **k-anonymous** over a set of quasi-identifiers when every combination of their values is shared by at least k rows (Sweeney's model, in a 2002 paper in the International Journal on Uncertainty, Fuzziness and Knowledge-based Systems). You raise k by **generalising** (zip to its first digits, age to bands) and by **suppressing** rows in groups that stay too small. The checks below show two pitfalls. Generalising enough to be safe destroys detail, and even a k-anonymous table leaks when everyone in a group shares the sensitive value, the **homogeneity** problem that motivated stronger definitions.

### Differential privacy

Differential privacy changes the question from "is this table safe?" to "how much can any one person change what I publish?". A randomised mechanism is **epsilon-differentially private** if, for any two datasets that differ in one person's data, the probability of any output differs by at most a factor of e to the epsilon (Dwork and Roth's monograph states the definition and the basic results). Small epsilon means strong privacy.

- **The Laplace mechanism** answers a numeric query by adding noise drawn from a Laplace distribution with scale equal to the query's **sensitivity** divided by epsilon. A count has sensitivity 1, because one person changes it by at most 1.
- **Composition.** When several private mechanisms run on the same data, their epsilons add up (the basic composition theorem). This is why a **privacy budget** exists: you decide the total epsilon in advance and spend it across questions.
- **DP-SGD** (Abadi and colleagues, 2016) makes training private. Each example's gradient is clipped to a maximum norm, Gaussian noise is added to the sum, and a **privacy accountant** tracks the epsilon spent over many steps. Opacus provides such an accountant for PyTorch.

NIST published SP 800-226, "Guidelines for Evaluating Differential Privacy Guarantees", in March 2025. It describes a "differential privacy pyramid" for evaluating a deployment and warns about **privacy hazards**, common implementation mistakes made when turning the mathematics into software.

### Memorisation and membership inference

A **membership-inference attack** asks: was this record in the training set? Shokri and colleagues showed such attacks work against models served by commercial machine-learning-as-a-service providers, using only black-box queries. The simplest version, used below, thresholds the model's loss: training rows tend to have lower loss than rows the model has never seen. The attack AUC is 0.5 when members and non-members look the same.

## A real system that works this way

The Netflix Prize release is the standing reminder that "we removed the names" fails for rich records, and Carlini and colleagues' extraction results are the evidence that language models can return training text. I make no claims about specific products' privacy deployments, which I did not verify.

## Code you can run

Everything is CPU only and seeded. The data are synthetic. Block 4 uses the privacy accountant from Opacus 1.6.0 (installed into the chapter's environment for this purpose) and numpy for the training loop, so it needs no GPU.

#### 1. PII detection and redaction, scored

400 synthetic lines contain emails, phone numbers, card numbers (generated to pass the Luhn checksum), names and plain 16-digit order numbers that are not cards. The order numbers are random, so about one in ten also passes the checksum.

```python
import hashlib
import hmac
import random
import re

rng = random.Random(11)
FIRST = ["Priya", "Tom", "Aisha", "Marco", "Chen", "Ingrid", "Omar", "Sofia"]
LAST = ["Sharma", "Becker", "Okafor", "Rossi", "Wang", "Larsen", "Haddad", "Silva"]
LABELS = ["Order", "Build", "Ticket"]


def luhn_ok(digits):
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def make_card():
    body = [rng.randint(0, 9) for _ in range(15)]
    for check in range(10):
        if luhn_ok("".join(map(str, body + [check]))):
            return "".join(map(str, body + [check]))


def make_document():
    first, last = rng.choice(FIRST), rng.choice(LAST)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@example.org"
    phone = f"+44 7{rng.randint(100, 999)} {rng.randint(100000, 999999)}"
    card = make_card()
    templates = [
        (f"Customer {name} wrote from {email} about a refund.", {"NAME": name, "EMAIL": email}),
        (f"Call {name} on {phone} after 5pm.", {"NAME": name, "PHONE": phone}),
        (f"Paid with card {card} by Dr {last}.", {"CARD": card, "NAME": f"Dr {last}"}),
        (f"{rng.choice(LABELS)} {rng.randint(10 ** 15, 10 ** 16 - 1)} closed today.", {}),
    ]
    return rng.choice(templates)


DOCS = [make_document() for _ in range(400)]
PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PHONE": re.compile(r"\+\d{2} \d{4} \d{6}"),
    "CARD": re.compile(r"\b\d{16}\b"),
    "NAME": re.compile(r"\b(?:Mr|Ms|Dr) [A-Z][a-z]+"),
}


def detect(text, use_luhn):
    found = set()
    for kind, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            if kind == "CARD" and use_luhn and not luhn_ok(m.group()):
                continue
            found.add((kind, m.group()))
    return found


def score(use_luhn):
    stats = {k: [0, 0, 0] for k in PATTERNS}
    for text, truth in DOCS:
        gold = set(truth.items())
        found = detect(text, use_luhn)
        for kind in PATTERNS:
            g = {x for x in gold if x[0] == kind}
            f = {x for x in found if x[0] == kind}
            stats[kind][0] += len(g & f)
            stats[kind][1] += len(f - g)
            stats[kind][2] += len(g - f)
    return stats


for use_luhn in (False, True):
    print(f"\nregex detectors, Luhn check on cards: {use_luhn}")
    print("type    found  false alarms  missed  precision  recall")
    for kind, (tp, fp, fn) in score(use_luhn).items():
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        print(f"{kind:6s}  {tp:5d}  {fp:12d}  {fn:6d}  {precision:9.3f}  {recall:6.3f}")

KEY = b"rotate-me-and-keep-me-out-of-the-repo"


def redact(text, mode):
    for kind, pattern in PATTERNS.items():
        def sub(m):
            if kind == "CARD" and not luhn_ok(m.group()):
                return m.group()
            if mode == "mask":
                return f"[{kind}]"
            return f"[{kind}:{hmac.new(KEY, m.group().encode(), hashlib.sha256).hexdigest()[:6]}]"
        text = pattern.sub(sub, text)
    return text


sample = "Customer Priya Sharma wrote from priya.sharma@example.org; Priya Sharma called again."
print("\nmasking and keyed pseudonyms on one line (the plain name slips through, as the table predicted)")
print(" ", redact(sample, "mask"))
print(" ", redact(sample, "pseudonym"))
print(" ", redact("a wrote from priya.sharma@example.org", "pseudonym"))
```

```text

regex detectors, Luhn check on cards: False
type    found  false alarms  missed  precision  recall
EMAIL     104             0       0      1.000   1.000
PHONE      93             0       0      1.000   1.000
CARD       99           104       0      0.488   1.000
NAME       99             0     197      1.000   0.334

regex detectors, Luhn check on cards: True
type    found  false alarms  missed  precision  recall
EMAIL     104             0       0      1.000   1.000
PHONE      93             0       0      1.000   1.000
CARD       99            11       0      0.900   1.000
NAME       99             0     197      1.000   0.334

masking and keyed pseudonyms on one line (the plain name slips through, as the table predicted)
  Customer Priya Sharma wrote from [EMAIL]; Priya Sharma called again.
  Customer Priya Sharma wrote from [EMAIL:b27510]; Priya Sharma called again.
  a wrote from [EMAIL:b27510]
```

Emails and the one phone format are found perfectly. Cards show why a **checksum** matters: without the Luhn test every 16-digit number is a card and precision is 0.488 (104 false alarms); with it precision rises to 0.900, and the remaining 11 false alarms are random numbers that pass the check by chance. Names are the weak spot. The pattern finds only titled names such as "Dr Silva", so recall is 0.334 and 197 plain names are missed; the redaction demo shows "Priya Sharma" surviving. That is where a trained recogniser earns its keep, and even then, as Presidio's own documentation says, not every item is found. The last two lines show that the keyed pseudonym is stable: the same email always becomes the same token.

#### 2. k-anonymity, linkage and homogeneity

A 3,000-row table has zip, age, sex and a diagnosis. An "attacker" holds a public list of names with the same three columns and joins it to the released table.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
n = 3000
prefix = rng.choice(np.arange(100, 120), n, p=np.random.default_rng(1).dirichlet(np.ones(20) * 3))
people = pd.DataFrame({
    "name": [f"person{i:04d}" for i in range(n)],
    "zip": [f"{p}{rng.integers(0, 100):02d}" for p in prefix],
    "age": rng.integers(18, 90, n),
    "sex": rng.choice(["F", "M"], n),
})
old = people["age"] > 70
diagnosis = np.where(old, rng.choice(["heart", "diabetes", "flu"], n, p=[0.5, 0.3, 0.2]),
                     rng.choice(["heart", "diabetes", "flu"], n, p=[0.1, 0.2, 0.7]))
cluster = np.isin(prefix, [105, 106, 107]) & (rng.random(n) < 0.95)
released = people.drop(columns="name").assign(diagnosis=np.where(cluster, "asthma", diagnosis))
QI = ["zip", "age", "sex"]


def generalise(frame, zip_digits, age_band):
    out = frame.copy()
    out["zip"] = out["zip"].str[:zip_digits] + "*" * (5 - zip_digits)
    out["age"] = (out["age"] // age_band * age_band).astype(str) + "+"
    return out


def k_anonymity(frame):
    return int(frame.groupby(QI).size().min())


def suppress(frame, k):
    sizes = frame.groupby(QI)["sex"].transform("size")
    return frame[sizes >= k]


def reidentified(zip_digits, age_band, k):
    public = generalise(people[["name"] + QI], zip_digits, age_band)
    table = suppress(generalise(released, zip_digits, age_band), k)
    matches = public.merge(table, on=QI).groupby("name").size()
    return float((matches == 1).sum() / len(people))


K = 5
print(f"generalisation                    k as is   re-identified as is   rows suppressed for k = {K}   re-identified after")
for zip_digits, age_band in ((5, 1), (4, 5), (3, 5), (3, 10), (3, 20)):
    table = generalise(released, zip_digits, age_band)
    dropped = 1 - len(suppress(table, K)) / len(table)
    label = f"zip {zip_digits} digits, age band {age_band}"
    print(f"{label:32s} {k_anonymity(table):7d}   {reidentified(zip_digits, age_band, 1):17.3f}   {dropped:21.3f}   {reidentified(zip_digits, age_band, K):.3f}")

table = suppress(generalise(released, 3, 20), K)
groups = table.groupby(QI)["diagnosis"].agg(size="size", top=lambda s: s.value_counts(normalize=True).iloc[0])
print(f"\nzip 3 digits, age band 20, suppressed to k = {K}: now k = {k_anonymity(table)} with {len(groups)} groups")
risky = groups[groups["top"] >= 0.9]
print(f"groups where 90% or more share one diagnosis: {len(risky)} covering {int(risky['size'].sum())} of {len(table)} people")
print("each of those people is hidden among at least 5 others and still has the diagnosis disclosed")
```

```text
generalisation                    k as is   re-identified as is   rows suppressed for k = 5   re-identified after
zip 5 digits, age band 1               1               0.985                   1.000   0.000
zip 4 digits, age band 5               1               0.539                   0.991   0.000
zip 3 digits, age band 5               1               0.017                   0.269   0.000
zip 3 digits, age band 10              1               0.004                   0.054   0.000
zip 3 digits, age band 20              1               0.003                   0.028   0.000

zip 3 digits, age band 20, suppressed to k = 5: now k = 5 with 161 groups
groups where 90% or more share one diagnosis: 23 covering 283 of 2917 people
each of those people is hidden among at least 5 others and still has the diagnosis disclosed
```

With full detail, 0.985 of people are unique on zip, age and sex and the join re-identifies 0.985 of them. Each coarser row makes people less unique (0.539, 0.017, 0.003), but notice the price: to guarantee k = 5 at four zip digits you would suppress 0.991 of rows. At three digits and age bands of 20 only 0.028 of rows go, and after suppression nobody is re-identified by the join. The final lines show the second pitfall: in this table 23 groups covering 283 of 2,917 people have 90% or more of one diagnosis, so someone hidden among at least five others still has their diagnosis disclosed with 90% or more confidence. The cluster of asthma in three zip prefixes is planted in the synthetic data to make the point; real data have such clusters too.

#### 3. The Laplace mechanism and the budget

```python
import numpy as np

rng = np.random.default_rng(5)
ages = rng.integers(18, 90, 10000)
true_count = int((ages > 65).sum())
print(f"true count of people over 65: {true_count} (counting query, sensitivity 1)")


def laplace_count(epsilon, size=1):
    return true_count + rng.laplace(0.0, 1.0 / epsilon, size)


print("\nepsilon   noise scale   mean absolute error over 20000 releases   one release")
for epsilon in (0.01, 0.1, 0.5, 1.0, 5.0):
    draws = laplace_count(epsilon, 20000)
    print(f"{epsilon:7.2f}   {1 / epsilon:11.1f}   {np.abs(draws - true_count).mean():38.2f}   {draws[0]:11.1f}")

print("\nindistinguishability: add or remove one person and compare output densities at the same value")
epsilon = 0.5
neighbour = true_count + 1
for out in (true_count - 2, true_count, true_count + 3):
    ratio = np.exp(-abs(out - true_count) * epsilon) / np.exp(-abs(out - neighbour) * epsilon)
    print(f"  output {out}: density ratio {ratio:.3f} (bound e^eps = {np.exp(epsilon):.3f}, 1/e^eps = {1 / np.exp(epsilon):.3f})")

print("\nrepeating one query: the analyst averages fresh noise away")
epsilon_each = 0.1
for k in (1, 10, 100, 1000):
    errors = [abs(laplace_count(epsilon_each, k).mean() - true_count) for _ in range(500)]
    print(f"  {k:4d} releases at epsilon {epsilon_each} each: total epsilon {k * epsilon_each:6.1f}, mean error of the average {np.mean(errors):6.2f}")

print("\nspending a budget of epsilon 1.0 on three different questions (sequential composition adds the epsilons)")
budget = 1.0
split = {"count over 65": 0.5, "count over 80": 0.3, "count under 25": 0.2}
for name, eps in split.items():
    print(f"  {name:16s} epsilon {eps:.1f}  expected error {1 / eps:.1f}")
print(f"  total epsilon {sum(split.values()):.1f} of {budget:.1f}; one more question needs a bigger budget or has to wait")
```

```text
true count of people over 65: 3331 (counting query, sensitivity 1)

epsilon   noise scale   mean absolute error over 20000 releases   one release
   0.01         100.0                                   100.98        3313.2
   0.10          10.0                                     9.95        3349.0
   0.50           2.0                                     1.99        3331.3
   1.00           1.0                                     1.00        3332.4
   5.00           0.2                                     0.20        3330.8

indistinguishability: add or remove one person and compare output densities at the same value
  output 3329: density ratio 1.649 (bound e^eps = 1.649, 1/e^eps = 0.607)
  output 3331: density ratio 1.649 (bound e^eps = 1.649, 1/e^eps = 0.607)
  output 3334: density ratio 0.607 (bound e^eps = 1.649, 1/e^eps = 0.607)

repeating one query: the analyst averages fresh noise away
     1 releases at epsilon 0.1 each: total epsilon    0.1, mean error of the average   9.23
    10 releases at epsilon 0.1 each: total epsilon    1.0, mean error of the average   3.53
   100 releases at epsilon 0.1 each: total epsilon   10.0, mean error of the average   1.15
  1000 releases at epsilon 0.1 each: total epsilon  100.0, mean error of the average   0.34

spending a budget of epsilon 1.0 on three different questions (sequential composition adds the epsilons)
  count over 65    epsilon 0.5  expected error 2.0
  count over 80    epsilon 0.3  expected error 3.3
  count under 25   epsilon 0.2  expected error 5.0
  total epsilon 1.0 of 1.0; one more question needs a bigger budget or has to wait
```

The mean absolute error of a Laplace release equals its scale, 1 over epsilon, as the first table shows (about 100.98 at epsilon 0.01 against 0.20 at 5). The density-ratio lines check the definition: moving from the true count to one person more changes the density at any output by at most e to the 0.5, which is 1.649 (or its reciprocal 0.607). The repeated-query table is the reason budgets exist: ten releases at epsilon 0.1 each cost a total epsilon of 1.0, and a thousand cost 100.0, but their average has an error of only 0.34, so unlimited queries average the privacy away. Spending a fixed budget means choosing which questions deserve the noise.

The lab draws the two densities and tracks the budget. Its defaults are the chapter's epsilon 0.5 and a probe 3 above the true count, giving noise scale 2.0 and a density ratio of 0.607 against the bound 1.649.

<PrivacyBudgetLab />

#### 4. Memorisation, a membership attack and DP-SGD

A logistic regression with 200 features and only 100 training rows can memorise them. Each configuration is trained on eight different datasets and the results are averaged. The attack uses the loss: members get a score of minus their loss.

```python
import numpy as np
from opacus.accountants import RDPAccountant
from sklearn.metrics import roc_auc_score

N, D, EPOCHS, BATCH, LR, SEEDS = 100, 200, 100, 10, 0.1, 8
W_TRUE = np.zeros(D)
W_TRUE[:5] = 1.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(w, X, y):
    p = np.clip(sigmoid(X @ w), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def draw(rng, n):
    X = rng.normal(0, 1, (n, D))
    return X, (X @ W_TRUE + rng.normal(0, 2.0, n) > 0).astype(float)


def train(Xtr, ytr, rng, clip=None, sigma=0.0):
    w = np.zeros(D)
    for _ in range(EPOCHS):
        for _ in range(N // BATCH):
            idx = rng.choice(N, BATCH, replace=False)
            grads = (sigmoid(Xtr[idx] @ w) - ytr[idx])[:, None] * Xtr[idx]
            if clip is not None:
                grads = grads / np.maximum(1.0, np.linalg.norm(grads, axis=1, keepdims=True) / clip)
            total = grads.sum(axis=0)
            if clip is not None:
                total = total + rng.normal(0, sigma * clip, D)
            w = w - LR * total / BATCH
    return w


def evaluate(clip, sigma):
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        Xtr, ytr = draw(rng, N)
        Xout, yout = draw(rng, N)
        Xte, yte = draw(rng, 2000)
        w = train(Xtr, ytr, rng, clip, sigma)
        members, others = loss(w, Xtr, ytr), loss(w, Xout, yout)
        auc = roc_auc_score(np.r_[np.ones(N), np.zeros(N)], np.r_[-members, -others])
        out.append([((sigmoid(Xtr @ w) > 0.5) == ytr).mean(), ((sigmoid(Xte @ w) > 0.5) == yte).mean(), auc])
    return np.mean(out, axis=0)


def epsilon(sigma, delta=1e-5):
    accountant = RDPAccountant()
    for _ in range(EPOCHS * (N // BATCH)):
        accountant.step(noise_multiplier=sigma, sample_rate=BATCH / N)
    return accountant.get_epsilon(delta=delta)


print(f"mean over {SEEDS} seeds; {N} training rows, {D} features, {EPOCHS * (N // BATCH)} steps, sampling rate {BATCH / N:.2f}")
print("model                         train acc  test acc  attack AUC   epsilon at delta 1e-5")
train_acc, test_acc, auc = evaluate(None, 0.0)
print(f"{'plain training':28s} {train_acc:9.3f}  {test_acc:8.3f}  {auc:10.3f}   none")
for sigma in (0.5, 1.0, 2.0, 4.0):
    train_acc, test_acc, auc = evaluate(1.0, sigma)
    print(f"{'DP-SGD, noise multiplier ' + str(sigma):28s} {train_acc:9.3f}  {test_acc:8.3f}  {auc:10.3f}   {epsilon(sigma):.2f}")
print("\nattack AUC 0.5 means the loss gives no hint whether a row was in training")
```

```text
mean over 8 seeds; 100 training rows, 200 features, 1000 steps, sampling rate 0.10
model                         train acc  test acc  attack AUC   epsilon at delta 1e-5
plain training                   1.000     0.585       0.865   none
DP-SGD, noise multiplier 0.5     0.996     0.583       0.830   136.05
DP-SGD, noise multiplier 1.0     0.979     0.571       0.781   27.16
DP-SGD, noise multiplier 2.0     0.885     0.552       0.691   8.94
DP-SGD, noise multiplier 4.0     0.720     0.528       0.614   3.74

attack AUC 0.5 means the loss gives no hint whether a row was in training
```

Plain training reaches a train accuracy of 1.000 against a test accuracy of 0.585, and the loss-threshold attack tells members from non-members with an AUC of 0.865. DP-SGD trades this away gradually: at noise multiplier 4.0 the attack AUC is 0.614 and the privacy accountant reports epsilon 3.74 at delta 1e-5, but test accuracy has fallen to 0.528, close to a coin flip. At noise multiplier 0.5 the guarantee is weak (epsilon 136.05) and the attack still falls from 0.865 to 0.830. Two cautions. The epsilon is a worst-case bound on what any attack could learn, while the AUC is one attack's measurement, so a low AUC is not a proof of privacy. And with 100 rows the cost of privacy is large; large datasets make DP-SGD far cheaper per unit of epsilon, which is why it is used with big training sets.

<Infographic src="/img/gov/privacy-pii-and-differential-privacy-dp.svg" alt="Tables for the Laplace mechanism at five epsilon values and for DP-SGD at four noise multipliers, with the privacy-utility trade-off summarised." caption="Blocks 3 and 4: epsilon sets the noise, composition spends it, and DP-SGD trades accuracy for a lower attack AUC." />

## Production snippets (not run here)

The recogniser pipeline below is the usual starting point for names and free text. It needs a downloaded language model for the NLP engine, so it is **Not run in this environment**.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
text = "Contact Priya Sharma on +44 7123 456789."
found = analyzer.analyze(text=text, language="en")
print(anonymizer.anonymize(text=text, analyzer_results=found).text)
```

For training with differential privacy in PyTorch, Opacus wraps the model, optimiser and data loader so that per-example clipping, noise and accounting happen for you; the accountant is the part this chapter ran.

## Designing with it

- **Minimise before you detect.** Do not log what you do not need; detection is a second line of defence with the recall you measured.
- **Measure recall on your own text**, with a labelled sample re-scored on every change.
- **Never publish a table because the names are gone.** Check k over realistic quasi-identifiers, suppress small groups, and check the sensitive column inside each group.
- **Decide the epsilon budget in advance** and account for every release, including repeated ones.
- **Treat memorisation as a test.** Deduplicate training data, run a membership or extraction test against your own model, and use DP training where the data justify its cost.
- **Set retention.** Data you delete cannot leak. Write down how long prompts, logs and training sets live.

## Where this stands in 2026

:::info Industry view

- **Differential privacy has official evaluation guidance.** NIST SP 800-226 (March 2025) gives practitioners a way to examine a differentially private deployment and lists the hazards that undo its guarantees.
- **PII tooling is mature but not perfect.** Presidio's documentation, at the project's new address, still carries the no-guarantee warning; the PyPI release of `presidio-analyzer` I checked is 2.2.364 of 22 July 2026.
- **Opacus is at 1.6.0 (May 2026)** and provides the RDP accountant used above.
- **Privacy and bias testing meet in the AI Act.** The amended Act allows exceptional processing of special categories of personal data to detect and correct bias under listed safeguards, including deletion once the bias is corrected; see [the regulation chapter](/docs/governance/regulation-and-model-documentation) and [the fairness chapter](/docs/governance/fairness-testing-in-practice).

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why does the Luhn check raise card precision from 0.488 to 0.900 but not to 1.000?</summary>

It removes the 16-digit numbers that fail the checksum, 104 false alarms down to 11. About one random 16-digit number in ten passes the checksum by chance, so some order numbers still look like cards. Context, such as nearby words, is the next signal.

</details>

<details>
<summary><strong>Q2.</strong> A released table has k = 5 over zip, age and sex. Is the diagnosis column safe?</summary>

Not necessarily. In the example 23 groups covering 283 people have 90% or more of one diagnosis, so being one of at least five similar rows does not hide the diagnosis. k-anonymity protects against picking out a row, not against learning an attribute.

</details>

<details>
<summary><strong>Q3.</strong> What is the sensitivity of "how many people are over 65" and what noise scale does epsilon 0.5 need?</summary>

Sensitivity is 1, because adding or removing one person changes the count by at most 1. The scale is sensitivity over epsilon, 1 / 0.5 = 2.0, and the expected absolute error is also about 2.

</details>

<details>
<summary><strong>Q4.</strong> A dashboard lets analysts re-run the same private count as often as they like. What goes wrong?</summary>

Each release spends epsilon, and the costs add. Averaging a thousand releases at epsilon 0.1 each gives an error of 0.34 while having spent a total epsilon of 100.0, so the protection is gone. Count every release against a budget and cache answers.

</details>

<details>
<summary><strong>Q5.</strong> The membership attack AUC is 0.614 after DP-SGD at epsilon 3.74. Can you say the model is private?</summary>

No. The AUC is one attack's result; epsilon is the worst-case guarantee, and a stronger attack may do better than the loss threshold. The epsilon at delta 1e-5 is the statement to defend, and its interpretation depends on the data size and the accounting.

</details>

<details>
<summary><strong>Q6.</strong> Why is keyed pseudonymisation better than plain hashing of an email address?</summary>

Anyone can hash a list of likely emails and compare. A keyed hash needs the secret key, so guessing needs the key. It still allows joins on the token, which is why the key must be protected and rotated.

</details>

## Further reading

All opened for this chapter in October 2026, except where stated.

- Dwork and Roth, [The algorithmic foundations of differential privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).
- Abadi and colleagues, [Deep learning with differential privacy](https://arxiv.org/abs/1607.00133), 2016.
- Shokri, Stronati, Song and Shmatikov, [Membership inference attacks against machine learning models](https://arxiv.org/abs/1610.05820), 2017.
- Carlini and colleagues, [Extracting training data from large language models](https://arxiv.org/abs/2012.07805), 2020.
- Narayanan and Shmatikov, [How to break anonymity of the Netflix Prize dataset](https://arxiv.org/abs/cs/0610105), 2006.
- NIST, [SP 800-226, Guidelines for evaluating differential privacy guarantees](https://csrc.nist.gov/pubs/sp/800/226/final), March 2025.
- [Presidio documentation](https://presidio.dataprivacystack.org/).
- Sweeney, [k-anonymity: a model for protecting privacy](https://dataprivacylab.org/people/sweeney/kanonymity.html), 2002 (I confirmed the citation and the definition through a search result summary, and did not open the paper itself).

## Check yourself

- I can build a detector with a checksum, score its precision and recall, and say what it cannot find.
- I can check k-anonymity over quasi-identifiers and explain generalisation, suppression and the homogeneity problem.
- I can state the epsilon definition, add noise with the Laplace mechanism and explain why repeated queries need a budget.
- I can describe DP-SGD in three steps and what the accountant reports.
- I can run a loss-threshold membership test and say what its AUC does and does not prove.
- I can name the retention and minimisation decisions that matter before any of these tools.
