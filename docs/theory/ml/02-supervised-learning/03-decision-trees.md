---
id: ml-decision-trees
title: "Decision Trees"
sidebar_label: "Decision trees"
sidebar_position: 3
slug: /theory/ml/decision-trees
description: "Grow a tree of yes/no questions by always asking the question that purifies the data most, measured with entropy, information gain and Gini, then prune it so it does not memorise."
tags: [decision-trees, entropy, information-gain, gini, pruning, cart, id3]
---

import Infographic from '@site/src/components/Infographic';
import ImpurityLab from '@site/src/components/viz/ImpurityLab';

**In one line.** A decision tree plays twenty questions with your data: at every step it asks the yes/no question that leaves the cleanest groups behind, and stops before it memorises.

## The idea in plain words

Imagine a doctor sorting patients with a flowchart. "Is the temperature above 38? If yes, is there a cough?" Each box asks one question, each arrow is an answer, and each end point is a decision. A **decision tree** is that flowchart, except that the questions are not written by an expert: the algorithm finds them from labelled examples.

Training is one simple greedy idea, repeated.

1. Look at the examples that reach this node. If they are all one class, stop: this is a **leaf** and its answer is that class.
2. Otherwise try every available question, and for each one look at the groups it would create.
3. Keep the question whose groups are the **purest**, and recurse into each group.

So everything hinges on a number that says how mixed a group is. A group of nine "yes" and five "no" is fairly mixed. A group of four "yes" and no "no" is pure. *Entropy* and the *Gini index* are two such numbers: both are zero for a pure group and largest for a 50/50 mix. The *information gain* of a question is how much the average mixedness drops after asking it.

The appeal is that the result is a model you can read aloud and audit. It handles numbers and categories with little preparation, and it needs no scaling. The catch is the one you met in regression: a tree allowed to grow until every leaf is pure has memorised its training set, noise included. The cure is the same family of ideas, simplicity as a virtue. Stop early, or grow fully and cut weak branches (**pruning**), preferring the smallest tree that still explains the data.

<Infographic src="/img/ml/decision-trees-outlook-split.svg" alt="The 14 play-tennis days split by Outlook with entropy at every node, a bar chart of the gain of each attribute, and the finished tree" caption="Choosing the question. Entropy 0.940, weighted child entropy 0.694 and gain 0.247 are printed by block 1 below." />

<Infographic src="/img/ml/decision-trees-impurity-curves.svg" alt="Entropy, Gini and misclassification error plotted against the share of positives, with the 9 to 5 node marked" caption="Three ways to score a node. The marked values are printed by block 6 below." />

## How it works

### Trees & expressiveness

A flowchart of attribute tests from root to leaf. Trees can represent any Boolean function, handle mixed types, and are interpretable.

:::note

**Suited for** problems needing explanation with meaningful features: credit approval, medical triage, churn.

:::

### Entropy & information gain

H(S) = −Σ p_c log₂ p_c. Gain(S,A) = H(S) − Σ (|Sᵥ|/|S|)·H(Sᵥ). Split on the highest gain (ID3).

#### Impurity calculator

Set the positive/negative counts and see entropy and Gini. The interactive calculator is in the code section below, next to the numbers it reproduces.

:::tip

**Worked.** [9+,5−] → H = **0.94**. Outlook split → weighted child entropy 0.694 → gain = 0.94 − 0.694 = **0.247**. Pure node H=0, 50/50 H=1.

:::

### Gini index & attribute types

- **Gini index**: Gini(S) = 1 − Σ p_c². CART uses it with binary splits; cheaper than entropy (no log). [9+,5−] → **0.459**.
- **Splitting types**: Nominal → one branch per value; continuous → threshold x ≤ t, with t chosen to maximise gain.

### MDL, Occam & pruning

A tree grown to purity memorises the data. Prefer the smallest tree that explains it.

:::tip

**MDL / Occam's razor.** Prefer the simplest hypothesis. In practice **prune**: stop early (pre-pruning) or grow fully then cut weak branches (post-pruning) using a validation set.

:::

### Key takeaways

- **1 · Impurity**: Entropy / Gini; 0 pure, max at 50/50.
- **2 · Split**: Highest information gain (ID3) or lowest Gini (CART).
- **3 · Prune**: MDL/Occam: smallest tree that fits.

:::note

**The thread.** Decision trees turn learning into a game of "which question purifies the data most?" Entropy and Gini score the questions, information gain picks the winner, and pruning stops the tree from memorising, giving a model you can actually read.

:::

## A real system that works this way

**scikit-learn's decision trees** are the system most readers will actually use. Its user guide states that it implements an optimised version of **CART**, which builds *binary* trees: at each node it picks one feature and one threshold. That differs from the lecture's ID3, which makes one branch per category, so on categorical data you one-hot encode first (block 2 shows the consequence). The same guide documents the knobs the lecture only names: `max_depth`, `min_samples_split` and `min_samples_leaf` for pre-pruning, and `ccp_alpha` for post-pruning by cost-complexity. It also lists the known weaknesses honestly: trees overfit without pruning, are unstable (small data changes can give a completely different tree), predict piecewise-constant values and so cannot extrapolate, and are hurt by imbalanced classes. Missing values are supported natively in current releases.

**Google's decision-forests course** teaches the same structure (a root, conditions at the internal nodes, predictions at the leaves) and notes that its training library uses a CART learner, which is a useful reminder that CART is the workhorse behind the forests and boosted trees of the later chapters.

**The pattern in regulated settings** is credit approval, medical triage and churn screening, the uses the lecture names. A short tree is attractive there because a reviewer can follow the exact path that led to one decision, and can trace which rule to change when the policy changes.

## Code you can run

Seven blocks. Blocks 1, 2, 3, 6 and 7 are about choosing questions; blocks 4 and 5 are about size and stability. The play-tennis table is the standard 14-row teaching example, and it is the data behind the lecture's $[9{+},5{-}]$ root.

### 1. Entropy, Gini and information gain

This block reproduces every number in the lecture: the root entropy 0.94, Gini 0.459, the three Outlook children, the weighted entropy 0.694 and the gain 0.247. It then scores all four attributes.

```python
import numpy as np

PLAY_TENNIS = [
    ("Sunny", "Hot", "High", "Weak", "No"),
    ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"),
    ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"),
    ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"),
    ("Rain", "Mild", "High", "Strong", "No"),
]
ATTRIBUTES = ["Outlook", "Temperature", "Humidity", "Wind"]

def entropy(pos, neg):
    n = pos + neg
    return 0.0 - sum(c / n * np.log2(c / n) for c in (pos, neg) if c)

def gini(pos, neg):
    n = pos + neg
    return 1 - (pos / n) ** 2 - (neg / n) ** 2

def counts(rows):
    pos = sum(r[-1] == "Yes" for r in rows)
    return pos, len(rows) - pos

def gain(rows, column, impurity):
    total = impurity(*counts(rows))
    for value in sorted({r[column] for r in rows}):
        part = [r for r in rows if r[column] == value]
        total -= len(part) / len(rows) * impurity(*counts(part))
    return total

pos, neg = counts(PLAY_TENNIS)
print(f"root [{pos}+,{neg}-]: entropy {entropy(pos, neg):.3f} (lecture 0.94), Gini {gini(pos, neg):.3f} (lecture 0.459)")
print(f"pure node entropy {entropy(5, 0):.1f}, 50/50 node entropy {entropy(7, 7):.1f}\n")

print("Outlook children:")
weighted = 0
for value in ("Sunny", "Overcast", "Rain"):
    part = [r for r in PLAY_TENNIS if r[0] == value]
    p, n = counts(part)
    weighted += len(part) / 14 * entropy(p, n)
    print(f"  {value:9s} [{p}+,{n}-]  entropy {entropy(p, n):.3f}")
print(f"weighted child entropy {weighted:.3f} (lecture 0.694)   gain {entropy(pos, neg) - weighted:.3f} (lecture 0.247)\n")

print("attribute     information gain   Gini gain")
for column, name in enumerate(ATTRIBUTES):
    print(f"{name:12s}  {gain(PLAY_TENNIS, column, entropy):14.3f}   {gain(PLAY_TENNIS, column, gini):9.3f}")
```

What it prints:

```text
root [9+,5-]: entropy 0.940 (lecture 0.94), Gini 0.459 (lecture 0.459)
pure node entropy 0.0, 50/50 node entropy 1.0

Outlook children:
  Sunny     [2+,3-]  entropy 0.971
  Overcast  [4+,0-]  entropy 0.000
  Rain      [3+,2-]  entropy 0.971
weighted child entropy 0.694 (lecture 0.694)   gain 0.247 (lecture 0.247)

attribute     information gain   Gini gain
Outlook                0.247       0.116
Temperature            0.029       0.019
Humidity               0.152       0.092
Wind                   0.048       0.031
```

Each lecture figure is printed beside its own: entropy 0.940, Gini 0.459, weighted child entropy 0.694, gain 0.247. The final table ranks the attributes. Outlook wins by entropy (0.247) and by Gini (0.116), and Temperature comes last under both, so the two criteria agree here. Read the Outlook children too: Overcast is a pure $[4{+},0{-}]$ node with entropy 0, the best kind of result a split can give.

### Try it: the impurity calculator

The lecture's "impurity calculator" widget is below. Its defaults give the root node $[9{+},5{-}]$: entropy 0.940 and Gini 0.459. The second half of the lab splits the 14 days on any attribute and reports the weighted child impurity and the gain: leave it on *Outlook* with *entropy* and you read 0.694 and 0.247, then switch to *Gini* to get the 0.116 above. Push the positives and negatives to equal counts and watch entropy reach 1.

<ImpurityLab />

### 2. ID3 from scratch, next to scikit-learn

A dozen lines of recursion choose the highest-gain attribute at every node, then scikit-learn's CART is trained on the same rows for comparison.

```python
from collections import Counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

ROWS = [
    ("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No"),
]
NAMES = ["Outlook", "Temperature", "Humidity", "Wind"]

def entropy(rows):
    counts = Counter(r[-1] for r in rows)
    n = len(rows)
    return 0.0 - sum(c / n * np.log2(c / n) for c in counts.values())

def gain(rows, column):
    total = entropy(rows)
    for value in {r[column] for r in rows}:
        part = [r for r in rows if r[column] == value]
        total -= len(part) / len(rows) * entropy(part)
    return total

def id3(rows, columns):
    labels = {r[-1] for r in rows}
    if len(labels) == 1 or not columns:
        return Counter(r[-1] for r in rows).most_common(1)[0][0]
    best = max(columns, key=lambda c: gain(rows, c))
    rest = [c for c in columns if c != best]
    return (best, {v: id3([r for r in rows if r[best] == v], rest) for v in sorted({r[best] for r in rows})})

def show(node, depth=0):
    if isinstance(node, str):
        print("  " * depth + f"-> {node}")
        return
    column, branches = node
    for value, child in branches.items():
        print("  " * depth + f"{NAMES[column]} = {value}")
        show(child, depth + 1)

tree = id3(ROWS, [0, 1, 2, 3])
show(tree)

def predict(node, row):
    while not isinstance(node, str):
        column, branches = node
        node = branches[row[column]]
    return node

hits = sum(predict(tree, r) == r[-1] for r in ROWS)
print(f"\nID3 reproduces {hits} of {len(ROWS)} training labels")

encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform([r[:4] for r in ROWS])
y = [r[-1] for r in ROWS]
cart = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
print("\nscikit-learn (CART, binary splits on one-hot columns):")
print(export_text(cart, feature_names=list(encoder.get_feature_names_out(NAMES))))
```

What it prints:

```text
Outlook = Overcast
  -> Yes
Outlook = Rain
  Wind = Strong
    -> No
  Wind = Weak
    -> Yes
Outlook = Sunny
  Humidity = High
    -> No
  Humidity = Normal
    -> Yes

ID3 reproduces 14 of 14 training labels

scikit-learn (CART, binary splits on one-hot columns):
|--- Outlook_Overcast <= 0.50
|   |--- Humidity_Normal <= 0.50
|   |   |--- Outlook_Rain <= 0.50
|   |   |   |--- class: No
|   |   |--- Outlook_Rain >  0.50
|   |   |   |--- Wind_Strong <= 0.50
|   |   |   |   |--- class: Yes
|   |   |   |--- Wind_Strong >  0.50
|   |   |   |   |--- class: No
|   |--- Humidity_Normal >  0.50
|   |   |--- Wind_Weak <= 0.50
|   |   |   |--- Temperature_Cool <= 0.50
|   |   |   |   |--- class: Yes
|   |   |   |--- Temperature_Cool >  0.50
|   |   |   |   |--- class: No
|   |   |--- Wind_Weak >  0.50
|   |   |   |--- class: Yes
|--- Outlook_Overcast >  0.50
|   |--- class: Yes
```

The hand-written ID3 tree is the one in the first board: Outlook at the root, Overcast is a pure Yes, Sunny splits on Humidity, Rain splits on Wind, and it reproduces all 14 training labels. scikit-learn needs more questions for the same data because it can only ask binary questions about one-hot columns ("is Outlook Overcast?"), so a three-way split costs two levels. The tree is different in shape but it is the same idea.

### 3. Splitting a number

For a continuous attribute the lecture says to test thresholds between sorted values. Here are six temperatures with their labels.

```python
import numpy as np

temperature = np.array([40, 48, 60, 72, 80, 90], dtype=float)
play = np.array([0, 0, 1, 1, 1, 0])

def entropy(labels):
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return 0.0 - sum(q * np.log2(q) for q in (p, 1 - p) if q)

parent = entropy(play)
print(f"parent entropy {parent:.3f}   (3 positives, 3 negatives in the sorted list: 40 48 60 72 80 90)\n")
print("candidate threshold   left   right   weighted entropy   gain")
best = None
for lo, hi in zip(temperature[:-1], temperature[1:]):
    t = (lo + hi) / 2
    left, right = play[temperature <= t], play[temperature > t]
    w = (len(left) * entropy(left) + len(right) * entropy(right)) / len(play)
    marker = ""
    if play[temperature == lo][0] != play[temperature == hi][0]:
        marker = "  <- label changes here"
        if best is None or parent - w > best[1]:
            best = (t, parent - w)
    print(f"  temperature <= {t:5.1f}   {len(left)}      {len(right)}      {w:12.3f}   {parent - w:8.3f}{marker}")
print(f"\nbest threshold: {best[0]:.0f} with gain {best[1]:.3f}")
```

What it prints:

```text
parent entropy 1.000   (3 positives, 3 negatives in the sorted list: 40 48 60 72 80 90)

candidate threshold   left   right   weighted entropy   gain
  temperature <=  44.0   1      5             0.809      0.191
  temperature <=  54.0   2      4             0.541      0.459  <- label changes here
  temperature <=  66.0   3      3             0.918      0.082
  temperature <=  76.0   4      2             1.000      0.000
  temperature <=  85.0   5      1             0.809      0.191  <- label changes here

best threshold: 54 with gain 0.459
```

<Infographic src="/img/ml/decision-trees-continuous-threshold.svg" alt="Six temperature readings on a number line with candidate cut points, their gains, and the best cut at 54" caption="Where to cut a number. The gains in the table are printed by block 3." />

Five candidate cuts sit midway between neighbours. The two that fall where the label changes (54 and 85) are the serious candidates, and 54 wins with a gain of 0.459 against 0.191 for 85. In practice the library sorts the feature once and sweeps the cut point along it, which is why trees handle numbers so cheaply.

### 4. Overfitting and pruning

A noisy problem (15% of the labels are flipped) where a tree grown to purity has nowhere to go but memorisation.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=800, n_features=10, n_informative=4, n_redundant=0,
                           flip_y=0.15, class_sep=1.0, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

def row(label, tree):
    print(f"{label:28s} leaves={tree.get_n_leaves():4d} depth={tree.get_depth():3d} "
          f"train={tree.score(X_train, y_train):.3f} test={tree.score(X_test, y_test):.3f}")

row("fully grown", DecisionTreeClassifier(random_state=0).fit(X_train, y_train))
for depth in (2, 3, 5):
    row(f"pre-pruned max_depth={depth}", DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train))
row("pre-pruned min_samples_leaf=15", DecisionTreeClassifier(min_samples_leaf=15, random_state=0).fit(X_train, y_train))

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas[:-1]
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=2)
val_scores = [DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_fit, y_fit).score(X_val, y_val) for a in alphas]
best_alpha = alphas[int(np.argmax(val_scores))]
print(f"\ncost-complexity path has {len(alphas)} candidate alphas; the validation set picks ccp_alpha = {best_alpha:.4f}")
row("post-pruned (ccp_alpha)", DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0).fit(X_train, y_train))
```

What it prints:

```text
fully grown                  leaves=  78 depth= 17 train=1.000 test=0.753
pre-pruned max_depth=2       leaves=   4 depth=  2 train=0.717 test=0.700
pre-pruned max_depth=3       leaves=   8 depth=  3 train=0.775 test=0.784
pre-pruned max_depth=5       leaves=  18 depth=  5 train=0.829 test=0.759
pre-pruned min_samples_leaf=15 leaves=  20 depth=  9 train=0.838 test=0.784

cost-complexity path has 41 candidate alphas; the validation set picks ccp_alpha = 0.0117
post-pruned (ccp_alpha)      leaves=   5 depth=  4 train=0.792 test=0.784
```

<Infographic src="/img/ml/decision-trees-pruning.svg" alt="Training and test accuracy of six trees from fully grown to pruned, with leaf counts" caption="Pruning in numbers. Every bar is printed by block 4." />

The fully grown tree has 78 leaves, scores 1.000 on its training data and only 0.753 on unseen data. Every pruned variant gives up training accuracy (0.717 to 0.838) and several gain test accuracy, up to 0.784. The post-pruned tree (the validation set chose `ccp_alpha` 0.0117) does as well as anything on the test set with only 5 leaves, and that is the MDL / Occam point from the lecture in numbers: the smallest tree that explains the data generalises as well as the big one. The depth-2 tree shows the other edge, too simple at 0.700.

### 5. Instability

Same 569-row dataset, twenty bootstrap resamples, one small tree per resample.

```python
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target
rng = np.random.default_rng(0)

roots = []
for _ in range(20):
    idx = rng.integers(0, len(X), len(X))
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[idx], y[idx])
    roots.append(data.feature_names[tree.tree_.feature[0]])

print("root question chosen by 20 trees, each trained on a bootstrap resample of the same 569 rows:")
for name, n in Counter(roots).most_common():
    print(f"  {n:2d} x  {name}")

shapes = set()
for _ in range(20):
    idx = rng.integers(0, len(X), len(X))
    tree = DecisionTreeClassifier(random_state=0).fit(X[idx], y[idx])
    shapes.add((tree.get_depth(), tree.get_n_leaves()))
print(f"\nfully grown trees on 20 resamples came out with {len(shapes)} different (depth, leaves) shapes")
```

What it prints:

```text
root question chosen by 20 trees, each trained on a bootstrap resample of the same 569 rows:
  10 x  worst perimeter
   4 x  worst area
   3 x  worst concave points
   2 x  mean concave points
   1 x  worst radius

fully grown trees on 20 resamples came out with 11 different (depth, leaves) shapes
```

Shuffle the rows a little and the tree's *first question* changes: ten resamples pick `worst perimeter`, but the other ten pick four different features. The fully grown trees come out in 11 different shapes. This is variance, and it is the reason the next group of chapters averages many trees instead of trusting one.

### 6. The impurity curves

```python
import numpy as np

print("   p     entropy    Gini    misclassification")
for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 9 / 14, 0.7, 0.9, 1.0):
    h = 0.0 - sum(q * np.log2(q) for q in (p, 1 - p) if q > 0)
    g = 1 - p ** 2 - (1 - p) ** 2
    e = 1 - max(p, 1 - p)
    note = "   <- the lecture's [9+,5-]" if abs(p - 9 / 14) < 1e-12 else ""
    print(f"{p:5.3f}   {h:7.3f}   {g:6.3f}   {e:8.3f}{note}")
```

What it prints:

```text
   p     entropy    Gini    misclassification
0.000     0.000    0.000      0.000
0.100     0.469    0.180      0.100
0.200     0.722    0.320      0.200
0.300     0.881    0.420      0.300
0.400     0.971    0.480      0.400
0.500     1.000    0.500      0.500
0.643     0.940    0.459      0.357   <- the lecture's [9+,5-]
0.700     0.881    0.420      0.300
0.900     0.469    0.180      0.100
1.000     0.000    0.000      0.000
```

Entropy peaks at 1.000 and Gini at 0.500, both at a 50/50 mix, and both are zero at the ends. The lecture's node is the highlighted row: $p = 9/14 = 0.643$ gives 0.940 and 0.459. Misclassification error has the same shape but is a straight line, which gives the search less to work with, and that is why trees do not split on it.

### 7. A trap: identifier columns

```python
import numpy as np

ROWS = [
    ("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No"),
]
rows = [(f"D{i + 1}",) + r for i, r in enumerate(ROWS)]
names = ["Day (unique id)", "Outlook", "Temperature", "Humidity", "Wind"]

def entropy(labels):
    n = len(labels)
    return 0.0 - sum(c / n * np.log2(c / n) for c in (labels.count("Yes"), labels.count("No")) if c)

def gain_and_ratio(column):
    labels = [r[-1] for r in rows]
    total, split_info = entropy(labels), 0.0
    for v in sorted({r[column] for r in rows}):
        part = [r[-1] for r in rows if r[column] == v]
        w = len(part) / len(rows)
        total -= w * entropy(part)
        split_info -= w * np.log2(w)
    return total, total / split_info, split_info

print("attribute          gain   split info   gain ratio")
for c, name in enumerate(names):
    g, r, si = gain_and_ratio(c)
    print(f"{name:16s} {g:6.3f}   {si:9.3f}   {r:9.3f}")
```

What it prints:

```text
attribute          gain   split info   gain ratio
Day (unique id)   0.940       3.807       0.247
Outlook           0.247       1.577       0.156
Temperature       0.029       1.557       0.019
Humidity          0.152       1.000       0.152
Wind              0.048       0.985       0.049
```

The unique-ID column splits the 14 rows into 14 pure single-row children and so achieves the maximum gain, 0.940, nearly four times the real winner. A tree that used it would "learn" nothing and generalise to nothing. Gain ratio narrows the gap but does not remove it here, so the real defence is not to feed the tree such columns.

## Designing with it

### Controlling the size of the tree

| Knob (scikit-learn) | What it does | Direction for less overfitting |
| --- | --- | --- |
| `max_depth` | Caps the number of questions on any path | Smaller |
| `min_samples_leaf` | Every leaf must hold at least this many rows | Larger (the guide suggests trying 5 first) |
| `min_samples_split` | A node needs this many rows before it may split | Larger |
| `ccp_alpha` | Cuts branches whose gain does not pay for their size | Larger |

Pre-pruning is cheap and predictable. Post-pruning with `ccp_alpha` is more adaptive: grow the full tree, ask `cost_complexity_pruning_path` for the sequence of candidate alphas, and choose with a validation set or cross-validation, as block 4 does.

### What to remember when you reach for a tree

- **No scaling needed**, because a threshold on a feature is unchanged by monotone rescaling.
- **Remove identifier-like columns** (row IDs, timestamps used as keys). Block 7 shows how an ID column wins the split.
- **Use trees to explain, ensembles to predict.** One shallow tree is a communication tool. Accuracy is what the ensemble chapter (bagging, random forests, boosting) is for, and block 5 shows why: a single tree is unstable.
- **Do not trust impurity-based feature importances blindly.** They favour features with many distinct values.
- **Know the failure shapes.** A tree approximates a diagonal boundary with a staircase, cannot extrapolate beyond the training range, and needs many levels to express parity-like rules such as XOR.

:::note Correction: the lecture says "ID3", the library does CART
The lecture picks splits by information gain (ID3) and says CART "uses Gini with binary splits". Both are right, but if you go on to use scikit-learn you are always using CART-style binary splits, even when you set `criterion="entropy"`. The two criteria usually choose the same question, as the table in block 1 shows (Outlook first, Temperature last, whichever you use).
:::

:::note Beyond the lecture: information gain likes many-valued attributes
Information gain is biased towards attributes with many distinct values. Give every row a unique ID and splitting on it makes every child pure, so its gain is the maximum possible. Block 7 shows it: the ID column scores a gain of 0.940, almost four times Outlook's 0.247. Dividing by the *split information* gives the **gain ratio** and shrinks the advantage, but here the ID column still edges ahead (0.247 against 0.156), so the practical fixes are to drop identifier columns and to forbid tiny leaves.
:::

:::note Beyond the lecture: regression trees
The same machinery predicts numbers. Replace entropy with the variance of the target inside a group, pick the split that reduces the weighted variance most, and have each leaf predict the mean of its rows. That is what `DecisionTreeRegressor` does.
:::

## Where this stands in 2026

:::info Industry view

- A single tree is now mostly a teaching and explanation tool. The strong tabular models are ensembles of trees, random forests and gradient-boosted trees, covered in the ensemble chapters.
- Benchmark evidence (Grinsztajn, Oyallon and Varoquaux, 2022, 45 datasets) reports tree-based models remaining state of the art on medium-sized tabular data of around ten thousand rows, ahead of tuned deep networks. TabPFN (Nature, January 2025) reports surpassing tuned tree ensembles on datasets up to 10,000 samples and 500 features. Treat both as benchmark results, and test on your own data.
- Tree libraries have moved on from the lecture: native handling of missing values is now documented in scikit-learn, and the histogram-based boosted-tree implementations scale far beyond what a plain tree can.
- Interpretability expectations have grown with regulation. A short pruned tree is among the few models you can print and hand to an auditor, which is why it persists in credit and clinical-triage settings.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Why are decision trees considered expressive and interpretable?</summary>

They can represent any Boolean function of the inputs, handle mixed attribute types with little preprocessing, and produce readable if/else rules from root to leaf.<br /><em>Module 5 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Write the entropy formula and compute the entropy of [9+, 5−].</summary>

H(S) = −Σ p_c log₂ p_c. H([9+,5−]) = −(9/14)log₂(9/14) − (5/14)log₂(5/14) = 0.94. (Pure node → 0; 50/50 → 1.)<br /><em>Module 5 · numeric</em>

</details>

<details>
<summary><strong>Q3.</strong> Define information gain and compute it for the Outlook split (parent H=0.94).</summary>

Gain(S,A) = H(S) − Σ (|Sᵥ|/|S|)H(Sᵥ). Outlook: (5/14)(0.971)+(4/14)(0)+(5/14)(0.971) = 0.694 → Gain = 0.94 − 0.694 = 0.247.<br /><em>Module 5 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Write the Gini index and compute it for [9+, 5−]. How does it compare to entropy?</summary>

Gini(S) = 1 − Σ p_c² = 1 − (9/14)² − (5/14)² = 0.459. It usually picks the same split as entropy but is cheaper (no logarithm); CART uses it with binary splits.<br /><em>Module 5 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> How are continuous attributes split in a decision tree?</summary>

By a threshold test x ≤ t, where t is chosen (from candidate values between sorted points) to maximise information gain.<br /><em>Module 5 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> What is the MDL principle and how does it relate to Occam's razor?</summary>

Minimum Description Length prefers the hypothesis that minimises the total description length of model + data; an information-theoretic Occam's razor favouring the simplest tree that explains the data.<br /><em>Module 5 · conceptual</em>

</details>

<details>
<summary><strong>Q7.</strong> How does pruning combat overfitting in trees?</summary>

A fully grown tree memorises noise. Pre-pruning stops growth early (depth/min-samples); post-pruning grows the full tree then cuts back weak branches using a validation set, reducing variance.<br /><em>Module 5 · conceptual</em>

</details>

## Further reading

- [scikit-learn user guide: Decision trees](https://scikit-learn.org/stable/modules/tree.html) is the primary reference for CART, criteria, pruning and the practical tips used here.
- [scikit-learn example: Post pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) walks through `ccp_alpha` end to end.
- [scikit-learn example: Permutation importance vs random forest feature importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html) shows with a demonstration why impurity-based importances favour high-cardinality features.
- [Google decision forests course: Decision trees](https://developers.google.com/machine-learning/decision-forests/decision-trees) is a clear visual introduction to the structure.
- [Induction of Decision Trees (J. R. Quinlan, Machine Learning, 1986)](https://doi.org/10.1023/A:1022643204877) is the paper that introduced ID3 and the information-gain criterion.
- The Elements of Statistical Learning treats trees (CART) with the full theory. The book is available to download from its authors' page: [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/).
- The 14-row play-tennis table in blocks 1, 2 and 7 is a widely used teaching example. The lecture's $[9{+},5{-}]$ root and its Outlook split come from it, as block 1 confirms by reproducing every figure.
- Built from the course lecture "ml-m5-decision-trees" (Lecture Library series).

- **[An Introduction to Statistical Learning](https://www.statlearning.com/)** `book`
  James, Witten, Hastie & Tibshirani: The friendliest rigorous intro to ML, free PDF plus R/Python labs.
- **[Stanford CS229 (Machine Learning)](https://cs229.stanford.edu/)** `course`
  Andrew Ng, Stanford: The rigorous derivations behind SVMs, GLMs, EM and learning theory.
- **[StatQuest](https://statquest.org/video-index/)** `▶ video`
  Josh Starmer: Short, wonderfully clear videos that build intuition step by step.

## What you should now be able to do

- [ ] I can compute the entropy 0.940 and the Gini index 0.459 of a node with nine positives and five negatives.
- [ ] I can compute the information gain of a split and reproduce the 0.247 for Outlook.
- [ ] I can explain how a split on a numeric attribute is chosen, and why only label changes need to be considered.
- [ ] I can explain why a tree grown to purity overfits, and tell pre-pruning from post-pruning.
- [ ] I can explain why information gain favours identifier-like columns and what to do about it.
- [ ] I can explain why a single tree is unstable, which is what motivates ensembles.
