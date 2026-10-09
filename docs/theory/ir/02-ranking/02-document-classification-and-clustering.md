---
id: ir-document-classification-clustering
title: "Information Retrieval · Session 6 — Document Classification and Clustering"
sidebar_label: "6 · Group documents"
sidebar_position: 2
slug: /theory/ir/document-classification-and-clustering
description: "How labelled categories differ from discovered clusters, with a nearest-centroid classifier and a small k-means loop."
tags: [information-retrieval, text-classification, clustering, k-means]
---

import Infographic from '@site/src/components/Infographic';
import DocumentGroupingLab from '@site/src/components/viz/DocumentGroupingLab';

**In one line.** Classification applies known labels to documents; clustering finds groups whose meaning must be inspected afterwards.

:::tip Before you start

**You should already know**

- How tf-idf vectors and cosine similarity work ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).
- The idea of supervised and unsupervised learning ([unsupervised learning](/docs/theory/ml/unsupervised-learning)).

**Reading time:** about 45 minutes, plus a minute to run the code. The first run downloads the 20 newsgroups archive.

**After this chapter you can**

- Work a Naive Bayes classification by hand with smoothing.
- Say what feature selection does to a text classifier, with numbers.
- Explain why k-means on raw tf-idf vectors disappoints and what fixes it.

:::

## In 30 seconds

A mail clerk sorts letters into labelled pigeonholes. That is classification: the pigeonholes exist before the letters arrive, and the clerk learns from examples of which letter goes where. A librarian handed a pile of unlabelled pamphlets, who sorts them into piles by similarity and names the piles afterwards, is doing clustering.

Both need documents turned into numbers. Classification has labels to learn from and can be scored against them. Clustering has none, so every pile needs a person to read it before it is trusted.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Label (category) | A known class assigned to a document | `sport`, `politics` |
| Centroid | The average vector of a group of documents | The mean tf-idf vector of all sport posts |
| Rocchio (nearest centroid) | Assign a document to the class whose centroid is closest | Cosine 0.7 to sport beats 0.4 to politics |
| Naive Bayes | Pick the class under which the document's words are most probable, assuming words are independent | $P(\text{sport}) \times P(\text{goal}\mid\text{sport}) \times \dots$ |
| Smoothing | Adding a small count so unseen words do not give probability zero | Add 1 to every word count |
| Feature selection | Keeping only the most useful words | The 2,000 best of 19,812 words by chi-squared |
| k-means | Repeatedly assign points to the nearest centre and move the centres | Six centres for six newsgroups |
| Adjusted Rand index (ARI) | How well clusters match true labels, 0 for chance, 1 for perfect | 0.455 in the experiment below |
| SVD (LSA) | Squashing many word dimensions into a few topic-like ones | 19,812 dimensions to 20 |


## The idea in plain words

Search does not always begin with a one-off query. A system may need to route each incoming document into a known topic, separate spam from wanted mail, or group a large collection so a person can explore it. These are related tasks because they use document representations, but their supervision differs.

**Classification** starts with labelled examples or predefined categories. A model learns to assign an incoming document to one or more of those categories. In a nearest-centroid example, a document has cosine similarity **0.7** to the sport centroid and **0.4** to the politics centroid, so it is assigned sport. This is a closed choice between known labels, assuming the product has decided that a single label is appropriate.

**Clustering** begins without category labels. It groups documents by a chosen similarity or distance. The groups may later turn out to correspond to sport and politics, or to something less obvious such as writing style, source or document length. A cluster number is not a semantic label. A person must inspect representative documents before naming it or using it as a navigation category.

<Infographic src="/img/ir/document-grouping.svg" alt="A document is assigned to the sport class because cosine 0.7 exceeds 0.4, while k-means discovers groups by assignment and centroid updates." caption="Supervised nearest-centroid classification and unsupervised k-means share a distance idea but answer different questions." />

Naive Bayes, Rocchio and k-nearest neighbours are common classification approaches; k-means and hierarchical clustering handle unlabelled groups. Rocchio's class centroid is an average vector from labelled examples; k-means centroids are repeatedly moved to the average of their current cluster members. The algorithms use similar geometry but the meaning of the centroids is different.

:::note Added for this site

The production example and evaluation guidance below extend the algorithm survey. They distinguish whether a grouping improves an actual retrieval or routing workflow from whether its geometry looks tidy.

:::

The classification view starts with **0.7 sport** and **0.4 politics** similarities, so the result is sport. Switch to clustering and run k-means steps to watch the unlabelled centres move.

<DocumentGroupingLab />

## Worked example, step by step

Four training documents: "goal match goal" and "match team" are `sport`; "vote law" and "law law vote" are `politics`. Classify the new document "goal law law" with Naive Bayes and add-one smoothing.

1. **Prior.** Two of four training documents are sport, so $P(\text{sport}) = P(\text{politics}) = 0.5$.
2. **Word counts.** Sport has goal 2, match 2, team 1: 5 words. Politics has vote 2, law 3: 5 words. The vocabulary has 5 words: goal, law, match, team, vote.
3. **Smoothed probabilities.** With add-one smoothing, $P(w \mid c) = (\text{count} + 1) / (5 + 5)$. Sport: goal 0.3, law 0.1. Politics: goal 0.1, law 0.4.
4. **Score each class.** Sport: $0.5 \times 0.3 \times 0.1 \times 0.1 = 0.0015$. Politics: $0.5 \times 0.1 \times 0.4 \times 0.4 = 0.008$.
5. **Normalise.** Politics gets $0.008 / (0.0015 + 0.008) = 0.842$ and sport gets 0.158. The document is classified `politics`: two strong political words outweigh one sport word.

In words: each class "explains" the words differently, and the class that explains them better wins. Smoothing means the word "team", unseen in politics, still gets a small non-zero probability. The first block below prints 0.842 and 0.158.

## How it works

### Supervised categories

Assign predefined categories from labelled data: Naive Bayes (probabilistic), Rocchio (nearest class centroid), k-NN. Evaluate with precision/recall/F1.

:::tip

**Worked.** cosine 0.7 to "sport" vs 0.4 to "politics" → classify sport.

:::

### Unsupervised groups

k-means: assign each doc to the nearest of k centroids → recompute → repeat. Hierarchical builds a dendrogram. The cluster hypothesis: similar docs answer the same queries.


## A real system that works this way

**Gmail's tabbed inbox** is a named production example of document classification. Google's [description of the feature](https://blog.google/products-and-platforms/products/gmail/gmail-ai-features/) says incoming messages are assigned to predefined tabs such as Primary, Promotions, Social, Updates and Forums using machine learning and other signals. A user can move a message when the chosen category is wrong. Those tabs have names and product meaning before a new message arrives, which is why this is classification rather than unsupervised clustering.

For a search product, classification can assign topic metadata to new documents. A query for `travel expenses` might be restricted to a travel-policy category, or results may be faceted so users can narrow to policies. The category can improve discovery only if its errors are understood. A relevant document misclassified outside the selected topic becomes invisible. If a classifier is uncertain, a product can show several candidate labels or avoid using the label as a hard retrieval filter.

Clustering serves a different workflow. An analyst can group a large set of search results to reveal themes without first defining them. The cluster hypothesis says documents that are similar tend to answer the same information needs, making grouping plausible. It is a hypothesis, not a guarantee: two documents may be similar because both contain a common template while addressing different policies.

## Code you can run

The first block reproduces the nearest-centroid decision. The cosine values are supplied measurements; the rule chooses the largest one. A real classifier would calculate them from the document and labelled class centroids.

```python
similarities = {"sport": 0.7, "politics": 0.4}
label = max(similarities, key=similarities.get)
print("sport cosine:", similarities["sport"])
print("politics cosine:", similarities["politics"])
print("assigned class:", label)
assert label == "sport"
```

The second block runs two-dimensional k-means on the same four points and initial centres as the lab. It makes no claims about semantic topic names. Each iteration assigns every point to the nearer centre, then moves each centre to the mean of its assigned points.

```python
from math import dist

points = [(0.1, 0.2), (0.2, 0.1), (0.85, 0.8), (0.75, 0.9)]
centres = [(0.1, 0.2), (0.25, 0.1)]

def assign(points, centres):
    return [min(range(len(centres)), key=lambda group: dist(point, centres[group])) for point in points]

def update(points, membership, centres):
    next_centres = []
    for group, old_centre in enumerate(centres):
        members = [point for point, assigned in zip(points, membership) if assigned == group]
        next_centres.append(tuple(sum(point[axis] for point in members) / len(members) for axis in range(2))
                            if members else old_centre)
    return next_centres

for step in range(3):
    membership = assign(points, centres)
    print("step", step, "groups", membership)
    centres = update(points, membership, centres)

assert assign(points, centres) == [0, 0, 1, 1]
```

The cluster numbers are arbitrary: swapping 0 and 1 would describe the same partition. K-means can converge to different groupings from different initial centres, so a serious use compares multiple initialisations and inspects the resulting documents.

### The worked example in code

This block fits scikit-learn's `MultinomialNB` on the four training documents.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = ["goal match goal", "match team", "vote law", "law law vote"]
labels = ["sport", "sport", "politics", "politics"]
vectoriser = CountVectorizer()
model = MultinomialNB(alpha=1.0).fit(vectoriser.fit_transform(docs), labels)
test = vectoriser.transform(["goal law law"])
print("vocabulary", list(vectoriser.get_feature_names_out()))
print("classes", model.classes_.tolist())
print("posterior", np.round(model.predict_proba(test)[0], 3).tolist())
print("prediction", model.predict(test)[0])
```

**Reading the output.** The vocabulary matches step 2, the classes are listed alphabetically (`politics` first), the posterior `[0.842, 0.158]` matches step 5 and the prediction is `politics`.

### An experiment on real newsgroup posts

Real posts are noisier. The block below uses six groups of the 20 newsgroups collection, chosen so that two pairs are easy to confuse: `rec.sport.hockey` and `rec.sport.baseball`, and `talk.politics.guns` and `talk.politics.mideast`, plus `sci.space` and `sci.med`. Headers, footers and quoted replies are removed, because otherwise the classifier can learn from email addresses and signatures. The training split has 3,494 posts and the test split 2,326, with 19,812 tf-idf features.

The supervised half scores Naive Bayes and a Rocchio nearest-centroid classifier while keeping only the top k words by chi-squared, with the selection fitted on the training posts only. The unsupervised half runs k-means on the test posts and scores it against the true groups, then repeats it after reducing the vectors with SVD.

scikit-learn's `NearestCentroid` measures Euclidean distance to each class mean, which on unit-length tf-idf vectors is close to, but not identical to, cosine similarity to the centroid.

Versions used: Python 3.14.6, scikit-learn 1.9.1. The run takes under 10 seconds once the data is cached.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid

categories = ["rec.sport.hockey", "rec.sport.baseball", "sci.space", "sci.med", "talk.politics.guns", "talk.politics.mideast"]
options = dict(categories=categories, remove=("headers", "footers", "quotes"))
train = fetch_20newsgroups(subset="train", **options)
test = fetch_20newsgroups(subset="test", **options)
vectoriser = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
x_train = vectoriser.fit_transform(train.data)
x_test = vectoriser.transform(test.data)
print(f"{len(train.data)} training and {len(test.data)} test posts, {x_train.shape[1]} features, {len(categories)} groups")

def report(name, model, a, b):
    model.fit(a, train.target)
    predicted = model.predict(b)
    print(f"{name:34}accuracy {np.mean(predicted == test.target):.3f}  macro-F1 {f1_score(test.target, predicted, average='macro'):.3f}")

print("Supervised, by number of chi-squared-selected features")
for k in (20, 100, 500, 2000, x_train.shape[1]):
    selector = SelectKBest(chi2, k=k).fit(x_train, train.target)
    a, b = selector.transform(x_train), selector.transform(x_test)
    report(f"Naive Bayes, {k} features", MultinomialNB(alpha=0.1), a, b)
    report(f"Rocchio centroid, {k} features", NearestCentroid(), a, b)

def purity(labels, truth):
    return sum(np.bincount(truth[labels == c]).max() for c in np.unique(labels)) / len(truth)

print("Unsupervised: k-means (k=6) on test posts, 10 random starts")
scores = []
for seed in range(10):
    labels = KMeans(6, n_init=1, init="random", random_state=seed).fit_predict(x_test)
    scores.append((adjusted_rand_score(test.target, labels), normalized_mutual_info_score(test.target, labels), purity(labels, test.target)))
scores = np.array(scores)
print(f"ARI mean {scores[:, 0].mean():.3f} (min {scores[:, 0].min():.3f}, max {scores[:, 0].max():.3f}); NMI {scores[:, 1].mean():.3f}; purity {scores[:, 2].mean():.3f}")
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

for dims in (20, 100):
    reduced = make_pipeline(TruncatedSVD(dims, random_state=0), Normalizer(copy=False)).fit_transform(x_test)
    ari = [adjusted_rand_score(test.target, KMeans(6, n_init=1, init="random", random_state=s).fit_predict(reduced)) for s in range(10)]
    print(f"k-means after SVD to {dims} dimensions: ARI mean {np.mean(ari):.3f} (min {min(ari):.3f}, max {max(ari):.3f})")
```

The output of the run:

```text
3494 training and 2326 test posts, 19812 features, 6 groups
Supervised, by number of chi-squared-selected features
Naive Bayes, 20 features          accuracy 0.325  macro-F1 0.336
Rocchio centroid, 20 features     accuracy 0.374  macro-F1 0.381
Naive Bayes, 100 features         accuracy 0.605  macro-F1 0.638
Rocchio centroid, 100 features    accuracy 0.620  macro-F1 0.637
Naive Bayes, 500 features         accuracy 0.791  macro-F1 0.800
Rocchio centroid, 500 features    accuracy 0.760  macro-F1 0.769
Naive Bayes, 2000 features        accuracy 0.852  macro-F1 0.853
Rocchio centroid, 2000 features   accuracy 0.810  macro-F1 0.816
Naive Bayes, 19812 features       accuracy 0.871  macro-F1 0.872
Rocchio centroid, 19812 features  accuracy 0.829  macro-F1 0.832
Unsupervised: k-means (k=6) on test posts, 10 random starts
ARI mean 0.166 (min 0.000, max 0.235); NMI 0.310; purity 0.450
k-means after SVD to 20 dimensions: ARI mean 0.455 (min 0.420, max 0.548)
k-means after SVD to 100 dimensions: ARI mean 0.330 (min 0.276, max 0.424)
```

**Reading the output.** Accuracy is the share of test posts labelled correctly; macro-F1 averages the per-class F1, so a weak class cannot hide. With six classes of similar size, guessing scores about 0.17. For k-means, ARI is 0 for random labelling and 1 for a perfect match, NMI (normalised mutual information) measures shared information and purity is the share of posts in the majority class of their cluster.

**Line by line.**

- `fetch_20newsgroups(... remove=("headers", "footers", "quotes"))` strips the parts of a post that identify its group without describing its topic.
- `SelectKBest(chi2, k=k).fit(x_train, train.target)` scores each word by its dependence on the class using training labels only, then `transform` applies the same selection to the test posts. Fitting the selector on test data would leak labels.
- `KMeans(6, n_init=1, init="random", ...)` runs ten single random starts so the spread shows. The default of several starts per fit would hide it.
- `make_pipeline(TruncatedSVD(dims), Normalizer())` reduces the vectors and rescales them to unit length, so Euclidean k-means behaves like cosine k-means.

### What the numbers say

Feature selection did not help accuracy. Naive Bayes was best with all 19,812 features (0.871), and every smaller set scored lower: 0.852 at 2,000, 0.791 at 500, 0.605 at 100 and 0.325 at 20. Keeping a tenth of the features cost 0.019 and kept the model much smaller, which is the real reason to select. Naive Bayes beat Rocchio at 500 features and above (0.871 against 0.829 with everything), while Rocchio was ahead with 100 or fewer.

The clustering result is the surprise. On raw tf-idf, ten random starts of k-means gave a mean ARI of 0.166, with one start no better than chance (0.000) and the best at 0.235. After reducing to 20 SVD dimensions the mean rose to 0.455 and even the worst start (0.420) beat the best raw start. At 100 dimensions it was 0.330. Labels buy a great deal: the supervised classifier reached 0.871 accuracy, while the best unsupervised grouping matched the real groups only moderately.

Limits: six hand-picked groups, one split, no tuning of the smoothing value (0.1), no cross-validation, and an SVD size chosen by trying two values. The scikit-learn documentation reports the same direction for LSA before k-means on a different text collection, with ARI rising from 0.203 to 0.317.

<Infographic src="/img/ir-enrich/ir1-feature-selection.svg" alt="Left, accuracy of Naive Bayes and Rocchio for five feature counts. Right, k-means adjusted Rand index for raw tf-idf and two SVD sizes." caption="Look first at the left column of numbers: accuracy rises with every extra feature. Then the right bars: SVD nearly triples k-means quality." />

## Designing with it

### Decide whether the labels already exist

Use classification when the product has a stable taxonomy and labelled examples. Use clustering when the goal is exploration or no agreed labels exist. If the problem is to retrieve the best passages for a query, neither task automatically replaces ranking: classification may narrow a candidate set and clustering may organise it, but relevance to a particular question still needs to be measured.

| Product question | Better starting task | Evidence needed |
| --- | --- | --- |
| "Which inbox tab should this message enter?" | Classification | Labelled examples and per-class errors |
| "What themes occur in these unfamiliar reports?" | Clustering | Representative documents and human inspection |
| "Which report answers this query?" | Retrieval ranking | Query-document relevance judgements |
| "Which topic facets help users narrow results?" | Classification or inspected clusters | User behaviour and missed-result checks |

The representation is as consequential as the algorithm. A tf-idf vector highlights exact vocabulary; an embedding can capture paraphrases but may group documents by broad theme while missing a precise distinction. Long documents with several topics can straddle clusters; chunk-level representations may be more coherent, but then the product must explain how chunk groups map back to whole documents. Choose the document unit before judging the grouping.

### Evaluate a classifier with the product's cost

Accuracy can hide a weak minority class. If 90% of messages are ordinary updates, a classifier that always predicts Updates has high accuracy and terrible value for finding rare security alerts. Inspect precision, recall and F1 per class, the confusion matrix and error examples. A spam filter may favour catching spam while avoiding false positives that hide important mail; a policy-topic tagger may prefer a different balance. For a multi-label taxonomy, a document can legitimately be both finance and travel, so a forced one-label model can be structurally wrong.

Keep training and test data separated by time or source where the workflow demands it. Near-duplicate documents in both sets can make scores look excellent while the model fails on genuinely new material. Monitor category proportions and error samples after launch as document language and topics change.

### Evaluate clusters as an aid to retrieval

K-means minimises within-cluster squared distance under its representation. That objective does not know whether the resulting groups help a user. Inspect the top terms and representative documents of each cluster, test stability across seeds and compare clusters with any available labels only as a diagnostic. For result grouping, ask whether users find relevant items faster, whether important results become hidden in a poorly named group, and whether the chosen number of clusters fits the interface. A clean geometric plot is not a substitute for those checks.

Hierarchical clustering yields a tree instead of a fixed flat partition. It can support drill-down exploration, but its link rule and distance measure influence every branch. K-means needs a chosen $k$; a hierarchy needs a chosen cut or navigation depth. Both choices should be reviewed with real documents, not inferred from the algorithm's name.

## Follow one document through both tasks

Take a short article about a football club's election for a new chair. A supervised sport classifier may assign sport because its labelled examples include clubs, teams and matches. A politics classifier may also fire on `election`, depending on the representation and labels. If the taxonomy allows one label only, the team must decide which information need the label serves. If users search for both sport and governance coverage, a multi-label result may be more useful than forcing a winner.

Now place the same article into an unlabelled collection containing match reports, club finances, national elections and company board changes. A clustering algorithm may group it with club finances because of shared organisation words, with national elections because of voting language, or with match reports because of football names. None of these groupings is automatically wrong in the abstract; each reflects the features and distance measure. The right question is whether the grouping helps the intended browsing or retrieval task.

### See the geometry behind Rocchio

For a labelled class, a centroid is the average of its training document vectors. A new document is compared with each class centroid, often by cosine. The values 0.7 and 0.4 are already similarity values, so the decision is sport. A production classifier must also consider whether 0.7 is strong enough at all. If both class similarities were 0.1, choosing the larger one would still return a label even though neither class may fit. An abstain or "other" path protects a closed taxonomy from silently absorbing unfamiliar topics.

Naive Bayes reaches the classification decision differently: it estimates how likely the observed terms are under each class, with smoothing for unseen terms. k-nearest neighbours compares the new document with labelled examples rather than only class averages. A nearest-centroid method is compact and explainable but can blur a class with several separate subtopics. None of these methods is universally best; evaluate them on the same labels, query tasks and time-separated data.

### See the loop behind k-means

The lab's initial centres are intentionally imperfect: both start near the lower-left points. In the first assignment, the second lower-left point forms one group with both upper-right points. Recomputing means moves that group's centre toward the upper right. A later assignment joins the two lower-left points and separates the two upper-right points. This is why one iteration is not the final answer and why initialisation matters. The objective can settle in a local minimum; repeated runs help reveal instability.

The method also assumes that a mean in the chosen vector space is meaningful. For sparse tf-idf vectors, normalisation and similarity choice affect the interpretation. For mixed text and metadata, simply averaging arbitrary numeric encodings can be nonsensical. Decide which fields express semantic proximity and which fields are filters. A document's department code might matter for access control but should not necessarily pull two unrelated policy texts into one topic cluster.

### Keep feedback from becoming a hidden label change

When users move messages between Gmail tabs or correct a topic tag, that feedback can help improve future classification. It can also be noisy: a user may move a message for workflow reasons unrelated to its topic. Record the meaning of each feedback action, sample corrected cases and avoid treating every click as ground truth. Similarly, if a cluster is named "Travel" by one analyst, the name is an interpretation of the current members. When the corpus changes, the same numeric cluster may drift, split or disappear. A stable product taxonomy should be maintained deliberately rather than inherited from raw cluster numbers.

If a category becomes a search filter, its mistakes have a measurable recall cost. Sample queries whose known relevant documents were filtered out and compare that loss with the navigation benefit. A soft category boost can be safer than a hard filter when the classifier is uncertain, provided permissions remain a separate exact constraint. This is a useful design distinction: a topic label is a prediction that may be wrong; an access rule is a policy that must be enforced. A cluster label is even less certain until people review its members. Keep those meanings visible in the interface and in logs so later engineers do not accidentally promote a convenient grouping into a source of truth.

## Where this stands in 2026

:::info Industry view

- Document classification remains a practical component of user-facing products: Gmail publicly describes ML-based routing into predefined inbox tabs.
- Embeddings can improve semantic grouping, but a cluster still requires human interpretation before it becomes a trustworthy navigation label.
- For search, category filters and clusters are supporting structures. The final result order needs query-specific relevance evaluation.

:::

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Selecting features on the whole data before splitting | It is one tidy preprocessing step | Fit the selector on training data only, as the experiment does. Chi-squared on test labels leaks the answer |
| Expecting feature selection to raise accuracy | Fewer features sound like less noise | Here accuracy fell with every cut. Select to shrink the model or speed up serving, then check the loss |
| Running k-means on raw tf-idf with one start | It is the standard recipe | Reduce dimensions first (SVD) and compare several starts. Raw starts ranged from ARI 0.000 to 0.235 |
| Reading cluster numbers as labels | Cluster 3 looks like a category | Inspect each cluster's top terms and sample posts before naming it |
| Judging a classifier by accuracy alone | One number is easy to report | Look at macro-F1 and a confusion matrix. A minority class can fail while accuracy looks fine |

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast classification and clustering for IR.</summary>

Classification is supervised; assign predefined categories from labelled data; clustering is unsupervised; group documents without labels.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> How does Rocchio classification work?</summary>

Compute a centroid (mean tf-idf vector) per class and assign a document to the nearest centroid (largest cosine).<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> A document has cosine 0.7 to the 'sport' centroid and 0.4 to 'politics'. Its class?</summary>

Nearest (largest cosine) centroid → sport.<br /><em>Session 6 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Describe the k-means clustering loop.</summary>

Assign each document to the nearest of k centroids → recompute each centroid as its members' mean → repeat until stable.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> State the cluster hypothesis and why it matters for IR.</summary>

Documents in the same cluster tend to be relevant to the same queries; motivating cluster-based retrieval and result grouping.<br /><em>Session 6 · conceptual</em>

</details>

<details>
<summary><strong>Q6.</strong> (Medium) Naive Bayes scored 0.871 with all features and 0.852 with 2,000. Why might you still ship the 2,000-feature model?</summary>

It gives up 0.019 of accuracy in return for a model about a tenth of the size, with faster scoring and a smaller footprint. Whether that trade is worth it depends on the product, so measure the loss on your own data first. The experiment shows only that accuracy did not improve from selection.

</details>

<details>
<summary><strong>Q7.</strong> (Stretch) K-means on raw tf-idf had mean ARI 0.166, but after SVD to 20 dimensions it rose to 0.455. What does the SVD change, and why is 100 dimensions worse than 20?</summary>

In the raw space most of the 19,812 dimensions are rare words, and distances between sparse vectors become nearly uniform, which gives k-means little structure to find. SVD keeps the directions that carry most of the variation, which resemble topics, and normalising makes the distance a cosine. With 100 dimensions more of the noisy, post-specific directions come back, so quality falls to 0.330. The best size must be tried, not assumed.

</details>

## Go deeper

- [Stanford IR book: text classification](https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html); labelled document routing and classifier families.
- [Stanford IR book: flat clustering](https://nlp.stanford.edu/IR-book/html/htmledition/flat-clustering-1.html); k-means and the cluster hypothesis.
- [Google's account of Gmail's tabbed inbox](https://blog.google/products-and-platforms/products/gmail/gmail-ai-features/); an official product example.
- [scikit-learn: clustering text documents using k-means](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html); shows LSA improving k-means on text and explains it by the curse of dimensionality.
- [scikit-learn: the 20 newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset); the loader, the by-date train and test split and the `remove` option.
- Built from the course lecture "ir-s6-classification-clustering" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can distinguish a known category from an unlabelled cluster.
- [ ] I can assign the example document to sport from cosine values 0.7 and 0.4.
- [ ] I can run one k-means assignment and centroid-update step.
- [ ] I can choose evaluation measures for category routing and explain why cluster geometry alone is insufficient.
- [ ] I can work a Naive Bayes classification by hand with add-one smoothing.
- [ ] I can say, with numbers, why chi-squared feature selection shrank the model but did not raise accuracy.
- [ ] I can explain why k-means on raw tf-idf was unstable and how SVD stabilised it, and I know ARI 0.455 is still only a moderate match.

## Where to go next

Next: [Session 7, evaluating ranked retrieval](/docs/theory/ir/evaluating-ranked-retrieval), which adds intervals and paired tests to scores like these. Related: [Bayesian learning](/docs/theory/ml/bayesian-learning), the probability behind Naive Bayes.
