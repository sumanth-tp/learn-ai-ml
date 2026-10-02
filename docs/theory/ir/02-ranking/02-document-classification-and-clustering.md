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

## The idea in plain words

Search does not always begin with a one-off query. A system may need to route each incoming document into a known topic, separate spam from wanted mail, or group a large collection so a person can explore it. These are related tasks because they use document representations, but their supervision differs.

**Classification** starts with labelled examples or predefined categories. A model learns to assign an incoming document to one or more of those categories. In the lecture's nearest-centroid example, a document has cosine similarity **0.7** to the sport centroid and **0.4** to the politics centroid, so it is assigned sport. This is a closed choice between known labels, assuming the product has decided that a single label is appropriate.

**Clustering** begins without category labels. It groups documents by a chosen similarity or distance. The groups may later turn out to correspond to sport and politics, or to something less obvious such as writing style, source or document length. A cluster number is not a semantic label. A person must inspect representative documents before naming it or using it as a navigation category.

<Infographic src="/img/ir/document-grouping.svg" alt="A document is assigned to the sport class because cosine 0.7 exceeds 0.4, while k-means discovers groups by assignment and centroid updates." caption="Supervised nearest-centroid classification and unsupervised k-means share a distance idea but answer different questions." />

The lecture names Naive Bayes, Rocchio and k-nearest neighbours as classification approaches, plus k-means and hierarchical clustering for unlabelled groups. Rocchio's class centroid is an average vector from labelled examples; k-means centroids are repeatedly moved to the average of their current cluster members. The algorithms use similar geometry but the meaning of the centroids is different.

:::note Beyond the lecture

The production example and evaluation guidance below extend the lecture's algorithm survey. They distinguish whether a grouping improves an actual retrieval or routing workflow from whether its geometry looks tidy.

:::

The classification view starts with the lecture's **0.7 sport** and **0.4 politics** similarities, so the result is sport. Switch to clustering and run k-means steps to watch the unlabelled centres move.

<DocumentGroupingLab />

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

The first block reproduces the lecture's nearest-centroid decision. The cosine values are the lecture's supplied measurements; the rule chooses the largest one. A real classifier would calculate them from the document and labelled class centroids.

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

For a labelled class, a centroid is the average of its training document vectors. A new document is compared with each class centroid, often by cosine. The lecture's 0.7 and 0.4 are already similarity values, so the decision is sport. A production classifier must also consider whether 0.7 is strong enough at all. If both class similarities were 0.1, choosing the larger one would still return a label even though neither class may fit. An abstain or "other" path protects a closed taxonomy from silently absorbing unfamiliar topics.

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

## Go deeper

- [Stanford IR book: text classification](https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html); labelled document routing and classifier families.
- [Stanford IR book: flat clustering](https://nlp.stanford.edu/IR-book/html/htmledition/flat-clustering-1.html); k-means and the cluster hypothesis.
- [Google's account of Gmail's tabbed inbox](https://blog.google/products-and-platforms/products/gmail/gmail-ai-features/); an official product example.
- Built from the course lecture "ir-s6-classification-clustering" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can distinguish a known category from an unlabelled cluster.
- [ ] I can assign the lecture's document to sport from cosine values 0.7 and 0.4.
- [ ] I can run one k-means assignment and centroid-update step.
- [ ] I can choose evaluation measures for category routing and explain why cluster geometry alone is insufficient.
