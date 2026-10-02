---
id: cv-visual-bag-of-words
title: "Computer Vision · Session 15; Visual Bag of Words"
sidebar_label: "2 · Visual bag of words"
sidebar_position: 2
slug: /theory/cv/visual-bag-of-words
description: "Build a visual vocabulary, count local descriptors, normalise a histogram and understand what spatial pooling loses."
tags: [computer-vision, visual-words, features, retrieval]
---

import Infographic from '@site/src/components/Infographic';
import VisualWordsLab from '@site/src/components/viz/VisualWordsLab';

**In one line.** A visual bag of words converts a variable number of local image features into one fixed-length count vector.

## The idea in plain words

:::note Beyond the lecture

The codebook leakage check, weighting caveats, comparison with learned representations and design discussion extend the lecture. Its pipeline, worked histogram and five practice questions remain below.

:::

The preceding feature chapters produce local descriptors at a variable number of keypoints. One image may contain ten reliable features, another thousands. A conventional classifier or nearest-neighbour index often expects one vector of a fixed dimension per image. The visual bag-of-words approach solves that shape problem by defining a vocabulary of representative descriptors, assigning each observed local descriptor to a word and counting the assignments. The result is a histogram whose length is the codebook size, even though the number of local observations varies.

The language analogy is useful but limited. A text document can count words without retaining their order. Here, a “word” is a cluster centre in a numerical descriptor space, not a human-named object or a semantic concept. A SIFT feature from a wheel and one from a circular logo might land near the same centre. The image histogram records that similar local patterns appeared, not why they appeared or where they were. Calling a codebook entry a “car part” without evidence would over-interpret the representation.

Build the codebook using descriptors from training images only. The usual teaching pipeline detects local features, extracts descriptors, samples a training subset, fits k-means, assigns each descriptor to its closest centre and counts word assignments per image. If the codebook is fit on the held-out evaluation images, the evaluation has already influenced the feature space, even when labels were hidden. The distance measure and descriptor normalisation affect assignments; a vocabulary built from one camera or domain may encode another domain poorly.

For the lecture's three-word example, counts $[4,1,3]$ sum to eight. Dividing by eight gives $[0.5,0.125,0.375]$. This term-frequency normalisation makes two images with the same proportions comparable even if one yielded more keypoints. It also discards absolute feature density, which may carry useful information in some tasks. A codebook with 500 centres yields a 500-dimensional whole-image histogram. A spatial pyramid concatenates histograms from several regions, so its dimension is **larger** than 500 if it contains more than one region. The image size does not determine that vector dimension.

<Infographic src="/img/cv/visual-words.svg" alt="Visual bag-of-words board: local descriptors are clustered into a vocabulary, counts four one three become frequencies point five point one two five point three seven five, and a spatial pyramid adds regional counts." caption="Fixed-length histograms simplify comparison; whole-image counts discard feature location." />

## How it works

### Codebook & histogram

Extract SIFT features → cluster with k-means into a visual vocabulary → assign each feature to its nearest word → count into a fixed-length histogram.

:::tip

**Worked.** counts [4,1,3] (total 8) → tf [0.5, 0.125, 0.375]; 500 words → 500-dim descriptor.

:::

### Spatial pyramid & concepts

A spatial pyramid adds coarse layout; grouping words into concepts builds a semantic hierarchy (parts → objects → scenes). Ideas persist in VLAD/pooling.

### Key takeaways

## A real system that works this way

Image search is a natural use case. A system can extract local descriptors from each catalogue image, quantise them against one trained codebook and index the resulting histograms. A new query is processed with the same detector, descriptor, codebook and weighting rules, then compared with stored histograms. This is closely related to the [recommendation-as-personalised-retrieval chapter](/docs/theory/ir/recommendation-as-personalised-retrieval): both systems turn source data into a comparable representation, retrieve candidates and rank them for a task. The visual words represent image patterns, while a recommender may also use user context and interaction history.

The lecture mentions term frequency and inverse document frequency. In visual retrieval, term frequency may be the raw or normalised count of a visual word in one image. Document frequency counts how many training images contain that word. An inverse-document-frequency factor can lower the weight of words that occur in almost every image, such as common background texture. It does not magically make every rare word useful: a rare detector artifact can receive a large weight. Fit the statistics on the training corpus, define the smoothing rule and compare retrieval quality on held-out queries. The toy code below demonstrates the exact frequency calculation only; it does not claim a trained image-search engine.

The source also mentions spatial pyramids. Imagine two images containing exactly four instances of word 1, one of word 2 and three of word 3. In one image, word 1 is concentrated on the left; in the other it is on the right. A whole-image histogram is identical. Divide each image into a left and right region, count separately and concatenate, and the two representations can differ. The improvement is conditional: a fixed grid can be brittle when objects move or cameras crop differently. The spatial subdivision and weighting are design choices to evaluate.

## Code you can run

The first block reproduces the lecture's histogram. The vector has three entries because this toy codebook has three words. Replacing the codebook with 500 words would yield 500 entries before any spatial subdivision. The assignment list is explicit so the count step can be checked without requiring image downloads.

```python
from collections import Counter

assignments = [0, 0, 0, 0, 1, 2, 2, 2]
vocabulary_size = 3
counts = Counter(assignments)
histogram = [counts[word] for word in range(vocabulary_size)]
total = sum(histogram)
term_frequency = [count / total for count in histogram]
print('Counts:', histogram)
print('Term frequency:', term_frequency)
print('Dimension:', len(histogram))
assert histogram == [4, 1, 3]
assert term_frequency == [0.5, 0.125, 0.375]
assert len(histogram) == vocabulary_size
```

The lab starts with the same counts and frequencies. Change each count to see how the normalised vector changes. The all-zero case has no defined frequency distribution, which the lab labels explicitly. A production system needs a policy for an image with no detected descriptors, such as a zero vector plus a missing-feature flag or a fallback representation.

<VisualWordsLab />

The second block demonstrates what global pooling loses. Both images have the same whole-image histogram, while their left-right concatenated histograms differ. The region labels are supplied in this construction; a real image pipeline obtains them from feature coordinates and a defined grid.

```python
from collections import Counter

image_a = {'left': [0, 0, 0, 0, 1], 'right': [2, 2, 2]}
image_b = {'left': [2, 2, 2], 'right': [0, 0, 0, 0, 1]}

def vector(parts):
    global_counts = Counter(word for region in parts.values() for word in region)
    whole = [global_counts[word] for word in range(3)]
    spatial = [Counter(parts[region])[word] for region in ('left', 'right') for word in range(3)]
    return whole, spatial

whole_a, spatial_a = vector(image_a)
whole_b, spatial_b = vector(image_b)
print('Whole-image vectors:', whole_a, whole_b)
print('Regional vectors:', spatial_a, spatial_b)
assert whole_a == whole_b == [4, 1, 3]
assert spatial_a != spatial_b
assert len(spatial_a) == 6
```

The six-value spatial vector comes from two regions times three visual words. Adding a whole-image level as well would make nine values in this toy pyramid. Normalise and weight each level deliberately; a simple concatenation can let a level with many cells dominate a similarity measure. The example isolates representation shape, not retrieval performance.

## Designing with it

Start with the source of local features. Classical pipelines often use SIFT because it offers a repeatable detector and 128-value descriptor under several transformations. The [SIFT chapter](/docs/theory/cv/sift-keypoints-and-descriptors) explains its approximate invariances and matching limits. A low-texture image may have few keypoints, while repeated patterns may have many nearly identical descriptors. Decide how to handle both cases before tuning the codebook. For example, sample at most a fixed number of descriptors per image while fitting the vocabulary so a few textured images do not dominate k-means.

Treat vocabulary size as a model capacity choice. A very small codebook merges visually different patterns and loses discrimination. A very large one can make each image histogram sparse and make assignments unstable under noise or minor appearance changes. Size also affects storage, assignment cost and index design. Choose it by held-out task performance and operational budget, not by assuming more bins are always better. Record the sampling seed, training images, descriptor settings and fitted centres so the transformation can be reproduced.

Use the same preprocessing path for queries and indexed images. If one side extracts SIFT after aggressive resizing and the other side uses full resolution, keypoint populations may differ. If one side uses a newer codebook, histogram coordinates no longer refer to the same visual words. Version the detector, descriptor, vocabulary and weighting as a single pipeline. Rebuilding a codebook requires re-encoding indexed images or carefully handling incompatible representations.

Choose weighting for the retrieval objective. Raw counts reward feature-rich images; term-frequency normalisation removes that volume difference. TF-IDF can suppress ubiquitous patterns, but its document frequency should come from the indexed training corpus and needs an explicit zero-frequency smoothing rule. Cosine similarity on normalised vectors and histogram-intersection similarity treat large bins differently. Evaluate actual query relevance, including cases with backgrounds, partial occlusion and repeated texture, rather than relying only on vector distances.

Use spatial information when position matters. A whole-image bag deliberately ignores arrangement, which can be helpful under translation and crop variation but harmful when two scenes share the same texture in different locations. Spatial pyramids restore coarse layout, at the cost of a larger vector and more sensitivity to shifts. A grid aligned to a camera frame may work in a fixed inspection station; an object-centred crop may be better if camera placement varies. Test both, especially when the object occupies a small fraction of the image.

Finally, compare with learned global embeddings for the actual task. Modern image encoders can produce one vector directly, often with semantic information that hand-built visual words do not capture. They also have training-data, domain-shift and compute considerations. A bag-of-visual-words baseline remains useful for understanding quantisation, histogram weighting and retrieval behaviour, and can be effective in controlled settings. A fair comparison uses the same evaluation queries, index scale, latency constraints and failure inspection. The source outline names VLAD and pooling as related ideas; they are different aggregation rules and are not implemented by the count code here.

## Where this stands in 2026

:::info Industry view

- Visual words are a classical, interpretable image-representation method. The arithmetic here is verified locally; no modern ranking or benchmark against learned image embeddings is claimed.
- The feature extraction and matching context is supported by the official OpenCV SIFT and feature-homography tutorials opened on 2026-10-02. The chapter code uses only standard-library collections.
- A visual word is a descriptor-cluster index. Treating it as an object concept requires separate evidence; the lecture’s “parts to objects to scenes” language describes a possible higher-level interpretation, not a guarantee of k-means.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Describe the bag-of-visual-words pipeline.</summary>

Extract local features (SIFT), cluster them with k-means into a visual vocabulary, assign each feature to its nearest word, and build a histogram of word counts per image.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What determines the dimension of a BoVW descriptor?</summary>

The vocabulary (codebook) size; e.g. 500 words → a 500-dimensional histogram, fixed regardless of feature count.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> For a 3-word codebook with counts [4,1,3], give the normalised histogram.</summary>

Total 8 → [4/8, 1/8, 3/8] = [0.5, 0.125, 0.375].<br /><em>Session 15 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> Why apply tf-idf weighting to visual words?</summary>

To down-weight common, uninformative visual words and emphasise distinctive ones; exactly as in text retrieval.<br /><em>Session 15 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> What does a spatial pyramid add to BoVW?</summary>

Coarse spatial layout; histograms computed over image sub-regions, which plain BoVW discards.<br /><em>Session 15 · conceptual</em>

</details>

## Further reading

- [OpenCV SIFT tutorial](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) for the local features that can feed a visual vocabulary.
- [OpenCV feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) for local-descriptor distances and matching context.
- Built from the course lecture "cv-s15-visual-bag-of-words" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source terminology

The lecture says that grouping visual words into concepts builds a hierarchy of parts, objects and scenes. A k-means visual word is only a cluster of descriptor values. It does not acquire a semantic name from clustering alone. A supervised or otherwise validated grouping would be needed to support a parts-to-objects interpretation. The lecture's codebook and histogram method remains valid without that extra interpretation.

:::

## Diagnosing a visual-word result

Suppose retrieval returns many images of brick walls for a query showing a brick house, but misses houses with smooth walls. This can happen if repeated brick texture produces numerous local features and dominates the histogram. Inspect which visual words carry the similarity, how many descriptors each image contributes and whether term-frequency normalisation or TF-IDF changes the ranking. The issue may be that the representation emphasises texture while the product needs building shape or semantics. Adding more vocabulary words may refine brick variants without solving the missing-house problem.

In another failure, two logos with the same colours and motifs but reversed arrangement have identical whole-image word counts. The six-value spatial example shows why: all counts survive the swap, but their regions do not. A spatial pyramid may separate these images if their positions are reasonably aligned. If the logos rotate freely or move within a larger scene, a fixed left-right grid may fail differently. Test position, scale and rotation variation before selecting a spatial layout.

A third case has an empty descriptor set after a low-light capture. Dividing counts by zero is not a meaningful normalisation. Decide whether to mark the image as low quality, use an alternative representation, or send it for review. The lab exposes this edge case directly. If a system silently emits an all-zero vector, downstream nearest-neighbour search may return arbitrary images with the same empty vector, creating a false appearance of a successful retrieval.

Check for data leakage in the unsupervised preprocessing too. A vocabulary fit on all available images, including the test collection, has seen the test distribution. This may be acceptable for a clearly defined transductive task, but it is not the same evaluation as learning a reusable pipeline for unseen future images. In the latter setting, fit centres and document frequencies on training data only. Store the fitted transformation and apply it unchanged to validation and test images. Changing the vocabulary during evaluation changes the meaning of every coordinate.

Vector distance should be interpreted with its weighting. The raw count vector `[4,1,3]` is twice `[2,0.5,1.5]`, and normalisation removes that scale difference, but only when fractional counts are meaningful as a comparison device. Euclidean distance on counts, cosine distance on TF-IDF and histogram intersection on nonnegative normalised values can rank pairs differently. Choose the measure using relevance labels and error inspection. A useful index must also be efficient enough for the target corpus, but a speed claim requires measuring the actual index and hardware.

Finally, recognise the representation boundary. The histogram can answer how often learned local patterns occur. It cannot prove which object made them, identify a relation between two parts or locate a specific match without retaining or rechecking local coordinates. A search system may use the histogram for candidate retrieval and then verify spatial correspondences using feature matching and RANSAC. That two-stage design uses the bag for speed and geometry for a stricter decision. The bag's loss of layout is a deliberate trade-off, not a bug that can be repaired by changing a normalisation constant.

## Check yourself

- I can describe vocabulary training, descriptor assignment and per-image counting in order.
- I can normalise counts `[4,1,3]` and state when that division is undefined.
- I can explain why codebook size controls whole-image histogram dimension.
- I can give an example where spatial pooling distinguishes two otherwise identical bags.
