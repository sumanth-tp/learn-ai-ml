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

:::tip Before you start
**You should already know**

- What a SIFT descriptor is and what the ratio test does: [SIFT keypoints and descriptors](/docs/theory/cv/sift-keypoints-and-descriptors).
- How k-means groups points: [unsupervised learning](/docs/theory/ml/unsupervised-learning).
- How term frequency and inverse document frequency weight words: [vector space and term weighting](/docs/theory/ir/vector-space-and-term-weighting).

**Reading time.** About 50 minutes, plus about a minute to run the large experiment.

**After this chapter you can**

- turn a set of local descriptors into a fixed-length histogram, by hand,
- say how vocabulary size, tf-idf weighting and the origin of the codebook change retrieval accuracy, with measured numbers,
- explain why a larger vocabulary stops being a vocabulary, and when to prefer direct matching.
:::

## In 30 seconds

A text search engine counts words. A photograph has no words, but it has thousands of small patches. Group all the patches you have ever seen into, say, 1,000 typical shapes, and give each shape a number. Now any image becomes a count of how many of its patches look like shape 1, shape 2 and so on. Two images of the same place give similar counts, so retrieving a picture becomes comparing histograms. This chapter builds that pipeline and measures how the size of the shape list, the weighting and the training data of the list change retrieval accuracy.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Local descriptor | Numbers describing a small patch | A 128-value SIFT vector |
| Codebook or vocabulary | A list of typical descriptors, found by k-means | 1,000 centres |
| Visual word | The index of the nearest centre | Word 17 |
| Histogram | Counts of visual words in one image | $[4,1,3]$ |
| Term frequency | Counts divided by their total | $[0.5,0.125,0.375]$ |
| Inverse document frequency | A weight that is small for words found in many images | 1.000 for a word in every image |
| Top-1 accuracy | The share of queries whose best match is the right scene | 0.87 |
| Gallery and query | Indexed images and the images used to search | 36 and 60 |

## The idea in plain words

:::note Added to the course material

The codebook leakage check, weighting caveats, comparison with learned representations and design discussion go beyond the course notes. The pipeline, worked histogram and five practice questions remain below.

:::

The preceding feature chapters produce local descriptors at a variable number of keypoints. One image may contain ten reliable features, another thousands. A conventional classifier or nearest-neighbour index often expects one vector of a fixed dimension per image. The visual bag-of-words approach solves that shape problem by defining a vocabulary of representative descriptors, assigning each observed local descriptor to a word and counting the assignments. The result is a histogram whose length is the codebook size, even though the number of local observations varies.

The language analogy is useful but limited. A text document can count words without retaining their order. Here, a “word” is a cluster centre in a numerical descriptor space, not a human-named object or a semantic concept. A SIFT feature from a wheel and one from a circular logo might land near the same centre. The image histogram records that similar local patterns appeared, not why they appeared or where they were. Calling a codebook entry a “car part” without evidence would over-interpret the representation.

Build the codebook using descriptors from training images only. The usual teaching pipeline detects local features, extracts descriptors, samples a training subset, fits k-means, assigns each descriptor to its closest centre and counts word assignments per image. If the codebook is fit on the held-out evaluation images, the evaluation has already influenced the feature space, even when labels were hidden. The distance measure and descriptor normalisation affect assignments; a vocabulary built from one camera or domain may encode another domain poorly.

For a three-word example, counts $[4,1,3]$ sum to eight. Dividing by eight gives $[0.5,0.125,0.375]$. This term-frequency normalisation makes two images with the same proportions comparable even if one yielded more keypoints. It also discards absolute feature density, which may carry useful information in some tasks. A codebook with 500 centres yields a 500-dimensional whole-image histogram. A spatial pyramid concatenates histograms from several regions, so its dimension is **larger** than 500 if it contains more than one region. The image size does not determine that vector dimension.

<Infographic src="/img/cv/visual-words.svg" alt="Visual bag-of-words board: local descriptors are clustered into a vocabulary, counts four one three become frequencies point five point one two five point three seven five, and a spatial pyramid adds regional counts." caption="Fixed-length histograms simplify comparison; whole-image counts discard feature location." />

## Worked example, step by step

Use a codebook of three words with centres $(0,0)$, $(4,0)$ and $(0,4)$, and an image with four descriptors, each a pair of numbers: $(0.5,0.5)$, $(1,0)$, $(3.5,0.5)$ and $(0,3)$.

**Assign and count.**

1. $(0.5,0.5)$ is $0.71$ from word 1, $3.54$ from word 2 and $3.54$ from word 3, so it goes to word 1.
2. $(1,0)$ is 1 from word 1, 3 from word 2 and about $4.12$ from word 3, so word 1.
3. $(3.5,0.5)$ is $3.54$ from word 1 and $0.71$ from word 2, so word 2.
4. $(0,3)$ is 3 from word 1 and 1 from word 3, so word 3.
5. The counts are $[2,1,1]$ and the term frequencies are $[0.5,0.25,0.25]$.

**Weight by idf.** Suppose the gallery has $N=4$ images, and the three words appear in 4, 2 and 1 of them. With the smoothed form $\mathrm{idf}=\ln\frac{N+1}{\mathrm{df}+1}+1$:

1. Word 1: $\ln(5/5)+1=1.000$.
2. Word 2: $\ln(5/3)+1=1.511$.
3. Word 3: $\ln(5/2)+1=1.916$.
4. Multiplying the counts $[2,1,1]$ by these gives $[2.000, 1.511, 1.916]$.

In words: the word found in every image still has the biggest count but gets the smallest weight, so rarer words carry more of the similarity.

## How it works

### Codebook & histogram

Extract SIFT features → cluster with k-means into a visual vocabulary → assign each feature to its nearest word → count into a fixed-length histogram.

:::tip

**Worked.** counts [4,1,3] (total 8) → tf [0.5, 0.125, 0.375]; 500 words → 500-dim descriptor.

:::

### Spatial pyramid & concepts

A spatial pyramid adds coarse layout; grouping words into concepts builds a semantic hierarchy (parts → objects → scenes). Ideas persist in VLAD/pooling.

### Key takeaways

- **1 · Vocabulary**; k-means centres of local descriptors; its size sets the histogram length.
- **2 · Histogram**; count assignments, normalise, optionally weight by tf-idf.
- **3 · Limits**; layout is lost unless regions are pooled separately.

## A real system that works this way

Image search is a natural use case. A system can extract local descriptors from each catalogue image, quantise them against one trained codebook and index the resulting histograms. A new query is processed with the same detector, descriptor, codebook and weighting rules, then compared with stored histograms. This is closely related to the [recommendation-as-personalised-retrieval chapter](/docs/theory/ir/recommendation-as-personalised-retrieval): both systems turn source data into a comparable representation, retrieve candidates and rank them for a task. The visual words represent image patterns, while a recommender may also use user context and interaction history.

Term frequency and inverse document frequency come from text retrieval. In visual retrieval, term frequency may be the raw or normalised count of a visual word in one image. Document frequency counts how many training images contain that word. An inverse-document-frequency factor can lower the weight of words that occur in almost every image, such as common background texture. It does not magically make every rare word useful: a rare detector artifact can receive a large weight. Fit the statistics on the training corpus, define the smoothing rule and compare retrieval quality on held-out queries. The toy code below demonstrates the exact frequency calculation only; it does not claim a trained image-search engine.

Spatial pyramids add layout. Imagine two images containing exactly four instances of word 1, one of word 2 and three of word 3. In one image, word 1 is concentrated on the left; in the other it is on the right. A whole-image histogram is identical. Divide each image into a left and right region, count separately and concatenate, and the two representations can differ. The improvement is conditional: a fixed grid can be brittle when objects move or cameras crop differently. The spatial subdivision and weighting are design choices to evaluate.

## Code you can run

The first block reproduces the three-word histogram. The vector has three entries because this toy codebook has three words. Replacing the codebook with 500 words would yield 500 entries before any spatial subdivision. The assignment list is explicit so the count step can be checked without requiring image downloads.

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

**What each control does.**

- *Visual word 1 count*, *2* and *3* set the three counts, from 0 to 12.
- The table shows the counts, their frequencies and the total.

**Try it yourself.**

1. Keep the defaults $[4,1,3]$ and read the frequencies 0.500, 0.125 and 0.375.
2. Double all three counts to $[8,2,6]$. The frequencies do not change, which is why normalising lets an image with more keypoints compare fairly with one with fewer.
3. Set all three counts to 0. The frequencies become undefined, the empty-image case a real pipeline must handle.

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

### Experiment: vocabulary size, weighting and where the codebook comes from

The task is to find which of 12 scenes a query view belongs to. Eight scenes are non-overlapping tiles of the Hubble deep field (public domain, NASA, per the scikit-image documentation), which look alike at a glance, and four are textures and the moon (brick, grass and gravel are CC0; the library documentation gives no licence statement for the moon image, which is used here only as a test input). Each scene yields 8 random views: a crop of 55 to 90% of the side, a rotation of up to 30 degrees, a zoom of 0.7 to 1.2, a brightness change and noise. Three views per scene form the gallery (36 images), five form the queries (60 images). Descriptors are SIFT, capped at 300 per image. The experiment is repeated for three seeds.

It tests three codebook sources: the gallery descriptors, the gallery plus the queries (a leak), and 48 views of six unrelated images. It also compares tf with tf-idf and with direct SIFT matching, where each gallery image is scored by the number of ratio-test matches.

Before the experiment, the worked example in code.

The next block reproduces the worked example with scikit-learn's nearest-centre routine.

```python
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
descriptors = np.array([[0.5, 0.5], [1.0, 0.0], [3.5, 0.5], [0.0, 3.0]])
words = pairwise_distances_argmin(descriptors, centres)
counts = np.bincount(words, minlength=3)
print('assigned words', words.tolist(), 'counts', counts.tolist(), 'term frequency', (counts / counts.sum()).tolist())

document_frequency = np.array([4, 2, 1])
idf = np.log((4 + 1) / (document_frequency + 1)) + 1
print('idf', idf.round(3).tolist(), 'tf-idf counts', (counts * idf).round(3).tolist())
```

**Reading the output.** The assigned words are 0, 0, 1, 2 (zero-based), the counts are $[2,1,1]$ and the term frequencies $[0.5,0.25,0.25]$. The idf values are 1.000, 1.511 and 1.916, and the weighted counts $[2.0, 1.511, 1.916]$.

The experiment block follows. It took about a minute on a laptop CPU.

```python
import cv2
import numpy as np
import sklearn
from skimage import data
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sift = cv2.SIFT_create(nfeatures=300)
sizes = (4, 16, 64, 256, 1024, 4096)
hubble = cv2.cvtColor(data.hubble_deep_field(), cv2.COLOR_RGB2GRAY)
scenes = [hubble[r * 430:(r + 1) * 430, c * 250:(c + 1) * 250] for r in range(2) for c in range(4)]
scenes += [data.brick(), data.grass(), data.gravel(), data.moon()]
unrelated = [cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY), data.camera(), data.coins(), data.text(), cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2GRAY), cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2GRAY)]


def random_view(image, rng):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    side = int(256 * rng.uniform(0.55, 0.9))
    y, x = rng.integers(0, 256 - side + 1, 2)
    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), rng.uniform(-30, 30), rng.uniform(0.7, 1.2))
    out = cv2.warpAffine(image[y:y + side, x:x + side], matrix, (side, side), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-15, 15) + rng.normal(0, 4, out.shape)
    return sift.detectAndCompute(np.clip(out, 0, 255).astype(np.uint8), None)[1]


def histograms(sets, model, size):
    counts = np.zeros((len(sets), size))
    for row, descriptors in zip(counts, sets):
        np.add.at(row, model.predict(descriptors), 1)
    return counts


def matching_accuracy(gallery, queries, glabels, qlabels, ratios=(0.6, 0.7, 0.8)):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    hits = dict.fromkeys(ratios, 0)
    for query, truth in zip(queries, qlabels):
        pairs = [matcher.knnMatch(query, g, k=2) for g in gallery]
        for ratio in ratios:
            votes = [sum(a.distance < ratio * b.distance for a, b in found) for found in pairs]
            hits[ratio] += glabels[int(np.argmax(votes))] == truth
    return [hits[ratio] / len(queries) for ratio in ratios]


table = {}
for seed in range(3):
    rng = np.random.default_rng(seed)
    views = [[random_view(scene, rng) for _ in range(8)] for scene in scenes]
    gallery = [d for v in views for d in v[:3]]
    queries = [d for v in views for d in v[3:]]
    glabels = np.repeat(np.arange(len(scenes)), 3)
    qlabels = np.repeat(np.arange(len(scenes)), 5)
    other = [random_view(image, rng) for image in unrelated for _ in range(8)]
    table.setdefault('direct', []).append(matching_accuracy(gallery, queries, glabels, qlabels))
    pools = {'gallery': gallery, 'gallery + queries': gallery + queries, 'unrelated photos': other}
    for source, pool in pools.items():
        stacked = np.vstack(pool)
        for size in sizes:
            model = KMeans(size, n_init=1, random_state=seed).fit(stacked)
            g, q = histograms(gallery, model, size), histograms(queries, model, size)
            idf = np.log((len(g) + 1) / ((g > 0).sum(0) + 1)) + 1
            for weighting, (gw, qw) in {'tf': (g, q), 'tf-idf': (g * idf, q * idf)}.items():
                if source != 'gallery' and weighting == 'tf':
                    continue
                prediction = glabels[(normalize(qw) @ normalize(gw).T).argmax(1)]
                table.setdefault((source, weighting, size), []).append((prediction == qlabels).mean())
print('sklearn', sklearn.__version__, '| scenes', len(scenes), '| gallery', len(gallery), '| queries', len(queries), '| gallery descriptors in the last draw', sum(len(d) for d in gallery))
direct = np.mean(table['direct'], axis=0)
print('direct SIFT matching, top-1 accuracy at ratio 0.6, 0.7, 0.8:', ' '.join(f'{value:.3f}' for value in direct))
print(f'{"codebook from":18s}{"weighting":>10s}' + ''.join(f'{size:>12d}' for size in sizes))
for source in ('gallery', 'gallery + queries', 'unrelated photos'):
    for weighting in ('tf', 'tf-idf'):
        if (source, weighting, 4) in table:
            print(f'{source:18s}{weighting:>10s}' + ''.join(f'{np.mean(table[(source, weighting, s)]):>7.2f}±{np.std(table[(source, weighting, s)]):.2f}' for s in sizes))
```

**Reading the output.** The first line gives the direct-matching accuracy at three ratio thresholds. The table has one row per codebook source and weighting and one column per vocabulary size. Each cell is the top-1 accuracy as mean plus or minus the standard deviation over three seeds. The last descriptor count says how many gallery descriptors the largest codebook was fitted to.

**Line by line.**

- `random_view` makes every query and gallery view from the same family of changes, with a seeded generator, so the three draws are reproducible.
- `KMeans(size, n_init=1, random_state=seed)` fits the codebook. One initialisation keeps the run fast, at some cost in stability.
- `idf = np.log((len(g) + 1) / ((g > 0).sum(0) + 1)) + 1` is the smoothed idf from the worked example, computed on the gallery only.
- `normalize(qw) @ normalize(gw).T` is cosine similarity, and `glabels[... .argmax(1)]` takes the scene of the most similar gallery image.

#### Reading the experiment

Size matters, then saturates into something else. With 4 words top-1 accuracy is 0.37. It rises to 0.64 at 16 words, 0.78 at 256 and 0.87 at 1,024 for tf-idf. The tf-idf weighting helps only at larger sizes: at 1,024 words tf gives 0.79 and tf-idf gives 0.87. At 4 and 16 words they are identical.

The 4,096-word codebook scores 1.00, but the gallery had about 5,000 descriptors in the last draw (4,984), so almost every descriptor owns a word. The histogram has become a lookup of exact local matches, not a summary. That is a gallery-specific index, not a vocabulary.

Leakage was small here. Fitting k-means on the gallery plus queries gave 0.89 at 1,024 words against 0.87 without, within one standard deviation (0.02). That does not show leakage is harmless, only that this test is too small to detect it. A codebook from unrelated photographs stalled at 0.59 to 0.64 from 64 words upward: a vocabulary learned from the wrong domain cannot represent star fields.

The surprise is the baseline. Direct SIFT matching with a ratio of 0.6 got 1.000, better than any codebook except the degenerate one. The ratio threshold mattered more than any codebook choice: 1.000 at 0.6, 0.889 at 0.7 and 0.700 at 0.8. The histogram index earns its place in cost, since it needs no pass over every gallery image at query time, not in accuracy.

Limits: 12 scenes, 60 queries, three seeds, synthetic views, a 300 descriptor cap and one clustering start. No spatial verification was used, and real image collections have far more scenes and descriptors.

<Infographic src="/img/cv-enrich/v2-bovw-vocabulary.svg" alt="Left: a table of retrieval accuracy against vocabulary size for four codebook and weighting settings. Right: bars of direct SIFT matching accuracy at three ratio thresholds." caption="Read the first row left to right: accuracy climbs with vocabulary size until 4096 words becomes a lookup table. Then compare it with the bars at the top right." />

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
- A visual word is a descriptor-cluster index. Treating it as an object concept requires separate evidence; the “parts to objects to scenes” language describes a possible higher-level interpretation, not a guarantee of k-means.

:::

## Common mistakes

1. **Growing the vocabulary until the score stops rising.** More words feel like more detail. With 4,096 words for 4,984 descriptors the histogram became a lookup table and scored 1.00 for a reason unrelated to vocabulary quality. Choose the size on held-out scenes and a fixed budget.
2. **Training the codebook on the wrong domain.** A generic vocabulary feels reusable. One built from unrelated photographs plateaued at 0.62 to 0.64 here, against 0.87 at the same size from the gallery's own domain.
3. **Fitting the codebook or idf on test images.** It is a quiet leak. Here it moved accuracy by only 0.02, inside the seed noise, so you cannot count on noticing it. Fit on training data and freeze.
4. **Skipping the direct-matching baseline.** A histogram index is fast, so it looks like the answer. Direct matching at ratio 0.6 reached 1.000, and the histogram needs spatial verification to compete on hard scenes.
5. **Ignoring empty images.** A frame with no descriptors gives an all-zero histogram, and every such frame matches every other. Flag them.

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

<details>
<summary><strong>Q6.</strong> Easy. A descriptor $(2.5, 0.5)$ and centres $(0,0)$ and $(4,0)$. Which visual word?</summary>

The distance to $(0,0)$ is $\sqrt{6.25+0.25}=2.55$. The distance to $(4,0)$ is $\sqrt{2.25+0.25}=1.58$. So it is assigned to the second word.<br /><em>Easy · numeric</em>

</details>

<details>
<summary><strong>Q7.</strong> Medium. tf-idf helped at 1,024 words (0.79 to 0.87) and made no difference at 16. Why?</summary>

With 16 words nearly every word appears in nearly every image, so the idf weights are all close to 1 and change nothing. With 1,024 words some words are common background and others are rare and distinctive, so down-weighting the common ones sharpens the similarity.<br /><em>Medium · interpretation</em>

</details>

<details>
<summary><strong>Q8.</strong> Stretch. A 4,096-word codebook scored 1.00 but a 1,024-word one 0.87 on the same data. Does this mean 4,096 is better?</summary>

Not as a general statement. There were about 5,000 gallery descriptors, so nearly each had its own word and the index acts as an exact lookup of the gallery. That fits this gallery and queries drawn from the same scenes. With more scenes, new scenes or a different codebook source it would not carry over, and the histogram would be very sparse.<br /><em>Stretch · interpretation</em>

</details>

## Further reading

- [OpenCV SIFT tutorial](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) for the local features that can feed a visual vocabulary.
- [OpenCV feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) for local-descriptor distances and matching context.
- Built from the course lecture "cv-s15-visual-bag-of-words" (Lecture Library series).

- Sivic and Zisserman, "Video Google: a text retrieval approach to object matching in videos", ICCV 2003, pages 1470 to 1477: the origin of visual words and text-style weighting for images (bibliographic record and abstract checked by search on 2026-10-09; the paper text was not opened, so the details of its clustering and weighting are not quoted).
- scikit-image 0.26.0 sample images: the Hubble deep field (NASA, public domain) and brick, grass and gravel (CC0), licences read from the library documentation.
- Library versions run for the experiment: OpenCV 5.0.0, scikit-learn 1.9.1, NumPy 2.5.3.

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source terminology

Grouping visual words into concepts is said to build a hierarchy of parts, objects and scenes. A k-means visual word is only a cluster of descriptor values. It does not acquire a semantic name from clustering alone. A supervised or otherwise validated grouping would be needed to support a parts-to-objects interpretation. The codebook and histogram method remains valid without that extra interpretation.

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
- I can assign descriptors to the nearest of three centres and compute the idf-weighted counts by hand.
- I can say how vocabulary size, tf-idf and the codebook's source moved retrieval accuracy, and why 4,096 words became a lookup table.
- I can explain why direct ratio-test matching is a baseline the histogram index must be compared with.
