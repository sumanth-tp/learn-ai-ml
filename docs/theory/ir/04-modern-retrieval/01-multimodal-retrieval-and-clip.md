---
id: ir-multimodal-retrieval-clip
title: "Information Retrieval · Session 13 — Multimodal Retrieval and CLIP"
sidebar_label: "13 · Multimodal"
sidebar_position: 1
slug: /theory/ir/multimodal-retrieval-and-clip
description: "How paired image and text encoders support cross-modal retrieval, with a worked cosine example and limits of zero-shot use."
tags: [information-retrieval, multimodal-retrieval, clip, contrastive-learning]
---

import Infographic from '@site/src/components/Infographic';
import CrossModalSimilarityLab from '@site/src/components/viz/CrossModalSimilarityLab';

**In one line.** Cross-modal search maps a text query and an image into a shared learned space so their similarity can be used for candidate ranking.

## The idea in plain words

Text search compares a query with documents in the same modality. Multimodal retrieval may ask a text query to find an image, an image to find related captions, or one image to find visually related items. Exact term overlap cannot compare pixels with words. A **dual-encoder** solution learns one representation for images and another for text, then trains them so paired items land near each other in a shared vector space.

The lecture names **CLIP** and **ALIGN** as examples. Both use image-text pairs and a contrastive objective: within a training batch, a caption should match its associated image more than unrelated images. After training, a text query can be encoded once and compared with stored image vectors. This is retrieval by similarity, usually cosine or a related dot product after normalisation. It does not mean the system has read the image with human certainty, and a high score is meaningful only relative to other candidates under the same model.

The lecture's arithmetic uses text vector $q=[1,0,1,0]$ and image vector $d=[1,1,1,0]$. Their dot product is 2; their lengths are $\sqrt{2}$ and $\sqrt{3}$; cosine is about **0.816**. These four-dimensional vectors are an illustration, not actual CLIP outputs. One score alone cannot establish that this image "ranks near the top"; ranking requires comparing it with other images for the same query.

<Infographic src="/img/ir/clip-training.svg" alt="Image and text encoders learn a shared space by raising similarity for matching pairs and lowering it for mismatched pairs." caption="Contrastive image-text training supplies the representation used later for cross-modal search." />

<Infographic src="/img/ir/image-cosine.svg" alt="The illustrative text vector [1,0,1,0] and image vector [1,1,1,0] have dot product two and cosine 0.816." caption="The lecture's cosine calculation is separate from how the encoders are trained." />

:::note Beyond the lecture

The caveat about ranking from one score, the toy contrastive batch and the design guidance below extend the lecture. The original CLIP and ALIGN publications are linked so historical model claims stay attributable.

:::

Adjust the image-only coordinate. At the lecture default, the text vector is `[1,0,1,0]`, the image vector is `[1,1,1,0]`, and the lab shows **dot product 2** and **cosine 0.816**. Increasing an unrelated coordinate lengthens the image vector and lowers cosine even though the shared coordinates stay fixed.

<CrossModalSimilarityLab />

## How it works

### Contrastive vision-language

Train an image encoder and a text encoder with a contrastive objective on image–caption pairs: matching pairs close, mismatched far. Retrieve by cosine similarity.

:::tip

**Worked.** text q=[1,0,1,0], image d=[1,1,1,0] → cos = 0.816 → ranks near top.

:::

### Describe any class

Because CLIP aligns arbitrary text with images, it does zero-shot retrieval/classification: describe a class in words ("a photo of a cat") and match images; no task-specific training.


## A real system that works this way

The [original CLIP publication](https://openai.com/index/clip/) describes OpenAI's research model trained to associate images with text from broad natural-language supervision. For zero-shot image classification, it compares an image with text descriptions of candidate labels, such as "a photo of a cat" and "a photo of a dog", without training a new classifier on that target label set. This is an actual model demonstration, not a guarantee that any prompt or any unseen category will be recognised reliably. The [paper](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf) details the paired-encoder contrastive training.

Google Research's [ALIGN paper](https://arxiv.org/abs/2102.05918) is another original example. It uses a simple dual-encoder architecture trained on large amounts of noisy image alt-text pairs and reports cross-modal search and zero-shot transfer in its experimental setting. Both papers are research from 2021; they show the training pattern, while any current product deployment would need separate evidence about model version, data rights and measured quality.

Consider an editorial image library. A user types `red bicycle near a river`, the text encoder produces a vector, and a vector index returns nearby image embeddings. A human editor still needs to check that the selected photo truly contains the requested bicycle and river and is licensed for the intended use. A semantic match can miss an exact colour, count or text in the scene. Search quality includes both candidate recall and the final visual inspection workflow.

## Code you can run

The first block reproduces the lecture's cosine value with only the Python standard library. The same vector pair appears in Session 5, but here the two sides represent different modalities after a shared-space encoder.

```python
from math import sqrt

text_vector = [1, 0, 1, 0]
image_vector = [1, 1, 1, 0]
dot = sum(a * b for a, b in zip(text_vector, image_vector))
text_norm = sqrt(sum(value * value for value in text_vector))
image_norm = sqrt(sum(value * value for value in image_vector))
cosine = dot / (text_norm * image_norm)
print(f"dot={dot} cosine={cosine:.3f}")
assert dot == 2 and round(cosine, 3) == 0.816
```

The second block shows a tiny **illustrative** contrastive choice. Two images and two captions have a fixed similarity matrix; the diagonal entries are the correct pairs. A softmax over each image's caption scores assigns more probability to the correct pair. The numbers are hand-picked, not produced by a trained vision model, and a real CLIP loss also considers both image-to-text and text-to-image directions.

```python
from math import exp

similarity = [[0.8, 0.2], [0.1, 0.9]]
correct_pair_probabilities = []
for image_index, caption_scores in enumerate(similarity):
    weights = [exp(score) for score in caption_scores]
    probability = weights[image_index] / sum(weights)
    correct_pair_probabilities.append(probability)
print([round(value, 3) for value in correct_pair_probabilities])
assert all(value > 0.5 for value in correct_pair_probabilities)
```

Training pushes the correct pairs above alternatives in a batch. Retrieval later compares a new text query with image vectors already stored in an index. The fixed four-number cosine example explains the comparison, not the learning that produced a meaningful shared space.

## Designing with it

### Distinguish training from serving

During contrastive training, paired image-caption data tells the model which two representations should align. A batch supplies mismatched alternatives. The image and text encoders are adjusted together. At serving time, image vectors can be precomputed and indexed, while a new query is encoded on demand. The system does not retrain for each query. Updating the model changes the vector space, so old image vectors should be rebuilt or kept in a versioned index rather than mixed with new query vectors.

### Decide what the vector represents

An image can contain several objects, text, a background and a visual style. A single vector compresses them into one representation. This works for broad semantic matching but can fail when the user asks for a small object, an exact number of people or an exception such as "without a helmet". Cropping, region-level embeddings, OCR or a reranker may help, but each adds complexity and needs evaluation. If the task is to find a specific product by SKU, metadata and exact lexical fields may be more reliable than image similarity.

The caption side is equally important. Web alt text can be incomplete, promotional or unrelated to the visible pixels. Training at scale can tolerate some noise, as the papers investigate, but dataset bias remains. A model may associate a background or style with a label rather than the actual object. Test with visually similar negatives, varied contexts and labels written in the language users employ.

### Build retrieval, not a magic classifier

For zero-shot classification, provide candidate text descriptions and compare an image against them. The candidate set and prompt wording influence the result. A high relative score among two poor labels still forces a winner unless the product offers an unknown or abstain option. For retrieval, collect several relevant images per query and rank them. The score should be evaluated with P@k, AP or NDCG as Session 7 describes, plus human checks for important constraints.

| Task | Query | Indexed items | What to validate |
| --- | --- | --- | --- |
| Text-to-image | Natural-language description | Image embeddings | Object, relation and attribute match |
| Image-to-text | Image or crop | Captions or documents | Grounding and caption specificity |
| Zero-shot labels | Candidate descriptions | Image embedding | Prompt sensitivity and unknown classes |
| Visual similarity | Reference image | Image embeddings | Whether visual resemblance serves user intent |

### Control score interpretation

Cosine 0.816 is a calculation in the toy space; it is not a universal relevance probability. A different model, prompt or normalisation changes score distributions. Compare candidates within a defined model and query, then label real results. If a system searches millions of images, approximate nearest-neighbour indexing may speed candidate retrieval but can miss the exact nearest items. Measure candidate recall before a more expensive visual-text reranker and keep a fallback for exact metadata filters.

### Preserve rights and provenance

An image result is not ready to use merely because it matches the text. Keep source URL, creator or licence metadata where available, collection time and any usage constraints. Remove or update images when rights change. For a production media library, the retrieval system should enforce access restrictions before display and export. These are product obligations separate from cosine similarity, yet they determine whether the retrieved item is useful.

## Follow one image through the system

An image enters a collection with an ID, pixels and metadata. An image encoder transforms its pixels into a vector, and the vector index stores that representation with a link to the original asset. At query time, a text encoder transforms `red bicycle near a river` into a vector from the same trained space. The index produces a candidate list by similarity. A final stage can check stricter attributes, source rights and human review. This is analogous to the text retrieval pipeline: representation, candidate generation, ranking and evaluation remain separate decisions.

If a new encoder replaces the old one, the existing image vectors no longer necessarily match new query vectors. A migration can build a parallel index and compare its results on held-out queries before switching traffic. If only some assets are re-encoded, record model versions and prevent cross-version scores from being mixed without calibration. This is the multimodal analogue of changing a text analyser in an inverted index.

### Read the contrastive objective

Imagine a batch with an image of a bicycle and a caption describing it, plus an image of a boat and its caption. The model should score each correct pairing above the cross-pairings. A contrastive loss converts similarities to probabilities across the batch and penalises the correct pair when alternatives are too competitive. The toy matrix in the code gives 0.8 and 0.9 to the diagonal, so the correct caption has a larger softmax share for each image. The example omits temperature scaling, large batches and the symmetric text-to-image part to keep the mechanism visible.

The loss is relative. It does not teach an image to satisfy every possible fact about a sentence, and it can learn shortcuts from correlations in the training data. Hard negatives, such as two images differing only by bicycle colour, can reveal whether the model learned the attribute. A model that does well on broad category labels may still be weak on exact relations such as "the dog is behind the bicycle".

### Interpret the cosine example honestly

The text and image vectors share two positive coordinates, giving dot product two. The image has one additional positive coordinate, making its length $\sqrt{3}$ while the text length is $\sqrt{2}$. Cosine divides by both lengths and yields 0.816. If the extra image coordinate grows, the dot product remains two but image length grows, so cosine falls. The lab shows this geometric effect. The toy coordinates have no learned semantic names; they only explain the arithmetic of comparing two encoded modalities.

One image score cannot determine a rank. If every other candidate has cosine below 0.816, this image ranks first; if many score above it, it does not. A threshold chosen on one model or collection may not transfer to another. The lecture's phrase "ranks near top" should therefore be read as a possible result after comparison, not as a conclusion from 0.816 alone. Report rankings against labelled image queries.

### Compare retrieval with classification

Text-to-image retrieval chooses among a stored collection of assets. Zero-shot classification compares an image with text descriptions of candidate classes. Both use the same shared-space scoring operation but have different success criteria. A classifier may need an unknown class or calibrated confidence; a search interface may need diverse images and useful filters. Prompt wording can affect a zero-shot label score, so use several representative descriptions and measure label errors on the target domain before deployment.

For an image library, combine semantic vectors with structured metadata. The user may need a portrait orientation, a minimum resolution, a licence permitting publication or an exact date. These are filters, not semantic guesses. Apply access and rights filters safely, then rank the eligible candidates. A highly similar image that cannot be used should not be shown as a valid result.

### Connect to later chapters

The dual-encoder design reappears in Session 15 for text-to-text dense retrieval. The expensive second-stage check resembles a cross-encoder reranker. Session 7's rank-aware metrics still evaluate the results, but labels must specify what visual match means to a user. The model is a way to produce candidates across modalities; the retrieval system around it determines freshness, rights, safety and whether the user can act on the result.

## Where this stands in 2026

:::info Industry view

- CLIP and ALIGN remain foundational demonstrations of contrastive image-text dual encoders; the original publications explain their training and reported zero-shot experiments.
- Image-text vector search is useful for broad semantic discovery, while exact attributes, OCR text and rights metadata often need additional fields or review.
- Current multimodal systems should be evaluated on their own images, languages, prompts and user constraints rather than inheriting a historical benchmark result.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is multimodal retrieval, and what enables it?</summary>

Retrieving one modality with a query in another (e.g. text→image); enabled by mapping modalities into a shared embedding space where similar items are close.<br /><em>Session 13 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> How are CLIP/ALIGN trained?</summary>

An image encoder and a text encoder are trained with a contrastive objective on image–caption pairs; matching pairs pulled together, mismatched pushed apart.<br /><em>Session 13 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Rank an image for a text query given text q=[1,0,1,0] and image d=[1,1,1,0].</summary>

cos = 2/(√2·√3) = 0.816; a strong match, so it ranks near the top.<br /><em>Session 13 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> What is CLIP's zero-shot capability?</summary>

Describe any class in words ('a photo of a cat') and match images to it without task-specific training; because arbitrary text and images share the space.<br /><em>Session 13 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> How is CLIP retrieval analogous to classic IR?</summary>

Both embed query and items as vectors and rank by cosine similarity; CLIP just uses learned cross-modal embeddings instead of tf-idf.<br /><em>Session 13 · conceptual</em>

</details>

## Go deeper

- [CLIP research page and paper](https://openai.com/index/clip/); original model and zero-shot explanation.
- [ALIGN paper](https://arxiv.org/abs/2102.05918); another large-scale image-text dual encoder.
- [Stanford IR book: vector-space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html); cosine as a retrieval score.
- Built from the course lecture "ir-s13-multimodal-clip" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check your understanding

- [ ] I can explain how contrastive image-text training creates a shared representation.
- [ ] I can compute the toy cross-modal cosine and avoid inferring a rank from one score.
- [ ] I can distinguish zero-shot label comparison from searching an image collection.
- [ ] I can identify metadata, rights and evaluation checks needed after vector retrieval.
