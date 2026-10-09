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

:::tip Before you start

**You should already know**

- What cosine similarity measures ([Session 5](/docs/theory/ir/vector-space-and-term-weighting)).
- What recall at k means ([Session 7](/docs/theory/ir/evaluating-ranked-retrieval)).

**Reading time:** about 50 minutes, plus about 1 minute to run the code (the first run downloads a 580 MB model and 135 MB of images).

**After this chapter you can**

- Say how a shared image and text space is trained and used for search.
- Measure text-to-image and image-to-text recall on a real set, and see how it falls as the collection grows.
- Explain the modality gap, and why a cosine of 0.3 can be a very good match.

:::

## In 30 seconds

You type "red bicycle by a river" and the system returns photographs. It can do that because a photo and a sentence were both turned into a list of numbers, and the two encoders were trained together so that a photo and its caption get similar lists. Searching is then just finding the nearest list.

Real scores are small. A caption and its own photo are not close in the way two similar photos are. What matters is that the right photo scores higher than the other photos in the collection, and that gets harder as the collection grows.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Dual encoder | Two networks, one for images and one for text, that output vectors of the same size | 512 numbers each |
| Contrastive training | Pull matching image-caption pairs together and push mismatched pairs apart | A bicycle photo and its caption |
| Gallery | The collection of items searched | 1,000 images |
| Recall at k (R@k) | Share of queries whose right item is in the top k | R@5 is 0.834 |
| Zero-shot | Using the model for a task it was not trained on, with no new training | Match a photo to "a photo of a cat" |
| Modality gap | Images and texts sit in separate regions of the shared space | Caption-to-image cosine below image-to-image cosine |
| Word order | Whether the sequence of words in a caption changes the result | "dog bites man" against "man bites dog" |

## The idea in plain words

Text search compares a query with documents in the same modality. Multimodal retrieval may ask a text query to find an image, an image to find related captions, or one image to find visually related items. Exact term overlap cannot compare pixels with words. A **dual-encoder** solution learns one representation for images and another for text, then trains them so paired items land near each other in a shared vector space.

**CLIP** and **ALIGN** are two examples. Both use image-text pairs and a contrastive objective: within a training batch, a caption should match its associated image more than unrelated images. After training, a text query can be encoded once and compared with stored image vectors. This is retrieval by similarity, usually cosine or a related dot product after normalisation. It does not mean the system has read the image with human certainty, and a high score is meaningful only relative to other candidates under the same model.

The worked arithmetic uses text vector $q=[1,0,1,0]$ and image vector $d=[1,1,1,0]$. Their dot product is 2; their lengths are $\sqrt{2}$ and $\sqrt{3}$; cosine is about **0.816**. These four-dimensional vectors are an illustration, not actual CLIP outputs. One score alone cannot establish that this image "ranks near the top"; ranking requires comparing it with other images for the same query.

<Infographic src="/img/ir/clip-training.svg" alt="Image and text encoders learn a shared space by raising similarity for matching pairs and lowering it for mismatched pairs." caption="Contrastive image-text training supplies the representation used later for cross-modal search." />

<Infographic src="/img/ir/image-cosine.svg" alt="The illustrative text vector [1,0,1,0] and image vector [1,1,1,0] have dot product two and cosine 0.816." caption="The cosine calculation is separate from how the encoders are trained." />

:::note Added for this site

The caveat about ranking from one score, the toy contrastive batch and the design guidance below are additions. The original CLIP and ALIGN publications are linked so historical model claims stay attributable.

:::

Adjust the image-only coordinate. At the default setting, the text vector is `[1,0,1,0]`, the image vector is `[1,1,1,0]`, and the lab shows **dot product 2** and **cosine 0.816**. Increasing an unrelated coordinate lengthens the image vector and lowers cosine even though the shared coordinates stay fixed.

<CrossModalSimilarityLab />

## Worked example, step by step

Three images (1, 2, 3) and their three captions (a, b, c, where caption a belongs to image 1, and so on). Suppose the model gives these cosines. Real values are small like these, near 0.3.

| | image 1 | image 2 | image 3 |
| --- | ---: | ---: | ---: |
| caption a | 0.30 | 0.25 | 0.10 |
| caption b | 0.20 | 0.31 | 0.28 |
| caption c | 0.15 | 0.33 | 0.29 |

1. **Text to image.** For each caption, find the image with the highest cosine. Caption a picks image 1 (0.30): correct. Caption b picks image 2 (0.31): correct. Caption c picks image 2 (0.33), but its own image 3 scored 0.29, so the right image is second.
2. R@1 is $2/3 = 0.667$. R@2 is $3/3 = 1.0$.
3. **Image to text.** For each image, look down its column. Image 1 picks caption a (0.30): correct. Image 2's column is 0.25, 0.31, 0.33 and picks caption c: wrong, caption b is second. Image 3 picks caption c (0.29): correct. R@1 is $2/3 = 0.667$.
4. **Chance.** If the ranking were random, R@1 would be $1/\text{gallery size}$: $1/3$ here, and $1/1{,}000 = 0.001$ for the real experiment.

In words: cross-modal search is a ranking test. The score of a pair means little alone, and a bigger gallery adds more rivals to beat.

<Infographic src="/img/ir-enrich/ir2-clip.svg" alt="Left, bars of text-to-image recall at 1 for galleries of 100, 300 and 1,000 images and for shuffled captions. Right, three cosine bars: caption with own image, caption with other captions, image with other images." caption="Look first at the bars on the left falling from 0.844 to 0.588 as the gallery grows, and the 0.313 bar on the right, much lower than 0.816 in the toy example." />

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

The first block reproduces the cosine value with only the Python standard library. The same vector pair appears in Session 5, but here the two sides represent different modalities after a shared-space encoder.

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

### The worked example in code

This block computes both directions of the 3 by 3 example.

```python
import numpy as np

sims = np.array([[0.30, 0.25, 0.10], [0.20, 0.31, 0.28], [0.15, 0.33, 0.29]])
own = np.arange(3)
text_rank = (sims > sims[own, own][:, None]).sum(1) + 1
image_rank = (sims.T > sims.T[own, own][:, None]).sum(1) + 1
print(text_rank, (text_rank == 1).mean().round(3), (text_rank <= 2).mean())
print(image_rank, (image_rank == 1).mean().round(3))
```

**Reading the output.** The text ranks print as `[1 1 2]` with R@1 0.667 and R@2 1.0. The image ranks print as `[1 2 1]` with R@1 0.667.

### An experiment: real recall, a growing gallery and shuffled words

How good is a small CLIP model at search, and what does it ignore? The block uses the 1,000-image test split of Flickr30k, with five human captions per image, so 5,000 text queries. It encodes the images and captions with OpenAI's `clip-vit-base-patch32`, then measures recall at 1, 5 and 10 in both directions for galleries of the first 100, 300 and 1,000 images. It repeats the 1,000-image run with every caption's words shuffled into a random order. It finishes with the average cosines that show the modality gap. An image-to-text query counts as a hit if any of its five captions is in the top k. Versions used: Python 3.14.6, transformers 5.18.0, torch 2.14.1, Pillow 12.3.0. The run takes about a minute on CPU.

:::warning Check the model's own limits first

The model card says the model is a research output and that "Any deployed use case of the model - whether commercial or not - is currently out of scope", including image search in a constrained environment without thorough in-domain testing. It also says the model was not evaluated on languages other than English. The card declares no licence field. The Hub copy of the Flickr30k split declares none either; the images are a research benchmark from Flickr, so use them here to measure and do not redistribute them.

:::

```python
import ast
import io
import zipfile

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

repo = "nlphuji/flickr_1k_test_image_text_retrieval"
table = pd.read_csv(hf_hub_download(repo, "test_1k_flickr.csv", repo_type="dataset"))
archive = zipfile.ZipFile(hf_hub_download(repo, "images_flickr_1k_test.zip", repo_type="dataset"))
captions = [c for raw in table["raw"] for c in ast.literal_eval(raw)]
owner = np.repeat(np.arange(len(table)), 5)
shuffler = np.random.default_rng(0)
scrambled = [" ".join(shuffler.permutation(c.split())) for c in captions]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def unit(x):
    return x / x.norm(dim=-1, keepdim=True)

def encode_images():
    out = []
    for start in range(0, len(table), 50):
        names = table["filename"][start:start + 50]
        batch = [Image.open(io.BytesIO(archive.read(f"images_flickr_1k_test/{n}"))).convert("RGB") for n in names]
        out.append(unit(model.get_image_features(**processor(images=batch, return_tensors="pt")).pooler_output))
    return torch.cat(out)

def encode_text(texts):
    out = []
    for start in range(0, len(texts), 250):
        tokens = processor(text=texts[start:start + 250], return_tensors="pt", padding=True, truncation=True)
        out.append(unit(model.get_text_features(**tokens).pooler_output))
    return torch.cat(out)

with torch.no_grad():
    pictures = encode_images()
    texts = {"original captions": encode_text(captions), "word order shuffled": encode_text(scrambled)}

def recall(text_vectors, gallery):
    sims = (text_vectors[owner < gallery] @ pictures[:gallery].T).numpy()
    text_rank = (sims > sims[np.arange(len(sims)), owner[owner < gallery]][:, None]).sum(1)
    by_image = sims.T
    own = np.arange(gallery)[:, None] * 5 + np.arange(5)
    best = np.take_along_axis(by_image, own, 1).max(1)
    image_rank = (by_image > best[:, None]).sum(1)
    return [float((rank < k).mean()) for rank in (text_rank, image_rank) for k in (1, 5, 10)]

print(f"{'queries':<22}{'gallery':>8}   text to image R@1  R@5  R@10    image to text R@1  R@5  R@10")
for name, vectors in texts.items():
    for gallery in ((100, 300, 1000) if name == "original captions" else (1000,)):
        r = recall(vectors, gallery)
        print(f"{name:<22}{gallery:>8}{r[0]:>21.3f}{r[1]:>6.3f}{r[2]:>6.3f}{r[3]:>20.3f}{r[4]:>6.3f}{r[5]:>6.3f}")

text = texts["original captions"]
off_diagonal = 1000 * 999
print(f"mean cosine, caption with its own image: {(text * pictures[owner]).sum(1).mean():.3f}")
print(f"mean cosine, image with other images: {((pictures @ pictures.T).sum() - 1000) / off_diagonal:.3f}")
print(f"mean cosine, caption with other captions: {((text[::5] @ text[::5].T).sum() - 1000) / off_diagonal:.3f}")
print(f"distance between image centroid and text centroid: {(pictures.mean(0) - text.mean(0)).norm():.3f}")
```

The output of the run:

```text

queries                gallery   text to image R@1  R@5  R@10    image to text R@1  R@5  R@10
original captions          100                0.844 0.976 0.994               0.970 0.990 0.990
original captions          300                0.721 0.935 0.969               0.877 0.987 0.997
original captions         1000                0.588 0.834 0.901               0.795 0.950 0.981
word order shuffled       1000                0.423 0.697 0.793               0.585 0.848 0.908
mean cosine, caption with its own image: 0.313
mean cosine, image with other images: 0.487
mean cosine, caption with other captions: 0.424
distance between image centroid and text centroid: 0.802
```

**Reading the output.** The first table gives recall at 1, 5 and 10. `text to image` uses each caption as a query. `image to text` uses each image. The last four lines are average cosines: a caption with its own image, an image with other images, a caption with other captions, and the distance between the two modality centres. All vectors have length 1.

**Line by line.**

- `.pooler_output` is the projected 512-number vector in this version of transformers, where the model methods return an output object.
- `(sims > ...[:, None]).sum(1)` counts how many items beat the right one, which is its rank minus one, without sorting.
- `shuffler.permutation(c.split())` puts the same words in random order, so only word order is destroyed.

### What the numbers say

Recall fell as the gallery grew. Text-to-image recall at 1 was 0.844 over 100 images, 0.721 over 300 and 0.588 over 1,000, and recall at 10 went from 0.994 to 0.901. Image-to-text was easier because five captions gave five chances: 0.970, 0.877 and 0.795 at 1. The same model on the same captions looks near perfect or mediocre depending only on how many images it must beat.

Shuffling word order cost less than expected. Text-to-image recall at 1 went from 0.588 to 0.423 and image-to-text from 0.795 to 0.585, so about 72% and 74% of the original recall survived scrambled captions, against a chance of 0.001. The words alone carry most of the signal. This fits the finding of Yuksekgonul and colleagues (2022) that vision-language models can do well on retrieval benchmarks without using composition or order. One shuffle at one seed is not a measurement of order sensitivity, only a hint.

The cosines show the modality gap. A caption and its own image averaged 0.313. An image and a different image averaged 0.487 and a caption and a different caption 0.424. The two centres were 0.802 apart on unit vectors. A matching pair is closer than random pairs but much further apart than same-modality items, which agrees with the modality gap described by Liang and colleagues (2022). The worked example's 0.816 is nowhere near this scale, and a threshold of 0.8 taken from the toy example would sit far above the average matching score of 0.313.

Limits: one small model, one benchmark of everyday scenes with English captions, no confidence intervals, one shuffle seed, and no rights, safety or latency checks.

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

One image score cannot determine a rank. If every other candidate has cosine below 0.816, this image ranks first; if many score above it, it does not. A threshold chosen on one model or collection may not transfer to another. The phrase "ranks near top" should therefore be read as a possible result after comparison, not as a conclusion from 0.816 alone. Report rankings against labelled image queries.

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

## Common mistakes

| Mistake | Why it feels right | What to do instead |
| --- | --- | --- |
| Quoting recall without the gallery size | The percentage looks self-contained | Always give the size. R@1 was 0.844 at 100 images and 0.588 at 1,000 |
| Setting a cosine threshold from the toy example | 0.816 looked like a good match | Real matching pairs averaged 0.313. Compare candidates within one query and measure on your own images |
| Assuming the model reads the sentence like a person | It retrieves the right picture | Scrambled captions kept about three quarters of recall. Test negation, counts and relations on your own data |
| Mixing vectors from different model versions | Both are 512 numbers | A new encoder is a new space. Re-encode the whole gallery |
| Deploying because the benchmark passed | The numbers are good | The model card marks deployment out of scope without in-domain testing, and image rights are a separate problem |

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

<details>
<summary><strong>Q6.</strong> (Medium) Text-to-image R@1 was 0.588 over 1,000 images and 0.844 over 100. Is the model worse on the larger set? Explain.</summary>

The model is the same. With more images, more of them look somewhat like each caption, so more rivals can outscore the right image. Recall at 1 falls even though no individual score has changed. Compare systems only on the same gallery, and report the gallery size.

</details>

<details>
<summary><strong>Q7.</strong> (Medium) The average cosine of a caption with its own image was 0.313 and an image with another image 0.487. Why is the first not a sign of a bad model?</summary>

The model is trained to make the right image score highest among the images for each caption, not to make cross-modal cosines large. Image vectors sit in one region of the space and caption vectors in another, the modality gap, so any caption-image cosine is lowered by the distance between the regions. The ranking, not the absolute value, is what search uses.

</details>

<details>
<summary><strong>Q8.</strong> (Stretch) With shuffled words, recall at 1 only fell from 0.588 to 0.423. What would you test next before claiming the model ignores word order?</summary>

Shuffle with several seeds and report the spread. Then use pairs that differ only in order or relation, such as "dog chases cat" against "cat chases dog", with the images that match each, because Flickr captions rarely include a confusable reversed caption. Compare against a bag-of-words baseline at the same gallery size. A fall of 28% is evidence that order matters somewhat, but not that the model understands relations.

</details>

## Go deeper

- [CLIP research page and paper](https://openai.com/index/clip/); original model and zero-shot explanation.
- [ALIGN paper](https://arxiv.org/abs/2102.05918); another large-scale image-text dual encoder.
- [Stanford IR book: vector-space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html); cosine as a retrieval score.
- [CLIP model card, openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) (opened 2026-10-09); intended use, out-of-scope uses and the language limit quoted above.
- [Liang et al.: Mind the Gap, the modality gap in multi-modal contrastive representation learning](https://arxiv.org/abs/2203.02053) (opened 2026-10-09); images and text sit at arm's length in the shared space.
- [Yuksekgonul et al.: When and why vision-language models behave like bags-of-words](https://arxiv.org/abs/2210.01936) (opened 2026-10-09); good retrieval is possible without composition or order information.
- [Young et al.: From image descriptions to visual denotations](https://aclanthology.org/Q14-1006) (2014); the Flickr30k data behind the 1,000-image test split.
- Built from the course lecture "ir-s13-multimodal-clip" (Lecture Library series).

- **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** `book`
  Manning, Raghavan & Schütze; The standard IR text; indexing, Boolean & vector models, ranking, evaluation.
- **[Stanford CS276](https://web.stanford.edu/class/cs276/)** `course`
  Stanford; Information retrieval and web search; slides that follow the IR book.

## Check yourself

- [ ] I can explain how contrastive image-text training creates a shared representation.
- [ ] I can compute the toy cross-modal cosine and avoid inferring a rank from one score.
- [ ] I can distinguish zero-shot label comparison from searching an image collection.
- [ ] I can identify metadata, rights and evaluation checks needed after vector retrieval.
- [ ] I can compute text-to-image and image-to-text recall by hand from a small similarity table.
- [ ] I can explain why recall depends on the size of the collection, and quote the measured fall from 0.844 to 0.588.
- [ ] I can describe the modality gap and say why cosine 0.313 is normal for a matching pair.
- [ ] I can say what the shuffled-caption result suggests and what it does not prove.

## Where to go next

Next: [Session 14, recommendation as personalised retrieval](/docs/theory/ir/recommendation-as-personalised-retrieval). Related: [Neural retrieval and reranking](/docs/theory/ir/neural-retrieval-and-reranking), where the same dual-encoder idea is applied to text only.
