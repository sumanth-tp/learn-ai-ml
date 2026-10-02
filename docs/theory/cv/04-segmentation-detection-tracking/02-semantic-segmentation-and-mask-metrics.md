---
id: cv-semantic-segmentation-and-mask-metrics
title: "Computer Vision · Session 12; Semantic Segmentation and Mask Metrics"
sidebar_label: "2 · Semantic masks and metrics"
sidebar_position: 2
slug: /theory/cv/semantic-segmentation-and-mask-metrics
description: "Compare semantic and instance masks, calculate IoU and Dice, and evaluate rare classes and boundaries honestly."
tags: [computer-vision, semantic-segmentation, iou, dice]
---

import Infographic from '@site/src/components/Infographic';
import MaskOverlapLab from '@site/src/components/viz/MaskOverlapLab';

**In one line.** Semantic segmentation labels pixels by class, while mask metrics compare predicted and reference pixel sets for a stated class.

## The idea in plain words

:::note Beyond the lecture

The count-only lab, empty-mask convention, per-class evaluation design and current model-family context extend the lecture. Its architecture outline, worked IoU and Dice example and all five practice questions remain below.

:::

An image-level classifier returns one label for a whole frame. A semantic segmenter returns a class for every pixel, producing a map of roads, sky, cars or other classes defined by the task. If two cars touch or overlap, their pixels may all share the same semantic label “car”. Instance segmentation adds a distinct identity for each object, often as a set of masks plus classes and scores. Panoptic segmentation combines things that have instances with stuff regions such as sky, under an explicit labelling policy. These output contracts cannot be substituted casually: a traffic-counting product needs separated cars, while a road-area estimator may need only the road class.

The lecture sketches fully convolutional networks and encoder-decoder designs. An encoder builds contextual features while reducing spatial resolution; a decoder produces an output map at the required resolution. Skip connections in U-Net-style designs carry higher-resolution features from earlier stages, which can help recover boundary detail. Upsampling does not reconstruct every detail that was absent or discarded in the input. The annotation resolution, image scale and choice of loss all influence thin structures and small objects. For a deeper treatment of convolution and pretrained vision backbones, follow the existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is) and [transfer-learning chapter](/docs/theory/dnn/transfer-learning-feature-extraction-vs-fine-tuning).

For one class, let $P$ be the set of pixels predicted positive and $G$ the reference set. The intersection contains pixels in both. The union contains pixels in either. Intersection over union is $|P\cap G|/|P\cup G|$. Dice is $2|P\cap G|/(|P|+|G|)$. When each mask has 100 pixels and they overlap in 50, their union is 150, so IoU is $50/150=0.333$ and Dice is $100/200=0.5$. These are two views of the same overlap: for nonempty union, Dice equals $2\mathrm{IoU}/(1+\mathrm{IoU})$ and is at least IoU. They cannot reveal whether the missed pixels lie at a harmless outer edge or across a clinically or operationally critical structure.

Mean IoU averages IoU over classes under a specified class-inclusion rule. Pixel accuracy counts correctly labelled pixels and can be dominated by the background. If a rare class occupies a small region, predicting background everywhere can score well on pixel accuracy while giving that class zero recall. Always show per-class support and IoU alongside an average, and explain whether absent classes are ignored, scored as perfect when both masks are empty, or treated another way. That convention can change a mean substantially in small datasets.

<Infographic src="/img/cv/semantic-metrics.svg" alt="Mask metric board: a 100-pixel predicted mask and 100-pixel reference mask overlap in 50 pixels; union 150 gives IoU one third and Dice one half, with a reminder to report each class." caption="The denominator and empty-class policy must be stated before an overlap score is compared." />

## How it works

### Semantic segmentation

Every pixel gets a class (road, car, sky). Instance segmentation also separates individual objects. Powers driving, medical imaging, scene understanding.

### FCN & encoder-decoder

FCNs output a spatial map. Encoder downsamples (context) → decoder upsamples (resolution); U-Net skip connections restore fine detail for sharp boundaries.

### IoU, Dice, mIoU

IoU = |P∩G|/|P∪G|; Dice = 2|P∩G|/(|P|+|G|); pixel accuracy; mean IoU averages over classes.

:::tip

**Worked.** |P|=|G|=100, overlap=50 → union=150 → IoU=50/150=0.333; Dice=100/200=0.5.

:::

### Key takeaways

- **1 · Task**; Per-pixel class; instance separates objects.
- **2 · Nets**; FCN, U-Net encoder-decoder + skips.
- **3 · Metrics**; IoU, Dice, mIoU.

## A real system that works this way

The current Torchvision documentation lists DeepLabV3 as a semantic segmentation model family and Mask R-CNN as an instance segmentation family. Their outputs answer different questions: a class map versus a set of object masks and detections. Meta's original SAM 3 publication, checked on 2026-10-02, describes concept-prompted detection, segmentation and tracking. Prompted masks introduce yet another input contract: quality depends on the concept, prompt and use case, and a returned mask still requires evaluation against the intended reference. These official sources verify the families' existence and scope; this chapter claims no relative accuracy or production suitability.

Imagine mapping water across aerial tiles. A semantic water mask can estimate covered area, but a shore boundary may move with image resolution and annotation judgement. If a 100-pixel predicted water patch overlaps the 100-pixel annotated patch by 50, the lecture metrics are exactly reproducible. However, one tile with a small stream and another with a huge lake should not necessarily receive equal operational weight. Macro-averaging tile scores gives each tile equal weight, while aggregating intersections and unions across tiles weights pixels differently. Choose the aggregation that matches the decision and report it.

The same project should stratify by cloud, shade, season and sensor. A water-looking roof may create a false positive; a shadowed stream may be missed. Review the actual masks over the source images and quantify small-object and boundary errors separately. If area is converted to physical units, account for pixel scale and projection. The toy overlap arithmetic is necessary to understand the metric, but does not provide those geospatial guarantees.

## Code you can run

The first block reproduces the lecture's numerical masks using counts. The intersection must not exceed either mask size. The union follows inclusion-exclusion, and the two metrics use different denominators.

```python
predicted_pixels = 100
reference_pixels = 100
intersection = 50
assert 0 <= intersection <= min(predicted_pixels, reference_pixels)
union = predicted_pixels + reference_pixels - intersection
iou = intersection / union
dice = 2 * intersection / (predicted_pixels + reference_pixels)
print('Union:', union)
print(f'IoU: {iou:.3f}')
print(f'Dice: {dice:.3f}')
assert union == 150
assert (round(iou, 3), round(dice, 3)) == (0.333, 0.5)
```

The lab begins with those three counts and exposes its arithmetic in the data table. It deliberately displays count bars rather than a Venn drawing: three cardinalities alone do not determine the geometry of two masks. Move the overlap down to see both metrics fall. The overlap slider is constrained not to exceed the smaller mask.

<MaskOverlapLab />

The second block compares two classes. It reports per-class IoU and an unweighted mean for this explicit example. Background has 900 predicted and 900 reference pixels with 850 overlapping; target has 100, 100 and 50. The target's one-third IoU is visible instead of being hidden by the larger background overlap. The example counts are synthetic and are not a consistent full confusion matrix over one fixed image; they are independent per-class set examples to isolate macro averaging.

```python
classes = {
    'background': (900, 900, 850),
    'target': (100, 100, 50),
}
scores = {}
for name, (predicted, reference, common) in classes.items():
    scores[name] = common / (predicted + reference - common)
mean_iou = sum(scores.values()) / len(scores)
print('Per-class IoU:', {name: round(score, 3) for name, score in scores.items()})
print(f'Mean IoU: {mean_iou:.3f}')
assert round(scores['target'], 3) == 0.333
assert round(mean_iou, 3) == 0.614
```

Two empty masks have zero union, so $0/0$ is undefined. Some benchmarks assign a perfect score to a class absent from both prediction and truth, while others exclude that class; either choice must be stated. The lab marks this case undefined and avoids an arbitrary score. A per-image average and a corpus-level aggregate can also differ even with the same empty-class rule.

## Designing with it

Write the annotation policy before training. Define class boundaries, occluded objects, uncertain pixels and ignored regions. If a lane marking is partly hidden by a car, decide whether the road class includes the hidden area or only visible pixels. Annotators can disagree around translucent, reflective or fuzzy edges. An evaluation metric cannot resolve a policy disagreement; it merely quantifies overlap under a chosen reference. Audit a sample of labels and record ambiguous cases.

Match output resolution to the smallest relevant structure. Downsampling a mask can erase a thin vessel or road marking; upsampling cannot recover it afterwards. Test the full preprocessing and postprocessing path at serving resolution, including interpolation and crop offsets. For discrete class maps, use a label-preserving resize when mapping predictions. Record the coordinate frame of annotations and outputs. A model with strong image-level accuracy may still have unacceptable boundary or small-object quality.

Report per-class counts. Mean IoU is helpful for balancing classes, but it is sensitive to which classes are included and what happens when a class is absent. Pixel accuracy can obscure rare classes. Dice is common for overlap-focused tasks, but its different denominator does not make it inherently better. State whether metrics are computed per image then averaged, or computed from pooled intersections and unions. Show the distribution, not only one mean, especially when tiles or scenes differ greatly in size and difficulty.

Inspect error topology. Two masks can share the same intersection and union yet differ in whether a narrow bridge is broken, an object is split or a large region is shifted. If connectivity or precise boundary matters, add component counts, boundary distance or task-specific measurements. For a navigation system, a small false road connection may be more dangerous than an equal number of missed pixels at an irrelevant edge. The metric suite should reflect such costs while retaining IoU and Dice for comparability.

Choose a model family based on output requirements and data. A semantic network returns class labels per pixel; an instance model separates countable objects; a promptable model may provide useful masks under a concept or exemplar input. These families can also be combined in a workflow. Compare on the same labelled evaluation set and record device, image size, preprocessing and latency. No public benchmark number proves fitness for the local annotation policy.

Monitor shift after deployment. Illumination, sensor type, season and object appearance can change the input distribution. Sample new images for ground-truth labelling, measure per-class performance and inspect failures. A model may continue to produce complete pixel maps while systematically missing a rare class. Track empty-mask frequency and predicted class area as diagnostics, but do not mistake them for accuracy when reference labels are unavailable.

## Where this stands in 2026

:::info Industry view

- Torchvision 0.29 DeepLabV3 and Mask R-CNN documentation and Meta’s SAM 3 publication were opened on 2026-10-02 to verify current semantic, instance and prompted segmentation families. No benchmark values are repeated.
- The lecture’s IoU and Dice figures reproduce from the stated counts. The empty-union case requires an evaluation convention and is explicitly undefined in the local lab.
- The water-mapping case is a design scenario, not a claim about a named deployed system. No image dataset or trained model was run.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> Contrast semantic and instance segmentation.</summary>

Semantic labels every pixel with a class (all cars = 'car'); instance additionally separates individual objects (car 1 vs car 2).<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What is the encoder-decoder / U-Net design, and what do skip connections do?</summary>

The encoder downsamples for context, the decoder upsamples to full resolution; skip connections carry fine detail from encoder to decoder for sharp boundaries.<br /><em>Session 12 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Prediction and ground-truth masks each cover 100 pixels, overlapping in 50. Compute IoU.</summary>

Union = 100+100−50 = 150; IoU = 50/150 = 0.333.<br /><em>Session 12 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> For the same masks, compute the Dice coefficient.</summary>

Dice = 2·50/(100+100) = 100/200 = 0.5 (Dice ≥ IoU always).<br /><em>Session 12 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What is mIoU, and why is pixel accuracy misleading?</summary>

mIoU is IoU averaged over all classes (the standard benchmark). Pixel accuracy is misleading under class imbalance; labelling everything 'background' can score high.<br /><em>Session 12 · conceptual</em>

</details>

## Further reading

- [Torchvision DeepLabV3](https://docs.pytorch.org/vision/stable/models/deeplabv3.html) for a current semantic family.
- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html) for an instance family.
- [Meta SAM 3 research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) for concept-prompted masks and tracking.
- Built from the course lecture "cv-s12-semantic-metrics" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Reading a mask report

A report that gives “mIoU 0.80” without a class list is incomplete. Ask whether background is included, how empty classes are handled and whether the result is a mean of images or a pooled class intersection and union. In a scene with many background pixels and a small critical object, a seemingly strong average may coexist with failure on the object. Per-class IoU, support and visual examples make the number interpretable. The synthetic two-class code shows how one-third target IoU remains visible despite a much larger background set.

Look at the unit of evaluation. A corpus of equally sized images can still contain objects of very different sizes. Averaging each image's Dice gives a tiny object the same image weight as a large object in another image. Pooling all pixels gives large objects more weight. Neither is automatically right. If each patient, product or road tile is the unit of decision, an image-level or entity-level aggregate may be appropriate. If total area error is the product measure, a pooled or physical-area-weighted result may be more useful. State the choice explicitly.

Review cases at the resolution of action. A mask may look good when shrunk into a report but miss a one-pixel gap that determines connectivity. It may have high IoU yet include a false thin spur into a forbidden region. Conversely, a minor boundary displacement may lower IoU on a thin target without changing the product decision. Add error categories and quantitative checks for the shapes that matter. Retain a gallery of representative true positives, false positives, false negatives and ambiguous labels for reviewers.

When a model predicts every pixel as background, pixel accuracy may still be high on a mostly background dataset. Compute the rare class's true positives, false positives and false negatives. Its IoU will expose the missed region when ground truth includes it. If both prediction and truth omit the class on many images, decide whether those images should contribute a perfect score or be excluded for that class. Reporting only nonempty examples can also overstate a system's ability to recognise absence, so present an absence metric or false-positive rate separately.

For instance segmentation, matching predicted objects to ground-truth objects adds another layer. A semantic mask of two adjacent cars can have a strong car-class IoU while failing to separate the cars. Object-level evaluation must specify a matching threshold, confidence sorting and handling of duplicates. The next detection chapter introduces box IoU and suppression; mask-level instance evaluation follows similar matching principles but measures mask overlap. Keep semantic and instance results in separate tables so a reader knows what was measured.

The lecture's algebra also supports a quick consistency check. For a fixed nonzero IoU, Dice is $2I/(1+I)$. If a report lists IoU 0.333 and Dice 0.333 for the exact same binary masks and aggregation, something is wrong: the corresponding Dice should be about 0.5. Different averaging across images can break this simple relationship between reported means, so check raw per-example values before declaring an arithmetic bug. The equation holds for each same pair of masks, not necessarily for two separately averaged summary columns.

Finally, keep capture and mask provenance. An annotation drawn on an upscaled image can be shifted when mapped back to original pixels. A model output may be cropped or padded before scoring. Store the transforms and evaluate after mapping both sets to the same grid. Tiny coordinate mistakes can dominate thin-object metrics and waste model-tuning effort. Test alignment with a synthetic shape whose location is known exactly before trusting a full dataset score.

## Check yourself

- I can distinguish semantic and instance masks from their output contracts.
- I can compute union, IoU and Dice from set sizes and overlap.
- I can explain why pixel accuracy and a single mIoU can hide a rare-class failure.
- I can state an empty-mask and aggregation convention in an evaluation report.
