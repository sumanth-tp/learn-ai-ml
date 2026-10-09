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

:::tip Before you start
**You should already know**

- What a mask is and how a region of pixels is labelled: [classical image segmentation](/docs/theory/cv/classical-image-segmentation).
- What a convolutional network does, in outline: [what a convolutional neural network is](/docs/theory/dnn/what-a-convolutional-neural-network-is).
- Precision and recall from a confusion matrix, since IoU is a close relative.

**Reading time.** About 45 minutes, plus a minute to run the code.

**After this chapter you can**

- compute IoU, Dice, pixel accuracy and mean IoU by hand from pixel counts,
- show why pixel accuracy hides a missed rare class,
- say which aggregation (per image or pooled) and which empty-class rule a report used.
:::

## In 30 seconds

A segmentation model colours every pixel. To grade it, you compare its colouring with a reference colouring. Pixel accuracy asks "what share of pixels match?", which a lazy model passes by painting everything as the common background. IoU asks "of all pixels either mask claims for this class, how many do both claim?", which a lazy model fails for the rare class. Think of grading a map of lakes: if lakes cover 1% of the map, a blank map is 99% right and completely useless.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Semantic mask | A class label for every pixel | Road, sky, car |
| Instance mask | A separate mask per object | Car 1 and car 2 |
| Intersection | Pixels both masks call the class | 50 pixels |
| Union | Pixels either mask calls the class | 150 pixels |
| IoU | Intersection divided by union | 50 / 150 = 0.333 |
| Dice | Twice the intersection over the sum of mask sizes | 100 / 200 = 0.5 |
| Pixel accuracy | Share of pixels labelled correctly | 0.96 for an all-background guess |
| Mean IoU (mIoU) | IoU averaged over classes | (0.96 + 0) / 2 = 0.48 |
| Support | How many reference pixels a class has | 4 pixels for the rare class |

## The idea in plain words

:::note Beyond the course material

The count-only lab, empty-mask convention, per-class evaluation design and current model-family context extend the course material. Its architecture outline, worked IoU and Dice example and all five practice questions remain below.

:::

An image-level classifier returns one label for a whole frame. A semantic segmenter returns a class for every pixel, producing a map of roads, sky, cars or other classes defined by the task. If two cars touch or overlap, their pixels may all share the same semantic label “car”. Instance segmentation adds a distinct identity for each object, often as a set of masks plus classes and scores. Panoptic segmentation combines things that have instances with stuff regions such as sky, under an explicit labelling policy. These output contracts cannot be substituted casually: a traffic-counting product needs separated cars, while a road-area estimator may need only the road class.

Fully convolutional networks and encoder-decoder designs are the usual starting point. An encoder builds contextual features while reducing spatial resolution; a decoder produces an output map at the required resolution. Skip connections in U-Net-style designs carry higher-resolution features from earlier stages, which can help recover boundary detail. Upsampling does not reconstruct every detail that was absent or discarded in the input. The annotation resolution, image scale and choice of loss all influence thin structures and small objects. For a deeper treatment of convolution and pretrained vision backbones, follow the existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is) and [transfer-learning chapter](/docs/theory/dnn/transfer-learning-feature-extraction-vs-fine-tuning).

For one class, let $P$ be the set of pixels predicted positive and $G$ the reference set. The intersection contains pixels in both. The union contains pixels in either. Intersection over union is $|P\cap G|/|P\cup G|$. Dice is $2|P\cap G|/(|P|+|G|)$. When each mask has 100 pixels and they overlap in 50, their union is 150, so IoU is $50/150=0.333$ and Dice is $100/200=0.5$. These are two views of the same overlap: for nonempty union, Dice equals $2\mathrm{IoU}/(1+\mathrm{IoU})$ and is at least IoU. They cannot reveal whether the missed pixels lie at a harmless outer edge or across a clinically or operationally critical structure.

Mean IoU averages IoU over classes under a specified class-inclusion rule. Pixel accuracy counts correctly labelled pixels and can be dominated by the background. If a rare class occupies a small region, predicting background everywhere can score well on pixel accuracy while giving that class zero recall. Always show per-class support and IoU alongside an average, and explain whether absent classes are ignored, scored as perfect when both masks are empty, or treated another way. That convention can change a mean substantially in small datasets.

<Infographic src="/img/cv/semantic-metrics.svg" alt="Mask metric board: a 100-pixel predicted mask and 100-pixel reference mask overlap in 50 pixels; union 150 gives IoU one third and Dice one half, with a reminder to report each class." caption="The denominator and empty-class policy must be stated before an overlap score is compared." />

## Worked example, step by step

One strip of 100 pixels. The reference has 96 background pixels and 4 rare-class pixels. Two models are graded. The block that uses `accuracy_score`, under "Code you can run", reproduces every number.

1. Model A paints everything background. Pixel accuracy is 96 / 100 = 0.96.
2. For the background class: intersection 96, union 96 + 4 − 96 = 100 (the four rare pixels are wrongly claimed), so IoU is 0.96.
3. For the rare class, model A claims nothing, so the intersection is 0 and IoU is 0. Mean IoU is (0.96 + 0) / 2 = 0.48.
4. Model B finds 2 of the 4 rare pixels and raises one false alarm at pixel 50. Rare class: intersection 2, union 4 + 3 − 2 = 5, so IoU is 2 / 5 = 0.4.
5. Rare-class Dice is 2 × 2 / (4 + 3) = 4 / 7 = 0.571. Check: 2 × 0.4 / 1.4 = 0.571, so Dice = 2 IoU / (1 + IoU).
6. Background for model B: 95 pixels are right, 2 rare pixels are wrongly called background and 1 background pixel is wrongly called rare. Intersection 95, union 95 + 2 + 1 = 98, IoU 95 / 98 = 0.969.
7. Pixel accuracy for B is (95 + 2) / 100 = 0.97 and mean IoU is (0.969 + 0.4) / 2 = 0.685.

In words: accuracy moved from 0.96 to 0.97, a gain of one point, while the rare class went from invisible to 0.4. The rare-class IoU is the number that shows the improvement.

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

Imagine mapping water across aerial tiles. A semantic water mask can estimate covered area, but a shore boundary may move with image resolution and annotation judgement. If a 100-pixel predicted water patch overlaps the 100-pixel annotated patch by 50, the overlap metrics are exactly reproducible. However, one tile with a small stream and another with a huge lake should not necessarily receive equal operational weight. Macro-averaging tile scores gives each tile equal weight, while aggregating intersections and unions across tiles weights pixels differently. Choose the aggregation that matches the decision and report it.

The same project should stratify by cloud, shade, season and sensor. A water-looking roof may create a false positive; a shadowed stream may be missed. Review the actual masks over the source images and quantify small-object and boundary errors separately. If area is converted to physical units, account for pixel scale and projection. The toy overlap arithmetic is necessary to understand the metric, but does not provide those geospatial guarantees.

## Code you can run

The first block reproduces the worked mask example using counts. The intersection must not exceed either mask size. The union follows inclusion-exclusion, and the two metrics use different denominators.

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

**What each control does.** "Predicted mask pixels" and "Ground-truth mask pixels" set the two mask sizes. "Overlapping pixels" sets the intersection and cannot exceed the smaller mask. The table shows union, IoU and Dice.

**Try it yourself.**

1. At the defaults (100, 100 and 50) the union is 150, IoU is 0.333 and Dice is 0.500, the numbers in the chapter's first block.
2. Raise the overlap to 100. IoU and Dice both reach 1.000, the only point where they are equal. Everywhere else Dice is larger, as Dice = 2 IoU / (1 + IoU) predicts.
3. Set the predicted mask to 200 pixels with overlap 50 and ground truth 100. The union is 250, IoU falls to 0.200 and Dice to 0.333. A mask that is too big is penalised even though it covers the whole reference.

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

The same strip in code, using scikit-learn's `accuracy_score`, `jaccard_score` (IoU) and `f1_score` (which equals Dice for a binary class). It prints the numbers from the worked example.

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

truth = np.zeros(100, int)
truth[:4] = 1
all_background = np.zeros(100, int)
partial = np.zeros(100, int)
partial[:2] = 1
partial[50] = 1
for name, pred in (('all background', all_background), ('finds 2 of 4, one false alarm', partial)):
    iou = jaccard_score(truth, pred, labels=[0, 1], average=None, zero_division=0)
    dice = f1_score(truth, pred, labels=[0, 1], average=None, zero_division=0)
    print(f'{name}: accuracy {accuracy_score(truth, pred):.3f}, IoU {np.round(iou, 3)}, mIoU {iou.mean():.3f}, rare Dice {dice[1]:.3f}')
```

**Reading the output.** Model A has accuracy 0.960 and mIoU 0.480. Model B has accuracy 0.970, rare IoU 0.4, mIoU 0.685 and rare Dice 0.571, as in steps 1 to 7. If your own report shows a high accuracy and a rare IoU of 0.000, the model is predicting the common class everywhere.

Two empty masks have zero union, so $0/0$ is undefined. Some benchmarks assign a perfect score to a class absent from both prediction and truth, while others exclude that class; either choice must be stated. The lab marks this case undefined and avoids an arbitrary score. A per-image average and a corpus-level aggregate can also differ even with the same empty-class rule.

### Experiment: what each metric hides

The question: given models with a known kind of mistake, which metric tells you? We build one 128 by 128 label map with three classes: background, a road band across the bottom (9.4% of pixels), and a rare class made of one thin bar and one small square (0.83% of pixels). Six predictions are made by hand with a known error each. Every metric is computed from a confusion matrix and then asserted equal to scikit-learn 1.9.1 (`jaccard_score`, `f1_score`). The last two parts test how IoU responds to a 1 pixel shift on bars of different widths, and how per-image and pooled averages differ over 20 squares of random size.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

def scene(size=128):
    truth = np.zeros((size, size), int)
    truth[116:, :] = 1
    truth[20:44, 60:63] = 2
    truth[100:108, 20:28] = 2
    return truth

def shift(mask, dx):
    return np.roll(mask, dx, axis=1)

def report(name, truth, pred):
    labels = [0, 1, 2]
    cm = confusion_matrix(truth.ravel(), pred.ravel(), labels=labels)
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.divide(inter, union, out=np.zeros(3), where=union > 0)
    dice = np.divide(2 * inter, cm.sum(0) + cm.sum(1), out=np.zeros(3), where=union > 0)
    assert np.allclose(iou, jaccard_score(truth.ravel(), pred.ravel(), labels=labels, average=None, zero_division=0))
    assert np.allclose(dice, f1_score(truth.ravel(), pred.ravel(), labels=labels, average=None, zero_division=0))
    accuracy = inter.sum() / cm.sum()
    print(f'{name:22s} pixel acc {accuracy:.3f}  mIoU {iou.mean():.3f}  rare IoU {iou[2]:.3f}  rare Dice {dice[2]:.3f}')

truth = scene()
rng = np.random.default_rng(0)
print('rare class share of pixels:', round(float((truth == 2).mean()), 4))
everything_background = np.zeros_like(truth)
report('all background', truth, everything_background)
rare_missing = truth.copy(); rare_missing[truth == 2] = 0
report('rare class missed', truth, rare_missing)
for dx in (1, 2):
    moved = truth.copy(); moved[truth == 2] = 0
    moved[shift(truth == 2, dx)] = 2
    report(f'rare shifted {dx} px', truth, moved)
halo = truth.copy()
big = np.zeros(truth.shape, bool); big[truth == 2] = True
for _ in range(2):
    big = big | np.roll(big, 1, 0) | np.roll(big, -1, 0) | np.roll(big, 1, 1) | np.roll(big, -1, 1)
halo[big & (truth == 0)] = 2
report('rare dilated 2 px', truth, halo)
noisy = truth.copy(); flip = rng.random(truth.shape) < 0.05
noisy[flip] = rng.integers(0, 3, flip.sum())
report('5% random noise', truth, noisy)

widths = [1, 2, 3, 5, 9, 17, 33]
print('width  IoU after a 1 px shift   Dice')
for w in widths:
    a = np.zeros((64, 64), bool); a[8:56, 20:20 + w] = True
    b = shift(a, 1)
    i = (a & b).sum() / (a | b).sum()
    print(f'{w:5d}  {i:.3f}                   {2 * i / (1 + i):.3f}')

ious, dices, inters, unions, sizes = [], [], 0, 0, 0
for k in range(20):
    side = int(rng.integers(3, 30))
    a = np.zeros((64, 64), bool); a[10:10 + side, 10:10 + side] = True
    b = shift(a, 2)
    inter, union = (a & b).sum(), (a | b).sum()
    ious.append(inter / union); dices.append(2 * inter / (a.sum() + b.sum()))
    inters += inter; unions += union
print(f'per-image mean IoU {np.mean(ious):.3f}  pooled IoU {inters / unions:.3f}')
print(f'mean Dice {np.mean(dices):.3f}  Dice implied by mean IoU {2 * np.mean(ious) / (1 + np.mean(ious)):.3f}')
```

**Reading the output.** The first six rows grade the six predictions: pixel accuracy, mIoU over the three classes, and the rare class's IoU and Dice. The width table shows the IoU of a vertical bar after a 1 pixel horizontal shift. The last two lines compare ways of averaging over 20 shifted squares.

**Line by line.**

- `report` builds the confusion matrix first. The diagonal is the intersection, and the union is row sum plus column sum minus the diagonal, which is the whole IoU definition.
- The two `assert np.allclose` lines compare the hand-built IoU and Dice with scikit-learn on every row, so a mistake in the formula would stop the script.
- `shift` rolls a mask sideways, which is a controlled stand-in for a boundary placed slightly wrong.
- In the last loop, `inters` and `unions` are summed over images before dividing, which is the pooled score.

The printed output was:

```text
rare class share of pixels: 0.0083
all background         pixel acc 0.898  mIoU 0.299  rare IoU 0.000  rare Dice 0.000
rare class missed      pixel acc 0.992  mIoU 0.664  rare IoU 0.000  rare Dice 0.000
rare shifted 1 px      pixel acc 0.996  mIoU 0.872  rare IoU 0.619  rare Dice 0.765
rare shifted 2 px      pixel acc 0.992  mIoU 0.784  rare IoU 0.360  rare Dice 0.529
rare dilated 2 px      pixel acc 0.989  mIoU 0.806  rare IoU 0.430  rare Dice 0.602
5% random noise        pixel acc 0.965  mIoU 0.698  rare IoU 0.301  rare Dice 0.463
width  IoU after a 1 px shift   Dice
    1  0.000                   0.000
    2  0.333                   0.500
    3  0.500                   0.667
    5  0.667                   0.800
    9  0.800                   0.889
   17  0.889                   0.941
   33  0.941                   0.970
per-image mean IoU 0.705  pooled IoU 0.807
mean Dice 0.816  Dice implied by mean IoU 0.827
```

**What the numbers say.** Pixel accuracy ranks the models almost backwards for a user who cares about the rare class. The model that misses the rare class entirely scores 0.992, higher than the model that has 5% random label noise (0.965), yet its rare IoU is 0.000 against 0.301 for the noisy one. Mean IoU reverses the ranking (0.664 against 0.698), because it counts the rare class as one third of the grade. Even mIoU is generous: a 2 pixel shift of the rare class leaves pixel accuracy at 0.992, identical to missing it, while rare IoU is 0.360.

The width table is the practical lesson on thin structures. A 1 pixel boundary error costs a 33 pixel wide object 0.059 of IoU (0.941) but costs a 3 pixel bar half of it (0.500) and a 1 pixel line all of it (0.000). A model that is visually right can still score badly on lines, vessels or lane markings, so thin classes need a boundary-tolerant metric alongside IoU.

The last two lines show that "the mean" is not one number. Averaging per image gave IoU 0.705 and pooling intersections and unions gave 0.807, because pooling lets the larger squares dominate. The per-image mean Dice is 0.816, while Dice computed from the mean IoU would be 0.827, so the relation Dice = 2 IoU / (1 + IoU) holds for each mask pair but not for two separately averaged summary columns.

<Infographic src="/img/cv-enrich/v3-mask-metrics.svg" alt="Bars compare pixel accuracy with rare-class IoU for four predictions, with a table of IoU after a one pixel shift for bars of width 1 to 33 and a card on per-image against pooled IoU." caption="Look first at the 'rare class missed' pair: accuracy 0.992, rare IoU 0.000." />

Limits: synthetic shapes, one fixed shift direction, a single seed for the noise and the square sizes, and no trained network. The numbers show how the metrics behave, not how any model performs.

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
- The IoU and Dice figures reproduce from the stated counts. The empty-union case requires an evaluation convention and is explicitly undefined in the local lab.
- The water-mapping case is a design scenario, not a claim about a named deployed system. No image dataset or trained model was run.

:::

## Common mistakes

- **Reporting pixel accuracy alone.** It feels natural because it is the classification metric. A model that misses a rare class entirely still scored 0.992. Report per-class IoU and the support of each class.
- **Quoting one mIoU without the rules.** The number looks complete. It changes with whether background is included, how classes absent from both masks are scored, and whether images are averaged or pooled (0.705 against 0.807 here). State all three.
- **Treating IoU as a boundary metric.** It seems to cover boundaries because it uses overlap. On thin objects, a 1 pixel shift costs 0.5 IoU at width 3. Add a boundary-distance measure when thin structures matter.
- **Averaging Dice and IoU columns and expecting the formula to hold.** The identity Dice = 2 IoU / (1 + IoU) is per mask pair. Compare raw per-example values before declaring a bug.
- **Scoring a resized mask against the original grid.** A one pixel misalignment looks harmless on screen but dominates thin-object scores. Map both masks to one grid first.

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

<details>
<summary><strong>Q6 (Easy).</strong> A strip has 96 background pixels and 4 rare pixels. A model paints everything background. What are its pixel accuracy and mean IoU?</summary>

Accuracy is 96 / 100 = 0.96. Background IoU is 96 / 100 = 0.96 and rare IoU is 0, so mean IoU is 0.48.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> In the experiment, "rare class missed" has pixel accuracy 0.992 and "5% random noise" has 0.965. Which has the higher mIoU, and why does the order flip?</summary>

The noisy model has the higher mIoU (0.698 against 0.664). Accuracy weights every pixel equally, and the rare class is 0.83% of pixels, so missing it costs almost nothing. mIoU weights each class equally, so the rare class's IoU of 0.000 pulls the mean down by a third of the grade.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> A road-map team averages IoU per image and another team pools pixels across images. Under what data does the pooled score exceed the per-image mean, and which would you report for a product that measures total road area?</summary>

Pooling lets big objects dominate. Here, per-image IoU was 0.705 and pooled IoU 0.807, because small squares lose more from the same 2 pixel shift. If the product measures total area, pooled or area-weighted is closer to the decision. If each image or patient is the unit of decision, report per-image and show the distribution.

</details>

## Further reading

- [Torchvision DeepLabV3](https://docs.pytorch.org/vision/stable/models/deeplabv3.html) for a current semantic family.
- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html) for an instance family.
- [Meta SAM 3 research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) for concept-prompted masks and tracking.
- [scikit-learn 1.9.1 `jaccard_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html), opened 2026-10-09: per-class scores with `average=None`, and the `zero_division` setting for empty classes.
- Built from the course lecture "cv-s12-semantic-metrics" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


## Reading a mask report

A report that gives “mIoU 0.80” without a class list is incomplete. Ask whether background is included, how empty classes are handled and whether the result is a mean of images or a pooled class intersection and union. In a scene with many background pixels and a small critical object, a seemingly strong average may coexist with failure on the object. Per-class IoU, support and visual examples make the number interpretable. The synthetic two-class code shows how one-third target IoU remains visible despite a much larger background set.

Look at the unit of evaluation. A corpus of equally sized images can still contain objects of very different sizes. Averaging each image's Dice gives a tiny object the same image weight as a large object in another image. Pooling all pixels gives large objects more weight. Neither is automatically right. If each patient, product or road tile is the unit of decision, an image-level or entity-level aggregate may be appropriate. If total area error is the product measure, a pooled or physical-area-weighted result may be more useful. State the choice explicitly.

Review cases at the resolution of action. A mask may look good when shrunk into a report but miss a one-pixel gap that determines connectivity. It may have high IoU yet include a false thin spur into a forbidden region. Conversely, a minor boundary displacement may lower IoU on a thin target without changing the product decision. Add error categories and quantitative checks for the shapes that matter. Retain a gallery of representative true positives, false positives, false negatives and ambiguous labels for reviewers.

When a model predicts every pixel as background, pixel accuracy may still be high on a mostly background dataset. Compute the rare class's true positives, false positives and false negatives. Its IoU will expose the missed region when ground truth includes it. If both prediction and truth omit the class on many images, decide whether those images should contribute a perfect score or be excluded for that class. Reporting only nonempty examples can also overstate a system's ability to recognise absence, so present an absence metric or false-positive rate separately.

For instance segmentation, matching predicted objects to ground-truth objects adds another layer. A semantic mask of two adjacent cars can have a strong car-class IoU while failing to separate the cars. Object-level evaluation must specify a matching threshold, confidence sorting and handling of duplicates. The next detection chapter introduces box IoU and suppression; mask-level instance evaluation follows similar matching principles but measures mask overlap. Keep semantic and instance results in separate tables so a reader knows what was measured.

The algebra also supports a quick consistency check. For a fixed nonzero IoU, Dice is $2I/(1+I)$. If a report lists IoU 0.333 and Dice 0.333 for the exact same binary masks and aggregation, something is wrong: the corresponding Dice should be about 0.5. Different averaging across images can break this simple relationship between reported means, so check raw per-example values before declaring an arithmetic bug. The equation holds for each same pair of masks, not necessarily for two separately averaged summary columns.

Finally, keep capture and mask provenance. An annotation drawn on an upscaled image can be shifted when mapped back to original pixels. A model output may be cropped or padded before scoring. Store the transforms and evaluate after mapping both sets to the same grid. Tiny coordinate mistakes can dominate thin-object metrics and waste model-tuning effort. Test alignment with a synthetic shape whose location is known exactly before trusting a full dataset score.

## Check yourself

- I can distinguish semantic and instance masks from their output contracts.
- I can compute union, IoU and Dice from set sizes and overlap.
- I can explain why pixel accuracy and a single mIoU can hide a rare-class failure.
- I can state an empty-mask and aggregation convention in an evaluation report.

- I can compute pixel accuracy, per-class IoU, Dice and mean IoU from counts and check them against scikit-learn.
- I can explain why pixel accuracy ranked a model that missed the rare class above one with random noise, and why mIoU reverses that.
- I can predict how much a 1 pixel shift costs IoU for a thin object and for a wide one.
- I can state whether a reported mean was per image or pooled, and why the two differ.

## Where to go next

Next: [object detection and box evaluation](/docs/theory/cv/object-detection-and-box-evaluation), where IoU becomes the rule that matches boxes. Related: [classical image segmentation](/docs/theory/cv/classical-image-segmentation), which produces the masks scored here.
