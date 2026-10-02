---
id: cv-image-classification
title: "Computer Vision · Sessions 9–10; Image Classification"
sidebar_label: "1 · Image classification"
sidebar_position: 1
slug: /theory/cv/image-classification
description: "Trace pixels to class scores, compute softmax and precision, recall and F1, and design a reliable evaluation."
tags: [computer-vision, classification, softmax, evaluation]
---

import Infographic from '@site/src/components/Infographic';
import ClassificationMetricsLab from '@site/src/components/viz/ClassificationMetricsLab';

**In one line.** Classification assigns an image-level label from learned visual evidence, and evaluation asks which errors that decision makes.

## The idea in plain words

:::note Beyond the lecture

The decision-threshold analysis, failure cases, runnable checks and design discussion extend the lecture. Its semantic-gap outline, model sequence, worked examples and all five practice questions remain below.

:::

A photograph is a grid of measured intensities, while a class label is a statement about an object or scene. The same object can occupy different positions, be viewed from different angles, appear under different lighting and be partly hidden. Conversely, two classes may look similar at the resolution available. The lecture calls this mismatch the **semantic gap**. An image classifier learns a rule from labelled examples that maps the observed pixels to class scores. It cannot infer a stable label merely because pixels have a particular brightness or an edge appears in one place. The training examples and evaluation set must represent the changes the deployed system will encounter.

The model sequence in the lecture illustrates successive ways to make that rule. A nearest-neighbour classifier compares the new image representation with saved examples, so the representation and distance measure determine which examples look close. A linear classifier maps a feature vector to class scores using a weighted sum. A convolutional neural network learns local filters and combines their responses over a larger receptive field. A vision transformer divides an image into patches and combines their representations using attention. These are architecture families, not an automatic ranking of quality. The right comparison holds the dataset, input resolution, compute budget and evaluation procedure fixed.

The deep-learning details of convolution, pretrained backbones and transfer learning are developed in the existing [CNN chapter](/docs/theory/dnn/what-a-convolutional-neural-network-is), [pretrained CNN chapter](/docs/theory/dnn/pretrained-cnn-models-and-imagenet) and [transfer-learning chapter](/docs/theory/dnn/transfer-learning-feature-extraction-vs-fine-tuning). Here the focus is the computer-vision decision: what is being classified, how scores become predictions and how to measure useful performance. An image-level label cannot tell an application where the object is. If the task requires a location or pixel boundary, detection or segmentation is a different output contract.

A model's logits are unrestricted class scores. Softmax exponentiates and normalises them to nonnegative values summing to one. Subtracting the largest logit before exponentiation avoids avoidable overflow without changing the result. For logits $[2,1,0]$, the probabilities round to $[0.665,0.245,0.090]$. A high softmax output is not by itself proof that a model is calibrated or that an image belongs to one of its known classes. Softmax only compares the supplied classes under this model. Distribution shift and unknown classes need separate evaluation.

<Infographic src="/img/cv/classification.svg" alt="Image classification board showing logits two, one and zero; softmax probabilities point six six five, point two four five and point zero nine zero; and precision, recall and F1 from forty true positives, ten false positives and twenty false negatives." caption="An individual score vector and an aggregate error measure are different pieces of evidence." />

## How it works

### The semantic gap

Pixels vary with viewpoint, lighting, deformation, clutter; but the label is constant. Data-driven learning bridges the gap.

### k-NN → CNN → ViT

- **k-NN / linear**; Majority of nearest neighbours; or f=Wx+b with softmax/SVM loss.
- **CNN**; Conv+ReLU, pooling, FC, softmax; learns edge→part→object features.
- **ViT**; Patches + self-attention; hardware-efficient, scales with data.

:::tip

**Worked.** Logits [2,1,0] → softmax [0.665, 0.245, 0.090]. TP=40,FP=10,FN=20 → P=0.80, R=0.667, F1=0.727.

:::

### Evaluation

Accuracy misleads on imbalance. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R). Softmax→probabilities; cross-entropy trains them.

### Key takeaways

- **1 · Semantic gap**; Learn label from varying pixels.
- **2 · Models**; k-NN→linear→CNN→ViT.
- **3 · Metrics**; Softmax; precision/recall/F1.

## A real system that works this way

The official Torchvision model catalogue lists image-classification model families and their preprocessing transforms. It is an example of why a deployed classifier must record more than the architecture name. A set of learned weights was trained with particular input sizes, colour conventions and normalisation. Changing a resize or channel order at serving time changes the data the model receives, even if the weight file and the class head are identical. The catalogue and model documentation were opened on 2026-10-02; this chapter does not claim to have benchmarked any of those model families.

Consider a factory line that flags images of damaged packages. The product decision is not necessarily the largest class score. A false negative might allow a damaged package through; a false positive might send a good package for manual inspection. The decision threshold therefore follows a cost and capacity policy. The confusion counts in the lecture, TP 40, FP 10 and FN 20, yield precision 0.8 and recall 2/3 at one such threshold. Increasing the threshold often reduces the number flagged, with a possible gain in precision and loss in recall. The exact trade-off must be measured on held-out examples; those three counts cannot predict what another threshold would do.

The same line may see several camera views, packaging colours and lighting conditions. A random image split can leak near-duplicates from the same product run into both training and validation. Split by the entity or acquisition period that will be new in production, and report metrics separately by camera, product type and lighting condition. Inspect the false negatives as images. Some may reflect missing evidence because the damage is outside the crop; no classifier architecture can recover an unseen defect. Others may show a consistent appearance that belongs in training. That distinction guides whether to change the camera, data, label policy or model.

## Code you can run

The first block computes the lecture's softmax with a stable shift. The three printed values reproduce its rounded probabilities, and the assertion checks that they sum to one. These values are a mathematical transformation of logits, not measured correctness or calibration.

```python
from math import exp, isclose

logits = [2.0, 1.0, 0.0]
shifted = [exp(value - max(logits)) for value in logits]
probabilities = [value / sum(shifted) for value in shifted]
print('Softmax:', [round(value, 3) for value in probabilities])
assert [round(value, 3) for value in probabilities] == [0.665, 0.245, 0.09]
assert isclose(sum(probabilities), 1.0)
```

The second block computes precision, recall and F1 directly from the same confusion counts. The algebraic F1 form, $2TP/(2TP+FP+FN)$, avoids a second calculation from rounded precision and recall. It produces **0.727** to three decimals. True negatives are not needed for these three metrics, but are needed to calculate accuracy and specificity.

```python
tp, fp, fn = 40, 10, 20
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * tp / (2 * tp + fp + fn)
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}')
assert (round(precision, 3), round(recall, 3), round(f1, 3)) == (0.8, 0.667, 0.727)
```

The lab begins with both of these worked examples. Move the first logit to see the single-image probabilities change; move TP, FP and FN to see the aggregate metrics change. They are separate controls because a single example's logits do not determine a dataset's confusion matrix. The table view exposes every value without relying on the lengths or colours of the bars.

<ClassificationMetricsLab />

If a selected class has no predicted positives, its precision denominator is zero; if there are no actual positives, its recall denominator is zero. An evaluation report must state its convention for undefined metrics. Setting them silently to one would reward a classifier that never predicts a rare class. The lab calls them undefined when the relevant denominator is zero. In a multiclass problem, define whether counts are per class, micro-averaged or macro-averaged before interpreting one F1 number.

## Designing with it

Define the unit of prediction. A classifier might label an entire image, a crop around a known object or a frame from a video. If the frame contains three objects, one image-level class can be ambiguous. Document whether multiple labels are allowed and how “unknown”, “unclear” and “no target” are represented. Label-policy ambiguity is often larger than a change between two model architectures. Review disagreements with domain experts and keep a small set of adjudicated examples for regression checks.

Choose a split that matches use. If future items come from new production lots, splitting individual images randomly can overstate generalisation because neighbouring frames and near-identical packages leak into both sets. Hold out acquisition sessions, devices or time periods as appropriate. Deduplicate before splitting and record how many images, entities and classes remain in each partition. A headline score without these details is hard to trust. A small rare class may have too few held-out cases for a stable estimate, so show counts and uncertainty rather than only three decimal places.

Pick metrics around the cost of errors. Accuracy counts all decisions equally and can look high when the majority class dominates. Precision asks how often a positive prediction is right, while recall asks how many actual positives were found. F1 balances the two by their harmonic mean but does not encode the asymmetric cost of a missed defect versus an unnecessary inspection. If a product has an explicit review capacity, measure how many examples can be sent to people per day and select a threshold on a validation set under that capacity. Report performance at the selected threshold on a separate test set.

Check calibration before using score magnitudes as risk estimates. Softmax outputs always sum to one even for nonsense inputs. A reliability diagram or held-out calibration measure can reveal whether predictions at a stated confidence level are correct at a corresponding rate. Calibration can drift when the camera or object mix changes. Track score distributions, class prevalence, human overrides and delayed ground truth. A model that still produces scores and labels can be silently failing if input quality changes.

Plan a fallback for uncertain or unfamiliar images. A low maximum score may be useful as one signal, but it is not a guaranteed detector of unknown classes. Consider explicit quality checks for blur, obstruction and invalid crops, and a review path for uncertain cases. The fallback must be tested in the full workflow, including operator load and latency. A classifier that works in an offline notebook may cause a queue of unreviewed images when used at line speed.

Finally, audit the preprocessing path. Record colour channel order, resize rule, crop, interpolation, intensity scaling and normalisation with the model artifact. Test the same example through training and serving paths and compare tensors before the model. Even a simple RGB/BGR mismatch can shift every class score. A model-family comparison is useful only after the input contract and evaluation protocol are stable.

## Where this stands in 2026

:::info Industry view

- Current Torchvision documentation, opened on 2026-10-02, provides several classification model families with weight-specific preprocessing. No speed or accuracy ranking is asserted here.
- The lecture describes ViTs as hardware-efficient. That is workload and implementation dependent; attention, token count and memory traffic can make a particular ViT more or less efficient than a particular CNN on a specific device.
- The two Python blocks use standard-library arithmetic and were run locally. No image dataset, trained weights or factory-line deployment was used for the examples.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What is the semantic gap in image classification?</summary>

The mismatch between low-level pixels (which vary with viewpoint, lighting, deformation, clutter) and the constant high-level label; data-driven learning bridges it.<br /><em>Sessions 9-10 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast k-NN, linear classifiers and CNNs.</summary>

k-NN: majority label of nearest neighbours (no training, slow test). Linear: f=Wx+b with softmax/SVM loss. CNN: learns a conv/pool feature hierarchy end-to-end.<br /><em>Sessions 9-10 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Compute softmax of logits [2, 1, 0].</summary>

e²=7.389, e¹=2.718, e⁰=1, sum=11.107 → [0.665, 0.245, 0.090].<br /><em>Sessions 9-10 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> TP=40, FP=10, FN=20. Compute precision, recall and F1.</summary>

Precision=40/50=0.80, recall=40/60=0.667, F1=2(0.8)(0.667)/(1.467)=0.727.<br /><em>Sessions 9-10 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> Why can accuracy be misleading, and what is a ViT?</summary>

Accuracy misleads on imbalanced data (a majority-class predictor scores high). A Vision Transformer splits the image into patches and uses self-attention; hardware-efficient and scalable.<br /><em>Sessions 9-10 · conceptual</em>

</details>

## Further reading

- [Torchvision classification models](https://docs.pytorch.org/vision/stable/models.html#classification) for current model-family and weights documentation.
- [Stanford CS231n notes](https://cs231n.github.io/) for classifier objectives and visual-recognition foundations.
- Built from the course lecture "cv-s9-10-image-classification" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source claim

The lecture calls ViTs “hardware-efficient” and scalable. The scalability description concerns how the architecture can use data and compute, but hardware efficiency is not an intrinsic guarantee. Compare a specific CNN and ViT at the same task quality, input size, device, batch size and latency target before drawing an efficiency conclusion. The lecture's model sequence is retained above.

:::

## Reading a classification report

Suppose a report says “99% accuracy” on a set with 99 normal images and one damaged image. Predicting normal every time achieves that accuracy while finding no damage. The report needs class prevalence, a confusion matrix, recall for the damaged class and the cost of a missed case. When there is only one positive test example, recall is either zero or one and is extremely unstable. A test with more independent positive cases is needed before shipping a threshold. The number 99% here is a counterexample constructed from counts, not an empirical benchmark for any named model.

A different report gives precision 0.8, recall 0.667 and F1 0.727. Those values are internally consistent with the lecture's TP, FP and FN, but they describe only the evaluated set and threshold. They say nothing about whether false positives cluster in a specific shift, whether all false negatives are tiny defects, or how many true negatives were present. Ask for the confusion matrix by slice, sample images and the threshold selection procedure. If a system exposes softmax outputs, ask whether they have been calibrated on data separated from training and whether the current inputs resemble that data.

Look for the annotation unit. A package with two defects may appear once in image-level counts, twice in object-level counts, or as several pixel regions in segmentation. Mixing these evaluation units can produce conflicting metrics without any arithmetic error. The unit must match the action the product takes: reject a package, draw a box for an inspector, or estimate defect area. Image classification is a good choice when the decision is genuinely global and enough evidence is visible in a standardised view.

Also distinguish validation from monitoring. During development, ground-truth labels permit precision and recall measurement. During live use, ground truth may arrive late or only for inspected cases. A dashboard of average softmax score is not a substitute for sampled labelled audits. Monitor the capture pipeline and review a designed sample of outputs to estimate ongoing errors. Otherwise a drift in lighting or crop can make the score distribution look confident while actual recall falls.

When comparing model families, include the full inference pipeline. Input resize and normalisation, CPU-to-device transfer, batch size and postprocessing can dominate latency. A model that runs quickly on a benchmark GPU may be slow on the target edge device. Conversely, an architecture with fewer parameters is not guaranteed to have lower latency if its operations map poorly to that hardware. State the measurement setup alongside any speed claim. The source outline names architecture families; it does not provide a fair deployment benchmark, so this chapter does not invent one.

Finally, inspect what the classifier learned. If damage examples were all photographed on a blue mat and normal examples on a grey mat, a high test score after a random split may reflect the mat rather than damage. Test counterexamples where the background is swapped or acquisition changes. Evaluate under occlusion and low light, and check whether the crop contains the feature that supports the label. Such tests connect the semantic gap back to data collection: the model receives pixels, so any repeated pixel cue can become its shortcut.

## Check yourself

- I can explain the semantic gap without assuming a fixed object appearance.
- I can turn logits into a stable softmax calculation and state what it does not prove.
- I can compute precision, recall and F1 from TP, FP and FN and handle zero denominators.
- I can choose a split and metric that match an image-level product decision.
