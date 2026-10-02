---
id: cv-object-detection-and-box-evaluation
title: "Computer Vision · Session 13; Object Detection and Box Evaluation"
sidebar_label: "3 · Object detection"
sidebar_position: 3
slug: /theory/cv/object-detection-and-box-evaluation
description: "Compare detector output families, calculate box IoU, understand suppression and interpret average precision."
tags: [computer-vision, object-detection, iou, nms]
---

import Infographic from '@site/src/components/Infographic';
import BoxIouNmsLab from '@site/src/components/viz/BoxIouNmsLab';

**In one line.** An object detector returns a variable set of boxes, classes and scores; overlap rules turn those candidates into evaluated detections.

## The idea in plain words

:::note Beyond the lecture

The coordinate convention, current-family check, score-threshold and AP explanation and deployment design extend the lecture. Its R-CNN sequence, box example and all five practice questions remain below.

:::

Image classification returns a class for the frame. Object detection returns a class, score and bounding box for each candidate object, so it can say both what and where. A detector must also handle a variable number of objects, small targets, occlusion and duplicate predictions. A box is a coarse localisation: it encloses the object but does not mark its pixel boundary. Instance segmentation adds a mask for each object when that boundary matters. Define which output the product needs before choosing a detector family.

The lecture uses the historical R-CNN family to explain how computation was shared. Original R-CNN classified region proposals with separate feature extraction; Fast R-CNN shared an image feature map across proposals; Faster R-CNN learned proposals with a region proposal network. Two-stage refers to a proposal step followed by prediction refinement. One-stage detectors predict candidates over image features without a separate proposal network. This distinction says little by itself about actual latency or accuracy. Backbone, resolution, implementation, device and postprocessing all matter.

The source names YOLO and SSD and describes both as grid-based and anchor-based. That captures some historical variants, but anchor boxes are not a universal requirement for one-stage detection. FCOS is an anchor-box-free one-stage detector according to its original paper and current Torchvision catalogue. Current Ultralytics YOLO26 documentation also describes an optional one-to-one head that can emit predictions without a separate non-maximum suppression pass; its default one-to-many path uses NMS. These examples show why a present-day design review should name a specific variant and configuration instead of treating “YOLO” as one fixed algorithm. This chapter does not rank those families.

For two axis-aligned boxes, intersection over union is intersection area divided by union area. Let A cover coordinates `[0,0,10,10]` and B `[5,0,15,10]`, using continuous coordinates with right and bottom boundaries excluded for area arithmetic. Each area is 100; the shared rectangle has width 5 and height 10, so its area is 50. Union area is 150 and IoU is one third. Non-maximum suppression, or NMS, sorts candidate boxes by score and removes lower-scoring boxes with overlap above a chosen threshold, usually within a class. At threshold 0.5, one-third overlap is below the threshold, so both boxes remain. If the candidates actually refer to the same physical object, this threshold may be too permissive; NMS cannot know identity from IoU alone.

<Infographic src="/img/cv/detection.svg" alt="Object detection board showing two boxes of area 100 with overlap 50, union 150 and IoU one third; NMS at threshold one half keeps both." caption="Box IoU is a geometric ratio; a detection decision also needs scores, class and matching policy." />

## How it works

### The R-CNN family

- **R-CNN**; ~2000 region proposals, one CNN each; accurate, slow.
- **Fast R-CNN**; One CNN pass; RoI pooling per region.
- **Faster R-CNN**; Learned Region Proposal Network + anchor boxes.

### YOLO & SSD

Predict boxes + classes in one pass over a grid using anchor boxes; real-time, small accuracy trade-off.

:::tip

**Worked.** A=[0,0,10,10], B=[5,0,15,10] overlap 50 → IoU=50/150=0.333. At NMS threshold 0.5, both kept.

:::

### NMS & mAP

NMS keeps the top box and drops high-IoU overlaps. A detection is correct if IoU with ground truth > threshold (e.g. 0.5). mAP = mean area under the precision-recall curve over classes.

### Key takeaways

- **1 · Two-stage**; R-CNN → Fast → Faster (RPN, anchors).
- **2 · Single-shot**; YOLO/SSD, real-time.
- **3 · Eval**; NMS by IoU; mAP.

## A real system that works this way

The current Torchvision documentation lists FCOS and Mask R-CNN model builders, while the official Ultralytics YOLO26 guide describes detection, segmentation and optional NMS-free inference heads. These pages and the original FCOS paper were opened on 2026-10-02. They establish that current implementations span anchor-free, two-stage and different postprocessing choices. They do not establish which one is best for a local dataset, licence requirement, device or latency budget. The local code checks geometry only; it does not run trained detector weights.

Imagine a camera monitoring boxes on a conveyor. The system needs one location per physical package. A high-confidence candidate may overlap a second lower-confidence candidate for the same package, and NMS can remove the duplicate. But two packages can overlap in the image, so an aggressive NMS threshold might remove a real second package. A fixed threshold should be chosen from labelled examples that include crowded scenes, unusual box sizes and partial occlusion. The product should measure missed packages, duplicates and processing time after the entire capture and postprocessing path.

A detector score threshold filters low-confidence predictions before matching them to ground truth. Sweeping this threshold produces precision and recall values; average precision, or AP, summarises a specified interpolation of the resulting precision-recall curve for a class at a particular IoU matching criterion. Mean AP averages over classes, and some evaluation protocols also average over several IoU criteria. The source's shorthand “area under the precision-recall curve” is useful intuition, but a reproducible AP number needs the exact interpolation, score ordering, duplicate-handling and IoU policy. A model's AP cannot be compared fairly if those protocols differ.

## Code you can run

The first block calculates the lecture's box overlap with a reusable axis-aligned function. It clamps negative widths and heights to zero, so disjoint boxes have zero intersection. The exact result is **1/3** and rounds to **0.333**.

```python
def area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_iou(first, second):
    shared = (
        max(first[0], second[0]), max(first[1], second[1]),
        min(first[2], second[2]), min(first[3], second[3]),
    )
    intersection = area(shared)
    union = area(first) + area(second) - intersection
    return intersection, union, intersection / union if union else None

a = (0, 0, 10, 10)
b = (5, 0, 15, 10)
intersection, union, iou = box_iou(a, b)
print('Intersection:', intersection, 'Union:', union)
print(f'IoU: {iou:.3f}')
assert (intersection, union) == (50, 150)
assert round(iou, 3) == 0.333
```

The lab starts with box B at x=5 and an NMS threshold of 0.5, so it keeps both candidates. Move B or the threshold to see the decision change. A is assumed to have the higher score; the lab isolates pairwise suppression rather than simulating a full detector.

<BoxIouNmsLab />

The second block applies pairwise NMS to those two candidate boxes at thresholds 0.5 and 0.3. It prints **two kept** at 0.5 and **one kept** at 0.3 because one-third overlap exceeds 0.3. This uses a strict greater-than suppression rule, which is stated so an equality case is unambiguous.

```python
a = (0, 0, 10, 10)
b = (5, 0, 15, 10)

def pair_iou(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)

candidates = [(0.9, a), (0.7, b)]

def pairwise_nms(items, threshold):
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    kept = [ordered[0]]
    for candidate in ordered[1:]:
        overlap = pair_iou(kept[0][1], candidate[1])
        if overlap <= threshold:
            kept.append(candidate)
    return kept

print('Kept at 0.5:', len(pairwise_nms(candidates, 0.5)))
print('Kept at 0.3:', len(pairwise_nms(candidates, 0.3)))
assert len(pairwise_nms(candidates, 0.5)) == 2
assert len(pairwise_nms(candidates, 0.3)) == 1
```

This two-box helper is intentionally narrow. Full NMS checks every next candidate against every previously kept candidate, uses class policy and handles score ties. A detector may use a different suppression method or an end-to-end head without conventional NMS. The geometry result remains useful for understanding matching and duplicate decisions.

## Designing with it

Define the box coordinate convention. Some pipelines use pixel-centre inclusive endpoints, others use continuous coordinates and half-open area intervals. The code uses continuous `[x1,y1,x2,y2]` areas with width `x2-x1`. Mixing conventions can produce off-by-one IoU differences, especially for tiny boxes. Document image origin, coordinate order, clipping and resize mapping. Test synthetic boxes for identical, disjoint, touching and partially overlapping cases before using an evaluation library.

Measure the right unit of success. A package-counting system cares about missed and duplicate physical items, while a safety system may care more about recall for a critical class. mAP is useful for model comparison under a standard protocol but may not describe performance at the product's chosen operating threshold. Report class-specific precision and recall, object size, crowding and latency at the actual deployment settings. Inspect false positives caused by reflections and false negatives under occlusion.

Choose suppression for scene density. If same-class objects are often close together, conventional NMS may remove a real neighbour. If the threshold is too high, duplicate boxes remain. Score calibration, class-wise rules and soft suppression can change the trade-off. Evaluate duplicates and misses together rather than tuning to make example images look clean. A source image with two overlapping packages is more informative than an isolated object for this choice.

Account for the entire inference path. Image decode, resize, normalisation, device transfer, network execution, box decoding and postprocessing all contribute to latency. A paper's model-only timing on one accelerator cannot be used as the product's line-speed guarantee. Benchmark on the target hardware and camera resolution, with representative batch size, warm-up and sustained load. Record tail latency and missed deadlines, not only average throughput.

Version the model with preprocessing and labels. A detector trained for a particular set of classes and box annotation rules cannot be compared directly with one using different definitions. If the image is letterboxed, map boxes back with the correct padding and scale. A one-pixel coordinate shift can matter for small objects. Keep annotated regression examples through the serving pipeline, and verify that class IDs, scores and boxes retain the expected meaning.

Finally, review evaluation protocol carefully. AP depends on an IoU threshold and precision-recall integration convention; mAP may include one or several IoU thresholds. Scores must be sorted, each ground-truth object matched according to protocol and duplicates counted as false positives. A single high-IoU prediction cannot be reused for two objects. Document ignore regions and difficult examples. Without this, two valid-looking mAP numbers may describe different tasks.

## Where this stands in 2026

:::info Industry view

- Torchvision 0.29 FCOS and Mask R-CNN documentation, the original FCOS paper, and Ultralytics YOLO26 documentation were checked on 2026-10-02. No model performance, speed or price is repeated from them.
- The lecture treats anchor boxes and NMS as universal to YOLO/SSD-style one-stage detection. FCOS is explicitly anchor-box-free, and documented YOLO26 has an optional NMS-free head; the source wording is qualified in the note below.
- The local runnable blocks test box geometry and two-box suppression only. No trained detector or real conveyor was run.

:::

## Practice questions

<details>
<summary><strong>Q1.</strong> What does object detection output, and how does it differ from classification?</summary>

A class label and a bounding box for each object (what + where), for a variable number of objects; classification gives only one label for the whole image.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> Contrast R-CNN, Fast R-CNN and Faster R-CNN.</summary>

R-CNN: ~2000 proposals, one CNN each (slow). Fast R-CNN: one CNN pass + RoI pooling. Faster R-CNN: a learned Region Proposal Network with anchor boxes.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> What is non-maximum suppression?</summary>

NMS keeps the highest-scoring box and removes others whose IoU with it exceeds a threshold, eliminating duplicate detections.<br /><em>Sessions 13-14 · conceptual</em>

</details>

<details>
<summary><strong>Q4.</strong> Boxes A=[0,0,10,10] and B=[5,0,15,10]. Compute IoU.</summary>

Each area 100; overlap [5,0,10,10] = 5×10 = 50; union = 100+100−50 = 150; IoU = 50/150 = 0.333.<br /><em>Sessions 13-14 · numeric</em>

</details>

<details>
<summary><strong>Q5.</strong> What is mAP?</summary>

Mean Average Precision; the area under the precision–recall curve (AP) averaged over classes (and often IoU thresholds); the standard detection benchmark.<br /><em>Sessions 13-14 · conceptual</em>

</details>

## Further reading

- [Torchvision FCOS](https://docs.pytorch.org/vision/stable/models/fcos.html) and the [original FCOS paper](https://arxiv.org/abs/1904.01355) for an anchor-free one-stage family.
- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html) for a two-stage detector with masks.
- [Ultralytics YOLO26 guide](https://docs.ultralytics.com/models/yolo26) for documented NMS and end-to-end options.
- Built from the course lecture "cv-s13-object-detection" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source model-family and AP claims

The lecture says YOLO and SSD predict with anchor boxes and describes a universal real-time versus accuracy trade-off. Those statements refer to particular historical designs, not all current one-stage detectors. FCOS is an official anchor-free counterexample, and Ultralytics documents an optional NMS-free YOLO26 inference path. “Real-time” and an accuracy trade-off require a named model, input, dataset, device and deadline. The lecture also compresses AP to “area under the precision-recall curve”; actual AP values depend on a stated matching and interpolation protocol.

:::

## Reading detection failures

If a scene shows two overlapping same-class objects but only one final box, inspect candidates before suppression. The network might have predicted both correctly and NMS removed one; it might have predicted only one. Those failures call for different changes. Plot candidate scores and pairwise IoU, then test the selected threshold on crowded validation scenes. A threshold change can recover one object while increasing duplicates elsewhere, so measure both outcomes.

If one object has several boxes, check the same stages. Scores may be close and their overlap below the suppression threshold, as in the lecture's one-third example at 0.5. But two non-identical boxes can also belong to genuinely separate objects. Apply a class-specific and scene-aware evaluation policy if needed, and inspect whether box annotations are consistent. NMS is a geometric heuristic, not an object-identity oracle.

If small objects have poor AP, verify their size in the model input after resize. A 10-pixel item in a 4K source image may become only a few pixels after aggressive downsampling. No box head can localise detail that preprocessing erased. Evaluate by object size and capture distance, inspect the underlying image and consider acquisition or crop changes before changing the model family. Small boxes are also sensitive to coordinate convention: a one-pixel edge change alters their IoU proportionally more than it alters a large box's IoU.

If mAP is high but the product misses its latency target, separate decode, preprocessing, inference and postprocessing timings. A detector may satisfy the model-only timing but fail after image transfer or NMS. Conversely, an end-to-end path that skips NMS may alter duplicate behaviour, so evaluate accuracy and speed together. A production decision needs a measured latency distribution on the actual hardware at the required resolution and traffic rate. The chapter's toy code provides no such timing.

When two teams report different AP for the same predictions, compare their class mapping, IoU threshold, interpolation method, score sorting, ignore-region handling and empty-class policy. One might average AP over classes and IoU thresholds while the other uses a single 0.5 threshold. The word “mAP” is insufficient for reproducibility. Save raw scored detections and reference boxes so a scorer can be rerun under an explicitly versioned protocol.

Finally, relate boxes back to the product action. A box around a person might be enough to trigger a review, but a robotic gripper may need a pose or precise outline. A detector can be the first stage that proposes a region for a mask model, tracker or human. In that pipeline, candidate recall may matter more than final precision at the proposal stage. Evaluate the whole workflow, including whether downstream stages correct or amplify detection errors.

## Check yourself

- I can distinguish image classification, detection and instance segmentation outputs.
- I can calculate intersection, union and IoU for axis-aligned boxes under a stated coordinate convention.
- I can predict a pairwise NMS decision from scores, IoU and threshold.
- I can explain why an AP or “real-time” claim needs its evaluation and hardware protocol.
