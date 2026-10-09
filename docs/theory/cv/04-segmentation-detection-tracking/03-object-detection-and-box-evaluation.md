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

:::tip Before you start
**You should already know**

- Intersection over union for masks, which is the same idea in two dimensions: [semantic masks and metrics](/docs/theory/cv/semantic-segmentation-and-mask-metrics).
- What a convolutional network outputs for one image: [what a convolutional neural network is](/docs/theory/dnn/what-a-convolutional-neural-network-is).
- Precision (how many alarms were real) and recall (how many real things were found).

**Reading time.** About 50 minutes, plus about 30 seconds to run the experiment once the model and data are cached.

**After this chapter you can**

- compute box IoU, run non-maximum suppression and average precision (AP) by hand,
- explain why a detector's raw output needs suppression before it is scored,
- say what changes AP and what barely does, using a pretrained detector measured on real photographs.
:::

## In 30 seconds

A detector draws boxes around things and gives each box a confidence. It draws many overlapping boxes around the same pedestrian, so you keep the most confident one and delete its near-duplicates. That deletion is non-maximum suppression. To grade the survivors, sort them by confidence, mark each one as a hit if it overlaps an unclaimed true box enough, and see how precision falls as you accept more of them. Average precision is the area under that curve. Think of a lifeguard shouting warnings: you want the loudest warnings to be the real ones.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Bounding box | A rectangle given by its corners | (0, 0, 10, 10) |
| Box IoU | Overlap area divided by union area of two boxes | 50 / 150 = 0.333 |
| Score | The detector's confidence for a box | 0.9 |
| NMS | Keep the best box, drop others overlapping it above a threshold | Two boxes at IoU 0.818 become one |
| True positive | A detection that matches an unclaimed true box | A box around a pedestrian |
| False positive | A detection that matches nothing, or a duplicate | A second box on the same person |
| Precision and recall | Real share of alarms, and share of real objects found | 0.6 and 1.0 |
| AP | Area under the precision-recall curve for one class | 0.7556 in the worked example |

## The idea in plain words

:::note Beyond the course material

The coordinate convention, current-family check, score-threshold and AP explanation and deployment design extend the course material. Its R-CNN sequence, box example and all five practice questions remain below.

:::

Image classification returns a class for the frame. Object detection returns a class, score and bounding box for each candidate object, so it can say both what and where. A detector must also handle a variable number of objects, small targets, occlusion and duplicate predictions. A box is a coarse localisation: it encloses the object but does not mark its pixel boundary. Instance segmentation adds a mask for each object when that boundary matters. Define which output the product needs before choosing a detector family.

The historical R-CNN family explains how computation was shared. Original R-CNN classified region proposals with separate feature extraction; Fast R-CNN shared an image feature map across proposals; Faster R-CNN learned proposals with a region proposal network. Two-stage refers to a proposal step followed by prediction refinement. One-stage detectors predict candidates over image features without a separate proposal network. This distinction says little by itself about actual latency or accuracy. Backbone, resolution, implementation, device and postprocessing all matter.

Course summaries often name YOLO and SSD and describe both as grid-based and anchor-based. That captures some historical variants, but anchor boxes are not a universal requirement for one-stage detection. FCOS is an anchor-box-free one-stage detector according to its original paper and current Torchvision catalogue. Current Ultralytics YOLO26 documentation also describes an optional one-to-one head that can emit predictions without a separate non-maximum suppression pass; its default one-to-many path uses NMS. These examples show why a present-day design review should name a specific variant and configuration instead of treating “YOLO” as one fixed algorithm. This chapter does not rank those families.

For two axis-aligned boxes, intersection over union is intersection area divided by union area. Let A cover coordinates `[0,0,10,10]` and B `[5,0,15,10]`, using continuous coordinates with right and bottom boundaries excluded for area arithmetic. Each area is 100; the shared rectangle has width 5 and height 10, so its area is 50. Union area is 150 and IoU is one third. Non-maximum suppression, or NMS, sorts candidate boxes by score and removes lower-scoring boxes with overlap above a chosen threshold, usually within a class. At threshold 0.5, one-third overlap is below the threshold, so both boxes remain. If the candidates actually refer to the same physical object, this threshold may be too permissive; NMS cannot know identity from IoU alone.

<Infographic src="/img/cv/detection.svg" alt="Object detection board showing two boxes of area 100 with overlap 50, union 150 and IoU one third; NMS at threshold one half keeps both." caption="Box IoU is a geometric ratio; a detection decision also needs scores, class and matching policy." />

## Worked example, step by step

Part one is suppression. Three boxes with scores: A = [0, 0, 10, 10] at 0.9, B = [1, 0, 11, 10] at 0.8, C = [5, 0, 15, 10] at 0.7. Part two is average precision for five detections against three true objects.

1. Sort by score: A, B, C. Keep A.
2. B overlaps A in a 9 by 10 rectangle: intersection 90, union 100 + 100 − 90 = 110, IoU 0.818. That is above 0.5, so B is suppressed.
3. C overlaps A in a 5 by 10 rectangle: intersection 50, union 150, IoU 0.333. That is at most 0.5, so C is kept. The survivors are A and C.
4. Now score five detections in order of confidence: hit, miss, hit, miss, hit. There are 3 true objects in total.
5. After each detection, precision is hits so far over detections so far: 1/1, 1/2, 2/3, 2/4, 3/5, which is 1.0, 0.5, 0.667, 0.5, 0.6. Recall is hits over 3: 0.333, 0.333, 0.667, 0.667, 1.0.
6. Replace each precision by the highest precision at that recall or beyond (the envelope): 1.0, 0.667, 0.667, 0.6, 0.6.
7. Sum the recall gains times the envelope: 0.333 × 1.0 + 0.333 × 0.667 + 0.333 × 0.6 = 0.333 + 0.222 + 0.2 = 0.7556.

In words: AP rewards a detector for putting real objects before false alarms, and every extra false alarm that outranks a real object pulls the area down.

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

The current Torchvision documentation lists FCOS and Mask R-CNN model builders, while the official Ultralytics YOLO26 guide describes detection, segmentation and optional NMS-free inference heads. These pages and the original FCOS paper were opened on 2026-10-02. They establish that current implementations span anchor-free, two-stage and different postprocessing choices. They do not establish which one is best for a local dataset, licence requirement, device or latency budget. The first code blocks check geometry only; the experiment later runs one small pretrained detector on 60 street photographs, and no other detector family was run.

Imagine a camera monitoring boxes on a conveyor. The system needs one location per physical package. A high-confidence candidate may overlap a second lower-confidence candidate for the same package, and NMS can remove the duplicate. But two packages can overlap in the image, so an aggressive NMS threshold might remove a real second package. A fixed threshold should be chosen from labelled examples that include crowded scenes, unusual box sizes and partial occlusion. The product should measure missed packages, duplicates and processing time after the entire capture and postprocessing path.

A detector score threshold filters low-confidence predictions before matching them to ground truth. Sweeping this threshold produces precision and recall values; average precision, or AP, summarises a specified interpolation of the resulting precision-recall curve for a class at a particular IoU matching criterion. Mean AP averages over classes, and some evaluation protocols also average over several IoU criteria. The shorthand “area under the precision-recall curve” is useful intuition, but a reproducible AP number needs the exact interpolation, score ordering, duplicate-handling and IoU policy. A model's AP cannot be compared fairly if those protocols differ.

## Code you can run

The first block calculates the worked box overlap with a reusable axis-aligned function. It clamps negative widths and heights to zero, so disjoint boxes have zero intersection. The exact result is **1/3** and rounds to **0.333**.

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

**What each control does.** "Box B horizontal start" slides box B across box A (A covers 0 to 10, B is 10 wide). "NMS threshold" is the overlap above which the lower-scoring box is suppressed.

**Try it yourself.**

1. At the defaults (start 5, threshold 0.50) the IoU is 0.333 and both boxes are kept.
2. Move the threshold to 0.30. The IoU of 0.333 is now above the threshold, so box B is suppressed. This is the second block's result.
3. Return the threshold to 0.50 and set the start to 1. The overlap is 9 by 10, so the IoU is 90 / 110 = 0.818 and B is suppressed, as in step 2 of the worked example. At start 10 the boxes only touch, the IoU is 0 and both are kept at any threshold.

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

The worked numbers in code. `torchvision.ops.box_iou` and `torchvision.ops.nms` are the library versions, and scikit-learn's `average_precision_score` is an independent check of the AP.

```python
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou, nms

boxes = torch.tensor([[0, 0, 10, 10], [1, 0, 11, 10], [5, 0, 15, 10]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7])
print('IoU A,B:', round(float(box_iou(boxes[:1], boxes[1:2])), 3), ' IoU A,C:', round(float(box_iou(boxes[:1], boxes[2:3])), 3))
print('kept at 0.5:', nms(boxes, scores, 0.5).tolist())

flags = np.array([1, 0, 1, 0, 1])
confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
tp, fp = np.cumsum(flags), np.cumsum(1 - flags)
precision, recall = tp / (tp + fp), tp / 3
envelope = np.maximum.accumulate(precision[::-1])[::-1]
steps = np.diff(np.concatenate([[0], recall]))
print('precision', np.round(precision, 3), 'recall', np.round(recall, 3))
print('AP with envelope:', round(float((steps * envelope).sum()), 4))
print('AP from scikit-learn:', round(float(average_precision_score(flags, confidence)), 4))
```

**Reading the output.** The IoUs are 0.818 and 0.333 and NMS keeps boxes 0 and 2, matching steps 2 and 3. The AP is 0.7556 from both the envelope calculation and scikit-learn, matching step 7. They agree here because the envelope does not change any step; on other curves they differ slightly, as the experiment below shows.

This two-box helper is intentionally narrow. Full NMS checks every next candidate against every previously kept candidate, uses class policy and handles score ties. A detector may use a different suppression method or an end-to-end head without conventional NMS. The geometry result remains useful for understanding matching and duplicate decisions.

### Experiment: what changes AP on real photographs

The question: with a real pretrained detector, how much does suppression matter, how much does the IoU matching rule matter, and are my own IoU, NMS and AP functions right? First a cross-check against the library on random boxes, then an evaluation on real images.

Model and data, with licences. The detector is torchvision's `ssdlite320_mobilenet_v3_large` with the `COCO_V1` weights: 3,440,060 parameters, a 13.4 MB file, and a box mAP of 21.3 on COCO val2017 according to the weights' own metadata (torchvision 0.29.1). The torchvision documentation, opened 2026-10-09, says its pretrained models may carry their own licences or terms derived from the training dataset and that you must decide whether your use is permitted, so read those terms before shipping anything. The images are the first 60 of the Penn-Fudan pedestrian database (about 54 MB, downloaded by the code and cached). Its readme says copyright stays with the authors and the material may not be reposted without permission, so the code downloads it and nothing here redistributes it. Ground-truth boxes are the tight boxes of each pedestrian's mask.

Block one compares my IoU matrix and NMS with the library's on 400 random boxes. It needs no download.

```python
import numpy as np
import torch
from torchvision.ops import box_iou, nms

def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)

def nms_scratch(boxes, scores, threshold):
    order = np.argsort(-scores, kind='stable')
    kept = []
    while len(order):
        best = order[0]
        kept.append(best)
        if len(order) == 1:
            break
        overlap = iou_matrix(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return np.array(kept)

rng = np.random.default_rng(0)
xy = rng.uniform(0, 100, (400, 2))
wh = rng.uniform(5, 60, (400, 2))
boxes = np.hstack([xy, xy + wh]).astype(np.float32)
scores = rng.uniform(0, 1, 400).astype(np.float32)
mine = iou_matrix(boxes, boxes)
theirs = box_iou(torch.from_numpy(boxes), torch.from_numpy(boxes)).numpy()
print('max abs IoU difference:', float(np.abs(mine - theirs).max()))
for t in (0.3, 0.5, 0.7):
    a = nms_scratch(boxes, scores, t)
    b = nms(torch.from_numpy(boxes), torch.from_numpy(scores), t).numpy()
    print(f'NMS {t}: scratch keeps {len(a)}, torchvision keeps {len(b)}, same set {set(a) == set(b)}')
```

**Reading the output.** The largest difference between the two IoU matrices is 0.0, and for thresholds 0.3, 0.5 and 0.7 the kept sets are the same indices (128, 250 and 371 of 400 boxes). This is the check that makes the experiment below trustworthy.

Block two runs the detector with its own suppression turned off (`nms_thresh=1.0`, score threshold 0.02), keeps the person class, and evaluates with my NMS at several thresholds. Average precision is computed twice: with the envelope rule, and with scikit-learn's step version.

```python
import os
import tempfile
import time
import urllib.request
import zipfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor

torch.set_num_threads(4)
URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
CACHE = os.path.join(tempfile.gettempdir(), 'PennFudanPed.zip')
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(URL, CACHE)
archive = zipfile.ZipFile(CACHE)
names = sorted(n for n in archive.namelist() if n.startswith('PennFudanPed/PNGImages/') and n.endswith('.png'))[:60]

def load(name):
    image = Image.open(archive.open(name)).convert('RGB')
    mask = np.array(Image.open(archive.open(name.replace('PNGImages', 'PedMasks').replace('.png', '_mask.png'))))
    boxes = []
    for pid in np.unique(mask)[1:]:
        ys, xs = np.nonzero(mask == pid)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return image, np.array(boxes, dtype=np.float32)

def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)

def nms_scratch(boxes, scores, threshold):
    order, kept = np.argsort(-scores, kind='stable'), []
    while len(order):
        best = order[0]; kept.append(best)
        overlap = iou_matrix(boxes[best][None], boxes[order[1:]])[0] if len(order) > 1 else np.array([])
        order = order[1:][overlap <= threshold]
    return np.array(kept, dtype=int)

model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1, score_thresh=0.02, nms_thresh=1.0, detections_per_img=300).eval()
raw, truth = [], []
start = time.perf_counter()
with torch.no_grad():
    for name in names:
        image, gt = load(name)
        out = model([to_tensor(image)])[0]
        keep = out['labels'] == 1
        raw.append((out['boxes'][keep].numpy(), out['scores'][keep].numpy()))
        truth.append(gt)
print(f'{len(names)} images, {sum(len(t) for t in truth)} pedestrians, {1000 * (time.perf_counter() - start) / len(names):.0f} ms per image')

def evaluate(nms_threshold, match_iou):
    flags, scores, missed = [], [], 0
    for (boxes, conf), gt in zip(raw, truth):
        if nms_threshold < 1:
            k = nms_scratch(boxes, conf, nms_threshold)
            boxes, conf = boxes[k], conf[k]
        order = np.argsort(-conf, kind='stable')
        taken = np.zeros(len(gt), bool)
        overlaps = iou_matrix(boxes, gt) if len(boxes) else np.zeros((0, len(gt)))
        for i in order:
            j = overlaps[i].argmax() if len(gt) else -1
            hit = j >= 0 and overlaps[i, j] >= match_iou and not taken[j]
            if hit:
                taken[j] = True
            flags.append(int(hit)); scores.append(conf[i])
        missed += int((~taken).sum())
    flags, scores = np.array(flags), np.array(scores)
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(flags[order]); fp = np.cumsum(1 - flags[order])
    recall = tp / (tp[-1] + missed); precision = tp / (tp + fp)
    mrec = np.concatenate([[0], recall, [1]]); mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(((mrec[1:] - mrec[:-1]) * mpre[1:]).sum())
    library = average_precision_score(np.concatenate([flags, np.ones(missed)]), np.concatenate([scores, np.full(missed, -1.0)]))
    return ap, float(library), int(flags.sum()), int(len(flags) - flags.sum()), missed

print('NMS thr  IoU  AP scratch  AP sklearn  TP   FP   missed')
for nms_t in (0.3, 0.5, 0.7, 1.0):
    for match in (0.5, 0.75):
        ap, lib, tp, fp, missed = evaluate(nms_t, match)
        label = 'none' if nms_t == 1.0 else f'{nms_t:.1f}'
        print(f'{label:>7s}  {match:.2f}  {ap:10.3f}  {lib:10.3f}  {tp:4d} {fp:4d} {missed:4d}')
```

**Reading the output.** Each row is one setting. `TP` is hits, `FP` is detections that matched nothing or duplicated a match, and `missed` is true pedestrians that no detection claimed. `AP scratch` uses the envelope rule and `AP sklearn` uses the step rule.

**Line by line.**

- `nms_thresh=1.0` stops the model suppressing its own output. A box is only dropped when its IoU is above 1.0, which never happens, so we see the raw candidates.
- `evaluate` walks detections from highest to lowest confidence, matches each to its best-overlapping true box, and counts it a hit only if that overlap reaches `match_iou` and the true box is still unclaimed. Anything else is a false positive, which is how duplicates are punished.
- The `library` line appends the missed pedestrians as positives with the lowest possible score, so scikit-learn sees the full set of 129 true objects.
- `mpre[i] = max(mpre[i], mpre[i + 1])` builds the envelope from right to left.

The printed output was:

```text
60 images, 129 pedestrians, 100 ms per image
NMS thr  IoU  AP scratch  AP sklearn  TP   FP   missed
    0.3  0.50       0.920       0.923   125  946    4
    0.3  0.75       0.859       0.870   117  954   12
    0.5  0.50       0.919       0.920   126 1803    3
    0.5  0.75       0.857       0.862   117 1812   12
    0.7  0.50       0.915       0.915   127 3203    2
    0.7  0.75       0.862       0.865   119 3211   10
   none  0.50       0.428       0.424   127 7338    2
   none  0.75       0.419       0.416   123 7342    6
```

**What the numbers say.** Suppression is not optional. Without it, AP at IoU 0.5 is 0.428, and with NMS at 0.5 it is 0.919. The same detector, scoring the same 129 pedestrians, loses more than half its AP to duplicates, because the 7,338 false positives bury the real ones in the ranking. All hits are nearly unchanged (127 against 126), so recall is not the problem, precision is.

The surprise is how little the threshold matters once suppression is on: 0.3, 0.5 and 0.7 give AP 0.920, 0.919 and 0.915 at IoU 0.5, though the number of false positives more than triples from 946 to 3,203. Their cost is hidden because those extra false positives all have low scores, below most real detections, so they barely dent the area. A lower threshold also risks deleting a real neighbour in a crowd: at 0.3, 4 pedestrians were missed against 2 at 0.7. These 60 images hold about two pedestrians each (129 in total), so they are a weak test of crowds.

Raising the matching IoU from 0.5 to 0.75 costs about 0.06 AP (0.919 to 0.857 at NMS 0.5): the boxes find the pedestrians but are not tight enough. The two AP calculations agree within 0.011 in every row, and the step version is a little higher in most rows, equal at NMS 0.7 and IoU 0.5, and lower when there is no suppression, which is why a stated interpolation rule matters when two teams compare numbers.

The detector ran at between 57 and 100 ms per image across three runs on 4 CPU threads (an Apple M3 Pro; the first run included warm-up and file caching, and the printed 100 ms is that run). That includes loading the image and its mask from the archive and the model's own resizing, so the model alone is faster.

<Infographic src="/img/cv-enrich/v3-detection-ap.svg" alt="Bars show average precision at IoU 0.5 for a pretrained detector with no suppression and with NMS at 0.3, 0.5 and 0.7, with cards on false positive counts and on the IoU 0.75 penalty." caption="Look first at the red bar: without suppression AP falls to 0.428." />

Limits: one class, one small dataset of street scenes, 60 images, one detector, one run, no tuning of the score threshold. The detector was trained on COCO, which contains pedestrians, so this is an easy case; the AP of 0.92 is not a claim about other classes or the 21.3 reported on COCO.

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
- Course summaries often treat anchor boxes and NMS as universal to YOLO/SSD-style one-stage detection. FCOS is explicitly anchor-box-free, and documented YOLO26 has an optional NMS-free head; the source wording is qualified in the note below.
- The first runnable blocks test box geometry and two-box suppression. The experiment runs one pretrained SSDlite detector on 60 pedestrian photographs; no conveyor, YOLO26 or FCOS model was run.

:::

## Common mistakes

- **Scoring raw detector output.** It looks like a list of predictions. Without suppression the same detector fell from AP 0.919 to 0.428 on the same images. Always apply the suppression the detector was designed with before scoring.
- **Choosing the NMS threshold from a clean picture.** A lower threshold removes more duplicates, which looks tidier. In crowds it deletes real neighbours (4 missed at 0.3 against 2 at 0.7). Tune it on crowded validation images and count both duplicates and misses.
- **Comparing AP numbers without the protocol.** "AP 0.92" looks like a fact. The matching IoU (0.5 or 0.75), the interpolation rule and the score cut-off all change it. Report all three, and save raw detections so the score can be recomputed.
- **Letting a hit be reused.** Matching each detection to the best box without marking it claimed makes duplicates look like hits. Each true box can be claimed once.
- **Quoting a detector's published mAP for your data.** The 21.3 on COCO val2017 does not carry over to a different camera, class or object size. Measure on your own labelled images.

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

<details>
<summary><strong>Q6 (Easy).</strong> Boxes A = [0, 0, 10, 10] at 0.9 and B = [1, 0, 11, 10] at 0.8. At an NMS threshold of 0.5, which survive?</summary>

IoU is 90 / 110 = 0.818, which is above 0.5, so B is suppressed and only A survives.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> Five detections in confidence order are hit, miss, hit, miss, hit, with 3 true objects. Compute AP with the envelope rule.</summary>

Precisions are 1, 0.5, 0.667, 0.5, 0.6 and recalls 0.333, 0.333, 0.667, 0.667, 1. The envelope is 1, 0.667, 0.667, 0.6, 0.6. AP is 0.333 × 1 + 0.333 × 0.667 + 0.333 × 0.6 = 0.7556.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> In the experiment, NMS thresholds 0.3 and 0.7 changed false positives from 946 to 3,203 but AP only from 0.920 to 0.915. Explain, and say when the gap would widen.</summary>

AP depends on where false positives rank, not how many there are. These extra false positives had low scores, so they sit after most real detections and add little to the area. The gap would widen in a crowd, where a high-confidence duplicate or a suppressed real neighbour changes the early part of the curve, or if the score threshold were raised and the low-score tail removed anyway.

</details>

## Further reading

- [Torchvision FCOS](https://docs.pytorch.org/vision/stable/models/fcos.html) and the [original FCOS paper](https://arxiv.org/abs/1904.01355) for an anchor-free one-stage family.
- [Torchvision Mask R-CNN](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html) for a two-stage detector with masks.
- [Ultralytics YOLO26 guide](https://docs.ultralytics.com/models/yolo26) for documented NMS and end-to-end options.
- [Torchvision SSDlite documentation](https://docs.pytorch.org/vision/stable/models/ssdlite.html), opened 2026-10-09, and the weights metadata of torchvision 0.29.1 (parameters, file size, COCO val2017 box mAP 21.3, and the note that pretrained weights may have their own licences).
- [Penn-Fudan pedestrian database](https://www.cis.upenn.edu/~jshi/ped_html/), the readme inside the downloaded archive, read 2026-10-09 (copyright retained by the authors).
- Built from the course lecture "cv-s13-object-detection" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of source model-family and AP claims

Course summaries say YOLO and SSD predict with anchor boxes and describe a universal real-time versus accuracy trade-off. Those statements refer to particular historical designs, not all current one-stage detectors. FCOS is an official anchor-free counterexample, and Ultralytics documents an optional NMS-free YOLO26 inference path. “Real-time” and an accuracy trade-off require a named model, input, dataset, device and deadline. Course summaries also compress AP to “area under the precision-recall curve”; actual AP values depend on a stated matching and interpolation protocol.

:::

## Reading detection failures

If a scene shows two overlapping same-class objects but only one final box, inspect candidates before suppression. The network might have predicted both correctly and NMS removed one; it might have predicted only one. Those failures call for different changes. Plot candidate scores and pairwise IoU, then test the selected threshold on crowded validation scenes. A threshold change can recover one object while increasing duplicates elsewhere, so measure both outcomes.

If one object has several boxes, check the same stages. Scores may be close and their overlap below the suppression threshold, as in the one-third example at 0.5. But two non-identical boxes can also belong to genuinely separate objects. Apply a class-specific and scene-aware evaluation policy if needed, and inspect whether box annotations are consistent. NMS is a geometric heuristic, not an object-identity oracle.

If small objects have poor AP, verify their size in the model input after resize. A 10-pixel item in a 4K source image may become only a few pixels after aggressive downsampling. No box head can localise detail that preprocessing erased. Evaluate by object size and capture distance, inspect the underlying image and consider acquisition or crop changes before changing the model family. Small boxes are also sensitive to coordinate convention: a one-pixel edge change alters their IoU proportionally more than it alters a large box's IoU.

If mAP is high but the product misses its latency target, separate decode, preprocessing, inference and postprocessing timings. A detector may satisfy the model-only timing but fail after image transfer or NMS. Conversely, an end-to-end path that skips NMS may alter duplicate behaviour, so evaluate accuracy and speed together. A production decision needs a measured latency distribution on the actual hardware at the required resolution and traffic rate. The chapter's toy code provides no such timing.

When two teams report different AP for the same predictions, compare their class mapping, IoU threshold, interpolation method, score sorting, ignore-region handling and empty-class policy. One might average AP over classes and IoU thresholds while the other uses a single 0.5 threshold. The word “mAP” is insufficient for reproducibility. Save raw scored detections and reference boxes so a scorer can be rerun under an explicitly versioned protocol.

Finally, relate boxes back to the product action. A box around a person might be enough to trigger a review, but a robotic gripper may need a pose or precise outline. A detector can be the first stage that proposes a region for a mask model, tracker or human. In that pipeline, candidate recall may matter more than final precision at the proposal stage. Evaluate the whole workflow, including whether downstream stages correct or amplify detection errors.

## Check yourself

- I can distinguish image classification, detection and instance segmentation outputs.
- I can calculate intersection, union and IoU for axis-aligned boxes under a stated coordinate convention.
- I can predict a pairwise NMS decision from scores, IoU and threshold.
- I can explain why an AP or “real-time” claim needs its evaluation and hardware protocol.

- I can run non-maximum suppression and compute average precision by hand and check both against a library.
- I can explain why unsuppressed detector output scores less than half the AP of the suppressed output, and why the NMS threshold changes false positives far more than AP.
- I can say which three protocol choices must accompany an AP number.

## Where to go next

Next: [multiple object tracking](/docs/theory/cv/multiple-object-tracking), which links the boxes found in each frame into identities. Related: [vision on edge devices](/docs/theory/cv/vision-on-edge-devices), for what it costs to run a detector like this one on a small machine.
