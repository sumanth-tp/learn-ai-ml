---
id: cv-multiple-object-tracking
title: "Computer Vision · Session 14; Multiple Object Tracking"
sidebar_label: "4 · Multiple object tracking"
sidebar_position: 4
slug: /theory/cv/multiple-object-tracking
description: "Link detections across frames with prediction and association, compute a toy IoU gate and interpret identity errors."
tags: [computer-vision, tracking, association, mota]
---

import Infographic from '@site/src/components/Infographic';
import TrackingAssociationLab from '@site/src/components/viz/TrackingAssociationLab';

**In one line.** Tracking keeps an object identity stable across frames by predicting motion and matching new observations to existing tracks.

:::tip Before you start
**You should already know**

- Box IoU and what a detector outputs per frame: [object detection and box evaluation](/docs/theory/cv/object-detection-and-box-evaluation).
- The mean and variance of a number, because a Kalman filter tracks both.
- What it means to pair two lists one to one (an assignment).

**Reading time.** About 50 minutes, plus a few seconds to run the code.

**After this chapter you can**

- predict, update and gate one track by hand and solve a small assignment,
- explain why MOTA barely changes when identity quality changes a great deal,
- choose a track lifetime that bridges an occlusion without inventing identities.
:::

## In 30 seconds

A detector sees each video frame fresh, like a person with no memory who is shown photographs one at a time. Tracking gives it memory. For each object you already follow, you guess where it will be next, look at what the detector found, and pair guesses with findings so that the same label follows the same object. If two people walk past each other, the pairing is where labels swap by mistake. Think of a coat-check: the ticket number must stay with the coat even when the queue shuffles.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Track | One object's identity and its position over time | Person number 3 |
| Detection | A box a detector found in this frame | One box at (120, 80) |
| Kalman filter | A predictor that blends a motion guess with a noisy measurement | Predict 12, measure 13, answer 12.5 |
| Kalman gain | How much to trust the measurement over the guess | 0.5 when both are equally uncertain |
| Hungarian algorithm | Finds the pairing with the lowest total cost | Pairs two tracks with two detections |
| Gate | A rule that rejects pairs that are too far apart | IoU below 0.1 is not allowed |
| Identity switch (IDSW) | A track label changes from one object to another or a new label starts | Person 3 becomes person 7 |
| MOTA | One minus misses, false alarms and switches over true objects | 0.82 for 18 errors in 100 |
| IDF1 | How consistently each true object keeps one label | 0.907 means very consistent |

## The idea in plain words

:::note Beyond the course material

The association-cost caveat, track lifecycle, latency and metric interpretation extend the course material. Its tracking-by-detection outline, SORT and Deep SORT references, worked IoU and all five practice questions remain below.

:::

An object detector treats each frame as a new image and returns boxes with classes and scores. Multiple object tracking asks which detections across frames refer to the same physical objects. The output is a set of tracks, each with a stable ID and time-indexed state. A correct box in each frame is not enough: if two people cross and their IDs swap, downstream counts, trajectories and event histories can be wrong. Tracking therefore adds a temporal data-association problem to detection.

Tracking by detection works in a fixed order. First detect objects in the current frame. Predict each existing track's next position using its previous state. Then score possible track-detection pairs, reject implausible pairs and choose a set of associations. Update matched tracks, start new tracks for unmatched detections and decide how long unmatched old tracks remain alive. A Kalman filter is one linear probabilistic motion predictor; it represents an estimated state and uncertainty. The Hungarian algorithm can solve a minimum-cost one-to-one assignment over a matrix of feasible pair costs. It does not decide what the cost should be or whether a detection is a real object.

Intersection over union is a simple pairwise cue. If a predicted box and a detected box intersect over 30 square pixels and their union covers 70, their IoU is $30/70=0.429$. Under a gate of at least 0.3, this pair is eligible for association. That does not mean it will be chosen when several tracks compete for the same detection. The assignment must consider all feasible pairs and a policy for unmatched rows and columns. A motion model, camera movement and frame interval all affect whether IoU remains useful: a fast object can move so far between frames that its boxes do not overlap at all.

The original SORT paper combines a simple motion predictor with assignment, and the Deep SORT paper adds appearance information to help maintain identities through longer occlusions. The original papers were opened on 2026-10-02. Appearance can distinguish similarly placed objects but may fail when uniforms or vehicles look alike; motion can bridge a short gap but accumulates uncertainty. Neither algorithm name guarantees performance on every camera or crowding level. Meta's SAM 3 publication documents concept-prompted video tracking as a more recent model family, but a prompted mask tracker has a different input and output contract from the box-based SORT pipeline.

<Infographic src="/img/cv/tracking.svg" alt="Tracking board: predict a track box, compare a 30-pixel intersection and 70-pixel union to get IoU point four two nine, then evaluate misses, false positives and identity switches." caption="A pair can pass an overlap gate yet still lose a global assignment to a stronger match." />

## Worked example, step by step

Part one is a Kalman update in one dimension. Part two is an assignment where taking the best pair first goes wrong.

1. A track moves at 2 pixels per frame and was at 10, so the motion guess is 12. After predicting, its variance (uncertainty) is 4.
2. The detector reports 13, with a measurement variance of 4.
3. The Kalman gain is predicted variance over total variance: 4 / (4 + 4) = 0.5. Both sources are equally uncertain, so we trust them equally.
4. The updated position is 12 + 0.5 × (13 − 12) = 12.5. The updated variance is (1 − 0.5) × 4 = 2, so the estimate is more certain than either source.
5. Now two tracks and two detections with IoU 0.80 (track 0, detection 0), 0.70 (track 0, detection 1), 0.75 (track 1, detection 0) and 0.00 (track 1, detection 1).
6. A greedy matcher takes the highest IoU first: track 0 with detection 0. Track 1 is then left with detection 1 at IoU 0.00, which the gate rejects, so track 1 is unmatched. Total IoU is 0.80 for one pair.
7. The Hungarian algorithm considers all pairings. It picks track 0 with detection 1 (0.70) and track 1 with detection 0 (0.75), total 1.45, and both tracks are matched.

In words: the Kalman filter says "how far should I trust what I just saw?", and the Hungarian algorithm says "what pairing is best for everyone together, not just for the first pair?"

## How it works

### Predict & associate

Detect each frame → Kalman-predict next box → associate detections to tracks by IoU (Hungarian) → maintain IDs. SORT = Kalman+IoU; DeepSORT adds appearance.

:::tip

**Worked.** ∩=30, ∪=70 → IoU 0.429; above the 0.3 threshold → linked.

:::

### MOTA & MOTP

MOTA = 1 − (FN+FP+IDSW)/GT penalises misses, false positives and ID switches; MOTP measures localisation. Stable identities matter as much as detection.

### Key takeaways

## A real system that works this way

Consider counting people entering a room from a fixed camera. A detector may produce one box per person per frame, but counting every detection would count one person many times. A tracker creates an ID and counts it only when it crosses an entry line under a specified direction rule. This design needs enough frame continuity to maintain IDs. If a person is hidden behind another, the track may remain alive for a few frames based on prediction; if the gap is too long, a new ID may be created when they reappear, causing a double count. Review occlusion sequences and entry-line crossings, not just individual frames.

The SORT and Deep SORT papers are concrete systems for the predict-and-associate pattern. Their historical results were obtained under their own detector and evaluation protocols; this chapter does not transfer any performance number to the room scenario. A current promptable tracker such as the one described in Meta's SAM 3 research may produce masks and track concepts through video, but it requires the appropriate prompt and task validation. Comparing a box tracker and a prompted mask tracker solely by their names would hide different annotation and integration costs.

For the room, camera motion creates another challenge. A fixed Kalman motion model in pixel coordinates may interpret a camera pan as all people moving together. Stabilisation, camera-motion compensation or a different coordinate frame can help. If the system is used for safety or access decisions, it should also have a review path for uncertain identity and a clear retention policy for track data. The toy IoU computation below is only a gate demonstration.

## Code you can run

The first block reproduces the pairwise IoU and threshold decision. The areas are given directly, so the computation does not invent box coordinates for those counts. It prints **0.429** after rounding and marks the pair eligible at a 0.3 threshold.

```python
intersection = 30
union = 70
threshold = 0.3
iou = intersection / union
eligible = iou >= threshold
print(f'IoU: {iou:.3f}')
print('Eligible pair:', eligible)
assert round(iou, 3) == 0.429
assert eligible
```

The worked numbers in code, with SciPy's `linear_sum_assignment` as the Hungarian solver. It prints the gain, the updated mean and variance, and both assignment outcomes.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

predicted, variance, measurement, noise = 12.0, 4.0, 13.0, 4.0
gain = variance / (variance + noise)
print('Kalman gain:', gain)
print('updated position:', predicted + gain * (measurement - predicted))
print('updated variance:', (1 - gain) * variance)

iou = np.array([[0.80, 0.70], [0.75, 0.00]])
greedy_first = tuple(int(v) for v in np.unravel_index(iou.argmax(), iou.shape))
print('greedy takes track, detection', greedy_first, 'and leaves track 1 with IoU', iou[1, 1])
rows, cols = linear_sum_assignment(-iou)
print('Hungarian pairs', list(zip(rows.tolist(), cols.tolist())), 'total IoU', round(float(iou[rows, cols].sum()), 2))
```

**Reading the output.** The gain is 0.5, the position 12.5 and the variance 2.0, matching steps 3 and 4. Greedy matching leaves track 1 with IoU 0.0. Hungarian pairs (0, 1) and (1, 0) for a total of 1.45, matching step 7. If you see greedy and Hungarian agreeing on every case you test, your matrices are too easy.

The lab uses the same numbers. Move intersection, union or gate to see when the pair becomes ineligible. Its output deliberately says “eligible”, not “assigned”, because another track may have a better feasible match. The data table includes every input and the decision.

<TrackingAssociationLab />

**What each control does.** "Intersection area" and "Union area" set the overlap of a predicted and a detected box. "Association threshold" is the minimum IoU allowed for a pair to be eligible.

**Try it yourself.**

1. At the defaults (30, 70 and 0.30) the IoU is 0.429 and the pair is eligible.
2. Raise the union to 140 with the intersection at 30. The IoU falls to 0.214 and the pair becomes ineligible. A fast object, whose boxes overlap less, fails a gate that a slow one passes.
3. Return the union to 70 and raise the threshold to 0.45. The same 0.429 now fails. Strict gates reject true matches, and loose gates (the experiment used 0.1) leave more work to the assignment.

The second block checks the MOTA formula on a clearly synthetic count example: 100 ground-truth object observations, ten misses, five false positives and three identity switches yield **0.820**. The score can be negative if errors exceed the number of ground-truth observations. It cannot by itself tell whether errors are concentrated on one person or spread evenly.

```python
ground_truth_observations = 100
false_negatives = 10
false_positives = 5
identity_switches = 3
mota = 1 - (false_negatives + false_positives + identity_switches) / ground_truth_observations
print(f'MOTA: {mota:.3f}')
assert round(mota, 3) == 0.82
assert 1 - (80 + 30 + 10) / 100 < 0
```

This formula aggregates errors across an evaluated sequence. A complete scorer needs frame-by-frame ground truth, an association policy and definitions for misses, false alarms and switches. If there are zero ground-truth object observations, the denominator is zero and the metric needs an explicit reporting convention. The 0.820 value is a mathematical example, not a result for SORT, Deep SORT or SAM 3.

### Experiment: what identity errors cost, and what hides them

The question: how much do a motion model and a track lifetime matter, and does MOTA tell you? We simulate six boxes the size of pedestrians (24 by 48 pixels) crossing a 640 pixel frame in opposite directions over 100 frames, so three left-to-right and three right-to-left paths cross. A vertical pole at x between 300 and 340 hides every object for about 7 frames. The detector adds Gaussian box noise (1.5 pixels), drops each visible detection with probability 0 to 0.4, and adds about 0.3 clutter boxes per frame. We compare an IoU-only tracker (last box, no motion) with a Kalman tracker (constant-velocity state, from scratch), both with Hungarian assignment, and track lifetimes of 3 and 12 missed frames. Scoring uses motmetrics 1.4.0 (installed into this environment with pip, MIT licence). Means over 5 seeds.

```python
import warnings
import numpy as np
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
FRAMES, W, H = 100, 24.0, 48.0

def simulate(seed, dropout):
    rng = np.random.default_rng(seed)
    starts = np.array([[0.0, 90], [0, 150], [0, 210], [640, 120], [640, 180], [640, 240]])
    speeds = np.array([5.5, 6.5, 6.0, -6.0, -5.0, -6.5]) * rng.uniform(0.9, 1.1, 6)
    truth, detections = [], []
    for t in range(FRAMES):
        centres = starts + np.stack([speeds * t, np.zeros(6)], 1)
        truth.append(centres.copy())
        frame = []
        for c in centres:
            hidden = 300 <= c[0] <= 340 or not 0 <= c[0] <= 640
            if not hidden and rng.random() >= dropout:
                frame.append(np.array([c[0], c[1], W, H]) + rng.normal(0, [1.5, 1.5, 1.0, 1.0]))
        for _ in range(rng.poisson(0.3)):
            frame.append(np.array([rng.uniform(0, 640), rng.uniform(60, 280), W, H]))
        detections.append(np.array(frame).reshape(-1, 4))
    return truth, detections

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih)

class Track:
    count = 0
    def __init__(self, box, kalman):
        Track.count += 1
        self.id, self.misses, self.kalman = Track.count, 0, kalman
        self.x = np.array([*box, 0.0, 0.0])
        self.P = np.diag([4, 4, 4, 4, 100, 100.0])
    def predict(self):
        if self.kalman:
            F = np.eye(6); F[0, 4] = F[1, 5] = 1
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + np.diag([1, 1, 0.1, 0.1, 0.5, 0.5])
        return self.x[:4]
    def update(self, box):
        if self.kalman:
            Hm = np.eye(4, 6)
            S = Hm @ self.P @ Hm.T + np.diag([2.0, 2.0, 1.0, 1.0])
            K = self.P @ Hm.T @ np.linalg.inv(S)
            self.x = self.x + K @ (box - Hm @ self.x)
            self.P = (np.eye(6) - K @ Hm) @ self.P
        else:
            self.x[:4] = box
        self.misses = 0

def run(truth, detections, kalman, max_age):
    Track.count = 0
    tracks, acc = [], mm.MOTAccumulator(auto_id=True)
    for t in range(FRAMES):
        dets = detections[t]
        predicted = [tr.predict() for tr in tracks]
        cost = np.array([[1 - iou(p, d) for d in dets] for p in predicted]).reshape(len(tracks), len(dets))
        rows, cols = linear_sum_assignment(cost) if cost.size else ([], [])
        matched = set()
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.9:
                tracks[r].update(dets[c]); matched.add(r)
        for r in range(len(tracks)):
            if r not in matched:
                tracks[r].misses += 1
        used = {c for r, c in zip(rows, cols) if r in matched}
        tracks += [Track(dets[c], kalman) for c in range(len(dets)) if c not in used]
        tracks = [tr for tr in tracks if tr.misses <= max_age]
        shown = [(tr.id, tr.x[:4]) for tr in tracks if tr.misses == 0]
        gt_boxes = [np.array([c[0], c[1], W, H]) for c in truth[t]]
        d = np.array([[1 - iou(g, b) if iou(g, b) > 0.3 else np.nan for _, b in shown] for g in gt_boxes]).reshape(6, len(shown))
        acc.update(list(range(6)), [i for i, _ in shown], d)
    s = mm.metrics.create().compute(acc, metrics=['mota', 'num_switches', 'num_false_positives', 'num_misses', 'num_objects', 'idf1'], name='x').iloc[0]
    return s

print('tracker        dropout max_age  MOTA   IDF1   IDSW   FP    FN  (mean of 5 seeds)')
for kalman in (False, True):
    for dropout in (0.0, 0.2, 0.4):
        for max_age in (3, 12):
            r = [run(*simulate(s, dropout), kalman, max_age) for s in range(5)]
            m = lambda k: np.mean([x[k] for x in r])
            own = 1 - (m('num_misses') + m('num_false_positives') + m('num_switches')) / m('num_objects')
            assert abs(own - m('mota')) < 1e-9
            print(f'{"Kalman+IoU" if kalman else "IoU only":13s} {dropout:6.1f} {max_age:7d} {m("mota"):6.3f} {m("idf1"):6.3f} {m("num_switches"):5.1f} {m("num_false_positives"):5.1f} {m("num_misses"):5.1f}')
```

**Reading the output.** Each row averages five simulated sequences of 600 true object observations. `MOTA` is 1 minus (misses + false positives + switches) over observations. `IDF1` is the share of detections that keep the correct identity under the best global labelling. `IDSW`, `FP` and `FN` are counts per sequence.

**Line by line.**

- `simulate` makes the detector: objects at the pole or outside the frame are hidden, and each other object is dropped with probability `dropout`.
- `Track.predict` and `Track.update` are a standard constant-velocity Kalman filter on box centre, width and height. The `if self.kalman` branches make the IoU-only tracker the same code without prediction or smoothing.
- `linear_sum_assignment(cost)` solves the pairing; `cost[r, c] < 0.9` is the gate, that is IoU above 0.1.
- Tracks are shown only on frames where they were matched, and they are dropped after `max_age` consecutive misses.
- The assertion checks that my own MOTA formula equals motmetrics' value, so the library and the formula in this chapter agree.

The printed output was:

```text
tracker        dropout max_age  MOTA   IDF1   IDSW   FP    FN  (mean of 5 seeds)
IoU only         0.0       3  0.857  0.505   8.2  31.6  46.2
IoU only         0.0      12  0.854  0.485   9.8  31.6  46.2
IoU only         0.2       3  0.664  0.424  12.8  34.4 154.6
IoU only         0.2      12  0.660  0.403  15.2  34.4 154.6
IoU only         0.4       3  0.439  0.263  33.4  34.6 268.6
IoU only         0.4      12  0.435  0.244  35.6  34.6 268.6
Kalman+IoU       0.0       3  0.860  0.528   6.4  31.6  46.2
Kalman+IoU       0.0      12  0.869  0.907   0.8  31.6  46.2
Kalman+IoU       0.2       3  0.673  0.472   7.6  34.2 154.4
Kalman+IoU       0.2      12  0.684  0.811   1.2  34.0 154.2
Kalman+IoU       0.4       3  0.466  0.334  17.6  34.4 268.4
Kalman+IoU       0.4      12  0.490  0.661   3.0  34.4 268.4
```

**What the numbers say.** With no dropout, the Kalman tracker with a 12-frame lifetime made 0.8 identity switches per sequence, against 9.8 for the IoU-only tracker with the same lifetime. MOTA barely moved: 0.869 against 0.854. IDF1 told the real story: 0.907 against 0.485. MOTA is dominated by misses (46 of the roughly 79 errors here are the frames behind the pole, which no tracker can fix) and one switch counts as one error however long the wrong label then persists. A reader who looked only at MOTA would conclude the motion model was worth 1.5 points.

The second result is that a longer lifetime helps only when there is a motion model. For IoU-only tracking, raising the lifetime from 3 to 12 frames increased switches (8.2 to 9.8) and lowered IDF1 (0.505 to 0.485), because a stale box left behind at the pole is never close enough to the reappearing object. For the Kalman tracker, the same change cut switches from 6.4 to 0.8, because the track coasts through the pole on its predicted velocity and picks the object up on the other side.

Detector dropout is what MOTA mostly measures. Going from 0 to 0.4 dropout takes the best tracker from MOTA 0.869 to 0.490 and from IDF1 0.907 to 0.661, with misses rising from 46 to 268, while the switches rise only from 0.8 to 3.0. If your MOTA falls, check the detector before the tracker.

<Infographic src="/img/cv-enrich/v3-tracking-idf1.svg" alt="Bars compare MOTA and IDF1 for an IoU-only tracker and a Kalman tracker at no dropout with lifetime 12, with a table of identity switches by dropout and a card on lifetime." caption="Look first at the pair of bars for the two trackers: MOTA is almost equal, IDF1 is not." />

Limits: six objects, straight paths at constant speed (which flatters a constant-velocity filter), perfect box size, a single pole as the only occluder, no appearance cues, a hand-set Kalman noise, five seeds, no real video. A real crowd with turning people would reduce the Kalman advantage.

## Designing with it

Define the identity task. Tracking a person through a short doorway occlusion differs from tracking a vehicle across cameras. A single-camera track ID may be valid only within one video session. Cross-camera re-identification requires additional evidence and raises a different privacy and error profile. Decide when a track starts, how many detections confirm it, how long it survives without a match and when its ID is retired. These lifecycle rules directly affect double counts and false tracks.

Build pair costs from relevant evidence. IoU works when objects move smoothly relative to frame rate and box sizes. A Kalman prediction gives a position and uncertainty that can gate impossible jumps. Appearance embeddings can help when objects cross or briefly disappear, but similar clothing or lighting changes can confuse them. Combine cues with calibrated weights or a learned cost, and record the choices. The Hungarian algorithm only optimises the supplied matrix; a globally optimal assignment under a bad cost can be systematically wrong.

Handle competition and no-match cases. When two tracks compete for one detection, a greedy pairwise IoU decision can allocate it poorly. A one-to-one assignment uses all candidate costs at once, but still needs dummy assignments or gating for a track with no suitable detection. Never force every track to match something. An occluded track should be allowed to remain unmatched and uncertain, then either reconnect or expire under a documented policy.

Evaluate identity and detection separately. Misses and false positives may come from the detector, while ID switches can arise in association. MOTA combines them, so an improved detector can raise MOTA even if identity matching is unchanged. Use identity-sensitive measures or explicit switch counts alongside detection measures, and inspect crossing and occlusion clips. The number of correct per-frame boxes does not prove that trajectories are reliable.

Test timing and dropped frames. A prediction step uses elapsed time; treating a 200-millisecond gap as one normal frame can make the motion estimate implausible. Record timestamps and camera frame rate. Measure full-pipeline latency and backlog under sustained load, because a tracker running on stale detections may return precise but late trajectories. Real-time is a system requirement with a deadline, not a property implied by a paper's algorithm name.

Finally, plan monitoring and correction. Track counts, unmatched-detection rates, unmatched-track ages and unusually frequent switches. Sample video clips for human review where labels can be obtained. A change in camera angle or detector version can alter association even when the tracking code is unchanged. Version the detector, appearance encoder, cost rules, thresholds and lifecycle policy together. Keep representative regression clips with crossing, occlusion and entry-line events so changes are judged against the intended product behaviour.

## Where this stands in 2026

:::info Industry view

- The original SORT and Deep SORT papers and Meta’s SAM 3 publication were opened on 2026-10-02. Their scopes are described without carrying their historical benchmark values into this chapter.
- The 30/70 IoU rounds to 0.429 and passes the stated 0.3 gate. That is a pairwise eligibility calculation, not a guaranteed global assignment.
- The MOTA example was executed locally with synthetic counts. A simulated-video tracker was run in the experiment, but no real video or trained model weights were.

:::

## Common mistakes

- **Judging a tracker by MOTA alone.** It is the standard headline, so it feels sufficient. Here two trackers differed by 0.015 MOTA and 0.42 IDF1. Report identity switches and an identity-sensitive score as well.
- **Raising the track lifetime without a motion model.** Keeping lost tracks longer sounds like it must bridge gaps. For the IoU-only tracker it increased switches from 8.2 to 9.8. Pair a longer lifetime with prediction.
- **Matching greedily.** Taking the best pair first is simple and often fine. When two tracks compete for one detection it can leave a track unmatched while a better total exists. Use an assignment solver and a gate.
- **Assuming per-frame motion in pixels.** A velocity of 2 pixels per frame means different speeds at 30 and 5 frames per second. Give the motion model elapsed time.
- **Blaming the tracker for detector misses.** A drop in MOTA often comes from the detector. At 0.4 dropout misses rose to 268 while switches stayed at 3.0. Separate the counts before tuning.

## Practice questions

<details>
<summary><strong>Q1.</strong> What is tracking-by-detection?</summary>

Detect objects in each frame, then link detections across frames (predict + associate) to maintain identities.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q2.</strong> What roles do the Kalman filter and Hungarian algorithm play?</summary>

The Kalman filter predicts each track's next box; the Hungarian algorithm matches detections to tracks to maximise total IoU.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q3.</strong> Predicted and detected boxes intersect in 30 px with union 70 px. Compute IoU.</summary>

30/70 = 0.429 (above a 0.3 threshold → linked).<br /><em>Session 14 · numeric</em>

</details>

<details>
<summary><strong>Q4.</strong> How do SORT and DeepSORT differ?</summary>

SORT uses Kalman + IoU; DeepSORT adds an appearance embedding to survive occlusions and reduce ID switches.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q5.</strong> Write the MOTA formula.</summary>

MOTA = 1 − (FN + FP + IDSW)/GT; penalising misses, false positives and ID switches.<br /><em>Session 14 · conceptual</em>

</details>

<details>
<summary><strong>Q6 (Easy).</strong> A track is predicted at 12 with variance 4. The detector says 13 with variance 4. What is the updated position and variance?</summary>

The gain is 4 / 8 = 0.5. The position is 12 + 0.5 × 1 = 12.5 and the variance is 0.5 × 4 = 2.

</details>

<details>
<summary><strong>Q7 (Medium).</strong> In the experiment, the IoU-only tracker with lifetime 12 had MOTA 0.854 and IDF1 0.485, and the Kalman tracker had 0.869 and 0.907. Which tracker would you ship and why?</summary>

The Kalman tracker. The MOTA gap is small because MOTA counts each identity switch as a single error and is dominated by the 46 misses, but the IDF1 gap shows the IoU-only tracker keeps the right label for only about half of the detections. A person-counting or trajectory product depends on identity, so IDF1 and switches are the numbers that matter.

</details>

<details>
<summary><strong>Q8 (Stretch).</strong> The pole hides objects for about 7 frames. Why did a lifetime of 3 give more switches than 12 for the Kalman tracker, and what would you change if the occlusion lasted 40 frames?</summary>

With lifetime 3 the track expires before the object reappears, so a new track and a new label begin: a switch. A lifetime of 12 outlasts the 7-frame gap. For a 40-frame occlusion a constant-velocity prediction drifts, because uncertainty grows each frame, so you would need either appearance evidence to reconnect the identity, a wider gate that grows with the predicted uncertainty, or a re-identification step after the gap.

</details>

## Further reading

- [Original SORT paper](https://arxiv.org/abs/1602.00763) for a motion-and-assignment tracker.
- [Original Deep SORT paper](https://arxiv.org/abs/1703.07402) for appearance-assisted association.
- [Meta SAM 3 research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) for a current prompted segmentation and tracking family.
- [py-motmetrics 1.4.0](https://pypi.org/project/motmetrics/): the package used to score the experiment, installed locally on 2026-10-09 (version and MIT licence read from the installed package metadata; the PyPI page itself did not load).
- [SORT paper abstract](https://arxiv.org/abs/1602.00763), opened 2026-10-09: a Kalman filter and the Hungarian algorithm as the tracking components, with detection quality as a key factor.
- Built from the course lecture "cv-s14-object-tracking" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.stanford.edu/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source association claim

Course summaries say the Hungarian algorithm matches detections to tracks “to maximise total IoU”. That is true only when the cost matrix is built solely from negative IoU or an equivalent transformation. A tracker can combine motion, appearance and gating, in which case Hungarian optimises that combined cost. The 0.429 example establishes one eligible pair under a 0.3 IoU gate, not the final assignment.

:::

## Diagnose an identity switch

Suppose two people cross, and the tracker exchanges their IDs. Inspect the predicted boxes and candidate association matrix just before the crossing. If each track's predicted box overlaps the other person's detection more, IoU-only matching can favour the wrong swap. Appearance evidence may help, but similar uniforms can defeat it. A longer motion history, explicit occlusion handling or a different camera viewpoint may be more useful. Review the entire clip because a switch visible after separation can be caused by an earlier missed detection.

Another failure is a track that vanishes for one frame and returns with a new ID. Determine whether the detector missed the object, a confidence threshold filtered it, or the track lifecycle expired too quickly. Keeping unmatched tracks longer can bridge short gaps, but also risks linking a different object later. The choice depends on how long occlusion typically lasts and how costly a wrong link is. Use held-out sequences with annotated identities to tune the survival interval.

If a tracker reports stable IDs but count totals are wrong, inspect the event rule. A person who turns around near an entry line may cross it twice; a box centre can wobble across the line because of detection jitter. Direction, hysteresis and a minimum track age can prevent repeated counting, but those rules must be tested on actual entry behaviour. Tracking quality and event quality are related but distinct. A visually smooth track does not guarantee a correct business count.

The MOTA formula can make interpretation subtle. Ten misses, five false positives and three switches over 100 reference observations give 0.820, but two trackers with the same MOTA can distribute those errors differently. One may miss a critical person for many frames, while another has many brief false positives. MOTA also counts an identity switch as one error under its protocol, which may not reflect the product cost of a broken trajectory. Report the component counts, identity-sensitive scores and representative clips rather than a single scalar.

When camera frame rate changes, a constant per-frame motion assumption changes meaning. A one-pixel-per-frame velocity at 30 frames per second and at five frames per second corresponds to different physical speeds. Feed elapsed time into the motion model and retune uncertainty for dropped frames. If camera motion is significant, stabilise or explicitly model it so all tracks are not displaced together. A pairwise IoU gate that worked in a static short-gap sequence can reject every true match after a fast pan.

Finally, assess how the tracker interacts with the detector. A new detector version may shift boxes or scores even when its frame-level AP improves, altering IoU association and track lifecycle. Run the same annotated clips through both complete pipelines. Compare detection, identity and event-level outcomes, and inspect failures that move from one category to another. The tracker consumes detections as measurements, so improvements at one stage do not automatically improve the whole system.

## Check yourself

- I can distinguish detection from stable identity across frames.
- I can compute the 30/70 association cue and say why it is only pairwise eligibility.
- I can explain what a Kalman predictor and an assignment solver each contribute.
- I can compute MOTA from error counts and list what it does not show.

- I can compute a one-dimensional Kalman update and a small Hungarian assignment by hand and check them in code.
- I can explain why MOTA stayed almost flat while IDF1 and the switch count changed a great deal, and what to report instead.
- I can say why a longer track lifetime needs a motion model, and what happens when an occlusion outlasts the lifetime.

## Where to go next

Next: [vision on edge devices](/docs/theory/cv/vision-on-edge-devices), which asks what it costs to run these models on small hardware. Related: [object detection and box evaluation](/docs/theory/cv/object-detection-and-box-evaluation), whose misses and false alarms feed the tracker.
