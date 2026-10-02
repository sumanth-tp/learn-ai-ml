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

## The idea in plain words

:::note Beyond the lecture

The association-cost caveat, track lifecycle, latency and metric interpretation extend the lecture. Its tracking-by-detection outline, SORT and Deep SORT references, worked IoU and all five practice questions remain below.

:::

An object detector treats each frame as a new image and returns boxes with classes and scores. Multiple object tracking asks which detections across frames refer to the same physical objects. The output is a set of tracks, each with a stable ID and time-indexed state. A correct box in each frame is not enough: if two people cross and their IDs swap, downstream counts, trajectories and event histories can be wrong. Tracking therefore adds a temporal data-association problem to detection.

The lecture presents tracking by detection. First detect objects in the current frame. Predict each existing track's next position using its previous state. Then score possible track-detection pairs, reject implausible pairs and choose a set of associations. Update matched tracks, start new tracks for unmatched detections and decide how long unmatched old tracks remain alive. A Kalman filter is one linear probabilistic motion predictor; it represents an estimated state and uncertainty. The Hungarian algorithm can solve a minimum-cost one-to-one assignment over a matrix of feasible pair costs. It does not decide what the cost should be or whether a detection is a real object.

Intersection over union is a simple pairwise cue. If a predicted box and a detected box intersect over 30 square pixels and their union covers 70, their IoU is $30/70=0.429$. Under a gate of at least 0.3, this pair is eligible for association. That does not mean it will be chosen when several tracks compete for the same detection. The assignment must consider all feasible pairs and a policy for unmatched rows and columns. A motion model, camera movement and frame interval all affect whether IoU remains useful: a fast object can move so far between frames that its boxes do not overlap at all.

The original SORT paper combines a simple motion predictor with assignment, and the Deep SORT paper adds appearance information to help maintain identities through longer occlusions. The original papers were opened on 2026-10-02. Appearance can distinguish similarly placed objects but may fail when uniforms or vehicles look alike; motion can bridge a short gap but accumulates uncertainty. Neither algorithm name guarantees performance on every camera or crowding level. Meta's SAM 3 publication documents concept-prompted video tracking as a more recent model family, but a prompted mask tracker has a different input and output contract from the lecture's box-based SORT pipeline.

<Infographic src="/img/cv/tracking.svg" alt="Tracking board: predict a track box, compare a 30-pixel intersection and 70-pixel union to get IoU point four two nine, then evaluate misses, false positives and identity switches." caption="A pair can pass an overlap gate yet still lose a global assignment to a stronger match." />

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

The SORT and Deep SORT papers are concrete systems for the lecture's predict-and-associate pattern. Their historical results were obtained under their own detector and evaluation protocols; this chapter does not transfer any performance number to the room scenario. A current promptable tracker such as the one described in Meta's SAM 3 research may produce masks and track concepts through video, but it requires the appropriate prompt and task validation. Comparing a box tracker and a prompted mask tracker solely by their names would hide different annotation and integration costs.

For the room, camera motion creates another challenge. A fixed Kalman motion model in pixel coordinates may interpret a camera pan as all people moving together. Stabilisation, camera-motion compensation or a different coordinate frame can help. If the system is used for safety or access decisions, it should also have a review path for uncertain identity and a clear retention policy for track data. The toy IoU computation below is only a gate demonstration.

## Code you can run

The first block reproduces the lecture's pairwise IoU and threshold decision. The areas are given directly, so the computation does not invent box coordinates for those counts. It prints **0.429** after rounding and marks the pair eligible at a 0.3 threshold.

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

The lab uses the same numbers. Move intersection, union or gate to see when the pair becomes ineligible. Its output deliberately says “eligible”, not “assigned”, because another track may have a better feasible match. The data table includes every input and the decision.

<TrackingAssociationLab />

The second block checks the lecture's MOTA formula on a clearly synthetic count example: 100 ground-truth object observations, ten misses, five false positives and three identity switches yield **0.820**. The score can be negative if errors exceed the number of ground-truth observations. It cannot by itself tell whether errors are concentrated on one person or spread evenly.

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
- The lecture’s 30/70 IoU rounds to 0.429 and passes the stated 0.3 gate. That is a pairwise eligibility calculation, not a guaranteed global assignment.
- The MOTA example was executed locally with synthetic counts. No video tracker or model weights were run.

:::

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

## Further reading

- [Original SORT paper](https://arxiv.org/abs/1602.00763) for a motion-and-assignment tracker.
- [Original Deep SORT paper](https://arxiv.org/abs/1703.07402) for appearance-assisted association.
- [Meta SAM 3 research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) for a current prompted segmentation and tracking family.
- Built from the course lecture "cv-s14-object-tracking" (Lecture Library series).

- **[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)** `book`
  Richard Szeliski; The standard computer-vision reference; the author posts the full PDF.
- **[Stanford CS231n notes](https://cs231n.github.io/)** `course`
  Stanford; The classic notes on neural nets, backprop and CNNs; clear and example-driven.
- **[OpenCV documentation](https://docs.opencv.org/)** `docs`
  OpenCV; Practical reference for filtering, edges, features and the algorithms in this course.


:::note Qualification of a source association claim

The lecture says the Hungarian algorithm matches detections to tracks “to maximise total IoU”. That is true only when the cost matrix is built solely from negative IoU or an equivalent transformation. A tracker can combine motion, appearance and gating, in which case Hungarian optimises that combined cost. The lecture's 0.429 example establishes one eligible pair under a 0.3 IoU gate, not the final assignment.

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
