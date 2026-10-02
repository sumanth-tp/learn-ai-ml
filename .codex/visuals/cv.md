# Computer Vision lab specifications

## PerspectiveProjectionLab

- Controls: focal length 1 to 5 in 0.5 steps; near point depth 1 to 5 in 0.5 steps. The two example points are `(1, 1, depth)` and `(2, 2, 2 × depth)`.
- Defaults: focal length 2 and near depth 2; both points map to image coordinate `(1, 1)`, as the chapter code prints.
- Drawing: side-by-side point coordinates and a shared projected pixel marker. Explain that one projection cannot recover a unique depth.
- Data view: point coordinates, depth and projected x/y for each point through VizPanel.

## SamplingQuantisationLab

- Controls: square image side 64 to 1024 pixels in 64-pixel steps, channels 1 or 3, bit depth 1 to 16 bits.
- Defaults: 512 pixels, 3 channels, 8 bits; 256 levels per channel, 786,432 bytes and 768 KiB before compression.
- Drawing: spatial grid and intensity-level count with a storage estimate. For bit depths not divisible by eight, storage is a packed-bit minimum estimate, clearly labelled.
- Data view: dimensions, channels, levels and minimum packed storage in bytes.

## GammaTransformLab

- Controls: input intensity 0 to 1 in steps of 0.01 and gamma 0.2 to 3 in steps of 0.1.
- Defaults: input 0.25 and gamma 0.5; output `0.25 ** 0.5 = 0.5`, matching the chapter's code.
- Drawing: input and output bars plus a sampled gamma curve.
- Data view: input, gamma, output and five sampled curve values.

## EdgeGradientLab

- Controls: horizontal gradient `Gx` from -10 to 10 and vertical gradient `Gy` from -10 to 10, integer steps.
- Defaults: `Gx=4`, `Gy=3`; magnitude `5` and orientation `36.87°`, matching the Session 3 worked number and code.
- Drawing: component bars and a vector arrow within an SVG coordinate plane.
- Data view: components, magnitude and angle in degrees.

## HoughVoteLab

- Controls: point `x,y` from -5 to 5 and line-normal angle 0 to 180 degrees in 15-degree steps.
- Defaults: point `(2,2)`, angle 45°; `rho=2.828` rounded, matching Session 4 code.
- Drawing: a point and the selected line-normal projection, with the rho calculation.
- Data view: x, y, angle, cosine, sine and rho.

## HarrisResponseLab

- Controls: two nonnegative structure-tensor eigenvalues from 0 to 2 in 0.1 steps and Harris k from 0.02 to 0.10 in 0.01 steps.
- Defaults: eigenvalues 1 and 1, k=0.04; response `0.84`, matching Session 6 code.
- Drawing: eigenvalue bars and response classification as corner, edge or flat for the selected values.
- Data view: eigenvalues, determinant, trace, k and response.

## SiftDescriptorLab

- Controls: spatial cells per side from 2 to 6 and orientation bins from 4 to 12.
- Defaults: 4 by 4 cells and 8 bins; dimension `128`, matching Session 7 code.
- Drawing: cell grid and a dimension multiplication expression; this illustrates descriptor layout rather than actual SIFT extraction.
- Data view: cells per side, total cells, bins and descriptor dimensions.

## RansacIterationsLab

- Controls: inlier fraction 0.1 to 0.9 in 0.05 steps, minimal sample size 2 to 5, target success probability 0.80 to 0.999 in 0.001 steps.
- Defaults: inlier fraction 0.5, sample size 2, target success 0.99; raw iteration value `16.008`, so the smallest integer meeting the target is `17`. Session 8's source rounds to 16 and misses the target slightly.
- Drawing: required iteration count and probability of drawing at least one all-inlier sample by the selected count.
- Data view: parameters, one-sample success probability, continuous bound, integer count and achieved probability.

## ClassificationMetricsLab

- Controls: true positives 0 to 100, false positives 0 to 100, false negatives 0 to 100, each in steps of 1. A separate three-logit selector keeps the worked logits `[2, 1, 0]` and lets the first logit vary from -2 to 4.
- Defaults: logits `[2, 1, 0]` give softmax values approximately `[0.665, 0.245, 0.090]`; TP 40, FP 10, FN 20 give precision 0.800, recall 0.667 and F1 0.727, matching chapter code.
- Drawing: a three-class softmax bar display plus precision and recall bars. Zero denominators show an undefined metric rather than a manufactured success rate.
- Data view: logits, each normalised probability, confusion counts and all three metrics through VizPanel.

## VisualWordsLab

- Controls: integer counts for three visual words from 0 to 12. All-zero input is explicitly labelled as having no normalised histogram.
- Defaults: counts `[4, 1, 3]`, total 8 and term frequencies `[0.5, 0.125, 0.375]`, matching chapter code.
- Drawing: three frequency bars, followed by a two-region comparison to show that a whole-image bag discards location.
- Data view: word, count and normalised frequency; total count and fixed descriptor dimension.

## PixelClusterLab

- Controls: pixel intensity 0 to 255 and two cluster centres 0 to 255. Defaults: pixel 120, centres 50 and 200; distances 70 and 80, so centre 1 wins, as in the lecture and code. An exact distance tie uses the first centre and is labelled as a tie.
- Drawing: one-dimensional intensity ruler with the pixel and both centres, and explicit distances.
- Data view: pixel, both centres, both absolute distances and assigned centre.

## MaskOverlapLab

- Controls: predicted-mask size and ground-truth-mask size 0 to 200, overlap 0 to the smaller size. Defaults: sizes 100 and 100, overlap 50; union 150, IoU 0.333 and Dice 0.500 as in the lecture and code. Empty-union metrics are labelled undefined.
- Drawing: two overlapping proportional horizontal bars or area blocks with numeric intersection, union and overlap scores; avoid implying a literal geometric mask from count-only inputs.
- Data view: both sizes, intersection, union, IoU and Dice.

## BoxIouNmsLab

- Controls: horizontal start of the second 10 by 10 box from 0 to 15 and NMS threshold 0.1 to 0.9. Defaults: box A `[0,0,10,10]`, B `[5,0,15,10]`, intersection 50, union 150, IoU 0.333 and threshold 0.5, so both survive. Equal scores have a fixed stated priority for A.
- Drawing: SVG boxes with overlap shaded and a keep/suppress decision at the selected threshold.
- Data view: both coordinates, intersection, union, IoU, threshold and decision.

## TrackingAssociationLab

- Controls: intersection area 0 to 70, union area 70 to 140 with intersection constrained to no more than union, and association threshold 0.1 to 0.9. Defaults: intersection 30, union 70, IoU 0.429, threshold 0.3, so association is accepted. This is a pairwise score illustration, not a full assignment solver.
- Drawing: score and threshold bars with an accept/reject result. Explain that multi-object assignment also needs a feasible cost matrix and identity policy.
- Data view: intersection, union, IoU, threshold and pairwise decision.

## EdgeWeightBudgetLab

- Controls: choose the source example VGG-16 at 138 million parameters or MobileNet V1 1.0-224 at 4.2 million parameters; choose 32, 16, 8 or 4 bits per stored parameter. Defaults: MobileNet example at 8 bits. The original paper's rounded 138/4.2 ratio is 32.857, about 33; idealised dense weight storage for 4.2 million parameters is 16.8 decimal MB at 32 bits and 4.2 decimal MB at 8 bits, exactly fourfold.
- Drawing: two bars for idealised 32-bit and selected-bit raw weight storage. State that packaging, scales, activations and operator support can change actual artifact size, memory and latency.
- Data view: source model and rounded parameter count, precision, idealised raw storage at both precisions, storage ratio and the two-model parameter ratio.

## HogBinLab

- Controls: unsigned gradient orientation 0 to 179 degrees in integer steps and magnitude 0 to 10 in integer steps. Defaults: orientation 50 degrees, magnitude 5; hard-assignment to unsigned 20-degree bins puts it in the 40–60-degree bin, index 2, with weight 5. This is a synthetic worked example because the mid-semester scan's gradients are unavailable.
- Drawing: nine bins with the selected bin filled according to the gradient magnitude. Label hard assignment as a teaching simplification; production HoG often interpolates between bins and normalises over blocks.
- Data view: orientation, magnitude, bin index, bin range and all nine bin weights.
