from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'cv'


def inverse_graphics():
    board = Board(1160, 500, 'Vision asks what produced the pixels', 'Two different 3D points can have the same pinhole projection')
    board.card(30, 130, 340, 145, 'Forward rendering', ['scene · surface · light', 'camera projection', 'one image'], 'blue', size=17)
    board.card(410, 130, 340, 145, 'Inverse vision', ['image measurements', 'candidate scene explanations', 'use priors and more views'], 'purple', size=17)
    board.card(790, 130, 340, 145, 'Ambiguity', ['(1, 1, 2) and (2, 2, 4)', 'focal length 2 → pixel (1, 1)', 'depth is not in one pixel'], 'orange', size=17)
    board.card(220, 350, 720, 90, 'The task determines the output', ['classification · detection · segmentation · tracking · 3D'], 'green', size=17)
    return board


def digital_image():
    board = Board(1160, 505, 'An image is sampled and quantised light', 'Spatial resolution and intensity resolution answer different questions')
    board.card(30, 135, 340, 150, 'Projection and sensor', ['scene light reaches sensor', 'illumination × reflectance', 'optics and exposure matter'], 'blue', size=17)
    board.card(410, 135, 340, 150, 'Sample in space', ['512 × 512 pixel grid', 'too sparse → aliasing', 'filter before downsampling'], 'purple', size=17)
    board.card(790, 135, 340, 150, 'Quantise intensity', ['8 bits → 256 levels', '3 channels per RGB pixel', 'too few → banding'], 'green', size=17)
    board.card(225, 360, 710, 85, 'Uncompressed teaching example', ['512 × 512 × 3 bytes = 786,432 bytes = 768 KiB'], 'yellow', size=17)
    return board


def colour_processing():
    board = Board(1160, 505, 'Remap pixels, then protect structure', 'A histogram describes intensities; a filter changes local neighbourhoods')
    board.card(30, 130, 340, 155, 'Global or point mapping', ['8 levels; CDF at 3 = 0.45', 'round(7 × 0.45) = 3', 'gamma 0.5 maps 0.25 → 0.5'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Neighbourhood smoothing', ['mean: equal pixel weights', 'Gaussian: favour near pixels', 'median: reject isolated spikes'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Colour representation', ['RGB: direct channel values', 'HSV: hue, saturation, value', 'lighting still changes colour'], 'green', size=17)
    board.card(215, 360, 730, 90, 'Check the task, not just visual appeal', ['contrast, noise, colour and edge preservation need separate tests'], 'yellow', size=17)
    return board


def gradients():
    board = Board(1160, 490, 'Edges are changes, not object names', 'The gradient gives strength and a direction across the edge')
    board.card(30, 130, 340, 150, 'Edge profile', ['step · ramp · roof', 'lighting can mimic a boundary', 'noise creates small changes'], 'blue', size=17)
    board.card(410, 130, 340, 150, 'Estimate gradients', ['Sobel or Prewitt masks', 'Gx = 4 and Gy = 3', 'magnitude = 5'], 'purple', size=17)
    board.card(790, 130, 340, 150, 'Interpret', ['angle = atan2(3, 4)', '36.87° across the edge', 'threshold needs context'], 'green', size=17)
    board.card(220, 350, 720, 85, 'Differentiate after controlling noise', ['a sharp response can come from texture, shadow or an object boundary'], 'yellow', size=17)
    return board


def canny_hough():
    board = Board(1160, 505, 'From a gradient ridge to a line vote', 'Canny thins and connects edges; Hough accumulates a line model')
    board.card(30, 130, 340, 155, 'Canny', ['smooth · gradient', 'non-max suppression', 'two thresholds · hysteresis'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Hough line', ['rho = x cosθ + y sinθ', 'each point votes for lines', 'peaks suggest candidates'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Worked point', ['point (2, 2) at 45°', 'rho = 2.828', 'a vote is not a proven line'], 'green', size=17)
    board.card(225, 360, 710, 90, 'Parameter choices control the result', ['resolution, thresholds and line support affect false detections'], 'yellow', size=17)
    return board


def harris_hog():
    board = Board(1160, 505, 'Locate corners; describe local shape', 'Harris responds to two-direction variation; HoG pools oriented gradients')
    board.card(30, 130, 340, 155, 'Flat, edge, corner', ['both eigenvalues small: flat', 'one large: edge', 'both large: corner'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Harris response', ['R = det(M) − k trace(M)²', 'lambda = (1, 1); k = 0.04', 'R = 0.84'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'HoG shape vector', ['9 bins × 4 cells × 105 blocks', '= 3,780 components', 'window geometry matters'], 'green', size=17)
    board.card(215, 360, 730, 90, 'A detector and descriptor solve different jobs', ['corner position is not yet a robust match or class decision'], 'yellow', size=17)
    return board


def sift_features():
    board = Board(1160, 505, 'SIFT finds scale-aware keypoints', 'Describe a neighbourhood relative to its detected scale and orientation')
    board.card(30, 130, 340, 155, 'Detect', ['Gaussian scale space', 'DoG extrema across scales', 'reject unstable points'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Orient and describe', ['dominant local direction', '4 × 4 spatial cells', '8 orientation bins'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Match carefully', ['4 × 4 × 8 = 128 values', 'ratio test screens ambiguity', 'geometry checks consistency'], 'green', size=17)
    board.card(225, 360, 710, 90, 'Invariance is approximate', ['blur, viewpoint, repetitive texture and low contrast still cause failure'], 'yellow', size=17)
    return board


def ransac_consensus():
    board = Board(1160, 505, 'RANSAC seeks agreement under outliers', 'The iteration formula counts a chance of an all-inlier minimal sample')
    board.card(30, 130, 340, 155, 'Sample and fit', ['choose a minimal set', 'fit a candidate geometry', 'reject degenerate samples'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Score and refine', ['count small residuals', 'keep strongest consensus', 'refit on its inliers'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Worked probability', ['w = 0.5; s = 2; p = 0.99', 'N raw = 16.008', 'ceil to 17 iterations'], 'green', size=17)
    board.card(225, 360, 710, 90, 'The lecture rounds this down to 16', ['16 trials reach about 98.998%; 17 reach at least 99%'], 'yellow', size=17)
    return board


def classification():
    board = Board(1160, 505, 'Classify an image, then measure errors', 'One image prediction and a dataset-level score answer different questions')
    board.card(30, 130, 340, 155, 'Represent and score', ['pixels → local or learned features', 'logits (2, 1, 0)', 'softmax → (.665, .245, .090)'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Choose a model', ['nearest neighbours or linear', 'CNN feature hierarchy', 'ViT patch attention'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Evaluate a threshold', ['TP 40 · FP 10 · FN 20', 'precision .800 · recall .667', 'F1 .727'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Inspect errors across real conditions', ['class balance, shifts, calibration and cost of each mistake'], 'yellow', size=17)
    return board


def visual_words():
    board = Board(1160, 505, 'Turn local descriptors into a fixed vector', 'A whole-image histogram counts visual words but forgets their locations')
    board.card(30, 130, 340, 155, 'Build a vocabulary', ['extract local descriptors', 'cluster training descriptors', 'each centre is a visual word'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Encode an image', ['assign each descriptor', 'counts (4, 1, 3); total 8', 'frequencies (.5, .125, .375)'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Restore some layout', ['split image into regions', 'histogram in each region', 'concatenate the vectors'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Dimension follows the codebook', ['500 words give 500 bins before a spatial-pyramid expansion'], 'yellow', size=17)
    return board


def classical_segmentation():
    board = Board(1160, 505, 'Partition an image into regions', 'A pixel-level partition can use intensity, seeds or graph structure')
    board.card(30, 130, 340, 155, 'Threshold and Otsu', ['choose one intensity split', 'between-class variance', 'uneven light can break it'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Cluster pixels', ['centres 50 and 200', 'pixel 120: distances 70, 80', 'assign centre 1'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Use neighbourhoods', ['region growth from seeds', 'watershed basins', 'graph cuts or superpixels'], 'green', size=17)
    board.card(210, 360, 740, 90, 'A region is not automatically a class', ['local grouping and semantic labels solve different tasks'], 'yellow', size=17)
    return board


def semantic_metrics():
    board = Board(1160, 505, 'Measure a pixel mask by its overlap', 'Semantic labels identify classes; instance labels separate objects')
    board.card(30, 130, 340, 155, 'Output contract', ['one class per pixel', 'instances need separate IDs', 'boundaries matter at small scale'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Worked masks', ['prediction 100 · truth 100', 'intersection 50 · union 150', 'IoU 0.333 · Dice 0.500'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Report honestly', ['per-class IoU and support', 'mean IoU needs class policy', 'inspect boundary errors'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Pixels have unequal practical cost', ['background accuracy can hide a missing rare object'], 'yellow', size=17)
    return board


def detection():
    board = Board(1160, 505, 'Detect objects and remove duplicate boxes', 'Boxes provide location, class and score for a variable number of objects')
    board.card(30, 130, 340, 155, 'Predict candidates', ['two-stage proposal families', 'one-stage or anchor-free families', 'preprocessing remains part of model'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Worked overlap', ['A: x 0–10; B: x 5–15', 'intersection 50 · union 150', 'IoU 0.333'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'NMS and evaluation', ['at threshold 0.5 keep both', 'AP from precision-recall', 'state IoU and class policy'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Real-time is a measured system target', ['device, image size, batch and postprocessing determine latency'], 'yellow', size=17)
    return board


def tracking():
    board = Board(1160, 505, 'Track identities over time', 'Detection creates observations; association maintains object IDs')
    board.card(30, 130, 340, 155, 'Predict', ['motion state and uncertainty', 'Kalman filter is one model', 'occlusion increases ambiguity'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Associate', ['candidate overlap 30/70', 'IoU 0.429 > gate 0.3', 'solve competing assignments'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Evaluate', ['misses · false positives', 'identity switches matter', 'MOTA = 1 − errors / GT'], 'green', size=17)
    board.card(210, 360, 740, 90, 'A pairwise overlap is only one cue', ['appearance, motion, timing and lifecycle policy protect identity'], 'yellow', size=17)
    return board


def edge_deployment():
    board = Board(1160, 505, 'Edge vision is a whole-system budget', 'Parameter counts and bit widths are useful estimates, not measured latency')
    board.card(30, 130, 340, 155, 'Choose an architecture', ['paper: VGG-16 138M', 'MobileNet V1 4.2M', '138 / 4.2 ≈ 32.9'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Store fewer bits', ['32-bit → 8-bit raw weights', '4× idealised storage reduction', '4.2M → 16.8 MB to 4.2 MB'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Test the device', ['capture · resize · inference', 'memory · heat · power', 'accuracy and tail latency'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Compression needs a measured acceptance test', ['scale metadata, activations and unsupported operators change real outcomes'], 'yellow', size=17)
    return board


def question_bank_map():
    board = Board(1160, 505, 'Thirty-six questions across the vision arc', 'The comprehensive source repeats the last seventeen main-bank questions')
    board.card(30, 130, 340, 155, 'Pixels and features', ['Q1–6: formation and sampling', 'Q7–13: edges and colour', 'Q14–19: SIFT, HoG, RANSAC'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Meaning and regions', ['Q20–24: classification', 'Q25–27: segmentation', 'Q28–30: detection'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Motion and deployment', ['Q31–32: tracking', 'Q33–34: visual words', 'Q35–36: edge devices'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Check the hidden assumptions', ['units, rounding, matching protocol and device speed need care'], 'yellow', size=17)
    return board


def midsem_methods():
    board = Board(1160, 505, 'Reconstruct the method, then insert scan values', 'The scan diagrams are unavailable; each panel is a labelled synthetic example')
    board.card(30, 130, 340, 155, 'Q4 · 150 / 50 step', ['5 × 5 vertical intensity split', '4-neighbour Laplacian −100, +100', 'Sobel first-derivative ridge'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Q5 · line voting', ['illustration: (1,1), (3,3)', 'line y = x', 'normal θ 135° · ρ 0'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Q6 · HoG and flow', ['toy gradient bin 40–60° wins', '9 unsigned 20° bins', 'Lucas–Kanade solves local flow'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Do not confuse examples with scan answers', ['the original pixel matrix, point values and cell gradients were not supplied'], 'yellow', size=17)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [
        ('inverse-graphics', inverse_graphics),
        ('digital-image', digital_image),
        ('colour-processing', colour_processing),
        ('gradients', gradients),
        ('canny-hough', canny_hough),
        ('harris-hog', harris_hog),
        ('sift-features', sift_features),
        ('ransac-consensus', ransac_consensus),
        ('classification', classification),
        ('visual-words', visual_words),
        ('classical-segmentation', classical_segmentation),
        ('semantic-metrics', semantic_metrics),
        ('detection', detection),
        ('tracking', tracking),
        ('edge-deployment', edge_deployment),
        ('question-bank-map', question_bank_map),
        ('midsem-methods', midsem_methods),
    ]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
