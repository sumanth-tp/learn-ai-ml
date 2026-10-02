"""Infographics for docs/theory/ml/03-ensembles-and-unsupervised-learning.

Run from the repo root:

    python3 scripts/infographics/ml_4.py            # all boards
    python3 scripts/infographics/ml_4.py kmeans     # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ml"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn
    return deco


def raw_text(b, x, y, text, size=12, fill=INK, anchor="middle", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def line(b, x1, y1, x2, y2, stroke=INK, width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{width}"{d} stroke-linecap="round"/>'
    )


def dot(b, cx, cy, r, fill, stroke, width=2):
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


@board("ensemble-learning-vote")
def ensemble_vote():
    b = Board(1100, 600, "Why a vote helps", "Three 70% models, a majority vote, and what shared mistakes do to it")
    b.group(20, 95, 440, 300, "Three independent voters", "blue")
    cards = []
    for i, name in enumerate(["Model A", "Model B", "Model C"]):
        cards.append(b.card(45, 140 + i * 78, 150, 62, name, ["right 70% of the time"], "blue", size=11))
    vote = b.card(255, 190, 170, 80, "majority vote", ["at least 2 of 3", "must be right"], "purple", size=12)
    for c in cards:
        b.arrow(c.right(), vote.left())
    b.card(255, 295, 170, 88, "78.4%", ["ensemble accuracy", "0.441 + 0.343"], "green", title_size=22)
    b.arrow(vote.bottom(), (340, 295))

    b.group(480, 95, 600, 300, "Accuracy of the vote, each voter 70% right", "teal")
    rows = [["voters", "independent", "rho = 0.5", "rho = 1.0"],
            ["1", "0.700", "0.700", "0.700"],
            ["3", "0.784", "0.742", "0.700"],
            ["5", "0.837", "0.768", "0.700"],
            ["11", "0.922", "0.811", "0.700"],
            ["25", "0.983", "0.841", "0.700"],
            ["101", "1.000", "0.850", "0.700"]]
    b.table(500, 140, [90, 150, 150, 150], rows, "teal", size=13, row_h=32)
    b.text(770, 382, "rho: chance that all voters copy one shared answer", 11, "teal", italic=True)

    b.card(20, 425, 530, 145, "Better than chance", ["Each voter must beat a coin flip.", "At 45% right, 101 voters reach only 0.156:", "adding voters makes the vote worse."], "red", size=13)
    b.card(570, 425, 510, 145, "Different mistakes", ["Fully shared errors (rho = 1.0) leave 101", "voters exactly as good as one: 0.700.", "Diversity is what the vote feeds on."], "orange", size=13)
    return b


@board("ensemble-learning-families")
def ensemble_families():
    b = Board(1240, 660, "Three ways to build an ensemble", "Bagging cuts variance, boosting cuts bias, stacking learns the blend")
    b.group(20, 95, 385, 430, "Bagging (parallel)", "blue")
    c1 = b.card(45, 140, 335, 70, "bootstrap resamples", ["63.2% of rows distinct,", "36.8% out-of-bag"], "blue", size=12)
    c2 = b.card(45, 240, 335, 56, "trees train independently", [], "blue", size=12)
    c3 = b.card(45, 326, 335, 56, "vote or average", [], "blue", size=12)
    b.arrow(c1.bottom(), c2.top())
    b.arrow(c2.bottom(), c3.top())
    b.card(45, 412, 335, 95, "30 fresh training sets", ["prediction variance:", "one tree 0.0761", "bagged trees 0.0207"], "green", size=12)

    b.group(425, 95, 390, 430, "Boosting (sequential)", "orange")
    d1 = b.card(450, 140, 340, 70, "round 1: error 0.300", ["alpha = 0.424"], "orange", size=12)
    d2 = b.card(450, 240, 340, 70, "re-weight the rows", ["wrong x1.53, right x0.65", "3 wrong rows carry 0.167 each"], "orange", size=12)
    d3 = b.card(450, 340, 340, 56, "next stump, repeat", [], "orange", size=12)
    b.arrow(d1.bottom(), d2.top())
    b.arrow(d2.bottom(), d3.top())
    b.card(450, 422, 340, 85, "same ten points", ["3 boosted stumps: 100%", "7 bagged stumps: 70%"], "green", size=12)

    b.group(835, 95, 385, 430, "Stacking (blend)", "purple")
    e1 = b.card(860, 140, 335, 80, "different model types", ["logistic, SVM, k-NN, forest"], "purple", size=12)
    e2 = b.card(860, 250, 335, 80, "meta-learner", ["trained on out-of-fold", "predictions"], "purple", size=12)
    b.arrow(e1.bottom(), e2.top())
    b.card(860, 360, 335, 147, "breast-cancer data", ["5-fold CV, repeated twice:", "best member   0.9754", "hard vote     0.9780", "stacking      0.9763"], "green", size=12, align="left")

    b.card(20, 555, 1200, 85, "Which problem do you have?", ["unstable flexible model: bagging   |   stable but too simple: boosting   |   several good, different models: blend"], "grey", size=13)
    return b


@board("gradient-boosting-in-practice-residual-loop")
def gb_loop():
    b = Board(1160, 580, "Gradient boosting is a residual loop", "80 noisy points from sin(x) + 0.5 sin(3x), depth-1 trees")
    s1 = b.card(30, 110, 250, 80, "1. start", ["predict the mean of y: -0.056", "RMSE 0.8800"], "blue", size=12)
    s2 = b.card(330, 110, 250, 80, "2. residuals", ["target minus prediction,", "one number per row"], "orange", size=12)
    s3 = b.card(330, 250, 250, 95, "3. fit a stump", ["first stump: split at 3.345,", "+0.764 left, -0.845 right"], "purple", size=12)
    s4 = b.card(30, 250, 250, 95, "4. add a fraction", ["prediction += learning rate", "x stump output"], "green", size=12)
    b.arrow(s1.right(), s2.left())
    b.arrow(s2.bottom(), s3.top())
    b.arrow(s3.left(), s4.right())
    b.arrow(s4.top(), s1.bottom(), label="repeat", color="red", dashed=True)
    b.card(30, 380, 550, 70, "check against the library", ["GradientBoostingRegressor, 100 stumps, rate 0.1:", "largest difference 2.2e-16"], "grey", size=12)
    b.card(30, 475, 550, 70, "stop rule", ["early stopping: quit when a held-out score", "has not improved for a set number of rounds"], "yellow", size=12)

    b.group(620, 100, 520, 445, "Error against the true curve (RMSE)", "teal")
    rows = [["rounds", "rate 1.0", "rate 0.1"],
            ["1", "0.2982", "0.7322"],
            ["5", "0.2772", "0.5072"],
            ["20", "0.2108", "0.2422"],
            ["100", "0.1672", "0.1778"],
            ["300", "0.1942", "0.1402"]]
    b.table(645, 145, [130, 170, 170], rows, "teal", size=14, row_h=36)
    b.card(645, 380, 470, 70, "rate 1.0 turns around", ["best near 100 rounds, then fits the noise", "(train RMSE keeps falling to 0.1154)"], "red", size=12)
    b.card(645, 465, 470, 60, "rate 0.1 is still improving", ["0.1402 at 300 rounds"], "green", size=12)
    return b


@board("gradient-boosting-in-practice-knobs")
def gb_knobs():
    b = Board(1240, 620, "The knobs that matter", "HistGradientBoostingClassifier, 8,000 rows, 20 features; names are scikit-learn's")
    b.group(20, 95, 560, 300, "Early stopping decides the number of trees", "teal")
    rows = [["setting", "trees", "AUC", "log loss"],
            ["rate 0.3, stop", "37", "0.9760", "0.1704"],
            ["rate 0.1, stop", "99", "0.9781", "0.1521"],
            ["rate 0.03, stop", "282", "0.9764", "0.1551"],
            ["rate 0.1, no stop", "600", "0.9787", "0.2977"]]
    b.table(40, 140, [200, 90, 120, 120], rows, "teal", size=13, row_h=36)
    b.card(40, 335, 520, 48, "Same ranking, nearly double the log loss without stopping", [], "red", size=12)

    b.group(600, 95, 620, 300, "Hyperparameters, by what they control", "purple")
    b.card(620, 140, 185, 235, "tree size", ["max_leaf_nodes 31", "max_depth none", "min_samples_leaf 20", "", "bigger trees need", "fewer rounds"], "blue", size=11)
    b.card(820, 140, 185, 235, "regularisation", ["learning_rate 0.1", "l2_regularization 0", "max_features 1.0", "early_stopping", "n_iter_no_change 10", "", "smaller steps, more trees"], "orange", size=11)
    b.card(1020, 140, 185, 235, "the data", ["NaN handled natively", "categorical_features", "monotonic_cst", "class_weight", "", "domain knowledge", "goes in here"], "green", size=11)

    b.card(20, 425, 1200, 75, "One knob at a time, early stopping on", ["every setting lands within 0.003 AUC (0.9757 to 0.9781): after early stopping, the rest is second order"], "yellow", size=13)
    b.card(20, 520, 590, 80, "Missing values", ["native NaN 0.8144, median 0.8114,", "median + indicator 0.8204, no gaps 0.7103"], "pink", size=12)
    b.card(630, 520, 590, 80, "Monotonic constraint", ["decreasing steps: free 522 of 1200, constrained 0", "RMSE vs truth 0.4359 -> 0.1645"], "pink", size=12)
    return b


@board("unsupervised-learning-kmeans")
def kmeans_board():
    b = Board(1160, 600, "k-means on the lecture's seven points", "k = 2, centroids start at 2 and 10; assign, update, repeat until nothing moves")
    x0, step, ay = 100.0, 90.0, 235.0
    px = lambda v: x0 + (v - 1) * step
    line(b, 70, ay, 1090, ay, FAINT, 2)
    for v in range(1, 13):
        line(b, px(v), ay - 5, px(v), ay + 5, FAINT, 1.4)
        raw_text(b, px(v), ay + 24, str(v), 11, FAINT)
    blue, orange = PALETTE["blue"], PALETTE["orange"]
    for v in (2, 3, 4, 5):
        dot(b, px(v), ay - 26, 13, blue["fill"], blue["stroke"])
    for v in (10, 11, 12):
        dot(b, px(v), ay - 26, 13, orange["fill"], orange["stroke"])
    for v in (2, 3, 4, 5):
        raw_text(b, px(v), ay - 21, str(v), 11, blue["text"], weight="700")
    for v in (10, 11, 12):
        raw_text(b, px(v), ay - 21, str(v), 11, orange["text"], weight="700")
    for v, col in ((3.5, blue), (11, orange)):
        cx = px(v)
        b.parts.append(f'<polygon points="{cx},{ay - 78} {cx - 12},{ay - 100} {cx + 12},{ay - 100}" fill="{col["stroke"]}"/>')
        raw_text(b, cx, ay - 108, f"centroid {v:g}", 12, col["text"], weight="700")
    for v, col in ((2, blue), (10, orange)):
        cx = px(v)
        b.parts.append(f'<polygon points="{cx},{ay + 36} {cx - 9},{ay + 54} {cx + 9},{ay + 54}" fill="none" stroke="{col["stroke"]}" stroke-width="2"/>')
        raw_text(b, cx, ay + 80, f"start {v}", 11, col["text"])
    b.arrow((px(2) + 16, ay + 46), (px(3.5) - 14, ay + 46), color="blue", label="update", dashed=True)
    b.arrow((px(10) + 16, ay + 46), (px(11) - 14, ay + 46), color="orange", label="update", dashed=True)

    c1 = b.card(30, 360, 340, 120, "pass 1: assign", ["nearest centroid wins:", "5 is 3 from centroid 2, 5 from 10", "{2,3,4,5} and {10,11,12}"], "blue", size=12)
    c2 = b.card(410, 360, 340, 120, "pass 1: update", ["mean of {2,3,4,5} = 3.5", "mean of {10,11,12} = 11", "WCSS = 7"], "orange", size=12)
    c3 = b.card(790, 360, 340, 120, "pass 2", ["same assignment, nothing moves:", "converged. scikit-learn gives", "centers 3.5 and 11, inertia 7.0"], "green", size=12)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    b.card(30, 510, 1100, 70, "WCSS = (1.5² + 0.5² + 0.5² + 1.5²) + (1² + 0² + 1²) = 5 + 2 = 7", ["the sum of squared distances to each point's own centroid"], "grey", size=13)
    return b


@board("unsupervised-learning-choosing-k")
def choosing_k():
    b = Board(1220, 620, "How many clusters?", "An elbow on 60 blob points, and a dendrogram on the lecture's seven")
    b.group(20, 95, 590, 500, "Elbow and silhouette (60 points)", "teal")
    inertia = [652.44, 345.74, 95.35, 79.43, 65.41, 52.99, 44.2, 36.86]
    ox, oy, ow, oh = 90.0, 520.0, 480.0, 250.0
    line(b, ox, oy, ox + ow, oy, FAINT, 1.6)
    line(b, ox, oy, ox, oy - oh, FAINT, 1.6)
    pts = []
    for i, w in enumerate(inertia):
        pts.append((ox + 30 + i * (ow - 60) / 7, oy - w / 652.44 * (oh - 20)))
    b.parts.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                   f'" fill="none" stroke="{PALETTE["teal"]["stroke"]}" stroke-width="2.4"/>')
    for i, ((x, y), w) in enumerate(zip(pts, inertia)):
        knee = i == 2
        dot(b, x, y, 7 if knee else 5, PALETTE["red"]["stroke"] if knee else "#ffffff", PALETTE["teal"]["stroke"] if not knee else PALETTE["red"]["stroke"])
        raw_text(b, x, oy + 20, str(i + 1), 12, INK)
        raw_text(b, x + (24 if i < 2 else 0), y - 12 if i else y - 12, f"{w:g}", 11, PALETTE["teal"]["text"])
    raw_text(b, ox + ow / 2, oy + 42, "k (number of clusters)", 12, FAINT)
    raw_text(b, ox - 14, oy - oh - 6, "WCSS", 12, FAINT, anchor="start")
    b.card(380, 135, 215, 70, "the knee is at k = 3", ["WCSS 345.74 -> 95.35"], "red", size=12)
    b.card(40, 135, 215, 70, "silhouette", ["k=2 0.474, k=3 0.658", "k=4 0.549: peak at 3"], "teal", size=11)

    b.group(630, 95, 570, 500, "Dendrogram, average linkage", "purple")
    leaves = [2, 3, 4, 5, 10, 11, 12]
    lx = {v: 690 + i * 70 for i, v in enumerate(leaves)}
    base, hs = 530.0, 38.0
    hy = lambda h: base - h * hs
    purple = PALETTE["purple"]["stroke"]

    def join(xa, ha, xb, hb, h):
        line(b, xa, hy(ha), xa, hy(h), purple, 2.2)
        line(b, xb, hy(hb), xb, hy(h), purple, 2.2)
        line(b, xa, hy(h), xb, hy(h), purple, 2.2)
        return (xa + xb) / 2

    m23 = join(lx[2], 0, lx[3], 0, 1.0)
    m45 = join(lx[4], 0, lx[5], 0, 1.0)
    m1011 = join(lx[10], 0, lx[11], 0, 1.0)
    left = join(m23, 1.0, m45, 1.0, 2.0)
    right = join(m1011, 1.0, lx[12], 0, 1.5)
    join(left, 2.0, right, 1.5, 7.5)
    for v in leaves:
        raw_text(b, lx[v], base + 20, str(v), 13, INK, weight="700")
    for h in (0, 2, 4, 6, 8):
        raw_text(b, 660, hy(h) + 4, str(h), 10, FAINT, anchor="end")
    line(b, 665, hy(4.2), 1185, hy(4.2), PALETTE["red"]["stroke"], 2, "8 6")
    raw_text(b, 926, hy(4.2) - 8, "cut here: 2 clusters", 12, PALETTE["red"]["text"], weight="700")
    raw_text(b, 915, hy(7.5) - 10, "last merge at 7.5", 11, PALETTE["purple"]["text"], weight="700")
    b.card(660, 145, 250, 56, "merge heights", ["1, 1, 1, 1.5, 2, 7.5"], "purple", size=12)
    return b


@board("unsupervised-learning-pca")
def pca_board():
    b = Board(1220, 640, "PCA keeps the high-variance directions", "The lecture's eigenvalues, a rotating axis, and the scaling trap")
    b.group(20, 95, 560, 330, "Eigenvalues 6.2, 2.4, 1.0, 0.4 (total 10)", "blue")
    vals = [6.2, 2.4, 1.0, 0.4]
    shares = ["62%", "24%", "10%", "4%"]
    cum = ["62%", "86%", "96%", "100%"]
    base = 365.0
    blue, green = PALETTE["blue"], PALETTE["green"]
    for i, (v, s, c) in enumerate(zip(vals, shares, cum)):
        x = 60 + i * 130
        h = v / 6.2 * 190
        keep = i < 2
        col = blue if keep else PALETTE["grey"]
        b.parts.append(f'<rect x="{x}" y="{base - h:.1f}" width="80" height="{h:.1f}" rx="6" fill="{col["fill"]}" stroke="{col["stroke"]}" stroke-width="2"/>')
        raw_text(b, x + 40, base - h - 8, f"{v:.1f}", 13, col["text"], weight="700")
        raw_text(b, x + 40, base + 20, f"PC{i + 1}", 12, INK, weight="700")
        raw_text(b, x + 40, base + 38, f"{s}, total {c}", 11, FAINT)
    b.card(330, 150, 235, 66, "keep 2 components", ["62% + 24% = 86%"], "green", size=12)

    b.group(600, 95, 600, 330, "The scaling trap, wine dataset", "orange")
    rows = [["", "PC1 share", "k-means ARI"],
            ["raw columns", "99.8%", "0.371"],
            ["standardised", "36.2%", "0.897"],
            ["standardised + PCA(2)", "55.4% (2 PCs)", "0.895"]]
    b.table(620, 140, [210, 180, 170], rows, "orange", size=13, row_h=40)
    b.card(620, 325, 560, 75, "one column (proline) is the whole raw PCA", ["PCA follows variance, and variance depends on units: standardise first"], "red", size=12)

    b.group(20, 450, 1180, 170, "Rotating the axis on 60 correlated points", "purple")
    b.card(45, 495, 270, 105, "angle 0 degrees", ["variance 2.626", "70.1% of total"], "grey", size=12)
    b.card(335, 495, 270, 105, "angle 45 degrees", ["variance 3.279", "87.5% of total"], "grey", size=12)
    b.card(625, 495, 270, 105, "angle 30.9 degrees = PC1", ["variance 3.468", "92.6% of total 3.746"], "green", size=12)
    b.card(915, 495, 265, 105, "PC2 at right angles", ["eigenvalue 0.278", "what a 1-D projection drops"], "orange", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
