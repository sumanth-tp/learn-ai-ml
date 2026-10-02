"""Infographics for docs/theory/ml/02-supervised-learning, chapters 4 to 6
(instance-based learning, support vector machines, Bayesian learning).

Every number on these boards is printed by the code blocks of the matching
chapter. Run from the repo root:

    python3 scripts/infographics/ml_3.py            # all boards
    python3 scripts/infographics/ml_3.py svm_margin # just one
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, INK, FAINT, MONO  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ml"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


def dot(b, x, y, color, shape="circle", r=7, ring=None, label="", dashed_ring=False):
    c = PALETTE[color]["stroke"]
    f = PALETTE[color]["fill"]
    if ring:
        b.parts.append(f'<circle cx="{x}" cy="{y}" r="{r + 5}" fill="none" stroke="{ring}" stroke-width="2.2"/>')
    if shape == "circle":
        b.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="{INK}" stroke-width="1"/>')
    else:
        b.parts.append(
            f'<rect x="{x - r}" y="{y - r}" width="{2 * r}" height="{2 * r}" fill="{c}" stroke="{INK}" stroke-width="1"/>'
        )
    if label:
        b.text(x + r + 6, y - r - 2, label, 12, "grey", "700", anchor="start")


def axes(b, x0, y0, w, h, xlabel="", ylabel="", ticks_x=(), ticks_y=(), fx=None, fy=None):
    b.parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#ffffff" stroke="#ced4da"/>')
    for t in ticks_x:
        px = fx(t)
        b.parts.append(f'<line x1="{px}" y1="{y0}" x2="{px}" y2="{y0 + h}" stroke="#eceff1"/>')
        b.text(px, y0 + h + 15, f"{t:g}", 11, FAINT)
    for t in ticks_y:
        py = fy(t)
        b.parts.append(f'<line x1="{x0}" y1="{py}" x2="{x0 + w}" y2="{py}" stroke="#eceff1"/>')
        b.text(x0 - 8, py + 4, f"{t:g}", 11, FAINT, anchor="end")
    if xlabel:
        b.text(x0 + w / 2, y0 + h + 32, xlabel, 12, "grey", "700")
    if ylabel:
        b.text(x0 - 8, y0 - 8, ylabel, 12, "grey", "700", anchor="end")


# ------------------------------------------------------------------ chapter 4


@board
def knn_lazy_vote():
    b = Board(1180, 560, "k-NN: store everything, vote at query time",
              "The lecture's five points, query q = (2, 3), k = 3 (output of knn_1.py)")

    b.group(30, 95, 330, 440, "Eager vs lazy", "blue")
    e = b.card(50, 140, 290, 150, "Eager learner", ["fit once: compress the", "data into a model", "predict: cheap", "e.g. linear model, tree"], "blue", title_size=15)
    lz = b.card(50, 320, 290, 190, "Lazy (instance-based)", ["training: store the data", "predict: measure the", "distance to every stored", "point, take the k closest", "and let them vote"], "orange", title_size=15)
    b.text(195, 308, "versus", 12, "grey", italic=True)

    b.group(385, 95, 765, 440, "Worked example", "green")
    x0, y0, w, h = 430, 150, 330, 330
    fx = lambda v: x0 + v / 6 * w
    fy = lambda v: y0 + h - v / 6 * h
    axes(b, x0, y0, w, h, "x", "", range(0, 7), range(0, 7), fx, fy)
    q = (2, 3)
    pts = {"A": (1, 1, "+"), "B": (2, 2, "+"), "C": (3, 3, "-"), "D": (5, 1, "-"), "E": (1, 4, "+")}
    r = math.sqrt(2)
    b.parts.append(
        f'<circle cx="{fx(q[0])}" cy="{fy(q[1])}" r="{r / 6 * w:.1f}" fill="none" stroke="{FAINT}" stroke-width="1.6" stroke-dasharray="6 5"/>'
    )
    for n, (px_, py_, s) in pts.items():
        if n in "BCE":
            b.parts.append(
                f'<line x1="{fx(q[0])}" y1="{fy(q[1])}" x2="{fx(px_)}" y2="{fy(py_)}" stroke="{INK}" stroke-width="1.4"/>'
            )
    for n, (px_, py_, s) in pts.items():
        dot(b, fx(px_), fy(py_), "blue" if s == "+" else "orange", "circle" if s == "+" else "square",
            ring=INK if n in "BCE" else None, label=n)
    cx, cy = fx(q[0]), fy(q[1])
    b.parts.append(
        f'<polygon points="{cx},{cy - 10} {cx + 10},{cy} {cx},{cy + 10} {cx - 10},{cy}" fill="{INK}"/>'
    )
    b.text(cx + 14, cy + 22, "q", 13, "grey", "700", anchor="start")

    b.table(790, 150, [110, 80, 80, 80],
            [["point", "class", "distance", "1/d²"],
             ["B", "+", "1.00", "1.0"],
             ["C", "-", "1.00", "1.0"],
             ["E", "+", "1.41", "0.5"],
             ["A (k=4)", "+", "2.24", "-"],
             ["D (k=5)", "-", "3.61", "-"]], "green", size=12)
    b.card(790, 350, 330, 62, "One vote each", ["+ 2 : - 1  ->  predicts +", "share of +  67%"], "green", size=12)
    b.card(790, 422, 330, 62, "Weighted 1/d²", ["+ 1.5 : - 1.0  ->  predicts +", "share of +  60%"], "yellow", size=12)
    b.text(955, 503, "weighting did not make the vote more lopsided:", 11, "grey", italic=True)
    b.text(955, 519, "the share of + fell from 67% to 60%", 11, "grey", italic=True)
    return b


@board
def knn_k_scale_dimensions():
    b = Board(1240, 600, "Three things that decide whether k-NN works",
              "Numbers from knn_2.py and knn_3.py (5-fold cross-validation, seeds fixed)")

    b.group(25, 95, 385, 480, "1. Choose k", "purple")
    rows = [["k", "train", "cv"], ["1", "1.000", "0.883"], ["3", "0.940", "0.920"], ["25", "0.920", "0.907"],
            ["101", "0.860", "0.827"], ["201", "0.760", "0.727"]]
    b.table(45, 140, [90, 130, 130], rows, "purple", size=13)
    b.card(45, 345, 345, 70, "k = 1: memorises", ["perfect on training data,", "only 0.883 outside it"], "purple", size=12)
    b.card(45, 430, 345, 70, "k = 201: over-smoothed", ["blurs the two moons;", "both scores fall"], "purple", size=12)
    b.text(217, 530, "two-moons data, 300 points, noise 0.35", 11, "grey", italic=True)
    b.text(217, 548, "k is the bias-variance knob", 11, "grey", italic=True)

    b.group(430, 95, 385, 480, "2. Put features on one scale", "orange")
    b.card(450, 140, 345, 90, "Wine, 13 features", ["range of alcohol 3.8", "range of proline 1402"], "orange", size=13)
    for i, (label, val, col) in enumerate([("raw features", 0.663, "red"), ("standardised", 0.961, "green")]):
        y = 275 + i * 80
        b.text(455, y, f"{label}: 5-NN accuracy", 13, "grey", "700", anchor="start")
        b.bar(455, y + 12, 270, val, color=col, h=18)
        b.text(740, y + 27, f"{val:.3f}", 14, col, "700", anchor="start")
    b.card(450, 445, 345, 100, "Why", ["Euclidean distance adds raw", "units, so proline drowns", "alcohol. Standardise inside", "the pipeline."], "orange", size=12)

    b.group(835, 95, 385, 480, "3. Beware many dimensions", "teal")
    b.table(855, 140, [100, 245], [["d", "nearest / farthest"], ["2", "0.020"], ["10", "0.290"], ["50", "0.619"],
                                  ["200", "0.789"], ["1000", "0.899"]], "teal", size=13)
    b.text(1027, 352, "closer to 1 = every point is about equally far", 11, "grey", italic=True)
    b.table(855, 372, [190, 155], [["useless columns", "5-NN accuracy"], ["0", "0.961"], ["50", "0.854"],
                                  ["200", "0.708"], ["500", "0.557"]], "teal", size=13)
    b.text(1027, 560, "wine data plus random noise columns", 11, "grey", italic=True)
    return b


# ------------------------------------------------------------------ chapter 5


@board
def svm_margin():
    b = Board(1220, 580, "SVM: the widest street between two classes",
              "Left: w = (1, 1), b = -3 (svm_1.py). Right: one stray + point and the C knob (svm_2.py)")

    b.group(25, 95, 580, 465, "Maximum margin", "blue")
    x0, y0, w, h = 85, 140, 380, 380
    lo, hi = -1.0, 5.0
    fx = lambda v: x0 + (v - lo) / (hi - lo) * w
    fy = lambda v: y0 + h - (v - lo) / (hi - lo) * h
    axes(b, x0, y0, w, h, "x1", "x2", range(-1, 6), range(-1, 6), fx, fy)
    def seg(c, color, dash=False, width=2.2):
        x_a, x_b = lo, hi
        dsh = ' stroke-dasharray="7 5"' if dash else ""
        b.parts.append(
            f'<line x1="{fx(x_a)}" y1="{fy(c - x_a)}" x2="{fx(x_b)}" y2="{fy(c - x_b)}" stroke="{color}" stroke-width="{width}"{dsh}/>'
        )
    b.parts.append(
        f'<clipPath id="cp"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath><g clip-path="url(#cp)">'
    )
    b.parts.append(
        f'<polygon points="{fx(lo)},{fy(4 - lo)} {fx(hi)},{fy(4 - hi)} {fx(hi)},{fy(2 - hi)} {fx(lo)},{fy(2 - lo)}" fill="#e7f5ff" opacity="0.8"/>'
    )
    seg(3, INK)
    seg(4, FAINT, True, 1.6)
    seg(2, FAINT, True, 1.6)
    b.parts.append("</g>")
    pos = [(2, 2), (3, 1), (4, 3), (3, 4)]
    neg = [(1, 1), (2, 0), (0, 0), (0, 1)]
    sv = {(2, 2), (3, 1), (1, 1), (2, 0)}
    for p in pos:
        dot(b, fx(p[0]), fy(p[1]), "blue", "circle", ring=INK if p in sv else None)
    for p in neg:
        dot(b, fx(p[0]), fy(p[1]), "orange", "square", ring=INK if p in sv else None)
    b.text(fx(1.7), fy(3.35), "w.x + b = 0", 12, "grey", "700", anchor="start")
    b.card(485, 150, 110, 120, "margin", ["2/||w||", "= sqrt(2)", "= 1.41"], "blue", size=12)
    b.card(485, 290, 110, 110, "4 support", ["vectors:", "(2,2) (3,1)", "(1,1) (2,0)"], "yellow", size=11)
    b.card(485, 420, 110, 100, "checks", ["(2,2): +1", "(1,1): -1"], "green", size=12)

    b.group(630, 95, 565, 465, "Soft margin: the C knob", "purple")
    b.text(912, 135, "extra + point at (1.2, 1.3), deep on the - side", 12, "grey", italic=True)
    rows = [["C", "margin", "support vectors", "train accuracy"],
            ["0.01", "22.63", "9", "0.56"], ["0.1", "4.80", "8", "0.89"], ["1", "1.41", "4", "0.89"],
            ["10", "0.55", "2", "1.00"], ["100", "0.36", "2", "1.00"]]
    b.table(655, 150, [70, 110, 190, 150], rows, "purple", size=13)
    b.card(655, 356, 255, 100, "small C: tolerant", ["wide street, many points", "inside it, more bias"], "purple", size=12)
    b.card(930, 356, 245, 100, "large C: strict", ["narrow street bent to", "fit the stray point"], "red", size=12)
    b.card(655, 474, 520, 66, "Always scale features first", ["wine, RBF SVC: 0.657 raw -> 0.983 scaled in a pipeline"], "yellow", size=12)
    return b


@board
def svm_kernel_lift():
    b = Board(1240, 625, "The kernel trick: lift, cut, map back",
              "Numbers from svm_3.py and svm_4.py")

    b.group(25, 95, 395, 270, "1. No line separates these", "red")
    xs_pos = [-1.5, -1, 0, 1, 1.5]
    xs_neg = [-4, -3, -2.5, 2.5, 3, 4]
    fx = lambda v: 60 + (v + 4.5) / 9 * 325
    b.parts.append(f'<line x1="50" y1="230" x2="395" y2="230" stroke="{INK}" stroke-width="2"/>')
    for v in xs_pos:
        dot(b, fx(v), 230, "blue", "circle")
    for v in xs_neg:
        dot(b, fx(v), 230, "orange", "square")
    for t in (-4, -2, 0, 2, 4):
        b.text(fx(t), 262, str(t), 11, FAINT)
    b.text(222, 305, "any single cut leaves errors:", 12, "red", "700")
    b.text(222, 325, "linear SVM accuracy 0.55", 13, "red", "700")

    b.arrow((425, 230), (480, 230), label="lift\n(x, x²)", color="purple", width=2.4)

    b.group(490, 95, 400, 270, "2. A straight cut now works", "green")
    px0, py0, pw, ph = 525, 135, 340, 195
    fxl = lambda v: px0 + (v + 4.5) / 9 * pw
    fyl = lambda v: py0 + ph - v / 18 * ph
    b.parts.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="#ffffff" stroke="#ced4da"/>')
    pts = [f"{fxl(x / 10):.1f},{fyl(x * x / 100):.1f}" for x in range(-42, 43)]
    b.parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#ced4da" stroke-width="1.5"/>')
    b.parts.append(f'<line x1="{px0}" y1="{fyl(4.25)}" x2="{px0 + pw}" y2="{fyl(4.25)}" stroke="{INK}" stroke-width="2.2"/>')
    for v in xs_pos:
        dot(b, fxl(v), fyl(v * v), "blue", "circle", r=6)
    for v in xs_neg:
        dot(b, fxl(v), fyl(v * v), "orange", "square", r=6)
    b.text(px0 + pw - 4, fyl(4.25) - 7, "cut at x² = 4.25", 12, "grey", "700", anchor="end")
    b.text(690, 350, "accuracy 1.00 on the lifted points", 12, "green", "700")

    b.arrow((895, 230), (940, 230), label="map\nback", color="purple", width=2.4)

    b.group(950, 95, 265, 270, "3. Back on the line", "blue")
    fx2 = lambda v: 975 + (v + 4.5) / 9 * 215
    b.parts.append(f'<rect x="{fx2(-2.062)}" y="195" width="{fx2(2.062) - fx2(-2.062)}" height="70" fill="#e7f5ff"/>')
    b.parts.append(f'<line x1="965" y1="230" x2="1200" y2="230" stroke="{INK}" stroke-width="2"/>')
    for v in (-2.062, 2.062):
        b.parts.append(f'<line x1="{fx2(v)}" y1="185" x2="{fx2(v)}" y2="275" stroke="{INK}" stroke-width="2.2"/>')
    for v in xs_pos:
        dot(b, fx2(v), 230, "blue", "circle", r=5)
    for v in xs_neg:
        dot(b, fx2(v), 230, "orange", "square", r=5)
    b.text(1082, 305, "cut falls at x = ±2.062", 13, "blue", "700")
    b.text(1082, 325, "a curved boundary,", 12, "grey", italic=True)
    b.text(1082, 341, "found with a straight cut", 12, "grey", italic=True)

    b.group(25, 385, 580, 220, "Never compute the lift", "purple")
    b.card(45, 435, 270, 145, "Same number, two routes", ["(x.z + 1)^2 = 4.0", "lift to 6 coordinates and", "take the dot product = 4.0", "RBF by hand = 0.001503"], "purple", size=12)
    b.card(335, 435, 255, 145, "Two rings, test accuracy", ["linear     0.522", "poly deg 2  1.000", "rbf        0.989"], "yellow", size=13)

    b.group(630, 385, 585, 220, "RBF: gamma sets each point's reach", "teal")
    b.table(650, 430, [100, 90, 90, 140], [["gamma", "train", "test", "support vectors"], ["0.01", "0.819", "0.789", "144"],
            ["1", "0.905", "0.933", "76"], ["10", "0.933", "0.944", "112"], ["100", "0.971", "0.878", "197"]],
            "teal", size=12)
    return b


# ------------------------------------------------------------------ chapter 6


@board
def bayes_base_rate():
    b = Board(1240, 600, "A positive test and the base rate",
              "0.1% prevalence, 99% sensitivity, 5% false positives (bayes_1.py)")

    b.group(25, 95, 800, 490, "Out of 100,000 people", "blue")
    pop = b.card(300, 140, 250, 60, "100,000 tested", [], "blue", title_size=16)
    sick = b.card(120, 250, 220, 80, "100 sick", ["0.1%"], "red", title_size=16)
    well = b.card(510, 250, 240, 80, "99,900 healthy", ["99.9%"], "green", title_size=16)
    b.arrow(pop.bottom(0.3), sick.top(), color="red")
    b.arrow(pop.bottom(0.7), well.top(), color="green")
    tp = b.card(60, 400, 140, 70, "99", ["test positive"], "red", title_size=18)
    fn = b.card(215, 400, 130, 70, "1", ["test negative"], "grey", title_size=18)
    fp = b.card(470, 400, 150, 70, "4,995", ["test positive"], "orange", title_size=18)
    tn = b.card(640, 400, 150, 70, "94,905", ["test negative"], "grey", title_size=16)
    b.arrow(sick.bottom(0.3), tp.top(), label="99%")
    b.arrow(sick.bottom(0.75), fn.top(), label="1%")
    b.arrow(well.bottom(0.3), fp.top(), label="5%")
    b.arrow(well.bottom(0.75), tn.top(), label="95%")
    b.card(60, 505, 560, 60, "Everyone who tests positive: 99 + 4,995 = 5,094", [], "yellow", title_size=14)
    b.card(640, 505, 160, 60, "99 / 5,094", ["= 0.0194"], "pink", title_size=14)

    b.group(850, 95, 365, 255, "Rarer disease, weaker belief", "purple")
    b.table(870, 140, [155, 175], [["prevalence", "P(disease | +)"], ["0.01%", "0.0020"], ["0.1%", "0.0194"],
                                  ["1%", "0.1667"], ["10%", "0.6875"], ["50%", "0.9519"]], "purple", size=13)

    b.group(850, 365, 365, 220, "Evidence accumulates", "teal")
    steps = [("prior", 0.001), ("1 positive", 0.0194), ("2 positives", 0.2818), ("3 positives", 0.8860)]
    for i, (label, v) in enumerate(steps):
        y = 410 + i * 35
        b.text(870, y + 14, label, 12, "grey", "700", anchor="start")
        b.bar(985, y, 140, v, color="teal", h=16)
        b.text(1135, y + 13, f"{v:.4f}" if v >= 0.01 else "0.0010", 12, "teal", "700", anchor="start")
    b.text(1032, 568, "each test assumed independent given the disease", 10, "grey", italic=True)
    return b


@board
def bayes_naive_bayes():
    b = Board(1240, 620, "Naive Bayes, MAP, and smoothing",
              "Numbers from bayes_2.py, bayes_3.py and bayes_4.py")

    b.group(25, 95, 600, 280, "The spam filter: multiply the likelihoods", "green")
    b.text(325, 140, "P(class | words) ∝ P(class) × P(word1 | class) × P(word2 | class)", 12, "green", "700")
    sp = b.card(50, 165, 265, 150, "spam", ["P(spam) = 0.4", "P(free | spam) = 0.8", "P(money | spam) = 0.6", "0.4 × 0.8 × 0.6 = 0.192"], "red", size=12)
    hm = b.card(335, 165, 265, 150, "ham", ["P(ham) = 0.6", "P(free | ham) = 0.1", "P(money | ham) = 0.2", "0.6 × 0.1 × 0.2 = 0.012"], "blue", size=12)
    b.card(50, 325, 550, 38, "P(spam | free, money) = 0.192 / (0.192 + 0.012) = 0.941", [], "green", title_size=13)

    b.group(650, 95, 565, 280, "MAP vs maximum likelihood", "purple")
    b.text(932, 138, "data: three heads in three flips", 12, "grey", italic=True)
    b.table(670, 152, [150, 130, 130, 125], [["", "fair coin", "biased 0.8", "winner"],
            ["likelihood", "0.125", "0.512", "biased"],
            ["× prior", "0.95", "0.05", ""],
            ["MAP score", "0.1187", "0.0256", "fair"]], "purple", size=12)
    b.card(670, 290, 525, 70, "ML = MAP with a flat prior", ["Beta(1,1): 1.000   Beta(2,2): 0.800   Beta(10,10): 0.571"], "purple", size=12)

    b.group(25, 395, 1190, 205, "Laplace smoothing: no probability is exactly zero", "orange")
    b.card(45, 440, 330, 140, "Message: free money meeting", ["'meeting' never seen in spam", "'free' never seen in ham", "so every product is 0"], "orange", size=12)
    b.card(400, 440, 360, 66, "alpha = 0", ["spam 0, ham 0  ->  0 / 0, no answer"], "red", size=12)
    b.card(400, 516, 360, 66, "alpha = 1", ["spam 6.150e-05, ham 7.688e-06", "P(spam) = 0.8889"], "green", size=12)
    b.card(785, 440, 410, 140, "Add alpha to every count", ["P(w | c) = (count + alpha) /", "(total + alpha × vocabulary)", "here 26 words per class,", "vocabulary of 32"], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    names_out = {
        "knn_lazy_vote": "instance-based-learning-lazy-vote",
        "knn_k_scale_dimensions": "instance-based-learning-k-scale-dimensions",
        "svm_margin": "support-vector-machines-margin",
        "svm_kernel_lift": "support-vector-machines-kernel-lift",
        "bayes_base_rate": "bayesian-learning-base-rate",
        "bayes_naive_bayes": "bayesian-learning-naive-bayes",
    }
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{names_out[name]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
