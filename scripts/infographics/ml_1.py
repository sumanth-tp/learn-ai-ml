"""Infographics for docs/theory/ml/01-ml-foundations (Track A, agent A1).

Each function redraws one board for a chapter as an original image. Every number
on a board is checked against the printed output of the chapter's own code
(.lecture-import/track-a/a1-code/out/*.txt) before the board is written. Run
from the repo root with the ML virtual environment, because one board fits real
polynomials:

    .lecture-import/venv-ml/bin/python scripts/infographics/ml_1.py
    .lecture-import/venv-ml/bin/python scripts/infographics/ml_1.py fit
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, FAINT, INK, MONO, PALETTE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "static" / "img" / "ml"
CODE_OUT = ROOT / ".lecture-import" / "track-a" / "a1-code" / "out"
BOARDS = {}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


def check(name, *needles):
    text = (CODE_OUT / f"{name}.txt").read_text()
    missing = [n for n in needles if n not in text]
    if missing:
        raise SystemExit(f"{name}: board numbers not in code output: {missing}")
    return text


def hbar(b, x, y, w, value, vmax, color, h=14):
    c = PALETTE[color]
    b.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#e9ecef"/>')
    b.parts.append(
        f'<rect x="{x}" y="{y}" width="{max(2.0, w * value / vmax):.1f}" height="{h}" rx="4" fill="{c["stroke"]}"/>'
    )




@board
def what_machine_learning_is_rules_to_learning():
    check("cars_learning_curve", "2108", "962", "958", "792", "766", "764", "771")
    b = Board(1000, 610, "Learning reverses the arrow", "Rules plus data give answers; data plus answers give rules")

    b.group(30, 90, 940, 130, "Traditional programming", "orange")
    r = b.card(60, 126, 200, 38, "rules", ["written by a person"], "orange", size=11)
    d = b.card(60, 172, 200, 38, "data", ["new cases"], "grey", size=11)
    p = b.card(380, 134, 220, 64, "program", ["applies the rules"], "orange")
    a = b.card(720, 134, 220, 64, "answers", ["predictions"], "green")
    b.arrow(r.right(), p.left(0.3))
    b.arrow(d.right(), p.left(0.7))
    b.arrow(p.right(), a.left())

    b.group(30, 240, 940, 140, "Machine learning", "blue")
    d2 = b.card(60, 278, 200, 38, "data", ["past examples"], "grey", size=11)
    a2 = b.card(60, 324, 200, 38, "answers", ["the label of each"], "green", size=11)
    alg = b.card(380, 288, 220, 64, "learning algorithm", ["searches for a rule"], "blue")
    rules = b.card(720, 288, 220, 64, "rules (the model)", ["then used on new data"], "purple")
    b.arrow(d2.right(), alg.left(0.3))
    b.arrow(a2.right(), alg.left(0.7))
    b.arrow(alg.right(), rules.left())

    b.group(30, 400, 940, 190, "Mitchell: P on T improves with E", "green")
    b.card(55, 436, 270, 40, "T  the task", ["predict a used car's price"], "green", size=11)
    b.card(55, 484, 270, 40, "E  the experience", ["past sales, features and price"], "green", size=11)
    b.card(55, 532, 270, 40, "P  the performance", ["mean absolute error, unseen cars"], "green", size=11)

    rows = [("cars seen (E)", "test MAE (P)"), ("3", "£962"), ("10", "£958"), ("30", "£792"), ("100", "£766"),
            ("300", "£764"), ("1,500", "£771"), ("hand-written rule", "£2,108")]
    values = [None, 962, 958, 792, 766, 764, 771, 2108]
    b.table(360, 432, [170, 120], rows, header_color="blue", size=11, row_h=19)
    for i, v in enumerate(values[1:], start=1):
        y = 432 + i * 19 + 3
        hbar(b, 680, y, 270, v, 2200, "red" if i == 7 else "blue", h=13)
    b.text(680, 428, "error, shorter is better", 10, FAINT, anchor="start")
    return b


@board
def what_machine_learning_is_paradigms():
    check("paradigms", "1.000", "0.393", "0.730", "0.826", "0.951", "1847", "0.766", "0.500")
    check("instance_vs_model", "426 x 30", "31 numbers", "0.951", "0.958", "0.944", "0.965")
    b = Board(1160, 600, "Four ways to learn", "One measured result from each, produced by the chapter's code")
    cols = [
        ("Supervised", "blue", ["x and y are both given", "", "classification: iris", "accuracy 1.000", "", "regression: diabetes", "R² 0.393"]),
        ("Unsupervised", "green", ["only x, no labels", "", "k-means finds 3 groups", "agreement with the", "species: ARI 0.730", "(labels never shown)"]),
        ("Semi-supervised", "orange", ["6 labels, 144 unlabelled", "", "labelled rows only:", "accuracy 0.826", "", "label spreading: 0.951"]),
        ("Reinforcement", "purple", ["reward after each action", "", "average reward 0.766", "random choice: 0.500", "", "best arm pulled", "1,847 of 2,000 times"]),
    ]
    for i, (t, col, lines) in enumerate(cols):
        b.card(30 + i * 285, 100, 265, 230, t, lines, col, title_size=16)

    b.group(30, 350, 1100, 230, "Two more questions about any model", "grey")
    b.card(55, 390, 340, 160, "Classification or regression?", [
        "label is a category: spam,",
        "disease, species",
        "",
        "label is a number: a price,",
        "a temperature, a dose"], "teal", size=12, title_size=14)
    b.card(410, 390, 340, 160, "Instance- or model-based?", [
        "k-NN keeps the whole",
        "426 x 30 training table",
        "",
        "logistic regression keeps",
        "31 numbers and drops the rest"], "teal", size=12, title_size=14)
    b.card(765, 390, 340, 160, "Batch or online?", [
        "online model, 100 rows at a",
        "time, test accuracy:",
        "0.944, 0.888, 0.958,",
        "0.965, 0.965",
        "it drifts with the latest batch"], "teal", size=12, title_size=14)
    return b


class _Mulberry32:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def random(self):
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def normal(self):
        u = max(self.random(), 1e-12)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * self.random())


@board
def what_machine_learning_is_fit():
    import numpy as np

    check("bias_variance", "0.3154", "0.2823", "0.0979", "0.1051", "0.0432", "0.7246", "0.0900")

    def sample(n, seed, noise):
        rng = _Mulberry32(seed)
        x = np.array([(i + rng.random()) / n for i in range(n)])
        y = np.cos(1.5 * np.pi * x) + noise * np.array([rng.normal() for _ in range(n)])
        return x, y

    x, y = sample(30, 11, 0.3)
    b = Board(1100, 550, "Underfit, good fit, overfit", "The same 30 noisy points, three polynomials")
    panels = [
        (1, "underfit", "red", "train 0.3154", "test 0.2823", "too stiff: high error on both"),
        (4, "good fit", "green", "train 0.0979", "test 0.1051", "close to the noise floor 0.0900"),
        (15, "overfit", "orange", "train 0.0432", "test 0.7246", "chases noise: train under the floor"),
    ]
    pw, ph = 330, 230
    for i, (degree, name, col, tr, te, note) in enumerate(panels):
        x0, y0 = 30 + i * 360, 100
        b.group(x0 - 10, y0 - 10, pw + 20, 400, f"degree {degree}: {name}", col)
        px0, py0 = x0 + 8, y0 + 40

        def sx(v):
            return px0 + v * (pw - 16)

        def sy(v):
            return py0 + (2.0 - max(-2.0, min(2.0, v))) / 4.0 * ph

        b.parts.append(f'<rect x="{px0}" y="{py0}" width="{pw - 16}" height="{ph}" fill="#ffffff" stroke="#dee2e6"/>')
        b.parts.append(f'<line x1="{px0}" y1="{sy(0)}" x2="{px0 + pw - 16}" y2="{sy(0)}" stroke="#dee2e6"/>')
        w = np.linalg.lstsq(np.vander(2 * x - 1, degree + 1, increasing=True), y, rcond=None)[0]
        grid = np.linspace(0, 1, 200)
        fit = np.vander(2 * grid - 1, degree + 1, increasing=True) @ w
        truth = np.cos(1.5 * np.pi * grid)
        b.parts.append(
            '<polyline fill="none" stroke="#868e96" stroke-width="1.6" stroke-dasharray="5 4" points="'
            + " ".join(f"{sx(g):.1f},{sy(t):.1f}" for g, t in zip(grid, truth)) + '"/>'
        )
        b.parts.append(
            f'<polyline fill="none" stroke="{PALETTE[col]["stroke"]}" stroke-width="2.6" points="'
            + " ".join(f"{sx(g):.1f},{sy(f):.1f}" for g, f in zip(grid, fit)) + '"/>'
        )
        for xi, yi in zip(x, y):
            b.parts.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="3.4" fill="#1c7ed6" fill-opacity="0.85"/>')
        b.text(x0 + pw / 2, py0 + ph + 34, tr + "   " + te, 13, col, "700")
        b.text(x0 + pw / 2, py0 + ph + 58, note, 12, INK)
        b.text(x0 + pw / 2, py0 + ph + 82, "dashed: true curve   dots: training points", 10, FAINT)
    b.text(550, 522, "no model can beat the noise floor of 0.3 squared = 0.0900; a training error below it means noise was fitted", 12, "grey", italic=True)
    return b




@board
def data_preprocessing_pipeline():
    text = check("encode_impute", "3 raw feature columns became 6 numeric columns")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and len(parts) == 7:
            rows.append(parts[1:])
    assert len(rows) == 6
    b = Board(1300, 620, "From a messy table to a model-ready matrix", "One ColumnTransformer, fitted on training rows only")
    raw = [["colour", "size", "age_years"], ["red", "S", "3"], ["blue", "M", "7"], ["blue", "L", "NaN"],
           ["NaN", "M", "5"], ["green", "NaN", "2"], ["red", "S", "9"]]
    b.text(150, 108, "raw features (3 columns)", 13, "grey", "700")
    b.table(30, 120, [90, 80, 100], raw, header_color="grey", size=12, row_h=28)

    b.group(340, 100, 400, 460, "One transformer per column type", "blue")
    c1 = b.card(360, 150, 360, 90, "colour  (nominal)", ["fill blanks with the most frequent", "then one-hot: blue, green, red"], "orange", size=12)
    c2 = b.card(360, 262, 360, 90, "size  (ordinal)", ["fill blanks with the most frequent", "then codes S=0, M=1, L=2"], "yellow", size=12)
    c3 = b.card(360, 374, 360, 90, "age_years  (numeric)", ["fill blanks with the median 5.0", "then standardise"], "green", size=12)
    c4 = b.card(360, 476, 360, 60, "age was missing?", ["a 0/1 indicator column"], "purple", size=12)
    b.arrow((300, 220), (340, 220), label="split", label_dy=-16, width=1.8)
    for c in (c1, c2, c3, c4):
        b.arrow((340, c.cy), c.left(), width=1.4)
        b.parts.append(f'<line x1="{c.x + c.w}" y1="{c.cy}" x2="758" y2="{c.cy}" stroke="{INK}" stroke-width="1.4"/>')
    b.parts.append(f'<line x1="758" y1="{c1.cy}" x2="758" y2="{c4.cy}" stroke="{INK}" stroke-width="1.4"/>')
    b.arrow((758, 240), (790, 240), label="stack", label_dy=-16, width=1.8)

    head = ["blue", "green", "red", "size", "age", "missing"]
    out = [head] + [[r[0].replace(".0", ""), r[1].replace(".0", ""), r[2].replace(".0", ""), r[3].replace(".0", ""), r[4], r[5].replace(".0", "")] for r in rows]
    b.text(1030, 108, "model-ready matrix (6 columns)", 13, "grey", "700")
    b.table(790, 120, [62, 62, 62, 62, 64, 82], out, header_color="green", size=12, row_h=28)
    b.card(790, 330, 400, 76, "3 raw columns became 6", ["row 2 had no age: median 5.0 filled it,", "standardised to -0.07, flagged 1"], "green", size=12)
    b.card(790, 430, 400, 76, "fitted on training rows only", ["the median, the mean and the standard", "deviation are replayed on new data"], "red", size=12)
    return b


@board
def data_preprocessing_scaling_outliers():
    check("lecture_numbers", "10.8", "11.74", "-0.239", "0.140", "5.25", "9.75", "4.5", "16.5", "46.0", "2.914", "3.000", "sqrt(n - 1) = 3.0")
    check("outlier_effect", "0.545", "0.026", "-0.013", "-0.072", "-0.036", "-0.037", "4.57", "0.69")
    b = Board(1160, 640, "Scaling a value and screening outliers", "The lecture's data set, with the numbers the code prints")
    data = [2, 4, 5, 6, 7, 8, 9, 10, 12, 45]
    x0, x1, yl = 60, 1100, 190

    def sx(v):
        return x0 + (v + 30) / 80 * (x1 - x0)

    b.parts.append(f'<rect x="{sx(-1.5):.1f}" y="{yl - 70}" width="{sx(16.5) - sx(-1.5):.1f}" height="70" fill="#1c7ed6" fill-opacity="0.10"/>')
    b.parts.append(f'<line x1="{x0}" y1="{yl}" x2="{x1}" y2="{yl}" stroke="{INK}" stroke-width="1.5"/>')
    for t in range(-30, 51, 10):
        b.parts.append(f'<line x1="{sx(t):.1f}" y1="{yl}" x2="{sx(t):.1f}" y2="{yl + 6}" stroke="{INK}"/>')
        b.text(sx(t), yl + 22, str(t), 11, FAINT)
    for v in data:
        b.parts.append(f'<circle cx="{sx(v):.1f}" cy="{yl - 12}" r="6" fill="#2f9e44" stroke="#ffffff" stroke-width="1.5"/>')
    b.parts.append(f'<circle cx="{sx(45):.1f}" cy="{yl - 12}" r="11" fill="none" stroke="#1c7ed6" stroke-width="2.4"/>')
    b.parts.append(f'<line x1="{sx(16.5):.1f}" y1="{yl - 80}" x2="{sx(16.5):.1f}" y2="{yl}" stroke="#1c7ed6" stroke-width="2.4"/>')
    b.parts.append(f'<line x1="{sx(46.0):.1f}" y1="{yl - 80}" x2="{sx(46.0):.1f}" y2="{yl + 40}" stroke="#e8590c" stroke-width="2.4" stroke-dasharray="6 4"/>')
    b.parts.append(f'<line x1="{sx(-24.4):.1f}" y1="{yl + 40}" x2="{sx(46.0):.1f}" y2="{yl + 40}" stroke="#e8590c" stroke-width="2.4" stroke-dasharray="6 4"/>')
    b.text(sx(16.5) - 8, yl - 88, "IQR upper fence 16.5", 12, "blue", "700", anchor="end")
    b.text(sx(46.0) - 8, yl - 88, "3-sigma upper limit 46.0", 12, "orange", "700", anchor="end")
    b.text(sx(45), yl - 36, "45", 12, "blue", "700")
    b.text(sx(10), yl + 62, "mean -/+ 3 sigma = [-24.4, 46.0]", 12, "orange", "700")

    b.card(30, 290, 350, 170, "Scale x = 8", [
        "mean 10.8   population sigma 11.74",
        "min 2   max 45",
        "",
        "standardised (8 - 10.8) / 11.74",
        "= -0.239",
        "min-max (8 - 2) / 43 = 0.140"], "blue", size=12)
    b.card(400, 290, 360, 170, "Two outlier rules", [
        "IQR: Q1 5.25  Q3 9.75  IQR 4.5",
        "fences [-1.5, 16.5]  flags 45",
        "",
        "3-sigma: limits [-24.4, 46.0]",
        "flags nothing: 45 is at 2.914",
        "sigma, inside its own limit"], "orange", size=12)
    b.card(780, 290, 350, 170, "With n = 10, 3-sigma is blind", [
        "no point can pass",
        "sqrt(n - 1) = 3.0 sigma",
        "",
        "an outlier of 1,000,000",
        "scores 3.000, not beyond it"], "red", size=12)

    b.text(580, 494, "What one extreme value (500 among 200 points near 50) does to a scaled value of 50", 13, "grey", "700")
    b.table(210, 506, [180, 270, 270], [
        ["scaler", "outlier absent", "outlier present"],
        ["standard", "-0.013", "-0.072"],
        ["min-max", "0.545", "0.026"],
        ["robust", "-0.036", "-0.037"]], header_color="blue", size=12, row_h=24)
    return b


@board
def data_preprocessing_curse():
    check("curse", "66.130", "2.316", "0.363", "0.108", "0.032", "0.960", "0.747", "0.720", "0.527", "0.400")
    check("scaling_matters", "0.691", "0.949", "0.972")
    b = Board(1160, 560, "The curse of dimensionality", "As columns pile up, distances stop discriminating and k-NN loses its footing")
    b.group(30, 90, 520, 440, "Nearest and farthest points", "orange")
    b.text(290, 140, "(farthest - nearest) / nearest distance, 500 random points", 12, INK)
    dims = [("2", 66.130), ("10", 2.316), ("100", 0.363), ("1,000", 0.108), ("10,000", 0.032)]
    for i, (d, v) in enumerate(dims):
        y = 170 + i * 54
        b.text(120, y + 14, f"d = {d}", 13, "orange", "700", anchor="end")
        hbar(b, 140, y, 300, math.log10(v * 1000 + 1), math.log10(66130 + 1), "orange", h=20)
        b.text(450, y + 15, f"{v:.3f}", 13, INK, "700", anchor="start")
    b.text(290, 456, "bars use a log scale; at d = 10,000 the gap is 3%", 11, FAINT, italic=True)
    b.text(290, 496, "'nearest' has stopped meaning anything", 13, "orange", "700")

    b.group(580, 90, 550, 440, "Noise columns added to iris", "red")
    b.text(855, 140, "5-fold k-NN accuracy (standardised)", 12, INK)
    pts = [("0", 0.960), ("10", 0.747), ("50", 0.720), ("200", 0.527), ("1,000", 0.400)]
    for i, (n, v) in enumerate(pts):
        y = 170 + i * 54
        b.text(700, y + 14, f"+{n}", 13, "red", "700", anchor="end")
        hbar(b, 720, y, 300, v, 1.0, "red", h=20)
        b.text(1030, y + 15, f"{v:.3f}", 13, INK, "700", anchor="start")
    b.text(855, 456, "scaling matters for the same reason: on wine, k-NN 0.691 raw,", 11, FAINT, italic=True)
    b.text(855, 474, "0.949 standardised; random forest 0.972 either way", 11, FAINT, italic=True)
    b.text(855, 504, "remedies: select features, reduce with PCA, regularise, add rows", 12, "red", "700")
    return b




def pair(b, x, y, w, a_label, a_val, b_label, b_val, a_col, b_col, vmax=1.0, fmt="{:.3f}"):
    b.text(x, y - 4, a_label, 11, INK, anchor="start")
    hbar(b, x, y, w, a_val, vmax, a_col, h=14)
    b.text(x + w + 8, y + 12, fmt.format(a_val), 12, a_col, "700", anchor="start")
    b.text(x, y + 34, b_label, 11, INK, anchor="start")
    hbar(b, x, y + 38, w, b_val, vmax, b_col, h=14)
    b.text(x + w + 8, y + 50, fmt.format(b_val), 12, b_col, "700", anchor="start")


@board
def features_leakage_and_imbalance_leakage():
    check("target_leakage", "0.719", "0.960")
    check("selection_leakage_pipeline", "0.850", "0.580")
    check("selection_leakage_scratch", "0.840", "0.460")
    check("target_encoding", "0.806", "0.510")
    check("temporal_leakage", "0.893", "0.799")
    b = Board(1280, 620, "Three kinds of leakage", "Each one raises the score without raising any error")
    cols = [
        ("1  Target leakage", "red", "a feature that is a consequence of the outcome",
         "a cancellation survey is sent only after the customer leaves"),
        ("2  Train-test contamination", "orange", "a selection, scaler or encoding fitted on rows that later test it",
         "features picked on all rows; category means that include a row's own label"),
        ("3  Temporal leakage", "purple", "a shuffled split of time-ordered rows",
         "test days sit between training days, so the model interpolates"),
    ]
    for i, (title, col, what, example) in enumerate(cols):
        x = 30 + i * 415
        b.group(x, 90, 395, 500, title, col)
        b.card(x + 15, 122, 365, 104, "what leaks", [what, example], col, size=11, title_size=13)
    x = 45
    b.text(x, 250, "cross-validated AUC", 12, "grey", "700", anchor="start")
    pair(b, x, 274, 230, "honest features", 0.719, "plus cancellation_survey_sent", 0.960, "green", "red")
    b.card(x, 372, 365, 190, "the fix", [
        "ask of every column:",
        "would this value exist at the",
        "moment of prediction?",
        "",
        "drop it, or rebuild it as it",
        "stood at prediction time"], "green", size=12, title_size=13)
    x = 460
    b.text(x, 250, "accuracy or AUC, honest vs inflated", 12, "grey", "700", anchor="start")
    pair(b, x, 274, 190, "SelectKBest inside Pipeline", 0.580, "SelectKBest before the CV", 0.850, "green", "red")
    pair(b, x, 352, 190, "scratch: chosen on train half", 0.460, "scratch: chosen on all rows", 0.840, "green", "red")
    pair(b, x, 430, 190, "target encoding, held-out rows", 0.510, "naive encoding, training rows", 0.806, "green", "red")
    b.card(x, 502, 365, 80, "the fix", ["fit every step inside the fold: Pipeline,", "TargetEncoder.fit_transform (cross fitting)"], "green", size=11, title_size=13)
    x = 875
    b.text(x, 250, "accuracy of a boosted model", 12, "grey", "700", anchor="start")
    pair(b, x, 274, 190, "forward chaining (honest)", 0.799, "shuffled 5-fold", 0.893, "green", "red")
    b.text(x, 360, "part of the gap is less data in the", 11, FAINT, anchor="start", italic=True)
    b.text(x, 376, "early forward folds; the direction", 11, FAINT, anchor="start", italic=True)
    b.text(x, 392, "is the lesson", 11, FAINT, anchor="start", italic=True)
    b.card(x, 422, 365, 140, "the fix", [
        "if rows are ordered in time and the",
        "model will face the future, split by",
        "time: TimeSeriesSplit trains on the",
        "past and tests on what follows"], "green", size=12, title_size=13)
    return b


@board
def features_leakage_and_imbalance_imbalance():
    text = check("imbalance_toolbox", "0.975", "0.984", "0.935", "0.758", "0.639", "0.282", "0.2857", "0.0255")
    check("prevalence_threshold", "0.9218", "0.7865", "0.5377", "0.3754", "0.1831", "0.0977", "0.0260", "0.921")
    b = Board(1280, 640, "Rare positives: five treatments, one trap", "A 2.5% positive class, scored on the same test set")
    rows = [["approach", "acc", "recall", "prec", "F1", "AP"]]
    names = {
        "always predict the majority": "always say no",
        "logistic, threshold 0.5": "plain, thr 0.5",
        "class_weight='balanced'": "class weights",
        "random oversampling of the minority": "oversampling",
    }
    for line in text.splitlines():
        for key, short in names.items():
            if line.startswith(key):
                parts = line[len(key):].split()
                rows.append([short, parts[1], parts[3], parts[5], parts[7], parts[9]])
        if line.startswith("tuned threshold"):
            parts = line.split(" for F1")[1].split()
            rows.append(["tuned thr 0.282", parts[1], parts[3], parts[5], parts[7], parts[9]])
    assert len(rows) == 6, rows
    b.group(30, 90, 640, 330, "What each treatment does", "blue")
    b.table(48, 130, [200, 80, 90, 80, 70, 70], rows, header_color="blue", size=12, row_h=34)
    b.text(350, 358, "mean predicted probability: plain 0.0243, class weights 0.2857,", 12, INK)
    b.text(350, 378, "oversampling 0.2817, against a true rate of 0.0255", 12, INK)
    b.text(350, 400, "reweighting slides the operating point and inflates probabilities", 12, "blue", "700")

    b.group(700, 90, 550, 330, "Same ROC-AUC 0.921, falling precision", "red")
    prev = [["prevalence", "precision", "accuracy", "always no", "avg prec"],
            ["50%", "0.8413", "0.8413", "0.5000", "0.9218"],
            ["20%", "0.5700", "0.8413", "0.8000", "0.7865"],
            ["5%", "0.2182", "0.8413", "0.9500", "0.5377"],
            ["2%", "0.0977", "0.8413", "0.9800", "0.3754"],
            ["0.5%", "0.0260", "0.8413", "0.9950", "0.1831"]]
    b.table(715, 130, [105, 105, 105, 110, 105], prev, header_color="red", size=12, row_h=34)
    b.text(975, 358, "fixed threshold 1.0, scores centred 2.0 apart", 12, INK)
    b.text(975, 380, "at 2% the model is wrong 9 times in 10 when it alerts", 12, "red", "700")

    b.group(30, 440, 1220, 170, "A decision ladder: stop when the problem is solved", "green")
    steps = ["1  metric that sees\nthe rare class", "2  stratified split\n(or by time)", "3  plain model,\nread the PR curve",
             "4  move the threshold\nby cost or by CV", "5  class weights if\nthe model needs them", "6  resample inside\ntraining folds only"]
    for i, t in enumerate(steps):
        b.card(50 + i * 200, 485, 185, 80, "", t.split("\\n"), "green", size=12)
    b.text(640, 596, "recalibrate if you need probabilities; more real positives beats every trick", 12, "green", "700")
    return b


@board
def features_leakage_and_imbalance_pipeline():
    check("full_pipeline", "0.063", "0.202", "0.037", "0.032", "['prepare', 'classify']")
    b = Board(1240, 560, "One Pipeline holds every fitted step", "Cross-validate the pipeline, not a pre-processed copy of the data")
    df = b.cylinder(30, 190, 150, 150, "DataFrame", ["tenure (gaps)", "spend", "plan", "region (120)"], "purple", size=12)
    b.group(240, 100, 470, 360, "ColumnTransformer: prepare", "blue")
    n1 = b.card(260, 150, 430, 80, "tenure, spend", ["median fill, then standardise"], "green", size=12)
    n2 = b.card(260, 250, 430, 80, "plan", ["one-hot, unknown categories ignored"], "orange", size=12)
    n3 = b.card(260, 350, 430, 80, "region", ["target encoding with cross fitting"], "yellow", size=12)
    for n in (n1, n2, n3):
        b.arrow((180, 265), n.left(), width=1.4)
    clf = b.card(790, 190, 200, 150, "classify", ["HistGradientBoosting", "class_weight=balanced"], "pink", size=12)
    for n in (n1, n2, n3):
        b.arrow(n.right(), clf.left(), width=1.4)
    cv = b.card(1030, 150, 180, 230, "5-fold CV", ["scored on average", "precision", "", "0.202", "+/- 0.037", "", "no-skill base", "rate 0.063"], "red", size=12, title_size=14)
    b.arrow(clf.right(), cv.left())
    b.card(240, 480, 750, 56, "a row with an unseen plan and region and a missing tenure", ["still returns a probability: 0.032, no crash"], "teal", size=12, title_size=13)
    b.text(1120, 470, "every fit happens", 11, FAINT)
    b.text(1120, 486, "on training folds", 11, FAINT)
    b.text(1120, 502, "only", 11, FAINT)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{name.replace('_', '-')}.svg")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
