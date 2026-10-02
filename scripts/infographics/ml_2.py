import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ml"
BOARDS = {}
REG = "regression-and-gradient-descent"
CLS = "classification-and-logistic-regression"
TREE = "decision-trees"


def board(name):
    def deco(fn):
        BOARDS[name] = fn
        return fn
    return deco


def poly(b, pts, color="blue", width=2.2, dash=False, opacity=1.0):
    col = PALETTE[color]["stroke"] if color in PALETTE else color
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    extra = ' stroke-dasharray="6 4"' if dash else ""
    b.parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}" '
                   f'stroke-linejoin="round" stroke-linecap="round" stroke-opacity="{opacity}"{extra}/>')


def dot(b, x, y, r=4.5, color="orange", ring=False):
    col = PALETTE[color]["stroke"] if color in PALETTE else color
    if ring:
        b.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    else:
        b.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" stroke="#fffdf7" stroke-width="1.2"/>')


def diamond_mark(b, x, y, r=4.6, color="orange"):
    col = PALETTE[color]["stroke"]
    b.parts.append(f'<path d="M{x:.1f},{y - r:.1f} L{x + r:.1f},{y:.1f} L{x:.1f},{y + r:.1f} L{x - r:.1f},{y:.1f} z" '
                   f'fill="{col}" stroke="#fffdf7" stroke-width="1"/>')


def line(b, x1, y1, x2, y2, color="#868e96", width=1.2, dash=False):
    extra = ' stroke-dasharray="4 4"' if dash else ""
    b.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                   f'stroke-width="{width}"{extra}/>')


def frame(b, x, y, w, h, xr, yr, xticks, yticks, xlabel="", ylabel="", fmt_x="{:g}", fmt_y="{:g}"):
    line(b, x, y + h, x + w, y + h, "#495057", 1.4)
    line(b, x, y, x, y + h, "#495057", 1.4)
    mx = lambda v: x + (v - xr[0]) / (xr[1] - xr[0]) * w
    my = lambda v: y + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    for t in xticks:
        line(b, mx(t), y + h, mx(t), y + h + 5, "#495057", 1.2)
        b.text(mx(t), y + h + 19, fmt_x.format(t), 11, "#868e96")
    for t in yticks:
        line(b, x - 5, my(t), x, my(t), "#495057", 1.2)
        b.text(x - 9, my(t) + 4, fmt_y.format(t), 11, "#868e96", anchor="end")
    if xlabel:
        b.text(x + w / 2, y + h + 38, xlabel, 12, "grey", "700")
    if ylabel:
        b.parts.append(f'<text xml:space="preserve" transform="translate({x - 38},{y + h / 2}) rotate(-90)" '
                       f'text-anchor="middle" font-family="monospace" font-size="12" font-weight="700" '
                       f'fill="{PALETTE["grey"]["text"]}">{ylabel}</text>')
    return mx, my


X1 = np.array([1.0, 2.0, 3.0])
Y1 = np.array([1.0, 2.0, 2.0])


def cost1(t):
    return float(np.sum((t * X1 - Y1) ** 2) / (2 * len(X1)))


def grad1(t):
    return float(np.sum((t * X1 - Y1) * X1) / len(X1))


def trail1(alpha, steps, start=0.0):
    out = [start]
    t = start
    for _ in range(steps):
        t -= alpha * grad1(t)
        out.append(t)
    return out


def problem2(Z, y):
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    m = len(Z)
    H = Zc.T @ Zc / m
    bvec = Zc.T @ yc / m
    best = np.linalg.solve(H, bvec)
    w, V = np.linalg.eigh(H)
    return H, bvec, best, w, V


AREA_AGE = np.array([[60, 12], [75, 3], [90, 18], [105, 6], [120, 15], [135, 2], [150, 9], [165, 20]], dtype=float)
PRICE = np.array([180, 260, 235, 310, 300, 395, 380, 390], dtype=float)


def steps_to_close(H, bvec, best, alpha, tol=1e-3):
    gap = lambda t: 0.5 * (t - best) @ H @ (t - best)
    t = np.zeros(2)
    g0 = gap(t)
    n = 0
    while gap(t) > tol * g0:
        t = t - alpha * (H @ t - bvec)
        n += 1
    return n




@board(f"{REG}-two-routes")
def two_routes():
    b = Board(1120, 600, "Two routes to the best line", "Fit y-hat = theta * x to (1,1) (2,2) (3,2): solve it exactly, or walk downhill")
    b.group(28, 92, 470, 480, "Route 1: solve it exactly", "blue")
    b.card(55, 138, 416, None, "theta = (X'X)^-1 X'y", ["the normal equation, one formula"], "blue", title_size=17)
    b.card(55, 232, 416, None, "What you give up", [
        "no learning rate and no iterations",
        "the inverse costs O(d^3) in the number of features",
        "wide data makes it slow; a duplicated column makes X'X singular"], "orange", align="left", bullets=True)
    b.card(55, 368, 416, None, "Checked on 442 diabetes rows", [
        "inverse, solve and lstsq all land within 1.5e-10 of scikit-learn's coefficients",
        "with a duplicated column, solve raises 'Singular matrix' but lstsq still answers"], "teal", align="left", bullets=True)
    b.text(263, 538, "scikit-learn's LinearRegression uses an SVD-based\nleast-squares solver, not the explicit inverse", 12, "blue", italic=True)

    b.group(530, 92, 562, 480, "Route 2: walk downhill", "green")
    mx, my = frame(b, 585, 148, 460, 230, (-0.2, 1.4), (0, 1.8), [0, 0.5, 1.0], [0, 0.5, 1.0, 1.5], "theta", "cost J", "{:g}", "{:g}")
    xs = np.linspace(-0.2, 1.4, 120)
    poly(b, [(mx(t), my(cost1(t))) for t in xs if cost1(t) <= 1.8], "blue", 2.6)
    best = float(X1 @ Y1 / (X1 @ X1))
    line(b, mx(best), 148, mx(best), 378, PALETTE["green"]["stroke"], 1.4, True)
    tr = trail1(0.1, 10)
    marks = [0, 1, 2, 5, 10]
    pts = [(mx(tr[k]), my(cost1(tr[k]))) for k in range(0, 11)]
    poly(b, pts, "orange", 1.4)
    for k in marks:
        dot(b, mx(tr[k]), my(cost1(tr[k])), 5 if k else 5.5, "orange")
    b.text(mx(0.72), my(1.62), "step : theta", 11, "orange", "700", anchor="start")
    for row, k in enumerate([0, 1, 2, 5, 10]):
        b.text(mx(0.72), my(1.62) + 18 + row * 17, f"{k:>4} : {tr[k]:.4f}", 11, "orange", "700", anchor="start")
        b.text(mx(tr[k]) + (-10 if k in (0, 5) else 7 if k < 10 else 11), my(cost1(tr[k])) - (8 if k < 5 else 11), str(k), 11, "orange", "700", anchor="end" if k in (0, 5) else "start")
    b.card(560, 420, 512, None, "theta <- theta - alpha * dJ/dtheta", [
        f"theta = 0, alpha = 0.1:  gradient = {grad1(0.0):.3f}",
        f"theta after step 1 = {tr[1]:.3f}    J: {cost1(0.0):.3f} -> {cost1(tr[1]):.3f}"], "green", title_size=15)
    b.card(560, 518, 512, 40, f"Both routes meet at theta = {best:.4f}, J = {cost1(best):.4f}", [], "yellow", title_size=14)
    return b


@board(f"{REG}-rate-and-scaling")
def rate_and_scaling():
    b = Board(1240, 640, "Learning rate and feature scaling", "Why gradient descent crawls, overshoots or flies away, and how scaling rounds the bowl")
    b.group(24, 92, 640, 520, "Learning rate alpha", "orange")
    curv = float(np.sum(X1 * X1) / len(X1))
    runs = [(0.01, "crawls", "yellow"), (0.10, "converges", "green"), (0.45, "diverges", "red")]
    for i, (alpha, verdict, col) in enumerate(runs):
        x0 = 40 + i * 205
        tr = trail1(alpha, 20)
        mx, my = frame(b, x0 + 30, 168, 150, 130, (-1, 3), (0, 8), [0, 1, 2], [0, 4, 8], "theta", "", "{:g}", "{:g}")
        xs = np.linspace(-1, 3, 100)
        poly(b, [(mx(t), my(min(cost1(t), 8))) for t in xs if cost1(t) <= 8], "blue", 2.2)
        pts = [(mx(t), my(cost1(t))) for t in tr if -1 <= t <= 3 and cost1(t) <= 8]
        if len(pts) > 1:
            poly(b, pts, PALETTE[col]["stroke"], 1.3)
        for p in pts[:12]:
            dot(b, p[0], p[1], 2.8, col)
        b.text(x0 + 105, 154, f"alpha = {alpha:.2f}", 14, col, "700")
        b.card(x0, 348, 190, None, verdict, [f"20 steps -> theta = {tr[-1]:.4f}", f"J = {cost1(tr[-1]):.4f}"], col, size=11, title_size=14)
    b.card(40, 486, 610, None, f"Stable only for alpha < 2 / (sum x^2 / m) = 2 / {curv:.3f} = {2 / curv:.4f}", [
        f"alpha = 0.10 reaches within 1e-6 of the minimum cost in 12 steps",
        "alpha = 0.40 is inside the limit but overshoots each time before settling"], "purple", size=11, title_size=13)

    b.group(690, 92, 526, 520, "Feature scaling rounds the bowl", "teal")
    for i, (name, Z, col) in enumerate([("raw (m2, age)", AREA_AGE, "orange"),
                                        ("standardised", (AREA_AGE - AREA_AGE.mean(0)) / AREA_AGE.std(0), "green")]):
        H, bvec, best, w, V = problem2(Z, PRICE)
        alpha = 1 / w[-1]
        n = steps_to_close(H, bvec, best, alpha)
        gap0 = 0.5 * best @ H @ best
        hx = math.sqrt(2 * gap0 * np.linalg.inv(H)[0, 0])
        hy = math.sqrt(2 * gap0 * np.linalg.inv(H)[1, 1])
        x0, y0, size = 712 + i * 252, 168, 220
        sc = min(size / (2.2 * hx), size / (2.2 * hy))
        cx = lambda t: x0 + size / 2 + (t - best[0]) * sc
        cy = lambda t: y0 + size / 2 - (t - best[1]) * sc
        b.parts.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" rx="8" fill="#fffdf7" stroke="#ced4da"/>')
        for lvl in (0.5, 0.2, 0.05, 0.01):
            r = math.sqrt(2 * lvl * gap0)
            ph = np.linspace(0, 2 * math.pi, 100)
            u = r / math.sqrt(w[0]) * np.cos(ph)
            v = r / math.sqrt(w[1]) * np.sin(ph)
            tx = best[0] + u * V[0, 0] + v * V[0, 1]
            ty = best[1] + u * V[1, 0] + v * V[1, 1]
            poly(b, [(cx(a), cy(c)) for a, c in zip(tx, ty)], "teal", 1.4, opacity=0.7)
        t = np.zeros(2)
        path = [t.copy()]
        for _ in range(min(n, 12)):
            t = t - alpha * (H @ t - bvec)
            path.append(t.copy())
        poly(b, [(cx(p[0]), cy(p[1])) for p in path], PALETTE[col]["stroke"], 1.6)
        for p in path:
            dot(b, cx(p[0]), cy(p[1]), 2.8, col)
        dot(b, cx(best[0]), cy(best[1]), 5, "teal")
        b.text(x0 + size / 2, y0 - 8, name, 13, col, "700")
        b.card(x0 - 4, y0 + size + 16, size + 8, None, f"{n} steps", [
            f"condition number {w[-1] / w[0]:.2f}",
            f"largest stable alpha {2 / w[-1]:.4f}",
            f"eigenvalues {w[0]:.2f} and {w[-1]:.2f}"], col, size=11, title_size=16)
    b.text(953, 568, "each run uses alpha = 1 / lambda_max; stops at 99.9% of the gap closed", 11, "teal", italic=True)
    b.text(953, 586, "steps shown are the first 12 of the path", 11, "teal", italic=True)
    return b


@board(f"{REG}-fit-diagnosis")
def fit_diagnosis():
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    rng = np.random.default_rng(0)

    def sample(n):
        x = rng.uniform(0, 1, n)
        return x.reshape(-1, 1), np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)

    Xtr, ytr = sample(30)
    Xte, yte = sample(500)
    configs = [("degree 1", LinearRegression(), 1, "underfit", "yellow"),
               ("degree 4", LinearRegression(), 4, "about right", "green"),
               ("degree 15", LinearRegression(), 15, "overfit", "red")]
    b = Board(1180, 730, "Diagnosing the fit", "Same 30 noisy points of a sine wave, three model sizes; test error is measured on 500 fresh points")
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    for i, (name, est, deg, verdict, col) in enumerate(configs):
        model = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(), est).fit(Xtr, ytr)
        tr = mean_squared_error(ytr, model.predict(Xtr))
        te = mean_squared_error(yte, model.predict(Xte))
        x0 = 60 + i * 370
        b.group(x0 - 36, 92, 350, 350, f"{name}: {verdict}", col)
        mx, my = frame(b, x0, 150, 270, 200, (0, 1), (-1.6, 1.6), [0, 0.5, 1], [-1, 0, 1], "x", "", "{:g}", "{:g}")
        poly(b, [(mx(x), my(np.sin(2 * np.pi * x))) for x in grid[:, 0]], "grey", 1.6, dash=True)
        pred = np.clip(model.predict(grid), -1.6, 1.6)
        poly(b, [(mx(x), my(v)) for x, v in zip(grid[:, 0], pred)], col, 2.6)
        for xv, yv in zip(Xtr[:, 0], ytr):
            dot(b, mx(xv), my(max(min(yv, 1.55), -1.55)), 3, "blue")
        b.text(x0 + 135, 420, f"train MSE {tr:.3f}    test MSE {te:.3f}", 13, col, "700")
    rows = [["model", "train MSE", "test MSE", "reading"]]
    for name, est, deg, reading in [("degree 1", LinearRegression(), 1, "underfit: both high"),
                                    ("degree 4", LinearRegression(), 4, "about right"),
                                    ("degree 15", LinearRegression(), 15, "overfit: train low, test high"),
                                    ("degree 15 + ridge 0.1", Ridge(alpha=0.1), 15, "regularised: gap shrinks"),
                                    ("degree 15 + ridge 1", Ridge(alpha=1.0), 15, "too much: underfits again")]:
        model = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(), est).fit(Xtr, ytr)
        rows.append([name, f"{mean_squared_error(ytr, model.predict(Xtr)):.3f}",
                     f"{mean_squared_error(yte, model.predict(Xte)):.3f}", reading])
    b.table(120, 472, [230, 140, 140, 400], rows, "blue", size=13)
    b.text(590, 712, "noise floor: the added noise has variance 0.09, so no model can beat a test MSE near 0.090", 12, "grey", italic=True)
    return b




def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]

    def draw():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return draw


def lab_data():
    draw = mulberry32(7)
    normal = lambda: math.sqrt(-2 * math.log(1 - draw())) * math.cos(2 * math.pi * draw())
    rows, labels = [], []
    for label, (cx, cy) in enumerate([(-1.0, -0.5), (1.0, 0.5)]):
        for _ in range(40):
            rows.append([cx + 1.1 * normal(), cy + 1.1 * normal()])
            labels.append(label)
    X, y = np.array(rows), np.array(labels)
    A = np.column_stack([np.ones(len(X)), X])
    theta = np.zeros(3)
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(A @ theta)))
        theta -= 0.5 * A.T @ (p - y) / len(y)
    return X, y, theta, 1 / (1 + np.exp(-(A @ theta)))


sigmoid = lambda z: 1 / (1 + math.exp(-z))


@board(f"{CLS}-score-to-class")
def score_to_class():
    X, y, theta, p = lab_data()
    b = Board(1240, 660, "From score to class", "Logistic regression: a linear score, squashed to a probability, then cut at a threshold")
    c1 = b.card(30, 100, 200, 78, "features x", ["x1, x2, ..."], "blue", title_size=15)
    c2 = b.card(290, 100, 230, 78, "score  z = theta . x", ["any real number,", "-inf to +inf"], "purple", title_size=15)
    c3 = b.card(580, 100, 240, 78, "sigmoid  1/(1+e^-z)", ["squashes to 0..1"], "teal", title_size=15)
    c4 = b.card(880, 100, 160, 78, "probability", ["p-hat = P(y=1|x)"], "green", title_size=15)
    d = b.diamond(1148, 139, 160, 100, "p-hat >= 0.5 ?", "yellow")
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    b.arrow(c3.right(), c4.left())
    b.arrow(c4.right(), (1068, 139))
    mx, my = frame(b, 580, 232, 240, 100, (-6, 6), (0, 1), [-6, -3, 0, 3, 6], [0, 0.5, 1], "z", "", "{:g}", "{:g}")
    poly(b, [(mx(z), my(sigmoid(z))) for z in np.linspace(-6, 6, 100)], "teal", 2.4)
    line(b, 580, my(0.5), 820, my(0.5), PALETTE["yellow"]["stroke"], 1.3, True)
    for z, col in [(1.0, "green"), (-0.5, "red")]:
        dot(b, mx(z), my(sigmoid(z)), 5, col)
    rows = [["score z", "p-hat", "class"]]
    for z in (1.0, -0.5, 0.0):
        rows.append([f"{z:+.1f}", f"{sigmoid(z):.3f}", str(int(sigmoid(z) >= 0.5))])
    b.table(40, 232, [110, 110, 90], rows, "purple", size=14)
    b.text(195, 396, "z = 0 is exactly p-hat = 0.5: the boundary", 12, "purple", italic=True)
    b.group(30, 410, 1180, 232, "Learned on 80 points (the lab's data)", "grey")
    x0, y0, w, h = 70, 450, 330, 170
    mx, my = frame(b, x0, y0, w, h, (-4, 4), (-4, 4), [-4, -2, 0, 2, 4], [-4, 0, 4], "feature 1", "", "{:g}", "{:g}")
    thr = 0.5
    level = math.log(thr / (1 - thr))
    ya, yb = (level - theta[0] - theta[1] * -4) / theta[2], (level - theta[0] - theta[1] * 4) / theta[2]
    b.parts.append(f'<clipPath id="scatterclip"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath><g clip-path="url(#scatterclip)">')
    b.parts.append(f'<polygon points="{mx(-4):.1f},{my(ya):.1f} {mx(4):.1f},{my(yb):.1f} {mx(4):.1f},{y0} {mx(-4):.1f},{y0}" '
                   f'fill="{PALETTE["orange"]["fill"]}" fill-opacity="0.8"/>')
    poly(b, [(mx(-4), my(ya)), (mx(4), my(yb))], "purple", 2.4)
    b.parts.append("</g>")
    for (a, c), lab in zip(X, y):
        if lab == 1:
            diamond_mark(b, mx(a), my(c), 4.4, "orange")
        else:
            dot(b, mx(a), my(c), 3.6, "blue")
    pred = p >= 0.5
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    b.card(440, 440, 360, None, "the fitted model", [
        f"theta = ({theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f})",
        "boundary: theta0 + theta1*x1 + theta2*x2 = 0",
        "a straight line (a hyperplane in more dimensions)"], "purple", size=11, title_size=14)
    b.card(440, 556, 360, None, "at threshold 0.5", [f"TP {tp}   FP {fp}   FN {fn}   TN {tn}"], "green", size=13, title_size=14)
    b.card(830, 440, 350, None, "trained with cross-entropy", [
        "loss = -mean( y log p + (1-y) log(1-p) )",
        "confident wrong answers cost a lot",
        "(squared error would barely punish them)"], "orange", size=11, title_size=14)
    b.card(830, 556, 350, None, "blue circles = class 0", ["orange diamonds = class 1", "shaded side is predicted class 1"], "grey", size=11, title_size=13)
    return b


@board(f"{CLS}-accuracy-lies")
def accuracy_lies():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    b = Board(1240, 660, "Accuracy lies on imbalanced data", "Count the four outcomes, then read precision, recall and F1 instead")
    b.group(24, 92, 560, 540, "The four outcomes (lecture example)", "blue")
    tp, fp, fn, tn = 40, 10, 5, 45
    b.text(345, 156, "predicted", 13, "grey", "700")
    b.text(265, 190, "positive", 12, "grey")
    b.text(425, 190, "negative", 12, "grey")
    b.text(60, 250, "actual", 13, "grey", "700")
    b.text(120, 250, "positive", 12, "grey")
    b.text(120, 340, "negative", 12, "grey")
    b.card(190, 205, 150, 80, f"TP {tp}", ["caught"], "green", title_size=22)
    b.card(350, 205, 150, 80, f"FN {fn}", ["missed"], "red", title_size=22)
    b.card(190, 295, 150, 80, f"FP {fp}", ["false alarm"], "orange", title_size=22)
    b.card(350, 295, 150, 80, f"TN {tn}", ["correctly ignored"], "grey", title_size=22)
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * pr * rc / (pr + rc)
    b.card(44, 410, 524, None, "", [
        f"precision = {tp}/{tp + fp} = {pr:.2f}     (alarms that were real)",
        f"recall    = {tp}/{tp + fn} = {rc:.3f}    (real cases caught)",
        f"F1 = 2PR / (P + R) = {f1:.3f}",
        f"accuracy  = {tp + tn}/100 = {(tp + tn) / 100:.2f}"], "blue", size=12, align="left")
    b.group(612, 92, 604, 540, "Why accuracy misleads", "red")
    y = np.array([1] * 50 + [0] * 950)
    b.card(634, 138, 560, None, "Always predict the majority class", [
        f"1000 rows, 5% positive: accuracy {accuracy_score(y, np.zeros(1000)):.2f}, recall {recall_score(y, np.zeros(1000)):.2f}",
        "a model that catches nothing scores 95%"], "red", size=12, title_size=14)
    X, yy = make_classification(n_samples=6000, n_features=8, n_informative=4, weights=[0.95, 0.05],
                                class_sep=1.5, flip_y=0.01, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.3, stratify=yy, random_state=0)
    plain = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    rows = [["model", "thr", "accuracy", "precision", "recall"]]
    for name, m, t in [("always 0", None, None), ("plain LR", plain, 0.5), ("plain LR, moved", plain, 0.2), ("balanced weights", weighted, 0.5)]:
        if m is None:
            pred = np.zeros(len(yte))
            rows.append([name, "-", f"{accuracy_score(yte, pred):.3f}", "0.000", f"{recall_score(yte, pred):.3f}"])
            continue
        pred = (m.predict_proba(Xte)[:, 1] >= t).astype(int)
        rows.append([name, f"{t:.2f}", f"{accuracy_score(yte, pred):.3f}", f"{precision_score(yte, pred, zero_division=0):.3f}",
                     f"{recall_score(yte, pred):.3f}"])
    b.text(914, 268, f"test set: {int(yte.sum())} positives in {len(yte)} rows ({yte.mean():.1%})", 12, "red", "700")
    b.table(632, 286, [200, 70, 100, 100, 90], rows, "red", size=12)
    b.card(634, 470, 560, None, "Read the row, not the headline", [
        "moving the threshold from 0.5 to 0.2 lowers accuracy yet nearly doubles recall",
        "class weights buy more recall again, at a steep price in precision",
        "pick the trade-off from the cost of each kind of mistake"], "orange", size=11, title_size=14, align="left", bullets=True)
    return b


@board(f"{CLS}-roc-and-threshold")
def roc_and_threshold():
    X, y, theta, p = lab_data()
    b = Board(1200, 660, "The ROC curve", "Sweep the threshold from 1 down to 0 and watch the true and false positive rates climb together")
    x0, y0, size = 100, 120, 400
    mx, my = frame(b, x0, y0, size, size, (0, 1), (0, 1), [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1],
                   "false positive rate", "true positive rate", "{:g}", "{:g}")
    order = np.argsort(-p)
    tpr = np.r_[0, np.cumsum(y[order] == 1) / np.sum(y == 1)]
    fpr = np.r_[0, np.cumsum(y[order] == 0) / np.sum(y == 0)]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    pts = [(mx(a), my(c)) for a, c in zip(fpr, tpr)]
    b.parts.append('<path d="M' + " L".join(f"{a:.1f},{c:.1f}" for a, c in pts) + f' L{mx(1):.1f},{my(0):.1f} z" fill="{PALETTE["blue"]["fill"]}" fill-opacity="0.8"/>')
    line(b, mx(0), my(0), mx(1), my(1), "#868e96", 1.4, True)
    poly(b, pts, "blue", 2.8)
    sweep = [(0.9, "right"), (0.7, "right"), (0.5, "right"), (0.3, "right"), (0.1, "right")]
    for t, side in sweep:
        pred = p >= t
        tp_r = np.sum(pred & (y == 1)) / np.sum(y == 1)
        fp_r = np.sum(pred & (y == 0)) / np.sum(y == 0)
        dot(b, mx(fp_r), my(tp_r), 6, "orange")
        dx = 12 if side == "right" else -12
        b.text(mx(fp_r) + dx, my(tp_r) + (20 if side == "right" else -8), f"t = {t:.1f}", 12, "orange", "700", anchor="start" if side == "right" else "end")
    b.text(mx(0.62), my(0.5), "diagonal: random guessing", 12, "grey", italic=True)
    b.text(mx(0.5), my(0.12), f"AUC = {auc:.4f}", 20, "blue", "700")
    b.card(560, 120, 600, None, f"AUC = {auc:.4f}  means three equivalent things", [
        "the area under the ROC curve",
        "the chance a random positive gets a higher score than a random negative",
        "how well the scores rank, whatever threshold you later pick"], "blue", size=12, title_size=15, align="left", bullets=True)
    rows = [["threshold", "TPR (recall)", "FPR", "precision"]]
    for t in (0.9, 0.7, 0.5, 0.3, 0.1):
        pred = p >= t
        tpn, fpn = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        rows.append([f"{t:.1f}", f"{tpn / np.sum(y == 1):.3f}", f"{fpn / np.sum(y == 0):.3f}", f"{tpn / (tpn + fpn):.3f}"])
    b.table(560, 280, [130, 160, 130, 150], rows, "orange", size=13)
    b.card(560, 500, 600, None, "How to read it", [
        "top-left corner = perfect, diagonal = no better than a coin",
        "a higher threshold is cautious: few false alarms, more misses",
        "a lower threshold is eager: more caught, more false alarms"], "green", size=12, title_size=14, align="left", bullets=True)
    return b



PLAY = [("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No")]


def entropy(pos, neg):
    n = pos + neg
    return 0.0 - sum(c / n * math.log2(c / n) for c in (pos, neg) if c)


def gini(pos, neg):
    n = pos + neg
    return 1 - (pos / n) ** 2 - (neg / n) ** 2


def counts(rows):
    pos = sum(r[-1] == "Yes" for r in rows)
    return pos, len(rows) - pos


def gain(rows, col, f=entropy):
    total = f(*counts(rows))
    for v in sorted({r[col] for r in rows}):
        part = [r for r in rows if r[col] == v]
        total -= len(part) / len(rows) * f(*counts(part))
    return total


@board(f"{TREE}-outlook-split")
def outlook_split():
    pos, neg = counts(PLAY)
    b = Board(1280, 710, "Choosing the question: Outlook", "Split the 14 play-tennis days on the attribute that purifies them most (ID3, information gain)")
    b.group(24, 92, 780, 322, "Entropy before and after splitting on Outlook", "blue")
    root = b.card(310, 128, 220, None, f"[{pos}+, {neg}-]  all 14 days", [f"entropy {entropy(pos, neg):.3f}", f"Gini {gini(pos, neg):.3f}"], "blue", size=12, title_size=14)
    weighted = 0.0
    for i, v in enumerate(("Sunny", "Overcast", "Rain")):
        part = [r for r in PLAY if r[0] == v]
        p, n = counts(part)
        weighted += len(part) / 14 * entropy(p, n)
        col = "green" if n == 0 else "yellow"
        c = b.card(44 + i * 250, 244, 230, None, f"{v}  [{p}+, {n}-]", [f"entropy {entropy(p, n):.3f}", f"weight {len(part)}/14"], col, size=12, title_size=14)
        b.arrow(root.bottom(), c.top(), label=v if False else "")
    g = entropy(pos, neg) - weighted
    b.card(44, 336, 740, None, f"weighted child entropy = {weighted:.3f}      gain = {entropy(pos, neg):.3f} - {weighted:.3f} = {g:.3f}", [
        "5/14 x 0.971 + 4/14 x 0 + 5/14 x 0.971"], "purple", size=12, title_size=14)
    b.group(830, 92, 426, 322, "Information gain of each attribute", "orange")
    gains = sorted([(gain(PLAY, i), name) for i, name in enumerate(["Outlook", "Temperature", "Humidity", "Wind"])], reverse=True)
    top = gains[0][0]
    for i, (gv, name) in enumerate(gains):
        y = 150 + i * 56
        b.text(850, y + 16, name, 13, "orange", "700", anchor="start")
        w = 190 * gv / top
        b.parts.append(f'<rect x="960" y="{y}" width="{w:.1f}" height="26" rx="5" fill="{PALETTE["orange"]["stroke"]}" fill-opacity="{1 if i == 0 else 0.45}"/>')
        b.text(960 + w + 8, y + 18, f"{gv:.3f}", 13, "#343a40", "700", anchor="start")
    b.text(1043, 392, "highest gain wins the root", 12, "orange", italic=True)

    b.group(24, 430, 1232, 262, "The finished tree: the same rule applied again inside every impure branch", "green")
    r = b.card(560, 466, 160, 44, "Outlook", [], "blue", title_size=16)
    o_s = b.card(160, 556, 160, 44, "Humidity", [], "purple", title_size=15)
    o_o = b.card(600, 562, 80, 40, "Yes", [], "green", title_size=15)
    o_r = b.card(960, 556, 160, 44, "Wind", [], "purple", title_size=15)
    b.arrow(r.bottom(0.2), o_s.top(), label="Sunny")
    b.arrow(r.bottom(0.5), o_o.top(), label="Overcast")
    b.arrow(r.bottom(0.8), o_r.top(), label="Rain")
    leaves = [(80, 646, "High: No", "red", o_s.bottom(0.3)), (250, 646, "Normal: Yes", "green", o_s.bottom(0.7)),
              (880, 646, "Strong: No", "red", o_r.bottom(0.3)), (1050, 646, "Weak: Yes", "green", o_r.bottom(0.7))]
    for x, y, t, col, src in leaves:
        c = b.card(x, y, 150, 36, t, [], col, title_size=13)
        b.arrow(src, c.top())
    return b


@board(f"{TREE}-impurity-curves")
def impurity_curves():
    b = Board(1180, 700, "Three ways to score a node", "All are 0 for a pure node and largest at 50/50; entropy and Gini are smooth, which is what the splitting search wants")
    mx, my = frame(b, 80, 120, 520, 440, (0, 1), (0, 1), [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], "share of positives p", "impurity", "{:g}", "{:g}")
    ps = np.linspace(0.001, 0.999, 300)
    poly(b, [(mx(q), my(1 - max(q, 1 - q))) for q in ps], "green", 2.2, dash=True)
    poly(b, [(mx(q), my(1 - q * q - (1 - q) ** 2)) for q in ps], "orange", 2.6)
    poly(b, [(mx(q), my(entropy(q, 1 - q))) for q in ps], "blue", 3)
    q = 9 / 14
    dot(b, mx(q), my(entropy(q, 1 - q)), 6.5, "blue")
    dot(b, mx(q), my(gini(q, 1 - q)), 6.5, "orange")
    line(b, mx(q), my(0), mx(q), my(entropy(q, 1 - q)), "#868e96", 1.2, True)
    b.text(mx(q) + 12, my(entropy(q, 1 - q)) - 10, f"[9+,5-]: entropy {entropy(9, 5):.3f}", 12, "blue", "700", anchor="start")
    b.text(mx(q) + 12, my(gini(q, 1 - q)) - 10, f"Gini {gini(9, 5):.3f}", 12, "orange", "700", anchor="start")
    b.text(mx(0.2), my(0.97), "entropy (bits)", 13, "blue", "700")
    b.text(mx(0.17), my(0.43), "Gini", 13, "orange", "700")
    b.text(mx(0.07), my(0.17), "error", 13, "green", "700")
    rows = [["p", "entropy", "Gini", "error"]]
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0):
        h = 0.0 if v in (0.0, 1.0) else entropy(v, 1 - v)
        gg = 1 - v * v - (1 - v) ** 2
        rows.append([f"{v:.1f}", f"{h:.3f}", f"{gg:.3f}", f"{1 - max(v, 1 - v):.3f}"])
    b.table(660, 120, [90, 120, 120, 120], rows, "blue", size=13)
    b.card(660, 480, 450, None, "Formulas", [
        "entropy = - sum p_c log2 p_c",
        "Gini = 1 - sum p_c^2",
        "error = 1 - max p_c"], "purple", size=12, title_size=14, align="left")
    b.card(660, 590, 450, None, "Why not just use error?", ["it is flat near the ends, so it can't tell a good split from a mediocre one"], "yellow", size=11, title_size=13, align="left")
    return b


@board(f"{TREE}-pruning")
def pruning():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_classification(n_samples=800, n_features=10, n_informative=4, n_redundant=0, flip_y=0.15, class_sep=1.0, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1)
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xtr, ytr)
    alphas = path.ccp_alphas[:-1]
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.3, random_state=2)
    vs = [DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(Xf, yf).score(Xv, yv) for a in alphas]
    best_alpha = alphas[int(np.argmax(vs))]
    models = [("fully grown", DecisionTreeClassifier(random_state=0)),
              ("max_depth 2", DecisionTreeClassifier(max_depth=2, random_state=0)),
              ("max_depth 3", DecisionTreeClassifier(max_depth=3, random_state=0)),
              ("max_depth 5", DecisionTreeClassifier(max_depth=5, random_state=0)),
              ("min_leaf 15", DecisionTreeClassifier(min_samples_leaf=15, random_state=0)),
              ("cost-complexity", DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0))]
    b = Board(1240, 640, "A tree grown to purity memorises the noise", "Same 800 rows with 15% label noise: training accuracy always wins, test accuracy shows what was learned")
    mx, my = frame(b, 80, 150, 760, 340, (0, 6), (0.5, 1.0), [], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "", "accuracy", "{:g}", "{:.1f}")
    for i, (name, m) in enumerate(models):
        m.fit(Xtr, ytr)
        tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
        cx = 80 + (i + 0.5) * 760 / 6
        for off, v, col in ((-24, tr, "blue"), (24, te, "orange")):
            h = my(0.5) - my(v)
            b.parts.append(f'<rect x="{cx + off - 20:.1f}" y="{my(v):.1f}" width="40" height="{h:.1f}" rx="4" fill="{PALETTE[col]["stroke"]}" fill-opacity="0.85"/>')
            b.text(cx + off, my(v) - 7, f"{v:.3f}", 11, col, "700")
        b.text(cx, 514, name, 11, "grey", "700")
        b.text(cx, 532, f"{m.get_n_leaves()} leaves", 11, "grey")
    b.pill(100, 96, "training accuracy", "blue", solid=True)
    b.pill(290, 96, "test accuracy", "orange", solid=True)
    b.card(880, 130, 330, None, "Pre-pruning", ["stop growing early:", "max_depth, min_samples_leaf,", "min_samples_split"], "green", size=12, title_size=14)
    b.card(880, 262, 330, None, "Post-pruning", ["grow fully, then cut weak branches", f"validation set chose ccp_alpha = {best_alpha:.4f}"], "purple", size=12, title_size=14)
    b.card(880, 372, 330, None, "MDL and Occam", ["prefer the smallest tree", "that still explains the data"], "yellow", size=12, title_size=14)
    b.card(80, 560, 1130, None, "Rule of thumb from the scikit-learn guide: try min_samples_leaf = 5 first, then tune depth or ccp_alpha on held-out data", [], "grey", title_size=12)
    return b


@board(f"{TREE}-continuous-threshold")
def continuous_threshold():
    temp = np.array([40, 48, 60, 72, 80, 90], dtype=float)
    play = np.array([0, 0, 1, 1, 1, 0])

    def ent(lbl):
        if len(lbl) == 0:
            return 0.0
        q = lbl.mean()
        return 0.0 - sum(v * math.log2(v) for v in (q, 1 - q) if v)

    parent = ent(play)
    b = Board(1140, 520, "Splitting a number: where to cut?", "Try a threshold between each pair of neighbouring values; only label changes can be the best cut")
    x0, y0, w = 90, 190, 620
    mx = lambda t: x0 + (t - 30) / 70 * w
    line(b, x0, y0, x0 + w, y0, "#495057", 1.6)
    for t, p in zip(temp, play):
        (dot(b, mx(t), y0, 8, "green") if p else dot(b, mx(t), y0, 8, "red"))
        b.text(mx(t), y0 - 18, f"{int(t)}", 13, "#343a40", "700")
        b.text(mx(t), y0 + 28, "Yes" if p else "No", 12, "green" if p else "red", "700")
    rows = [["cut at", "left / right", "weighted entropy", "gain"]]
    best = None
    for lo, hi, pl, ph in zip(temp[:-1], temp[1:], play[:-1], play[1:]):
        t = (lo + hi) / 2
        left, right = play[temp <= t], play[temp > t]
        wv = (len(left) * ent(left) + len(right) * ent(right)) / len(play)
        change = pl != ph
        if change and (best is None or parent - wv > best[1]):
            best = (t, parent - wv)
        line(b, mx(t), y0 - 40, mx(t), y0 + 44, PALETTE["orange"]["stroke"] if change else "#ced4da", 2 if change else 1, not change)
        rows.append([f"<= {t:g}" + ("  *" if change else ""), f"{len(left)} / {len(right)}", f"{wv:.3f}", f"{parent - wv:.3f}"])
    b.table(90, 270, [190, 150, 190, 130], rows, "orange", size=13)
    b.card(790, 190, 320, None, f"best cut: temperature <= {best[0]:g}", [f"gain {best[1]:.3f}", f"parent entropy {parent:.3f}", "* marks a label change"], "orange", size=12, title_size=15)
    b.card(790, 330, 320, None, "Why only label changes", ["for entropy the best cut always", "sits where the label changes,", "so tools skip the other cuts"], "yellow", size=12, title_size=13)
    return b


def main(names):
    todo = names or list(BOARDS)
    for key in todo:
        match = [n for n in BOARDS if n == key or n.endswith(key)]
        for name in match:
            path = BOARDS[name]().save(OUT / f"{name}.svg")
            print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
