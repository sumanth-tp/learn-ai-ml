"""Infographics for docs/theory/causal.

Run from the repo root:

    python3 scripts/infographics/causal_1.py              # all boards
    python3 scripts/infographics/causal_1.py worked       # boards whose name ends with the argument
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "causal"
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


def node(b, cx, cy, label, color="blue", w=120, h=40):
    return b.card(cx - w / 2, cy - h / 2, w, h, label, [], color, size=13, title_size=14)


@board("potential-outcomes")
def potential_outcomes():
    b = Board(1160, 560, "Every user has two outcomes, and we see one", "Simulated: 20,000 users, true effect built in as +2.0 spend")
    rows = [
        ["user", "engaged", "notified", "spend if\nnot notified", "spend if\nnotified", "effect"],
        ["0", "1", "1", "?", "15.81", "?"],
        ["1", "1", "0", "14.13", "?", "?"],
        ["2", "1", "1", "?", "15.34", "?"],
        ["3", "0", "0", "9.60", "?", "?"],
        ["4", "0", "0", "9.25", "?", "?"],
        ["5", "1", "0", "13.17", "?", "?"],
    ]
    b.group(20, 90, 640, 400, "What the data holds", "blue")
    b.table(40, 130, [60, 90, 90, 150, 130, 80], rows, "blue", size=14, row_h=44)
    raw_text(b, 340, 470, "the question marks are the outcomes that never happened", 13, FAINT)
    b.group(700, 90, 440, 400, "What we can compute", "orange")
    b.card(724, 130, 392, 100, "Simulator's view", ["knows both columns", "mean of (Y1 - Y0) = 2.000"], "green", size=15, title_size=16)
    b.card(724, 262, 392, 100, "Analyst's view", ["sees one column per user", "notified minus not: 4.369"], "red", size=15, title_size=16)
    b.card(724, 394, 392, 78, "Why they differ", ["79.9% of notified users were engaged", "against 20.4% of the others"], "yellow", size=14, title_size=15)
    return b


@board("worked-example")
def worked_example():
    b = Board(1160, 600, "Worked example: 1,000 users, one confounder", "Averages are exact by construction, so every step can be checked by hand")
    b.group(20, 90, 540, 300, "Step 1: who gets notified", "blue")
    b.card(40, 130, 240, 110, "Engaged (500)", ["notified 400", "not notified 100"], "green", size=15, title_size=16)
    b.card(300, 130, 240, 110, "Others (500)", ["notified 100", "not notified 400"], "grey", size=15, title_size=16)
    raw_text(b, 290, 285, "Engaged users are four times more likely to be notified.", 13, INK)
    raw_text(b, 290, 310, "Notified group: 400 engaged + 100 others.", 13, INK)
    raw_text(b, 290, 335, "Not notified: 100 engaged + 400 others.", 13, INK)
    b.group(580, 90, 560, 300, "Step 2: average spend in each cell", "orange")
    b.table(600, 135, [190, 170, 170], [
        ["", "notified", "not notified"],
        ["engaged", "16", "14"],
        ["others", "12", "10"],
    ], "orange", size=15, row_h=46)
    raw_text(b, 860, 320, "Inside each row the gap is 2.", 14, "#c2410c", weight="700")
    raw_text(b, 860, 345, "That is the true effect.", 14, "#c2410c", weight="700")
    b.group(20, 410, 1120, 170, "Step 3: the two ways to compare", "purple")
    b.card(40, 450, 540, 110, "Naive: notified minus not notified", ["notified: (400 x 16 + 100 x 12) / 500 = 15.2", "not notified: (100 x 14 + 400 x 10) / 500 = 10.8", "gap = 4.4"], "red", size=14, title_size=15, align="left")
    b.card(600, 450, 520, 110, "Adjusted: compare inside each row, then average", ["engaged: 16 - 14 = 2", "others: 12 - 10 = 2", "average of 2 and 2 = 2.0"], "green", size=14, title_size=15, align="left")
    return b


@board("three-roles")
def three_roles():
    b = Board(1160, 600, "Which variables to adjust for", "Printed by the bad-controls experiment: true total effect is 2.0")
    cols = [
        ("Outcome-only cause: harmless", "green", 20),
        ("Mediator: leave alone", "orange", 400),
        ("Collider: leave alone", "red", 780),
    ]
    for title, color, x in cols:
        b.group(x, 90, 360, 500, title, color)

    node(b, 200, 200, "quality", "green", 100)
    node(b, 90, 330, "treated", "blue", 100)
    node(b, 310, 330, "spend", "purple", 100)
    b.arrow((220, 222), (290, 308), color="grey")
    b.arrow((140, 330), (260, 330), color="blue")
    raw_text(b, 200, 400, "quality moves spend but not treatment,", 12, INK)
    raw_text(b, 200, 418, "so adjusting changes little.", 12, INK)
    raw_text(b, 200, 470, "unadjusted  +2.017", 15, INK, weight="700")
    raw_text(b, 200, 498, "adjusted    +2.004", 15, "#2b8a3e", weight="700")
    raw_text(b, 200, 540, "A cause of the outcome only", 12, FAINT)
    raw_text(b, 200, 558, "tightens the error bar (0.015 to 0.013).", 12, FAINT)

    node(b, 480, 330, "treated", "blue", 100)
    node(b, 590, 200, "usage", "orange", 100)
    node(b, 700, 330, "spend", "purple", 100)
    b.arrow((510, 308), (568, 222), color="orange")
    b.arrow((612, 222), (670, 308), color="orange")
    b.arrow((530, 330), (650, 330), color="blue")
    raw_text(b, 590, 400, "usage carries part of the effect.", 12, INK)
    raw_text(b, 590, 418, "Holding it fixed removes that part.", 12, INK)
    raw_text(b, 590, 470, "total effect        2.000", 15, INK, weight="700")
    raw_text(b, 590, 498, "adjusted for usage +1.013", 15, "#c2410c", weight="700")
    raw_text(b, 590, 540, "Right for the direct effect,", 12, FAINT)
    raw_text(b, 590, 558, "wrong for the total effect.", 12, FAINT)

    node(b, 860, 200, "treated", "blue", 100)
    node(b, 1060, 200, "spend", "purple", 100)
    node(b, 960, 340, "reviewed", "red", 110)
    b.arrow((910, 200), (1010, 200), color="blue")
    b.arrow((880, 222), (930, 318), color="red")
    b.arrow((1040, 222), (990, 318), color="red")
    raw_text(b, 960, 410, "Reviews rise with treatment and with spend:", 12, INK)
    raw_text(b, 960, 428, "both arrows point into reviewed.", 12, INK)
    raw_text(b, 960, 470, "unadjusted         +2.017", 15, INK, weight="700")
    raw_text(b, 960, 498, "adjusted for review -0.257", 15, "#c92a2a", weight="700")
    raw_text(b, 960, 540, "The sign flips, in a randomised", 12, FAINT)
    raw_text(b, 960, 558, "experiment, by one bad control.", 12, FAINT)
    return b

@board("coin-or-choice")
def coin_or_choice():
    b = Board(1160, 560, "A coin balances people, a choice does not", "Standardised mean difference of each covariate between treated and untreated; below 0.1 is balanced")
    b.group(20, 90, 540, 440, "Assigned by a coin", "green")
    b.group(600, 90, 540, 440, "Assigned by covariates", "red")
    for x0, label, vals, gap, ci, col in (
        (40, "coin", [("x1", 0.040), ("x2", 0.011), ("x3 squared", 0.000)], "2.145", "1.806 to 2.484", "green"),
        (620, "choice", [("x1", 0.666), ("x2", 0.417), ("x3 squared", 0.640)], "5.462", "5.165 to 5.759", "red"),
    ):
        y = 150
        for name, v in vals:
            raw_text(b, x0 + 70, y + 14, name, 14, INK, anchor="end")
            b.bar(x0 + 90, y, 300, abs(v), threshold=0.1, color=col, h=18)
            raw_text(b, x0 + 410, y + 14, f"{v:+.3f}", 14, INK, anchor="start", weight="700")
            y += 62
        b.card(x0 + 20, 350, 460, 120, "Difference in means", [f"{gap}", f"95% interval {ci}", "true effect 2.000"], col, size=15, title_size=16)
        raw_text(b, x0 + 250, 505, "the interval contains the true 2.0" if label == "coin" else "the interval misses the true 2.0 by a wide margin", 12, FAINT)
    return b


@board("pseudo-population")
def pseudo_population():
    b = Board(1160, 600, "Inverse-probability weights build a randomised-looking crowd", "Same 1,000 users as the worked example: each user counts for 1 divided by the chance of the treatment they got")
    b.group(20, 90, 760, 400, "Counts, weights and weighted counts", "blue")
    cells = [
        (40, 140, "Engaged, notified", ["400 users x weight 1.25", "= 500 weighted", "mean spend 16"], "green"),
        (410, 140, "Engaged, not notified", ["100 users x weight 5", "= 500 weighted", "mean spend 14"], "teal"),
        (40, 310, "Others, notified", ["100 users x weight 5", "= 500 weighted", "mean spend 12"], "orange"),
        (410, 310, "Others, not notified", ["400 users x weight 1.25", "= 500 weighted", "mean spend 10"], "yellow"),
    ]
    for x, y, title, lines, col in cells:
        b.card(x, y, 350, 140, title, lines, col, size=15, title_size=16)
    raw_text(b, 400, 470, "weight = 1 / P(the treatment the user actually got)", 13, FAINT)
    b.group(810, 90, 330, 400, "What the weights achieve", "purple")
    b.card(830, 140, 290, 100, "Weighted notified", ["(500 x 16 + 500 x 12) / 1000", "= 14.0"], "blue", size=14, title_size=15)
    b.card(830, 260, 290, 100, "Weighted not notified", ["(500 x 14 + 500 x 10) / 1000", "= 12.0"], "blue", size=14, title_size=15)
    b.card(830, 380, 290, 90, "Weighted effect", ["14.0 - 12.0 = 2.0", "the truth"], "green", size=15, title_size=16)
    b.card(20, 510, 1120, 70, "Why it works", ["each row of the crowd now has 500 notified and 500 not notified, so engagement no longer predicts treatment"], "yellow", size=14, title_size=15)
    return b


@board("estimator-scoreboard")
def estimator_scoreboard():
    b = Board(1160, 600, "Which estimator survives which mistake", "200 repeats, n = 2,000, true effect 2.000, confounded assignment (block 3, strength 1.0)")
    rows = [
        ["estimator", "mean", "spread (sd)", "rmse", "what must be right"],
        ["naive difference", "5.736", "0.154", "3.739", "nothing can fix it"],
        ["regression, wrong form", "4.103", "0.144", "2.108", "the outcome model"],
        ["regression, right form", "2.004", "0.053", "0.053", "the outcome model"],
        ["weighting, right propensity", "1.978", "0.913", "0.913", "the propensity model"],
        ["matching, right propensity", "2.111", "0.099", "0.149", "the propensity model"],
        ["doubly robust, wrong outcome", "1.990", "0.595", "0.595", "either model"],
        ["doubly robust, wrong propensity", "2.003", "0.053", "0.053", "either model"],
        ["doubly robust, both wrong", "4.167", "0.150", "2.172", "at least one model"],
    ]
    b.table(30, 100, [330, 110, 150, 110, 400], rows, "blue", size=15, row_h=44)
    b.card(30, 520, 540, 64, "Two honest surprises", ["correct weighting is 17x noisier than a right-form regression"], "yellow", size=13, title_size=14)
    b.card(590, 520, 540, 64, "Matching leaves a small bias", ["2.111 against 2.000, from imperfect nearest neighbours"], "orange", size=13, title_size=14)
    return b

def polyline(b, pts, stroke, width=3, dash=None):
    d = " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(f'<path d="M{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{da} stroke-linejoin="round" stroke-linecap="round"/>')


def dot(b, cx, cy, r, fill, stroke):
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


@board("did-lines")
def did_lines():
    b = Board(1160, 600, "Difference-in-differences borrows the control group's trend", "Stores with a rollout (treated) and without (control), 4 periods before and 4 after; true effect 3.0")
    b.group(20, 90, 640, 490, "Parallel trends: the estimate is right", "green")
    x0, x1 = 90, 560
    def y(v):
        return 520 - (v - 50) * 20
    polyline(b, [(x0, y(50)), (x1, y(54))], PALETTE["blue"]["stroke"])
    polyline(b, [(x0, y(56)), (x1, y(60))], PALETTE["orange"]["stroke"], 2, "7 5")
    polyline(b, [(x0, y(56)), (x1, y(63))], PALETTE["orange"]["stroke"])
    for (xx, v, c) in [(x0, 50, "blue"), (x1, 54, "blue"), (x0, 56, "orange"), (x1, 63, "orange")]:
        dot(b, xx, y(v), 6, PALETTE[c]["fill"], PALETTE[c]["stroke"])
    line(b, x1 + 25, y(60), x1 + 25, y(63), PALETTE["red"]["stroke"], 3)
    raw_text(b, x1 + 35, y(61.5) + 4, "effect", 13, "#c92a2a", anchor="start", weight="700")
    raw_text(b, x0, 548, "before", 13, FAINT)
    raw_text(b, x1, 548, "after", 13, FAINT)
    raw_text(b, 330, y(47.5), "dashed: where treated stores would have gone", 12, FAINT)
    b.card(40, 120, 600, 90, "Estimate 2.996", ["95% interval 2.679 to 3.314", "pre-period trend gap 0.157 per period, p = 0.133"], "green", size=14, title_size=15)
    b.group(700, 90, 440, 490, "Treated stores were already speeding up", "red")
    b.card(720, 130, 400, 120, "Estimate 4.734", ["true effect 3.000", "95% interval 4.412 to 5.056", "the interval excludes the truth"], "red", size=15, title_size=17)
    b.card(720, 280, 400, 120, "The warning sign", ["pre-period trend gap 0.426 per period", "p = 0.0001", "visible before the rollout"], "yellow", size=15, title_size=16)
    b.card(720, 430, 400, 120, "The rule", ["estimate = true effect + extra trend gained", "over the gap between before and after", "0.4 x 4 periods = 1.6, so 4.6 expected, 4.734 seen"], "purple", size=14, title_size=15)
    return b


@board("iv-path")
def iv_path():
    b = Board(1160, 620, "An instrument moves treatment without touching the outcome", "Nudge to a training course, take-up of the course, earnings; ability is never observed")
    b.group(20, 90, 700, 400, "The causal path", "blue")
    node(b, 110, 330, "nudge", "green", 120)
    node(b, 370, 330, "take-up", "blue", 120)
    node(b, 630, 330, "earnings", "purple", 120)
    node(b, 500, 160, "ability", "grey", 120)
    b.arrow((170, 330), (310, 330), color="green", label="first stage\n+0.388")
    b.arrow((430, 330), (570, 330), color="blue", label="effect = ?")
    b.arrow((450, 182), (390, 308), color="grey", dashed=True)
    b.arrow((550, 182), (610, 308), color="grey", dashed=True)
    b.arrow((110, 350), (630, 350), via=[(110, 420), (630, 420)], color="red", dashed=True, label="forbidden: exclusion restriction", label_at=0.5)
    raw_text(b, 370, 120 + 50, "hidden", 12, FAINT)
    b.group(740, 90, 400, 400, "Estimates (true effect 2.000)", "orange")
    b.table(756, 135, [200, 168], [
        ["method", "estimate"],
        ["OLS", "3.508"],
        ["Wald / 2SLS", "1.981"],
        ["nudge leaks +0.5", "3.329"],
    ], "orange", size=15, row_h=44)
    raw_text(b, 940, 350, "Wald = (earnings gap by nudge)", 12, INK)
    raw_text(b, 940, 370, "divided by (take-up gap by nudge)", 12, INK)
    raw_text(b, 940, 410, "leak bias = 0.5 / 0.388 = 1.29", 12, "#c2410c", weight="700")
    b.group(20, 510, 1120, 90, "Weak instruments, 300 repeats of n = 2,000 (10th to 90th percentile of the estimate)", "red")
    raw_text(b, 200, 570, "first-stage F 381.8:   1.711 to 2.228", 14, INK)
    raw_text(b, 580, 570, "F 13.9:   0.281 to 3.113", 14, INK)
    raw_text(b, 950, 570, "F 1.6:   -5.627 to 7.484", 14, "#c92a2a", weight="700")
    return b


@board("rdd-cutoff")
def rdd_cutoff():
    b = Board(1160, 600, "Regression discontinuity compares units just either side of a cutoff", "Score of 50 or more gets the programme; true jump 3.0; 300 repeats per bandwidth")
    b.group(20, 90, 640, 490, "Outcome against score", "blue")
    def px(sc):
        return 70 + (sc - 30) * 14
    def py(v):
        return 520 - (v - 17) * 24
    left = [(px(sc), py(20 + 0.15 * (sc - 50))) for sc in range(30, 51)]
    right = [(px(sc), py(23 + 0.15 * (sc - 50) + 0.012 * (sc - 50) ** 2)) for sc in range(50, 71)]
    polyline(b, left, PALETTE["blue"]["stroke"])
    polyline(b, right, PALETTE["orange"]["stroke"])
    line(b, px(50), 130, px(50), 540, INK, 1.6, "5 5")
    line(b, px(50) - 8, py(20), px(50) - 8, py(23), PALETTE["red"]["stroke"], 3)
    raw_text(b, px(50) - 16, py(21.5) + 4, "jump 3.0", 13, "#c92a2a", anchor="end", weight="700")
    raw_text(b, px(50), 560, "cutoff 50", 13, INK)
    raw_text(b, px(36), 560, "score 30", 12, FAINT)
    raw_text(b, px(66), 560, "score 70", 12, FAINT)
    b.group(690, 90, 450, 330, "Bandwidth: bias against noise", "orange")
    b.table(706, 135, [130, 120, 100, 94], [
        ["half-width", "mean jump", "spread", "rows"],
        ["20", "2.507", "0.144", "4,000"],
        ["10", "2.884", "0.200", "1,964"],
        ["5", "2.959", "0.278", "1,037"],
        ["2.5", "3.000", "0.387", "513"],
    ], "orange", size=15, row_h=44)
    raw_text(b, 915, 395, "wide windows are biased by curvature, narrow ones are noisy", 11, FAINT)
    b.card(690, 440, 450, 130, "Check before you trust it", ["a background variable should not jump (p = 0.839)", "no pile-up of rows just above the cutoff", "[266, 256, 257, 258] in 2.5-point bins"], "green", size=13, title_size=15)
    return b

@board("uplift-segments")
def uplift_segments():
    b = Board(1160, 600, "Four kinds of customer, and only one should get the coupon", "Coupon costs 0.5; the effect is the extra spend caused by the coupon (a randomised experiment, 10,000 customers)")
    cards = [
        (20, 100, "Persuadables", "17.6% of customers", "effect +2.5", "net of cost +2.0", "green"),
        (300, 100, "Older customers", "25.3% of customers", "effect +0.5", "net of cost 0.0", "yellow"),
        (580, 100, "Indifferent", "44.6% of customers", "effect 0.0", "net of cost -0.5", "grey"),
        (860, 100, "Sleeping dogs", "12.5% of customers", "effect -1.36", "net of cost -1.86", "red"),
    ]
    for x, y, title, share, eff, net, col in cards:
        b.card(x, y, 270, 150, title, [share, eff, net], col, size=15, title_size=17)
    b.group(20, 280, 1120, 300, "What targeting is worth, per customer", "blue")
    b.table(40, 330, [330, 130, 170, 450], [
        ["policy", "profit", "share treated", "note"],
        ["treat nobody", "0.000", "0%", "the baseline"],
        ["treat everyone", "-0.104", "100%", "sleeping dogs and cost eat the gain"],
        ["random 20%", "-0.021", "20%", "no better than the average customer"],
        ["perfect ranking, top 20%", "0.352", "20%", "all persuadables"],
    ], "blue", size=15, row_h=44)
    raw_text(b, 580, 572, "exact shares and effects; the same figures drive the lab", 13, FAINT)
    return b


@board("learner-scoreboard")
def learner_scoreboard():
    b = Board(1160, 600, "Three ways to estimate who responds", "Held-out customers; truth known because the data are simulated; 8 fresh datasets for the mean and spread")
    b.group(20, 90, 1120, 310, "How well each learner ranks customers by true effect", "purple")
    b.table(40, 135, [250, 180, 180, 220, 240], [
        ["learner", "corr with truth", "rmse", "profit, top 20%", "8-dataset corr (sd)"],
        ["S-learner", "0.853", "0.601", "0.308", "0.854 (0.026)"],
        ["X-learner", "0.746", "0.949", "0.274", "0.711 (0.046)"],
        ["T-learner", "0.648", "1.282", "0.210", "0.637 (0.038)"],
        ["oracle (true effect)", "1.000", "0.000", "0.355", "1.000"],
    ], "purple", size=15, row_h=44)
    b.card(20, 440, 360, 140, "S-learner", ["one model with the coupon as a feature", "simple, shrinks the effect", "won here"], "green", size=14, title_size=16)
    b.card(400, 440, 360, 140, "T-learner", ["two models, one per arm", "subtracts two noisy fits", "worst here"], "red", size=14, title_size=16)
    b.card(780, 440, 360, 140, "X-learner", ["fits the other arm's gap", "then blends the two", "in between"], "yellow", size=14, title_size=16)
    return b


@board("dml-recipe")
def dml_recipe():
    b = Board(1160, 640, "Double machine learning: remove what X explains, then compare what is left", "True effect of a price cut on sales is 1.000; 20 datasets of 1,000 rows, nonlinear confounding")
    b.group(20, 90, 1120, 150, "The recipe", "blue")
    steps = [(40, "1. Split", ["five folds"]), (250, "2. Predict sales from X", ["model fitted on other folds"]), (490, "3. Predict price cut from X", ["same folds"]), (740, "4. Residuals", ["sales minus guess", "price cut minus guess"]), (950, "5. Regress", ["sales residual on", "price cut residual"])]
    for x, title, lines in steps:
        b.card(x, 135, 190 if x < 940 else 170, 90, title, lines, "blue", size=12, title_size=13)
    b.group(20, 260, 1120, 360, "What each version returns", "orange")
    b.table(40, 305, [440, 160, 160, 330], [
        ["estimator", "mean", "sd", "what went wrong"],
        ["naive regression", "1.961", "0.052", "confounded by X"],
        ["linear controls", "2.095", "0.061", "X acts nonlinearly"],
        ["plug-in forest for sales only", "0.296", "0.016", "regularisation bias"],
        ["DML, forest, cross-fitted", "1.161", "0.073", "small forest bias"],
        ["DML, boosting, no cross-fitting", "0.633", "0.089", "memorised residuals"],
        ["DML, boosting, cross-fitted", "0.923", "0.066", "best of the boosted pair"],
    ], "orange", size=14, row_h=40)
    return b


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for key, fn in BOARDS.items():
        if argv and not NAMES[key].endswith(argv[0]):
            continue
        path = fn().save(OUT / f"{NAMES[key]}.svg")
        print("wrote", path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
