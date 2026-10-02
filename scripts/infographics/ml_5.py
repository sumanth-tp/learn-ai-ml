import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ml"
BOARDS = {}


def board(name):
    def wrap(fn):
        BOARDS[name] = fn
        return fn
    return wrap


@board("model-evaluation-confusion-metrics")
def confusion_metrics():
    b = Board(1120, 520, "One matrix, five metrics", "The lecture's example: TP 40, FP 10, FN 5, TN 45")
    b.text(235, 108, "predicted", 14, "grey", "700")
    b.text(155, 138, "negative", 12, "grey")
    b.text(315, 138, "positive", 12, "grey")
    b.text(40, 255, "actual", 14, "grey", "700", anchor="middle")
    tn = b.card(80, 150, 150, 110, "TN", ["45"], "green", size=22, title_size=18)
    fp = b.card(240, 150, 150, 110, "FP", ["10"], "orange", size=22, title_size=18)
    fn = b.card(80, 270, 150, 110, "FN", ["5"], "red", size=22, title_size=18)
    tp = b.card(240, 270, 150, 110, "TP", ["40"], "green", size=22, title_size=18)
    b.text(60, 210, "neg", 12, "grey", anchor="end")
    b.text(60, 330, "pos", 12, "grey", anchor="end")
    b.text(235, 410, "100 cases in all", 13, "grey", italic=True)
    rows = [
        ("accuracy", "(TP + TN) / all", "85 / 100 = 0.850", "blue"),
        ("precision", "TP / (TP + FP)", "40 / 50 = 0.800", "teal"),
        ("recall", "TP / (TP + FN)", "40 / 45 = 0.889", "purple"),
        ("F1", "2PR / (P + R)", "1.422 / 1.689 = 0.842", "orange"),
        ("specificity", "TN / (TN + FP)", "45 / 55 = 0.818", "green"),
    ]
    for i, (name, formula, value, col) in enumerate(rows):
        y = 100 + i * 66
        c = b.card(470, y, 600, 54, "", [], col)
        b.text(490, y + 33, name, 17, col, "700", anchor="start")
        b.text(640, y + 33, formula, 14, anchor="start")
        b.text(1050, y + 33, value, 16, col, "700", anchor="end")
    b.card(80, 440, 960, 56, "", ["An always-legit model on 1% fraud: accuracy 0.990, recall 0.000. Accuracy alone hides the failure."], "yellow", size=14)
    return b


@board("model-evaluation-honest-splits")
def honest_splits():
    b = Board(1240, 580, "Three ways to hold data back", "Breast-cancer data, logistic regression; drift data for the time split")
    b.group(25, 95, 380, 440, "One split", "red")
    b.parts.append('<rect x="50" y="150" width="240" height="30" rx="4" fill="#74c0fc" stroke="#1c7ed6"/>')
    b.parts.append('<rect x="290" y="150" width="90" height="30" rx="4" fill="#ffa94d" stroke="#e8590c"/>')
    b.text(170, 170, "train 80%", 12, "blue", "700")
    b.text(335, 170, "test 20%", 12, "orange", "700")
    b.card(50, 215, 330, 120, "Luck of the draw", ["20 different random splits:", "accuracy 0.947 to 0.991", "std 0.011"], "red", size=13)
    b.card(50, 360, 330, 150, "Why it matters", ["One number from one split can", "look better or worse than the", "truth by several points."], "yellow", size=13)

    b.group(425, 95, 390, 440, "5-fold cross-validation", "green")
    for r in range(5):
        for k in range(5):
            x = 450 + k * 68
            y = 150 + r * 38
            fill = "#ffa94d" if k == r else "#74c0fc"
            stroke = "#e8590c" if k == r else "#1c7ed6"
            b.parts.append(f'<rect x="{x}" y="{y}" width="64" height="30" rx="4" fill="{fill}" stroke="{stroke}"/>')
    b.text(620, 358, "orange = held-out fold, rotated", 12, "grey", italic=True)
    b.card(450, 385, 340, 125, "Every row tested once", ["fold scores 0.956 to 1.000", "mean 0.979, std 0.014", "stratified keeps class mix"], "green", size=13)

    b.group(835, 95, 380, 440, "Time-ordered split", "purple")
    for r in range(3):
        y = 150 + r * 38
        train_w = 70 + r * 50
        b.parts.append(f'<rect x="860" y="{y}" width="{train_w}" height="30" rx="4" fill="#74c0fc" stroke="#1c7ed6"/>')
        b.parts.append(f'<rect x="{860 + train_w + 6}" y="{y}" width="58" height="30" rx="4" fill="#ffa94d" stroke="#e8590c"/>')
    b.text(1025, 278, "the test block is always later", 12, "grey", italic=True)
    b.card(860, 300, 330, 90, "Shuffled KFold", ["mean accuracy 0.730", "(neighbours in time leak)"], "orange", size=13)
    b.card(860, 405, 330, 105, "TimeSeriesSplit", ["mean accuracy 0.681", "the honest, lower number when", "the world drifts"], "purple", size=13)
    return b


@board("model-evaluation-roc-vs-pr")
def roc_vs_pr():
    b = Board(1180, 540, "ROC ignores prevalence; precision does not", "Positives ~ N(1.5, 1), negatives ~ N(0, 1), threshold 1.0")
    b.card(40, 100, 330, 150, "Same ranking quality", ["ROC AUC = Phi(1.5 / sqrt 2)", "= 0.856", "at every prevalence"], "blue", size=14, title_size=16)
    b.table(410, 100, [150, 130, 190, 160], [
        ["prevalence", "AUC", "precision at t = 1", "area under PR"],
        ["0.50", "0.856", "0.813", "0.854"],
        ["0.10", "0.856", "0.326", "0.478"],
        ["0.01", "0.856", "0.042", "0.115"],
    ], header_color="blue", size=13, row_h=40)
    b.card(40, 290, 520, 120, "Prevalence 0.10, per 1000 cases", ["TP 69   FN 31   FP 143   TN 757", "TPR 0.691   FPR 0.159", "precision 69 / 212 = 0.326"], "teal", size=14, title_size=15)
    b.card(600, 290, 540, 120, "A real imbalanced model (2.5% positives)", ["ROC AUC 0.838", "average precision 0.575", "a random ranking would score 0.025"], "orange", size=14, title_size=15)
    b.card(40, 440, 1100, 70, "", ["When positives are rare, read the precision-recall curve: it shows what a flagged case is worth."], "yellow", size=15)
    return b


@board("model-evaluation-calibration-threshold")
def calibration_threshold():
    b = Board(1200, 560, "Honest probabilities, then a threshold from costs", "Beyond the lecture")
    b.group(25, 95, 560, 440, "Calibration (Gaussian Naive Bayes)", "teal")
    b.table(45, 140, [170, 110, 120, 110], [
        ["model", "Brier", "log loss", "ECE"],
        ["uncalibrated", "0.1663", "0.5514", "0.1066"],
        ["sigmoid (Platt)", "0.1574", "0.4843", "0.0314"],
        ["isotonic", "0.1520", "0.4692", "0.0236"],
    ], header_color="teal", size=13, row_h=38)
    b.card(45, 320, 520, 90, "Before calibrating", ["said 0.06, was positive 0.19", "said 0.96, was positive 0.88"], "orange", size=14)
    b.card(45, 430, 520, 85, "", ["Isotonic needs plenty of cases; with few it", "wobbles. Platt is steadier on small data."], "yellow", size=13)
    b.group(615, 95, 560, 440, "Cost-based threshold", "purple")
    b.card(635, 140, 520, 80, "Costs", ["miss a positive 500, false alarm 20", "optimum = 20 / (20 + 500) = 0.038"], "purple", size=14)
    b.table(635, 240, [260, 130, 130], [
        ["threshold", "total cost", "F1"],
        ["0.50 (the default)", "119,440", "0.607"],
        ["0.038 (formula)", "68,740", ""],
        ["0.08 (best on grid)", "63,500", "0.460"],
    ], header_color="purple", size=13, row_h=38)
    b.card(635, 420, 520, 95, "", ["TunedThresholdClassifierCV picked 0.07;", "test cost 63,920. F1 falls while cost falls by almost half."], "yellow", size=13)
    return b


@board("explaining-predictions-global-local")
def global_local():
    b = Board(1220, 580, "Which question are you asking?", "Credit-default model: gradient boosting on six features")
    b.group(25, 95, 580, 460, "Global: how does the model behave?", "blue")
    b.card(45, 145, 540, 150, "Permutation importance (test AUC drop)", ["late_payments 0.0940", "debt_ratio 0.0727   age 0.0527", "income 0.0212   tenure 0.0068   utilisation 0.0048"], "blue", size=13, title_size=14)
    b.card(45, 315, 540, 110, "Partial dependence on debt_ratio", ["average default probability rises", "0.065 at 0.069 to 0.306 at 0.565"], "teal", size=13, title_size=14)
    b.card(45, 445, 540, 90, "", ["Answers: what matters overall, and in which direction on average."], "yellow", size=13)
    b.group(635, 95, 560, 460, "Local: why this applicant?", "red")
    b.card(655, 145, 520, 110, "ICE curves", ["rise over the range: 2+ late payments 0.609", "0 or 1 late payments 0.193 (average 0.240)"], "orange", size=13, title_size=14)
    b.card(655, 275, 520, 130, "SHAP (TreeExplainer)", ["base -2.008 + 1.394 + 1.358 + 0.871", "+ 0.526 - 0.412 + 0.029 = +1.758 log-odds", "= 0.853 probability"], "red", size=13, title_size=14)
    b.card(655, 425, 520, 110, "LIME-style local surrogate", ["a line fitted to the model near one point;", "the slopes move with the kernel width"], "purple", size=13, title_size=14)
    return b


@board("explaining-predictions-pitfalls")
def pitfalls():
    b = Board(1220, 560, "When an importance number misleads", "Same model class, three versions of the data")
    b.card(30, 100, 370, 70, "1. Baseline", ["test AUC 0.765"], "blue", size=13, title_size=16)
    b.card(30, 190, 370, 140, "debt_ratio", ["permutation 0.0727", "mean |SHAP| 0.404"], "blue", size=15, title_size=16)
    b.card(425, 100, 370, 70, "2. Add a near-copy of debt_ratio", ["test AUC 0.764"], "orange", size=13, title_size=15)
    b.card(425, 190, 370, 140, "credit is split", ["debt_ratio 0.0317, copy 0.0116", "|SHAP| 0.242 and 0.159", "neither looks as big as the original"], "orange", size=13, title_size=15)
    b.card(820, 100, 370, 70, "3. Add collections_calls", ["a consequence of default; test AUC 0.970"], "red", size=12, title_size=15)
    b.card(820, 190, 370, 140, "tops every ranking", ["permutation 0.3163", "mean |SHAP| 2.055", "the next largest is 0.0104"], "red", size=14, title_size=15)
    b.arrow((215, 330), (215, 370), color="blue")
    b.arrow((610, 330), (610, 370), color="orange")
    b.arrow((1005, 330), (1005, 370), color="red")
    b.card(30, 375, 370, 100, "Read it as", ["a measure of how much THIS model", "relies on the column"], "blue", size=13, title_size=14)
    b.card(425, 375, 370, 100, "Cluster correlated columns", ["explain the group, or drop all", "but one before ranking"], "orange", size=13, title_size=14)
    b.card(820, 375, 370, 100, "Not a lever", ["importance is not cause; asking staff to", "cut collection calls would change nothing"], "red", size=13, title_size=14)
    b.card(30, 495, 1160, 45, "", ["Explanations describe the model, not the world."], "yellow", size=15)
    return b


@board("capstone-tabular-pipeline-flow")
def capstone_flow():
    b = Board(1300, 560, "From raw table to a checked, saved model", "Synthetic churn data: 8000 rows, churn rate 0.334")
    steps = [
        ("1 Split", ["6400 development", "1600 locked test"], "blue"),
        ("2 Baselines", ["majority AUC 0.500", "logistic 0.844", "boosting 0.840"], "teal"),
        ("3 Tune", ["12 random draws", "CV AUC 0.854"], "purple"),
        ("4 Calibrate", ["OOF Brier 0.1402", "to 0.1401: no gain"], "orange"),
        ("5 Threshold", ["economics: 0.167", "out-of-fold: 0.18"], "yellow"),
    ]
    prev = None
    for i, (t, lines, col) in enumerate(steps):
        c = b.card(30 + i * 252, 110, 225, 120, t, lines, col, size=13, title_size=16)
        if prev:
            b.arrow(prev.right(), c.left())
        prev = c
    b.arrow(prev.bottom(), (142, 300), via=[(prev.cx, 265), (142, 265)], color="blue")
    steps2 = [
        ("6 Test once", ["AUC 0.852", "precision 0.518", "recall 0.876"], "green"),
        ("7 Explain", ["contract 0.1570", "usage_trend 0.0675"], "red"),
        ("8 Model card", ["limits and slices", "region AUC 0.814 to 0.889"], "pink"),
        ("9 Save + check", ["joblib bundle", "6 checks pass"], "grey"),
    ]
    prev = None
    for i, (t, lines, col) in enumerate(steps2):
        c = b.card(30 + i * 252, 300, 225, 120, t, lines, col, size=13, title_size=16)
        if prev:
            b.arrow(prev.right(), c.left())
        prev = c
    b.card(30, 455, 1230, 70, "", ["Net benefit on the test set, with assumed economics: contact everyone 16,040; model at threshold 19,050."], "yellow", size=15)
    return b


@board("question-bank-worked-numbers")
def worked_numbers():
    b = Board(1240, 620, "The bank's arithmetic, checked", "Every number below was reproduced in code")
    cards = [
        ("k-NN", ["distance (1,2) to (4,6) = 5", "query (3,4): B 1.41, A 2.83,", "C 3.61, D 5.00 -> class +"], "blue"),
        ("SVM margin 2 / ||w||", ["w = (3,4): 2 / 5 = 0.4", "||w|| = 0.5: margin 4", "w=(2,1), b=-5, x=(2,3): +2"], "purple"),
        ("Bayes", ["spam: 0.32 / 0.38 = 0.842", "cavity: 0.12 / 0.20 = 0.6"], "teal"),
        ("k-means", ["centroid of 3 points (1.5, 1.33)", "first pass: {2,3} and the rest", "new centroids 2.5 and 16.0"], "orange"),
        ("Metrics (Q50)", ["TP 40, FP 10, FN 20, TN 30", "accuracy 0.700, precision 0.800", "recall 0.667, F1 0.727"], "green"),
        ("Tie in Q40", ["3 is exactly between 2 and 4", "tie to c1: 2.5 and 16.0", "tie to c2: 2.0 and 14.375"], "red"),
    ]
    for i, (t, lines, col) in enumerate(cards):
        x = 30 + (i % 3) * 405
        y = 100 + (i // 3) * 230
        b.card(x, y, 385, 200, t, lines, col, size=14, title_size=17)
    b.card(30, 565, 1180, 40, "", ["Modules 6 to 11: 55 questions, with 19 duplicate questions kept as short forms inside the matching answer."], "yellow", size=13)
    return b


def main(names):
    for name in names or list(BOARDS):
        path = BOARDS[name]().save(OUT / f"{name}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
