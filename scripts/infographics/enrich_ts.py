"""Second boards for docs/theory/timeseries, drawn from the numbers the chapters' experiments print.

Run from the repo root:

    python3 scripts/infographics/enrich_ts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "ts-enrich"


def raw_text(b, x, y, text, size=12, fill=INK, anchor="middle", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def line(b, x1, y1, x2, y2, color=INK, width=1.6, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{extra}/>')


def hbar(b, x, y, w, h, value, vmax, color, label, label_w=250, fmt="{:.2f}"):
    rect(b, x, y, w, h, "#f1f3f5", "#ced4da", 1, 3)
    rect(b, x, y, max(2, w * value / vmax), h, PALETTE[color]["stroke"], PALETTE[color]["stroke"], 1, 3)
    raw_text(b, x - 12, y + h / 2 + 4, label, 12, INK, "end")
    raw_text(b, x + w * value / vmax + 8, y + h / 2 + 4, fmt.format(value), 13, PALETTE[color]["text"], "start", "700")


def leakage_inflation():
    b = Board(1160, 560, "The same model, two splits, two stories", "Mean absolute error, seven days ahead, 215 test rows (lower is better)")
    panels = [
        (40, "Features include the time index t", [("seasonal naive baseline", 2.63, "grey"), ("random 5-fold split", 2.07, "green"), ("honest time split", 3.76, "red")]),
        (600, "Time index t removed", [("seasonal naive baseline", 2.63, "grey"), ("random 5-fold split", 2.16, "green"), ("honest time split", 2.76, "orange")]),
    ]
    for x0, title, rows in panels:
        b.group(x0 - 10, 95, 540, 300, title, "blue")
        for i, (label, value, color) in enumerate(rows):
            hbar(b, x0 + 210, 160 + i * 70, 250, 34, value, 4.0, color, label, fmt="{:.2f}")
    b.card(30, 420, 540, 110, "What the random split claims", ["with t: 21% better than the baseline", "(2.07 against 2.63)", "it can borrow rows from the future"], "green", size=14, title_size=15)
    b.card(590, 420, 540, 110, "What deployment gets", ["with t: 43% worse than the baseline (3.76)", "without t: 5% worse (2.76 against 2.63)", "trees cannot extrapolate a rising level"], "red", size=14, title_size=15)
    return b


def backtest_mase():
    b = Board(1160, 520, "Rolling-origin backtest: MASE by series and model", "15 origins, 14-day horizon, scaled by the training seasonal-naive error; lower is better, best per row in green")
    rows = [
        ["series", "naive", "seasonal naive", "ETS", "ARIMA", "ARIMA beats snaive"],
        ["quiet weekly (noise 1)", "3.568", "1.090", "0.998", "1.073", "7 of 15 origins"],
        ["noisy weekly (noise 6)", "0.981", "0.975", "0.905", "0.738", "12 of 15 origins"],
        ["very noisy weekly (12)", "0.957", "0.913", "0.794", "0.686", "14 of 15 origins"],
        ["no season, noisy (6)", "0.866", "0.910", "0.845", "0.732", "11 of 15 origins"],
    ]
    widths = [270, 130, 190, 130, 130, 240]
    b.table(30, 100, widths, rows, "blue", size=15, row_h=50, zebra=True)
    best = {1: 3, 2: 4, 3: 4, 4: 4}
    x_edges = [30]
    for w in widths:
        x_edges.append(x_edges[-1] + w)
    for r, c in best.items():
        rect(b, x_edges[c] + 2, 100 + r * 50 + 2, widths[c] - 4, 46, "none", PALETTE["green"]["stroke"], 3, 6)
    b.card(30, 380, 520, 110, "Where the baseline holds", ["quiet weekly series: all three structured", "models land within 10% of seasonal naive"], "yellow", size=14, title_size=15)
    b.card(580, 380, 550, 110, "Where it fails", ["noise copied from one week ago costs accuracy;", "averaging models win by about 20% to 25%"], "orange", size=14, title_size=15)
    return b


def lag_ladder():
    b = Board(1160, 600, "One week ahead, 24 series: what moved the score", "MASE on 288 forecasts (12 origins), lower is better; every bar is a printed number")
    items = [
        ("seasonal naive", 1.004, "grey"),
        ("ETS, no promo information", 0.695, "blue"),
        ("global HGB, no promo columns", 0.724, "blue"),
        ("local HGB, one model per series", 0.674, "purple"),
        ("global HGB, all honest features", 0.505, "green"),
        ("global HGB, 6 series never seen", 0.460, "teal"),
        ("global HGB + leaky rolling mean", 0.420, "red"),
    ]
    for i, (label, value, color) in enumerate(items):
        hbar(b, 360, 105 + i * 58, 560, 36, value, 1.1, color, label, fmt="{:.3f}")
    b.card(40, 530, 520, 56, "Known promo plan: 0.724 to 0.505", [], "green", size=14, title_size=15)
    b.card(600, 530, 520, 56, "Leaky feature: 0.505 to 0.420, gone in serving", [], "red", size=14, title_size=15)
    return b


def pretrained_honest():
    b = Board(1160, 560, "Chronos-2 (120M, CPU) against seasonal naive and ETS", "MASE over 10 synthetic series x 8 origins, 7-day horizon (lower is better)")
    panels = [
        (40, "540 days of context", [("seasonal naive", 0.855, "grey"), ("ETS", 0.655, "blue"), ("Chronos-2", 0.634, "purple"), ("Chronos-2 + promo plan", 0.522, "green")]),
        (600, "28 days of context", [("seasonal naive", 0.855, "grey"), ("ETS", 0.736, "blue"), ("Chronos-2", 0.758, "purple"), ("Chronos-2 + promo plan", 0.772, "orange")]),
    ]
    for x0, title, rows in panels:
        b.group(x0 - 10, 95, 540, 330, title, "blue")
        for i, (label, value, color) in enumerate(rows):
            hbar(b, x0 + 230, 150 + i * 62, 230, 34, value, 1.0, color, label, fmt="{:.3f}")
    b.card(30, 450, 540, 90, "Long history", ["Chronos-2 ties ETS (0.634 against 0.655);", "the future promo plan is what helps (0.522)"], "green", size=14, title_size=15)
    b.card(590, 450, 540, 90, "Short history", ["no foundation-model gain over ETS here;", "28 days hold too few promos to learn from"], "orange", size=14, title_size=15)
    return b


def intervals_cusum():
    b = Board(1160, 600, "Intervals and alarms need measuring too", "Nominal 80% intervals, 14 test windows x 8 series; CUSUM against a plain threshold on 60 simulated runs")
    b.group(20, 95, 540, 395, "ARIMA 80% interval coverage by lead day", "blue")
    cov = [0.69, 0.74, 0.83, 0.88, 0.87, 0.89, 0.87]
    x0, ybase, hmax = 70, 440, 270
    for i, v in enumerate(cov):
        h = hmax * v
        color = "red" if v < 0.78 else "green"
        rect(b, x0 + i * 68, ybase - h, 46, h, PALETTE[color]["fill"], PALETTE[color]["stroke"], 2, 4)
        raw_text(b, x0 + i * 68 + 23, ybase - h + 46, f"{v:.2f}", 13, PALETTE[color]["text"], "middle", "700")
        raw_text(b, x0 + i * 68 + 23, ybase + 20, f"day {i + 1}", 12, INK)
    line(b, 55, ybase - hmax * 0.8, 545, ybase - hmax * 0.8, INK, 1.8, "6 5")
    raw_text(b, 62, ybase - hmax * 0.8 - 8, "nominal 0.80", 12, INK, "start")
    b.group(590, 95, 550, 395, "Detecting a level shift (days of delay)", "orange")
    rows = [
        ["shift", "CUSUM: caught / delay", "threshold: caught / delay"],
        ["1.4 (3 units)", "54 / 10 days", "18 / 27 days"],
        ["2.8 (6 units)", "55 / 3 days", "36 / 0 days"],
        ["4.6 (10 units)", "55 / 1 day", "46 / 0 days"],
        ["no shift", "5 false alarms", "13 false alarms"],
    ]
    b.table(595, 150, [130, 200, 210], rows, "orange", size=12, row_h=48, zebra=True)
    b.card(20, 510, 1120, 70, "Overall coverage looked fine (ARIMA 0.823, ETS 0.885); the per-day view and the conformal fix (0.811, 0.824) are the finding", [], "yellow", size=13, title_size=14)
    return b


BOARDS = {
    "leakage-inflation": leakage_inflation,
    "backtest-mase": backtest_mase,
    "lag-ladder": lag_ladder,
    "pretrained-honest": pretrained_honest,
    "intervals-cusum": intervals_cusum,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, maker in BOARDS.items():
        if only in name:
            maker().save(OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
