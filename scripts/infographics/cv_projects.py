import math
from pathlib import Path

from board import INK, FAINT, MONO, PALETTE, Board, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "cv-projects"


def frame(board: Board, x, y, w, h, color="grey", fill_opacity=0.0, radius=8):
    c = PALETTE[color]
    board.parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{c["fill"]}" fill-opacity="{fill_opacity}" '
        f'stroke="{c["stroke"]}" stroke-width="1.4"/>'
    )


def line(board: Board, x1, y1, x2, y2, color=INK, width=1.4, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    board.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')


def dot(board: Board, x, y, color="blue", r=5):
    board.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{PALETTE[color]["stroke"]}" stroke="#fffdf7" stroke-width="1.5"/>')


def label(board: Board, x, y, text, size=12, color=INK, anchor="middle", weight="400"):
    board.text(x, y, text, size, color, weight, anchor)


def defect_architecture():
    b = Board(1240, 640, "Surface inspection: from a 64 px crop to a decision", "Every number on this board was measured on the synthetic plant; the budget is the requirement it has to meet")
    g1 = b.group(20, 88, 1200, 262, "Decision path, one crop at a time", "blue")
    cam = b.card(40, 150, 170, 150, "Line-scan camera", ["64 x 64 grey crops", "50 crops per second", "left on site"], "grey", size=12)
    pre = b.card(240, 150, 150, 150, "Preprocess", ["uint8 to float", "minus 0.5", "no resizing"], "grey", size=12)
    net = b.card(420, 150, 190, 150, "CNN as ONNX", ["8,345 parameters", "57 KiB file", "0.22 ms median", "one CPU thread"], "blue", size=12)
    cal = b.card(640, 150, 190, 150, "Calibrate", ["prior offset -1.904", "temperature 1.018", "logit to probability"], "purple", size=12)
    dia = b.diamond(930, 225, 170, 140, "p >= 0.023?", "yellow", size=13)
    rej = b.card(1040, 150, 160, 60, "Reject", ["to manual review"], "red", size=12)
    acc = b.card(1040, 240, 160, 60, "Accept", ["coil continues"], "green", size=12)
    for a, c in ((cam, pre), (pre, net), (net, cal), (cal, dia)):
        b.arrow(a.right(), c.left() if hasattr(c, "left") else c.left())
    b.arrow(dia.right(0.3), rej.left(), label="yes")
    b.arrow(dia.right(0.7), acc.left(), label="no")
    b.card(40, 312, 560, 30, "", ["Supplier C coils skip the model and go to manual review (v3 rule)"], "orange", size=12)
    g2 = b.group(20, 380, 1200, 240, "Around the model", "green")
    b.card(40, 436, 270, 160, "Drift monitor", ["PSI on brightness, texture,", "sharpness and score", "reject rate limits 0.166 to 0.227", "alarm at PSI 0.25"], "teal", size=12)
    b.card(340, 436, 270, 160, "Registry and gate", ["v1, v2, v3 as ONNX + meta", "gate: recall, cost, review share", "current.json and history.json", "rollback in one command"], "purple", size=12)
    b.card(640, 436, 270, 160, "Weekly audit", ["label a sample of accepted", "crops: 3,445 give an escape", "rate within 0.2 points", "feeds the next golden set"], "orange", size=12)
    b.card(940, 436, 260, 160, "Service and CLI", ["POST /inspect with PNG", "GET /healthz and /stats", "scan, drift, rollback, serve", "round trip p95 0.76 ms"], "blue", size=12)
    b.arrow(net.bottom(), (net.cx, 380), dashed=True, color="grey")
    return b


def defect_requirements():
    b = Board(1160, 560, "The requirement written as constraints", "Budget on the left, what the shipped system measured on the right")
    rows = [
        ["Constraint", "Budget", "Measured", "Verdict"],
        ["Cost of errors", "miss 50 : false reject 1", "0.395 per crop vs 0.938 for reject-all", "met"],
        ["Escapes", "recall >= 0.92", "0.942 on the test set", "met; interval [0.917, 0.960]"],
        ["Human review", "<= 30% of crops", "0.281 with supplier C routed", "met"],
        ["Latency", "p95 <= 10 ms per crop", "0.23 ms model, 0.76 ms round trip", "met, over 10x spare"],
        ["Hardware", "one CPU core, no GPU", "ONNX Runtime, one thread", "met"],
        ["Model size", "<= 5 MiB", "57 KiB", "met"],
        ["Privacy", "no image leaves the plant", "service binds to localhost", "met by design"],
        ["Hard case", "supplier C coarse grain", "recall 0.80, 40% false rejects", "not met: routed to manual"],
    ]
    b.table(30, 100, [200, 270, 400, 260], rows, "blue", size=13)
    return b


def defect_worked_example():
    b = Board(1180, 520, "One threshold, six crops, three costs", "Miss costs 50, a false reject costs 1, so reject when p > 1 / 51 = 0.0196")
    crops = [("A", "defect", "0.40", "red"), ("B", "good", "0.30", "green"), ("E", "good", "0.05", "green"), ("D", "defect", "0.03", "red"), ("C", "good", "0.01", "green"), ("F", "good", "0.002", "green")]
    xs = []
    for i, (name, kind, p, colour) in enumerate(crops):
        x = 70 + i * 180
        xs.append(x)
        b.card(x, 130, 130, 90, f"crop {name}", [kind, f"p = {p}"], colour, size=13)
    cuts = [(60, "0.5", "grey"), (70 + 0 * 180 + 155, "0.35", "orange"), (70 + 3 * 180 + 155, "0.0196", "purple"), (70 + 5 * 180 + 155, "0.001", "grey")]
    for x, text, colour in cuts:
        line(b, x, 110, x, 245, PALETTE[colour]["stroke"], 2.4, "6 4")
        label(b, x, 102, f"t = {text}", 12, PALETTE[colour]["text"], "middle", "700")
    b.card(30, 290, 340, 190, "t = 0.5, the default", ["rejects nothing", "A and D escape", "2 x 50 = 100", "16.67 per crop"], "grey", size=14)
    b.card(420, 290, 340, 190, "t = 0.35", ["rejects A only", "D escapes: 50", "no false rejects", "8.33 per crop"], "orange", size=14)
    b.card(810, 290, 340, 190, "t = 0.0196, the cost-based one", ["rejects A, B, E, D", "B and E are false rejects: 2", "nothing escapes", "0.33 per crop"], "purple", size=14)
    return b


COST_CURVE = [(0.002, 0.852), (0.004, 0.708), (0.007, 0.56), (0.01, 0.456), (0.015, 0.401), (0.02, 0.401), (0.023, 0.404), (0.03, 0.43), (0.04, 0.495), (0.06, 0.53), (0.09, 0.582), (0.13, 0.717), (0.2, 0.852), (0.3, 0.955), (0.45, 1.033), (0.6, 1.12), (0.8, 1.244), (0.95, 1.544)]


def defect_cost_curve():
    b = Board(1160, 520, "Cost per crop against the threshold", "Test set, 8,000 crops, miss 50 and false reject 1; lower is better")
    x0, y0, w, h = 110, 110, 620, 330
    lo, hi = math.log10(0.001), math.log10(1.0)

    def px(t):
        return x0 + (math.log10(t) - lo) / (hi - lo) * w

    def py(c):
        return y0 + h - c / 1.6 * h

    frame(b, x0, y0, w, h, "grey")
    for c in (0.0, 0.4, 0.8, 1.2, 1.6):
        line(b, x0, py(c), x0 + w, py(c), "#dee2e6", 1)
        label(b, x0 - 10, py(c) + 4, f"{c:.1f}", 11, FAINT, "end")
    for t in (0.001, 0.01, 0.1, 1.0):
        line(b, px(t), y0 + h, px(t), y0 + h + 6, INK, 1.2)
        label(b, px(t), y0 + h + 22, f"{t:g}", 11, FAINT)
    label(b, x0 + w / 2, y0 + h + 44, "threshold on the calibrated probability (log scale)", 12, INK)
    label(b, 52, y0 + h / 2, "cost per crop", 12, INK)
    reject_all = 0.938
    line(b, x0, py(reject_all), x0 + w, py(reject_all), PALETTE["red"]["stroke"], 2, "7 5")
    label(b, x0 + w - 4, py(reject_all) - 8, "reject everything 0.938", 12, PALETTE["red"]["text"], "end", "700")
    points = " ".join(f"{px(t):.1f},{py(c):.1f}" for t, c in COST_CURVE)
    b.parts.append(f'<polyline points="{points}" fill="none" stroke="{PALETTE["blue"]["stroke"]}" stroke-width="3" stroke-linejoin="round"/>')
    marks = [(0.5, 1.064, "default 0.5", "grey", -10, -14), (0.26, 0.905, "best F1 0.26", "orange", -20, 26), (0.0196, 0.406, "Bayes 0.0196", "purple", -4, 36), (0.023, 0.405, "chosen 0.023", "green", 70, -22)]
    for t, c, text, colour, dx, dy in marks:
        dot(b, px(t), py(c), colour, 6)
        label(b, px(t) + dx, py(c) + dy, text, 12, PALETTE[colour]["text"], "middle", "700")
    b.card(790, 120, 340, 120, "Why the minimum is so low", ["A miss costs 50 times a false reject.", "So the system rejects whenever", "the chance of a defect exceeds 1 / 51."], "yellow", size=13)
    b.card(790, 262, 340, 118, "What it buys", ["0.405 per crop, 95% CI [0.324, 0.485]", "against 0.938 for rejecting everything", "and 2.919 for accepting everything"], "green", size=13)
    b.card(790, 400, 340, 90, "The default 0.5 is worse than doing nothing clever", ["1.064 per crop: 36% of defects escape"], "red", size=13)
    return b


SLICES = [
    ("supplier A", 0.92, 0.88, 0.95, 0.115),
    ("supplier B", 0.94, 0.90, 0.97, 0.094),
    ("supplier C", 0.80, 0.70, 0.87, 0.403),
    ("dim light", 0.90, 0.83, 0.95, 0.136),
    ("normal light", 0.91, 0.88, 0.94, 0.155),
    ("pit", 0.94, 0.89, 0.97, None),
    ("scratch", 0.93, 0.88, 0.96, None),
    ("stain", 0.85, 0.79, 0.90, None),
]


def defect_slices():
    b = Board(1160, 560, "Error analysis by slice", "Recall with 95% Wilson intervals, and the false reject rate, at the chosen threshold")
    left, top, w, row = 200, 140, 430, 44
    label(b, left + w / 2, 112, "recall of defects", 14, INK, "middle", "700")
    label(b, 900, 112, "false reject rate of good crops", 14, INK, "middle", "700")
    for r in (0.6, 0.7, 0.8, 0.9, 1.0):
        x = left + (r - 0.6) / 0.4 * w
        line(b, x, top - 10, x, top + row * len(SLICES), "#dee2e6", 1)
        label(b, x, top + row * len(SLICES) + 18, f"{r:.1f}", 11, FAINT)
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        x = 760 + t / 0.5 * 300
        line(b, x, top - 10, x, top + row * len(SLICES), "#dee2e6", 1)
        label(b, x, top + row * len(SLICES) + 18, f"{t:.1f}", 11, FAINT)
    line(b, left + (0.92 - 0.6) / 0.4 * w, top - 10, left + (0.92 - 0.6) / 0.4 * w, top + row * len(SLICES), PALETTE["red"]["stroke"], 1.6, "6 4")
    label(b, left + (0.92 - 0.6) / 0.4 * w, top - 16, "target 0.92", 11, PALETTE["red"]["text"], "middle", "700")
    for i, (name, rec, lo, hi, frr) in enumerate(SLICES):
        y = top + row * i + row / 2
        bad = name == "supplier C"
        colour = "red" if bad else "blue"
        label(b, left - 14, y + 4, name, 13, PALETTE[colour]["text"] if bad else INK, "end", "700" if bad else "400")
        xl, xh, xr = (left + (v - 0.6) / 0.4 * w for v in (lo, hi, rec))
        line(b, xl, y, xh, y, PALETTE[colour]["stroke"], 3)
        line(b, xl, y - 6, xl, y + 6, PALETTE[colour]["stroke"], 2)
        line(b, xh, y - 6, xh, y + 6, PALETTE[colour]["stroke"], 2)
        dot(b, xr, y, colour, 6)
        label(b, xh + 36, y + 4, f"{rec:.2f}", 12, PALETTE[colour]["text"], "middle", "700")
        if frr is not None:
            xf = 760 + frr / 0.5 * 300
            dot(b, xf, y, "red" if bad else "teal", 6)
            label(b, xf + 34, y + 4, f"{frr:.3f}", 12, PALETTE["red"]["text"] if bad else PALETTE["teal"]["text"], "middle", "700")
    b.card(60, 505, 1040, 40, "", ["Supplier C is the outlier on both sides: recall 0.80 and 40% of good coil crops rejected. Stains are the weakest defect type."], "yellow", size=12)
    return b


def defect_release():
    b = Board(1180, 540, "Release gate and rollback", "A candidate must beat the serving version on a golden set it never trained on")
    v1 = b.card(40, 120, 230, 110, "v1  serving", ["golden cost 0.384", "recall 0.925", "review share 0.208"], "blue", size=13)
    v2 = b.card(40, 270, 230, 110, "v2  candidate", ["more supplier C data", "cost 0.418, recall 0.909", "review share 0.191"], "orange", size=13)
    v3 = b.card(40, 410, 230, 110, "v3  candidate", ["v1 weights, C to manual", "cost 0.377, recall 0.952", "review share 0.286"], "purple", size=13)
    gate = b.diamond(500, 330, 210, 150, "gate", "yellow", size=15)
    b.card(350, 130, 300, 100, "Gate rules", ["recall >= 0.85 and no worse", "than serving minus 0.01", "cost <= serving + 0.01", "review share <= 0.30"], "grey", size=12)
    b.arrow(v2.right(), gate.left(0.35))
    b.arrow(v3.right(), gate.left(0.65))
    no = b.card(700, 250, 220, 90, "v2 refused", ["recall 0.909 is worse", "than 0.925"], "red", size=13)
    yes = b.card(700, 380, 220, 90, "v3 promoted", ["current.json = v3", "history = [v1]"], "green", size=13)
    b.arrow(gate.right(0.35), no.left(), label="fail")
    b.arrow(gate.right(0.65), yes.left(), label="pass")
    mon = b.card(960, 380, 190, 90, "Monitor alarm", ["PSI 7.03 on brightness", "recall falls to 0.83"], "teal", size=12)
    b.arrow(mon.left(), yes.right())
    rb = b.card(960, 250, 190, 90, "Rollback", ["rollback command", "current.json = v1"], "pink", size=12)
    b.arrow(mon.top(), rb.bottom(), label="decide")
    b.arrow(rb.left(), v1.right(), via=[(930, 200), (300, 175)], dashed=True, color="pink")
    return b


ALL = {
    "defect-architecture": defect_architecture,
    "defect-requirements": defect_requirements,
    "defect-worked-example": defect_worked_example,
    "defect-cost-curve": defect_cost_curve,
    "defect-slices": defect_slices,
    "defect-release": defect_release,
}


def main():
    for name, make in ALL.items():
        make().save(OUT / f"{name}.svg")
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
