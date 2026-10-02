"""Infographics for docs/governance (Track B, agent L).

Run from the repo root:

    python3 scripts/infographics/gov_1.py            # all boards
    python3 scripts/infographics/gov_1.py layers     # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "gov"
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


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=6, opacity=1.0):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def heat(b, x, y, w, h, value, label, size=13):
    t = max(0.0, min(1.0, value))
    r = int(235 + (224 - 235) * t)
    g = int(247 + (49 - 247) * t)
    bl = int(238 + (49 - 238) * t)
    rect(b, x, y, w, h, f"rgb({r},{g},{bl})", "#ced4da", 1, 3)
    raw_text(b, x + w / 2, y + h / 2 + size * 0.36, label, size, "#7a1212" if t > 0.6 else INK, weight="700" if t > 0.6 else "400")


@board("red-teaming-llm-systems-attack-surface")
def attack_surface():
    b = Board(1240, 740, "Where an attack enters an LLM application", "Four doors, five defence layers; the red numbers are the attack categories the harness runs")
    b.group(20, 95, 1200, 480, "The shop assistant under test (a stub)", "blue")
    user = b.card(45, 160, 190, 90, "user turn", ["chat messages,", "possibly several"], "grey", size=12)
    docs = b.card(45, 330, 190, 90, "retrieved text", ["FAQ pages, emails,", "web pages, tool output"], "grey", size=12)
    prompt = b.card(330, 215, 230, 130, "prompt assembly", ["system prompt with a", "secret key, user turn,", "retrieved text"], "purple", size=12)
    model = b.card(655, 215, 190, 130, "model", ["follows instructions", "wherever it finds", "them"], "orange", size=13)
    out = b.card(945, 160, 240, 90, "reply to the user", ["text shown, logged,", "maybe rendered"], "grey", size=12)
    tools = b.card(945, 330, 240, 90, "tool calls", ["send_email,", "issue_refund"], "grey", size=12)
    b.arrow(user.right(), prompt.left(0.3))
    b.arrow(docs.right(), prompt.left(0.7))
    b.arrow(prompt.right(), model.left())
    b.arrow(model.right(0.3), out.left())
    b.arrow(model.right(0.7), tools.left())

    pills = [(45, 125, "1", "direct injection, jailbreak"), (45, 435, "2", "indirect injection, poisoning"),
             (330, 180, "3", "exfiltration"), (945, 435, "4", "tool abuse")]
    for x, y, n, text in pills:
        b.pill(x, y, f"{n}  {text}", "red", size=11, solid=True)

    layers = [(60, 480, "input filter", "blocks known phrases in", "user turn and documents"),
              (275, 480, "delimit untrusted", "marks retrieved text as", "data, not instructions"),
              (490, 480, "no secrets in prompt", "nothing to leak if the", "key is never there"),
              (705, 480, "output filter", "scans the reply for the", "key and forbidden text"),
              (920, 480, "tool policy", "recipient allow-list,", "refund cap per call")]
    for x, y, t, l1, l2 in layers:
        b.card(x, y, 200, 52, t, [], "green", size=12)
        raw_text(b, x + 100, y + 68, l1, 10, FAINT)
        raw_text(b, x + 100, y + 81, l2, 10, FAINT)

    b.card(20, 605, 590, 115, "Why four doors matter", ["A filter on the user turn leaves retrieved text,", "the secret and the tool calls unguarded. Indirect", "injection needs no user: just text in a document."], "orange", size=13, align="left")
    b.card(630, 605, 590, 115, "What the harness measures", ["Attack success rate per category, each attack", "tried 10 times, scored by code that looks for the", "leaked key, the bad tool call or the planted fact."], "teal", size=13, align="left")
    return b


@board("red-teaming-llm-systems-layers")
def layers():
    b = Board(1240, 700, "What each defence layer buys", "Attack success rate, 8 seeded attacks x 10 trials per category, stubbed assistant")
    cols = ["direct", "jailbreak", "exfil", "indirect", "tools", "poison", "overall"]
    rows = [("no defences", [0.713, 0.713, 0.812, 0.850, 0.625, 0.800, 0.752]),
            ("input filter only", [0.300, 0.338, 0.200, 0.200, 0.188, 0.150, 0.229]),
            ("output filter only", [0.713, 0.000, 0.200, 0.850, 0.625, 0.800, 0.531]),
            ("tool policy only", [0.713, 0.713, 0.812, 0.000, 0.037, 0.800, 0.512]),
            ("delimit untrusted only", [0.713, 0.713, 0.812, 0.175, 0.625, 0.188, 0.537]),
            ("no secrets only", [0.713, 0.713, 0.000, 0.850, 0.625, 0.800, 0.617]),
            ("all five layers", [0.300, 0.000, 0.000, 0.000, 0.037, 0.025, 0.060])]
    x0, y0, lw, cw, rh = 82, 120, 250, 118, 42
    rect(b, x0, y0 - 40, lw + cw * len(cols), 40, PALETTE["blue"]["stroke"], PALETTE["blue"]["stroke"], 1, 6)
    raw_text(b, x0 + 12, y0 - 14, "configuration", 13, "#ffffff", "start", "700")
    for i, c in enumerate(cols):
        raw_text(b, x0 + lw + cw * i + cw / 2, y0 - 14, c, 13, "#ffffff", weight="700")
    for r, (name, vals) in enumerate(rows):
        y = y0 + r * rh
        raw_text(b, x0 + 12, y + rh / 2 + 5, name, 13, INK, "start", "700" if r in (0, 6) else "400")
        for i, v in enumerate(vals):
            heat(b, x0 + lw + cw * i + 3, y + 3, cw - 6, rh - 6, v, f"{v:.3f}")
    b.card(40, 440, 380, 120, "Layers are not interchangeable", ["The output filter ends jailbreak output", "(0.713 to 0.000) and does nothing", "for indirect injection (0.850)."], "purple", size=12, align="left")
    b.card(430, 440, 380, 120, "Residual: 29 of 480 still land", ["24 direct-injection override attempts,", "3 split refunds that pass the per-call cap,", "2 base64 poisoned documents."], "red", size=12, align="left")
    b.card(820, 440, 380, 120, "Not every goal can be filtered", ["Making the bot say a word leaks nothing", "and calls no tool, so no layer here", "stops it: 0.300 stays."], "orange", size=12, align="left")
    b.card(40, 585, 1160, 90, "Zero successes is not zero risk", ["0 of 20 probes: true rate could still be 13.9% (95% one-sided).  0 of 59 probes brings the bound just under 5%.", "A defence with a true rate of 4% shows zero successes in 25 probes 36.0% of the time."], "yellow", size=12)
    return b


@board("fairness-testing-in-practice-metrics")
def fairness_metrics():
    b = Board(1240, 690, "Four group metrics, one model", "20,000 synthetic applicants, logistic regression, one threshold of 0.30, test half of 10,000 rows")
    b.group(20, 95, 640, 330, "What each metric compares between groups", "blue")
    rows = [["metric", "group A", "group B", "gap"],
            ["selection rate", "0.352", "0.275", "0.077"],
            ["true positive rate", "0.579", "0.575", "0.004"],
            ["false positive rate", "0.248", "0.215", "0.033"],
            ["precision", "0.515", "0.349", "0.166"],
            ["base rate (actual)", "0.312", "0.167", "0.145"]]
    b.table(40, 140, [230, 130, 130, 110], rows, "blue", size=14, row_h=40)
    b.text(340, 398, "demographic parity gap 0.0765 | equalised odds gap 0.0332", 12, "blue", italic=True)

    b.group(680, 95, 540, 330, "Calibration within groups", "purple")
    rows2 = [["score band", "mean score", "observed A", "observed B"],
             ["(0.0, 0.2]", "0.12 / 0.11", "0.148", "0.067"],
             ["(0.2, 0.4]", "0.29 / 0.29", "0.343", "0.206"],
             ["(0.4, 0.6]", "0.48 / 0.48", "0.540", "0.401"],
             ["(0.6, 1.0]", "0.69 / 0.68", "0.758", "0.617"]]
    b.table(700, 140, [120, 140, 120, 120], rows2, "purple", size=13, row_h=40)
    b.card(700, 345, 500, 66, "same score, different outcome", ["at score 0.29: A observed 0.343,", "B observed 0.206"], "red", size=12)

    b.card(20, 450, 400, 110, "Equal opportunity roughly holds", ["TPR 0.579 against 0.575: qualified people", "in both groups are found at the same rate."], "green", size=13, align="left")
    b.card(440, 450, 400, 110, "Predictive parity does not", ["Precision 0.515 against 0.349: a flag", "means less in group B."], "orange", size=13, align="left")
    b.card(860, 450, 360, 110, "Why they cannot all match", ["FPR = base rate odds x", "(1 - precision) / precision x TPR"], "purple", size=13, align="left")
    b.card(20, 585, 1200, 85, "Equal precision 0.70 and equal TPR 0.60 with base rates 0.312 and 0.167", ["force a false positive rate of 0.117 in group A and 0.052 in group B. Only equal base rates (or a perfect model) let all three hold."], "yellow", size=13)
    return b


@board("fairness-testing-in-practice-mitigation")
def fairness_mitigation():
    b = Board(1280, 700, "Mitigations move one gap and cost another", "40,000 synthetic applicants; every row measured on the same 10,000 held-out rows")
    cols = ["method", "accuracy", "sel A", "sel B", "DP gap", "EO gap"]
    rows = [("baseline, threshold 0.50", [0.764, 0.117, 0.076, 0.041, 0.050]),
            ("baseline, threshold 0.30", [0.710, 0.376, 0.271, 0.105, 0.092]),
            ("per-group thresholds, equal selection", [0.701, 0.340, 0.330, 0.010, 0.038]),
            ("ThresholdOptimizer, parity", [0.759, 0.061, 0.060, 0.001, 0.017]),
            ("ThresholdOptimizer, equalised odds", [0.758, 0.055, 0.041, 0.014, 0.028]),
            ("ExponentiatedGradient, parity", [0.764, 0.103, 0.089, 0.014, 0.011]),
            ("group added as a feature, 0.30", [0.719, 0.459, 0.156, 0.302, 0.324])]
    x0, y0, lw, cw, rh = 60, 120, 460, 140, 42
    rect(b, x0, y0 - 40, lw + cw * 5, 40, PALETTE["teal"]["stroke"], PALETTE["teal"]["stroke"], 1, 6)
    raw_text(b, x0 + 12, y0 - 14, cols[0], 13, "#ffffff", "start", "700")
    for i, c in enumerate(cols[1:]):
        raw_text(b, x0 + lw + cw * i + cw / 2, y0 - 14, c, 13, "#ffffff", weight="700")
    for r, (name, vals) in enumerate(rows):
        y = y0 + r * rh
        raw_text(b, x0 + 12, y + rh / 2 + 5, name, 13, INK, "start")
        for i, v in enumerate(vals):
            heat(b, x0 + lw + cw * i + 3, y + 3, cw - 6, rh - 6, min(1.0, v * 3.2) if i >= 3 else 0.0, f"{v:.3f}")
    b.card(40, 440, 380, 100, "Pre-, in- or post-processing", ["Post: thresholds per group.", "In: a constrained learner.", "Pre: reweigh or repair the data."], "blue", size=12, align="left")
    b.card(440, 440, 380, 100, "Parity is cheap here", ["ExponentiatedGradient cuts the DP gap", "0.041 to 0.014 at the same 0.764 accuracy."], "green", size=12, align="left")
    b.card(840, 440, 400, 100, "Awareness is not fairness", ["Adding the group as a feature gives a DP gap", "of 0.302: the model now sees the real base rate."], "red", size=12, align="left")
    b.card(40, 565, 1200, 110, "Small slices invent gaps: two groups with the same true TPR of 0.60", ["20 positives per group: mean gap 0.123, 45.6% of audits show a gap above 0.10.   200 positives: 0.039 and 3.7%.   5,000 positives: 0.008 and 0%.", "Three attributes with 4, 4 and 5 values make 80 slices: expect 4.0 spurious flags at a 5% false-alarm rate each."], "yellow", size=12)
    return b


@board("privacy-pii-and-differential-privacy-anonymisation")
def privacy_anon():
    b = Board(1240, 700, "Detecting and hiding identity", "Synthetic text and a synthetic table; every number is printed by blocks 1 and 2 of the chapter")
    b.group(20, 95, 580, 330, "Regex PII detectors on 400 synthetic lines", "blue")
    rows = [["type", "precision", "recall", "note"],
            ["email", "1.000", "1.000", "structured"],
            ["phone", "1.000", "1.000", "one format"],
            ["card, no check", "0.488", "1.000", "104 false alarms"],
            ["card, Luhn", "0.900", "1.000", "11 false alarms"],
            ["name (Mr, Ms, Dr)", "1.000", "0.334", "197 names missed"]]
    b.table(40, 140, [170, 110, 100, 170], rows, "blue", size=13, row_h=40)
    b.text(310, 400, "plain names need a trained recogniser, not a pattern", 12, "blue", italic=True)

    b.group(620, 95, 600, 330, "k-anonymity by generalising a 3,000-row table", "purple")
    rows2 = [["generalisation", "unique as is", "rows cut for k=5", "re-id after"],
             ["zip 5 digits, age 1", "0.985", "1.000", "0.000"],
             ["zip 4 digits, age 5", "0.539", "0.991", "0.000"],
             ["zip 3 digits, age 5", "0.017", "0.269", "0.000"],
             ["zip 3 digits, age 20", "0.003", "0.028", "0.000"]]
    b.table(640, 140, [190, 115, 145, 110], rows2, "purple", size=12, row_h=40)
    b.card(640, 350, 560, 62, "the cost of anonymity", ["safer tables lose rows or detail: 0.991 of rows at zip 4 digits"], "orange", size=12)

    b.card(20, 450, 400, 120, "Mask or pseudonymise", ["Masking gives [EMAIL]. A keyed hash gives", "[EMAIL:b27510]: the same address maps to", "the same token, so joins still work."], "green", size=12, align="left")
    b.card(440, 450, 360, 120, "k-anonymity is not enough", ["After suppressing to k = 5, 23 groups hold", "283 of 2,917 people, and 90% or more of", "each group share one diagnosis."], "red", size=12, align="left")
    b.card(820, 450, 400, 120, "Linkage beats removal of names", ["Without generalisation 0.985 of people are", "unique on zip, age and sex, so a public list", "with those three columns names them."], "orange", size=12, align="left")
    b.card(20, 595, 1200, 85, "Automated detection finds most, never all", ["Presidio's documentation gives no guarantee of finding all sensitive information: add retention limits and access control."], "yellow", size=13)
    return b


@board("privacy-pii-and-differential-privacy-dp")
def privacy_dp():
    b = Board(1280, 720, "Epsilon, noise and what a leak attack sees", "Laplace counting query (block 3) and DP-SGD against a loss-threshold membership attack (block 4)")
    b.group(20, 95, 560, 300, "Laplace mechanism, count of 3,331", "teal")
    rows = [["epsilon", "noise scale", "mean abs error"],
            ["0.01", "100.0", "100.98"],
            ["0.1", "10.0", "9.95"],
            ["0.5", "2.0", "1.99"],
            ["1.0", "1.0", "1.00"],
            ["5.0", "0.2", "0.20"]]
    b.table(40, 140, [150, 170, 200], rows, "teal", size=14, row_h=38)
    b.text(300, 386, "scale = sensitivity / epsilon; sensitivity of a count is 1", 12, "teal", italic=True)

    b.group(600, 95, 660, 300, "DP-SGD against a membership attack (mean of 8 seeds)", "purple")
    rows2 = [["model", "test acc", "attack AUC", "epsilon"],
             ["plain training", "0.585", "0.865", "none"],
             ["noise multiplier 0.5", "0.583", "0.830", "136.05"],
             ["noise multiplier 1.0", "0.571", "0.781", "27.16"],
             ["noise multiplier 2.0", "0.552", "0.691", "8.94"],
             ["noise multiplier 4.0", "0.528", "0.614", "3.74"]]
    b.table(620, 140, [240, 120, 140, 120], rows2, "purple", size=13, row_h=38)
    b.text(930, 386, "100 rows, 200 features, 1,000 steps, delta 1e-5", 12, "purple", italic=True)

    b.card(20, 420, 400, 130, "Composition adds epsilons", ["One count at epsilon 0.1 has error 9.23.", "Ten releases cost epsilon 1.0 and a", "thousand cost 100.0 (error 0.34): the", "average removes the noise."], "orange", size=12, align="left")
    b.card(440, 420, 400, 130, "The guarantee", ["Adding or removing one person changes", "any output probability by at most a", "factor e^epsilon: 1.649 at epsilon 0.5,", "1/1.649 = 0.607 the other way."], "green", size=12, align="left")
    b.card(860, 420, 400, 130, "Privacy costs accuracy", ["At 100 rows, noise multiplier 4 cuts the", "attack AUC from 0.865 to 0.614 and test", "accuracy from 0.585 to 0.528. Small data", "makes privacy expensive."], "red", size=12, align="left")
    b.card(20, 575, 1240, 120, "Epsilon is a bound, the attack AUC is one measurement", ["Epsilon 136 is a weak guarantee and still cut the attack from 0.865 to 0.830. A low attack AUC does not prove privacy: a stronger attack", "may do better, which is why the epsilon accountant matters. Carlini and colleagues extracted hundreds of verbatim training sequences from GPT-2 with queries alone."], "yellow", size=12)
    return b


@board("regulation-and-model-documentation-timeline")
def reg_timeline():
    b = Board(1400, 760, "EU AI Act: what applies when", "Regulation (EU) 2024/1689 as amended by Regulation (EU) 2026/1744 (Digital Omnibus on AI), dates read from the Official Journal texts, October 2026")
    events = [
        ("1 Aug 2024", "entry into force", "of the Act (OJ 12 July 2024)", "grey", 0),
        ("2 Feb 2025", "Chapters I and II", "prohibitions (Art 5), AI literacy (Art 4)", "red", 1),
        ("2 Aug 2025", "GPAI, governance,", "penalties (not Art 101), notified bodies", "purple", 0),
        ("27 Jul 2026", "Omnibus in force", "published 24 July 2026; Arts 102 to 110 apply", "grey", 1),
        ("2 Aug 2026", "general date", "Art 50 transparency; Art 101 GPAI fines", "orange", 0),
        ("2 Dec 2026", "new prohibitions", "Art 5(1)(ba), (bb); Art 50(2) for older gen AI", "red", 1),
        ("2 Dec 2027", "high-risk, Annex III", "Chapter III Sections 1 to 3", "blue", 0),
        ("2 Aug 2028", "high-risk, Annex I", "products under Article 6(1)", "teal", 1),
    ]
    x0, x1, ay = 120, 1280, 380
    line(b, x0 - 20, ay, x1 + 20, ay, INK, 3)
    step = (x1 - x0) / (len(events) - 1)
    for i, (d, t1, t2, col, up) in enumerate(events):
        cx = x0 + i * step
        c = PALETTE[col]
        b.parts.append(f'<circle cx="{cx:.1f}" cy="{ay}" r="9" fill="{c["stroke"]}" stroke="#fffdf7" stroke-width="3"/>')
        y = 130 if up == 0 else 430
        line(b, cx, y + 120 if up == 0 else y, cx, ay - 10 if up == 0 else ay + 10, c["stroke"], 2, "4 4")
        b.card(cx - 85, y, 170, 120, d, [t1, t2], col, size=11, title_size=14)
    b.card(20, 590, 450, 150, "Fines (Article 99)", ["EUR 35m or 7% for prohibited practices", "EUR 15m or 3% for most operator duties,", "including Article 50; EUR 7.5m or 1% for", "wrong information. SMEs: the lower figure."], "red", size=12, align="left")
    b.card(490, 590, 430, 150, "Later dates in the text", ["2 Aug 2027: GPAI models placed before", "2 Aug 2025 must comply (Art 111(3)).", "2 Aug 2030: high-risk systems for public", "authorities placed earlier (Art 111(2))."], "purple", size=12, align="left")
    b.card(940, 590, 440, 150, "What moved in the Omnibus", ["Annex III high-risk: 2 Aug 2026 became", "2 Dec 2027. Annex I: 2 Aug 2028.", "Article 6(5) is excluded from the delay.", "AI literacy (Art 4) reworded."], "orange", size=12, align="left")
    return b


@board("regulation-and-model-documentation-tiers")
def reg_tiers():
    b = Board(1300, 760, "Risk tiers, frameworks and the paper trail", "A teaching summary of the Act's structure, three voluntary frameworks and what a model card records; not legal advice")
    tiers = [("prohibited", "Art 5", "social scoring, workplace emotion inference, untargeted face scraping", "red", 600),
             ("high-risk", "Art 6, Annex I and III", "risk management, data governance, documentation, logging, oversight, accuracy", "orange", 500),
             ("transparency", "Art 50", "tell people it is an AI; mark synthetic content; label deep fakes", "yellow", 420),
             ("general-purpose models", "Chapter V", "documentation, copyright policy, training summary; systemic risk above 10^25 FLOP adds testing", "purple", 540),
             ("minimal", "no specific duty", "spam filters and the like; AI literacy still applies (Art 4)", "green", 380)]
    for i, (name, art, what, col, w) in enumerate(tiers):
        y = 105 + i * 78
        x = 20 + (600 - w) / 2
        b.card(x, y, w, 66, f"{name}  ({art})", [what], col, size=11, title_size=13)
    b.group(660, 95, 620, 240, "Voluntary frameworks", "blue")
    b.card(680, 140, 185, 180, "NIST AI RMF 1.0", ["Jan 2023", "GOVERN, MAP,", "MEASURE, MANAGE", "GenAI profile:", "AI 600-1, July 2024"], "blue", size=11)
    b.card(880, 140, 185, 180, "ISO/IEC 42001", ["published", "18 Dec 2023", "AI management", "system standard,", "51 pages"], "teal", size=11)
    b.card(1080, 140, 185, 180, "Model card", ["Mitchell et al.", "9 sections,", "2019 (FAT*)", "datasheets for", "datasets: 2018"], "purple", size=11)
    b.group(660, 355, 620, 200, "A model card has nine sections", "purple")
    secs = ["Model details", "Intended use", "Factors", "Metrics", "Evaluation data", "Training data", "Quantitative analyses", "Ethical considerations", "Caveats and recommendations"]
    for i, t in enumerate(secs):
        b.pill([675, 855, 1045][i % 3], 400 + (i // 3) * 48, f"{i + 1} {t}", "purple", size=11)
    b.group(20, 520, 620, 215, "Audit trail: hash-chained log", "green")
    b.card(40, 565, 580, 70, "edit entry 2: verification fails at position 2", ["delete entry 1: fails at position 1"], "green", size=12)
    b.card(40, 650, 580, 70, "rewrite the last entry and recompute its hash: passes", ["only a copy of the previous head held elsewhere exposes it"], "red", size=12)
    b.card(660, 575, 620, 160, "Deployers keep logs at least six months", ["Art 12 requires high-risk systems to record events;", "Art 26(6) sets at least six months for deployers", "and Art 19 the same for providers, unless other", "law provides otherwise. Art 73: serious incidents", "reported within 15 days (10 for a death, 2 for widespread)."], "orange", size=12, align="left")
    return b


@board("ai-incident-response-lifecycle")
def inc_lifecycle():
    b = Board(1300, 740, "Five steps of AI incident response, and three real write-ups", "Severity scores are from the chapter's teaching rubric; incident facts are from the cited write-ups")
    steps = [("1 detect", "monitors, tickets,", "red team findings", "red"), ("2 triage", "score severity,", "name a commander", "orange"),
             ("3 contain", "kill switch, rollback,", "narrow the tools", "yellow"), ("4 communicate", "users, regulators,", "internal channel", "teal"),
             ("5 learn", "root cause, blameless", "review, regression test", "green")]
    prev = None
    for i, (t, l1, l2, col) in enumerate(steps):
        c = b.card(30 + i * 252, 105, 220, 96, t, [l1, l2], col, size=12, title_size=15)
        if prev:
            b.arrow(prev.right(), c.left())
        prev = c
    b.group(20, 225, 620, 270, "Severity rubric (scores add up)", "blue")
    rows = [["dimension", "scale", "max"], ["scope", "one 1, some 3, many 5", "5"], ["data", "none 0, personal 3, payment or health 5", "5"],
            ["harm", "embarrassment 1, financial 3, safety or legal 5", "5"], ["reversible", "fully 0, with effort 2, no 4", "4"], ["public", "internal 0, customers 1, press or regulator 3", "3"]]
    b.table(35, 270, [110, 410, 60], rows, "blue", size=11, row_h=34)
    b.text(330, 485, "SEV1 from 14, SEV2 from 9, SEV3 from 5, SEV4 below", 12, "blue", italic=True)
    b.group(660, 225, 620, 270, "Scored examples", "purple")
    rows2 = [["incident", "score", "level"],
             ["OpenAI, Mar 2023: titles and payment details", "15", "SEV1"],
             ["Air Canada, 2024: invented refund rule", "9", "SEV2"],
             ["Tay, 2016: coordinated abuse", "9", "SEV2"],
             ["typo in a canned answer", "3", "SEV4"],
             ["document leaked to one employee", "6", "SEV3"]]
    b.table(675, 270, [400, 90, 100], rows2, "purple", size=11, row_h=34)
    b.card(20, 515, 400, 205, "OpenAI, March 2023", ["A bug in an open-source Redis client library", "let some users see another user's chat titles.", "Payment details of 1.2% of ChatGPT Plus", "subscribers active in a nine-hour window were", "possibly visible. Service taken offline, patched,", "affected users notified."], "orange", size=11, align="left")
    b.card(440, 515, 400, 205, "Air Canada, February 2024", ["The tribunal rejected the idea that the chatbot", "was a separate legal entity and held the airline", "responsible for what it told a customer about", "bereavement fares. Total ordered: $812.02,", "of which $650.88 damages."], "red", size=11, align="left")
    b.card(860, 515, 420, 205, "Microsoft Tay, March 2016", ["Microsoft said a coordinated attack by a subset", "of people exploited a vulnerability, and that it", "had made a critical oversight for this specific", "attack. The lesson: a red team that never imagined", "the attack cannot test the defence."], "purple", size=11, align="left")
    return b


@board("ai-incident-response-minutes")
def inc_minutes():
    b = Board(1300, 740, "Where the minutes go", "A worked incident and 40 seeded incidents; containment minutes are model parameters, not measurements")
    b.group(20, 95, 620, 300, "Worked incident: harm starts 09:00", "teal")
    phases = [("detect", 47, "teal"), ("ack", 5, "blue"), ("decide", 18, "purple"), ("contain", 15, "orange")]
    x = 45
    total = 85
    for name, m, col in phases:
        w = m / total * 570
        c = PALETTE[col]
        b.parts.append(f'<rect x="{x:.1f}" y="150" width="{w:.1f}" height="60" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2"/>')
        raw_text(b, x + w / 2, 177, f"{m}", 15, c["text"], weight="700")
        raw_text(b, x + w / 2, 196, name, 11, c["text"])
        x += w
    raw_text(b, 330, 250, "harm window 85 minutes = 1,020 harmful responses at 12 per minute", 13, INK, weight="700")
    rows = [["containment", "min", "window", "responses"], ["kill switch", "2", "72", "864"], ["rollback", "15", "85", "1,020"], ["hotfix", "120", "190", "2,280"]]
    b.table(45, 270, [170, 90, 130, 150], rows, "teal", size=12, row_h=28)
    b.group(660, 95, 620, 300, "40 seeded incidents", "purple")
    rows2 = [["phase", "median min", "90th pct", "share of window"], ["detect", "29.3", "138.6", "41.0%"], ["acknowledge", "6.0", "13.4", "5.3%"],
             ["decide", "16.7", "39.7", "15.2%"], ["contain", "17.3", "168.2", "38.5%"]]
    b.table(680, 145, [170, 130, 120, 160], rows2, "purple", size=12, row_h=34)
    b.text(970, 350, "total window: median 111 min, 90th percentile 242 min", 12, "purple", italic=True)
    b.group(20, 415, 1260, 150, "Detecting a shift from 0.5% to 4.0% harmful outputs (300 runs)", "orange")
    rows3 = [["detector", "false alarms", "mean delay", "95th pct"], ["rolling mean of 200 >= 0.03", "0.020", "143.5", "276.1"], ["likelihood-ratio CUSUM, limit 7", "0.023", "131.6", "296.2"]]
    b.table(40, 460, [400, 150, 150, 150], rows3, "orange", size=12, row_h=30)
    b.card(900, 458, 360, 90, "a lower threshold alarms sooner", ["rolling >= 0.02: delay 81.9 requests,", "false alarms in 47.7% of runs"], "red", size=11)
    b.card(20, 585, 1260, 135, "Lessons the numbers carry", ["A kill switch saved 13 minutes here; detection was the largest slice of the mean window (41.0%), ahead of containment (38.5%).", "Communication began 35 minutes after containment, root cause came 29.6 hours later and the review on day 4: do those in parallel, not in sequence.", "Alarm settings are a trade between delay and false alarms; report both."], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
