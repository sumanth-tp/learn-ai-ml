"""Infographics for docs/senior/02-engineering-craft.

Run from the repo root:

    python3 scripts/infographics/senior_3.py            # all boards
    python3 scripts/infographics/senior_3.py eval-noise # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "senior"
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


def rect(b, x, y, w, h, color, width=1.6, opacity=1.0):
    c = PALETTE[color]
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{c["fill"]}" '
        f'stroke="{c["stroke"]}" stroke-width="{width}" opacity="{opacity}"/>'
    )


@board("build-vs-buy-and-model-selection-break-even")
def bb_break_even():
    b = Board(1200, 640, "Build or buy: where hosting starts to pay",
              "Assumed prices and throughput (replace them with your quotes); 1,500 input and 300 output tokens per request")
    b.group(20, 95, 650, 335, "Monthly cost at five volumes", "teal")
    rows = [["requests / month", "replicas", "buy (API)", "host", "cheaper"],
            ["100,000", "2", "1,140", "12,050", "buy"],
            ["1,000,000", "2", "11,400", "20,150", "buy"],
            ["5,000,000", "2", "57,000", "56,150", "host"],
            ["20,000,000", "8", "228,000", "202,100", "host"],
            ["100,000,000", "39", "1,140,000", "978,675", "host"]]
    b.table(40, 140, [165, 95, 125, 130, 105], rows, "teal", size=13, row_h=40)
    b.text(345, 410, "buy: tokens + errors at 6%   |   host: GPUs + 0.5 engineer + errors at 9%", 11, "teal", italic=True)

    b.group(695, 95, 485, 335, "Break-even volume, requests per month", "purple")
    b.card(715, 140, 445, 82, "no quality gap", ["2,064,815", "open model as accurate as the API"], "blue", size=12, title_size=17)
    b.card(715, 237, 445, 82, "3 points more errors, 0.10 per error", ["4,645,833", "the gap eats more than half the saving"], "orange", size=12, title_size=17)
    b.card(715, 334, 445, 80, "API output price halved", ["no break-even below 500,000,000", "the saving per request falls under the gap"], "green", size=12, title_size=17)

    b.card(20, 455, 380, 165, "Hosting is a staircase", ["Replicas are whole GPUs:", "2 at 5M requests, 8 at 20M,", "39 at 100M. The minimum of 2", "is there for availability."], "blue", size=12)
    b.card(410, 455, 380, 165, "Errors have a price", ["API cost per request is 0.0054.", "3 extra points at 0.10 per error", "add 0.0030 to hosting: more", "than half of what it saves."], "orange", size=12)
    b.card(800, 455, 380, 165, "Buying has a clock", ["Docs opened 2026-10-02: Sonnet 4.5", "deprecated 2026-09-30, retires", "2026-11-30. At least 60 days notice;", "keep prompts and evals portable."], "red", size=12)
    return b


@board("build-vs-buy-and-model-selection-eval-noise")
def bb_noise():
    b = Board(1200, 620, "Is the better model really better?", "300 graded items, three candidate models; 95% intervals on the gap between scores")
    b.group(20, 95, 710, 410, "Gap in pass rate, 95% interval", "blue")
    x0, scale = 105.0, 2800.0
    px = lambda v: x0 + (v + 0.05) * scale
    ay = 465
    line(b, px(-0.05), ay, px(0.15), ay, FAINT, 2)
    for v in (-0.05, 0.0, 0.05, 0.10, 0.15):
        line(b, px(v), ay - 5, px(v), ay + 5, FAINT, 1.4)
        raw_text(b, px(v), ay + 22, f"{v:+.2f}" if v else "0", 11, FAINT)
    line(b, px(0), 135, px(0), ay, "#e03131", 1.8, "5 4")
    items = [("A against B, unpaired", 0.067, -0.008, 0.142, "grey", 175),
             ("A against B, paired", 0.067, 0.020, 0.113, "green", 235),
             ("A against C, unpaired", 0.047, -0.028, 0.121, "grey", 320),
             ("A against C, paired", 0.047, -0.001, 0.094, "orange", 380)]
    for label, mid, lo, hi, col, y in items:
        c = PALETTE[col]
        raw_text(b, 45, y - 22, label, 12, c["text"], anchor="start", weight="700")
        line(b, px(lo), y, px(hi), y, c["stroke"], 5)
        dot(b, px(mid), y, 7, c["fill"], c["stroke"])
        raw_text(b, px(hi) + 10, y + 4, f"[{lo:+.3f}, {hi:+.3f}]", 11, INK, anchor="start")
    raw_text(b, px(0) + 6, 150, "no difference", 11, "#c92a2a", anchor="start")

    b.group(750, 95, 430, 410, "Items needed for 80% power, 5% size", "orange")
    rows = [["gap to detect", "items"],
            ["0.100", "133"],
            ["0.067", "296"],
            ["0.030", "1,478"],
            ["0.020", "3,325"]]
    b.table(770, 140, [200, 190], rows, "orange", size=14, row_h=40)
    b.card(770, 360, 390, 125, "300 items can detect 0.067", ["A against B clears it (paired CI above 0).", "A against C, a gap of 0.047, does not:", "the interval touches zero."], "yellow", size=12)

    b.card(20, 525, 1160, 80, "Pairing is free", ["both models answered the same 300 items and their item scores correlate (0.615 and 0.589), so differencing item by item narrows the interval"], "green", size=12)
    return b


@board("cost-modelling-and-roi-agent-loop")
def roi_agent():
    b = Board(1200, 640, "An agent resends its history every step", "Measured with tiktoken o200k_base: prefix 1,154 tokens, each tool step adds 172, one answer is 55 tokens")
    b.group(20, 95, 520, 330, "Prompt tokens at each of six steps", "blue")
    base_y, per = 395.0, 0.115
    for i in range(6):
        x = 50 + i * 82
        prefix = 1154 * per
        added = 172 * i * per
        rect(b, x, base_y - prefix, 60, prefix, "blue")
        if i:
            rect(b, x, base_y - prefix - added, 60, added, "orange")
        raw_text(b, x + 30, base_y + 18, f"step {i + 1}", 11, INK, weight="700")
        raw_text(b, x + 30, base_y - prefix - added - 8, f"{1154 + 172 * i:,}", 11, INK)
    b.text(280, 150, "blue: the stable prefix   orange: history added", 11, FAINT, italic=True)

    b.group(560, 95, 620, 330, "Cost of one task, assumed prices", "teal")
    rows = [["steps", "input tokens", "no cache", "cached", "cached / plain"],
            ["1", "1,154", "0.00275", "0.00332", "1.21"],
            ["3", "3,978", "0.00928", "0.00556", "0.60"],
            ["6", "9,504", "0.02165", "0.00917", "0.42"],
            ["10", "19,280", "0.04296", "0.01447", "0.34"],
            ["20", "55,760", "0.12032", "0.03012", "0.25"]]
    b.table(580, 140, [80, 140, 120, 120, 130], rows, "teal", size=13, row_h=40)
    b.text(870, 410, "input 2.00 and output 8.00 per million tokens, assumed", 11, "teal", italic=True)

    b.card(20, 450, 380, 170, "Total input grows with steps squared", ["6 steps bill 9,504 input tokens,", "20 steps bill 55,760: 3.3 times", "the steps, 5.9 times the tokens."], "orange", size=12)
    b.card(410, 450, 380, 170, "Caching pays from step 2", ["Anthropic docs, opened 2026-10-02:", "write 1.25x, read 0.1x the input price", "(some newer models differ). One step", "costs 21% more cached."], "green", size=12)
    b.card(800, 450, 380, 170, "Check the minimum", ["A prefix below the model's minimum", "cacheable length is not cached at all;", "the docs list 512 to 4,096 tokens", "depending on the model."], "red", size=12)
    return b


@board("cost-modelling-and-roi-roi-distribution")
def roi_dist():
    b = Board(1200, 660, "One ROI number hides the risk", "Support-assist feature, 120 users, 24 months, 10% a year discount; every input is an assumption")
    b.group(20, 95, 380, 360, "Base case", "blue")
    cards = [("monthly benefit", "16,200"), ("monthly cost", "6,514"), ("monthly net", "9,686"), ("payback", "9.3 months"), ("24-month NPV", "120,810"), ("24-month ROI", "1.58")]
    for i, (t, v) in enumerate(cards):
        b.card(40 + (i % 2) * 175, 140 + (i // 2) * 100, 165, 85, t, [v], "blue" if i < 3 else "green", size=14, title_size=12)

    b.group(420, 95, 420, 360, "What moves the NPV (low to high)", "orange")
    tor = [("realisation 0.3 to 0.9", 352586), ("minutes saved 1 to 3.5", 352586), ("adoption 0.3 to 0.85", 317935),
           ("rework 0.5 to 0.1", 188046), ("fixed run cost 10,000 to 4,000", 130587), ("build cost 160,000 to 90,000", 70000)]
    for i, (label, swing) in enumerate(tor):
        y = 150 + i * 49
        raw_text(b, 440, y, label, 11, INK, anchor="start", weight="700")
        w = swing / 352586 * 330
        rect(b, 440, y + 8, w, 20, "orange")
        raw_text(b, 440 + w + 8, y + 23, f"{swing:,}", 11, INK, anchor="start")

    b.group(860, 95, 320, 360, "Monte Carlo, 20,000 draws", "purple")
    b.card(880, 140, 280, 75, "median NPV", ["18,562"], "purple", size=14, title_size=12)
    b.card(880, 228, 280, 75, "10th to 90th percentile", ["-103,410 to 194,734"], "purple", size=13, title_size=12)
    b.card(880, 316, 280, 120, "chance it pays", ["NPV above zero: 0.564", "payback within a year: 0.257"], "red", size=13, title_size=12)

    b.card(20, 480, 590, 160, "Why the two answers differ", ["The base case multiplies the middle value of every input.", "The inputs are skewed (a build overrun has no upside),", "and the result is a product of them, so the typical", "outcome sits far below the arithmetic one."], "yellow", size=12)
    b.card(630, 480, 550, 160, "Spend the next day on the top bars", ["Realisation and minutes saved swing the NPV by", "352,586 each, build cost by 70,000. Measure", "those two in a pilot before polishing the build plan."], "green", size=12)
    return b


@board("estimation-and-planning-sum-of-ranges")
def est_ranges():
    b = Board(1200, 640, "Summing the likely values undercounts", "Eight tasks in working days: optimistic, likely, pessimistic; one engineer, one after another")
    b.group(20, 95, 700, 400, "Task ranges", "blue")
    tasks = [("data audit", 2, 3, 8), ("label the eval set", 4, 6, 15), ("baseline", 3, 5, 12), ("eval harness", 3, 4, 7),
             ("iterate to target", 5, 10, 30), ("integration", 4, 6, 12), ("safety review", 2, 3, 8), ("rollout", 2, 3, 6)]
    x0, sc = 230.0, 12.0
    for i, (name, lo, mode, hi) in enumerate(tasks):
        y = 150 + i * 42
        raw_text(b, 40, y + 4, name, 12, INK, anchor="start", weight="700")
        line(b, x0 + lo * sc, y, x0 + hi * sc, y, PALETTE["blue"]["stroke"], 5)
        dot(b, x0 + mode * sc, y, 7, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"])
        raw_text(b, x0 + hi * sc + 8, y + 4, f"{lo} / {mode} / {hi}", 11, FAINT, anchor="start")

    b.group(740, 95, 440, 400, "The total, in days", "orange")
    rows = [["measure", "days"],
            ["sum of likely values", "40"],
            ["sum of PERT means", "47.2"],
            ["simulated P50", "53.9"],
            ["simulated P85", "61.7"],
            ["simulated P95", "66.2"]]
    b.table(760, 140, [250, 140], rows, "orange", size=14, row_h=40)
    b.card(760, 385, 400, 90, "Finish within 40 days", ["0.6% of 100,000 simulated projects"], "red", size=13)

    b.card(20, 515, 560, 105, "Right-skewed tasks", ["Each task can run over much more than it can", "run under, so the sum drifts above the sum of modes."], "yellow", size=12)
    b.card(600, 515, 580, 105, "A date needs a percentile", ["Quote the P50 for planning and the P85 for a promise;", "never the sum of likely values."], "green", size=12)
    return b


@board("estimation-and-planning-shared-risk")
def est_shared():
    b = Board(1200, 660, "Shared risk widens the tail", "Three tasks run 1.5 times longer if the data is worse than assumed; 35% chance; 4,000 simulated projects")
    b.group(20, 95, 640, 270, "Same average, different tail (days)", "blue")
    rows = [["scenario", "mean", "P50", "P85", "P95"],
            ["data as expected", "54.3", "53.8", "61.7", "66.4"],
            ["35% bad, one shared draw", "59.6", "58.1", "70.9", "81.2"],
            ["35% bad, independent draws", "59.5", "58.6", "69.0", "76.4"]]
    b.table(40, 140, [270, 80, 80, 80, 80], rows, "blue", size=13, row_h=42)
    b.card(40, 320, 600, 38, "Shared: one verdict on the data decides all three tasks", [], "orange", size=12)

    b.group(680, 95, 500, 270, "P85 as the slowdown grows", "orange")
    rows = [["slowdown", "shared", "independent"],
            ["1.25", "65.5", "65.1"],
            ["1.50", "70.9", "69.0"],
            ["2.00", "86.0", "78.1"],
            ["2.50", "101.2", "88.0"]]
    b.table(700, 140, [140, 160, 160], rows, "orange", size=14, row_h=42)

    b.group(20, 385, 1160, 255, "Outside view: ten past projects, estimate against actual (illustrative history)", "purple")
    b.card(40, 430, 270, 90, "median overrun", ["1.55 times the estimate"], "purple", size=14, title_size=12)
    b.card(325, 430, 270, 90, "80th percentile", ["2.02 times the estimate"], "purple", size=14, title_size=12)
    b.card(610, 430, 270, 90, "within estimate", ["2 of 10 projects"], "purple", size=14, title_size=12)
    b.card(895, 430, 265, 90, "engineers say 40 days", ["62 days median, 81 at P80"], "red", size=13, title_size=12)
    b.card(40, 540, 1120, 80, "Use both views", ["The simulation prices the tasks you listed; the history prices the ones nobody listed. When they disagree, believe the history until you know why."], "yellow", size=12)
    return b


@board("design-docs-and-reviews-template")
def dd_template():
    b = Board(1200, 700, "A design doc for an LLM feature, and what a linter sees", "Ten sections; the checker counts quantities, missing sections and vague words")
    b.group(20, 95, 640, 585, "Sections, in order", "blue")
    secs = [("context", "who hurts, how often, measured today"), ("goals", "a number and a date, per goal"),
            ("non-goals", "what a reader might assume and should not"), ("evaluation", "set size, graders, metric, interval"),
            ("alternatives", "two or more, with the reason each lost"), ("design", "the call path, the checks, the data flow"),
            ("risks", "named failures, each with a control"), ("rollout", "stages, and the gate between them"),
            ("rollback", "trigger numbers and time to undo"), ("cost", "tokens per call, volume, budget, alarm")]
    for i, (name, prompt) in enumerate(secs):
        y = 140 + i * 53
        b.card(40, y, 160, 42, name, [], "blue", size=12)
        raw_text(b, 215, y + 26, prompt, 12, INK, anchor="start")

    b.group(680, 95, 500, 585, "Linter output on two drafts", "orange")
    rows = [["", "bad draft", "good draft"],
            ["words", "78", "269"],
            ["quantities", "0", "15"],
            ["sections", "5", "10"],
            ["missing sections", "5", "0"],
            ["vague phrases", "7", "0"],
            ["problems found", "14", "0"]]
    b.table(700, 140, [180, 140, 130], rows, "orange", size=13, row_h=38)
    b.card(700, 430, 460, 100, "Vague words it flags", ["fast, scalable, robust, high quality,", "state of the art, best practices, as needed"], "red", size=12)
    b.card(700, 545, 460, 115, "What a linter cannot see", ["Whether the numbers are right, whether", "option A really lost for the reason given,", "or whether the risk list is complete."], "yellow", size=12)
    return b


@board("design-docs-and-reviews-matrix")
def dd_matrix():
    b = Board(1200, 660, "A decision matrix, then a test of the weights", "Three options for ticket summaries; scores 1 to 5; weights 5, 4, 3, 4, 3, 2")
    b.group(20, 95, 770, 330, "Scores", "blue")
    rows = [["criterion (weight)", "API, prompt", "API + retrieval", "small self-hosted"],
            ["answer quality (5)", "3", "5", "3"],
            ["time to ship (4)", "5", "4", "1"],
            ["run cost at volume (3)", "3", "2", "5"],
            ["privacy, residency (4)", "2", "2", "5"],
            ["on-call load (3)", "5", "3", "2"],
            ["reversibility (2)", "4", "4", "3"],
            ["weighted score", "3.571", "3.429", "3.143"]]
    b.table(40, 140, [250, 150, 170, 170], rows, "blue", size=13, row_h=36)

    b.group(810, 95, 370, 330, "Win share, 2,000 weight draws", "orange")
    for i, (name, v) in enumerate((("API, prompt", 0.722), ("API + retrieval", 0.203), ("small self-hosted", 0.075))):
        y = 160 + i * 80
        raw_text(b, 830, y, name, 12, INK, anchor="start", weight="700")
        rect(b, 830, y + 10, v * 320 + 2, 24, "orange")
        raw_text(b, 830 + v * 320 + 12, y + 28, f"{v:.3f}", 12, INK, anchor="start", weight="700")
    b.text(995, 405, "each weight scaled by 0.5 to 1.5 at random", 11, "orange", italic=True)

    b.card(20, 450, 380, 190, "A close call is a result", ["The leader wins 72% of draws. The other two", "win 28% between them: say so in the doc and", "pick on something the matrix leaves out."], "yellow", size=12)
    b.card(410, 450, 380, 190, "A gate is not a weight", ["Privacy needs a score of 3 or more. That", "removes both API options and leaves the", "self-hosted model, the lowest average."], "red", size=12)
    b.card(800, 450, 380, 190, "Write the losers down", ["The alternatives section records why each", "option lost, so the reviewer can disagree", "with a reason instead of a hunch."], "green", size=12)
    return b


@board("technical-leadership-and-mentoring-archetypes")
def lead_arch():
    b = Board(1200, 640, "Four ways to lead without managing", "Archetypes from Will Larson's staffeng.com; decision speed from Bezos' 2016 letter; feedback from CCL's SBI model")
    b.group(20, 95, 1160, 265, "Staff-plus archetypes", "blue")
    arch = [("Tech Lead", "guides the approach and execution of one team", "blue"), ("Architect", "owns direction and quality in a critical area", "purple"),
            ("Solver", "digs into hard problems and moves where needed", "orange"), ("Right Hand", "borrows an executive's scope to run a complex org", "green")]
    for i, (t, d, c) in enumerate(arch):
        b.card(40 + i * 285, 140, 270, 100, t, [d], c, size=12, title_size=16)
    b.card(40, 255, 1120, 85, "For AI work, the same split shows up as", ["tech lead of the LLM feature team  |  architect of the eval and serving platform  |  solver on the incident nobody can explain"], "grey", size=12)

    b.group(20, 380, 560, 245, "Decide at the right speed", "orange")
    b.card(40, 425, 255, 120, "two-way door", ["reversible: a prompt, a", "threshold, a model alias.", "Decide fast, with about", "70% of the information."], "green", size=12)
    b.card(305, 425, 255, 120, "one-way door", ["costly to undo: a data", "contract, a vendor deal, a", "schema. Slow down and", "write it up."], "red", size=12)
    b.text(300, 585, "disagree and commit: say the gamble out loud, then back the call", 11, "orange", italic=True)

    b.group(600, 380, 580, 245, "Feedback that can be acted on (SBI)", "teal")
    b.card(620, 425, 170, 160, "Situation", ["Tuesday's review of", "the eval set"], "teal", size=11)
    b.card(805, 425, 170, 160, "Behaviour", ["you graded 12 items", "yourself, with no", "second grader"], "teal", size=11)
    b.card(990, 425, 170, 160, "Impact", ["the 8-point gain", "may be the grader's", "taste, not the model"], "teal", size=11)
    return b


@board("technical-leadership-and-mentoring-eval-discipline")
def lead_eval():
    b = Board(1200, 660, "Mentoring on evaluation discipline: let the numbers teach", "Two seeded simulations a junior can rerun in a minute")
    b.group(20, 95, 600, 400, "A real 10-point regression, a small check", "red")
    rows = [["items", "new looks same or better", "interval shows it is worse"],
            ["5", "0.506", "0.067"],
            ["10", "0.393", "0.112"],
            ["20", "0.282", "0.112"],
            ["50", "0.151", "0.214"],
            ["100", "0.058", "0.378"],
            ["300", "0.003", "0.812"]]
    b.table(40, 140, [90, 250, 230], rows, "red", size=13, row_h=40)
    b.text(320, 460, "pass rate 0.80 old against 0.70 new, 20,000 simulated reviews per row", 11, "red", italic=True)

    b.group(640, 95, 540, 400, "Best of k prompts, all truly 0.70", "orange")
    rows = [["variants tried", "best score on 50 items"],
            ["1", "0.700"],
            ["3", "0.755"],
            ["5", "0.773"],
            ["10", "0.796"],
            ["20", "0.817"],
            ["50", "0.839"]]
    b.table(660, 140, [210, 290], rows, "orange", size=13, row_h=40)
    b.text(910, 460, "winner re-run on 400 fresh items: 0.701", 12, "orange", weight="700")

    b.card(20, 520, 380, 120, "Five examples catch nothing", ["With 5 items a real 10-point drop looks", "fine half the time and shows up in", "the interval 6.7% of the time."], "red", size=12)
    b.card(410, 520, 380, 120, "The winner's curse", ["Pick the best of 20 and its score is", "inflated by 0.116 on average, then", "falls back to 0.701 on fresh items."], "orange", size=12)
    b.card(800, 520, 380, 120, "Teach the habit", ["Fix the eval set first, count the variants", "tried, and confirm on items the search", "never saw."], "green", size=12)
    return b


@board("postmortems-and-on-call-for-ml-error-budget")
def pm_budget():
    b = Board(1200, 700, "Error budgets and burn rates", "Google SRE definitions; 30-day window; the quality events are an illustrative example")
    b.group(20, 95, 560, 270, "Budget for a 30-day window", "blue")
    rows = [["SLO", "budget", "full-outage minutes"],
            ["99.00%", "1.000%", "432.0"],
            ["99.50%", "0.500%", "216.0"],
            ["99.90%", "0.100%", "43.2"],
            ["99.95%", "0.050%", "21.6"],
            ["99.99%", "0.010%", "4.3"]]
    b.table(40, 140, [130, 150, 250], rows, "blue", size=13, row_h=36)

    b.group(600, 95, 580, 270, "Burn-rate rules, 99.9% SLO", "orange")
    rows = [["rate", "long / short", "budget used", "action"],
            ["14.4x", "1 h / 5 min", "2.0%", "page"],
            ["6x", "6 h / 30 min", "5.0%", "page"],
            ["1x", "72 h / 6 h", "10.0%", "ticket"]]
    b.table(620, 140, [90, 190, 150, 120], rows, "orange", size=13, row_h=40)
    b.text(890, 335, "budget used = rate x long window / period", 11, "orange", italic=True)

    b.group(20, 385, 1160, 205, "Three quality events against a 99% good-response SLO", "red")
    rows = [["bad share", "hours", "burn rate", "budget used", "rules that fire"],
            ["8%", "6.0", "8.0x", "6.7%", "page 6x / 6 h"],
            ["3%", "20.0", "3.0x", "8.3%", "none"],
            ["100%", "0.5", "100.0x", "6.9%", "page 14.4x / 1 h, page 6x / 6 h"]]
    b.table(40, 430, [140, 110, 150, 170, 530], rows, "red", size=13, row_h=36)

    b.card(20, 610, 570, 75, "Together: 21.9% of the budget", ["the quiet 3% regression cost more than either loud one"], "yellow", size=12)
    b.card(610, 610, 570, 75, "Alert on burn, not on the threshold", ["slow burns need a sampled grade, not only a rate rule"], "green", size=12)
    return b


@board("postmortems-and-on-call-for-ml-postmortem-anatomy")
def pm_anatomy():
    b = Board(1200, 700, "Faster detection is the cheapest budget", "Seeded simulations: 3 incidents a month, 99.5% SLO (216 full-outage minutes), and a graded-sample monitor")
    b.group(20, 95, 590, 300, "Monthly budget used by detection time", "orange")
    rows = [["mean time to detect", "mean budget used", "P(exhausted)"],
            ["15 min", "44.6%", "0.112"],
            ["60 min", "67.7%", "0.255"],
            ["240 min", "160.0%", "0.493"],
            ["720 min", "406.3%", "0.692"]]
    b.table(40, 140, [200, 200, 160], rows, "orange", size=13, row_h=42)
    b.text(315, 375, "30% hard outages, 70% regressions hurting 5% to 15% of responses", 11, "orange", italic=True)

    b.group(630, 95, 550, 300, "Graded samples a day, 0.95 to 0.90", "teal")
    rows = [["graded / day", "caught day 1", "day 3", "day 7"],
            ["20", "0.132", "0.388", "0.652"],
            ["50", "0.232", "0.642", "0.940"],
            ["100", "0.418", "0.899", "0.997"],
            ["200", "0.715", "0.994", "1.000"],
            ["400", "0.947", "1.000", "1.000"]]
    b.table(650, 140, [140, 140, 110, 100], rows, "teal", size=13, row_h=40)

    b.group(20, 415, 1160, 270, "A blameless postmortem, in order", "purple")
    steps = [("summary", "one paragraph"), ("impact", "users, share, hours"), ("timeline", "UTC, with detection"), ("causes", "root and contributing"),
             ("went well / badly", "no names"), ("actions", "owner, date, type")]
    for i, (t, d) in enumerate(steps):
        b.card(40 + i * 188, 460, 175, 80, t, [d], "purple", size=11)
    b.card(40, 560, 1120, 105, "Added for a model regression", ["what the evals covered and missed  |  which data, prompt or weights changed", "how the bad share was measured  |  which new check would have caught it, and its false-alarm rate"], "yellow", size=12)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        key = next(k for k, v in NAMES.items() if k == name or v == name or v.endswith(name))
        path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
