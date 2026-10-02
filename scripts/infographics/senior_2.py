"""Infographics for docs/senior/01-system-design-cases, chapters 04 to 06.

Run from the repo root:

    python3 scripts/infographics/senior_2.py            # all boards
    python3 scripts/infographics/senior_2.py context    # boards whose name contains "context"
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


def rect(b, x, y, w, h, fill, stroke=None, opacity=1.0):
    s = f' stroke="{stroke}" stroke-width="1.2"' if stroke else ""
    b.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="{opacity}"{s}/>')


@board("ai-coding-assistant-architecture")
def coding_architecture():
    b = Board(1240, 700, "An AI coding assistant is three products", "4,000 developers, 5 active hours a day; the numbers come from the sizing block")
    b.group(20, 95, 1200, 300, "Three latency tiers, three different systems", "blue")
    rows = [["tier", "requests per dev-hour", "prompt tokens", "output tokens", "peak requests/s", "peak prefill tok/s"],
            ["completion", "250", "1,500", "40", "833.3", "625,000"],
            ["chat", "4", "6,000", "400", "13.3", "56,000"],
            ["agent step", "24", "20,000", "300", "80.0", "320,000"]]
    b.table(45, 140, [150, 230, 190, 190, 200, 170], rows, "blue", size=13, row_h=36)
    b.card(45, 295, 380, 85, "completion: tens of milliseconds matter", ["debounce, cancel stale requests,", "small model, tiny context"], "teal", size=12)
    b.card(445, 295, 380, 85, "chat: first token in about a second", ["retrieved context, streamed answer", "human reads while it writes"], "orange", size=12)
    b.card(845, 295, 355, 85, "agent: minutes, many steps", ["tools run tests, loop until green,", "prefix caching decides the cost"], "purple", size=12)

    a = b.card(20, 430, 250, 95, "editor", ["cursor, open files,", "selection, diagnostics"], "grey", size=12)
    c = b.card(320, 430, 250, 95, "context assembly", ["rank, pack under a budget,", "prefix first (cacheable)"], "green", size=12)
    d = b.card(620, 430, 250, 95, "privacy gate", ["path exclusions, secret", "redaction, then send"], "red", size=12)
    e = b.card(920, 430, 300, 95, "model by tier", ["small fast model for completion,", "larger model for chat and agent"], "purple", size=12)
    b.arrow(a.right(), c.left())
    b.arrow(c.right(), d.left())
    b.arrow(d.right(), e.left())
    b.arrow(e.bottom(), (1070, 560))
    b.card(20, 560, 560, 120, "How you know it works: execution", ["generated code runs against tests in a sandbox.", "pass@k is estimated from n samples with c correct;", "text similarity cannot tell a one-character bug from a copy."], "yellow", size=12)
    b.card(600, 560, 620, 120, "Total load at the stated peak", ["about 1,001,000 prefill tokens/s across the three tiers:", "101 GPUs at 10,000 tokens/s each, 34 at 30,000, 11 at 100,000", "(per-GPU rates are parameters, not measurements)"], "grey", size=12)
    return b


@board("ai-coding-assistant-context-budget")
def coding_context():
    b = Board(1240, 640, "Context is a budget, not a bucket", "30 completion points in httpx 0.28.1; recall of the definitions the finished function needed")
    b.group(20, 95, 700, 330, "Body recall by strategy and budget (tokens)", "teal")
    rows = [["strategy", "250", "500", "1000", "2000", "4000"],
            ["neighbour", "0.167", "0.217", "0.267", "0.283", "0.350"],
            ["bm25", "0.083", "0.167", "0.233", "0.300", "0.433"],
            ["bm25+graph", "0.133", "0.300", "0.500", "0.700", "0.733"],
            ["map+graph", "0.150", "0.300", "0.383", "0.517", "0.683"]]
    b.table(45, 140, [170, 100, 100, 100, 100, 100], rows, "teal", size=14, row_h=38)
    b.card(45, 345, 650, 62, "signature recall, map+graph", ["0.400  0.617  0.583  0.817  0.933   (the map makes signatures cheap)"], "yellow", size=12)

    b.group(745, 95, 475, 330, "How each strategy builds its list", "purple")
    b.card(765, 140, 435, 55, "neighbour", ["same file by distance from the cursor, then the rest"], "grey", size=11)
    b.card(765, 205, 435, 55, "bm25", ["keyword match on identifiers in the last 20 lines"], "blue", size=11)
    b.card(765, 270, 435, 66, "bm25+graph", ["boost the callees of the top 3 hits and of calls", "already visible before the cursor"], "green", size=11)
    b.card(765, 346, 435, 66, "map+graph", ["30% of the budget on one-line signatures of the", "most-called functions, then the graph list"], "orange", size=11)

    b.card(20, 450, 400, 170, "Keyword search alone stalls", ["at 4,000 tokens bm25 reaches 0.433,", "and neighbour only 0.350:", "word overlap with the line being", "typed is a weak guide."], "red", size=12)
    b.card(440, 450, 380, 170, "Following calls pays", ["bm25+graph reaches 0.700 at 2,000", "tokens, the same budget where bm25", "has 0.300: a call edge finds what", "words cannot."], "green", size=12)
    b.card(840, 450, 380, 170, "A map spends tokens on breadth", ["map+graph has lower body recall than", "bm25+graph at every budget from 1,000", "up, yet sees 0.933 of signatures at", "4,000 tokens."], "orange", size=12)
    return b


@board("support-agent-architecture")
def support_architecture():
    b = Board(1240, 720, "A support agent is a pipeline with a hard edge", "The model proposes; a gateway in ordinary code decides what is allowed to happen")
    a = b.card(20, 110, 190, 90, "customer message", ["chat, email, voice", "transcript"], "grey", size=12)
    i = b.card(250, 110, 190, 90, "intake", ["verify identity,", "mask card numbers"], "teal", size=12)
    r = b.card(480, 110, 190, 90, "router", ["intent + risk class,", "risky goes to a person"], "orange", size=12)
    g = b.card(710, 110, 220, 90, "agent loop", ["model + memory + tools,", "bounded steps and tokens"], "purple", size=12)
    d = b.diamond(1090, 155, 220, 110, "confident and\nin policy?", "yellow", size=12)
    b.arrow(a.right(), i.left())
    b.arrow(i.right(), r.left())
    b.arrow(r.right(), g.left())
    b.arrow(g.right(), d.left())
    ans = b.card(970, 250, 250, 70, "send the answer", ["log it, sample it for review"], "green", size=12)
    hand = b.card(970, 345, 250, 80, "hand over to a person", ["summary, evidence, what", "was tried"], "red", size=12)
    b.arrow((1090, 210), ans.top(), label="yes")
    b.arrow((1200, 155), hand.right(), via=[(1232, 155), (1232, 385)], color="red", label="no", label_at=0.12)

    b.group(20, 235, 930, 190, "Memory", "blue")
    b.card(40, 275, 290, 120, "this conversation", ["append-only history keeps the", "prefix cache; compact only when", "the window forces it"], "blue", size=11)
    b.card(350, 275, 290, 120, "this customer", ["orders, plan, past tickets,", "read through the same gateway", "so access rules apply"], "blue", size=11)
    b.card(660, 275, 270, 120, "never", ["card numbers in logs:", "masked to [CARD]"], "red", size=11)

    b.group(20, 450, 1200, 250, "Tool gateway: what the scripted scenarios did", "red")
    rows = [["call", "outcome"],
            ["refund 30 on own order, then the same key again", "refunded; duplicate ignored"],
            ["refund on another customer's order (injected text)", "denied: not found for this customer"],
            ["refund 60 with 50 left on the order", "denied: more than the balance"],
            ["refund above the tier cap, then past the daily cap", "escalated, escalated"],
            ["refund by an unverified caller", "denied: not verified"]]
    b.table(45, 495, [560, 560], rows, "red", size=13, row_h=30)
    return b


@board("support-agent-escalation-and-cost")
def support_cost():
    b = Board(1240, 680, "Containment is not resolution, and resolution is not cost", "2,000 synthetic tickets; agent 0.05, person 4.0, wrong automated answer 15.0 (placeholder units)")
    b.group(20, 95, 640, 300, "Same agent, different threshold", "orange")
    rows = [["policy", "contained", "resolved", "wrong", "cost/ticket"],
            ["automate all", "1.000", "0.648", "703", "6.729"],
            ["0.6, risky to person", "0.557", "0.479", "156", "3.304"],
            ["0.8, risky to person", "0.371", "0.342", "60", "3.134"],
            ["0.75, risky to person", "0.424", "0.387", "75", "3.067"],
            ["all to people", "-", "-", "-", "4.000"]]
    b.table(40, 140, [210, 105, 105, 75, 100], rows, "orange", size=13, row_h=36)
    raw_text(b, 340, 375, "0.75 is the cheapest of 21 thresholds, risky intents escalated", 11, FAINT)

    b.group(680, 95, 540, 300, "Reliability: all k trials, or any of k", "teal")
    rows = [["k", "pass^k (all)", "pass@k (any)"],
            ["1", "0.642", "0.642"],
            ["2", "0.523", "0.761"],
            ["4", "0.405", "0.837"],
            ["8", "0.300", "0.883"]]
    b.table(705, 140, [100, 200, 200], rows, "teal", size=14, row_h=38)
    b.card(705, 340, 490, 45, "customers experience pass^k, not pass@k", [], "teal", size=12)

    b.group(20, 420, 790, 240, "Input tokens over a conversation (placeholder cache price 0.1)", "purple")
    rows = [["turns", "full history", "rolling summary", "full, cached", "summary, cached"],
            ["8", "23,520", "20,320", "6,510", "11,302"],
            ["12", "46,800", "32,720", "10,566", "19,382"],
            ["20", "116,400", "57,520", "20,982", "35,542"]]
    b.table(40, 465, [90, 150, 170, 150, 190], rows, "purple", size=13, row_h=36)
    raw_text(b, 415, 640, "a sliding summary rewrites the prefix, so it loses the cache", 11, FAINT)

    b.card(830, 420, 390, 240, "Three lessons", ["1. Letting the agent answer everything", "is the most expensive policy tried.", "2. Escalating risky intents is cheap", "insurance when confidence is blind to risk.", "3. Measure resolved tickets and cost,", "not containment."], "yellow", size=12, align="left")
    return b


@board("ml-platform-layers")
def platform_layers():
    b = Board(1240, 720, "A platform is four shared layers and a paved road", "Eight teams, one platform: what is shared, what stays with the team")
    b.group(20, 95, 1200, 380, "Shared layers", "blue")
    f = b.card(45, 140, 270, 200, "1. features", ["offline store for training,", "online store for serving,", "point-in-time joins so a", "training row sees only what", "was known then"], "teal", size=12)
    t = b.card(335, 140, 270, 200, "2. training", ["one GPU pool, per-team", "quotas, borrowing of idle", "capacity, reclaim by", "preemption, dominant-share", "fairness across resources"], "orange", size=12)
    r = b.card(625, 140, 270, 200, "3. registry", ["versions, aliases, lineage", "(run, code, data snapshot),", "a promotion gate, and", "rollback as one alias move"], "purple", size=12)
    v = b.card(915, 140, 285, 200, "4. serving", ["online endpoints and batch", "jobs from the same artefact,", "latency budget checked at", "promotion, shadow and", "canary release"], "green", size=12)
    b.arrow(f.right(), t.left())
    b.arrow(t.right(), r.left())
    b.arrow(r.right(), v.left())
    b.card(45, 365, 1155, 85, "cost attribution across all four", ["every job and endpoint carries owner and cost-centre labels; idle capacity is charged by a stated rule", "(usage only, split equally, by usage or by quota), so a team can see what its choices cost"], "yellow", size=12)

    b.group(20, 500, 1200, 200, "The golden path: the supported route, not the only route", "green")
    b.card(45, 545, 360, 130, "a project template", ["owner, cost centre, eval set and", "registry name are required;", "over 4 GPUs needs an approval"], "green", size=12)
    b.card(425, 545, 360, 130, "gate on promotion", ["5 versions tried: 2 promoted,", "3 blocked (incomplete lineage,", "a slice regression, over the", "50 ms p95 budget)"], "green", size=12)
    b.card(805, 545, 395, 130, "off-path is allowed, unsupported", ["a team may choose its own stack;", "the platform team does not carry its", "pager, and it gives up the shared layers"], "grey", size=12)
    return b


@board("ml-platform-pooling-and-cost")
def platform_pooling():
    b = Board(1240, 700, "Pool the GPUs, then decide who pays for idle", "Four teams, one week; placeholder price of 1.0 per GPU-hour; synthetic demand and jobs")
    b.group(20, 95, 600, 280, "Demand: separate peaks against one pool", "teal")
    rows = [["team", "quota", "peak", "GPU-hours"],
            ["search", "24", "34", "2,249"],
            ["ads", "16", "24", "1,589"],
            ["vision", "16", "24", "1,048"],
            ["nlp", "8", "22", "979"]]
    b.table(40, 140, [150, 120, 120, 150], rows, "teal", size=13, row_h=32)
    b.card(40, 320, 275, 45, "partitions: 104 GPUs", ["utilisation 0.336"], "red", size=11)
    b.card(335, 320, 270, 45, "one pool: 56 GPUs", ["utilisation 0.623"], "green", size=11)

    b.group(640, 95, 580, 280, "Scheduling policies, 64 GPUs, 927 jobs (waits in hours)", "orange")
    rows = [["policy", "utilisation", "preempted", "ads p50", "search p95"],
            ["static quotas", "0.651", "0", "24.2 h", "4.8 h"],
            ["shared FIFO pool", "0.774", "0", "0.5 h", "4.8 h"],
            ["quota + borrow", "0.790", "156", "0.3 h", "2.0 h"]]
    b.table(660, 140, [160, 100, 90, 90, 100], rows, "orange", size=12, row_h=34)
    b.card(660, 295, 540, 70, "borrowing wasted 163 GPU-hours", ["the price of reclaiming quota: restarted jobs", "(1.5% of 10,752 GPU-hours)"], "yellow", size=11)

    b.group(20, 400, 760, 280, "Who pays the 4,887 idle GPU-hours? (64 GPUs, week cost 10,752)", "purple")
    rows = [["method", "search", "ads", "vision", "nlp"],
            ["usage only", "2,249", "1,589", "1,048", "979"],
            ["idle split equally", "3,471", "2,811", "2,270", "2,201"],
            ["idle by usage", "4,123", "2,913", "1,921", "1,795"],
            ["idle by quota", "4,082", "2,811", "2,270", "1,590"]]
    b.table(40, 445, [210, 130, 130, 130, 130], rows, "purple", size=13, row_h=36)
    raw_text(b, 400, 665, "usage only leaves 4,887 GPU-hours unallocated; the other rows sum to 10,752", 11, FAINT)

    b.group(800, 400, 420, 280, "Training on yesterday's truth", "red")
    rows = [["training join", "offline AUC", "served AUC"],
            ["point-in-time", "0.597", "0.597"],
            ["latest value", "0.670", "0.616"]]
    b.table(820, 445, [160, 120, 100], rows, "red", size=12, row_h=36)
    b.card(820, 570, 380, 90, "the leak flatters the offline number", ["0.670 offline against 0.616 when the", "same model sees values as they were known"], "red", size=11)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        keys = [k for k, v in NAMES.items() if k == name or v == name or name in v]
        for key in keys:
            path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
            print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
