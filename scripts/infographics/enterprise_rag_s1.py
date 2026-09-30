"""Infographics for docs/projects/enterprise-rag/01-session-1.md (Enterprise RAG, Session 1).

Each function redraws one slide, whiteboard page or demo-app screen from the
session video (bjkjaqUZl4E) as an original board-style image. Timestamps in the
docstrings are video times (H:MM). Run from the repo root:

    python3 scripts/infographics/enterprise_rag_s1.py              # all boards
    python3 scripts/infographics/enterprise_rag_s1.py s1_security  # just one
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import CHAR_W, FAINT, INK, MONO, PALETTE, SANS, Board, Box, esc, wrap  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "enterprise-rag"
BOARDS = {}

# Handwriting stack for the live pen notes the presenters add on top of slides.
HAND = "'Segoe Print','Chalkboard SE','Comic Sans MS','Comic Neue',cursive"
PEN = {
    "orange": "#f76707",
    "red": "#e03131",
    "pink": "#e64980",
    "green": "#37b24d",
    "yellow": "#f59f00",
    "blue": "#1c7ed6",
    "purple": "#7048e8",
}


def board(fn):
    BOARDS[fn.__name__] = fn
    return fn


# --------------------------------------------------------------------- helpers


def tone(c, key="text"):
    """Palette colour by name, or pass a hex straight through."""
    if c in PALETTE:
        return PALETTE[c][key]
    return c or INK


def raw(b, s):
    b.parts.append(s)


def heading(b, segments, y=44, size=24, x=None, anchor="middle", family=MONO):
    """One line of bold text made of differently coloured segments."""
    x = b.w / 2 if x is None else x
    spans = "".join(f'<tspan fill="{tone(c)}">{esc(t)}</tspan>' for t, c in segments)
    raw(b, f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" font-size="{size}" '
           f'font-weight="700" xml:space="preserve">{spans}</text>')


def hand(b, x, y, text, color="orange", size=15, anchor="middle"):
    """A live pen note, in handwriting, in the presenter's ink colour."""
    fill = PEN.get(color, tone(color, "stroke"))
    for i, t in enumerate(text.split("\n")):
        raw(b, f'<text x="{x}" y="{y + i * size * 1.25:.1f}" text-anchor="{anchor}" font-family="{HAND}" '
               f'font-size="{size}" font-weight="700" fill="{fill}">{esc(t)}</text>')


def tick(b, x, y, color="orange", s=1.0):
    """A hand-drawn tick whose left end sits at (x, y)."""
    c = PEN.get(color, color)
    raw(b, f'<path d="M{x},{y} l{4 * s:.1f},{5 * s:.1f} l{10 * s:.1f},{-12 * s:.1f}" fill="none" stroke="{c}" '
           f'stroke-width="{2.6 * s:.1f}" stroke-linecap="round" stroke-linejoin="round"/>')


def cross(b, x, y, color="red", s=1.0):
    c = PEN.get(color, color)
    d = 6 * s
    raw(b, f'<path d="M{x - d},{y - d} L{x + d},{y + d} M{x + d},{y - d} L{x - d},{y + d}" stroke="{c}" '
           f'stroke-width="{2.6 * s:.1f}" stroke-linecap="round"/>')


def ring(b, x, y, w, h, color="green", width=3.0, rx=6):
    """A highlighter outline the presenter drew round part of a slide."""
    c = PEN.get(color, color)
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{c}" '
           f'stroke-width="{width}" stroke-opacity="0.9"/>')


def ellipse(b, cx, cy, rx, ry, color="pink", width=2.2, fill="none"):
    c = PEN.get(color, color)
    raw(b, f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{c}" stroke-width="{width}"/>')


def underline(b, x1, x2, y, color="orange", double=False, width=2.2):
    c = PEN.get(color, color)
    raw(b, f'<path d="M{x1},{y} Q{(x1 + x2) / 2},{y + 3} {x2},{y - 1}" fill="none" stroke="{c}" '
           f'stroke-width="{width}" stroke-linecap="round"/>')
    if double:
        raw(b, f'<path d="M{x1 + 4},{y + 5} Q{(x1 + x2) / 2},{y + 8} {x2 - 2},{y + 4}" fill="none" '
               f'stroke="{c}" stroke-width="{width}" stroke-linecap="round"/>')


def line(b, x1, y1, x2, y2, color=INK, width=1.6, dashed=False):
    c = tone(color, "stroke") if color in PALETTE else color
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    raw(b, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{width}"{dash}/>')


def zone(b, x, y, w, h, label, color="grey", size=15, dashed=True, fill_opacity=0.35, align="middle"):
    """A group box whose label may run to two lines ("\\n")."""
    c = PALETTE[color]
    dash = ' stroke-dasharray="9 6"' if dashed else ""
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c["fill"]}" '
           f'fill-opacity="{fill_opacity}" stroke="{c["stroke"]}" stroke-width="2"{dash}/>')
    tx = x + w / 2 if align == "middle" else x + 14
    for i, t in enumerate(label.split("\n")):
        raw(b, f'<text x="{tx}" y="{y + 8 + size + i * size * 1.25:.1f}" text-anchor="{align}" '
               f'font-family="{MONO}" font-size="{size}" font-weight="700" fill="{c["text"]}">{esc(t)}</text>')
    return Box(x, y, w, h)


def sub(b, x, y, w, h, label, color="grey", size=12):
    """A thin dashed sub-box with a small label, inside a zone."""
    c = PALETTE[color]
    raw(b, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" fill-opacity="0.75" '
           f'stroke="{c["stroke"]}" stroke-width="1.3" stroke-dasharray="5 4"/>')
    if label:
        raw(b, f'<text x="{x + w / 2}" y="{y + size + 7}" text-anchor="middle" font-family="{MONO}" '
               f'font-size="{size}" font-weight="700" fill="{c["text"]}">{esc(label)}</text>')
    return Box(x, y, w, h)


def node(b, x, y, w, h, lines, color="blue", size=11, bold_first=True, first_size=None, dashed=False,
         fg=None):
    """A card with centred text lines and no automatic wrapping."""
    box = b.card(x, y, w, h, "", [], color, dashed=dashed)
    first_size = first_size or size + 1
    sizes = [first_size if (i == 0 and bold_first) else size for i in range(len(lines))]
    total = sum(s * 1.35 for s in sizes)
    ty = y + (h - total) / 2 + sizes[0] if lines else 0
    for i, t in enumerate(lines):
        weight = "700" if (i == 0 and bold_first) else "400"
        fill = fg or (tone(color) if (i == 0 and bold_first) else INK)
        if color == "dark":
            fill = "#ffffff" if i == 0 else "#e9ecef"
        raw(b, f'<text x="{x + w / 2}" y="{ty:.1f}" text-anchor="middle" font-family="{MONO}" '
               f'font-size="{sizes[i]}" font-weight="{weight}" fill="{fill}">{esc(t)}</text>')
        if i + 1 < len(lines):
            ty += sizes[i + 1] * 1.35
    return box


def items(b, x, y, entries, size=11, gap=None, color=INK, mark="", mark_color=None, weight="400",
          anchor="start", width=None):
    """Draw text lines top-down; returns the baseline y of each entry and the next free y."""
    gap = gap or size * 1.5
    ys = []
    for e in entries:
        wrapped = wrap(e, int(width / (size * CHAR_W))) if width else [e]
        ys.append(y)
        for j, t in enumerate(wrapped):
            prefix = ""
            if mark:
                prefix = mark + " " if j == 0 else " " * (len(mark) + 1)
            if mark and mark_color:
                raw(b, f'<text x="{x}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
                       f'font-weight="{weight}" xml:space="preserve"><tspan fill="{tone(mark_color, "stroke")}">'
                       f'{esc(prefix)}</tspan><tspan fill="{tone(color)}">{esc(t)}</tspan></text>')
            else:
                raw(b, f'<text x="{x}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
                       f'font-weight="{weight}" fill="{tone(color)}" xml:space="preserve">{esc(prefix + t)}</text>')
            y += gap
    return ys, y


def panel(b, x, y, w, h, title, lines=(), color="blue", size=11, title_size=None, footer="",
          footer_color=None, align="left", mark="", mark_color=None, pad=12, gap=None, dashed=False,
          title_color=None, footer_italic=False):
    """A fixed-size card: title at the top, body lines below it, optional footer at the bottom."""
    box = b.card(x, y, w, h, "", [], color, dashed=dashed)
    title_size = title_size or size + 2
    tcol = title_color or tone(color)
    ty = y + pad + title_size
    tx = x + w / 2 if align == "center" else x + pad
    anchor = "middle" if align == "center" else "start"
    if title:
        for t in wrap(title, int((w - 2 * pad) / (title_size * CHAR_W))):
            raw(b, f'<text x="{tx}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
                   f'font-size="{title_size}" font-weight="700" fill="{tone(tcol)}">{esc(t)}</text>')
            ty += title_size * 1.3
        ty += 6
    else:
        ty = y + pad + size
    ys = []
    gap = gap or size * 1.45
    chars = int((w - 2 * pad) / (size * CHAR_W)) - (len(mark) + 1 if mark else 0)
    for ln in lines:
        ys.append(ty)
        for j, t in enumerate(wrap(ln, chars) if ln else [""]):
            prefix = (mark + " " if j == 0 else " " * (len(mark) + 1)) if mark else ""
            mc = tone(mark_color or color, "stroke")
            raw(b, f'<text x="{tx}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
                   f'xml:space="preserve"><tspan fill="{mc}" font-weight="700">{esc(prefix)}</tspan>'
                   f'<tspan fill="{INK}">{esc(t)}</tspan></text>')
            ty += gap
    if footer:
        fc = tone(footer_color or color)
        flines = wrap(footer, int((w - 2 * pad) / (size * CHAR_W)))
        fy = y + h - pad - (len(flines) - 1) * size * 1.35
        style = ' font-style="italic"' if footer_italic else ""
        for t in flines:
            raw(b, f'<text x="{tx}" y="{fy:.1f}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
                   f'font-weight="700" fill="{fc}"{style}>{esc(t)}</text>')
            fy += size * 1.35
    return box, ys


def note(b, x, y, text, size=11, color=FAINT, anchor="start"):
    b.text(x, y, text, size, color, "400", anchor=anchor, family=SANS, italic=True)


def curve(b, p1, p2, bend=0.3, color="pink", width=2.0, dashed=False, head=True, pen=True):
    """A curved (quadratic) pen arrow; bend > 0 bows to the left of the direction of travel."""
    import math
    c = PEN.get(color, color) if pen else tone(color, "stroke")
    (x1, y1), (x2, y2) = p1, p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    nx, ny = -(y2 - y1), x2 - x1
    k = bend
    qx, qy = mx + nx * k, my + ny * k
    mid = b._marker(c) if head else None
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    end = f' marker-end="url(#{mid})"' if head else ""
    raw(b, f'<path d="M{x1:.1f},{y1:.1f} Q{qx:.1f},{qy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{c}" '
           f'stroke-width="{width}"{dash}{end}/>')
    _ = math


def pen_arrow(b, p1, p2, color="red", via=None, width=2.0, dashed=False, both=False, label="", label_dy=0):
    b.arrow(p1, p2, via=via, color=PEN.get(color, color), width=width, dashed=dashed, both=both, label=label,
            label_dy=label_dy)


def link(b, a, c, axis="x", overlap=12, **kw):
    """A two-headed arrow between neighbouring boxes, reaching a little into each so both heads show."""
    if axis == "x":
        y = kw.pop("y", (max(a.y, c.y) + min(a.y + a.h, c.y + c.h)) / 2)
        b.arrow((a.x + a.w - overlap, y), (c.x + overlap, y), both=True, **kw)
    else:
        x = kw.pop("x", (max(a.x, c.x) + min(a.x + a.w, c.x + c.w)) / 2)
        b.arrow((x, a.y + a.h - overlap), (x, c.y + overlap), both=True, **kw)


def star(b, cx, cy, r=9, color="#f59f00"):
    import math
    pts = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy - rr * math.sin(a):.1f}")
    raw(b, f'<polygon points="{" ".join(pts)}" fill="{color}" stroke="#e8590c" stroke-width="0.8"/>')


def brace(b, x, y1, y2, color="red", facing="right"):
    c = PEN.get(color, color)
    m = (y1 + y2) / 2
    d = 10 if facing == "right" else -10
    raw(b, f'<path d="M{x},{y1} q{d},0 {d},14 V{m - 8} q0,8 {d},8 q{-d},0 {-d},8 V{y2 - 14} q0,14 {-d},14" '
           f'fill="none" stroke="{c}" stroke-width="2"/>')


# ------------------------------------------------------------ 1. Opening slides


@board
def s1_security():
    """0:07-0:19 prepared slide 'Why security is the biggest concern', with Krish's pen notes."""
    W, H = 1400, 1050
    b = Board(W, H)
    heading(b, [("WHY ", INK), ("SECURITY", "red"), (" IS THE BIGGEST CONCERN", INK)], 46, 26)
    heading(b, [("IN GEN AI APPLICATIONS", "blue")], 82, 24)
    b.text(882, 84, "⚠", 28, "red", "700")
    hand(b, 520, 84, "AI Agents", "orange", 18, anchor="end")

    # Row 1: powerful-but-risky · typical architecture · new risks
    b.card(20, 104, 240, 200, "GEN AI IS POWERFUL, BUT ALSO RISKY!",
           ["Gen AI applications have access to sensitive data, connect to external systems, make "
            "decisions, and interact with users at scale.", "",
            "With great power comes great responsibility."], "blue", size=11, align="left")

    zone(b, 272, 104, 876, 200, "TYPICAL GEN AI APPLICATION ARCHITECTURE", "grey", size=14)
    chain = [["User"], ["Chat /", "App Interface"], ["AI Application", "(Orchestration)"], ["LLM", "(Model)"],
             ["Data Sources", "(RAG / DB / APIs)"], ["Tools &", "Integrations"], ["Output /", "Action"]]
    nodes = []
    for i, lines in enumerate(chain):
        x = 284 + i * 124
        nodes.append(node(b, x, 146, 106, 72, lines, "blue", size=10, first_size=11))
    for a, c in zip(nodes, nodes[1:]):
        b.text((a.x + a.w + c.x) / 2, a.cy + 6, "↔", 18, INK, "700")
    # pen notes on the chain
    underline(b, nodes[1].x + 6, nodes[1].x + nodes[1].w - 6, 228, "orange")
    pen_arrow(b, (nodes[2].cx + 10, 246), (nodes[3].cx - 10, 246), "orange", both=True, width=2.4)
    ring(b, nodes[5].x - 5, 141, nodes[5].w + 10, 82, "orange", 2.4)
    hand(b, nodes[5].cx, 256, "{Hallucination} ⇒", "red", 14)
    pen_arrow(b, (nodes[4].cx, 124), (nodes[4].cx, 144), "orange", width=2.4)
    pen_arrow(b, (nodes[2].cx, 222), (nodes[2].cx, 316), "orange", width=2.4)

    rk, ys = panel(b, 1160, 104, 220, 200, "THIS ARCHITECTURE INTRODUCES NEW RISKS",
                   ["Untrusted Inputs", "Unpredictable Outputs", "Data Leakage", "Prompt Injection",
                    "Insecure Integrations", "Lack of Visibility", "Compliance & Legal Risks"], "red",
                   size=11, title_size=12, mark="•")
    for k in (0, 1):
        pen_arrow(b, (1372, ys[k] - 4), (1340, ys[k] - 4), "red", width=2)
    for k in (4, 5, 6):
        tick(b, 1352 if k != 6 else 1360, ys[k] - 5, "orange", 0.9)

    # Row 2: seven reasons
    b.text(W / 2, 336, "WHY SECURITY IS THE BIGGEST CONCERN?", 18, "red", "700")
    hand(b, 150, 336, "Tools, Techniques", "green", 17)
    hand(b, 1120, 336, "AI Agents", "orange", 17)
    reasons = [
        ("1  NEW ATTACK SURFACE", "Gen AI apps combine models, data, tools & users, creating a huge new attack "
         "surface that didn't exist in traditional apps.", "", "blue"),
        ("2  UNTRUSTED INPUTS", "Attackers can manipulate prompts, upload files, or inject hidden instructions "
         "to trick the model and steal or damage data.", "Example: Prompt Injection, Jailbreaking", "purple"),
        ("3  SENSITIVE DATA AT RISK", "Gen AI apps often process PII, confidential business data, source code, "
         "and proprietary information.", "One leak can cause massive financial and reputational loss.", "red"),
        ("4  UNPREDICTABLE OUTPUTS", "Models can hallucinate, generate harmful, biased, or incorrect responses "
         "that can mislead users or violate policies.", "Trust without verification is dangerous.", "orange"),
        ("5  COMPLEX INTEGRATIONS", "Gen AI apps connect to APIs, databases, tools, browsers, and agents. A "
         "single weak integration can be exploited.", "More integrations = More risk", "teal"),
        ("6  LACK OF VISIBILITY", "Without proper logging, tracing and monitoring, it's impossible to detect "
         "attacks, debug issues or ensure reliability.", "You can't protect what you can't see.", "green"),
        ("7  COMPLIANCE & LEGAL RISKS", "Data privacy laws (GDPR, CCPA), industry regulations and emerging AI "
         "laws make security & governance a must-have.", "Non-compliance = Fines, Bans, Lawsuits", "pink"),
    ]
    for i, (t, body, foot, colr) in enumerate(reasons):
        panel(b, 20 + i * 195, 352, 186, 245, t, [body], colr, size=11, title_size=12, footer=foot)

    # Row 3: examples · impact · what we must do · callout
    hand(b, 420, 628, "Production Database", "pink", 17)
    ex = zone(b, 20, 636, 580, 244, "REAL WORLD EXAMPLES", "red", size=15, fill_opacity=0.25)
    examples = ["Samsung employees leaked sensitive code via ChatGPT.", "Prompt injection used to bypass safeguards.",
                "Malicious prompts tricked AI into leaking hidden data.",
                "AI agents manipulated to perform unintended actions.",
                "Public data in vector stores exposed via poor access control.",
                "Hallucinations caused financial & legal misinformation."]
    ys, _ = items(b, 34, 690, examples, size=11, gap=30, mark="⚠", mark_color="red", width=420)
    brace(b, 462, 676, 866, "red")
    b.text(530, 730, "These are not\ntheoretical.\nThey are\nhappening", 12, INK, "400", line_gap=1.45)
    b.text(530, 832, "TODAY!", 16, "red", "700")
    _ = ex

    zone(b, 615, 636, 375, 244, "THE IMPACT OF POOR SECURITY", "blue", size=14, fill_opacity=0.25)
    impacts = ["Financial Loss", "Data Breaches", "Reputation Damage", "Loss of Customer Trust",
               "Legal Consequences", "Business Disruption"]
    for i, t in enumerate(impacts):
        x = 630 + (i % 2) * 178
        y = 676 + (i // 2) * 50
        node(b, x, y, 168, 38, [t], "blue", size=11, first_size=11)
        if i < 4:
            tick(b, x + 150, y + 4, "orange", 0.9)
    b.text(802, 858, "One vulnerability can undo years of trust and hard work.", 10, "blue", "700")

    zone(b, 1005, 636, 260, 244, "WHAT WE MUST DO", "green", size=15, fill_opacity=0.25)
    must = ["Build with Security by Design", "Validate Inputs & Outputs", "Implement Guardrails & Policies",
            "Use LLM Gateways for Control", "Evaluate Continuously", "Monitor Everything (Observability)",
            "Secure Data, Tools & Integrations", "Stay Compliant & Governed"]
    ys, _ = items(b, 1016, 682, must, size=10, gap=24.5, mark="☑", mark_color="green")
    for k in (0, 1, 3, 5, 7):
        tick(b, 1240, ys[k] - 7, "orange", 0.8)
    hand(b, 1000, 626, "Guardrails", "red", 17, anchor="end")
    for k in (1, 2, 3, 4):
        pen_arrow(b, (993, ys[k] - 4), (1012, ys[k] - 4), "red", width=1.8)

    b.card(1277, 646, 103, 224, "", [], "yellow")
    star(b, 1364, 664, 10)
    b.text(1328, 704, "Security\nis not a\nfeature.\nIt's the", 12, INK, "400", line_gap=1.4)
    b.text(1328, 782, "FOUNDATION", 12, "blue", "700")
    b.text(1328, 804, "of trust-\nworthy AI.", 12, INK, "400", line_gap=1.4)
    star(b, 1296, 862, 8)

    # Row 4: closing banner
    b.card(20, 898, 1360, 104, "", [], "grey")
    b.text(W / 2, 928, "The future of Gen AI is huge, but so are the risks.   Secure AI systems are the only AI "
           "systems that will be trusted, adopted and used at scale.", 12, INK, "400")
    heading(b, [("BUILD FAST, BUT BUILD ", INK), ("SAFE.", "red"), ("      SECURITY TODAY, ", INK),
                ("TRUST", "green"), (" FOREVER.", INK)], 972, 22)
    note(b, 20, 1030, "Handwriting in colour marks the pen notes Krish added live over the slide.")
    return b


@board
def s1_agentic_architecture():
    """0:22-0:32 prepared slide 'Architecture of agentic AI application', page 2/7, with live highlights."""
    W, H = 1600, 1350
    b = Board(W, H)
    heading(b, [("ARCHITECTURE OF ", INK), ("AGENTIC AI APPLICATION", "purple")], 46, 28)
    b.text(W / 2, 76, "A Production-Ready, Secure, Observable and Scalable Architecture", 15, INK, "400",
           family=SANS)

    # Row A ---------------------------------------------------------------
    g1 = zone(b, 20, 96, 185, 380, "1. USERS &\nCHANNELS", "blue", size=14)
    chans = ["Web App", "Mobile App", "Desktop App", "Slack / Teams", "API Clients", "Voice / IoT",
             "Enterprise Users"]
    for i, t in enumerate(chans):
        node(b, 32, 150 + i * 45, 161, 36, [t], "blue", size=11, first_size=11, fg=INK)

    g2 = zone(b, 235, 96, 1015, 380, "2. ORCHESTRATION LAYER (AGENT RUNTIME)", "blue", size=15)
    rh = sub(b, 250, 132, 222, 330, "Request Handling", "blue")
    reqs = ["Request Receiver", "Session Manager", "Authentication", "Authorization (RBAC/ABAC)",
            "Rate Limiting", "Request Validation"]
    for i, t in enumerate(reqs):
        node(b, 262, 162 + i * 49, 198, 38, [t], "grey", size=11, first_size=11, fg=INK)

    ao = sub(b, 500, 132, 490, 330, "Agent Orchestrator", "blue")
    pl = node(b, 640, 158, 210, 40, ["Planner / Reasoner"], "purple", size=12, first_size=12)
    agents = []
    for i, name in enumerate(["Researcher", "Analyst", "Coder"]):
        agents.append(node(b, 520 + i * 160, 244, 130, 58, ["Sub-Agent", name], "blue", size=11, first_size=11))
    b.arrow(pl.bottom(), agents[1].top())
    b.arrow(pl.left(), agents[0].top(), via=[(agents[0].cx, pl.cy)])
    b.arrow(pl.right(), agents[2].top(), via=[(agents[2].cx, pl.cy)])
    for a, c in zip(agents, agents[1:]):
        b.arrow(a.right(), c.left(), both=True, dashed=True, width=1.3)
    line(b, agents[0].cx, 336, agents[2].cx, 336, INK, 1.8)
    for a in agents:
        b.arrow((a.cx, a.y + a.h + 2), (a.cx, 334), both=True, width=1.4)
    mm = node(b, 520, 382, 450, 46, ["Memory Manager (Short-term / Long-term)"], "blue", size=12, first_size=12)
    b.arrow((agents[1].cx, 338), mm.top(), both=True, width=1.4)
    link(b, rh, ao, width=1.5)

    ae = sub(b, 1020, 132, 215, 330, "Action Executor", "green")
    acts = ["Tool Selection", "Action Execution", "Response Builder", "Output Formatter", "Stream / Return"]
    for i, t in enumerate(acts):
        node(b, 1032, 166 + i * 56, 191, 42, [t], "green", size=11, first_size=11, fg=INK)
    link(b, ao, ae, width=1.5)
    b.arrow(g1.right(0.5), g2.left(0.5), width=2)

    g3 = zone(b, 1290, 96, 290, 380, "3. TOOLS & INTEGRATIONS\nLAYER", "orange", size=14)
    tools = ["Search Engines", "Databases (SQL / NoSQL / Vector DB)", "APIs & Web Services",
             "Enterprise Systems (CRM, ERP, HRMS…)", "Code Execution Environment",
             "File Storage (S3, GCS, Azure Blob…)", "Email / Messaging / Notifications"]
    tboxes = []
    for i, t in enumerate(tools):
        tboxes.append(node(b, 1300, 150 + i * 45, 270, 36, [t], "orange", size=10, first_size=10, fg=INK))
    for tb in tboxes:
        b.arrow((1250, tb.cy), (1298, tb.cy), width=1.5)
    ring(b, 1284, 90, 302, 392, "green", 3.2)
    for k in (2, 3, 4):
        ring(b, tboxes[k].x - 3, tboxes[k].y - 3, tboxes[k].w + 6, tboxes[k].h + 6, "green", 2.6)

    # Row B ---------------------------------------------------------------
    y0 = 506
    g4 = zone(b, 20, y0, 248, 296, "4. CONTEXT & MEMORY\nLAYER", "purple", size=14)
    mems = [("Short-term Memory", ["(Conversation History)"]),
            ("Long-term Memory", ["(User Preferences,", "Knowledge, Past Actions)"]),
            ("Vector Store", ["(Embeddings + Index)"]),
            ("Knowledge Graph", ["(Entities, Relationships)"])]
    my = y0 + 52
    for t, body in mems:
        bx = node(b, 32, my, 224, 44 + 15 * (len(body) - 1), [t] + body, "purple", size=10, first_size=11)
        my += bx.h + 8
    b.arrow(g1.bottom(0.5), (g1.cx, y0), width=1.6)
    b.arrow((330, 476), (250, y0), dashed=True, color="blue", width=1.5)

    g5 = zone(b, 296, y0, 466, 296, "5. GUARDRAILS LAYER (SAFETY & CONTROL)", "red", size=14)
    sub(b, 306, y0 + 38, 234, 246, "Input Guardrails", "red")
    items(b, 314, y0 + 80, ["Prompt Injection Detection", "PII / Secrets Detection",
                            "Toxicity / Hate / Abuse Detection", "Input Validation",
                            "Policy / Compliance Check", "Context Sanitization"],
          size=10, gap=30, mark="☑", mark_color="red")
    so = sub(b, 548, y0 + 38, 204, 246, "Output Guardrails", "red")
    items(b, 556, y0 + 80, ["Toxicity / Harm Detection", "Hallucination Detection", "PII / Secrets Masking",
                            "Fact Verification", "Policy / Compliance Check", "Response Validation",
                            "Refusal / Safe Response"], size=10, gap=28, mark="☑", mark_color="red")
    ring(b, 290, y0 - 6, 478, 308, "green", 3.2)
    ring(b, so.x - 4, so.y - 4, so.w + 8, so.h + 8, "green", 2.6)
    link(b, g4, g5, color="blue", width=1.6)

    g6 = zone(b, 790, y0, 400, 296, "6. LLM GATEWAY LAYER", "green", size=15)
    items(b, 802, y0 + 62, ["Model Routing", "Load Balancing", "Failover & Fallback", "Rate Limiting & Quotas",
                            "Caching (Prompt/Response)", "Cost Optimization", "Model Versioning",
                            "Policy Enforcement", "Audit Logging"], size=11, gap=25, mark="•", mark_color="green")
    sub(b, 990, y0 + 38, 188, 246, "Supported Models", "green", size=11)
    items(b, 1000, y0 + 76, ["OpenAI (GPT-4o)", "Anthropic (Claude)", "Google (Gemini)", "Meta (Llama)",
                             "Mistral AI", "Cohere", "Azure OpenAI", "Self-Hosted Models", "… and more"],
          size=10, gap=23)
    ring(b, 784, y0 - 6, 412, 308, "green", 3.2)
    link(b, g5, g6, color="green", width=1.6)

    g7 = zone(b, 1216, y0, 364, 296, "7. MODEL LAYER", "purple", size=15)
    sub(b, 1228, y0 + 40, 166, 200, "Proprietary Models", "purple", size=11)
    items(b, 1240, y0 + 80, ["GPT-4o / 4.1", "Claude 3.5", "Gemini 1.5", "…"], size=11, gap=30)
    sub(b, 1400, y0 + 40, 168, 200, "Open Source Models", "purple", size=11)
    items(b, 1412, y0 + 80, ["Llama 3 / 3.1", "Mistral", "Mixtral", "Phi / Qwen", "…"], size=11, gap=30)
    link(b, g6, g7, width=1.6)

    # Row C ---------------------------------------------------------------
    y1 = 832
    g8 = zone(b, 20, y1, 980, 186, "8. OBSERVABILITY & EVALUATION LAYER", "orange", size=15)
    obs = [("Logging", ["Requests", "Responses", "Errors", "Audit Logs"], 140),
           ("Tracing", ["Request Traces", "Agent Steps", "Tool Calls", "Latency"], 150),
           ("Metrics & Monitoring", ["Token Usage", "Throughput", "Cost Tracking", "Success Rate",
                                     "User Feedback"], 180),
           ("Evaluation", ["Offline Evals", "Online Evals (A/B, Shadow)", "Quality Metrics (RAGAS, etc.)",
                           "Human Feedback", "Regression Testing"], 232),
           ("Dashboards & Alerts", ["Real-time Dashboards", "Alerts & Notifications", "SLO / SLA Monitoring"], 198)]
    ox = 34
    for t, lst, w in obs:
        sub(b, ox, y1 + 38, w, 138, t, "orange", size=11)
        items(b, ox + 10, y1 + 76, lst, size=10, gap=19, mark="•", mark_color="orange")
        ring(b, ox - 3, y1 + 35, w + 6, 144, "pink", 2.4)
        ox += w + 12

    g9 = zone(b, 1020, y1, 560, 186, "9. SECURITY & GOVERNANCE LAYER", "blue", size=15)
    sub(b, 1032, y1 + 38, 300, 140, "Security Controls", "blue", size=11)
    items(b, 1042, y1 + 74, ["Data Encryption (In-Transit / At-Rest)", "Secrets Management",
                             "Network Security (VPC, Firewalls)", "Least Privilege Access", "MCP / Tool Permissions",
                             "Sandboxing / Code Isolation", "Content Safety & Filtering"], size=10, gap=15.5,
          mark="•", mark_color="blue")
    sub(b, 1342, y1 + 38, 226, 140, "Governance", "blue", size=11)
    items(b, 1352, y1 + 74, ["RBAC / ABAC", "Audit Trails", "Policy Management", "Compliance (GDPR, SOC2, ISO)",
                             "Data Classification", "Retention Policies"], size=10, gap=17.5, mark="•",
          mark_color="blue")
    link(b, g8, g9, dashed=True, color="orange", width=1.6)
    b.arrow(g5.bottom(0.5), (g5.cx, y1), both=True, width=1.4)
    b.arrow(g7.bottom(0.5), (g7.cx, y1), both=True, width=1.4)
    b.arrow((g2.x + 610, 476), (g2.x + 610, y0), both=True, width=1.4)

    # Row D ---------------------------------------------------------------
    y2 = 1046
    g10 = zone(b, 20, y2, 420, 190, "10. INFRASTRUCTURE LAYER", "grey", size=14)
    infra = [("Cloud / On-Prem", "AWS, Azure, GCP"), ("Compute", "VMs / Kubernetes, Containers / Serverless"),
             ("Storage", "Object Storage, Block / File / DB"), ("Networking", "VPC / CDN, Load Balancer")]
    for i, (t, d) in enumerate(infra):
        b.card(32 + (i % 2) * 202, y2 + 40 + (i // 2) * 72, 194, 64, t, [d], "grey", size=10, title_size=11)
    g11 = zone(b, 460, y2, 720, 190, "11. DEVOPS & DELIVERY PIPELINE", "teal", size=14)
    steps = [["Code", "Commit"], ["Build"], ["Tests"], ["Security", "Scan", "(SAST/DAST)"], ["Container", "Build"],
             ["Deploy"], ["Monitor"], ["Feedback", "Loop"]]
    prev = None
    for i, s in enumerate(steps):
        bx = node(b, 474 + i * 88, y2 + 48, 72, 70, s, "teal", size=10, first_size=10, fg=INK)
        if prev:
            b.arrow(prev.right(), bx.left(), width=1.4)
        prev = bx
    sub(b, 474, y2 + 136, 692, 40, "", "teal")
    b.text(820, y2 + 161, "IaC (Terraform / CloudFormation) | Config Mgmt | Secrets Mgmt | Blue-Green / Canary "
           "Deployments", 10, INK, "400")
    b.arrow(g10.right(0.5), g11.left(0.5), width=1.6)
    b.arrow((700, y2), (700, y1 + 186), dashed=True, color="orange", width=1.6)
    b.arrow((1100, y2), (1100, y1 + 186), dashed=True, color="orange", width=1.6)
    b.text(710, y2 - 8, "monitoring / feedback", 10, "orange", "700", anchor="start")

    g12 = zone(b, 1195, y2, 195, 190, "12. HUMAN-IN-THE-\nLOOP (OPTIONAL)", "green", size=13)
    items(b, 1207, y2 + 76, ["Human Review", "Approval Workflows", "Feedback Capture", "Override Actions"],
          size=10, gap=24, mark="•", mark_color="green")
    _ = g12

    zone(b, 1400, y2, 180, 190, "LEGEND", "grey", size=13)
    legend = [("Request Flow", INK, False), ("Data / Context Flow", "blue", False), ("Control Flow", "green", False),
              ("Monitoring / Feedback", "orange", True), ("External Interaction", "blue", True)]
    for i, (t, c, d) in enumerate(legend):
        ly = y2 + 60 + i * 26
        b.arrow((1410, ly), (1440, ly), color=c, dashed=d, width=1.6)
        b.text(1446, ly + 4, t, 10, INK, "400", anchor="start")

    b.card(20, 1256, 1560, 44, "", [], "yellow")
    heading(b, [("★ AGENTIC AI IS POWERFUL, BUT SECURE ARCHITECTURE MAKES IT ", INK),
                ("TRUSTWORTHY, RELIABLE & PRODUCTION READY.", "red"), (" ★", "yellow")], 1284, 15)
    note(b, 20, 1330, "Green and pink outlines are the highlight boxes Krish drew live while explaining each layer.")
    return b


@board
def s1_gateway_sketch():
    """0:28-0:31 whiteboard below the architecture slide: why a gateway sits between clients and providers."""
    W, H = 1300, 560
    b = Board(W, H, "Why an LLM gateway?", "Krish's whiteboard sketch under the architecture slide (layer 6)")

    # Before: clients call providers directly
    zone(b, 20, 96, 380, 400, "First: every client calls a provider", "grey", size=13)
    ui = b.card(40, 150, 130, 250, "", [], "green")
    _ = ui
    names = ["Chat", "API", "Coworker"]
    provs = ["OpenAI API", "Gemini API", "Anthropic API"]
    for i, (n, p) in enumerate(zip(names, provs)):
        y = 190 + i * 80
        b.text(105, y + 5, n, 15, INK, "700")
        pb = node(b, 250, y - 17, 136, 36, [p], "grey", size=12, first_size=12, fg=INK)
        b.arrow((170, y), pb.left(), width=1.6)
    b.text(210, 450, "each client is wired\nto one provider", 12, FAINT, "700", line_gap=1.3)

    # After: the gateway in between
    zone(b, 420, 96, 860, 440, "Then: one LLM gateway in the middle", "pink", size=13)
    hand(b, 470, 170, "UI", "pink", 22)
    uib = b.card(500, 176, 150, 260, "", [], "green")
    ys = [226, 306, 386]
    for n, y in zip(names, ys):
        b.text(575, y + 6, n, 16, INK, "700")
    ring(b, 530, 204, 90, 36, "pink", 2.2)
    curve(b, (560, 204), (606, 158), 0.5, "pink")
    curve(b, (640, 150), (612, 202), 0.4, "pink")
    for y in ys:
        line(b, 650, y, 700, y, INK, 1.8)
    hand(b, 700, 170, "Config", "pink", 18)
    curve(b, (655, 236), (738, 262), -0.35, "pink")
    curve(b, (655, 380), (738, 340), 0.35, "pink")
    gw = b.card(745, 150, 210, 330, "", [], "pink")
    hand(b, 770, 205, "LLM Caching", "pink", 16, anchor="start")
    hand(b, 770, 245, "Guardrails", "pink", 16, anchor="start")
    b.text(850, 322, "LLM Gateway", 22, "red", "700")
    hand(b, 770, 400, "Evaluation", "pink", 16, anchor="start")
    note(b, 770, 422, "(half-written on the board)", 10)
    _ = uib, gw

    hand(b, 1010, 168, "(Config)", "red", 17)
    hand(b, 1005, 205, "Routing", "red", 18)
    underline(b, 970, 1040, 214, "red", double=True)
    pts = [(1060, 250), (1060, 322), (1060, 400)]
    for (px, py) in pts:
        pen_arrow(b, (958, 320), (px - 6, py), "red", width=2.2)
    for (px, py), p in zip(pts, provs):
        node(b, px, py - 20, 150, 40, [p], "grey", size=13, first_size=13, fg=INK)
    curve(b, (1214, 262), (1214, 306), -0.6, "pink")
    curve(b, (1214, 338), (1214, 384), -0.6, "pink")
    hand(b, 1244, 240, "↓↓↓", "pink", 16)
    b.text(1135, 468, "fallback order:", 11, "pink", "700")
    b.text(1135, 486, "OpenAI → Gemini → Anthropic", 11, "pink", "700")
    return b


@board
def s1_production_goals():
    """0:46-0:47 and 1:10-1:11 Scribble Ink pages 4-5 (Divesh): the target and its four goals."""
    W, H = 1240, 800
    b = Board(W, H)
    # centre title, as typed on the board
    for i, t in enumerate(["PRODUCTION GRADE", "SCALABLE", "ADVANCE RAG"]):
        b.text(620, 250 + i * 50, t, 34, "green", "700")
    underline(b, 468, 772, 260, "red", width=2.6)
    underline(b, 510, 730, 360, "pink", width=2.6)

    # left: user traffic and data
    b.text(215, 118, "100000", 30, "yellow", "700")
    underline(b, 150, 290, 128, "yellow", double=True)
    ut = node(b, 140, 150, 150, 50, ["User Traffic"], "yellow", size=14, first_size=15)
    note(b, 215, 222, "how many users will use it", 11, FAINT, "middle")
    da = node(b, 140, 330, 150, 50, ["Data"], "yellow", size=14, first_size=15)
    pen_arrow(b, (455, 290), (ut.x + ut.w + 6, ut.cy + 6), "yellow", width=2.4)
    pen_arrow(b, (455, 300), (da.x + da.w + 6, da.cy - 4), "yellow", width=2.4)
    n90 = node(b, 60, 440, 150, 56, ["90 %", "noise"], "grey", size=12, first_size=16)
    n10 = node(b, 240, 440, 170, 56, ["10 %  True", "useful data"], "green", size=12, first_size=16)
    pen_arrow(b, (185, 382), (135, 438), "yellow", width=2.2)
    pen_arrow(b, (230, 382), (310, 438), "yellow", width=2.2)
    tick(b, 418, 452, "red", 1.1)
    tick(b, 426, 462, "red", 1.1)
    _ = n90, n10

    # top right: org, not public
    og = node(b, 690, 88, 110, 46, ["org"], "red", size=15, first_size=17)
    pb = node(b, 900, 70, 130, 46, ["public"], "red", size=15, first_size=17)
    underline(b, 910, 1020, 124, "red", double=True)
    pen_arrow(b, (600, 222), (og.x + 10, og.y + og.h + 4), "red", width=2.2)
    curve(b, (og.x + og.w - 10, og.y), (pb.x + 10, pb.y + 4), -0.35, "red")
    pen_arrow(b, (pb.x - 6, pb.y + pb.h + 14), (og.x + og.w + 6, og.y + og.h - 6), "red", width=2.2)
    tick(b, 846, 140, "red", 0.9)
    note(b, 860, 170, "an assistant for the organisation,", 11, FAINT, "middle")
    note(b, 860, 186, "not a public chatbot", 11, FAINT, "middle")
    b.text(1135, 250, "90 %", 24, "red", "700")
    underline(b, 1095, 1180, 262, "red")
    b.text(1100, 330, "Assistance", 24, "red", "700")
    underline(b, 1010, 1190, 342, "red")

    # advanced = RAG + agentic behaviour
    curve(b, (780, 372), (955, 452), 0.35, "pink", 2.4)
    rg = node(b, 960, 428, 100, 46, ["Rag"], "pink", size=15, first_size=17)
    pen_arrow(b, (rg.x + rg.w + 4, rg.cy), (1112, rg.cy), "pink", width=2.2)
    b.person(1150, 412, "pink", 0.95)
    b.text(1100, 506, "agentic behaviour is involved:", 11, "pink", "700")
    b.text(1100, 522, "that is why it is advanced RAG", 11, "pink", "700")

    # page break, then the four goals (1:10 to 1:11)
    line(b, 20, 560, W - 20, 560, "#adb5bd", 1.5)
    note(b, 24, 580, "Written at 1:10 to 1:11, below the target")
    goals = [("1", "Robust", ""), ("2", "Reliable", ""),
             ("3", "Secure", "no one can bypass our chatbots or AI systems"),
             ("4", "Scalable", "serves many users")]
    for i, (n, t, gloss) in enumerate(goals):
        cx = 120 + (i // 2) * 600
        cy = 630 + (i % 2) * 100
        ellipse(b, cx, cy, 28, 28, "pink", 2.4, fill="#fff0f6")
        b.text(cx, cy + 9, n, 24, "pink", "700")
        b.text(cx + 50, cy + 11, t, 30, "pink", "700", anchor="start")
        tick(b, cx + 60 + len(t) * 18 + 18, cy - 4, "yellow", 1.6)
        if gloss:
            note(b, cx + 52, cy + 34, gloss, 12, FAINT)
    return b


@board
def s1_request_path():
    """1:04-1:07 prepared image 'Production grade advanced RAG', with Divesh's pen notes (more at 5:00-5:02)."""
    W, H = 1400, 800
    b = Board(W, H)
    b.text(W / 2, 48, "PRODUCTION GRADE ADVANCED RAG", 28, INK, "700")
    b.card(230, 70, 820, 644, "", [], "grey")

    b.person(300, 96, "blue", 1.0, "User")
    st = node(b, 370, 104, 140, 56, ["Streamlit UI"], "blue", size=13, first_size=14)
    fa = node(b, 560, 100, 140, 64, ["FastAPI", "/query"], "blue", size=13, first_size=14)
    gr = b.diamond(800, 140, 160, 108, "NeMo\nGuardrails", "red", size=13)
    b.arrow((330, 132), st.left())
    _ = fa
    b.arrow(st.right(), fa.left())
    b.arrow(fa.right(), gr.left(0.5))
    b.text(935, 116, "Blocked", 14, "red", "700")
    raw(b, f'<path d="M{gr.x + gr.w},{gr.cy} H1010 V690" fill="none" stroke="{INK}" stroke-width="1.8"/>')

    pl = node(b, 735, 250, 130, 60, ["Planner", "Node"], "orange", size=13, first_size=13)
    b.arrow(gr.bottom(0.5), pl.top(), label="Pass", label_color="green", label_dx=26)
    rs = node(b, 610, 380, 130, 60, ["Responder", "Node"], "purple", size=13, first_size=13)
    rt = node(b, 850, 380, 130, 60, ["Retriever", "Node"], "blue", size=13, first_size=13)
    b.arrow(pl.left(), rs.top(), via=[(rs.cx, pl.cy)])
    b.arrow(pl.right(), rt.top(), via=[(rt.cx, pl.cy)])
    b.text(615, 350, "Conversational", 12, "green", "700")
    b.text(962, 350, "Technical", 12, "green", "700")
    fr = node(b, 835, 480, 160, 60, ["FlashRank", "Local Reranker"], "blue", size=12, first_size=13)
    b.arrow(rt.bottom(), fr.top())
    b.arrow(fr.left(), rs.bottom(0.55), via=[(rs.x + rs.w * 0.55, fr.cy)])
    b.arrow(rs.left(), st.bottom(), via=[(st.cx, rs.cy)], both=True)
    ms = b.cylinder(845, 600, 140, 84, "LangGraph", ["MemorySaver"], "purple", size=12)
    b.arrow(rs.bottom(0.3), ms.left(0.55), via=[(rs.x + rs.w * 0.3, ms.y + ms.h * 0.55)], dashed=True,
            color="purple")

    # pen notes, 1:04-1:07
    for bx in (st, fa, pl, rt, fr):
        tick(b, bx.x + bx.w + 4, bx.y + 6, "red", 1.0)
    hand(b, 330, 230, "FE", "red", 30)
    pen_arrow(b, (360, 210), (st.cx - 6, st.y + st.h + 6), "red", width=2.2)
    hand(b, 612, 240, "BE", "red", 30)
    pen_arrow(b, (560, 196), (fa.cx + 10, fa.y + fa.h + 4), "red", width=2.2)
    hand(b, 1110, 92, "safe", "yellow", 24)
    underline(b, 1070, 1150, 102, "yellow", double=True)
    cross(b, 1052, 110, "red", 0.9)
    # side sketch: what an agent looks like
    hand(b, 1160, 200, "User", "red", 22)
    pen_arrow(b, (1160, 212), (1160, 250), "red", width=2)
    hand(b, 1160, 278, "DB", "red", 22)
    underline(b, 1138, 1182, 286, "red", double=True)
    pen_arrow(b, (1160, 296), (1160, 330), "red", width=2)
    hand(b, 1160, 358, "LLM", "red", 22)
    underline(b, 1128, 1192, 366, "red", double=True)
    for (x1, y1, x2, y2) in [(1160, 384, 1130, 414), (1160, 384, 1190, 414), (1190, 414, 1170, 444),
                             (1190, 414, 1212, 444)]:
        line(b, x1, y1, x2, y2, PEN["red"], 2)
    for (cx, cy) in [(1160, 384), (1130, 414), (1190, 414), (1170, 444), (1212, 444)]:
        ellipse(b, cx, cy, 7, 7, "red", 2, fill="#fff5f5")
    note(b, 1160, 474, "user, data, model,", 11, FAINT, "middle")
    note(b, 1160, 489, "then a graph of nodes", 11, FAINT, "middle")

    # pen notes added when the diagram returns at 5:00-5:02
    hand(b, 1300, 110, "simple ✗", "pink", 22)
    hand(b, 945, 176, "↳ sec", "pink", 18, anchor="start")
    hand(b, 120, 380, "K8S", "pink", 30)
    hand(b, 120, 440, "query", "pink", 28)
    underline(b, 70, 180, 452, "pink", double=True)
    hand(b, 400, 610, "Data", "pink", 26)
    hand(b, 420, 646, "retrieval", "pink", 26)
    hand(b, 1300, 560, "Qdrant", "pink", 22)
    underline(b, 1255, 1345, 568, "pink")
    b.cylinder(1265, 584, 70, 70, "DB", [], "pink", size=11)
    pen_arrow(b, (1265, 670), (1016, 690), "pink", width=2.2)
    note(b, 20, 760, "Red and yellow pen notes were added at 1:04 to 1:07; pink ones when the diagram "
         "returned at 5:00 to 5:02 (FE = front end, BE = back end).")
    return b


@board
def s1_full_architecture():
    """1:07-1:08, again 5:02-5:03 and 5:15-5:17: the full project architecture, seven zones and a legend."""
    W, H = 1440, 830
    b = Board(W, H, "Enterprise Agentic RAG: full target architecture")

    z1 = zone(b, 20, 90, 220, 270, "1. User Interface", "blue", size=14)
    ch = node(b, 40, 138, 180, 56, ["Streamlit", "Chat UI"], "blue", size=12, first_size=13)
    ev = node(b, 40, 262, 180, 56, ["Streamlit", "Eval App"], "blue", size=12, first_size=13)
    b.arrow(ev.top(), ch.bottom())

    z2 = zone(b, 260, 90, 330, 270, "2. API + Safety Gate", "red", size=14)
    fa = node(b, 280, 174, 110, 60, ["FastAPI", "/query"], "red", size=12, first_size=13)
    gr = b.diamond(490, 204, 184, 144, "NeMo\nGuardrails", "red", size=12)
    b.text(490, 238, "Blocks · Jailbreak ·", 10, INK, "400")
    b.text(490, 252, "Off-topic · Injection", 10, INK, "400")
    b.arrow(ch.right(), fa.left(0.3), via=[(250, ch.cy), (250, fa.y + fa.h * 0.3)], label="user query",
            label_dy=-4)
    b.arrow(ev.right(), fa.left(0.8), via=[(262, ev.cy), (262, fa.y + fa.h * 0.8)], label="phase 1 query",
            label_dy=6)
    b.arrow(fa.right(), gr.left(0.5))
    b.arrow(gr.bottom(), fa.bottom(), via=[(gr.cx, 312), (fa.cx, 312)], label="✗ blocked", label_color="red")

    z3 = zone(b, 610, 90, 340, 270, "3. Agent Engine — LangGraph", "purple", size=14)
    pl = node(b, 690, 126, 190, 50, ["Planner Node", "Intent Classification"], "purple", size=10, first_size=12)
    rt = node(b, 625, 218, 150, 50, ["Retriever Node", "Vector Search"], "blue", size=10, first_size=12)
    rs = node(b, 790, 218, 150, 50, ["Responder Node", "Answer Generation"], "purple", size=10, first_size=12)
    ms = b.cylinder(760, 288, 140, 62, "MemorySaver", ["Conversation History"], "purple", size=10)
    b.arrow(gr.right(0.5), pl.left(), via=[(640, gr.cy), (640, pl.cy)], label="✓ pass", label_color="green",
            label_dy=-2)
    b.arrow(pl.bottom(0.2), rt.top(0.6), label="technical", label_dx=-26)
    b.arrow(pl.bottom(0.8), rs.top(0.4), label="conversational", label_dx=34)
    b.arrow(rt.right(), rs.left())
    b.arrow(rs.bottom(0.35), (rs.x + rs.w * 0.35, ms.y + 4), both=True, dashed=True, color="purple")
    b.arrow(rs.right(0.3), pl.right(0.6), via=[(944, rs.y + rs.h * 0.3), (944, pl.y + pl.h * 0.6)],
            dashed=True, color="purple")

    z4 = zone(b, 970, 90, 450, 270, "4. Knowledge & LLMs", "orange", size=14)
    qd = b.cylinder(985, 126, 120, 70, "Qdrant Cloud", ["Vector DB"], "orange", size=10)
    pk = node(b, 1130, 128, 140, 58, ["Portkey Gateway", "Routing + Fallback"], "orange", size=10, first_size=11)
    gp = node(b, 1292, 128, 118, 58, ["Groq Primary", "Llama 3.3 · 70B"], "orange", size=10, first_size=11)
    fr = node(b, 985, 238, 120, 56, ["FlashRank", "Local Reranker"], "orange", size=10, first_size=11)
    gf = node(b, 1140, 262, 124, 58, ["Groq Fallback", "Llama 3.1 · 8B"], "orange", size=10, first_size=11)
    b.arrow(qd.right(0.45), pk.left(0.45))
    b.arrow(pk.right(), gp.left())
    b.arrow(pk.bottom(), gf.top(), dashed=True, label="fallback", label_dx=30)
    b.arrow(qd.bottom(), fr.top())
    b.arrow(fr.left(0.7), ms.right(0.5), via=[(960, fr.y + fr.h * 0.7), (960, ms.y + ms.h * 0.5)], both=True)
    b.arrow(pl.top(), pk.top(), via=[(pl.cx, 76), (pk.cx, 76)])
    b.text(1040, 70, "planner / responder LLM calls", 10, FAINT, "700")

    z5 = zone(b, 20, 392, 610, 170, "5. Data Ingestion", "blue", size=14)
    dl = node(b, 36, 440, 180, 70, ["Document Loaders", "PDF · HTML · DOCX", "PPTX · TXT"], "blue", size=10,
              first_size=12)
    pd = b.cylinder(245, 432, 150, 86, "processed_data/", ["Local JSON", "Chunks"], "blue", size=10)
    ge = node(b, 422, 440, 196, 70, ["Gemini Embeddings", "gemini-embedding-2-", "preview · 3072-dim"], "blue",
              size=10, first_size=12)
    b.arrow(dl.right(), pd.left(0.55))
    b.arrow(pd.right(0.55), ge.left())
    b.arrow(ge.right(0.4), rt.bottom(0.3), via=[(650, ge.y + ge.h * 0.4), (650, 330), (rt.x + rt.w * 0.3, 330)])
    b.arrow(z1.bottom(0.5), (z1.cx, 392), dashed=True, color="blue", width=1.4)

    z6 = zone(b, 670, 392, 750, 190, "6. Evaluation Suite — RAGAS", "pink", size=14)
    gd = b.cylinder(690, 440, 150, 96, "Golden Dataset", ["15 RAG Samples", "6 Guardrail Tests"], "pink", size=10)
    rm = node(b, 890, 430, 250, 70, ["RAGAS Metrics", "Faithfulness · Relevancy ·", "Precision · Recall · Correctness"],
              "pink", size=10, first_size=12)
    jd = node(b, 1200, 440, 200, 56, ["Judge LLM", "Groq · Separate Key"], "pink", size=10, first_size=12)
    tc = node(b, 890, 514, 250, 50, ["Tool Correctness", "Jaccard · Zero LLM Cost"], "pink", size=10, first_size=12)
    b.arrow(gd.right(0.35), rm.left(0.5), color="red")
    b.arrow(gd.right(0.75), tc.left(0.5), color="red")
    b.arrow(rm.right(0.5), jd.left(0.4), color="red")

    z7 = zone(b, 20, 620, 610, 150, "7. Monitoring & Observability", "green", size=14)
    lf = node(b, 60, 680, 220, 56, ["Pydantic Logfire", "Distributed Tracing"], "green", size=11, first_size=12)
    ls = node(b, 380, 680, 220, 56, ["LangSmith", "Agent Step Tracing"], "green", size=11, first_size=12)
    b.arrow(dl.bottom(0.5), lf.top(0.5), dashed=True, color="green", label="spans")
    b.arrow(ge.bottom(0.5), ls.top(0.6), dashed=True, color="green", label="traces")

    zone(b, 670, 610, 400, 190, "Legend (Flow Types)", "grey", size=13)
    flows = [("Query / Response Flow", INK, False), ("Feedback / Memory Flow", INK, True),
             ("Data Ingestion Flow", "blue", False), ("Evaluation Flow", "red", False),
             ("Observability Flow", "green", True)]
    for i, (t, c, d) in enumerate(flows):
        y = 660 + i * 27
        b.arrow((690, y), (760, y), color=c, dashed=d, width=1.8)
        b.text(775, y + 4, t, 11, INK, "400", anchor="start")
    zone(b, 1090, 610, 330, 190, "Colour key", "grey", size=13)
    key = [("User Interface", "blue"), ("API + Safety", "red"), ("Agent Engine", "purple"),
           ("Knowledge & LLMs", "orange"), ("Data Ingestion", "blue"), ("Evaluation", "pink"),
           ("Observability", "green"), ("Memory (cylinder)", "purple")]
    for i, (t, c) in enumerate(key):
        x = 1106 + (i // 4) * 160
        y = 650 + (i % 4) * 34
        raw(b, f'<rect x="{x}" y="{y}" width="18" height="14" rx="3" fill="{PALETTE[c]["fill"]}" '
               f'stroke="{PALETTE[c]["stroke"]}" stroke-width="1.5"/>')
        b.text(x + 26, y + 12, t, 10, INK, "400", anchor="start")
    note(b, 20, 816, "Session 1 builds the chat UI, FastAPI, the LangGraph engine, ingestion, Qdrant with FlashRank, "
         "and tracing. The eval app, the NeMo gate, the Portkey gateway and the RAGAS suite arrive in Session 2.")
    _ = z2, z3, z4, z5, z6, z7
    return b


@board
def s1_prototype_to_cloud():
    """1:09-1:11 Scribble Ink page 5, below the goals; 'gemini', 'sent' and 'jina' added at 5:03."""
    W, H = 1240, 630
    b = Board(W, H, "From prototype to cloud", "Divesh's roadmap for the project")
    row = [("Prototype", "yellow", True), ("laptop", "grey", False), ("groq", "yellow", False),
           ("Test", "yellow", False), ("MVP", "yellow", False)]
    boxes = []
    for i, (t, c, dbl) in enumerate(row):
        bx = node(b, 60 + i * 230, 96, 170, 52, [t], c, size=16, first_size=18, fg=INK)
        boxes.append(bx)
    underline(b, boxes[0].x + 20, boxes[0].x + boxes[0].w - 20, 158, "yellow", double=True)
    underline(b, boxes[2].x + 40, boxes[2].x + boxes[2].w - 40, 158, "yellow", double=True)
    underline(b, boxes[4].x + 40, boxes[4].x + boxes[4].w - 40, 158, "yellow")
    for a, c in zip(boxes, boxes[1:]):
        b.arrow(a.right(), c.left(), color="#adb5bd", width=1.4)
    note(b, 620, 186, "build the first version on a laptop with Groq, test it, reach an MVP", 12, FAINT, "middle")

    ld = node(b, 260, 226, 290, 130, ["Local", "Dev"], "red", size=30, first_size=30, fg=INK)
    cd = node(b, 700, 226, 290, 130, ["Cloud", "deploy"], "red", size=30, first_size=30, fg=INK)
    pen_arrow(b, (150, ld.cy), (ld.x - 6, ld.cy), "yellow", width=2.4)
    pen_arrow(b, (ld.x + ld.w + 8, ld.cy), (cd.x - 6, cd.cy), "yellow", width=2.4)
    pen_arrow(b, (cd.x + cd.w + 8, cd.cy), (1060, cd.cy), "yellow", width=2.4)
    hand(b, 1120, cd.cy + 10, "AWS", "yellow", 30)

    pen_arrow(b, (ld.cx - 40, ld.y + ld.h + 6), (ld.cx - 40, 410), "yellow", width=2.4)
    hand(b, ld.cx - 30, 444, "Opensource", "yellow", 28)
    underline(b, ld.cx - 120, ld.cx + 70, 456, "yellow", double=True)
    hand(b, ld.cx - 10, 500, "↳ avg", "yellow", 22, anchor="start")
    hand(b, ld.cx + 20, 536, "↳ good", "yellow", 22, anchor="start")
    note(b, ld.cx - 30, 570, "open-source models: average to good", 12, FAINT, "middle")

    hand(b, 110, 440, "gemini", "yellow", 24)
    underline(b, 60, 160, 450, "yellow")
    hand(b, 110, 504, "sent", "yellow", 24)
    underline(b, 80, 140, 514, "yellow", double=True)
    note(b, 110, 540, "embeddings: Gemini,", 11, FAINT, "middle")
    note(b, 110, 555, "sentence-transformers", 11, FAINT, "middle")
    note(b, 110, 570, "as the fallback", 11, FAINT, "middle")

    hand(b, cd.cx + 20, 470, "CI – CD", "yellow", 34)
    underline(b, cd.cx - 70, cd.cx + 110, 486, "yellow", double=True)
    hand(b, 1120, 380, "jina", "yellow", 28)
    underline(b, 1080, 1160, 392, "yellow", double=True)
    tick(b, 1170, 410, "yellow", 1.2)
    note(b, 1120, 430, "Jina embeddings and", 11, FAINT, "middle")
    note(b, 1120, 445, "reranker in the cloud", 11, FAINT, "middle")
    note(b, 20, 614, "'gemini', 'sent' and 'jina' were added when the page was revisited at 5:03.")
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        path = BOARDS[name]().save(OUT / f"{name.replace('_', '-')}.svg")
        print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
