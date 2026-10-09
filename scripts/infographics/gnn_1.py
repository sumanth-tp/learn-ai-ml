"""Infographics for docs/theory/gnn.

Run from the repo root:

    python3 scripts/infographics/gnn_1.py              # all boards
    python3 scripts/infographics/gnn_1.py steps        # boards whose name ends with the argument
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "gnn"
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


def vertex(b, cx, cy, label, color="blue", r=22, sub=None):
    c = PALETTE[color]
    b.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="2.2"/>')
    raw_text(b, cx, cy + 5, label, 15, c["text"], weight="700")
    if sub:
        raw_text(b, cx, cy + r + 18, sub, 13, INK)


def polyline(b, pts, stroke, width=3, dash=None):
    d = " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(f'<path d="M{d}" fill="none" stroke="{stroke}" stroke-width="{width}"{da} stroke-linejoin="round" stroke-linecap="round"/>')


TINY = {0: (110, 190), 1: (110, 330), 2: (260, 260), 3: (410, 260), 4: (540, 260)}
TINY_EDGES = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]


@board("message-passing-steps")
def message_passing_steps():
    b = Board(1160, 600, "One round of message passing on five nodes", "Every node starts with a number; each round replaces it by a summary of its neighbours")
    b.group(20, 90, 640, 490, "The graph and the three summaries", "blue")
    for i, j in TINY_EDGES:
        line(b, *TINY[i], *TINY[j], PALETTE["grey"]["stroke"], 2.2)
    for i, (x, y) in TINY.items():
        vertex(b, x, y, str(i + 1), "blue", 24, f"node {i}")
    raw_text(b, 340, 140, "circles show the start values 1 to 5; labels give node numbers", 13, FAINT)
    b.table(40, 430, [150, 90, 90, 90, 90, 90], [
        ["summary", "node 0", "node 1", "node 2", "node 3", "node 4"],
        ["sum", "5", "4", "7", "8", "4"],
        ["mean", "2.5", "2.0", "2.333", "4.0", "4.0"],
        ["GCN step", "1.866", "1.866", "2.771", "4.241", "4.133"],
    ], "blue", size=14, row_h=34)
    b.group(690, 90, 450, 490, "How node 2 is computed", "orange")
    b.card(710, 130, 410, 120, "Sum", ["node 2 has neighbours 0, 1, 3", "start values 1, 2, 4", "1 + 2 + 4 = 7"], "teal", size=14, title_size=16)
    b.card(710, 270, 410, 120, "Mean", ["7 divided by 3 neighbours", "= 2.333"], "green", size=14, title_size=16)
    b.card(710, 410, 410, 150, "GCN step", ["weights 0.289 for each of nodes 0, 1, 3", "and 0.25 for node 2 itself", "0.289 x 1 + 0.289 x 2 + 0.25 x 3", "+ 0.289 x 4 = 2.771 (exact weights)"], "orange", size=14, title_size=16)
    return b


@board("karate-hops")
def karate_hops():
    b = Board(1160, 600, "Two labelled nodes classify a club, until there are too many layers", "Zachary's karate club: 34 members, 78 friendships, two factions of 17; seeds are members 0 and 33")
    b.group(20, 90, 560, 490, "Propagating the two labels", "green")
    b.table(40, 135, [100, 170, 150, 120], [
        ["hops", "nodes reached", "accuracy", "guesses"],
        ["1", "31", "0.941", "2"],
        ["2", "34", "0.941", "2"],
        ["3", "34", "0.971", "2"],
        ["10", "34", "0.971", "2"],
        ["20", "34", "0.941", "2"],
        ["50", "34", "0.500", "1"],
        ["200", "34", "0.500", "1"],
    ], "green", size=15, row_h=44)
    raw_text(b, 300, 530, "after 50 hops every node gets the same guess", 14, "#c92a2a", weight="700")
    b.group(610, 90, 530, 230, "Receptive field of member 0", "blue")
    for k, (n, label) in enumerate([(17, "1 layer"), (26, "2 layers"), (34, "3 layers")]):
        y = 150 + k * 50
        raw_text(b, 700, y + 13, label, 14, INK, anchor="end")
        b.bar(720, y, 330, n / 34, color="blue", h=20)
        raw_text(b, 1060, y + 15, f"{n} of 34", 14, INK, anchor="start", weight="700")
    b.group(610, 340, 530, 240, "What to take from it", "yellow")
    b.card(630, 385, 490, 80, "Three hops reach everyone", ["a deep stack is not needed to see the whole graph"], "yellow", size=14, title_size=15)
    b.card(630, 480, 490, 80, "Too many hops erase the signal", ["repeated averaging pulls all nodes together"], "red", size=14, title_size=15)
    return b


@board("wl-limits")
def wl_limits():
    b = Board(1160, 600, "What neighbour summaries can and cannot tell apart", "Colour refinement (the Weisfeiler-Leman test) bounds what a message-passing network can distinguish")
    b.group(20, 90, 560, 240, "Same colours, different graphs", "red")
    cx, cy, r = 150, 220, 55
    pts = [(cx + r * math.cos(math.pi / 3 * k - math.pi / 2), cy + r * math.sin(math.pi / 3 * k - math.pi / 2)) for k in range(6)]
    for k in range(6):
        line(b, *pts[k], *pts[(k + 1) % 6], PALETTE["grey"]["stroke"], 2.2)
    for k, (x, y) in enumerate(pts):
        vertex(b, x, y, "", "red", 9)
    raw_text(b, cx, 305, "hexagon", 13, INK)
    for ox in (350, 470):
        tri = [(ox, 170), (ox - 38, 245), (ox + 38, 245)]
        for k in range(3):
            line(b, *tri[k], *tri[(k + 1) % 3], PALETTE["grey"]["stroke"], 2.2)
        for x, y in tri:
            vertex(b, x, y, "", "red", 9)
    raw_text(b, 410, 305, "two triangles", 13, INK)
    raw_text(b, 300, 135, "every node has degree 2, so every colour round is identical", 12, FAINT)
    b.group(600, 90, 540, 240, "Different colours, told apart", "green")
    b.table(620, 135, [210, 300], [
        ["pair", "colours after 3 rounds"],
        ["path of 6", "[0, 0, 1, 1, 2, 2]"],
        ["double star", "[0, 0, 0, 0, 1, 1]"],
        ["hexagon vs triangles", "identical lists"],
    ], "green", size=14, row_h=44)
    b.group(20, 350, 1120, 230, "Which summary keeps the information", "purple")
    b.table(40, 395, [200, 140, 140, 140, 460], [
        ["neighbour values", "sum", "mean", "max", "what it shows"],
        ["{1, 1, 2}", "4", "1.333", "2", "sum and mean differ from {1, 2}; max does not"],
        ["{1, 2}", "3", "1.5", "2", "max cannot tell it from {1, 1, 2}"],
        ["{1, 1}", "2", "1.0", "1", "mean cannot tell it from {1}"],
        ["{1}", "1", "1.0", "1", "sum counts the neighbours"],
    ], "purple", size=14, row_h=32)
    return b

@board("three-layers")
def three_layers():
    b = Board(1160, 620, "Three rules for how much a node listens to each neighbour", "Weights node 2 gives its inputs in the five-node graph (itself, then nodes 0, 1 and 3)")
    cols = [
        (20, "GCN", "blue", ["weights fixed by degree", "0.250, 0.289, 0.289, 0.289", "one matrix W for everyone", "45 parameters (8 in, 5 out)"], "h' = relu(S h W)"),
        (400, "GraphSAGE", "green", ["mean of neighbours, self kept apart", "0.333, 0.333, 0.333 (+ own value)", "two matrices", "85 parameters (8 in, 5 out)"], "h' = relu(W_self h + W_nb mean(h_nb))"),
        (780, "GAT", "orange", ["weights learned from the pair", "0.588, 0.131, 0.065, 0.216", "one matrix plus scoring vectors", "110 parameters (2 heads of 5)"], "h' = relu(sum alpha_ij W h_j)"),
    ]
    for x, name, col, lines, formula in cols:
        b.group(x, 90, 360, 330, name, col)
        b.card(x + 20, 135, 320, 70, "Rule", [formula], col, size=12, title_size=14)
        b.card(x + 20, 225, 320, 170, "On this graph", lines, col, size=13, title_size=14)
    b.group(20, 440, 1120, 160, "Checked against the library", "purple")
    b.table(40, 485, [360, 240, 240, 240], [
        ["largest difference from torch_geometric", "GCNConv", "SAGEConv", "GATConv"],
        ["same weights, same graph", "1.2e-07", "0.0e+00", "2.4e-07"],
    ], "purple", size=14, row_h=40)
    return b


@board("homophily-sweep")
def homophily_sweep():
    b = Board(1160, 600, "The graph helps only while its edges join similar nodes", "240 nodes, 3 communities, 30 labels, 10 seeds per cell; share of edges within a class falls from 0.81 to 0.40")
    cols = [("0.812 of edges within a class", 20, [("mlp", 0.647), ("gcn", 0.974), ("sage", 0.973), ("gat", 0.969)]),
            ("0.566 within a class", 400, [("mlp", 0.647), ("gcn", 0.669), ("sage", 0.795), ("gat", 0.679)]),
            ("0.400 within a class", 780, [("mlp", 0.647), ("gcn", 0.385), ("sage", 0.572), ("gat", 0.305)])]
    colors = {"mlp": "grey", "gcn": "blue", "sage": "green", "gat": "orange"}
    for title, x, vals in cols:
        b.group(x, 90, 360, 400, title, "grey")
        for k, (name, v) in enumerate(vals):
            y = 150 + k * 80
            raw_text(b, x + 50, y + 15, name, 14, INK, anchor="end")
            b.bar(x + 65, y, 230, v, color=colors[name], h=22)
            raw_text(b, x + 305, y + 17, f"{v:.3f}", 14, INK, anchor="start", weight="700")
    b.card(20, 510, 1120, 80, "Reading it", ["with clean edges all three graph models beat the feature-only MLP by 0.32; with mostly cross-class edges GCN and GAT fall below it, GraphSAGE falls less"], "yellow", size=14, title_size=15)
    return b


@board("attention-steps")
def attention_steps():
    b = Board(1160, 560, "How GAT turns four scores into four weights", "Node 2 scoring itself and nodes 0, 1 and 3; LeakyReLU slope 0.2")
    b.group(20, 90, 760, 440, "Step by step", "orange")
    b.table(40, 135, [190, 140, 140, 140, 110], [
        ["input", "raw score", "after LeakyReLU", "exp", "weight"],
        ["itself", "2.0", "2.0", "7.389", "0.5876"],
        ["node 0", "0.5", "0.5", "1.649", "0.1311"],
        ["node 1", "-1.0", "-0.2", "0.819", "0.0651"],
        ["node 3", "1.0", "1.0", "2.718", "0.2162"],
        ["total", "", "", "12.575", "1.0000"],
    ], "orange", size=14, row_h=44)
    raw_text(b, 400, 440, "weight = exp(value) divided by the total of exp(values)", 13, FAINT)
    raw_text(b, 400, 470, "a negative score is shrunk to a fifth, not removed", 13, FAINT)
    b.group(800, 90, 340, 440, "What training learned", "green")
    b.card(820, 140, 300, 150, "Share of attention on same-class neighbours", ["0.823 against 0.817 of neighbours", "with clean edges (p_out 0.01)"], "green", size=13, title_size=14)
    b.card(820, 310, 300, 150, "With noisy edges", ["0.576 against 0.567 of neighbours", "attention stayed close to uniform"], "yellow", size=13, title_size=14)
    return b

@board("task-types")
def task_types():
    b = Board(1160, 640, "Three kinds of graph task, each with its own trap", "Numbers printed by blocks 1, 2 and 5 on simulated data with known structure")
    cols = [
        (20, "Node: fraud rings", "red", "Is this account part of a ring?", [
            ("features only", "0.228"), ("+ degree and clustering", "0.603"), ("GNN", "0.852"), ("GNN, edges shuffled", "0.176")],
         "average precision, random ranking 0.05", "the graph is the signal; shuffle it and the model falls below features alone"),
        (400, "Link: recommendation", "green", "Will these two nodes be linked?", [
            ("common neighbours", "0.604"), ("Adamic-Adar", "0.604"), ("GNN, honest", "0.762"), ("GNN, test edges in graph", "0.813")],
         "AUC on held-out edges", "leaving test edges in the message-passing graph inflates AUC"),
        (780, "Graph: whole-graph label", "blue", "Is this graph large?", [
            ("sum readout", "0.842"), ("mean readout", "0.594"), ("max readout", "0.594"), ("majority guess", "0.571")],
         "test accuracy, 3-regular graphs", "mean and max cannot see size"),
    ]
    for x, title, col, question, rows, unit, trap in cols:
        b.group(x, 90, 360, 520, title, col)
        raw_text(b, x + 180, 135, question, 13, INK)
        b.table(x + 15, 155, [215, 115], [["model", "score"]] + [[a, v] for a, v in rows], col, size=13, row_h=38)
        raw_text(b, x + 180, 385, unit, 12, FAINT)
        b.card(x + 20, 410, 320, 170, "The trap", [trap], col, size=13, title_size=15)
    return b


@board("depth-collapse")
def depth_collapse():
    b = Board(1160, 600, "Depth without skip connections erases the nodes", "GCN with 1 to 32 hidden graph layers, 30 labels, 4 seeds; similarity is the mean cosine between nodes of different classes")
    b.group(20, 90, 1120, 400, "Accuracy and similarity by depth", "purple")
    b.table(40, 135, [140, 200, 260, 260, 220], [
        ["layers", "plain accuracy", "plain similarity", "skip accuracy", "skip similarity"],
        ["1", "0.948", "0.295", "0.937", "0.304"],
        ["2", "0.985", "0.239", "0.965", "0.378"],
        ["4", "0.890", "0.360", "0.982", "0.341"],
        ["8", "0.347", "1.000", "0.972", "0.254"],
        ["16", "0.345", "1.000", "0.917", "0.368"],
        ["32", "0.318", "1.000", "0.485", "0.781"],
    ], "purple", size=15, row_h=44)
    b.card(20, 510, 540, 70, "Plain stack", ["from 8 layers every node has the same embedding"], "red", size=14, title_size=15)
    b.card(600, 510, 540, 70, "Skip connections", ["hold to 16 layers, then they too begin to blur"], "green", size=14, title_size=15)
    return b


@board("sampling-footprint")
def sampling_footprint():
    b = Board(1160, 600, "One batch of 64 nodes, and how much of the graph it drags in", "Barabasi-Albert graph, 100,000 nodes, 499,975 edges, mean degree 10, largest degree 1,141")
    b.group(20, 90, 1120, 380, "Nodes loaded to embed 64 targets", "orange")
    rows = [("1 layer, all neighbours", 862), ("2 layers, all neighbours", 20871), ("3 layers, all neighbours", 94068),
            ("2 layers, sample 10 then 5", 2530), ("3 layers, sample 10, 5, 5", 10378)]
    for k, (label, n) in enumerate(rows):
        y = 140 + k * 64
        raw_text(b, 300, y + 17, label, 14, INK, anchor="end")
        b.bar(320, y, 560, n / 100000, color="red" if "all" in label else "green", h=24)
        raw_text(b, 900, y + 18, f"{n:,}  ({n / 1000:.1f}%)", 14, INK, anchor="start", weight="700")
    b.card(20, 490, 540, 90, "Three layers touch almost everything", ["94.1% of the graph for just 64 targets"], "red", size=14, title_size=15)
    b.card(600, 490, 540, 90, "Fan-out 10, 5, 5 keeps it to a tenth", ["10.4%, at the price of a noisier neighbour average"], "green", size=14, title_size=15)
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
