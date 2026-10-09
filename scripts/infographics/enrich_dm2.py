import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "dm-enrich"
BOARDS = {}


def board(name):
    def deco(fn):
        BOARDS[name] = fn
        return fn
    return deco


def label(b, x, y, text, size=13, fill=INK, anchor="start", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def hbar(b, x, y, width, value, maximum, color, text, name="", name_w=0, h=20):
    c = PALETTE[color]
    if name:
        label(b, x, y + h - 5, name, 13, INK)
    bx = x + name_w
    b.parts.append(f'<rect x="{bx}" y="{y}" width="{width}" height="{h}" rx="5" fill="#e9ecef"/>')
    b.parts.append(
        f'<rect x="{bx}" y="{y}" width="{max(1.5, width * value / maximum):.1f}" height="{h}" rx="5" fill="{c["stroke"]}"/>'
    )
    label(b, bx + width + 10, y + h - 5, text, 13, c["text"], weight="700")


@board("dm2-scd-history")
def scd_history():
    b = Board(1120, 520, "Three joins to a customer history, measured", "20,300 synthetic orders, 2,000 customers, 3,145 Type 2 rows")
    label(b, 40, 112, "Share of orders in the wrong region (latest-row join)", 14, INK, weight="700")
    hbar(b, 40, 124, 520, 23.68, 100, "red", "23.68%", "", 0, h=24)
    label(b, 40, 190, "Error in each region's revenue total (latest-row join)", 14, INK, weight="700")
    regions = [("North", -1.99), ("East", -0.90), ("South", 0.60), ("West", 2.37)]
    zero = 380
    scale = 90
    b.parts.append(f'<line x1="{zero}" y1="205" x2="{zero}" y2="335" stroke="{INK}" stroke-width="1.5"/>')
    for i, (name, v) in enumerate(regions):
        y = 212 + i * 30
        label(b, 40, y + 15, name, 13, INK)
        c = PALETTE["orange"]["stroke"]
        x0 = zero if v >= 0 else zero + v * scale
        b.parts.append(f'<rect x="{x0:.1f}" y="{y}" width="{abs(v) * scale:.1f}" height="20" rx="4" fill="{c}"/>')
        label(b, zero + (v * scale) + (10 if v >= 0 else -10), y + 15, f"{v:+.2f}%", 13, PALETTE["orange"]["text"], "start" if v >= 0 else "end", "700")
    label(b, 40, 372, "Join row counts against 20,300 orders", 14, INK, weight="700")
    b.card(40, 388, 250, 84, "Half-open", ["20,300 rows", "one match per order"], "green", size=13)
    b.card(310, 388, 250, 84, "Closed BETWEEN", ["20,638 rows", "338 orders double counted"], "red", size=13)
    b.card(660, 130, 410, 120, "Why the totals hide it", ["Customers leave one region", "and arrive in another.", "Row errors cancel in the sums."], "yellow", size=13)
    b.card(660, 270, 410, 100, "Closed join revenue", ["1,245,149.75 against 1,227,800.55", "inflation 1.41%"], "red", size=13)
    b.card(660, 390, 410, 84, "Weekly snapshot", ["sees 3,133 of 3,145 versions", "misses 12"], "teal", size=13)
    return b


@board("dm2-pit-auc")
def pit_auc():
    b = Board(1120, 500, "A leaky join looks great offline and falls apart served", "4,000 synthetic users, logistic regression, 1,000 held out")
    rows = [("latest value", 0.991, 0.678, "red"), ("event-time as-of", 0.995, 0.678, "orange"), ("availability as-of", 0.678, 0.678, "green")]
    label(b, 40, 118, "offline AUC (same join)", 13, FAINT, weight="700")
    label(b, 40, 282, "served AUC (what the live service sees)", 13, FAINT, weight="700")
    for i, (name, off, srv, colour) in enumerate(rows):
        y = 134 + i * 44
        hbar(b, 40, y, 440, off, 1.0, colour, f"{off:.3f}", name, 190, h=26)
        y2 = 298 + i * 44
        hbar(b, 40, y2, 440, srv, 1.0, "teal", f"{srv:.3f}", name, 190, h=26)
    b.card(800, 130, 280, 130, "The surprise", ["the event-time join,", "the textbook rule,", "leaked as much as", "the latest value"], "yellow", size=13)
    b.card(800, 290, 280, 130, "Why", ["decision-day value", "published two days", "late, so a cutoff on", "event time let it in"], "red", size=13)
    b.card(40, 450, 1040, 40, "Scaler refit on a shifted live batch: AUC 0.678 both ways, mean prediction 0.400 against 0.286", [], "teal", size=13)
    return b


@board("dm2-retry-recovery")
def retry_recovery():
    b = Board(1120, 500, "Retries rescue runs; keyed writes keep the backfill honest", "4,000 simulated runs per policy; 30-day backfill into two DuckDB tables")
    label(b, 40, 112, "Share of runs that finish", 14, INK, weight="700")
    rows = [("no retry", 0.546, "red", "34.0 min work"), ("whole DAG, 3 x 1 min", 0.963, "orange", "58.8 min work"), ("per task, 3 x 1 min", 0.968, "yellow", "47.9 min work"), ("per task, 1,2,4", 0.988, "teal", "48.5 min work"), ("per task, 1,2,4,8", 0.998, "green", "48.9 min work")]
    for i, (name, v, colour, note) in enumerate(rows):
        y = 130 + i * 50
        hbar(b, 40, y, 330, v, 1.0, colour, f"{v:.3f}", name, 190, h=24)
        label(b, 40, y + 40, note, 11, FAINT)
    label(b, 700, 112, "30-day backfill, rows in the table", 14, INK, weight="700")
    hbar(b, 700, 130, 300, 38951, 38951, "red", "38,951", "", 0, h=26)
    label(b, 700, 176, "append (9 duplicate writes, 8 days doubled)", 12, INK)
    hbar(b, 700, 196, 300 * 30054 / 38951, 30054, 30054, "green", "30,054", "", 0, h=26)
    label(b, 700, 242, "replace by day, equals the expected total", 12, INK)
    b.card(700, 280, 380, 90, "Finish time of a successful run", ["resume from failed task: 49.1 min", "restart the whole chain: 57.7 min"], "yellow", size=13)
    b.card(700, 390, 380, 90, "Where backoff helps", ["total wait 3 min: 0.968", "total wait 7 min: 0.988, 15 min: 0.998"], "teal", size=13)
    return b


@board("dm2-lineage-impact")
def lineage_impact():
    b = Board(1120, 480, "A lineage graph over-reports, and silent jobs hide the rest", "synthetic estate: 119 nodes, 172 edges, 6,060 run events, 60 models")
    label(b, 40, 112, "Models flagged when a source is bad from day 60 (mean over 10 sources)", 14, INK, weight="700")
    hbar(b, 40, 130, 420, 21.0, 21.0, "red", "21.0", "plain graph", 150, h=26)
    hbar(b, 40, 170, 420, 9.8, 21.0, "green", "9.8", "time-aware", 150, h=26)
    label(b, 40, 232, "factor 2.14; worst source: 39 flagged against 20 affected", 12, FAINT)
    label(b, 40, 290, "Recall of affected models as jobs stop emitting lineage", 14, INK, weight="700")
    for i, (share, rec, colour) in enumerate([("0%", 1.000, "green"), ("5%", 0.869, "teal"), ("10%", 0.689, "yellow"), ("20%", 0.481, "red")]):
        hbar(b, 40, 308 + i * 40, 420, rec, 1.0, colour, f"{rec:.3f}", f"{share} jobs silent", 150, h=26)
    b.card(680, 130, 400, 130, "Why the graph over-reports", ["It says could reach.", "Models trained before the", "change read good data, so", "they are not affected."], "yellow", size=13)
    b.card(680, 290, 400, 150, "Why recall falls so fast", ["A path crosses four or five jobs.", "One silent job cuts the path.", "20% silent jobs lose more than", "half of the affected models."], "red", size=13)
    return b


@board("dm2-skew-salting")
def skew_salting():
    b = Board(1120, 520, "More partitions do not split a hot key; salting does", "2,000,000 synthetic events, merchant 0 holds 50%, rows per hash bucket against the ideal")
    label(b, 40, 112, "Largest partition divided by the ideal share (1.0 is perfectly even)", 14, INK, weight="700")
    groups = [("hash by merchant", [(16, 9.94, "red"), (80, 40.14, "red"), (400, 200.09, "red")]), ("merchant + salt 8", [(16, 2.34, "orange"), (80, 5.48, "orange"), (400, 25.97, "orange")]), ("merchant + salt 64", [(16, 1.51, "green"), (80, 2.64, "green"), (400, 6.75, "green")])]
    y = 135
    for name, bars in groups:
        label(b, 40, y + 14, name, 13, INK, weight="700")
        for j, (parts, value, colour) in enumerate(bars):
            hbar(b, 40, y + 22 + j * 26, 520, value, 200.09, colour, f"{value:.2f}", f"{parts} partitions", 150, h=18)
        y += 112
    b.card(830, 135, 250, 130, "Hot key bound", ["1,000,000 rows in one", "partition against an ideal", "of 25,000 at 80 partitions", "gives at least 40"], "yellow", size=12)
    b.card(830, 285, 250, 100, "Map-side combine", ["126,159 rows shuffled,", "6.3% of 2,000,000"], "teal", size=12)
    b.card(830, 405, 250, 100, "Distinct count trap", ["sum of partials 743,776", "true count 198,666"], "red", size=12)
    return b


@board("dm2-chunking-recall")
def chunking_recall():
    b = Board(1120, 540, "Chunk counting and document counting disagree", "533 SciFact abstracts, 300 queries, all-MiniLM-L6-v2, recall at 5")
    label(b, 40, 108, "dense recall at 5", 14, INK, weight="700")
    label(b, 400, 108, "teal: top 5 chunks     green: top 5 distinct documents", 12, FAINT)
    rows = [("whole abstract", 0.884, 0.884, "118,466 words"), ("40 words", 0.859, 0.896, "111,534 words"), ("100 words", 0.861, 0.891, "111,534 words"), ("100, overlap 50", 0.848, 0.908, "182,184 words"), ("100 + title", 0.879, 0.903, "130,211 words")]
    for i, (name, chunk, doc, words) in enumerate(rows):
        y = 130 + i * 74
        label(b, 40, y + 18, name, 14, INK, weight="700")
        label(b, 40, y + 40, words, 11, FAINT)
        hbar(b, 200, y, 420, chunk, 1.0, "teal", f"{chunk:.3f}", "", 0, h=24)
        hbar(b, 200, y + 30, 420, doc, 1.0, "green", f"{doc:.3f}", "", 0, h=24)
    b.card(800, 140, 280, 130, "Overlap 50", ["worst counted in chunks,", "best counted in documents:", "neighbouring chunks fill", "the five slots"], "yellow", size=12)
    b.card(800, 290, 280, 110, "Input limit", ["72.4% of abstracts exceed", "the 256-token limit,", "so whole-abstract vectors", "are cut off"], "red", size=12)
    b.card(800, 420, 280, 90, "Title on each chunk", ["+16.7% words,", "0.861 to 0.879"], "teal", size=12)
    return b


@board("dm2-k-anonymity-cost")
def k_anonymity_cost():
    b = Board(1120, 520, "Unique ids are not anonymity; generalising is cheap, k is not enough", "UCI Adult extract, 45,222 complete rows, hashed pseudonymous ids")
    label(b, 40, 108, "Share of rows that are unique as quasi-identifiers are added", 14, INK, weight="700")
    steps = [("age", 0.000), ("+ sex", 0.000), ("+ race", 0.001), ("+ marital status", 0.012), ("+ country", 0.056), ("+ education", 0.138), ("+ occupation", 0.303)]
    for i, (name, v) in enumerate(steps):
        colour = "green" if v < 0.02 else ("yellow" if v < 0.1 else ("orange" if v < 0.2 else "red"))
        hbar(b, 40, 128 + i * 32, 360, v, 0.35, colour, f"{v * 100:.1f}%", name, 150, h=22)
    label(b, 40, 372, "After generalising and suppressing groups under 5:", 14, INK, weight="700")
    hbar(b, 40, 390, 360, 0.0, 0.35, "green", "0.0% unique", "", 0, h=22)
    label(b, 40, 440, "2.48% of rows suppressed (1,121); AUC 0.9276 to 0.9245", 12, FAINT)
    b.card(660, 130, 420, 100, "Attacker guess success", ["mean 0.436 before,", "0.0123 after generalising"], "teal", size=13)
    b.card(660, 250, 420, 120, "What k = 5 did not fix", ["7.66% of kept rows sit in a group", "where everyone shares one", "income class: the class is disclosed"], "red", size=13)
    b.card(660, 390, 420, 100, "Cost of the fix", ["AUC fell 0.0031 on the same rows;", "the signal is outside the identifiers"], "yellow", size=13)
    return b


@board("dm2-monitor-tradeoff")
def monitor_tradeoff():
    b = Board(1120, 560, "Every monitor trades missed incidents against false alarms", "120 simulated days of hourly batches, 36 incidents; caught per severity out of 4")
    label(b, 40, 108, "monitor", 13, FAINT, weight="700")
    label(b, 340, 108, "caught at 0.25 / 0.5 / 1.0", 13, FAINT, weight="700")
    label(b, 640, 108, "false alarms per 1,000 healthy batches (square-root scale)", 13, FAINT, weight="700")
    rows = [("delay over 60 min", (0, 2, 4), 0.5), ("delay over 20 min", (4, 4, 4), 29.1), ("volume 20% off trailing mean", (3, 4, 4), 712.2), ("volume 15% off same hour", (3, 4, 4), 2.4), ("volume 15%, twice in a row", (1, 4, 4), 0.0), ("KS p below 0.05", (4, 4, 4), 50.4), ("KS p below 1e-6", (0, 2, 4), 0.0), ("KS statistic over 0.10", (0, 3, 4), 1.0), ("null rate over 6%", (0, 4, 4), 0.0)]
    for i, (name, caught, false) in enumerate(rows):
        y = 128 + i * 46
        label(b, 40, y + 22, name, 13, INK, weight="700")
        for j, c in enumerate(caught):
            colour = "green" if c == 4 else ("yellow" if c >= 2 else "red")
            b.card(340 + j * 90, y, 80, 34, f"{c}/4", [], colour, size=13)
        colour = "red" if false > 100 else ("orange" if false > 20 else "green")
        hbar(b, 640, y + 6, 330, false ** 0.5, 712.2 ** 0.5, colour, f"{false:g}", "", 0, h=22)
    return b


def main(names):
    for name in names or list(BOARDS):
        print(BOARDS[name]().save(OUT / f"{name}.svg").relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
