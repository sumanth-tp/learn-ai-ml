"""Boards for the three data-management project chapters in docs/mlops/data/98-projects.

Run from the repo root:

    python3 scripts/infographics/dm_projects.py            # all boards
    python3 scripts/infographics/dm_projects.py orders     # boards whose name contains "orders"

Every number drawn here was printed by a run of the project's own scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "dm-projects"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn

    return deco


@board("orders-architecture")
def orders_architecture():
    b = Board(1200, 760, "Orders and payments: one day, end to end", "Validate at the edge, keep every row, rebuild only what changed, publish by pointer")
    b.group(20, 90, 250, 410, "Sources", "orange")
    b.card(40, 130, 210, 100, "Orders system", ["CSV, daily changes", "every status change", "is a new row"], "orange", size=11)
    b.card(40, 250, 210, 100, "Payments provider", ["JSON lines, webhooks", "duplicates, late events"], "orange", size=11)
    b.card(40, 370, 210, 100, "Settlement file", ["the provider's own daily", "totals, used only to", "check our numbers"], "yellow", size=11)

    b.group(290, 90, 690, 410, "Warehouse (one DuckDB file)", "blue")
    ingest = b.card(305, 130, 150, 100, "ingest", ["manifest check", "contract check", "hash the email"], "blue", size=11)
    stage = b.card(480, 130, 150, 100, "stage", ["latest version per", "key, drop exact", "duplicates"], "blue", size=11)
    marts = b.card(655, 130, 150, 100, "marts", ["one verdict per", "payment: counted,", "orphan or late"], "blue", size=11)
    gate = b.card(830, 130, 135, 100, "gate", ["reconciled?", "closed days kept?"], "red", size=11)
    raw = b.cylinder(320, 290, 120, 110, "raw.*", ["every row kept", "reject_reason"], "purple", size=11)
    stg = b.cylinder(495, 290, 120, 110, "stg.*", ["one row per", "order, payment"], "purple", size=11)
    mart = b.cylinder(670, 290, 120, 110, "mart.*", ["fct_payments", "daily work table"], "purple", size=11)
    pub = b.cylinder(875, 290, 90, 110, "versions", ["v1 to v29", "view=pointer"], "green", size=10)
    recon = b.card(305, 425, 500, 55, "reconcile", ["counts balance, money matches the settlement file"], "teal", size=11)
    b.arrow(ingest.right(), stage.left())
    b.arrow(stage.right(), marts.left())
    b.arrow(marts.right(), gate.left())
    b.arrow(ingest.bottom(), raw.top(), label="append")
    b.arrow(stage.bottom(), stg.top(), label="by key")
    b.arrow(marts.bottom(), mart.top(), label="by day")
    b.arrow(gate.bottom(0.65), pub.top(), label="pass")
    b.arrow(raw.right(), stg.left(), dashed=True)
    b.arrow(stg.right(), mart.left(), dashed=True)
    b.arrow((250, 180), ingest.left(0.3))
    b.arrow((250, 300), ingest.left(0.8))
    b.arrow((250, 420), recon.left(), dashed=True)
    b.arrow(mart.bottom(), (730, 425), dashed=True)
    b.arrow(recon.right(), gate.bottom(0.1), via=[(843, 452)], dashed=True)

    b.group(1000, 90, 180, 410, "Consumers", "green")
    b.card(1015, 140, 150, 100, "Finance report", ["reads the view", "never a work table"], "green", size=11)
    b.card(1015, 270, 150, 100, "Analytics", ["same view, same", "version"], "green", size=11)
    b.arrow(pub.right(), (1015, 320), label="read")
    b.arrow((985, 320), (1015, 190), via=[(985, 190)])

    b.group(20, 530, 1160, 200, "Operations around the data flow", "grey")
    b.card(40, 570, 270, 140, "run metadata", ["ops.runs: task, status, attempt,", "metrics, code version", "ops.lineage: output <- inputs"], "grey", size=11)
    b.card(330, 570, 270, 140, "reconciliation", ["counts balance at every step", "money agrees with the provider", "to the minor unit"], "teal", size=11)
    b.card(620, 570, 270, 140, "alerts with owners", ["9 rules in alerts.yaml", "owner comes from the contract", "or is named in the rule"], "pink", size=11)
    b.card(910, 570, 250, 140, "runbooks", ["one per alert, in the repo", "rollback = move the pointer"], "yellow", size=11)
    return b


@board("orders-journey")
def orders_journey():
    b = Board(1200, 700, "One payment, four fates", "Event day D. The day stays open for 3 days, closes, and is reconciled on D+7")
    x0, step = 300, 105
    days = ["D", "D+1", "D+2", "D+3", "D+4", "D+5", "D+6", "D+7"]
    b.group(x0 - 40, 100, 4 * step, 440, "", "green")
    b.group(x0 - 40 + 4 * step, 100, 4 * step, 440, "", "red")
    b.text(x0 - 40 + 2 * step, 126, "open: the day may be restated", 13, "green", "700")
    b.text(x0 - 40 + 6 * step, 126, "closed: a change needs an approver", 13, "red", "700")
    for i, d in enumerate(days):
        x = x0 + i * step
        b.text(x, 170, d, 14, "grey", "700")
        b.arrow((x, 182), (x, 520), color="grey", dashed=True, width=1)
    b.text(x0 + 3 * step, 196, "day closes", 11, "orange", "700")
    b.text(x0 + 7 * step, 196, "reconciled", 11, "teal", "700")
    lanes = [
        (250, 0, "green", "On time", "57,756 payments", "counted at once"),
        (330, 2, "yellow", "1 to 3 days late", "2,147 payments", "counted; the published day is restated"),
        (410, 5, "red", "4 to 6 days late", "475 payments", "held back: 2,695,995 minor units"),
        (490, None, "purple", "Order never arrives", "20 payments", "orphan: 144,015 minor units"),
    ]
    for y, arrive, colour, head, count, result in lanes:
        b.text(30, y - 6, head, 13, colour, "700", anchor="start")
        b.text(30, y + 14, count, 12, "grey", "400", anchor="start")
        b.pill(x0 - 28, y - 12, "event", colour, 11, solid=True)
        if arrive is None:
            b.arrow((x0 + 38, y), (x0 + 7 * step - 10, y), color=colour, dashed=True)
            b.pill(x0 + 3 * step - 20, y - 36, "no order to match", colour, 11)
            b.text(x0 + 50, y + 30, result, 12, "grey", "400", anchor="start")
        elif arrive == 0:
            b.text(x0 + 50, y + 4, result, 12, "grey", "400", anchor="start")
        else:
            xe = x0 + arrive * step
            b.arrow((x0 + 38, y), (xe - 36, y), color=colour)
            b.pill(xe - 34, y - 12, "arrives", colour, 11)
            b.text(x0 + 50, y + 30, result, 12, "grey", "400", anchor="start")
    b.card(30, 560, 560, 120, "Why this policy", ["Finance needs a number that stops moving. Three days", "covers 82% of the late payments here (2,147 of 2,622).", "The rest wait for a person: a number that changes", "silently cannot be audited."], "yellow", size=12, align="left")
    b.card(620, 560, 550, 120, "What the pipeline guarantees", ["Every payment lands in exactly one bucket: counted, held", "back or orphan. The three buckets plus the quarantined", "rows add up to the provider's total, to the minor unit."], "teal", size=12, align="left")
    return b


@board("orders-gate")
def orders_gate():
    b = Board(1200, 720, "The daily run as a graph, and the gate before publication", "Seven tasks, one gate, one pointer. Any task can be re-run for the same date")
    names = [("ingest_orders", "blue", 30, 110), ("ingest_payments", "blue", 30, 220), ("stage", "blue", 290, 165), ("marts", "blue", 480, 165), ("reconcile", "teal", 670, 165), ("publish", "red", 860, 165), ("alerts", "pink", 1030, 165)]
    boxes = {}
    for name, colour, x, y in names:
        boxes[name] = b.card(x, y, 150 if name not in ("alerts", "publish") else 130, 60, name, [], colour, size=12)
    b.arrow(boxes["ingest_orders"].right(), boxes["stage"].left(0.3))
    b.arrow(boxes["ingest_payments"].right(), boxes["stage"].left(0.7))
    b.arrow(boxes["stage"].right(), boxes["marts"].left())
    b.arrow(boxes["marts"].right(), boxes["reconcile"].left())
    b.arrow(boxes["reconcile"].right(), boxes["publish"].left())
    b.arrow(boxes["publish"].right(), boxes["alerts"].left())
    b.text(105, 100, "arrivals", 12, "grey", "700")
    b.text(1095, 252, "runs even if an", 11, "pink", "700")
    b.text(1095, 268, "upstream task failed", 11, "pink", "700")
    b.group(20, 300, 560, 400, "The gate: publish only if none of these blocks", "red")
    rules = [
        ("1  a rejected batch is waiting", "orders:2026-06-09 blocked it for 2 days"),
        ("2  a count does not balance", "7 checks, e.g. staged = fact rows"),
        ("3  closed days differ from the provider", "unexplained money must be exactly zero"),
        ("4  a batch has more than 2% rejected rows", "limit set in the runner"),
        ("5  a closed day would change unapproved", "a restatement needs a named approver"),
    ]
    for i, (head, tail) in enumerate(rules):
        b.card(40, 338 + i * 70, 520, 60, head, [tail], "red", size=11, align="left")
    b.group(610, 300, 570, 400, "What a pass and a block do", "green")
    pub = b.cylinder(640, 345, 140, 120, "v29", ["table + pointer"], "green", size=12)
    b.card(820, 345, 340, 120, "pass", ["copy the work table to version N+1", "move the view to N+1", "record fingerprint and date"], "green", size=11, align="left")
    b.card(640, 495, 520, 90, "block", ["consumers keep reading the last good version", "the alert names the blocker and the owner"], "red", size=11, align="left")
    b.card(640, 598, 520, 84, "rollback", ["move the pointer back, freeze publishing, fix,", "backfill, check, unfreeze"], "yellow", size=11, align="left")
    return b


@board("orders-alerts")
def orders_alerts():
    b = Board(1200, 700, "Alert rules, thresholds and owners", "The owner is written into the contract or the rule, so nobody has to guess whom to wake")
    rows = [
        ["Rule", "Fires when", "Severity", "Owner", "First step"],
        ["batch_rejected", "a batch is rejected and not yet redelivered", "page", "orders-team or payments-team", "ask for a corrected file"],
        ["unexplained_money", "closed days differ from settlement", "page", "finance-data", "list payments that differ from their order"],
        ["publish_blocked_day", "publication blocked today", "ticket", "data-platform", "read the blocker in ops.runs"],
        ["publish_blocked_two_days", "publication blocked for 2 or more days", "page", "data-platform", "read the blocker in ops.runs"],
        ["task_failed", "a task failed 3 times", "page", "data-platform", "fix, then re-run the same date"],
        ["file_missing", "no file for a dataset on its day", "page", "dataset owner", "check the source job"],
        ["quarantine_rate_high", "more than 2% of a batch's rows rejected", "ticket", "dataset owner", "group rejects by reason"],
        ["payment_rows_quarantined", "any payment row rejected today", "ticket", "payments-team", "ask the provider to correct and resend"],
        ["late_money_waiting", "more than 100,000 minor units late beyond the window in a day", "ticket", "finance-data", "decide: restate or leave"],
    ]
    b.table(20, 100, [190, 330, 90, 230, 300], rows, header_color="pink", size=11, row_h=42)
    b.card(20, 540, 560, 140, "Suppression is part of the design", ["While a batch is waiting, unexplained_money is silent:", "the missing rows are the cause, and one page is enough.", "The blocked publication escalates from ticket to page", "on the second day."], "yellow", size=12, align="left")
    b.card(610, 540, 570, 140, "Measured on the 36 days", ["29 alerts: 6 pages and 23 tickets. Every page falls", "inside the two incident windows (June 9 to 10 and", "June 12 to 13). No page fired on a healthy day."], "teal", size=12, align="left")
    return b


@board("orders-results")
def orders_results():
    b = Board(1200, 760, "What the controls caught, and where the money went", "Every injected defect was caught by one named control, with no good row rejected")
    rows = [
        ["Defect", "Injected", "Caught", "Control"],
        ["customer_id missing", "40", "40", "contract: not_nullable"],
        ["negative total", "25", "25", "contract: min"],
        ["currency not allowed", "30", "30", "contract: allowed"],
        ["order time in the future", "15", "15", "contract: not_after_arrival"],
        ["duplicate order rows", "60", "60", "staging: dedupe on key"],
        ["duplicate payment webhooks", "80", "80", "staging: dedupe on payment_id"],
        ["payment currency not allowed", "20", "20", "contract: allowed"],
        ["orders that never arrive", "20", "20", "mart: orphan verdict"],
        ["payments 4 to 6 days late", "475", "475", "mart: late verdict"],
        ["renamed column (1 file)", "1", "1", "batch: rejected_schema"],
        ["truncated file (1 file)", "1", "1", "batch: rejected_rowcount"],
    ]
    b.table(20, 100, [260, 80, 80, 270], rows, header_color="green", size=12, row_h=36)
    b.card(20, 560, 690, 170, "False alarms", ["0 good rows rejected, 0 false duplicates, 0 false orphans.", "The count balances: 178,733 valid order rows = 178,673 versions", "+ 60 duplicates, and 60,458 valid payment rows = 60,378 staged", "+ 80 duplicates."], "teal", size=12, align="left")
    settled, in_warehouse, late, quarantined, orphan = 306761252, 304010944, 2534214, 72079, 144015
    held = late + quarantined + orphan
    b.text(910, 124, "Provider settled, 29 closed days", 13, "blue", "700")
    b.text(910, 146, f"{settled:,} minor units", 15, "blue", "700")
    b.bar(740, 170, 340, in_warehouse / settled, color="green", h=26)
    b.text(740, 222, f"in the warehouse {in_warehouse:,} ({100 * in_warehouse / settled:.2f}%)", 12, "green", "700", anchor="start")
    b.text(910, 290, f"held back {held:,} ({100 * held / settled:.2f}%)", 13, "red", "700")
    x = 740
    for label, value, colour in [("late", late, "red"), ("orphan", orphan, "purple"), ("quarantined", quarantined, "yellow")]:
        w = 340 * value / held
        b.card(x, 310, max(w, 6), 40, "", [], colour, size=10)
        x += w
    b.text(740, 380, f"late beyond the window  {late:,}  ({100 * late / held:.1f}% of held back)", 12, "red", "700", anchor="start")
    b.text(740, 406, f"orders never arrived    {orphan:,}  ({100 * orphan / held:.1f}%)", 12, "purple", "700", anchor="start")
    b.text(740, 432, f"rows quarantined        {quarantined:,}  ({100 * quarantined / held:.1f}%)", 12, "yellow", "700", anchor="start")
    b.text(910, 500, "unexplained: 0", 18, "green", "700")
    b.text(910, 526, "warehouse + held back = settlement total", 11, "grey")
    return b


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [name for name in BOARDS if not argv or any(a in NAMES[name] for a in argv)]
    for name in wanted:
        path = OUT / f"{NAMES[name]}.svg"
        BOARDS[name]().save(path)
        print("wrote", path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
