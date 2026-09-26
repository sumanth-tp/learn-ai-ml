"""Build a deterministic sample warehouse in DuckDB.

Raw data (with real-looking PII) lives in the ``raw`` schema. The agent only ever sees
views in ``main`` whose PII columns are masked by expressions generated from the
catalog, so a query cannot return a raw email however it is written.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from data_analyst.logging_setup import get_logger
from data_analyst.warehouse.catalog import TABLES, Column

log = get_logger(__name__)

FIRST = ["Asha", "Ben", "Chen", "Dara", "Elif", "Farid", "Gita", "Hugo", "Ines", "Jonas",
         "Kavya", "Liam", "Mei", "Nia", "Omar", "Priya", "Quinn", "Ravi", "Sofia", "Tariq"]
LAST = ["Ahmed", "Brown", "Costa", "Dubois", "Evans", "Fischer", "Garcia", "Hansen",
        "Iyer", "Jensen", "Kim", "Lopez", "Mehta", "Nakamura", "Okafor", "Patel"]
COUNTRIES = ["United States", "India", "Germany", "United Kingdom", "Brazil", "Japan",
             "France", "Canada", "Australia", "Nigeria"]
CATEGORIES = ["Electronics", "Home", "Sports", "Beauty", "Books", "Toys"]
START, END = date(2022, 1, 1), date(2024, 12, 31)


def _day(rng: random.Random, start: date = START, end: date = END) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate every raw table. Same seed, same data: the golden eval set relies on it."""
    rng = random.Random(seed)
    n_customers, n_products, n_suppliers, n_orders = 2000, 120, 25, 30000

    customers = []
    for cid in range(1, n_customers + 1):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        customers.append({
            "customer_id": cid,
            "full_name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{cid}@example.com",
            "phone": f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
            "country": rng.choices(COUNTRIES, weights=[30, 18, 10, 10, 8, 7, 6, 5, 4, 2])[0],
            "segment": rng.choices(["consumer", "smb", "enterprise"], weights=[70, 22, 8])[0],
            "signup_date": _day(rng, date(2021, 1, 1), date(2024, 6, 30)),
        })

    suppliers = [{
        "supplier_id": sid,
        "supplier_name": f"Supplier {sid:02d} Ltd",
        "country": rng.choice(COUNTRIES),
        "contact_email": f"sales{sid}@supplier{sid}.example.org",
    } for sid in range(1, n_suppliers + 1)]

    products = []
    for pid in range(1, n_products + 1):
        price = round(rng.uniform(5, 400), 2)
        products.append({
            "product_id": pid,
            "product_name": f"{rng.choice(CATEGORIES)} item {pid:03d}",
            "category": CATEGORIES[pid % len(CATEGORIES)],
            "supplier_id": rng.randint(1, n_suppliers),
            "list_price": price,
            "unit_cost": round(price * rng.uniform(0.4, 0.75), 2),
        })

    orders, items = [], []
    item_id = 1
    for oid in range(1, n_orders + 1):
        orders.append({
            "order_id": oid,
            "customer_id": rng.randint(1, n_customers),
            "order_date": _day(rng),
            "status": rng.choices(["completed", "cancelled", "refunded", "pending"],
                                  weights=[82, 8, 5, 5])[0],
            "channel": rng.choices(["web", "mobile", "store"], weights=[50, 35, 15])[0],
        })
        for _ in range(rng.randint(1, 4)):
            prod = products[rng.randint(0, n_products - 1)]
            items.append({
                "order_item_id": item_id,
                "order_id": oid,
                "product_id": prod["product_id"],
                "quantity": rng.randint(1, 5),
                "unit_price": prod["list_price"],
                "discount_pct": rng.choice([0, 0, 0, 5, 10, 15, 20, 30]),
            })
            item_id += 1

    inventory = []
    for year in (2022, 2023, 2024):
        for m in range(1, 13):
            month_end = date(year + m // 12, m % 12 + 1, 1) - timedelta(days=1)
            for pid in range(1, n_products + 1):
                inventory.append({"snapshot_date": month_end, "product_id": pid,
                                  "units_on_hand": rng.randint(0, 500)})

    campaigns = []
    for cid in range(1, 41):
        start = _day(rng)
        campaigns.append({
            "campaign_id": cid,
            "campaign_name": f"Campaign {cid:02d}",
            "channel": rng.choice(["email", "social", "search", "tv"]),
            "start_date": start,
            "end_date": start + timedelta(days=rng.randint(7, 60)),
            "budget": round(rng.uniform(2_000, 80_000), 2),
        })

    tickets = []
    for tid in range(1, 3001):
        opened = datetime.combine(_day(rng), datetime.min.time()) + timedelta(
            minutes=rng.randint(0, 1439))
        priority = rng.choices(["low", "medium", "high"], weights=[50, 35, 15])[0]
        hours = {"low": 72, "medium": 36, "high": 12}[priority]
        resolved = None if rng.random() < 0.05 else opened + timedelta(
            hours=rng.uniform(0.5, hours * 2))
        tickets.append({
            "ticket_id": tid,
            "customer_id": rng.randint(1, n_customers),
            "opened_at": opened,
            "resolved_at": resolved,
            "category": rng.choice(["billing", "delivery", "product", "account"]),
            "priority": priority,
        })

    sessions = []
    for sid in range(1, 20001):
        device = rng.choices(["desktop", "mobile", "tablet"], weights=[45, 45, 10])[0]
        rate = {"desktop": 0.06, "mobile": 0.035, "tablet": 0.045}[device]
        sessions.append({
            "session_id": sid,
            "customer_id": rng.randint(1, n_customers) if rng.random() < 0.6 else None,
            "session_date": _day(rng),
            "device": device,
            "pages_viewed": rng.randint(1, 25),
            "converted": rng.random() < rate,
        })

    employees = [{"employee_id": i, "full_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                  "salary": rng.randint(40_000, 180_000)} for i in range(1, 51)]

    return {
        "customers": pd.DataFrame(customers),
        "products": pd.DataFrame(products),
        "suppliers": pd.DataFrame(suppliers),
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(items),
        "inventory_snapshots": pd.DataFrame(inventory),
        "marketing_campaigns": pd.DataFrame(campaigns),
        "support_tickets": pd.DataFrame(tickets),
        "web_sessions": pd.DataFrame(sessions),
        "employees": pd.DataFrame(employees),
    }


def mask_expression(col: Column) -> str:
    """SQL that replaces a PII column with a masked value, keeping analytic use.

    Emails keep the domain (useful for "which email providers"), phones keep the last
    four digits, names keep the first initial. The raw value never leaves the view.
    """
    n = col.name
    match col.pii:
        case "email":
            return f"'***@' || split_part({n}, '@', 2) AS {n}"
        case "phone":
            return f"'***-***-' || right({n}, 4) AS {n}"
        case "name":
            return f"left({n}, 1) || '***' AS {n}"
        case _:
            return n


def build_warehouse(path: Path, seed: int = 42) -> Path:
    """(Re)create the warehouse file atomically: build to a temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    frames = generate(seed)
    con = duckdb.connect(str(tmp))
    try:
        con.execute("CREATE SCHEMA raw")
        for name, df in frames.items():
            con.register("frame", df)
            spec = next((t for t in TABLES if t.name == name), None)
            if spec is None:
                con.execute(f"CREATE TABLE raw.{name} AS SELECT * FROM frame")
            else:
                select = ", ".join(
                    f"CAST({c.name} AS {c.type}) AS {c.name}"
                    if c.type.startswith(("DECIMAL", "DATE", "TIMESTAMP")) else c.name
                    for c in spec.columns
                )
                con.execute(f"CREATE TABLE raw.{name} AS SELECT {select} FROM frame")
            con.unregister("frame")
        for t in TABLES:
            cols = ", ".join(mask_expression(c) for c in t.columns)
            con.execute(f"CREATE VIEW main.{t.name} AS SELECT {cols} FROM raw.{t.name}")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    tmp.replace(path)
    log.info("warehouse_built", path=str(path), tables=len(TABLES), seed=seed)
    return path


def ensure_warehouse(path: Path) -> Path:
    if not path.exists():
        build_warehouse(path)
    return path
