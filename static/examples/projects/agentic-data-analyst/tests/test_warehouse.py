from __future__ import annotations

import duckdb

from data_analyst.warehouse.catalog import (
    ALLOWED_TABLES,
    close_join_paths,
    join_path,
    render_schema,
    schema_version,
)
from data_analyst.warehouse.seed import generate


def test_seed_is_deterministic() -> None:
    a, b = generate(42), generate(42)
    assert a["orders"].equals(b["orders"])
    assert a["order_items"].equals(b["order_items"])


def test_views_mask_pii_and_raw_keeps_it(warehouse_file) -> None:
    con = duckdb.connect(str(warehouse_file), read_only=True)
    try:
        masked = con.execute("SELECT full_name, email, phone FROM customers LIMIT 1").fetchone()
        raw = con.execute("SELECT full_name, email, phone FROM raw.customers LIMIT 1").fetchone()
    finally:
        con.close()
    assert masked[0].endswith("***") and len(masked[0]) == 4
    assert masked[1].startswith("***@")
    assert masked[2].startswith("***-***-")
    assert "@" in raw[1] and not raw[1].startswith("***")


def test_employees_exist_only_in_raw(warehouse_file) -> None:
    assert "employees" not in ALLOWED_TABLES
    con = duckdb.connect(str(warehouse_file), read_only=True)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        con.close()
    assert tables == set(ALLOWED_TABLES)


def test_join_path_finds_bridge() -> None:
    assert join_path("products", "customers") == ["products", "order_items", "orders", "customers"]
    assert close_join_paths(["products", "orders"]) == ["products", "orders", "order_items"]


def test_render_schema_flags_pii_and_joins() -> None:
    text = render_schema(["customers", "orders"])
    assert "email VARCHAR" in text and "[PII, masked]" in text
    assert "orders.customer_id = customers.customer_id" in text


def test_schema_version_is_stable() -> None:
    assert schema_version() == schema_version()
    assert len(schema_version()) == 12
