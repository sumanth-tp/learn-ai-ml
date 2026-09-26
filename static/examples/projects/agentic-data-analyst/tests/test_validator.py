from __future__ import annotations

import pytest

from data_analyst.validator import SQLValidator
from data_analyst.warehouse.catalog import ALLOWED_TABLES


@pytest.fixture
def v() -> SQLValidator:
    return SQLValidator(ALLOWED_TABLES, default_limit=200, max_limit=1000)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE orders", "only SELECT"),
        ("DELETE FROM orders", "only SELECT"),
        ("INSERT INTO orders VALUES (1)", "only SELECT"),
        ("UPDATE orders SET status = 'x'", "only SELECT"),
        ("CREATE TABLE x AS SELECT * FROM orders", "only SELECT"),
        ("COPY orders TO '/tmp/o.csv'", "only SELECT"),
        ("ATTACH '/tmp/other.db'", "only SELECT"),
        ("PRAGMA database_list", "only SELECT"),
        ("SET enable_external_access = true", "only SELECT"),
        ("SELECT 1 FROM orders; DROP TABLE orders", "exactly one statement"),
        ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
        ("SELECT * FROM query('SELECT * FROM raw.customers')", "query"),
        ("SELECT read_text('/etc/hosts') FROM orders", "read_text"),
        ("SELECT * FROM raw.customers", "schema-qualified"),
        ("SELECT * FROM memory.main.orders", "schema-qualified"),
        ("SELECT * FROM information_schema.tables", "schema-qualified"),
        ("SELECT * FROM employees", "not allowed"),
        ("SELECT * FROM duckdb_settings()", "duckdb_settings"),
        ("SELECT 1", "at least one allowed table"),
        ("SELECT * FROM orders LIMIT (SELECT 5)", "integer literal"),
        ("SELEC oops", ""),
    ],
)
def test_blocks_unsafe_sql(v: SQLValidator, sql: str, reason: str) -> None:
    r = v.validate(sql)
    assert not r.ok
    assert reason in (r.error or "")


def test_comment_tricks_do_not_help(v: SQLValidator) -> None:
    r = v.validate("SELECT * FROM orders /* harmless */ ; /* */ DROP TABLE orders")
    assert not r.ok


def test_adds_limit_when_missing(v: SQLValidator) -> None:
    r = v.validate("SELECT status, COUNT(*) FROM orders GROUP BY status")
    assert r.ok and r.sql and r.sql.endswith("LIMIT 200")
    assert r.warnings == ["added LIMIT 200"]


def test_clamps_large_limit(v: SQLValidator) -> None:
    r = v.validate("SELECT * FROM orders LIMIT 50000")
    assert r.ok and r.sql and r.sql.endswith("LIMIT 1000")


def test_keeps_small_limit(v: SQLValidator) -> None:
    r = v.validate("SELECT * FROM orders LIMIT 5")
    assert r.ok and r.sql and r.sql.endswith("LIMIT 5") and not r.warnings


def test_cte_names_are_not_tables(v: SQLValidator) -> None:
    r = v.validate("WITH recent AS (SELECT * FROM orders) SELECT COUNT(*) FROM recent")
    assert r.ok and r.tables == ["orders"]


def test_cte_cannot_hide_forbidden_table(v: SQLValidator) -> None:
    r = v.validate("WITH x AS (SELECT * FROM employees) SELECT * FROM x")
    assert not r.ok and "employees" in (r.error or "")


def test_union_and_subqueries_are_checked(v: SQLValidator) -> None:
    assert v.validate("SELECT customer_id FROM orders UNION SELECT customer_id FROM customers").ok
    bad = v.validate(
        "SELECT * FROM orders WHERE customer_id IN (SELECT employee_id FROM employees)"
    )
    assert not bad.ok
