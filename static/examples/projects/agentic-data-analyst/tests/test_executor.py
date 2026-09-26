from __future__ import annotations

import pytest

from data_analyst.executor import QueryError, QueryTimeout, WarehouseExecutor, redact


def test_runs_read_only_query(executor: WarehouseExecutor) -> None:
    res = executor.run("SELECT COUNT(*) AS n FROM customers")
    assert res.columns == ["n"] and res.rows == [[2000]] and not res.truncated


def test_row_cap_truncates(settings) -> None:
    ex = WarehouseExecutor(settings.warehouse_path, timeout_s=5, row_cap=10)
    res = ex.run("SELECT order_id FROM orders")
    assert res.row_count == 10 and res.truncated


def test_timeout_interrupts_query(settings) -> None:
    ex = WarehouseExecutor(settings.warehouse_path, timeout_s=0.5, row_cap=10)
    with pytest.raises(QueryTimeout):
        ex.run("SELECT COUNT(*) FROM order_items a, order_items b, orders c")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM raw.orders",  # even if the validator were bypassed
        "CREATE TABLE main.x AS SELECT 1",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SET enable_external_access = true",
        "COPY (SELECT 1) TO 'leak.csv'",
    ],
)
def test_database_layer_blocks_unsafe_sql(executor: WarehouseExecutor, sql: str) -> None:
    with pytest.raises(QueryError):
        executor.run(sql)


def test_estimate_sees_through_limit(executor: WarehouseExecutor) -> None:
    est = executor.estimate("SELECT * FROM products CROSS JOIN customers LIMIT 200")
    assert est.estimated_rows == 240_000


def test_estimate_reports_binder_errors(executor: WarehouseExecutor) -> None:
    with pytest.raises(QueryError, match="price"):
        executor.estimate("SELECT price FROM order_items")


def test_values_are_json_safe(executor: WarehouseExecutor) -> None:
    res = executor.run("SELECT signup_date, CAST(1.5 AS DECIMAL(4,2)) AS d FROM customers LIMIT 1")
    assert isinstance(res.rows[0][0], str) and res.rows[0][1] == 1.5


def test_dlp_redacts_leaked_contact_details() -> None:
    assert redact("mail bob@corp.com now") == ("mail ***@corp.com now", 1)
    assert redact("+1-555-123-4567")[0] == "***-***-4567"
    assert redact("2024-01-31") == ("2024-01-31", 0)
    assert redact("***@example.com") == ("***@example.com", 0)
