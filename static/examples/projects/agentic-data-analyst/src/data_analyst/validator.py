"""Static SQL validation with sqlglot, before anything touches the database.

The model's SQL is untrusted input. We parse it into an AST and enforce policy on the
tree, not on the text: regexes are defeated by comments, casing and quoting, the AST
is not.
"""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from pydantic import BaseModel
from sqlglot import exp
from sqlglot.errors import ParseError

DIALECT = "duckdb"

# Statement or clause types that write, change configuration or reach outside the DB.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
    exp.Set,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Into,
    exp.Grant,
    exp.Export,
    exp.LoadData,
)

# Functions that read files, run nested SQL or expose settings and internals.
FORBIDDEN_FUNCTION_PREFIXES: tuple[str, ...] = (
    "read_",
    "glob",
    "query",
    "sniff_",
    "duckdb_",
    "pragma_",
    "current_setting",
    "getvariable",
    "setvariable",
    "install",
    "load",
    "parquet_",
    "iceberg_",
    "delta_",
    "sqlite_",
    "postgres_",
    "mysql_",
    "http",
    "system",
)


class ValidationResult(BaseModel):
    ok: bool
    sql: str | None = None
    tables: list[str] = []
    error: str | None = None
    warnings: list[str] = []


class SQLValidator:
    def __init__(self, allowed_tables: Iterable[str], default_limit: int, max_limit: int) -> None:
        self.allowed = frozenset(t.lower() for t in allowed_tables)
        self.default_limit = default_limit
        self.max_limit = max_limit

    def validate(self, sql: str) -> ValidationResult:
        try:
            statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
        except ParseError as e:
            return _fail(f"SQL does not parse: {_first_line(str(e))}")
        if len(statements) != 1:
            return _fail(f"exactly one statement is allowed, got {len(statements)}")
        stmt = statements[0]
        if not isinstance(stmt, exp.Query):
            return _fail(f"only SELECT queries are allowed, got {stmt.key.upper()}")

        for node in stmt.walk():
            if isinstance(node, FORBIDDEN_NODES):
                return _fail(f"forbidden operation: {node.key.upper()}")
            if isinstance(node, exp.Func):
                name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
                if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                    return _fail(f"function {name}() is not allowed")

        cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
        tables: set[str] = set()
        for table in stmt.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                return _fail("table functions are not allowed in FROM")
            name = table.name.lower()
            if table.catalog or (table.db and table.db.lower() != "main"):
                return _fail(f"schema-qualified table {table.sql(DIALECT)} is not allowed")
            if name in cte_names and not table.db:
                continue
            if name not in self.allowed:
                allowed = ", ".join(sorted(self.allowed))
                return _fail(f"table {name!r} is not allowed; allowed tables: {allowed}")
            tables.add(name)
        if not tables:
            return _fail("the query must read from at least one allowed table")

        stmt, warnings = self._enforce_limit(stmt)
        if isinstance(stmt, ValidationResult):
            return stmt
        return ValidationResult(
            ok=True, sql=stmt.sql(dialect=DIALECT), tables=sorted(tables), warnings=warnings
        )

    def _enforce_limit(
        self, stmt: exp.Query
    ) -> tuple[exp.Query, list[str]] | tuple[ValidationResult, list[str]]:
        limit = stmt.args.get("limit")
        if limit is None:
            return stmt.limit(self.default_limit), [f"added LIMIT {self.default_limit}"]
        value = limit.expression
        if not (isinstance(value, exp.Literal) and value.is_int):
            return _fail("LIMIT must be an integer literal"), []
        if int(value.this) > self.max_limit:
            stmt = stmt.copy()
            stmt.set("limit", exp.Limit(expression=exp.Literal.number(self.max_limit)))
            return stmt, [f"LIMIT clamped to {self.max_limit}"]
        return stmt, []


def _fail(message: str) -> ValidationResult:
    return ValidationResult(ok=False, error=message)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else text
