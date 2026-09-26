"""The semantic layer: what each exposed table and column means, which columns are PII,
and how tables join. This is the single source of truth for schema retrieval, the
table allow-list and column masking."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Literal

from pydantic import BaseModel

PiiKind = Literal["name", "email", "phone"]


class Column(BaseModel):
    name: str
    type: str
    description: str
    pii: PiiKind | None = None


class Table(BaseModel):
    name: str
    description: str
    columns: list[Column]
    glossary: list[str] = []

    def document(self) -> str:
        """Text that gets embedded for schema retrieval.

        The glossary holds the business words people actually use ("revenue", "units
        sold") so a question finds the table even when no column is named that way.
        """
        cols = "; ".join(f"{c.name}: {c.description}" for c in self.columns)
        terms = f" Business terms: {', '.join(self.glossary)}." if self.glossary else ""
        return f"Table {self.name}. {self.description}{terms} Columns: {cols}"


class Relationship(BaseModel):
    left: str
    left_column: str
    right: str
    right_column: str


def _c(name: str, type_: str, description: str, pii: PiiKind | None = None) -> Column:
    return Column(name=name, type=type_, description=description, pii=pii)


TABLES: list[Table] = [
    Table(
        name="customers",
        description="One row per customer account: who buys from us, their country and segment.",
        columns=[
            _c("customer_id", "INTEGER", "primary key of the customer"),
            _c("full_name", "VARCHAR", "customer name (masked)", "name"),
            _c("email", "VARCHAR", "customer email address (masked)", "email"),
            _c("phone", "VARCHAR", "customer phone number (masked)", "phone"),
            _c("country", "VARCHAR", "customer country, ISO name such as Germany or India"),
            _c("segment", "VARCHAR", "customer segment: consumer, smb or enterprise"),
            _c("signup_date", "DATE", "date the customer signed up"),
        ],
        glossary=["customer", "customers", "buyer", "account", "segment", "country"],
    ),
    Table(
        name="products",
        description="Product catalogue: product names, category, supplier and list price.",
        columns=[
            _c("product_id", "INTEGER", "primary key of the product"),
            _c("product_name", "VARCHAR", "product name"),
            _c("category", "VARCHAR", "product category such as Electronics, Home, Sports"),
            _c("supplier_id", "INTEGER", "supplier who provides the product"),
            _c("list_price", "DECIMAL(10,2)", "current list price in USD"),
            _c("unit_cost", "DECIMAL(10,2)", "cost to us per unit in USD, for margin"),
        ],
        glossary=["product", "products", "category", "catalogue", "price", "margin"],
    ),
    Table(
        name="orders",
        description=(
            "Sales orders header: one row per order with customer, order date, status and "
            "sales channel. Only status = 'completed' counts towards revenue and sales."
        ),
        columns=[
            _c("order_id", "INTEGER", "primary key of the order"),
            _c("customer_id", "INTEGER", "customer who placed the order"),
            _c("order_date", "DATE", "date the order was placed; use year() or month() on it"),
            _c("status", "VARCHAR", "completed, cancelled, refunded or pending"),
            _c("channel", "VARCHAR", "sales channel: web, mobile or store"),
        ],
        glossary=["order", "orders", "sales", "revenue", "year", "month", "channel", "cancelled"],
    ),
    Table(
        name="order_items",
        description=(
            "Order lines: one row per product in an order. Revenue of a line is "
            "quantity * unit_price * (1 - discount_pct / 100). Units sold is quantity."
        ),
        columns=[
            _c("order_item_id", "INTEGER", "primary key of the order line"),
            _c("order_id", "INTEGER", "order this line belongs to"),
            _c("product_id", "INTEGER", "product sold on this line"),
            _c("quantity", "INTEGER", "units sold on this line"),
            _c("unit_price", "DECIMAL(10,2)", "price per unit actually charged, in USD"),
            _c("discount_pct", "DECIMAL(5,2)", "discount percentage applied, 0 to 30"),
        ],
        glossary=[
            "revenue",
            "sales",
            "units sold",
            "quantity",
            "order value",
            "basket",
            "discount",
        ],
    ),
    Table(
        name="suppliers",
        description="Suppliers who provide our products, with their country.",
        columns=[
            _c("supplier_id", "INTEGER", "primary key of the supplier"),
            _c("supplier_name", "VARCHAR", "supplier company name"),
            _c("country", "VARCHAR", "supplier country"),
            _c("contact_email", "VARCHAR", "supplier contact email (masked)", "email"),
        ],
        glossary=["supplier", "suppliers", "vendor"],
    ),
    Table(
        name="inventory_snapshots",
        description="Month-end stock levels: units on hand per product per snapshot date.",
        columns=[
            _c("snapshot_date", "DATE", "month-end date of the stock count"),
            _c("product_id", "INTEGER", "product counted"),
            _c("units_on_hand", "INTEGER", "units in the warehouse at that date"),
        ],
        glossary=["inventory", "stock", "on hand"],
    ),
    Table(
        name="marketing_campaigns",
        description="Marketing campaigns with their channel, dates and budget spend.",
        columns=[
            _c("campaign_id", "INTEGER", "primary key of the campaign"),
            _c("campaign_name", "VARCHAR", "campaign name"),
            _c("channel", "VARCHAR", "marketing channel: email, social, search or tv"),
            _c("start_date", "DATE", "campaign start date"),
            _c("end_date", "DATE", "campaign end date"),
            _c("budget", "DECIMAL(12,2)", "campaign budget in USD"),
        ],
        glossary=["marketing", "campaign", "budget", "spend"],
    ),
    Table(
        name="support_tickets",
        description="Customer support tickets: category, priority, open and resolution times.",
        columns=[
            _c("ticket_id", "INTEGER", "primary key of the ticket"),
            _c("customer_id", "INTEGER", "customer who raised the ticket"),
            _c("opened_at", "TIMESTAMP", "when the ticket was opened"),
            _c("resolved_at", "TIMESTAMP", "when it was resolved; NULL if still open"),
            _c("category", "VARCHAR", "billing, delivery, product or account"),
            _c("priority", "VARCHAR", "low, medium or high"),
        ],
        glossary=["support", "ticket", "resolution time", "priority", "complaint"],
    ),
    Table(
        name="web_sessions",
        description="Website and app visits: device, pages viewed and whether the visit converted.",
        columns=[
            _c("session_id", "INTEGER", "primary key of the session"),
            _c("customer_id", "INTEGER", "logged-in customer, NULL for anonymous visitors"),
            _c("session_date", "DATE", "date of the visit"),
            _c("device", "VARCHAR", "desktop, mobile or tablet"),
            _c("pages_viewed", "INTEGER", "number of pages viewed in the session"),
            _c("converted", "BOOLEAN", "true when the session ended in an order"),
        ],
        glossary=["web", "session", "visit", "conversion", "converted", "device", "traffic"],
    ),
]

RELATIONSHIPS: list[Relationship] = [
    Relationship(
        left="orders", left_column="customer_id", right="customers", right_column="customer_id"
    ),
    Relationship(
        left="order_items", left_column="order_id", right="orders", right_column="order_id"
    ),
    Relationship(
        left="order_items", left_column="product_id", right="products", right_column="product_id"
    ),
    Relationship(
        left="products", left_column="supplier_id", right="suppliers", right_column="supplier_id"
    ),
    Relationship(
        left="inventory_snapshots",
        left_column="product_id",
        right="products",
        right_column="product_id",
    ),
    Relationship(
        left="support_tickets",
        left_column="customer_id",
        right="customers",
        right_column="customer_id",
    ),
    Relationship(
        left="web_sessions",
        left_column="customer_id",
        right="customers",
        right_column="customer_id",
    ),
]

# Governed metrics: a business term implies the tables its definition needs. This is
# what a semantic layer (dbt metrics, Cube, LookML) gives you, in miniature.
METRICS: dict[str, list[str]] = {
    "revenue": ["order_items", "orders"],
    "sales": ["order_items", "orders"],
    "order value": ["order_items", "orders"],
    "units sold": ["order_items", "products"],
    "margin": ["order_items", "products", "orders"],
    "conversion": ["web_sessions"],
    "resolution time": ["support_tickets"],
}

TABLES_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}
ALLOWED_TABLES: frozenset[str] = frozenset(TABLES_BY_NAME)


def schema_version() -> str:
    """Hash of the catalog. Cache entries made against an older schema are ignored."""
    payload = json.dumps(
        [t.model_dump() for t in TABLES] + [r.model_dump() for r in RELATIONSHIPS], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _adjacency() -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {t: set() for t in TABLES_BY_NAME}
    for r in RELATIONSHIPS:
        adj[r.left].add(r.right)
        adj[r.right].add(r.left)
    return adj


def join_path(a: str, b: str) -> list[str]:
    """Shortest chain of tables linking ``a`` to ``b`` (inclusive), or [] if unconnected."""
    adj = _adjacency()
    prev: dict[str, str | None] = {a: None}
    queue = deque([a])
    while queue:
        node = queue.popleft()
        if node == b:
            path = [node]
            while (p := prev[path[-1]]) is not None:
                path.append(p)
            return list(reversed(path))
        for nxt in sorted(adj[node]):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    return []


def close_join_paths(tables: list[str]) -> list[str]:
    """Add the bridge tables needed so every selected table can be joined to the first.

    Retrieval may return ``products`` and ``orders`` for "revenue by category", but the
    join needs ``order_items`` in between. Without this the model invents a join key.
    """
    if not tables:
        return []
    result = list(dict.fromkeys(tables))
    anchor = result[0]
    for t in list(result[1:]):
        for bridge in join_path(anchor, t):
            if bridge not in result:
                result.append(bridge)
    return result


def render_schema(tables: list[str]) -> str:
    """Compact DDL-like context for the prompt, with PII columns flagged."""
    lines: list[str] = []
    for name in tables:
        t = TABLES_BY_NAME[name]
        lines.append(f"TABLE {t.name} -- {t.description}")
        for c in t.columns:
            flag = " [PII, masked]" if c.pii else ""
            lines.append(f"  {c.name} {c.type} -- {c.description}{flag}")
    joins = [
        f"  {r.left}.{r.left_column} = {r.right}.{r.right_column}"
        for r in RELATIONSHIPS
        if r.left in tables and r.right in tables
    ]
    if joins:
        lines.append("JOINS")
        lines.extend(joins)
    return "\n".join(lines)
