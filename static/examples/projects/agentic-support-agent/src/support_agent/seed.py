"""Idempotent seed data: running it twice leaves the same rows."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session, sessionmaker

from support_agent.db import Customer, Order, session_scope, utcnow

CUSTOMERS = [
    ("cust_001", "Asha Patel", "asha.patel@example.com", "+44 7700 900123"),
    ("cust_002", "Ben Carter", "ben.carter@example.com", "+44 7700 900456"),
    ("cust_003", "Chloe Martin", "chloe.martin@example.com", None),
]


def _orders(now: datetime) -> list[Order]:
    def d(days: int) -> datetime:
        return now - timedelta(days=days)

    return [
        Order(
            id="ORD-1001",
            customer_id="cust_001",
            status="delivered",
            total=Decimal("49.99"),
            items=[{"sku": "MUG-01", "name": "Ceramic mug", "qty": 2}],
            placed_at=d(9),
            delivered_at=d(5),
            carrier="Royal Mail",
            tracking_number="RM123GB",
        ),
        Order(
            id="ORD-1002",
            customer_id="cust_001",
            status="delivered",
            total=Decimal("249.00"),
            items=[{"sku": "HEAD-02", "name": "Noise-cancelling headphones", "qty": 1}],
            placed_at=d(14),
            delivered_at=d(10),
            carrier="DPD",
            tracking_number="DPD998877",
        ),
        Order(
            id="ORD-1003",
            customer_id="cust_001",
            status="shipped",
            total=Decimal("19.50"),
            items=[{"sku": "BOOK-07", "name": "Paperback novel", "qty": 1}],
            placed_at=d(2),
            delivered_at=None,
            carrier="Royal Mail",
            tracking_number="RM555GB",
        ),
        Order(
            id="ORD-1004",
            customer_id="cust_001",
            status="delivered",
            total=Decimal("75.00"),
            items=[{"sku": "LAMP-03", "name": "Desk lamp", "qty": 1}],
            placed_at=d(50),
            delivered_at=d(45),
            carrier="DPD",
            tracking_number="DPD112233",
        ),
        Order(
            id="ORD-2001",
            customer_id="cust_002",
            status="delivered",
            total=Decimal("89.50"),
            items=[{"sku": "SHOE-11", "name": "Running shoes", "qty": 1}],
            placed_at=d(6),
            delivered_at=d(3),
            carrier="Evri",
            tracking_number="EV777",
        ),
        Order(
            id="ORD-2002",
            customer_id="cust_002",
            status="placed",
            total=Decimal("12.00"),
            items=[{"sku": "SOCK-05", "name": "Socks (3 pack)", "qty": 1}],
            placed_at=d(0),
            delivered_at=None,
            carrier=None,
            tracking_number=None,
        ),
    ]


def seed(factory: sessionmaker[Session], now: datetime | None = None) -> int:
    """Insert customers and orders that are missing. Returns rows inserted."""
    now = now or utcnow()
    inserted = 0
    with session_scope(factory) as s:
        for cid, name, email, phone in CUSTOMERS:
            if s.get(Customer, cid) is None:
                s.add(Customer(id=cid, name=name, email=email, phone=phone))
                inserted += 1
        s.flush()
        for order in _orders(now):
            if s.get(Order, order.id) is None:
                s.add(order)
                inserted += 1
    return inserted
