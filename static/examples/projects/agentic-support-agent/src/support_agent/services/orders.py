"""Order lookups and returns, always scoped to the authenticated customer."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from support_agent.db import Order, ReturnRequest, session_scope, utcnow
from support_agent.errors import NotAllowedError, NotFoundError, TransientError

RETURN_WINDOW_DAYS = 30


def as_utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo on read; treat naive values as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.id,
        "status": order.status,
        "total": f"{order.total:.2f}",
        "refunded": f"{order.refunded_amount:.2f}",
        "refundable": f"{order.refundable:.2f}",
        "currency": order.currency,
        "items": order.items,
        "placed_at": as_utc(order.placed_at).date().isoformat(),
        "delivered_at": as_utc(order.delivered_at).date().isoformat()
        if order.delivered_at
        else None,
        "carrier": order.carrier,
        "tracking_number": order.tracking_number,
    }


class OrderService:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self._factory = factory

    def _owned(self, s: Session, user_id: str, order_id: str) -> Order:
        order = s.get(Order, order_id.strip().upper())
        # Same message for "missing" and "someone else's": do not leak existence.
        if order is None or order.customer_id != user_id:
            raise NotFoundError(f"No order {order_id} found on your account.")
        return order

    def get_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        try:
            with session_scope(self._factory) as s:
                return order_to_dict(self._owned(s, user_id, order_id))
        except OperationalError as exc:
            raise TransientError("order database unavailable") from exc

    def list_orders(self, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            with session_scope(self._factory) as s:
                rows = s.scalars(
                    select(Order)
                    .where(Order.customer_id == user_id)
                    .order_by(Order.placed_at.desc())
                    .limit(max(1, min(limit, 20)))
                ).all()
                return [order_to_dict(o) for o in rows]
        except OperationalError as exc:
            raise TransientError("order database unavailable") from exc

    def return_eligibility(
        self, user_id: str, order_id: str, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or utcnow()
        with session_scope(self._factory) as s:
            order = self._owned(s, user_id, order_id)
            existing = s.scalar(select(ReturnRequest).where(ReturnRequest.order_id == order.id))
            if existing is not None:
                return {"order_id": order.id, "eligible": False,
                        "reason": f"Return {existing.id} is already open for this order."}
            if order.status != "delivered" or order.delivered_at is None:
                return {"order_id": order.id, "eligible": False,
                        "reason": f"The order is '{order.status}', not delivered yet."}
            deadline = as_utc(order.delivered_at) + timedelta(days=RETURN_WINDOW_DAYS)
            if now > deadline:
                return {"order_id": order.id, "eligible": False,
                        "reason": f"The {RETURN_WINDOW_DAYS}-day return window closed on "
                        f"{deadline.date().isoformat()}."}
            return {"order_id": order.id, "eligible": True,
                    "return_by": deadline.date().isoformat()}

    def create_return(self, user_id: str, order_id: str, reason: str) -> dict[str, Any]:
        """Idempotent: a second call for the same order returns the existing RMA."""
        check = self.return_eligibility(user_id, order_id)
        with session_scope(self._factory) as s:
            order = self._owned(s, user_id, order_id)
            existing = s.scalar(select(ReturnRequest).where(ReturnRequest.order_id == order.id))
            if existing is not None:
                return {"return_id": existing.id, "order_id": order.id,
                        "status": existing.status, "replayed": True}
            if not check["eligible"]:
                raise NotAllowedError(check["reason"])
            rma = "RMA-" + hashlib.sha256(order.id.encode()).hexdigest()[:8].upper()
            s.add(ReturnRequest(id=rma, order_id=order.id, reason=reason[:500]))
            try:
                s.flush()
            except IntegrityError:
                # A concurrent request created it between our check and insert.
                s.rollback()
                existing = s.scalar(
                    select(ReturnRequest).where(ReturnRequest.order_id == order.id)
                )
                assert existing is not None
                return {"return_id": existing.id, "order_id": order.id,
                        "status": existing.status, "replayed": True}
            return {"return_id": rma, "order_id": order.id, "status": "open",
                    "replayed": False,
                    "instructions": "Print the label from your order page and drop the "
                    "parcel at any post office within 14 days."}
