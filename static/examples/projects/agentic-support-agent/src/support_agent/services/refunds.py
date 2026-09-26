"""Refunds: a provider interface (HTTP + stub) and an idempotent ledger service."""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from support_agent.db import Order, Refund, session_scope
from support_agent.errors import (
    NotAllowedError,
    NotFoundError,
    PermanentGatewayError,
    TransientError,
)

log = logging.getLogger(__name__)


def money(value: float | str | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def refund_idempotency_key(thread_id: str, order_id: str, amount: Decimal) -> str:
    """Same conversation + same order + same amount = the same refund.

    It is stable across a resumed interrupt, a retried tool call, a replay
    from an old checkpoint and a customer asking twice in one thread.
    """
    raw = f"{thread_id}|{order_id.upper()}|{money(amount)}"
    return "rf_" + hashlib.sha256(raw.encode()).hexdigest()[:40]


@dataclass(frozen=True)
class GatewayResult:
    provider_ref: str
    status: str


class RefundGateway(Protocol):
    """The payment provider. It must honour the idempotency key itself."""

    def refund(
        self, *, idempotency_key: str, order_id: str, amount: Decimal, currency: str
    ) -> GatewayResult: ...


class HttpRefundGateway:
    """Talks to a real refund API that accepts an Idempotency-Key header."""

    def __init__(self, base_url: str, api_key: str | None, timeout_s: float) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    def refund(
        self, *, idempotency_key: str, order_id: str, amount: Decimal, currency: str
    ) -> GatewayResult:
        try:
            resp = self._client.post(
                "/refunds",
                json={"order_id": order_id, "amount": str(amount), "currency": currency},
                headers={"Idempotency-Key": idempotency_key},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"refund provider unreachable: {exc!r}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientError(f"refund provider returned {resp.status_code}")
        if resp.status_code >= 400:
            raise PermanentGatewayError(f"refund provider rejected: {resp.text[:200]}")
        body = resp.json()
        return GatewayResult(provider_ref=body["id"], status=body.get("status", "succeeded"))


@dataclass
class StubRefundGateway:
    """In-process fake of the provider, with the same idempotency semantics.

    `fail_next` queues exceptions to raise on the next calls (for tests);
    `failure_rate` injects random transient failures (for chaos in dev).
    """

    failure_rate: float = 0.0
    fail_next: list[Exception] = field(default_factory=list)
    calls: int = 0
    _by_key: dict[str, GatewayResult] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _rng: random.Random = field(default_factory=lambda: random.Random(7))

    def refund(
        self, *, idempotency_key: str, order_id: str, amount: Decimal, currency: str
    ) -> GatewayResult:
        with self._lock:
            self.calls += 1
            if self.fail_next:
                raise self.fail_next.pop(0)
            if self.failure_rate and self._rng.random() < self.failure_rate:
                raise TransientError("stub provider: injected timeout")
            if idempotency_key not in self._by_key:
                ref = "re_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]
                self._by_key[idempotency_key] = GatewayResult(provider_ref=ref, status="succeeded")
            return self._by_key[idempotency_key]

    @property
    def distinct_refunds(self) -> int:
        return len(self._by_key)


class RefundService:
    def __init__(self, factory: sessionmaker[Session], gateway: RefundGateway) -> None:
        self._factory = factory
        self._gateway = gateway

    def issue_refund(
        self,
        *,
        user_id: str,
        order_id: str,
        amount: Decimal,
        reason: str,
        idempotency_key: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        amount = money(amount)
        order_id = order_id.strip().upper()
        try:
            refund_id, currency, replay = self._claim(
                user_id, order_id, amount, reason, idempotency_key, approved_by
            )
        except OperationalError as exc:
            raise TransientError("refund ledger unavailable") from exc
        if replay is not None:
            log.info("refund replayed", extra={"order_id": order_id, "refund_id": refund_id})
            return replay

        # Network call happens outside any DB transaction.
        try:
            result = self._gateway.refund(
                idempotency_key=idempotency_key, order_id=order_id, amount=amount,
                currency=currency,
            )
        except PermanentGatewayError as exc:
            self._mark(refund_id, "failed", error=str(exc))
            raise NotAllowedError(f"The payment provider declined the refund: {exc}") from exc
        except TransientError as exc:
            # Leave the row 'pending': a retry with the same key finishes it.
            self._mark(refund_id, "pending", error=str(exc))
            raise

        self._complete(refund_id, order_id, amount, result.provider_ref)
        log.info("refund issued", extra={"order_id": order_id, "amount": str(amount)})
        return {"refund_id": refund_id, "order_id": order_id, "amount": str(amount),
                "currency": currency, "status": "succeeded",
                "provider_ref": result.provider_ref, "replayed": False}

    # -- internals ----------------------------------------------------------

    def _claim(
        self, user_id: str, order_id: str, amount: Decimal, reason: str, key: str,
        approved_by: str | None,
    ) -> tuple[int, str, dict[str, Any] | None]:
        """Find or create the ledger row for this key.

        Returns (refund_id, currency, replay_result). replay_result is set when
        the refund already succeeded, in which case the provider is not called.
        """
        with session_scope(self._factory) as s:
            order = s.get(Order, order_id)
            if order is None or order.customer_id != user_id:
                raise NotFoundError(f"No order {order_id} found on your account.")
            existing = s.scalar(select(Refund).where(Refund.idempotency_key == key))
            if existing is not None:
                if existing.status == "succeeded":
                    return existing.id, order.currency, {
                        "refund_id": existing.id, "order_id": order_id,
                        "amount": str(existing.amount), "currency": order.currency,
                        "status": "succeeded", "provider_ref": existing.provider_ref,
                        "replayed": True,
                    }
                return existing.id, order.currency, None  # pending/failed: finish it
            if amount <= 0:
                raise NotAllowedError("Refund amount must be positive.")
            in_flight = s.scalar(
                select(func.coalesce(func.sum(Refund.amount), 0)).where(
                    Refund.order_id == order_id, Refund.status == "pending"
                )
            )
            available = order.refundable - Decimal(in_flight)
            if amount > available:
                raise NotAllowedError(
                    f"Only {available:.2f} {order.currency} can still be refunded on {order_id}."
                )
            row = Refund(idempotency_key=key, order_id=order_id, amount=amount,
                         reason=reason[:500], status="pending", approved_by=approved_by)
            s.add(row)
            try:
                s.flush()
            except IntegrityError:
                s.rollback()
                raced = s.scalar(select(Refund).where(Refund.idempotency_key == key))
                assert raced is not None
                return raced.id, order.currency, None
            return row.id, order.currency, None

    def _mark(self, refund_id: int, status: str, error: str | None = None) -> None:
        with session_scope(self._factory) as s:
            s.execute(
                update(Refund)
                .where(Refund.id == refund_id, Refund.status != "succeeded")
                .values(status=status, error=(error or "")[:500])
            )

    def _complete(self, refund_id: int, order_id: str, amount: Decimal, ref: str) -> None:
        with session_scope(self._factory) as s:
            # Conditional update: only the first finisher moves the order balance.
            res = s.execute(
                update(Refund)
                .where(Refund.id == refund_id, Refund.status != "succeeded")
                .values(status="succeeded", provider_ref=ref, error=None)
            )
            if res.rowcount == 1:  # type: ignore[attr-defined]
                s.execute(
                    update(Order)
                    .where(Order.id == order_id)
                    .values(refunded_amount=Order.refunded_amount + amount)
                )

    def count_succeeded(self, order_id: str) -> int:
        with session_scope(self._factory) as s:
            return int(
                s.scalar(
                    select(func.count()).select_from(Refund).where(
                        Refund.order_id == order_id, Refund.status == "succeeded"
                    )
                )
                or 0
            )
