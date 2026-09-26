"""Order and refund services against a real (SQLite) database with the stub provider."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from support_agent.container import Container
from support_agent.db import Refund, session_scope, utcnow
from support_agent.errors import (
    NotAllowedError,
    NotFoundError,
    PermanentGatewayError,
    TransientError,
)
from support_agent.services.refunds import StubRefundGateway, refund_idempotency_key


def _refund(
    c: Container, key: str, amount: str = "10.00", order: str = "ORD-1001", user: str = "cust_001"
) -> dict:
    return c.deps.tools.refunds.issue_refund(
        user_id=user, order_id=order, amount=Decimal(amount), reason="test", idempotency_key=key
    )


def test_same_key_refunds_once(container: Container, gateway: StubRefundGateway) -> None:
    first = _refund(container, "k1")
    second = _refund(container, "k1")
    assert first["replayed"] is False and second["replayed"] is True
    assert first["provider_ref"] == second["provider_ref"]
    assert gateway.calls == 1  # replay never reaches the provider
    assert container.deps.tools.refunds.count_succeeded("ORD-1001") == 1


def test_cannot_refund_more_than_remaining(container: Container) -> None:
    _refund(container, "k1", "40.00")
    with pytest.raises(NotAllowedError, match=r"Only 9\.99"):
        _refund(container, "k2", "20.00")


def test_other_customers_order_is_not_found(container: Container) -> None:
    with pytest.raises(NotFoundError):
        _refund(container, "k1", order="ORD-2001", user="cust_001")


def test_transient_failure_leaves_pending_then_retry_completes(
    container: Container, gateway: StubRefundGateway
) -> None:
    gateway.fail_next.append(TransientError("timeout"))
    with pytest.raises(TransientError):
        _refund(container, "k1")
    with session_scope(container.sessions) as s:
        assert s.query(Refund).one().status == "pending"
    result = _refund(container, "k1")  # the retry, same key
    assert result["status"] == "succeeded"
    assert gateway.distinct_refunds == 1
    assert container.deps.tools.refunds.count_succeeded("ORD-1001") == 1


def test_permanent_gateway_error_marks_failed(
    container: Container, gateway: StubRefundGateway
) -> None:
    gateway.fail_next.append(PermanentGatewayError("card expired"))
    with pytest.raises(NotAllowedError, match="declined"):
        _refund(container, "k1")
    with session_scope(container.sessions) as s:
        assert s.query(Refund).one().status == "failed"


def test_idempotency_key_is_stable_and_amount_sensitive() -> None:
    a = refund_idempotency_key("t1", "ord-1001", Decimal("10"))
    assert a == refund_idempotency_key("t1", "ORD-1001", Decimal("10.00"))
    assert a != refund_idempotency_key("t1", "ORD-1001", Decimal("10.01"))
    assert a != refund_idempotency_key("t2", "ORD-1001", Decimal("10"))


def test_return_eligibility_and_idempotent_create(container: Container) -> None:
    orders = container.deps.tools.orders
    assert orders.return_eligibility("cust_001", "ORD-1001")["eligible"] is True
    assert orders.return_eligibility("cust_001", "ORD-1003")["eligible"] is False  # shipped
    assert orders.return_eligibility("cust_001", "ORD-1004")["eligible"] is False  # 45 days
    later = utcnow() + timedelta(days=40)
    assert orders.return_eligibility("cust_001", "ORD-1001", now=later)["eligible"] is False
    first = orders.create_return("cust_001", "ORD-1001", "chipped")
    again = orders.create_return("cust_001", "ORD-1001", "chipped")
    assert first["return_id"] == again["return_id"] and again["replayed"] is True


def test_faq_retriever_finds_policy_and_rejects_unrelated(container: Container) -> None:
    faq = container.deps.tools.faq
    assert faq.search("how long does a refund take")[0]["id"] == "refund-timing"
    assert faq.search("bitcoin price prediction") == []
