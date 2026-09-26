"""Payments upstream, served over **streamable HTTP** with bearer auth.

Stands in for a payments provider. ``refund`` is a side-effecting write and
is idempotent on ``idempotency_key``: repeating a call returns the original
refund instead of paying twice.
"""

from __future__ import annotations

import argparse
import os
import threading
import uuid

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from mcp_gateway.demo_upstreams._auth import SharedSecretVerifier


def create_server(token: str | None = None) -> FastMCP:
    auth = SharedSecretVerifier(token) if token else None
    mcp = FastMCP("payments", instructions="Balances and refunds.", auth=auth)
    balances = {"ACC-1001": 1250.00, "ACC-1002": 80.50, "ACC-2001": 99_000.00}
    refunds: dict[str, dict[str, object]] = {}
    lock = threading.Lock()

    @mcp.tool(annotations={"readOnlyHint": True})
    def get_balance(account: str) -> dict[str, object]:
        """Current balance of an account such as ACC-1001."""
        if account not in balances:
            raise ToolError(f"unknown account {account}")
        return {"account": account, "balance": balances[account], "currency": "EUR"}

    @mcp.tool(annotations={"destructiveHint": True, "idempotentHint": True})
    def refund(order_id: str, amount: float, currency: str, idempotency_key: str) -> dict[str, object]:
        """Refund part or all of an order. Repeating the same idempotency_key is safe."""
        if amount <= 0:
            raise ToolError("amount must be positive")
        with lock:
            if idempotency_key in refunds:
                return refunds[idempotency_key] | {"replayed": True}
            record: dict[str, object] = {
                "refund_id": f"rf_{uuid.uuid4().hex[:10]}",
                "order_id": order_id,
                "amount": amount,
                "currency": currency,
                "status": "succeeded",
            }
            refunds[idempotency_key] = record
            return record | {"replayed": False}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9101")))
    a = parser.parse_args()
    token = os.environ.get("PAYMENTS_UPSTREAM_TOKEN")
    if not token:
        raise SystemExit("PAYMENTS_UPSTREAM_TOKEN is required")
    create_server(token).run(transport="http", host=a.host, port=a.port, show_banner=False)


if __name__ == "__main__":
    main()
