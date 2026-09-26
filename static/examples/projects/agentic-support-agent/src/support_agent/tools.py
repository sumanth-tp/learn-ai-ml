"""The agent's tools, the ToolNode that runs them, and its retry/permission wrapper.

No `from __future__ import annotations` here on purpose: ToolNode inspects the
error handler's annotation at runtime to decide which exceptions it handles.
"""

import asyncio
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolInvocationError
from langgraph.types import Command
from sqlalchemy.exc import OperationalError

from support_agent import metrics
from support_agent.config import Settings
from support_agent.errors import DomainError, NotAllowedError, TransientError
from support_agent.intent import Intent, normalise_order_id
from support_agent.services.faq import FaqRetriever
from support_agent.services.orders import OrderService
from support_agent.services.refunds import RefundService, money, refund_idempotency_key
from support_agent.state import Context, SupportState

log = logging.getLogger(__name__)

# Least privilege: the model for each intent only sees, and may only run, these.
TOOLS_BY_INTENT: dict[str, frozenset[str]] = {
    Intent.ORDER_STATUS: frozenset({"get_order", "list_my_orders", "search_faq"}),
    Intent.RETURNS: frozenset(
        {"get_order", "list_my_orders", "check_return_eligibility", "create_return", "search_faq"}
    ),
    Intent.REFUND: frozenset({"get_order", "list_my_orders", "issue_refund", "search_faq"}),
}


@dataclass
class ToolServices:
    settings: Settings
    orders: OrderService
    refunds: RefundService
    faq: FaqRetriever


Runtime = ToolRuntime[Context, SupportState]


def build_tools(svc: ToolServices) -> list[BaseTool]:
    threshold = money(svc.settings.refund_approval_threshold)

    @tool
    def get_order(order_id: str, runtime: Runtime) -> str:
        """Look up one of the customer's orders by id (for example ORD-1001).

        Returns status, items, totals, the refundable amount and tracking details.
        """
        user_id = runtime.context.user_id
        return json.dumps(svc.orders.get_order(user_id, normalise_order_id(order_id)))

    @tool
    def list_my_orders(runtime: Runtime, limit: int = 5) -> str:
        """List the customer's most recent orders, newest first."""
        return json.dumps(svc.orders.list_orders(runtime.context.user_id, limit))

    @tool
    def check_return_eligibility(order_id: str, runtime: Runtime) -> str:
        """Check whether an order can still be returned, and until when."""
        return json.dumps(
            svc.orders.return_eligibility(runtime.context.user_id, normalise_order_id(order_id))
        )

    @tool
    def create_return(order_id: str, reason: str, runtime: Runtime) -> str:
        """Open a return (RMA) for an eligible order. Safe to call twice for one order."""
        return json.dumps(
            svc.orders.create_return(runtime.context.user_id, normalise_order_id(order_id), reason)
        )

    @tool
    def issue_refund(order_id: str, amount: float, reason: str, runtime: Runtime) -> str:
        """Refund an amount on an order to the original payment method.

        Refunds above the approval threshold are held for a human reviewer
        automatically; call this tool normally and the system handles it.
        """
        order_id = normalise_order_id(order_id)
        amt = money(amount)
        approved_by: str | None = None
        if amt > threshold:
            # Defence in depth: the approval node should have run, but the tool
            # re-checks so no routing bug can refund a large amount unreviewed.
            decision = (runtime.state.get("approvals") or {}).get(runtime.tool_call_id or "")
            if decision is None:
                raise NotAllowedError(f"Refunds over {threshold} need a reviewer's approval.")
            if not decision["approved"]:
                metrics.REFUNDS.labels(status="rejected").inc()
                return json.dumps(
                    {"status": "rejected", "order_id": order_id, "note": decision.get("note", "")}
                )
            approved_by = decision["reviewer"]
        thread_id = str(runtime.config["configurable"]["thread_id"])
        result = svc.refunds.issue_refund(
            user_id=runtime.context.user_id,
            order_id=order_id,
            amount=amt,
            reason=reason,
            idempotency_key=refund_idempotency_key(thread_id, order_id, amt),
            approved_by=approved_by,
        )
        metrics.REFUNDS.labels(status="replayed" if result["replayed"] else "succeeded").inc()
        return json.dumps(result)

    @tool
    def search_faq(query: str) -> str:
        """Search the help centre for store policies (delivery, returns, payments)."""
        return json.dumps(svc.faq.search(query))

    return [
        get_order,
        list_my_orders,
        check_return_eligibility,
        create_return,
        issue_refund,
        search_faq,
    ]


def handle_tool_error(e: DomainError | ToolInvocationError) -> str:
    """Business-rule and bad-argument errors become an error ToolMessage.

    The model reads it and explains or corrects itself. Anything else
    (transient errors, bugs) is raised to the retry wrapper.
    """
    if isinstance(e, ToolInvocationError):
        return f"Error: invalid arguments. {e.message}"
    return f"Error: {e}"


RETRYABLE = (TransientError, OperationalError, TimeoutError)


def make_tool_wrapper(
    settings: Settings, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
) -> Callable[..., Awaitable[ToolMessage | Command]]:
    """Permission check, per-call timeout and retries with exponential backoff and jitter."""

    async def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        call = request.tool_call
        name = call["name"]
        intent = (request.state or {}).get("intent") if isinstance(request.state, dict) else None
        allowed = TOOLS_BY_INTENT.get(str(intent), frozenset())
        if name not in allowed:
            metrics.TOOL_CALLS.labels(tool=name, status="denied").inc()
            log.warning("tool denied", extra={"tool": name, "intent": intent})
            return ToolMessage(
                content=f"Error: tool '{name}' is not permitted for this request.",
                tool_call_id=call["id"],
                name=name,
                status="error",
            )
        delay = settings.tool_backoff_initial_s
        for attempt in range(1, settings.tool_max_attempts + 1):
            try:
                result = await asyncio.wait_for(execute(request), settings.tool_timeout_s)
            except RETRYABLE as exc:
                metrics.TOOL_CALLS.labels(tool=name, status="retry").inc()
                log.warning(
                    "tool transient failure",
                    extra={"tool": name, "attempt": attempt, "error": repr(exc)},
                )
                if attempt == settings.tool_max_attempts:
                    metrics.TOOL_CALLS.labels(tool=name, status="failed").inc()
                    return ToolMessage(
                        content=f"Error: the {name} service is temporarily unavailable. "
                        "Apologise and offer to try again or hand over to the team.",
                        tool_call_id=call["id"],
                        name=name,
                        status="error",
                    )
                await sleep(min(delay, settings.tool_backoff_max_s) * random.uniform(0.5, 1.5))
                delay *= 2
                continue
            status = getattr(result, "status", "success")
            metrics.TOOL_CALLS.labels(tool=name, status=str(status)).inc()
            return result
        raise AssertionError("unreachable")

    return wrapper


def build_tool_node(tools: list[BaseTool], settings: Settings) -> ToolNode:
    return ToolNode(
        tools, handle_tool_errors=handle_tool_error, awrap_tool_call=make_tool_wrapper(settings)
    )


def pending_tool_calls(state: SupportState) -> list[dict[str, Any]]:
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    return list(getattr(last, "tool_calls", None) or [])
