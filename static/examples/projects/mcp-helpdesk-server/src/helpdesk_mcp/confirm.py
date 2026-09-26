"""Human confirmation for destructive actions, on every kind of client.

MCP offers *elicitation*: the server asks the client to show the user a form.
How you use it depends on the negotiated protocol era, and some clients do
not support it at all, so there are three paths:

============================  ==============================================
Client                         What happens
============================  ==============================================
2026-07-28 + elicitation       Return ``InputRequiredResult``; the client asks
                               the user and retries the call with the answer.
2025-xx (handshake era) +      ``await ctx.elicit(...)`` over the SSE back
elicitation                    channel, inside the same call.
No elicitation support         Two-step: return a single-use confirm token;
                               the model must show the user and call again.
============================  ==============================================

``ctx.elicit`` raises on 2026-07-28 connections (that era removed
server-initiated requests), and ``InputRequiredResult`` is rejected on older
ones, which is why the era check comes first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mcp_types as mt
from fastmcp import Context
from mcp_types.version import MODERN_PROTOCOL_VERSIONS

from helpdesk_mcp.identity import Identity
from helpdesk_mcp.repository import HelpdeskRepository

CONFIRM_KEY = "confirm"


@dataclass
class ConfirmOutcome:
    decision: Literal["confirmed", "declined", "input_required", "token_issued"]
    input_required: mt.InputRequiredResult | None = None
    token: str | None = None


def _client_can_elicit(ctx: Context) -> bool:
    try:
        caps = ctx.session.client_capabilities
    except RuntimeError:
        return False
    return caps is not None and caps.elicitation is not None


def _is_modern(ctx: Context) -> bool:
    rc = ctx.request_context
    return rc is not None and rc.protocol_version in MODERN_PROTOCOL_VERSIONS


async def confirm_destructive(
    ctx: Context,
    repo: HelpdeskRepository,
    who: Identity,
    *,
    action: str,
    target_id: int,
    message: str,
    confirm_token: str | None,
    token_ttl_s: int,
) -> ConfirmOutcome:
    # Path 0: the caller is completing a two-step confirmation.
    if confirm_token:
        await repo.consume_confirm_token(who, confirm_token, action, target_id)
        return ConfirmOutcome("confirmed")

    if _client_can_elicit(ctx):
        if _is_modern(ctx):
            # Path 1: multi-round-trip. request_state is sealed by FastMCP, so a
            # client cannot swap the target between the question and the answer.
            expected_state = f"{action}:{target_id}"
            responses = ctx.input_responses
            if responses and ctx.request_state == expected_state:
                answer = responses.get(CONFIRM_KEY)
                accepted = (
                    isinstance(answer, mt.ElicitResult)
                    and answer.action == "accept"
                    and bool((answer.content or {}).get(CONFIRM_KEY))
                )
                return ConfirmOutcome("confirmed" if accepted else "declined")
            request = mt.ElicitRequest(
                params=mt.ElicitRequestFormParams(
                    message=message,
                    requested_schema={
                        "type": "object",
                        "properties": {
                            CONFIRM_KEY: {
                                "type": "boolean",
                                "title": "Yes, I am sure",
                                "description": message,
                            }
                        },
                        "required": [CONFIRM_KEY],
                    },
                )
            )
            return ConfirmOutcome(
                "input_required",
                input_required=mt.InputRequiredResult(
                    input_requests={CONFIRM_KEY: request}, request_state=expected_state
                ),
            )
        # Path 2: handshake-era imperative elicitation.
        result = await ctx.elicit(message, bool, response_title="Yes, I am sure")
        accepted = result.action == "accept" and bool(getattr(result, "data", False))
        return ConfirmOutcome("confirmed" if accepted else "declined")

    # Path 3: no elicitation. Hand back a token bound to user, action and target.
    token = await repo.issue_confirm_token(who, action, target_id, token_ttl_s)
    return ConfirmOutcome("token_issued", token=token)
