"""Graph state, run context and the custom reducers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

RESET = -1
"""Send this to an add_or_reset channel to zero it at the start of a turn."""


def add_or_reset(current: int | None, update: int | None) -> int:
    """Counter reducer: adds, except that RESET zeroes it.

    Plain operator.add cannot be reset, and a per-request budget must start
    from zero on every new user message while still accumulating inside it.
    """
    if update is None:
        return current or 0
    if update == RESET:
        return 0
    return (current or 0) + update


def merge_dicts(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge reducer so two nodes can add approvals without clobbering."""
    return {**(current or {}), **(update or {})}


class SupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str | None
    order_id: str | None
    blocked_reason: str | None
    summary: str
    user_context: str
    steps: Annotated[int, add_or_reset]
    tokens: Annotated[int, add_or_reset]
    approvals: Annotated[dict[str, Any], merge_dicts]
    outcome: str | None


@dataclass(frozen=True)
class Context:
    """Per-run, non-persisted context. The user id comes from auth, never from the model."""

    user_id: str
    request_id: str = "-"
