"""Pydantic models that form the public contract of the server.

These types become the JSON Schemas that MCP clients (and the LLMs behind
them) see in ``tools/list``. Field descriptions are written for a model
reader: they say what a value means and when to use it, not how it is stored.
Changing them is an API change; see ``schema_compat.py``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Status(StrEnum):
    open = "open"
    in_progress = "in_progress"
    waiting_on_user = "waiting_on_user"
    resolved = "resolved"
    closed = "closed"


class Priority(StrEnum):
    p1 = "p1"  # service down for many users
    p2 = "p2"  # one team blocked
    p3 = "p3"  # one user blocked / degraded
    p4 = "p4"  # question or minor annoyance


class Category(StrEnum):
    access = "access"
    hardware = "hardware"
    software = "software"
    network = "network"
    email = "email"
    other = "other"


class CommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    author: str
    body: str
    internal: bool = Field(description="True if only helpdesk agents can see this comment.")
    created_at: datetime


class TicketOut(BaseModel):
    """A helpdesk ticket as returned to clients."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    status: Status
    priority: Priority
    category: Category
    requester: str
    assignee: str | None
    version: int = Field(
        description="Increments on every change. Pass it back as expected_version when updating."
    )
    created_at: datetime
    updated_at: datetime
    uri: str = Field(description="Resource URI for this ticket, readable with resources/read.")


class TicketDetail(TicketOut):
    comments: list[CommentOut] = Field(default_factory=list)


class Page[T](BaseModel):
    """One page of results. Pass next_cursor back to get the next page."""

    items: list[T]
    next_cursor: str | None = Field(
        default=None, description="Opaque cursor for the next page; null when there are no more."
    )


class TicketPage(Page[TicketOut]):
    pass


class DeleteResult(BaseModel):
    """Outcome of a destructive request."""

    status: str = Field(description="'deleted', 'cancelled' or 'confirmation_required'.")
    ticket_id: int
    confirm_token: str | None = Field(
        default=None,
        description=(
            "Present when status is confirmation_required. Show the user what will be deleted, "
            "and only if they agree call delete_ticket again with this token."
        ),
    )
    expires_in_s: int | None = None
    message: str


class TriageSuggestion(BaseModel):
    """A suggested classification. Advisory only; nothing is changed until update_ticket."""

    category: Category
    priority: Priority
    rationale: str = Field(max_length=500)
    suggested_kb_slugs: list[str] = Field(default_factory=list)
    source: str = Field(description="'llm' when a model produced it, 'fallback' when rules did.")


class IncidentReport(BaseModel):
    window_hours: int
    tickets_scanned: int
    by_category: dict[str, int]
    by_priority: dict[str, int]
    open_p1_ids: list[int]
    top_category: str | None


class KBArticleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    title: str
    category: Category
    body: str


class KBArticleSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    title: str
    category: Category
    uri: str


class Me(BaseModel):
    user: str
    tenant: str
    roles: list[str]
