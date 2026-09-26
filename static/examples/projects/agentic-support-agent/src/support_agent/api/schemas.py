"""Request and response bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{4,64}$")


class ResumeRequest(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=500)
    interrupt_id: str | None = None


class ForkRequest(BaseModel):
    checkpoint_id: str
    message: str = Field(min_length=1, max_length=4000)


class ReplayRequest(BaseModel):
    checkpoint_id: str
