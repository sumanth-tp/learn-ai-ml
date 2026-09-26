"""Test doubles shared across test modules."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class ToolCallingFake(FakeMessagesListChatModel):
    """LangChain's list-replay fake, made tool-capable, for scripting exact model turns."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFake:  # type: ignore[override]
        return self
