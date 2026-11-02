"""Shared test helpers."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class ToolCallingFake(FakeMessagesListChatModel):
    """Scripted chat model that accepts ``bind_tools`` (returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self
