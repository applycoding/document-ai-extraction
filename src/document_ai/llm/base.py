"""Chat-model interface used by the extractor."""

from __future__ import annotations

from typing import Protocol, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMClient(Protocol):
    def complete(self, messages: list[ChatMessage]) -> str:
        """Return the assistant message content."""
