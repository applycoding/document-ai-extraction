"""Scripted model for tests. Each call returns the next canned string."""

from __future__ import annotations

from document_ai.errors import DocumentAIError
from document_ai.llm.base import ChatMessage


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[ChatMessage]) -> str:
        if self.calls >= len(self._responses):
            raise DocumentAIError("FakeLLM has no remaining scripted responses")
        response = self._responses[self.calls]
        self.calls += 1
        return response
