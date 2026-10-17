"""Build an extractor by name."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

from document_ai.errors import DocumentAIError
from document_ai.llm.base import LLMClient
from document_ai.llm.fake import FakeLLM
from document_ai.llm.openai_compat import OpenAICompatibleLLM
from document_ai.llm.parser import ParserLLM


class LLMName(StrEnum):
    PARSER = "parser"
    OPENAI = "openai"
    FAKE = "fake"


def build_llm(name: str, *, scripted: list[str] | None = None) -> LLMClient:
    try:
        kind = LLMName(name)
    except ValueError as exc:
        raise DocumentAIError(
            f"unknown extractor {name!r}. Choose parser, openai, or fake."
        ) from exc
    match kind:
        case LLMName.PARSER:
            return ParserLLM()
        case LLMName.OPENAI:
            return OpenAICompatibleLLM.from_env()
        case LLMName.FAKE:
            return FakeLLM(scripted or [])
        case _ as unreachable:
            assert_never(unreachable)
