from __future__ import annotations

import json
from typing import Any

import pytest

from document_ai.errors import DocumentAIError
from document_ai.llm.fake import FakeLLM
from document_ai.llm.openai_compat import OpenAICompatibleLLM
from document_ai.llm.registry import build_llm


def test_fake_llm_returns_scripted_responses_in_order() -> None:
    fake = FakeLLM(["one", "two"])
    assert fake.complete([{"role": "user", "content": "hi"}]) == "one"
    assert fake.complete([{"role": "user", "content": "hi"}]) == "two"
    with pytest.raises(DocumentAIError):
        fake.complete([{"role": "user", "content": "hi"}])


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(DocumentAIError, match="OPENAI_API_KEY"):
        build_llm("openai")


def test_openai_client_posts_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Body:
        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "{\"invoice\": { }}"}}]}
            ).encode()

    def _urlopen(request: Any, timeout: float) -> _Body:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode())
        return _Body()

    monkeypatch.setattr("document_ai.llm.openai_compat.urllib.request.urlopen", _urlopen)
    client = OpenAICompatibleLLM(
        api_key="test-key",
        model="fake-model",
        base_url="https://example.test/v1",
        json_mode=True,
        timeout=5,
    )
    text = client.complete([{"role": "user", "content": "extract"}])
    assert text == '{"invoice": { }}'
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["timeout"] == 5
    assert captured["body"]["model"] == "fake-model"
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_unknown_llm() -> None:
    with pytest.raises(DocumentAIError, match="unknown extractor"):
        build_llm("local-gguf")
