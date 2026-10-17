"""OpenAI-compatible chat client using the standard library HTTP stack."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from document_ai import __version__
from document_ai.errors import DocumentAIError
from document_ai.llm.base import ChatMessage


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        json_mode: bool = True,
        timeout: float = 60,
    ) -> None:
        if not api_key:
            raise DocumentAIError("OPENAI_API_KEY is empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.json_mode = json_mode
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> OpenAICompatibleLLM:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise DocumentAIError(
                "OPENAI_API_KEY is not set. Use --llm parser for the offline baseline."
            )
        return cls(
            api_key=api_key,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            json_mode=os.environ.get("OPENAI_JSON_MODE", "1") != "0",
        )

    def complete(self, messages: list[ChatMessage]) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": list(messages),
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"document-ai/{__version__}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise DocumentAIError(f"model request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise DocumentAIError(f"model request failed: {exc.reason}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DocumentAIError("model response did not include message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise DocumentAIError("model response content was empty")
        return content
