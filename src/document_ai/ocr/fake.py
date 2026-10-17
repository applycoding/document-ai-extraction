"""OCR stand-in that returns scripted text. Tests use this instead of a model."""

from __future__ import annotations

from PIL import Image


class FakeOCR:
    name = "fake"

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def extract(self, image: Image.Image) -> str:
        self.calls += 1
        return self.text
