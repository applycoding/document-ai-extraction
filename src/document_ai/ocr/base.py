"""OCR engine interface."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class OCREngine(Protocol):
    name: str

    def extract(self, image: Image.Image) -> str:
        """Return plain text for one image."""
