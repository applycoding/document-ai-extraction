"""Tesseract adapter. Imported even when pytesseract is not installed."""

from __future__ import annotations

from PIL import Image

from document_ai.errors import OCRUnavailable

try:
    import pytesseract
except ImportError:  # optional extra: document-ai[ocr-tesseract]
    pytesseract = None


class TesseractOCR:
    name = "tesseract"

    def __init__(self) -> None:
        if pytesseract is None:
            raise OCRUnavailable(
                "pytesseract is not installed. Install the ocr-tesseract extra "
                "and the tesseract binary."
            )

    def extract(self, image: Image.Image) -> str:
        if pytesseract is None:
            raise OCRUnavailable("pytesseract is not installed.")
        try:
            text = pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRUnavailable("The tesseract binary was not found on PATH.") from exc
        return text or ""
