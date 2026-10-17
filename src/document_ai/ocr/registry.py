"""Build an OCR engine by name."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

from document_ai.errors import DocumentAIError
from document_ai.ocr.base import OCREngine
from document_ai.ocr.doctr import DocTROCR
from document_ai.ocr.fake import FakeOCR
from document_ai.ocr.tesseract import TesseractOCR


class OCRName(StrEnum):
    TESSERACT = "tesseract"
    DOCTR = "doctr"
    FAKE = "fake"


def build_ocr(name: str, *, fake_text: str = "") -> OCREngine:
    try:
        kind = OCRName(name)
    except ValueError as exc:
        raise DocumentAIError(
            f"unknown OCR engine {name!r}. Choose tesseract, doctr, or fake."
        ) from exc
    match kind:
        case OCRName.TESSERACT:
            return TesseractOCR()
        case OCRName.DOCTR:
            return DocTROCR()
        case OCRName.FAKE:
            return FakeOCR(fake_text)
        case _ as unreachable:
            assert_never(unreachable)
