"""Pluggable OCR backends."""

from document_ai.ocr.base import OCREngine
from document_ai.ocr.registry import build_ocr

__all__ = ["OCREngine", "build_ocr"]
