"""Wire ingestion and extraction into one callable pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from document_ai.extract import DEFAULT_MAX_RETRIES, extract_invoice
from document_ai.ingest import Page, ingest_bytes, needs_ocr
from document_ai.llm.base import LLMClient
from document_ai.llm.registry import build_llm
from document_ai.models import ExtractionResult
from document_ai.ocr.base import OCREngine
from document_ai.ocr.registry import build_ocr


class Pipeline:
    def __init__(
        self,
        *,
        llm: LLMClient,
        ocr_factory: Callable[[], OCREngine],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.llm = llm
        self._ocr_factory = ocr_factory
        self._ocr: OCREngine | None = None
        self.max_retries = max_retries

    def run_path(self, path: Path) -> ExtractionResult:
        return self.run_bytes(path.read_bytes(), path.name)

    def run_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        pages = self.read_pages(data, filename)
        return extract_invoice(self.llm, pages, max_retries=self.max_retries)

    def read_pages(self, data: bytes, filename: str) -> list[Page]:
        ocr = self._ocr_engine() if needs_ocr(data, filename) else None
        return ingest_bytes(data, filename, ocr)

    def _ocr_engine(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = self._ocr_factory()
        return self._ocr


def build_pipeline(
    ocr_name: str,
    llm_name: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    scripted: list[str] | None = None,
    fake_text: str = "",
) -> Pipeline:
    return Pipeline(
        llm=build_llm(llm_name, scripted=scripted),
        ocr_factory=lambda: build_ocr(ocr_name, fake_text=fake_text),
        max_retries=max_retries,
    )


def pipeline_from_env() -> Pipeline:
    retries = int(os.environ.get("DOCUMENT_AI_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    return build_pipeline(
        os.environ.get("DOCUMENT_AI_OCR", "tesseract"),
        os.environ.get("DOCUMENT_AI_LLM", "parser"),
        max_retries=retries,
    )
