import pytest

from document_ai.errors import DocumentAIError, OCRUnavailable
from document_ai.ocr import doctr, tesseract
from document_ai.ocr.doctr import render_doctr_export
from document_ai.ocr.registry import build_ocr


def test_render_doctr_export_joins_words() -> None:
    exported = {
        "pages": [
            {
                "blocks": [
                    {
                        "lines": [
                            {"words": [{"value": "Vendor:"}, {"value": "Northwind"}]},
                            {"words": [{"value": "Total:"}, {"value": "9.72"}]},
                        ]
                    }
                ]
            }
        ]
    }
    assert render_doctr_export(exported) == "Vendor: Northwind\nTotal: 9.72"


def test_tesseract_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tesseract, "pytesseract", None)
    with pytest.raises(OCRUnavailable):
        tesseract.TesseractOCR()


def test_doctr_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctr, "ocr_predictor", None)
    monkeypatch.setattr(doctr, "np", None)
    with pytest.raises(OCRUnavailable):
        doctr.DocTROCR()


def test_unknown_ocr_engine() -> None:
    with pytest.raises(DocumentAIError, match="unknown OCR"):
        build_ocr("azure")
