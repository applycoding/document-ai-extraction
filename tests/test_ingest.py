import io

import pytest
from PIL import Image
from pypdf import PdfReader

from document_ai.dataset.generate import render_invoice
from document_ai.errors import IngestError
from document_ai.ingest import ingest_bytes
from document_ai.llm.parser import ParserLLM
from document_ai.models import TextSource
from document_ai.ocr.fake import FakeOCR
from document_ai.pipeline import Pipeline
from document_ai.textpdf import build_text_pdf
from tests.helpers import embedded_pages


def test_pdf_text_layer_keeps_labels() -> None:
    _invoice, sheets = render_invoice(1)
    data = build_text_pdf(sheets)
    reader = PdfReader(io.BytesIO(data))
    text = reader.pages[0].extract_text() or ""
    assert "Vendor:" in text
    assert "INV-1001" in text
    assert "SYNTHETIC INVOICE" in text


def test_pipeline_reads_text_pdf_without_calling_ocr() -> None:
    invoice, sheets = render_invoice(1)
    calls = {"count": 0}

    def factory() -> FakeOCR:
        calls["count"] += 1
        return FakeOCR("unused")

    result = Pipeline(llm=ParserLLM(), ocr_factory=factory).run_bytes(
        build_text_pdf(sheets), "invoice.pdf"
    )
    assert calls["count"] == 0
    assert result.invoice == invoice
    assert result.text_sources == [TextSource.EMBEDDED]


def test_multipage_pdf_round_trip() -> None:
    invoice, sheets = render_invoice(4)
    result = Pipeline(llm=ParserLLM(), ocr_factory=FakeOCR).run_bytes(
        build_text_pdf(sheets), "invoice.pdf"
    )
    assert result.page_count == 2
    assert result.invoice == invoice


def test_image_uses_ocr_text() -> None:
    invoice, pages = embedded_pages(1)
    image = Image.new("RGB", (80, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    ocr = FakeOCR(pages[0].text)
    result = Pipeline(llm=ParserLLM(), ocr_factory=lambda: ocr).run_bytes(
        buffer.getvalue(), "receipt.png"
    )
    assert ocr.calls == 1
    assert result.text_sources == [TextSource.OCR]
    assert result.invoice == invoice


def test_blank_pdf_explains_missing_text_layer() -> None:
    with pytest.raises(IngestError, match="no text layer"):
        ingest_bytes(build_text_pdf([[]]), "scan.pdf", None)


def test_empty_file_is_rejected() -> None:
    with pytest.raises(IngestError, match="empty"):
        ingest_bytes(b"", "invoice.pdf", None)


def test_unsupported_type_is_rejected() -> None:
    with pytest.raises(IngestError, match="Unsupported"):
        ingest_bytes(b"hello", "notes.txt", None)
