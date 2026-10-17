"""Shared fixtures for pipeline tests."""

from document_ai.dataset.generate import render_invoice
from document_ai.ingest import Page
from document_ai.models import Invoice, TextSource


def embedded_pages(index: int = 1) -> tuple[Invoice, list[Page]]:
    invoice, sheets = render_invoice(index)
    pages = [
        Page(number=number, text="\n".join(lines), source=TextSource.EMBEDDED)
        for number, lines in enumerate(sheets, start=1)
    ]
    return invoice, pages
