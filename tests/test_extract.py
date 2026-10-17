import json

import pytest

from document_ai.errors import ExtractionFailed
from document_ai.extract import extract_invoice, format_pages
from document_ai.llm.fake import FakeLLM
from document_ai.llm.parser import ParserLLM
from document_ai.models import required_citation_fields
from tests.helpers import embedded_pages


def _good_json(index: int = 1) -> str:
    _invoice, pages = embedded_pages(index)
    return ParserLLM().complete([{"role": "user", "content": format_pages(pages)}])


def test_parser_extracts_generated_invoice() -> None:
    invoice, pages = embedded_pages(2)
    result = extract_invoice(ParserLLM(), pages, max_retries=0)
    assert result.invoice == invoice
    assert result.attempts == 1
    assert 0 <= result.confidence <= 1
    assert [item.field for item in result.citations] == required_citation_fields(invoice)


def test_split_invoice_cites_totals_on_page_two() -> None:
    invoice, pages = embedded_pages(4)
    assert len(pages) == 2
    result = extract_invoice(ParserLLM(), pages, max_retries=0)
    assert result.invoice == invoice
    totals = {item.field: item.page for item in result.citations}
    assert totals["vendor_name"] == 1
    assert totals["total"] == 2
    assert totals["tax"] == 2


def test_receipt_kind_is_read_from_banner() -> None:
    invoice, pages = embedded_pages(5)
    result = extract_invoice(ParserLLM(), pages, max_retries=0)
    assert invoice.document_kind == "receipt"
    assert result.invoice.document_kind == "receipt"


def test_markdown_fence_is_accepted() -> None:
    _invoice, pages = embedded_pages(1)
    fenced = "```json\n" + _good_json(1) + "\n```"
    result = extract_invoice(FakeLLM([fenced]), pages, max_retries=0)
    assert result.attempts == 1
    assert result.invoice.invoice_number == "INV-1001"


def test_invalid_json_is_repaired() -> None:
    invoice, pages = embedded_pages(1)
    fake = FakeLLM(["not json", _good_json(1)])
    result = extract_invoice(fake, pages, max_retries=2)
    assert result.attempts == 2
    assert fake.calls == 2
    assert result.invoice == invoice


def test_bad_citation_quote_is_repaired() -> None:
    _invoice, pages = embedded_pages(1)
    payload = json.loads(_good_json(1))
    payload["citations"][0]["quote"] = "this quote is not on the page"
    fake = FakeLLM([json.dumps(payload), _good_json(1)])
    result = extract_invoice(fake, pages)
    assert result.attempts == 2


def test_retries_exhausted() -> None:
    _invoice, pages = embedded_pages(1)
    fake = FakeLLM(["nope", "still nope"])
    with pytest.raises(ExtractionFailed) as raised:
        extract_invoice(fake, pages, max_retries=1)
    assert raised.value.attempts == 2
    assert len(raised.value.errors) == 2
