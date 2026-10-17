"""Offline parser for the synthetic invoice template.

This is a baseline extractor so demos and evaluation run without an API key.
It is not a language model. Citation confidences are fixed heuristics, not
calibrated probabilities.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from document_ai.llm.base import ChatMessage
from document_ai.models import Citation, CitedExtraction, Invoice, LineItem

PAGE_BLOCK = re.compile(r"--- page (\d+) ---\n(.*?)(?=\n--- page \d+ ---|\Z)", re.DOTALL)

HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "vendor_name": re.compile(r"^\s*Vendor:\s*(.+?)\s*$", re.MULTILINE),
    "invoice_number": re.compile(r"^\s*Invoice Number:\s*(.+?)\s*$", re.MULTILINE),
    "invoice_date": re.compile(r"^\s*Invoice Date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
    "due_date": re.compile(r"^\s*Due Date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE),
    "currency": re.compile(r"^\s*Currency:\s*([A-Za-z]{3})\s*$", re.MULTILINE),
    "subtotal": re.compile(r"^\s*Subtotal:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
    "tax": re.compile(r"^\s*Tax:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
    "total": re.compile(r"^\s*Total:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
}

LINE_RE = re.compile(
    r"^\s*-\s*(.+?)\s*\|\s*qty\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*unit\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*\|\s*amount\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)

HEADER_CONFIDENCE = {
    "document_kind": 0.9,
    "vendor_name": 0.95,
    "invoice_number": 0.99,
    "invoice_date": 0.98,
    "due_date": 0.98,
    "currency": 0.99,
    "subtotal": 0.99,
    "tax": 0.97,
    "total": 0.99,
}

LINE_CONFIDENCE = {
    "description": 0.96,
    "quantity": 0.99,
    "unit_price": 0.99,
    "amount": 0.99,
}


class ParserLLM:
    """Read the synthetic template out of the prompt and return schema JSON."""

    def complete(self, messages: list[ChatMessage]) -> str:
        pages = pages_from_messages(messages)
        extraction = parse_pages(pages)
        return extraction.model_dump_json()


def pages_from_messages(messages: list[ChatMessage]) -> list[tuple[int, str]]:
    for message in messages:
        if message["role"] != "user":
            continue
        found = [(int(number), text) for number, text in PAGE_BLOCK.findall(message["content"])]
        if found:
            return found
    for message in messages:
        if message["role"] == "user" and "Vendor:" in message["content"]:
            return [(1, message["content"])]
    raise ValueError("no invoice text was found in the prompt")


def parse_pages(pages: list[tuple[int, str]]) -> CitedExtraction:
    kind_page, kind = _document_kind(pages)
    vendor = _require(pages, "vendor_name")
    number = _require(pages, "invoice_number")
    issued = _require(pages, "invoice_date")
    currency = _require(pages, "currency")
    subtotal = _require(pages, "subtotal")
    tax = _require(pages, "tax")
    total = _require(pages, "total")
    due = _search(pages, HEADER_PATTERNS["due_date"])
    line_matches = _line_items(pages)

    items: list[LineItem] = []
    citations: list[Citation] = [
        _cite("document_kind", kind_page, kind[1], HEADER_CONFIDENCE["document_kind"]),
        _cite("vendor_name", vendor[0], vendor[1].group(0), HEADER_CONFIDENCE["vendor_name"]),
        _cite(
            "invoice_number",
            number[0],
            number[1].group(0),
            HEADER_CONFIDENCE["invoice_number"],
        ),
        _cite("invoice_date", issued[0], issued[1].group(0), HEADER_CONFIDENCE["invoice_date"]),
        _cite(
            "currency",
            currency[0],
            currency[1].group(0),
            HEADER_CONFIDENCE["currency"],
        ),
        _cite("subtotal", subtotal[0], subtotal[1].group(0), HEADER_CONFIDENCE["subtotal"]),
        _cite("tax", tax[0], tax[1].group(0), HEADER_CONFIDENCE["tax"]),
        _cite("total", total[0], total[1].group(0), HEADER_CONFIDENCE["total"]),
    ]
    due_date = None
    if due is not None:
        due_date = due[1].group(1)
        citations.append(_cite("due_date", due[0], due[1].group(0), HEADER_CONFIDENCE["due_date"]))

    for index, (page, match) in enumerate(line_matches):
        description, quantity, unit_price, amount = match.groups()
        items.append(
            LineItem(
                description=description.strip(),
                quantity=Decimal(quantity),
                unit_price=Decimal(unit_price),
                amount=Decimal(amount),
            )
        )
        prefix = f"line_items[{index}]"
        for name, confidence in LINE_CONFIDENCE.items():
            citations.append(_cite(f"{prefix}.{name}", page, match.group(0), confidence))

    invoice = Invoice(
        document_kind=kind[0],
        vendor_name=vendor[1].group(1).strip(),
        invoice_number=number[1].group(1).strip(),
        invoice_date=issued[1].group(1),
        due_date=due_date,
        currency=currency[1].group(1).upper(),
        subtotal=Decimal(subtotal[1].group(1)),
        tax=Decimal(tax[1].group(1)),
        total=Decimal(total[1].group(1)),
        line_items=items,
    )
    return CitedExtraction(invoice=invoice, citations=citations)


def _document_kind(
    pages: list[tuple[int, str]],
) -> tuple[int, tuple[Literal["invoice", "receipt"], str]]:
    for number, text in pages:
        receipt = re.search(r"^\s*SYNTHETIC RECEIPT\s*$", text, re.MULTILINE)
        if receipt:
            return number, ("receipt", receipt.group(0))
        invoice = re.search(r"^\s*SYNTHETIC INVOICE\s*$", text, re.MULTILINE)
        if invoice:
            return number, ("invoice", invoice.group(0))
    raise ValueError("missing SYNTHETIC INVOICE or SYNTHETIC RECEIPT banner")


def _require(pages: list[tuple[int, str]], field: str) -> tuple[int, re.Match[str]]:
    found = _search(pages, HEADER_PATTERNS[field])
    if found is None:
        raise ValueError(f"missing {field}")
    return found


def _search(
    pages: list[tuple[int, str]], pattern: re.Pattern[str]
) -> tuple[int, re.Match[str]] | None:
    for number, text in pages:
        match = pattern.search(text)
        if match:
            return number, match
    return None


def _line_items(pages: list[tuple[int, str]]) -> list[tuple[int, re.Match[str]]]:
    found: list[tuple[int, re.Match[str]]] = []
    for number, text in pages:
        found.extend((number, match) for match in LINE_RE.finditer(text))
    if not found:
        raise ValueError("no line items found")
    return found


def _cite(field: str, page: int, quote: str, confidence: float) -> Citation:
    cleaned = quote.strip()
    if not cleaned:
        raise ValueError(f"empty quote for {field}")
    return Citation(field=field, page=page, quote=cleaned, confidence=confidence)
