"""Ask an extractor for an invoice and retry when validation fails."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from document_ai.errors import DocumentAIError, ExtractionFailed
from document_ai.ingest import Page
from document_ai.llm.base import ChatMessage, LLMClient
from document_ai.models import (
    Citation,
    CitedExtraction,
    ExtractionResult,
    Invoice,
    required_citation_fields,
)

DEFAULT_MAX_RETRIES = 2

SYSTEM_PROMPT = """You extract one invoice or receipt from document text into JSON.
Return a single JSON object and nothing else. Do not use markdown fences.

Rules:
- Keys are invoice and citations, and no others.
- document_kind is "invoice" or "receipt".
- Money fields are decimal strings such as "12.50".
- Dates are YYYY-MM-DD.
- currency is a 3-letter ISO code.
- Each line item amount equals quantity times unit price, within 0.01.
- subtotal equals the sum of line amounts, within 0.01.
- total equals subtotal plus tax, within 0.01.
- due_date may be null. If it is null, do not cite it.
- Cite every other invoice field and every line-item field.
- field paths look like vendor_name or line_items[0].amount.
- page is the 1-based page number from the prompt.
- quote is a short excerpt copied from that page that contains the value.
- confidence is a number from 0 to 1.

JSON schema:
""" + json.dumps(CitedExtraction.model_json_schema())


def extract_invoice(
    llm: LLMClient,
    pages: list[Page],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ExtractionResult:
    if max_retries < 0:
        raise ExtractionFailed(["max_retries must be zero or greater"], 0)
    if not pages:
        raise ExtractionFailed(["no pages to extract"], 0)
    messages: list[ChatMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_pages(pages)},
    ]
    errors: list[str] = []
    attempts = 0
    budget = max_retries + 1
    while attempts < budget:
        attempts += 1
        raw = ""
        try:
            raw = llm.complete(messages)
            invoice, citations = _validate_payload(raw, pages)
        except (ValueError, DocumentAIError) as exc:
            errors.append(_format_exception(exc))
        else:
            return ExtractionResult(
                invoice=invoice,
                citations=citations,
                confidence=_overall_confidence(citations),
                attempts=attempts,
                page_count=len(pages),
                text_sources=[page.source for page in pages],
            )
        if attempts < budget:
            messages.append({"role": "assistant", "content": raw[:100_000]})
            messages.append({"role": "user", "content": _repair_prompt(errors[-1])})
    raise ExtractionFailed(errors, attempts)


def format_pages(pages: list[Page]) -> str:
    blocks = [f"--- page {page.number} ---\n{page.text}" for page in pages]
    return "\n".join(blocks)


def _validate_payload(raw: str, pages: list[Page]) -> tuple[Invoice, list[Citation]]:
    payload = CitedExtraction.model_validate(_load_json(raw))
    _check_citations(payload.invoice, payload.citations, pages)
    return payload.invoice, payload.citations


def _load_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("response JSON must be an object")
    return data


def _check_citations(invoice: Invoice, citations: list[Citation], pages: list[Page]) -> None:
    required = required_citation_fields(invoice)
    by_field: dict[str, Citation] = {}
    problems: list[str] = []
    for citation in citations:
        if citation.field in by_field:
            problems.append(f"duplicate citation for {citation.field}")
            continue
        by_field[citation.field] = citation
    page_by_number = {page.number: page for page in pages}
    for field in required:
        citation = by_field.get(field)
        if citation is None:
            problems.append(f"missing citation for {field}")
            continue
        page = page_by_number.get(citation.page)
        if page is None:
            problems.append(f"{field} cites page {citation.page}, which is not in the document")
            continue
        if _normalize(citation.quote) not in _normalize(page.text):
            problems.append(f"quote for {field} was not found on page {citation.page}")
    for field in sorted(set(by_field) - set(required)):
        problems.append(f"unexpected citation for {field}")
    if problems:
        raise ValueError("; ".join(problems))


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _overall_confidence(citations: list[Citation]) -> float:
    mean = sum(item.confidence for item in citations) / len(citations)
    rounded = round(mean, 4)
    if rounded < 0:
        return 0.0
    if rounded > 1:
        return 1.0
    return float(rounded)


def _repair_prompt(error: str) -> str:
    return (
        "The previous JSON failed validation. Fix every error and return the full "
        f"JSON object only.\nErrors: {error}"
    )


def _format_exception(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for item in exc.errors():
            location = ".".join(str(piece) for piece in item["loc"])
            parts.append(f"{location}: {item['msg']}")
        return "; ".join(parts)
    return str(exc)
