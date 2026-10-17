"""Pydantic schema for invoices, citations, and extraction results."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)


def _coerce_decimal(value: object) -> object:
    if isinstance(value, float):
        return str(value)
    return value


def _decimal_to_json(value: Decimal) -> str:
    return format(value, "f")


DecimalValue = Annotated[
    Decimal,
    BeforeValidator(_coerce_decimal),
    PlainSerializer(_decimal_to_json, return_type=str, when_used="json"),
]


class TextSource(StrEnum):
    """Where a page's text came from."""

    EMBEDDED = "embedded"
    OCR = "ocr"


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: DecimalValue = Field(gt=0)
    unit_price: DecimalValue = Field(ge=0)
    amount: DecimalValue = Field(ge=0)

    @model_validator(mode="after")
    def amount_matches_quantity(self) -> LineItem:
        expected = (self.quantity * self.unit_price).quantize(Decimal("0.01"))
        actual = self.amount.quantize(Decimal("0.01"))
        if abs(actual - expected) > Decimal("0.01"):
            raise ValueError("amount does not match quantity * unit_price")
        return self


class Invoice(BaseModel):
    """One invoice or a receipt that uses the same fields.

    Receipts use `document_kind="receipt"`. A receipt number goes in
    `invoice_number`. `due_date` may be null.
    """

    model_config = ConfigDict(extra="forbid")

    document_kind: Literal["invoice", "receipt"]
    vendor_name: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    subtotal: DecimalValue = Field(ge=0)
    tax: DecimalValue = Field(ge=0)
    total: DecimalValue = Field(ge=0)
    line_items: list[LineItem] = Field(min_length=1)

    @model_validator(mode="after")
    def totals_are_consistent(self) -> Invoice:
        rolled = sum((item.amount for item in self.line_items), Decimal("0"))
        subtotal = self.subtotal.quantize(Decimal("0.01"))
        if abs(rolled.quantize(Decimal("0.01")) - subtotal) > Decimal("0.01"):
            raise ValueError("subtotal does not match the sum of line item amounts")
        expected_total = (self.subtotal + self.tax).quantize(Decimal("0.01"))
        if abs(expected_total - self.total.quantize(Decimal("0.01"))) > Decimal("0.01"):
            raise ValueError("total does not match subtotal + tax")
        if self.due_date is not None and self.due_date < self.invoice_date:
            raise ValueError("due_date is earlier than invoice_date")
        return self


class Citation(BaseModel):
    """A field value tied back to a page excerpt."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class CitedExtraction(BaseModel):
    """JSON object the extractor is asked to return."""

    model_config = ConfigDict(extra="forbid")

    invoice: Invoice
    citations: list[Citation] = Field(min_length=1)


class ExtractionResult(BaseModel):
    invoice: Invoice
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1)
    attempts: int = Field(ge=1)
    page_count: int = Field(ge=1)
    text_sources: list[TextSource]


def required_citation_fields(invoice: Invoice) -> list[str]:
    """Field paths that must each have one citation."""
    fields = [
        "document_kind",
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "currency",
        "subtotal",
        "tax",
        "total",
    ]
    if invoice.due_date is not None:
        fields.append("due_date")
    for index, _item in enumerate(invoice.line_items):
        prefix = f"line_items[{index}]"
        fields.extend(
            [
                f"{prefix}.description",
                f"{prefix}.quantity",
                f"{prefix}.unit_price",
                f"{prefix}.amount",
            ]
        )
    return fields
