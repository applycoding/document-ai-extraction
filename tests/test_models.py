from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from document_ai.models import Invoice, LineItem


def _invoice(**overrides: object) -> Invoice:
    item = LineItem(
        description="Grid notebook",
        quantity=Decimal("2"),
        unit_price=Decimal("6.25"),
        amount=Decimal("12.50"),
    )
    payload: dict[str, object] = {
        "document_kind": "invoice",
        "vendor_name": "Northwind Supplies",
        "invoice_number": "INV-1001",
        "invoice_date": date(2026, 1, 5),
        "due_date": date(2026, 2, 4),
        "currency": "USD",
        "subtotal": Decimal("12.50"),
        "tax": Decimal("1.00"),
        "total": Decimal("13.50"),
        "line_items": [item],
    }
    payload.update(overrides)
    return Invoice.model_validate(payload)


def test_invoice_accepts_consistent_totals() -> None:
    invoice = _invoice()
    assert invoice.total == Decimal("13.50")


def test_invoice_coerces_float_money() -> None:
    invoice = _invoice(tax=1.0, total=13.5, subtotal=12.5)
    assert invoice.tax == Decimal("1.0")


def test_invoice_rejects_bad_total() -> None:
    with pytest.raises(ValidationError):
        _invoice(total=Decimal("99.00"))


def test_invoice_rejects_bad_line_amount() -> None:
    item = LineItem(
        description="Desk lamp",
        quantity=Decimal("1"),
        unit_price=Decimal("18.00"),
        amount=Decimal("18.00"),
    )
    broken = item.model_copy(update={"amount": Decimal("1.00")})
    with pytest.raises(ValidationError):
        _invoice(
            line_items=[broken],
            subtotal=Decimal("1.00"),
            tax=Decimal("0.00"),
            total=Decimal("1.00"),
        )


def test_invoice_rejects_due_date_before_issue_date() -> None:
    with pytest.raises(ValidationError):
        _invoice(due_date=date(2026, 1, 1))


def test_invoice_rejects_lowercase_currency() -> None:
    with pytest.raises(ValidationError):
        _invoice(currency="usd")
