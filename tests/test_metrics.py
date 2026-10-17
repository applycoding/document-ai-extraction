from datetime import date
from decimal import Decimal

from document_ai.eval.metrics import score_document
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


def test_exact_document_scores_true_positives_only() -> None:
    gold = _invoice()
    scores, exact = score_document(gold, gold)
    assert exact is True
    assert scores["vendor_name"].tp == 1
    assert scores["vendor_name"].fp == 0
    assert scores["vendor_name"].fn == 0
    assert scores["total"].precision() == 1
    assert scores["line_items.amount"].recall() == 1


def test_wrong_vendor_is_false_positive_and_false_negative() -> None:
    gold = _invoice()
    predicted = _invoice(vendor_name="Other Vendor")
    scores, exact = score_document(gold, predicted)
    assert exact is False
    assert scores["vendor_name"].tp == 0
    assert scores["vendor_name"].fp == 1
    assert scores["vendor_name"].fn == 1
    assert scores["total"].tp == 1


def test_line_items_align_by_description() -> None:
    gold = _invoice()
    extra = LineItem(
        description="Desk lamp",
        quantity=Decimal("1"),
        unit_price=Decimal("18.00"),
        amount=Decimal("18.00"),
    )
    swapped = _invoice(
        line_items=[extra, gold.line_items[0]],
        subtotal=Decimal("30.50"),
        tax=Decimal("1.00"),
        total=Decimal("31.50"),
    )
    scores, exact = score_document(gold, swapped)
    assert exact is False
    assert scores["line_items.description"].tp == 1
    assert scores["line_items.description"].fp == 1
    assert scores["line_items.description"].fn == 0
    assert scores["line_items.amount"].tp == 1


def test_missing_prediction_counts_false_negatives() -> None:
    scores, exact = score_document(_invoice(), None)
    assert exact is False
    assert scores["invoice_number"].fn == 1
    assert scores["invoice_number"].tp == 0
    assert scores["line_items.quantity"].fn == 1


def test_quantity_two_matches_two_point_zero() -> None:
    gold = _invoice()
    predicted = _invoice(
        line_items=[
            LineItem(
                description="Grid notebook",
                quantity=Decimal("2.0"),
                unit_price=Decimal("6.250"),
                amount=Decimal("12.5"),
            )
        ]
    )
    scores, exact = score_document(gold, predicted)
    assert exact is True
    assert scores["line_items.quantity"].tp == 1
    assert scores["subtotal"].tp == 1
