"""Field-level precision and recall for invoice extraction.

A header field is one slot. A wrong value counts as both a false positive and
a false negative. Line items are aligned greedily by description, case-folded.
Unmatched gold line items are false negatives. Unmatched predicted line items
are false positives. Numeric money is compared at cent precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from document_ai.models import Invoice


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        if denominator == 0:
            return None
        return self.tp / denominator

    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        if denominator == 0:
            return None
        return self.tp / denominator

    def f1(self) -> float | None:
        precision = self.precision()
        recall = self.recall()
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def score_document(gold: Invoice, predicted: Invoice | None) -> tuple[dict[str, Counts], bool]:
    """Score one document. `predicted=None` marks every gold field as missed."""
    if predicted is None:
        return _missed(gold), False
    scores: dict[str, Counts] = {}
    matched = True
    pairs = [
        ("document_kind", gold.document_kind, predicted.document_kind),
        ("vendor_name", _text(gold.vendor_name), _text(predicted.vendor_name)),
        ("invoice_number", _text(gold.invoice_number), _text(predicted.invoice_number)),
        ("invoice_date", gold.invoice_date.isoformat(), predicted.invoice_date.isoformat()),
        ("currency", gold.currency.upper(), predicted.currency.upper()),
        ("subtotal", _money(gold.subtotal), _money(predicted.subtotal)),
        ("tax", _money(gold.tax), _money(predicted.tax)),
        ("total", _money(gold.total), _money(predicted.total)),
    ]
    for field, gold_value, pred_value in pairs:
        matched = _slot(scores, field, gold_value, pred_value) and matched
    gold_due = gold.due_date.isoformat() if gold.due_date else None
    pred_due = predicted.due_date.isoformat() if predicted.due_date else None
    matched = _slot(scores, "due_date", gold_due, pred_due) and matched
    matched = _score_lines(gold, predicted, scores) and matched
    return scores, matched


def merge_scores(parts: list[dict[str, Counts]]) -> dict[str, Counts]:
    merged: dict[str, Counts] = {}
    for part in parts:
        for field, counts in part.items():
            merged.setdefault(field, Counts()).add(counts)
    return dict(sorted(merged.items()))


def micro_counts(fields: dict[str, Counts]) -> Counts:
    total = Counts()
    for counts in fields.values():
        total.add(counts)
    return total


def _score_lines(gold: Invoice, predicted: Invoice, scores: dict[str, Counts]) -> bool:
    used: set[int] = set()
    matched = True
    for gold_item in gold.line_items:
        match_at = _find_line(predicted, gold_item.description, used)
        if match_at is None:
            matched = False
            for field in ("description", "quantity", "unit_price", "amount"):
                scores.setdefault(f"line_items.{field}", Counts()).fn += 1
            continue
        used.add(match_at)
        pred_item = predicted.line_items[match_at]
        matched = (
            _slot(
                scores,
                "line_items.description",
                _text(gold_item.description),
                _text(pred_item.description),
            )
            and matched
        )
        matched = (
            _slot(
                scores,
                "line_items.quantity",
                _qty(gold_item.quantity),
                _qty(pred_item.quantity),
            )
            and matched
        )
        matched = (
            _slot(
                scores,
                "line_items.unit_price",
                _money(gold_item.unit_price),
                _money(pred_item.unit_price),
            )
            and matched
        )
        matched = (
            _slot(
                scores,
                "line_items.amount",
                _money(gold_item.amount),
                _money(pred_item.amount),
            )
            and matched
        )
    for index, _item in enumerate(predicted.line_items):
        if index in used:
            continue
        matched = False
        for field in ("description", "quantity", "unit_price", "amount"):
            scores.setdefault(f"line_items.{field}", Counts()).fp += 1
    return matched


def _find_line(predicted: Invoice, description: str, used: set[int]) -> int | None:
    target = description.strip().casefold()
    for index, item in enumerate(predicted.line_items):
        if index in used:
            continue
        if item.description.strip().casefold() == target:
            return index
    return None


def _missed(gold: Invoice) -> dict[str, Counts]:
    scores, _matched = score_document(gold, gold)
    missed: dict[str, Counts] = {}
    for field, counts in scores.items():
        missed[field] = Counts(fn=counts.tp + counts.fn)
    return missed


def _slot(scores: dict[str, Counts], field: str, gold: str | None, predicted: str | None) -> bool:
    counts = scores.setdefault(field, Counts())
    if gold is None and predicted is None:
        return True
    if gold is None:
        counts.fp += 1
        return False
    if predicted is None:
        counts.fn += 1
        return False
    if gold == predicted:
        counts.tp += 1
        return True
    counts.fp += 1
    counts.fn += 1
    return False


def _text(value: str) -> str:
    return value.strip().casefold()


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _qty(value: Decimal) -> str:
    return format(value.normalize(), "f")
