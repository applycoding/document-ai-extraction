"""Write fictional invoices with labels. No personal data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from document_ai.models import Invoice, LineItem
from document_ai.textpdf import build_text_pdf

VENDORS = [
    "Northwind Supplies",
    "Brightline Studio",
    "Harbor Paper Co",
    "Cobalt Fixtures",
    "Little Oak Pantry",
    "Metro Ribbon Goods",
    "Silver Finch Printing",
    "Cedar Circuit Parts",
]

CATALOG = [
    ("Kraft mailer", "4.50"),
    ("Binder clip box", "3.75"),
    ("Desk lamp", "18.00"),
    ("Grid notebook", "6.25"),
    ("Ink cartridge", "27.40"),
    ("Cotton tote", "9.00"),
    ("Shelf bracket", "12.25"),
    ("Label roll", "5.10"),
]

TAX_RATE = Decimal("0.08")


def render_invoice(index: int) -> tuple[Invoice, list[list[str]]]:
    """Return a deterministic invoice and its page lines. Index starts at 1."""
    if index < 1:
        raise ValueError("index starts at 1")
    document_kind, banner = _kind(index)
    vendor = VENDORS[(index - 1) % len(VENDORS)]
    item_count = 2 + ((index - 1) % 3)
    start = (index * 3) % len(CATALOG)
    quantities = [Decimal(1 + ((index + offset) % 4)) for offset in range(item_count)]
    line_items: list[LineItem] = []
    for offset, quantity in enumerate(quantities):
        description, price_text = CATALOG[(start + offset) % len(CATALOG)]
        unit_price = Decimal(price_text)
        amount = _cents(quantity * unit_price)
        line_items.append(
            LineItem(
                description=description,
                quantity=quantity,
                unit_price=unit_price,
                amount=amount,
            )
        )
    subtotal = _cents(sum((item.amount for item in line_items), Decimal("0")))
    tax = _cents(subtotal * TAX_RATE)
    total = _cents(subtotal + tax)
    issued = date(2026, 1, 5) + timedelta(days=(index - 1) * 17)
    due = issued + timedelta(days=30)
    invoice = Invoice(
        document_kind=document_kind,
        vendor_name=vendor,
        invoice_number=f"INV-{1000 + index}",
        invoice_date=issued,
        due_date=due,
        currency="USD",
        subtotal=subtotal,
        tax=tax,
        total=total,
        line_items=line_items,
    )
    header = [
        banner,
        "Not a real bill. No personal data.",
        f"Vendor: {invoice.vendor_name}",
        f"Invoice Number: {invoice.invoice_number}",
        f"Invoice Date: {invoice.invoice_date.isoformat()}",
        f"Due Date: {invoice.due_date.isoformat() if invoice.due_date else ''}",
        "Currency: USD",
        "Items:",
    ]
    for item in invoice.line_items:
        header.append(
            f"- {item.description} | qty {format(item.quantity, 'f')} | "
            f"unit {_money(item.unit_price)} | amount {_money(item.amount)}"
        )
    totals = [
        f"Subtotal: {_money(invoice.subtotal)}",
        f"Tax: {_money(invoice.tax)}",
        f"Total: {_money(invoice.total)}",
    ]
    if index % 4 == 0:
        return invoice, [header, totals]
    return invoice, [header + totals]


def write_dataset(out_dir: Path, count: int) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    for index in range(1, count + 1):
        invoice, pages = render_invoice(index)
        document_id = f"inv_{index:03d}"
        pdf_name = f"{document_id}.pdf"
        label_name = f"{document_id}.json"
        text_name = f"{document_id}.txt"
        (out_dir / pdf_name).write_bytes(build_text_pdf(pages))
        label = invoice.model_dump_json(indent=2) + "\n"
        (out_dir / label_name).write_text(label, encoding="utf-8")
        (out_dir / text_name).write_text(_page_dump(pages), encoding="utf-8")
        documents.append(
            {"id": document_id, "pdf": pdf_name, "label": label_name, "text": text_name}
        )
    manifest = {
        "description": (
            "Synthetic invoices for evaluation. Fictional vendors and amounts. "
            "No personal data."
        ),
        "documents": documents,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write synthetic labeled invoices")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        write_dataset(args.out, args.count)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {args.count} synthetic invoices to {args.out}")
    return 0


def _kind(index: int) -> tuple[Literal["invoice", "receipt"], str]:
    if index % 5 == 0:
        return "receipt", "SYNTHETIC RECEIPT"
    return "invoice", "SYNTHETIC INVOICE"


def _page_dump(pages: list[list[str]]) -> str:
    chunks: list[str] = []
    for number, lines in enumerate(pages, start=1):
        chunks.append(f"--- page {number} ---")
        chunks.extend(lines)
    return "\n".join(chunks) + "\n"


def _cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _money(value: Decimal) -> str:
    return format(_cents(value), "f")


if __name__ == "__main__":
    raise SystemExit(main())
