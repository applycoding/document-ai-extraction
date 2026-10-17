"""Minimal text-layer PDF writer for synthetic invoices.

The writer emits Helvetica text with one line per `Td` move. It does not
rasterize pages. Real scanned PDFs are out of scope; see the README.
"""

from __future__ import annotations


def build_text_pdf(pages: list[list[str]]) -> bytes:
    """Build a one-font PDF whose text layer is the given lines per page."""
    if not pages:
        raise ValueError("at least one page is required")
    font_id = 3 + len(pages) * 2
    kids: list[str] = []
    body: list[bytes] = []
    for index, lines in enumerate(pages):
        page_id = 3 + index * 2
        content_id = page_id + 1
        kids.append(f"{page_id} 0 R")
        stream = _page_stream(lines)
        body.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode("ascii")
        )
        header = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        body.append(header + stream + b"\nendstream")
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages_obj = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode("ascii")
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    return _assemble([catalog, pages_obj, *body, font])


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "54 750 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("0 -16 Td")
        commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def _escape(text: str) -> str:
    if not text.isascii():
        raise ValueError("synthetic PDF text must be ASCII")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _assemble(objects: list[bytes]) -> bytes:
    chunks = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(chunks))
        chunks.extend(f"{number} 0 obj\n".encode("ascii"))
        chunks.extend(obj)
        chunks.extend(b"\nendobj\n")
    xref_at = len(chunks)
    size = len(objects) + 1
    lines = ["xref", f"0 {size}", "0000000000 65535 f "]
    lines.extend(f"{offset:010d} 00000 n " for offset in offsets[1:])
    trailer = (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    return bytes(chunks + ("\n".join(lines) + "\n").encode("ascii") + trailer)
