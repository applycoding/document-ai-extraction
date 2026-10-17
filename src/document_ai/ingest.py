"""Load a PDF or an image into page text.

PDFs use the embedded text layer. This demo does not rasterize scanned PDFs.
Images are sent to the configured OCR engine.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from document_ai.errors import IngestError, OCRUnavailable
from document_ai.models import TextSource
from document_ai.ocr.base import OCREngine

MAX_BYTES = 10 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    source: TextSource


def ingest_bytes(data: bytes, filename: str, ocr: OCREngine | None) -> list[Page]:
    if not data:
        raise IngestError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise IngestError("The file is larger than 10MB.")
    kind = _detect_kind(data, filename)
    if kind == "pdf":
        return _ingest_pdf(data)
    if kind == "image":
        return _ingest_image(data, ocr)
    raise IngestError(f"Unsupported file type for {filename!r}. Use a PDF or an image.")


def needs_ocr(data: bytes, filename: str) -> bool:
    return _detect_kind(data, filename) == "image"


def _detect_kind(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if data.startswith(b"%PDF") or suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES or _looks_like_image(data):
        return "image"
    return "unknown"


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith((b"GIF87a", b"GIF89a", b"BM")):
        return True
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return True
    return data.startswith(b"RIFF") and data[8:12] == b"WEBP"


def _ingest_pdf(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise IngestError("The PDF could not be read.") from exc
    if reader.is_encrypted:
        raise IngestError("Encrypted PDFs are not supported.")
    if not reader.pages:
        raise IngestError("The PDF has no pages.")
    pages: list[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not _has_text(text):
            raise IngestError(
                f"Page {index} has no text layer. This demo does not rasterize "
                "scanned PDFs. Export the page as an image and pass an OCR engine."
            )
        pages.append(Page(number=index, text=text, source=TextSource.EMBEDDED))
    return pages


def _ingest_image(data: bytes, ocr: OCREngine | None) -> list[Page]:
    if ocr is None:
        raise IngestError("An image requires an OCR engine.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError("The image could not be read.") from exc
    pages: list[Page] = []
    frame_count = int(getattr(image, "n_frames", 1))
    for index in range(frame_count):
        image.seek(index)
        frame = image.copy().convert("RGB")
        try:
            text = ocr.extract(frame)
        except OCRUnavailable:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise IngestError(f"OCR failed on frame {index + 1}: {exc}") from exc
        if not _has_text(text):
            raise IngestError(f"OCR returned no text for frame {index + 1}.")
        pages.append(Page(number=index + 1, text=text, source=TextSource.OCR))
    return pages


def _has_text(text: str) -> bool:
    return any(character.isalnum() for character in text)
