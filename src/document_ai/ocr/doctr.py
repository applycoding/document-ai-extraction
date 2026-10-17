"""docTR adapter. The package is optional and is not imported by tests."""

from __future__ import annotations

from PIL import Image

from document_ai.errors import OCRUnavailable

try:
    import numpy as np
    from doctr.models import ocr_predictor
except ImportError:  # optional extra: document-ai[ocr-doctr]
    np = None
    ocr_predictor = None


def render_doctr_export(exported: dict[str, object]) -> str:
    """Join a docTR `export()` tree into plain text. Exposed for unit tests."""
    pages_out: list[str] = []
    pages = exported.get("pages", [])
    if not isinstance(pages, list):
        return ""
    for page in pages:
        if not isinstance(page, dict):
            continue
        lines_out: list[str] = []
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            lines = block.get("lines", [])
            if not isinstance(lines, list):
                continue
            for line in lines:
                if not isinstance(line, dict):
                    continue
                words = line.get("words", [])
                if not isinstance(words, list):
                    continue
                tokens = [
                    str(word.get("value", ""))
                    for word in words
                    if isinstance(word, dict) and word.get("value")
                ]
                if tokens:
                    lines_out.append(" ".join(tokens))
        pages_out.append("\n".join(lines_out))
    return "\n\n".join(part for part in pages_out if part)


class DocTROCR:
    name = "doctr"

    def __init__(self) -> None:
        if ocr_predictor is None or np is None:
            raise OCRUnavailable(
                "python-doctr is not installed. Install the ocr-doctr extra. "
                "The first call downloads pretrained weights."
            )
        self._predictor: object | None = None

    def extract(self, image: Image.Image) -> str:
        if ocr_predictor is None or np is None:
            raise OCRUnavailable("python-doctr is not installed.")
        if self._predictor is None:
            self._predictor = ocr_predictor(pretrained=True)
        array = np.asarray(image.convert("RGB"))
        result = self._predictor([array])  # type: ignore[operator]
        rendered = result.render()
        if isinstance(rendered, str) and rendered.strip():
            return rendered
        exported = result.export()
        if isinstance(exported, dict):
            return render_doctr_export(exported)
        return ""
