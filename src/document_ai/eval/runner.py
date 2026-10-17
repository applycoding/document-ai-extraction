"""Score a labeled dataset with a pipeline and render the measured report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from document_ai.eval.metrics import Counts, merge_scores, micro_counts, score_document
from document_ai.models import Invoice
from document_ai.pipeline import Pipeline, build_pipeline


@dataclass
class DocumentError:
    document_id: str
    message: str


@dataclass
class EvalReport:
    documents: int
    exact_match: float | None
    fields: dict[str, Counts]
    micro: Counts
    errors: list[DocumentError]

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": self.documents,
            "exact_match": self.exact_match,
            "fields": {name: _counts_dict(counts) for name, counts in self.fields.items()},
            "micro": _counts_dict(self.micro),
            "errors": [
                {"document_id": error.document_id, "message": error.message}
                for error in self.errors
            ],
        }


def evaluate_dataset(dataset_dir: Path, pipeline: Pipeline | None = None) -> EvalReport:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {dataset_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    active = pipeline or build_pipeline("tesseract", "parser")
    parts: list[dict[str, Counts]] = []
    errors: list[DocumentError] = []
    exact = 0
    for document in documents:
        document_id = str(document["id"])
        gold = Invoice.model_validate_json((dataset_dir / document["label"]).read_text())
        try:
            predicted = active.run_path(dataset_dir / document["pdf"]).invoice
        except Exception as exc:
            errors.append(DocumentError(document_id, str(exc)))
            parts.append(score_document(gold, None)[0])
            continue
        scores, matched = score_document(gold, predicted)
        parts.append(scores)
        if matched:
            exact += 1
    fields = merge_scores(parts)
    document_count = len(documents)
    rate = exact / document_count if document_count else None
    return EvalReport(
        documents=document_count,
        exact_match=rate,
        fields=fields,
        micro=micro_counts(fields),
        errors=errors,
    )


def render_report(report: EvalReport) -> str:
    lines = [
        f"documents: {report.documents}",
        f"exact_match: {_format_ratio(report.exact_match)}",
        f"{'field':<28} {'tp':>5} {'fp':>5} {'fn':>5} "
        f"{'precision':>10} {'recall':>10} {'f1':>10}",
    ]
    for name, counts in report.fields.items():
        lines.append(_row(name, counts))
    lines.append(_row("micro", report.micro))
    if report.errors:
        lines.append("errors:")
        for error in report.errors:
            lines.append(f"- {error.document_id}: {error.message}")
    return "\n".join(lines) + "\n"


def _row(name: str, counts: Counts) -> str:
    return (
        f"{name:<28} {counts.tp:5d} {counts.fp:5d} {counts.fn:5d} "
        f"{_format_ratio(counts.precision()):>10} {_format_ratio(counts.recall()):>10} "
        f"{_format_ratio(counts.f1()):>10}"
    )


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _counts_dict(counts: Counts) -> dict[str, object]:
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": counts.precision(),
        "recall": counts.recall(),
        "f1": counts.f1(),
    }
