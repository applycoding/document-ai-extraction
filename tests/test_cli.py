import json
from pathlib import Path

from pytest import CaptureFixture

from document_ai.cli import main
from document_ai.dataset.generate import render_invoice, write_dataset
from document_ai.eval.runner import evaluate_dataset
from document_ai.pipeline import build_pipeline
from document_ai.textpdf import build_text_pdf

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "data" / "synthetic"


def test_cli_extract(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    invoice, sheets = render_invoice(1)
    path = tmp_path / "inv.pdf"
    path.write_bytes(build_text_pdf(sheets))
    code = main(["extract", str(path), "--llm", "parser", "--ocr", "fake"])
    assert code == 0
    captured = capsys.readouterr()
    body = json.loads(captured.out)
    assert body["invoice"]["invoice_number"] == invoice.invoice_number


def test_cli_generate_and_eval(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    dataset = tmp_path / "set"
    assert main(["generate-dataset", "--out", str(dataset), "--count", "8"]) == 0
    code = main(["eval", "--dataset", str(dataset), "--llm", "parser", "--ocr", "fake"])
    assert code == 0
    captured = capsys.readouterr()
    assert "precision" in captured.out
    assert "micro" in captured.out
    report = evaluate_dataset(dataset, build_pipeline("fake", "parser"))
    assert report.documents == 8
    assert report.errors == []
    assert report.micro.fp == 0
    assert report.micro.fn == 0
    assert report.micro.tp > 0


def test_committed_dataset_matches_generator(tmp_path: Path) -> None:
    write_dataset(tmp_path, count=8)
    committed = json.loads((COMMITTED / "manifest.json").read_text(encoding="utf-8"))
    generated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert generated["documents"] == committed["documents"]
    for document in committed["documents"]:
        name = document["pdf"]
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes()
        label = document["label"]
        assert (tmp_path / label).read_text(encoding="utf-8") == (
            COMMITTED / label
        ).read_text(encoding="utf-8")


def test_cli_missing_file(tmp_path: Path) -> None:
    code = main(["extract", str(tmp_path / "missing.pdf")])
    assert code == 2
