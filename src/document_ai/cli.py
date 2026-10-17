"""Command line for extraction, evaluation, and dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from document_ai.dataset.generate import write_dataset
from document_ai.errors import DocumentAIError
from document_ai.eval.runner import evaluate_dataset, render_report
from document_ai.pipeline import build_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="document-ai",
        description="Extract invoices from PDFs or images and score a labeled set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract one document to JSON")
    extract.add_argument("path", type=Path)
    extract.add_argument("--ocr", default="tesseract", choices=["tesseract", "doctr", "fake"])
    extract.add_argument("--llm", default="parser", choices=["parser", "openai"])
    extract.add_argument("--max-retries", type=int, default=2)
    extract.add_argument("--out", default="-", help="Output path, or - for stdout")

    evaluate = sub.add_parser("eval", help="Score a synthetic dataset and print the measurements")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/synthetic"))
    evaluate.add_argument("--llm", default="parser", choices=["parser", "openai"])
    evaluate.add_argument("--ocr", default="tesseract", choices=["tesseract", "doctr", "fake"])
    evaluate.add_argument("--out", type=Path, default=None, help="Optional JSON report path")

    generate = sub.add_parser("generate-dataset", help="Write fictional labeled invoices")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--count", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "extract":
        return _extract(args)
    if args.command == "eval":
        return _evaluate(args)
    if args.command == "generate-dataset":
        return _generate(args)
    print(f"unknown command {args.command}", file=sys.stderr)
    return 2


def _extract(args: argparse.Namespace) -> int:
    if not args.path.is_file():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 2
    pipeline = build_pipeline(args.ocr, args.llm, max_retries=args.max_retries)
    try:
        result = pipeline.run_path(args.path)
    except DocumentAIError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = result.model_dump_json(indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(args.ocr, args.llm)
    try:
        report = evaluate_dataset(args.dataset, pipeline)
    except (FileNotFoundError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"could not read dataset: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render_report(report))
    if args.out is not None:
        args.out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    if report.errors:
        return 1
    return 0


def _generate(args: argparse.Namespace) -> int:
    try:
        write_dataset(args.out, args.count)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {args.count} synthetic invoices to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
