"""HTTP API for one-file extraction."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile

from document_ai import __version__
from document_ai.errors import ExtractionFailed, IngestError, OCRUnavailable
from document_ai.models import ExtractionResult
from document_ai.pipeline import Pipeline, pipeline_from_env


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="Document AI Extraction",
        version=__version__,
        summary="Extract a validated invoice from a PDF or image.",
    )
    app.state.pipeline = pipeline or pipeline_from_env()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/extract", response_model=ExtractionResult)
    async def extract(file: UploadFile) -> ExtractionResult:
        data = await file.read()
        try:
            return app.state.pipeline.run_bytes(data, file.filename or "upload")
        except (IngestError, OCRUnavailable) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ExtractionFailed as exc:
            raise HTTPException(
                status_code=422,
                detail={"message": str(exc), "errors": exc.errors, "attempts": exc.attempts},
            ) from exc

    return app


app = create_app()
