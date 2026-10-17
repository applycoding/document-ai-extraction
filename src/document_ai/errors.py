"""Expected failures raised by the pipeline."""


class DocumentAIError(Exception):
    """Base error for failures a caller can show to a user."""


class IngestError(DocumentAIError):
    """The upload could not be read as a PDF or an image."""


class OCRUnavailable(DocumentAIError):
    """The selected OCR backend is not installed or has no binary."""


class ExtractionFailed(DocumentAIError):
    """Model output did not pass validation within the retry budget."""

    def __init__(self, errors: list[str], attempts: int) -> None:
        self.errors = errors
        self.attempts = attempts
        summary = errors[-1] if errors else "unknown error"
        super().__init__(f"extraction failed after {attempts} attempt(s): {summary}")
