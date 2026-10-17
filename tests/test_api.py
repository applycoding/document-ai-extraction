import io

from fastapi.testclient import TestClient
from PIL import Image
from tests.helpers import embedded_pages

from document_ai.api import create_app
from document_ai.dataset.generate import render_invoice
from document_ai.pipeline import build_pipeline
from document_ai.textpdf import build_text_pdf


def test_health() -> None:
    client = TestClient(create_app(build_pipeline("fake", "parser")))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_pdf() -> None:
    invoice, sheets = render_invoice(1)
    client = TestClient(create_app(build_pipeline("fake", "parser")))
    response = client.post(
        "/v1/extract",
        files={"file": ("inv.pdf", build_text_pdf(sheets), "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["invoice"]["vendor_name"] == invoice.vendor_name
    assert body["invoice"]["total"] == format(invoice.total, "f")
    assert body["page_count"] == 1
    assert body["citations"]


def test_extract_image() -> None:
    invoice, pages = embedded_pages(3)
    image = Image.new("RGB", (32, 32), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pipeline = build_pipeline("fake", "parser", fake_text=pages[0].text)
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/v1/extract",
        files={"file": ("page.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["invoice"]["invoice_number"] == invoice.invoice_number


def test_empty_upload_is_400() -> None:
    client = TestClient(create_app(build_pipeline("fake", "parser")))
    response = client.post("/v1/extract", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_invalid_model_output_is_422() -> None:
    _invoice, sheets = render_invoice(1)
    pipeline = build_pipeline("fake", "fake", scripted=["{}"], max_retries=0)
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/v1/extract",
        files={"file": ("inv.pdf", build_text_pdf(sheets), "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["attempts"] == 1
