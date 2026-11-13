"""
Security, validation, and error-handling tests for CivicLens backend API.

Covers:
- Oversized PDF uploads (HTTP 413)
- Invalid PDF extensions and magic-byte checks (HTTP 400)
- Corrupted/unreadable PDFs (HTTP 400)
- Empty PDF uploads (HTTP 400)
- Malformed document IDs / path traversal protection (HTTP 400)
- Missing metadata / missing index handling (HTTP 404)
- Safe document deletion & artifact cleanup
"""

import io
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import config

client = TestClient(app)


def test_upload_invalid_extension():
    """Reject files without .pdf extension."""
    response = client.post(
        "/documents/upload",
        files={"file": ("malicious.exe", b"%PDF-1.4 fake content", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_empty_file():
    """Reject zero-byte PDF uploads."""
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_upload_invalid_magic_bytes():
    """Reject files renamed to .pdf that lack the %PDF- magic header."""
    response = client.post(
        "/documents/upload",
        files={"file": ("not_a_pdf.pdf", b"<html><body>Not a PDF</body></html>", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Magic header missing" in response.json()["detail"]


def test_upload_oversized_file(monkeypatch):
    """Enforce maximum upload size limit."""
    # Temporarily set upload limit to 100 bytes for testing
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)

    oversized_data = b"%PDF-" + b"A" * 150
    response = client.post(
        "/documents/upload",
        files={"file": ("oversized.pdf", oversized_data, "application/pdf")}
    )
    assert response.status_code == 413
    assert "exceeds maximum upload size" in response.json()["detail"]


def test_upload_corrupted_pdf(monkeypatch, tmp_path):
    """Handle corrupted PDFs that fail text extraction gracefully."""
    # Point storage to tmp_path
    monkeypatch.setattr(config, "DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(config, "METADATA_DIR", tmp_path / "metadata")
    (tmp_path / "documents").mkdir(parents=True)
    (tmp_path / "metadata").mkdir(parents=True)

    # Mock extract_pdf to raise an exception simulating PyMuPDF extraction failure
    def mock_extract_fail(path):
        raise ValueError("Corrupted PDF structure")

    monkeypatch.setattr("app.api.documents.extract_pdf", mock_extract_fail)

    fake_pdf = b"%PDF-1.4 header but broken content"
    response = client.post(
        "/documents/upload",
        files={"file": ("corrupt.pdf", fake_pdf, "application/pdf")}
    )
    assert response.status_code == 400
    assert "Corrupted or unreadable" in response.json()["detail"]


def test_malformed_document_id_validation():
    """Reject non-UUID document IDs to prevent path traversal across all endpoints."""
    bad_id = "malformed-id-12345"

    # GET file
    r1 = client.get(f"/documents/{bad_id}/file")
    assert r1.status_code == 400
    assert "Invalid document ID format" in r1.json()["detail"]

    # POST index
    r2 = client.post(f"/documents/{bad_id}/index")
    assert r2.status_code == 400
    assert "Invalid document ID format" in r2.json()["detail"]

    # GET search
    r3 = client.get(f"/documents/{bad_id}/search?q=test")
    assert r3.status_code == 400
    assert "Invalid document ID format" in r3.json()["detail"]

    # POST ask
    r4 = client.post(f"/documents/{bad_id}/ask", json={"question": "Test question?", "top_k": 5})
    assert r4.status_code == 400
    assert "Invalid document ID format" in r4.json()["detail"]

    # DELETE document
    r5 = client.delete(f"/documents/{bad_id}")
    assert r5.status_code == 400
    assert "Invalid document ID format" in r5.json()["detail"]


def test_missing_document_handling():
    """Return 404 cleanly when valid UUID document is not found."""
    random_id = str(uuid.uuid4())

    r_file = client.get(f"/documents/{random_id}/file")
    assert r_file.status_code == 404

    r_index = client.post(f"/documents/{random_id}/index")
    assert r_index.status_code == 404

    r_ask = client.post(f"/documents/{random_id}/ask", json={"question": "Test question?", "top_k": 5})
    assert r_ask.status_code == 404

    r_del = client.delete(f"/documents/{random_id}")
    assert r_del.status_code == 404


def test_safe_document_deletion(monkeypatch, tmp_path):
    """Ensure delete_document safely cleans up PDF, metadata, and FAISS index files."""
    doc_id = str(uuid.uuid4())

    docs_dir = tmp_path / "documents"
    meta_dir = tmp_path / "metadata"
    idx_dir = tmp_path / "indexes"
    docs_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    idx_dir.mkdir(parents=True)

    monkeypatch.setattr(config, "DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(config, "METADATA_DIR", meta_dir)
    monkeypatch.setattr(config, "INDEX_DIR", idx_dir)
    monkeypatch.setattr("app.api.documents.DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr("app.api.documents.METADATA_DIR", meta_dir)
    monkeypatch.setattr("app.api.documents.INDEX_DIR", idx_dir)

    pdf_file = docs_dir / f"{doc_id}.pdf"
    meta_file = meta_dir / f"{doc_id}.json"
    faiss_file = idx_dir / f"{doc_id}.faiss"

    pdf_file.write_bytes(b"%PDF-1.4 test")
    meta_file.write_text('{"document_id": "' + doc_id + '"}', encoding="utf-8")
    faiss_file.write_bytes(b"faiss index")

    assert pdf_file.exists()
    assert meta_file.exists()
    assert faiss_file.exists()

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["document_id"] == doc_id
    assert "original_pdf" in resp.json()["deleted_artifacts"]
    assert "metadata" in resp.json()["deleted_artifacts"]
    assert "faiss_index" in resp.json()["deleted_artifacts"]

    assert not pdf_file.exists()
    assert not meta_file.exists()
    assert not faiss_file.exists()
