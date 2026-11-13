
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import httpx

from app.core import config
from app.core.config import (
    DOCUMENTS_DIR,
    METADATA_DIR,
    INDEX_DIR,
)
from app.retrieval.indexer import build_document_index
from app.ingestion.pdf_processor import extract_pdf
from app.retrieval.search import search_document
from app.ingestion.chunker import create_chunks
from app.models.answer import AskRequest, AskResponse, Citation
from app.generation.answer_generator import generate_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def validate_document_uuid(document_id: str) -> str:
    """Validate that document_id is a valid UUID string to prevent path traversal."""
    try:
        return str(uuid.UUID(document_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID format. A valid UUID is required."
        )


@router.get("")
async def list_documents():
    documents = []
    for metadata_path in METADATA_DIR.glob("*.json"):
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict):
                continue

            document_id = metadata.get("document_id")
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id != metadata_path.stem
            ):
                continue

            pdf_path = DOCUMENTS_DIR / f"{document_id}.pdf"
            index_path = INDEX_DIR / f"{document_id}.faiss"
            if not pdf_path.is_file():
                continue

            documents.append(
                {
                    "document_id": document_id,
                    "filename": metadata.get("filename", pdf_path.name),
                    "page_count": metadata.get("page_count", 0),
                    "chunk_count": metadata.get("chunk_count", 0),
                    "indexed": index_path.is_file(),
                    "uploaded_at": datetime.fromtimestamp(
                        pdf_path.stat().st_mtime
                    ).astimezone().isoformat(),
                }
            )
        except (json.JSONDecodeError, OSError, TypeError):
            continue

    return {"documents": documents}


@router.get("/{document_id}/file")
async def get_document_file(document_id: str):
    valid_id = validate_document_uuid(document_id)

    pdf_path = DOCUMENTS_DIR / f"{valid_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document PDF not found",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Filename & extension validation
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    pdf_path = DOCUMENTS_DIR / f"{document_id}.pdf"
    metadata_path = METADATA_DIR / f"{document_id}.json"

    try:
        # 2. Upload size enforcement during stream/read
        max_bytes = config.MAX_UPLOAD_BYTES
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB."
            )

        # 3. Non-empty check
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 4. PDF magic-byte validation
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF file. Magic header missing.")

        pdf_path.write_bytes(content)

        # 5. Extract text & validate readability
        try:
            extracted = extract_pdf(str(pdf_path))
        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", document_id, e)
            raise HTTPException(status_code=400, detail="Corrupted or unreadable PDF file.")

        if extracted.get("page_count", 0) == 0:
            raise HTTPException(status_code=400, detail="PDF contains no readable pages.")

        chunks = create_chunks(extracted["pages"])

        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "page_count": extracted["page_count"],
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "message": "PDF processed successfully.",
            "document_id": document_id,
            "filename": file.filename,
            "page_count": extracted["page_count"],
            "chunk_count": len(chunks),
        }

    except HTTPException:
        pdf_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise

    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        logger.error("Unexpected error during PDF upload for %s: %s", document_id, e)
        raise HTTPException(status_code=500, detail="PDF processing failed.")

    finally:
        await file.close()


@router.post("/{document_id}/index")
async def index_document(document_id: str):
    valid_id = validate_document_uuid(document_id)

    pdf_path = DOCUMENTS_DIR / f"{valid_id}.pdf"
    metadata_path = METADATA_DIR / f"{valid_id}.json"

    if not pdf_path.exists() or not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        # Re-extract the original PDF.
        extracted = extract_pdf(str(pdf_path))

        # Generate improved structural chunks.
        chunks = create_chunks(extracted["pages"])

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        metadata["page_count"] = extracted["page_count"]
        metadata["chunk_count"] = len(chunks)
        metadata["chunks"] = chunks

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Rebuild FAISS using the new chunks.
        result = build_document_index(valid_id)

        return {
            "message": "Document reprocessed and indexed successfully.",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {str(error)}"
        )


@router.get("/{document_id}/search")
async def search_document_endpoint(
    document_id: str,
    q: str,
    top_k: int = 5
):
    valid_id = validate_document_uuid(document_id)

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20."
        )

    try:
        results = search_document(valid_id, q, top_k)

        return {
            "document_id": valid_id,
            "query": q,
            "result_count": len(results),
            "results": results
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document search failed."
        )


@router.post("/{document_id}/ask", response_model=AskResponse)
async def ask_document(document_id: str, request: AskRequest):
    valid_id = validate_document_uuid(document_id)

    metadata_path = METADATA_DIR / f"{valid_id}.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        retrieved_chunks = search_document(
            document_id=valid_id,
            query=request.question,
            top_k=request.top_k
        )

        if not retrieved_chunks:
            return AskResponse(
                document_id=valid_id,
                question=request.question,
                answer=(
                    "I couldn't find relevant information in this "
                    "document to answer your question."
                ),
                insufficient_evidence=True,
                citations=[]
            )

        generated = await generate_answer(
            question=request.question,
            chunks=retrieved_chunks
        )

        # Resolve citations against the actual retrieved evidence.
        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in retrieved_chunks
        }

        verified_citations = []
        seen_ids = set()

        for chunk_id in generated.citations:
            if chunk_id in seen_ids:
                continue

            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            seen_ids.add(chunk_id)

            verified_citations.append(
                Citation(
                    chunk_id=chunk["chunk_id"],
                    page_number=chunk["page_number"],
                    section=chunk.get("section", "General"),
                    text=chunk["text"],
                    bboxes=chunk.get("bboxes", [])
                )
            )

        insufficient = generated.insufficient_evidence
        answer = generated.answer

        # A substantive answer without a verified source is not accepted.
        if not insufficient and not verified_citations:
            insufficient = True
            answer = (
                "I couldn't verify a supporting passage in the document "
                "for this answer. Please try rephrasing your question."
            )

        return AskResponse(
            document_id=valid_id,
            question=request.question,
            answer=answer,
            insufficient_evidence=insufficient,
            citations=verified_citations
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document index not found. Please index the document first."
        )

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not communicate with the local LLM: {str(error)}"
        )

    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not process the generated answer: {str(error)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    valid_id = validate_document_uuid(document_id)

    document_files = {
        "original_pdf": DOCUMENTS_DIR / f"{valid_id}.pdf",
        "metadata": METADATA_DIR / f"{valid_id}.json",
        "faiss_index": INDEX_DIR / f"{valid_id}.faiss"
    }

    existing_files = [
        path for path in document_files.values()
        if path.is_file()
    ]

    if not existing_files:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    deleted_artifacts = []

    try:
        for artifact, path in document_files.items():
            if path.is_file():
                path.unlink()
                deleted_artifacts.append(artifact)

    except OSError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document deletion was incomplete. "
                "Some files may still exist."
            )
        )

    return {
        "message": "Document and associated files deleted successfully.",
        "document_id": valid_id,
        "deleted_artifacts": deleted_artifacts
    }

