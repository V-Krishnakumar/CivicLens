
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import METADATA_DIR, INDEX_DIR

INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load once and reuse while the API process is running.
_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)

    return _embedding_model


def build_document_index(document_id: str) -> dict:
    metadata_path = METADATA_DIR / f"{document_id}.json"

    if not metadata_path.exists():
        raise FileNotFoundError("Document metadata not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata["chunks"]

    if not chunks:
        raise ValueError("No text chunks found in this document.")

    texts = [chunk["text"] for chunk in chunks]

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = INDEX_DIR / f"{document_id}.faiss"
    faiss.write_index(index, str(index_path))

    return {
        "document_id": document_id,
        "indexed_chunks": index.ntotal,
        "embedding_dimensions": embeddings.shape[1]
    }
