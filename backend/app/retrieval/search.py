
import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import METADATA_DIR, INDEX_DIR
from app.retrieval.indexer import get_embedding_model


def reciprocal_rank_fusion(
    dense_results: list,
    keyword_results: list,
    k: int = 60
) -> list:
    scores = {}

    for ranking in (dense_results, keyword_results):
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0) + (
                1 / (k + rank)
            )

    return sorted(scores, key=scores.get, reverse=True)


def search_document(
    document_id: str,
    query: str,
    top_k: int = 5
) -> list:
    metadata_path = METADATA_DIR / f"{document_id}.json"
    index_path = INDEX_DIR / f"{document_id}.faiss"

    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError("Document index not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata["chunks"]

    if not chunks:
        return []

    index = faiss.read_index(str(index_path))

    # Dense retrieval
    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    search_count = min(top_k * 3, len(chunks))

    _, dense_indices = index.search(query_embedding, search_count)
    dense_results = [
        int(i) for i in dense_indices[0] if i >= 0
    ]

    # Keyword retrieval
    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    keyword_scores = bm25.get_scores(query.lower().split())

    keyword_results = [
        int(i)
        for i in np.argsort(keyword_scores)[::-1][:search_count]
        if keyword_scores[i] > 0
    ]

    # Combine rankings
    fused_results = reciprocal_rank_fusion(
        dense_results,
        keyword_results
    )

    # Deduplicate retrieved chunks that share heavy overlap (e.g. >= 70% character overlap)
    selected_chunks = []
    seen_texts = []

    for index in fused_results:
        chunk = chunks[index]
        chunk_text = chunk["text"]
        
        # Check for near-duplicate or heavy containment with already selected chunks
        is_duplicate = False
        for selected in selected_chunks:
            # If one is almost entirely contained inside another
            if chunk_text in selected["text"] or selected["text"] in chunk_text:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_chunks.append({
                **chunk,
                "retrieval_score": round(
                    1 / (60 + fused_results.index(index) + 1),
                    6
                )
            })

        if len(selected_chunks) >= top_k:
            break

    return selected_chunks
