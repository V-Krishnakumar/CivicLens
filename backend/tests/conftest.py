"""
Shared pytest setup for CivicLens tests.

Mocks heavy native dependencies (faiss, sentence_transformers, rank_bm25)
before any app module is imported. This allows unit tests to run without
installing GPU/native packages or downloading embedding models.
"""
import sys
from unittest.mock import MagicMock

# --- Mock native/heavy dependencies FIRST ---
_MOCK_MODULES = [
    "faiss",
    "sentence_transformers",
    "rank_bm25",
    "pymupdf",
    "fitz",
]
for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
