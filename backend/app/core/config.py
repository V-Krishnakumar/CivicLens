import os
from pathlib import Path

# Base data directory configurable via environment variable
DATA_DIR = Path(os.getenv("CIVICLENS_DATA_DIR", "/app/data"))
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_DIR = DATA_DIR / "metadata"
INDEX_DIR = DATA_DIR / "indexes"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Maximum upload size in bytes (default: 25 MB)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

# LLM inference settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M")
