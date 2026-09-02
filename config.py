"""
Central configuration for DocuRAG.

All tunable values live here and are overridable via environment variables
(loaded from a local, git-ignored `.env` file). Nothing sensitive is
hard-coded in source: `.env.example` documents every variable a deployer
might want to change, and `.env` itself is excluded from version control.

Even though DocuRAG runs fully locally (no API keys), this pattern is kept
deliberately so that swapping in a hosted LLM/embedding provider later only
means adding a variable here — no code changes elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. Safe no-op in
# environments (CI, Docker) where config is injected another way.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


# --- Paths -------------------------------------------------------------
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SUPPORTED_EXTENSIONS = {".pdf"}

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "metadata.json"
DOCUMENT_REGISTRY_PATH = VECTOR_STORE_DIR / "documents.json"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "docurag.log"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)

# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL_NAME = _env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Retrieval ------------------------------------------------------------
# DEFAULT_TOP_K: how many chunks to retrieve per question by default.
DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 4)
MAX_TOP_K = _env_int("MAX_TOP_K", 10)

# SIMILARITY_THRESHOLD: minimum cosine similarity (inner product on
# normalized vectors, range ~[-1, 1]) for a retrieved chunk to be treated
# as relevant. Chunks scoring below this are discarded before being shown
# to the LLM. This is a blunt, empirically-tuned heuristic, not a
# calibrated probability — see README "Limitations". Set to 0 (or leave
# unset with SIMILARITY_THRESHOLD_ENABLED=false) to disable filtering.
SIMILARITY_THRESHOLD_ENABLED = _env_str("SIMILARITY_THRESHOLD_ENABLED", "true").lower() == "true"
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.3) if SIMILARITY_THRESHOLD_ENABLED else None

# --- Generation ------------------------------------------------------------
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_REQUEST_TIMEOUT_SECONDS = _env_int("LLM_REQUEST_TIMEOUT_SECONDS", 120)

# --- Logging ------------------------------------------------------------
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")
LOG_TO_FILE = _env_str("LOG_TO_FILE", "true").lower() == "true"
