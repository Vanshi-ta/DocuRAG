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
# DEFAULT_TOP_K: how many chunks are ultimately handed to the LLM.
# Raised from 4 -> 6 in this phase: with MAX_CHUNKS_PER_SOURCE=2 below, 6
# slots guarantee room for at least 3 distinct documents' chunks in a
# multi-document question, instead of one document being able to fill
# every slot. See docs/RETRIEVAL.md for the reasoning and before/after
# test results.
DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 6)
MAX_TOP_K = _env_int("MAX_TOP_K", 10)

# CANDIDATE_POOL_SIZE: how many candidates FAISS returns BEFORE diversity
# selection narrows them down to DEFAULT_TOP_K. Must be >= DEFAULT_TOP_K.
# A larger pool gives the diversity step more to choose from (so a second
# or third document's best chunk can be pulled in even if it wasn't in the
# raw top-6), at the cost of a few extra vector comparisons — negligible
# for a flat index at this project's scale.
CANDIDATE_POOL_SIZE = _env_int("CANDIDATE_POOL_SIZE", 15)

# MAX_CHUNKS_PER_SOURCE: hard cap on how many chunks from any single
# source_filename can appear in the final selection. This is the direct
# fix for one document dominating top-k on multi-document questions.
MAX_CHUNKS_PER_SOURCE = _env_int("MAX_CHUNKS_PER_SOURCE", 2)

# ENTITY_FANOUT_ENABLED: when a question mentions 2+ capitalized
# entities (e.g. "Vanshita" and "Manas"), run one sub-retrieval per
# entity and merge results, instead of a single query embedding that can
# semantically drift toward whichever entity's wording is closer to the
# rest of the question. See src/retrieval/query_processing.py.
ENTITY_FANOUT_ENABLED = _env_str("ENTITY_FANOUT_ENABLED", "true").lower() == "true"

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
