"""
Central configuration for DocuRAG.

All tunable values live here and are overridable via environment variables
(loaded from a local, git-ignored `.env` file). Nothing sensitive is
hard-coded in source: `.env.example` documents every variable a deployer
might want to change, and `.env` itself is excluded from version control.

DocuRAG runs fully locally and needs no API keys, but every tunable value is
still read from the environment so deployment-specific settings never require
a code change.
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw != "" else default


def _env_bool(name: str, default: bool) -> bool:
    # Only the literal string "true" (any case) enables a flag. Unlike
    # _env_int/_env_float, an empty value counts as False, not "unset".
    return os.getenv(name, "true" if default else "false").lower() == "true"


#  Paths -
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
SUPPORTED_EXTENSIONS = {".pdf"}

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "metadata.json"
DOCUMENT_REGISTRY_PATH = VECTOR_STORE_DIR / "documents.json"

LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "docurag.log"

#  Chunking 
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)

#  Embeddings 
EMBEDDING_MODEL_NAME = _env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

#  Retrieval 
# DEFAULT_TOP_K: how many chunks are ultimately handed to the LLM.
# With MAX_CHUNKS_PER_SOURCE=2 below, 6 slots leave room for at least 3
# distinct documents in a multi-document question, instead of one document
# being able to fill every slot. See docs/RETRIEVAL.md for the reasoning.
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
ENTITY_FANOUT_ENABLED = _env_bool("ENTITY_FANOUT_ENABLED", True)

# SIMILARITY_THRESHOLD: minimum cosine similarity (inner product on
# normalized vectors, range ~[-1, 1]) for a retrieved chunk to be treated
# as relevant. Chunks scoring below this are discarded before being shown
# to the LLM. This is a blunt, empirically-tuned heuristic, not a
# calibrated probability. Filtering is OFF by default; it only applies when
# SIMILARITY_THRESHOLD_ENABLED=true. SIMILARITY_THRESHOLD_VALUE is the raw
# number (the UI's "ignore weak passages" toggle uses it too), while
# SIMILARITY_THRESHOLD is that number or None depending on the env flag.
SIMILARITY_THRESHOLD_ENABLED = _env_bool("SIMILARITY_THRESHOLD_ENABLED", False)
SIMILARITY_THRESHOLD_VALUE = _env_float("SIMILARITY_THRESHOLD", 0.3)
SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD_VALUE if SIMILARITY_THRESHOLD_ENABLED else None

#  Generation 
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 4096)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_REQUEST_TIMEOUT_SECONDS = _env_int("LLM_REQUEST_TIMEOUT_SECONDS", 300)

#  Logging 
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")
LOG_TO_FILE = _env_bool("LOG_TO_FILE", True)
