"""
Central configuration for DocuRAG.

Keeping paths and constants here (instead of scattered across files) means
every module — ingestion now, chunking/embedding/retrieval later — reads
from one source of truth.
"""

from pathlib import Path

# Project root = the folder this file lives in.
BASE_DIR = Path(__file__).resolve().parent

# Where uploaded PDFs live. In Phase 2 you'll manually drop PDFs here;
# in later phases, Streamlit's file uploader will save uploads to this path.
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# File types the ingestion pipeline is allowed to process.
SUPPORTED_EXTENSIONS = {".pdf"}

# --- Chunking (Phase 3) ---------------------------------------------------
# CHUNK_SIZE:    max characters per chunk.
# CHUNK_OVERLAP: characters repeated between consecutive chunks, so a fact
#                sitting right on a chunk boundary still appears whole in
#                at least one chunk.
# See the Phase 3 guide for why these specific values were chosen for
# technical PDFs.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Embeddings & Vector Store (Phase 4) -----------------------------------
# EMBEDDING_MODEL_NAME: a local Sentence Transformers model — downloaded
#                        once from Hugging Face, then cached locally. No API
#                        key, no per-call cost, no internet needed after the
#                        first download. See Phase 4 guide Sections 6-7.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Where the persistent FAISS index and its parallel metadata file live.
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_STORE_PATH = VECTOR_STORE_DIR / "metadata.json"

# --- Retrieval (Phase 5) ---------------------------------------------------
# DEFAULT_TOP_K: how many chunks to retrieve per question by default.
# See Phase 5 guide Section 4 for the reasoning behind this value.
DEFAULT_TOP_K = 4

# --- Generation (Phase 6) ---------------------------------------------------
# OLLAMA_BASE_URL: Ollama's local REST API — no internet, no API key.
# OLLAMA_MODEL:    the local LLM to use. See Phase 6 guide Sections 2/6 for
#                   why this model was chosen and what to swap in on a
#                   lower-RAM laptop.
# LLM_TEMPERATURE: low value = more deterministic, less "creative" output —
#                   appropriate for grounded factual QA. See Section 16.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
LLM_TEMPERATURE = 0.1
LLM_REQUEST_TIMEOUT_SECONDS = 120
