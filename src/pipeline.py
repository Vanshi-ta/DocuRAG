"""
Pipeline orchestration for DocuRAG.

This module is the ONLY place that chains ingestion (Phase 2), chunking
(Phase 3), and embedding + vector storage (Phase 4) together. Each stage
module stays independently testable and swappable — this file just wires
them in sequence, in the correct order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    UPLOAD_DIR,
)
from src.ingestion.embedder import Embedder
from src.ingestion.pdf_loader import IngestionResult, load_pdfs_from_directory
from src.ingestion.text_splitter import ChunkingResult, split_documents
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.index_builder import build_vector_index
from src.vectorstore.metadata_store import MetadataStore


def ingest_and_chunk(
    directory: Path = UPLOAD_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[IngestionResult, ChunkingResult]:
    """
    Run the Phase 2 + Phase 3 pipeline: load every PDF in `directory`, then
    split the resulting page-level Documents into overlapping chunks.
    """
    ingestion_result = load_pdfs_from_directory(directory)
    chunking_result = split_documents(
        ingestion_result.documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return ingestion_result, chunking_result


def run_full_ingestion_pipeline(
    directory: Path = UPLOAD_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder: Embedder | None = None,
    persist: bool = True,
) -> Tuple[IngestionResult, ChunkingResult, FaissVectorStore, MetadataStore]:
    """
    Run the complete Phase 2 + 3 + 4 pipeline end to end: PDFs -> Documents
    -> chunks -> embeddings -> FAISS index (+ metadata store), optionally
    persisting the index and metadata to disk.

    `embedder` can be passed in so a caller (e.g. a future Streamlit app)
    can load the model ONCE and reuse it across multiple pipeline runs,
    rather than this function silently constructing a new one every call —
    see Phase 4 guide Section 19 for why that matters.
    """
    ingestion_result, chunking_result = ingest_and_chunk(directory, chunk_size, chunk_overlap)

    if embedder is None:
        embedder = Embedder()

    faiss_store, metadata_store = build_vector_index(chunking_result.chunks, embedder)

    if persist:
        faiss_store.save(FAISS_INDEX_PATH)
        metadata_store.save(METADATA_STORE_PATH)

    return ingestion_result, chunking_result, faiss_store, metadata_store


def load_vector_store(embedding_dimension: int) -> Tuple[FaissVectorStore, MetadataStore]:
    """
    Reload a previously persisted FAISS index + metadata store from disk,
    without re-running ingestion/chunking/embedding — this is what the
    Streamlit app calls on startup if an index already exists, instead of
    forcing the user to re-upload every session.
    """
    faiss_store = FaissVectorStore.load(FAISS_INDEX_PATH, embedding_dimension)
    metadata_store = MetadataStore.load(METADATA_STORE_PATH)
    return faiss_store, metadata_store


def reset_all(directory: Path = UPLOAD_DIR) -> None:
    """
    Delete all uploaded PDFs and the persisted FAISS index + metadata
    store, returning the project to a clean, unindexed state.

    This is a Phase 7 addition: the Streamlit "Reset index" button calls
    this instead of doing raw filesystem cleanup itself, keeping app.py
    free of anything beyond UI/session-state concerns (Phase 7 guide's
    architecture requirement).
    """
    for pdf_path in directory.glob("*.pdf"):
        pdf_path.unlink()
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if METADATA_STORE_PATH.exists():
        METADATA_STORE_PATH.unlink()


if __name__ == "__main__":
    # Run from the project root with:
    #   python -m src.pipeline
    ingestion_result, chunking_result, faiss_store, metadata_store = run_full_ingestion_pipeline()

    print("\n--- Ingestion (Phase 2) ---")
    print(ingestion_result.summary())

    print("\n--- Chunking (Phase 3) ---")
    print(chunking_result.summary())

    print("\n--- Embedding + Vector Store (Phase 4) ---")
    print(f"Vectors in FAISS index: {faiss_store.index.ntotal}")
    print(f"Records in metadata store: {len(metadata_store)}")
    print(f"Embedding dimension: {faiss_store.embedding_dimension}")
    print(f"Persisted to: {FAISS_INDEX_PATH} and {METADATA_STORE_PATH}")

