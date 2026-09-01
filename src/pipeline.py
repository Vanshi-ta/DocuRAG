"""
Pipeline orchestration for DocuRAG.

This module is the ONLY place that chains ingestion (Phase 2) and chunking
(Phase 3) together. Neither pdf_loader.py nor text_splitter.py imports the
other — pdf_loader produces Documents, text_splitter consumes Documents,
and this file wires them in sequence. That keeps each stage independently
testable and swappable (e.g., you could later add a non-PDF loader here
without touching text_splitter.py at all).
"""

from __future__ import annotations

from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from src.ingestion.pdf_loader import IngestionResult, load_pdfs_from_directory
from src.ingestion.text_splitter import ChunkingResult, split_documents


def ingest_and_chunk(
    directory: Path = UPLOAD_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[IngestionResult, ChunkingResult]:
    """
    Run the full Phase 2 + Phase 3 pipeline: load every PDF in `directory`,
    then split the resulting page-level Documents into overlapping chunks.

    Returns both results (not just the chunks) so callers/tests can inspect
    what happened at each stage — e.g. which files were skipped during
    ingestion, independent of how chunking went.
    """
    ingestion_result = load_pdfs_from_directory(directory)
    chunking_result = split_documents(
        ingestion_result.documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return ingestion_result, chunking_result


if __name__ == "__main__":
    # Run from the project root with:
    #   python -m src.pipeline
    ingestion_result, chunking_result = ingest_and_chunk()

    print("\n--- Ingestion (Phase 2) ---")
    print(ingestion_result.summary())

    print("\n--- Chunking (Phase 3) ---")
    print(chunking_result.summary())
