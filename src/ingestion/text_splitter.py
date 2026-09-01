"""
Text chunking module for DocuRAG — Phase 3.

Responsible ONLY for:
    - splitting page-level LangChain Documents into smaller, overlapping
      chunks using RecursiveCharacterTextSplitter
    - preserving every piece of metadata Phase 2 attached (source_filename,
      doc_id, page_number, page_raw)
    - adding chunk-specific metadata (chunk_id, chunk_index, chunk_char_count)
    - validating chunk_size/chunk_overlap so a common misconfiguration
      (overlap >= size) fails loudly instead of silently producing garbage

This module has NO knowledge of embeddings, FAISS, or Streamlit — same
boundary discipline as src/ingestion/pdf_loader.py. It takes Documents in,
returns (more, smaller) Documents out.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass
class ChunkingResult:
    """Summary of one chunking run, for reporting/debugging."""

    chunks: List[Document] = field(default_factory=list)
    source_document_count: int = 0
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    def summary(self) -> str:
        lengths = [len(c.page_content) for c in self.chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        lines = [
            "=" * 60,
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}",
            f"Original page-level Documents: {self.source_document_count}",
            f"Resulting chunks: {len(self.chunks)}",
            f"Chunk length — min: {min(lengths, default=0)}, "
            f"max: {max(lengths, default=0)}, avg: {avg_len:.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """
    Catch the most common chunking misconfiguration before it silently
    produces bad results: overlap must be strictly smaller than chunk size.

    If overlap >= chunk_size, RecursiveCharacterTextSplitter can end up
    re-emitting the same text (or making near-zero forward progress through
    the document), producing far more chunks than expected and wasting
    embedding/storage work in later phases for no retrieval benefit.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}). A common mistake is setting overlap "
            f"too large relative to size — as a rule of thumb, keep overlap "
            f"at roughly 10-20% of chunk_size."
        )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> ChunkingResult:
    """
    Split page-level Documents into smaller, overlapping chunks.

    Every metadata key already present on a page Document (source_filename,
    doc_id, page_number, page_raw, source, page) is automatically copied onto
    every chunk produced from that page — this is RecursiveCharacterTextSplitter's
    default behavior via split_documents(), not something we implement
    manually. We only ADD to that inherited metadata.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Tried in order: split on paragraph breaks first, then line breaks,
        # then sentence-ish boundaries, then words, and only as a last
        # resort mid-word on any character. This is what makes it
        # "recursive" — it tries to respect natural text structure before
        # falling back to a hard cut.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        logger.warning(
            "Splitting produced zero chunks from %d input Documents — "
            "check whether the input Documents actually contain text.",
            len(documents),
        )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

    logger.info(
        "Split %d page-level Documents into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return ChunkingResult(
        chunks=chunks,
        source_document_count=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    # Debug entry point. Run from the project root with:
    #   python -m src.ingestion.text_splitter
    from config import UPLOAD_DIR
    from src.ingestion.pdf_loader import load_pdfs_from_directory

    ingestion_result = load_pdfs_from_directory(UPLOAD_DIR)
    if not ingestion_result.documents:
        print("No documents were loaded — add a PDF to data/uploads/ first.")
        raise SystemExit(0)

    chunking_result = split_documents(ingestion_result.documents)
    print("\n" + chunking_result.summary())

    print("\n--- Sample chunks ---")
    for chunk in chunking_result.chunks[:3]:
        print("\nmetadata:", chunk.metadata)
        print("content preview:", chunk.page_content[:200].replace("\n", " "))
