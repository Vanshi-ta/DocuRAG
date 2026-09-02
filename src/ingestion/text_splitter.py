"""
Text chunking module for DocuRAG.

Responsible ONLY for:
    - splitting page-level LangChain Documents into smaller, overlapping
      chunks using RecursiveCharacterTextSplitter
    - preserving every piece of metadata pdf_loader attached
    - adding chunk-specific metadata (chunk_id, chunk_index, chunk_char_count)
    - validating chunk_size/chunk_overlap so a common misconfiguration
      (overlap >= size) fails loudly instead of silently producing garbage
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


@dataclass
class ChunkingResult:
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
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}). Keep overlap at roughly 10-20% of chunk_size."
        )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> ChunkingResult:
    validate_chunk_params(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        logger.warning(
            "Splitting produced zero chunks from %d input Documents.", len(documents)
        )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

    logger.info(
        "Split %d page-level Documents into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )

    return ChunkingResult(
        chunks=chunks,
        source_document_count=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
