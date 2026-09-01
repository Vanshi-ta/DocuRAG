"""
Index builder for DocuRAG — Phase 4.

This is the ONLY module that knows about all three pieces: chunk Documents,
the Embedder, and the FaissVectorStore + MetadataStore pair. Each of those
stays independently reusable and testable; this module wires them together
in the correct order — same orchestration pattern as pipeline.py wiring
ingestion + chunking together in Phase 3.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from src.ingestion.embedder import Embedder
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore


def build_vector_index(
    chunks: List[Document],
    embedder: Embedder,
) -> Tuple[FaissVectorStore, MetadataStore]:
    """
    Embed every chunk and store the resulting vectors in FAISS, with a
    parallel MetadataStore so each vector's FAISS position maps back to its
    source chunk's text and metadata.

    The ordering guarantee this function relies on: `texts` is built from
    `chunks` in order, `embedder.embed_texts(texts)` preserves that order,
    `faiss_store.add_vectors(vectors)` adds them to FAISS in that same
    order, and `records` is built from that same `chunks` list in that same
    order. As long as nothing reorders any of these lists independently,
    FAISS position i and metadata_store.records[i] always describe the same
    chunk.
    """
    faiss_store = FaissVectorStore(embedder.embedding_dimension)
    metadata_store = MetadataStore()

    if not chunks:
        return faiss_store, metadata_store

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_texts(texts)

    faiss_store.add_vectors(vectors)

    records = [
        ChunkRecord(
            chunk_id=chunk.metadata["chunk_id"],
            chunk_text=chunk.page_content,
            source_filename=chunk.metadata["source_filename"],
            doc_id=chunk.metadata["doc_id"],
            page_number=chunk.metadata["page_number"],
            chunk_index=chunk.metadata["chunk_index"],
        )
        for chunk in chunks
    ]
    metadata_store.add_records(records)

    # Defensive check: if these two ever drift apart in size, every later
    # lookup silently returns the WRONG chunk's metadata for a given FAISS
    # match — a bug you'd only notice by manually spot-checking answers.
    # Fail loudly here instead, immediately after building the index.
    assert len(metadata_store) == faiss_store.index.ntotal, (
        "FAISS index and metadata store are out of sync in size — "
        "this should never happen if add_vectors and add_records are "
        "always called together with the same ordered list of chunks."
    )

    return faiss_store, metadata_store
