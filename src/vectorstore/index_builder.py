"""
Index builder for DocuRAG.

The only module that knows about all four pieces: chunk Documents, the
Embedder, FaissVectorStore, and MetadataStore. It allocates vector IDs from
the DocumentRegistry, embeds the chunks, and adds them to FAISS and the
metadata store under those same IDs — the ordering/ID-matching guarantee
every downstream lookup depends on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.documents import Document

from src.vectorstore.document_registry import DocumentRegistry
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore

if TYPE_CHECKING:  # pragma: no cover
    from src.ingestion.embedder import Embedder


def add_chunks_to_index(
    chunks: List[Document],
    embedder: "Embedder",
    faiss_store: FaissVectorStore,
    metadata_store: MetadataStore,
    registry: DocumentRegistry,
) -> List[int]:
    """
    Embed `chunks` and add them to faiss_store/metadata_store under newly
    allocated vector IDs. Returns the list of vector IDs used, so the
    caller can register them against a DocumentEntry.
    """
    if not chunks:
        return []

    vector_ids = registry.allocate_vector_ids(len(chunks))

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_texts(texts)

    faiss_store.add_vectors(vectors, vector_ids)

    records = [
        ChunkRecord(
            vector_id=vid,
            chunk_id=chunk.metadata["chunk_id"],
            chunk_text=chunk.page_content,
            source_filename=chunk.metadata["source_filename"],
            doc_id=chunk.metadata["doc_id"],
            page_number=chunk.metadata["page_number"],
            chunk_index=chunk.metadata["chunk_index"],
        )
        for vid, chunk in zip(vector_ids, chunks)
    ]
    metadata_store.add_records(records)

    assert faiss_store.ntotal == len(metadata_store), (
        "FAISS index and metadata store are out of sync in size — "
        "this should never happen if add_chunks_to_index is the only "
        "path used to add vectors."
    )

    return vector_ids
