

from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.documents import Document

from src.vectorstore.document_registry import DocumentRegistry
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import ChunkRecord, MetadataStore

if TYPE_CHECKING:                    
    from src.ingestion.embedder import Embedder


def add_chunks_to_index(
    chunks: List[Document],
    embedder: "Embedder",
    faiss_store: FaissVectorStore,
    metadata_store: MetadataStore,
    registry: DocumentRegistry,
) -> List[int]:
    
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
