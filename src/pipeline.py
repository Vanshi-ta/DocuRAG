"""
Pipeline orchestration for DocuRAG.

The only module that chains ingestion, chunking, embedding, and vector
storage together, and the only module that knows about duplicate
detection and document deletion/re-indexing. Each stage module (pdf_loader,
text_splitter, embedder, faiss_store, metadata_store, document_registry)
stays independently testable and swappable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_REGISTRY_PATH,
    FAISS_INDEX_PATH,
    METADATA_STORE_PATH,
    UPLOAD_DIR,
)
from src.ingestion.hashing import compute_file_hash
from src.ingestion.pdf_loader import PDFLoadError, load_single_pdf
from src.ingestion.text_splitter import split_documents
from src.vectorstore.document_registry import DocumentRegistry
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.index_builder import add_chunks_to_index
from src.vectorstore.metadata_store import MetadataStore

if TYPE_CHECKING:  # pragma: no cover
    from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class PipelineRunResult:
    """Summary of one ingestion run, for UI display and logging."""

    indexed_files: List[str] = field(default_factory=list)
    skipped_duplicate_files: List[str] = field(default_factory=list)  # (filename)
    reindexed_files: List[str] = field(default_factory=list)          # content changed
    failed_files: List[Tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    total_chunks_added: int = 0

    def summary(self) -> str:
        lines = [
            f"Indexed: {self.indexed_files}",
            f"Re-indexed (content changed): {self.reindexed_files}",
            f"Skipped duplicates: {self.skipped_duplicate_files}",
            f"Failed: {self.failed_files}",
            f"Chunks added: {self.total_chunks_added}",
        ]
        return "\n".join(lines)


def _load_or_create_store(embedding_dimension: int) -> Tuple[FaissVectorStore, MetadataStore, DocumentRegistry]:
    try:
        faiss_store = FaissVectorStore.load(FAISS_INDEX_PATH, embedding_dimension)
        metadata_store = MetadataStore.load(METADATA_STORE_PATH)
        registry = DocumentRegistry.load(DOCUMENT_REGISTRY_PATH)
    except FileNotFoundError:
        faiss_store = FaissVectorStore(embedding_dimension)
        metadata_store = MetadataStore()
        registry = DocumentRegistry()
    return faiss_store, metadata_store, registry


def _persist(faiss_store: FaissVectorStore, metadata_store: MetadataStore, registry: DocumentRegistry) -> None:
    faiss_store.save(FAISS_INDEX_PATH)
    metadata_store.save(METADATA_STORE_PATH)
    registry.save(DOCUMENT_REGISTRY_PATH)


def run_full_ingestion_pipeline(
    directory: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder: Embedder | None = None,
    persist: bool = True,
) -> Tuple[PipelineRunResult, FaissVectorStore, MetadataStore, DocumentRegistry]:
    """
    Ingest every PDF in `directory` into the persistent vector store,
    per file:

      1. Hash the file's raw bytes.
      2. If that hash is already registered under any filename -> skip
         (duplicate content, log it, do not re-embed).
      3. If a document with the same *filename* is already registered but
         with a *different* hash -> the file changed; remove the old
         document's vectors/metadata/registry entry first, then proceed
         (clean re-indexing, not silent duplication or stale data).
      4. Load, chunk, embed, and add the new/changed file to the index.

    Existing indexed documents not present in `directory` this run are left
    untouched — this function only adds/updates, it never deletes based on
    absence. Use `remove_document()` for explicit deletion.
    """
    if embedder is None:
        embedder = Embedder()
    if directory is None:
        directory = UPLOAD_DIR

    faiss_store, metadata_store, registry = _load_or_create_store(embedder.embedding_dimension)
    result = PipelineRunResult()

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found in %s", directory)
        return result, faiss_store, metadata_store, registry

    for pdf_path in pdf_paths:
        content_hash = compute_file_hash(pdf_path)

        existing_by_hash = registry.find_by_hash(content_hash)
        if existing_by_hash is not None:
            logger.info(
                "Skipping '%s' — identical content already indexed as '%s'",
                pdf_path.name, existing_by_hash.source_filename,
            )
            result.skipped_duplicate_files.append(pdf_path.name)
            continue

        existing_by_name = registry.find_by_filename(pdf_path.name)
        is_reindex = existing_by_name is not None
        if is_reindex:
            logger.info(
                "Content of '%s' changed since last index — re-indexing", pdf_path.name
            )
            stale_vector_ids = registry.remove_document(existing_by_name.doc_id)
            faiss_store.remove_ids(stale_vector_ids)
            metadata_store.remove(stale_vector_ids)

        try:
            pages = load_single_pdf(pdf_path)
        except PDFLoadError as exc:
            logger.error("Failed to load '%s': %s", pdf_path.name, exc)
            result.failed_files.append((pdf_path.name, str(exc)))
            continue

        chunking_result = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunking_result.chunks:
            logger.error("'%s' produced zero chunks — skipping", pdf_path.name)
            result.failed_files.append((pdf_path.name, "Produced zero chunks after splitting."))
            continue

        vector_ids = add_chunks_to_index(chunking_result.chunks, embedder, faiss_store, metadata_store, registry)

        doc_id = pages[0].metadata["doc_id"]
        page_count = len({p.metadata["page_number"] for p in pages})
        registry.add_document(
            doc_id=doc_id,
            source_filename=pdf_path.name,
            content_hash=content_hash,
            page_count=page_count,
            chunk_count=len(vector_ids),
            vector_ids=vector_ids,
        )

        result.total_chunks_added += len(vector_ids)
        if is_reindex:
            result.reindexed_files.append(pdf_path.name)
        else:
            result.indexed_files.append(pdf_path.name)

    if persist:
        _persist(faiss_store, metadata_store, registry)

    logger.info("Ingestion run complete:\n%s", result.summary())
    return result, faiss_store, metadata_store, registry


def load_vector_store(embedding_dimension: int) -> Tuple[FaissVectorStore, MetadataStore, DocumentRegistry]:
    """Reload a previously persisted store without re-running ingestion."""
    faiss_store = FaissVectorStore.load(FAISS_INDEX_PATH, embedding_dimension)
    metadata_store = MetadataStore.load(METADATA_STORE_PATH)
    registry = DocumentRegistry.load(DOCUMENT_REGISTRY_PATH)
    return faiss_store, metadata_store, registry


def remove_document(
    doc_id: str,
    faiss_store: FaissVectorStore,
    metadata_store: MetadataStore,
    registry: DocumentRegistry,
    persist: bool = True,
) -> int:
    """
    Explicitly delete one indexed document: removes its vectors from FAISS,
    its records from the metadata store, and its entry from the registry.
    Every other document's vector IDs are untouched. Returns the number of
    chunks removed.
    """
    vector_ids = registry.remove_document(doc_id)
    faiss_store.remove_ids(vector_ids)
    metadata_store.remove(vector_ids)

    if persist:
        _persist(faiss_store, metadata_store, registry)

    return len(vector_ids)


def reset_all(directory: Path | None = None) -> None:
    """Delete all uploaded PDFs and the persisted index/metadata/registry,
    returning the project to a clean, unindexed state."""
    if directory is None:
        directory = UPLOAD_DIR
    for pdf_path in directory.glob("*.pdf"):
        pdf_path.unlink()
    for path in (FAISS_INDEX_PATH, METADATA_STORE_PATH, DOCUMENT_REGISTRY_PATH):
        if path.exists():
            path.unlink()
    logger.info("Reset complete — uploads and persisted store cleared.")


if __name__ == "__main__":
    from src.logging_config import configure_logging

    configure_logging()
    run_result, store, metadata, doc_registry = run_full_ingestion_pipeline()
    print("\n" + run_result.summary())
    print(f"\nVectors in FAISS index: {store.ntotal}")
    print(f"Records in metadata store: {len(metadata)}")
    print(f"Documents in registry: {len(doc_registry.documents)}")
