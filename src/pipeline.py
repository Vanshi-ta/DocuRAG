

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

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

if TYPE_CHECKING:                    
    from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

                                                                               
STATUS_INDEXED = "indexed"
STATUS_REINDEXED = "reindexed"
STATUS_SKIPPED_DUPLICATE = "skipped_duplicate"
STATUS_FAILED = "failed"

STAGE_HASHING = "hashing"
STAGE_LOADING = "loading"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"                                                     


@dataclass
class FileResult:
    

    filename: str
    status: str                                 
    chunks_added: int = 0
    page_count: int = 0
    error_type: Optional[str] = None                                                
    error: Optional[str] = None


@dataclass
class IngestionProgress:
    

    filename: str
    file_index: int           
    total_files: int
    stage: str
    message: str = ""


ProgressCallback = Callable[[IngestionProgress], None]


@dataclass
class PipelineRunResult:
    

    indexed_files: List[str] = field(default_factory=list)
    skipped_duplicate_files: List[str] = field(default_factory=list)              
    reindexed_files: List[str] = field(default_factory=list)                           
    failed_files: List[Tuple[str, str]] = field(default_factory=list)                      
    total_chunks_added: int = 0
    file_results: List[FileResult] = field(default_factory=list)                               

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


def _emit(callback: Optional[ProgressCallback], event: IngestionProgress) -> None:
    
    if callback is None:
        return
    try:
        callback(event)
    except Exception:                                
        logger.exception("progress_callback raised; ignoring so ingestion can continue")


def run_full_ingestion_pipeline(
    directory: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder: Embedder | None = None,
    persist: bool = True,
    paths: Optional[Sequence[Path]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[PipelineRunResult, FaissVectorStore, MetadataStore, DocumentRegistry]:
    
    if embedder is None:
                                                                           
                                                                     
        from src.ingestion.embedder import Embedder as _Embedder

        embedder = _Embedder()

    faiss_store, metadata_store, registry = _load_or_create_store(embedder.embedding_dimension)
    result = PipelineRunResult()

    if paths is not None:
        pdf_paths = [Path(p) for p in paths]
    else:
        if directory is None:
            directory = UPLOAD_DIR
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        pdf_paths = sorted(directory.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDF files to ingest")
        return result, faiss_store, metadata_store, registry

    total = len(pdf_paths)

    def progress(index: int, name: str, stage: str, message: str = "") -> None:
        _emit(progress_callback, IngestionProgress(name, index, total, stage, message))

    def fail(index: int, name: str, error_type: str, reason: str) -> None:
        result.failed_files.append((name, reason))
        result.file_results.append(
            FileResult(filename=name, status=STATUS_FAILED, error_type=error_type, error=reason)
        )
        progress(index, name, STATUS_FAILED, reason)

    for index, pdf_path in enumerate(pdf_paths, start=1):
        name = pdf_path.name

        progress(index, name, STAGE_HASHING)
        content_hash = compute_file_hash(pdf_path)

        existing_by_hash = registry.find_by_hash(content_hash)
        if existing_by_hash is not None:
            logger.info(
                "Skipping '%s' — identical content already indexed as '%s'",
                name, existing_by_hash.source_filename,
            )
            result.skipped_duplicate_files.append(name)
            result.file_results.append(
                FileResult(
                    filename=name,
                    status=STATUS_SKIPPED_DUPLICATE,
                    error=f"Identical content already indexed as '{existing_by_hash.source_filename}'.",
                )
            )
            progress(index, name, STATUS_SKIPPED_DUPLICATE, existing_by_hash.source_filename)
            continue

        progress(index, name, STAGE_LOADING)
        try:
            pages = load_single_pdf(pdf_path)
        except PDFLoadError as exc:
            logger.error("Failed to load '%s': %s", name, exc)
            fail(index, name, type(exc).__name__, str(exc))
            continue

        progress(index, name, STAGE_CHUNKING)
        chunking_result = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunking_result.chunks:
            logger.error("'%s' produced zero chunks — skipping", name)
            fail(index, name, "ZeroChunksError", "Produced zero chunks after splitting.")
            continue

                                                                                
        existing_by_name = registry.find_by_filename(name)
        is_reindex = existing_by_name is not None
        if is_reindex:
            logger.info("Content of '%s' changed since last index — re-indexing", name)
            stale_vector_ids = registry.remove_document(existing_by_name.doc_id)
            faiss_store.remove_ids(stale_vector_ids)
            metadata_store.remove(stale_vector_ids)

        progress(index, name, STAGE_EMBEDDING)
        vector_ids = add_chunks_to_index(chunking_result.chunks, embedder, faiss_store, metadata_store, registry)

        doc_id = pages[0].metadata["doc_id"]
        page_count = len({p.metadata["page_number"] for p in pages})
        registry.add_document(
            doc_id=doc_id,
            source_filename=name,
            content_hash=content_hash,
            page_count=page_count,
            chunk_count=len(vector_ids),
            vector_ids=vector_ids,
        )

        result.total_chunks_added += len(vector_ids)
        status = STATUS_REINDEXED if is_reindex else STATUS_INDEXED
        (result.reindexed_files if is_reindex else result.indexed_files).append(name)
        result.file_results.append(
            FileResult(filename=name, status=status, chunks_added=len(vector_ids), page_count=page_count)
        )
        progress(index, name, status)

    if persist:
        _persist(faiss_store, metadata_store, registry)

    logger.info("Ingestion run complete:\n%s", result.summary())
    return result, faiss_store, metadata_store, registry


def load_vector_store(embedding_dimension: int) -> Tuple[FaissVectorStore, MetadataStore, DocumentRegistry]:
    
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
    
    vector_ids = registry.remove_document(doc_id)
    faiss_store.remove_ids(vector_ids)
    metadata_store.remove(vector_ids)

    if persist:
        _persist(faiss_store, metadata_store, registry)

    return len(vector_ids)


def reset_all(directory: Path | None = None) -> None:
    
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
