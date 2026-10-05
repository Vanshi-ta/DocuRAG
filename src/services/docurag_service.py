

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD_VALUE,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
)
from src.errors import EmptyQuestionError, IndexNotReadyError
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import RAGAnswer, RAGStream, answer_question, answer_question_stream
from src.pipeline import (
    PipelineRunResult,
    ProgressCallback,
    load_vector_store,
    remove_document,
    reset_all,
    run_full_ingestion_pipeline,
)
from src.retrieval.retriever import Retriever
from src.vectorstore.document_registry import DocumentEntry, DocumentRegistry

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    

    index_loaded: bool
    document_count: int
    chunk_count: int
    llm: Dict[str, Any]                                      


class DocuRAGService:
    def __init__(
        self,
        embedder: Any = None,
        llm_client: Optional[OllamaClient] = None,
        upload_dir: Path = UPLOAD_DIR,
    ):
        self._embedder = embedder                                             
        self.llm_client = llm_client or OllamaClient()
        self.upload_dir = Path(upload_dir)

        self.faiss_store = None
        self.metadata_store = None
        self.registry: Optional[DocumentRegistry] = None

        self._lock = threading.RLock()                                              

                                                                             
    @property
    def embedder(self):
        if self._embedder is None:
            from src.ingestion.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

                                                                             
    def load_existing(self) -> bool:
        
        with self._lock:
            if self.faiss_store is not None:
                return True
            try:
                self.faiss_store, self.metadata_store, self.registry = load_vector_store(
                    self.embedder.embedding_dimension
                )
                return True
            except FileNotFoundError:
                return False

    @property
    def has_index(self) -> bool:
        return self.faiss_store is not None and self.faiss_store.ntotal > 0

    def list_documents(self) -> List[DocumentEntry]:
        return self.registry.list_documents() if self.registry is not None else []

    def status(self, check_llm: bool = True) -> ServiceStatus:
        llm = self.llm_client.health_check() if check_llm else {}
        return ServiceStatus(
            index_loaded=self.faiss_store is not None,
            document_count=len(self.registry.documents) if self.registry is not None else 0,
            chunk_count=self.faiss_store.ntotal if self.faiss_store is not None else 0,
            llm=llm,
        )

                                                                             
    def save_uploads(self, files: Mapping[str, bytes]) -> Tuple[List[Path], List[str]]:
        
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        rejected: List[str] = []
        for raw_name, data in files.items():
            name = Path(raw_name).name
            if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected.append(raw_name)
                continue
            target = self.upload_dir / name
            target.write_bytes(data)
            saved.append(target)
        return saved, rejected

    def ingest(
        self,
        paths: Optional[Sequence[Path]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineRunResult:
        
        with self._lock:
            result, faiss_store, metadata_store, registry = run_full_ingestion_pipeline(
                directory=self.upload_dir,
                embedder=self.embedder,
                persist=True,
                paths=paths,
                progress_callback=progress_callback,
            )
            self.faiss_store, self.metadata_store, self.registry = faiss_store, metadata_store, registry
            return result

    def delete_document(self, doc_id: str) -> int:
        
        with self._lock:
            if self.registry is None:
                return 0
            return remove_document(doc_id, self.faiss_store, self.metadata_store, self.registry)

    def reset(self) -> None:
        with self._lock:
            reset_all(self.upload_dir)
            self.faiss_store = self.metadata_store = self.registry = None

                                                                             
    def _prepare_question(self, question: str, use_threshold: bool) -> Tuple[str, Retriever, Optional[float]]:
        question = (question or "").strip()
        if not question:
            raise EmptyQuestionError("Please enter a non-empty question.")
        if not self.has_index:
            raise IndexNotReadyError("No documents are indexed yet.")
        retriever = Retriever(self.embedder, self.faiss_store, self.metadata_store)
        threshold = SIMILARITY_THRESHOLD_VALUE if use_threshold else None
        return question, retriever, threshold

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K, use_threshold: bool = False) -> RAGAnswer:
        question, retriever, threshold = self._prepare_question(question, use_threshold)
        return answer_question(
            question, retriever, self.llm_client, top_k=top_k, similarity_threshold=threshold
        )

    def ask_stream(self, question: str, top_k: int = DEFAULT_TOP_K, use_threshold: bool = False) -> RAGStream:
        question, retriever, threshold = self._prepare_question(question, use_threshold)
        return answer_question_stream(
            question, retriever, self.llm_client, top_k=top_k, similarity_threshold=threshold
        )
