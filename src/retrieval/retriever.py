"""
Retrieval module for DocuRAG.

Responsible ONLY for: embed a question, search FAISS for the top-k most
similar chunks, look up their text/metadata, optionally drop chunks below
a similarity threshold, and return a clean, ranked list. No LLM knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from config import DEFAULT_TOP_K, MAX_TOP_K, SIMILARITY_THRESHOLD
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import MetadataStore

if TYPE_CHECKING:  # pragma: no cover - avoids importing sentence-transformers
    # just to type-hint a constructor argument, so retrieval logic stays
    # unit-testable with lightweight fakes (see tests/test_retriever.py).
    from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_text: str
    source_filename: str
    page_number: int
    similarity_score: float
    chunk_id: str


class Retriever:
    def __init__(
        self,
        embedder: "Embedder",
        faiss_store: FaissVectorStore,
        metadata_store: MetadataStore,
    ):
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.metadata_store = metadata_store

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> List[RetrievedChunk]:
        """
        Embed `question` and return the top_k most similar chunks (best
        first), each with its source filename, page number, and similarity
        score.

        `similarity_threshold`, if not None, drops any chunk scoring below
        it — these are typically near-random matches FAISS returns anyway
        (a flat index always returns *something*, even for an unrelated
        question) and would otherwise be handed to the LLM as if they were
        relevant context. Pass `similarity_threshold=None` to disable
        filtering entirely.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if top_k > MAX_TOP_K:
            logger.warning("top_k=%d exceeds MAX_TOP_K=%d; clamping.", top_k, MAX_TOP_K)
            top_k = MAX_TOP_K

        query_vector = self.embedder.embed_query(question)
        scores, ids = self.faiss_store.search(query_vector, top_k=top_k)

        results: List[RetrievedChunk] = []
        n_filtered = 0
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue  # FAISS pads with -1 when the index has fewer than top_k vectors
            score = float(score)
            if similarity_threshold is not None and score < similarity_threshold:
                n_filtered += 1
                continue
            record = self.metadata_store.get(int(vid))
            results.append(
                RetrievedChunk(
                    chunk_text=record.chunk_text,
                    source_filename=record.source_filename,
                    page_number=record.page_number,
                    similarity_score=score,
                    chunk_id=record.chunk_id,
                )
            )

        logger.info(
            "Retrieved %d chunks for question %r (top_k=%d, threshold=%s, %d filtered out)",
            len(results), question, top_k, similarity_threshold, n_filtered,
        )
        return results
