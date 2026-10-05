

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
    ENTITY_FANOUT_ENABLED,
    MAX_CHUNKS_PER_SOURCE,
    MAX_TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.retrieval.query_processing import extract_entities
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import MetadataStore

if TYPE_CHECKING:
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
                continue
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

    def retrieve_raw(
        self,
        question: str,
        search_width: int,
    ) -> List[RetrievedChunk]:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if search_width < 1:
            raise ValueError(f"search_width must be at least 1, got {search_width}")

        query_vector = self.embedder.embed_query(question)
        scores, ids = self.faiss_store.search(query_vector, top_k=search_width)

        results: List[RetrievedChunk] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue
            record = self.metadata_store.get(int(vid))
            results.append(
                RetrievedChunk(
                    chunk_text=record.chunk_text,
                    source_filename=record.source_filename,
                    page_number=record.page_number,
                    similarity_score=float(score),
                    chunk_id=record.chunk_id,
                )
            )
        return results

    def retrieve_diverse(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> List[RetrievedChunk]:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if top_k > MAX_TOP_K:
            logger.warning("top_k=%d exceeds MAX_TOP_K=%d; clamping.", top_k, MAX_TOP_K)
            top_k = MAX_TOP_K

        pool_size = max(candidate_pool_size, top_k)
        query_vector = self.embedder.embed_query(question)
        scores, ids = self.faiss_store.search(query_vector, top_k=pool_size)

        raw_pool: List[RetrievedChunk] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue
            record = self.metadata_store.get(int(vid))
            raw_pool.append(
                RetrievedChunk(
                    chunk_text=record.chunk_text,
                    source_filename=record.source_filename,
                    page_number=record.page_number,
                    similarity_score=float(score),
                    chunk_id=record.chunk_id,
                )
            )

        if similarity_threshold is not None:
            candidates = [c for c in raw_pool if c.similarity_score >= similarity_threshold]
        else:
            candidates = raw_pool
        n_filtered = len(raw_pool) - len(candidates)

        selected: List[RetrievedChunk] = []
        per_source_count: Dict[str, int] = {}
        for c in candidates:
            if len(selected) >= top_k:
                break
            if per_source_count.get(c.source_filename, 0) >= max_chunks_per_source:
                continue
            selected.append(c)
            per_source_count[c.source_filename] = per_source_count.get(c.source_filename, 0) + 1

        if len(selected) < top_k:
            selected_ids = {c.chunk_id for c in selected}
            for c in candidates:
                if len(selected) >= top_k:
                    break
                if c.chunk_id in selected_ids:
                    continue
                selected.append(c)
                selected_ids.add(c.chunk_id)

        selected.sort(key=lambda c: c.similarity_score, reverse=True)

        n_sources = len({c.source_filename for c in selected})
        logger.info(
            "retrieve_diverse: %d raw candidates from FAISS -> %d passed threshold -> "
            "%d selected from %d source(s) for %r (pool=%d, top_k=%d, cap=%d, "
            "threshold=%s, %d filtered out)",
            len(raw_pool), len(candidates), len(selected), n_sources, question,
            pool_size, top_k, max_chunks_per_source, similarity_threshold, n_filtered,
        )

        if not selected and raw_pool:
            preview = "; ".join(
                f"{c.source_filename} p.{c.page_number} score={c.similarity_score:.4f}"
                for c in raw_pool[:5]
            )
            logger.warning(
                "retrieve_diverse: ALL %d raw candidates scored below "
                "similarity_threshold=%s for %r — top raw candidates were: %s. "
                "If a document you expect to match (e.g. a specific resume) does not "
                "appear here at all, it may not be indexed; if it appears but with a "
                "low score, the threshold may be too high for this query's phrasing — "
                "see scripts/tune_threshold.py.",
                len(raw_pool), similarity_threshold, question, preview,
            )

        return selected


def retrieve_for_question(
    question: str,
    retriever: Retriever,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    entity_fanout_enabled: bool = ENTITY_FANOUT_ENABLED,
) -> List[RetrievedChunk]:
    entities = extract_entities(question) if entity_fanout_enabled else []

    if len(entities) < 2:
        return retriever.retrieve_diverse(
            question,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            max_chunks_per_source=max_chunks_per_source,
            similarity_threshold=similarity_threshold,
        )

    per_entity_k = max(2, -(-top_k // len(entities)))
    merged: List[RetrievedChunk] = []
    seen_chunk_ids = set()

    for entity in entities:
        subquery = f"{entity} {question}"
        sub_results = retriever.retrieve_diverse(
            subquery,
            top_k=per_entity_k,
            candidate_pool_size=candidate_pool_size,
            max_chunks_per_source=max_chunks_per_source,
            similarity_threshold=similarity_threshold,
        )
        for c in sub_results:
            if c.chunk_id not in seen_chunk_ids:
                merged.append(c)
                seen_chunk_ids.add(c.chunk_id)

    merged.sort(key=lambda c: c.similarity_score, reverse=True)
    merged = merged[:MAX_TOP_K]

    logger.info(
        "retrieve_for_question: entity fan-out for %r -> entities=%s, %d merged chunks from %d source(s)",
        question, entities, len(merged), len({c.source_filename for c in merged}),
    )
    return merged

