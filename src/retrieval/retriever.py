"""
Retrieval module for DocuRAG.

Responsible ONLY for: embed a question, search FAISS for the top-k most
similar chunks, look up their text/metadata, optionally drop chunks below
a similarity threshold, and return a clean, ranked list. No LLM knowledge.
"""

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
        Single-stage retrieval: embed `question`, return the top_k globally
        highest-scoring chunks. Kept as the simple baseline used by
        scripts/evaluate.py (retrieval-accuracy measurement wants the raw
        ranking, not the diversity-adjusted one). For actual question
        answering, `retrieve_diverse` (or `retrieve_for_question`, which
        adds entity fan-out) should be used instead — see their docstrings
        for why a single top-k search is insufficient for multi-document
        questions.
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

    def retrieve_raw(
        self,
        question: str,
        search_width: int,
    ) -> List[RetrievedChunk]:
        """
        Diagnostic-only method: return up to `search_width` raw FAISS
        candidates, UNFILTERED and UNCLAMPED against MAX_TOP_K.

        `retrieve()` and `retrieve_diverse()` both clamp their `top_k`
        argument against MAX_TOP_K because that limit is meant to bound how
        much context ultimately reaches the LLM — a real production
        concern. It is NOT meant to limit how deep a diagnostic script can
        inspect the raw ranking (e.g. "show me the top 15 candidates and
        their scores" for debugging a threshold value), so this method
        intentionally does not apply that clamp. It exists purely to
        support scripts/tune_threshold.py — nothing in the answer-serving
        path (rag_engine.py) calls this.
        """
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
        """
        Two-stage retrieval that fixes the "one document fills every slot"
        failure mode a plain top-k search has on multi-document questions:

          Stage 1 (candidate retrieval): pull `candidate_pool_size`
          candidates from FAISS instead of just `top_k` — a wider net to
          select diversity from.

          Stage 2 (diversity selection): walk the candidates best-score-
          first (FAISS already returns them sorted) and greedily select up
          to `top_k`, skipping any chunk that would push its source
          filename's count past `max_chunks_per_source`. If the cap leaves
          slots unfilled (e.g. because fewer than top_k/cap distinct
          sources exist above the threshold at all), the remaining slots
          are backfilled from leftover candidates in score order, so a
          single-document corpus still gets a full top_k instead of being
          artificially starved by its own cap.

        This is NOT full Maximal Marginal Relevance (MMR) — it doesn't
        compute pairwise semantic redundancy between candidate chunks, only
        a hard per-source count cap. That's a deliberate simplification:
        the actual failure mode observed (one resume's chunks filling
        every slot) is a *source*-diversity problem, not a *semantic*-
        redundancy problem, so a per-source cap solves it directly with
        O(candidate_pool_size) work and zero extra embedding computation.
        Full MMR (computing chunk-to-chunk cosine similarity to penalize
        near-duplicate content even within one source) would be the next
        step if within-document near-duplicate chunks became a problem —
        see docs/RETRIEVAL.md for that discussion.
        """
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

        # Build the FULL raw pool first (before any threshold filtering) so
        # that if filtering ends up discarding everything, we still have
        # the actual scores/sources on hand to log — this is what the old
        # version of this method could NOT do, since it discarded a
        # below-threshold chunk's score the moment it failed the check.
        raw_pool: List[RetrievedChunk] = []
        for score, vid in zip(scores[0], ids[0]):
            if vid == -1:
                continue  # FAISS pads with -1 when the index has fewer than pool_size vectors
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

        n_sources = len({c.source_filename for c in selected})
        logger.info(
            "retrieve_diverse: %d raw candidates from FAISS -> %d passed threshold -> "
            "%d selected from %d source(s) for %r (pool=%d, top_k=%d, cap=%d, "
            "threshold=%s, %d filtered out)",
            len(raw_pool), len(candidates), len(selected), n_sources, question,
            pool_size, top_k, max_chunks_per_source, similarity_threshold, n_filtered,
        )

        if not selected and raw_pool:
            # This is the exact situation being diagnosed: the index is not
            # empty and DID return candidates, but every one of them scored
            # below similarity_threshold. Log the actual raw scores so this
            # is diagnosable directly from the running app's log next time,
            # without needing a separate script run.
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
    """
    The retrieval strategy actually used for answering questions (called by
    rag_engine.answer_question). Decides between two strategies:

      - 0 or 1 detected entities -> a single `retrieve_diverse` call on the
        question as written. This covers direct questions and questions
        about one document.

      - 2+ detected entities (e.g. "Vanshita" and "Manas") -> one
        `retrieve_diverse` sub-call per entity, using a query of
        "<entity> <original question>" for each, then the results are
        merged (deduplicated by chunk_id, re-sorted by score, capped at
        MAX_TOP_K). This exists because a single query embedding for
        "What skills does Vanshita have that Manas does not?" tends to
        drift toward whichever entity's name/context is semantically
        closer to the rest of the sentence — fanning out guarantees each
        named entity gets its own dedicated retrieval pass rather than
        competing for the same top_k slots.

    Why not just always increase top_k instead of doing entity fan-out?
    Because raising top_k for every question (not just comparison ones)
    means single-document questions get padded with irrelevant chunks too
    — more prompt tokens, slower generation, and more chances for the LLM
    to get confused by irrelevant context. Fan-out only activates when the
    question's own wording signals it needs multiple documents' worth of
    evidence.
    """
    entities = extract_entities(question) if entity_fanout_enabled else []

    if len(entities) < 2:
        return retriever.retrieve_diverse(
            question,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            max_chunks_per_source=max_chunks_per_source,
            similarity_threshold=similarity_threshold,
        )

    per_entity_k = max(2, -(-top_k // len(entities)))  # ceil division
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

