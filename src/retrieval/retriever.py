"""
Retrieval module for DocuRAG — Phase 5.

Responsible ONLY for:
    - embedding a user's natural-language question with the SAME embedder
      used at ingestion time
    - searching the FAISS index for the top-k most similar chunk vectors
    - looking up each match's text and metadata via the metadata store
    - returning a clean, ordered list of RetrievedChunk objects

This module has NO knowledge of the LLM or Streamlit. It takes a question
in, returns ranked chunks out — the same "pure function, no UI knowledge"
discipline as every prior-phase module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from config import DEFAULT_TOP_K
from src.ingestion.embedder import Embedder
from src.vectorstore.faiss_store import FaissVectorStore
from src.vectorstore.metadata_store import MetadataStore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass
class RetrievedChunk:
    """One retrieved chunk, ready to display or hand to a future LLM prompt."""

    chunk_text: str
    source_filename: str
    page_number: int
    similarity_score: float
    chunk_id: str


class Retriever:
    """
    Wires together an Embedder, a FaissVectorStore, and a MetadataStore to
    answer: "given this question, what are the top-k most relevant chunks?"

    Pass in already-constructed instances rather than building them inside
    __init__ — the embedder, in particular, should be loaded once and
    shared across ingestion AND retrieval (Phase 4 guide Section 19), not
    reloaded here.
    """

    def __init__(
        self,
        embedder: Embedder,
        faiss_store: FaissVectorStore,
        metadata_store: MetadataStore,
    ):
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.metadata_store = metadata_store

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
        """
        Embed `question` and return the top_k most similar chunks, ranked
        best-first, each with its source filename, page number, and
        similarity score attached.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        query_vector = self.embedder.embed_query(question)
        scores, indices = self.faiss_store.search(query_vector, top_k=top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS pads results with -1 when the index has fewer than
                # top_k vectors total — skip those, they're not real matches.
                continue
            record = self.metadata_store.get(int(idx))
            results.append(
                RetrievedChunk(
                    chunk_text=record.chunk_text,
                    source_filename=record.source_filename,
                    page_number=record.page_number,
                    similarity_score=float(score),
                    chunk_id=record.chunk_id,
                )
            )

        logger.info(
            "Retrieved %d chunks for question %r (top_k=%d)",
            len(results), question, top_k,
        )
        return results
