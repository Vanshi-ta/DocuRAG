"""
RAG answer engine for DocuRAG.

The only module that knows about all three query-time pieces: retrieval
strategy, prompt construction, and the LLM client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
    ENTITY_FANOUT_ENABLED,
    MAX_CHUNKS_PER_SOURCE,
    SIMILARITY_THRESHOLD,
)
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import RetrievedChunk, Retriever, retrieve_for_question

logger = logging.getLogger(__name__)

NO_RELEVANT_CONTEXT_MESSAGE = "I could not find this information in the provided documents."


@dataclass
class RAGAnswer:
    """The generated answer plus the chunks it was grounded in, and a
    transparency flag on whether retrieval was confident enough to even
    call the LLM."""

    question: str
    answer: str
    sources: List[RetrievedChunk]
    used_llm: bool


def answer_question(
    question: str,
    retriever: Retriever,
    llm_client: OllamaClient,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    entity_fanout_enabled: bool = ENTITY_FANOUT_ENABLED,
) -> RAGAnswer:
    """
    Full query-time RAG flow:
      1. retrieve chunks via `retrieve_for_question` — a single
         `retrieve_diverse` call for questions about one entity/document,
         or an entity-fan-out merge for questions naming 2+ entities (see
         retriever.py for why the distinction matters). Either way, chunks
         below `similarity_threshold` are already excluded.
      2. if NOTHING survives retrieval, short-circuit: return the standard
         "not found" message WITHOUT calling the LLM. This is the ONLY
         place "not found" is decided before generation — it fires purely
         on "zero relevant chunks exist", never on "the question's exact
         wording wasn't matched" (that distinction lives entirely in
         retrieval; by the time chunks reach here, they were judged
         semantically relevant, which is why the LLM prompt is now
         instructed not to second-guess them into another "not found").
      3. otherwise, build a grounded prompt and call the LLM
      4. return the answer alongside the chunks it was grounded in
    """
    retrieved_chunks = retrieve_for_question(
        question,
        retriever,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        max_chunks_per_source=max_chunks_per_source,
        similarity_threshold=similarity_threshold,
        entity_fanout_enabled=entity_fanout_enabled,
    )

    if not retrieved_chunks:
        logger.info("No chunks passed the relevance threshold for %r — skipping LLM call.", question)
        return RAGAnswer(
            question=question,
            answer=NO_RELEVANT_CONTEXT_MESSAGE,
            sources=[],
            used_llm=False,
        )

    prompt = build_prompt(question, retrieved_chunks)
    answer_text = llm_client.generate(prompt)

    return RAGAnswer(question=question, answer=answer_text, sources=retrieved_chunks, used_llm=True)

