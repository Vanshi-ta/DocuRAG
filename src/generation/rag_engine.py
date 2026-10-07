

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Iterator

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
    

    question: str
    answer: str
    sources: list[RetrievedChunk]
    used_llm: bool


@dataclass
class RAGStream:
    

    question: str
    sources: list[RetrievedChunk]
    used_llm: bool
    tokens: Iterator[str]


def answer_question(
    question: str,
    retriever: Retriever,
    llm_client: OllamaClient,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    similarity_threshold: float | None = SIMILARITY_THRESHOLD,
    entity_fanout_enabled: bool = ENTITY_FANOUT_ENABLED,
) -> RAGAnswer:
    
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


def answer_question_stream(
    question: str,
    retriever: Retriever,
    llm_client: OllamaClient,
    top_k: int = DEFAULT_TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    similarity_threshold: float | None = SIMILARITY_THRESHOLD,
    entity_fanout_enabled: bool = ENTITY_FANOUT_ENABLED,
) -> RAGStream:
    
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
        return RAGStream(
            question=question,
            sources=[],
            used_llm=False,
            tokens=iter([NO_RELEVANT_CONTEXT_MESSAGE]),
        )

    prompt = build_prompt(question, retrieved_chunks)
    tokens = llm_client.generate_stream(prompt)
    return RAGStream(question=question, sources=retrieved_chunks, used_llm=True, tokens=tokens)
