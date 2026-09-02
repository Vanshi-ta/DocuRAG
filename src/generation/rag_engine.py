"""
RAG answer engine for DocuRAG.

The only module that knows about all three query-time pieces: Retriever,
prompt construction, and the LLM client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import RetrievedChunk, Retriever

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
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
) -> RAGAnswer:
    """
    Full query-time RAG flow:
      1. retrieve the top_k most relevant chunks for `question`, dropping
         any below `similarity_threshold`
      2. if NOTHING survives filtering, short-circuit: return the standard
         "not found" message WITHOUT calling the LLM. This is both a
         reliability measure (no chance of the LLM hallucinating an answer
         from irrelevant context) and a cost/latency optimization.
      3. otherwise, build a grounded prompt and call the LLM
      4. return the answer alongside the chunks it was grounded in
    """
    retrieved_chunks = retriever.retrieve(question, top_k=top_k, similarity_threshold=similarity_threshold)

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
