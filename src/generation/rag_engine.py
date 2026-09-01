"""
RAG answer engine for DocuRAG — Phase 6.

This is the ONLY module that knows about all three query-time pieces: the
Retriever (Phase 5), prompt construction, and the LLM client. Each stays
independently testable and swappable — this module just wires them
together in the correct order, same orchestration pattern as
index_builder.py wiring embedder + FAISS + metadata together in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config import DEFAULT_TOP_K
from src.generation.llm_client import OllamaClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class RAGAnswer:
    """The final result of one question: the generated answer, plus the
    exact chunks it was grounded in — everything needed to display both
    the answer and its sources."""

    question: str
    answer: str
    sources: List[RetrievedChunk]


def answer_question(
    question: str,
    retriever: Retriever,
    llm_client: OllamaClient,
    top_k: int = DEFAULT_TOP_K,
) -> RAGAnswer:
    """
    Full query-time RAG flow:
      1. retrieve the top_k most relevant chunks for `question`
      2. build a grounded prompt from those chunks + the question
      3. send the prompt to the local LLM
      4. return the generated answer alongside the chunks it was grounded
         in, so sources can be displayed without re-running retrieval
    """
    retrieved_chunks = retriever.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved_chunks)
    answer_text = llm_client.generate(prompt)

    return RAGAnswer(question=question, answer=answer_text, sources=retrieved_chunks)
