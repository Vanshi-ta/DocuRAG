"""
Automated tests for src/generation/rag_engine.py.

Uses fake Retriever and OllamaClient stand-ins (duck-typed, no real vector
store or model needed) to test that answer_question() wires retrieval,
prompt construction, and generation together correctly.

Run from the project root with:
    pytest tests/test_rag_engine.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.rag_engine import answer_question
from src.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    """Stands in for Retriever — returns a fixed list of chunks and records
    what top_k it was called with, so tests can assert on it."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_top_k = None

    def retrieve(self, question, top_k):
        self.last_top_k = top_k
        return self._chunks


class FakeLLMClient:
    """Stands in for OllamaClient — returns a fixed answer and records the
    prompt it was called with, so tests can assert on prompt content
    without needing a real model."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.response_text


def make_chunk(text="content", filename="doc.pdf", page=1):
    return RetrievedChunk(
        chunk_text=text, source_filename=filename, page_number=page,
        similarity_score=0.9, chunk_id="c1",
    )


def test_answer_question_wires_retrieval_prompt_and_llm_together():
    chunks = [make_chunk("Relevant fact.")]
    retriever = FakeRetriever(chunks)
    llm_client = FakeLLMClient("The answer is X.")

    result = answer_question("What is X?", retriever, llm_client, top_k=3)

    assert retriever.last_top_k == 3
    assert "Relevant fact." in llm_client.last_prompt
    assert "What is X?" in llm_client.last_prompt
    assert result.answer == "The answer is X."
    assert result.sources == chunks
    assert result.question == "What is X?"


def test_answer_question_uses_default_top_k_when_not_specified():
    from config import DEFAULT_TOP_K

    retriever = FakeRetriever([make_chunk()])
    llm_client = FakeLLMClient("answer")

    answer_question("a question", retriever, llm_client)

    assert retriever.last_top_k == DEFAULT_TOP_K


def test_answer_question_preserves_sources_even_when_answer_says_not_found():
    chunks = [make_chunk("Unrelated content.")]
    retriever = FakeRetriever(chunks)
    llm_client = FakeLLMClient("I could not find this information in the provided documents.")

    result = answer_question("Some question", retriever, llm_client, top_k=2)

    # Sources are still returned even when the LLM says it couldn't find an
    # answer — so a UI can still show "here's what was checked" (Phase 7+).
    assert result.sources == chunks
    assert "could not find" in result.answer.lower()
