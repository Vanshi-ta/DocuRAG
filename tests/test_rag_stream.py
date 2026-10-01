"""Tests for answer_question_stream: same retrieval/short-circuit behavior as
answer_question, delivered as sources + token iterator."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.rag_engine import NO_RELEVANT_CONTEXT_MESSAGE, answer_question_stream
from src.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve_diverse(self, question, top_k, candidate_pool_size=None,
                          max_chunks_per_source=None, similarity_threshold=None):
        return self._chunks


class FakeStreamingLLM:
    def __init__(self, tokens):
        self.tokens = tokens
        self.stream_calls = 0
        self.last_prompt = None

    def generate_stream(self, prompt):
        self.stream_calls += 1
        self.last_prompt = prompt
        return iter(self.tokens)


def chunk(text="fact"):
    return RetrievedChunk(chunk_text=text, source_filename="a.pdf", page_number=2,
                          similarity_score=0.8, chunk_id="c1")


def test_stream_returns_sources_first_and_tokens_after():
    llm = FakeStreamingLLM(["The ", "answer."])
    result = answer_question_stream("What?", FakeRetriever([chunk()]), llm, top_k=3)
    assert result.used_llm is True
    assert [s.source_filename for s in result.sources] == ["a.pdf"]
    assert "".join(result.tokens) == "The answer."
    assert "fact" in llm.last_prompt


def test_stream_short_circuits_without_calling_llm():
    llm = FakeStreamingLLM(["never"])
    result = answer_question_stream("What?", FakeRetriever([]), llm, top_k=3)
    assert llm.stream_calls == 0
    assert result.used_llm is False
    assert result.sources == []
    assert "".join(result.tokens) == NO_RELEVANT_CONTEXT_MESSAGE
