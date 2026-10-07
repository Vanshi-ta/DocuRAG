"""Tests for answer_question_stream: same retrieval/short-circuit behavior as
answer_question, delivered as sources + token iterator."""

from src.generation.rag_engine import NO_RELEVANT_CONTEXT_MESSAGE, answer_question_stream
from tests.helpers import FakeRetriever, make_chunk


class FakeStreamingLLM:
    def __init__(self, tokens):
        self.tokens = tokens
        self.stream_calls = 0
        self.last_prompt = None

    def generate_stream(self, prompt):
        self.stream_calls += 1
        self.last_prompt = prompt
        return iter(self.tokens)


def test_stream_returns_sources_first_and_tokens_after():
    llm = FakeStreamingLLM(["The ", "answer."])
    result = answer_question_stream("What?", FakeRetriever([make_chunk("fact", "a.pdf", 2, 0.8)]), llm, top_k=3)
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
