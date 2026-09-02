import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.rag_engine import NO_RELEVANT_CONTEXT_MESSAGE, answer_question
from src.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_call = None

    def retrieve(self, question, top_k, similarity_threshold=None):
        self.last_call = (question, top_k, similarity_threshold)
        return self._chunks


class FakeLLMClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt):
        self.call_count += 1
        self.last_prompt = prompt
        return self.response_text


def make_chunk(text="content", filename="doc.pdf", page=1, score=0.9):
    return RetrievedChunk(chunk_text=text, source_filename=filename, page_number=page,
                           similarity_score=score, chunk_id="c1")


def test_answer_question_calls_llm_when_chunks_found():
    retriever = FakeRetriever([make_chunk("Relevant fact.")])
    llm_client = FakeLLMClient("The answer is X.")

    result = answer_question("What is X?", retriever, llm_client, top_k=3)

    assert llm_client.call_count == 1
    assert result.used_llm is True
    assert result.answer == "The answer is X."
    assert len(result.sources) == 1


def test_answer_question_short_circuits_when_no_chunks_survive_threshold():
    retriever = FakeRetriever([])  # simulates every chunk filtered out by threshold
    llm_client = FakeLLMClient("should never be returned")

    result = answer_question("Some question", retriever, llm_client, top_k=3, similarity_threshold=0.9)

    assert llm_client.call_count == 0  # LLM never called
    assert result.used_llm is False
    assert result.answer == NO_RELEVANT_CONTEXT_MESSAGE
    assert result.sources == []


def test_answer_question_passes_threshold_through_to_retriever():
    retriever = FakeRetriever([make_chunk()])
    llm_client = FakeLLMClient("answer")

    answer_question("q", retriever, llm_client, top_k=5, similarity_threshold=0.42)

    assert retriever.last_call == ("q", 5, 0.42)
