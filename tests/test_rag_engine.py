from src.generation.rag_engine import NO_RELEVANT_CONTEXT_MESSAGE, answer_question
from tests.helpers import FakeRetriever, make_chunk


class FakeLLMClient:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt):
        self.call_count += 1
        self.last_prompt = prompt
        return self.response_text


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
