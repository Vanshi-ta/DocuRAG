"""Tests for retrieve_for_question: entity fan-out routing, per-entity top_k
math, de-duplication by chunk_id, and the MAX_TOP_K cap on merged results."""

import math

from config import MAX_TOP_K
from src.retrieval.query_processing import extract_entities
from src.retrieval.retriever import retrieve_for_question
from tests.helpers import make_chunk

TWO_ENTITY_QUESTION = "What are the skills of Manas and Vanshita?"


class RecordingRetriever:
    """Records every retrieve_diverse call; `responder(question)` supplies the chunks."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def retrieve_diverse(self, question, top_k, candidate_pool_size, max_chunks_per_source, similarity_threshold):
        self.calls.append({"question": question, "top_k": top_k, "threshold": similarity_threshold})
        return self._responder(question)


def test_two_entities_fan_out_into_one_subquery_each():
    entities = extract_entities(TWO_ENTITY_QUESTION)
    assert len(entities) == 2
    retriever = RecordingRetriever(lambda q: [])

    retrieve_for_question(TWO_ENTITY_QUESTION, retriever, top_k=6, entity_fanout_enabled=True)

    assert [c["question"] for c in retriever.calls] == [f"{e} {TWO_ENTITY_QUESTION}" for e in entities]


def test_per_entity_top_k_is_ceil_split_with_floor_of_two():
    for top_k, expected in [(6, 3), (7, 4), (2, 2), (1, 2)]:
        retriever = RecordingRetriever(lambda q: [])
        retrieve_for_question(TWO_ENTITY_QUESTION, retriever, top_k=top_k, entity_fanout_enabled=True)
        assert expected == max(2, math.ceil(top_k / 2))
        assert [c["top_k"] for c in retriever.calls] == [expected, expected]


def test_threshold_is_passed_through_to_every_subquery():
    retriever = RecordingRetriever(lambda q: [])
    retrieve_for_question(
        TWO_ENTITY_QUESTION, retriever, top_k=6, similarity_threshold=0.42, entity_fanout_enabled=True
    )
    assert [c["threshold"] for c in retriever.calls] == [0.42, 0.42]


def test_merged_results_are_deduplicated_by_chunk_id_and_sorted_by_score():
    shared = make_chunk("shared", "a.pdf", 1, score=0.70, chunk_id="shared")

    def responder(question):
        if question.startswith("Manas"):
            return [make_chunk("m", "m.pdf", 1, score=0.50, chunk_id="m1"), shared]
        return [make_chunk("v", "v.pdf", 1, score=0.90, chunk_id="v1"), shared]

    retriever = RecordingRetriever(responder)
    results = retrieve_for_question(TWO_ENTITY_QUESTION, retriever, top_k=6, entity_fanout_enabled=True)

    assert [c.chunk_id for c in results] == ["v1", "shared", "m1"]


def test_merged_results_are_capped_at_max_top_k():
    def responder(question):
        prefix = question.split()[0]
        return [make_chunk("x", f"{prefix}.pdf", 1, score=0.9 - i * 0.01, chunk_id=f"{prefix}-{i}") for i in range(MAX_TOP_K)]

    retriever = RecordingRetriever(responder)
    results = retrieve_for_question(TWO_ENTITY_QUESTION, retriever, top_k=6, entity_fanout_enabled=True)

    assert len(results) == MAX_TOP_K


def test_fan_out_disabled_or_single_entity_uses_one_plain_diverse_search():
    retriever = RecordingRetriever(lambda q: [])
    retrieve_for_question(TWO_ENTITY_QUESTION, retriever, top_k=6, entity_fanout_enabled=False)
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["question"] == TWO_ENTITY_QUESTION and retriever.calls[0]["top_k"] == 6

    retriever = RecordingRetriever(lambda q: [])
    retrieve_for_question("What are the skills of Manas?", retriever, top_k=6, entity_fanout_enabled=True)
    assert len(retriever.calls) == 1 and retriever.calls[0]["top_k"] == 6
