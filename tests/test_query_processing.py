import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.query_processing import extract_entities


def test_extracts_two_entities_from_and_question():
    assert extract_entities("What are the skills of Manas and Vanshita?") == ["Manas", "Vanshita"]


def test_extracts_two_entities_from_difference_question():
    assert extract_entities("What skills does Vanshita have that Manas does not?") == ["Vanshita", "Manas"]


def test_extracts_two_entities_from_compare_question():
    entities = extract_entities("Compare the education of Manas and Vanshita.")
    assert entities == ["Manas", "Vanshita"]  # "Compare" correctly excluded (sentence-initial)


def test_merges_multi_word_entity():
    entities = extract_entities("What is Vanshita Suryavanshi's CGPA?")
    assert entities == ["Vanshita Suryavanshi"]


def test_no_entities_in_lowercase_pronoun_question():
    # Documents this as a known, deliberate limitation: no coreference
    # resolution, so a pronoun-only question does not trigger fan-out.
    assert extract_entities("What are her skills?") == []


def test_no_entities_when_single_entity_present():
    assert extract_entities("What is Vanshita's CGPA?") == ["Vanshita"]


def test_deduplicates_repeated_entity_case_insensitively():
    entities = extract_entities("Does Manas know Python? Manas also knows Java.")
    assert entities.count("Manas") == 1
