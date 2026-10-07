from src.generation.prompt_builder import SYSTEM_INSTRUCTIONS, build_prompt, format_context
from tests.helpers import make_chunk


def test_format_context_handles_empty_list():
    assert "no relevant context" in format_context([]).lower()


def test_format_context_labels_each_chunk_with_source_and_page():
    chunks = [make_chunk("Text A", "a.pdf", 1), make_chunk("Text B", "b.pdf", 5)]
    context = format_context(chunks)
    assert "SOURCE: a.pdf" in context and "PAGE: 1" in context
    assert "SOURCE: b.pdf" in context and "PAGE: 5" in context


def test_build_prompt_includes_instructions_context_and_question():
    chunks = [make_chunk("The meeting is at 3pm.")]
    prompt = build_prompt("When is the meeting?", chunks)
    assert SYSTEM_INSTRUCTIONS.strip() in prompt
    assert "The meeting is at 3pm." in prompt
    assert "When is the meeting?" in prompt
    assert prompt.strip().endswith("Answer:")


def test_system_instructions_cover_required_rules():
    lowered = SYSTEM_INSTRUCTIONS.lower()
    assert "only" in lowered and "context" in lowered
    assert "invent" in lowered or "guess" in lowered
    assert "could not find" in lowered
    assert "concise" in lowered
