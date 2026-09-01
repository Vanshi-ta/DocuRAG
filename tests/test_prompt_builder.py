"""
Automated tests for src/generation/prompt_builder.py.

Pure text-in/text-out logic — no LLM, no vector store, no network needed.

Run from the project root with:
    pytest tests/test_prompt_builder.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.prompt_builder import SYSTEM_INSTRUCTIONS, build_prompt, format_context
from src.retrieval.retriever import RetrievedChunk


def make_chunk(text: str, filename: str = "doc.pdf", page: int = 1, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_text=text,
        source_filename=filename,
        page_number=page,
        similarity_score=score,
        chunk_id="c1",
    )


def test_format_context_handles_empty_list():
    assert "no relevant context" in format_context([]).lower()


def test_format_context_labels_each_chunk_with_source_and_page():
    chunks = [make_chunk("Text A", "a.pdf", 1), make_chunk("Text B", "b.pdf", 5)]
    context = format_context(chunks)

    assert "a.pdf" in context and "page 1" in context
    assert "b.pdf" in context and "page 5" in context
    assert "Text A" in context and "Text B" in context


def test_build_prompt_includes_instructions_context_and_question():
    chunks = [make_chunk("The meeting is at 3pm.")]
    prompt = build_prompt("When is the meeting?", chunks)

    assert SYSTEM_INSTRUCTIONS.strip() in prompt
    assert "The meeting is at 3pm." in prompt
    assert "When is the meeting?" in prompt
    assert prompt.strip().endswith("Answer:")


def test_build_prompt_with_no_chunks_still_produces_a_valid_prompt():
    prompt = build_prompt("Any question", [])

    assert "no relevant context" in prompt.lower()
    assert "Any question" in prompt


def test_system_instructions_cover_all_four_required_rules():
    # Not a test of behavior, but a guard against accidentally weakening
    # the prompt later: confirms the required grounding rules are present.
    lowered = SYSTEM_INSTRUCTIONS.lower()
    assert "only" in lowered and "context" in lowered          # use only retrieved context
    assert "invent" in lowered or "guess" in lowered            # do not invent facts
    assert "could not find" in lowered                          # explicit "not found" phrase
    assert "concise" in lowered                                 # answer concisely
