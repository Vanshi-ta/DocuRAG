import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.text_splitter import split_documents, validate_chunk_params


def make_document(text, page_number=1, source_filename="test.pdf"):
    return Document(
        page_content=text,
        metadata={"source_filename": source_filename, "doc_id": "fake-doc-id",
                  "page_number": page_number, "page_raw": page_number - 1},
    )


def test_validate_chunk_params_rejects_overlap_larger_than_size():
    with pytest.raises(ValueError, match="must be smaller"):
        validate_chunk_params(chunk_size=500, chunk_overlap=600)


def test_validate_chunk_params_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="must be positive"):
        validate_chunk_params(chunk_size=0, chunk_overlap=0)


def test_split_documents_produces_more_chunks_than_input_for_long_text():
    long_text = "This is a sentence about RAG systems. " * 200
    doc = make_document(long_text)
    result = split_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(result.chunks) > 1


def test_split_documents_keeps_short_text_as_one_chunk():
    short_text = "This PDF page has very little text on it."
    doc = make_document(short_text)
    result = split_documents([doc], chunk_size=1000, chunk_overlap=150)
    assert len(result.chunks) == 1
    assert result.chunks[0].page_content == short_text


def test_chunks_inherit_original_metadata():
    doc = make_document("Some content. " * 100, page_number=3, source_filename="handbook.pdf")
    result = split_documents([doc], chunk_size=200, chunk_overlap=30)
    for chunk in result.chunks:
        assert chunk.metadata["source_filename"] == "handbook.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["doc_id"] == "fake-doc-id"


def test_chunks_get_unique_chunk_ids():
    doc = make_document("Some content. " * 100)
    result = split_documents([doc], chunk_size=200, chunk_overlap=30)
    chunk_ids = {c.metadata["chunk_id"] for c in result.chunks}
    assert len(chunk_ids) == len(result.chunks)
