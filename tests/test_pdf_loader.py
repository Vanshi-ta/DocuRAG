"""
Automated tests for src/ingestion/pdf_loader.py.

Run from the project root with:
    pytest tests/test_pdf_loader.py -v

Most of these tests are dependency-free (no PDF file required) and check
the validation/error-handling logic. The last test only runs if you've
placed at least one real PDF in data/uploads/, since we can't generate a
real PDF's binary content without an extra library.
"""

import sys
from pathlib import Path

import pytest

# Allow `pytest` to find `config` and `src` when run from the project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import UPLOAD_DIR
from src.ingestion.pdf_loader import (
    load_pdfs_from_directory,
    load_single_pdf,
    validate_pdf_path,
)


def test_validate_pdf_path_raises_on_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        validate_pdf_path(missing_file)


def test_validate_pdf_path_raises_on_wrong_extension(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("this is not a pdf")
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_pdf_path(text_file)


def test_validate_pdf_path_raises_on_directory(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        validate_pdf_path(tmp_path)


def test_load_pdfs_from_directory_raises_on_missing_directory(tmp_path):
    missing_dir = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError):
        load_pdfs_from_directory(missing_dir)


def test_load_pdfs_from_directory_handles_empty_directory(tmp_path):
    # An empty (but existing) directory should return an empty result,
    # not raise — no PDFs is a valid, if unhelpful, state.
    result = load_pdfs_from_directory(tmp_path)
    assert result.documents == []
    assert result.loaded_files == []
    assert result.skipped_files == []


@pytest.mark.skipif(
    not any(UPLOAD_DIR.glob("*.pdf")),
    reason="No PDF found in data/uploads/ — add a real PDF to run this test.",
)
def test_load_single_pdf_on_a_real_sample():
    sample_pdf = next(UPLOAD_DIR.glob("*.pdf"))
    pages = load_single_pdf(sample_pdf)

    assert len(pages) > 0
    first_page = pages[0]

    # Every page must carry the metadata later phases (chunking, citations)
    # depend on.
    assert first_page.metadata["source_filename"] == sample_pdf.name
    assert "doc_id" in first_page.metadata
    assert first_page.metadata["page_number"] == 1  # 1-indexed, not 0
    assert first_page.metadata["page_raw"] == 0

    # All pages of the same document must share the same doc_id.
    doc_ids = {page.metadata["doc_id"] for page in pages}
    assert len(doc_ids) == 1
