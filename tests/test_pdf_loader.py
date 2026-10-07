import pytest


from src.ingestion.pdf_loader import (
    EmptyFileError,
    CorruptedPDFError,
    NoExtractableTextError,
    load_pdfs_from_directory,
    load_single_pdf,
    validate_pdf_path,
)
from tests.helpers import make_pdf


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


def test_validate_pdf_path_raises_on_empty_file(tmp_path):
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        validate_pdf_path(empty_file)


def test_load_single_pdf_raises_corrupted_error_on_garbage_bytes(tmp_path):
    fake_pdf = tmp_path / "garbage.pdf"
    fake_pdf.write_bytes(b"this is not a real PDF structure at all")
    with pytest.raises(CorruptedPDFError):
        load_single_pdf(fake_pdf)


def test_load_pdfs_from_directory_raises_on_missing_directory(tmp_path):
    missing_dir = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError):
        load_pdfs_from_directory(missing_dir)


def test_load_pdfs_from_directory_handles_empty_directory(tmp_path):
    result = load_pdfs_from_directory(tmp_path)
    assert result.documents == []
    assert result.loaded_files == []
    assert result.skipped_files == []


def test_load_pdfs_from_directory_skips_one_bad_file_without_crashing(tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a real pdf")

    result = load_pdfs_from_directory(tmp_path)

    assert result.loaded_files == []
    assert len(result.skipped_files) == 1
    assert result.skipped_files[0][0] == "bad.pdf"


def test_load_single_pdf_returns_one_document_per_page_with_one_based_page_numbers(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_pdf(pdf_path, ["Hello from page one."])

    pages = load_single_pdf(pdf_path)

    assert len(pages) == 1
    assert "Hello from page one." in pages[0].page_content
    assert pages[0].metadata["page_number"] == 1


def test_load_single_pdf_raises_when_no_page_has_extractable_text(tmp_path):
    blank_pdf = tmp_path / "blank.pdf"
    make_pdf(blank_pdf, [])

    with pytest.raises(NoExtractableTextError):
        load_single_pdf(blank_pdf)
