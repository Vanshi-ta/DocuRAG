import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_loader import (
    EmptyFileError,
    CorruptedPDFError,
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
