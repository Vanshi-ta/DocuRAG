import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.hashing import compute_file_hash


def test_identical_content_same_hash_different_filenames(tmp_path):
    content = b"%PDF-1.4 fake pdf bytes for hashing test"
    f1 = tmp_path / "report.pdf"
    f2 = tmp_path / "report_copy.pdf"
    f1.write_bytes(content)
    f2.write_bytes(content)

    assert compute_file_hash(f1) == compute_file_hash(f2)


def test_different_content_different_hash(tmp_path):
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(b"content A")
    f2.write_bytes(b"content B")

    assert compute_file_hash(f1) != compute_file_hash(f2)


def test_hash_changes_when_file_content_changes(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"version 1")
    h1 = compute_file_hash(f)
    f.write_bytes(b"version 2")
    h2 = compute_file_hash(f)

    assert h1 != h2
