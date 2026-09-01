"""
Automated test for src/pipeline.py's reset_all(), used by the Streamlit
"Reset index" button (Phase 7).

Run from the project root with:
    pytest tests/test_pipeline_reset.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src.pipeline import reset_all


def test_reset_all_removes_pdfs_and_persisted_store(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "a.pdf").write_bytes(b"fake pdf content")
    (upload_dir / "b.pdf").write_bytes(b"fake pdf content")
    (upload_dir / "notes.txt").write_bytes(b"not a pdf")  # should be left alone

    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir()
    fake_index_path = vector_store_dir / "index.faiss"
    fake_metadata_path = vector_store_dir / "metadata.json"
    fake_index_path.write_bytes(b"fake index")
    fake_metadata_path.write_text("[]")

    # Point the module's persisted-file constants at our temp files for
    # the duration of this test only.
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", fake_index_path)
    monkeypatch.setattr(config, "METADATA_STORE_PATH", fake_metadata_path)
    monkeypatch.setattr("src.pipeline.FAISS_INDEX_PATH", fake_index_path)
    monkeypatch.setattr("src.pipeline.METADATA_STORE_PATH", fake_metadata_path)

    reset_all(directory=upload_dir)

    assert not (upload_dir / "a.pdf").exists()
    assert not (upload_dir / "b.pdf").exists()
    assert (upload_dir / "notes.txt").exists()  # non-PDF untouched
    assert not fake_index_path.exists()
    assert not fake_metadata_path.exists()


def test_reset_all_does_not_raise_when_nothing_exists_yet(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty_uploads"
    empty_dir.mkdir()

    missing_index_path = tmp_path / "does_not_exist" / "index.faiss"
    missing_metadata_path = tmp_path / "does_not_exist" / "metadata.json"

    monkeypatch.setattr("src.pipeline.FAISS_INDEX_PATH", missing_index_path)
    monkeypatch.setattr("src.pipeline.METADATA_STORE_PATH", missing_metadata_path)

    reset_all(directory=empty_dir)  # should simply do nothing, not raise
