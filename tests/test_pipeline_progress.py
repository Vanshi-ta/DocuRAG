"""Tests for the UI-facing additions to src/pipeline.py: progress callbacks,
per-file results, `paths=`, and failed re-index leaving the old version intact.
Uses the same real stores + fake embedder approach as test_pipeline_integration.py."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from reportlab.pdfgen import canvas

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src.pipeline import run_full_ingestion_pipeline


class DeterministicFakeEmbedder:
    embedding_dimension = 16

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).random(self.embedding_dimension, dtype=np.float64)
            vectors.append((v / np.linalg.norm(v)).astype("float32"))
        return np.vstack(vectors) if vectors else np.empty((0, self.embedding_dimension), dtype="float32")

    def embed_query(self, text):
        return self.embed_texts([text])


def make_pdf(path: Path, lines):
    c = canvas.Canvas(str(path))
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    store_dir = tmp_path / "vector_store"
    upload_dir.mkdir()
    store_dir.mkdir()
    for prefix in ("config", "src.pipeline"):
        monkeypatch.setattr(f"{prefix}.UPLOAD_DIR", upload_dir)
        monkeypatch.setattr(f"{prefix}.FAISS_INDEX_PATH", store_dir / "index.faiss")
        monkeypatch.setattr(f"{prefix}.METADATA_STORE_PATH", store_dir / "metadata.json")
        monkeypatch.setattr(f"{prefix}.DOCUMENT_REGISTRY_PATH", store_dir / "documents.json")
    return upload_dir


def test_progress_events_and_file_results(isolated_dirs):
    make_pdf(isolated_dirs / "a.pdf", ["Alpha document text."])
    make_pdf(isolated_dirs / "b.pdf", ["Beta document text."])
    events = []

    result, *_ = run_full_ingestion_pipeline(
        embedder=DeterministicFakeEmbedder(), progress_callback=events.append
    )

    a_stages = [e.stage for e in events if e.filename == "a.pdf"]
    assert a_stages == ["hashing", "loading", "chunking", "embedding", "indexed"]
    assert {e.total_files for e in events} == {2}
    assert [e.file_index for e in events if e.filename == "b.pdf"][0] == 2
    assert [(r.filename, r.status) for r in result.file_results] == [("a.pdf", "indexed"), ("b.pdf", "indexed")]
    assert all(r.chunks_added > 0 and r.page_count == 1 for r in result.file_results)


def test_callback_exception_does_not_abort_ingestion(isolated_dirs):
    make_pdf(isolated_dirs / "a.pdf", ["Alpha document text."])

    def boom(event):
        raise RuntimeError("UI bug")

    result, *_ = run_full_ingestion_pipeline(embedder=DeterministicFakeEmbedder(), progress_callback=boom)
    assert result.indexed_files == ["a.pdf"]


def test_paths_argument_ingests_only_those_files(isolated_dirs):
    make_pdf(isolated_dirs / "a.pdf", ["Alpha document text."])
    make_pdf(isolated_dirs / "b.pdf", ["Beta document text."])

    result, _, _, registry = run_full_ingestion_pipeline(
        embedder=DeterministicFakeEmbedder(), paths=[isolated_dirs / "b.pdf"]
    )

    assert result.indexed_files == ["b.pdf"]
    assert [d.source_filename for d in registry.list_documents()] == ["b.pdf"]


def test_duplicate_is_reported_with_status(isolated_dirs):
    make_pdf(isolated_dirs / "a.pdf", ["Alpha document text."])
    embedder = DeterministicFakeEmbedder()
    run_full_ingestion_pipeline(embedder=embedder)

    result, *_ = run_full_ingestion_pipeline(embedder=embedder)
    assert result.file_results[0].status == "skipped_duplicate"


def test_failed_reindex_keeps_previous_version_searchable(isolated_dirs):
    pdf = isolated_dirs / "policy.pdf"
    make_pdf(pdf, ["The refund window is 30 days."])
    embedder = DeterministicFakeEmbedder()
    _, faiss_store, _, registry = run_full_ingestion_pipeline(embedder=embedder)
    old_chunks = faiss_store.ntotal
    old_doc_id = registry.find_by_filename("policy.pdf").doc_id

    pdf.write_bytes(b"this is not a pdf at all")  # same name, new (broken) content
    result, faiss_store2, metadata_store2, registry2 = run_full_ingestion_pipeline(embedder=embedder)

    assert result.failed_files and result.failed_files[0][0] == "policy.pdf"
    assert result.file_results[0].status == "failed"
    assert result.file_results[0].error_type  # e.g. CorruptedPDFError
    assert registry2.find_by_filename("policy.pdf").doc_id == old_doc_id
    assert faiss_store2.ntotal == old_chunks == len(metadata_store2)
