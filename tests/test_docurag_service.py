"""Smoke tests for DocuRAGService with a fake embedder and fake LLM client."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from reportlab.pdfgen import canvas

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.errors import EmptyQuestionError, IndexNotReadyError
from src.services.docurag_service import DocuRAGService


class FakeEmbedder:
    embedding_dimension = 16

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).random(self.embedding_dimension)
            out.append((v / np.linalg.norm(v)).astype("float32"))
        return np.vstack(out)

    def embed_query(self, t):
        return self.embed_texts([t])


class FakeLLM:
    def generate(self, prompt):
        return "fake answer"

    def generate_stream(self, prompt):
        return iter(["fake ", "answer"])

    def health_check(self):
        return {"reachable": True, "model": "fake", "model_available": True, "available_models": ["fake"]}


def pdf_bytes(tmp_path, text):
    p = tmp_path / "tmp.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(72, 750, text)
    c.showPage()
    c.save()
    return p.read_bytes()


@pytest.fixture
def service(tmp_path, monkeypatch):
    store_dir = tmp_path / "vector_store"
    store_dir.mkdir()
    for prefix in ("config", "src.pipeline"):
        monkeypatch.setattr(f"{prefix}.FAISS_INDEX_PATH", store_dir / "index.faiss")
        monkeypatch.setattr(f"{prefix}.METADATA_STORE_PATH", store_dir / "metadata.json")
        monkeypatch.setattr(f"{prefix}.DOCUMENT_REGISTRY_PATH", store_dir / "documents.json")
    return DocuRAGService(embedder=FakeEmbedder(), llm_client=FakeLLM(), upload_dir=tmp_path / "uploads")


def test_ask_before_indexing_raises_index_not_ready(service):
    with pytest.raises(IndexNotReadyError):
        service.ask("anything?")


def test_empty_question_raises(service):
    with pytest.raises(EmptyQuestionError):
        service.ask("   ")


def test_save_uploads_sanitizes_names_and_rejects_non_pdf(service, tmp_path):
    saved, rejected = service.save_uploads(
        {"../../evil.pdf": pdf_bytes(tmp_path, "x"), "notes.txt": b"hi"}
    )
    assert [p.name for p in saved] == ["evil.pdf"]
    assert saved[0].parent == service.upload_dir
    assert rejected == ["notes.txt"]


def test_full_flow_upload_ingest_ask_stream_delete_reset(service, tmp_path):
    saved, _ = service.save_uploads({"policy.pdf": pdf_bytes(tmp_path, "The refund window is 30 days.")})
    events = []
    result = service.ingest(paths=saved, progress_callback=events.append)

    assert result.indexed_files == ["policy.pdf"]
    assert events and events[-1].stage == "indexed"
    status = service.status()
    assert status.index_loaded and status.document_count == 1 and status.chunk_count > 0
    assert status.llm["reachable"] is True

    answer = service.ask("What is the refund window?", top_k=3)
    assert answer.answer == "fake answer" and answer.used_llm

    stream = service.ask_stream("What is the refund window?", top_k=3)
    assert "".join(stream.tokens) == "fake answer"
    assert stream.sources[0].source_filename == "policy.pdf"

    doc_id = service.list_documents()[0].doc_id
    assert service.delete_document(doc_id) > 0
    assert service.list_documents() == []
    with pytest.raises(IndexNotReadyError):
        service.ask("again?")

    service.reset()
    assert service.registry is None and not any(service.upload_dir.glob("*.pdf"))


def test_load_existing_restores_persisted_index(service, tmp_path):
    saved, _ = service.save_uploads({"a.pdf": pdf_bytes(tmp_path, "Alpha text here.")})
    service.ingest(paths=saved)

    fresh = DocuRAGService(embedder=FakeEmbedder(), llm_client=FakeLLM(), upload_dir=service.upload_dir)
    assert fresh.load_existing() is True
    assert len(fresh.list_documents()) == 1
