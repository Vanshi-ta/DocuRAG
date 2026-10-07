"""Smoke tests for DocuRAGService with a fake embedder and fake LLM client."""


import pytest


from src.errors import EmptyQuestionError, IndexNotReadyError
from src.services.docurag_service import DocuRAGService
from tests.helpers import DeterministicFakeEmbedder, make_pdf


class FakeLLM:
    def generate(self, prompt):
        return "fake answer"

    def generate_stream(self, prompt):
        return iter(["fake ", "answer"])

    def health_check(self):
        return {"reachable": True, "model": "fake", "model_available": True, "available_models": ["fake"]}


def pdf_bytes(tmp_path, text):
    path = tmp_path / "tmp.pdf"
    make_pdf(path, [text])
    return path.read_bytes()


@pytest.fixture
def service(isolated_dirs):
    return DocuRAGService(embedder=DeterministicFakeEmbedder(), llm_client=FakeLLM(), upload_dir=isolated_dirs)


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

    fresh = DocuRAGService(embedder=DeterministicFakeEmbedder(), llm_client=FakeLLM(), upload_dir=service.upload_dir)
    assert fresh.load_existing() is True
    assert len(fresh.list_documents()) == 1


def test_ask_passes_similarity_threshold_only_when_requested(service, tmp_path, monkeypatch):
    from config import SIMILARITY_THRESHOLD_VALUE

    saved, _ = service.save_uploads({"policy.pdf": pdf_bytes(tmp_path, "The refund window is 30 days.")})
    service.ingest(paths=saved)

    seen = []

    def fake_answer_question(question, retriever, llm_client, top_k, similarity_threshold):
        seen.append(similarity_threshold)

    monkeypatch.setattr("src.services.docurag_service.answer_question", fake_answer_question)

    service.ask("What is the refund window?", use_threshold=False)
    service.ask("What is the refund window?", use_threshold=True)

    assert seen == [None, SIMILARITY_THRESHOLD_VALUE]
