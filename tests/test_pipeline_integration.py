"""
Integration test for src/pipeline.py covering the three riskiest ingestion
behaviors: duplicate-content detection, re-indexing on content
change, and clean document deletion — exercised against the REAL
FaissVectorStore, MetadataStore, and DocumentRegistry, with only the
embedding model swapped for a fast deterministic fake (hashing text into a
vector), so this test runs in milliseconds without downloading a real
Sentence Transformers model.
"""

from src.pipeline import remove_document, run_full_ingestion_pipeline
from tests.helpers import DeterministicFakeEmbedder, make_pdf


def test_ingest_two_documents_then_duplicate_upload_is_skipped(isolated_dirs):
    upload_dir = isolated_dirs
    make_pdf(upload_dir / "policy.pdf", ["The refund window is 30 days."])
    make_pdf(upload_dir / "faq.pdf", ["Support is available 24/7 via chat."])

    embedder = DeterministicFakeEmbedder()
    result, faiss_store, metadata_store, registry = run_full_ingestion_pipeline(embedder=embedder)

    assert set(result.indexed_files) == {"policy.pdf", "faq.pdf"}
    assert faiss_store.ntotal == len(metadata_store) == result.total_chunks_added
    assert len(registry.documents) == 2

    # Re-run ingestion on the exact same directory (same files, same bytes):
    # both should now be recognized as duplicates and skipped, not re-embedded.
    result2, faiss_store2, metadata_store2, registry2 = run_full_ingestion_pipeline(embedder=embedder)
    assert set(result2.skipped_duplicate_files) == {"policy.pdf", "faq.pdf"}
    assert result2.indexed_files == []
    assert result2.total_chunks_added == 0
    assert faiss_store2.ntotal == faiss_store.ntotal  # unchanged


def test_content_change_triggers_reindex_not_duplication(isolated_dirs):
    upload_dir = isolated_dirs
    pdf_path = upload_dir / "policy.pdf"
    make_pdf(pdf_path, ["The refund window is 30 days."])

    embedder = DeterministicFakeEmbedder()
    result1, faiss_store1, metadata_store1, registry1 = run_full_ingestion_pipeline(embedder=embedder)
    original_chunk_count = faiss_store1.ntotal
    original_doc_id = registry1.find_by_filename("policy.pdf").doc_id

    # Overwrite same filename with different content.
    make_pdf(pdf_path, ["The refund window is now 45 days, updated policy."])

    result2, faiss_store2, metadata_store2, registry2 = run_full_ingestion_pipeline(embedder=embedder)

    assert result2.reindexed_files == ["policy.pdf"]
    assert result2.skipped_duplicate_files == []
    # Still exactly one document registered for this filename — no duplication.
    assert len([d for d in registry2.documents.values() if d.source_filename == "policy.pdf"]) == 1
    new_doc_id = registry2.find_by_filename("policy.pdf").doc_id
    assert new_doc_id != original_doc_id  # it's a genuinely new document record
    # Old vectors are gone, new ones took their place (not both).
    assert faiss_store2.ntotal == original_chunk_count
    assert len(metadata_store2) == original_chunk_count


def test_remove_document_deletes_only_its_own_vectors(isolated_dirs):
    upload_dir = isolated_dirs
    make_pdf(upload_dir / "policy.pdf", ["The refund window is 30 days."])
    make_pdf(upload_dir / "faq.pdf", ["Support is available 24/7 via chat."])

    embedder = DeterministicFakeEmbedder()
    result, faiss_store, metadata_store, registry = run_full_ingestion_pipeline(embedder=embedder)

    policy_doc = registry.find_by_filename("policy.pdf")
    faq_doc = registry.find_by_filename("faq.pdf")
    total_before = faiss_store.ntotal

    n_removed = remove_document(policy_doc.doc_id, faiss_store, metadata_store, registry)

    assert n_removed == len(policy_doc.vector_ids)
    assert faiss_store.ntotal == total_before - n_removed
    assert len(metadata_store) == total_before - n_removed
    assert registry.find_by_filename("policy.pdf") is None
    # faq.pdf's chunks are untouched.
    for vid in faq_doc.vector_ids:
        assert metadata_store.get(vid).source_filename == "faq.pdf"
