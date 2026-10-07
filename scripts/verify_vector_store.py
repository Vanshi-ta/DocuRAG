"""
Verification script for the persisted vector store.

Builds the FAISS index + metadata store + document registry from whatever
PDFs are in data/uploads/, persists them to disk, then reloads them from disk
as a SEPARATE, fresh set of objects (simulating an app restart) and confirms
everything still matches. Also runs one sample similarity search on the
reloaded store.

Note: ingestion is incremental, so PDFs that are already indexed (same
content) are skipped rather than re-embedded; "this run" counts below only
cover files that were newly indexed or updated.

Run from the project root with:
    python scripts/verify_vector_store.py
"""

import _common  # noqa: F401  (makes the project root importable)

from config import DOCUMENT_REGISTRY_PATH, FAISS_INDEX_PATH, METADATA_STORE_PATH
from src.ingestion.embedder import Embedder
from src.pipeline import load_vector_store, run_full_ingestion_pipeline


def main() -> None:
    embedder = Embedder()

    print("Step 1: Building the vector store from data/uploads/ ...")
    run_result, faiss_store, metadata_store, registry = run_full_ingestion_pipeline(
        embedder=embedder, persist=True
    )

    if faiss_store.ntotal == 0:
        print("No chunks were embedded — add a PDF to data/uploads/ first.")
        return

    print(f"  Documents indexed this run : {len(run_result.indexed_files) + len(run_result.reindexed_files)}")
    print(f"  Chunks added this run      : {run_result.total_chunks_added}")
    print(f"  Documents in registry      : {len(registry.documents)}")
    print(f"  Vectors in FAISS           : {faiss_store.ntotal}")
    print(f"  Embedding dimension        : {faiss_store.embedding_dimension}")
    print(f"  Metadata records           : {len(metadata_store)}")
    assert faiss_store.ntotal == len(metadata_store), "OUT OF SYNC before save!"

    print(
        "\nStep 2: Persisted to:"
        f"\n  {FAISS_INDEX_PATH}\n  {METADATA_STORE_PATH}\n  {DOCUMENT_REGISTRY_PATH}"
    )

    print("\nStep 3: Reloading from disk as fresh objects (simulating a restart)...")
    reloaded_faiss_store, reloaded_metadata_store, reloaded_registry = load_vector_store(
        embedding_dimension=faiss_store.embedding_dimension
    )

    assert reloaded_faiss_store.ntotal == faiss_store.ntotal, (
        "Reloaded FAISS index has a different vector count than before saving!"
    )
    assert len(reloaded_metadata_store) == len(metadata_store), (
        "Reloaded metadata store has a different record count than before saving!"
    )
    assert len(reloaded_registry.documents) == len(registry.documents), (
        "Reloaded document registry has a different document count than before saving!"
    )
    print(f"  Reloaded vectors   : {reloaded_faiss_store.ntotal}  (matches ✓)")
    print(f"  Reloaded records   : {len(reloaded_metadata_store)}  (matches ✓)")
    print(f"  Reloaded documents : {len(reloaded_registry.documents)}  (matches ✓)")

    print("\nStep 4: Running one sample similarity search on the RELOADED store...")
    sample_query = "What is this document about?"
    query_vector = embedder.embed_query(sample_query)

    scores, indices = reloaded_faiss_store.search(query_vector, top_k=3)

    print(f"  Query: {sample_query!r}")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        record = reloaded_metadata_store.get(int(idx))
        preview = record.chunk_text[:120].replace("\n", " ")
        print(
            f"  #{rank}  score={score:.3f}  "
            f"{record.source_filename} (page {record.page_number})  "
            f"-> {preview}..."
        )

    print("\nAll checks passed — the vector store persists and reloads correctly.")


if __name__ == "__main__":
    main()
