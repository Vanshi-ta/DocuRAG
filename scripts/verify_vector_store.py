"""
Verification script for the Phase 4 vector store.

Builds the FAISS index + metadata store from whatever PDFs are in
data/uploads/, persists them to disk, then reloads them from disk as a
SEPARATE, fresh pair of objects (simulating an app restart) and confirms
everything still matches. Also runs one sample similarity search so you
can see retrieval working end-to-end, even though we haven't built the
retrieval module itself yet (that's Phase 5+).

Run from the project root with:
    python scripts/verify_vector_store.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import FAISS_INDEX_PATH, METADATA_STORE_PATH
from src.ingestion.embedder import Embedder
from src.pipeline import load_vector_store, run_full_ingestion_pipeline


def main() -> None:
    print("Step 1: Building the vector store from data/uploads/ ...")
    ingestion_result, chunking_result, faiss_store, metadata_store = (
        run_full_ingestion_pipeline(persist=True)
    )

    if faiss_store.index.ntotal == 0:
        print("No chunks were embedded — add a PDF to data/uploads/ first.")
        return

    print(f"  Documents ingested : {len(ingestion_result.loaded_files)}")
    print(f"  Chunks created      : {len(chunking_result.chunks)}")
    print(f"  Vectors in FAISS    : {faiss_store.index.ntotal}")
    print(f"  Embedding dimension : {faiss_store.embedding_dimension}")
    print(f"  Metadata records    : {len(metadata_store)}")
    assert faiss_store.index.ntotal == len(metadata_store), "OUT OF SYNC before save!"

    print(f"\nStep 2: Persisted to:\n  {FAISS_INDEX_PATH}\n  {METADATA_STORE_PATH}")

    print("\nStep 3: Reloading from disk as fresh objects (simulating a restart)...")
    reloaded_faiss_store, reloaded_metadata_store = load_vector_store(
        embedding_dimension=faiss_store.embedding_dimension
    )

    assert reloaded_faiss_store.index.ntotal == faiss_store.index.ntotal, (
        "Reloaded FAISS index has a different vector count than before saving!"
    )
    assert len(reloaded_metadata_store) == len(metadata_store), (
        "Reloaded metadata store has a different record count than before saving!"
    )
    print(f"  Reloaded vectors : {reloaded_faiss_store.index.ntotal}  (matches ✓)")
    print(f"  Reloaded records : {len(reloaded_metadata_store)}  (matches ✓)")

    print("\nStep 4: Running one sample similarity search on the RELOADED store...")
    embedder = Embedder()  # loaded once, reused for this one query
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
