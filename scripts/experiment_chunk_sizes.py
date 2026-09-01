"""
Experiment: compare chunking behavior across several chunk_size /
chunk_overlap combinations, on whatever real PDFs you have in
data/uploads/.

Run from the project root with:
    python scripts/experiment_chunk_sizes.py

This does NOT change your config.py defaults — it's a read-only experiment
so you can visually compare outcomes before deciding on final values.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import UPLOAD_DIR
from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.text_splitter import split_documents

# Each tuple is (chunk_size, chunk_overlap). Feel free to add/remove rows.
EXPERIMENTS = [
    (300, 30),     # too small — see Phase 3 guide for what to expect
    (500, 75),
    (1000, 150),   # current recommended default for technical PDFs
    (2000, 200),
    (4000, 300),   # too large — see Phase 3 guide for what to expect
]


def main() -> None:
    ingestion_result = load_pdfs_from_directory(UPLOAD_DIR)
    if not ingestion_result.documents:
        print("No documents were loaded — add a PDF to data/uploads/ first.")
        return

    print(f"Source: {len(ingestion_result.documents)} page-level Documents "
          f"from {len(ingestion_result.loaded_files)} file(s)\n")

    print(f"{'chunk_size':>10} | {'overlap':>7} | {'#chunks':>7} | "
          f"{'min len':>7} | {'max len':>7} | {'avg len':>7}")
    print("-" * 60)

    for chunk_size, chunk_overlap in EXPERIMENTS:
        result = split_documents(
            ingestion_result.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        lengths = [len(c.page_content) for c in result.chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        print(
            f"{chunk_size:>10} | {chunk_overlap:>7} | {len(result.chunks):>7} | "
            f"{min(lengths, default=0):>7} | {max(lengths, default=0):>7} | "
            f"{avg_len:>7.0f}"
        )


if __name__ == "__main__":
    main()
