"""
Metadata store module for DocuRAG — Phase 4.

FAISS returns only integer positions when you search it — it never gives
you back text or metadata. This module keeps an ORDERED list of metadata
records, one per vector, such that FAISS position `i` always corresponds to
`MetadataStore.records[i]`. This ordered-list mapping is the entire
mechanism that turns "FAISS found vector #42" into "that means chunk 42 of
handbook.pdf, page 7." There is no cleverness beyond keeping the two lists
in the same order — which is exactly why it's critical that vectors and
records are always added together, from the same ordered list of chunks
(see index_builder.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass
class ChunkRecord:
    """
    One metadata record, corresponding 1:1 with one vector at the same
    position in the FAISS index. Every field here is something a later
    phase (retrieval, source citation) needs and cannot get from FAISS
    itself.
    """

    chunk_id: str
    chunk_text: str
    source_filename: str
    doc_id: str
    page_number: int
    chunk_index: int


class MetadataStore:
    def __init__(self):
        self.records: List[ChunkRecord] = []

    def add_records(self, records: List[ChunkRecord]) -> None:
        """
        Append records. MUST be called with records in the exact same order
        their corresponding vectors were added to FAISS in the same call —
        position i here must always describe FAISS vector i.
        """
        self.records.extend(records)

    def get(self, faiss_index: int) -> ChunkRecord:
        """Look up the metadata record for a given FAISS position."""
        if faiss_index < 0 or faiss_index >= len(self.records):
            raise IndexError(
                f"FAISS index {faiss_index} out of range for metadata store "
                f"of size {len(self.records)} — this means the FAISS index "
                f"and metadata store have gone out of sync, which should "
                f"never happen if they were built and saved together."
            )
        return self.records[faiss_index]

    def __len__(self) -> int:
        return len(self.records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, ensure_ascii=False, indent=2)
        logger.info("Saved %d metadata records to %s", len(self.records), path)

    @classmethod
    def load(cls, path: Path) -> "MetadataStore":
        if not path.exists():
            raise FileNotFoundError(f"No metadata store found at {path}")
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        store.records = [ChunkRecord(**r) for r in raw_records]
        logger.info("Loaded %d metadata records from %s", len(store.records), path)
        return store
