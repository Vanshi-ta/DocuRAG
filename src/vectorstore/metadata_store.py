"""
Metadata store module for DocuRAG.

FAISS (via IndexIDMap2) returns caller-assigned int64 vector IDs from a
search — never text. This module keeps a dict of {vector_id: ChunkRecord}
so a search hit turns into "that means chunk X of handbook.pdf, page 7."

Keyed by ID rather than position (as an earlier version of this module was)
specifically because IDs are stable across deletions — removing document A's
three chunks does not renumber document B's chunks, so no desync is
possible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """One metadata record, keyed by its vector_id (the same ID used in FAISS)."""

    vector_id: int
    chunk_id: str
    chunk_text: str
    source_filename: str
    doc_id: str
    page_number: int
    chunk_index: int


class MetadataStore:
    def __init__(self):
        self.records: Dict[int, ChunkRecord] = {}

    def add_records(self, records: Iterable[ChunkRecord]) -> None:
        for record in records:
            self.records[record.vector_id] = record

    def remove(self, vector_ids: Iterable[int]) -> int:
        """Remove records for the given vector IDs. Returns count removed."""
        removed = 0
        for vid in vector_ids:
            if vid in self.records:
                del self.records[vid]
                removed += 1
        return removed

    def get(self, vector_id: int) -> ChunkRecord:
        if vector_id not in self.records:
            raise KeyError(
                f"vector_id {vector_id} not found in metadata store "
                f"(size {len(self.records)}) — this means the FAISS index "
                f"and metadata store have gone out of sync."
            )
        return self.records[vector_id]

    def __len__(self) -> int:
        return len(self.records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records.values()], f, ensure_ascii=False, indent=2)
        logger.info("Saved %d metadata records to %s", len(self.records), path)

    @classmethod
    def load(cls, path: Path) -> "MetadataStore":
        if not path.exists():
            raise FileNotFoundError(f"No metadata store found at {path}")
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        store.records = {r["vector_id"]: ChunkRecord(**r) for r in raw_records}
        logger.info("Loaded %d metadata records from %s", len(store.records), path)
        return store
