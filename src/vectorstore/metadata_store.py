

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    

    vector_id: int
    chunk_id: str
    chunk_text: str
    source_filename: str
    doc_id: str
    page_number: int
    chunk_index: int


class MetadataStore:
    def __init__(self):
        self.records: dict[int, ChunkRecord] = {}

    def add_records(self, records: Iterable[ChunkRecord]) -> None:
        for record in records:
            self.records[record.vector_id] = record

    def remove(self, vector_ids: Iterable[int]) -> int:
        
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
    def load(cls, path: Path) -> MetadataStore:
        if not path.exists():
            raise FileNotFoundError(f"No metadata store found at {path}")
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        store.records = {r["vector_id"]: ChunkRecord(**r) for r in raw_records}
        logger.info("Loaded %d metadata records from %s", len(store.records), path)
        return store
