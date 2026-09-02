"""
Document registry for DocuRAG.

FaissVectorStore and MetadataStore both operate at the *chunk* level. This
module tracks state at the *document* level:

    - which documents are already indexed, keyed by content hash (so the
      same PDF uploaded twice — even under a different filename — is
      detected and skipped rather than re-embedded)
    - which vector IDs belong to which document (so deleting or
      re-indexing a document means "remove exactly these IDs," not a full
      rebuild)
    - a single monotonically increasing counter that hands out the int64
      vector IDs FaissVectorStore and MetadataStore use, so IDs are never
      reused even after documents are deleted
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentEntry:
    doc_id: str
    source_filename: str
    content_hash: str
    page_count: int
    chunk_count: int
    vector_ids: List[int]
    indexed_at: str


@dataclass
class DocumentRegistry:
    documents: Dict[str, DocumentEntry] = field(default_factory=dict)  # doc_id -> entry
    next_vector_id: int = 0

    # --- lookups -----------------------------------------------------------
    def find_by_hash(self, content_hash: str) -> Optional[DocumentEntry]:
        for entry in self.documents.values():
            if entry.content_hash == content_hash:
                return entry
        return None

    def find_by_filename(self, filename: str) -> Optional[DocumentEntry]:
        for entry in self.documents.values():
            if entry.source_filename == filename:
                return entry
        return None

    def list_documents(self) -> List[DocumentEntry]:
        return sorted(self.documents.values(), key=lambda e: e.source_filename)

    # --- mutation -----------------------------------------------------------
    def allocate_vector_ids(self, n: int) -> List[int]:
        ids = list(range(self.next_vector_id, self.next_vector_id + n))
        self.next_vector_id += n
        return ids

    def add_document(
        self,
        doc_id: str,
        source_filename: str,
        content_hash: str,
        page_count: int,
        chunk_count: int,
        vector_ids: List[int],
    ) -> DocumentEntry:
        entry = DocumentEntry(
            doc_id=doc_id,
            source_filename=source_filename,
            content_hash=content_hash,
            page_count=page_count,
            chunk_count=chunk_count,
            vector_ids=vector_ids,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.documents[doc_id] = entry
        logger.info(
            "Registered document '%s' (doc_id=%s, %d chunks, %d pages)",
            source_filename, doc_id, chunk_count, page_count,
        )
        return entry

    def remove_document(self, doc_id: str) -> List[int]:
        """Remove a document from the registry. Returns its vector IDs so
        the caller can remove them from FaissVectorStore and MetadataStore."""
        entry = self.documents.pop(doc_id, None)
        if entry is None:
            return []
        logger.info(
            "Removed document '%s' (doc_id=%s) from registry, freeing %d vectors",
            entry.source_filename, doc_id, len(entry.vector_ids),
        )
        return entry.vector_ids

    # --- persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_vector_id": self.next_vector_id,
            "documents": [asdict(e) for e in self.documents.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved document registry (%d documents) to %s", len(self.documents), path)

    @classmethod
    def load(cls, path: Path) -> "DocumentRegistry":
        if not path.exists():
            raise FileNotFoundError(f"No document registry found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        registry = cls(next_vector_id=payload.get("next_vector_id", 0))
        for raw in payload.get("documents", []):
            entry = DocumentEntry(**raw)
            registry.documents[entry.doc_id] = entry
        logger.info("Loaded document registry (%d documents) from %s", len(registry.documents), path)
        return registry
