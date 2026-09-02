"""
Content hashing for duplicate-document detection.

Hashing raw file bytes (not filename) means the same PDF uploaded twice
under different names is still recognized as a duplicate, and a file whose
content changed but keeps the same name is correctly recognized as *not*
a duplicate (triggering re-indexing instead of a silent skip).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MB read chunks, so large PDFs don't blow up memory


def compute_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(_CHUNK_SIZE)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()
