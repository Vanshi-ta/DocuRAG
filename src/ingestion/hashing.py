

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024                                                        


def compute_file_hash(file_path: Path) -> str:
    
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(_CHUNK_SIZE)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()
