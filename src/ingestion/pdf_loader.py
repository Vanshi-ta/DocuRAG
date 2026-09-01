"""
PDF ingestion module for DocuRAG — Phase 2.

Responsible ONLY for:
    - discovering PDF files in a directory
    - validating that a path is a usable PDF
    - extracting per-page text using LangChain's PyPDFLoader (backed by pypdf)
    - normalizing and enriching each page's metadata

This module has NO knowledge of chunking, embeddings, FAISS, or Streamlit.
That boundary is intentional: this file should be fully testable and usable
from a plain Python script, a Jupyter notebook, or (later) a Streamlit app,
without ever importing `streamlit`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SUPPORTED_EXTENSIONS = {".pdf"}


@dataclass
class IngestionResult:
    """Summary of one directory-level ingestion run, for reporting/debugging."""

    documents: List[Document] = field(default_factory=list)
    loaded_files: List[str] = field(default_factory=list)
    skipped_files: List[Tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    empty_pages: List[Tuple[str, int]] = field(default_factory=list)    # (filename, page_number)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Loaded files   ({len(self.loaded_files)}): {self.loaded_files}",
            f"Skipped files  ({len(self.skipped_files)}): {self.skipped_files}",
            f"Empty pages    ({len(self.empty_pages)}): {self.empty_pages}",
            f"Total pages extracted: {len(self.documents)}",
            "=" * 60,
        ]
        return "\n".join(lines)


def validate_pdf_path(file_path: Path) -> None:
    """
    Raise a clear, specific exception if `file_path` is not a usable PDF path.

    This only checks the *filesystem-level* things: existence, that it's a
    file (not a directory), and that the extension is supported. It does NOT
    guarantee the PDF is actually readable — a file can look valid and still
    be corrupted or encrypted, which is caught later when pypdf tries to
    parse it.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}' for {file_path.name}. "
            f"Only {SUPPORTED_EXTENSIONS} are supported in Phase 2."
        )


def load_single_pdf(file_path: Path) -> List[Document]:
    """
    Load one PDF and return a list of LangChain Document objects, one per page.

    Each Document's metadata is enriched with:
        - source_filename : original file's name (e.g. "handbook.pdf"),
                             not the full filesystem path
        - doc_id           : a stable UUID shared by every page of this
                              document, so later phases can group chunks
                              back to "the same source document"
        - page_number      : 1-indexed page number, for human-facing citations
                              ("see page 4"), which is what non-programmers
                              expect
        - page_raw         : the original 0-indexed page number PyPDFLoader
                              returns, kept around for debugging

    Any exception pypdf/PyPDFLoader raises on an unreadable or corrupted file
    propagates up — the caller (load_pdfs_from_directory) is responsible for
    catching it, so that one bad file never stops the whole batch.
    """
    validate_pdf_path(file_path)

    loader = PyPDFLoader(str(file_path))
    pages: List[Document] = loader.load()

    if not pages:
        raise ValueError(f"No pages could be extracted from {file_path.name}")

    doc_id = str(uuid.uuid4())

    for page in pages:
        raw_page_number = page.metadata.get("page", 0)  # PyPDFLoader: 0-indexed
        page.metadata["source_filename"] = file_path.name
        page.metadata["doc_id"] = doc_id
        page.metadata["page_number"] = raw_page_number + 1  # normalized, 1-indexed
        page.metadata["page_raw"] = raw_page_number

        if not page.page_content or not page.page_content.strip():
            logger.warning(
                "Empty text extracted from %s, page %d — likely a scanned/"
                "image-only page. OCR would be needed to extract this text; "
                "that's out of scope for Phase 2, so this page is kept with "
                "empty content rather than dropped.",
                file_path.name,
                raw_page_number + 1,
            )

    return pages


def load_pdfs_from_directory(directory: Path) -> IngestionResult:
    """
    Load every supported PDF in `directory` (non-recursive).

    A single bad file (corrupted, encrypted, wrong extension) is logged and
    skipped rather than raising and stopping the whole batch — ingestion of
    N documents should not fail entirely because document 7 is broken.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    result = IngestionResult()
    pdf_paths = sorted(directory.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDF files found in %s", directory)
        return result

    for pdf_path in pdf_paths:
        try:
            pages = load_single_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001 — intentionally broad on purpose:
            # ANY failure for one file (corrupted PDF, encrypted PDF, permission
            # error, etc.) must not crash ingestion of the other files.
            logger.error("Skipping %s — %s", pdf_path.name, exc)
            result.skipped_files.append((pdf_path.name, str(exc)))
            continue

        for page in pages:
            if not page.page_content.strip():
                result.empty_pages.append((pdf_path.name, page.metadata["page_number"]))

        result.documents.extend(pages)
        result.loaded_files.append(pdf_path.name)
        logger.info("Loaded %s (%d pages)", pdf_path.name, len(pages))

    return result


if __name__ == "__main__":
    # Debug entry point. Run from the project root with:
    #   python -m src.ingestion.pdf_loader
    from config import UPLOAD_DIR

    result = load_pdfs_from_directory(UPLOAD_DIR)
    print("\n" + result.summary())

    if result.documents:
        first = result.documents[0]
        print("\n--- Preview of the first Document object ---")
        print("metadata:", first.metadata)
        print("page_content (first 300 chars):")
        print(first.page_content[:300])
    else:
        print("\nNo documents were loaded — check the warnings above.")
