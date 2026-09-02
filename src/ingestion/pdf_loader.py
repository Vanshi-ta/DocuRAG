"""
PDF ingestion module for DocuRAG.

Responsible ONLY for:
    - discovering PDF files in a directory
    - validating that a path is a usable PDF
    - extracting per-page text using LangChain's PyPDFLoader (backed by pypdf)
    - normalizing and enriching each page's metadata
    - classifying failures (missing / wrong type / empty file / corrupted /
      encrypted / no extractable text) into specific exception types so
      callers (pipeline, UI) can show a precise, actionable message instead
      of a generic "something went wrong"

This module has NO knowledge of chunking, embeddings, FAISS, or Streamlit.
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

SUPPORTED_EXTENSIONS = {".pdf"}


class PDFLoadError(Exception):
    """Base class for every PDF-loading failure DocuRAG classifies explicitly."""


class EmptyFileError(PDFLoadError):
    """The file exists but contains zero bytes."""


class CorruptedPDFError(PDFLoadError):
    """The file could not be parsed as a valid PDF (bad structure, or encrypted)."""


class NoExtractableTextError(PDFLoadError):
    """The PDF parsed successfully but every page came back with no text
    (typically a scanned/image-only PDF — DocuRAG does not do OCR)."""


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
    Filesystem-level validation only: existence, that it's a file, that the
    extension is supported, and that it isn't a zero-byte file. Does NOT
    guarantee the PDF is actually parseable — that's caught in
    load_single_pdf, which distinguishes corrupted/encrypted/empty-text
    failures explicitly.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}' for {file_path.name}. "
            f"Only {SUPPORTED_EXTENSIONS} are supported."
        )
    if file_path.stat().st_size == 0:
        raise EmptyFileError(f"{file_path.name} is a 0-byte file.")


def load_single_pdf(file_path: Path) -> List[Document]:
    """
    Load one PDF and return a list of LangChain Document objects, one per
    page. Raises a specific PDFLoadError subclass on failure so callers can
    show a precise message rather than a bare traceback.

    Each Document's metadata is enriched with:
        - source_filename : original file's name
        - doc_id           : a stable UUID shared by every page of this document
        - page_number      : 1-indexed page number, for human-facing citations
        - page_raw         : the original 0-indexed page number from PyPDFLoader
    """
    validate_pdf_path(file_path)

    try:
        loader = PyPDFLoader(str(file_path))
        pages: List[Document] = loader.load()
    except Exception as exc:  # pypdf raises a variety of internal error types
        # for malformed structure, unsupported encryption, etc. — normalize
        # all of them into one DocuRAG-specific, user-facing exception.
        raise CorruptedPDFError(
            f"{file_path.name} could not be parsed as a valid PDF "
            f"(it may be corrupted or password-protected): {exc}"
        ) from exc

    if not pages:
        raise CorruptedPDFError(f"No pages could be extracted from {file_path.name}")

    doc_id = str(uuid.uuid4())

    for page in pages:
        raw_page_number = page.metadata.get("page", 0)  # PyPDFLoader: 0-indexed
        page.metadata["source_filename"] = file_path.name
        page.metadata["doc_id"] = doc_id
        page.metadata["page_number"] = raw_page_number + 1  # normalized, 1-indexed
        page.metadata["page_raw"] = raw_page_number

    if all(not page.page_content.strip() for page in pages):
        raise NoExtractableTextError(
            f"{file_path.name} parsed successfully but no text could be "
            f"extracted from any of its {len(pages)} page(s) — it is likely "
            f"a scanned/image-only PDF. DocuRAG does not perform OCR."
        )

    for page in pages:
        if not page.page_content.strip():
            logger.warning(
                "Empty text on %s page %d — likely a scanned/image-only page.",
                file_path.name, page.metadata["page_number"],
            )

    return pages


def load_pdfs_from_directory(directory: Path) -> IngestionResult:
    """
    Load every supported PDF in `directory` (non-recursive). A single bad
    file is logged and skipped rather than raising and stopping the whole
    batch.
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
        except PDFLoadError as exc:
            logger.error("Skipping %s — %s", pdf_path.name, exc)
            result.skipped_files.append((pdf_path.name, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — last-resort catch-all so one
            # unexpected failure never crashes ingestion of the other files.
            logger.error("Skipping %s — unexpected error: %s", pdf_path.name, exc)
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
    from config import UPLOAD_DIR

    result = load_pdfs_from_directory(UPLOAD_DIR)
    print("\n" + result.summary())
