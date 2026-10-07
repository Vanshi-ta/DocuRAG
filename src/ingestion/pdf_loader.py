

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    pass


class EmptyFileError(PDFLoadError):
    pass


class CorruptedPDFError(PDFLoadError):
    pass


class NoExtractableTextError(PDFLoadError):
    pass


@dataclass
class IngestionResult:
    documents: list[Document] = field(default_factory=list)
    loaded_files: list[str] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)                      
    empty_pages: list[tuple[str, int]] = field(default_factory=list)                             

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


def load_single_pdf(file_path: Path) -> list[Document]:
    
    validate_pdf_path(file_path)

    try:
        loader = PyPDFLoader(str(file_path))
        pages: list[Document] = loader.load()
    except Exception as exc:                                                  
                                                                           
                                                                       
        raise CorruptedPDFError(
            f"{file_path.name} could not be parsed as a valid PDF "
            f"(it may be corrupted or password-protected): {exc}"
        ) from exc

    if not pages:
        raise CorruptedPDFError(f"No pages could be extracted from {file_path.name}")

    doc_id = str(uuid.uuid4())

    for page in pages:
        raw_page_number = page.metadata.get("page", 0)                          
        page.metadata["source_filename"] = file_path.name
        page.metadata["doc_id"] = doc_id
        page.metadata["page_number"] = raw_page_number + 1                         
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
        except Exception as exc:                                               
                                                                            
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
