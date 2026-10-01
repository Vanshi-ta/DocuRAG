"""
Structured, UI-friendly exceptions for DocuRAG.

Every class here inherits from the built-in exception the old code raised
(ConnectionError, TimeoutError, ValueError), so existing `except` blocks
and existing tests keep working unchanged. What's new is that a frontend can
now tell failures apart with `isinstance(...)` or the stable `code` string,
and show a clean message (`str(exc)`) plus an optional short `hint`
without parsing error text.
"""

from __future__ import annotations

from typing import Optional


class DocuRAGError(Exception):
    """Base class for errors DocuRAG raises deliberately."""

    code: str = "docurag_error"
    default_hint: Optional[str] = None

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint if hint is not None else self.default_hint


# --- LLM (Ollama) --------------------------------------------------------
class LLMUnavailableError(DocuRAGError, ConnectionError):
    """Ollama could not be reached at all."""

    code = "llm_unavailable"
    default_hint = "Start Ollama (e.g. `ollama serve`) and try again."


class LLMModelNotFoundError(DocuRAGError, ConnectionError):
    """Ollama is running, but the configured model isn't pulled."""

    code = "llm_model_not_found"
    default_hint = "Pull the model with `ollama pull <model>` and try again."


class LLMTimeoutError(DocuRAGError, TimeoutError):
    """Ollama accepted the request but did not answer in time."""

    code = "llm_timeout"
    default_hint = "The model may still be loading. Try again in a moment."


class LLMResponseError(DocuRAGError, ConnectionError):
    """Ollama answered with an error, or a stream broke mid-way."""

    code = "llm_response_error"


# --- Query / service layer -------------------------------------------------
class EmptyQuestionError(DocuRAGError, ValueError):
    code = "empty_question"


class IndexNotReadyError(DocuRAGError):
    """A question was asked before any document was indexed."""

    code = "index_not_ready"
    default_hint = "Upload and process at least one PDF first."
