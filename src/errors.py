

from __future__ import annotations

from typing import Optional


class DocuRAGError(Exception):
    

    code: str = "docurag_error"
    default_hint: Optional[str] = None

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint if hint is not None else self.default_hint


                                                                           
class LLMUnavailableError(DocuRAGError, ConnectionError):
    

    code = "llm_unavailable"
    default_hint = "Start Ollama (e.g. `ollama serve`) and try again."


class LLMModelNotFoundError(DocuRAGError, ConnectionError):
    

    code = "llm_model_not_found"
    default_hint = "Pull the model with `ollama pull <model>` and try again."


class LLMTimeoutError(DocuRAGError, TimeoutError):
    

    code = "llm_timeout"
    default_hint = "The model may still be loading. Try again in a moment."


class LLMResponseError(DocuRAGError, ConnectionError):
    

    code = "llm_response_error"


                                                                             
class EmptyQuestionError(DocuRAGError, ValueError):
    code = "empty_question"


class IndexNotReadyError(DocuRAGError):
    

    code = "index_not_ready"
    default_hint = "Upload and process at least one PDF first."
