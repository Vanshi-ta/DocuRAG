"""
Shared UI state and cached resources.

`get_service()` is a process-wide singleton (st.cache_resource), so the
embedding model and the loaded index are created once and shared by every
page and rerun, instead of being rebuilt per session.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD
from src.services.conversation_store import ConversationStore
from src.services.docurag_service import DocuRAGService


@st.cache_resource(show_spinner="Loading embedding model and index…")
def get_service() -> DocuRAGService:
    service = DocuRAGService()
    service.load_existing()
    return service


@st.cache_resource
def get_store() -> ConversationStore:
    return ConversationStore()


@st.cache_data(ttl=10, show_spinner=False)
def llm_status() -> dict:
    """Ollama health, cached briefly so every rerun doesn't ping it.
    Call `llm_status.clear()` to force a re-check."""
    return get_service().llm_client.health_check()


def init_state() -> None:
    ss = st.session_state
    # Settings live in one dict, not in widget keys: Streamlit drops widget
    # state for widgets that aren't rendered on the current page.
    ss.setdefault(
        "settings",
        {"top_k": DEFAULT_TOP_K, "use_threshold": SIMILARITY_THRESHOLD is not None},
    )
    ss.setdefault("active_conversation_id", None)  # None = a new, not-yet-saved chat
    ss.setdefault("uploader_nonce", 0)             # bump to clear the file uploader
    ss.setdefault("last_ingest", None)             # per-file results of the last upload


def require_service() -> "DocuRAGService":
    """Return the service, or show a friendly full-page error and stop."""
    from ui.components import error_card

    try:
        return get_service()
    except Exception as exc:  # e.g. embedding model can't be loaded
        error_card(
            "DocuRAG couldn't start",
            str(exc),
            "Check that the embedding model is installed (it downloads on first run, "
            "which needs internet once) and that your dependencies are installed.",
        )
        st.stop()
