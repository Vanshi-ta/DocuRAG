"""Settings view: retrieval options, system status, and destructive actions."""

from __future__ import annotations

import streamlit as st

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    MAX_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SIMILARITY_THRESHOLD_VALUE,
)
from ui.components import describe_exception, error_card, esc, html, page_header, status_pill
from ui.state import get_store, llm_status, require_service
from ui.styles import inject_css

inject_css("narrow")
service = require_service()
store = get_store()
ss = st.session_state


def _on_top_k() -> None:
    ss.settings["top_k"] = ss.w_top_k


def _on_threshold() -> None:
    ss.settings["use_threshold"] = ss.w_use_threshold


@st.dialog("Reset the document index?")
def confirm_reset() -> None:
    st.write("This deletes **all uploaded PDFs and the search index**. Your saved conversations are kept.")
    yes, no = st.columns(2)
    if yes.button("Delete everything", type="primary", use_container_width=True, key="confirm_reset"):
        try:
            service.reset()
        except Exception as exc:  # noqa: BLE001
            error_card(*describe_exception(exc))
            return
        ss.last_ingest = None
        st.rerun()
    if no.button("Cancel", use_container_width=True, key="cancel_reset"):
        st.rerun()


@st.dialog("Delete all conversations?")
def confirm_clear_chats() -> None:
    st.write("Every saved conversation will be permanently deleted.")
    yes, no = st.columns(2)
    if yes.button("Delete conversations", type="primary", use_container_width=True, key="confirm_clear"):
        for summary in store.list():
            store.delete(summary.id)
        ss.active_conversation_id = None
        st.rerun()
    if no.button("Cancel", use_container_width=True, key="cancel_clear"):
        st.rerun()


page_header("Settings", "Tune retrieval and check that everything DocuRAG depends on is running.")

# --- retrieval ---------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Retrieval**")
    st.slider(
        "Passages used per answer",
        min_value=1,
        max_value=MAX_TOP_K,
        value=ss.settings["top_k"],
        key="w_top_k",
        on_change=_on_top_k,
        help="More passages give the model more context but make answers slower.",
    )
    st.toggle(
        "Ignore weakly related passages",
        value=ss.settings["use_threshold"],
        key="w_use_threshold",
        on_change=_on_threshold,
        help=(
            f"Drops passages below a similarity of {SIMILARITY_THRESHOLD_VALUE} before they reach the model. "
            "If nothing passes, DocuRAG says it couldn't find the answer instead of guessing."
        ),
    )

# --- system status -------------------------------------------------------------------
with st.container(border=True):
    head, refresh = st.columns([4, 1], vertical_alignment="center")
    head.markdown("**System**")
    if refresh.button("Re-check", icon=":material/refresh:", key="settings_recheck", use_container_width=True):
        llm_status.clear()
        st.rerun()

    status = llm_status()
    if not status.get("reachable"):
        pill = status_pill("Ollama offline", "err")
    elif not status.get("model_available"):
        pill = status_pill("Model not installed", "warn")
    else:
        pill = status_pill("Ready", "ok")
    html(pill)

    if not status.get("reachable"):
        st.caption("Start Ollama with `ollama serve`.")
    elif not status.get("model_available"):
        st.caption(f"Install the model with `ollama pull {OLLAMA_MODEL}`.")

    rows = [
        ("Language model", OLLAMA_MODEL),
        ("Ollama server", OLLAMA_BASE_URL),
        ("Embedding model", EMBEDDING_MODEL_NAME),
        ("Passage size", f"{CHUNK_SIZE} characters ({CHUNK_OVERLAP} overlap)"),
    ]
    html('<div class="kv" style="margin-top:.8rem">' + "".join(
        f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in rows
    ) + "</div>")
    st.caption("These come from your .env / config.py. Edit them there and restart the app.")

# --- danger zone -------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Data**")
    st.caption("Destructive actions. You'll be asked to confirm.")
    a, b = st.columns(2)
    with a:
        if st.button("Reset document index", icon=":material/delete_forever:", use_container_width=True, key="reset_btn"):
            confirm_reset()
    with b:
        if st.button("Delete all conversations", icon=":material/delete_sweep:", use_container_width=True, key="clear_btn"):
            confirm_clear_chats()
