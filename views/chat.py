"""Chat view: conversation history sidebar, streaming answers, source cards."""

from __future__ import annotations

import logging

import streamlit as st

from src.errors import DocuRAGError
from ui.components import (
    describe_exception,
    empty_state,
    error_card,
    esc,
    html,
    pack_error,
    render_sources,
    status_pill,
    unpack_error,
)
from ui.state import get_store, llm_status, require_service
from ui.styles import inject_css

logger = logging.getLogger(__name__)

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"
DOCS_PAGE = "views/documents.py"

SUGGESTIONS = [
    (":material/summarize:", "Summarize each document in a few sentences."),
    (":material/topic:", "What are the main topics covered across my documents?"),
    (":material/format_list_numbered:", "List the key facts, dates, or numbers mentioned."),
]

inject_css("narrow")
service = require_service()
store = get_store()
ss = st.session_state


# --- sidebar: history -----------------------------------------------------------
def _new_chat() -> None:
    ss.active_conversation_id = None


def _open(conversation_id: str) -> None:
    ss.active_conversation_id = conversation_id


def _delete(conversation_id: str) -> None:
    store.delete(conversation_id)
    if ss.active_conversation_id == conversation_id:
        ss.active_conversation_id = None


def _rename(conversation_id: str) -> None:
    new_title = ss.get(f"rename_{conversation_id}", "").strip()
    if new_title:
        store.rename(conversation_id, new_title)


def render_history() -> None:
    with st.sidebar:
        st.button(
            "New chat",
            icon=":material/add:",
            use_container_width=True,
            type="primary",
            on_click=_new_chat,
            key="new_chat",
        )
        html('<div class="side-label">Recent</div>')
        conversations = store.list()[:25]
        if not conversations:
            st.caption("Your conversations will appear here.")
        for conv in conversations:
            active = conv.id == ss.active_conversation_id
            main, more = st.columns([6, 1], vertical_alignment="center")
            with main:
                st.button(
                    conv.title,
                    key=f"{'convactive' if active else 'conv'}_{conv.id}",
                    use_container_width=True,
                    on_click=_open,
                    args=(conv.id,),
                    help=conv.title,
                )
            with more:
                with st.popover(":material/more_vert:", help="Conversation actions"):
                    st.text_input("Title", value=conv.title, key=f"rename_{conv.id}")
                    st.button("Rename", key=f"do_rename_{conv.id}", on_click=_rename, args=(conv.id,),
                              use_container_width=True)
                    st.button("Delete chat", key=f"do_delete_{conv.id}", on_click=_delete, args=(conv.id,),
                              icon=":material/delete:", use_container_width=True)


# --- message rendering ----------------------------------------------------------
def render_message(message) -> None:
    avatar = USER_AVATAR if message.role == "user" else ASSISTANT_AVATAR
    with st.chat_message(message.role, avatar=avatar):
        if message.is_error:
            error_card(*unpack_error(message.content))
            return
        st.markdown(message.content)
        if message.role == "assistant":
            render_sources(message.sources, message.used_llm)


def header(title: str, status: dict) -> None:
    if not status.get("reachable"):
        pill = status_pill("Ollama offline", "err")
    elif not status.get("model_available"):
        pill = status_pill(f"Model {status.get('model', '')} missing", "warn")
    else:
        pill = status_pill(status.get("model", "model ready"), "ok")
    html(f'<div class="chat-head"><div class="t">{esc(title)}</div>{pill}</div>')


def llm_banner(status: dict) -> None:
    if status.get("reachable") and status.get("model_available"):
        return
    if not status.get("reachable"):
        error_card(
            "Ollama isn't running",
            "DocuRAG needs a local Ollama server to write answers.",
            "Start it with `ollama serve`, then press Re-check.",
            kind="warn",
        )
    else:
        error_card(
            f"Model '{status.get('model')}' isn't installed",
            "Ollama is running but doesn't have the configured model.",
            f"Install it with `ollama pull {status.get('model')}`, then press Re-check.",
            kind="warn",
        )
    if st.button("Re-check", icon=":material/refresh:", key="recheck"):
        llm_status.clear()
        st.rerun()


# --- asking a question -----------------------------------------------------------
def answer(question: str, conversation_id: str) -> None:
    settings = ss.settings
    store.append_message(conversation_id, "user", question)

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        status = st.status("Searching your documents…", expanded=False)
        parts: list[str] = []
        rag = None
        try:
            rag = service.ask_stream(
                question, top_k=settings["top_k"], use_threshold=settings["use_threshold"]
            )
            if rag.used_llm:
                n = len(rag.sources)
                status.update(label=f"Found {n} relevant passage{'s' if n != 1 else ''} — writing answer…")
            else:
                status.update(label="No relevant passages found", state="complete")

            def collect():
                for token in rag.tokens:
                    parts.append(token)
                    yield token

            st.write_stream(collect())
            if rag.used_llm:
                status.update(label="Answer ready", state="complete")
            render_sources(rag.sources, rag.used_llm)
            store.append_message(
                conversation_id, "assistant", "".join(parts), sources=rag.sources, used_llm=rag.used_llm
            )
        except Exception as exc:  # noqa: BLE001 - every failure must end in a visible state
            if not isinstance(exc, DocuRAGError):
                logger.exception("Unexpected error while answering")
            status.update(label="Couldn't finish the answer", state="error")
            title, detail, hint = describe_exception(exc)
            error_card(title, detail, hint)
            if parts:  # keep whatever was written before the failure
                store.append_message(
                    conversation_id, "assistant", "".join(parts),
                    sources=rag.sources if rag else None, used_llm=True,
                )
            store.append_message(
                conversation_id, "assistant", pack_error(title, detail, hint), is_error=True
            )


# --- page ---------------------------------------------------------------------------
render_history()

conversation = None
if ss.active_conversation_id:
    try:
        conversation = store.get(ss.active_conversation_id)
    except (FileNotFoundError, ValueError):
        ss.active_conversation_id = None

status = llm_status()
header(conversation.title if conversation else "New chat", status)
llm_banner(status)

messages = conversation.messages if conversation else []
documents = service.list_documents()

if not service.has_index:
    empty_state(
        "📄",
        "Add a document to get started",
        "Upload one or more PDFs and DocuRAG will answer questions using only what's in them.",
    )
    st.page_link(DOCS_PAGE, label="Upload documents", icon=":material/upload_file:")
elif not messages:
    names = ", ".join(d.source_filename for d in documents[:3])
    more = f" and {len(documents) - 3} more" if len(documents) > 3 else ""
    empty_state(
        "💬",
        "Ask anything about your documents",
        "Answers are grounded in your PDFs, with the exact passages shown as sources.",
    )
    html(f'<div class="scope">Searching {len(documents)} document{"s" if len(documents) != 1 else ""}: {esc(names)}{esc(more)}</div>')
    with st.container(key="chat_suggestions"):
        columns = st.columns(len(SUGGESTIONS))
        for i, (column, (icon, text)) in enumerate(zip(columns, SUGGESTIONS)):
            with column:
                if st.button(text, icon=icon, key=f"suggest_{i}", use_container_width=True):
                    ss.pending_question = text
                    st.rerun()

for message in messages:
    render_message(message)

typed = st.chat_input(
    "Ask a question about your documents…" if service.has_index else "Upload a document to start asking questions",
    disabled=not service.has_index,
)
question = (typed or ss.pop("pending_question", None) or "").strip()

if question:
    if conversation is None:
        conversation = store.create()
        ss.active_conversation_id = conversation.id
    answer(question, conversation.id)
    # Re-render once so the sidebar history (new/updated title) reflects this turn.
    st.rerun()
