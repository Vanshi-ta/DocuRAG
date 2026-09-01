"""
DocuRAG — Streamlit UI (Phase 7).

This file handles ONLY presentation and orchestration: rendering widgets,
managing session state, and calling into src/ modules for every piece of
actual RAG logic. It contains NO PDF parsing, chunking, embedding,
retrieval, or prompt-construction code of its own — every one of those
already exists, already tested, in src/. See the Phase 7 guide for why
that separation matters and how each responsibility maps to a specific
function call below.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_TOP_K, UPLOAD_DIR
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import answer_question
from src.ingestion.embedder import Embedder
from src.pipeline import load_vector_store, reset_all, run_full_ingestion_pipeline
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="DocuRAG", page_icon="📄", layout="wide")


# --- Cached resources -------------------------------------------------------
# @st.cache_resource holds these across EVERY rerun of the script (button
# clicks, chat input, etc.) and across every user session — the model
# weights are loaded from disk exactly once per running app process, not
# once per interaction. See Phase 7 guide Sections 3 and 5.
@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_llm_client() -> OllamaClient:
    return OllamaClient()


# --- Session state initialization -------------------------------------------
# st.session_state persists per-browser-session data across reruns. Unlike
# the cached resources above (one shared instance for the whole app), this
# data is specific to THIS user's current session — their chat history and
# their currently-loaded index. See Phase 7 guide Section 4.
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = None
if "indexed_file_count" not in st.session_state:
    st.session_state.indexed_file_count = 0


def try_load_existing_index() -> None:
    """
    On first load in a session, if a vector store already exists on disk
    from a previous run (Phase 4's persistence), load it automatically
    instead of forcing the user to re-upload and re-process every time
    they open the app.
    """
    if st.session_state.faiss_store is not None:
        return
    embedder = get_embedder()
    try:
        faiss_store, metadata_store = load_vector_store(embedder.embedding_dimension)
        st.session_state.faiss_store = faiss_store
        st.session_state.metadata_store = metadata_store
        st.session_state.indexed_file_count = len(
            {record.source_filename for record in metadata_store.records}
        )
    except FileNotFoundError:
        pass  # no existing index yet — completely normal on a fresh project


try_load_existing_index()


# --- Header (#1) -------------------------------------------------------------
st.title("📄 DocuRAG")
st.caption("Domain-specific document Q&A — fully local, no paid APIs.")


# --- Sidebar: upload, process, reset (#2, #3, #4, #10) -----------------------
with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", disabled=not uploaded_files):
        # Save uploaded files to disk so the existing (unmodified) Phase 2
        # pdf_loader — which reads from a directory of real files — can
        # work exactly as it already does. See Phase 7 guide Section 6.
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Extracting text, chunking, embedding, and indexing..."):
            embedder = get_embedder()
            ingestion_result, chunking_result, faiss_store, metadata_store = (
                run_full_ingestion_pipeline(embedder=embedder, persist=True)
            )

        st.session_state.faiss_store = faiss_store
        st.session_state.metadata_store = metadata_store
        st.session_state.indexed_file_count = len(ingestion_result.loaded_files)

        if ingestion_result.skipped_files:
            st.warning(
                f"Skipped {len(ingestion_result.skipped_files)} file(s) — "
                f"check the terminal log for details."
            )

    # Visual indication of indexed state (#4).
    if st.session_state.faiss_store is not None:
        st.success(
            f"✅ {st.session_state.indexed_file_count} document(s) indexed "
            f"({st.session_state.faiss_store.index.ntotal} chunks)"
        )
    else:
        st.info("No documents indexed yet — upload PDFs above and click "
                 "'Process Documents'.")

    st.divider()

    # Reset (#10).
    if st.button("🗑️ Reset index"):
        reset_all()
        st.session_state.faiss_store = None
        st.session_state.metadata_store = None
        st.session_state.indexed_file_count = 0
        st.session_state.messages = []
        st.rerun()


# --- Main area: chat interface (#5, #6, #7, #8, #9) --------------------------
def render_sources(sources) -> None:
    """Render a list of RetrievedChunk objects as a source-citation block."""
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"- **{s.source_filename}**, page {s.page_number} "
                f"(similarity: {s.similarity_score:.3f})"
            )


# Replay conversation history (#6) — this loop is what makes the chat look
# persistent across reruns; without session_state, it would vanish on the
# very next interaction (see Phase 7 guide Sections 1-2).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask a question about your documents...")

if question:
    if st.session_state.faiss_store is None:
        st.error("Please upload and process at least one document first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question, "sources": None}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = get_embedder()
                retriever = Retriever(
                    embedder,
                    st.session_state.faiss_store,
                    st.session_state.metadata_store,
                )
                llm_client = get_llm_client()

                try:
                    result = answer_question(
                        question, retriever, llm_client, top_k=DEFAULT_TOP_K
                    )
                    st.markdown(result.answer)  # (#7)
                    if result.sources:
                        render_sources(result.sources)  # (#8)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )
                except (ConnectionError, TimeoutError) as exc:
                    # (#9's spirit extended to infrastructure failures, not
                    # just "answer not found" — both are clearly surfaced
                    # rather than silently failing.)
                    error_message = f"⚠️ {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "sources": None}
                    )
