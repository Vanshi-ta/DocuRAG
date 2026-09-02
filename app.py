"""
DocuRAG — Streamlit UI.

This file handles ONLY presentation and orchestration: rendering widgets,
managing session state, and calling into src/ and config for every piece
of actual RAG logic. No PDF parsing, chunking, embedding, retrieval, or
prompt-construction code lives here.
"""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_TOP_K, MAX_TOP_K, SIMILARITY_THRESHOLD, UPLOAD_DIR
from src.generation.llm_client import OllamaClient
from src.generation.rag_engine import answer_question
from src.ingestion.embedder import Embedder
from src.logging_config import configure_logging
from src.pipeline import load_vector_store, remove_document, reset_all, run_full_ingestion_pipeline
from src.retrieval.retriever import Retriever

configure_logging()

st.set_page_config(page_title="DocuRAG", page_icon="📄", layout="wide")


# --- Cached resources -------------------------------------------------------
@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_llm_client() -> OllamaClient:
    return OllamaClient()


# --- Session state initialization -------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources", "used_llm"}
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = None
if "registry" not in st.session_state:
    st.session_state.registry = None
if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K
if "use_threshold" not in st.session_state:
    st.session_state.use_threshold = SIMILARITY_THRESHOLD is not None


def try_load_existing_index() -> None:
    """On first load in a session, load a previously persisted index from
    disk if one exists, instead of forcing a re-upload every time."""
    if st.session_state.faiss_store is not None:
        return
    embedder = get_embedder()
    try:
        faiss_store, metadata_store, registry = load_vector_store(embedder.embedding_dimension)
        st.session_state.faiss_store = faiss_store
        st.session_state.metadata_store = metadata_store
        st.session_state.registry = registry
    except FileNotFoundError:
        pass  # no existing index yet — completely normal on a fresh project


try_load_existing_index()


# --- Header -------------------------------------------------------------
st.title("📄 DocuRAG")
st.caption("Domain-specific document Q&A — fully local, no paid APIs.")


# --- Sidebar: upload, process, manage documents -----------------------
def render_document_manager() -> None:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents", disabled=not uploaded_files):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("Extracting text, chunking, embedding, and indexing..."):
            embedder = get_embedder()
            result, faiss_store, metadata_store, registry = run_full_ingestion_pipeline(
                embedder=embedder, persist=True
            )

        st.session_state.faiss_store = faiss_store
        st.session_state.metadata_store = metadata_store
        st.session_state.registry = registry

        if result.indexed_files:
            st.success(f"Indexed: {', '.join(result.indexed_files)}")
        if result.reindexed_files:
            st.info(f"Re-indexed (content changed): {', '.join(result.reindexed_files)}")
        if result.skipped_duplicate_files:
            st.warning(
                f"Skipped (identical content already indexed): "
                f"{', '.join(result.skipped_duplicate_files)}"
            )
        if result.failed_files:
            for filename, reason in result.failed_files:
                st.error(f"Could not index **{filename}**: {reason}")

    st.divider()

    # --- Indexed document list with per-document delete ---------------------
    registry = st.session_state.registry
    if registry is not None and registry.documents:
        st.subheader(f"Indexed documents ({len(registry.documents)})")
        for entry in registry.list_documents():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"**{entry.source_filename}**  \n"
                    f"{entry.page_count} pages · {entry.chunk_count} chunks"
                )
            with col2:
                if st.button("🗑️", key=f"delete_{entry.doc_id}", help="Remove this document"):
                    remove_document(
                        entry.doc_id,
                        st.session_state.faiss_store,
                        st.session_state.metadata_store,
                        st.session_state.registry,
                    )
                    st.rerun()
        total_chunks = st.session_state.faiss_store.ntotal if st.session_state.faiss_store else 0
        st.caption(f"Total: {total_chunks} chunks indexed.")
    else:
        st.info("No documents indexed yet — upload PDFs above and click 'Process Documents'.")

    st.divider()

    # --- Retrieval settings ---------------------------------------------------
    st.subheader("Retrieval settings")
    st.session_state.top_k = st.slider(
        "Chunks to retrieve (top-k)", min_value=1, max_value=MAX_TOP_K, value=st.session_state.top_k
    )
    st.session_state.use_threshold = st.checkbox(
        "Filter out low-relevance chunks",
        value=st.session_state.use_threshold,
        help=(
            f"Discards retrieved chunks below a similarity threshold "
            f"(currently {SIMILARITY_THRESHOLD}) before they reach the LLM. "
            f"Disabling this may surface less relevant context."
        ),
    )

    st.divider()

    if st.button("🗑️ Reset everything"):
        reset_all()
        st.session_state.faiss_store = None
        st.session_state.metadata_store = None
        st.session_state.registry = None
        st.session_state.messages = []
        st.rerun()


with st.sidebar:
    render_document_manager()


# --- Main area: chat interface --------------------------
def render_sources(sources, used_llm: bool) -> None:
    if not used_llm:
        st.caption("No sources met the relevance threshold — the LLM was not called.")
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(
                f"- **{s.source_filename}**, page {s.page_number} "
                f"(cosine similarity: {s.similarity_score:.3f})"
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") is not None:
            render_sources(message["sources"], message.get("used_llm", True))

question = st.chat_input("Ask a question about your documents...")

if question is not None:
    question = question.strip()

    if st.session_state.faiss_store is None:
        st.error("Please upload and process at least one document first.")
    elif not question:
        st.error("Please enter a non-empty question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = get_embedder()
                retriever = Retriever(
                    embedder, st.session_state.faiss_store, st.session_state.metadata_store
                )
                llm_client = get_llm_client()
                threshold = SIMILARITY_THRESHOLD if st.session_state.use_threshold else None

                try:
                    result = answer_question(
                        question,
                        retriever,
                        llm_client,
                        top_k=st.session_state.top_k,
                        similarity_threshold=threshold,
                    )
                    st.markdown(result.answer)
                    render_sources(result.sources, result.used_llm)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                            "used_llm": result.used_llm,
                        }
                    )
                except ValueError as exc:
                    # e.g. an empty question slipping through, or a
                    # misconfigured top_k — surfaced clearly rather than
                    # as a raw traceback.
                    error_message = f"⚠️ {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "sources": None}
                    )
                except (ConnectionError, TimeoutError) as exc:
                    # LLM (Ollama) unavailable or slow to respond.
                    error_message = f"⚠️ {exc}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "sources": None}
                    )
