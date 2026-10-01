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

st.markdown(
    """
    <style>
        :root {
            --bg: #f4f1ec;
            --bg-strong: #ecf3fa;
            --panel: rgba(255, 255, 255, 0.82);
            --panel-soft: rgba(248, 249, 252, 0.9);
            --panel-strong: rgba(228, 239, 252, 0.9);
            --border: rgba(153, 170, 188, 0.45);
            --border-strong: rgba(103, 138, 182, 0.55);
            --text: #1d2433;
            --muted: #5f6d7a;
            --primary: #2f5d8f;
            --primary-strong: #234d7a;
            --accent: #dfeefb;
            --accent-strong: #b9d8f5;
            --success: #2d8f6d;
            --warning: #d38c1e;
            --shadow: rgba(15, 23, 42, 0.08);
        }

        html[data-theme="dark"],
        .theme-dark {
            --bg: #0f1724;
            --bg-strong: #111b2b;
            --panel: rgba(19, 27, 38, 0.86);
            --panel-soft: rgba(24, 34, 48, 0.9);
            --panel-strong: rgba(27, 47, 67, 0.92);
            --border: rgba(143, 163, 188, 0.28);
            --border-strong: rgba(116, 160, 211, 0.45);
            --text: #e5edf8;
            --muted: #a7b7cc;
            --primary: #7bb0e6;
            --primary-strong: #9cc7f1;
            --accent: rgba(86, 129, 177, 0.25);
            --accent-strong: rgba(110, 165, 220, 0.34);
            --success: #6bc4a0;
            --warning: #f0ba60;
            --shadow: rgba(3, 7, 18, 0.35);
        }

        @media (prefers-color-scheme: dark) {
            :root:not([data-theme="light"]) {
                --bg: #0f1724;
                --bg-strong: #111b2b;
                --panel: rgba(19, 27, 38, 0.86);
                --panel-soft: rgba(24, 34, 48, 0.9);
                --panel-strong: rgba(27, 47, 67, 0.92);
                --border: rgba(143, 163, 188, 0.28);
                --border-strong: rgba(116, 160, 211, 0.45);
                --text: #e5edf8;
                --muted: #a7b7cc;
                --primary: #7bb0e6;
                --primary-strong: #9cc7f1;
                --accent: rgba(86, 129, 177, 0.25);
                --accent-strong: rgba(110, 165, 220, 0.34);
                --success: #6bc4a0;
                --warning: #f0ba60;
                --shadow: rgba(3, 7, 18, 0.35);
            }
        }

        .stApp {
            background: linear-gradient(180deg, var(--bg) 0%, var(--bg-strong) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.02);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div {
            background: var(--panel-soft);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        h1 {
            color: var(--text);
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem !important;
        }

        .stCaption {
            color: var(--muted) !important;
            font-size: 0.95rem !important;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--border-strong);
            background: linear-gradient(180deg, var(--panel-soft) 0%, var(--panel-strong) 100%);
            color: var(--primary-strong);
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .stButton > button:hover {
            border-color: var(--primary);
            box-shadow: 0 8px 18px var(--shadow);
            transform: translateY(-1px);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(47, 93, 143, 0.2);
        }

        .stFileUploader > div {
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        [data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 4px 18px var(--shadow);
            margin-bottom: 0.8rem;
        }

        [data-testid="stChatMessage"] .stMarkdown {
            color: var(--text);
        }

        [data-testid="stChatMessage"] .stMarkdown p {
            margin-top: 0;
            margin-bottom: 0.25rem;
        }

        [data-testid="stChatMessage"] .stExpander {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--panel-soft);
        }

        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(47, 93, 143, 0.18);
            background: var(--panel-strong);
            color: var(--text);
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            color: var(--muted);
            border-radius: 10px 10px 0 0;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: var(--panel);
            color: var(--primary-strong);
            border: 1px solid var(--border);
            border-bottom: none;
        }

        .stChatInput {
            border: 1px solid var(--border-strong);
            border-radius: 16px;
            background: var(--panel);
            box-shadow: 0 8px 26px var(--shadow);
        }

        .research-shell {
            margin-top: 0.25rem;
        }

        .source-panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 0.9rem 0.3rem 0.9rem;
            box-shadow: 0 6px 22px var(--shadow);
            position: sticky;
            top: 1rem;
        }

        .source-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--text);
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
            letter-spacing: -0.02em;
        }

        .source-count {
            font-size: 0.72rem;
            color: var(--muted);
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
        }

        .source-item {
            background: var(--panel-soft);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 0.75rem;
            margin-bottom: 0.55rem;
            color: var(--text);
        }

        .source-item strong {
            color: var(--primary-strong);
            font-weight: 700;
        }

        .source-meta {
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.72rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }

        .source-badge {
            display: inline-flex;
            align-items: center;
            background: var(--accent);
            color: var(--primary-strong);
            border: 1px solid var(--border-strong);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-weight: 600;
            font-size: 0.7rem;
        }

        .source-empty {
            color: var(--muted);
            background: var(--panel-soft);
            border: 1px dashed var(--border-strong);
            border-radius: 12px;
            padding: 0.75rem;
            line-height: 1.5;
        }

        textarea {
            background: transparent !important;
            color: var(--text) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }

        .stSlider > div > div > div {
            background: linear-gradient(90deg, var(--accent-strong) 0%, var(--primary) 100%);
        }

        .stCheckbox {
            color: var(--text);
        }

        .stNumberInput, .stTextInput, .stSelectbox, .stMultiSelect {
            background: var(--panel-soft);
            color: var(--text);
        }

        .stDataFrame, .stTable {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
st.title("DocuRAG")
st.caption("Local document research assistant for PDF collections")


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
