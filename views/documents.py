"""Documents view: upload + indexing progress, results, and the document library."""

from __future__ import annotations

import logging

import streamlit as st

from src.pipeline import (
    STAGE_CHUNKING,
    STAGE_EMBEDDING,
    STAGE_HASHING,
    STAGE_LOADING,
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_REINDEXED,
    STATUS_SKIPPED_DUPLICATE,
    IngestionProgress,
)
from ui.components import chip, describe_exception, empty_state, error_card, esc, fmt_date, html, page_header
from ui.state import require_service
from ui.styles import inject_css

logger = logging.getLogger(__name__)

inject_css("wide")
service = require_service()
ss = st.session_state

STAGE_LABELS = {
    STAGE_HASHING: "Checking for duplicates",
    STAGE_LOADING: "Reading pages",
    STAGE_CHUNKING: "Splitting into passages",
    STAGE_EMBEDDING: "Embedding and indexing",
}
STAGE_FRACTION = {STAGE_HASHING: 0.05, STAGE_LOADING: 0.2, STAGE_CHUNKING: 0.4, STAGE_EMBEDDING: 0.6}

RESULT_CHIPS = {
    STATUS_INDEXED: ("Indexed", "ok"),
    STATUS_REINDEXED: ("Updated", "info"),
    STATUS_SKIPPED_DUPLICATE: ("Already indexed", "warn"),
    STATUS_FAILED: ("Failed", "err"),
}


# --- actions ---------------------------------------------------------------------
def run_ingest(files) -> None:
    saved, rejected = service.save_uploads({f.name: f.getvalue() for f in files})
    if rejected:
        st.warning("Skipped (not a PDF): " + ", ".join(rejected))
    if not saved:
        return

    with st.status("Indexing documents…", expanded=True) as status:
        bar = st.progress(0.0)
        line = st.empty()

        def on_progress(event: IngestionProgress) -> None:
            fraction = STAGE_FRACTION.get(event.stage, 1.0)
            bar.progress(min(1.0, ((event.file_index - 1) + fraction) / event.total_files))
            label = STAGE_LABELS.get(event.stage, "Done")
            line.markdown(f"**{event.filename}** · {label} ({event.file_index} of {event.total_files})")

        try:
            result = service.ingest(paths=saved, progress_callback=on_progress)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion crashed")
            status.update(label="Indexing failed", state="error")
            error_card(*describe_exception(exc))
            return
        bar.progress(1.0)
        status.update(label="Indexing complete", state="complete", expanded=False)

    ss.last_ingest = [vars(r) for r in result.file_results]
    ss.uploader_nonce += 1  # clears the uploader widget
    st.rerun()


def _dismiss_results() -> None:
    ss.last_ingest = None


@st.dialog("Remove this document?")
def confirm_remove(doc_id: str, name: str) -> None:
    st.write(f"**{name}** will be removed from the index and its uploaded file deleted from disk.")
    st.caption("Past conversations keep their saved citations.")
    yes, no = st.columns(2)
    if yes.button("Remove", type="primary", use_container_width=True, key=f"confirm_rm_{doc_id}"):
        try:
            service.delete_document(doc_id)
            (service.upload_dir / name).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Delete failed")
            error_card(*describe_exception(exc))
            return
        st.rerun()
    if no.button("Cancel", use_container_width=True, key=f"cancel_rm_{doc_id}"):
        st.rerun()


# --- page ---------------------------------------------------------------------------
page_header("Documents", "Upload PDFs and manage what DocuRAG can answer from.")

documents = service.list_documents()

if documents:
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(documents))
    c2.metric("Pages", sum(d.page_count for d in documents))
    c3.metric("Passages", sum(d.chunk_count for d in documents))
    st.write("")

# Upload
with st.container(border=True):
    st.markdown("**Add documents**")
    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_nonce}",
        label_visibility="collapsed",
        help="Text-based PDFs only. Scanned/image-only PDFs can't be read (no OCR).",
    )
    count = len(files) if files else 0
    if st.button(
        f"Index {count} file{'s' if count != 1 else ''}" if count else "Index files",
        type="primary",
        disabled=not count,
        icon=":material/upload_file:",
        key="index_btn",
    ):
        run_ingest(files)

# Last upload results
if ss.last_ingest:
    with st.container(border=True):
        head, dismiss = st.columns([5, 1], vertical_alignment="center")
        head.markdown("**Last upload**")
        dismiss.button("Dismiss", key="dismiss_results", on_click=_dismiss_results, use_container_width=True)
        rows = []
        for r in ss.last_ingest:
            label, kind = RESULT_CHIPS.get(r["status"], (r["status"], "info"))
            if r["status"] in (STATUS_INDEXED, STATUS_REINDEXED):
                detail = f'{r["chunks_added"]} passages from {r["page_count"]} page{"s" if r["page_count"] != 1 else ""}'
            else:
                detail = r.get("error") or ""
            rows.append(
                f'<div class="file-row"><div><div class="nm">{esc(r["filename"])}</div>'
                f'<div class="dt">{esc(detail)}</div></div>{chip(label, kind)}</div>'
            )
        html("".join(rows))

# Library
st.write("")
st.markdown(f"#### Library ({len(documents)})" if documents else "#### Library")
if not documents:
    empty_state(
        "📚",
        "No documents yet",
        "Drop a PDF above and press Index. Once it's indexed you can ask questions about it in Chat.",
    )
else:
    for doc in documents:
        with st.container(border=True):
            info, action = st.columns([6, 1.4], vertical_alignment="center")
            with info:
                meta = f"{doc.page_count} page{'s' if doc.page_count != 1 else ''} · {doc.chunk_count} passages"
                added = fmt_date(doc.indexed_at)
                if added:
                    meta += f" · added {added}"
                html(f'<div class="doc-name">📄 {esc(doc.source_filename)}</div><div class="doc-meta">{esc(meta)}</div>')
            with action:
                if st.button("Remove", key=f"rm_{doc.doc_id}", icon=":material/delete:", use_container_width=True):
                    confirm_remove(doc.doc_id, doc.source_filename)
    st.page_link("views/chat.py", label="Ask questions about these documents", icon=":material/chat_bubble:")
