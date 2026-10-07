"""
Reusable UI building blocks: headers, chips, empty states, error cards,
source/citation display.

SECURITY: anything that originates from a PDF (filenames, passage text) or
from an exception is passed through `esc()` before being placed in HTML,
because these strings are rendered with unsafe_allow_html=True.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

from src.errors import DocuRAGError

PASSAGE_PREVIEW_CHARS = 420


def esc(value: Any) -> str:
    return escape(str(value), quote=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --- headers / small elements -------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sub">{esc(subtitle)}</div>' if subtitle else ""
    html(f'<div class="page-head"><div class="title">{esc(title)}</div>{sub}</div>')


def chip(text: str, kind: str = "info") -> str:
    """Return chip HTML. kind: ok | warn | err | info."""
    return f'<span class="chip {kind}">{esc(text)}</span>'


def status_pill(text: str, kind: str = "ok") -> str:
    return f'<span class="pill {kind}"><span class="dot"></span>{esc(text)}</span>'


def fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y")
    except (ValueError, TypeError):
        return ""


def empty_state(icon: str, title: str, body: str) -> None:
    html(
        f'<div class="empty"><div class="ico">{esc(icon)}</div>'
        f'<div class="ttl">{esc(title)}</div><div class="bd">{esc(body)}</div></div>'
    )


# --- errors ---------------------------------------------------------------------
_ERROR_TITLES = {
    "llm_unavailable": "Can't reach Ollama",
    "llm_model_not_found": "Model not installed",
    "llm_timeout": "The model took too long to answer",
    "llm_response_error": "The model returned an error",
    "empty_question": "Type a question first",
    "index_not_ready": "No documents indexed yet",
}


def error_card(title: str, detail: str = "", hint: str | None = None, kind: str = "err") -> None:
    parts = [f'<div class="card-note {kind}"><div class="ttl">{esc(title)}</div>']
    if detail:
        parts.append(f'<div class="dt">{esc(detail)}</div>')
    if hint:
        parts.append(f'<div class="ht">{esc(hint)}</div>')
    parts.append("</div>")
    html("".join(parts))


def describe_exception(exc: BaseException) -> tuple[str, str, str | None]:
    """Map an exception to (title, detail, hint) for display."""
    if isinstance(exc, DocuRAGError):
        return _ERROR_TITLES.get(exc.code, "Something went wrong"), str(exc), exc.hint
    return "Something went wrong", str(exc) or exc.__class__.__name__, None


def pack_error(title: str, detail: str, hint: str | None) -> str:
    """Serialize an error into a chat message's text (so it survives reloads)."""
    return "\n\n".join(p for p in (title, detail, hint or "") if p)


def unpack_error(text: str) -> tuple[str, str, str | None]:
    parts = text.split("\n\n")
    title = parts[0]
    detail = parts[1] if len(parts) > 1 else ""
    hint = parts[2] if len(parts) > 2 else None
    return title, detail, hint


# --- sources / citations ------------------------------------------------------------
def _unique_locations(sources: Iterable[Any]) -> list[tuple[str, int]]:
    seen, out = set(), []
    for s in sources:
        key = (s.source_filename, s.page_number)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > PASSAGE_PREVIEW_CHARS:
        flat = flat[:PASSAGE_PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"
    return flat


def render_sources(sources: Sequence[Any] | None, used_llm: bool | None) -> None:
    """Chips for each cited file/page, plus an expander with the actual passages.
    Works with RetrievedChunk or StoredSource objects (same attribute names)."""
    if sources is None:
        return
    if used_llm is False or not sources:
        html(
            '<div class="note">No passages in your documents were relevant enough, '
            "so no answer was generated.</div>"
        )
        return

    chips = "".join(
        f'<span class="src-chip" title="{esc(name)}"><span class="nm">{esc(name)}</span>'
        f'<span class="pg">p.{esc(page)}</span></span>'
        for name, page in _unique_locations(sources)
    )
    html(f'<div class="src-label">Sources</div><div class="src-chips">{chips}</div>')

    with st.expander(f"View retrieved passages ({len(sources)})"):
        for s in sources:
            pct = max(0.0, min(1.0, float(s.similarity_score))) * 100
            html(
                f'<div class="passage-head"><span class="nm">{esc(s.source_filename)}'
                f'<span class="pg">page {esc(s.page_number)}</span></span>'
                f'<span class="sc">relevance {float(s.similarity_score):.2f}</span></div>'
                f'<div class="rel"><span style="width:{pct:.0f}%"></span></div>'
                f'<div class="passage">{esc(_preview(s.chunk_text))}</div>'
            )
