"""
Global styling for the DocuRAG UI.

Design notes
- Colours are neutral translucent greys + one accent, so the same CSS works in
  both Streamlit's light and dark themes (no theme hacks, no `data-theme`).
- The accent comes from `.streamlit/config.toml` (primaryColor) for native
  widgets and from `--accent` below for custom HTML. Keep the two in sync.
- System font stack only: no external font requests, so the app stays fully
  local/offline.
"""

from __future__ import annotations

import streamlit as st

_WIDTHS = {"narrow": "52rem", "wide": "66rem"}

_BASE_CSS = """
<style>
:root {
  --accent: #6366f1;
  --accent-soft: rgba(99, 102, 241, 0.12);
  --line: rgba(128, 128, 128, 0.28);
  --surface: rgba(128, 128, 128, 0.07);
  --surface-2: rgba(128, 128, 128, 0.14);
  --muted: rgba(128, 128, 128, 0.95);
  --ok: #16a34a;
  --warn: #d97706;
  --err: #dc2626;
  --radius: 14px;
}

html, body, [class*="st-"], .stApp {
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
  font-feature-settings: "liga" !important;
  -webkit-font-feature-settings: "liga" !important;
}

/* ---- chrome ---- */
footer, [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { font-size: 1.75rem !important; }

/* ---- page header ---- */
.page-head { margin-bottom: 1.2rem; }
.page-head .title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }
.page-head .sub { color: var(--muted); font-size: 0.95rem; margin-top: 0.25rem; }

/* ---- chat header ---- */
.chat-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding-bottom: 0.8rem; margin-bottom: 0.4rem; border-bottom: 1px solid var(--line); }
.chat-head .t { font-weight: 650; font-size: 1.05rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---- pills / chips ---- */
.pill { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; padding: 0.22rem 0.65rem;
  border-radius: 999px; border: 1px solid var(--line); background: var(--surface); white-space: nowrap; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.pill.ok .dot { background: var(--ok); } .pill.err .dot { background: var(--err); } .pill.warn .dot { background: var(--warn); }

.chip { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.78rem; font-weight: 600;
  padding: 0.18rem 0.6rem; border-radius: 999px; border: 1px solid transparent; }
.chip.ok { color: var(--ok); background: rgba(22,163,74,.12); }
.chip.warn { color: var(--warn); background: rgba(217,119,6,.14); }
.chip.err { color: var(--err); background: rgba(220,38,38,.12); }
.chip.info { color: var(--accent); background: var(--accent-soft); }

/* ---- chat messages ---- */
[data-testid="stChatMessage"] { background: transparent; border: none; padding: 0.7rem 0.25rem; gap: 0.75rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.75rem 0.9rem; }
[data-testid="stChatMessage"] p { line-height: 1.65; margin-bottom: 0.55rem; }
[data-testid="stChatInput"] { margin-top: 1rem; border-radius: 12px; border: 1px solid var(--line);
  box-shadow: 0 5px 20px rgba(0,0,0,.06); transition: border-color .15s ease, box-shadow .15s ease; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.st-key-chat_suggestions [data-testid="stHorizontalBlock"] { gap: 0.65rem; }
.st-key-chat_suggestions [data-testid="stColumn"] button { min-height: 4rem; height: 100%; justify-content: flex-start;
  text-align: left; white-space: normal; padding: 0.65rem 0.75rem; line-height: 1.35; }
.st-key-chat_suggestions [data-testid="stColumn"] button p { white-space: normal; line-height: 1.35; }

/* ---- sources ---- */
.src-label { color: var(--muted); font-size: 0.75rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin: 0.9rem 0 0.4rem; }
.src-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.src-chip { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; padding: 0.25rem 0.65rem;
  border-radius: 8px; background: var(--accent-soft); border: 1px solid transparent; max-width: 100%; }
.src-chip .nm { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 16rem; }
.src-chip .pg { opacity: .75; }
.passage-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; font-size: 0.85rem; margin-top: 0.6rem; }
.passage-head .nm { font-weight: 650; }
.passage-head .pg { color: var(--muted); font-weight: 400; margin-left: .4rem; }
.passage-head .sc { color: var(--muted); font-size: 0.75rem; white-space: nowrap; }
.rel { height: 4px; border-radius: 999px; background: var(--surface-2); overflow: hidden; margin: 0.3rem 0 0.4rem; }
.rel > span { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
.passage { border-left: 3px solid var(--accent); padding: 0.35rem 0.8rem; margin-bottom: 0.5rem;
  font-size: 0.87rem; line-height: 1.55; opacity: 0.92; word-break: break-word; }
.note { color: var(--muted); font-size: 0.85rem; margin-top: 0.6rem; }

/* ---- empty states ---- */
.empty { border: 1px dashed var(--line); border-radius: var(--radius); background: var(--surface);
  padding: 2.4rem 1.5rem; text-align: center; margin: 1rem 0 1.2rem; }
.empty .ico { font-size: 2.1rem; line-height: 1; margin-bottom: 0.6rem; }
.empty .ttl { font-size: 1.15rem; font-weight: 650; margin-bottom: 0.3rem; }
.empty .bd { color: var(--muted); max-width: 30rem; margin: 0 auto; line-height: 1.55; }
.scope { color: var(--muted); font-size: 0.85rem; text-align: center; margin: 0.2rem 0 1rem; }

/* ---- error / notice cards ---- */
.card-note { border: 1px solid var(--line); border-left-width: 4px; border-radius: 10px; padding: 0.75rem 1rem; margin: 0.4rem 0; background: var(--surface); }
.card-note.err { border-left-color: var(--err); } .card-note.warn { border-left-color: var(--warn); } .card-note.ok { border-left-color: var(--ok); }
.card-note .ttl { font-weight: 650; margin-bottom: 0.15rem; }
.card-note .dt { font-size: 0.9rem; opacity: .9; }
.card-note .ht { font-size: 0.85rem; color: var(--muted); margin-top: 0.35rem; }

/* ---- documents ---- */
.doc-name { font-weight: 650; font-size: 1rem; word-break: break-word; }
.doc-meta { color: var(--muted); font-size: 0.83rem; margin-top: 0.15rem; }
.file-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid var(--line); }
.file-row:last-child { border-bottom: none; }
.file-row .nm { font-weight: 600; word-break: break-word; }
.file-row .dt { color: var(--muted); font-size: 0.82rem; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 0.8rem 1rem; }
[data-testid="stFileUploaderDropzone"] { border-radius: var(--radius); border: 1.5px dashed var(--line); background: var(--surface); padding: 1.6rem; }

/* ---- settings ---- */
.kv { display: grid; grid-template-columns: 11rem 1fr; gap: 0.35rem 1rem; font-size: 0.9rem; }
.kv .k { color: var(--muted); } .kv .v { word-break: break-word; }

/* ---- sidebar ---- */
.side-label { color: var(--muted); font-size: 0.72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; margin: 1rem 0 0.3rem; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 550; }
[class*="st-key-conv_"] button, [class*="st-key-convactive_"] button { justify-content: flex-start; text-align: left; border: 1px solid transparent; background: transparent; }
[class*="st-key-conv_"] button:hover { background: var(--surface-2); }
[class*="st-key-convactive_"] button { background: var(--accent-soft); font-weight: 650; }
[class*="st-key-conv_"] button p, [class*="st-key-convactive_"] button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---- small screens ---- */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
  .page-head .title { font-size: 1.35rem; }
  .empty { padding: 1.6rem 1rem; }
  .kv { grid-template-columns: 1fr; gap: 0.1rem 0; } .kv .v { margin-bottom: 0.5rem; }
  .src-chip .nm { max-width: 10rem; }
  .chat-head { flex-direction: column; align-items: flex-start; gap: 0.4rem; }
  .st-key-chat_suggestions [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  .st-key-chat_suggestions [data-testid="stColumn"] { min-width: min(100%, 12rem); flex: 1 1 12rem; }
}
</style>
"""


def inject_css(width: str = "narrow") -> None:
    """Inject the global stylesheet. `width` is 'narrow' (chat/settings) or 'wide' (documents)."""
    max_width = _WIDTHS.get(width, _WIDTHS["narrow"])
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<style>.block-container {{ max-width: {max_width} !important; }}</style>",
        unsafe_allow_html=True,
    )
