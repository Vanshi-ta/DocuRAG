"""
DocuRAG — Streamlit entry point.

Sets up the page, theme, navigation and shared state, then hands off to the
selected view in views/. All RAG logic lives in src/ (via
src.services.DocuRAGService); the files in ui/ and views/ only render.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.logging_config import configure_logging
from ui.state import init_state

configure_logging()

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(
    page_title="DocuRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"))

init_state()

navigation = st.navigation(
    [
        st.Page("views/chat.py", title="Chat", icon=":material/chat_bubble:", default=True),
        st.Page("views/documents.py", title="Documents", icon=":material/folder_open:"),
        st.Page("views/settings.py", title="Settings", icon=":material/tune:"),
    ]
)
navigation.run()
