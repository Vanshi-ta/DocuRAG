"""Every user-facing DocuRAGError code must have a friendly title in the UI."""

from src.errors import DocuRAGError
from ui.components import _ERROR_TITLES


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def test_every_error_code_has_a_ui_title():
    codes = {sub.code for sub in _all_subclasses(DocuRAGError)}
    assert codes, "expected DocuRAGError to have subclasses"
    assert codes <= set(_ERROR_TITLES), f"codes without a UI title: {codes - set(_ERROR_TITLES)}"
