from types import SimpleNamespace

import pytest


from src.services.conversation_store import DEFAULT_TITLE, ConversationStore


def fake_chunk(name="a.pdf", page=1):
    return SimpleNamespace(source_filename=name, page_number=page, similarity_score=0.5,
                           chunk_id="c1", chunk_text="some text")


def test_create_append_and_reload_roundtrip(tmp_path):
    store = ConversationStore(tmp_path / "convs")
    conv = store.create()
    assert conv.title == DEFAULT_TITLE

    store.append_message(conv.id, "user", "What is the refund window?")
    store.append_message(conv.id, "assistant", "30 days.", sources=[fake_chunk()], used_llm=True)

    loaded = store.get(conv.id)
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].sources[0].source_filename == "a.pdf"
    assert loaded.messages[1].sources[0].chunk_text == "some text"
    assert loaded.messages[1].used_llm is True


def test_first_user_message_becomes_title_and_is_truncated(tmp_path):
    store = ConversationStore(tmp_path)
    conv = store.create()
    store.append_message(conv.id, "user", "x" * 100)
    title = store.get(conv.id).title
    assert title.endswith("…") and len(title) == 61


def test_list_orders_most_recent_first_and_counts_messages(tmp_path):
    store = ConversationStore(tmp_path)
    a, b = store.create(), store.create()
    store.append_message(a.id, "user", "older chat touched last")
    summaries = store.list()
    assert [s.id for s in summaries][0] == a.id
    assert {s.id for s in summaries} == {a.id, b.id}
    assert next(s for s in summaries if s.id == a.id).message_count == 1


def test_error_messages_are_flagged(tmp_path):
    store = ConversationStore(tmp_path)
    conv = store.create()
    store.append_message(conv.id, "assistant", "Ollama is down", is_error=True)
    assert store.get(conv.id).messages[0].is_error is True


def test_rename_and_delete(tmp_path):
    store = ConversationStore(tmp_path)
    conv = store.create()
    assert store.rename(conv.id, "  Policies  ").title == "Policies"
    assert store.delete(conv.id) is True
    assert store.delete(conv.id) is False
    assert store.list() == []


def test_invalid_ids_are_rejected_to_prevent_path_traversal(tmp_path):
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../../etc/passwd")
    with pytest.raises(ValueError):
        store.append_message("abc", "user", "hi")


def test_corrupt_file_is_skipped_in_listing(tmp_path):
    store = ConversationStore(tmp_path)
    store.create()
    (tmp_path / "broken.json").write_text("{not json")
    assert len(store.list()) == 1
