"""Tests for config.py's environment-variable parsing helpers."""

import pytest

from config import _env_bool, _env_float, _env_int


def test_env_int_and_float_fall_back_to_default_when_unset_or_empty(monkeypatch):
    monkeypatch.delenv("DOCURAG_TEST_VAR", raising=False)
    assert _env_int("DOCURAG_TEST_VAR", 7) == 7
    assert _env_float("DOCURAG_TEST_VAR", 0.5) == 0.5

    monkeypatch.setenv("DOCURAG_TEST_VAR", "")
    assert _env_int("DOCURAG_TEST_VAR", 7) == 7
    assert _env_float("DOCURAG_TEST_VAR", 0.5) == 0.5


def test_env_int_and_float_parse_values(monkeypatch):
    monkeypatch.setenv("DOCURAG_TEST_VAR", "12")
    assert _env_int("DOCURAG_TEST_VAR", 7) == 12
    assert _env_float("DOCURAG_TEST_VAR", 0.5) == 12.0


def test_env_int_rejects_non_numeric_values(monkeypatch):
    monkeypatch.setenv("DOCURAG_TEST_VAR", "abc")
    with pytest.raises(ValueError):
        _env_int("DOCURAG_TEST_VAR", 7)


@pytest.mark.parametrize("default, expected", [(True, True), (False, False)])
def test_env_bool_uses_default_when_unset(monkeypatch, default, expected):
    monkeypatch.delenv("DOCURAG_TEST_VAR", raising=False)
    assert _env_bool("DOCURAG_TEST_VAR", default) is expected


@pytest.mark.parametrize("raw, expected", [("true", True), ("TRUE", True), ("True", True),
                                           ("false", False), ("1", False), ("yes", False)])
def test_env_bool_only_the_word_true_enables_a_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("DOCURAG_TEST_VAR", raw)
    assert _env_bool("DOCURAG_TEST_VAR", False) is expected


def test_env_bool_empty_string_is_false_even_when_default_is_true(monkeypatch):
    # Documents a deliberate difference from _env_int/_env_float, where an
    # empty value means "unset".
    monkeypatch.setenv("DOCURAG_TEST_VAR", "")
    assert _env_bool("DOCURAG_TEST_VAR", True) is False
