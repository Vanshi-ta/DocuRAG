"""Tests for OllamaClient.generate_stream, health_check and structured errors.
Fully offline: requests is mocked."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.errors import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from src.generation.llm_client import OllamaClient


def stream_response(lines):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = [json.dumps(x).encode() if isinstance(x, dict) else x for x in lines]
    return resp


def http_error_response(status):
    resp = MagicMock()
    resp.status_code = status
    err = requests.exceptions.HTTPError(response=resp)
    resp.raise_for_status.side_effect = err
    return resp


@patch("src.generation.llm_client.requests.post")
def test_generate_stream_yields_tokens_in_order(mock_post):
    mock_post.return_value = stream_response(
        [{"response": "Hel", "done": False}, {"response": "lo", "done": False}, {"response": "", "done": True}]
    )
    tokens = list(OllamaClient(model="m").generate_stream("prompt"))
    assert tokens == ["Hel", "lo"]
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["stream"] is True
    assert kwargs["stream"] is True
    mock_post.return_value.close.assert_called_once()


@patch("src.generation.llm_client.requests.post")
def test_generate_stream_skips_blank_and_garbage_lines(mock_post):
    mock_post.return_value = stream_response([b"", b"not json", {"response": "ok", "done": True}])
    assert list(OllamaClient().generate_stream("p")) == ["ok"]


@patch("src.generation.llm_client.requests.post")
def test_generate_stream_raises_unavailable_at_call_time(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()
    with pytest.raises(LLMUnavailableError, match="Ollama"):
        OllamaClient().generate_stream("p")  # raised here, not on first next()


@patch("src.generation.llm_client.requests.post")
def test_new_errors_are_still_builtin_errors(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()
    with pytest.raises(ConnectionError):
        OllamaClient().generate("p")
    mock_post.side_effect = requests.exceptions.Timeout()
    with pytest.raises(TimeoutError):
        OllamaClient().generate("p")
    with pytest.raises(LLMTimeoutError):
        OllamaClient().generate("p")


@patch("src.generation.llm_client.requests.post")
def test_404_maps_to_model_not_found_and_other_http_to_response_error(mock_post):
    mock_post.return_value = http_error_response(404)
    with pytest.raises(LLMModelNotFoundError):
        OllamaClient(model="nope").generate("p")
    mock_post.return_value = http_error_response(500)
    with pytest.raises(LLMResponseError):
        OllamaClient().generate("p")


@patch("src.generation.llm_client.requests.post")
def test_error_line_mid_stream_raises(mock_post):
    mock_post.return_value = stream_response([{"response": "a"}, {"error": "model crashed"}])
    gen = OllamaClient().generate_stream("p")
    assert next(gen) == "a"
    with pytest.raises(LLMResponseError, match="model crashed"):
        next(gen)


def test_errors_expose_code_and_hint():
    err = LLMUnavailableError("down")
    assert err.code == "llm_unavailable"
    assert err.hint
    assert str(err) == "down"


@patch("src.generation.llm_client.requests.get")
def test_health_check_reports_model_availability(mock_get):
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": "llama3.2:3b"}, {"name": "mistral:latest"}]}
    mock_get.return_value = resp

    ok = OllamaClient(model="llama3.2:3b").health_check()
    assert ok["reachable"] and ok["model_available"]
    assert OllamaClient(model="mistral").health_check()["model_available"]  # implicit :latest
    missing = OllamaClient(model="qwen").health_check()
    assert missing["reachable"] and not missing["model_available"]


@patch("src.generation.llm_client.requests.get")
def test_health_check_never_raises_when_ollama_is_down(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()
    status = OllamaClient().health_check()
    assert status["reachable"] is False
    assert status["model_available"] is False
