"""
Automated tests for src/generation/llm_client.py.

Uses unittest.mock to fake HTTP responses from Ollama's REST API, so these
tests run fast and offline — no real Ollama installation or running model
required. This tests OUR error-handling and payload-construction logic,
not Ollama itself.

Run from the project root with:
    pytest tests/test_llm_client.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.generation.llm_client import OllamaClient


def make_mock_response(json_data: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


@patch("src.generation.llm_client.requests.post")
def test_generate_returns_stripped_response_text(mock_post):
    mock_post.return_value = make_mock_response({"response": "  The answer is 42.  "})
    client = OllamaClient(model="fake-model")

    result = client.generate("some prompt")

    assert result == "The answer is 42."
    mock_post.assert_called_once()


@patch("src.generation.llm_client.requests.post")
def test_generate_sends_correct_payload(mock_post):
    mock_post.return_value = make_mock_response({"response": "answer"})
    client = OllamaClient(model="llama3.2:3b", temperature=0.2)

    client.generate("my prompt")

    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3.2:3b"
    assert payload["prompt"] == "my prompt"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.2


@patch("src.generation.llm_client.requests.post")
def test_generate_raises_connection_error_with_helpful_message(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()
    client = OllamaClient()

    with pytest.raises(ConnectionError, match="Ollama"):
        client.generate("some prompt")


@patch("src.generation.llm_client.requests.post")
def test_generate_raises_timeout_error_with_helpful_message(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()
    client = OllamaClient()

    with pytest.raises(TimeoutError, match="Ollama"):
        client.generate("some prompt")


@patch("src.generation.llm_client.requests.post")
def test_generate_returns_empty_string_without_crashing_on_empty_response(mock_post):
    mock_post.return_value = make_mock_response({"response": ""})
    client = OllamaClient()

    result = client.generate("some prompt")

    assert result == ""
