"""
LLM client module for DocuRAG.

Responsible ONLY for sending a fully-built prompt string to a local Ollama
model and returning its generated text.
"""

from __future__ import annotations

import logging

import requests

from config import (
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around Ollama's local REST API. Ollama must already be
    running and the target model already pulled (`ollama pull <model>`).
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. Is Ollama running? "
                f"Try `ollama list` in a terminal to check, or `ollama serve` "
                f"to start it manually."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama did not respond within {LLM_REQUEST_TIMEOUT_SECONDS}s. "
                f"The model may still be loading on first use, or your machine "
                f"may be under heavy load — try again."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise ConnectionError(
                f"Ollama returned an error ({response.status_code}). "
                f"Is model '{self.model}' pulled? Try `ollama pull {self.model}`."
            ) from exc

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            logger.warning("Ollama returned an empty response for this prompt")

        return answer
