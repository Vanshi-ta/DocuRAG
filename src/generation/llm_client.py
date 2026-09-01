"""
LLM client module for DocuRAG — Phase 6.

Responsible ONLY for sending a fully-built prompt string to a local Ollama
model and returning its generated text. No prompt construction, no
retrieval logic lives here — this module's only job is "send prompt
string, get response string back," talking to Ollama's local REST API.
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
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class OllamaClient:
    """
    Thin wrapper around Ollama's local REST API (http://localhost:11434 by
    default). Ollama must already be running — on Windows it runs as a
    background service that starts automatically after installation — and
    the target model must already be pulled (`ollama pull <model>`); see
    Phase 6 guide Sections 3-4.
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
        """
        Send `prompt` to the local Ollama model and return its full
        generated text.

        Uses Ollama's /api/generate endpoint with stream=False — we wait
        for the complete response rather than handling a token-by-token
        stream. That keeps this module simple (one request, one response)
        at the cost of not showing partial output while the model is still
        generating; acceptable for this phase, and easy to swap to
        streaming later without touching any other module.
        """
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
                f"On Windows it should start automatically after installation — "
                f"try running `ollama list` in a terminal to check, or "
                f"`ollama serve` to start it manually."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Ollama did not respond within {LLM_REQUEST_TIMEOUT_SECONDS}s. "
                f"The model may still be loading into memory on first use, or "
                f"your machine may be under heavy load — try again."
            ) from exc

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            logger.warning("Ollama returned an empty response for this prompt")

        return answer
