"""
LLM client module for DocuRAG.

Responsible ONLY for sending a fully-built prompt string to a local Ollama
model and returning its generated text, either all at once (`generate`) or
token-by-token (`generate_stream`), plus a non-raising `health_check`.

Failures are raised as the structured exceptions in src/errors.py. Each one
subclasses the built-in (ConnectionError / TimeoutError) the old code raised,
so existing callers and tests are unaffected.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator

import requests

from config import (
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
)
from src.errors import (
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 3
CHARS_PER_TOKEN_ESTIMATE = 3.5
ANSWER_RESERVE_TOKENS = 300

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
        num_ctx: int = OLLAMA_NUM_CTX,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx

    
    # --- diagnostics -------------------------------------------------------
    def _warn_if_prompt_may_not_fit(self, prompt: str) -> None:
        estimated = int(len(prompt) / CHARS_PER_TOKEN_ESTIMATE)
        budget = self.num_ctx - ANSWER_RESERVE_TOKENS
        if estimated > budget:
            logger.warning(
                "Prompt is ~%d tokens (estimated) but only ~%d fit in num_ctx=%d "
                "after reserving %d for the answer; Ollama will silently truncate it. "
                "Lower top_k or raise OLLAMA_NUM_CTX.",
                estimated, budget, self.num_ctx, ANSWER_RESERVE_TOKENS,
            )

    @staticmethod
    def _log_usage(data: Dict[str, Any]) -> None:
        def secs(key: str) -> float:
            return round(data.get(key, 0) / 1e9, 1)

        logger.info(
            "Ollama usage: prompt_tokens=%s output_tokens=%s prompt_eval_s=%s generation_s=%s total_s=%s",
            data.get("prompt_eval_count"), data.get("eval_count"),
            secs("prompt_eval_duration"), secs("eval_duration"), secs("total_duration"),
        )

    # --- request building / error mapping ---------------------------------
    def _payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

    def _post(self, prompt: str, stream: bool) -> requests.Response:
        """POST to /api/generate, translating transport failures into
        DocuRAG's structured errors."""
        self._warn_if_prompt_may_not_fit(prompt)
        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(
                url,
                json=self._payload(prompt, stream),
                timeout=LLM_REQUEST_TIMEOUT_SECONDS,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}. Is Ollama running? "
                f"Try `ollama list` in a terminal to check, or `ollama serve` "
                f"to start it manually."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {LLM_REQUEST_TIMEOUT_SECONDS}s. "
                f"The model may still be loading on first use, or your machine "
                f"may be under heavy load — try again."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                raise LLMModelNotFoundError(
                    f"Ollama returned an error ({status}). "
                    f"Is model '{self.model}' pulled? Try `ollama pull {self.model}`."
                ) from exc
            raise LLMResponseError(
                f"Ollama returned an error ({status}). "
                f"Is model '{self.model}' pulled? Try `ollama pull {self.model}`."
            ) from exc

    # --- generation --------------------------------------------------------
    def generate(self, prompt: str) -> str:
        response = self._post(prompt, stream=False)
        data = response.json()
        self._log_usage(data)
        answer = data.get("response", "").strip()

        if not answer:
            logger.warning("Ollama returned an empty response for this prompt")

        return answer

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the answer as text fragments.

        The HTTP request is opened immediately, so connection / model errors
        are raised from THIS call (before a UI has started drawing an empty
        answer bubble). Errors that happen mid-stream are raised while
        iterating.
        """
        response = self._post(prompt, stream=True)
        return self._iter_tokens(response)

    def _iter_tokens(self, response: requests.Response) -> Iterator[str]:
        produced_any = False
        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line in Ollama stream")
                    continue

                if data.get("error"):
                    raise LLMResponseError(f"Ollama reported an error: {data['error']}")

                token = data.get("response", "")
                if token:
                    produced_any = True
                    yield token
                    
                if data.get("done"):
                    self._log_usage(data)
                    break
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Ollama stopped responding mid-answer (timeout {LLM_REQUEST_TIMEOUT_SECONDS}s)."
            ) from exc
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            raise LLMUnavailableError(
                f"Lost connection to Ollama at {self.base_url} while generating the answer."
            ) from exc
        finally:
            response.close()

        if not produced_any:
            logger.warning("Ollama returned an empty streamed response for this prompt")

    # --- health ------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Never raises. Lets a UI show "Ollama offline" / "model missing"
        states up front instead of discovering them on the first question.

        Returns: {"reachable": bool, "model": str, "model_available": bool,
                  "available_models": [str, ...]}
        """
        result: Dict[str, Any] = {
            "reachable": False,
            "model": self.model,
            "model_available": False,
            "available_models": [],
        }
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:  # deliberately broad: a health probe must not raise
            logger.info("Ollama health check failed: %s", exc)
            return result

        result["reachable"] = True
        result["available_models"] = names
        result["model_available"] = self.model in names or (
            ":" not in self.model and f"{self.model}:latest" in names
        )
        return result
