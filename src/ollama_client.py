"""Shared OpenAI-compatible client for the local Ollama server."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Ollama's OpenAI-compatible endpoint ignores this value.
OLLAMA_MODEL = "llama3.2"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return the process-wide client; create it once on first use."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    return _client


def validate_ollama() -> None:
    """Fail early with a useful message if Ollama or llama3.2 is unavailable."""
    from openai import APIConnectionError, APIStatusError

    try:
        models = get_client().models.list()
    except APIConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. Start Ollama "
            "and confirm it is serving requests, then run `ollama pull llama3.2`."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"Ollama at http://localhost:11434 returned HTTP {exc.status_code}. "
            "Check that the local Ollama service is running."
        ) from exc

    available = {model.id for model in models.data}
    if not any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in available):
        raise RuntimeError(
            "Ollama is running, but model `llama3.2` is not installed. "
            "Run `ollama pull llama3.2` and try again."
        )


def create_chat_completion(*, messages: list[dict[str, Any]], temperature: float = 0.2,
                           max_tokens: int | None = 2048, stream: bool = False) -> Any:
    """Create a completion with a friendly error for local-server failures."""
    from openai import APIConnectionError, APIStatusError

    try:
        request: dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if stream:
            request["stream"] = True
        return get_client().chat.completions.create(**request)
    except APIConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. Start Ollama "
            "and confirm `ollama list` shows llama3.2."
        ) from exc
    except APIStatusError as exc:
        detail = getattr(exc, "message", str(exc))
        if exc.status_code == 404 or "model" in str(detail).lower():
            raise RuntimeError(
                "Ollama could not find model `llama3.2`. Run `ollama pull llama3.2`."
            ) from exc
        raise RuntimeError(f"Ollama request failed (HTTP {exc.status_code}): {detail}") from exc
