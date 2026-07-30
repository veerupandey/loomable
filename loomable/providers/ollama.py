"""loomable.providers.ollama - Ollama local inference provider.

Ollama exposes an OpenAI-compatible API at ``http://localhost:11434/v1``.
Subclasses :class:`OpenAIProvider` so ``complete()`` and ``stream()``
work out of the box with any locally running Ollama model.
"""

from __future__ import annotations

from .openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """A ``ModelProvider`` for Ollama local models (OpenAI-compatible).

    Parameters
    ----------
    model:
        The Ollama model name (e.g. ``"llama3.2"``, ``"mistral"``,
        ``"codellama"``).
    base_url:
        The Ollama API base URL. Defaults to ``http://localhost:11434/v1``.
    timeout:
        Per-request timeout. Ollama can be slow on first load.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        # Ollama doesn't require an API key
        super().__init__(
            model=model,
            api_key="ollama",  # placeholder — Ollama ignores this
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    @property
    def _provider_id(self) -> str:
        return f"ollama:{self.model}"
