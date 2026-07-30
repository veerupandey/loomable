"""loomable.providers.groq - Groq inference provider.

Groq uses the OpenAI-compatible Chat Completions API with a different base URL
and API key. Subclasses :class:`OpenAIProvider` so ``complete()`` and ``stream()``
work out of the box.
"""

from __future__ import annotations

import os

from .openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """A ``ModelProvider`` for Groq inference (OpenAI-compatible).

    Parameters
    ----------
    model:
        The Groq model name (e.g. ``"llama-3.3-70b-versatile"``,
        ``"mixtral-8x7b-32768"``).
    api_key:
        The Groq API key. Defaults to the ``GROQ_API_KEY`` environment variable.
    base_url:
        The API base URL. Defaults to ``https://api.groq.com/openai/v1``.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
        **kwargs,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    @property
    def _provider_id(self) -> str:
        return f"groq:{self.model}"
