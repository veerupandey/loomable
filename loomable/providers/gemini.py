"""loomable.providers.gemini - Google Gemini provider.

Google exposes Gemini models through an OpenAI-compatible endpoint at
``https://generativelanguage.googleapis.com/v1beta/openai``. This provider
subclasses :class:`OpenAIProvider` and adapts the auth (API key as query param
or bearer token depending on the endpoint variant).

For Vertex AI, point ``base_url`` at the Vertex OpenAI-compatible endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from .openai import OpenAIProvider


class GeminiProvider(OpenAIProvider):
    """A ``ModelProvider`` for Google Gemini via the OpenAI-compatible API.

    Parameters
    ----------
    model:
        The Gemini model name (e.g. ``"gemini-2.0-flash"``, ``"gemini-1.5-pro"``).
    api_key:
        The Google AI API key. Defaults to the ``GOOGLE_API_KEY`` or
        ``GEMINI_API_KEY`` environment variable.
    base_url:
        The API base URL. Defaults to Google's OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        resolved_key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        )
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

    @property
    def _provider_id(self) -> str:
        return f"gemini:{self.model}"
