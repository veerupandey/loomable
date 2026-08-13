"""Shared provider helper for simple use-case examples.

Picks the first available backend from env:
  1) Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)
  2) Z.AI / OpenAI-compatible (ZAI_API_KEY or OPENAI_API_KEY + optional base URL)
  3) Azure OpenAI (AZURE_OPENAI_*)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def make_provider():
    """Return a configured model provider for demos."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        from loomable.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            api_key=gemini_key,
            timeout=120.0,
        )

    zai_key = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if zai_key:
        from loomable.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=os.environ.get("ZAI_MODEL")
            or os.environ.get("OPENAI_MODEL", "glm-5.2"),
            api_key=zai_key,
            base_url=os.environ.get(
                "ZAI_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
            ),
            timeout=120.0,
        )

    from loomable.providers.openai import AzureOpenAIProvider

    return AzureOpenAIProvider()
