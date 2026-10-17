"""Shared Gemini and OpenAI provider configuration for public examples."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo-root .env even when the script cwd is examples/<subdir>
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()


def has_live_provider() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def make_provider():
    """Return a configured live model provider for demos."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        from loomable.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            api_key=gemini_key,
            timeout=120.0,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from loomable.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            timeout=120.0,
        )

    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY to run this example.")


def require_provider():
    """Like ``make_provider``, but fail clearly when no credentials are set."""
    if not has_live_provider():
        raise SystemExit(
            "Set GEMINI_API_KEY or OPENAI_API_KEY in the environment or repo-root .env."
        )
    return make_provider()


def make_embedder():
    """Return an embedder aligned with ``make_provider`` credential priority."""
    require_provider()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        from loomable.providers import GeminiEmbedder

        return GeminiEmbedder(api_key=gemini_key)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from loomable.providers import OpenAIEmbedder

        return OpenAIEmbedder(api_key=openai_key)

    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY to run this example.")
