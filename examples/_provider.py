"""Select a real OpenAI or Gemini model for examples that support both."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _backend():
    selected = os.getenv("LOOMABLE_PROVIDER")
    keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    }
    if selected and selected not in keys:
        raise SystemExit("Set LOOMABLE_PROVIDER to openai or gemini.")
    backend = selected or ("gemini" if keys["gemini"] else "openai")
    return backend, keys[backend]


def make_provider():
    backend, key = _backend()
    if not key:
        raise SystemExit(f"Set {backend.upper()}_API_KEY in your environment or the repo-root .env.")
    if backend == "gemini":
        from loomable.providers import GeminiProvider

        return GeminiProvider(os.getenv("GEMINI_MODEL", "gemini-flash-latest"), api_key=key)
    from loomable.providers import OpenAIProvider

    return OpenAIProvider(os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), api_key=key)


require_provider = make_provider


def make_embedder():
    backend, key = _backend()
    if not key:
        raise SystemExit(f"Set {backend.upper()}_API_KEY to run the embedding example.")
    if backend == "gemini":
        from loomable.providers import GeminiEmbedder

        return GeminiEmbedder(api_key=key)
    from loomable.providers import OpenAIEmbedder

    return OpenAIEmbedder(api_key=key)
