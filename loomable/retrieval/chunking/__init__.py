"""Built-in chunk strategies for loomable.retrieval."""

from __future__ import annotations

from loomable.retrieval.chunking.base import (
    ChunkStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
    resolve_strategy,
)

_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Side-effect imports register strategies
    from loomable.retrieval.chunking import code as _code  # noqa: F401
    from loomable.retrieval.chunking import html_pdf as _html_pdf  # noqa: F401
    from loomable.retrieval.chunking import markdown as _markdown  # noqa: F401
    from loomable.retrieval.chunking import text as _text  # noqa: F401

    _BUILTINS_LOADED = True


__all__ = [
    "ChunkStrategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "resolve_strategy",
    "_ensure_builtins",
]
