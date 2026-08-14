"""Chunk strategy protocol and registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from loomable.retrieval.types import Chunk, Document

__all__ = [
    "ChunkStrategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "resolve_strategy",
]


@runtime_checkable
class ChunkStrategy(Protocol):
    """Turn a :class:`Document` into retrieval :class:`Chunk`s."""

    name: str

    def chunk(self, document: Document) -> list[Chunk]:
        ...


_REGISTRY: dict[str, ChunkStrategy] = {}


def register_strategy(strategy: ChunkStrategy) -> ChunkStrategy:
    _REGISTRY[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> ChunkStrategy:
    key = (name or "auto").strip().lower()
    if key not in _REGISTRY:
        from loomable.retrieval.chunking import _ensure_builtins

        _ensure_builtins()
    if key not in _REGISTRY:
        raise KeyError(f"unknown chunk strategy: {name!r}")
    return _REGISTRY[key]


def list_strategies() -> list[str]:
    from loomable.retrieval.chunking import _ensure_builtins

    _ensure_builtins()
    return sorted(_REGISTRY)


def resolve_strategy(strategy: str | ChunkStrategy | None) -> ChunkStrategy:
    if strategy is None:
        return get_strategy("auto")
    if isinstance(strategy, str):
        return get_strategy(strategy)
    return strategy
