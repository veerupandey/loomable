"""Pluggable retrieval protocols (rewrite / rerank / route).

Every agentic step is a Protocol so callers can swap in custom LLMs,
cross-encoders, or heuristics without changing :class:`AgenticRetriever`.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, runtime_checkable

RetrievalMode = Literal["chunks", "file", "auto"]
BaseMode = Literal["vector", "lexical", "hybrid"]


@runtime_checkable
class QueryRewriter(Protocol):
    """Expand or transform a user query into one or more search queries."""

    name: str

    async def rewrite(self, query: str) -> list[str]:
        """Return search queries (may include the original)."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-order / filter retrieved hits before they reach the agent."""

    name: str

    async def rerank(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        ...


@runtime_checkable
class ModeRouter(Protocol):
    """Choose ``chunks`` vs ``file`` retrieval for a query."""

    name: str

    async def choose_mode(self, query: str) -> Literal["chunks", "file"]:
        ...


@runtime_checkable
class CorpusRouter(Protocol):
    """Choose which named corpora to query (multi-corpus agentic routing)."""

    name: str

    async def choose_corpora(
        self,
        query: str,
        corpora: Sequence[dict[str, str]],
    ) -> list[str]:
        """Return corpus names to query. ``corpora`` items have ``name`` + ``description``."""
        ...


@runtime_checkable
class HitCompressor(Protocol):
    """Optional post-step: shrink hit content to query-relevant spans."""

    name: str

    async def compress(
        self,
        query: str,
        hits: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...
