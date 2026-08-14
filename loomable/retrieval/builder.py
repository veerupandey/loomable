"""High-level ``build_retriever`` / ``build_corpus`` APIs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence

from loomable.codeindex.embedders import HashingEmbedder
from loomable.kernel.contracts import Retriever, VectorBackend
from loomable.kernel.long_term import InMemoryVectorBackend, LongTermStore
from loomable.retrieval.chunking.base import ChunkStrategy, resolve_strategy
from loomable.retrieval.ingest import load_sources
from loomable.retrieval.retrievers import HybridRetriever, LexicalRetriever, VectorRetriever
from loomable.retrieval.types import Chunk, Document


async def chunk_documents(
    documents: Sequence[Document],
    *,
    strategy: str | ChunkStrategy = "auto",
) -> list[Chunk]:
    """Apply a chunk strategy to many documents."""
    strat = resolve_strategy(strategy)
    out: list[Chunk] = []
    for doc in documents:
        out.extend(strat.chunk(doc))
    return out


async def build_corpus(
    sources: Sequence[Any],
    *,
    strategy: str | ChunkStrategy = "auto",
) -> tuple[list[Document], list[Chunk]]:
    """Load sources and chunk them. Returns ``(documents, chunks)``."""
    docs = load_sources(sources)
    chunks = await chunk_documents(docs, strategy=strategy)
    return docs, chunks


def _make_store(
    *,
    store: LongTermStore | None,
    backend: VectorBackend | None,
    persist_path: str | Path | None,
) -> LongTermStore:
    if store is not None:
        return store
    if backend is not None:
        return LongTermStore(backend=backend, backend_name="custom")
    if persist_path is not None:
        return LongTermStore(path=persist_path, backend_name="zvec")
    # Ephemeral retrieval corpus — not agent L3; keep zero-dep in-memory.
    return LongTermStore(backend=InMemoryVectorBackend(), backend_name="memory")


async def build_retriever(
    sources: Sequence[Any],
    *,
    name: str = "search_docs",
    mode: str = "hybrid",
    strategy: str | ChunkStrategy = "auto",
    embedder: Any | None = None,
    store: LongTermStore | None = None,
    backend: VectorBackend | None = None,
    persist_path: str | Path | None = None,
    vector_weight: float = 0.7,
) -> Retriever:
    """Build a ready-to-ship :class:`~loomable.kernel.contracts.Retriever` tool.

    Parameters
    ----------
    sources:
        Files, directories, inline strings, dicts, or :class:`Document`s.
    name:
        Agent tool name (normalized to ``search_*``).
    mode:
        ``"hybrid"`` (default, RRF), ``"vector"``, or ``"lexical"``.
    strategy:
        Chunk strategy name (``auto`` / ``text`` / ``markdown`` / ``code`` /
        ``html`` / ``pdf``) or a custom :class:`ChunkStrategy`.
    embedder:
        Any object with ``async embed(text) -> list[float]``. Defaults to
        :class:`~loomable.codeindex.embedders.HashingEmbedder` for offline use.
    store / backend / persist_path:
        Pluggable vector storage. ``persist_path`` uses **Alibaba zvec**
        (``pip install loomable[zvec]``). Pass ``store=open_vector_store(
        engine="faiss", ...)`` for FAISS, ``postgres_url=...`` for Postgres,
        or ``backend=`` for any :class:`~loomable.kernel.contracts.VectorBackend`.
        Omit both for an in-memory store (tests / ephemeral).

    Examples
    --------
    ::

        retriever = await build_retriever(
            ["./docs", "./README.md"],
            name="search_docs",
            mode="hybrid",
            persist_path="./.loomable/docs_zvec",  # Alibaba zvec on disk
        )
        agent = Agent(model=..., retrievers=[retriever])  # tool: search_docs
    """
    from loomable.retrieval.naming import ensure_search_tool_name

    tool_name = ensure_search_tool_name(name)
    _docs, chunks = await build_corpus(sources, strategy=strategy)
    mode_key = (mode or "hybrid").strip().lower()
    emb = embedder or HashingEmbedder()

    if mode_key == "lexical":
        return LexicalRetriever(tool_name, chunks)

    lt = _make_store(store=store, backend=backend, persist_path=persist_path)
    vector = VectorRetriever(
        tool_name if mode_key == "vector" else f"{tool_name}__vector",
        store=lt,
        embedder=emb,
    )
    await vector.index_chunks(chunks)

    if mode_key == "vector":
        vector.name = tool_name
        return vector

    if mode_key == "hybrid":
        lexical = LexicalRetriever(f"{tool_name}__lexical", chunks)
        return HybridRetriever(
            tool_name, vector=vector, lexical=lexical, vector_weight=vector_weight
        )

    raise ValueError(f"mode must be 'vector', 'lexical', or 'hybrid', got {mode!r}")


def build_retriever_sync(sources: Sequence[Any], **kwargs: Any) -> Retriever:
    """Sync wrapper around :func:`build_retriever`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build_retriever(sources, **kwargs))
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(lambda: asyncio.run(build_retriever(sources, **kwargs))).result()
