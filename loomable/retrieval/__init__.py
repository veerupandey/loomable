"""loomable.retrieval — document ingest, chunking, and retriever builders.

High-level::

    from loomable.retrieval import build_retriever

    retriever = await build_retriever(
        ["./docs", "./README.md", {"id": "note", "text": "…"}],
        name="docs",
        mode="hybrid",          # vector | lexical | hybrid
        strategy="auto",        # text | markdown | code | html | pdf | auto
        persist_path="./.loomable/docs.zvec.json",
    )
    agent = Agent(model=..., retrievers=[retriever])

Deep code uses the same stack via :class:`~loomable.codeindex.CodeIndex`.
"""

from __future__ import annotations

from loomable.retrieval.builder import (
    build_corpus,
    build_retriever,
    build_retriever_sync,
    chunk_documents,
)
from loomable.retrieval.chunking import (
    ChunkStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
    resolve_strategy,
)
from loomable.retrieval.ingest import coerce_source, load_directory, load_file, load_sources
from loomable.retrieval.retrievers import HybridRetriever, LexicalRetriever, VectorRetriever
from loomable.retrieval.types import Chunk, Document

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "Document",
    "HybridRetriever",
    "LexicalRetriever",
    "VectorRetriever",
    "build_corpus",
    "build_retriever",
    "build_retriever_sync",
    "chunk_documents",
    "coerce_source",
    "get_strategy",
    "list_strategies",
    "load_directory",
    "load_file",
    "load_sources",
    "register_strategy",
    "resolve_strategy",
]
