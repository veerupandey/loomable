"""loomable.retrieval — ingest, chunking, and pluggable agentic retrievers.

High-level::

    from loomable.retrieval import ingest, build_agentic_retriever
    from loomable.providers.vector_store import open_vector_store

    corpus = await ingest(
        ["./docs", "./README.md"],
        name="docs",
        store=open_vector_store(path="./.loomable/docs_zvec"),  # or faiss/postgres
        strategy="auto",
        base_mode="hybrid",
    )

    retriever = await build_agentic_retriever(
        corpus,
        mode="auto",              # chunks | file | auto | custom ModeRouter
        rewrite="off",            # off | multi_query | hyde | custom QueryRewriter
        rerank=True,              # off | score | llm | custom Reranker
        compress="off",           # off | llm | custom HitCompressor
        # llm=provider,          # required for multi_query / hyde / llm rerank
    )
    agent = Agent(model=..., retrievers=[retriever])

Everything is pluggable: store, chunk strategy, base mode, rewrite, mode router,
rerank, compress, and multi-corpus ``CorpusRouter``.

Deep code uses the same stack via :class:`~loomable.codeindex.CodeIndex`.
"""

from __future__ import annotations

from loomable.retrieval.agentic import (
    AgenticRetriever,
    CompositeRetriever,
    build_agentic_retriever,
)
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
from loomable.retrieval.corpus import Corpus, ingest
from loomable.retrieval.ingest import coerce_source, load_directory, load_file, load_sources
from loomable.retrieval.plugins import (
    CorpusRouter,
    HitCompressor,
    ModeRouter,
    QueryRewriter,
    Reranker,
)
from loomable.retrieval.rerank import (
    IdentityCompressor,
    IdentityReranker,
    LLMCompressor,
    LLMReranker,
    ScoreReranker,
)
from loomable.retrieval.retrievers import HybridRetriever, LexicalRetriever, VectorRetriever
from loomable.retrieval.rewrite import HyDERewriter, IdentityRewriter, MultiQueryRewriter
from loomable.retrieval.route import (
    AllCorporaRouter,
    DescriptionCorpusRouter,
    FixedModeRouter,
    HeuristicModeRouter,
    LLMModeRouter,
)
from loomable.retrieval.types import Chunk, Document
from loomable.providers.vector_store import open_vector_store

__all__ = [
    # Core types
    "Chunk",
    "ChunkStrategy",
    "Document",
    "Corpus",
    "open_vector_store",
    # Ingest / build
    "ingest",
    "build_corpus",
    "build_retriever",
    "build_retriever_sync",
    "build_agentic_retriever",
    "chunk_documents",
    "coerce_source",
    "load_directory",
    "load_file",
    "load_sources",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "resolve_strategy",
    # Base retrievers
    "HybridRetriever",
    "LexicalRetriever",
    "VectorRetriever",
    "AgenticRetriever",
    "CompositeRetriever",
    # Protocols
    "QueryRewriter",
    "Reranker",
    "ModeRouter",
    "CorpusRouter",
    "HitCompressor",
    # Built-ins
    "IdentityRewriter",
    "MultiQueryRewriter",
    "HyDERewriter",
    "IdentityReranker",
    "ScoreReranker",
    "LLMReranker",
    "IdentityCompressor",
    "LLMCompressor",
    "FixedModeRouter",
    "HeuristicModeRouter",
    "LLMModeRouter",
    "AllCorporaRouter",
    "DescriptionCorpusRouter",
]
