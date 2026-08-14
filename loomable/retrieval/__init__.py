"""loomable.retrieval — ingest, chunking, and pluggable agentic retrievers.

Ship any :class:`~loomable.kernel.contracts.Retriever` to the agent::

    from loomable.retrieval import ingest, build_agentic_retriever
    from loomable.providers.vector_store import open_vector_store

    corpus = await ingest(
        ["./docs", "./README.md"],
        name="docs",                          # corpus id
        store=open_vector_store(path="./.loomable/docs_zvec"),
        strategy="auto",
        base_mode="hybrid",                   # RRF hybrid (beats naive vector top-k)
    )

    retriever = await build_agentic_retriever(
        corpus,
        name="search_docs",                   # agent tool name
        mode="auto",
        rewrite="off",
        rerank="mmr",                         # default; diversity + relevance
        compress="off",
    )
    agent = Agent(model=..., retrievers=[retriever])  # LLM calls search_docs

Or attach a vector DB directly (the knowledge base *is* the store)::

    agent = Agent(model=..., knowledge_base=store)          # search_knowledge
    agent = Agent(model=..., knowledge_base={"policy": store, "notes": ["./notes"]})
    create_deep_agent(model, knowledge_base=store, retrievers=[custom])

Custom retrievers work the same — implement ``name`` + ``async retrieve`` and
pass ``Agent(retrievers=[my_retriever])``.
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
from loomable.retrieval.ingest import (
    coerce_source,
    is_http_url,
    load_directory,
    load_file,
    load_sources,
    load_url,
)
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
    MMRReranker,
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
from loomable.retrieval.metadata import matches_filters, shape_hit
from loomable.retrieval.types import Chunk, Document
from loomable.retrieval.knowledge import (
    KnowledgeBase,
    build_knowledge_retriever,
    resolve_knowledge_base,
)
from loomable.retrieval.naming import (
    DEFAULT_SEARCH_DOCS,
    DEFAULT_SEARCH_KNOWLEDGE,
    ensure_search_tool_name,
)
from loomable.providers.vector_store import open_vector_store

__all__ = [
    # Core types
    "Chunk",
    "ChunkStrategy",
    "Document",
    "Corpus",
    "KnowledgeBase",
    "open_vector_store",
    # Ingest / build
    "ingest",
    "build_knowledge_retriever",
    "resolve_knowledge_base",
    "build_corpus",
    "build_retriever",
    "build_retriever_sync",
    "build_agentic_retriever",
    "chunk_documents",
    "coerce_source",
    "is_http_url",
    "load_directory",
    "load_file",
    "load_sources",
    "load_url",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "resolve_strategy",
    "ensure_search_tool_name",
    "DEFAULT_SEARCH_DOCS",
    "DEFAULT_SEARCH_KNOWLEDGE",
    "matches_filters",
    "shape_hit",
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
    "MMRReranker",
    "LLMReranker",
    "IdentityCompressor",
    "LLMCompressor",
    "FixedModeRouter",
    "HeuristicModeRouter",
    "LLMModeRouter",
    "AllCorporaRouter",
    "DescriptionCorpusRouter",
]
