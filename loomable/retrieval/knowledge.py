"""Knowledge bases — a vector store the agent can search.

Industry pattern (Agno Knowledge, LlamaIndex VectorStoreIndex, LangChain
VectorStoreRetriever): the knowledge base *is* a vector DB. Optional sources
are ingested into that store; optional :class:`~loomable.kernel.contracts.Retriever`
objects ship as extra ``search_*`` tools.

Pass the same object to :class:`~loomable.agent.builder.Agent`,
:func:`~loomable.agent.deep.create_deep_agent` (a thin Agent harness),
:class:`~loomable.agent.team.Team`, :class:`~loomable.case.Case`,
:class:`~loomable.flow.workflow.Workflow`, and :class:`~loomable.flow.flow.Flow`.

::

    from loomable import Agent, create_deep_agent
    from loomable.providers.vector_store import open_vector_store
    from loomable.retrieval import KnowledgeBase

    store = open_vector_store(engine="faiss", path="./.loomable/kb", dimensions=384)

    agent = Agent(
        model=...,
        knowledge_base=KnowledgeBase(store=store, sources=["./handbook.pdf"]),
        retrievers=[custom],          # extra search tools, same agent
    )

    # Named collections (personal vs company) → search_personal, search_company
    Agent(model=..., knowledge_base={"personal": ["./notes"], "company": store})

    # Deep agent is Agent — same kwargs
    create_deep_agent(model, knowledge_base=store, embedder=embedder)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loomable.kernel.contracts import Retriever
from loomable.kernel.long_term import LongTermStore
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval.naming import DEFAULT_SEARCH_KNOWLEDGE, ensure_search_tool_name

__all__ = [
    "KnowledgeBase",
    "build_knowledge_retriever",
    "is_vector_store",
    "resolve_knowledge_base",
    "run_sync",
]

_STORE_URI_PREFIXES = (
    "zvec:",
    "faiss:",
    "chroma:",
    "milvus:",
    "postgres:",
    "postgresql:",
    "memory:",
    "file:",
)

_COLLECTION_SPEC_KEYS = frozenset(
    {
        "sources",
        "store",
        "uri",
        "backend",
        "vector_store",
        "retriever",
        "embedder",
        "description",
        "metadata",
        "strategy",
        "base_mode",
        "name",
    }
)

_STORE_HINT_KEYS = frozenset(
    {"sources", "store", "uri", "backend", "vector_store", "retriever"}
)


def run_sync(coro: Any) -> Any:
    """Run *coro* from sync code (Agent.build), including inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def is_vector_store(obj: Any) -> bool:
    """True for :class:`LongTermStore` or a VectorBackend (``index`` + ``query``)."""
    if obj is None or _is_retriever(obj):
        return False
    if isinstance(obj, LongTermStore):
        return True
    return callable(getattr(obj, "index", None)) and callable(getattr(obj, "query", None))


def _is_retriever(obj: Any) -> bool:
    return obj is not None and callable(getattr(obj, "retrieve", None)) and bool(
        getattr(obj, "name", None)
    )


def _is_store_uri(value: str) -> bool:
    low = (value or "").strip().lower()
    return any(low.startswith(p) for p in _STORE_URI_PREFIXES)


def _is_collection_spec(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = set(value.keys())
    return bool(keys & _STORE_HINT_KEYS) and keys <= (_COLLECTION_SPEC_KEYS | _STORE_HINT_KEYS)


def _as_store(store: Any) -> LongTermStore:
    if store is None:
        return open_vector_store(engine="memory")
    if isinstance(store, LongTermStore):
        return store
    if isinstance(store, str) and _is_store_uri(store):
        return open_vector_store(uri=store)
    if is_vector_store(store):
        return LongTermStore(backend=store, backend_name="custom")
    raise TypeError(
        "knowledge_base store must be a LongTermStore, VectorBackend, or "
        f"vector URI, got {type(store).__name__}"
    )


def _default_embedder(embedder: Any | None) -> Any:
    if embedder is not None:
        return embedder
    from loomable.codeindex.embedders import HashingEmbedder

    return HashingEmbedder()


async def build_knowledge_retriever(
    sources: Sequence[Any],
    *,
    name: str = "knowledge",
    description: str = "",
    user_id: str | None = None,
    scope: str | None = None,
    embedder: Any | None = None,
    store: Any | None = None,
    metadata: dict[str, Any] | None = None,
    base_mode: str = "hybrid",
    strategy: str = "auto",
) -> Retriever:
    """Ingest *sources* into a vector store and return an agentic ``search_*`` tool."""
    from loomable.retrieval.agentic import AgenticRetriever
    from loomable.retrieval.corpus import ingest

    tool_name = ensure_search_tool_name(name, default=DEFAULT_SEARCH_KNOWLEDGE)
    corpus_id = (scope or name or "knowledge").strip() or "knowledge"
    if corpus_id.lower().startswith("search_"):
        corpus_id = corpus_id[7:] or "knowledge"
    meta = dict(metadata or {})
    if user_id:
        meta.setdefault("user_id", user_id)
    meta.setdefault("scope", corpus_id)
    corpus = await ingest(
        list(sources),
        name=corpus_id,
        description=description,
        store=_as_store(store),
        embedder=embedder,
        strategy=strategy,
        base_mode=base_mode,
        metadata=meta,
    )
    return AgenticRetriever(
        corpus,
        name=tool_name,
        description=description
        or f"Search the '{corpus_id}' knowledge base and cite filename/page.",
        mode="auto",
        rewrite="off",
        rerank="mmr",
    )


def _store_retriever(
    store: Any,
    *,
    name: str,
    description: str = "",
    embedder: Any | None = None,
) -> Retriever:
    from loomable.retrieval.retrievers import VectorRetriever

    tool_name = ensure_search_tool_name(name, default=DEFAULT_SEARCH_KNOWLEDGE)
    retriever = VectorRetriever(
        tool_name,
        store=_as_store(store),
        embedder=_default_embedder(embedder),
    )
    retriever.description = (  # type: ignore[attr-defined]
        description or "Search the knowledge base (vector store) and cite sources."
    )
    return retriever


def _corpus_retriever(corpus: Any, *, name: str | None = None, description: str = "") -> Retriever:
    from loomable.retrieval.agentic import AgenticRetriever
    from loomable.retrieval.corpus import Corpus

    if not isinstance(corpus, Corpus):
        raise TypeError(f"expected Corpus, got {type(corpus).__name__}")
    tool_name = ensure_search_tool_name(
        name or corpus.name, default=DEFAULT_SEARCH_KNOWLEDGE
    )
    return AgenticRetriever(
        corpus,
        name=tool_name,
        description=description
        or corpus.description
        or f"Search the '{corpus.name}' knowledge base and cite filename/page.",
        mode="auto",
        rewrite="off",
        rerank="mmr",
    )


def _bind_name(retriever: Retriever, name: str | None) -> Retriever:
    if not name:
        return retriever
    retriever.name = ensure_search_tool_name(name, default=retriever.name)
    return retriever


@dataclass
class KnowledgeBase:
    """A searchable knowledge base backed by a vector store.

    Parameters
    ----------
    store:
        :func:`~loomable.providers.vector_store.open_vector_store` result,
        a :class:`~loomable.kernel.contracts.VectorBackend`, or a vector URI
        (``faiss:./kb``, ``zvec:./kb``, ``postgresql://...``).
    sources:
        Optional files, directories, URLs, or inline text to ingest into *store*.
        Omit when the vector DB is already populated.
    retriever:
        Existing :class:`~loomable.kernel.contracts.Retriever` — used as-is
        (skips ingest). ``Agent(retrievers=[...])`` is the other way to attach
        extra search tools alongside a knowledge base.
    """

    store: Any | None = None
    sources: Sequence[Any] | None = None
    name: str = "knowledge"
    description: str = ""
    embedder: Any | None = None
    metadata: dict[str, Any] | None = None
    strategy: str = "auto"
    base_mode: str = "hybrid"
    retriever: Any | None = None
    uri: str | None = None
    backend: Any | None = None

    async def to_retriever(
        self,
        *,
        embedder: Any | None = None,
        user_id: str | None = None,
        name: str | None = None,
    ) -> Retriever:
        """Materialize this KB as a ``search_*`` retriever tool."""
        tool_name = name or self.name
        if self.retriever is not None:
            if _is_retriever(self.retriever):
                return _bind_name(self.retriever, tool_name)
            raise TypeError("KnowledgeBase.retriever must implement retrieve()")
        resolved_embedder = embedder if embedder is not None else self.embedder
        store = self.store
        if store is None and self.uri:
            store = self.uri
        if store is None and self.backend is not None:
            store = self.backend
        if self.sources:
            return await build_knowledge_retriever(
                list(self.sources),
                name=tool_name,
                description=self.description,
                user_id=user_id,
                scope=tool_name,
                embedder=resolved_embedder,
                store=store,
                metadata=self.metadata,
                base_mode=self.base_mode,
                strategy=self.strategy,
            )
        if store is None:
            raise ValueError(
                "KnowledgeBase needs store= (vector DB), sources= to ingest, "
                "or retriever="
            )
        return _store_retriever(
            store,
            name=tool_name,
            description=self.description,
            embedder=resolved_embedder,
        )


async def _from_spec(
    spec: Mapping[str, Any],
    *,
    name: str,
    embedder: Any | None,
    user_id: str | None,
) -> Retriever:
    kb = KnowledgeBase(
        store=spec.get("store") or spec.get("vector_store"),
        sources=spec.get("sources"),
        name=str(spec.get("name") or name),
        description=str(spec.get("description") or ""),
        embedder=spec.get("embedder", embedder),
        metadata=spec.get("metadata"),
        strategy=str(spec.get("strategy") or "auto"),
        base_mode=str(spec.get("base_mode") or "hybrid"),
        retriever=spec.get("retriever"),
        uri=spec.get("uri"),
        backend=spec.get("backend"),
    )
    return await kb.to_retriever(embedder=embedder, user_id=user_id, name=name)


async def resolve_knowledge_base(
    spec: Any,
    *,
    embedder: Any | None = None,
    user_id: str | None = None,
    name: str | None = None,
) -> list[Retriever]:
    """Normalize ``knowledge_base=`` into retriever tools.

    Accepts a vector store, URI, sources, :class:`KnowledgeBase`,
    :class:`~loomable.retrieval.corpus.Corpus`, retriever, named mapping, or
    collection spec dict.
    """
    if spec is None or spec is False:
        return []
    tool_name = name or "knowledge"

    if _is_retriever(spec):
        return [_bind_name(spec, name)]

    from loomable.retrieval.corpus import Corpus

    if isinstance(spec, Corpus):
        return [_corpus_retriever(spec, name=name)]

    if isinstance(spec, KnowledgeBase):
        return [await spec.to_retriever(embedder=embedder, user_id=user_id, name=name)]

    if isinstance(spec, Mapping) and not _is_collection_spec(spec):
        # Named collections: {"personal": [...], "company": store}
        out: list[Retriever] = []
        for key, value in spec.items():
            out.extend(
                await resolve_knowledge_base(
                    value,
                    embedder=embedder,
                    user_id=user_id,
                    name=str(key),
                )
            )
        return out

    if isinstance(spec, Mapping) and _is_collection_spec(spec):
        return [
            await _from_spec(
                spec, name=tool_name, embedder=embedder, user_id=user_id
            )
        ]

    if is_vector_store(spec) or (isinstance(spec, str) and _is_store_uri(spec)):
        return [
            _store_retriever(
                spec, name=tool_name, embedder=embedder
            )
        ]

    if isinstance(spec, (str, Path)):
        return [
            await build_knowledge_retriever(
                [spec],
                name=tool_name,
                user_id=user_id,
                scope=tool_name,
                embedder=embedder,
            )
        ]

    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        items = list(spec)
        if not items:
            return []
        if all(_is_retriever(x) for x in items):
            return [_bind_name(x, None) for x in items]
        return [
            await build_knowledge_retriever(
                items,
                name=tool_name,
                user_id=user_id,
                scope=tool_name,
                embedder=embedder,
            )
        ]

    raise TypeError(
        "knowledge_base must be a vector store, URI, KnowledgeBase, Corpus, "
        f"retriever, sources, or named mapping, got {type(spec).__name__}"
    )
