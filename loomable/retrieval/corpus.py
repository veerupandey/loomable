"""Corpus — ingest + upsert over pluggable chunk strategy / vector store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from loomable.codeindex.embedders import HashingEmbedder
from loomable.kernel.contracts import VectorBackend
from loomable.kernel.long_term import InMemoryVectorBackend, LongTermStore
from loomable.retrieval.chunking.base import ChunkStrategy, resolve_strategy
from loomable.retrieval.ingest import load_sources
from loomable.retrieval.retrievers import HybridRetriever, LexicalRetriever, VectorRetriever
from loomable.retrieval.types import Chunk, Document

__all__ = ["Corpus", "ingest"]


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
    return LongTermStore(backend=InMemoryVectorBackend(), backend_name="memory")


def _content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


@dataclass
class Corpus:
    """Named, updatable document collection with a base :class:`~loomable.kernel.contracts.Retriever`.

    All pieces are pluggable: ``strategy``, ``store`` / ``backend``, ``embedder``,
    and base retrieval ``mode`` (``hybrid`` recommended).
    """

    name: str
    description: str = ""
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    store: LongTermStore | None = None
    embedder: Any = None
    base_mode: str = "hybrid"
    vector_weight: float = 0.7
    strategy: str | ChunkStrategy = "auto"
    _vector: VectorRetriever | None = field(default=None, repr=False)
    _lexical: LexicalRetriever | None = field(default=None, repr=False)
    _base: Any = field(default=None, repr=False)
    _doc_hashes: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def retriever(self) -> Any:
        if self._base is None:
            raise RuntimeError("Corpus has no indexed retriever; call ingest/upsert first")
        return self._base

    def info(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description or self.name}

    async def upsert(
        self,
        sources: Sequence[Any],
        *,
        strategy: str | ChunkStrategy | None = None,
        rebuild: bool = False,
    ) -> int:
        """Load sources, chunk, and index. Skips unchanged docs unless ``rebuild``."""
        strat = resolve_strategy(strategy if strategy is not None else self.strategy)
        docs = load_sources(sources)
        if not docs:
            return 0

        emb = self.embedder or HashingEmbedder()
        self.embedder = emb
        if self.store is None:
            self.store = _make_store(store=None, backend=None, persist_path=None)

        new_chunks: list[Chunk] = []
        removed_ids: list[str] = []
        for doc in docs:
            digest = _content_hash(doc.text)
            if not rebuild and self._doc_hashes.get(doc.id) == digest:
                continue
            # Drop prior chunks for this document id
            stale = [c for c in self.chunks if c.document_id == doc.id]
            removed_ids.extend(c.id for c in stale)
            self.chunks = [c for c in self.chunks if c.document_id != doc.id]
            self.documents = [d for d in self.documents if d.id != doc.id]
            self.documents.append(doc)
            self._doc_hashes[doc.id] = digest
            produced = list(strat.chunk(doc))
            for c in produced:
                c.metadata.setdefault("source", doc.source or doc.id)
                c.metadata.setdefault("path", doc.source or doc.id)
                c.metadata["corpus"] = self.name
            new_chunks.extend(produced)
            self.chunks.extend(produced)

        if self.store is not None:
            for cid in removed_ids:
                try:
                    await self.store.delete(cid)
                except Exception:
                    pass

        to_index = self.chunks if rebuild else new_chunks
        await self._rebind_retrievers(index_chunks=to_index, rebuild=rebuild)
        return len(new_chunks)

    async def _rebind_retrievers(
        self, *, index_chunks: Sequence[Chunk], rebuild: bool = False
    ) -> None:
        assert self.store is not None
        emb = self.embedder or HashingEmbedder()
        mode = (self.base_mode or "hybrid").strip().lower()

        if mode == "lexical":
            self._lexical = LexicalRetriever(f"{self.name}__lexical", self.chunks)
            self._base = self._lexical
            self._base.name = self.name
            return

        if self._vector is None or rebuild:
            self._vector = VectorRetriever(
                f"{self.name}__vector",
                store=self.store,
                embedder=emb,
                chunks=self.chunks,
            )
        else:
            self._vector._chunks.update({c.id: c for c in self.chunks})

        if index_chunks:
            await self._vector.index_chunks(index_chunks)

        if mode == "vector":
            self._vector.name = self.name
            self._base = self._vector
            return

        if mode == "hybrid":
            self._lexical = LexicalRetriever(f"{self.name}__lexical", self.chunks)
            self._base = HybridRetriever(
                self.name,
                vector=self._vector,
                lexical=self._lexical,
                vector_weight=self.vector_weight,
            )
            return

        raise ValueError(f"base_mode must be vector|lexical|hybrid, got {self.base_mode!r}")

    def documents_by_source(self) -> dict[str, Document]:
        out: dict[str, Document] = {}
        for d in self.documents:
            key = d.source or d.id
            out[key] = d
        return out

    def chunks_for_sources(self, sources: Sequence[str]) -> list[Chunk]:
        wanted = {s.lower() for s in sources}
        bases = {Path(s).name.lower() for s in sources}
        out: list[Chunk] = []
        for c in self.chunks:
            src = str(c.metadata.get("source") or c.metadata.get("path") or "").lower()
            if src in wanted or Path(src).name.lower() in bases:
                out.append(c)
                continue
            # document_id match
            if c.document_id.lower() in wanted:
                out.append(c)
        return out


async def ingest(
    sources: Sequence[Any],
    *,
    name: str = "docs",
    description: str = "",
    strategy: str | ChunkStrategy = "auto",
    embedder: Any | None = None,
    store: LongTermStore | None = None,
    backend: VectorBackend | None = None,
    persist_path: str | Path | None = None,
    base_mode: str = "hybrid",
    vector_weight: float = 0.7,
) -> Corpus:
    """Ingest sources into a named :class:`Corpus` (pluggable store/strategy/mode).

    ``strategy="auto"`` (default) is format-aware: PDFs are extracted with page
    markers, then page-chunked (oversized pages split with overlap — never
    truncated). Pass a ``.pdf`` path or directory; chunking is internal.
    """
    corpus = Corpus(
        name=name,
        description=description or name,
        store=_make_store(store=store, backend=backend, persist_path=persist_path),
        embedder=embedder or HashingEmbedder(),
        base_mode=base_mode,
        vector_weight=vector_weight,
        strategy=strategy,
    )
    await corpus.upsert(sources, strategy=strategy, rebuild=True)
    return corpus
