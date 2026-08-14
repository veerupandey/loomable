"""Agentic retriever — pluggable rewrite / route / base / rerank / compress."""

from __future__ import annotations

from typing import Any, Sequence

from loomable.kernel.contracts import Retriever
from loomable.retrieval.corpus import Corpus
from loomable.retrieval.rerank import resolve_compressor, resolve_reranker
from loomable.retrieval.rewrite import resolve_rewriter
from loomable.retrieval.route import (
    match_file_sources,
    resolve_corpus_router,
    resolve_mode_router,
)

__all__ = ["AgenticRetriever", "CompositeRetriever", "build_agentic_retriever"]


def _merge_hits(groups: Sequence[Sequence[dict[str, Any]]], *, k: int) -> list[dict[str, Any]]:
    """RRF-merge hit lists from multiple query rewrites / corpora."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hits in groups:
        for rank, hit in enumerate(hits):
            key = str(hit.get("id") or hit.get("content"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            payloads.setdefault(key, hit)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[dict[str, Any]] = []
    for key, score in ordered[: max(1, int(k))]:
        row = dict(payloads[key])
        row["score"] = score
        out.append(row)
    return out


class AgenticRetriever(Retriever):
    """Single-corpus agentic retriever with pluggable stages.

    Pipeline::

        query → rewrite → mode route (chunks|file) → base retrieve → rerank → compress

    Attach to an agent as a tool via ``Agent(retrievers=[...])``.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        name: str | None = None,
        mode: str | Any = "auto",
        rewrite: str | Any | None = "off",
        rerank: str | bool | Any | None = True,
        compress: str | bool | Any | None = "off",
        llm: Any | None = None,
        fetch_k: int | None = None,
    ) -> None:
        self.corpus = corpus
        self.name = name or corpus.name
        self.rewriter = resolve_rewriter(rewrite, llm=llm)
        self.mode_router = resolve_mode_router(mode, llm=llm)
        self.reranker = resolve_reranker(rerank, llm=llm)
        self.compressor = resolve_compressor(compress, llm=llm)
        self.fetch_k = fetch_k
        self.llm = llm

    async def _retrieve_chunks(self, query: str, k: int) -> list[dict[str, Any]]:
        return await self.corpus.retriever.retrieve(query, k)

    async def _retrieve_file(self, query: str, k: int) -> list[dict[str, Any]]:
        sources = list(self.corpus.documents_by_source().keys())
        matched = match_file_sources(query, sources)
        if not matched:
            # Fall back to chunks if no filename cue matched.
            return await self._retrieve_chunks(query, k)
        chunks = self.corpus.chunks_for_sources(matched)
        if not chunks:
            # Return whole documents as single hits
            docs = self.corpus.documents_by_source()
            out: list[dict[str, Any]] = []
            for src in matched:
                doc = docs.get(src)
                if doc is None:
                    continue
                out.append(
                    {
                        "id": doc.id,
                        "content": doc.text[:12_000],
                        "score": 1.0,
                        "source": doc.source or doc.id,
                        "path": doc.source or doc.id,
                        "retrieval_mode": "file",
                        "kind": "document",
                        "name": doc.id,
                    }
                )
            return out[: max(1, int(k))]
        # Prefer lexical over matched file chunks for ranking within file
        from loomable.retrieval.retrievers import LexicalRetriever

        lex = LexicalRetriever(f"{self.name}__file", chunks)
        hits = await lex.retrieve(query, k)
        for h in hits:
            h["retrieval_mode"] = "file"
        if hits:
            return hits
        return [c.as_result(score=1.0) | {"retrieval_mode": "file"} for c in chunks[:k]]

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        k = max(1, int(k))
        fetch = int(self.fetch_k) if self.fetch_k else max(k * 4, k)
        queries = await self.rewriter.rewrite(query)
        if not queries:
            queries = [query]
        mode = await self.mode_router.choose_mode(query)

        groups: list[list[dict[str, Any]]] = []
        for q in queries:
            if mode == "file":
                hits = await self._retrieve_file(q, fetch)
            else:
                hits = await self._retrieve_chunks(q, fetch)
            for h in hits:
                h.setdefault("retrieval_mode", mode)
                h.setdefault("corpus", self.corpus.name)
            groups.append(hits)

        merged = _merge_hits(groups, k=fetch) if len(groups) > 1 else (groups[0][:fetch] if groups else [])
        ranked = await self.reranker.rerank(query, merged, top_n=k)
        compressed = await self.compressor.compress(query, ranked)
        # Ensure citation-friendly fields
        out: list[dict[str, Any]] = []
        for hit in compressed[:k]:
            row = dict(hit)
            row.setdefault("content", row.get("text") or "")
            row.setdefault("corpus", self.corpus.name)
            row.setdefault("retrieval_mode", mode)
            out.append(row)
        return out


class CompositeRetriever(Retriever):
    """Multi-corpus agentic router (pluggable :class:`CorpusRouter`)."""

    def __init__(
        self,
        corpora: Sequence[Corpus | AgenticRetriever],
        *,
        name: str = "knowledge",
        corpus_router: str | Any = "all",
        llm: Any | None = None,
        # Defaults applied when wrapping bare Corpus objects
        mode: str | Any = "auto",
        rewrite: str | Any | None = "off",
        rerank: str | bool | Any | None = True,
        compress: str | bool | Any | None = "off",
    ) -> None:
        self.name = name
        self.llm = llm
        self.router = resolve_corpus_router(corpus_router, llm=llm)
        self._children: dict[str, AgenticRetriever] = {}
        for item in corpora:
            if isinstance(item, AgenticRetriever):
                self._children[item.corpus.name] = item
            elif isinstance(item, Corpus):
                self._children[item.name] = AgenticRetriever(
                    item,
                    mode=mode,
                    rewrite=rewrite,
                    rerank=rerank,
                    compress=compress,
                    llm=llm,
                )
            else:
                raise TypeError("corpora must be Corpus or AgenticRetriever instances")

    def infos(self) -> list[dict[str, str]]:
        return [c.corpus.info() for c in self._children.values()]

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        k = max(1, int(k))
        chosen = await self.router.choose_corpora(query, self.infos())
        if not chosen:
            chosen = list(self._children.keys())[:1]
        per = max(k, (k + len(chosen) - 1) // max(1, len(chosen)))
        groups: list[list[dict[str, Any]]] = []
        for name in chosen:
            child = self._children.get(name)
            if child is None:
                continue
            hits = await child.retrieve(query, per)
            groups.append(hits)
        return _merge_hits(groups, k=k)


async def build_agentic_retriever(
    sources: Sequence[Any] | Corpus | Sequence[Corpus],
    *,
    name: str = "docs",
    description: str = "",
    # ingestion
    strategy: Any = "auto",
    embedder: Any | None = None,
    store: Any | None = None,
    backend: Any | None = None,
    persist_path: Any | None = None,
    base_mode: str = "hybrid",
    vector_weight: float = 0.6,
    # agentic stages (all pluggable)
    mode: str | Any = "auto",
    rewrite: str | Any | None = "off",
    rerank: str | bool | Any | None = True,
    compress: str | bool | Any | None = "off",
    corpus_router: str | Any = "all",
    llm: Any | None = None,
) -> Retriever:
    """Build an :class:`AgenticRetriever` or :class:`CompositeRetriever`.

    ``sources`` may be raw ingest inputs, a single :class:`Corpus`, or a list
    of corpora for multi-corpus routing.
    """
    from loomable.retrieval.corpus import ingest

    # Multi-corpus: list of Corpus
    if (
        isinstance(sources, (list, tuple))
        and sources
        and all(isinstance(s, Corpus) for s in sources)
    ):
        return CompositeRetriever(
            list(sources),  # type: ignore[arg-type]
            name=name,
            corpus_router=corpus_router,
            llm=llm,
            mode=mode,
            rewrite=rewrite,
            rerank=rerank,
            compress=compress,
        )

    if isinstance(sources, Corpus):
        corpus = sources
    else:
        corpus = await ingest(
            sources,  # type: ignore[arg-type]
            name=name,
            description=description,
            strategy=strategy,
            embedder=embedder,
            store=store,
            backend=backend,
            persist_path=persist_path,
            base_mode=base_mode,
            vector_weight=vector_weight,
        )

    return AgenticRetriever(
        corpus,
        name=name,
        mode=mode,
        rewrite=rewrite,
        rerank=rerank,
        compress=compress,
        llm=llm,
    )
