"""Concrete Retriever implementations (vector / lexical / hybrid)."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from loomable.kernel.contracts import Retriever
from loomable.kernel.long_term import LongTermStore
from loomable.retrieval.types import Chunk

_TOKEN = re.compile(r"[a-z0-9_]+", re.I)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class VectorRetriever(Retriever):
    """Similarity search over chunks indexed in a :class:`LongTermStore`."""

    def __init__(
        self,
        name: str,
        *,
        store: LongTermStore,
        embedder: Any,
        chunks: Sequence[Chunk] | None = None,
    ) -> None:
        self.name = name
        self.store = store
        self.embedder = embedder
        self._chunks = {c.id: c for c in (chunks or [])}

    async def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        from loomable.providers.embedders import embed_many

        n = 0
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        texts = [f"{c.name} {c.kind}\n{c.text}" for c in chunk_list]
        vectors = await embed_many(self.embedder, texts)
        for chunk, vector in zip(chunk_list, vectors):
            self._chunks[chunk.id] = chunk
            meta = chunk.as_result(score=0.0)
            meta.pop("score", None)  # never persist score into vector metadata
            meta["text"] = chunk.text[:8_000]
            meta["source_type"] = "retrieval"
            await self.store.index(id=chunk.id, vector=vector, metadata=meta)
            n += 1
        return n

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        vector = await self.embedder.embed(query)
        rows = await self.store.query(vector, k=max(1, int(k)))
        out: list[dict[str, Any]] = []
        for row in rows:
            content = row.get("content") or row.get("text") or ""
            out.append(
                {
                    "id": row.get("id"),
                    "content": content,
                    "score": float(row.get("score") or 0.0),
                    "source": row.get("source") or row.get("path") or "",
                    "path": row.get("path") or row.get("source") or "",
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "kind": row.get("kind"),
                    "name": row.get("name"),
                }
            )
        return out


class LexicalRetriever(Retriever):
    """In-memory BM25-lite retriever over chunk text."""

    def __init__(self, name: str, chunks: Sequence[Chunk] | None = None) -> None:
        self.name = name
        self._chunks: list[Chunk] = list(chunks or [])
        self._docs_tokens: list[list[str]] = [_tokenize(c.text) for c in self._chunks]
        self._avg_len = (
            sum(len(t) for t in self._docs_tokens) / max(1, len(self._docs_tokens))
        )
        self._df: dict[str, int] = {}
        for tokens in self._docs_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self._chunks.append(c)
            tokens = _tokenize(c.text)
            self._docs_tokens.append(tokens)
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._avg_len = (
            sum(len(t) for t in self._docs_tokens) / max(1, len(self._docs_tokens))
        )

    def _score(self, query: str, doc_tokens: list[str]) -> float:
        q = _tokenize(query)
        if not q or not doc_tokens:
            return 0.0
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        n_docs = max(1, len(self._chunks))
        avg = self._avg_len or 1.0
        k1, b = 1.5, 0.75
        score = 0.0
        for term in q:
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = self._df.get(term, 0) or 1
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * len(doc_tokens) / avg)
            score += idf * (freq * (k1 + 1)) / denom
        return score

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        scored = [
            (self._score(query, tokens), chunk)
            for tokens, chunk in zip(self._docs_tokens, self._chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, chunk in scored[: max(1, int(k))]:
            if score <= 0:
                continue
            out.append(chunk.as_result(score=score))
        return out


class HybridRetriever(Retriever):
    """Blend vector + lexical scores (RRF-style)."""

    def __init__(
        self,
        name: str,
        *,
        vector: VectorRetriever,
        lexical: LexicalRetriever,
        vector_weight: float = 0.6,
    ) -> None:
        self.name = name
        self.vector = vector
        self.lexical = lexical
        self.vector_weight = min(1.0, max(0.0, float(vector_weight)))

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        k_fetch = max(int(k) * 3, int(k))
        v_hits = await self.vector.retrieve(query, k_fetch)
        l_hits = await self.lexical.retrieve(query, k_fetch)
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        vw = self.vector_weight
        lw = 1.0 - vw
        for rank, hit in enumerate(v_hits):
            key = str(hit.get("id") or hit.get("content"))
            scores[key] = scores.get(key, 0.0) + vw * (1.0 / (60 + rank))
            payloads[key] = hit
        for rank, hit in enumerate(l_hits):
            key = str(hit.get("id") or hit.get("content"))
            scores[key] = scores.get(key, 0.0) + lw * (1.0 / (60 + rank))
            payloads.setdefault(key, hit)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: list[dict[str, Any]] = []
        for key, score in ordered[: max(1, int(k))]:
            row = dict(payloads[key])
            row["score"] = score
            out.append(row)
        return out
