"""ChromaDB VectorBackend — persistent local path or HttpClient.

Install::

    pip install loomable[chroma]   # chromadb

File-based::

    open_vector_store(engine="chroma", path="./.loomable/chroma", dimensions=384)

Server::

    open_vector_store(engine="chroma", uri="http://localhost:8000", dimensions=384)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loomable.kernel.errors import MemoryBackendError

__all__ = ["ChromaVectorBackend"]


def _require_chroma() -> Any:
    try:
        import chromadb  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chromadb is required for ChromaVectorBackend. "
            "Install with: pip install loomable[chroma]"
        ) from exc
    return chromadb


def _flat_meta(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    out: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        k = str(key)
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[k] = value
        elif isinstance(value, str):
            out[k] = value[:8_000]
        elif value is None:
            continue
        else:
            out[k] = json.dumps(value, default=str)[:8_000]
    return out


def _inflate_meta(metadata: dict[str, Any] | None) -> dict[str, Any]:
    meta = dict(metadata or {})
    # Best-effort: leave JSON strings as strings; callers use content/text keys.
    return meta


class ChromaVectorBackend:
    """Chroma persistent / HTTP vector store implementing :class:`VectorBackend`."""

    def __init__(
        self,
        *,
        dimensions: int,
        path: str | Path | None = None,
        uri: str | None = None,
        collection: str = "loomable",
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        if int(dimensions) <= 0:
            raise ValueError("dimensions must be a positive int")
        self.dimensions = int(dimensions)
        self.collection_name = collection or "loomable"
        self.path = str(Path(path).expanduser().resolve()) if path is not None else None
        self.uri = (uri or "").strip() or None
        self.host = host
        self.port = int(port) if port is not None else None
        self._backend_id = (
            f"chroma:{self.path or self.uri or f'{host}:{port}'}:{self.collection_name}"
        )
        chroma = _require_chroma()
        if self.path:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._client = chroma.PersistentClient(path=self.path)
        elif self.uri:
            # chromadb HttpClient via Settings / URL
            from urllib.parse import urlparse

            parsed = urlparse(self.uri)
            h = parsed.hostname or "localhost"
            p = parsed.port or 8000
            self._client = chroma.HttpClient(host=h, port=p)
        elif self.host:
            self._client = chroma.HttpClient(
                host=self.host, port=self.port or 8000
            )
        else:
            # Ephemeral in-process
            self._client = chroma.Client()
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def _validate_dims(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"ChromaVectorBackend expected {self.dimensions} dims, got {len(vector)}"
            )

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._validate_dims(vector)
        try:
            item_id = str(id)
            meta = _flat_meta(metadata)
            doc = str(meta.get("content") or meta.get("text") or "")
            self._col.upsert(
                ids=[item_id],
                embeddings=[[float(x) for x in vector]],
                metadatas=[meta] if meta else None,
                documents=[doc] if doc else None,
            )
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        self._validate_dims(vector)
        if k <= 0:
            return []
        try:
            raw = self._col.query(
                query_embeddings=[[float(x) for x in vector]],
                n_results=max(1, int(k)),
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc
        ids = (raw.get("ids") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for item_id, meta, doc, dist in zip(ids, metas, docs, dists):
            row = _inflate_meta(meta)
            row.pop("score", None)
            if doc and not row.get("content") and not row.get("text"):
                row["content"] = doc
            # cosine space distance → similarity
            try:
                score = 1.0 - float(dist)
            except (TypeError, ValueError):
                score = 0.0
            out.append({**row, "id": str(item_id), "score": score})
        return out

    async def delete(self, id: str) -> None:
        try:
            self._col.delete(ids=[str(id)])
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def get(self, id: str) -> dict[str, Any] | None:
        try:
            raw = self._col.get(
                ids=[str(id)],
                include=["metadatas", "documents"],
            )
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc
        ids = raw.get("ids") or []
        if not ids:
            return None
        metas = raw.get("metadatas") or [None]
        docs = raw.get("documents") or [None]
        meta = _inflate_meta(metas[0])
        meta.pop("score", None)
        doc = docs[0] if docs else None
        if doc and not meta.get("content") and not meta.get("text"):
            meta["content"] = doc
        return {**meta, "id": str(ids[0])}

    async def scan(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        try:
            raw = self._col.get(
                include=["metadatas", "documents"],
                limit=max(1, int(limit)),
            )
        except TypeError:
            # Older chroma clients may not accept limit= on get.
            try:
                raw = self._col.get(include=["metadatas", "documents"])
            except Exception as exc:
                raise MemoryBackendError(self._backend_id) from exc
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc
        ids = raw.get("ids") or []
        metas = raw.get("metadatas") or []
        docs = raw.get("documents") or []
        out: list[dict[str, Any]] = []
        for item_id, meta, doc in zip(ids, metas, docs):
            if len(out) >= max(0, int(limit)):
                break
            row = _inflate_meta(meta)
            row.pop("score", None)
            if doc and not row.get("content") and not row.get("text"):
                row["content"] = doc
            out.append({**row, "id": str(item_id)})
        return out

    def close(self) -> None:
        return None
