"""Milvus VectorBackend — Milvus Lite (``.db`` file) or server URI.

Install::

    pip install loomable[milvus]   # pymilvus[milvus_lite]

File-based (Milvus Lite)::

    open_vector_store(engine="milvus", path="./.loomable/milvus.db", dimensions=384)

Server::

    open_vector_store(engine="milvus", uri="http://localhost:19530", dimensions=384)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loomable.kernel.errors import MemoryBackendError

__all__ = ["MilvusVectorBackend"]

_ID_FIELD = "pk"
_VECTOR_FIELD = "embedding"


def _require_milvus_client() -> Any:
    try:
        from pymilvus import MilvusClient  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pymilvus is required for MilvusVectorBackend. "
            "Install with: pip install loomable[milvus]  "
            "(includes milvus-lite for file:// .db URIs)."
        ) from exc
    return MilvusClient


class MilvusVectorBackend:
    """Milvus / Milvus Lite store implementing :class:`VectorBackend`.

    Metadata is kept in a JSON sidecar next to file-based ``.db`` URIs (Milvus
    Lite's quick schema is vector-only). Server mode keeps an in-process cache.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        path: str | Path | None = None,
        uri: str | None = None,
        collection: str = "loomable",
        token: str | None = None,
    ) -> None:
        if int(dimensions) <= 0:
            raise ValueError("dimensions must be a positive int")
        self.dimensions = int(dimensions)
        self.collection_name = collection or "loomable"
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._meta_path: Path | None = None

        if path is not None:
            p = Path(path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.uri = str(p)
            self._meta_path = Path(str(p) + ".meta.json")
        elif uri:
            self.uri = uri.strip()
            if self.uri.endswith(".db") or self.uri.startswith("file:"):
                file_uri = self.uri.removeprefix("file:")
                self._meta_path = Path(file_uri + ".meta.json")
        else:
            self.uri = str(Path("/tmp/loomable_milvus_ephemeral.db"))
            self._meta_path = Path(self.uri + ".meta.json")

        self._backend_id = f"milvus:{self.uri}:{self.collection_name}"
        MilvusClient = _require_milvus_client()
        kwargs: dict[str, Any] = {"uri": self.uri}
        if token:
            kwargs["token"] = token
        try:
            self._client = MilvusClient(**kwargs)
        except Exception as exc:
            msg = str(exc)
            if "milvus-lite" in msg or "milvus_lite" in msg:
                raise ImportError(
                    "Milvus Lite is required for file-based Milvus URIs. "
                    "Install with: pip install 'pymilvus[milvus_lite]' or loomable[milvus]"
                ) from exc
            raise
        self._load_meta()
        self._ensure_collection()

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def _load_meta(self) -> None:
        if self._meta_path and self._meta_path.is_file():
            try:
                raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._meta_cache = {str(k): dict(v) for k, v in (raw or {}).items()}
            except (OSError, json.JSONDecodeError, TypeError):
                self._meta_cache = {}

    def _save_meta(self) -> None:
        if self._meta_path is None:
            return
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(self._meta_cache, default=str), encoding="utf-8"
        )

    def _ensure_collection(self) -> None:
        name = self.collection_name
        if self._client.has_collection(collection_name=name):
            return
        self._client.create_collection(
            collection_name=name,
            dimension=self.dimensions,
            primary_field_name=_ID_FIELD,
            id_type="string",
            max_length=512,
            vector_field_name=_VECTOR_FIELD,
            metric_type="COSINE",
            auto_id=False,
        )

    def _validate_dims(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"MilvusVectorBackend expected {self.dimensions} dims, got {len(vector)}"
            )

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._validate_dims(vector)
        item_id = str(id)[:512]
        try:
            try:
                self._client.delete(
                    collection_name=self.collection_name, ids=[item_id]
                )
            except Exception:
                pass
            self._client.insert(
                collection_name=self.collection_name,
                data=[
                    {
                        _ID_FIELD: item_id,
                        _VECTOR_FIELD: [float(x) for x in vector],
                    }
                ],
            )
            self._meta_cache[item_id] = dict(metadata or {})
            self._save_meta()
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        self._validate_dims(vector)
        if k <= 0:
            return []
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                data=[[float(x) for x in vector]],
                limit=max(1, int(k)),
                output_fields=[_ID_FIELD],
            )
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

        group = hits[0] if hits else []
        out: list[dict[str, Any]] = []
        for hit in group:
            item_id = str(
                hit.get("id")
                or hit.get(_ID_FIELD)
                or (hit.get("entity") or {}).get(_ID_FIELD)
                or ""
            )
            distance = hit.get("distance")
            meta = dict(self._meta_cache.get(item_id) or {})
            meta.pop("score", None)
            try:
                score = float(distance) if distance is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            out.append({**meta, "id": item_id, "score": score})
        return out

    async def delete(self, id: str) -> None:
        item_id = str(id)[:512]
        try:
            self._client.delete(
                collection_name=self.collection_name, ids=[item_id]
            )
            self._meta_cache.pop(item_id, None)
            self._save_meta()
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def get(self, id: str) -> dict[str, Any] | None:
        item_id = str(id)[:512]
        if item_id not in self._meta_cache:
            return None
        meta = dict(self._meta_cache.get(item_id) or {})
        meta.pop("score", None)
        return {**meta, "id": item_id}

    async def scan(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item_id, metadata in self._meta_cache.items():
            if len(out) >= max(0, int(limit)):
                break
            meta = dict(metadata or {})
            meta.pop("score", None)
            out.append({**meta, "id": str(item_id)})
        return out

    def close(self) -> None:
        try:
            self._save_meta()
            self._client.close()
        except Exception:
            pass
