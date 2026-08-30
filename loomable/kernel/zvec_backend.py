"""Alibaba Zvec — file-based embedded vector store (``pip install zvec``).

Zvec is Alibaba's local vector database: vectors + scalar fields on disk with
HNSW indexes. This module wraps it as a Loomable :class:`VectorBackend`.

Docs: https://github.com/alibaba/zvec
Install: ``pip install loomable[zvec]`` or ``pip install zvec``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_META_FIELD = "metadata_json"
_VECTOR_FIELD = "embedding"
_COLLECTION_NAME = "loomable"
_ORIG_ID_KEY = "_loomable_id"
# Alibaba zvec: max 64 chars; limited charset (no `/`, `:`, spaces, …).
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_@#%=+\.\-!]")
_MAX_ID_LEN = 64

# One live collection per path so Agent() × N in one process does not deadlock
# on zvec's exclusive LOCK file.
_LIVE: dict[str, Any] = {}
_REFS: dict[str, int] = {}


def _safe_doc_id(item_id: str) -> str:
    """Map arbitrary Loomable ids onto zvec-legal document ids."""
    raw = str(item_id)
    if len(raw) <= _MAX_ID_LEN and not _SAFE_ID_RE.search(raw):
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"h_{digest}"


class ZvecVectorBackend:
    """Alibaba Zvec collection on disk (default file-based vector store).

    Parameters
    ----------
    path:
        Directory for the Zvec collection. Must not already exist when
        creating a new collection (Zvec creates the path).
    dimensions:
        Embedding width. Required to *create* a new collection; optional when
        opening an existing one. If omitted on a new path, the collection is
        created lazily on the first :meth:`index` call from the vector length.
    """

    def __init__(self, path: str | Path, *, dimensions: int | None = None) -> None:
        self.path = str(Path(path).expanduser().resolve())
        self._dimensions = dimensions
        self._collection: Any | None = None
        if self._collection_exists():
            self._open()
        elif dimensions is not None:
            self._create(dimensions)

    def _require(self) -> Any:
        try:
            import zvec  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Alibaba zvec is required for file-based vector storage. "
                "Install with: pip install loomable[zvec]"
            ) from exc
        return zvec

    def _collection_exists(self) -> bool:
        root = Path(self.path)
        return root.is_dir() and any(root.iterdir())

    def _schema(self, zvec: Any, dimensions: int) -> Any:
        return zvec.CollectionSchema(
            name=_COLLECTION_NAME,
            fields=[zvec.FieldSchema(_META_FIELD, zvec.DataType.STRING)],
            vectors=[
                zvec.VectorSchema(
                    _VECTOR_FIELD,
                    zvec.DataType.VECTOR_FP32,
                    dimension=int(dimensions),
                )
            ],
        )

    def _attach(self, collection: Any) -> None:
        self._collection = collection
        _LIVE[self.path] = collection
        _REFS[self.path] = _REFS.get(self.path, 0) + 1

    def _create(self, dimensions: int) -> None:
        cached = _LIVE.get(self.path)
        if cached is not None:
            self._attach(cached)
            return
        zvec = self._require()
        # create_and_open requires the path to *not* exist yet.
        parent = Path(self.path).parent
        parent.mkdir(parents=True, exist_ok=True)
        if Path(self.path).exists():
            # Empty dir from a prior mkdir — remove so zvec can create it.
            if Path(self.path).is_dir() and not any(Path(self.path).iterdir()):
                Path(self.path).rmdir()
            elif Path(self.path).exists():
                raise FileExistsError(
                    f"Zvec collection path already exists and is not empty: {self.path}"
                )
        self._dimensions = int(dimensions)
        self._attach(
            zvec.create_and_open(
                self.path,
                self._schema(zvec, self._dimensions),
            )
        )

    def _open(self) -> None:
        cached = _LIVE.get(self.path)
        if cached is not None:
            self._attach(cached)
            return
        zvec = self._require()
        self._attach(zvec.open(self.path))

    def _ensure(self, dimensions: int | None = None) -> Any:
        if self._collection is not None:
            return self._collection
        if self._collection_exists():
            self._open()
            return self._collection
        if dimensions is None:
            raise RuntimeError(
                "Zvec collection has no vectors yet and dimensions were not set; "
                "index at least one item or pass dimensions= to ZvecVectorBackend."
            )
        self._create(dimensions)
        return self._collection

    def close(self) -> None:
        """Release this handle. The on-disk lock is freed when the last handle closes."""
        if self._collection is None:
            return
        n = _REFS.get(self.path, 1) - 1
        if n <= 0:
            try:
                self._collection.flush()
            except Exception:
                pass
            _LIVE.pop(self.path, None)
            _REFS.pop(self.path, None)
        else:
            _REFS[self.path] = n
        self._collection = None

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        dim = len(vector)
        if self._dimensions is not None and dim != self._dimensions:
            raise ValueError(
                f"vector length {dim} does not match zvec dimensions={self._dimensions}"
            )
        col = self._ensure(dim)
        zvec = self._require()
        meta = dict(metadata or {})
        meta[_ORIG_ID_KEY] = str(id)
        doc = zvec.Doc(
            id=_safe_doc_id(id),
            vectors={_VECTOR_FIELD: [float(x) for x in vector]},
            fields={_META_FIELD: json.dumps(meta, default=str)},
        )
        col.upsert([doc])
        col.flush()

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        try:
            col = self._ensure(self._dimensions or len(vector))
        except RuntimeError:
            return []
        if col is None:
            return []
        stats = col.stats
        if getattr(stats, "doc_count", 0) == 0:
            return []
        zvec = self._require()
        hits = col.query(
            queries=zvec.Query(field_name=_VECTOR_FIELD, vector=[float(x) for x in vector]),
            topk=int(k),
            output_fields=[_META_FIELD],
        )
        out: list[dict[str, Any]] = []
        for hit in hits:
            raw = ""
            try:
                raw = str(hit.field(_META_FIELD) or "")
            except Exception:
                raw = ""
            try:
                meta = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                meta = {"raw": raw}
            if not isinstance(meta, dict):
                meta = {"value": meta}
            # Default HNSW cosine returns similarity (1 = identical, 0 = orthogonal).
            similarity = float(hit.score if hit.score is not None else 0.0)
            orig = str(meta.pop(_ORIG_ID_KEY, hit.id))
            meta.pop("score", None)
            row = {**meta, "id": orig, "score": similarity}
            out.append(row)
        return out

    def _row_from_hit(self, hit: Any) -> dict[str, Any]:
        raw = ""
        try:
            raw = str(hit.field(_META_FIELD) or "")
        except Exception:
            raw = ""
        try:
            meta = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            meta = {"raw": raw}
        if not isinstance(meta, dict):
            meta = {"value": meta}
        meta = dict(meta)
        orig = str(meta.pop(_ORIG_ID_KEY, getattr(hit, "id", "")))
        meta.pop("score", None)
        return {**meta, "id": orig}

    async def get(self, id: str) -> dict[str, Any] | None:
        if self._collection is None and not self._collection_exists():
            return None
        try:
            col = self._ensure(self._dimensions)
        except RuntimeError:
            return None
        if col is None:
            return None
        safe = _safe_doc_id(id)
        getter = getattr(col, "get", None)
        if callable(getter):
            try:
                docs = getter([safe])
            except Exception:
                docs = None
            if docs:
                hit = docs[0] if isinstance(docs, (list, tuple)) else docs
                row = self._row_from_hit(hit)
                if row.get("id") == str(id) or getattr(hit, "id", None) == safe:
                    row["id"] = str(id)
                    return row
        for row in await self.scan(limit=10_000):
            if row.get("id") == str(id):
                return row
        return None

    async def scan(self, *, limit: int = 10_000) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if self._collection is None and not self._collection_exists():
            return []
        try:
            col = self._ensure(self._dimensions)
        except RuntimeError:
            return []
        if col is None:
            return []
        stats = col.stats
        doc_count = int(getattr(stats, "doc_count", 0) or 0)
        if doc_count == 0:
            return []
        topk = min(int(limit), doc_count)
        # Prefer native enumerate when available.
        lister = getattr(col, "list", None) or getattr(col, "get_all", None)
        if callable(lister):
            try:
                docs = lister()
                out: list[dict[str, Any]] = []
                for hit in docs or []:
                    out.append(self._row_from_hit(hit))
                    if len(out) >= topk:
                        break
                return out
            except Exception:
                pass
        # Fallback: similarity scan with a zero vector of known width.
        dim = self._dimensions
        if dim is None:
            return []
        zvec = self._require()
        hits = col.query(
            queries=zvec.Query(
                field_name=_VECTOR_FIELD,
                vector=[0.0] * int(dim),
            ),
            topk=topk,
            output_fields=[_META_FIELD],
        )
        return [self._row_from_hit(hit) for hit in hits]

    async def delete(self, id: str) -> None:
        if self._collection is None and not self._collection_exists():
            return
        col = self._ensure(self._dimensions)
        col.delete([_safe_doc_id(id)])
        col.flush()
