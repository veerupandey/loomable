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

    def _create(self, dimensions: int) -> None:
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
        self._collection = zvec.create_and_open(
            self.path,
            self._schema(zvec, self._dimensions),
        )

    def _open(self) -> None:
        zvec = self._require()
        self._collection = zvec.open(self.path)

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
        """Release the collection handle so another process can open the path."""
        if self._collection is not None:
            try:
                self._collection.flush()
            except Exception:
                pass
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

    async def delete(self, id: str) -> None:
        if self._collection is None and not self._collection_exists():
            return
        col = self._ensure(self._dimensions)
        col.delete([_safe_doc_id(id)])
        col.flush()
