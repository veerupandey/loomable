"""FAISS VectorBackend — CPU by default, GPU when faiss-gpu is available.

Install one of::

    pip install loomable[faiss]          # pulls faiss-cpu
    pip install faiss-gpu                # GPU wheels (same ``import faiss``)

Do not install ``faiss-cpu`` and ``faiss-gpu`` together.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from loomable.kernel.errors import MemoryBackendError

__all__ = ["FaissVectorBackend"]

DeviceKind = Literal["cpu", "gpu", "auto"]

_INDEX_NAME = "index.faiss"
_META_NAME = "meta.json"


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FAISS is required for FaissVectorBackend. "
            "Install with: pip install loomable[faiss]  (CPU) "
            "or pip install faiss-gpu  (GPU; same import name)."
        ) from exc
    return faiss


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "numpy is required for FaissVectorBackend. "
            "It is pulled in by faiss-cpu / faiss-gpu."
        ) from exc
    return np


class FaissVectorBackend:
    """FAISS ANN store with optional disk persistence and GPU offload.

    Parameters
    ----------
    dimensions:
        Embedding width. Required up front (FAISS indexes are fixed-width).
    path:
        Optional directory for ``index.faiss`` + ``meta.json``. When omitted,
        the index lives only in process memory.
    device:
        ``"cpu"`` (default), ``"gpu"`` (requires faiss-gpu + a CUDA device),
        or ``"auto"`` (GPU if ``faiss.get_num_gpus() > 0``, else CPU).
    gpu_id:
        CUDA device index when using GPU.
    """

    def __init__(
        self,
        *,
        dimensions: int,
        path: str | Path | None = None,
        device: DeviceKind = "cpu",
        gpu_id: int = 0,
    ) -> None:
        if int(dimensions) <= 0:
            raise ValueError("dimensions must be a positive int")
        self.dimensions = int(dimensions)
        self.path = str(Path(path).expanduser().resolve()) if path is not None else None
        self.device = device
        self.gpu_id = int(gpu_id)
        self._backend_id = f"faiss:{self.device}:{self.path or 'memory'}"

        self._faiss = _require_faiss()
        self._np = _require_numpy()
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_int = 1
        self._gpu_resources: Any | None = None
        self._on_gpu = False
        self._index: Any = self._new_cpu_index()

        if self.path and (Path(self.path) / _INDEX_NAME).is_file():
            self._load()
        self._maybe_to_gpu()

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def using_gpu(self) -> bool:
        return self._on_gpu

    def _new_cpu_index(self) -> Any:
        # Inner product on L2-normalized vectors == cosine similarity.
        flat = self._faiss.IndexFlatIP(self.dimensions)
        return self._faiss.IndexIDMap2(flat)

    def _gpu_available(self) -> bool:
        if not hasattr(self._faiss, "StandardGpuResources"):
            return False
        if not hasattr(self._faiss, "index_cpu_to_gpu"):
            return False
        try:
            return int(self._faiss.get_num_gpus()) > 0
        except Exception:
            return False

    def _resolve_want_gpu(self) -> bool:
        if self.device == "cpu":
            return False
        if self.device == "gpu":
            if not self._gpu_available():
                raise RuntimeError(
                    "device='gpu' requested but FAISS GPU is unavailable. "
                    "Install faiss-gpu and ensure a CUDA device is visible."
                )
            return True
        # auto
        return self._gpu_available()

    def _maybe_to_gpu(self) -> None:
        if not self._resolve_want_gpu():
            self._on_gpu = False
            return
        self._gpu_resources = self._faiss.StandardGpuResources()
        cpu = self._as_cpu_index()
        self._index = self._faiss.index_cpu_to_gpu(
            self._gpu_resources, self.gpu_id, cpu
        )
        self._on_gpu = True

    def _as_cpu_index(self) -> Any:
        if self._on_gpu and hasattr(self._faiss, "index_gpu_to_cpu"):
            return self._faiss.index_gpu_to_cpu(self._index)
        return self._index

    def _validate_dims(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"FaissVectorBackend expected {self.dimensions} dims, got {len(vector)}"
            )

    def _normalize(self, rows: Any) -> Any:
        # faiss.normalize_L2 mutates in place
        self._faiss.normalize_L2(rows)
        return rows

    def _persist(self) -> None:
        if not self.path:
            return
        root = Path(self.path)
        root.mkdir(parents=True, exist_ok=True)
        cpu = self._as_cpu_index()
        self._faiss.write_index(cpu, str(root / _INDEX_NAME))
        payload = {
            "dimensions": self.dimensions,
            "next_int": self._next_int,
            "id_to_int": self._id_to_int,
            "metadata": self._metadata,
        }
        (root / _META_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _load(self) -> None:
        assert self.path is not None
        root = Path(self.path)
        meta_path = root / _META_NAME
        index_path = root / _INDEX_NAME
        if not index_path.is_file():
            return
        raw = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        dims = int(raw.get("dimensions") or self.dimensions)
        if dims != self.dimensions:
            raise ValueError(
                f"FAISS index at {self.path} has dimensions={dims}, "
                f"but backend was constructed with dimensions={self.dimensions}"
            )
        self._index = self._faiss.read_index(str(index_path))
        self._id_to_int = {str(k): int(v) for k, v in (raw.get("id_to_int") or {}).items()}
        self._int_to_id = {v: k for k, v in self._id_to_int.items()}
        self._metadata = {
            str(k): dict(v) for k, v in (raw.get("metadata") or {}).items()
        }
        self._next_int = int(raw.get("next_int") or (max(self._int_to_id) + 1 if self._int_to_id else 1))
        self._on_gpu = False

    def close(self) -> None:
        """Flush to disk (if path set) and drop GPU resources."""
        try:
            self._persist()
        except Exception:
            pass
        self._gpu_resources = None
        self._on_gpu = False

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._validate_dims(vector)
        try:
            item_id = str(id)
            np = self._np
            row = np.asarray([vector], dtype="float32")
            self._normalize(row)
            if item_id in self._id_to_int:
                old = np.asarray([self._id_to_int[item_id]], dtype="int64")
                self._index.remove_ids(old)
            else:
                self._id_to_int[item_id] = self._next_int
                self._int_to_id[self._next_int] = item_id
                self._next_int += 1
            ids = np.asarray([self._id_to_int[item_id]], dtype="int64")
            self._index.add_with_ids(row, ids)
            self._metadata[item_id] = dict(metadata or {})
            self._persist()
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        self._validate_dims(vector)
        if k <= 0 or self._index.ntotal == 0:
            return []
        try:
            np = self._np
            row = np.asarray([vector], dtype="float32")
            self._normalize(row)
            top_k = min(int(k), int(self._index.ntotal))
            scores, ids = self._index.search(row, top_k)
            out: list[dict[str, Any]] = []
            for score, int_id in zip(scores[0].tolist(), ids[0].tolist()):
                if int_id < 0:
                    continue
                # Guard against FAISS empty-slot sentinel scores.
                if not math.isfinite(float(score)):
                    continue
                item_id = self._int_to_id.get(int(int_id))
                if item_id is None:
                    continue
                meta = dict(self._metadata.get(item_id) or {})
                out.append({"id": item_id, "score": float(score), **meta})
            return out
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc

    async def delete(self, id: str) -> None:
        item_id = str(id)
        int_id = self._id_to_int.pop(item_id, None)
        self._metadata.pop(item_id, None)
        if int_id is None:
            return
        self._int_to_id.pop(int_id, None)
        try:
            np = self._np
            self._index.remove_ids(np.asarray([int_id], dtype="int64"))
            self._persist()
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self._backend_id) from exc
