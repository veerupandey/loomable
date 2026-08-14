"""loomable.kernel.long_term - Long-Term Store with pluggable vector backend.

Default **file** backend is Alibaba **zvec** (``pip install loomable[zvec]``).
Pass ``backend=`` / ``engine=`` for Postgres, FAISS (CPU/GPU), or any object
satisfying :class:`~loomable.kernel.contracts.VectorBackend`.

Zero-dependency tests / ephemeral use: :class:`InMemoryVectorBackend`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from loomable.kernel.contracts import VectorBackend
from loomable.kernel.errors import MemoryBackendError
from loomable.kernel.zvec_backend import ZvecVectorBackend

__all__ = [
    "InMemoryVectorBackend",
    "LongTermStore",
    "ZvecVectorBackend",
    "open_vector_store",
]

EngineKind = Literal["zvec", "faiss", "postgres", "memory"]
FaissDevice = Literal["cpu", "gpu", "auto"]


class InMemoryVectorBackend:
    """Simple in-process cosine store (tests / no optional deps).

    Not for production corpora — use :class:`ZvecVectorBackend` (Alibaba zvec),
    :class:`~loomable.providers.backends.faiss.FaissVectorBackend`, or
    :class:`~loomable.providers.backends.postgres.PgVectorBackend`.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._store[id] = {"vector": list(vector), "metadata": dict(metadata or {})}

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        if not self._store:
            return []
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item_id, item in self._store.items():
            score = _cosine_similarity(vector, item["vector"])
            scored.append((score, item_id, item["metadata"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": item_id, "score": score, **metadata}
            for score, item_id, metadata in scored[:k]
        ]

    async def delete(self, id: str) -> None:
        self._store.pop(id, None)


class LongTermStore:
    """Long-term memory store backed by a pluggable VectorBackend.

    Resolution order:
    1. Explicit ``backend=`` (Postgres, custom, …)
    2. ``path=`` → Alibaba :class:`ZvecVectorBackend` (file-based)
    3. else → :class:`InMemoryVectorBackend`
    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
        backend_name: str = "zvec",
        *,
        path: str | Path | None = None,
        dimensions: int | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
            self.backend_name = backend_name
        elif path is not None:
            self.backend = ZvecVectorBackend(path, dimensions=dimensions)
            self.backend_name = "zvec"
        else:
            self.backend = InMemoryVectorBackend()
            self.backend_name = "memory" if backend_name == "zvec" else backend_name

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        try:
            await self.backend.index(id, vector, metadata)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        try:
            return await self.backend.query(vector, k)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc

    async def delete(self, id: str) -> None:
        try:
            await self.backend.delete(id)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc

    def close(self) -> None:
        """Release backend resources (e.g. Alibaba zvec collection lock)."""
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    async def aclose(self) -> None:
        closer = getattr(self.backend, "aclose", None) or getattr(self.backend, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result


def open_vector_store(
    *,
    path: str | Path | None = None,
    backend: VectorBackend | None = None,
    postgres_url: str | None = None,
    dimensions: int | None = None,
    user_id: str | None = None,
    engine: EngineKind | None = None,
    device: FaissDevice = "cpu",
    gpu_id: int = 0,
) -> LongTermStore:
    """Factory: Alibaba zvec, FAISS (CPU/GPU), Postgres, or explicit backend.

    ::

        open_vector_store(path="./.loomable/vectors")                    # Alibaba zvec
        open_vector_store(engine="faiss", path="./.loomable/faiss",
                          dimensions=1536, device="auto")               # FAISS
        open_vector_store(postgres_url=DSN, dimensions=1536)            # Postgres
        open_vector_store(backend=my_backend)                           # anything VectorBackend

    ``engine`` selects the store when multiple options could apply. Defaults:
    ``postgres_url`` → postgres; ``path`` → zvec; else memory. Pass
    ``engine="faiss"`` to use FAISS (``pip install loomable[faiss]``; for GPU
    install ``faiss-gpu`` instead of ``faiss-cpu``).
    """
    if backend is not None:
        return LongTermStore(backend=backend, backend_name="custom")

    eng = (engine or "").strip().lower() or None
    if eng == "postgres" or (postgres_url and eng is None):
        if not postgres_url:
            raise ValueError("engine='postgres' requires postgres_url=")
        from loomable.providers.backends.postgres import PgVectorBackend

        dims = int(dimensions or 1536)
        return LongTermStore(
            backend=PgVectorBackend(
                postgres_url, dimensions=dims, user_id=user_id
            ),
            backend_name="postgres",
        )
    if eng == "faiss":
        from loomable.providers.backends.faiss import FaissVectorBackend

        if dimensions is None:
            raise ValueError("engine='faiss' requires dimensions=")
        return LongTermStore(
            backend=FaissVectorBackend(
                dimensions=int(dimensions),
                path=path,
                device=device,
                gpu_id=gpu_id,
            ),
            backend_name="faiss",
        )
    if eng == "memory":
        return LongTermStore(backend_name="memory")
    if eng == "zvec" or (path is not None and eng is None):
        if path is None:
            raise ValueError("engine='zvec' requires path=")
        return LongTermStore(path=path, dimensions=dimensions, backend_name="zvec")
    return LongTermStore(backend_name="memory")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
