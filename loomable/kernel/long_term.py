"""loomable.kernel.long_term - Long-Term Store with pluggable vector backend.

**Default (agent memory L3):** Alibaba **zvec** on disk under
``.loomable/memory_zvec`` (``pip install loomable[zvec]``).

For FAISS / Postgres / convenience factory use
:func:`loomable.providers.vector_store.open_vector_store` (kept out of the
kernel so provider backends do not violate kernel independence).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from loomable.kernel.contracts import VectorBackend
from loomable.kernel.errors import MemoryBackendError
from loomable.kernel.zvec_backend import ZvecVectorBackend

__all__ = [
    "DEFAULT_ZVEC_PATH",
    "InMemoryVectorBackend",
    "LongTermStore",
    "ZvecVectorBackend",
]

# Default on-disk Alibaba zvec collection for agent L3 notes / LongTermStore().
DEFAULT_ZVEC_PATH = Path(".loomable") / "memory_zvec"


class InMemoryVectorBackend:
    """Simple in-process cosine store (tests / no optional deps).

    Not the product default — use :class:`ZvecVectorBackend` (Alibaba zvec),
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
            {**metadata, "id": item_id, "score": score}
            for score, item_id, metadata in scored[:k]
        ]

    async def delete(self, id: str) -> None:
        self._store.pop(id, None)


class LongTermStore:
    """Long-term memory store backed by a pluggable VectorBackend.

    Resolution order:
    1. Explicit ``backend=`` (FAISS, Postgres, in-memory, custom, …)
    2. ``path=`` → Alibaba :class:`ZvecVectorBackend` at that directory
    3. else → Alibaba zvec at :data:`DEFAULT_ZVEC_PATH` (``.loomable/memory_zvec``);
       falls back to :class:`InMemoryVectorBackend` if ``zvec`` is not installed.
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
            # Probe zvec now: ZvecVectorBackend defers ImportError until index()
            # when dimensions are unknown, which would skip this fallback.
            try:
                import zvec as _zvec  # noqa: F401
            except ImportError:
                self.backend = InMemoryVectorBackend()
                self.backend_name = "memory"
            else:
                self.backend = ZvecVectorBackend(DEFAULT_ZVEC_PATH, dimensions=dimensions)
                self.backend_name = "zvec"

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
        if not callable(closer):
            return
        result = closer()
        if hasattr(result, "__await__"):
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(result)
            # If a loop is already running, prefer ``await store.aclose()``.

    async def aclose(self) -> None:
        closer = getattr(self.backend, "aclose", None) or getattr(self.backend, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
