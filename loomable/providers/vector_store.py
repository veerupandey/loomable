"""Factory for pluggable vector stores (zvec / FAISS / Postgres / memory).

Lives outside ``loomable.kernel`` so the kernel stays free of provider imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loomable.kernel.contracts import VectorBackend
from loomable.kernel.long_term import (
    DEFAULT_ZVEC_PATH,
    InMemoryVectorBackend,
    LongTermStore,
)

__all__ = ["open_vector_store"]

EngineKind = Literal["zvec", "faiss", "postgres", "memory"]
FaissDevice = Literal["cpu", "gpu", "auto"]


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
    """Factory for agent L3 / retrieval vector stores.

    Defaults to **Alibaba zvec** (``path`` or :data:`~loomable.kernel.long_term.DEFAULT_ZVEC_PATH`).

    ::

        open_vector_store()                                             # zvec default path
        open_vector_store(path="./.loomable/notes_zvec")                # zvec custom path
        open_vector_store(engine="faiss", path="./.loomable/faiss",
                          dimensions=1536, device="auto")               # FAISS
        open_vector_store(postgres_url=DSN, dimensions=1536)            # Postgres
        open_vector_store(engine="memory")                              # tests / ephemeral
        open_vector_store(backend=my_backend)                           # anything VectorBackend
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
        return LongTermStore(
            backend=InMemoryVectorBackend(),
            backend_name="memory",
        )
    if eng == "zvec" or eng is None:
        return LongTermStore(
            path=path if path is not None else DEFAULT_ZVEC_PATH,
            dimensions=dimensions,
            backend_name="zvec",
        )
    raise ValueError(f"unknown engine={engine!r}; use zvec|faiss|postgres|memory")
