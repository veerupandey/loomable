"""Factory for pluggable vector stores.

Engines: ``zvec`` (default) · ``faiss`` · ``postgres`` / pgvector · ``chroma`` ·
``milvus`` · ``memory``.

Also accepts a ``uri`` shorthand::

    open_vector_store(uri="zvec:./.loomable/notes_zvec")
    open_vector_store(uri="faiss:./.loomable/faiss", dimensions=384)
    open_vector_store(uri="chroma:./.loomable/chroma", dimensions=384)
    open_vector_store(uri="milvus:./.loomable/milvus.db", dimensions=384)
    open_vector_store(uri="postgresql://user:pass@localhost:5432/db", dimensions=384)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from loomable.kernel.contracts import VectorBackend
from loomable.kernel.long_term import (
    DEFAULT_ZVEC_PATH,
    InMemoryVectorBackend,
    LongTermStore,
)

__all__ = ["open_vector_store", "parse_vector_uri"]

EngineKind = Literal[
    "zvec", "faiss", "postgres", "memory", "chroma", "milvus"
]
FaissDevice = Literal["cpu", "gpu", "auto"]


def parse_vector_uri(uri: str) -> dict[str, Any]:
    """Parse a vector-store URI into factory kwargs.

    Supported::

        zvec:/path | zvec:///path | file:/path (treated as zvec when no scheme engine)
        faiss:/path
        chroma:/path | chroma://host:8000
        milvus:/path/to.db | milvus://host:19530 | http://host:19530 (milvus)
        postgresql://... | postgres://...
        memory:
    """
    raw = (uri or "").strip()
    if not raw:
        raise ValueError("uri must be non-empty")

    # engine:path forms (zvec:./x, faiss:/tmp/x, chroma:./c, milvus:./m.db)
    for eng in ("zvec", "faiss", "chroma", "milvus", "memory", "postgres", "postgresql"):
        prefix = f"{eng}:"
        if raw.lower().startswith(prefix) and not raw.lower().startswith(f"{eng}://"):
            rest = raw[len(prefix) :]
            if eng in {"postgres", "postgresql"}:
                return {"engine": "postgres", "postgres_url": rest}
            if eng == "memory":
                return {"engine": "memory"}
            return {"engine": "zvec" if eng == "zvec" else eng, "path": rest or None}

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"postgres", "postgresql"}:
        return {"engine": "postgres", "postgres_url": raw}
    if scheme == "memory":
        return {"engine": "memory"}
    if scheme in {"zvec", "faiss"}:
        path = parsed.path or ""
        if parsed.netloc and path:
            path = f"/{parsed.netloc}{path}"
        elif parsed.netloc and not path:
            path = parsed.netloc
        return {"engine": scheme, "path": path or None}
    if scheme == "chroma":
        # chroma://host:8000 or chroma:///abs/path
        if parsed.port or (parsed.hostname and not raw.startswith("chroma:///")):
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            return {"engine": "chroma", "uri": f"http://{host}:{port}"}
        path = parsed.path or ""
        return {"engine": "chroma", "path": path or None}
    if scheme == "milvus":
        if parsed.port or (
            parsed.hostname
            and not str(parsed.path).endswith(".db")
            and not raw.startswith("milvus:///")
        ):
            host = parsed.hostname or "localhost"
            port = parsed.port or 19530
            return {"engine": "milvus", "uri": f"http://{host}:{port}"}
        path = parsed.path or ""
        if parsed.netloc and path:
            path = f"/{parsed.netloc}{path}"
        return {"engine": "milvus", "path": path or None}
    if scheme in {"http", "https"}:
        # Heuristic: 19530 → milvus, 8000 → chroma, else chroma
        port = parsed.port
        if port == 19530:
            return {"engine": "milvus", "uri": raw}
        return {"engine": "chroma", "uri": raw}
    if scheme == "file":
        return {"engine": "zvec", "path": parsed.path}
    # Bare path → zvec
    return {"engine": "zvec", "path": raw}


def open_vector_store(
    *,
    path: str | Path | None = None,
    backend: VectorBackend | None = None,
    postgres_url: str | None = None,
    uri: str | None = None,
    dimensions: int | None = None,
    user_id: str | None = None,
    engine: EngineKind | str | None = None,
    device: FaissDevice = "cpu",
    gpu_id: int = 0,
    collection: str | None = None,
    token: str | None = None,
) -> LongTermStore:
    """Factory for agent L3 / retrieval vector stores.

    Defaults to **Alibaba zvec** (``path`` or :data:`~loomable.kernel.long_term.DEFAULT_ZVEC_PATH`).

    ::

        open_vector_store()
        open_vector_store(engine="faiss", path="./.loomable/faiss", dimensions=384)
        open_vector_store(engine="chroma", path="./.loomable/chroma", dimensions=384)
        open_vector_store(engine="milvus", path="./.loomable/milvus.db", dimensions=384)
        open_vector_store(postgres_url=DSN, dimensions=1536)
        open_vector_store(uri="milvus:./.loomable/m.db", dimensions=384)
        open_vector_store(engine="memory")
    """
    if backend is not None:
        return LongTermStore(backend=backend, backend_name="custom")

    if uri:
        parsed = parse_vector_uri(uri)
        engine = engine or parsed.get("engine")
        path = path if path is not None else parsed.get("path")
        postgres_url = postgres_url or parsed.get("postgres_url")
        # remote chroma/milvus
        remote_uri = parsed.get("uri")
    else:
        remote_uri = None

    eng = (engine or "").strip().lower() or None
    coll = collection or "loomable"

    if eng == "postgres" or (postgres_url and eng is None):
        if not postgres_url:
            raise ValueError("engine='postgres' requires postgres_url= or uri=")
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
    if eng == "chroma":
        from loomable.providers.backends.chroma import ChromaVectorBackend

        if dimensions is None:
            raise ValueError("engine='chroma' requires dimensions=")
        return LongTermStore(
            backend=ChromaVectorBackend(
                dimensions=int(dimensions),
                path=path,
                uri=remote_uri,
                collection=coll,
            ),
            backend_name="chroma",
        )
    if eng == "milvus":
        from loomable.providers.backends.milvus import MilvusVectorBackend

        if dimensions is None:
            raise ValueError("engine='milvus' requires dimensions=")
        return LongTermStore(
            backend=MilvusVectorBackend(
                dimensions=int(dimensions),
                path=path,
                uri=remote_uri,
                collection=coll,
                token=token,
            ),
            backend_name="milvus",
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
    raise ValueError(
        f"unknown engine={engine!r}; use zvec|faiss|postgres|chroma|milvus|memory"
    )
