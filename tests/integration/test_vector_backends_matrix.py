"""Cross-backend vector store + agentic RAG tests.

Covers file-based and Docker/server URIs:

- **zvec** — Alibaba file dir
- **faiss** — local index dir
- **chroma** — PersistentClient path (file)
- **milvus** — Milvus Lite ``.db`` file
- **postgres / pgvector** — Docker ``loomable-pg`` when ``POSTGRES_URL`` is set

Run::

    export POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable
    pytest tests/integration/test_vector_backends_matrix.py -q
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from loomable.providers.vector_store import open_vector_store, parse_vector_uri
from loomable.retrieval import AgenticRetriever, ingest

DIMS = 8
POSTGRES_URL = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


class _FixedEmbedder:
    """Deterministic tiny embedder for backend matrix tests."""

    def __init__(self) -> None:
        self._table = {
            "oauth": _unit([1, 0, 0, 0, 0, 0, 0, 0]),
            "auth": _unit([0.95, 0.05, 0, 0, 0, 0, 0, 0]),
            "bearer": _unit([0.9, 0.1, 0, 0, 0, 0, 0, 0]),
            "invoice": _unit([0, 1, 0, 0, 0, 0, 0, 0]),
            "billing": _unit([0, 0.95, 0.05, 0, 0, 0, 0, 0]),
            "discount": _unit([0, 0.9, 0.1, 0, 0, 0, 0, 0]),
            "roast": _unit([0, 0, 1, 0, 0, 0, 0, 0]),
            "gravy": _unit([0, 0, 0.95, 0.05, 0, 0, 0, 0]),
        }

    async def embed(self, text: str) -> list[float]:
        low = (text or "").lower()
        acc = [0.0] * DIMS
        hits = 0
        for token, vec in self._table.items():
            if token in low:
                hits += 1
                acc = [a + b for a, b in zip(acc, vec)]
        if hits == 0:
            # stable hash-ish fallback
            for i, ch in enumerate(low[:DIMS]):
                acc[i % DIMS] += (ord(ch) % 13) / 13.0
        return _unit(acc)


def _seed_docs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "auth.md").write_text(
        "# Auth\n\nOAuth2 bearer tokens for API authentication.\n",
        encoding="utf-8",
    )
    (root / "billing.md").write_text(
        "# Billing\n\nEnterprise invoice discounts.\n",
        encoding="utf-8",
    )
    (root / "cooking.md").write_text(
        "# Cooking\n\nService the roast with gravy.\n",
        encoding="utf-8",
    )
    return root


async def _roundtrip(store_factory: Callable[[], Any], tmp: Path) -> None:
    docs = _seed_docs(tmp / "docs")
    emb = _FixedEmbedder()
    store = store_factory()
    try:
        corpus = await ingest(
            [docs],
            name="docs",
            store=store,
            embedder=emb,
            strategy="markdown",
            base_mode="hybrid",
        )
        rag = AgenticRetriever(
            corpus, name="search_docs", mode="chunks", rewrite="off", rerank="mmr"
        )
        hits = await rag.retrieve("OAuth2 bearer authentication", k=2)
        assert hits, "expected retrieval hits"
        blob = " ".join(str(h.get("content") or "") for h in hits).lower()
        assert "oauth" in blob or "bearer" in blob or "auth" in blob, blob
        # cooking must not beat auth for this query
        top = (hits[0].get("content") or "").lower()
        assert "roast" not in top and "gravy" not in top, top
    finally:
        aclose = getattr(store, "aclose", None)
        if callable(aclose):
            await aclose()
        else:
            close = getattr(store, "close", None)
            if callable(close):
                close()


# ---------------------------------------------------------------------------
# URI parser
# ---------------------------------------------------------------------------


def test_parse_vector_uri_matrix() -> None:
    assert parse_vector_uri("zvec:./.loomable/z")["engine"] == "zvec"
    assert parse_vector_uri("faiss:/tmp/f")["engine"] == "faiss"
    assert parse_vector_uri("chroma:./.loomable/c")["engine"] == "chroma"
    assert parse_vector_uri("milvus:./.loomable/m.db")["engine"] == "milvus"
    assert parse_vector_uri("postgresql://u:p@localhost:5432/db")["engine"] == "postgres"
    assert parse_vector_uri("http://localhost:19530")["engine"] == "milvus"
    assert parse_vector_uri("http://localhost:8000")["engine"] == "chroma"
    assert parse_vector_uri("memory:")["engine"] == "memory"


# ---------------------------------------------------------------------------
# File-based backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zvec_file_backend(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    path = tmp_path / "zvec_store"

    def factory():
        return open_vector_store(engine="zvec", path=path, dimensions=DIMS)

    await _roundtrip(factory, tmp_path)


@pytest.mark.asyncio
async def test_faiss_file_backend(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    path = tmp_path / "faiss_store"

    def factory():
        return open_vector_store(
            engine="faiss", path=path, dimensions=DIMS, device="cpu"
        )

    await _roundtrip(factory, tmp_path)


@pytest.mark.asyncio
async def test_chroma_file_backend(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    path = tmp_path / "chroma_store"

    def factory():
        return open_vector_store(
            uri=f"chroma:{path}",
            dimensions=DIMS,
            collection=f"c_{tmp_path.name[-8:]}",
        )

    await _roundtrip(factory, tmp_path)


@pytest.mark.asyncio
async def test_milvus_lite_file_backend(tmp_path: Path) -> None:
    pytest.importorskip("pymilvus")
    try:
        import milvus_lite  # noqa: F401
    except ImportError:
        pytest.skip("milvus_lite not installed (pip install 'pymilvus[milvus_lite]')")
    path = tmp_path / "milvus.db"

    def factory():
        return open_vector_store(
            engine="milvus",
            path=path,
            dimensions=DIMS,
            collection=f"m_{tmp_path.name[-8:]}",
        )

    await _roundtrip(factory, tmp_path)


# ---------------------------------------------------------------------------
# Postgres (Docker loomable-pg)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_pgvector_docker_backend(tmp_path: Path) -> None:
    if not POSTGRES_URL:
        pytest.skip("POSTGRES_URL not set (Docker: postgresql://loomable:loomable@127.0.0.1:5432/loomable)")
    pytest.importorskip("asyncpg")

    def factory():
        return open_vector_store(
            uri=POSTGRES_URL,
            dimensions=DIMS,
            user_id=f"matrix_{tmp_path.name[-8:]}",
        )

    await _roundtrip(factory, tmp_path)


# ---------------------------------------------------------------------------
# Parametric smoke: all available engines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_available_engines_retrieve(tmp_path: Path) -> None:
    """Run every installed backend; skip only when the optional dep is missing."""
    engines: list[tuple[str, Callable[[], Any]]] = [
        ("memory", lambda: open_vector_store(engine="memory")),
    ]

    try:
        import zvec  # noqa: F401

        engines.append(
            (
                "zvec",
                lambda: open_vector_store(
                    engine="zvec", path=tmp_path / "all_zvec", dimensions=DIMS
                ),
            )
        )
    except ImportError:
        pass
    try:
        import faiss  # noqa: F401

        engines.append(
            (
                "faiss",
                lambda: open_vector_store(
                    engine="faiss", path=tmp_path / "all_faiss", dimensions=DIMS
                ),
            )
        )
    except ImportError:
        pass
    try:
        import chromadb  # noqa: F401

        engines.append(
            (
                "chroma",
                lambda: open_vector_store(
                    engine="chroma",
                    path=tmp_path / "all_chroma",
                    dimensions=DIMS,
                    collection="all_chroma",
                ),
            )
        )
    except ImportError:
        pass
    try:
        import milvus_lite  # noqa: F401
        from pymilvus import MilvusClient  # noqa: F401

        engines.append(
            (
                "milvus",
                lambda: open_vector_store(
                    engine="milvus",
                    path=tmp_path / "all_milvus.db",
                    dimensions=DIMS,
                    collection="all_milvus",
                ),
            )
        )
    except ImportError:
        pass
    if POSTGRES_URL:
        try:
            import asyncpg  # noqa: F401

            engines.append(
                (
                    "postgres",
                    lambda: open_vector_store(
                        postgres_url=POSTGRES_URL,
                        dimensions=DIMS,
                        user_id="all_engines",
                    ),
                )
            )
        except ImportError:
            pass

    if not engines:
        pytest.skip("no vector backends available")
    failures: list[str] = []
    for name, factory in engines:
        try:
            await _roundtrip(factory, tmp_path / name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    assert not failures, "backend matrix failures:\n" + "\n".join(failures)
