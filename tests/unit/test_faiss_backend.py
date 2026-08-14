"""FAISS VectorBackend unit tests (CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("faiss")

from loomable.providers.vector_store import open_vector_store
from loomable.providers.backends.faiss import FaissVectorBackend


@pytest.mark.asyncio
async def test_faiss_index_query_delete_cpu() -> None:
    backend = FaissVectorBackend(dimensions=3, device="cpu")
    await backend.index("a", [1.0, 0.0, 0.0], {"text": "alpha"})
    await backend.index("b", [0.0, 1.0, 0.0], {"text": "beta"})
    hits = await backend.query([1.0, 0.0, 0.0], k=2)
    assert hits and hits[0]["id"] == "a"
    assert hits[0]["text"] == "alpha"
    assert float(hits[0]["score"]) >= 0.99
    assert not backend.using_gpu
    await backend.delete("a")
    hits2 = await backend.query([1.0, 0.0, 0.0], k=2)
    assert all(h["id"] != "a" for h in hits2)


@pytest.mark.asyncio
async def test_faiss_persist_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "faiss_store"
    store = open_vector_store(
        engine="faiss", path=path, dimensions=2, device="cpu"
    )
    await store.index("doc/1", [1.0, 0.0], {"text": "hello"})
    store.close()
    store2 = open_vector_store(
        engine="faiss", path=path, dimensions=2, device="cpu"
    )
    hits = await store2.query([1.0, 0.0], k=1)
    assert hits and hits[0]["id"] == "doc/1"
    assert hits[0]["text"] == "hello"
    assert float(hits[0]["score"]) >= 0.99
    store2.close()


@pytest.mark.asyncio
async def test_faiss_upsert_replaces_vector() -> None:
    backend = FaissVectorBackend(dimensions=2, device="cpu")
    await backend.index("x", [1.0, 0.0], {"n": 1})
    await backend.index("x", [0.0, 1.0], {"n": 2})
    hits = await backend.query([0.0, 1.0], k=1)
    assert hits[0]["id"] == "x"
    assert hits[0]["n"] == 2
    assert float(hits[0]["score"]) >= 0.99


def test_faiss_gpu_unavailable_raises_when_forced() -> None:
    import faiss

    if int(faiss.get_num_gpus()) > 0 and hasattr(faiss, "StandardGpuResources"):
        pytest.skip("GPU available in this environment")
    with pytest.raises(RuntimeError, match="GPU"):
        FaissVectorBackend(dimensions=2, device="gpu")


def test_faiss_auto_stays_cpu_without_gpu() -> None:
    import faiss

    if int(faiss.get_num_gpus()) > 0 and hasattr(faiss, "StandardGpuResources"):
        pytest.skip("GPU available — auto may select GPU")
    backend = FaissVectorBackend(dimensions=2, device="auto")
    assert backend.using_gpu is False
