"""VectorBackend get/scan and L3 memory_error event coverage."""

from __future__ import annotations

from typing import Any

import pytest

from loomable.agent.builder import BuiltAgent
from loomable.agent.events import Event, JSONTracer
from loomable.kernel.long_term import InMemoryVectorBackend, LongTermStore
from loomable.providers.vector_store import open_vector_store


@pytest.mark.asyncio
async def test_inmemory_backend_get_scan() -> None:
    backend = InMemoryVectorBackend()
    await backend.index("a", [1.0, 0.0], {"text": "alpha"})
    await backend.index("b", [0.0, 1.0], {"text": "beta"})
    got = await backend.get("a")
    assert got is not None
    assert got["id"] == "a"
    assert got["text"] == "alpha"
    assert await backend.get("missing") is None
    scanned = await backend.scan(limit=10)
    ids = {row["id"] for row in scanned}
    assert ids == {"a", "b"}
    limited = await backend.scan(limit=1)
    assert len(limited) == 1


@pytest.mark.asyncio
async def test_long_term_store_get_scan_wrappers() -> None:
    store = open_vector_store(engine="memory")
    await store.index("doc-1", [0.5, 0.5], {"note_id": "doc-1", "text": "hi"})
    row = await store.get("doc-1")
    assert row is not None and row["id"] == "doc-1"
    rows = await store.scan(limit=5)
    assert any(r["id"] == "doc-1" for r in rows)


@pytest.mark.asyncio
async def test_faiss_get_scan() -> None:
    pytest.importorskip("faiss")
    from loomable.providers.backends.faiss import FaissVectorBackend

    backend = FaissVectorBackend(dimensions=3, device="cpu")
    await backend.index("a", [1.0, 0.0, 0.0], {"text": "alpha"})
    await backend.index("b", [0.0, 1.0, 0.0], {"text": "beta"})
    got = await backend.get("a")
    assert got is not None and got["text"] == "alpha"
    # Dim-strict query must still reject zero-vector; get/scan must not use it.
    with pytest.raises(ValueError):
        await backend.query([0.0], k=10)
    scanned = await backend.scan()
    assert {r["id"] for r in scanned} == {"a", "b"}


@pytest.mark.asyncio
async def test_memory_error_event_on_user_recall_failure() -> None:
    import io

    class _BoomNotes:
        async def recall(self, query: str, k: int = 3) -> list[Any]:
            raise RuntimeError("vector backend down")

    class _Stub:
        pass

    tracer = JSONTracer(stream=io.StringIO())
    fake = _Stub()
    fake.memory_auto_recall = True
    fake.note_store = _BoomNotes()
    fake._cached_user_facts = []
    fake.events = tracer

    await BuiltAgent._refresh_user_memory_context(fake, "who am I")  # type: ignore[arg-type]

    kinds = [e.kind for e in tracer.trace]
    assert "memory_error" in kinds
    err = next(e for e in tracer.trace if e.kind == "memory_error")
    assert err.attributes["op"] == "user_recall"
    assert err.attributes["error_type"] == "RuntimeError"
    assert "vector backend down" in err.attributes["error"]


@pytest.mark.asyncio
async def test_memory_error_event_on_auto_extract_failure() -> None:
    import io

    class _BoomNotes:
        async def write(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("cannot write note")

    class _Stub:
        pass

    tracer = JSONTracer(stream=io.StringIO())
    fake = _Stub()
    fake.persist_session = True
    fake.session_store = type("S", (), {"save": staticmethod(lambda s: None)})()
    fake.session = type("Sess", (), {"l1": [], "l2": [], "step": 0})()
    fake.session.l1 = []
    fake.session.l2 = []
    fake.session.step = 0
    fake.summarizer = None
    fake.context_policy = None
    fake.memory_window = 8
    fake.compaction_threshold = 16
    fake.pinned_steps = set()
    fake.events = tracer
    fake.memory_auto_extract = True
    fake.note_store = _BoomNotes()
    fake._memory_user_id = "u1"
    fake._memory_scope = None
    fake._cached_user_facts = []
    fake._token_budget = 8192

    # Patch auto_extract to raise so we hit the emit path cleanly.
    import loomable.memory.compose as compose_mod

    async def _boom(*_a: Any, **_k: Any) -> list[str]:
        raise RuntimeError("extract failed")

    orig = compose_mod.auto_extract_into_notes
    compose_mod.auto_extract_into_notes = _boom  # type: ignore[assignment]
    try:
        BuiltAgent._persist_session(fake, "My name is Sam", "Hello Sam")  # type: ignore[arg-type]
    finally:
        compose_mod.auto_extract_into_notes = orig  # type: ignore[assignment]

    err = next(e for e in tracer.trace if e.kind == "memory_error")
    assert err.attributes["op"] == "auto_extract"
    assert err.attributes["error_type"] == "RuntimeError"
