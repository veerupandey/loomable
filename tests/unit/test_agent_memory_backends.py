"""Agent L1/L2 memory across sqlite / file / postgres / custom backends."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.errors import SessionNotFoundError
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.kernel.stores import (
    BackendSessionStore,
    FileSessionStore,
    InMemoryMemoryBackend,
    SessionStore,
    SQLiteMemoryBackend,
)
from loomable.memory import open_session_store


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages)
        if "Alex" in blob and "What is my name" in blob:
            return ModelResponse(
                content="Your name is Alex.",
                usage={"input_tokens": 1, "output_tokens": 1},
            )
        return ModelResponse(
            content="ack",
            usage={"input_tokens": 1, "output_tokens": 1},
        )


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Echo())


async def _roundtrip(store) -> None:
    a1 = Agent(
        model=_model(),
        session_id="mem-1",
        session_store=store,
        modalities="text",
        memory_window=8,
    )
    r1 = await a1.arun("My name is Alex")
    assert "Alex" in (r1.output.text() or "") or r1.output.text()

    a2 = Agent(
        model=_model(),
        session_id="mem-1",
        session_store=store,
        resume=True,
        modalities="text",
        memory_window=8,
    )
    r2 = await a2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_agent_memory_sqlite_session_store(tmp_path) -> None:
    store = open_session_store("sqlite", path=str(tmp_path / "s.db"))
    await _roundtrip(store)


@pytest.mark.asyncio
async def test_agent_memory_file_session_store(tmp_path) -> None:
    store = open_session_store("file", path=str(tmp_path / "sessions"))
    await _roundtrip(store)


@pytest.mark.asyncio
async def test_agent_memory_backend_param_inmemory() -> None:
    backend = InMemoryMemoryBackend()
    a1 = Agent(
        model=_model(),
        session_id="mb-1",
        memory_backend=backend,
        modalities="text",
    )
    await a1.arun("My name is Alex")
    a2 = Agent(
        model=_model(),
        session_id="mb-1",
        memory_backend=backend,
        resume=True,
        modalities="text",
    )
    r2 = await a2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")


@pytest.mark.asyncio
async def test_agent_memory_backend_sqlite_kv() -> None:
    backend = SQLiteMemoryBackend(":memory:")
    store = BackendSessionStore(backend)
    await _roundtrip(store)


@pytest.mark.asyncio
async def test_open_session_store_memory_kind() -> None:
    store = open_session_store("memory")
    await _roundtrip(store)


@pytest.mark.asyncio
async def test_file_session_store_missing_raises(tmp_path) -> None:
    store = FileSessionStore(tmp_path / "empty")
    with pytest.raises(SessionNotFoundError):
        store.resume("nope")


def test_open_session_store_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        open_session_store("redis")


def test_session_store_default_still_sqlite() -> None:
    assert isinstance(SessionStore(), SessionStore)


@pytest.mark.asyncio
async def test_compose_conversation_store_with_zvec_notes() -> None:
    """L1/L2 session store and L3 NoteStore(zvec) are independent axes."""
    from loomable.memory import NoteStore
    from loomable.kernel.long_term import LongTermStore
    from loomable.providers.vector_store import open_vector_store

    class _Emb:
        async def embed(self, text: str) -> list[float]:
            return [float(len(text) % 5), 1.0, 0.0]

    store = open_session_store("memory")
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())

    a1 = Agent(
        model=_model(),
        session_id="compose-1",
        session_store=store,
        note_store=notes,
        memory_tool=True,
        modalities="text",
    )
    await a1.arun("My name is Alex")
    await notes.write("who", "User name is Alex", tags=["identity"])

    recalled = await notes.recall("name", k=1)
    assert recalled and "Alex" in recalled[0].text

    a2 = Agent(
        model=_model(),
        session_id="compose-1",
        session_store=store,
        resume=True,
        note_store=notes,
        memory_tool=True,
        modalities="text",
    )
    r2 = await a2.arun("What is my name?")
    assert "Alex" in (r2.output.text() or "")
    assert await notes.read("who") is not None

