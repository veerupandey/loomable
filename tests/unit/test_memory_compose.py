"""Composable Memory API — conversation / user / knowledge layers for Agent."""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec
from loomable.kernel.long_term import LongTermStore
from loomable.providers.vector_store import open_vector_store
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import (
    ConversationMemory,
    Memory,
    MemoryScope,
    NoteStore,
    ScopedNoteStore,
    UserMemory,
    extract_user_facts,
    open_session_store,
)


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages)
        if "Alex" in blob and ("name" in blob.lower() or "Who" in blob):
            return ModelResponse(content="Your name is Alex.")
        if "Known facts about this user" in blob and "Alex" in blob:
            return ModelResponse(content="From memory: your name is Alex.")
        return ModelResponse(content="ack")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0]


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Echo())


@pytest.mark.asyncio
async def test_scoped_recall_under_crowded_multitenant() -> None:
    """Owned notes must surface even when global top-k*4 would exclude them."""
    base = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    alice = ScopedNoteStore(base, scope=MemoryScope.of(user_id="alice"))
    bob = ScopedNoteStore(base, scope=MemoryScope.of(user_id="bob"))
    # Flood Alice's namespace with many "injury" notes so vector top-k is Alice-dominated.
    for i in range(40):
        await alice.write(f"injury-{i}", f"Alice injury note number {i} soft tissue")
    await bob.write("injury", "Bob exclusive rear bumper claim detail")

    hits = await bob.recall("injury soft tissue bumper", k=3)
    assert hits, "Bob's scoped recall must return owned notes"
    assert all("Bob" in n.text for n in hits)
    assert all("Alice" not in n.text for n in hits)


def test_extract_user_facts() -> None:
    facts = extract_user_facts("Hi! My name is Alex. I prefer dark mode.")
    assert any("Alex" in f for f in facts)
    assert any("prefer" in f.lower() for f in facts)


@pytest.mark.asyncio
async def test_scoped_note_store_isolates_users() -> None:
    base = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    alice = ScopedNoteStore(base, scope=MemoryScope.of(user_id="alice"))
    bob = ScopedNoteStore(base, scope=MemoryScope.of(user_id="bob"))
    await alice.write("prefs", "Alice likes teal")
    await bob.write("prefs", "Bob likes crimson")
    a = await alice.recall("likes", k=3)
    b = await bob.recall("likes", k=3)
    assert a and "Alice" in a[0].text
    assert b and "Bob" in b[0].text
    assert all("Alice" not in n.text for n in b)


@pytest.mark.asyncio
async def test_memory_scope_claim_id_isolation() -> None:
    """Insurance-style scopes: same user, different claim_id must not leak."""
    from loomable.memory import MemoryScope

    base = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    c1 = ScopedNoteStore(
        base, scope=MemoryScope.of(user_id="alice", claim_id="CLM-1")
    )
    c2 = ScopedNoteStore(
        base, scope=MemoryScope.of(user_id="alice", claim_id="CLM-2")
    )
    await c1.write("injury", "Claim 1: soft-tissue neck injury")
    await c2.write("injury", "Claim 2: rear bumper only")

    h1 = await c1.recall("injury", k=5)
    h2 = await c2.recall("injury", k=5)
    assert h1 and "Claim 1" in h1[0].text
    assert h2 and "Claim 2" in h2[0].text
    assert all("Claim 2" not in n.text for n in h1)
    assert all("Claim 1" not in n.text for n in h2)

    memory = Memory.compose(
        conversation=ConversationMemory(store=open_session_store("memory")),
        user=UserMemory(note_store=base, memory_tool=True, auto_extract=False),
    )
    agent = Agent(
        model=_model(),
        memory=memory,
        session_id="claim-CLM-1",
        user_id="alice",
        scopes={"claim_id": "CLM-1"},
        modalities="text",
    )
    assert isinstance(agent._note_store, ScopedNoteStore)
    assert agent._note_store.scope.get("claim_id") == "CLM-1"
    assert "user_id=alice" in agent._note_store.scope.prefix
    assert "claim_id=CLM-1" in agent._note_store.scope.prefix


@pytest.mark.asyncio
async def test_memory_compose_conversation_and_user_auto_extract() -> None:
    store = open_session_store("memory")
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    memory = Memory.compose(
        conversation=ConversationMemory(store=store, window=6),
        user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
    )
    agent = Agent(
        model=_model(),
        memory=memory,
        session_id="compose-1",
        user_id="alice",
        modalities="text",
    )
    assert agent._session_store is store
    assert isinstance(agent._note_store, ScopedNoteStore)
    assert agent._memory_tool is True
    assert agent._memory_auto_extract is True

    kwargs = memory.to_agent_kwargs()
    assert kwargs.get("memory_auto_extract") is True
    flat = Agent(
        model=_model(),
        session_id="compose-flat",
        user_id="alice",
        modalities="text",
        **{k: v for k, v in kwargs.items() if k != "memory"},
    )
    assert flat._memory_auto_extract is True

    await agent.arun("My name is Alex and I prefer dark mode.")
    # Auto-extract should have written scoped notes
    listed = await agent._note_store.list()
    assert listed, "expected auto-extracted user facts"
    assert all("user_id=alice" in n.note_id for n in listed)

    # New agent, same memory bundle → recall injects facts
    agent2 = Agent(
        model=_model(),
        memory=memory,
        session_id="compose-2",  # different chat
        user_id="alice",
        modalities="text",
    )
    r2 = await agent2.arun("What do you know about me?")
    # Either cached facts in prompt or model sees them
    text = r2.output.text() or ""
    assert "Alex" in text or agent2._get_built()._cached_user_facts


@pytest.mark.asyncio
async def test_memory_compose_rejects_flat_store_kwargs() -> None:
    from loomable.agent.errors import AgentConfigError

    store_a = open_session_store("memory")
    store_b = open_session_store("memory")
    memory = Memory.compose(conversation=ConversationMemory(store=store_a))
    with pytest.raises(AgentConfigError, match="Memory.compose"):
        Agent(
            model=_model(),
            memory=memory,
            session_store=store_b,
            session_id="x",
            modalities="text",
        )


@pytest.mark.asyncio
async def test_memory_compose_rejects_working_on_agent() -> None:
    from loomable.agent.errors import AgentConfigError
    from loomable.memory import WorkingMemory

    memory = Memory.compose(working=WorkingMemory.tiered())
    with pytest.raises(AgentConfigError, match="WorkingMemory"):
        Agent(model=_model(), memory=memory, modalities="text")


@pytest.mark.asyncio
async def test_memory_compose_conversation_and_user() -> None:
    store = open_session_store("memory")
    notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=_Emb())
    memory = Memory.compose(
        conversation=ConversationMemory(store=store),
        user=UserMemory(note_store=notes, memory_tool=False),
    )
    assert memory.conversation is not None
    assert memory.user is not None
    agent = Agent(model=_model(), memory=memory, session_id="s", user_id="u1", modalities="text")
    assert agent._note_store is not None
    assert agent._memory_tool is False
