"""Exercise a multi-turn support console with real chat and embedding APIs."""

import asyncio
import os

import httpx
import pytest

from loomable import Agent, ConversationMemory, Memory, MemoryScope, UserMemory, open_session_store
from loomable.memory import NoteStore, ScopedNoteStore
from loomable.providers import GeminiEmbedder, OpenAIEmbedder, open_vector_store
from loomable.serve import FastAPIAdapter
from tests.integration.test_model_live import live_provider

pytestmark = pytest.mark.live


async def test_support_console_remembers_and_forgets_across_sessions(tmp_path):
    pytest.importorskip("faiss")
    backend = os.environ.get("LOOMABLE_LIVE_PROVIDER", "gemini")
    if backend == "openai":
        embedder, dimensions = OpenAIEmbedder(), 1536
    elif backend == "gemini":
        embedder, dimensions = GeminiEmbedder(), 3072
    else:
        pytest.skip("this app uses OpenAI or Gemini embeddings")

    def storage():
        vectors = open_vector_store(engine="faiss", path=tmp_path / "notes", dimensions=dimensions)
        return vectors, NoteStore(vectors, embedder)

    def console(provider, notes, customer, *, conversation=True, memory_tool=False):
        memory = Memory.compose(
            conversation=ConversationMemory(
                store=open_session_store("file", path=str(tmp_path / customer)),
                enabled=conversation, window=8,
            ),
            user=UserMemory(note_store=notes, auto_extract=True, memory_tool=memory_tool),
        )
        agent = Agent(
            model=provider, memory=memory, user_id=customer, max_cached_sessions=1,
            instructions=(
                "You are a support console. Use conversation history and saved user facts. "
                "For lookup questions return only the requested value or UNKNOWN if absent. "
                "Never guess. When asked to save a note, call memory with the supplied ID and text."
            ),
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=FastAPIAdapter(agent).app()), base_url="http://support",
        )

    async def turn(client, text, session=None):
        response = await client.post("/run", json={"input_text": text, "session_id": session})
        assert response.status_code == 200, response.text
        return "".join(part["text"] for part in response.json()["output"]).strip().upper()

    async with live_provider() as provider:
        vectors, notes = storage()
        alice = ScopedNoteStore(notes, scope=MemoryScope.of(user_id="alice"))
        try:
            async with console(provider, notes, "alice", memory_tool=True) as client:
                await turn(client, "The order reference for this ticket is QZ-8421. Acknowledge.", "ticket")
                assert "QZ-8421" in await turn(client, "What is the order reference?", "ticket")
                await turn(client, "My favorite color is persimmon. Acknowledge.", "profile")
                await turn(client, "Save a note with ID shipping and text Shipping service code: UP-7429. Acknowledge.", "profile")
                assert "UP-7429" in (await alice.read("shipping")).text
                assert any("persimmon" in note.text.lower() for note in await alice.list())
                assert "QZ-8421" in await turn(client, "What is the order reference?", "ticket")
        finally:
            vectors.close()

        vectors, notes = storage()
        alice = ScopedNoteStore(notes, scope=MemoryScope.of(user_id="alice"))
        try:
            async with console(provider, notes, "alice") as reopened:
                assert "QZ-8421" in await turn(reopened, "What is the order reference?", "ticket")
                assert "PERSIMMON" in await turn(reopened, "What is my favorite color?", "new-ticket")
                assert "QZ-8421" not in await turn(reopened, "What is the order reference?", "new-ticket")
            async with console(provider, notes, "bob") as other_customer:
                assert "PERSIMMON" not in await turn(other_customer, "What is my favorite color?", "ticket")
                assert "QZ-8421" not in await turn(other_customer, "What is the order reference?", "ticket")

            bob = ScopedNoteStore(notes, scope=MemoryScope.of(user_id="bob"))
            await bob.write("shipping", "Shipping service code: FX-9021", tags=["scope:user_id=alice"])
            shipping = [note for note in await alice.list() if "Shipping service code:" in note.text]
            assert len(shipping) == 1 and "UP-7429" in shipping[0].text
            claim = ScopedNoteStore(notes, scope=MemoryScope.of(claim_id="C1", user_id="alice"))
            delimiter = ScopedNoteStore(notes, scope=MemoryScope.of(claim_id="C1|user_id=alice"))
            await claim.write("reference", "Claim reference: CLAIM-8217")
            assert await delimiter.read("reference") is None
            assert await delimiter.list() == []
            async with console(provider, notes, "alice", conversation=False) as notes_only:
                assert "UP-7429" in await turn(notes_only, "What is my shipping service code?")
                await alice.write("shipping", "Shipping service code: DH-6123")
                answer = await turn(notes_only, "What is my shipping service code?")
                assert "DH-6123" in answer and "UP-7429" not in answer
                await alice.delete("shipping")
                answer = await turn(notes_only, "What is my shipping service code?")
                assert all(carrier not in answer for carrier in ("UP-7429", "DH-6123", "FX-9021"))
                await turn(notes_only, "My favorite snack is tamarind. Acknowledge.")
                assert "TAMARIND" in await turn(notes_only, "What is my favorite snack?", "another-ticket")
        finally:
            vectors.close()


async def test_support_ticket_compacts_with_a_pooled_model(tmp_path):
    store = open_session_store("file", path=str(tmp_path))
    memory = Memory.compose(conversation=ConversationMemory(
        store=store, window=4, compaction_threshold=4, use_llm_summarizer=True,
    ))
    async with live_provider() as provider, asyncio.timeout(30):
        agent = Agent(
            model=provider, memory=memory, session_id="ticket",
            instructions="You are a support agent. Keep exact identifiers. Acknowledge facts concisely.",
        )
        await agent.arun(
            "Acknowledge without repeating the details. The customer has scheduled the repair, "
            "and the public repair ticket label to retain exactly is XY-7693."
        )
        await agent.arun("The repair is scheduled for Friday. Acknowledge.")
        await agent.arun("The technician will bring a spare. Acknowledge.")
        saved = store.resume("ticket")
        assert len(saved.l1) == 4 and len(saved.l2) == 1
        summary = saved.l2[0]
        assert any("XY-7693" in part for part in [summary.text, *summary.objectives, *summary.decisions])
        reopened = Agent(model=provider, memory=memory, session_id="ticket", resume=True)
        answer = await reopened.arun("Repeat the exact public repair ticket label, including letters and the hyphen.")
        assert "XY-7693" in answer.text
