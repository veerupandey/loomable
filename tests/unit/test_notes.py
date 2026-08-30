"""Unit tests for loomable.agent.notes - NoteStore and memory tool."""

from __future__ import annotations

import json

import pytest

from loomable.memory import Note, NoteStore, make_memory_tool
from loomable.kernel.long_term import LongTermStore
from loomable.providers.vector_store import open_vector_store


# ---------------------------------------------------------------------------
# Fake embedder for testing (deterministic, dimension-1 vectors)
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """A fake embedder that returns a simple hash-based vector for testing."""

    async def embed(self, text: str) -> list[float]:
        # Return a single-dimension vector based on text hash for simplicity
        return [float(hash(text) % 1000) / 1000.0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def long_term_store() -> LongTermStore:
    return open_vector_store(engine="memory")  # in-memory (no Alibaba zvec required)


@pytest.fixture
def note_store(long_term_store: LongTermStore, embedder: FakeEmbedder) -> NoteStore:
    return NoteStore(long_term_store, embedder)


# ---------------------------------------------------------------------------
# Note dataclass tests
# ---------------------------------------------------------------------------


class TestNote:
    def test_creation(self):
        note = Note(note_id="abc", text="hello", tags=["x", "y"])
        assert note.note_id == "abc"
        assert note.text == "hello"
        assert note.tags == ["x", "y"]

    def test_default_tags(self):
        note = Note(note_id="abc", text="hello")
        assert note.tags == []


# ---------------------------------------------------------------------------
# NoteStore tests
# ---------------------------------------------------------------------------


class TestNoteStore:
    @pytest.mark.asyncio
    async def test_write_creates_note(self, note_store: NoteStore):
        note = await note_store.write("lesson-1", "Always validate input", ["coding"])
        assert note.note_id == "lesson-1"
        assert note.text == "Always validate input"
        assert note.tags == ["coding"]

    @pytest.mark.asyncio
    async def test_write_upserts_by_id(self, note_store: NoteStore):
        """Req 7.1: listing contains exactly one note per distinct identifier with latest text."""
        await note_store.write("lesson-1", "First version", ["v1"])
        await note_store.write("lesson-1", "Updated version", ["v2"])
        notes = await note_store.list()
        matching = [n for n in notes if n.note_id == "lesson-1"]
        assert len(matching) == 1
        assert matching[0].text == "Updated version"
        assert matching[0].tags == ["v2"]

    @pytest.mark.asyncio
    async def test_read_existing(self, note_store: NoteStore):
        await note_store.write("lesson-1", "Some text", ["tag"])
        note = await note_store.read("lesson-1")
        assert note is not None
        assert note.note_id == "lesson-1"
        assert note.text == "Some text"
        assert note.tags == ["tag"]

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, note_store: NoteStore):
        note = await note_store.read("nonexistent")
        assert note is None

    @pytest.mark.asyncio
    async def test_list_all(self, note_store: NoteStore):
        await note_store.write("a", "Note A", ["x"])
        await note_store.write("b", "Note B", ["y"])
        notes = await note_store.list()
        ids = {n.note_id for n in notes}
        assert ids == {"a", "b"}

    @pytest.mark.asyncio
    async def test_list_by_tag(self, note_store: NoteStore):
        await note_store.write("a", "Note A", ["coding", "python"])
        await note_store.write("b", "Note B", ["design"])
        notes = await note_store.list(tag="coding")
        assert len(notes) == 1
        assert notes[0].note_id == "a"

    @pytest.mark.asyncio
    async def test_delete(self, note_store: NoteStore):
        """Req 7.2: deleted note is removed."""
        await note_store.write("a", "Note A")
        await note_store.delete("a")
        note = await note_store.read("a")
        assert note is None

    @pytest.mark.asyncio
    async def test_recall_vector_search(self, note_store: NoteStore):
        """Req 7.3: recall returns notes relevant to the query."""
        await note_store.write("a", "Python is great for scripting")
        await note_store.write("b", "Java is good for enterprise")
        results = await note_store.recall("Python scripting", k=2)
        # Should return results (order depends on embedding similarity)
        assert len(results) <= 2
        assert all(isinstance(n, Note) for n in results)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_silent(self, note_store: NoteStore):
        # Should not raise
        await note_store.delete("does-not-exist")

    @pytest.mark.asyncio
    async def test_agent_notes_shim_still_exports(self):
        from loomable.agent import NoteStore as AgentNoteStore
        from loomable.agent.notes import NoteStore as ShimNoteStore
        from loomable.memory import NoteStore as MemNoteStore

        assert AgentNoteStore is MemNoteStore
        assert ShimNoteStore is MemNoteStore


# ---------------------------------------------------------------------------
# make_memory_tool tests
# ---------------------------------------------------------------------------


class TestMemoryTool:
    @pytest.fixture
    def memory_tool(self, note_store: NoteStore):
        return make_memory_tool(note_store)

    def test_tool_name(self, memory_tool):
        assert memory_tool.name == "memory"

    def test_tool_schema(self, memory_tool):
        schema = memory_tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "memory"
        assert "action" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_write_action(self, memory_tool):
        """Req 7.4: Memory_Tool exposes write action."""
        result = await memory_tool.invoke({
            "action": "write",
            "note_id": "tip-1",
            "text": "Use type hints",
            "tags": "coding,python",
        })
        data = json.loads(result.content)
        assert data["note_id"] == "tip-1"
        assert data["text"] == "Use type hints"
        assert data["tags"] == ["coding", "python"]

    @pytest.mark.asyncio
    async def test_read_action(self, memory_tool):
        """Req 7.4: Memory_Tool exposes read action."""
        await memory_tool.invoke({
            "action": "write",
            "note_id": "tip-1",
            "text": "Always test",
        })
        result = await memory_tool.invoke({"action": "read", "note_id": "tip-1"})
        data = json.loads(result.content)
        assert data["note_id"] == "tip-1"
        assert data["text"] == "Always test"

    @pytest.mark.asyncio
    async def test_read_not_found(self, memory_tool):
        result = await memory_tool.invoke({"action": "read", "note_id": "missing"})
        data = json.loads(result.content)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_list_action(self, memory_tool):
        """Req 7.4: Memory_Tool exposes list action."""
        await memory_tool.invoke({
            "action": "write",
            "note_id": "a",
            "text": "Note A",
            "tags": "x",
        })
        result = await memory_tool.invoke({"action": "list"})
        data = json.loads(result.content)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["note_id"] == "a"

    @pytest.mark.asyncio
    async def test_delete_action(self, memory_tool):
        """Req 7.4: Memory_Tool exposes delete action."""
        await memory_tool.invoke({
            "action": "write",
            "note_id": "a",
            "text": "Note A",
        })
        result = await memory_tool.invoke({"action": "delete", "note_id": "a"})
        data = json.loads(result.content)
        assert data["deleted"] == "a"

    @pytest.mark.asyncio
    async def test_recall_action(self, memory_tool):
        """Req 7.4: Memory_Tool exposes recall action."""
        await memory_tool.invoke({
            "action": "write",
            "note_id": "tip-1",
            "text": "Use pytest for testing",
        })
        result = await memory_tool.invoke({
            "action": "recall",
            "query": "testing",
            "k": 3,
        })
        data = json.loads(result.content)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_unknown_action(self, memory_tool):
        result = await memory_tool.invoke({"action": "unknown"})
        data = json.loads(result.content)
        assert "error" in data
        assert "Unknown action" in data["error"]

    @pytest.mark.asyncio
    async def test_persists_via_long_term_store(self, note_store: NoteStore):
        """Req 7.5: Notes persist through kernel LongTermStore without modifying kernel."""
        tool = make_memory_tool(note_store)
        await tool.invoke({
            "action": "write",
            "note_id": "persist-1",
            "text": "Persisted note",
        })
        # Verify directly via NoteStore (backed by LongTermStore)
        note = await note_store.read("persist-1")
        assert note is not None
        assert note.text == "Persisted note"
