"""Unit tests for MemoryStore protocol and TieredMemoryStore (Req 12).

Tests:
- Write defaults to episodic tier (Req 12.3)
- Recall returns records across tiers (Req 12.4)
- Working/episodic persist per session_id across instantiations (Req 12.5)
- MemoryStore is swappable — protocol check (Req 12.7)
- TieredMemoryStore satisfies MemoryStore protocol (Req 12.1)
"""

from __future__ import annotations

from typing import Any

import pytest

from loomable.flow.memory import MemoryStore, Tier, TieredMemoryStore, _session_stores


@pytest.fixture(autouse=True)
def _clear_session_stores():
    """Clear the module-level session store between tests."""
    _session_stores.clear()
    yield
    _session_stores.clear()


class TestTieredMemoryStoreProtocol:
    """TieredMemoryStore satisfies the MemoryStore protocol (Req 12.1)."""

    def test_is_instance_of_protocol(self):
        store = TieredMemoryStore()
        assert isinstance(store, MemoryStore)

    def test_protocol_is_runtime_checkable(self):
        """The MemoryStore protocol is runtime_checkable."""
        assert hasattr(MemoryStore, "__protocol_attrs__") or hasattr(
            MemoryStore, "__abstractmethods__"
        )
        # Key: isinstance check works
        store = TieredMemoryStore()
        assert isinstance(store, MemoryStore)


class TestWriteDefaultsToEpisodic:
    """write() defaults to EPISODIC tier (Req 12.3)."""

    async def test_write_without_tier_goes_to_episodic(self):
        store = TieredMemoryStore()
        await store.write("test record")

        results = await store.recall("test record", tiers=[Tier.EPISODIC])
        assert len(results) == 1
        assert results[0]["record"] == "test record"
        assert results[0]["tier"] == "episodic"

    async def test_write_without_tier_not_in_working(self):
        store = TieredMemoryStore()
        await store.write("test record")

        results = await store.recall("test record", tiers=[Tier.WORKING])
        assert len(results) == 0

    async def test_write_without_tier_not_in_procedural(self):
        store = TieredMemoryStore()
        await store.write("test record")

        results = await store.recall("test record", tiers=[Tier.PROCEDURAL])
        assert len(results) == 0


class TestWriteToSpecificTiers:
    """Records go to the specified tier."""

    async def test_write_to_working(self):
        store = TieredMemoryStore()
        await store.write("working item", tier=Tier.WORKING)

        results = await store.recall("working item", tiers=[Tier.WORKING])
        assert len(results) == 1
        assert results[0]["tier"] == "working"

    async def test_write_to_procedural(self):
        store = TieredMemoryStore()
        await store.write("always greet the user", tier=Tier.PROCEDURAL)

        results = await store.recall("greet", tiers=[Tier.PROCEDURAL])
        assert len(results) == 1
        assert results[0]["record"] == "always greet the user"

    async def test_write_to_semantic_fallback(self):
        """Without LongTermStore, semantic writes go to fallback."""
        store = TieredMemoryStore()
        await store.write("earth orbits the sun", tier=Tier.SEMANTIC)

        results = await store.recall("earth", tiers=[Tier.SEMANTIC])
        assert len(results) == 1
        assert results[0]["record"] == "earth orbits the sun"


class TestRecallAcrossTiers:
    """recall() returns relevant records across requested tiers (Req 12.4)."""

    async def test_recall_across_all_tiers(self):
        store = TieredMemoryStore()
        await store.write("working note about Python", tier=Tier.WORKING)
        await store.write("episodic event about Python", tier=Tier.EPISODIC)
        await store.write("procedure for Python style", tier=Tier.PROCEDURAL)

        # Recall with no tier filter searches all
        results = await store.recall("Python")
        assert len(results) == 3

    async def test_recall_filters_by_tier(self):
        store = TieredMemoryStore()
        await store.write("working note about cats", tier=Tier.WORKING)
        await store.write("episodic event about cats", tier=Tier.EPISODIC)

        results = await store.recall("cats", tiers=[Tier.WORKING])
        assert len(results) == 1
        assert results[0]["tier"] == "working"

    async def test_recall_with_multiple_tiers(self):
        store = TieredMemoryStore()
        await store.write("working note about dogs", tier=Tier.WORKING)
        await store.write("episodic event about dogs", tier=Tier.EPISODIC)
        await store.write("procedure about dogs", tier=Tier.PROCEDURAL)

        results = await store.recall("dogs", tiers=[Tier.WORKING, Tier.EPISODIC])
        assert len(results) == 2
        tiers_found = {r["tier"] for r in results}
        assert tiers_found == {"working", "episodic"}

    async def test_recall_respects_k_limit(self):
        store = TieredMemoryStore()
        for i in range(10):
            await store.write(f"event {i} about testing", tier=Tier.EPISODIC)

        results = await store.recall("testing", k=3)
        assert len(results) == 3

    async def test_recall_no_results_for_unmatched_query(self):
        store = TieredMemoryStore()
        await store.write("hello world", tier=Tier.EPISODIC)

        results = await store.recall("xyz_not_found")
        assert len(results) == 0


class TestSessionPersistence:
    """Working/episodic persist per session_id across instantiations (Req 12.5)."""

    async def test_same_session_shares_episodic(self):
        store1 = TieredMemoryStore(session_id="session-a")
        await store1.write("remember this")

        # New instance with same session_id sees the record
        store2 = TieredMemoryStore(session_id="session-a")
        results = await store2.recall("remember", tiers=[Tier.EPISODIC])
        assert len(results) == 1
        assert results[0]["record"] == "remember this"

    async def test_same_session_shares_working(self):
        store1 = TieredMemoryStore(session_id="session-b")
        await store1.write("scratch note", tier=Tier.WORKING)

        store2 = TieredMemoryStore(session_id="session-b")
        results = await store2.recall("scratch", tiers=[Tier.WORKING])
        assert len(results) == 1

    async def test_different_sessions_isolated(self):
        store1 = TieredMemoryStore(session_id="session-x")
        await store1.write("secret for session x")

        store2 = TieredMemoryStore(session_id="session-y")
        results = await store2.recall("secret", tiers=[Tier.EPISODIC])
        assert len(results) == 0

    async def test_no_session_is_private(self):
        """Without a session_id, each store has its own private memory."""
        store1 = TieredMemoryStore()
        await store1.write("private data")

        store2 = TieredMemoryStore()
        results = await store2.recall("private", tiers=[Tier.EPISODIC])
        assert len(results) == 0


class TestMemoryStoreIsSwappable:
    """MemoryStore is swappable — any conforming implementation works (Req 12.7)."""

    async def test_custom_implementation_satisfies_protocol(self):
        """A custom class implementing write and recall is a MemoryStore."""

        class CustomMemory:
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            async def write(self, record: str, *, tier: Tier = Tier.EPISODIC, **meta: Any) -> None:
                self.records.append({"record": record, "tier": tier.value, **meta})

            async def recall(
                self, query: str, *, tiers: list[Tier] | None = None, k: int = 5
            ) -> list[dict[str, Any]]:
                return [r for r in self.records if query.lower() in r["record"].lower()][:k]

        custom = CustomMemory()
        assert isinstance(custom, MemoryStore)

        # It works as a store
        await custom.write("hello from custom")
        results = await custom.recall("hello")
        assert len(results) == 1
        assert results[0]["record"] == "hello from custom"

    async def test_flow_accepts_custom_memory_store(self):
        """A Flow can accept any MemoryStore implementation."""
        from loomable.flow import Flow

        class MinimalMemory:
            async def write(self, record: str, *, tier: Tier = Tier.EPISODIC, **meta: Any) -> None:
                pass

            async def recall(
                self, query: str, *, tiers: list[Tier] | None = None, k: int = 5
            ) -> list[dict[str, Any]]:
                return []

        custom = MinimalMemory()
        assert isinstance(custom, MemoryStore)

        # Flow construction should accept it without error
        async def noop(x: Any) -> str:
            return "done"

        flow = Flow([noop], memory=custom)
        assert flow._memory is custom


class TestMetadataStorage:
    """Metadata passed via **meta is stored and returned."""

    async def test_metadata_stored_on_write(self):
        store = TieredMemoryStore()
        await store.write("tagged record", tier=Tier.EPISODIC, source="user", importance="high")

        results = await store.recall("tagged")
        assert len(results) == 1
        assert results[0]["source"] == "user"
        assert results[0]["importance"] == "high"


class TestMemoryOnContext:
    """Memory is attached to RunContext when Flow runs (Req 12.2)."""

    async def test_memory_available_on_context(self):
        from loomable.agent.context import RunContext
        from loomable.flow import Flow

        captured_memory = {}

        async def capture_node(x: Any, *, context: RunContext | None = None) -> str:
            if context and context.memory:
                captured_memory["store"] = context.memory
            return "done"

        store = TieredMemoryStore(session_id="ctx-test")
        flow = Flow([capture_node], memory=store)
        await flow.arun("go")

        assert captured_memory.get("store") is store
