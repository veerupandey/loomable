"""Unit tests for loomable.kernel.memory.MemoryManager.

Tests cover:
- L1: record_turn appends turns
- L2: add_summary appends summaries
- L3: recall returns similarity-ranked items from the LongTermStore
"""

import pytest

from loomable.kernel.long_term import InMemoryVectorBackend, LongTermStore
from loomable.providers.vector_store import open_vector_store
from loomable.kernel.memory import MemoryManager
from loomable.kernel.models import StructuredSummary, Turn


def _mm() -> MemoryManager:
    """MemoryManager with in-memory L3 (unit tests; product default is Alibaba zvec)."""
    return MemoryManager(long_term_store=open_vector_store(engine="memory"))


class TestMemoryManagerL1:
    """Tests for L1 (raw recent turns) tier."""

    def test_record_turn_appends_to_l1(self) -> None:
        mm = _mm()
        turn = Turn(role="user", content="hello", tokens=5, step=1)
        mm.record_turn(turn)
        assert mm.l1 == [turn]

    def test_record_multiple_turns_preserves_order(self) -> None:
        mm = _mm()
        t1 = Turn(role="user", content="hello", tokens=5, step=1)
        t2 = Turn(role="assistant", content="hi", tokens=3, step=2)
        t3 = Turn(role="user", content="how are you?", tokens=8, step=3)
        mm.record_turn(t1)
        mm.record_turn(t2)
        mm.record_turn(t3)
        assert mm.l1 == [t1, t2, t3]

    def test_l1_starts_empty(self) -> None:
        mm = _mm()
        assert mm.l1 == []


class TestMemoryManagerL2:
    """Tests for L2 (compressed summaries/entities) tier."""

    def test_add_summary_appends_to_l2(self) -> None:
        mm = _mm()
        summary = StructuredSummary(
            covers_steps=range(1, 6),
            objectives=["complete task"],
            decisions=["use approach A"],
            text="Summary of steps 1-5",
            tokens=20,
        )
        mm.add_summary(summary)
        assert mm.l2 == [summary]

    def test_add_multiple_summaries_preserves_order(self) -> None:
        mm = _mm()
        s1 = StructuredSummary(
            covers_steps=range(1, 6),
            objectives=["task 1"],
            decisions=["decision A"],
            text="Summary 1",
            tokens=15,
        )
        s2 = StructuredSummary(
            covers_steps=range(6, 11),
            objectives=["task 2"],
            decisions=["decision B"],
            text="Summary 2",
            tokens=18,
        )
        mm.add_summary(s1)
        mm.add_summary(s2)
        assert mm.l2 == [s1, s2]

    def test_l2_starts_empty(self) -> None:
        mm = _mm()
        assert mm.l2 == []


class TestMemoryManagerL3:
    """Tests for L3 (vector episodic memory) tier."""

    async def test_recall_returns_empty_when_no_items_indexed(self) -> None:
        mm = _mm()
        results = await mm.recall([1.0, 0.0, 0.0], k=5)
        assert results == []

    async def test_recall_returns_similarity_ranked_items(self) -> None:
        mm = _mm()
        # Index some items directly into L3
        await mm.l3.index("item1", [1.0, 0.0, 0.0], {"text": "first"})
        await mm.l3.index("item2", [0.0, 1.0, 0.0], {"text": "second"})
        await mm.l3.index("item3", [0.9, 0.1, 0.0], {"text": "third"})

        # Query with a vector close to item1 and item3
        results = await mm.recall([1.0, 0.0, 0.0], k=3)

        assert len(results) == 3
        # item1 should be first (exact match), item3 second (close), item2 last
        assert results[0]["id"] == "item1"
        assert results[1]["id"] == "item3"
        assert results[2]["id"] == "item2"

    async def test_recall_respects_k_limit(self) -> None:
        mm = _mm()
        await mm.l3.index("a", [1.0, 0.0], {"text": "a"})
        await mm.l3.index("b", [0.9, 0.1], {"text": "b"})
        await mm.l3.index("c", [0.0, 1.0], {"text": "c"})

        results = await mm.recall([1.0, 0.0], k=2)
        assert len(results) == 2

    async def test_recall_results_contain_score_and_metadata(self) -> None:
        mm = _mm()
        await mm.l3.index("ep1", [1.0, 0.0], {"topic": "memory", "step": 5})

        results = await mm.recall([1.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0]["id"] == "ep1"
        assert "score" in results[0]
        assert results[0]["topic"] == "memory"
        assert results[0]["step"] == 5


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_default_long_term_store_is_alibaba_zvec(self) -> None:
        mm = MemoryManager()
        assert isinstance(mm.l3, LongTermStore)
        assert mm.l3.backend_name == "zvec"

    def test_custom_long_term_store_used(self) -> None:
        custom_store = LongTermStore(
            backend=InMemoryVectorBackend(), backend_name="custom"
        )
        mm = MemoryManager(long_term_store=custom_store)
        assert mm.l3 is custom_store

    @pytest.mark.asyncio
    async def test_two_default_stores_do_not_deadlock_on_zvec_lock(self) -> None:
        pytest.importorskip("zvec")
        first = LongTermStore()
        await first.index("note-1", [1.0, 0.0, 0.0], {"topic": "lock"})
        second = LongTermStore()
        hits = await second.query([1.0, 0.0, 0.0], k=1)
        assert hits
        assert hits[0]["id"] == "note-1"
